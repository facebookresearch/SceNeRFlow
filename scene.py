# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import logging

import numpy as np
import torch
from canonical_model import get_canonical_model
from deformation_model import get_deformation_model
from pruning import Pruning
from utils import Returns, infill_masked

LOGGER = logging.getLogger(__name__)


class Scene(torch.nn.Module):
    def __init__(self, settings, data_handler):
        super().__init__()
        self.canonical_model = get_canonical_model(settings)

        self.use_viewdirs = settings.use_viewdirs

        self.use_deformations = True
        if not self.use_deformations:
            self.deformation_model = None
        else:
            self.deformation_model = get_deformation_model(
                settings, data_handler.get_timeline_range()
            )

        self.pruning = Pruning(settings, data_handler)

        # necessary for determine_nerf_volume_extent() to work. will be overwritten with correct values
        self.register_buffer(
            "pos_max", torch.from_numpy(np.array([1e5, 1e5, 1e5], dtype=np.float32))
        )
        self.register_buffer(
            "pos_min", torch.from_numpy(np.array([-1e5, -1e5, -1e5], dtype=np.float32))
        )

    def forward(
        self, positions, view_directions, timesteps, mip_scales, is_training, returns=None, **kwargs
    ):
        # timesteps in [0,1]
        timesteps = torch.clamp(timesteps, min=0.0, max=1.0)

        if returns is None:
            returns = Returns(restricted=[])
            returns.activate_mode("coarse")

        # normalize positions into unit cube for NGP
        positions = (positions - self.pos_min) / (self.pos_max - self.pos_min)  # in [0,1]
        positions.requires_grad = True  # necessary for gradient-based losses
        returns.add_return("normalized_undeformed_positions", positions, clone=False)

        mask = self.pruning(positions, is_training=is_training)
        returns.set_mask(mask=mask)  # fills in tensors to num_rays x num_points_per_ray

        visualize_pruning = False
        if visualize_pruning:
            self._visualize(positions, mask)

        # removing all samples in tensors can lead to issues, keep at least one dummy
        completely_pruned = not torch.any(mask)
        if completely_pruned:
            mask_shape = mask.shape
            mask = mask.view(-1)
            mask[0] = True
            mask = mask.view(mask_shape)

        positions = positions[mask].view(-1, 3)
        view_directions = view_directions[mask].view(-1, 3)
        timesteps = timesteps[mask].view(-1, 1)

        if not self.use_deformations:
            pref_timesteps = None
        else:
            positions, view_directions, pref_timesteps = self.deformation_model(
                positions, timesteps, mask, is_training=is_training, returns=returns
            )

            view_directions = view_directions[mask].view(-1, 3)
            if self.use_viewdirs:
                returns.add_return("view_directions", view_directions)

        rgb, alpha = self.canonical_model(
            positions,
            view_directions,
            timesteps=timesteps,
            pref_timesteps=pref_timesteps,
            mip_scales=mip_scales,
            returns=returns,
            **kwargs
        )

        if completely_pruned:
            alpha *= 0.0

        returns.set_mask(mask=None)

        # get infilled num_rays x num_points_per_ray tensors
        rgb = infill_masked(mask, rgb, infill_value=0)
        alpha = infill_masked(mask, alpha, infill_value=0)

        return rgb, alpha, mask

    def _visualize(self, positions, mask):
        if self.pruning.voxel_grid is not None:

            from tqdm import tqdm

            LOGGER.info("writing debug point clouds")

            debug_pos = positions * (self.pos_max - self.pos_min) + self.pos_min
            prune = debug_pos[~mask].view(-1, 3)
            keep = debug_pos[mask].view(-1, 3)
            samples_list = []
            for x, y, z in tqdm(prune.cpu().detach().numpy()[::100]):
                samples_list.append("v " + str(x) + " " + str(y) + " " + str(z) + " 1 0 0")
            for x, y, z in tqdm(keep.cpu().detach().numpy()[::100]):
                samples_list.append("v " + str(x) + " " + str(y) + " " + str(z) + " 0 1 0")
            samples_string = "\n".join(samples_list)
            output_file = "samples.obj"
            with open(output_file, "w") as output_file:
                output_file.write(samples_string)

            voxel_grid_list = []
            size = self.pruning.voxel_grid_size
            for x in tqdm(range(size)):
                for y in range(size):
                    for z in range(size):
                        occ = self.pruning.voxel_grid[x, y, z]
                        if occ:
                            x2 = x + 0.5
                            y2 = y + 0.5
                            z2 = z + 0.5
                            x2 = (x2 / size) * (self.pos_max[0] - self.pos_min[0]) + self.pos_min[0]
                            y2 = (y2 / size) * (self.pos_max[1] - self.pos_min[1]) + self.pos_min[1]
                            z2 = (z2 / size) * (self.pos_max[2] - self.pos_min[2]) + self.pos_min[2]
                            voxel = (
                                "v "
                                + str(x2.item())
                                + " "
                                + str(y2.item())
                                + " "
                                + str(z2.item())
                                + " 0 0 1"
                            )
                            voxel_grid_list.append(voxel)
            voxel_grid_string = "\n".join(voxel_grid_list)
            output_file = "voxel_grid.obj"
            with open(output_file, "w") as output_file:
                output_file.write(voxel_grid_string)

    def set_pos_max_min(self, pos_max, pos_min):
        self.pos_max = pos_max
        self.pos_min = pos_min

    def get_pos_max(self):
        return self.pos_max

    def get_pos_min(self):
        return self.pos_min

    def state_dict(self):
        state_dict = super().state_dict()
        state_dict["pos_max"] = self.pos_max
        state_dict["pos_min"] = self.pos_min
        return state_dict

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.pos_max = state_dict["pos_max"]
        self.pos_min = state_dict["pos_min"]

    def _add_tags(self, parameters, tags):
        for parameter in parameters:
            parameter["tags"] += tags
        return parameters

    def step(self):
        self.canonical_model.step()
        if self.use_deformations:
            self.deformation_model.step()

    def get_parameters_with_optimization_information(self):
        canonical_parameters = self.canonical_model.get_parameters_with_optimization_information()
        canonical_parameters = self._add_tags(canonical_parameters, ["canonical"])

        if not self.use_deformations:
            deformation_parameters = []
        else:
            deformation_parameters = (
                self.deformation_model.get_parameters_with_optimization_information()
            )
            deformation_parameters = self._add_tags(deformation_parameters, ["deformation"])

        return canonical_parameters + deformation_parameters

    def get_regularization_losses(self):

        regularization_losses = {}

        regularization_losses["canonical_model"] = self.canonical_model.get_regularization_losses()

        if self.use_deformations:
            regularization_losses[
                "deformation_model"
            ] = self.deformation_model.get_regularization_losses()

        return regularization_losses
