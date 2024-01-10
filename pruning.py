# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import logging
import os

import numpy as np
import torch
from binary_dataset import BinaryDataset
from utils import szudzik

LOGGER = logging.getLogger(__name__)


def get_pruning_key(voxel_grid_size, timestep):
    key = szudzik(voxel_grid_size, timestep)
    return key


class Pruning(torch.nn.Module):
    def __init__(self, settings, data_handler):
        super().__init__()

        self.do_use_pruning = settings.use_pruning
        self.voxel_grid_size = settings.voxel_grid_size

        self.no_pruning_probability = settings.no_pruning_probability

        self.voxel_grid = None  # register_buffer("voxel_grid", torch.empty((voxel_grid_size, voxel_grid_size, voxel_grid_size), dtype=torch.bool)) # registering as buffer would store this in the checkpoint, which just wastes disk space

        self.current_timestep = None

        if self.do_use_pruning:
            self._dataset = BinaryDataset(
                data_handler.data_loader.get_dataset_folder(),
                name="foreground_voxel_grids",
                read_only=True,
            )

    def forward(self, positions, is_training):
        # positions are normalized to unit cube
        # positions: num_rays x num_points_per_ray x 3

        input_shape = positions.shape[
            :-1
        ]  # num_rays x num_points_per_ray or num_rays * num_points_per_ray

        if (
            not self.do_use_pruning
            or self.voxel_grid is None
            or (torch.rand(1) < self.no_pruning_probability and is_training)
        ):
            mask = torch.ones(input_shape, device=positions.device, dtype=torch.bool)
            return mask

        voxel_indices = positions.view(-1, 3)
        voxel_indices = torch.floor(
            voxel_indices * self.voxel_grid_size
        ).long()  # num_rays * num_points_per_ray x 3
        mask = self.voxel_grid[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]]
        mask = mask.view(input_shape)

        return mask

    def load_voxel_grid(self, timestep):

        self.current_timestep = timestep

        if timestep is None:
            self.voxel_grid = None
            return

        if timestep == "all":
            key = "all"
        else:
            key = get_pruning_key(self.voxel_grid_size, float(timestep))
            if key not in self._dataset:
                LOGGER.warning(
                    "undesirable flow. got wrong timestep, fall back to 'all' voxel grid pruning"
                )
                key = "all"

        from io import BytesIO

        voxel_grid_bytes = BytesIO(self._dataset.get_entry(key))
        voxel_grid_bytes.seek(0)
        try:
            voxel_grid = np.load(voxel_grid_bytes)
        except Exception as exception:
            LOGGER.exception(
                "failed to load pruning at time: " + str(timestep) + " " + str(self.voxel_grid_size)
            )
            raise exception
        self.voxel_grid = torch.from_numpy(voxel_grid["foreground_voxel_grid"]).cuda()

    def get_parameters_with_optimization_information(self):
        return []

    def get_regularization_losses(self):
        return {}
