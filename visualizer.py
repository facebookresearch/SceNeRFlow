# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import logging
import os

import numpy as np
import torch
from multi_gpu import (
    multi_gpu_receive_returns_from_rank_pathrenderer,
    multi_gpu_send_returns_to_rank_pathrenderer,
)
from tqdm import trange
from utils import Returns

logging.getLogger("matplotlib").setLevel(logging.WARNING)
LOGGER = logging.getLogger(__name__)


class Visualizer:
    def __init__(self, settings, data_handler, rank, world_size):
        self.batch_builder = data_handler.batch_builder

        self.smooth_deformations_type = settings.smooth_deformations_type
        self.weight_coarse_smooth_deformations = settings.weight_coarse_smooth_deformations

        self.points_per_chunk = settings.points_per_chunk // 8
        self.default_num_points_per_ray = settings.num_points_per_ray

        self.rank = rank
        self.world_size = world_size

    def render(
        self,
        only_first_frame,
        extrins,
        intrins,
        timesteps,
        scene,
        renderer,
        backgrounds=None,
        points_per_ray=None,
        **kwargs
    ):

        if points_per_ray is None:
            points_per_ray = self.default_num_points_per_ray

        all_relevant_results = []
        for counter in trange(len(extrins)):

            relevant_results = {}

            extrin = extrins[counter]
            intrin = intrins[counter]
            timestep = timesteps[counter]
            if backgrounds is not None:
                background = backgrounds[counter]

            single_image = {
                "extrin": extrin,
                "intrin": intrin,
                "timestep": timestep,
            }
            if backgrounds is not None:
                single_image["background"] = background

            batch = self.batch_builder.build(single_image=single_image)
            num_rays = batch["rays_origin"].shape[0]
            rays_per_chunk = max(1, self.points_per_chunk // points_per_ray)

            for chunk_start in range(0, num_rays, rays_per_chunk):

                # render

                returns = Returns()  # dummy
                returns.activate_mode("coarse")
                subreturns = Returns()

                subbatch = {
                    key: tensor[chunk_start : chunk_start + rays_per_chunk]
                    for key, tensor in batch.items()
                }

                renderer.render(
                    batch=subbatch,
                    scene=scene,
                    points_per_ray=points_per_ray,
                    returns=returns,
                    subreturns=subreturns,
                )

                del returns

                # compute losses

                if self.smooth_deformations_type not in [
                    "finite",
                    "jacobian",
                    "divergence",
                    "nerfies",
                ]:
                    raise NotImplementedError

                for mode in subreturns.get_modes():

                    def wrapper_to_free_memory_quickly(relevant_results, mode, subreturns):
                        subreturns.activate_mode(mode)
                        position_offsets = subreturns.get_returns()["coarse_position_offsets"]
                        normalized_undeformed_positions = subreturns.get_returns()[
                            "normalized_undeformed_positions"
                        ]

                        if "coarse_position_offsets" not in relevant_results:
                            relevant_results["coarse_position_offsets"] = []
                        relevant_results["coarse_position_offsets"].append(
                            position_offsets.detach().cpu()
                        )
                        if "normalized_undeformed_positions" not in relevant_results:
                            relevant_results["normalized_undeformed_positions"] = []
                        relevant_results["normalized_undeformed_positions"].append(
                            normalized_undeformed_positions.detach().cpu()
                        )
                        if "opacity" not in relevant_results:
                            relevant_results["opacity"] = []
                        relevant_results["opacity"].append(
                            subreturns.get_returns()["alpha"].detach().cpu()
                        )

                        if self.smooth_deformations_type in ["jacobian", "nerfies"]:
                            from utils import get_minibatch_jacobian

                            jacobian = get_minibatch_jacobian(
                                position_offsets, normalized_undeformed_positions
                            )  # num_points x 3 x 3
                            if self.smooth_deformations_type == "jacobian_broken":
                                this_loss = jacobian**2  # if using position_offsets
                                this_loss = this_loss.view(position_offsets.shape[:-1] + (-1,))
                                this_loss = this_loss.mean(
                                    dim=-1
                                )  # num_rays_in_chunk x num_points_per_ray
                                eps = 1e-6
                                this_loss = torch.sqrt(this_loss + eps)
                            elif self.smooth_deformations_type == "jacobian":
                                R_times_Rt = torch.matmul(
                                    jacobian, torch.transpose(jacobian, -1, -2)
                                )
                                this_loss = torch.abs(
                                    R_times_Rt - torch.eye(3, device=jacobian.device).view(-1, 3, 3)
                                )
                                this_loss = this_loss.view(position_offsets.shape[:-1] + (-1,))
                                this_loss = this_loss.mean(
                                    dim=-1
                                )  # num_rays_in_chunk x num_points_per_ray
                            else:
                                singular_values = torch.linalg.svdvals(jacobian)  # num_points x 3
                                eps = 1e-6
                                stable_singular_values = torch.maximum(
                                    singular_values, eps * torch.ones_like(singular_values)
                                )
                                log_singular_values = torch.log(stable_singular_values)
                                this_loss = torch.mean(
                                    log_singular_values**2, dim=-1
                                )  # num_points

                        else:  # divergence
                            from utils import divergence_exact, divergence_approx

                            exact = False
                            divergence_fn = divergence_exact if exact else divergence_approx
                            divergence = divergence_fn(
                                inputs=normalized_undeformed_positions, outputs=position_offsets
                            )
                            this_loss = torch.abs(
                                divergence
                            )  # num_rays_in_chunk x num_points_per_ray

                        weigh_by_opacity = False
                        if weigh_by_opacity:
                            opacity = subreturns.get_returns()["alpha"]
                            max_windowed = True
                            if max_windowed:
                                window_fraction = 0.01
                                points_per_ray = opacity.shape[1]
                                kernel_size = max(1, int(window_fraction * points_per_ray))
                                if kernel_size % 2 == 0:
                                    kernel_size += 1  # needed for integer padding
                                padding = (kernel_size - 1) // 2
                                opacity = torch.nn.functional.max_pool1d(
                                    opacity,
                                    kernel_size=kernel_size,
                                    stride=1,
                                    padding=padding,
                                    dilation=1,
                                    ceil_mode=True,
                                    return_indices=False,
                                )
                            this_loss = opacity.detach() * this_loss

                        only_for_large_deformations = 0.0001
                        if only_for_large_deformations is not None:
                            offset_magnitude = torch.linalg.norm(position_offsets.detach(), dim=-1)
                            mode = "sigmoid"
                            if mode == "binary":
                                offset_weights = offset_magnitude > only_for_large_deformations
                            elif mode == "sigmoid":
                                offset_weights = torch.sigmoid(
                                    (4.0 * offset_magnitude / only_for_large_deformations) - 2.0
                                )
                            this_loss = this_loss * offset_weights.detach()

                        if "smooth_deformations_loss" not in relevant_results:
                            relevant_results["smooth_deformations_loss"] = []
                        relevant_results["smooth_deformations_loss"].append(
                            this_loss.detach().cpu()
                        )

                    wrapper_to_free_memory_quickly(relevant_results, mode, subreturns)

                del subreturns

            for key in relevant_results.keys():
                relevant_results[key] = torch.cat(relevant_results[key], dim=0)

            all_relevant_results.append(relevant_results)

            if only_first_frame:
                break

        return all_relevant_results

    def render_and_store(self, state_loader_saver, output_name, only_first_frame=True, **kwargs):

        if self.rank != 0:
            return

        all_relevant_results = self.render(only_first_frame=only_first_frame, **kwargs)

        output_folder = os.path.join(state_loader_saver.get_results_folder(), "3_visualization")
        state_loader_saver.create_folder(output_folder)

        for counter, relevant_results in enumerate(all_relevant_results):

            loss = relevant_results["smooth_deformations_loss"]
            opacity = relevant_results["opacity"]
            undeformed_positions = relevant_results["normalized_undeformed_positions"]
            offsets = relevant_results["coarse_position_offsets"]

            # flatten
            loss = loss.reshape(-1)
            opacity = opacity.reshape(-1)
            undeformed_positions = undeformed_positions.reshape(-1, 3)
            offsets = offsets.reshape(-1, 3)

            # opacity filtering
            use_opacity_thresholding = False
            opacity_threshold = 0.01
            if use_opacity_thresholding:
                mask = opacity > opacity_threshold
                loss = loss[mask]
                opacity = opacity[mask]
                undeformed_positions = undeformed_positions[mask, :]
                offsets = offsets[mask, :]

            # weigh the loss
            weigh_by_loss_weight = False
            if weigh_by_loss_weight:
                loss_weight = self.weight_coarse_smooth_deformations
                loss = loss_weight * loss

            # random subsampling
            only_keep_n_points = 10000
            if only_keep_n_points is not None:
                random_indices = torch.randperm(n=loss.shape[0])[:only_keep_n_points]
                loss = loss[random_indices]
                opacity = opacity[random_indices]
                undeformed_positions = undeformed_positions[random_indices, :]
                offsets = offsets[random_indices, :]

            # convert loss to color
            max_loss = 10.0
            loss[loss > max_loss] = max_loss
            loss = loss / max_loss
            loss = (255 * loss.numpy()).astype(np.uint8)
            from matplotlib import cm

            colors = cm.jet(loss)[:, :3]  # num_points x 3

            # mesh generation
            mesh_lines = []
            for (x, y, z), (dx, dy, dz), (r, g, b) in zip(
                undeformed_positions.numpy(), offsets.numpy(), colors
            ):
                mesh_lines.append(
                    "v "
                    + str(x)
                    + " "
                    + str(y)
                    + " "
                    + str(z)
                    + " "
                    + str(r)
                    + " "
                    + str(g)
                    + " "
                    + str(b)
                )
                mesh_lines.append(
                    "v "
                    + str(x + 0.000001)
                    + " "
                    + str(y)
                    + " "
                    + str(z)
                    + " "
                    + str(r)
                    + " "
                    + str(g)
                    + " "
                    + str(b)
                )
                mesh_lines.append(
                    "v "
                    + str(x + dx)
                    + " "
                    + str(y + dy)
                    + " "
                    + str(z + dz)
                    + " "
                    + str(r)
                    + " "
                    + str(g)
                    + " "
                    + str(b)
                )

            for i in range(offsets.shape[0]):
                # faces are 1-indexed
                mesh_lines.append(
                    "f " + str(1 + 3 * i) + " " + str(1 + 3 * i + 1) + " " + str(1 + 3 * i + 2)
                )

            mesh_lines = "\n".join(mesh_lines)

            # write out mesh
            with open(
                os.path.join(output_folder, output_name + "_" + str(counter).zfill(6)) + ".obj", "w"
            ) as mesh_file:
                mesh_file.write(mesh_lines)
