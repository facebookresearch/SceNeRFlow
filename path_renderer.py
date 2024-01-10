# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import logging
import os

import imageio
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


class PathRenderer:
    def __init__(self, data_handler, rank, world_size):
        self.batch_builder = data_handler.batch_builder

        self.rank = rank
        self.world_size = world_size

    def render(
        self,
        extrins,
        intrins,
        timesteps,
        scene,
        renderer,
        backgrounds=None,
        points_per_ray=None,
        reduce_memory_for_correspondences=False,
        returns=None,
        hacky_checkpoint_loading=None,
        **kwargs
    ):

        if returns is None:
            returns = Returns(restricted=["corrected_rgb"])

        for counter in trange(len(extrins)):

            returns.activate_mode(counter)

            if counter % self.world_size != self.rank:
                if self.rank == 0:
                    multi_gpu_receive_returns_from_rank_pathrenderer(
                        self.rank, self.world_size, counter, returns
                    )
                continue

            extrin = extrins[counter]
            intrin = intrins[counter]
            timestep = timesteps[counter]
            if backgrounds is not None:
                background = backgrounds[counter]

            float_timestep = float(timestep.numpy())

            if hacky_checkpoint_loading is not None:
                hacky_checkpoint_loading.load(float_timestep)

            uses_pruning = (
                scene.pruning.do_use_pruning and scene.pruning.current_timestep != timestep
            )
            if uses_pruning:
                original_timestep = scene.pruning.current_timestep
                scene.pruning.load_voxel_grid(
                    float_timestep if "only_canonical" not in kwargs else "all"
                )

            single_image = {
                "extrin": extrin,
                "intrin": intrin,
                "timestep": timestep,
            }
            if backgrounds is not None:
                single_image["background"] = background

            with torch.no_grad():

                batch = self.batch_builder.build(single_image=single_image)

                rendering_generator_wrapper = renderer.render(
                    batch=batch,
                    scene=scene,
                    points_per_ray=points_per_ray,
                    returns=returns,
                    pull_to_cpu=True,
                    iterator_yield=True,
                    **kwargs
                )

                for subreturns in rendering_generator_wrapper():

                    if (
                        reduce_memory_for_correspondences
                        and "deformed_positions" in subreturns
                        and "weights" in subreturns
                    ):
                        correspondences_rgb = self._visualize_correspondences(
                            subreturns,
                            number_of_small_rgb_voxels=kwargs["number_of_small_rgb_voxels"]
                            if "number_of_small_rgb_voxels" in kwargs
                            else 30,
                            blend_with_corrected_rgb=kwargs["blend_with_corrected_rgb"]
                            if "blend_with_corrected_rgb" in kwargs
                            else 0.0,
                        )
                        subreturns.delete_return("deformed_positions")
                        subreturns.delete_return("weights")
                        subreturns.add_return("correspondences_rgb", correspondences_rgb)

                returns.add_returns(subreturns.concatenate_returns())

            returns.reshape_returns(width=intrin["width"], height=intrin["height"])

            if uses_pruning:
                scene.pruning.load_voxel_grid(original_timestep)

            if self.rank != 0:
                multi_gpu_send_returns_to_rank_pathrenderer(target_rank=0, returns=returns)

        return returns.get_returns()["corrected_rgb"], returns

    def _visualize_correspondences(
        self,
        returns,
        number_of_small_rgb_voxels=30,
        background_threshold=0.4,
        blend_with_corrected_rgb=0.0,
    ):

        device = "cpu"  # self.rank

        deformed_positions = returns.get_returns()["deformed_positions"].to(
            device
        )  # num_rays x num_points x 3
        weights = returns.get_returns()["weights"].to(device)  # num_rays x num_points

        # visibility_weight is the weight of the influence that each sample has on the final rgb value. so they sum to at most 1.
        accumulated_visibility = torch.cumsum(weights, dim=-1)  # num_rays x num_points
        background_mask = accumulated_visibility[:, -1] < background_threshold  # num_rays
        median_indices = torch.min(torch.abs(accumulated_visibility - 0.5), dim=-1)[
            1
        ]  # num_rays. visibility goes from 0 to 1. 0.5 is the median, so treat it as "most likely to be on the actually visible surface"
        num_rays = median_indices.shape[0]
        # median_indices contains the index of one ray sample for each pixel.
        # this ray sample is selected in this line of code.
        surface_pixels = deformed_positions[
            torch.arange(num_rays, device=device), median_indices, :
        ]  # num_rays x 3
        correspondences_rgb = surface_pixels

        # break the canonical space into smaller voxels.
        # each voxel covers the entire RGB space [0,1]^3.
        # makes it easier to visualize small changes. leads to a 3D checkerboard pattern.
        if number_of_small_rgb_voxels > 1:
            correspondences_rgb *= number_of_small_rgb_voxels
            correspondences_rgb = correspondences_rgb - correspondences_rgb.long()

        # correspondences_rgb[background_mask] = 0.0

        corrected_rgb = returns.get_returns()["corrected_rgb"].to(device)  # num_rays
        correspondences_rgb = (
            1.0 - blend_with_corrected_rgb
        ) * correspondences_rgb + blend_with_corrected_rgb * corrected_rgb
        correspondences_rgb[background_mask] = corrected_rgb[background_mask]  # modified

        z_vals = returns.get_returns()["z_vals"]
        depth = z_vals[torch.arange(num_rays, device=device), median_indices]
        depth = torch.clamp(depth, min=0.0, max=1.0)
        depth[background_mask] = 1.0
        returns.delete_return("depth")
        returns.add_return("depth", depth)
        min_normalized_depth = 0.1
        disparity = 1.0 / torch.max(min_normalized_depth * torch.ones_like(depth), depth)
        disparity *= min_normalized_depth
        returns.delete_return("disparity")
        returns.add_return("disparity", disparity)

        return correspondences_rgb.cpu()

    def render_and_store(
        self,
        state_loader_saver,
        output_name,
        returns=None,
        rgb=None,
        visualize_correspondences=True,
        reduce_memory_for_correspondences=True,
        also_store_images=False,
        output_folder=None,
        hacky_checkpoint_loading=None,
        only_render_if_file_does_not_exist=True,
        **kwargs
    ):

        if output_folder is None and self.rank == 0:
            output_folder = os.path.join(state_loader_saver.get_results_folder(), "0_renderings")
            state_loader_saver.create_folder(output_folder)

        if only_render_if_file_does_not_exist:
            check_output_file = os.path.join(output_folder, output_name + "_rgb.mp4")
            if os.path.exists(check_output_file):
                LOGGER.info("already rendered. will not render again: " + output_name)
                return

        if returns is None:
            if visualize_correspondences:
                correspondences = ["deformed_positions", "weights", "correspondences_rgb"]
                # deformed_positions and weights get deleted if reduce_memory_for_correspondences==True
            else:
                correspondences = []
            returns = Returns(
                restricted=["corrected_rgb", "disparity", "depth", "z_vals"] + correspondences
            )
        else:
            reduce_memory_for_correspondences = False

        _, returns = self.render(
            returns=returns,
            reduce_memory_for_correspondences=reduce_memory_for_correspondences,
            hacky_checkpoint_loading=hacky_checkpoint_loading,
            **kwargs
        )

        if self.rank != 0:
            return

        def store(output_file, images, fps=30, quality=10):
            imageio.mimwrite(output_file, images, fps=fps, quality=quality)
            if also_store_images:
                for counter, image in enumerate(images):
                    imageio.imsave(
                        output_file + "_" + str(counter).zfill(5) + ".jpg", image, quality=100
                    )

        def saveable(tensor):
            try:
                tensor = tensor.numpy()
            except Exception:
                pass
            return (255 * np.clip(tensor, 0, 1)).astype(np.uint8)

        def jet_color_scheme(tensor):
            # tensor: values in [0,1]
            from matplotlib import cm

            tensor = cm.jet(saveable(tensor))[:, :, :, :3]
            return tensor  # values in [0,1]

        def stack_images_for_video(name):
            stacked = []
            for counter in returns.get_modes():
                returns.activate_mode(counter)
                this_result = returns.get_returns()[name]
                stacked.append(this_result)
            return torch.stack(stacked, dim=0)

        # rgb
        if "corrected_rgb" in returns.get_returns():
            corrected_rgb = stack_images_for_video("corrected_rgb")

            output_file = os.path.join(output_folder, output_name + "_rgb.mp4")
            store(output_file, saveable(corrected_rgb))

            if rgb is not None:
                # save groundtruth only once
                save_groundtruth = True
                # save_groundtruth = not any("_rgb_gt" in file for file in os.listdir(output_folder))
                if save_groundtruth:
                    output_file = os.path.join(output_folder, output_name + "_rgb_gt.mp4")
                    store(output_file, saveable(rgb))

                error_map = np.linalg.norm(rgb - corrected_rgb, axis=-1) / np.sqrt(3)
                error_map = np.clip(
                    error_map / 0.10, 0.0, 1.0
                )  # emphasize small errors, clip at 10% max error
                error_map = jet_color_scheme(error_map)
                output_file = os.path.join(output_folder, output_name + "_error.mp4")
                store(output_file, saveable(error_map), quality=6)

        # depth
        if "depth" in returns.get_returns():
            depth = stack_images_for_video("depth")
            depth = depth / torch.max(depth)
            output_file = os.path.join(output_folder, output_name + "_depth.mp4")
            store(output_file, saveable(depth))

        # disparity
        if "disparity" in returns.get_returns():
            disparity = stack_images_for_video("disparity")
            disparity = jet_color_scheme(disparity)
            output_file = os.path.join(output_folder, output_name + "_disparity.mp4")
            store(output_file, saveable(disparity))

        # correspondences
        if "correspondences_rgb" in returns.get_returns():
            correspondences_rgb = stack_images_for_video("correspondences_rgb")
            output_file = os.path.join(output_folder, output_name + "_correspondences_rgb.mp4")
            store(output_file, saveable(correspondences_rgb), quality=5)

        if hacky_checkpoint_loading is not None:
            raise AssertionError  # make sure that hacky_checkpoint_loading is only used intentionally, so throw an error that needs to be caught
