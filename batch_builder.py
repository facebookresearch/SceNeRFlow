# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import logging

import numpy as np
import torch

LOGGER = logging.getLogger(__name__)


class BatchBuilder:
    def __init__(self, settings, ray_builder):

        self.ray_builder = ray_builder

        self.do_vignetting_correction = settings.do_vignetting_correction
        self.do_ngp_mip_nerf = settings.do_ngp_mip_nerf
        self.debug = settings.debug

    def build(self, batch_size=None, active_imageids=None, precomputed=None, single_image=None):

        batch = {}

        if single_image is not None:
            # this assumes that a single full image is requested
            # single_image: extrin, intrin, timestep, intrinid, (background)

            rays_dict = self.ray_builder.build(
                single_image["extrin"], single_image["intrin"]
            )  # cuda
            rays_origin = rays_dict["rays_origin"].view(-1, 3)  # H * W x 3
            rays_dir = rays_dict["rays_dir"].view(-1, 3)

            num_rays = rays_origin.shape[0]
            timesteps = single_image["timestep"].repeat(num_rays)
            near = torch.tensor(single_image["extrin"]["near"]).repeat(num_rays)
            far = torch.tensor(single_image["extrin"]["far"]).repeat(num_rays)
            intrinids = torch.tensor(single_image["intrin"]["intrinid"]).repeat(num_rays)
            if "background" in single_image:
                batch["background"] = single_image["background"].view(-1, 3)

            if self.do_vignetting_correction or self.do_ngp_mip_nerf:
                width = single_image["intrin"]["width"]
                height = single_image["intrin"]["height"]
                y_coordinates = torch.arange(height).repeat_interleave(
                    width
                )  # [0,0,0,1,1,1,2,2,2,3,3,3]
                x_coordinates = torch.arange(width).repeat(height)  # [0,1,2,0,1,2,0,1,2,0,1,2]

                x_center = torch.tensor(single_image["intrin"]["center_x"])
                y_center = torch.tensor(single_image["intrin"]["center_y"])

            if self.do_ngp_mip_nerf:
                x_center = x_center.repeat(num_rays)
                y_center = y_center.repeat(num_rays)
                x_focal = torch.tensor(single_image["intrin"]["focal_x"]).repeat(num_rays)
                y_focal = torch.tensor(single_image["intrin"]["focal_y"]).repeat(num_rays)

                batch["rotation"] = single_image["extrin"]["rotation"].repeat(
                    num_rays, 1, 1
                )  # num_rays x 3 x 3

        elif precomputed is not None:

            num_images = precomputed["rays_origin"].shape[0]
            if active_imageids is None:
                active_imageids = torch.arange(num_images)

            def flatten_pixel_dimensions(tensor):
                # turns N x H x W x F into N x H * W x F
                return tensor.view(tensor.shape[0], -1, tensor.shape[-1])

            rgb = flatten_pixel_dimensions(precomputed["rgb"])
            rays_origin = flatten_pixel_dimensions(precomputed["rays_origin"])
            rays_dir = flatten_pixel_dimensions(precomputed["rays_dir"])

            num_rays_per_image = rgb.shape[1]
            if batch_size is None:
                # used by scheduler and state_loader_saver, which use a named subset as batch. see data_handler.
                assert num_images == len(active_imageids)
                image_indices = torch.arange(num_images).repeat_interleave(
                    num_rays_per_image
                )  # [0,0,0,1,1,1,2,2,2,3,3,3]
                flattened_indices = torch.arange(num_rays_per_image).repeat(
                    num_images
                )  # [0,1,2,0,1,2,0,1,2,0,1,2]
            else:
                # standard training batch
                image_indices = torch.randint(
                    len(active_imageids), size=(batch_size,)
                )  # among active training images
                flattened_indices = torch.randint(num_rays_per_image, size=(batch_size,))

            all_train_image_indices = active_imageids[image_indices]  # among all training images
            image_indices = precomputed["train_to_loaded_train_ids"][
                all_train_image_indices
            ]  # among loaded training images
            if self.debug and torch.any(image_indices == -1):
                raise AssertionError("mapping is broken")

            rgb = rgb[image_indices, flattened_indices]
            batch["rgb"] = rgb

            rays_origin = rays_origin[image_indices, flattened_indices]
            rays_dir = rays_dir[image_indices, flattened_indices]
            timesteps = precomputed["timesteps"][image_indices]
            near = precomputed["near"][image_indices]
            far = precomputed["far"][image_indices]
            intrinids = precomputed["intrinids"][image_indices]

            if "background" in precomputed or self.do_vignetting_correction or self.do_ngp_mip_nerf:
                if "coordinate_subsets" in precomputed:
                    # not using full images
                    yx_coordinates = precomputed["coordinate_subsets"][
                        image_indices, flattened_indices
                    ]
                    y_coordinates = yx_coordinates[:, 0]
                    x_coordinates = yx_coordinates[:, 1]
                    if self.do_vignetting_correction:
                        width = precomputed["intrins"]["width"][0]
                else:
                    # when using a standard training batch and data_handler loaded full images.
                    width = precomputed["background"].shape[2]
                    y_coordinates = torch.div(flattened_indices, width, rounding_mode="floor")
                    x_coordinates = flattened_indices % width

            if self.do_vignetting_correction or self.do_ngp_mip_nerf:
                x_center = precomputed["intrins"]["center_x"][intrinids]
                y_center = precomputed["intrins"]["center_y"][intrinids]

            if "background" in precomputed:
                exintrinids = precomputed["exintrinids"][image_indices]
                batch["background"] = precomputed["background"][
                    exintrinids, y_coordinates, x_coordinates
                ]

            if self.do_ngp_mip_nerf:
                x_focal = precomputed["intrins"]["focal_x"][intrinids]
                y_focal = precomputed["intrins"]["focal_y"][intrinids]
                batch["rotation"] = precomputed["extrins"]["rotation"][
                    image_indices, :, :
                ]  # num_rays x 3 x 3

            batch["image_indices"] = all_train_image_indices

        if self.do_vignetting_correction:
            batch["normalized_x_coordinate"] = (x_coordinates - x_center) / width
            batch["normalized_y_coordinate"] = (
                y_coordinates - y_center
            ) / width  # same normalization as x

        if self.do_ngp_mip_nerf:
            batch["x_coordinate"] = x_coordinates
            batch["y_coordinate"] = y_coordinates
            batch["x_center"] = x_center
            batch["y_center"] = y_center
            batch["x_focal"] = x_focal
            batch["y_focal"] = y_focal

        batch.update(
            {
                "rays_origin": rays_origin,
                "rays_dir": rays_dir,
                "timesteps": timesteps,
                "near": near,
                "far": far,
                "intrinids": intrinids,
            }
        )

        batch = {key: tensor.cuda() for key, tensor in batch.items()}

        return batch
