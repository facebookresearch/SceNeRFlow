# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import logging

import numpy as np
import torch
from batch_builder import BatchBuilder
from batch_composer import BatchComposer
from data_loader import get_data_loader
from ray_builder import RayBuilder

LOGGER = logging.getLogger(__name__)


class DataHandler:
    def __init__(self, settings, rank=0, world_size=1, precomputation_mode=False):

        self.rank = rank
        self.world_size = world_size
        self.optimization_mode = settings.optimization_mode
        self.do_ngp_mip_nerf = settings.do_ngp_mip_nerf

        self.data_loader = get_data_loader(settings, self.rank, self.world_size)

        self.ray_builder = RayBuilder(rank=self.rank, multi_gpu=settings.multi_gpu)
        self.ray_builder.use_precomputed_dataset(
            self.data_loader.get_dataset_folder(), create=precomputation_mode
        )

        self.batch_builder = BatchBuilder(settings, self.ray_builder)
        self.batch_composer = BatchComposer(self.batch_builder)

    def get_batch(self, batch_size=None, subset_name=None):
        if subset_name is None:
            return self.batch_composer.compose(
                batch_size, precomputed=self.get_precomputed(subset_name="main")
            )
        else:
            # used by scheduler and state_loader_saver
            return self.batch_builder.build(precomputed=self.get_precomputed(subset_name))

    def get_timeline_range(self):
        return self.data_loader.get_timeline_range()

    def get_training_set_size(self):
        return len(self.data_loader.get_train_imageids())

    def get_train_imageids(self):
        return self.data_loader.get_train_imageids()

    def get_precomputed(self, subset_name=None):

        if subset_name is None:
            subset_name = "main"

        needs_special_handling = ["rgb", "rays_origin", "rays_dir", "coordinate_subsets"]

        precomputed = {
            key: value
            for key, value in self.precomputed.items()
            if key not in needs_special_handling
        }

        for key in needs_special_handling:
            value = self.precomputed[key][subset_name]
            # remove the key "coordinate_subsets" if we use all pixels.
            # this serves as a signal for functions that they are getting all pixels.
            if value is not None:
                precomputed[key] = value

        return precomputed

    def load_training_set(
        self,
        factor,
        num_total_rays_to_precompute=None,
        num_pixels_per_image=None,
        foreground_focused=False,
        imageids=None,
        also_load_top_left_corner_and_four_courners=False,
    ):

        self.precomputed = {}  # pytorch cpu

        training_imageids = torch.from_numpy(self.data_loader.get_train_imageids())
        training_imageids = training_imageids.cuda()
        if imageids is None:
            imageids = training_imageids.clone()

        # indexing for batch_builder (all training images -> loaded training images)
        # imageids: loaded training images -> all images
        # training_imageids: all training images -> all images
        num_total_images = self.data_loader.num_total_images()
        all_to_loaded_train = torch.zeros(num_total_images, dtype=np.long) - 1
        all_to_loaded_train[imageids] = torch.arange(len(imageids))
        train_to_loaded_train = all_to_loaded_train[training_imageids.cpu()]
        self.precomputed["train_to_loaded_train_ids"] = train_to_loaded_train

        # background
        if self.data_loader.has_background_images():
            LOGGER.info("loading background images...")
            self.precomputed["background"] = torch.from_numpy(
                self.data_loader.load_background_images(factor=factor)
            )
            self.precomputed["exintrinids"] = self.data_loader.get_exintrinids(imageids).cpu()

        # decide which pixels to load
        if num_total_rays_to_precompute is not None:
            if num_pixels_per_image is not None:
                raise RuntimeError(
                    "only provide one of num_total_rays_to_precompute or num_pixels_per_image"
                )
            num_pixels_per_image = max(1, num_total_rays_to_precompute // len(imageids))

        if foreground_focused:
            desired_subsets = {
                "main": {
                    "mode": "foreground_focused",
                    "foreground_fraction": 0.8,
                    "num_pixels_per_image": num_pixels_per_image,
                }
            }
        else:
            if num_pixels_per_image is None:
                desired_subsets = {"main": {"mode": "all"}}
            else:
                desired_subsets = {
                    "main": {"mode": "random", "num_pixels_per_image": num_pixels_per_image}
                }

        if also_load_top_left_corner_and_four_courners:
            desired_subsets["top_left_corner"] = {
                "mode": "specific",
                "y_coordinates": [0],
                "x_coordinates": [0],
            }
            desired_subsets["four_corners"] = {
                "mode": "specific",
                "y_coordinates": [0, 0, -1, -1],
                "x_coordinates": [0, -1, 0, -1],
            }

        # rgb
        LOGGER.info("loading training rgb...")
        rgb_subsets, coordinate_subsets = self.data_loader.load_images(
            factor=factor,
            imageids=imageids,
            desired_subsets=desired_subsets,
            background_images_dict=self.precomputed,
        )
        self.precomputed["rgb"] = {
            subset_name: torch.from_numpy(rgb) for subset_name, rgb in rgb_subsets.items()
        }
        self.precomputed["coordinate_subsets"] = coordinate_subsets

        # rays_origin, rays_dir
        LOGGER.info("loading training rays...")

        extrinids = self.data_loader.get_extrinids(imageids)
        extrins = self.data_loader.get_extrinsics(extrinids=extrinids)
        if self.do_ngp_mip_nerf:
            self.precomputed["extrins"] = {}
            self.precomputed["extrins"]["rotation"] = torch.stack(
                [extrin["rotation"] for extrin in extrins], dim=0
            )

        intrinids = self.data_loader.get_intrinids(imageids)
        self.precomputed["intrinids"] = intrinids.cpu()
        all_intrins = self.data_loader.get_intrinsics(factor=factor)
        self.precomputed["intrins"] = {}
        for key in all_intrins[0].keys():
            if key in ["distortion"]:
                continue
            values = torch.from_numpy(np.array([intrin[key] for intrin in all_intrins]))
            if values.dtype == torch.float64:
                values = values.float()
            self.precomputed["intrins"][key] = values
        image_intrins = self.data_loader.get_intrinsics(intrinids=intrinids, factor=factor)

        rays_dict = self.ray_builder.build_multiple(
            extrins, image_intrins, coordinate_subsets=coordinate_subsets
        )
        self.precomputed["rays_origin"] = rays_dict["rays_origin"]
        self.precomputed["rays_dir"] = rays_dict["rays_dir"]

        # timesteps
        self.precomputed["timesteps"] = self.data_loader.get_timesteps(imageids).cpu()

        # near, far
        self.precomputed["near"] = torch.from_numpy(
            np.array([extrin["near"] for extrin in extrins], dtype=np.float32)
        )
        self.precomputed["far"] = torch.from_numpy(
            np.array([extrin["far"] for extrin in extrins], dtype=np.float32)
        )

    def get_test_cameras_for_rendering(self, factor=None):

        imageids = self.data_loader.get_test_imageids()

        only_render_current_timestep = self.optimization_mode == "per_timestep"
        if only_render_current_timestep:
            timesteps = self.data_loader.get_timesteps(imageids).cpu()
            imageids_with_right_timestep = []
            right_timesteps = list(set(self.precomputed["timesteps"].numpy()))
            for imageid, timestep in zip(imageids, timesteps):
                if timestep in right_timesteps:
                    imageids_with_right_timestep.append(imageid)
            imageids = np.array(imageids_with_right_timestep)

            # sort by (extrinid, timestep) such that renderings can be concatenated easily
            extrinids = self.data_loader.get_extrinids(imageids).cpu()
            timesteps = self.data_loader.get_timesteps(imageids).cpu()
            image_info = [
                (int(extrinid), float(timestep), int(imageid))
                for extrinid, timestep, imageid in zip(extrinids, timesteps, imageids)
            ]
            image_info = sorted(image_info, key=lambda x: (x[0], x[1]))
            imageids = np.array([image[2] for image in image_info])

            LOGGER.debug("test set: " + str(imageids) + " " + str(right_timesteps))

        test_cameras = {}

        test_cameras["timesteps"] = self.data_loader.get_timesteps(imageids).cpu()

        extrinids = self.data_loader.get_extrinids(imageids)
        test_cameras["extrins"] = self.data_loader.get_extrinsics(extrinids=extrinids)

        intrinids = self.data_loader.get_intrinids(imageids)
        test_cameras["intrins"] = self.data_loader.get_intrinsics(
            intrinids=intrinids, factor=factor
        )

        test_cameras["rgb"] = torch.from_numpy(
            self.data_loader.load_images(factor=factor, imageids=imageids)[0]["everything"]
        )

        if self.data_loader.has_background_images():
            exintrinids = self.data_loader.get_exintrinids(imageids)
            test_cameras["backgrounds"] = torch.from_numpy(
                self.data_loader.load_background_images(factor=factor, exintrinids=exintrinids)
            )

        return test_cameras

    def visualize_images_in_3D(self, results_folder):
        # results_folder = state_loader_saver.get_results_folder()
        origin = self.precomputed["rays_origin"][:, 0, 0]  # N x 3
        top_left = self.precomputed["rays_dir"][:, 0, 0]  # N x 3
        top_right = self.precomputed["rays_dir"][:, 0, -1]  # N x 3
        bottom_right = self.precomputed["rays_dir"][:, -1, -1]  # N x 3
        bottom_left = self.precomputed["rays_dir"][:, -1, 0]  # N x 3

        rescale_factor = 0.5

        beginning = np.tile(origin, [4, 1])  # 4*N x 3
        end = np.concatenate([top_left, top_right, bottom_right, bottom_left], 0)  # 4*N x 3
        end = beginning + rescale_factor * end

        mesh_string = ""
        for x, y, z in beginning:
            mesh_string += "v " + str(x) + " " + str(y) + " " + str(z) + " 0.0 1.0 0.0\n"
        for x, y, z in end:
            mesh_string += "v " + str(x) + " " + str(y) + " " + str(z) + " 1.0 0.0 0.0\n"
        for x, y, z in end:
            mesh_string += "v " + str(x + 0.00001) + " " + str(y) + " " + str(z) + " 1.0 0.0 0.0\n"
        num_vertices = beginning.shape[0]
        for i in range(num_vertices):
            i += 1
            mesh_string += (
                "f " + str(i) + " " + str(i + num_vertices) + " " + str(i + 2 * num_vertices) + "\n"
            )

        import os

        with open(os.path.join(results_folder, "cameras_2.obj"), "w") as mesh_file:
            mesh_file.write(mesh_string)

        num_images = 20
        total_num_images = self.precomputed["rgb"].shape[0]
        step = total_num_images // num_images
        from tqdm import trange

        for image in trange(num_images):
            image = image * step
            rgb = self.precomputed["rgb"][image].numpy()  # H x W x 3
            origin = self.precomputed["rays_origin"][image].numpy()  # H x W x 3
            directions = self.precomputed["rays_dir"][image].numpy()  # H x W x 3
            pos = origin + rescale_factor * directions  # H x W x 3

            mesh_string = ""
            for (x, y, z), (r, g, b) in zip(pos.reshape(-1, 3), rgb.reshape(-1, 3)):
                mesh_string += (
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
                    + "\n"
                )

            with open(
                os.path.join(results_folder, "test_" + str(image).zfill(3) + ".obj"), "w"
            ) as mesh_file:
                mesh_file.write(mesh_string)
