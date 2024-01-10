# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import logging

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import pathlib
import shutil

import torch
from multi_gpu import multi_gpu_barrier
from tqdm import tqdm
from utils import Returns

LOGGER = logging.getLogger(__name__)


class StateLoaderSaver:
    def __init__(self, settings, rank):

        self.basedir = settings.basedir
        if settings.temporary_basedir is None:
            self.temporary_basedir = settings.basedir
        else:
            self.temporary_basedir = settings.temporary_basedir
        self.expname = settings.expname
        self.reload = not settings.no_reload

        self.save_checkpoint_every = settings.save_checkpoint_every
        self.save_intermediate_checkpoint_every = settings.save_intermediate_checkpoint_every
        self.save_temporary_checkpoint_every = settings.save_temporary_checkpoint_every

        self.multi_gpu = settings.multi_gpu
        self.rank = rank

    @staticmethod
    def create_folder(folder):
        pathlib.Path(folder).mkdir(parents=True, exist_ok=True)

    def get_experiment_name(self):
        return self.expname

    def get_results_folder(self):
        return os.path.join(self.basedir, self.get_experiment_name())

    def get_checkpoint_folder(self):
        return os.path.join(self.get_results_folder(), "1_checkpoints/")

    def get_latest_checkpoint_file(self):
        return os.path.join(self.get_checkpoint_folder(), "latest.pth")

    def get_temporary_results_folder(self):
        return os.path.join(self.temporary_basedir, self.get_experiment_name())

    def get_temporary_checkpoint_folder(self):
        return os.path.join(self.get_temporary_results_folder(), "1_checkpoints/")

    def get_latest_temporary_checkpoint_file(self):
        return os.path.join(self.get_temporary_checkpoint_folder(), "latest.pth")

    ### BACKUP

    def backup_files(self, settings):

        if self.rank != 0:
            return

        LOGGER.info("backing up... ")

        results_folder = self.get_results_folder()
        temporary_results_folder = self.get_temporary_results_folder()
        if not self.reload and os.path.exists(temporary_results_folder):
            shutil.rmtree(temporary_results_folder)
        if os.path.exists(results_folder):
            if self.reload:
                LOGGER.info("already exists.")
                return
            else:
                shutil.rmtree(results_folder)
        self.create_folder(results_folder)

        self.create_folder(self.get_checkpoint_folder())
        self.create_folder(self.get_temporary_checkpoint_folder())

        f = os.path.join(results_folder, "settings.txt")
        with open(f, "w") as file:
            for arg in sorted(vars(settings)):
                attr = getattr(settings, arg)
                file.write("{} = {}\n".format(arg, attr))
        if settings.config is not None:
            f = os.path.join(results_folder, "config.txt")
            with open(f, "w") as file:
                with open(settings.config, "r") as original_config:
                    file.write(original_config.read())

        target_folder = os.path.join(results_folder, "2_backup/")

        special_files_to_copy = []
        filetypes_to_copy = [".py", ".txt"]
        subfolders_to_copy = ["", "configs/"]

        this_file = os.path.realpath(__file__)
        this_folder = os.path.dirname(this_file) + "/"
        self.create_folder(target_folder)
        # special files
        [
            self.create_folder(os.path.join(target_folder, os.path.split(file)[0]))
            for file in special_files_to_copy
        ]
        [
            shutil.copyfile(os.path.join(this_folder, file), os.path.join(target_folder, file))
            for file in special_files_to_copy
        ]
        # folders
        for subfolder in subfolders_to_copy:
            self.create_folder(os.path.join(target_folder, subfolder))
            files = os.listdir(os.path.join(this_folder, subfolder))
            files = [
                file
                for file in files
                if os.path.isfile(os.path.join(this_folder, subfolder, file))
                and file[file.rfind(".") :] in filetypes_to_copy
            ]
            [
                shutil.copyfile(
                    os.path.join(this_folder, subfolder, file),
                    os.path.join(target_folder, subfolder, file),
                )
                for file in files
            ]

    ### LOGGING

    def print_log(self, training_iteration, logging):
        logging_string = "[TRAIN] Iter: " + str(training_iteration)
        if logging is not None and "psnr" in logging:
            logging_string += " PSNR: " + str(logging["psnr"])
        LOGGER.info(logging_string)

    ### CHECKPOINTS

    def save(
        self, training_iteration, scene, renderer, scheduler, trainer, force_save_in_stable=False
    ):

        if self.rank != 0:
            return

        checkpoint = {
            "training_iteration": training_iteration,
            "scene": scene.state_dict(),
            "renderer": renderer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "trainer": trainer.state_dict(),
        }

        # save in temporary /scratch storage
        checkpoint_file = self.get_latest_temporary_checkpoint_file()
        LOGGER.info("saving to scratch storage: " + checkpoint_file)
        temporary_file = checkpoint_file + "_TEMP"
        torch.save(
            checkpoint, temporary_file
        )  # try to avoid getting interrupted while writing to checkpoint
        os.rename(temporary_file, checkpoint_file)

        # save in stable storage
        if force_save_in_stable or (training_iteration % self.save_checkpoint_every == 0):
            stable_checkpoint_file = self.get_latest_checkpoint_file()
            if checkpoint_file != stable_checkpoint_file:
                LOGGER.info("copying to stable storage: " + stable_checkpoint_file)
                shutil.copyfile(checkpoint_file, stable_checkpoint_file)

        # keep copy of this intermediate checkpoint
        if training_iteration % self.save_intermediate_checkpoint_every == 0:
            intermediate_checkpoint_file = os.path.join(
                self.get_checkpoint_folder(), str(training_iteration).zfill(9) + ".pth"
            )
            shutil.copyfile(checkpoint_file, intermediate_checkpoint_file)

    def save_for_only_test(self, timestep, scene, renderer, stable_storage=True):

        if self.rank != 0:
            return

        checkpoint = {
            "timestep": timestep,
            "renderer": renderer.state_dict(),
        }

        timevariant_canonical_model = (
            scene.canonical_model.variant in ["snfa", "snfag"]
            or scene.canonical_model.brightness_variability > 0.0
        )
        if timevariant_canonical_model or timestep == 0.0:
            checkpoint["scene"] = scene.state_dict()
        else:
            if scene.deformation_model is not None:
                checkpoint["deformation_model"] = scene.deformation_model.state_dict()

        if stable_storage:
            # save in stable storage
            checkpoint_file = os.path.join(
                self.get_checkpoint_folder(), "timestep_" + str(timestep) + ".pth"
            )
            LOGGER.info("saving timestep to stable storage: " + checkpoint_file)
            temporary_file = checkpoint_file + "_TEMP"
            torch.save(
                checkpoint, temporary_file
            )  # try to avoid getting interrupted while writing to checkpoint
            os.rename(temporary_file, checkpoint_file)
        else:
            # save in temporary /scratch storage
            checkpoint_file = os.path.join(
                self.get_temporary_checkpoint_folder(), "timestep_" + str(timestep) + ".pth"
            )
            LOGGER.info("saving timestep to scratch storage: " + checkpoint_file)
            temporary_file = checkpoint_file + "_TEMP"
            torch.save(
                checkpoint, temporary_file
            )  # try to avoid getting interrupted while writing to checkpoint
            os.rename(temporary_file, checkpoint_file)

    def get_last_stored_training_iteration(self):
        return self.last_stored_training_iteration

    def latest_checkpoint(self):
        try:
            checkpoint_file = self.get_latest_temporary_checkpoint_file()
            if not os.path.exists(checkpoint_file):
                checkpoint_file = self.get_latest_checkpoint_file()

            map_location = {
                "cuda:0": "cuda:" + str(self.rank)
            }  # rank=0 stores the checkpoint, but when loading, we want to load into rank=self.rank
            LOGGER.info("trying to load from: " + checkpoint_file)
            checkpoint = torch.load(checkpoint_file, map_location=map_location)
        except FileNotFoundError:
            checkpoint = None
        return checkpoint

    def initialize_parameters(self, scene, renderer, scheduler, trainer, data_handler):

        checkpoint = self.latest_checkpoint()
        if checkpoint is None:

            pos_max, pos_min = self.determine_nerf_volume_extent(scene, renderer, data_handler)
            scene.set_pos_max_min(pos_max=pos_max, pos_min=pos_min)

            # share initial parameters from rank 0 to all other ranks by loading and storing a checkpoint
            if self.multi_gpu:
                multi_gpu_barrier(self.rank)  # all processes need to see that checkpoint is None
            if self.rank == 0:
                self.save(-1, scene, renderer, scheduler, trainer)
            if self.multi_gpu:
                multi_gpu_barrier(self.rank)
            checkpoint = self.latest_checkpoint()

        scene.load_state_dict(checkpoint["scene"])
        renderer.load_state_dict(checkpoint["renderer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        trainer.load_state_dict(checkpoint["trainer"])

        self.last_stored_training_iteration = checkpoint["training_iteration"]

    ### INITIALIZATION

    def determine_nerf_volume_extent(
        self, scene, renderer, data_handler, output_camera_visualization=True
    ):
        # the nerf volume has some extent, but this extent is not fixed. this function computes (somewhat approximate) minimum and maximum coordinates along each axis. it considers all cameras (their positions and point samples along the rays of their corners).

        batch = data_handler.get_batch(subset_name="four_corners")

        returns = Returns()
        returns.activate_mode("extent")
        with torch.no_grad():
            renderer.render(
                batch, scene=scene, points_per_ray=4, is_training=False, returns=returns
            )

        critical_ray_points = returns.get_returns()["unnormalized_undeformed_positions"].view(-1, 3)
        consider_camera_positions = False
        if consider_camera_positions:
            camera_positions = batch["rays_origin"].view(-1, 3)
            critical_points = torch.cat([critical_ray_points, camera_positions], dim=0)
        else:
            critical_points = critical_ray_points
        pos_min = torch.min(critical_points, dim=0)[0]
        pos_max = torch.max(critical_points, dim=0)[0]

        # add some extra space around the volume. stretch away from the center of the volume.
        stretching_factor = 1.1
        center = (pos_min + pos_max) / 2.0
        pos_min -= center
        pos_max -= center
        pos_min *= stretching_factor
        pos_max *= stretching_factor
        pos_min += center
        pos_max += center

        if output_camera_visualization:
            camera_positions = batch["rays_origin"].view(-1, 3)
            rays_near = returns.get_returns()["unnormalized_undeformed_positions"][:, 0, :]
            rays_far = returns.get_returns()["unnormalized_undeformed_positions"][:, -1, :]
            self.visualize_cameras(camera_positions, rays_near, rays_far, filename="cameras.obj")

        return pos_max, pos_min

    def visualize_cameras(self, camera_positions, rays_near, rays_far, filename):

        if self.rank != 0:
            return

        cameras = camera_positions.detach().cpu().numpy()
        beginning = rays_near.detach().cpu().numpy()
        end = rays_far.detach().cpu().numpy()

        mesh_string = ""
        for x, y, z in beginning:
            mesh_string += "v " + str(x) + " " + str(y) + " " + str(z) + " 0.0 1.0 0.0\n"
        for x, y, z in end:
            mesh_string += "v " + str(x) + " " + str(y) + " " + str(z) + " 1.0 0.0 0.0\n"
        for x, y, z in end:
            mesh_string += "v " + str(x + 0.00001) + " " + str(y) + " " + str(z) + " 1.0 0.0 0.0\n"
        for x, y, z in cameras:
            mesh_string += "v " + str(x) + " " + str(y) + " " + str(z) + " 0.0 0.0 1.0\n"
        for x, y, z in cameras:
            mesh_string += "v " + str(x + 0.00001) + " " + str(y) + " " + str(z) + " 0.0 0.0 1.0\n"
        for x, y, z in cameras:
            mesh_string += "v " + str(x) + " " + str(y + 0.00001) + " " + str(z) + " 0.0 0.0 1.0\n"
        num_vertices = beginning.shape[0]
        for i in range(num_vertices):
            i += 1
            mesh_string += (
                "f " + str(i) + " " + str(i + num_vertices) + " " + str(i + 2 * num_vertices) + "\n"
            )
        offset = 3 * num_vertices
        num_cameras = cameras.shape[0]
        for i in range(num_cameras):
            i += 1
            mesh_string += (
                "f "
                + str(offset + i)
                + " "
                + str(offset + i + num_cameras)
                + " "
                + str(offset + i + 2 * num_cameras)
                + "\n"
            )

        with open(os.path.join(self.get_results_folder(), filename), "w") as mesh_file:
            mesh_file.write(mesh_string)
