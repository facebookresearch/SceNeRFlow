# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
python rendering.py ./results/toy_example
For a circular camera path, use:
python rendering.py ./results/toy_example --circular
"""

import logging
import os
import sys

import coloredlogs
import imageio
import numpy as np
import torch

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
LOGGER = logging.getLogger(__name__)


def create_folder(folder):
    import pathlib

    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)


# RENDERING


def _editing(joints, joint_name_to_joint_id):

    # example: joints = {"person_one": {0: (0.3, 0.0, 0.5), 1: (0.3, 0.1, 0.5)}}, need to be float32 cuda arrays in normalized canonical space
    # example: joint_name_to_joint_id = {"right_toe": 0}

    import torch
    from utils import project_to_correct_range

    def actual_editing(positions, rgb, alpha):

        people = list(joints.keys())
        device = positions.device

        # MODIFY HERE
        # radius = 0.007 # physical size of the modification, in world-space units
        # transparency_factor = 0.1
        # per_person_joints_to_delete = {people[1]: ["right_toe", "right_foot_tip", "right_ankle"], people[0]: []}
        # per_person_joints_to_recolor = {people[1]: [], people[0]: []}
        # per_person_joints_to_amplify = {people[1]: [("right_shoulder", (0.3, 0.3, 3.0))], people[0]: []} # amplification uses a factor each for the RGB channels, here 0.3 for red
        # blending_alpha = 0.6
        # per_person_joints_to_blend = {people[1]: [], people[0]: [("left_knee", (0.3,0.6,0.3))]} # blend (0.3,0.6,0.3) with the scene color

        for person in people:

            joints_to_delete = per_person_joints_to_delete[person]
            joints_to_recolor = per_person_joints_to_recolor[person]
            joints_to_amplify = per_person_joints_to_amplify[person]
            joints_to_blend = per_person_joints_to_blend[person]

            do_delete = len(joints_to_delete) > 0
            do_recolor = len(joints_to_recolor) > 0
            do_amplify = len(joints_to_amplify) > 0
            do_blend = len(joints_to_blend) > 0

            if do_delete:
                positions_to_delete = torch.stack(
                    [
                        joints[person][joint_name_to_joint_id[joint_name]]
                        for joint_name in joints_to_delete
                    ],
                    dim=0,
                )  # num_positions_to_delete x 3

            def _convert(joint_list):
                positions_to_modify, modifications = zip(
                    *[
                        (
                            joints[person][joint_name_to_joint_id[joint_name]],
                            torch.from_numpy(np.array(modification)).float().to(device),
                        )
                        for joint_name, modification in joint_list
                    ]
                )
                positions_to_modify = torch.stack(
                    list(positions_to_modify), dim=0
                )  # num_positions_to_recolor x 3
                modifications = torch.stack(
                    list(modifications), dim=0
                )  # num_positions_to_recolor x 3
                if rgb.dtype == torch.float16:
                    modifications = modifications.half()
                return positions_to_modify, modifications

            if do_recolor:
                positions_to_recolor, new_colors = _convert(joints_to_recolor)
            if do_amplify:
                positions_to_amplify, color_factors = _convert(joints_to_amplify)
            if do_blend:
                positions_to_blend, blend_colors = _convert(joints_to_blend)

            def within_radius(joints, input_positions):
                pairwise_distances = torch.cdist(
                    input_positions.float().unsqueeze(0),
                    joints.float().unsqueeze(0),
                    compute_mode="donot_use_mm_for_euclid_dist",
                ).squeeze(
                    0
                )  # num_points x num_joints
                max_values, max_indices = torch.max(
                    (pairwise_distances < radius).float(), dim=-1
                )  # num_points

                to_edit_mask = max_values > 0.0
                max_indices = max_indices[to_edit_mask]  # which joint
                return to_edit_mask, max_indices

            # delete
            if do_delete:
                to_delete_mask, _ = within_radius(positions_to_delete, positions)
                alpha[to_delete_mask] *= transparency_factor

            # recolor
            if do_recolor:
                to_recolor_mask, recolor_indices = within_radius(positions_to_recolor, positions)
                rgb[to_recolor_mask] = new_colors[recolor_indices]

            # amplify
            if do_amplify:
                to_amplify_mask, amplify_indices = within_radius(positions_to_amplify, positions)
                rgb[to_amplify_mask] = rgb[to_amplify_mask] * color_factors[amplify_indices]

            # blend
            if do_blend:
                to_blend_mask, blend_indices = within_radius(positions_to_blend, positions)
                rgb[to_blend_mask] = (1.0 - blending_alpha) * rgb[
                    to_blend_mask
                ] + blending_alpha * blend_colors[blend_indices]

        rgb = project_to_correct_range(rgb, mode="zick_zack")
        return rgb, alpha

    return actual_editing


def test_time_rendering_test_cameras(results_folder):

    output_name = "test_cameras"

    only_fine = False
    only_coarse = False
    only_canonical = False
    do_edit = False

    number_of_small_rgb_voxels = 30  # for correspondence visualization. along each axis.
    only_render_if_file_does_not_exist = (
        False  # False saves time when re-running an interrupted rendering
    )
    factor = 2  # downsampling factor for the image resolution

    output_folder = os.path.join(results_folder, "4_outputs", output_name)

    create_folder(output_folder)

    # code_folder = os.path.join(input_results_folder, "2_backup/")
    # sys.path = [code_folder] + sys.path

    use_latest_checkpoint = False

    from settings import config_parser

    settings, _ = config_parser(
        config_file=os.path.join(results_folder, "settings.txt")
    ).parse_known_args()
    do_pref = settings.do_pref
    if do_pref:
        from utils import overwrite_settings_for_pref

        settings = overwrite_settings_for_pref(settings)
        use_latest_checkpoint = True
    do_nrnerf = settings.do_nrnerf
    if do_nrnerf:
        from utils import overwrite_settings_for_nrnerf

        settings = overwrite_settings_for_nrnerf(settings)
        use_latest_checkpoint = True
    from state_loader_saver import StateLoaderSaver

    rank = 0
    state_loader_saver = StateLoaderSaver(settings, rank)
    from data_handler import DataHandler

    data_handler = DataHandler(settings)
    data_handler.load_training_set(
        factor=16,
        num_pixels_per_image=5,
        also_load_top_left_corner_and_four_courners=False,
    )
    data_loader = data_handler.data_loader
    from scene import Scene

    scene = Scene(settings, data_handler).cuda()
    from renderer import Renderer

    renderer = Renderer(settings).cuda()

    if use_latest_checkpoint:
        filename = "latest.pth"
    else:
        filename = "timestep_0.0.pth"
    verify_checkpoint_file = os.path.join(state_loader_saver.get_checkpoint_folder(), filename)
    if os.path.exists(verify_checkpoint_file):
        checkpoint_folder = state_loader_saver.get_checkpoint_folder()
    else:
        checkpoint_folder = state_loader_saver.get_temporary_checkpoint_folder()

    checkpoint_file = os.path.join(checkpoint_folder, filename)
    checkpoint = torch.load(checkpoint_file)
    scene.load_state_dict(checkpoint["scene"])
    renderer.load_state_dict(checkpoint["renderer"])

    class _load_checkpoint_for_timestep:
        def __init__(self):
            self.loaded_timestep = 0.0

        def load(self, timestep):
            if timestep == self.loaded_timestep or use_latest_checkpoint:
                return

            self.loaded_timestep = timestep
            checkpoint_file = os.path.join(checkpoint_folder, "timestep_" + str(timestep) + ".pth")
            if not os.path.exists(checkpoint_file):
                raise FileNotFoundError
            checkpoint = torch.load(checkpoint_file)
            try:
                timevariant_canonical_model = (
                    scene.canonical_model.variant in ["snfa", "snfag"]
                    or scene.canonical_model.brightness_variability > 0.0
                )
            except:
                try:
                    timevariant_canonical_model = (
                        scene.canonical_model.use_time_conditioning
                        or scene.canonical_model.brightness_variability > 0.0
                    )
                except:
                    timevariant_canonical_model = scene.canonical_model.brightness_variability > 0.0
            if timevariant_canonical_model or timestep == 0.0:
                scene.load_state_dict(checkpoint["scene"])
            else:
                if scene.deformation_model is not None:
                    scene.deformation_model.load_state_dict(checkpoint["deformation_model"])

    hacky_checkpoint_loading = _load_checkpoint_for_timestep()

    all_test_imageids = data_loader.get_test_imageids()

    all_rgbs = []
    from tqdm import tqdm

    for test_imageid in tqdm(all_test_imageids):
        groundtruth = data_loader.load_images(
            factor=factor, imageids=np.array([test_imageid]).astype(np.int32)
        )[0]["everything"][
            0
        ]  # H x W x 3

        imageids = np.array([test_imageid]).astype(np.int32)
        test_cameras = {}
        test_cameras["timesteps"] = data_loader.get_timesteps(imageids).cpu()
        extrinids = data_loader.get_extrinids(imageids)
        test_cameras["extrins"] = data_loader.get_extrinsics(extrinids=extrinids)
        intrinids = data_loader.get_intrinids(imageids)
        test_cameras["intrins"] = data_loader.get_intrinsics(intrinids=intrinids, factor=factor)
        test_cameras["rgb"] = torch.from_numpy(groundtruth).unsqueeze(0)
        if data_loader.has_background_images():
            exintrinids = data_loader.get_exintrinids(imageids)
            test_cameras["backgrounds"] = torch.from_numpy(
                data_loader.load_background_images(factor=factor, exintrinids=exintrinids)
            )

        if do_edit:
            joints_file = os.path.join(settings.datadir, "joints.json")
            import json

            with open(joints_file, "r") as json_file:
                joints_dict = json.load(json_file)
                people = joints_dict.keys()

            joint_ids = list(joints_dict["0"].keys())
            joint_name_to_joint_id = {
                joints_dict["0"][joint_id]["joint_name"]: int(joint_id) for joint_id in joint_ids
            }  # str(joint_name): str(joint_id)

            def get_canonical_joints(person, timestep=0):
                canonical_joints = joints_dict[person]
                joint_ids = sorted([int(joint_id) for joint_id in canonical_joints.keys()])
                canonical_joints = [canonical_joints[str(joint_id)] for joint_id in joint_ids]
                canonical_joints = np.array(canonical_joints).astype(np.float32)  # num_joints x 3
                canonical_joints = torch.from_numpy(canonical_joints).cuda().float()
                # convert to normalized NeRF space
                canonical_joints = (canonical_joints - scene.pos_min) / (
                    scene.pos_max - scene.pos_min
                )
                return canonical_joints  # num_joints x 3

            joints = {person: get_canonical_joints(person) for person in people}

            editing = _editing(joints, joint_name_to_joint_id)
        else:
            editing = None

        output_name = str(test_imageid).zfill(8)
        if only_canonical:
            scene.use_deformations = False
            test_cameras["only_canonical"] = True
        if only_fine:
            scene.deformation_model.zero_out_coarse_deformations = True
            test_cameras["only_canonical"] = True  # for correct pruning
        if only_coarse:
            scene.deformation_model.zero_out_fine_deformations = True
        if number_of_small_rgb_voxels is not None:
            test_cameras["number_of_small_rgb_voxels"] = number_of_small_rgb_voxels
        from path_renderer import PathRenderer

        world_size = 1
        path_rendering = PathRenderer(data_handler, rank, world_size)
        try:
            path_rendering.render_and_store(
                state_loader_saver,
                output_name,
                output_folder=output_folder,
                scene=scene,
                renderer=renderer,
                hacky_checkpoint_loading=hacky_checkpoint_loading,
                also_store_images=True,
                only_render_if_file_does_not_exist=only_render_if_file_does_not_exist,
                editing=editing,
                **test_cameras
            )
        except AssertionError:
            pass  # hacky workaround to make sure hacky_checkpoint_loading is only used intentionally
        except FileNotFoundError:
            continue
        if only_canonical:
            scene.use_deformations = True

        file_name = os.path.join(output_folder, output_name + "_rgb.mp4")
        rendered_image = imageio.imread(file_name + "_00000.jpg")
        all_rgbs.append(rendered_image.copy())
        rendered_image = rendered_image.astype(np.float32) / 255.0

    if len(all_rgbs) > 0:
        imageio.mimwrite(
            os.path.join(output_folder, "video.mp4"), np.stack(all_rgbs, axis=0), fps=25, quality=10
        )

    # sys.path.remove(code_folder)


def test_time_rendering_circular(results_folder):

    output_name = "circular"
    factor = 8  # downsampling factor for the image resolution
    circular = {
        "near": 1,  # near plane in world units
        "far": 20,  # far plane in world units
        "num_steps": 60,  # how many timesteps to play. if more than the scene contains, will go back in time again, see below for how this is used
        "center": torch.from_numpy(np.array([0.0, 0.0, 1.0]))
        .float()
        .cuda(),  # the center of the circle in world space
        "radius": 1.0,  # heuristic size of the circle, see below for how this is used
    }
    single_timestep = None  # None: all timesteps, float: bullet-time single timestep
    do_edit = False

    output_folder = os.path.join(results_folder, "4_outputs", output_name)

    create_folder(output_folder)

    # code_folder = os.path.join(results_folder, "2_backup/")
    # sys.path = [code_folder] + sys.path

    from settings import config_parser

    settings, _ = config_parser(
        config_file=os.path.join(results_folder, "settings.txt")
    ).parse_known_args()
    do_pref = settings.do_pref
    if do_pref:
        from utils import overwrite_settings_for_pref

        settings = overwrite_settings_for_pref(settings)
    do_nrnerf = settings.do_nrnerf
    if do_nrnerf:
        from utils import overwrite_settings_for_nrnerf

        settings = overwrite_settings_for_nrnerf(settings)
    from state_loader_saver import StateLoaderSaver

    rank = 0
    state_loader_saver = StateLoaderSaver(settings, rank)
    from data_handler import DataHandler

    data_handler = DataHandler(settings)
    data_handler.load_training_set(
        factor=16,
        num_pixels_per_image=5,
        also_load_top_left_corner_and_four_courners=False,
    )
    data_loader = data_handler.data_loader
    from scene import Scene

    scene = Scene(settings, data_handler).cuda()
    from renderer import Renderer

    renderer = Renderer(settings).cuda()

    if do_pref or do_nrnerf:
        filename = "latest.pth"
    else:
        filename = "timestep_0.0.pth"
    verify_checkpoint_file = os.path.join(state_loader_saver.get_checkpoint_folder(), filename)
    if os.path.exists(verify_checkpoint_file):
        checkpoint_folder = state_loader_saver.get_checkpoint_folder()
    else:
        checkpoint_folder = state_loader_saver.get_temporary_checkpoint_folder()

    checkpoint_file = os.path.join(checkpoint_folder, filename)
    checkpoint = torch.load(checkpoint_file)
    scene.load_state_dict(checkpoint["scene"])
    renderer.load_state_dict(checkpoint["renderer"])

    class _load_checkpoint_for_timestep:
        def __init__(self):
            self.loaded_timestep = 0.0

        def load(self, timestep):
            if timestep == self.loaded_timestep or do_pref or do_nrnerf:
                return

            self.loaded_timestep = timestep

            checkpoint_file = os.path.join(checkpoint_folder, "timestep_" + str(timestep) + ".pth")
            checkpoint = torch.load(checkpoint_file)
            try:
                timevariant_canonical_model = (
                    scene.canonical_model.variant in ["snfa", "snfag"]
                    or scene.canonical_model.brightness_variability > 0.0
                )
            except:
                try:
                    timevariant_canonical_model = (
                        scene.canonical_model.use_timevarying_appearance
                        or scene.canonical_model.use_timevarying_geometry
                        or scene.canonical_model.brightness_variability > 0.0
                    )
                except:
                    timevariant_canonical_model = (
                        scene.canonical_model.use_time_conditioning
                        or scene.canonical_model.brightness_variability > 0.0
                    )
            if timevariant_canonical_model or timestep == 0.0:
                scene.load_state_dict(checkpoint["scene"])
            else:
                if scene.deformation_model is not None:
                    scene.deformation_model.load_state_dict(checkpoint["deformation_model"])

    hacky_checkpoint_loading = _load_checkpoint_for_timestep()

    def get_circular_trajectory(num_steps, near, far, single_timestep, center=None, radius=None):

        mode = "z"  # vertical axis

        imageids = torch.arange(data_loader.num_total_images()).cuda()
        extrinids = data_loader.get_extrinids(imageids)
        input_extrins = data_loader.get_extrinsics(extrinids=torch.unique(extrinids))

        output_extrins = []  # translation, rotation, near, far

        # poses: N x 3 x 4
        translations = torch.stack(
            [extrin["translation"] for extrin in input_extrins], dim=0
        )  # N x 3

        # assume that cameras are somewhat circular
        # figure out bounding box
        min_translation = torch.min(translations, dim=0)[0]
        max_translation = torch.max(translations, dim=0)[0]

        studio_center = (min_translation + max_translation) / 2.0  # 3
        if center is None:
            center = studio_center

        # radius
        if mode == "y":
            distances = translations[:, [0, 2]] - studio_center[[0, 2]].view(
                1, 2
            )  # N x 2. ignore vertical y direction
        elif mode == "z":
            distances = translations[:, [0, 1]] - studio_center[[0, 1]].view(
                1, 2
            )  # N x 2. ignore vertical z direction
        distances = torch.linalg.norm(distances, dim=-1)  # N
        if radius is None:
            radius = 0.6 * torch.max(distances)
        else:
            radius = radius * torch.max(distances)

        # translations
        angles = torch.linspace(0, 2 * np.pi, num_steps)  # shape: num_steps
        for angle in angles:
            if mode == "y":
                extrin = {
                    "translation": torch.stack(
                        [radius * np.cos(angle), torch.zeros(1).cuda()[0], radius * np.sin(angle)],
                        dim=0,
                    ).float()
                    + center
                }
            elif mode == "z":
                extrin = {
                    "translation": torch.stack(
                        [radius * np.cos(angle), radius * np.sin(angle), torch.zeros(1).cuda()[0]],
                        dim=0,
                    ).float()
                    + center
                }
            output_extrins.append(extrin)

        # rotations
        offset = -np.pi / 2
        for angle, extrin in zip(angles + offset, output_extrins):
            if mode == "y":
                rotation = torch.from_numpy(
                    np.array(
                        [
                            [np.cos(angle), 0.0, -np.sin(angle)],
                            [0.0, 1.0, 0.0],
                            [np.sin(angle), 0.0, np.cos(angle)],
                        ]
                    )
                )  # 3x3
            elif mode == "z":
                rotation = torch.from_numpy(
                    np.array(
                        [
                            [np.cos(angle), 0.0, -np.sin(angle)],
                            [np.sin(angle), 0.0, np.cos(angle)],
                            [0.0, 1.0, 0.0],
                        ]
                    )
                )  # 3x3
            extrin["rotation"] = rotation.float()

        # near, far
        for extrin in output_extrins:
            extrin["near"] = near
            extrin["far"] = far

        # intrins
        intrinids = data_loader.get_intrinids(imageids)
        input_intrins = data_loader.get_intrinsics(intrinids=torch.unique(intrinids), factor=factor)
        intrin = input_intrins[0]
        output_intrins = [intrin for _ in range(num_steps)]

        # timesteps
        input_timesteps = torch.sort(torch.unique(data_loader.get_timesteps(imageids).cpu()))[0]
        if single_timestep is None:
            input_timesteps = torch.cat(
                [input_timesteps, torch.flip(input_timesteps, dims=[0])], dim=0
            )
            output_timesteps = []
            while len(output_timesteps) < num_steps:
                counter = len(output_timesteps)
                output_timesteps.append(input_timesteps[counter % len(input_timesteps)])
        else:
            output_timesteps = [
                torch.from_numpy(np.array([single_timestep], dtype=float)) for _ in range(num_steps)
            ]
        output_timesteps = torch.stack(output_timesteps, dim=0)

        return output_extrins, output_intrins, output_timesteps

    if circular is not None:
        extrins, intrins, timesteps = get_circular_trajectory(
            num_steps=circular["num_steps"],
            near=circular["near"],
            far=circular["far"],
            single_timestep=single_timestep,
            center=circular["center"],
            radius=circular["radius"],
        )

    use_backgrounds = False
    if use_backgrounds:
        background_images_dict = {}
        background_images_dict["background"] = torch.from_numpy(
            data_loader.load_background_images(factor=factor)
        )
    else:
        background_images_dict = None

    def visualize_cameras(extrins, intrins, timesteps):

        rays_dict = data_handler.ray_builder.build_multiple(extrins, intrins)
        rays_origin = rays_dict["rays_origin"]["everything"]  # torch cpu, N x H x W x 3
        rays_dir = rays_dict["rays_dir"]["everything"]  # torch cpu, N x H x W x 3

        top_left = rays_origin[:, 0, 0, :]
        top_right = rays_origin[:, 0, -1, :]
        bottom_left = rays_origin[:, -1, 0, :]
        bottom_right = rays_origin[:, -1, -1, :]
        rays_origin = torch.cat([top_left, top_right, bottom_left, bottom_right], dim=0)

        top_left = rays_dir[:, 0, 0, :]
        top_right = rays_dir[:, 0, -1, :]
        bottom_left = rays_dir[:, -1, 0, :]
        bottom_right = rays_dir[:, -1, -1, :]
        rays_dir = torch.cat([top_left, top_right, bottom_left, bottom_right], dim=0)

        batch = {
            "rays_origin": rays_origin,
            "rays_dir": rays_dir,
            "timesteps": timesteps.repeat_interleave(4),
            "near": torch.from_numpy(
                np.array([extrin["near"] for extrin in extrins])
            ).repeat_interleave(4),
            "far": torch.from_numpy(
                np.array([extrin["far"] for extrin in extrins])
            ).repeat_interleave(4),
            "intrinids": torch.from_numpy(
                np.array([intrin["intrinid"] for intrin in intrins])
            ).repeat_interleave(4),
        }
        batch = {key: tensor.cuda() for key, tensor in batch.items()}

        from utils import Returns

        returns = Returns()
        returns.activate_mode("extent")
        with torch.no_grad():
            renderer.render(
                batch, scene=scene, points_per_ray=4, is_training=False, returns=returns
            )

        camera_positions = batch["rays_origin"].view(-1, 3)
        rays_near = returns.get_returns()["unnormalized_undeformed_positions"][:, 0, :]
        rays_far = returns.get_returns()["unnormalized_undeformed_positions"][:, -1, :]

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

        with open(os.path.join(output_folder, "cameras.obj"), "w") as mesh_file:
            mesh_file.write(mesh_string)

    visualize_cameras(extrins, intrins, timesteps)

    if do_edit:
        joints_file = os.path.join(settings.datadir, "joints.json")
        import json

        with open(joints_file, "r") as json_file:
            joints_dict = json.load(json_file)
        people = joints_dict.keys()

        joint_ids = list(joints_dict["0"].keys())
        joint_name_to_joint_id = {
            joints_dict["0"][joint_id]["joint_name"]: int(joint_id) for joint_id in joint_ids
        }  # str(joint_name): str(joint_id)

        def get_canonical_joints(person, timestep=0):
            canonical_joints = joints_dict[person]
            joint_ids = sorted([int(joint_id) for joint_id in canonical_joints.keys()])
            canonical_joints = [canonical_joints[str(joint_id)] for joint_id in joint_ids]
            canonical_joints = np.array(canonical_joints).astype(np.float32)  # num_joints x 3
            canonical_joints = torch.from_numpy(canonical_joints).cuda().float()
            # convert to normalized NeRF space
            canonical_joints = (canonical_joints - scene.pos_min) / (scene.pos_max - scene.pos_min)
            return canonical_joints  # num_joints x 3

        joints = {person: get_canonical_joints(person) for person in people}

        editing = _editing(joints, joint_name_to_joint_id)
    else:
        editing = None

    from path_renderer import PathRenderer

    world_size = 1
    path_rendering = PathRenderer(data_handler, rank, world_size)
    try:
        path_rendering.render_and_store(
            state_loader_saver,
            output_name,
            extrins=extrins,
            intrins=intrins,
            timesteps=timesteps,
            scene=scene,
            renderer=renderer,
            backgrounds=background_images_dict,
            output_folder=output_folder,
            hacky_checkpoint_loading=hacky_checkpoint_loading,
            only_render_if_file_does_not_exist=False,
            editing=editing,
        )
    except AssertionError:
        pass  # related to hacky_checkpoint_loading

    # sys.path.remove(code_folder)


if __name__ == "__main__":

    # logging_level = logging.DEBUG
    logging_level = logging.INFO
    coloredlogs.install(level=logging_level, fmt="%(name)s[%(process)d] %(levelname)s %(message)s")
    logging.basicConfig(level=logging_level)

    results_folder = sys.argv[1]

    if len(sys.argv) > 2:
        LOGGER.info("Will render using circular trajectory.")
        test_time_rendering_circular(results_folder)
    else:
        LOGGER.info("Will render into test cameras")
        test_time_rendering_test_cameras(results_folder)
