# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
python preprocess.py INPUT_FOLDER OUTPUT_FOLDER
# shorthand alternative:
python preprocess.py INPUT_FOLDER
# will use INPUT_FODLER as OUTPUT_FOLDER
"""
import json
import logging
import os
import shutil
import sys
from io import BytesIO

import coloredlogs
import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from tqdm import tqdm

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
LOGGER = logging.getLogger(__name__)


def create_folder(folder):
    import pathlib

    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)


# PREPROCESSING


def _visualize_voxel_grid(
    output_file,
    voxel_grid,
    voxel_grid_pos=None,
    voxel_grid_min=None,
    voxel_grid_max=None,
    mode="rgb",
    remove_zeros=False,
    has_unit_range=False,
):

    # modes: rgb, binary, grey

    if voxel_grid_pos is None:
        if voxel_grid_min is None:  # position of minimum corner
            voxel_grid_min = torch.from_numpy(np.array([0.0, 0.0, 0.0]))
        if voxel_grid_max is None:
            voxel_grid_max = torch.from_numpy(np.array([1.0, 1.0, 1.0]))

    voxel_grid = voxel_grid.clone()
    num_x, num_y, num_z = voxel_grid.shape[:3]

    if mode == "binary":
        true_color = torch.from_numpy(np.array([0.0, 1.0, 0.0]))
        false_color = torch.from_numpy(np.array([1.0, 0.0, 0.0]))
        has_unit_range = True

        mask = voxel_grid > 0  # convert to boolean
        mask = mask.view(-1)

        if len(voxel_grid.shape) == 3:
            voxel_grid = torch.stack([voxel_grid, voxel_grid, voxel_grid], dim=-1)

        flattened_voxel_grid = voxel_grid.view(-1, 3).long()
        flattened_voxel_grid[mask, :] = true_color
        flattened_voxel_grid[~mask, :] = false_color

        voxel_grid = flattened_voxel_grid.view([num_x, num_y, num_z, 3])
        if remove_zeros:
            voxel_grid_for_zeros = voxel_grid[:, :, :, 0]

    if mode == "grey":
        use_color_scheme = True
        if use_color_scheme:
            from matplotlib import cm

            if len(voxel_grid.shape) == 4:
                voxel_grid = voxel_grid[:, :, :, 0]
            if remove_zeros:
                voxel_grid_for_zeros = voxel_grid.clone()
            if has_unit_range:
                voxel_grid = 255.0 * voxel_grid
            voxel_grid = torch.clamp(voxel_grid, min=0.0, max=255.0)
            # voxel_grid = cm.jet(voxel_grid.cpu().numpy().astype(np.int32))[:,:,:,:3]
            voxel_grid = cm.turbo(voxel_grid.cpu().numpy().astype(np.uint8))[:, :, :, :3]
            has_unit_range = True
        else:
            if len(voxel_grid.shape) == 3:
                voxel_grid = torch.stack([voxel_grid, voxel_grid, voxel_grid], dim=-1)
            if remove_zeros:
                voxel_grid_for_zeros = voxel_grid

    if not has_unit_range:
        voxel_grid /= 255.0

    try:
        voxel_grid = voxel_grid.cpu().numpy()
    except AttributeError:
        pass

    output_strings = []
    if voxel_grid_pos is None:
        scale_x, scale_y, scale_z = voxel_grid_max - voxel_grid_min
        min_x, min_y, min_z = voxel_grid_min
    else:
        this_voxel_grid_pos = voxel_grid_pos
        try:
            this_voxel_grid_pos = this_voxel_grid_pos.clone().cpu().numpy()
        except AttributeError:
            pass
    from tqdm import tqdm

    for x in tqdm(range(num_x)):
        for y in range(num_y):
            for z in range(num_z):
                if remove_zeros and voxel_grid_for_zeros[x, y, z] == 0:
                    continue
                if voxel_grid_pos is None:
                    pos_x = (x / num_x) * scale_x + min_x
                    pos_y = (y / num_y) * scale_y + min_y
                    pos_z = (z / num_z) * scale_z + min_z
                else:
                    pos_x, pos_y, pos_z = this_voxel_grid_pos[x, y, z]
                r, g, b = voxel_grid[x, y, z]
                output_strings.append(
                    "v "
                    + str(pos_x)
                    + " "
                    + str(pos_y)
                    + " "
                    + str(pos_z)
                    + " "
                    + str(r)
                    + " "
                    + str(g)
                    + " "
                    + str(b)
                )

    with open(output_file, "w") as output_file:
        output_file.write("\n".join(output_strings))


def _combine_pruned_voxel_grids(dataset=None, voxel_grid_size=None):

    if voxel_grid_size is None:
        voxel_grid_size = 128

    from binary_dataset import BinaryDataset

    voxel_grid_dataset = BinaryDataset(dataset, name="foreground_voxel_grids", read_only=False)
    voxel_grid_names = list(voxel_grid_dataset.keys())

    if "all" in voxel_grid_names:
        raise RuntimeError("already computed")

    voxel_grids = []
    for voxel_grid_name in voxel_grid_names:
        from io import BytesIO

        voxel_grid_bytes = BytesIO(voxel_grid_dataset.get_entry(voxel_grid_name))
        voxel_grid_bytes.seek(0)
        voxel_grid = np.load(voxel_grid_bytes)
        voxel_grid = torch.from_numpy(
            voxel_grid["foreground_voxel_grid"]
        ).cuda()  # voxel_grid_size x voxel_grid_size x voxel_grid_size. boolean
        voxel_grids.append(voxel_grid)

    voxel_grids = torch.stack(
        voxel_grids, dim=-1
    )  # voxel_grid_size x voxel_grid_size x voxel_grid_size x num_timesteps. boolean
    all_voxel_grid = torch.any(
        voxel_grids, dim=-1
    )  # voxel_grid_size x voxel_grid_size x voxel_grid_size. boolean

    from io import BytesIO

    voxel_grid_bytes = BytesIO()
    np.savez_compressed(voxel_grid_bytes, foreground_voxel_grid=all_voxel_grid.cpu().numpy())
    voxel_grid_dataset.maybe_add_entry(voxel_grid_bytes, key="all")

    voxel_grid_dataset.close()


def _space_carving(dataset=None, voxel_grid_size=None, debug=None, append_to_existing_file=False):

    if debug is None:
        debug = True
    if debug:
        factor = 1
        if voxel_grid_size is None:
            voxel_grid_size = 128

        background_threshold = 20.0 / 255.0  # background subtraction difference threshold
        foreground_mask_dilation_kernel_size = (
            71  # in pixels. will be rounded up to nearest odd number
        )
        visualize_foreground_masks = True

        fraction_threshold = 0.9  # minimum fraction of cameras that see a point that have to say it's foreground in order for it to stay in the voxel grid
        visibility_threshold = 4  # minimum number of cameras that need to see a point in order for the point to be considered valuable at all
        visualize_voxel_grids = True
        voxel_grid_dilation_kernel_size = 5  # in voxels. will be rounded up to nearest odd number

        studio_bounding_box = False

        use_binary_dataset = False

    else:
        factor = 1
        if voxel_grid_size is None:
            voxel_grid_size = 128

        background_threshold = 20.0 / 255.0
        foreground_mask_dilation_kernel_size = (
            71  # in pixels. will be rounded up to nearest odd number
        )
        visualize_foreground_masks = False

        fraction_threshold = 0.9  # minimum fraction of cameras that see a point that have to say it's foreground in order for it to stay in the voxel grid
        visibility_threshold = 4  # minimum number of cameras that need to see a point in order for the point to be considered valuable at all
        visualize_voxel_grids = False
        voxel_grid_dilation_kernel_size = 5  # in voxels. will be rounded up to nearest odd number

        studio_bounding_box = False

        use_binary_dataset = True

    from settings import config_parser

    settings, _ = config_parser().parse_known_args()
    settings.datadir = dataset
    settings.multi_gpu = False
    settings.use_background = True
    settings.do_vignetting_correction = False
    settings.debug = False

    settings.basedir = None
    settings.temporary_basedir = None
    settings.expname = None
    settings.no_reload = None

    settings.save_checkpoint_every = None
    settings.save_intermediate_checkpoint_every = None
    settings.save_temporary_checkpoint_every = None

    settings.use_time_conditioning = True
    settings.use_viewdirs = False
    settings.brightness_variability = 0.0
    settings.activation_function = "ReLU"
    settings.use_pruning = False
    settings.voxel_grid_size = 0

    settings.prefer_cutlass_over_fullyfused_mlp = False

    settings.learning_rate_decay_autodecoding_fraction = None
    settings.learning_rate_decay_autodecoding_iterations = None
    settings.learning_rate_decay_mlp_fraction = None
    settings.learning_rate_decay_mlp_iterations = None
    settings.weight_parameter_regularization = None

    settings.color_calibration_mode = "none"
    settings.points_per_chunk = 4194304
    settings.num_points_per_ray = 1024
    settings.disparity_sampling = False
    settings.raw_noise_std = 0.0
    settings.use_half_precision = False

    from data_handler import DataHandler

    data_handler = DataHandler(settings)
    data_loader = data_handler.data_loader
    imageids = torch.from_numpy(data_loader.get_train_imageids()).cuda()
    device = imageids.device

    # determine nerf volume grid extent
    data_handler.load_training_set(
        factor=16,
        num_pixels_per_image=5,  # some dummy value
        also_load_top_left_corner_and_four_courners=True,
    )  # four_corners subset need to be loaded
    from scene import Scene
    from renderer import Renderer

    scene = Scene(settings, data_handler).cuda()  # incl. time line
    renderer = Renderer(settings).cuda()

    from state_loader_saver import StateLoaderSaver

    state_loader_saver = StateLoaderSaver(settings, rank=0)
    grid_pos_max, grid_pos_min = state_loader_saver.determine_nerf_volume_extent(
        scene, renderer, data_handler, output_camera_visualization=False
    )

    # create auxiliary voxel grids containing voxel indices and positions of voxel centers
    voxel_grid_index_x, voxel_grid_index_y, voxel_grid_index_z = torch.meshgrid(
        torch.arange(voxel_grid_size, device=device),
        torch.arange(voxel_grid_size, device=device),
        torch.arange(voxel_grid_size, device=device),
        indexing="ij",
    )  # voxel_grid_index_x: size x size x size
    voxel_grid_index = (
        torch.stack([voxel_grid_index_x, voxel_grid_index_y, voxel_grid_index_z], dim=-1)
        .view(-1, 3)
        .cuda()
    )  # size * size * size x 3
    # apply min-max Nerf volume extent: turn into voxel center (+0.5), divide by voxel_grid_size, scale by (max-min), shift by min
    voxel_grid_pos = (voxel_grid_index + 0.5) / voxel_grid_size  # in [0,1]
    voxel_grid_pos = voxel_grid_pos * (grid_pos_max - grid_pos_min).view(1, 3) + grid_pos_min.view(
        1, 3
    )  # size * size * size x 3

    # load static background images
    background_images_dict = {}
    background_images_dict["background"] = torch.from_numpy(
        data_loader.load_background_images(factor=factor)
    )
    # background_images_dict["exintrinids"] = data_loader.get_exintrinids(imageids).cpu()
    background_images_dict["background_threshold"] = background_threshold

    # studio bounding box
    if studio_bounding_box:

        extrinids = data_loader.get_extrinids(imageids)
        extrins = data_loader.get_extrinsics(extrinids=torch.unique(extrinids))
        all_translations = torch.stack(
            [extrin["translation"] for extrin in extrins], dim=0
        )  # N x 3
        studio_min = torch.min(all_translations, dim=0)[0]
        studio_max = torch.max(all_translations, dim=0)[0]

        studio_center = (studio_min + studio_max) / 2.0
        studio_dilation_factor = 1.3
        studio_min = studio_center + studio_dilation_factor * (studio_min - studio_center)
        studio_max = studio_center + studio_dilation_factor * (studio_max - studio_center)

        studio_min = torch.max(torch.stack([studio_min, grid_pos_min], dim=0), dim=0)[0]
        studio_max = torch.min(torch.stack([studio_max, grid_pos_max], dim=0), dim=0)[0]

        valid_min_mask = voxel_grid_pos >= studio_min.view(1, 3)  # num_voxels x 3
        valid_min_mask = torch.all(valid_min_mask, dim=-1)  # num_voxels
        valid_max_mask = voxel_grid_pos <= studio_max.view(1, 3)
        valid_max_mask = torch.all(valid_max_mask, dim=-1)  # num_voxels

        inside_studio_mask = torch.logical_and(valid_min_mask, valid_max_mask)  # num_voxels

    # determine timestamp per image
    timesteps = data_loader.get_timesteps(imageids).cpu()

    # open BinaryDataset
    if use_binary_dataset:
        from binary_dataset import BinaryDataset

        output_dataset = BinaryDataset(
            folder=dataset,
            name="foreground_voxel_grids",
            delete_existing=not append_to_existing_file,
            read_only=False,
        )

    # for loop over time
    from tqdm import tqdm

    for current_timestep in tqdm(sorted(torch.unique(timesteps).numpy())):

        # get timestamp's imageids
        current_indices_among_loaded_images = torch.where(timesteps == current_timestep)[0]
        current_imageids = imageids[current_indices_among_loaded_images]

        # load foreground masks
        desired_subsets = {"main": {"mode": "background_masks"}}

        # desired_subsets["DEBUG"] = {"mode": "foreground_focused",
        #                            "foreground_fraction": 0.8,
        #                            "num_pixels_per_image": 5,
        #                            "write_out_masks": None,
        #                            }
        background_images_dict["exintrinids"] = data_loader.get_exintrinids(current_imageids).cpu()
        rgb_subsets, _ = data_loader.load_images(
            factor=factor,
            imageids=current_imageids,
            desired_subsets=desired_subsets,
            background_images_dict=background_images_dict,
        )
        current_foreground_masks = (
            1 - torch.from_numpy(rgb_subsets["main"]).cuda()
        )  # 0: background, 1: foreground

        # determine extrinsics and intrinsics per mask

        extrinids = data_loader.get_extrinids(current_imageids)
        extrins = data_loader.get_extrinsics(extrinids=extrinids)

        intrinids = data_loader.get_intrinids(current_imageids)
        intrins = data_loader.get_intrinsics(intrinids=intrinids, factor=factor)

        # dilate foreground of masks (0: background, 1: foreground)
        current_foreground_masks = torch.unsqueeze(
            current_foreground_masks, dim=1
        )  # num_masks x 1 x height x width
        kernel_size = foreground_mask_dilation_kernel_size
        if kernel_size % 2 == 0:
            kernel_size += 1  # needed for integer padding
        padding = (kernel_size - 1) // 2
        current_foreground_masks = torch.nn.functional.max_pool2d(
            current_foreground_masks,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=1,
            ceil_mode=True,
            return_indices=False,
        )

        # create voxel grid
        background_voxel_grid = torch.zeros(
            (voxel_grid_size, voxel_grid_size, voxel_grid_size), dtype=int
        ).cuda()
        visible_voxel_grid = torch.zeros(
            (voxel_grid_size, voxel_grid_size, voxel_grid_size), dtype=int
        ).cuda()

        # for loop over all masks
        for current_foreground_mask, extrin, intrin in zip(
            current_foreground_masks, extrins, intrins
        ):

            current_foreground_mask = torch.squeeze(
                current_foreground_mask, dim=0
            )  # height x width

            if visualize_foreground_masks:
                foreground_masks_for_voxel_grids = os.path.join(
                    dataset, "foreground_masks_for_voxel_grids"
                )
                import pathlib

                pathlib.Path(foreground_masks_for_voxel_grids).mkdir(parents=True, exist_ok=True)
                if "counter" not in locals():
                    counter = -1
                counter += 1
                import imageio

                imageio.imwrite(
                    os.path.join(
                        foreground_masks_for_voxel_grids,
                        str(current_timestep) + "_" + str(counter) + ".jpg",
                    ),
                    (255 * current_foreground_mask.cpu().numpy()).astype(np.uint8),
                )

            # project each voxel center
            # ignore distortion parameters

            # subtract translation
            positions = voxel_grid_pos - extrin["translation"].view(1, 3)

            # left-apply inverse rotation
            positions = torch.matmul(
                extrin["rotation"].view(3, 3).t(), positions.t()
            )  # 3 x num_voxels

            # flip axes
            positions[1, :] *= -1
            positions[2, :] *= -1

            # divide by resulting z coordinate
            behind_camera_mask = positions[2, :] >= 0.0  # positive z is behind camera
            positions = (positions[:2, :] / positions[2:, :]).t()  # num_voxels x 2

            # multiply x coordinate by focal_x, then add center_x
            positions[:, 0] = intrin["focal_x"] * positions[:, 0] + intrin["center_x"]

            # multiply y coordinate by focal_y, then add center_y
            positions[:, 1] = intrin["focal_y"] * positions[:, 1] + intrin["center_y"]

            # check whether it is inside the image, using height and width
            inside_image = torch.logical_and(
                behind_camera_mask,
                torch.logical_and(
                    (positions[:, 0] >= 0),
                    torch.logical_and(
                        (positions[:, 0] < intrin["width"]),
                        torch.logical_and(
                            (positions[:, 1] >= 0), (positions[:, 1] < intrin["height"])
                        ),
                    ),
                ),
            )  # num_voxels

            # for these, round down to nearest (u,v) coordinate and read out mask and see if it is background
            inside_positions = positions[inside_image]  # num_inside_voxels x 2
            inside_positions = torch.floor(inside_positions).long()  # num_inside_voxels x 2
            is_background = (
                current_foreground_mask[inside_positions[:, 1], inside_positions[:, 0]] == 0
            )  # num_inside_voxels

            # combine both of these masks
            visible_voxels_1D_index = torch.where(inside_image)[
                0
            ]  # num_inside_voxels. flattened index among all voxels
            background_voxels_1D_index = visible_voxels_1D_index[
                is_background
            ]  # voxels that are inside and background

            # determine 3D indices of these background voxels using voxel_grid_index
            background_voxels_3D_index = voxel_grid_index[
                background_voxels_1D_index, :
            ]  # num_background_voxels x 3
            visible_voxels_3D_index = voxel_grid_index[visible_voxels_1D_index, :]

            # increment background_voxel_grid
            background_voxel_grid[
                background_voxels_3D_index[:, 0],
                background_voxels_3D_index[:, 1],
                background_voxels_3D_index[:, 2],
            ] += 1
            visible_voxel_grid[
                visible_voxels_3D_index[:, 0],
                visible_voxels_3D_index[:, 1],
                visible_voxels_3D_index[:, 2],
            ] += 1

        # threshold voxel grid to determine empty voxels
        background_grid = background_voxel_grid / visible_voxel_grid
        foreground_grid = 1 - background_grid
        # at least "visibility_threshold" many cameras need to see the voxel at all
        foreground_grid[torch.where(visible_voxel_grid <= visibility_threshold)] = 0
        # at least "fraction_threshold" percent many cameras need to declare the voxel foreground for it to be foreground
        foreground_grid[torch.where(foreground_grid < fraction_threshold)] = 0

        # dilate on voxel grid
        kernel_size = voxel_grid_dilation_kernel_size
        if kernel_size % 2 == 0:
            kernel_size += 1  # needed for integer padding
        padding = (kernel_size - 1) // 2
        foreground_grid = torch.nn.functional.max_pool3d(
            foreground_grid.unsqueeze(0),
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=1,
            ceil_mode=True,
            return_indices=False,
        ).squeeze(0)

        # studio bounding box
        if studio_bounding_box:
            original_shape = foreground_grid.shape
            foreground_grid = foreground_grid.view(-1)
            foreground_grid[~inside_studio_mask] = 0
            foreground_grid = foreground_grid.view(original_shape)

        # convert voxel grid datatype to boolean
        foreground_grid = foreground_grid > 0

        # visualize
        if visualize_voxel_grids:
            voxel_grid_folder = os.path.join(dataset, "voxel_grids")
            import pathlib

            pathlib.Path(voxel_grid_folder).mkdir(parents=True, exist_ok=True)
            output_file = os.path.join(voxel_grid_folder, "grid_" + str(current_timestep) + ".obj")
            _visualize_voxel_grid(
                output_file,
                foreground_grid,
                voxel_grid_pos=voxel_grid_pos.view(
                    voxel_grid_size, voxel_grid_size, voxel_grid_size, 3
                ),
                mode="grey",
                remove_zeros=True,
                has_unit_range=True,
            )

        # key for BinaryDataset: voxel_grid_size and timestamp
        from pruning import get_pruning_key

        key = get_pruning_key(voxel_grid_size, float(current_timestep))

        # store in BinaryDataset
        if use_binary_dataset:
            from io import BytesIO

            voxel_grid_bytes = BytesIO()
            np.savez_compressed(
                voxel_grid_bytes, foreground_voxel_grid=foreground_grid.cpu().numpy()
            )
            output_dataset.maybe_add_entry(voxel_grid_bytes, key=key)

    # close BinaryDataset
    if use_binary_dataset:
        output_dataset.close()

    _combine_pruned_voxel_grids(dataset=dataset, voxel_grid_size=voxel_grid_size)


def preprocess_dataset(input_folder, output_folder):

    # currently ignores pixel aspect ratios

    broken_camera_distortion_threshold = 0.2
    precompute_factors = [1, 2, 4, 8, 16]

    if input_folder != output_folder:
        create_folder(output_folder)
        shutil.copyfile(
            os.path.join(input_folder, "extrinsics.json"),
            os.path.join(output_folder, "extrinsics.json"),
        )
        shutil.copyfile(
            os.path.join(input_folder, "intrinsics.json"),
            os.path.join(output_folder, "intrinsics.json"),
        )
        shutil.copyfile(
            os.path.join(input_folder, "frame_to_extrin_intrin_time.json"),
            os.path.join(output_folder, "frame_to_extrin_intrin_time.json"),
        )

    # background
    LOGGER.info("processing background images...")
    with open(os.path.join(input_folder, "background_to_extrin_intrin.json"), "r") as json_file:
        background_to_extrin_intrin = json.load(json_file)

    from binary_dataset import BinaryDataset

    background_dataset = BinaryDataset(
        folder=output_folder, name="background", delete_existing=True
    )

    background_folder = os.path.join(input_folder, "background")
    background_images = sorted(
        os.listdir(background_folder)
    )  # sort is not required but makes it deterministic
    for filename in tqdm(background_images):

        background_id, extension = os.path.splitext(filename)

        # read
        if extension == "png":
            background_image = imageio.imread(
                os.path.join(background_folder, filename),
                ignoregamma=True,
            )
        else:
            background_image = imageio.imread(os.path.join(background_folder, filename))
        background_image = background_image.astype(np.uint8)

        # encode
        background_image = cv2.cvtColor(background_image, cv2.COLOR_RGB2BGR)
        _, encoded_image = cv2.imencode(
            ".jpeg", background_image, [cv2.IMWRITE_JPEG_QUALITY, 98]
        )  # this jpeg will be a normal RGB (not BGR) jpeg as long as background_image is BGR
        image_bytes = BytesIO(encoded_image)
        image_bytes.seek(0)

        # write
        from data_loader_blender import combine_rawextrinid_rawintrinid

        extrin_intrin = background_to_extrin_intrin[background_id]
        key = combine_rawextrinid_rawintrinid(
            extrin_intrin["extrin"], extrin_intrin["intrin"]
        )  # rawextrinid + rawintrinid
        background_dataset.maybe_add_entry(image_bytes, key)

    background_dataset.close()

    # images
    LOGGER.info("processing images...")
    dataset = BinaryDataset(folder=output_folder, name="dataset", delete_existing=True)

    images_folder = os.path.join(input_folder, "images")
    image_files = sorted(
        os.listdir(images_folder)
    )  # sort is not required but makes it deterministic
    for filename in tqdm(image_files):

        image_id, extension = os.path.splitext(filename)

        # read
        if extension == "png":
            image = imageio.imread(
                os.path.join(images_folder, filename),
                ignoregamma=True,
            )
        else:
            image = imageio.imread(os.path.join(images_folder, filename))
        image = image.astype(np.uint8)

        # encode
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        _, encoded_image = cv2.imencode(
            ".jpeg", image, [cv2.IMWRITE_JPEG_QUALITY, 98]
        )  # this jpeg will be a normal RGB (not BGR) jpeg as long as image is BGR
        image_bytes = BytesIO(encoded_image)
        image_bytes.seek(0)

        # write
        key = image_id
        dataset.maybe_add_entry(image_bytes, key)

    dataset.close()

    # precompute factors
    LOGGER.info("precompute downsampling...")
    from settings import config_parser
    from data_handler import DataHandler

    settings, _ = config_parser().parse_known_args()
    settings.datadir = output_folder
    settings.multi_gpu = False
    settings.use_background = True
    settings.do_vignetting_correction = False
    settings.debug = False
    data_handler = DataHandler(settings, precomputation_mode=True)
    all_imageids = torch.arange(data_handler.data_loader.num_total_images()).cuda()
    for factor in precompute_factors:
        data_handler.load_training_set(factor, num_pixels_per_image=5, imageids=all_imageids)

    # precompute voxel grids for pruning
    LOGGER.info("compute voxel grids for pruning...")
    _space_carving(
        dataset=output_folder, voxel_grid_size=128, debug=False, append_to_existing_file=False
    )


if __name__ == "__main__":

    # logging_level = logging.DEBUG
    logging_level = logging.INFO
    coloredlogs.install(level=logging_level, fmt="%(name)s[%(process)d] %(levelname)s %(message)s")
    logging.basicConfig(level=logging_level)

    if len(sys.argv) == 2:
        input_folder = sys.argv[1]
        output_folder = input_folder
    elif len(sys.argv) == 3:
        input_folder = sys.argv[1]
        output_folder = sys.argv[2]
    else:
        raise RuntimeError("takes either one or two arguments as input")

    preprocess_dataset(input_folder, output_folder)
