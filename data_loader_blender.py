# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import json

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import logging
import os

import imageio
import numpy as np
import torch
import tqdm
from binary_dataset import BinaryDataset
from data_loader import DataLoader
from multi_gpu import multi_gpu_barrier
from utils import check_scratch_for_dataset_copy

logging.getLogger("PIL").setLevel(logging.WARNING)
LOGGER = logging.getLogger(__name__)


def combine_rawextrinid_rawintrinid(rawextrinid, rawintrinid):
    return str(rawextrinid) + str(rawintrinid)


class DataLoaderBlender(DataLoader):
    def __init__(self, settings, rank, world_size):
        super().__init__()

        self.rank = rank
        self.world_size = world_size
        self.multi_gpu = settings.multi_gpu

        if settings.allow_scratch_datadir_copy:
            self.datadir = check_scratch_for_dataset_copy(settings.datadir)
        else:
            self.datadir = settings.datadir
        LOGGER.info("will use dataset in " + self.datadir)
        self.use_background = settings.use_background and os.path.exists(
            os.path.join(self.datadir, "background.bin")
        )

        if settings.do_pref and settings.pref_dataset_index >= 0:
            self._pref_dataset_index = settings.pref_dataset_index
            self._pref_num_timesteps_per_index = 25
            self._pref_index_overlap = 3
        else:
            self._pref_dataset_index = None

        self._init_imageid_to_extrin_intrin_timestep()
        self._init_intrinsics()
        self._init_extrinsics()

        def unquote(s):
            if s[0] in ["'", '"']:
                s = s[1:]
            if s[-1] in ["'", '"']:
                s = s[:-1]
            return s

        test_cameras = [unquote(test_camera) for test_camera in settings.test_cameras]
        self.test_imageids = np.array(
            [
                imageid
                for imageid in range(self.num_total_images())
                if self.extrinid_to_rawextrinid[self.imageid_to_extrinid[imageid]] in test_cameras
            ]
        )
        self.train_imageids = np.array(
            [
                imageid
                for imageid in range(self.num_total_images())
                if imageid not in self.test_imageids
            ]
        )

        few_deformation_cameras = False
        if few_deformation_cameras:
            extrinids = self.get_extrinids(self.train_imageids)
            timesteps = self.get_timesteps(self.train_imageids)
            few_cameras = torch.unique(extrinids)[::10]
            LOGGER.info([self.extrinid_to_rawextrinid[extrinid] for extrinid in few_cameras])
            LOGGER.info(len(self.train_imageids))
            self.train_imageids = np.array(
                [
                    imageid
                    for imageid, extrinid, timestep in zip(
                        self.train_imageids, extrinids, timesteps
                    )
                    if extrinid in few_cameras or timestep == 0.0
                ]
            )
            LOGGER.info(len(self.train_imageids))

        self._is_png = True

    def _init_pref_dataset_split(self, imageid_to_timestep):
        all_timesteps = sorted(list(set(list(imageid_to_timestep))))
        first_timestep = self._pref_dataset_index * (
            self._pref_num_timesteps_per_index - self._pref_index_overlap
        )
        last_timestep = (self._pref_dataset_index + 1) * (
            self._pref_num_timesteps_per_index - self._pref_index_overlap
        ) + self._pref_index_overlap

        timestep_subset = all_timesteps[first_timestep:last_timestep]
        LOGGER.info(str(first_timestep) + " " + str(last_timestep))
        LOGGER.info("timestep_subset: " + str(timestep_subset))
        if len(timestep_subset) == 0:
            LOGGER.warning("pref index out of bounds")
            raise RuntimeError("pref index out of bounds")
        self._pref_subset = [
            imageid
            for imageid, timestep in enumerate(imageid_to_timestep)
            if timestep in timestep_subset
        ]

    def _maybe_apply_pref_subset(self, some_imageid_list):
        if self._pref_dataset_index is None:
            return some_imageid_list
        if type(some_imageid_list) == type([]):
            # list
            subset = [x for index, x in enumerate(some_imageid_list) if index in self._pref_subset]
        else:
            # numpy array
            subset = some_imageid_list[np.array(self._pref_subset).astype(np.int32)]
        return subset

    def _init_imageid_to_extrin_intrin_timestep(self):

        multi_view_mapping = os.path.join(self.datadir, "frame_to_extrin_intrin_time.json")

        with open(multi_view_mapping, "r") as multi_view_mapping:
            multi_view_mapping = json.load(multi_view_mapping)

        raw_extrin_intrin_time_list = []
        for key in sorted(multi_view_mapping.keys()):
            raw_extrin_intrin_time_list.append(multi_view_mapping[key])

        # convert to consecutive numerical ids

        def as_tensor(some_list):
            return torch.from_numpy(np.array(some_list)).cuda()

        # time
        self.imageid_to_timestep = np.array(
            [extrin_intrin_time["time"] for extrin_intrin_time in raw_extrin_intrin_time_list]
        )
        # imageid_to_timestep
        if self._pref_dataset_index is not None:
            self._init_pref_dataset_split(self.imageid_to_timestep)
        self.imageid_to_timestep = self._maybe_apply_pref_subset(self.imageid_to_timestep)
        self._normalize_timesteps()
        self.imageid_to_timestep_torch = as_tensor(self.imageid_to_timestep).float()

        # extrinsics
        self.extrinid_to_rawextrinid = sorted(
            list(
                set(
                    [
                        extrin_intrin_time["extrin"]
                        for extrin_intrin_time in raw_extrin_intrin_time_list
                    ]
                )
            )
        )
        self.rawextrinid_to_extrinid = dict(
            [(extrin, i) for i, extrin in enumerate(self.extrinid_to_rawextrinid)]
        )
        self.imageid_to_extrinid = [
            self.rawextrinid_to_extrinid[extrin_intrin_time["extrin"]]
            for extrin_intrin_time in raw_extrin_intrin_time_list
        ]
        # imageid_to_extrinid
        self.imageid_to_extrinid = self._maybe_apply_pref_subset(self.imageid_to_extrinid)
        self.imageid_to_extrinid_torch = as_tensor(self.imageid_to_extrinid)

        # intrinsics
        self.intrinid_to_rawintrinid = sorted(
            list(
                set(
                    [
                        extrin_intrin_time["intrin"]
                        for extrin_intrin_time in raw_extrin_intrin_time_list
                    ]
                )
            )
        )
        self.rawintrinid_to_intrinid = dict(
            [(intrin, i) for i, intrin in enumerate(self.intrinid_to_rawintrinid)]
        )
        self.imageid_to_intrinid = [
            self.rawintrinid_to_intrinid[extrin_intrin_time["intrin"]]
            for extrin_intrin_time in raw_extrin_intrin_time_list
        ]
        # imageid_to_intrinid
        self.imageid_to_intrinid = self._maybe_apply_pref_subset(self.imageid_to_intrinid)
        self.imageid_to_intrinid_torch = as_tensor(self.imageid_to_intrinid)

        # extrinsics x intrinsics # for background
        self.exintrinid_to_rawexintrinid = sorted(
            list(
                set(
                    [
                        combine_rawextrinid_rawintrinid(
                            extrin_intrin_time["extrin"], extrin_intrin_time["intrin"]
                        )
                        for extrin_intrin_time in raw_extrin_intrin_time_list
                    ]
                )
            )
        )
        self.rawexintrinid_to_exintrinid = dict(
            [
                (rawexintrin, exintrinid)
                for exintrinid, rawexintrin in enumerate(self.exintrinid_to_rawexintrinid)
            ]
        )
        self.imageid_to_exintrinid = [
            self.rawexintrinid_to_exintrinid[
                combine_rawextrinid_rawintrinid(
                    extrin_intrin_time["extrin"], extrin_intrin_time["intrin"]
                )
            ]
            for extrin_intrin_time in raw_extrin_intrin_time_list
        ]
        # imageid_to_exintrinid
        self.imageid_to_exintrinid = self._maybe_apply_pref_subset(self.imageid_to_exintrinid)
        self.imageid_to_exintrinid_torch = as_tensor(self.imageid_to_exintrinid)

    def _normalize_timesteps(self):
        min_time, max_time = np.min(self.imageid_to_timestep), np.max(self.imageid_to_timestep)
        self.timeline_without_normalization = max_time - min_time
        if max_time != min_time:
            epsilon = 0.0  # needs to be 0.0!
            self.imageid_to_timestep = (self.imageid_to_timestep - min_time + epsilon) / (
                max_time - min_time + 2.0 * epsilon
            )  # in [0,1]
        else:
            self.imageid_to_timestep *= 0.0  # if min_time == max_time, all timesteps are set to 0.0

    def _init_intrinsics(self):
        with open(os.path.join(self.datadir, "intrinsics.json"), "r") as json_file:
            orig_intrinsics = json.load(json_file)
        self.intrinsics = {}
        for key in orig_intrinsics.keys():
            intrinid = self.rawintrinid_to_intrinid[key]
            self.intrinsics[intrinid] = orig_intrinsics[key]
            self.intrinsics[intrinid]["intrinid"] = intrinid

    def _init_extrinsics(self):
        with open(os.path.join(self.datadir, "extrinsics.json"), "r") as json_file:
            orig_extrinsics = json.load(json_file)

        if self.multi_gpu:
            multi_gpu_barrier(self.rank)

        self.extrinsics = {}
        for key in orig_extrinsics.keys():
            orig_extrin = orig_extrinsics[key]
            orig_extrin = {
                "translation": torch.from_numpy(
                    np.array(orig_extrin["translation"], dtype=np.float32)
                ).cuda(),
                "rotation": torch.from_numpy(
                    np.array(orig_extrin["rotation"], dtype=np.float32)
                ).cuda(),
                "near": orig_extrin["near"],
                "far": orig_extrin["far"],
            }
            self.extrinsics[self.rawextrinid_to_extrinid[key]] = orig_extrin

    # returns numpy arrays
    def images_iterator(self, factor, imageids=None):

        dataset_index = os.path.join(self.datadir, "dataset_index.json")
        if os.path.exists(dataset_index):
            return self._images_iterator_binary(factor=factor, imageids=imageids)
        else:
            return self._images_iterator_png(factor=factor, imageids=imageids)

    def _images_iterator_png(self, factor, imageids=None):

        if factor == 1:
            images_folder = os.path.join(self.datadir, "images")
        else:
            images_folder = os.path.join(self.datadir, "images_" + str(factor))
            if not os.path.exists(images_folder):
                if self.rank == 0:
                    self._downsample_images_png(images_folder, factor, dataset_index=None)
                if self.multi_gpu:
                    multi_gpu_barrier(self.rank)

        image_files = sorted(os.listdir(images_folder))
        # image_files
        image_files = self._maybe_apply_pref_subset(image_files)
        if imageids is not None:
            image_files = [image_files[imageid] for imageid in imageids]

        for i in tqdm.trange(len(image_files)):
            image_file = image_files[i]
            image = imageio.imread(
                os.path.join(images_folder, image_file),
                ignoregamma=True,
            )
            yield image_file, image[..., :3].astype(np.float32) / 255.0  # throw away alpha channel

    def _downsample_images_png(self, images_folder, factor):
        import cv2

        os.makedirs(images_folder)
        for image_file, image in self.images_iterator(factor=1):

            image_filename = os.path.split(image_file)[1]
            output_file = os.path.join(images_folder, image_filename)

            H, W = image.shape[:2]
            H = H // factor
            W = W // factor

            downsampled_image = cv2.resize(image, (W, H), interpolation=cv2.INTER_AREA)

            imageio.imsave(output_file, (255.0 * downsampled_image).astype(np.uint8))

    def _images_iterator_binary(self, factor, imageids=None, dataset_index=None):

        if factor == 1:
            dataset_file = os.path.join(self.datadir, "dataset.bin")
            dataset_index = os.path.join(self.datadir, "dataset_index.json")
        else:
            dataset_file = os.path.join(self.datadir, "dataset_" + str(factor) + ".bin")
            dataset_index = os.path.join(self.datadir, "dataset_index_" + str(factor) + ".json")
            if not os.path.exists(dataset_file):
                if self.rank == 0:
                    self._downsample_images_binary(dataset_file, dataset_index, factor)
                if self.multi_gpu:
                    multi_gpu_barrier(self.rank)

        with open(dataset_index, "r") as dataset_index:
            dataset_index = json.load(dataset_index)

        image_files = sorted(list(dataset_index.keys()))
        # image_files
        image_files = self._maybe_apply_pref_subset(image_files)
        if imageids is not None:
            image_files = [image_files[imageid] for imageid in imageids]

        with open(dataset_file, "rb") as dataset_bin:
            for i in tqdm.trange(len(image_files)):
                image_file = image_files[i]
                dataset_bin.seek(dataset_index[image_file]["start"])
                image_bytes = dataset_bin.read(
                    dataset_index[image_file]["end"] - dataset_index[image_file]["start"]
                )
                try:
                    if self._is_png:
                        image = imageio.imread(image_bytes, ignoregamma=True)  # png
                    else:
                        image = imageio.imread(image_bytes)  # jpg
                except TypeError:
                    self._is_png = False
                    image = imageio.imread(image_bytes)
                    LOGGER.debug("using jpg during data loading")
                yield image_file, image[..., :3].astype(
                    np.float32
                ) / 255.0  # throw away alpha channel

    def _downsample_images_binary(self, dataset_file, dataset_index_file, factor):
        import cv2
        from io import BytesIO

        dataset_index = {}
        start = 0
        with open(dataset_file, "wb") as dataset_bin:
            for image_file, image in self.images_iterator(factor=1):

                image_filename = os.path.split(image_file)[1]

                H, W = image.shape[:2]
                H = H // factor
                W = W // factor

                downsampled_image = cv2.resize(image, (W, H), interpolation=cv2.INTER_AREA)
                downsampled_image = (255.0 * downsampled_image).astype(np.uint8)

                if self._is_png:
                    image_bytes = BytesIO()
                    imageio.imwrite(image_bytes, downsampled_image, format="png")
                else:
                    downsampled_image = cv2.cvtColor(downsampled_image, cv2.COLOR_RGB2BGR)
                    _, encoded_image = cv2.imencode(
                        ".jpeg", downsampled_image, [cv2.IMWRITE_JPEG_QUALITY, 98]
                    )  # this jpeg will be a normal RGB (not BGR) jpeg as long as downsampled_image is BGR
                    image_bytes = BytesIO(encoded_image)
                    image_bytes.seek(0)

                end = start + image_bytes.getbuffer().nbytes
                dataset_index[image_filename] = {
                    "start": start,  # inclusive
                    "end": end,  # exclusive
                }
                start = end

                dataset_bin.write(image_bytes.getbuffer())

        with open(dataset_index_file, "w", encoding="utf-8") as json_file:
            json.dump(dataset_index, json_file, ensure_ascii=False, indent=4)

    # return numpy array
    def load_images(
        self, factor=None, imageids=None, desired_subsets=None, background_images_dict=None
    ):

        if imageids is None:
            imageids = self.train_imageids
        if factor is None:
            factor = 1

        num_images = len(imageids)

        if desired_subsets is None:
            desired_subsets = {"everything": {"mode": "all"}}

        rgb_subsets = {subset_name: None for subset_name in desired_subsets.keys()}
        coordinates_subsets = {subset_name: None for subset_name in desired_subsets.keys()}

        for total_index, (_, image) in enumerate(
            self.images_iterator(factor=factor, imageids=imageids)
        ):

            for subset_name, desired_subset in desired_subsets.items():

                mode = desired_subset["mode"]
                height, width = image.shape[:2]

                def get_background_mask():

                    if background_images_dict is None:
                        raise RuntimeError("require background images!")

                    exintrinid = background_images_dict["exintrinids"][total_index]
                    background_image = background_images_dict["background"][exintrinid]

                    if "background_threshold" in background_images_dict:
                        background_threshold = background_images_dict["background_threshold"]
                    else:
                        background_threshold = 5.0 / 255.0

                    difference_image = torch.mean(
                        torch.abs(torch.from_numpy(image) - background_image), axis=-1
                    )  # height x width
                    background_mask = difference_image < background_threshold

                    return background_mask, background_image

                if mode == "background_masks":

                    background_mask, _ = get_background_mask()
                    y_coordinates = None

                elif mode == "random":
                    num_pixels_per_image = desired_subset["num_pixels_per_image"]
                    if (
                        num_pixels_per_image >= height * width
                    ):  # keep all pixels, so no need to randomly permute them
                        y_coordinates = None
                        x_coordinates = None
                        if total_index == 0:
                            LOGGER.info("can use all pixels out of " + str(height * width))
                    else:
                        random_flattened_indices = torch.randperm(n=height * width)[
                            :num_pixels_per_image
                        ]
                        y_coordinates = torch.div(
                            random_flattened_indices, width, rounding_mode="floor"
                        )
                        x_coordinates = random_flattened_indices % width
                        if total_index == 0:
                            LOGGER.info(
                                "using "
                                + str(len(random_flattened_indices))
                                + " pixels out of "
                                + str(height * width)
                            )

                elif mode == "foreground_focused":

                    background_mask, background_image = get_background_mask()

                    if "write_out_masks" in desired_subset:  # hacky flag for debugging
                        some_folder = os.path.join(self.datadir, "debug")
                        import pathlib

                        pathlib.Path(some_folder).mkdir(parents=True, exist_ok=True)
                        imageio.imwrite(
                            os.path.join(some_folder, str(total_index) + ".jpg"),
                            (255 * background_mask.numpy()).astype(np.uint8),
                        )
                        imageio.imwrite(
                            os.path.join(some_folder, str(total_index) + "_bg.jpg"),
                            (255 * background_image.numpy()).astype(np.uint8),
                        )
                        imageio.imwrite(
                            os.path.join(some_folder, str(total_index) + "_gt.jpg"),
                            (255 * image).astype(np.uint8),
                        )

                    num_pixels_per_image = desired_subset["num_pixels_per_image"]
                    num_pixels_per_image = min(num_pixels_per_image, 3 * height * width)
                    num_foreground_pixels = int(
                        num_pixels_per_image * desired_subset["foreground_fraction"]
                    )

                    all_y_foreground, all_x_foreground = torch.where(~background_mask)
                    if len(all_y_foreground) > 0:  # some foreground exists
                        if len(all_y_foreground) == height * width:  # no background exists
                            num_foreground_pixels = num_pixels_per_image
                        foreground_indices = torch.randint(
                            low=0, high=len(all_y_foreground), size=(num_foreground_pixels,)
                        )
                        y_foreground = all_y_foreground[foreground_indices]
                        x_foreground = all_x_foreground[foreground_indices]
                    else:  # no foreground exists
                        y_foreground = torch.empty((0,), dtype=torch.int64)
                        x_foreground = torch.empty((0,), dtype=torch.int64)

                    # fill up with background pixels
                    num_background_pixels = num_pixels_per_image - len(y_foreground)
                    if num_background_pixels > 0:
                        all_y_background, all_x_background = torch.where(background_mask)
                        background_indices = torch.randint(
                            low=0, high=len(all_y_background), size=(num_background_pixels,)
                        )
                        y_background = all_y_background[background_indices]
                        x_background = all_x_background[background_indices]
                    else:
                        y_background = torch.empty((0,), dtype=torch.int64)
                        x_background = torch.empty((0,), dtype=torch.int64)

                    y_coordinates = torch.cat([y_foreground, y_background], axis=-1)
                    x_coordinates = torch.cat([x_foreground, x_background], axis=-1)

                elif mode == "specific":
                    y_coordinates = torch.from_numpy(
                        np.array(desired_subset["y_coordinates"])
                    ).long()
                    x_coordinates = torch.from_numpy(
                        np.array(desired_subset["x_coordinates"])
                    ).long()

                elif mode == "all":
                    y_coordinates = None
                    x_coordinates = None

                # coordinates
                if y_coordinates is None:
                    if mode == "background_masks":
                        this_image = background_mask
                    else:
                        this_image = image
                else:
                    # initialization
                    if coordinates_subsets[subset_name] is None:
                        num_coordinates = y_coordinates.shape[0]
                        coordinates_subsets[subset_name] = torch.empty(
                            (num_images, num_coordinates, 2), dtype=torch.int64, device="cpu"
                        )
                    # store
                    coordinates_subsets[subset_name][total_index, :, 0] = y_coordinates
                    coordinates_subsets[subset_name][total_index, :, 1] = x_coordinates

                    this_image = image[y_coordinates, x_coordinates]

                    if "with_background_info" in desired_subset:
                        this_background_mask = (
                            background_mask[y_coordinates, x_coordinates].unsqueeze(-1).float()
                        )
                        this_background_image = background_image[
                            y_coordinates, x_coordinates
                        ].float()
                        this_image = np.concatenate(
                            [this_image, this_background_image, this_background_mask], axis=-1
                        )

                # RGB
                # initialization
                if rgb_subsets[subset_name] is None:
                    rgb_subsets[subset_name] = np.empty(
                        (num_images,) + this_image.shape, dtype=np.float32
                    )

                # store
                rgb_subsets[subset_name][total_index] = this_image

        return rgb_subsets, coordinates_subsets  # N x H x W x 3 or N x num_pixels_per_image x 3

    ### BACKGROUND

    def has_background_images(self):
        return self.use_background

    # returns numpy arrays
    def background_images_iterator(self, factor, exintrinids=None):

        if factor == 1:
            name = "background"
        else:
            name = "background_" + str(factor)
            dataset_file = os.path.join(self.datadir, name + ".bin")
            if not os.path.exists(dataset_file):
                if self.rank == 0:
                    self._downsample_background_images(name, factor)
                if self.multi_gpu:
                    multi_gpu_barrier(self.rank)

        background_dataset = BinaryDataset(folder=self.datadir, name=name)

        rawexintrinids = sorted(list(background_dataset.keys()))
        if exintrinids is not None:
            rawexintrinids = [
                self.exintrinid_to_rawexintrinid[exintrinid] for exintrinid in exintrinids
            ]

        for i in tqdm.trange(len(rawexintrinids)):
            rawexintrinid = rawexintrinids[i]
            image_bytes = background_dataset.get_entry(key=rawexintrinid)
            try:
                if self._is_png:
                    image = imageio.imread(image_bytes, ignoregamma=True)  # png
                else:
                    image = imageio.imread(image_bytes)  # jpg
            except TypeError:
                self._is_png = False
                image = imageio.imread(image_bytes)
                LOGGER.debug("using jpg during data loading")
            yield rawexintrinid, image[..., :3].astype(
                np.float32
            ) / 255.0  # throw away alpha channel

    def _downsample_background_images(self, name, factor):
        import cv2
        from io import BytesIO

        background_dataset = BinaryDataset(folder=self.datadir, name=name)

        for rawexintrinid, image in self.background_images_iterator(factor=1):

            H, W = image.shape[:2]
            H = H // factor
            W = W // factor

            downsampled_image = cv2.resize(image, (W, H), interpolation=cv2.INTER_AREA)
            downsampled_image = (255.0 * downsampled_image).astype(np.uint8)

            if self._is_png:
                image_bytes = BytesIO()
                imageio.imwrite(image_bytes, downsampled_image, format="png")
            else:
                downsampled_image = cv2.cvtColor(downsampled_image, cv2.COLOR_RGB2BGR)
                _, encoded_image = cv2.imencode(
                    ".jpeg", downsampled_image, [cv2.IMWRITE_JPEG_QUALITY, 98]
                )  # this jpeg will be a normal RGB (not BGR) jpeg as long as downsampled_image is BGR
                image_bytes = BytesIO(encoded_image)
                image_bytes.seek(0)

            background_dataset.maybe_add_entry(image_bytes, key=rawexintrinid)

        background_dataset.close()

    # return numpy array
    def load_background_images(self, factor=None, exintrinids=None):

        if factor is None:
            factor = 1
        if exintrinids is None:
            exintrinids = torch.arange(len(self.exintrinid_to_rawexintrinid)).long().cuda()

        num_images = len(exintrinids)

        images_array = None

        for total_index, (_, image) in enumerate(
            self.background_images_iterator(factor=factor, exintrinids=exintrinids)
        ):

            if images_array is None:
                images_array = np.empty((num_images,) + image.shape, dtype=np.float32)

            images_array[total_index] = image

        return images_array  # N x H x W x 3

    # uses pytorch
    def get_extrinids(self, imageids):
        return self.imageid_to_extrinid_torch[imageids]

    def get_intrinids(self, imageids):
        return self.imageid_to_intrinid_torch[imageids]

    def get_exintrinids(self, imageids):
        return self.imageid_to_exintrinid_torch[imageids]

    def get_timesteps(self, imageids):
        return self.imageid_to_timestep_torch[imageids]

    def get_extrinsics(self, extrinids):
        return [self.extrinsics[extrinid] for extrinid in extrinids.cpu().numpy()]

    def get_intrinsics(self, intrinids=None, factor=None):
        if intrinids is None:
            intrinids = torch.arange(max(list(self.intrinsics.keys())) + 1)

        if factor is None:
            factor = 1

        if factor == 1:
            intrinsics = self.intrinsics
        else:
            intrinsics = {}
            for intrinid, intrin in self.intrinsics.items():
                new_height = intrin["height"] // factor
                new_width = intrin["width"] // factor
                exact_factor_x = new_width / intrin["width"]
                exact_factor_y = new_height / intrin["height"]
                new_intrin = {
                    "center_x": exact_factor_x * intrin["center_x"],
                    "center_y": exact_factor_y * intrin["center_y"],
                    "focal_x": exact_factor_x * intrin["focal_x"],
                    "focal_y": exact_factor_y * intrin["focal_y"],
                    "height": new_height,
                    "width": new_width,
                    "intrinid": intrin["intrinid"],
                }
                if "distortion" in intrin:
                    new_intrin["distortion"] = intrin["distortion"]
                intrinsics[intrinid] = new_intrin

        return [intrinsics[intrinid] for intrinid in intrinids.cpu().numpy()]

    def num_total_images(self):
        return len(self.imageid_to_intrinid)

    def get_timeline_range(self):
        return self.timeline_without_normalization

    def get_train_imageids(self):
        return self.train_imageids

    def get_test_imageids(self):
        return self.test_imageids

    def get_dataset_folder(self):
        return self.datadir
