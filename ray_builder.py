# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import logging

import numpy as np
import torch
from binary_dataset import BinaryDataset
from tqdm import trange
from utils import szudzik
from multi_gpu import multi_gpu_barrier

LOGGER = logging.getLogger(__name__)


class RayBuilder:
    def __init__(self, rank=0, multi_gpu=False, max_num_precomputed_undistortions=None):

        if max_num_precomputed_undistortions is None:
            max_num_precomputed_undistortions = 1000
        self.max_num_precomputed_undistortions = max_num_precomputed_undistortions
        self.precomputed_undistortion = {}

        self.rank = rank
        self.multi_gpu = multi_gpu

        self._precomputed_dataset = None
        self.only_use_precomputed_dataset_for_nonzero_distortions = True

    def use_precomputed_dataset(self, dataset_folder, create=False):

        if create:
            if self.rank == 0:
                self._precomputed_dataset = BinaryDataset(
                    dataset_folder, name="precomputed_rays", read_only=False
                )
                self._precomputed_dataset.flush()
            if self.multi_gpu:
                multi_gpu_barrier(self.rank)  # wait for dataset creation if not existing
                if self.rank > 0:
                    self._precomputed_dataset = BinaryDataset(
                        dataset_folder, name="precomputed_rays", read_only=True
                    )
        else:
            self._precomputed_dataset = BinaryDataset(
                dataset_folder, name="precomputed_rays", read_only=True
            )

    def undistort(self, intrin, i, j):
        original_i_shape = i.shape
        original_j_shape = j.shape

        i = i.reshape(-1)
        j = j.reshape(-1)

        target_i = i.clone()
        target_j = j.clone()

        def _get_value(name):
            if name in intrin["distortion"]:
                return intrin["distortion"][name]
            else:
                return 0.0

        k1 = _get_value("k1")
        k2 = _get_value("k2")
        p1 = _get_value("p1")
        p2 = _get_value("p2")
        k3 = _get_value("k3")
        s1 = _get_value("s1")
        s2 = _get_value("s2")
        s3 = _get_value("s3")
        s4 = _get_value("s4")

        if all(x == 0.0 for x in [k1, k2, p1, p2, k3, s1, s2, s3, s4]):
            max_num_iterations = 0
            mean_error_i = torch.zeros(1)
            mean_error_j = torch.zeros(1)
        else:
            max_num_iterations = 20000
        custom_max_update = 2.0  # if above this value, downscale the update values to this value
        stability_mask_threshold = 1e-10
        convergence_threshold = 1e-8  # stop optimization if mean error below this threshold
        acceptance_threshold = (
            1e-4  # after optimization is over, accept the result if its error is below this value
        )

        for _current_iter in range(max_num_iterations):

            if _current_iter % 10000 == 0 and _current_iter > 0:
                custom_max_update /= 5.0

            # components of current position
            r = i * i + j * j
            radial = k1 * r + k2 * r * r + k3 * r * r * r
            tangential_i = 2.0 * p1 * i * j + p2 * (r + 2.0 * i * i)
            tangential_j = p1 * (r + 2.0 * j * j) + 2.0 * p2 * i * j
            thin_prism_i = s1 * r + s2 * r * r
            thin_prism_j = s3 * r + s4 * r * r

            current_i = i + radial * i + tangential_i + thin_prism_i
            current_j = j + radial * j + tangential_j + thin_prism_j

            # residual

            error_i = current_i - target_i
            error_j = current_j - target_j

            # build 2x2 Jacobi matrix (error_i and error_j wrt. i and j)

            d_radial_wrt_i = (k1 + 2.0 * k2 * r + 3 * k3 * r * r) * 2.0 * i
            d_radial_wrt_j = (k1 + 2.0 * k2 * r + 3 * k3 * r * r) * 2.0 * j

            # i wrt i
            d_tangential_i_wrt_i = 2.0 * p1 * j + p2 * 6.0 * i
            d_thin_prism_i_wrt_i = s1 * 2.0 * i + (s2 * 2.0 * r) * 2.0 * i
            d_current_i_wrt_i = (
                1.0
                + (d_radial_wrt_i * i + radial * 1.0)
                + d_tangential_i_wrt_i
                + d_thin_prism_i_wrt_i
            )

            # i wrt j
            d_tangential_i_wrt_j = 2.0 * p1 * i + p2 * 2.0 * j
            d_thin_prism_i_wrt_j = s1 * 2.0 * j + (s2 * 2.0 * r) * 2.0 * j
            d_current_i_wrt_j = d_radial_wrt_j * i + d_tangential_i_wrt_j + d_thin_prism_i_wrt_j

            # j wrt i
            d_tangential_j_wrt_i = p1 * 2.0 * i + 2.0 * p2 * j
            d_thin_prism_j_wrt_i = s3 * 2.0 * i + (s4 * 2.0 * r) * 2.0 * i
            d_current_j_wrt_i = d_radial_wrt_i * j + d_tangential_j_wrt_i + d_thin_prism_j_wrt_i

            # j wrt j
            d_tangential_j_wrt_j = p1 * 6.0 * j + 2.0 * p2 * i
            d_thin_prism_j_wrt_j = s3 * 2.0 * j + (s4 * 2.0 * r) * 2.0 * j
            d_current_j_wrt_j = (
                1.0
                + (d_radial_wrt_j * j + radial * 1.0)
                + d_tangential_j_wrt_j
                + d_thin_prism_j_wrt_j
            )

            # Gauss-Newton with n=m
            denominator = (
                d_current_i_wrt_i * d_current_j_wrt_j - d_current_i_wrt_j * d_current_j_wrt_i
            )
            update_i = d_current_j_wrt_j * error_i - d_current_i_wrt_j * error_j
            update_j = -d_current_j_wrt_i * error_i + d_current_i_wrt_i * error_j

            update_i /= denominator
            update_j /= denominator

            # update
            stability_mask = torch.abs(denominator) > stability_mask_threshold

            max_update = torch.max(
                torch.abs(torch.cat([update_i[stability_mask], update_j[stability_mask]], 0))
            )
            if max_update > custom_max_update:
                update_i[stability_mask] = torch.clamp(
                    update_i[stability_mask], min=-custom_max_update, max=custom_max_update
                )
                update_j[stability_mask] = torch.clamp(
                    update_j[stability_mask], min=-custom_max_update, max=custom_max_update
                )

            i[stability_mask] -= update_i[stability_mask]
            j[stability_mask] -= update_j[stability_mask]

            mean_error_i = torch.mean(torch.abs(error_i))
            mean_error_j = torch.mean(torch.abs(error_j))
            if mean_error_i < convergence_threshold and mean_error_j < convergence_threshold:
                break

        LOGGER.debug(
            "undistortion error for "
            + str(intrin["intrinid"])
            + ": "
            + str(mean_error_i.item())
            + " "
            + str(mean_error_j.item())
        )
        if (
            not torch.isfinite(mean_error_i)
            or mean_error_i > acceptance_threshold
            or not torch.isfinite(mean_error_j)
            or mean_error_j > acceptance_threshold
        ):
            LOGGER.warning("did not converge: " + str([k1, k2, p1, p2, k3, s1, s2, s3, s4]))
            LOGGER.warning(
                "undistortion error for "
                + str(intrin["intrinid"])
                + ": "
                + str(mean_error_i.item())
                + " "
                + str(mean_error_j.item())
            )
            raise RuntimeError("undistortion did not converge")

        i = i.reshape(original_i_shape)
        j = j.reshape(original_j_shape)

        return i, j

    def _convert_intrin_to_key(self, intrin):

        a = intrin["intrinid"]
        b = intrin["height"]  # a proxy for the image rescaling "factor"

        key = szudzik(a, b)
        return key

    def maybe_get_precomputed_undistortion(self, intrin, device):

        key = self._convert_intrin_to_key(intrin)

        if key in self.precomputed_undistortion:
            i, j = self.precomputed_undistortion[key]
            return i.clone().to(device), j.clone().to(device)

        if self.only_use_precomputed_dataset_for_nonzero_distortions and (
            "distortion" not in intrin
            or ("distortion" in intrin and all(x == 0 for x in intrin["distortion"].values()))
        ):
            return None, None

        if self._precomputed_dataset is not None:
            if key in self._precomputed_dataset:
                i_j_bytes = self._precomputed_dataset.get_entry(key)
                try:
                    i_j = np.frombuffer(i_j_bytes, dtype=np.float32)
                except Exception as exception:
                    LOGGER.warning("failed at: " + str(intrin))
                    raise exception
                i, j = torch.from_numpy(i_j["i"]), torch.from_numpy(i_j["j"])
                self.maybe_store_precomputed_undistortion(
                    intrin, i, j
                )  # maybe load into RAM dictionary (self.precomputed_undistortion)
                return i.clone().to(device), j.clone().to(
                    device
                )  # in case it didn't get stored, need to return here

        return None, None

    def maybe_store_precomputed_undistortion(self, intrin, i, j):

        key = self._convert_intrin_to_key(intrin)

        if (
            key not in self.precomputed_undistortion
            and len(self.precomputed_undistortion) < self.max_num_precomputed_undistortions
        ):
            self.precomputed_undistortion[key] = (i.clone().cpu(), j.clone().cpu())

        if self.only_use_precomputed_dataset_for_nonzero_distortions and (
            "distortion" not in intrin
            or ("distortion" in intrin and all(x == 0 for x in intrin["distortion"].values()))
        ):
            return

        if self._precomputed_dataset is not None:
            if key not in self._precomputed_dataset:
                from io import BytesIO

                i_j_bytes = BytesIO()
                np.savez_compressed(
                    i_j_bytes,
                    i=i.cpu().numpy().astype(np.float32),
                    j=j.cpu().numpy().astype(np.float32),
                )
                self._precomputed_dataset.maybe_add_entry(i_j_bytes, key=key)
                self._precomputed_dataset.flush()

    def build(self, extrin, intrin):

        device = extrin["rotation"].device

        i, j = self.maybe_get_precomputed_undistortion(intrin, device)
        if i is None:
            if self.rank == 0:

                # (0, 0) is top left (?)
                i, j = torch.meshgrid(
                    torch.linspace(0, intrin["width"] - 1, intrin["width"], device=device),
                    torch.linspace(0, intrin["height"] - 1, intrin["height"], device=device),
                    indexing="ij",
                )  # pytorch's meshgrid has indexing='ij'
                i = i.t()
                j = j.t()

                i = (i - intrin["center_x"]) / intrin["focal_x"]
                j = (j - intrin["center_y"]) / intrin["focal_y"]

                if "distortion" in intrin:
                    i, j = self.undistort(intrin, i, j)

                self.maybe_store_precomputed_undistortion(intrin, i, j)

            if self.multi_gpu:
                multi_gpu_barrier(self.rank)
                if self.rank > 0:
                    self._precomputed_dataset.flush()  # get updated dataset with current undistortion
                    i, j = self.maybe_get_precomputed_undistortion(intrin, device)

        dirs = torch.stack([i, -j, -torch.ones_like(i, device=device)], -1)
        # Rotate ray directions from camera frame to the world frame
        rays_dir = torch.sum(
            dirs[..., np.newaxis, :] * extrin["rotation"], -1
        )  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
        # Translate camera frame's origin to the world frame. It is the origin of all rays.
        rays_origin = extrin["translation"].expand(rays_dir.shape)

        return {
            "rays_origin": rays_origin,  # pytorch, H x W x 3
            "rays_dir": rays_dir,
        }

    def build_multiple(self, extrins, intrins, coordinate_subsets=None):

        num_images = len(extrins)

        if coordinate_subsets is None:
            coordinate_subsets = {"everything": None}

        rays_origin_subsets = {subset_name: None for subset_name in coordinate_subsets.keys()}
        rays_dir_subsets = {subset_name: None for subset_name in coordinate_subsets.keys()}

        for index in trange(len(extrins)):

            extrin = extrins[index]
            intrin = intrins[index]

            rays_dict = self.build(extrin, intrin)
            rays_origin = rays_dict["rays_origin"]
            rays_dir = rays_dict["rays_dir"]

            for subset_name, coordinate_subset in coordinate_subsets.items():

                if coordinate_subset is None:
                    this_rays_origin = rays_origin
                    this_rays_dir = rays_dir
                else:
                    y_coordinates = coordinate_subset[index, :, 0]
                    x_coordinates = coordinate_subset[index, :, 1]
                    this_rays_origin = rays_origin[y_coordinates, x_coordinates]
                    this_rays_dir = rays_dir[y_coordinates, x_coordinates]

                if rays_origin_subsets[subset_name] is None:
                    rays_origin_subsets[subset_name] = torch.empty(
                        (num_images,) + this_rays_origin.shape, dtype=torch.float32, device="cpu"
                    )
                    rays_dir_subsets[subset_name] = torch.empty(
                        (num_images,) + this_rays_dir.shape, dtype=torch.float32, device="cpu"
                    )

                rays_origin_subsets[subset_name][index] = this_rays_origin.cpu()
                rays_dir_subsets[subset_name][index] = this_rays_dir.cpu()

        return {
            "rays_origin": rays_origin_subsets,  # torch cpu, N x H x W x 3
            "rays_dir": rays_dir_subsets,
        }
