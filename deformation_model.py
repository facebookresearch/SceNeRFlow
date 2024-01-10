# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch


def get_deformation_model(settings, timeline_range):
    from deformation_model_ngp import DeformationModelNGP

    if settings.backbone == "ngp":
        return DeformationModelNGP(settings, timeline_range)


class DeformationModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def viewdirs_via_finite_differences(self, positions, returns):
        # positions: num_rays x points_per_ray x 3
        num_rays, points_per_ray, _ = positions.shape

        eps = 1e-6
        difference_type = "backward"
        if difference_type == "central":
            # central differences (except for first and last sample since one neighbor is missing for them)
            unnormalized_central_differences = (
                positions[:, 2:, :] - positions[:, :-2, :]
            )  # rays x (samples-2) x 3
            central_differences = unnormalized_central_differences / (
                torch.norm(unnormalized_central_differences, dim=-1, keepdim=True) + eps
            )
            # fill in first and last sample by duplicating neighboring direction
            view_directions = torch.cat(
                [
                    central_differences[:, 0, :].view(-1, 1, 3),
                    central_differences,
                    central_differences[:, -1, :].view(-1, 1, 3),
                ],
                axis=1,
            )  # rays x samples x 3
        elif difference_type == "backward":
            unnormalized_backward_differences = (
                positions[:, 1:, :] - positions[:, :-1, :]
            )  # rays x (samples-1) x 3. 0-th sample has no direction.
            backward_differences = unnormalized_backward_differences / (
                torch.norm(unnormalized_backward_differences, dim=-1, keepdim=True) + eps
            )
            # fill in first sample by duplicating neighboring direction
            view_directions = torch.cat(
                [backward_differences[:, 0, :].view(-1, 1, 3), backward_differences],
                axis=1,
            )  # rays x samples x 3

        return view_directions

    def _apply_se3(self, undeformed_positions, network_output):
        w, v, pivot, translation = torch.split(
            network_output, [3, 3, 3, 3], dim=1
        )  # all: num_points x 3
        eps = 10e-7
        theta = torch.norm(w, dim=-1, keepdim=True) + eps  # num_points x 1
        w = w / theta
        v = v / theta
        skew = torch.zeros((w.shape[0], 3, 3), device=w.device)
        skew[:, 0, 1] = -w[:, 2]
        skew[:, 0, 2] = w[:, 1]
        skew[:, 1, 0] = w[:, 2]
        skew[:, 1, 2] = -w[:, 0]
        skew[:, 2, 0] = -w[:, 1]
        skew[:, 2, 1] = w[:, 0]
        eye = torch.zeros((w.shape[0], 3, 3), device=w.device)
        eye[:, 0, 0] = 1.0
        eye[:, 1, 1] = 1.0
        eye[:, 2, 2] = 1.0
        skew_squared = torch.matmul(skew, skew)
        exp_so3 = (
            eye
            + torch.sin(theta).view(-1, 1, 1) * skew
            + (1.0 - torch.cos(theta)).view(-1, 1, 1) * skew_squared
        )  # num_points x 3 x 3
        p = (
            theta.view(-1, 1, 1) * eye
            + (1.0 - torch.cos(theta)).view(-1, 1, 1) * skew
            + (theta - torch.sin(theta)).view(-1, 1, 1) * skew_squared
        )  # num_points x 3 x 3
        p = torch.matmul(p, v.view(-1, 3, 1))  # num_points x 3 x 1
        se3_transform = torch.cat([exp_so3, p], -1)  # num_points x 3 x 4
        se3_transform = torch.cat(
            [
                se3_transform,
                torch.zeros((se3_transform.shape[0], 1, 4), device=se3_transform.device),
            ],
            1,
        )  # num_points x 4 x 4
        se3_transform[:, 3, 3] = 1.0
        warped_pts = undeformed_positions + pivot  # num_points x 3
        # in homogenuous coordinates
        warped_pts = torch.cat(
            [
                warped_pts,
                torch.ones((warped_pts.shape[0], 1), device=warped_pts.device),
            ],
            -1,
        )
        warped_pts = torch.matmul(se3_transform, warped_pts.view(-1, 4, 1)).view(
            -1, 4
        )  # num_points x 4
        warped_pts = warped_pts[:, :3] / warped_pts[:, 3].view(-1, 1)  # num_points x 3

        warped_pts = warped_pts - pivot
        warped_pts = warped_pts + translation
        unmasked_offsets = warped_pts - undeformed_positions
        return unmasked_offsets
