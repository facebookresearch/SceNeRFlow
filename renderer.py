# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import logging

import torch
from post_correction import PostCorrection
from pre_correction import PreCorrection
from utils import Returns, project_to_correct_range

LOGGER = logging.getLogger(__name__)


class Renderer(torch.nn.Module):
    def __init__(self, settings):
        super().__init__()

        self.pre_correction = PreCorrection(settings)
        self.post_correction = PostCorrection(settings)

        self.points_per_chunk = settings.points_per_chunk
        self.default_num_points_per_ray = settings.num_points_per_ray
        self.default_disparity_sampling = settings.disparity_sampling  # boolean
        self.raw_noise_std = settings.raw_noise_std
        self.do_ngp_mip_nerf = settings.do_ngp_mip_nerf

        self.use_half_precision = settings.use_half_precision

    def _generate_points_on_rays(
        self, batch, num_points_per_ray, scene, is_training, returns, disparity_sampling=None
    ):

        if disparity_sampling is None:
            disparity_sampling = self.default_disparity_sampling

        device = batch["rays_origin"].device
        num_rays = batch["rays_origin"].shape[0]

        # near/far
        t_vals = torch.linspace(0.0, 1.0, steps=num_points_per_ray, device=device)
        t_vals = t_vals.expand([num_rays, num_points_per_ray])  # num_rays x num_points_per_ray
        if disparity_sampling:  # linear in inverse depth
            z_vals = 1.0 / (
                1.0 / batch["near"].view(-1, 1) * (1.0 - t_vals)
                + 1.0 / batch["far"].view(-1, 1) * (t_vals)
            )
        else:  # linear in depth
            z_vals = batch["near"].view(-1, 1) * (1.0 - t_vals) + batch["far"].view(-1, 1) * (
                t_vals
            )

        if is_training:
            # get intervals between samples
            mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape, device=device)
            z_vals = lower + (upper - lower) * t_rand

        positions = (
            batch["rays_origin"][..., None, :]
            + batch["rays_dir"][..., None, :] * z_vals[..., :, None]
        )  # num_rays x num_points_per_ray x 3
        batch["positions"] = positions
        returns.add_return("unnormalized_undeformed_positions", positions)

        do_normalize_z_vals = True  # relevant for _compose_rays
        if do_normalize_z_vals:
            scaling = torch.mean(scene.get_pos_max() - scene.get_pos_min())
            z_vals = z_vals / scaling

        batch["timesteps"] = (
            batch["timesteps"].view(-1, 1).tile([1, num_points_per_ray])
        )  # num_rays x num_points_per_ray

        batch["intrinids"] = (
            batch["intrinids"].view(-1, 1).tile([1, num_points_per_ray])
        )  # num_rays x num_points_per_ray

        view_directions = batch["rays_dir"] / torch.norm(
            batch["rays_dir"], dim=-1, keepdim=True
        )  # num_rays x 3
        batch["view_directions"] = view_directions.view(num_rays, 1, 3).tile(
            [1, num_points_per_ray, 1]
        )  # num_rays x num_points_per_ray x 3

        batch["mip_scale"] = None

        return batch, z_vals

    def _compose_rays(
        self,
        raw_rgb_per_point,
        raw_alpha,
        z_vals,
        rays_dir,
        pruning_mask,
        is_training,
        points_per_ray,
        returns,
    ):

        device = raw_rgb_per_point.device
        corrective_factor = float(points_per_ray) / 1024.0
        raw2alpha = lambda raw_alpha, dists, act_fn=torch.nn.functional.relu: 1.0 - torch.exp(
            -act_fn(raw_alpha) * dists * corrective_factor
        )

        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat(
            [dists, torch.tensor([1e10], device=device).expand(dists[..., :1].shape)], -1
        )  # [N_rays, N_samples]

        dists = dists * torch.norm(rays_dir[..., None, :], dim=-1)

        if is_training and self.raw_noise_std > 0.0:
            noise = torch.randn(raw_alpha.shape, device=device) * self.raw_noise_std
            noise[~pruning_mask] = 0.0  # pruned samples get no noise
        else:
            noise = 0.0

        # noise is added to alpha during accumulation
        alpha = raw2alpha(raw_alpha + noise, dists)  # [N_rays, N_samples]
        returns.add_return("alpha", alpha)

        weights = (
            alpha
            * torch.cumprod(
                torch.cat(
                    [torch.ones((alpha.shape[0], 1), device=device), 1.0 - alpha + 1e-10], -1
                ),
                -1,
            )[:, :-1]
        )
        returns.add_return("weights", weights)

        rgb = torch.sum(weights[..., None] * raw_rgb_per_point, -2)  # [N_rays, 3]
        returns.add_return("uncorrected_rgb", rgb)

        accumulated_weights = torch.sum(weights, -1)
        returns.add_return("accumulated_weights", accumulated_weights)

        depth = (
            torch.sum(weights[:, :-1] * z_vals[:, :-1], -1)
            + (1.0 - accumulated_weights + weights[:, -1]) * z_vals[:, -1]
        )
        returns.add_return("depth", depth)

        returns.add_return("z_vals", z_vals)

        disparity = 1.0 / torch.max(1e-4 * torch.ones_like(depth), depth / torch.sum(weights, -1))
        returns.add_return("disparity", disparity)

        return rgb, accumulated_weights

    def render(self, *args, **kwargs):
        if self.use_half_precision:
            with torch.autocast("cuda"):  # tag:half_precision
                return self._render(*args, **kwargs)
        else:
            return self._render(*args, **kwargs)

    def _render(
        self,
        batch,
        scene,
        returns=None,
        subreturns=None,
        points_per_ray=None,
        is_training=False,
        pull_to_cpu=False,
        iterator_yield=False,
        **kwargs
    ):

        if points_per_ray is None:
            points_per_ray = self.default_num_points_per_ray

        if returns is None:
            returns = Returns(restricted=["corrected_rgb"])
            returns.activate_mode("coarse")
            subreturns = Returns(restricted=["corrected_rgb"])
        elif subreturns is None:
            subreturns = Returns(restricted=returns.get_restricted_list())

        num_rays = batch["rays_origin"].shape[0]
        rays_per_chunk = max(1, self.points_per_chunk // points_per_ray)

        def generator_wrapper():  # hacky way to allow path_renderer to save memory
            for chunkid, chunk_start in enumerate(range(0, num_rays, rays_per_chunk)):

                subreturns.activate_mode(chunkid)

                subbatch = {
                    key: tensor[chunk_start : chunk_start + rays_per_chunk]
                    for key, tensor in batch.items()
                }

                subbatch = self.pre_correction(subbatch, returns=subreturns)

                subbatch, z_vals = self._generate_points_on_rays(
                    subbatch, points_per_ray, scene, is_training, returns=subreturns
                )

                raw_rgb_per_point, raw_alpha, pruning_mask = scene(
                    subbatch["positions"],
                    subbatch["view_directions"],
                    subbatch["timesteps"],
                    mip_scales=subbatch["mip_scale"],
                    is_training=is_training,
                    returns=subreturns,
                    **kwargs
                )

                rgb, accumulated_weights = self._compose_rays(
                    raw_rgb_per_point,
                    raw_alpha,
                    z_vals,
                    subbatch["rays_dir"],
                    pruning_mask,
                    is_training,
                    points_per_ray,
                    returns=subreturns,
                )

                rgb = self.post_correction(
                    rgb, subbatch, accumulated_weights, is_training, returns=subreturns
                )

                if pull_to_cpu:
                    subreturns.pull_to_cpu()

                if iterator_yield:  # used by path_renderer to save memory
                    yield subreturns

        if iterator_yield:  # only for path_renderer
            return generator_wrapper
        else:  # main branch
            for _ in generator_wrapper():  # hacky way to run the generator
                pass
            returns.add_returns(subreturns.concatenate_returns())
            rgb = returns.get_returns()["corrected_rgb"]
            return rgb

    def get_parameters_with_optimization_information(self):
        return (
            self.pre_correction.get_parameters_with_optimization_information()
            + self.post_correction.get_parameters_with_optimization_information()
        )

    def get_regularization_losses(self):
        regularization_losses = {}
        regularization_losses["pre_correction"] = self.pre_correction.get_regularization_losses()
        regularization_losses["post_correction"] = self.post_correction.get_regularization_losses()
        return regularization_losses
