# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import logging

import torch
from utils import Returns

LOGGER = logging.getLogger(__name__)


class Losses:
    def __init__(self, settings, world_size):

        self.reconstruction_loss_type = settings.reconstruction_loss_type  # L1, L2
        self.smooth_deformations_type = (
            settings.smooth_deformations_type
        )  # "divergence", "jacobian", "finite", "norm_preserving"

        self.weight_smooth_deformations = settings.weight_smooth_deformations
        self.weight_background_loss = settings.weight_background_loss
        self.weight_brightness_change_regularization = (
            settings.weight_brightness_change_regularization
        )
        self.weight_hard_surface_loss = settings.weight_hard_surface_loss
        self.weight_coarse_smooth_deformations = settings.weight_coarse_smooth_deformations
        self.weight_fine_smooth_deformations = settings.weight_fine_smooth_deformations
        self.weight_small_fine_offsets_loss = settings.weight_small_fine_offsets_loss
        self.weight_similar_coarse_and_total_offsets_loss = (
            settings.weight_similar_coarse_and_total_offsets_loss
        )
        self.do_pref = settings.do_pref
        self.do_nrnerf = settings.do_nrnerf

        self.variant = settings.variant

        self.smoothness_robustness_threshold = settings.smoothness_robustness_threshold

        self.use_half_precision = settings.use_half_precision

        self.world_size = world_size

        self.debug = settings.debug

        self.variant_reconstruction_mode = None  # init

    def _only_for_large_deformations(
        self, loss, position_offsets, only_for_large_deformations=None, mode="sigmoid"
    ):
        if only_for_large_deformations is None:
            only_for_large_deformations = 0.001
        offset_magnitude = torch.linalg.norm(position_offsets.detach(), dim=-1)
        if mode == "binary":
            offset_weights = offset_magnitude > only_for_large_deformations
        elif mode == "sigmoid":
            offset_weights = torch.sigmoid(
                (4.0 * offset_magnitude / only_for_large_deformations) - 2.0
            )
        if len(loss.shape) == 3:
            offset_weights = torch.unsqueeze(offset_weights, dim=-1)
        return loss * offset_weights.detach()

    def smoothness_loss(
        self,
        deformation_smoothness_offsets,
        input_positions_name,
        returns,
        subreturns,
        L2=False,
        max_windowed=False,
        smart_maxpool=False,
        gaussian_filtered=False,
        only_to_opaque_foreground=False,
        use_minimum_weight=False,
        second_smart_maxwindowed=False,
        params=None,
    ):
        if self.smooth_deformations_type not in [
            "finite",
            "jacobian",
            "divergence",
            "nerfies",
            "norm_preserving",
        ]:
            raise NotImplementedError

        if params is None:
            params = {}

        if self.smooth_deformations_type == "finite":

            def ray_based_smooth_deformations(position_offsets, opacity, L2=False):
                deformation_differences = torch.linalg.norm(
                    position_offsets[:, 1:, :] - position_offsets[:, :-1, :], dim=-1
                )
                weigh_by_opacity = False
                if weigh_by_opacity:
                    deformation_differences = opacity.detach() * deformation_differences
                only_for_large_deformations = None
                if only_for_large_deformations is not None:
                    offset_magnitude = torch.linalg.norm(position_offsets.detach(), dim=-1)
                    mask = offset_magnitude > only_for_large_deformations
                    deformation_differences = deformation_differences * mask
                return torch.mean(deformation_differences ** (2 if L2 else 1))

            return ray_based_smooth_deformations(
                returns.get_returns()[deformation_smoothness_offsets],
                returns.get_returns()["alpha"],
                L2=L2,
            )
        else:

            def gradient_based_smooth_deformations(subreturns, L2=False):
                smooth_deformations_losses = []
                for mode in subreturns.get_modes():
                    subreturns.activate_mode(mode)
                    position_offsets = subreturns.get_returns()[deformation_smoothness_offsets]
                    input_positions = subreturns.get_returns()[input_positions_name]

                    if self.smooth_deformations_type in ["jacobian", "nerfies"]:
                        from utils import get_minibatch_jacobian

                        jacobian = get_minibatch_jacobian(
                            position_offsets, input_positions
                        )  # num_points x 3 x 3
                        # this_loss = (jacobian - torch.eye(3, device=jacobian.device).view(1,3,3)) ** 2 # if using normalized_deformed_positions
                        if self.smooth_deformations_type == "jacobian":
                            identity = torch.eye(3, device=jacobian.device).view(-1, 3, 3)
                            jacobian = (
                                jacobian + identity
                            )  # turn from jacobian of offsets to jacobian of positions
                            R_times_Rt = torch.matmul(jacobian, torch.transpose(jacobian, -1, -2))
                            this_loss = torch.abs(R_times_Rt - identity)
                            this_loss = this_loss.view(position_offsets.shape[:-1] + (-1,))
                            if L2:
                                this_loss = this_loss**2
                            this_loss = this_loss.mean(
                                dim=-1
                            )  # num_rays_in_chunk x num_points_per_ray
                        else:  # nerfies
                            singular_values = torch.linalg.svdvals(jacobian)  # num_points x 3
                            eps = 1e-6
                            stable_singular_values = torch.maximum(
                                singular_values, eps * torch.ones_like(singular_values)
                            )
                            log_singular_values = torch.log(stable_singular_values)
                            this_loss = torch.mean(log_singular_values**2, dim=-1)  # num_points
                    elif self.smooth_deformations_type == "norm_preserving":
                        inputs = input_positions
                        outputs = position_offsets
                        e = torch.randn_like(
                            outputs, device=outputs.get_device()
                        )  # num_rays_in_chunk x num_points_per_ray x 3
                        eps = 1e-5
                        e = e / (
                            torch.linalg.vector_norm(e, dim=-1, keepdim=True) + eps
                        )  # unit sphere
                        eT_dydx = torch.autograd.grad(outputs, inputs, e, create_graph=True)[
                            0
                        ]  # num_rays_in_chunk x num_points_per_ray x 3
                        eT_dydx_pe = (
                            eT_dydx + e
                        )  # turn from Jacobian of offsets into Jacobian of positions: J_pos x = (J_off+I)x = J_off x + x
                        this_loss = torch.abs(
                            torch.linalg.norm(eT_dydx_pe, dim=-1) - 1
                        )  # num_rays_in_chunk x num_points_per_ray
                        if L2:
                            this_loss = this_loss**2
                    else:  # divergence
                        from utils import divergence_exact, divergence_approx

                        exact = False
                        divergence_fn = divergence_exact if exact else divergence_approx
                        divergence = divergence_fn(inputs=input_positions, outputs=position_offsets)
                        this_loss = torch.abs(divergence)  # num_rays_in_chunk x num_points_per_ray
                        if L2:
                            this_loss = this_loss**2

                    weigh_by_opacity = True
                    if weigh_by_opacity:
                        opacity = subreturns.get_returns()["alpha"]
                        if max_windowed:
                            if "window_fraction" in params:
                                window_fraction = params["window_fraction"]
                            else:
                                window_fraction = 0.01
                            points_per_ray = opacity.shape[1]
                            kernel_size = max(1, int(window_fraction * points_per_ray))
                            if kernel_size % 2 == 0:
                                kernel_size += 1  # needed for integer padding
                            padding = (kernel_size - 1) // 2
                            opacity = torch.nn.functional.max_pool1d(
                                opacity,
                                kernel_size=kernel_size,
                                stride=1,
                                padding=padding,
                                dilation=1,
                                ceil_mode=True,
                                return_indices=False,
                            )
                        if smart_maxpool:
                            if not max_windowed:
                                raise RuntimeError("need max_windowed")
                            difference_factor = 10.0
                            original_opacity = subreturns.get_returns()["alpha"]
                            mask = (
                                opacity >= difference_factor * original_opacity
                            )  # num_rays_in_chunk x num_points_per_ray
                            opacity[mask] = opacity[mask] / difference_factor
                        if second_smart_maxwindowed:

                            window_fraction = 0.1
                            difference_factor = 300.0

                            points_per_ray = opacity.shape[1]
                            kernel_size = max(1, int(window_fraction * points_per_ray))
                            if kernel_size % 2 == 0:
                                kernel_size += 1  # needed for integer padding
                            padding = (kernel_size - 1) // 2
                            opacity = torch.nn.functional.max_pool1d(
                                opacity,
                                kernel_size=kernel_size,
                                stride=1,
                                padding=padding,
                                dilation=1,
                                ceil_mode=True,
                                return_indices=False,
                            )

                            original_opacity = subreturns.get_returns()["alpha"]
                            mask = (
                                opacity >= difference_factor * original_opacity
                            )  # num_rays_in_chunk x num_points_per_ray
                            opacity[mask] = opacity[mask] / difference_factor

                        if gaussian_filtered:
                            sigma_fraction = 0.02
                            gaussian_window_fraction = 0.1  # not that important
                            points_per_ray = opacity.shape[1]
                            kernel_size = max(1, int(gaussian_window_fraction * points_per_ray))
                            if kernel_size % 2 == 0:
                                kernel_size += 1  # needed for integer padding
                            sigma = sigma_fraction * float(points_per_ray)
                            gaussian_filter = torch.linspace(
                                -int(kernel_size / 2),
                                int(kernel_size / 2),
                                step=kernel_size,
                                device=opacity.device,
                            )
                            gaussian_filter = torch.exp(-(gaussian_filter**2) / (2 * sigma**2))
                            gaussian_filter = gaussian_filter.view(
                                1, 1, -1
                            )  # 1 x 1 x gaussian_support
                            gaussian_filter = gaussian_filter / torch.sum(
                                gaussian_filter
                            )  # normalize
                            opacity = torch.unsqueeze(
                                opacity, dim=1
                            )  # num_rays x 1 x num_points_per_ray
                            opacity = torch.nn.functional.conv1d(
                                input=opacity, weight=gaussian_filter, padding="same"
                            )
                            opacity = torch.squeeze(opacity, dim=1)
                        if only_to_opaque_foreground:
                            accumulated_weights_threshold = 0.5
                            accumulated_weights = subreturns.get_returns()[
                                "accumulated_weights"
                            ]  # num_rays_in_chunk
                            is_opaque = accumulated_weights >= accumulated_weights_threshold
                            opacity[~is_opaque, :] = 0.0
                        if use_minimum_weight:
                            minimum_weight = 1e-2
                            opacity[torch.where(opacity < minimum_weight)] = minimum_weight
                        if not max_windowed and not smart_maxpool:
                            opacity = opacity.clone()
                        this_loss = opacity.detach() * this_loss

                    only_for_large_deformations = 0.001
                    if only_for_large_deformations is not None:
                        this_loss = self._only_for_large_deformations(
                            this_loss,
                            position_offsets,
                            only_for_large_deformations=only_for_large_deformations,
                            mode="sigmoid",
                        )

                    factor_above_threshold = 0.1
                    if (
                        self.smoothness_robustness_threshold is not None
                        and self.smoothness_robustness_threshold > 0.0
                    ):
                        offset_magnitude = torch.linalg.norm(position_offsets.detach(), dim=-1)
                        above_threshold = offset_magnitude > self.smoothness_robustness_threshold
                        weights = above_threshold * factor_above_threshold + ~above_threshold

                        this_loss = this_loss * weights

                    smooth_deformations_losses.append(this_loss)
                smooth_deformations_loss = torch.cat(smooth_deformations_losses, dim=0)
                return torch.mean(smooth_deformations_loss)

            return gradient_based_smooth_deformations(subreturns, L2=L2)

    def compute(self, batch, scene, renderer, scheduler, training_iteration):

        device = batch["rgb"].device

        returns = Returns()
        returns.activate_mode("train")
        subreturns = Returns()  # to allow for gradient-based losses

        rgb = renderer.render(
            batch, scene, returns=returns, subreturns=subreturns, is_training=True
        )

        losses = []
        log = {}

        if self.variant_reconstruction_mode is not None:

            if self.variant_reconstruction_mode == "simple":
                losses.append(
                    {
                        "name": "reconstruction_L2",
                        "weight": 1.0,
                        "loss": torch.mean((batch["rgb"] - rgb) ** 2),
                    }
                )
            elif self.variant_reconstruction_mode == "huber":
                losses.append(
                    {
                        "name": "reconstruction_huber",
                        "weight": 1.0,
                        "loss": torch.nn.functional.smooth_l1_loss(
                            rgb, batch["rgb"], reduction="mean", beta=0.1
                        ),
                    }
                )

        if self.reconstruction_loss_type == "L1":
            losses.append(
                {
                    "name": "reconstruction_L1",
                    "weight": 1.0,
                    "loss": torch.mean(torch.abs(batch["rgb"] - rgb)),
                }
            )
        elif self.reconstruction_loss_type == "L2":
            losses.append(
                {
                    "name": "reconstruction_L2",
                    "weight": 1.0,
                    "loss": torch.mean((batch["rgb"] - rgb) ** 2),
                }
            )

        log["psnr"] = (
            -10.0
            * torch.log(torch.mean((batch["rgb"] - rgb) ** 2))
            / torch.log(torch.tensor([10.0], device=device))
        ).item()
        log["all_psnr"] = (
            (
                -10.0
                * torch.log(torch.mean((batch["rgb"] - rgb), dim=-1) ** 2)
                / torch.log(torch.tensor([10.0], device=device))
            )
            .detach()
            .cpu()
            .numpy()
        )

        do_total_deformation_smoothness = (
            self.weight_smooth_deformations > 0.0 and "position_offsets" in returns.get_returns()
        )
        if do_total_deformation_smoothness:
            deformation_smoothness_offsets = "position_offsets"
            input_positions = "normalized_undeformed_positions"
            L2 = True if self.do_nrnerf else False
            losses.append(
                {
                    "name": "total_smooth_deformations",
                    "weight": self.weight_smooth_deformations,
                    "loss": self.smoothness_loss(
                        deformation_smoothness_offsets,
                        input_positions,
                        returns,
                        subreturns,
                        L2=L2,
                        max_windowed=True,
                        smart_maxpool=True,
                    ),
                }
            )

        do_coarse_deformation_smoothness = (
            self.weight_coarse_smooth_deformations > 0.0
            and "coarse_position_offsets" in returns.get_returns()
        )
        if do_coarse_deformation_smoothness:
            deformation_smoothness_offsets = "coarse_position_offsets"
            input_positions = "normalized_undeformed_positions"
            losses.append(
                {
                    "name": "coarse_smooth_deformations",
                    "weight": self.weight_coarse_smooth_deformations,
                    "loss": self.smoothness_loss(
                        deformation_smoothness_offsets,
                        input_positions,
                        returns,
                        subreturns,
                        L2=False,
                        max_windowed=True,
                        smart_maxpool=True,
                        gaussian_filtered=False,
                        only_to_opaque_foreground=True,
                        use_minimum_weight=False,
                        second_smart_maxwindowed=False,
                    ),
                }
            )

        do_fine_deformation_smoothness = (
            self.weight_fine_smooth_deformations > 0.0
            and "coarse_positions" in returns.get_returns()
        )
        if do_fine_deformation_smoothness:
            deformation_smoothness_offsets = "fine_position_offsets"
            input_positions = "coarse_positions"

            params = {}

            losses.append(
                {
                    "name": "fine_smooth_deformations",
                    "weight": self.weight_fine_smooth_deformations,
                    "loss": self.smoothness_loss(
                        deformation_smoothness_offsets,
                        input_positions,
                        returns,
                        subreturns,
                        L2=False,
                        max_windowed=True,
                        smart_maxpool=True,
                        gaussian_filtered=False,
                        only_to_opaque_foreground=True,
                        use_minimum_weight=False,
                        second_smart_maxwindowed=False,
                        params=params,
                    ),
                }
            )

        if (
            "background" in batch
            and self.weight_background_loss > 0.0
            and "accumulated_weights" in returns.get_returns()
        ):

            LOGGER.debug(str(torch.mean(returns.get_returns()["accumulated_weights"])))

            def beta_distribution_prior(x):
                # as in Neural Volumes
                shift = 0.1
                loss = torch.log(shift + x) + torch.log(shift + (1.0 - x))
                return torch.mean(loss) - -2.20727  # corrected such that a loss of 0.0 is optimal

            losses.append(
                {
                    "name": "background",
                    "weight": self.weight_background_loss,
                    "loss": beta_distribution_prior(returns.get_returns()["accumulated_weights"]),
                }
            )

        if self.weight_hard_surface_loss > 0.0 and "weights" in returns.get_returns():

            LOGGER.debug(str(torch.mean(returns.get_returns()["weights"])))

            def laplacian_distribution_prior(x):
                # as in LOLNeRF
                loss = -torch.log(torch.exp(-x) + torch.exp(-(1 - x)))
                return torch.mean(loss)

            losses.append(
                {
                    "name": "hard_surface",
                    "weight": self.weight_hard_surface_loss,
                    "loss": laplacian_distribution_prior(returns.get_returns()["weights"]),
                }
            )

        if (
            "brightness_change" in returns.get_returns()
            and self.weight_brightness_change_regularization > 0.0
        ):
            losses.append(
                {
                    "name": "brightness_change",
                    "weight": self.weight_brightness_change_regularization,
                    # "loss": torch.mean((returns.get_returns()["accumulated_weights"].detach() * returns.get_returns()["brightness_change"]) ** 2),
                    "loss": torch.mean((returns.get_returns()["brightness_change"]) ** 2),
                }
            )

        if self.do_pref:
            pref_prediction_loss_weight = 0.01

            deformation_model = scene.deformation_model

            predictor = deformation_model.pref_predictor_mlp

            num_latents = deformation_model.latent_encoding_config[
                "base_resolution"
            ]  # 0, ..., max_timestep, "num_latents"
            helper_indices = (
                torch.arange(num_latents, device=device) / (num_latents - 1)
                + deformation_model.latent_code_index_correction
            )  # from 0 to max_timestep (incl.)
            learned_coefficients = deformation_model.latent_codes(helper_indices.view(-1, 1))

            # get predicted coefficients
            coefficient_size = learned_coefficients.shape[1]
            tau = deformation_model.pref_tau_window
            padding_coefficients = torch.zeros((tau - 1, coefficient_size), device=device)
            padded_coefficients = torch.cat([padding_coefficients, learned_coefficients], dim=0)

            tau_indexing = torch.arange(tau, device=device)
            tau_indexing = tau_indexing.repeat(num_latents - 1).view(num_latents - 1, tau)
            tau_indexing += torch.arange(num_latents - 1, device=device).view(-1, 1)
            tau_indexing = tau_indexing.view(-1)

            prior_coefficients = padded_coefficients[tau_indexing].view(
                num_latents - 1, tau * coefficient_size
            )

            predicted_coefficients = predictor(prior_coefficients)

            prediction_loss = torch.linalg.norm(
                learned_coefficients[1:] - predicted_coefficients, dim=-1
            )  # first latent does not get predicted
            prediction_loss = torch.mean(prediction_loss**2)

            losses.append(
                {
                    "name": "pref_prediction",
                    "weight": pref_prediction_loss_weight,
                    "loss": prediction_loss,
                }
            )

        if self.debug:
            for loss_dict in losses:
                if not torch.isfinite(loss_dict["loss"]):
                    LOGGER.debug(
                        loss_dict["name"] + " is not finite: " + str(loss_dict["loss"].item())
                    )

        loss = sum(loss_dict["weight"] * loss_dict["loss"] for loss_dict in losses) / float(
            self.world_size
        )

        # tag:half_precision
        half_precision_float_gradient_upscaling = self.use_half_precision
        if half_precision_float_gradient_upscaling:
            # keep the gradient in a good numerical range.
            # scaling is only unproblematic because Adam is invariant under global scaling.
            num_rays, num_points_per_ray = returns.get_returns()[
                "normalized_undeformed_positions"
            ].shape[:2]
            default_num_rays = 1024
            default_num_points_per_ray = 4096
            # loss_scaling_factor = 128.0 * ( float(num_rays * num_points_per_ray) / float(default_num_rays * default_num_points_per_ray) )
            loss_scaling_factor = 8.0 * (
                float(num_rays * num_points_per_ray)
                / float(default_num_rays * default_num_points_per_ray)
            )
            loss = loss * loss_scaling_factor
        else:
            loss_scaling_factor = 1.0

        return loss, loss_scaling_factor, log
