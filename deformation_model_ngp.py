# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import logging

import tinycudann as tcnn
import torch
from deformation_model import DeformationModel
from utils import (
    build_pytorch_mlp_from_tinycudann,
    infill_masked,
    positional_encoding,
    project_to_correct_range,
)

LOGGER = logging.getLogger(__name__)


class DeformationModelNGP(DeformationModel):
    def __init__(self, settings, timeline_range):
        super().__init__()

        self.debug = settings.debug

        # initialization
        self.fine_levels_to_detach = 0

        # model
        self.use_temporal_latent_codes = settings.use_temporal_latent_codes
        self.pure_mlp_bending = settings.pure_mlp_bending
        self.use_viewdirs = settings.use_viewdirs
        self.use_nerfies_se3 = settings.use_nerfies_se3
        self.activation_function = settings.activation_function
        self.do_nrnerf = settings.do_nrnerf
        self.do_dnerf = settings.do_dnerf
        self.do_pref = settings.do_pref
        self.pref_tau_window = settings.pref_tau_window

        self.coarse_and_fine = settings.coarse_and_fine
        self.coarse_parametrization = settings.coarse_parametrization
        self.fine_range = settings.fine_range
        self.ignore_time = settings.optimization_mode == "per_timestep"
        self.hierarchical_application = True
        if self.hierarchical_application and self.use_nerfies_se3:
            raise NotImplementedError  # SE3 might need slightly more careful application

        if self.pure_mlp_bending and self.coarse_and_fine and self.coarse_parametrization != "MLP":
            raise RuntimeError("need to turn off pure_mlp_bending when using coarse hashgrid")

        self.explicit_deformations = False
        self.restrict_explicit_deformations = False

        self.skip_connections = settings.coarse_mlp_skip_connections  # no skip: 0

        # implementation
        self.half_precision = settings.use_half_precision  # tag:half_precision
        self.use_pytorch_mlp = True
        self.prefer_cutlass_over_fullyfused_mlp = settings.prefer_cutlass_over_fullyfused_mlp

        # optimization
        self.learning_rate_decay_autodecoding_fraction = (
            settings.learning_rate_decay_autodecoding_fraction
        )
        self.learning_rate_decay_autodecoding_iterations = (
            settings.learning_rate_decay_autodecoding_iterations
        )
        self.learning_rate_decay_mlp_fraction = settings.learning_rate_decay_mlp_fraction
        self.learning_rate_decay_mlp_iterations = settings.learning_rate_decay_mlp_iterations
        self.weight_parameter_regularization = settings.weight_parameter_regularization
        self.coarse_mlp_weight_decay = settings.coarse_mlp_weight_decay

        # state
        self.zero_out_fine_deformations = False  # initialization
        self.zero_out_coarse_deformations = False

        if self.use_nerfies_se3:
            self.output_dims = 12
        else:
            self.output_dims = 3

        if self.do_pref:
            self.coarse_and_fine = False
            self.pure_mlp_bending = True
            self.activation_function = "ReLU"
            self.skip_connections = 1

            if self.ignore_time or not self.use_temporal_latent_codes:
                raise AssertionError("wrong settings for PREF")

        if self.do_nrnerf:
            self.coarse_and_fine = False
            self.pure_mlp_bending = True
            self.activation_function = "ReLU"
            self.skip_connections = 0

            if self.ignore_time or not self.use_temporal_latent_codes:
                raise AssertionError("wrong settings for nr-nerf")

        if self.do_dnerf:
            self.coarse_and_fine = False
            self.pure_mlp_bending = True
            self.activation_function = "ReLU"
            self.skip_connections = 1

            if self.ignore_time or self.use_temporal_latent_codes:
                raise AssertionError("wrong settings for D-NeRF")

        if self.ignore_time:
            self.use_temporal_latent_codes = False

        self.latent_encoding_config = {
            "otype": "DenseGrid",
            "n_levels": 16 if self.do_nrnerf else 1,
            "n_features_per_level": 2,
            "log2_hashmap_size": 0,
            "base_resolution": int(timeline_range) + 1,
            "per_level_scale": 1.0,
        }
        if self.do_pref:
            self.latent_encoding_config["n_levels"] = 5
            self.latent_encoding_config["n_features_per_level"] = 1
            self.pref_full_latent_dimension = 32
        if self.use_temporal_latent_codes:
            self.positional_encoding_config = {
                "otype": "Composite",
                "nested": [
                    {
                        "n_dims_to_encode": 3,
                        "otype": "Frequency",
                        "n_frequencies": 9,
                    },
                    {"otype": "Identity"},
                ],
            }
        else:
            self.positional_encoding_config = {
                "otype": "Frequency",
                "n_frequencies": 9,
            }
        if self.do_dnerf:
            self.positional_encoding_config = {
                "otype": "Composite",
                "nested": [
                    {
                        "n_dims_to_encode": 3,
                        "otype": "Frequency",
                        "n_frequencies": 10,
                    },
                    {
                        "n_dims_to_encode": 1,
                        "otype": "Frequency",
                        "n_frequencies": 4,
                    },
                ],
            }
        self.pure_mlp_bending_network_config = {
            "otype": "CutlassMLP" if self.prefer_cutlass_over_fullyfused_mlp else "FullyFusedMLP",
            "activation": self.activation_function,
            "output_activation": "None",
            "n_neurons": 128,
            "n_hidden_layers": 4 if self.skip_connections == 0 else 2,
        }
        if self.do_pref:
            self.pure_mlp_bending_network_config["n_neurons"] = 256
            self.pure_mlp_bending_network_config["n_hidden_layers"] = 4

            self.pref_predictor_mlp_config = {
                "otype": "CutlassMLP"
                if self.prefer_cutlass_over_fullyfused_mlp
                else "FullyFusedMLP",
                "activation": self.activation_function,
                "output_activation": "None",
                "n_neurons": 128,
                "n_hidden_layers": 5,
            }

        if self.do_nrnerf:
            self.pure_mlp_bending_network_config["n_neurons"] = 64
            self.pure_mlp_bending_network_config["n_hidden_layers"] = 5

        if self.do_dnerf:
            self.pure_mlp_bending_network_config["n_neurons"] = 256

        self.hash_encoding_config = {
            "otype": "HashGrid",
            "n_levels": 16,
            "n_features_per_level": 2,
            "log2_hashmap_size": 20,
            "base_resolution": 32,
            "per_level_scale": 1.3819,
        }
        if self.explicit_deformations:
            self.hash_encoding_config["n_features_per_level"] = self.output_dims
        self.base_network_config = {
            "otype": "CutlassMLP" if self.prefer_cutlass_over_fullyfused_mlp else "FullyFusedMLP",
            "activation": self.activation_function,
            "output_activation": "None",
            "n_neurons": 64,
            "n_hidden_layers": 1,
        }
        self._init_networks(settings)

    def _init_networks(self, settings):

        if self.ignore_time:
            temporal_conditioning_size = 0
        else:
            if self.use_temporal_latent_codes:
                if self.do_pref:
                    temporal_conditioning_size = self.pref_full_latent_dimension
                else:
                    temporal_conditioning_size = (
                        self.latent_encoding_config["n_features_per_level"]
                        * self.latent_encoding_config["n_levels"]
                    )
            else:
                temporal_conditioning_size = 1

        if self.pure_mlp_bending or (self.coarse_and_fine and self.coarse_parametrization == "MLP"):

            def _get_mlp(counter):
                if not self.use_pytorch_mlp:
                    raise NotImplementedError("need to split hashgrid and mlp for skip connection")
                first_layer = counter == 0
                last_layer = counter == self.skip_connections
                encoding_config = self.positional_encoding_config

                if self.do_dnerf:
                    n_frequencies = encoding_config["nested"][0]["n_frequencies"]
                elif self.use_temporal_latent_codes:
                    n_frequencies = encoding_config["nested"][0]["n_frequencies"]
                else:
                    n_frequencies = encoding_config["n_frequencies"]
                skip_connection_dimension = 3 + n_frequencies * 2 * 3
                if self.use_temporal_latent_codes:
                    skip_connection_dimension += temporal_conditioning_size
                elif self.do_dnerf:
                    n_frequencies = encoding_config["nested"][1]["n_frequencies"]
                    skip_connection_dimension += (
                        temporal_conditioning_size + n_frequencies * 2 * temporal_conditioning_size
                    )
                else:
                    skip_connection_dimension += (
                        temporal_conditioning_size + n_frequencies * 2 * temporal_conditioning_size
                    )

                hidden_dimension = self.pure_mlp_bending_network_config["n_neurons"]

                original_output_activation = self.pure_mlp_bending_network_config[
                    "output_activation"
                ]
                if not last_layer:
                    self.pure_mlp_bending_network_config[
                        "output_activation"
                    ] = self.pure_mlp_bending_network_config["activation"]

                mlp_dict = {
                    "n_input_dims": skip_connection_dimension + hidden_dimension,
                    "n_output_dims": hidden_dimension,
                    "network_config": self.pure_mlp_bending_network_config,
                }

                if first_layer:
                    mlp_dict["n_input_dims"] = 3 + temporal_conditioning_size
                    mlp_dict["encoding_config"] = encoding_config
                if last_layer:
                    mlp_dict["n_output_dims"] = self.output_dims

                if self.use_pytorch_mlp:
                    network = build_pytorch_mlp_from_tinycudann(
                        mlp_dict, last_layer_zero_init=last_layer
                    )
                else:
                    if first_layer:
                        network = tcnn.NetworkWithInputEncoding(**mlp_dict)
                    else:
                        network = tcnn.Network(**mlp_dict)

                if not last_layer:
                    self.pure_mlp_bending_network_config[
                        "output_activation"
                    ] = original_output_activation

                return network

            self.pure_mlp_bending_model = torch.nn.ModuleList(
                [_get_mlp(counter) for counter in range(1 + self.skip_connections)]
            )

        if not self.pure_mlp_bending or self.coarse_and_fine:
            self.bending_enc_model = tcnn.Encoding(
                n_input_dims=3 + temporal_conditioning_size,
                encoding_config=self.hash_encoding_config,
            )

            if self.explicit_deformations:
                with torch.no_grad():
                    self.bending_enc_model.params *= 0.0

            bending_enc_output_size = (
                self.hash_encoding_config["n_levels"]
                * self.hash_encoding_config["n_features_per_level"]
            )

            mlp_dict = {
                "n_input_dims": bending_enc_output_size,
                "n_output_dims": self.output_dims,
                "network_config": self.base_network_config,
            }
            if self.use_pytorch_mlp:
                self.bending_mlp_model = build_pytorch_mlp_from_tinycudann(
                    mlp_dict, last_layer_zero_init=True
                )
            else:
                self.bending_mlp_model = tcnn.Network(**mlp_dict)

        if self.coarse_parametrization == "hashgrid" and self.coarse_and_fine:
            self.coarse_bending_enc_model = tcnn.Encoding(
                n_input_dims=3 + temporal_conditioning_size,
                encoding_config=self.hash_encoding_config,
            )

            if self.explicit_deformations:
                with torch.no_grad():
                    self.coarse_bending_enc_model.params *= 0.0

            bending_enc_output_size = (
                self.hash_encoding_config["n_levels"]
                * self.hash_encoding_config["n_features_per_level"]
            )

            mlp_dict = {
                "n_input_dims": bending_enc_output_size,
                "n_output_dims": self.output_dims,
                "network_config": self.base_network_config,
            }
            if self.use_pytorch_mlp:
                self.coarse_bending_mlp_model = build_pytorch_mlp_from_tinycudann(
                    mlp_dict, last_layer_zero_init=True
                )
            else:
                self.coarse_bending_mlp_model = tcnn.Network(**mlp_dict)

        if self.use_temporal_latent_codes:
            self.latent_codes = tcnn.Encoding(
                n_input_dims=1, encoding_config=self.latent_encoding_config, dtype=torch.float32
            )
            if not self.do_pref:
                with torch.no_grad():
                    self.latent_codes.params *= 0.0
                    self.latent_codes.params += 0.1  # to prevent zero gradient if the value is 0.0

            # if num_latents == 30, then 0.5/29.0, 1.5/29.0, ..., 29.5/29.0 are the indices that give the individual latent codes
            # without mixing/interpolation. there is a padding latent code on each side, at -0.5/29.0 and 30.5/29.0, respectively.
            # to map the dataloader's range of [0,1] such that 0 goes to 0.5/29.0 and 1 to 29.5/29.0, we have to do:
            # [0,1] + 0.5/(num_latents-1)
            num_latents = self.latent_encoding_config["base_resolution"]
            self.latent_code_index_correction = +0.5 / float(num_latents - 1)

        if self.do_pref:

            # latent basis
            coefficients_size = (
                self.latent_encoding_config["n_features_per_level"]
                * self.latent_encoding_config["n_levels"]
            )
            full_latent_size = self.pref_full_latent_dimension

            pref_latent_basis = torch.nn.Linear(coefficients_size, full_latent_size, bias=False)
            with torch.no_grad():
                torch.nn.init.normal_(pref_latent_basis.weight)
            self.pref_latent_basis = pref_latent_basis

            # predictor MLP
            mlp_dict = {
                "n_input_dims": self.pref_tau_window * coefficients_size,
                "n_output_dims": coefficients_size,
                "network_config": self.pref_predictor_mlp_config,
            }
            if self.use_pytorch_mlp:
                self.pref_predictor_mlp = build_pytorch_mlp_from_tinycudann(
                    mlp_dict, last_layer_zero_init=False
                )
            else:
                self.pref_predictor_mlp = tcnn.Network(**mlp_dict)

    def forward(self, positions, timesteps, mask, is_training, returns):

        if self.use_temporal_latent_codes:
            if self.do_pref:
                num_latents = self.latent_encoding_config[
                    "base_resolution"
                ]  # 0, ..., max_timestep, "num_latents"
                max_timestep = num_latents - 1
                helper_indices = (
                    torch.arange(num_latents, device=positions.device) / (num_latents - 1)
                    + self.latent_code_index_correction
                )  # from 0 to max_timestep (incl.)
                coefficients = self.latent_codes(helper_indices.view(-1, 1))
                all_latent_codes = self.pref_latent_basis(
                    coefficients
                )  # cofficient_size x pref_full_latent_dimension

                original_timesteps = timesteps

                unnormalized_timesteps = timesteps * max_timestep  # from [0,1] to [0,max_timestep]
                pref_int_timesteps = (unnormalized_timesteps + 0.0000001).long()

                latent_codes = all_latent_codes[pref_int_timesteps.view(-1)]
                timesteps = latent_codes
            else:
                latent_indices = timesteps + self.latent_code_index_correction
                latent_codes = self.latent_codes(latent_indices)
                latent_codes = project_to_correct_range(latent_codes, mode="zick_zack")
                timesteps = latent_codes

        if self.ignore_time:
            model_input = positions
        else:
            model_input = torch.cat([positions, timesteps], -1)

        # large bending MLP
        if self.pure_mlp_bending or (self.coarse_and_fine and self.coarse_parametrization == "MLP"):
            if self.use_pytorch_mlp:
                if self.use_temporal_latent_codes:
                    # don't apply positional encoding to latent codes
                    pos_enc_input_dimensions = self.positional_encoding_config["nested"][0][
                        "n_dims_to_encode"
                    ]
                    pos_enc_model_input, raw_model_input = torch.split(
                        model_input,
                        [
                            pos_enc_input_dimensions,
                            model_input.shape[-1] - pos_enc_input_dimensions,
                        ],
                        dim=-1,
                    )
                    pos_enc_model_input = positional_encoding(
                        pos_enc_model_input,
                        num_frequencies=self.positional_encoding_config["nested"][0][
                            "n_frequencies"
                        ],
                    )
                    pure_model_input = torch.cat([pos_enc_model_input, raw_model_input], dim=-1)
                elif self.do_dnerf:
                    pos_enc_positions = positional_encoding(
                        positions,
                        num_frequencies=self.positional_encoding_config["nested"][0][
                            "n_frequencies"
                        ],
                    )
                    pos_enc_timesteps = positional_encoding(
                        timesteps,
                        num_frequencies=self.positional_encoding_config["nested"][1][
                            "n_frequencies"
                        ],
                    )
                    pure_model_input = torch.cat([pos_enc_positions, pos_enc_timesteps], dim=-1)
                else:
                    pure_model_input = positional_encoding(
                        model_input,
                        num_frequencies=self.positional_encoding_config["n_frequencies"],
                    )
            else:
                pure_model_input = model_input

            network_output = pure_model_input
            for counter, pure_mlp_bending_model in enumerate(self.pure_mlp_bending_model):
                if counter > 0:  # skip connection
                    network_output = torch.cat([pure_model_input, network_output], dim=-1)
                network_output = pure_mlp_bending_model(network_output)
            if not self.use_pytorch_mlp:
                network_output = 0.01 * network_output

        # coarse hashgrid
        if self.coarse_and_fine and self.coarse_parametrization == "hashgrid":
            position_features = self.coarse_bending_enc_model(model_input)

            if self.explicit_deformations:
                num_points = position_features.shape[0]
                position_features = position_features.reshape(num_points, -1, self.output_dims)
                if self.restrict_explicit_deformations:
                    raise NotImplementedError
                network_output = torch.sum(position_features, dim=1)  # num_points x output_dims
            else:
                if not self.half_precision:
                    position_features = position_features.float()
                network_output = 0.01 * self.coarse_bending_mlp_model(position_features)

        # handling coarse deformations
        if self.coarse_and_fine:
            coarse_network_output = network_output

            if self.hierarchical_application:
                if self.use_nerfies_se3:
                    coarse_position_offsets = self._apply_se3(positions, network_output)
                else:
                    coarse_position_offsets = network_output
                coarse_positions = positions + coarse_position_offsets

                # workaround for masking and gradient-based losses: the coarse_positions that is stored in "returns" needs to be in the computational graph for fine deformations
                successful = returns.add_return("coarse_positions", coarse_positions)
                if successful:
                    coarse_positions = returns.get_returns()["coarse_positions"]
                    original_mask = returns.get_mask()
                    returns.set_mask(None)
                    returns.add_return("coarse_positions", coarse_positions, clone=False)
                    returns.set_mask(original_mask)
                    coarse_positions = coarse_positions[mask].view(-1, 3)

                returns.add_return("coarse_position_offsets", coarse_position_offsets)

        # fine/main hashgrid
        if not self.pure_mlp_bending or self.coarse_and_fine:
            if self.coarse_and_fine and self.hierarchical_application:
                if self.ignore_time:
                    fine_model_input = coarse_positions
                else:
                    fine_model_input = torch.cat([coarse_positions, timesteps], -1)
            else:
                fine_model_input = model_input
            position_features = self.bending_enc_model(fine_model_input)

            if self.fine_levels_to_detach > 0:
                fine_levels_to_detach = min(
                    self.fine_levels_to_detach, self.hash_encoding_config["n_levels"]
                )
                split_dimension = (
                    fine_levels_to_detach * self.hash_encoding_config["n_features_per_level"]
                )
                attached_position_features, detached_position_features = torch.split(
                    position_features,
                    [position_features.shape[-1] - split_dimension, split_dimension],
                    dim=-1,
                )
                detached_position_features = detached_position_features.detach()
                position_features = torch.cat(
                    [attached_position_features, detached_position_features], dim=-1
                )

            if self.explicit_deformations:
                num_points = position_features.shape[0]
                position_features = position_features.reshape(num_points, -1, self.output_dims)
                if self.restrict_explicit_deformations:
                    raise NotImplementedError
                network_output = torch.sum(position_features, dim=1)  # num_points x output_dims
            else:
                if not self.half_precision:
                    position_features = position_features.float()
                network_output = 0.01 * self.bending_mlp_model(position_features)

            if self.coarse_and_fine:
                fine_network_output = network_output

        # combining coarse and fine
        if self.coarse_and_fine:
            if self.zero_out_fine_deformations:
                network_output = coarse_network_output
            elif self.zero_out_coarse_deformations:
                if self.fine_range > 0.0:
                    fine_network_output = project_to_correct_range(
                        fine_network_output,
                        min_=-self.fine_range,
                        max_=+self.fine_range,
                        mode="zick_zack",
                    )
                network_output = fine_network_output
            else:
                if self.fine_range > 0.0:
                    fine_network_output = project_to_correct_range(
                        fine_network_output,
                        min_=-self.fine_range,
                        max_=+self.fine_range,
                        mode="zick_zack",
                    )
                returns.add_return("fine_position_offsets", fine_network_output)
                network_output = coarse_network_output + fine_network_output

        # SE(3)
        if self.use_nerfies_se3:
            position_offsets = self._apply_se3(positions, network_output)
        else:
            position_offsets = network_output

        # PREF mixed batches
        if self.do_pref:
            num_points = position_offsets.shape[0]

            downscale_pref = 1000.0
            if downscale_pref is not None:
                position_offsets = position_offsets / downscale_pref

            if is_training:
                # first half with offsets, second half without
                half_num_points = num_points // 2

                position_offsets[half_num_points:, :] = 0.0

                pref_int_timesteps = (
                    pref_int_timesteps.clone()
                )  # necessary to avoid some in-place gradient problem
                pref_int_timesteps[:half_num_points] = pref_int_timesteps[:half_num_points] + 1
                pref_int_timesteps = torch.clamp(pref_int_timesteps, min=0, max=max_timestep)

                pref_timesteps = pref_int_timesteps / max_timestep

            else:  # no offsets anywhere
                position_offsets[:, :] = 0.0
                pref_timesteps = original_timesteps
        else:
            pref_timesteps = None

        # D-NeRF enforce canonical model to be 0-th timestep
        if self.do_dnerf:
            mask = timesteps != 0.0  # num_points x 1
            position_offsets = mask * position_offsets

        returns.add_return("position_offsets", position_offsets)

        positions = positions + position_offsets

        if self.debug:
            before_clamp = positions.clone()
        positions = project_to_correct_range(positions, mode="zick_zack")
        if self.debug and torch.any(before_clamp != positions):
            LOGGER.debug("clamping " + str(torch.mean((before_clamp != positions).float()).item()))

        returns.add_return("deformed_positions", positions)

        if self.use_viewdirs:
            # returns.get_returns()["deformed_positions"] is infilled to num_rays x num_points_per_ray
            if "deformed_positions" in returns.get_returns():
                infilled_positions = returns.get_returns()["deformed_positions"]
            else:
                infilled_positions = infill_masked(mask, positions, infill_value=0)
            view_directions = self.viewdirs_via_finite_differences(infilled_positions, returns)
        else:
            view_directions = torch.zeros(
                mask.shape + positions.shape[1:], dtype=positions.dtype, device=positions.device
            )

        return positions, view_directions, pref_timesteps

    def get_parameters_with_optimization_information(self):

        params = []

        if self.pure_mlp_bending or (self.coarse_and_fine and self.coarse_parametrization == "MLP"):
            params.append(
                {
                    "name": "pure_mlp_bending_model",
                    "tags": ["mlp"],
                    "parameters": list(self.pure_mlp_bending_model.parameters()),
                    "optimizer": "Adam",
                    "learning_rate": 1e-4,
                    # *pure* MLP has less need for slow-down, similar to autodecoded parameters
                    "decay_steps": self.learning_rate_decay_autodecoding_iterations,
                    "decay_rate": self.learning_rate_decay_autodecoding_fraction,
                    "weight_decay": self.weight_parameter_regularization,
                }
            )
            if self.coarse_and_fine:
                params[-1]["weight_decay"] = self.coarse_mlp_weight_decay
                params[-1]["tags"].append("coarse_deformation")
        if self.coarse_and_fine and self.coarse_parametrization == "hashgrid":
            params.append(
                {
                    "name": "coarse_bending_enc_model",
                    "tags": ["autodecoding", "coarse_deformation"],
                    "parameters": list(self.coarse_bending_enc_model.parameters()),
                    "optimizer": "SmartAdam",
                    "learning_rate": 1e-3,
                    "decay_steps": self.learning_rate_decay_autodecoding_iterations,
                    "decay_rate": self.learning_rate_decay_autodecoding_fraction,
                    "weight_decay": self.weight_parameter_regularization,
                }
            )
            params.append(
                {
                    "name": "coarse_bending_mlp_model",
                    "tags": ["mlp", "coarse_deformation"],
                    "parameters": list(self.coarse_bending_mlp_model.parameters()),
                    "optimizer": "Adam",
                    "learning_rate": 1e-3,
                    "decay_steps": self.learning_rate_decay_mlp_iterations,
                    "decay_rate": self.learning_rate_decay_mlp_fraction,
                    "weight_decay": self.weight_parameter_regularization,
                }
            )
        if not self.pure_mlp_bending or self.coarse_and_fine:
            params.append(
                {
                    "name": "bending_enc_model",
                    "tags": ["autodecoding"],
                    "parameters": list(self.bending_enc_model.parameters()),
                    "optimizer": "SmartAdam",
                    "learning_rate": 1e-3,
                    "decay_steps": self.learning_rate_decay_autodecoding_iterations,
                    "decay_rate": self.learning_rate_decay_autodecoding_fraction,
                    "weight_decay": self.weight_parameter_regularization,
                }
            )
            if self.coarse_and_fine:
                params[-1]["tags"].append("fine_deformation")
            params.append(
                {
                    "name": "bending_mlp_model",
                    "tags": ["mlp"],
                    "parameters": list(self.bending_mlp_model.parameters()),
                    "optimizer": "Adam",
                    "learning_rate": 1e-3,
                    "decay_steps": self.learning_rate_decay_mlp_iterations,
                    "decay_rate": self.learning_rate_decay_mlp_fraction,
                    "weight_decay": self.weight_parameter_regularization,
                }
            )
            if self.coarse_and_fine:
                params[-1]["tags"].append("fine_deformation")

        if self.use_temporal_latent_codes:
            params.append(
                {
                    "name": "latent_codes",
                    "tags": ["autodecoding"],
                    "parameters": list(self.latent_codes.parameters()),
                    "optimizer": "SmartAdam",
                    "learning_rate": 1e-4,
                    "decay_steps": self.learning_rate_decay_autodecoding_iterations,
                    "decay_rate": self.learning_rate_decay_autodecoding_fraction,
                    "weight_decay": self.weight_parameter_regularization,
                }
            )
        if self.do_pref:
            params.append(
                {
                    "name": "pref_latent_basis",
                    "tags": ["mlp", "pref"],
                    "parameters": list(self.pref_latent_basis.parameters()),
                    "optimizer": "Adam",
                    "learning_rate": 1e-3,
                    "decay_steps": self.learning_rate_decay_mlp_iterations,
                    "decay_rate": self.learning_rate_decay_mlp_fraction,
                    "weight_decay": self.weight_parameter_regularization,
                }
            )
            params.append(
                {
                    "name": "pref_predictor_mlp",
                    "tags": ["mlp", "pref"],
                    "parameters": list(self.pref_predictor_mlp.parameters()),
                    "optimizer": "Adam",
                    "learning_rate": 1e-3,
                    "decay_steps": self.learning_rate_decay_mlp_iterations,
                    "decay_rate": self.learning_rate_decay_mlp_fraction,
                    "weight_decay": self.weight_parameter_regularization,
                }
            )

        return params

    def step(self):

        # if self.use_global_transform:
        #    with torch.no_grad():
        #        # project to unit sphere for valid quaternion
        #        self.global_rotation[:] = self.global_rotation[:] / torch.linalg.norm(self.global_rotation, ord=2, dim=0)
        pass

    def get_regularization_losses(self):
        regularization_losses = {}

        # use weight decay instead.
        # if self.pure_mlp_bending:
        #    regularization_losses["pure_mlp_bending_model"] = torch.sum(self.pure_mlp_bending_model.params ** 2)
        # else:
        #    regularization_losses["bending_mlp_model"] = torch.sum(self.bending_mlp_model.params ** 2)

        return regularization_losses

    def get_deformation_representation_at_unnormalized_timestep(self, timestep):
        n_features_per_level = self.latent_encoding_config["n_features_per_level"]
        level_size = self.latent_codes.params.shape[0] // self.latent_encoding_config["n_levels"]

        deformation_representation = []

        assert timestep.long().float() == timestep  # timestep needs to be an integer

        latentid = (
            timestep.long() + 1
        )  # tiny cuda nn allocates a dummy latent code before the actual latent codes, on each level
        for level in range(self.latent_encoding_config["n_levels"]):
            level_offset = level * level_size
            deformation_representation.append(
                self.latent_codes.params[
                    latentid * n_features_per_level
                    + level_offset : (latentid + 1) * n_features_per_level
                    + level_offset
                ]
            )

        return deformation_representation

    def set_deformation_representation_at_unnormalized_timestep(
        self, timestep, deformation_representation
    ):
        n_features_per_level = self.latent_encoding_config["n_features_per_level"]
        level_size = self.latent_codes.params.shape[0] // self.latent_encoding_config["n_levels"]

        assert timestep.long().float() == timestep  # timestep needs to be an integer

        latentid = (
            timestep.long() + 1
        )  # tiny cuda nn allocates a dummy latent code before the actual latent codes, on each level
        for level, partial_latent in zip(
            range(self.latent_encoding_config["n_levels"]), deformation_representation
        ):
            level_offset = level * level_size
            self.latent_codes.params[
                latentid * n_features_per_level
                + level_offset : (latentid + 1) * n_features_per_level
                + level_offset
            ] = partial_latent
