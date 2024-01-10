# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import logging

import tinycudann as tcnn
import torch
from canonical_model import CanonicalModel
from utils import build_pytorch_mlp_from_tinycudann, positional_encoding, project_to_correct_range

LOGGER = logging.getLogger(__name__)


class CanonicalModelNGP(CanonicalModel):
    def __init__(self, settings):
        super().__init__()

        self.debug = settings.debug

        # model
        self.use_timevarying_appearance = False
        self.use_timevarying_geometry = False
        self.use_viewdirs = settings.use_viewdirs
        self.brightness_variability = settings.brightness_variability
        self.variant = settings.variant

        self.do_pref = settings.do_pref
        self.do_nrnerf = settings.do_nrnerf
        self.do_dnerf = settings.do_dnerf
        self.do_pure_mlp = self.do_pref or self.do_nrnerf or self.do_dnerf

        self.activation_function = settings.activation_function
        if self.activation_function == "LeakyReLU":
            self.activation_function = "ReLU"

        self.separate_brightness_model = True
        self.separate_appearance_model = self.variant == "snfa"

        if self.use_timevarying_geometry and not self.use_timevarying_appearance:
            raise NotImplementedError
        if self.use_timevarying_appearance and self.brightness_variability > 0.0:
            raise NotImplementedError

        # implementation
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

        if self.variant in ["snfa", "snfag"]:
            self.weight_parameter_regularization *= 1.0

        self.hash_encoding_config = {
            "otype": "HashGrid",
            "n_levels": 13,
            "n_features_per_level": 2,
            "log2_hashmap_size": 20,
            "base_resolution": 128,
            "per_level_scale": 1.3819,
        }
        if self.variant in ["snfa", "snfag"]:
            self.hash_encoding_config["base_resolution"] = 16
            self.hash_encoding_config["per_level_scale"] = 1.5
            self.hash_encoding_config["n_levels"] = 16

        self.base_network_config = {
            "otype": "CutlassMLP" if self.prefer_cutlass_over_fullyfused_mlp else "FullyFusedMLP",
            "activation": self.activation_function,
            "output_activation": "None",
            "n_neurons": 64,
            "n_hidden_layers": 1,
        }
        self.dir_encoding_config = {
            "otype": "Composite",
            "nested": [
                {"n_dims_to_encode": 3, "otype": "SphericalHarmonics", "degree": 4},
                {"otype": "Identity"},
            ],
        }
        self.rgb_network_config = {
            "otype": "CutlassMLP" if self.prefer_cutlass_over_fullyfused_mlp else "FullyFusedMLP",
            "activation": self.activation_function,
            "output_activation": "None",
            "n_neurons": 64,
            "n_hidden_layers": 2,
        }

        self.brightness_positional_encoding_config = {
            "otype": "Composite",
            "nested": [
                {
                    "n_dims_to_encode": 1,
                    "otype": "Frequency",
                    "n_frequencies": 3,
                },
                {"otype": "Identity"},
            ],
        }
        self.brightness_network_config = {
            "otype": "CutlassMLP" if self.prefer_cutlass_over_fullyfused_mlp else "FullyFusedMLP",
            "activation": self.activation_function,
            "output_activation": "None",
            "n_neurons": 64,
            "n_hidden_layers": 2,
        }
        self.num_feats_from_base_to_rgb = 15

        if self.do_pure_mlp:
            if self.use_viewdirs:
                self.positional_encoding_config = {
                    "otype": "Composite",
                    "nested": [
                        {
                            "n_dims_to_encode": 3,
                            "otype": "Frequency",
                            "n_frequencies": 10,
                        },
                        {
                            "n_dims_to_encode": 3,
                            "otype": "Frequency",
                            "n_frequencies": 4,
                        },
                    ],
                }
            else:
                self.positional_encoding_config = {
                    "otype": "Frequency",
                    "n_frequencies": 10,
                }
            self.pure_mlp_config = {
                "otype": "CutlassMLP"
                if self.prefer_cutlass_over_fullyfused_mlp
                else "FullyFusedMLP",
                "activation": self.activation_function,
                "output_activation": "None",
                "n_neurons": 256,
                "n_hidden_layers": 4,
            }
            self.skip_connections = 1

        self._init_networks()

    def _init_networks(self):
        self.base_enc_model = tcnn.Encoding(
            n_input_dims=3 + (1 if self.use_timevarying_geometry else 0),
            encoding_config=self.hash_encoding_config,
        )
        base_enc_output_size = (
            self.hash_encoding_config["n_levels"]
            * self.hash_encoding_config["n_features_per_level"]
        )
        self.base_mlp_model = tcnn.Network(
            n_input_dims=base_enc_output_size,
            n_output_dims=1 + self.num_feats_from_base_to_rgb,
            network_config=self.base_network_config,
        )
        self.rgb_mlp_model = tcnn.NetworkWithInputEncoding(
            n_input_dims=3 + self.num_feats_from_base_to_rgb,
            n_output_dims=3,
            encoding_config=self.dir_encoding_config,
            network_config=self.rgb_network_config,
        )
        if self.separate_appearance_model:
            self.separate_base_enc_model = tcnn.Encoding(
                n_input_dims=3 + (1 if self.use_timevarying_appearance else 0),
                encoding_config=self.hash_encoding_config,
            )
            base_enc_output_size = (
                self.hash_encoding_config["n_levels"]
                * self.hash_encoding_config["n_features_per_level"]
            )
            self.separate_base_mlp_model = tcnn.Network(
                n_input_dims=base_enc_output_size,
                n_output_dims=self.num_feats_from_base_to_rgb,
                network_config=self.base_network_config,
            )
            self.separate_rgb_mlp_model = tcnn.NetworkWithInputEncoding(
                n_input_dims=3 + self.num_feats_from_base_to_rgb,
                n_output_dims=3,
                encoding_config=self.dir_encoding_config,
                network_config=self.rgb_network_config,
            )
        if self.brightness_variability > 0.0:
            if self.separate_brightness_model:
                self.brightness_enc_model = tcnn.Encoding(
                    n_input_dims=4, encoding_config=self.hash_encoding_config
                )
                brightness_enc_output_size = (
                    self.hash_encoding_config["n_levels"]
                    * self.hash_encoding_config["n_features_per_level"]
                )
                self.brightness_mlp_model = tcnn.Network(
                    n_input_dims=brightness_enc_output_size,
                    n_output_dims=1,
                    network_config=self.brightness_network_config,
                )
            else:
                self.brightness_mlp_model = tcnn.NetworkWithInputEncoding(
                    n_input_dims=1 + self.num_feats_from_base_to_rgb,
                    n_output_dims=1,
                    encoding_config=self.brightness_positional_encoding_config,
                    network_config=self.brightness_network_config,
                )
        if self.do_pure_mlp:

            def _get_mlp(counter):
                first_layer = counter == 0
                last_layer = counter == self.skip_connections
                encoding_config = self.positional_encoding_config

                if self.do_pref:
                    time_conditioning_size = 1
                elif self.do_nrnerf or self.do_dnerf:
                    time_conditioning_size = 0

                if self.use_viewdirs:
                    if time_conditioning_size != 0:
                        raise NotImplementedError
                    # position
                    n_frequencies = encoding_config["nested"][0]["n_frequencies"]
                    skip_connection_dimension = 3 + n_frequencies * 2 * 3
                    # viewdirs
                    n_frequencies = encoding_config["nested"][1]["n_frequencies"]
                    skip_connection_dimension += 3 + n_frequencies * 2 * 3
                else:
                    n_frequencies = encoding_config["n_frequencies"]
                    skip_connection_dimension = (
                        3
                        + time_conditioning_size
                        + n_frequencies * 2 * (3 + time_conditioning_size)
                    )

                hidden_dimension = self.pure_mlp_config["n_neurons"]

                original_output_activation = self.pure_mlp_config["output_activation"]
                if not last_layer:
                    self.pure_mlp_config["output_activation"] = self.pure_mlp_config["activation"]

                mlp_dict = {
                    "n_input_dims": skip_connection_dimension + hidden_dimension,
                    "n_output_dims": hidden_dimension,
                    "network_config": self.pure_mlp_config,
                }

                if first_layer:
                    mlp_dict["n_input_dims"] = 3 + time_conditioning_size  # position and time
                    if self.use_viewdirs:
                        mlp_dict["n_input_dims"] += 3
                    mlp_dict["encoding_config"] = encoding_config
                if last_layer:
                    mlp_dict["n_output_dims"] = 4  # alpha and RGB

                network = build_pytorch_mlp_from_tinycudann(
                    mlp_dict, last_layer_zero_init=last_layer
                )

                if not last_layer:
                    self.pure_mlp_config["output_activation"] = original_output_activation

                return network

            self.pure_mlp = torch.nn.ModuleList(
                [_get_mlp(counter) for counter in range(1 + self.skip_connections)]
            )

    def forward(
        self,
        positions,
        view_directions,
        timesteps=None,
        pref_timesteps=None,
        mip_scales=None,
        returns=None,
        **kwargs
    ):

        if self.do_pure_mlp:
            if self.do_pref:
                base_input = torch.cat([positions, pref_timesteps], -1)
            elif self.do_nrnerf:
                base_input = positions
            elif self.do_dnerf:
                pass
            else:
                raise NotImplementedError
            if self.use_viewdirs:
                # dnerf
                pos_enc_positions = positional_encoding(
                    positions,
                    num_frequencies=self.positional_encoding_config["nested"][0]["n_frequencies"],
                )
                pos_enc_view_directions = positional_encoding(
                    view_directions,
                    num_frequencies=self.positional_encoding_config["nested"][1]["n_frequencies"],
                )
                model_input = torch.cat([pos_enc_positions, pos_enc_view_directions], dim=-1)
            else:
                model_input = positional_encoding(
                    base_input, num_frequencies=self.positional_encoding_config["n_frequencies"]
                )
            network_output = model_input
            for counter, pure_mlp_model in enumerate(self.pure_mlp):
                if counter > 0:  # skip connection
                    network_output = torch.cat([model_input, network_output], dim=-1)
                network_output = pure_mlp_model(network_output)
            alpha = network_output[:, 0]
            rgb = 0.1 * network_output[:, 1:] + 0.5
            rgb = project_to_correct_range(rgb, mode="zick_zack")
        else:

            timevarying_base_input = torch.cat([positions, timesteps], -1)
            constant_base_input = positions

            # geometry

            if self.use_timevarying_geometry:
                base_input = timevarying_base_input
            else:
                base_input = constant_base_input

            base_enc_output = self.base_enc_model(base_input)
            base_output = self.base_mlp_model(base_enc_output)

            alpha = base_output[:, 0]

            # RGB, including view-dependence

            if not self.use_viewdirs:
                view_directions = 0.0 * view_directions

            if self.separate_appearance_model:

                if self.use_timevarying_appearance:
                    base_input = timevarying_base_input
                else:
                    base_input = constant_base_input

                base_enc_output = self.separate_base_enc_model(base_input)
                base_output = self.separate_base_mlp_model(base_enc_output)

                concat_feats = torch.cat([view_directions, base_output], 1)
                rgb = self.separate_rgb_mlp_model(concat_feats)

            else:
                concat_feats = torch.cat([view_directions, base_output[:, 1:]], 1)
                rgb = self.rgb_mlp_model(concat_feats)
            standard = True
            if standard:
                rgb = 0.1 * rgb + 0.5
                rgb = project_to_correct_range(rgb, mode="zick_zack")
            else:
                rgb = 0.1 * rgb
                rgb = torch.sigmoid(rgb)

        # brightness

        if self.brightness_variability > 0.0:
            if self.separate_brightness_model:
                concat_feats_2 = self.brightness_enc_model(torch.cat([positions, timesteps], -1))
            else:
                concat_feats_2 = torch.cat([timesteps, base_output[:, 1:]], 1)
            brightness_change = self.brightness_mlp_model(concat_feats_2)
            brightness_change = brightness_change * 0.001

            brightness_change = project_to_correct_range(
                brightness_change,
                min_=-self.brightness_variability,
                max_=self.brightness_variability,
                mode="zick_zack",
            )

            returns.add_return("brightness_change", brightness_change)

            if self.debug and torch.rand(1) < 0.01:
                LOGGER.debug(
                    "brightness change: "
                    + str(torch.mean(brightness_change).item())
                    + " +- "
                    + str(torch.std(brightness_change).item())
                )

            rgb = rgb * (1.0 + brightness_change)
            rgb = project_to_correct_range(rgb, mode="zick_zack")

        if "editing" in kwargs and kwargs["editing"] is not None:
            rgb, alpha = kwargs["editing"](positions, rgb, alpha)

        # returns

        returns.add_return("rgb_per_point", rgb)
        returns.add_return("raw_alpha", alpha)
        return rgb, alpha

    def get_parameters_with_optimization_information(self):

        params = []

        params.append(
            {
                "name": "base_enc_model",
                "tags": ["autodecoding"],
                "parameters": list(self.base_enc_model.parameters()),
                "optimizer": "SmartAdam",
                "learning_rate": 1e-2,
                "decay_steps": self.learning_rate_decay_autodecoding_iterations,
                "decay_rate": self.learning_rate_decay_autodecoding_fraction,
                "weight_decay": self.weight_parameter_regularization,  # 0.0,
            }
        )

        params.append(
            {
                "name": "base_mlp_model",
                "tags": ["mlp"],
                "parameters": list(self.base_mlp_model.parameters()),
                "optimizer": "Adam",
                "learning_rate": 1e-2,
                "decay_steps": self.learning_rate_decay_mlp_iterations,
                "decay_rate": self.learning_rate_decay_mlp_fraction,
                "weight_decay": self.weight_parameter_regularization,  # 0.0,
            }
        )

        params.append(
            {
                "name": "rgb_mlp_model",
                "tags": ["mlp"],
                "parameters": list(self.rgb_mlp_model.parameters()),
                "optimizer": "Adam",
                "learning_rate": 1e-2,
                "decay_steps": self.learning_rate_decay_mlp_iterations,
                "decay_rate": self.learning_rate_decay_mlp_fraction,
                "weight_decay": self.weight_parameter_regularization,  # 1e-6,
            }
        )

        if self.separate_appearance_model:
            params.append(
                {
                    "name": "separate_base_enc_model",
                    "tags": ["autodecoding", "separate_appearance"],
                    "parameters": list(self.separate_base_enc_model.parameters()),
                    "optimizer": "SmartAdam",
                    "learning_rate": 1e-2,
                    "decay_steps": self.learning_rate_decay_autodecoding_iterations,
                    "decay_rate": self.learning_rate_decay_autodecoding_fraction,
                    "weight_decay": self.weight_parameter_regularization,  # 0.0,
                }
            )

            params.append(
                {
                    "name": "separate_base_mlp_model",
                    "tags": ["mlp", "separate_appearance"],
                    "parameters": list(self.separate_base_mlp_model.parameters()),
                    "optimizer": "Adam",
                    "learning_rate": 1e-2,
                    "decay_steps": self.learning_rate_decay_mlp_iterations,
                    "decay_rate": self.learning_rate_decay_mlp_fraction,
                    "weight_decay": self.weight_parameter_regularization,  # 0.0,
                }
            )

            params.append(
                {
                    "name": "separate_rgb_mlp_model",
                    "tags": ["mlp", "separate_appearance"],
                    "parameters": list(self.separate_rgb_mlp_model.parameters()),
                    "optimizer": "Adam",
                    "learning_rate": 1e-2,
                    "decay_steps": self.learning_rate_decay_mlp_iterations,
                    "decay_rate": self.learning_rate_decay_mlp_fraction,
                    "weight_decay": self.weight_parameter_regularization,  # 1e-6,
                }
            )

        if self.brightness_variability > 0.0:
            if self.separate_brightness_model:
                params.append(
                    {
                        "name": "brightness_enc_model",
                        "tags": ["autodecoding", "separate_brightness"],
                        "parameters": list(self.brightness_enc_model.parameters()),
                        "optimizer": "SmartAdam",
                        "learning_rate": 1e-2,
                        "decay_steps": self.learning_rate_decay_autodecoding_iterations,
                        "decay_rate": self.learning_rate_decay_autodecoding_fraction,
                        "weight_decay": self.weight_parameter_regularization,
                    }
                )
            params.append(
                {
                    "name": "brightness_mlp_model",
                    "tags": ["mlp"],
                    "parameters": list(self.brightness_mlp_model.parameters()),
                    "optimizer": "Adam",
                    "learning_rate": 1e-2,
                    "decay_steps": self.learning_rate_decay_mlp_iterations,
                    "decay_rate": self.learning_rate_decay_mlp_fraction,
                    "weight_decay": self.weight_parameter_regularization,
                }
            )
            if self.separate_brightness_model:
                params[-1]["tags"].append("separate_brightness")

        if self.do_pure_mlp:
            params.append(
                {
                    "name": "pure_mlp",
                    "tags": ["mlp"],
                    "parameters": list(self.pure_mlp.parameters()),
                    "optimizer": "Adam",
                    "learning_rate": 1e-3,
                    "decay_steps": self.learning_rate_decay_mlp_iterations,
                    "decay_rate": self.learning_rate_decay_mlp_fraction,
                    "weight_decay": self.weight_parameter_regularization,  # 0.0,
                }
            )

        return params

    def step(self):
        pass

    def get_regularization_losses(self):
        regularization_losses = {}

        return regularization_losses
