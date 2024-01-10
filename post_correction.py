# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import logging

import torch
from utils import project_to_correct_range

LOGGER = logging.getLogger(__name__)


class ColorCalibration(torch.nn.Module):
    def __init__(self, settings):

        super().__init__()

        self.color_calibration_mode = settings.color_calibration_mode

        if self.color_calibration_mode == "none":
            pass

        elif self.color_calibration_mode == "full_matrix":

            num_train_images = data_handler.get_training_set_size()

            self.register_parameter(
                name="imageids_to_bias",
                param=torch.nn.Parameter(torch.zeros(num_train_images, 3, 1)),
            )
            self.register_parameter(
                name="imageids_to_full_matrix",
                param=torch.nn.Parameter(torch.zeros(num_train_images, 3, 3)),
            )
            with torch.no_grad():
                self.imageids_to_full_matrix[:, 0, 0] = 1.0
                self.imageids_to_full_matrix[:, 1, 1] = 1.0
                self.imageids_to_full_matrix[:, 2, 2] = 1.0

        elif self.color_calibration_mode == "neural_volumes":

            num_train_images = data_handler.get_training_set_size()

            self.register_parameter(
                name="imageids_to_scalings",
                param=torch.nn.Parameter(torch.ones(num_train_images, 3)),
            )
            self.register_parameter(
                name="imageids_to_biases",
                param=torch.nn.Parameter(torch.zeros(num_train_images, 3)),
            )

    def forward(self, rgb, batch, returns=None):

        if self.color_calibration_mode == "none":
            return rgb

        elif self.color_calibration_mode == "full_matrix":

            biases = self.imageids_to_bias[batch["image_indices"]]  # N x 3 x 1
            matrices = self.imageids_to_full_matrix[batch["image_indices"]]  # N x 3 x 3
            rgb = torch.matmul(matrices, rgb.view(-1, 3, 1)) + biases  # N x 3 x 1
            rgb = rgb.view(-1, 3)  # N x 3

            rgb = project_to_correct_range(rgb, mode="zick_zack")

            return rgb

        elif self.color_calibration_mode == "neural_volumes":

            biases = self.imageids_to_bias[batch["image_indices"]]  # N x 3
            scalings = self.imageids_to_scalings[batch["image_indices"]]  # N x 3
            rgb = scalings * rgb + biases  # N x 3

            rgb = project_to_correct_range(rgb, mode="zick_zack")

            return rgb

    def get_parameters_with_optimization_information(self):
        params = []

        if self.color_calibration_mode == "full_matrix":
            params.append(
                {
                    "name": "color_calibration",
                    "parameters": self.parameters(),
                    "optimizer": "SmartAdam",
                    "learning_rate": 1e-6,
                    "decay_steps": self.learning_rate_decay_iterations,
                    "decay_rate": self.learning_rate_decay_fraction,
                    "weight_decay": 0.0,
                }
            )
        elif self.color_calibration_mode == "neural_volumes":
            params.append(
                {
                    "name": "color_calibration",
                    "parameters": self.parameters(),
                    "optimizer": "SmartAdam",
                    "learning_rate": 1e-6,
                    "decay_steps": self.learning_rate_decay_iterations,
                    "decay_rate": self.learning_rate_decay_fraction,
                    "weight_decay": 0.0,
                }
            )

        return params

    def get_regularization_losses(self):
        return {}


class Background(torch.nn.Module):
    def __init__(self, settings):
        super().__init__()

    def forward(self, rgb, batch, accumulated_weights, returns=None):

        if "background" in batch:
            rgb = rgb + (1.0 - accumulated_weights).view(-1, 1) * batch["background"]
        return rgb

    def get_parameters_with_optimization_information(self):
        return []

    def get_regularization_losses(self):
        return {}


class VignettingCorrection(torch.nn.Module):
    def __init__(self, settings):
        super().__init__()

        self.do_vignetting_correction = settings.do_vignetting_correction

        self.learning_rate_decay_autodecoding_iterations = (
            settings.learning_rate_decay_autodecoding_iterations
        )
        self.learning_rate_decay_autodecoding_fraction = (
            settings.learning_rate_decay_autodecoding_fraction
        )

        # assume all cameras follow the same vignetting
        self.vignetting_parameters = torch.nn.Parameter(torch.zeros((3,), dtype=torch.float32))

    def forward(self, rgb, batch, returns=None):

        if not self.do_vignetting_correction or "normalized_x_coordinate" not in batch:
            return rgb

        k1, k2, k3 = torch.unbind(self.vignetting_parameters)
        r = batch["normalized_x_coordinate"] ** 2 + batch["normalized_y_coordinate"] ** 2

        offset = k1 * r + k2 * r**2 + k3 * r**3

        rgb = rgb * (1.0 + offset.view(-1, 1))

        rgb = project_to_correct_range(rgb, mode="zick_zack")

        return rgb

    def get_parameters_with_optimization_information(self):
        params = []
        if self.do_vignetting_correction:
            params.append(
                {
                    "name": "vignetting_parameters",
                    "tags": ["autodecoding", "vignetting"],
                    "parameters": [self.vignetting_parameters],
                    "optimizer": "Adam",
                    "learning_rate": 1e-2,
                    "decay_steps": self.learning_rate_decay_autodecoding_iterations,
                    "decay_rate": self.learning_rate_decay_autodecoding_fraction,
                    "weight_decay": 0.0,
                }
            )
        return params

    def get_regularization_losses(self):
        return {}


class PostCorrection(torch.nn.Module):
    def __init__(self, settings):
        super().__init__()

        self.color_calibration = ColorCalibration(settings)
        self.background = Background(settings)
        self.vignetting_correction = VignettingCorrection(settings)

    def forward(self, rgb, batch, accumulated_weights, is_training, returns=None):

        rgb = self.vignetting_correction(rgb, batch, returns=returns)

        if is_training:
            rgb = self.color_calibration(rgb, batch, returns=returns)

        rgb = self.background(rgb, batch, accumulated_weights, returns=returns)

        returns.add_return("corrected_rgb", rgb)

        return rgb

    def get_parameters_with_optimization_information(self):
        color_calibration_parameters = (
            self.color_calibration.get_parameters_with_optimization_information()
        )
        background_parameters = self.background.get_parameters_with_optimization_information()
        vignetting_correction_parameters = (
            self.vignetting_correction.get_parameters_with_optimization_information()
        )
        return (
            color_calibration_parameters + background_parameters + vignetting_correction_parameters
        )

    def get_regularization_losses(self):
        regularization_losses = {}
        regularization_losses[
            "color_calibration"
        ] = self.color_calibration.get_regularization_losses()
        regularization_losses["background"] = self.background.get_regularization_losses()
        regularization_losses[
            "vignetting_correction"
        ] = self.vignetting_correction.get_regularization_losses()
        return regularization_losses
