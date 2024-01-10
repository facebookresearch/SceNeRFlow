# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import logging

import torch
from smart_adam import SmartAdam

LOGGER = logging.getLogger(__name__)


class Optimizer:
    def __init__(self, settings, scene, renderer):

        self.debug = settings.debug

        self.parameter_sources = [scene, renderer]

        self.optimizers_with_information = []
        for param_with_info in self.get_parameters_with_information():

            # tag:half_precision : we need Adam because it's invariant under global scaling of the loss.

            if param_with_info["optimizer"] == "Adam":
                optimizer = torch.optim.AdamW(
                    params=param_with_info["parameters"],
                    lr=param_with_info["learning_rate"],
                    weight_decay=param_with_info["weight_decay"],
                    betas=(0.9, 0.99),
                    eps=1e-15,
                )
            elif param_with_info["optimizer"] == "SmartAdam":
                optimizer = SmartAdam(
                    params=param_with_info["parameters"],
                    lr=param_with_info["learning_rate"],
                    weight_decay=param_with_info["weight_decay"],
                    betas=(0.9, 0.99),
                    eps=1e-15,
                )

            self.optimizers_with_information.append(
                {
                    "name": param_with_info["name"],
                    "tags": param_with_info["tags"],
                    "optimizer": optimizer,
                    "parameters": param_with_info["parameters"],
                    "initial_learning_rate": param_with_info["learning_rate"],
                    "decay_steps": param_with_info["decay_steps"],
                    "decay_rate": param_with_info["decay_rate"],
                }
            )

    def get_parameters_with_information(self):
        parameters_with_information = []
        for parameter_source in self.parameter_sources:
            parameters_with_information += (
                parameter_source.get_parameters_with_optimization_information()
            )
        return parameters_with_information

    def zero_grad(self, set_to_none=None):
        if set_to_none is None:
            set_to_none = True
        for optimizer_with_info in self.optimizers_with_information:
            optimizer_with_info["optimizer"].zero_grad(set_to_none=set_to_none)

    def scale_gradients(self, factor):
        if factor == 1.0:
            return
        with torch.no_grad():
            for param_with_info in self.get_parameters_with_information():
                for param in param_with_info["parameters"]:
                    if param.grad is not None:
                        param.grad *= factor

    def step(self, scaler=None, use_gradient_scaling=False):
        for optimizer_with_info in self.optimizers_with_information:
            if use_gradient_scaling:
                scaler.step(optimizer_with_info["optimizer"])
            else:
                optimizer_with_info["optimizer"].step()

        if self.debug and torch.rand(1) < 0.01:
            max_abs_grad_value = torch.zeros(1).cuda()
            max_abs_grad_name = "-"

            max_abs_value = torch.zeros(1).cuda()
            max_abs_name = "-"

            for param_with_info in self.get_parameters_with_information():
                for param in param_with_info["parameters"]:

                    if not torch.all(torch.isfinite(param)):
                        try:
                            mask = torch.where(~torch.isfinite(param))
                            LOGGER.debug(
                                "Non-finite value in "
                                + param_with_info["name"]
                                + ": "
                                + str(param[mask])
                            )
                        except Exception:
                            LOGGER.debug("Non-finite value in " + param_with_info["name"])

                    if param.grad is not None and not torch.all(torch.isfinite(param.grad)):
                        try:
                            mask = torch.where(~torch.isfinite(param.grad))
                            LOGGER.debug(
                                "Non-finite value in "
                                + param_with_info["name"]
                                + ".grad: "
                                + str(param.grad[mask])
                            )
                        except Exception:
                            LOGGER.debug("Non-finite value in " + param_with_info["name"] + ".grad")

                    this_max = torch.max(torch.abs(param))
                    if this_max > max_abs_value:
                        max_abs_value = this_max
                        max_abs_name = param_with_info["name"]

                    if param.grad is not None:
                        this_max = torch.max(torch.abs(param.grad))
                        if this_max > max_abs_grad_value:
                            max_abs_grad_value = this_max
                            max_abs_grad_name = param_with_info["name"]

            LOGGER.debug("Parameter max value " + str(max_abs_value.item()) + " in " + max_abs_name)
            LOGGER.debug(
                "Gradient max value " + str(max_abs_grad_value.item()) + " in " + max_abs_grad_name
            )

    def load_state_dict(self, state_dict):
        for optimizer_with_info, this_state_dict in zip(
            self.optimizers_with_information, state_dict["optimizers"]
        ):
            optimizer_with_info["optimizer"].load_state_dict(this_state_dict["optimizer"])
            optimizer_with_info["initial_learning_rate"] = this_state_dict["initial_learning_rate"]
            optimizer_with_info["decay_steps"] = this_state_dict["decay_steps"]
            optimizer_with_info["decay_rate"] = this_state_dict["decay_rate"]

    def state_dict(self):
        optimizers = []

        for optimizer_with_info in self.optimizers_with_information:
            optimizers.append(
                {
                    "optimizer": optimizer_with_info["optimizer"].state_dict(),
                    "initial_learning_rate": optimizer_with_info["initial_learning_rate"],
                    "decay_steps": optimizer_with_info["decay_steps"],
                    "decay_rate": optimizer_with_info["decay_rate"],
                }
            )

        return {"optimizers": optimizers}
