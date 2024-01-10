# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import logging
import sys

import numpy as np
import torch
from losses import Losses
from multi_gpu import multi_gpu_sync_gradients
from optimizer import Optimizer

logging.getLogger("matplotlib").setLevel(logging.WARNING)
LOGGER = logging.getLogger(__name__)


class Trainer:
    def __init__(self, settings, scene, renderer, world_size):

        super().__init__()

        self.multi_gpu = settings.multi_gpu

        self.use_gradient_scaling = False
        self.scaler = torch.cuda.amp.GradScaler()

        self.scene = scene
        self.optimizer = Optimizer(settings, scene, renderer)
        self.losses = Losses(settings, world_size)

        # virtual batches
        self.num_virtual_batches = 1  # init
        self.iterations_since_last_change = 0  # init
        self.automatically_adjust_num_virtual_batches = True
        self.try_fewer_virtual_batches_every = 100
        self.max_num_virtual_batches = 16

    def get_optimizer(self):
        return self.optimizer

    def zero_grad(self, set_to_none=None):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def backward(self, loss, multi_gpu_sync=True):

        if self.use_gradient_scaling:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        if self.multi_gpu and multi_gpu_sync:
            multi_gpu_sync_gradients(self.optimizer.get_parameters())

    def losses_and_backward_with_virtual_batches(self, *args):

        while True:
            log = self._losses_and_backward_with_virtual_batches(
                *args
            )  # need to call this from here instead of from within itself, such that the raised exception is out of scope
            if (
                log is None
            ):  # out of memory happened. can only happen until num_virtual_batches > max_num_virtual_batches.
                # cleanup, free memory
                self.zero_grad(set_to_none=True)  # sets gradients to None
                torch.cuda.empty_cache()
            else:
                return log

    def _losses_and_backward_with_virtual_batches(
        self, batch, scene, renderer, scheduler, training_iteration
    ):

        manually_accumulate_gradients = True
        if manually_accumulate_gradients:
            accumulated_gradients = {}
            for param_with_info in self.optimizer.get_parameters_with_information():
                accumulated_gradients[param_with_info["name"]] = {
                    param_index: None for param_index in range(len(param_with_info["parameters"]))
                }

        num_rays = batch["rays_origin"].shape[0]
        rays_per_virtual_batch = max(1, int(np.ceil(num_rays / self.num_virtual_batches)))

        for virtual_batch_index, virtual_batch_start in enumerate(
            range(0, num_rays, rays_per_virtual_batch)
        ):

            try:
                virtual_batch = {
                    key: tensor[virtual_batch_start : virtual_batch_start + rays_per_virtual_batch]
                    for key, tensor in batch.items()
                }

                LOGGER.debug(
                    "virtual_batch_index: "
                    + str(virtual_batch_index)
                    + " | virtual batch start: "
                    + str(virtual_batch_start)
                    + " | virtual batch size: "
                    + str(virtual_batch["rays_origin"].shape)
                )

                training_loss, loss_scaling_factor, log = self.losses.compute(
                    virtual_batch, scene, renderer, scheduler, training_iteration
                )

                self.backward(training_loss, multi_gpu_sync=False)

                if manually_accumulate_gradients:
                    with torch.no_grad():
                        for param_with_info in self.optimizer.get_parameters_with_information():
                            this_acc_grad = accumulated_gradients[param_with_info["name"]]
                            for param_index, this_param in enumerate(param_with_info["parameters"]):
                                if this_param.grad is not None:
                                    if this_acc_grad[param_index] is None:
                                        this_acc_grad[param_index] = this_param.grad.clone()
                                    else:
                                        this_acc_grad[param_index] += this_param.grad

                    self.zero_grad()

            except RuntimeError as exception:  # handle out of memory
                if self.num_virtual_batches > self.max_num_virtual_batches:

                    LOGGER.warning(
                        "trying to exceed maximum number of virtual batches: "
                        + str(self.num_virtual_batches)
                        + " / "
                        + str(self.max_num_virtual_batches)
                    )
                    raise exception

                elif (
                    any(oom in str(exception) for oom in ["out of memory", "OUT_OF_MEMORY"])
                    and self.automatically_adjust_num_virtual_batches
                ):

                    sys.stderr.flush()

                    LOGGER.info(
                        "virtual batch too large, need more than "
                        + str(self.num_virtual_batches)
                        + " virtual batches"
                    )

                    self.num_virtual_batches += 1
                    self.iterations_since_last_change = 0

                    return None

                else:
                    raise exception

        if manually_accumulate_gradients:
            with torch.no_grad():
                for param_with_info in self.optimizer.get_parameters_with_information():
                    this_acc_grad = accumulated_gradients[param_with_info["name"]]
                    for param_index, this_param in enumerate(param_with_info["parameters"]):
                        this_param.grad = this_acc_grad[param_index]
            del accumulated_gradients

        self.optimizer.scale_gradients(
            factor=1.0 / (float(self.num_virtual_batches) * loss_scaling_factor)
        )

        if self.multi_gpu:
            multi_gpu_sync_gradients(self.optimizer.get_parameters())

        self.iterations_since_last_change += 1
        if (
            self.iterations_since_last_change >= self.try_fewer_virtual_batches_every
            and self.num_virtual_batches > 1
            and self.automatically_adjust_num_virtual_batches
        ):
            self.num_virtual_batches -= 1
            self.iterations_since_last_change = 0  # not really necessary but might reduce the number of OOM exceptions in some weird edge cases

        return log

    def step(self):

        self.optimizer.step(self.scaler, self.use_gradient_scaling)
        if self.use_gradient_scaling:
            self.scaler.update()

        self.scene.step()  # e.g. for enforcing hard constraints via projection

    def state_dict(self):
        return {"optimizer": self.optimizer.state_dict(), "scaler": self.scaler.state_dict()}

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.scaler.load_state_dict(state_dict["scaler"])
