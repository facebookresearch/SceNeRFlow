# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch


class PreCorrection(torch.nn.Module):
    def __init__(self, settings):
        super().__init__()

    def forward(self, batch, returns=None):
        return batch

    def get_parameters_with_optimization_information(self):
        return []

    def get_regularization_losses(self):
        return {}
