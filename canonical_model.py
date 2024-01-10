# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch


def get_canonical_model(settings):
    from canonical_model_ngp import CanonicalModelNGP

    if settings.backbone == "ngp":
        return CanonicalModelNGP(settings)


class CanonicalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
