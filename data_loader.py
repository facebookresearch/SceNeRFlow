# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
class DataLoader:
    def __init__(self):
        pass


def get_data_loader(settings, rank, world_size):
    from data_loader_blender import DataLoaderBlender

    if settings.dataset_type == "blender":
        return DataLoaderBlender(settings, rank, world_size)
