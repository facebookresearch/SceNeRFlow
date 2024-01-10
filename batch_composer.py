# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from random import shuffle

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch


class BatchComposer:
    def __init__(self, batch_builder):
        self.batch_builder = batch_builder

    def set_batch_composition(self, batch_composition):
        self.batch_composition = batch_composition

    def _determine_subbatch_sizes(self, batch_size):

        shuffle(self.batch_composition)  # in-place shuffling

        subbatch_sizes = []
        for subbatch_info in self.batch_composition:
            if len(subbatch_info["imageids"]) == 0:
                subbatch_size = 0
            else:
                subbatch_size = int(subbatch_info["fraction"] * batch_size)
            subbatch_sizes.append(subbatch_size)

        final_subbatch_sizes = []
        remaining = batch_size - sum(subbatch_sizes)
        for subbatch_size in subbatch_sizes:
            if subbatch_size > 0 and remaining > 0:
                final_subbatch_sizes.append(subbatch_size + remaining)
            else:
                final_subbatch_sizes.append(subbatch_size)

        return final_subbatch_sizes

    def compose(self, batch_size, precomputed):

        subbatch_sizes = self._determine_subbatch_sizes(batch_size)

        subbatches = []
        for subbatch_size, subbatch_info in zip(subbatch_sizes, self.batch_composition):
            if subbatch_size > 0:
                imageids = subbatch_info["imageids"]  # among all training images

                subbatch = self.batch_builder.build(
                    active_imageids=imageids, batch_size=subbatch_size, precomputed=precomputed
                )

                subbatches.append(subbatch)

        batch = {
            key: torch.cat([subbatch[key] for subbatch in subbatches], axis=0)
            for key in subbatches[0].keys()
        }

        return batch
