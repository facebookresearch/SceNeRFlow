# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import json
import os


class BinaryDataset:
    def __init__(self, folder, name=None, delete_existing=False, read_only=False):

        if name is None:
            name = "dataset"
        self.name = name
        self.folder = folder

        self.read_only = read_only

        self.open(delete_existing=delete_existing)

    def _get_dataset_index_file_name(self):
        return os.path.join(self.folder, self.name + "_index.json")

    def _get_dataset_file_name(self):
        return os.path.join(self.folder, self.name + ".bin")

    def open(self, delete_existing=False):

        dataset_file = self._get_dataset_file_name()
        if self.read_only:
            mode = "br"
        else:
            if delete_existing:
                mode = "bw+"
            else:
                mode = "ba+"
        self._dataset_bin = open(dataset_file, mode)

        dataset_index_file = self._get_dataset_index_file_name()
        if os.path.exists(dataset_index_file) and not delete_existing:
            with open(dataset_index_file, "r") as json_file:
                self._dataset_index = json.load(json_file)
            if len(self._dataset_index) > 0:
                self._start = max([entry["end"] for entry in self._dataset_index.values()])
            else:
                self._start = 0
            self._modified = False
        else:
            self._dataset_index = {}
            self._start = 0
            self._modified = True

    def maybe_add_entry(self, entry_bytes, key):

        if self.read_only:
            raise RuntimeError("trying to add to BinaryDataset in read_only mode")

        key = str(key)

        if key in self:
            return

        self._modified = True

        self._dataset_bin.seek(self._start)

        self._end = self._start + entry_bytes.getbuffer().nbytes
        self._dataset_index[key] = {
            "start": self._start,  # inclusive
            "end": self._end,  # exclusive
        }
        self._start = self._end

        self._dataset_bin.write(entry_bytes.getbuffer())

    def __contains__(self, key):
        return str(key) in self._dataset_index

    def keys(self):
        return self._dataset_index.keys()

    def get_entry(self, key):

        key = str(key)

        start = self._dataset_index[key]["start"]
        end = self._dataset_index[key]["end"]

        self._dataset_bin.seek(start)
        entry_bytes = self._dataset_bin.read(end - start)

        return entry_bytes

    def close(self):

        self._dataset_bin.close()

        if self._modified:
            dataset_index_file = self._get_dataset_index_file_name()
            with open(dataset_index_file, "w", encoding="utf-8") as json_file:
                json.dump(self._dataset_index, json_file, ensure_ascii=False, indent=4)

    def flush(self):
        self.close()
        self.open()
