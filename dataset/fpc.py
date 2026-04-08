import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


class FpcDataset(Dataset):

    def __init__(self, data_root: str, split: str, target_field: str = "velocity"):
        """
        Args:
            data_root:    Directory containing parsed .npz and .dat files.
            split:        'train', 'valid', or 'test'.
            target_field: 'velocity' (default) or 'pressure'.
        """
        if target_field not in ("velocity", "pressure"):
            raise ValueError("target_field must be 'velocity' or 'pressure', got: %s" % target_field)

        self.target_field = target_field
        meta_path = os.path.join(data_root, split + ".npz")
        data_path = os.path.join(data_root, split + ".dat")

        meta_keys = ("pos", "node_type", "cells", "indices", "cindices", "all_velocity_shape")
        tmp = np.load(meta_path, allow_pickle=True)
        self.meta = {key: tmp[key] for key in meta_keys}

        vel_shape = self.meta["all_velocity_shape"]
        self.fp = np.memmap(data_path, dtype="float32", mode="r", shape=vel_shape)

        # Pressure field (optional)
        self.fp_pressure = None
        if target_field == "pressure":
            pressure_path = os.path.join(data_root, split + "_pressure.dat")
            if not os.path.exists(pressure_path):
                raise FileNotFoundError(
                    "Pressure data not found: %s\n"
                    "Re-run parse_tfrecord.py to extract the pressure field." % pressure_path
                )
            pressure_shape = (vel_shape[0], vel_shape[1], 1)
            self.fp_pressure = np.memmap(pressure_path, dtype="float32", mode="r",
                                         shape=pressure_shape)

        self.tra_len = self.fp.shape[1]
        self.num_sampes_per_tra = self.tra_len - 1
        tras_nums = len(self.meta["indices"]) - 1
        self.total_samples = tras_nums * self.num_sampes_per_tra

    def __getitem__(self, index: int) -> Data:
        tra_index        = index // self.num_sampes_per_tra
        tra_sample_index = index % (self.tra_len - 1)
        tra_start_index  = self.meta["indices"][tra_index]
        tra_end_index    = self.meta["indices"][tra_index + 1]
        ctra_start_index = self.meta["cindices"][tra_index]
        ctra_end_index   = self.meta["cindices"][tra_index + 1]

        pos       = self.meta["pos"][tra_start_index:tra_end_index]
        node_type = self.meta["node_type"][tra_start_index:tra_end_index]
        cells     = self.meta["cells"][ctra_start_index:ctra_end_index]

        if self.target_field == "pressure":
            pressure_t   = self.fp_pressure[tra_start_index:tra_end_index, tra_sample_index]     # [N, 1]
            pressure_tp1 = self.fp_pressure[tra_start_index:tra_end_index, tra_sample_index + 1] # [N, 1]
            x = np.concatenate([node_type, pressure_t], axis=-1)   # [N, 2]
            y = pressure_tp1                                         # [N, 1]
        else:
            tra_velocity = self.fp[tra_start_index:tra_end_index, tra_sample_index]       # [N, 2]
            tra_target   = self.fp[tra_start_index:tra_end_index, tra_sample_index + 1]  # [N, 2]
            x = np.concatenate([node_type, tra_velocity], axis=-1)  # [N, 3]
            y = tra_target                                            # [N, 2]

        graph = Data(
            x    = torch.as_tensor(x.copy(),   dtype=torch.float32),
            pos  = torch.as_tensor(pos.copy(), dtype=torch.float32),
            face = torch.as_tensor(cells.T.copy(), dtype=torch.int64),
            y    = torch.as_tensor(y.copy(),   dtype=torch.float32),
        )
        return graph

    def __len__(self) -> int:
        return self.total_samples
