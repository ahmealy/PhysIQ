"""
FlagDataset — loads parsed flag_simple cloth simulation data.

Data files produced by data/parse_flag_tfrecord.py:
    {data_root}/{split}/traj_{i:05d}.npz  — one file per trajectory containing:
        world_pos  [T, N, 3]  float32
        mesh_pos   [N, 2]     float32
        node_type  [N, 1]     int32
        cells      [F, 3]     int32

    {data_root}/{split}_index.npz — index file with n_traj, steps_per_traj

Returns a PyG Data object per (trajectory, timestep) pair with:
    graph.x         [N, 4]  — concat(world_pos_t[3], node_type[1])
    graph.prev_x    [N, 3]  — world_pos_{t-1}  (= world_pos_t at t=0)
    graph.pos       [N, 2]  — mesh_pos (2D rest configuration)
    graph.world_pos [N, 3]  — world_pos_t
    graph.face      [3, F]  — triangle connectivity (int64)
    graph.y         [N, 3]  — world_pos_{t+1} (regression target)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


class FlagDataset(Dataset):

    def __init__(self, data_root: str, split: str):
        self._split_dir  = os.path.join(data_root, split)
        index_path = os.path.join(data_root, f"{split}_index.npz")

        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"Flag index file not found: {index_path}\n"
                "Re-run: python data/parse_flag_tfrecord.py"
            )
        if not os.path.isdir(self._split_dir):
            raise FileNotFoundError(
                f"Flag trajectory directory not found: {self._split_dir}\n"
                "Re-run: python data/parse_flag_tfrecord.py"
            )

        idx = np.load(index_path)
        self.n_traj         = int(idx["n_traj"])
        steps_per_traj      = idx["steps_per_traj"].tolist()   # T per trajectory

        # Each trajectory yields T-1 (t, t+1) pairs
        self.steps_per_traj = [s - 1 for s in steps_per_traj]
        self.total_samples  = sum(self.steps_per_traj)

        bad_trajs = [i for i, s in enumerate(self.steps_per_traj) if s < 1]
        if bad_trajs:
            raise ValueError(
                f"Trajectories {bad_trajs} have fewer than 2 timesteps (T < 2). "
                "Data may be corrupted. Re-run parse_flag_tfrecord.py."
            )

        # Cumulative step counts for index → (traj_idx, t) mapping
        self._cum_steps = np.cumsum([0] + self.steps_per_traj)

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, index: int) -> Data:
        traj_idx = int(np.searchsorted(self._cum_steps[1:], index, side="right"))
        t = index - self._cum_steps[traj_idx]

        # Load trajectory on demand — no RAM accumulation
        traj_path = os.path.join(self._split_dir, f"traj_{traj_idx:05d}.npz")
        traj = np.load(traj_path)

        world_pos = traj["world_pos"]   # [T, N, 3]  float32
        mesh_pos  = traj["mesh_pos"]    # [N, 2]     float32
        node_type = traj["node_type"]   # [N, 1]     int32
        cells     = traj["cells"]       # [F, 3]     int32

        world_pos_t    = world_pos[t].astype(np.float32)          # [N, 3]
        world_pos_tp1  = world_pos[t + 1].astype(np.float32)      # [N, 3]
        # At t=0 there is no previous frame — use current as previous (zero velocity)
        world_pos_prev = world_pos[t - 1 if t > 0 else t].astype(np.float32)  # [N, 3]

        mesh_pos_f  = mesh_pos.astype(np.float32)
        node_type_f = node_type.astype(np.float32)
        cells_i     = cells.astype(np.int64)

        # Node features: concat(world_pos_t, node_type) → [N, 4]
        x = np.concatenate([world_pos_t, node_type_f], axis=-1)

        graph = Data(
            x          = torch.as_tensor(x,               dtype=torch.float32),
            prev_x     = torch.as_tensor(world_pos_prev,  dtype=torch.float32),
            pos        = torch.as_tensor(mesh_pos_f,       dtype=torch.float32),
            world_pos  = torch.as_tensor(world_pos_t,     dtype=torch.float32),
            face       = torch.as_tensor(cells_i.T,        dtype=torch.int64),
            y          = torch.as_tensor(world_pos_tp1,   dtype=torch.float32),
        )
        return graph
