"""
FlagDataset — loads parsed flag_simple cloth simulation data.

Data files produced by data/parse_flag_tfrecord.py:
    {data_root}/{split}_pos.npz   — world_pos per trajectory (object array of [T, N, 3])
    {data_root}/{split}_mesh.npz  — mesh_pos, node_type, cells per trajectory

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
        pos_path  = os.path.join(data_root, f"{split}_pos.npz")
        mesh_path = os.path.join(data_root, f"{split}_mesh.npz")

        if not os.path.exists(pos_path):
            raise FileNotFoundError(
                f"Flag position data not found: {pos_path}\n"
                "Re-run: python data/parse_flag_tfrecord.py"
            )
        if not os.path.exists(mesh_path):
            raise FileNotFoundError(
                f"Flag mesh data not found: {mesh_path}\n"
                "Re-run: python data/parse_flag_tfrecord.py"
            )

        # Keep NpzFile handles open for lazy per-item access
        self._pos_data  = np.load(pos_path,  allow_pickle=True)
        self._mesh_data = np.load(mesh_path, allow_pickle=True)

        # Only materialize the object array headers (not the inner float arrays)
        # to compute trajectory lengths for __len__ and index mapping
        world_pos_arr = self._pos_data["world_pos"]     # object array: [n_traj] of [T, N, 3]
        self.mesh_pos_list   = self._mesh_data["mesh_pos"]    # [n_traj] of [N, 2]
        self.node_type_list  = self._mesh_data["node_type"]   # [n_traj] of [N, 1]
        self.cells_list      = self._mesh_data["cells"]       # [n_traj] of [F, 3]

        self.n_traj = len(world_pos_arr)
        # Each trajectory has T-1 timestep pairs (t, t+1)
        self.steps_per_traj = [arr.shape[0] - 1 for arr in world_pos_arr]
        self.total_samples = sum(self.steps_per_traj)

        bad_trajs = [i for i, s in enumerate(self.steps_per_traj) if s < 1]
        if bad_trajs:
            raise ValueError(
                f"Trajectories {bad_trajs} have fewer than 2 timesteps (T < 2). "
                "Data may be corrupted. Re-run parse_flag_tfrecord.py."
            )

        # Cumulative step counts for index mapping
        self._cum_steps = np.cumsum([0] + self.steps_per_traj)

        # Store the object array for lazy per-trajectory access in __getitem__
        self._world_pos_arr = world_pos_arr

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, index: int) -> Data:
        traj_idx = int(np.searchsorted(self._cum_steps[1:], index, side="right"))
        t = index - self._cum_steps[traj_idx]

        world_pos  = self._world_pos_arr[traj_idx]      # [T, N, 3] — accessed lazily
        mesh_pos   = self.mesh_pos_list[traj_idx]        # [N, 2]
        node_type  = self.node_type_list[traj_idx]       # [N, 1]
        cells      = self.cells_list[traj_idx]           # [F, 3]

        # Arrays extracted from a numpy object array retain dtype=object; cast explicitly.
        world_pos_t    = np.asarray(world_pos[t],     dtype=np.float32)   # [N, 3]
        world_pos_tp1  = np.asarray(world_pos[t + 1], dtype=np.float32)   # [N, 3]
        # At t=0 there is no previous frame — use current as previous (zero velocity)
        world_pos_prev = np.asarray(
            world_pos[t - 1] if t > 0 else world_pos[t], dtype=np.float32
        )  # [N, 3]
        mesh_pos_f  = np.asarray(mesh_pos,  dtype=np.float32)             # [N, 2]
        node_type_f = np.asarray(node_type, dtype=np.float32)             # [N, 1]
        cells_i     = np.asarray(cells,     dtype=np.int64)               # [F, 3]

        # Node features: concat(world_pos_t, node_type) → [N, 4]
        x = np.concatenate([world_pos_t, node_type_f], axis=-1)

        graph = Data(
            x          = torch.as_tensor(x,                dtype=torch.float32),
            prev_x     = torch.as_tensor(world_pos_prev,   dtype=torch.float32),
            pos        = torch.as_tensor(mesh_pos_f,        dtype=torch.float32),
            world_pos  = torch.as_tensor(world_pos_t,      dtype=torch.float32),
            face       = torch.as_tensor(cells_i.T,         dtype=torch.int64),
            y          = torch.as_tensor(world_pos_tp1,    dtype=torch.float32),
        )
        return graph
