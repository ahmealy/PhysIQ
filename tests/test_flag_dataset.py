import os
import numpy as np
import torch
import pytest
from torch_geometric.data import Data


def _make_fake_data(data_dir: str, n_traj: int = 2, T: int = 10, N: int = 15, F: int = 20):
    """Write synthetic flag data files matching the parse_flag_tfrecord.py output format."""
    os.makedirs(data_dir, exist_ok=True)
    world_pos = np.array([
        np.random.randn(T, N, 3).astype(np.float32) for _ in range(n_traj)
    ], dtype=object)
    np.savez_compressed(os.path.join(data_dir, "train_pos.npz"), world_pos=world_pos)

    mesh_pos  = np.array([np.random.randn(N, 2).astype(np.float32)  for _ in range(n_traj)], dtype=object)
    node_type = np.array([np.zeros((N, 1), dtype=np.int32)           for _ in range(n_traj)], dtype=object)
    cells     = np.array([np.random.randint(0, N, (F, 3)).astype(np.int64) for _ in range(n_traj)], dtype=object)
    np.savez_compressed(os.path.join(data_dir, "train_mesh.npz"),
                        mesh_pos=mesh_pos, node_type=node_type, cells=cells)
    return n_traj, T, N, F


def test_flag_dataset_length(tmp_path):
    data_dir = str(tmp_path)
    n_traj, T, N, F = _make_fake_data(data_dir)
    from dataset.flag_dataset import FlagDataset
    ds = FlagDataset(data_dir, split="train")
    # Each trajectory has T-1 timestep pairs
    assert len(ds) == n_traj * (T - 1)


def test_flag_dataset_item_shapes(tmp_path):
    data_dir = str(tmp_path)
    n_traj, T, N, F = _make_fake_data(data_dir)
    from dataset.flag_dataset import FlagDataset
    ds = FlagDataset(data_dir, split="train")
    graph = ds[0]
    assert isinstance(graph, Data)
    # graph.x: [N, 4] — concat(world_pos_t[3], node_type[1])
    assert graph.x.shape == (N, 4)
    # graph.prev_x: [N, 3] — world_pos_{t-1}
    assert graph.prev_x.shape == (N, 3)
    # graph.pos: [N, 2] — mesh_pos (2D rest configuration)
    assert graph.pos.shape == (N, 2)
    # graph.world_pos: [N, 3] — world_pos_t
    assert graph.world_pos.shape == (N, 3)
    # graph.face: [3, F] — triangle connectivity
    assert graph.face.shape == (3, F)
    # graph.y: [N, 3] — world_pos_{t+1} (target)
    assert graph.y.shape == (N, 3)


def test_flag_dataset_first_step_prev_equals_cur(tmp_path):
    """At t=0, prev_world_pos should equal world_pos_t (no history available)."""
    data_dir = str(tmp_path)
    _make_fake_data(data_dir)
    from dataset.flag_dataset import FlagDataset
    ds = FlagDataset(data_dir, split="train")
    graph = ds[0]
    # t=0: prev_x should equal world_pos
    assert torch.allclose(graph.prev_x, graph.world_pos)


def test_flag_dataset_second_step_has_history(tmp_path):
    """At t=1, prev_world_pos should equal world_pos at t=0."""
    data_dir = str(tmp_path)
    n_traj, T, N, F = _make_fake_data(data_dir, T=5)
    from dataset.flag_dataset import FlagDataset
    ds = FlagDataset(data_dir, split="train")
    graph_t0 = ds[0]   # t=0
    graph_t1 = ds[1]   # t=1
    # t=1: prev_x == world_pos at t=0
    assert torch.allclose(graph_t1.prev_x, graph_t0.world_pos)
