"""Tests for FpcDataset velocity and pressure modes."""
import os
import numpy as np
import torch
import pytest


def _make_fake_cfd_data(data_dir: str, n_traj: int = 2, T: int = 5, N: int = 10, F: int = 8):
    """Write synthetic CFD data files matching parse_tfrecord.py output."""
    os.makedirs(data_dir, exist_ok=True)
    shape0 = n_traj * N

    # Velocity memmap
    vel = np.random.randn(shape0, T, 2).astype(np.float32)
    fp = np.memmap(os.path.join(data_dir, "train.dat"), dtype="float32", mode="w+",
                   shape=(shape0, T, 2))
    fp[:] = vel
    fp.flush()
    del fp

    # Pressure memmap
    pres = np.random.randn(shape0, T, 1).astype(np.float32)
    fp2 = np.memmap(os.path.join(data_dir, "train_pressure.dat"), dtype="float32", mode="w+",
                    shape=(shape0, T, 1))
    fp2[:] = pres
    fp2.flush()
    del fp2

    # indices: each trajectory has N nodes
    indices = np.array([0, N, N * 2])
    cindices = np.array([0, F, F * 2])

    all_pos       = np.random.randn(shape0, 2).astype(np.float32)
    all_node_type = np.zeros((shape0, 1), dtype=np.float32)
    all_cells     = np.random.randint(0, N, (F * n_traj, 3)).astype(np.int64)

    np.savez_compressed(
        os.path.join(data_dir, "train.npz"),
        pos=all_pos,
        node_type=all_node_type,
        cells=all_cells,
        indices=indices,
        cindices=cindices,
        all_velocity_shape=(shape0, T, 2),
        all_pressure_shape=(shape0, T, 1),
    )
    return shape0, T, N, F


def test_velocity_mode_default(tmp_path):
    """Default (velocity) mode: graph.x = [N, 3], graph.y = [N, 2]."""
    data_dir = str(tmp_path)
    _make_fake_cfd_data(data_dir)
    from dataset.fpc import FpcDataset
    ds = FpcDataset(data_root=data_dir, split="train")
    graph = ds[0]
    assert graph.x.shape[1] == 3
    assert graph.y.shape[1] == 2


def test_pressure_mode_shapes(tmp_path):
    """Pressure mode: graph.x = [N, 2], graph.y = [N, 1]."""
    data_dir = str(tmp_path)
    _make_fake_cfd_data(data_dir)
    from dataset.fpc import FpcDataset
    ds = FpcDataset(data_root=data_dir, split="train", target_field="pressure")
    graph = ds[0]
    assert graph.x.shape[1] == 2, f"x should be [N, 2] for pressure, got {graph.x.shape}"
    assert graph.y.shape[1] == 1, f"y should be [N, 1] for pressure, got {graph.y.shape}"


def test_pressure_mode_values_differ_from_velocity(tmp_path):
    """Pressure graph.x contains different values than velocity graph.x."""
    data_dir = str(tmp_path)
    _make_fake_cfd_data(data_dir)
    from dataset.fpc import FpcDataset
    ds_vel  = FpcDataset(data_root=data_dir, split="train")
    ds_pres = FpcDataset(data_root=data_dir, split="train", target_field="pressure")
    g_vel  = ds_vel[0]
    g_pres = ds_pres[0]
    assert torch.allclose(g_vel.x[:, 0], g_pres.x[:, 0]), "node_type column should match"
    assert g_vel.x.shape[1] != g_pres.x.shape[1]


def test_pressure_missing_dat_raises(tmp_path):
    """If pressure .dat file missing and target_field='pressure', raise FileNotFoundError."""
    data_dir = str(tmp_path)
    _make_fake_cfd_data(data_dir)
    os.remove(os.path.join(data_dir, "train_pressure.dat"))
    from dataset.fpc import FpcDataset
    with pytest.raises(FileNotFoundError, match="pressure"):
        FpcDataset(data_root=data_dir, split="train", target_field="pressure")


def test_velocity_mode_backward_compat(tmp_path):
    """Existing velocity-only data (no pressure .dat) still works in velocity mode."""
    data_dir = str(tmp_path)
    _make_fake_cfd_data(data_dir)
    os.remove(os.path.join(data_dir, "train_pressure.dat"))
    from dataset.fpc import FpcDataset
    ds = FpcDataset(data_root=data_dir, split="train")
    graph = ds[0]
    assert graph.x.shape[1] == 3
