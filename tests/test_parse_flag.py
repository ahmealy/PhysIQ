"""
Test that parse_flag_tfrecord output files have the expected structure.
This test does NOT parse TFRecords (no TF needed) — it validates pre-existing output files.
Skip if data_flag/ doesn't exist.
"""
import os
import numpy as np
import pytest

DATA_DIR = "data_flag"


@pytest.mark.skipif(
    not os.path.exists(os.path.join(DATA_DIR, "train_index.npz")),
    reason="data_flag/train_index.npz not present — run parse_flag_tfrecord.py first",
)
def test_train_index_structure():
    idx = np.load(os.path.join(DATA_DIR, "train_index.npz"))
    assert "n_traj" in idx
    assert "steps_per_traj" in idx
    n_traj = int(idx["n_traj"])
    assert n_traj > 0
    steps = idx["steps_per_traj"]
    assert len(steps) == n_traj
    assert all(s > 1 for s in steps), "All trajectories must have T > 1"


@pytest.mark.skipif(
    not os.path.exists(os.path.join(DATA_DIR, "train_index.npz")),
    reason="data_flag/train_index.npz not present — run parse_flag_tfrecord.py first",
)
def test_first_train_trajectory_structure():
    traj_path = os.path.join(DATA_DIR, "train", "traj_00000.npz")
    assert os.path.exists(traj_path), f"First trajectory file missing: {traj_path}"

    traj = np.load(traj_path)
    assert "world_pos" in traj
    assert "mesh_pos"  in traj
    assert "node_type" in traj
    assert "cells"     in traj

    world_pos = traj["world_pos"]    # [T, N, 3]
    mesh_pos  = traj["mesh_pos"]     # [N, 2]
    node_type = traj["node_type"]    # [N, 1]
    cells     = traj["cells"]        # [F, 3]

    assert world_pos.ndim == 3
    assert world_pos.shape[2] == 3        # 3D positions
    assert world_pos.dtype == np.float32

    N = world_pos.shape[1]
    assert mesh_pos.shape == (N, 2)       # 2D rest coords
    assert node_type.shape[1] == 1        # scalar type per node
    assert cells.shape[1] == 3            # triangles
