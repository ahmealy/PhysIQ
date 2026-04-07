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
    not os.path.exists(os.path.join(DATA_DIR, "train_pos.npz")),
    reason="data_flag/train_pos.npz not present — run parse_flag_tfrecord.py first",
)
def test_train_pos_structure():
    data = np.load(os.path.join(DATA_DIR, "train_pos.npz"), allow_pickle=True)
    world_pos = data["world_pos"]
    # Object array of trajectories
    assert world_pos.dtype == object
    assert len(world_pos) > 0
    # First trajectory: [T, N, 3]
    first = world_pos[0]
    assert first.ndim == 3
    assert first.shape[2] == 3   # 3D positions


@pytest.mark.skipif(
    not os.path.exists(os.path.join(DATA_DIR, "train_mesh.npz")),
    reason="data_flag/train_mesh.npz not present — run parse_flag_tfrecord.py first",
)
def test_train_mesh_structure():
    data = np.load(os.path.join(DATA_DIR, "train_mesh.npz"), allow_pickle=True)
    mesh_pos  = data["mesh_pos"]
    node_type = data["node_type"]
    cells     = data["cells"]

    assert mesh_pos.dtype == object
    first_mesh = mesh_pos[0]
    assert first_mesh.ndim == 2
    assert first_mesh.shape[1] == 2   # 2D rest coords

    first_cells = cells[0]
    assert first_cells.ndim == 2
    assert first_cells.shape[1] == 3  # triangles
