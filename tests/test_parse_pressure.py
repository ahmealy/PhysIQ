"""
Test that pressure .dat files exist and have the right shape.
Skips if re-parsing hasn't been run yet.
"""
import os
import numpy as np
import pytest


@pytest.mark.skipif(
    not os.path.exists("data/train_pressure.dat"),
    reason="data/train_pressure.dat not present — re-run parse_tfrecord.py first",
)
def test_pressure_dat_loadable():
    """Pressure .dat file loads as memmap with correct shape."""
    meta = np.load("data/train.npz", allow_pickle=True)
    vel_shape = tuple(meta["all_velocity_shape"])  # (total_nodes, T, 2)
    pressure_shape = (vel_shape[0], vel_shape[1], 1)
    fp = np.memmap("data/train_pressure.dat", dtype="float32", mode="r",
                   shape=pressure_shape)
    assert fp.shape == pressure_shape
    assert fp.dtype == np.float32


@pytest.mark.skipif(
    not os.path.exists("data/train_pressure.dat"),
    reason="data/train_pressure.dat not present",
)
def test_pressure_values_not_all_zero():
    """Pressure values are non-trivial (not zeroed out)."""
    meta = np.load("data/train.npz", allow_pickle=True)
    vel_shape = tuple(meta["all_velocity_shape"])
    pressure_shape = (vel_shape[0], vel_shape[1], 1)
    fp = np.memmap("data/train_pressure.dat", dtype="float32", mode="r",
                   shape=pressure_shape)
    assert np.any(fp != 0.0), "All pressure values are zero — extraction may have failed"
