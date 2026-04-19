"""
tests/test_phase2_storage.py — Comprehensive tests for Phase 2 storage layer.
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from storage.pkl_repository import PklResultRepository
from storage.factory import StorageFactory, get_repository
from storage.protocols import ResultRepository

h5py = pytest.importorskip("h5py", reason="h5py not installed")
from storage.hdf5_repository import HDF5ResultRepository


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_arrays(T=5, N=20, D=2):
    rng = np.random.default_rng(42)
    predictions = rng.standard_normal((T, N, D)).astype(np.float32)
    targets     = rng.standard_normal((T, N, D)).astype(np.float32)
    coords      = rng.standard_normal((N, 2)).astype(np.float32)
    meta        = {"domain": "flag", "score": 0.91}
    return predictions, targets, coords, meta


# ── PklResultRepository ───────────────────────────────────────────────────────

def test_pkl_save_load_roundtrip(tmp_path):
    repo = PklResultRepository(tmp_path)
    preds, targets, coords, meta = _make_arrays()
    repo.save("test_run", preds, targets, coords, meta)
    p2, t2, c2, m2 = repo.load("test_run")
    np.testing.assert_array_equal(preds, p2)
    np.testing.assert_array_equal(targets, t2)
    np.testing.assert_array_equal(coords, c2)
    assert m2 == meta


def test_pkl_load_legacy_2tuple(tmp_path):
    repo = PklResultRepository(tmp_path)
    preds, targets, coords, _ = _make_arrays()
    path = tmp_path / "legacy.pkl"
    with open(path, "wb") as f:
        pickle.dump([[preds, targets], coords], f)
    p2, t2, c2, meta = repo.load("legacy")
    np.testing.assert_array_equal(preds, p2)
    assert meta == {}


def test_pkl_list_sorted_newest_first(tmp_path):
    repo = PklResultRepository(tmp_path)
    preds, targets, coords, meta = _make_arrays()
    for name in ["run_20240101_000000", "run_20240303_000000", "run_20240202_000000"]:
        repo.save(name, preds, targets, coords, meta)
    names = repo.list()
    assert names == ["run_20240303_000000", "run_20240202_000000", "run_20240101_000000"]


def test_pkl_exists(tmp_path):
    repo = PklResultRepository(tmp_path)
    preds, targets, coords, meta = _make_arrays()
    assert not repo.exists("x")
    repo.save("x", preds, targets, coords, meta)
    assert repo.exists("x")


def test_pkl_delete(tmp_path):
    repo = PklResultRepository(tmp_path)
    preds, targets, coords, meta = _make_arrays()
    repo.save("x", preds, targets, coords, meta)
    repo.delete("x")
    assert not repo.exists("x")


def test_pkl_load_timestep(tmp_path):
    repo = PklResultRepository(tmp_path)
    preds, targets, coords, meta = _make_arrays(T=5)
    repo.save("x", preds, targets, coords, meta)
    frame = repo.load_timestep("x", t=2)
    np.testing.assert_array_equal(frame, preds[2])


def test_pkl_get_path(tmp_path):
    repo = PklResultRepository(tmp_path)
    preds, targets, coords, meta = _make_arrays()
    repo.save("x", preds, targets, coords, meta)
    assert repo.get_path("x") == tmp_path / "x.pkl"


def test_pkl_file_not_found(tmp_path):
    repo = PklResultRepository(tmp_path)
    with pytest.raises(FileNotFoundError):
        repo.load("nonexistent")


# ── HDF5ResultRepository ──────────────────────────────────────────────────────

def test_hdf5_save_load_roundtrip(tmp_path):
    repo = HDF5ResultRepository(tmp_path)
    preds, targets, coords, meta = _make_arrays()
    repo.save("test_run", preds, targets, coords, meta)
    p2, t2, c2, m2 = repo.load("test_run")
    np.testing.assert_array_almost_equal(preds, p2)
    np.testing.assert_array_almost_equal(targets, t2)
    np.testing.assert_array_almost_equal(coords, c2)
    assert m2 == meta


def test_hdf5_load_timestep_partial(tmp_path):
    repo = HDF5ResultRepository(tmp_path)
    preds, targets, coords, meta = _make_arrays(T=5)
    repo.save("x", preds, targets, coords, meta)
    frame = repo.load_timestep("x", t=2)
    assert frame.shape == preds[2].shape
    np.testing.assert_array_almost_equal(frame, preds[2])


def test_hdf5_list_sorted(tmp_path):
    repo = HDF5ResultRepository(tmp_path)
    preds, targets, coords, meta = _make_arrays()
    for name in ["run_20240101_000000", "run_20240303_000000", "run_20240202_000000"]:
        repo.save(name, preds, targets, coords, meta)
    names = repo.list()
    assert names == ["run_20240303_000000", "run_20240202_000000", "run_20240101_000000"]


def test_hdf5_exists(tmp_path):
    repo = HDF5ResultRepository(tmp_path)
    preds, targets, coords, meta = _make_arrays()
    assert not repo.exists("x")
    repo.save("x", preds, targets, coords, meta)
    assert repo.exists("x")


def test_hdf5_delete(tmp_path):
    repo = HDF5ResultRepository(tmp_path)
    preds, targets, coords, meta = _make_arrays()
    repo.save("x", preds, targets, coords, meta)
    repo.delete("x")
    assert not repo.exists("x")


def test_hdf5_metadata_roundtrip(tmp_path):
    repo = HDF5ResultRepository(tmp_path)
    preds, targets, coords, _ = _make_arrays()
    meta = {"domain": "flag", "nested": {"a": 1, "b": [1, 2, 3]}, "score": 0.99}
    repo.save("x", preds, targets, coords, meta)
    _, _, _, m2 = repo.load("x")
    assert m2 == meta


def test_hdf5_file_not_found(tmp_path):
    repo = HDF5ResultRepository(tmp_path)
    with pytest.raises(FileNotFoundError):
        repo.load("nonexistent")


# ── StorageFactory ────────────────────────────────────────────────────────────

def test_factory_default_is_pkl(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    repo = StorageFactory.create(tmp_path)
    assert isinstance(repo, PklResultRepository)


def test_factory_hdf5_backend(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "storage_config.json").write_text(
        json.dumps({"result_backend": "hdf5"})
    )
    repo = StorageFactory.create(tmp_path)
    assert isinstance(repo, HDF5ResultRepository)


def test_get_repository_convenience(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    repo = get_repository(tmp_path)
    assert repo is not None


# ── Protocol conformance ──────────────────────────────────────────────────────

def test_pkl_implements_protocol(tmp_path):
    repo = PklResultRepository(tmp_path)
    assert isinstance(repo, ResultRepository)


def test_hdf5_implements_protocol(tmp_path):
    repo = HDF5ResultRepository(tmp_path)
    assert isinstance(repo, ResultRepository)
