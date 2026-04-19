"""Tests for Phase 3: composable ingest pipeline."""
import json
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from ingest.protocols import SolverAdapter
from ingest.stages import harvest, validate, normalise, write, index
from ingest.adapters.openfoam import OpenFOAMAdapter
from ingest.adapters.tfrecord import TFRecordAdapter
from ingest.pipeline import IngestPipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_data(S=4, T=10, N=20, D=2):
    return {
        "positions": np.random.randn(S, T + 1, N, D).astype(np.float32),
        "velocities": np.random.randn(S, T, N, D).astype(np.float32),
        "node_types": np.random.randint(0, 4, (S, N)).astype(np.float32),
        "metadata": {},
    }


def _make_adapter(splits=None, data=None):
    if splits is None:
        splits = ["train", "valid", "test"]
    if data is None:
        data = _make_data()
    adapter = MagicMock()
    adapter.name = "MockAdapter"
    adapter.list_splits.return_value = splits
    adapter.load_split.return_value = data
    adapter.source_path = Path(".")
    return adapter


# ---------------------------------------------------------------------------
# Protocol tests
# ---------------------------------------------------------------------------

def test_solver_adapter_protocol():
    """A class with the right methods satisfies isinstance(obj, SolverAdapter)."""
    class GoodAdapter:
        @property
        def name(self):
            return "Good"

        def list_splits(self):
            return ["train"]

        def load_split(self, split):
            return {}

        @property
        def source_path(self):
            return Path(".")

    assert isinstance(GoodAdapter(), SolverAdapter)


# ---------------------------------------------------------------------------
# Stage 1: harvest
# ---------------------------------------------------------------------------

def test_harvest_calls_adapter():
    """harvest() calls adapter.load_split with the given split."""
    data = _make_data()
    adapter = _make_adapter(data=data)
    result = harvest.harvest(adapter, "train")
    adapter.load_split.assert_called_once_with("train")
    assert result is data


# ---------------------------------------------------------------------------
# Stage 2: validate
# ---------------------------------------------------------------------------

def test_validate_passes_valid_data():
    data = _make_data()
    result = validate.validate(data, "train")
    # Same object returned unchanged
    assert result is data


def test_validate_missing_key_raises():
    data = _make_data()
    del data["velocities"]
    with pytest.raises(ValueError, match="Missing required keys"):
        validate.validate(data, "train")


def test_validate_shape_mismatch_raises():
    data = _make_data()
    # Give positions a different trajectory count
    data["positions"] = np.random.randn(7, 11, 20, 2).astype(np.float32)
    with pytest.raises(ValueError, match="Trajectory count mismatch"):
        validate.validate(data, "train")


# ---------------------------------------------------------------------------
# Stage 3: normalise
# ---------------------------------------------------------------------------

def test_normalise_returns_stats():
    data = _make_data()
    _, stats = normalise.normalise(data)
    for key in ("vel_mean", "vel_std", "pos_mean", "pos_std"):
        assert key in stats, f"Missing stat: {key}"


def test_normalise_with_pressures():
    data = _make_data()
    S, T, N = 4, 10, 20
    data["pressures"] = np.random.randn(S, T, N, 1).astype(np.float32)
    _, stats = normalise.normalise(data)
    assert "pressure_mean" in stats
    assert "pressure_std" in stats


# ---------------------------------------------------------------------------
# Stage 4: write
# ---------------------------------------------------------------------------

def test_write_npz_creates_file(tmp_path, monkeypatch):
    # Redirect manifest to tmp_path
    monkeypatch.setattr(write, "MANIFEST_PATH", tmp_path / "manifest.json")
    data = _make_data()
    _, stats = normalise.normalise(data)
    out = write.write_npz(data, stats, tmp_path, "train")
    assert out.exists()
    assert out.name == "train_ingest.npz"


def test_write_npz_manifest_content(tmp_path, monkeypatch):
    monkeypatch.setattr(write, "MANIFEST_PATH", tmp_path / "manifest.json")
    data = _make_data(S=6)
    _, stats = normalise.normalise(data)
    write.write_npz(data, stats, tmp_path, "valid")
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert "valid" in manifest["splits"]
    assert manifest["splits"]["valid"]["num_trajectories"] == 6


# ---------------------------------------------------------------------------
# IngestPipeline
# ---------------------------------------------------------------------------

def test_pipeline_run_success(tmp_path, monkeypatch):
    monkeypatch.setattr(write, "MANIFEST_PATH", tmp_path / "manifest.json")
    data = _make_data()
    adapter = _make_adapter(splits=["train", "valid"], data=data)
    pipeline = IngestPipeline(adapter, out_dir=tmp_path)
    results = pipeline.run(rebuild_index=False)
    assert results["train"]["status"] == "ok"
    assert results["valid"]["status"] == "ok"
    assert Path(results["train"]["npz"]).exists()


def test_pipeline_run_error_handling(tmp_path, monkeypatch):
    monkeypatch.setattr(write, "MANIFEST_PATH", tmp_path / "manifest.json")
    good_data = _make_data()

    def side_effect(split):
        if split == "test":
            raise RuntimeError("simulated failure")
        return good_data

    adapter = _make_adapter(splits=["train", "test"])
    adapter.load_split.side_effect = side_effect

    pipeline = IngestPipeline(adapter, out_dir=tmp_path)
    results = pipeline.run(rebuild_index=False)
    assert results["train"]["status"] == "ok"
    assert results["test"]["status"] == "error"
    assert "simulated failure" in results["test"]["error"]


# ---------------------------------------------------------------------------
# Adapter tests
# ---------------------------------------------------------------------------

def test_openfoam_adapter_raises():
    adapter = OpenFOAMAdapter()
    with pytest.raises(NotImplementedError):
        adapter.load_split("train")


def test_tfrecord_adapter_missing_dat_raises(tmp_path):
    """TFRecordAdapter raises RuntimeError when .dat files are absent."""
    adapter = TFRecordAdapter(data_dir=tmp_path)
    with pytest.raises(RuntimeError, match="parse_tfrecord.py"):
        adapter.load_split("train")
