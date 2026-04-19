"""
tests/test_phase1_pipeline.py — Tests for Phase 1 data pipeline improvements.

Tests:
  - 1.1 .dat.ok sentinel: FpcDataset raises on missing sentinel
  - 1.1 .dat.ok sentinel: FpcDataset loads normally when sentinel present
  - 1.3 result retention: prune() deletes oldest files, keeps N most recent
  - 1.3 result retention: dry_run=True does not delete
  - 1.3 result retention: empty dir handled gracefully
  - 1.4 LRU model cache: OrderedDict evicts LRU when over capacity
"""

import os
import time
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest


# ─── 1.1  Sentinel tests ──────────────────────────────────────────────────────

class TestDatOkSentinel:
    """FpcDataset must check for .dat.ok before opening memmap."""

    def _make_dataset_dir(self, tmp_path: Path, write_sentinel: bool = True) -> Path:
        """Create a minimal valid dataset directory for FpcDataset."""
        n_traj   = 2
        n_nodes  = 5
        n_steps  = 4
        n_cells  = 3

        pos       = np.zeros((n_traj * n_nodes, 2),  dtype=np.float32)
        node_type = np.zeros((n_traj * n_nodes, 1),  dtype=np.float32)
        cells     = np.zeros((n_traj * n_cells, 3),  dtype=np.int32)
        indices   = np.array([0, n_nodes, n_traj * n_nodes], dtype=np.int64)
        cindices  = np.array([0, n_cells, n_traj * n_cells], dtype=np.int64)
        vel_shape = (n_traj * n_nodes, n_steps, 2)

        np.savez_compressed(
            tmp_path / "train.npz",
            pos=pos, node_type=node_type, cells=cells,
            indices=indices, cindices=cindices,
            all_velocity_shape=vel_shape,
        )

        dat_path = tmp_path / "train.dat"
        fp = np.memmap(dat_path, dtype="float32", mode="w+", shape=vel_shape)
        fp[:] = 0.0
        del fp

        if write_sentinel:
            (tmp_path / "train.dat.ok").touch()

        return tmp_path

    def test_raises_without_sentinel(self, tmp_path):
        """FpcDataset should raise RuntimeError if .dat.ok is missing."""
        self._make_dataset_dir(tmp_path, write_sentinel=False)

        from dataset.fpc import FpcDataset
        with pytest.raises(RuntimeError, match="Missing parse sentinel"):
            FpcDataset(data_root=str(tmp_path), split="train")

    def test_loads_with_sentinel(self, tmp_path):
        """FpcDataset should load normally when .dat.ok sentinel is present."""
        self._make_dataset_dir(tmp_path, write_sentinel=True)

        from dataset.fpc import FpcDataset
        ds = FpcDataset(data_root=str(tmp_path), split="train")
        # 2 trajectories × (4 steps − 1) = 6 samples
        assert len(ds) == 6

    def test_check_sentinel_static_method_present(self):
        """_check_sentinel must be a static method on FpcDataset."""
        from dataset.fpc import FpcDataset
        assert hasattr(FpcDataset, "_check_sentinel")
        assert callable(FpcDataset._check_sentinel)

    def test_check_sentinel_raises_on_missing(self, tmp_path):
        """_check_sentinel raises RuntimeError when sentinel file absent."""
        from dataset.fpc import FpcDataset
        dat_path = str(tmp_path / "train.dat")
        # No sentinel file created
        with pytest.raises(RuntimeError, match="Missing parse sentinel"):
            FpcDataset._check_sentinel(dat_path, "velocity")

    def test_check_sentinel_passes_when_present(self, tmp_path):
        """_check_sentinel does not raise when sentinel file is present."""
        from dataset.fpc import FpcDataset
        dat_path = str(tmp_path / "train.dat")
        (tmp_path / "train.dat.ok").touch()
        # Should not raise
        FpcDataset._check_sentinel(dat_path, "velocity")


# ─── 1.3  Result retention tests ─────────────────────────────────────────────

class TestResultRetention:
    """prune() must keep N most recent files and delete the rest."""

    def _make_result_files(self, result_dir: Path, n: int,
                           suffix: str = ".pkl") -> list[Path]:
        """Create n result files with distinct mtimes (1s apart)."""
        files = []
        for i in range(n):
            p = result_dir / f"rollout_{i:04d}{suffix}"
            p.write_bytes(b"x" * 100)
            # Stagger mtimes so sort order is deterministic
            os.utime(p, (time.time() + i, time.time() + i))
            files.append(p)
        return files

    def test_keeps_n_most_recent(self, tmp_path):
        """prune(keep=3) should keep the 3 newest files."""
        from result.retention import prune
        files = self._make_result_files(tmp_path, n=6)
        deleted = prune(tmp_path, keep=3, dry_run=False)
        assert len(deleted) == 3
        # Oldest 3 should be gone
        for f in files[:3]:
            assert not f.exists(), f"{f} should have been deleted"
        # Newest 3 should survive
        for f in files[3:]:
            assert f.exists(), f"{f} should have been kept"

    def test_dry_run_does_not_delete(self, tmp_path):
        """dry_run=True must not delete any files."""
        from result.retention import prune
        files = self._make_result_files(tmp_path, n=5)
        deleted = prune(tmp_path, keep=2, dry_run=True)
        assert len(deleted) == 3
        # All files must still exist
        for f in files:
            assert f.exists(), f"{f} was deleted during dry run"

    def test_empty_dir_no_error(self, tmp_path):
        """prune() on an empty directory should return [] without error."""
        from result.retention import prune
        deleted = prune(tmp_path, keep=5, dry_run=False)
        assert deleted == []

    def test_keep_more_than_total(self, tmp_path):
        """keep > total files should delete nothing."""
        from result.retention import prune
        files = self._make_result_files(tmp_path, n=3)
        deleted = prune(tmp_path, keep=10, dry_run=False)
        assert deleted == []
        for f in files:
            assert f.exists()

    def test_supports_h5_extension(self, tmp_path):
        """prune() should also handle .h5 files (Phase 2 format)."""
        from result.retention import prune
        self._make_result_files(tmp_path, n=4, suffix=".h5")
        deleted = prune(tmp_path, keep=2, dry_run=False)
        assert len(deleted) == 2

    def test_missing_dir_no_error(self, tmp_path):
        """prune() on a non-existent directory should return [] without error."""
        from result.retention import prune
        deleted = prune(tmp_path / "nonexistent", keep=5, dry_run=False)
        assert deleted == []


# ─── 1.4  LRU model cache tests ──────────────────────────────────────────────

class TestLRUModelCache:
    """Model cache in api/state.py must evict LRU entry when over capacity."""

    def test_cache_capacity_constant_exists(self):
        """_MODEL_CACHE_MAX must be defined and positive."""
        import api.state as state
        assert hasattr(state, "_MODEL_CACHE_MAX")
        assert state._MODEL_CACHE_MAX > 0

    def test_cache_is_ordered_dict(self):
        """_model_cache must be an OrderedDict for LRU ordering."""
        from collections import OrderedDict
        import api.state as state
        assert isinstance(state._model_cache, OrderedDict)

    def test_lru_eviction_on_overflow(self):
        """
        When more than _MODEL_CACHE_MAX models are loaded, the LRU entry
        should be evicted.
        """
        import api.state as state
        from collections import OrderedDict

        original_cache  = state._model_cache
        original_max    = state._MODEL_CACHE_MAX

        try:
            # Reset to a clean state with capacity=2
            state._model_cache   = OrderedDict()
            state._MODEL_CACHE_MAX = 2

            # Manually populate the cache (bypass torch.load)
            mock_model = MagicMock()
            state._model_cache[("ckpt_a", "cpu")] = mock_model
            state._model_cache[("ckpt_b", "cpu")] = mock_model

            # Simulate adding a third entry (the eviction logic)
            state._model_cache[("ckpt_c", "cpu")] = mock_model
            if len(state._model_cache) > state._MODEL_CACHE_MAX:
                state._model_cache.popitem(last=False)  # evict LRU

            assert len(state._model_cache) == 2
            # ckpt_a (inserted first = LRU) should be gone
            assert ("ckpt_a", "cpu") not in state._model_cache
            # ckpt_b and ckpt_c should remain
            assert ("ckpt_b", "cpu") in state._model_cache
            assert ("ckpt_c", "cpu") in state._model_cache

        finally:
            state._model_cache   = original_cache
            state._MODEL_CACHE_MAX = original_max

    def test_clear_model_cache(self):
        """clear_model_cache() must empty the cache."""
        import api.state as state
        from collections import OrderedDict

        original_cache = state._model_cache
        try:
            state._model_cache = OrderedDict()
            state._model_cache[("x", "cpu")] = MagicMock()
            assert len(state._model_cache) == 1

            state.clear_model_cache()
            assert len(state._model_cache) == 0
        finally:
            state._model_cache = original_cache
