"""
tests/test_phase2_zarr.py — Tests for storage/zarr_archive.py (Phase 2.6).
"""

import logging
from unittest.mock import patch

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")

from storage.zarr_archive import ZarrArchive  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

S, T, N, D = 3, 10, 5, 2  # small shapes


@pytest.fixture()
def arrays():
    rng = np.random.default_rng(0)
    positions = rng.random((S, T, N, D)).astype(np.float32)
    velocities = rng.random((S, T, N, D)).astype(np.float32)
    node_types = rng.integers(0, 3, size=(S, N)).astype(np.int32)
    pressures = rng.random((S, T, N, 1)).astype(np.float32)
    return positions, velocities, node_types, pressures


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_zarr_write_read_roundtrip(tmp_path, arrays):
    positions, velocities, node_types, _ = arrays
    arch = ZarrArchive(tmp_path / "zarr_root")
    arch.write_split("train", positions=positions, velocities=velocities, node_types=node_types)

    result = arch.read_split("train")
    assert len(result) == 3
    pos_r, vel_r, nt_r = result

    np.testing.assert_array_equal(pos_r, positions)
    np.testing.assert_array_equal(vel_r, velocities)
    np.testing.assert_array_equal(nt_r, node_types)


def test_zarr_sentinel(tmp_path, arrays):
    positions, velocities, node_types, _ = arrays
    arch = ZarrArchive(tmp_path / "zarr_root")
    sentinel = arch._sentinel_path("train")
    assert not sentinel.exists()

    arch.write_split("train", positions=positions, velocities=velocities, node_types=node_types)
    assert sentinel.exists()


def test_zarr_exists(tmp_path, arrays):
    positions, velocities, node_types, _ = arrays
    arch = ZarrArchive(tmp_path / "zarr_root")

    assert arch.exists("train") is False
    arch.write_split("train", positions=positions, velocities=velocities, node_types=node_types)
    assert arch.exists("train") is True


def test_zarr_list_splits(tmp_path, arrays):
    positions, velocities, node_types, _ = arrays
    arch = ZarrArchive(tmp_path / "zarr_root")

    assert arch.list_splits() == []

    arch.write_split("train", positions=positions, velocities=velocities, node_types=node_types)
    arch.write_split("valid", positions=positions, velocities=velocities, node_types=node_types)

    splits = sorted(arch.list_splits())
    assert splits == ["train", "valid"]


def test_zarr_pressures(tmp_path, arrays):
    positions, velocities, node_types, pressures = arrays
    arch = ZarrArchive(tmp_path / "zarr_root")
    arch.write_split(
        "train",
        positions=positions,
        velocities=velocities,
        node_types=node_types,
        pressures=pressures,
    )

    result = arch.read_split("train")
    assert len(result) == 4
    pos_r, vel_r, nt_r, pres_r = result

    np.testing.assert_array_equal(pos_r, positions)
    np.testing.assert_array_equal(pres_r, pressures)


def test_zarr_import_error_graceful(tmp_path, arrays, caplog):
    """write_split should log a warning and return silently if zarr is unavailable."""
    positions, velocities, node_types, _ = arrays
    arch = ZarrArchive(tmp_path / "zarr_root")

    with patch.dict("sys.modules", {"zarr": None}):
        with caplog.at_level(logging.WARNING, logger="storage.zarr_archive"):
            arch.write_split(
                "train",
                positions=positions,
                velocities=velocities,
                node_types=node_types,
            )

    assert not arch.exists("train"), "sentinel should NOT be written when zarr is unavailable"
    assert any("zarr" in msg.lower() for msg in caplog.messages), (
        "Expected a warning mentioning zarr"
    )
