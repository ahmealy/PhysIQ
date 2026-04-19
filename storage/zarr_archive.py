"""
storage/zarr_archive.py — Write/read Zarr archives of dataset splits.

The Zarr archive mirrors the memmap .dat layout:
  Each split (train/valid/test) → one Zarr group with datasets:
    positions  [S, T+1, N, 3]   (or 2D for cylinder_flow)
    velocities [S, T, N, D]
    node_types [S, N]
    pressures  [S, T, N, 1]     (only for pressure splits)
  Metadata stored as group attributes.

Usage:
    arch = ZarrArchive("data/zarr/cylinder_flow_fpc")
    arch.write_split("train", positions=..., velocities=..., node_types=...)
    pos, vel, nt = arch.read_split("train")
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class ZarrArchive:
    """Zarr-backed archive for dataset splits."""

    def __init__(self, zarr_root):
        self.zarr_root = Path(zarr_root)

    def _store_path(self, split_name: str) -> Path:
        return self.zarr_root / f"{split_name}.zarr"

    def _sentinel_path(self, split_name: str) -> Path:
        return self.zarr_root / f"{split_name}.zarr.ok"

    def write_split(
        self,
        split_name: str,
        *,
        positions: np.ndarray,
        velocities: np.ndarray,
        node_types: np.ndarray,
        pressures: Optional[np.ndarray] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """Write arrays for one split to Zarr store."""
        try:
            import zarr
        except ImportError:
            logger.warning(
                "zarr is not installed — skipping Zarr archive write for split '%s'.",
                split_name,
            )
            return

        self.zarr_root.mkdir(parents=True, exist_ok=True)
        store_path = self._store_path(split_name)
        try:
            from zarr.codecs import BloscCodec
            compressor = BloscCodec(cname="lz4", clevel=5)
        except ImportError:
            try:
                from numcodecs import Blosc
                compressor = Blosc(cname="lz4", clevel=5)
            except ImportError:
                compressor = None

        store = zarr.open_group(str(store_path), mode="w")

        def _write(name, arr):
            kwargs = dict(name=name, data=arr, overwrite=True)
            if compressor is not None:
                kwargs["compressors"] = compressor
            store.create_array(**kwargs)

        _write("positions", positions)
        _write("velocities", velocities)
        _write("node_types", node_types)
        if pressures is not None:
            _write("pressures", pressures)

        if metadata:
            store.attrs.update(metadata)

        # Write sentinel
        self._sentinel_path(split_name).touch()
        logger.info("Zarr split '%s' written to %s", split_name, store_path)

    def read_split(self, split_name: str):
        """Read arrays for one split from Zarr store.

        Returns (positions, velocities, node_types) or
                (positions, velocities, node_types, pressures) if pressures present.
        """
        import zarr

        store_path = self._store_path(split_name)
        store = zarr.open_group(str(store_path), mode="r")

        positions = store["positions"][:]
        velocities = store["velocities"][:]
        node_types = store["node_types"][:]

        if "pressures" in store:
            pressures = store["pressures"][:]
            return positions, velocities, node_types, pressures

        return positions, velocities, node_types

    def exists(self, split_name: str) -> bool:
        """Return True if the split has been successfully written (sentinel present)."""
        return self._sentinel_path(split_name).exists()

    def list_splits(self) -> list:
        """Return list of split names that have been successfully written."""
        if not self.zarr_root.exists():
            return []
        return [
            p.name.replace(".zarr.ok", "")
            for p in self.zarr_root.glob("*.zarr.ok")
        ]
