"""
storage/hdf5_repository.py — HDF5-backed result repository.

File layout:
    /predictions   float32 [T, N, D]
    /targets       float32 [T, N, D]
    /coords        float32 [N, 2|3]
    /meta          (attributes on root group, JSON-encoded dict)
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

_TS_RE = re.compile(r'(\d{8}_\d{6})')


def _sort_key(name: str, result_dir: Path) -> str:
    """Newest-first sort key — parse YYYYMMDD_HHMMSS from name, fall back to mtime."""
    m = _TS_RE.search(name)
    if m:
        return m.group(1)
    try:
        mtime = (result_dir / (name + ".h5")).stat().st_mtime
        return datetime.fromtimestamp(mtime).strftime("%Y%m%d_%H%M%S")
    except OSError:
        return "00000000_000000"


class HDF5ResultRepository:
    """ResultRepository backed by HDF5 files."""

    SUFFIX = ".h5"

    def __init__(self, result_dir: Path | str):
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(parents=True, exist_ok=True)

    # ── write ────────────────────────────────────────────────────────────────

    def save(
        self,
        name: str,
        predictions: np.ndarray,
        targets: np.ndarray,
        coords: np.ndarray,
        metadata: dict,
    ) -> Path:
        path = self.result_dir / f"{name}{self.SUFFIX}"
        predictions = np.asarray(predictions, dtype=np.float32)
        targets = np.asarray(targets, dtype=np.float32)
        coords = np.asarray(coords, dtype=np.float32)
        T, N, D = predictions.shape
        with h5py.File(path, "w") as f:
            f.create_dataset(
                "predictions", data=predictions,
                compression="gzip", compression_opts=4,
                chunks=(1, N, D),
            )
            f.create_dataset(
                "targets", data=targets,
                compression="gzip", compression_opts=4,
                chunks=(1, N, D),
            )
            f.create_dataset(
                "coords", data=coords,
                compression="gzip", compression_opts=4,
            )
            for k, v in metadata.items():
                f.attrs[k] = json.dumps(v)
        return path

    # ── read ─────────────────────────────────────────────────────────────────

    def load(self, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        path = self._resolve(name)
        with h5py.File(path, "r") as f:
            predictions = f["predictions"][:]
            targets = f["targets"][:]
            coords = f["coords"][:]
            meta = {k: json.loads(v) for k, v in f.attrs.items()}
        return predictions, targets, coords, meta

    def load_timestep(self, name: str, t: int) -> np.ndarray:
        """True partial read — only loads one timestep from disk."""
        path = self._resolve(name)
        with h5py.File(path, "r") as f:
            frame = f["predictions"][t:t+1, ...]
        return frame[0]

    # ── query ────────────────────────────────────────────────────────────────

    def list(self) -> list[str]:
        names = [p.stem for p in self.result_dir.glob(f"*{self.SUFFIX}")]
        names.sort(key=lambda n: _sort_key(n, self.result_dir), reverse=True)
        return names

    def exists(self, name: str) -> bool:
        stem = name if not name.endswith(self.SUFFIX) else name[: -len(self.SUFFIX)]
        return (self.result_dir / f"{stem}{self.SUFFIX}").exists()

    def delete(self, name: str) -> None:
        path = self._resolve(name)
        path.unlink()

    def get_path(self, name: str) -> Path:
        return self._resolve(name)

    # ── internal ─────────────────────────────────────────────────────────────

    def _resolve(self, name: str) -> Path:
        stem = name if not name.endswith(self.SUFFIX) else name[: -len(self.SUFFIX)]
        path = self.result_dir / f"{stem}{self.SUFFIX}"
        if not path.exists():
            raise FileNotFoundError(f"Result not found: {path}")
        return path
