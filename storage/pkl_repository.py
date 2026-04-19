"""
storage/pkl_repository.py — Pickle-based result repository.

This is a thin wrapper around the existing pickle format so that all
API routes can be migrated to the ResultRepository interface while
keeping identical on-disk behaviour.

File format (unchanged from the original codebase):
    pickle.dump( [[predictions, targets], coords, metadata], f )
    i.e. a 3-tuple: (result_list, coords, meta)
    where result_list[0] = predictions, result_list[1] = targets
"""

from __future__ import annotations

import os
import pickle
import re
from datetime import datetime
from pathlib import Path

import numpy as np

_TS_RE = re.compile(r'(\d{8}_\d{6})')


def _sort_key(name: str, result_dir: Path) -> str:
    """Newest-first sort key — parse YYYYMMDD_HHMMSS from name, fall back to mtime."""
    m = _TS_RE.search(name)
    if m:
        return m.group(1)
    try:
        mtime = (result_dir / (name + ".pkl")).stat().st_mtime
        return datetime.fromtimestamp(mtime).strftime("%Y%m%d_%H%M%S")
    except OSError:
        return "00000000_000000"


class PklResultRepository:
    """
    ResultRepository backed by pickle files in a local directory.

    Supports the legacy 2-tuple format (result, crds) for read-back
    so existing files are never broken.
    """

    SUFFIX = ".pkl"

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
        with open(path, "wb") as f:
            pickle.dump([[predictions, targets], coords, metadata], f)
        return path

    # ── read ─────────────────────────────────────────────────────────────────

    def load(self, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        path = self._resolve(name)
        with open(path, "rb") as f:
            data = pickle.load(f)
        return self._unpack(data)

    def load_timestep(self, name: str, t: int) -> np.ndarray:
        """PKL must deserialise everything — no partial-read shortcut."""
        predictions, _, _, _ = self.load(name)
        return predictions[t]

    # ── query ────────────────────────────────────────────────────────────────

    def list(self) -> list[str]:
        names = [p.stem for p in self.result_dir.glob(f"*{self.SUFFIX}")]
        names.sort(key=lambda n: _sort_key(n, self.result_dir), reverse=True)
        return names

    def exists(self, name: str) -> bool:
        return (self.result_dir / f"{name}{self.SUFFIX}").exists()

    def delete(self, name: str) -> None:
        path = self._resolve(name)
        path.unlink()

    def get_path(self, name: str) -> Path:
        return self._resolve(name)

    # ── internal ─────────────────────────────────────────────────────────────

    def _resolve(self, name: str) -> Path:
        """Return the path for `name`, accepting names with or without suffix."""
        # Strip any suffix the caller might have included
        stem = name if not name.endswith(self.SUFFIX) else name[: -len(self.SUFFIX)]
        path = self.result_dir / f"{stem}{self.SUFFIX}"
        if not path.exists():
            raise FileNotFoundError(f"Result not found: {path}")
        return path

    @staticmethod
    def _unpack(data) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """Unpack legacy 2-tuple or current 3-tuple format."""
        if len(data) == 2:
            result, crds = data
            meta = {}
        else:
            result, crds, meta = data
        predictions = np.asarray(result[0])
        targets     = np.asarray(result[1])
        coords      = np.asarray(crds)
        return predictions, targets, coords, meta
