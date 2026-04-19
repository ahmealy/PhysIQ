"""
storage/protocols.py — Abstract interface for result storage.

Repository Pattern: all API routes depend only on ResultRepository,
never on a specific backend (pkl, hdf5, etc.).

Strategy Pattern: storage/factory.py selects the implementation at
runtime from config so the backend can be swapped with one config key.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ResultRepository(Protocol):
    """
    Read/write interface for rollout result files.

    Each result stores:
      - predictions : float32 [T, N, D]   GNN predicted field
      - targets     : float32 [T, N, D]   ground-truth field (may equal predictions)
      - coords      : float32 [N, 2|3]    mesh node coordinates
      - metadata    : dict                 domain, target_field, confidence_score, …
    """

    def save(
        self,
        name: str,
        predictions: np.ndarray,
        targets: np.ndarray,
        coords: np.ndarray,
        metadata: dict,
    ) -> Path:
        """Persist a rollout result and return the file path."""
        ...

    def load(self, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """
        Return (predictions, targets, coords, metadata).
        Raises FileNotFoundError if the result does not exist.
        """
        ...

    def load_timestep(self, name: str, t: int) -> np.ndarray:
        """
        Return predictions[t] — single timestep.
        Backends that support partial reads (HDF5) are dramatically faster here;
        PKL falls back to a full load.
        """
        ...

    def list(self) -> list[str]:
        """Return result names (no extension) sorted newest-first."""
        ...

    def exists(self, name: str) -> bool:
        """Return True if a result with this name is stored."""
        ...

    def delete(self, name: str) -> None:
        """Remove the result file.  Raises FileNotFoundError if absent."""
        ...

    def get_path(self, name: str) -> Path:
        """Return the absolute path to the backing file for this result."""
        ...
