"""Protocols for ingest adapters."""
from typing import Protocol, runtime_checkable
import numpy as np
from pathlib import Path


@runtime_checkable
class SolverAdapter(Protocol):
    """Adapter that reads raw simulation data from a source format."""

    @property
    def name(self) -> str:
        """Human-readable name of the adapter (e.g. 'TFRecord', 'OpenFOAM')."""
        ...

    def list_splits(self) -> list[str]:
        """Return available split names (e.g. ['train', 'valid', 'test'])."""
        ...

    def load_split(self, split: str) -> dict:
        """
        Load a split and return a dict with keys:
            positions  np.ndarray [S, T+1, N, D]
            velocities np.ndarray [S, T, N, D]
            node_types np.ndarray [S, N]
            pressures  np.ndarray [S, T, N, 1]  (optional)
            metadata   dict
        """
        ...

    @property
    def source_path(self) -> Path:
        """Path to the raw data source."""
        ...
