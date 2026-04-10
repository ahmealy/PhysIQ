"""
CFD Shape Parameter Extractor
==============================
Extracts structured design parameters (cx, cy, r, v_inlet) from the
cylinder_flow dataset by analysing per-trajectory node positions and types.

Design principles
-----------------
- **Single Responsibility**: each class does exactly one thing.
- **Open / Closed**: add new parameter extractors by subclassing
  ``BaseParamExtractor`` without touching existing code.
- **Dependency Inversion**: ``TrajectoryParamExtractor`` depends on the
  abstract ``BaseParamExtractor``, not on concrete implementations.
- **Interface Segregation**: ``CircleFitter`` exposes only ``fit(pts)``;
  callers need not know about the optimisation internals.
"""
from __future__ import annotations

import sys
import os
import argparse
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from typing import Optional

# Allow running as a script from any CWD
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.utils import NodeType  # noqa: E402


# ---------------------------------------------------------------------------
# Data transfer objects
# ---------------------------------------------------------------------------

@dataclass
class CylinderParams:
    """Structured representation of one cylinder_flow design point."""
    cx: float         # cylinder centre x  (normalised to [0, 1])
    cy: float         # cylinder centre y  (normalised to [0, 1])
    r:  float         # cylinder radius    (normalised units)
    v_inlet: float    # inlet velocity magnitude (m/s)

    def to_array(self) -> np.ndarray:
        """Return params as a 1-D float32 array [cx, cy, r, v_inlet]."""
        return np.array([self.cx, self.cy, self.r, self.v_inlet], dtype=np.float32)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "CylinderParams":
        return cls(cx=float(arr[0]), cy=float(arr[1]),
                   r=float(arr[2]), v_inlet=float(arr[3]))

    @classmethod
    def feature_names(cls) -> list[str]:
        return [f.name for f in fields(cls)]


# ---------------------------------------------------------------------------
# Circle fitting
# ---------------------------------------------------------------------------

class CircleFitter:
    """
    Algebraic least-squares circle fit (Kåsa method).

    Fits the equation  (x - cx)² + (y - cy)² = r²  to a point cloud.
    Single Responsibility: only circle fitting, nothing else.
    """

    def fit(self, pts: np.ndarray) -> tuple[float, float, float]:
        """
        Fit a circle to 2-D points.

        Args:
            pts: [M, 2] array of (x, y) coordinates.

        Returns:
            (cx, cy, r) — centre and radius in the same units as pts.
        """
        if pts.shape[0] < 3:
            raise ValueError("Need at least 3 points to fit a circle.")

        x, y = pts[:, 0], pts[:, 1]
        A = np.column_stack([x, y, np.ones(len(x))])
        b = x ** 2 + y ** 2
        # Least-squares solution: A @ [2cx, 2cy, cx²+cy²-r²]ᵀ = b
        result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        cx  = result[0] / 2.0
        cy  = result[1] / 2.0
        r   = float(np.sqrt(max(result[2] + cx ** 2 + cy ** 2, 0.0)))
        return float(cx), float(cy), r


# ---------------------------------------------------------------------------
# Abstract extractor (Open / Closed principle base)
# ---------------------------------------------------------------------------

class BaseParamExtractor(ABC):
    """
    Abstract base for trajectory-level design parameter extraction.

    Subclass and implement ``extract`` to add new domains.
    """

    @abstractmethod
    def extract(self,
                pos:       np.ndarray,   # [N, 2]
                node_type: np.ndarray,   # [N, 1] int
                velocity:  np.ndarray,   # [N, 2] at t=0
                ) -> Optional[CylinderParams]:
        """
        Extract design parameters from one trajectory's static data.

        Returns None if extraction fails (e.g. no obstacle nodes).
        """


class CylinderParamExtractor(BaseParamExtractor):
    """
    Extracts (cx, cy, r, v_inlet) from a cylinder_flow trajectory.

    Algorithm
    ---------
    In the DeepMind cylinder_flow dataset, the cylinder surface nodes are
    labelled ``NodeType.WALL_BOUNDARY`` (6) — the same type as the channel
    top/bottom walls.  To isolate the cylinder nodes we:

    1. Select all WALL_BOUNDARY nodes.
    2. Build the domain bounding box (xmin, xmax, ymin, ymax).
    3. Discard nodes that lie on the domain boundary edges (within ``eps``).
       The remaining "interior wall" nodes are the cylinder surface.
    4. Fit a circle to those interior wall nodes via algebraic least-squares.
    5. Extract v_inlet as the mean x-velocity of INFLOW nodes at t=0.
    """

    # Tolerance (in mesh units) for deciding whether a node lies on the
    # axis-aligned domain boundary.  The mesh spacing at the walls is
    # ~0.018 so 1e-3 is safely smaller than any edge gap.
    BOUNDARY_EPS: float = 1e-3

    def __init__(self, circle_fitter: Optional[CircleFitter] = None):
        # Dependency injection — allows swapping the fitter in tests
        self._fitter = circle_fitter or CircleFitter()

    def _isolate_cylinder_nodes(self, pos: np.ndarray,
                                 node_type_1d: np.ndarray) -> np.ndarray:
        """
        Return the (x, y) positions of cylinder surface nodes.

        These are WALL_BOUNDARY nodes that do NOT sit on any of the four
        axis-aligned domain edges.
        """
        wall_mask = node_type_1d == int(NodeType.WALL_BOUNDARY)
        if wall_mask.sum() == 0:
            return np.empty((0, 2), dtype=np.float32)

        wall_pos = pos[wall_mask]  # [M, 2]

        xmin, xmax = float(pos[:, 0].min()), float(pos[:, 0].max())
        ymin, ymax = float(pos[:, 1].min()), float(pos[:, 1].max())
        eps = self.BOUNDARY_EPS

        on_top    = np.abs(wall_pos[:, 1] - ymax) < eps
        on_bottom = np.abs(wall_pos[:, 1] - ymin) < eps
        on_left   = np.abs(wall_pos[:, 0] - xmin) < eps
        on_right  = np.abs(wall_pos[:, 0] - xmax) < eps

        interior = ~(on_top | on_bottom | on_left | on_right)
        return wall_pos[interior]  # [K, 2] — cylinder surface nodes only

    def extract(self,
                pos:       np.ndarray,
                node_type: np.ndarray,
                velocity:  np.ndarray,
                ) -> Optional[CylinderParams]:
        node_type_1d = node_type.squeeze(-1)  # [N]

        # ── 1. Cylinder surface nodes ──────────────────────────────────────
        cylinder_pts = self._isolate_cylinder_nodes(pos, node_type_1d)
        if len(cylinder_pts) < 3:
            return None  # not enough points to fit circle

        try:
            cx, cy, r = self._fitter.fit(cylinder_pts)
        except (np.linalg.LinAlgError, ValueError):
            return None

        # ── 2. Inlet velocity ─────────────────────────────────────────────
        inflow_mask = node_type_1d == int(NodeType.INFLOW)
        if inflow_mask.sum() > 0:
            v_inlet = float(np.mean(velocity[inflow_mask, 0]))
        else:
            # Fallback: mean x-velocity of all nodes
            v_inlet = float(np.mean(velocity[:, 0]))

        return CylinderParams(cx=cx, cy=cy, r=r, v_inlet=v_inlet)


# ---------------------------------------------------------------------------
# Trajectory-level driver
# ---------------------------------------------------------------------------

class TrajectoryParamExtractor:
    """
    Iterates over all trajectories in a dataset and extracts design params.

    Depends on the *abstract* BaseParamExtractor (Dependency Inversion).
    The concrete extractor is injected at construction time.
    """

    def __init__(self, extractor: BaseParamExtractor):
        self._extractor = extractor

    def extract_all(self, meta: dict, velocity_memmap: np.ndarray,
                    verbose: bool = True) -> np.ndarray:
        """
        Extract one param vector per trajectory.

        Args:
            meta:            dict loaded from .npz with keys:
                             'pos', 'node_type', 'cells', 'indices', 'cindices'
            velocity_memmap: np.memmap of shape [N_total, T, 2]
            verbose:         print progress every 100 trajectories

        Returns:
            params [N_traj, 4] float32 array (NaN rows = failed extraction)
        """
        indices   = meta["indices"]
        n_traj    = len(indices) - 1
        params    = np.full((n_traj, 4), np.nan, dtype=np.float32)
        n_failed  = 0

        for i in range(n_traj):
            start, end = int(indices[i]), int(indices[i + 1])
            pos        = meta["pos"][start:end]        # [N, 2]
            node_type  = meta["node_type"][start:end]  # [N, 1]
            velocity   = velocity_memmap[start:end, 0] # [N, 2] at t=0

            result = self._extractor.extract(pos, node_type, velocity)
            if result is not None:
                params[i] = result.to_array()
            else:
                n_failed += 1

            if verbose and (i + 1) % 100 == 0:
                print(f"  [{i + 1}/{n_traj}] trajectories processed "
                      f"({n_failed} failed so far)")

        if verbose:
            valid_rows = np.isfinite(params[:, 0]).sum()
            print(f"Extraction complete: {valid_rows}/{n_traj} succeeded, "
                  f"{n_failed} failed.")

        return params


# ---------------------------------------------------------------------------
# Dataset loader (thin wrapper around FpcDataset's raw numpy files)
# ---------------------------------------------------------------------------

class CylinderDatasetLoader:
    """
    Loads the raw numpy arrays from a cylinder_flow split directory.

    Does NOT create a PyG Dataset — this is intentional (we only need
    the static node arrays, not the full sample-level Data objects).
    """

    def __init__(self, data_root: str, split: str = "train"):
        self._meta_path = os.path.join(data_root, f"{split}.npz")
        self._dat_path  = os.path.join(data_root, f"{split}.dat")

        if not os.path.exists(self._meta_path):
            raise FileNotFoundError(f"Metadata file not found: {self._meta_path}")
        if not os.path.exists(self._dat_path):
            raise FileNotFoundError(f"Velocity data file not found: {self._dat_path}")

    def load(self) -> tuple[dict, np.ndarray]:
        """
        Returns (meta_dict, velocity_memmap).

        meta_dict keys: 'pos', 'node_type', 'cells', 'indices', 'cindices',
                        'all_velocity_shape'
        velocity_memmap shape: [N_total, T, 2]
        """
        keys = ("pos", "node_type", "cells", "indices", "cindices",
                "all_velocity_shape")
        raw  = np.load(self._meta_path, allow_pickle=True)
        meta = {k: raw[k] for k in keys}

        vel_shape = tuple(int(x) for x in meta["all_velocity_shape"])
        memmap    = np.memmap(self._dat_path, dtype="float32", mode="r",
                              shape=vel_shape)
        return meta, memmap


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract cylinder shape parameters from the cylinder_flow dataset."
    )
    p.add_argument("--data-dir",  default="data",
                   help="Root directory of the cylinder_flow dataset (default: data)")
    p.add_argument("--split",     default="train",
                   choices=["train", "valid", "test"],
                   help="Which split to process (default: train)")
    p.add_argument("--out-dir",   default=None,
                   help="Directory to save design_params.npy "
                        "(default: <data-dir>/cylinder_flow/)")
    p.add_argument("--quiet",     action="store_true",
                   help="Suppress per-trajectory progress output")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)

    data_dir = args.data_dir
    # If the user points at a root that contains 'cylinder_flow/', use that.
    # Otherwise assume data_dir itself is the cylinder_flow directory.
    candidate = os.path.join(data_dir, "cylinder_flow")
    if os.path.exists(candidate):
        data_dir = candidate

    out_dir = args.out_dir or data_dir
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "design_params.npy")

    print(f"Loading {args.split} split from: {data_dir}")
    loader  = CylinderDatasetLoader(data_root=data_dir, split=args.split)
    meta, vel = loader.load()

    n_traj = len(meta["indices"]) - 1
    print(f"Found {n_traj} trajectories.")

    extractor = TrajectoryParamExtractor(
        extractor=CylinderParamExtractor(circle_fitter=CircleFitter())
    )
    params = extractor.extract_all(meta, vel, verbose=not args.quiet)

    np.save(out_path, params)
    print(f"\nSaved design_params.npy to: {out_path}")
    print(f"Shape: {params.shape}  — columns: {CylinderParams.feature_names()}")

    # Preview first 5 valid rows
    valid = params[np.isfinite(params[:, 0])]
    print("\nFirst 5 valid rows:")
    header = "  " + "  ".join(f"{n:>10}" for n in CylinderParams.feature_names())
    print(header)
    for row in valid[:5]:
        print("  " + "  ".join(f"{v:10.4f}" for v in row))


if __name__ == "__main__":
    main()
