"""
Cloth Pose Extractor
=====================
Extracts compact design parameters from flag_simple (cloth) trajectories.

Design parameter
----------------
The cloth dataset has a fixed mesh topology (N=1579 nodes, same for all
trajectories) but varied initial cloth configurations (draping poses).

We represent each design as the PCA-compressed initial world position:

    design_vector [K] = PCA_K( world_pos_t0.flatten() )

K defaults to 16 (explains > 95% of variance empirically).

The "stress proxy" (optimisation target) is the mean deformation magnitude
at the steady-state final timestep:

    stress = mean_N( ||world_pos_T - mesh_extended_pos||₂ )

where mesh_extended_pos is the 3D rest position (z=0 plane).

Design principles
-----------------
- **Single Responsibility**: ``PosePCA`` fits/applies PCA;
  ``StressProxyComputer`` computes the stress proxy;
  ``ClothTrajectoryLoader`` handles file I/O.
- **Open / Closed**: subclass ``BaseClothExtractor`` to swap the pose
  representation without changing callers.
- **Dependency Inversion**: callers depend on the abstract interface.
"""
from __future__ import annotations

import os
import sys
import argparse
import pickle

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.utils import NodeType  # noqa: E402


# ---------------------------------------------------------------------------
# Data transfer objects
# ---------------------------------------------------------------------------

@dataclass
class ClothParams:
    """
    Compressed representation of one cloth design point.

    Fields
    ------
    pose_pca:   [K] PCA coefficients of the initial world position
    stress:     scalar stress proxy at final timestep
    traj_idx:   trajectory index (for back-reference)
    """
    pose_pca: np.ndarray    # [K]
    stress:   float
    traj_idx: int


# ---------------------------------------------------------------------------
# PCA for pose compression (Single Responsibility)
# ---------------------------------------------------------------------------

class PosePCA:
    """
    Fits a PCA on flattened initial cloth world positions and
    transforms / inverse-transforms them.

    Uses numpy SVD directly to avoid sklearn dependency.
    """

    def __init__(self, n_components: int = 16) -> None:
        self.n_components = n_components
        self.mean_:    Optional[np.ndarray] = None  # [N*3]
        self.components_: Optional[np.ndarray] = None  # [K, N*3]
        self._is_fitted: bool = False

    def fit(self, X: np.ndarray) -> "PosePCA":
        """
        Fit PCA to a matrix of flattened poses.

        Args:
            X: [n_samples, N*3] float32

        Returns:
            self
        """
        self.mean_ = X.mean(axis=0)
        X_c = X - self.mean_
        # Economy SVD
        _, s, Vt = np.linalg.svd(X_c, full_matrices=False)
        self.components_ = Vt[:self.n_components]  # [K, N*3]
        self._is_fitted = True

        # Report explained variance
        explained = (s[:self.n_components] ** 2).sum() / (s ** 2).sum()
        print(f"PosePCA fitted: {self.n_components} components explain "
              f"{explained * 100:.1f}% of variance  (n_samples={len(X)})")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """[n, N*3] → [n, K]"""
        return (X - self.mean_) @ self.components_.T

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """[n, K] → [n, N*3]"""
        return Z @ self.components_ + self.mean_

    def save(self, path: str) -> None:
        """Save PCA components as a portable .npz file.

        Avoids pickle module-path issues (``__main__.PosePCA``) when the
        extractor is run as a CLI script and the saved object is later loaded
        from a different entry point (e.g. ``train_cvae.py`` or the API).
        """
        np.savez(path,
                 mean_=self.mean_,
                 components_=self.components_,
                 n_components=np.array(self.n_components))

    @classmethod
    def load(cls, path: str) -> "PosePCA":
        """Load a previously saved PosePCA.

        Supports both the new .npz format and the legacy pickled format
        (backward compatibility).
        """
        # Try new .npz format first (path may or may not include .npz extension)
        npz_path = path if path.endswith(".npz") else path + ".npz"
        if os.path.exists(npz_path):
            data = np.load(npz_path)
            obj = cls(n_components=int(data["n_components"]))
            obj.mean_        = data["mean_"]
            obj.components_  = data["components_"]
            obj._is_fitted   = True
            return obj
        # Fall back to legacy pickle format
        if os.path.exists(path):
            import pickle as _pickle
            with open(path, "rb") as f:
                return _pickle.load(f)
        raise FileNotFoundError(f"PosePCA file not found: {path}")


# ---------------------------------------------------------------------------
# Stress proxy (Single Responsibility)
# ---------------------------------------------------------------------------

class StressProxyComputer:
    """
    Computes a deformation-magnitude stress proxy from cloth trajectories.

    Stress proxy = mean nodal displacement from the extended rest mesh
    at the final timestep.  Only NORMAL nodes are included.
    """

    def __call__(self,
                 world_pos: np.ndarray,   # [T, N, 3]
                 mesh_pos:  np.ndarray,   # [N, 2]
                 node_type: np.ndarray,   # [N, 1]
                 ) -> float:
        """
        Args:
            world_pos: [T, N, 3]
            mesh_pos:  [N, 2]   — 2D rest-config coordinates (z=0 plane)
            node_type: [N, 1]

        Returns:
            scalar stress proxy
        """
        # Rest position in 3D (z-extended: z=0)
        rest_3d = np.concatenate(
            [mesh_pos, np.zeros((len(mesh_pos), 1), dtype=np.float32)], axis=-1
        )  # [N, 3]

        nt = node_type.squeeze(-1)
        normal_mask = nt == int(NodeType.NORMAL)

        # Use final timestep (near steady-state)
        wp_final = world_pos[-1]  # [N, 3]
        disp     = np.linalg.norm(wp_final[normal_mask] - rest_3d[normal_mask], axis=-1)
        return float(disp.mean())


# ---------------------------------------------------------------------------
# Trajectory loader (Single Responsibility)
# ---------------------------------------------------------------------------

class ClothTrajectoryLoader:
    """Loads individual cloth trajectory .npz files."""

    def __init__(self, split_dir: str) -> None:
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Cloth split directory not found: {split_dir}")
        self._split_dir = split_dir

    def load(self, traj_idx: int) -> dict:
        """
        Load one trajectory.

        Returns dict with keys:
            world_pos [T, N, 3], mesh_pos [N, 2],
            node_type [N, 1],    cells [F, 3]
        """
        path = os.path.join(self._split_dir, f"traj_{traj_idx:05d}.npz")
        return dict(np.load(path))

    def count_trajectories(self, index_path: str) -> int:
        idx = np.load(index_path)
        return int(idx["n_traj"])


# ---------------------------------------------------------------------------
# Abstract extractor
# ---------------------------------------------------------------------------

class BaseClothExtractor(ABC):
    """Abstract base for cloth design parameter extraction."""

    @abstractmethod
    def extract_all(self, n_traj: int) -> tuple[np.ndarray, np.ndarray, "PosePCA"]:
        """
        Extract design vectors and stress proxies for all trajectories.

        Returns:
            pose_pca  [n_traj, K]  PCA coefficients
            stress    [n_traj]     stress proxy values
            pca       fitted PosePCA object (for inverse-transform at generation time)
        """


class ClothPoseExtractor(BaseClothExtractor):
    """
    Extracts PCA-compressed initial cloth pose + stress proxy.
    """

    def __init__(self,
                 loader:        ClothTrajectoryLoader,
                 pca:           Optional[PosePCA]           = None,
                 stress_fn:     Optional[StressProxyComputer] = None,
                 n_components:  int = 16) -> None:
        self._loader    = loader
        self._pca       = pca or PosePCA(n_components=n_components)
        self._stress_fn = stress_fn or StressProxyComputer()

    def extract_all(self, n_traj: int,
                    verbose: bool = True) -> tuple[np.ndarray, np.ndarray, PosePCA]:
        """
        Args:
            n_traj:  total number of trajectories to process
            verbose: print progress every 100 trajectories

        Returns:
            pose_pca [n_traj, K], stress [n_traj], fitted PosePCA
        """
        # ── Pass 1: collect all initial world positions for PCA fitting ──
        all_poses = []
        for i in range(n_traj):
            traj  = self._loader.load(i)
            pose0 = traj["world_pos"][0].astype(np.float32)  # [N, 3]
            all_poses.append(pose0.flatten())                 # [N*3]

        all_poses_arr = np.stack(all_poses, axis=0)  # [n_traj, N*3]
        self._pca.fit(all_poses_arr)

        # ── Pass 2: project + compute stress ─────────────────────────────
        pose_pca = np.zeros((n_traj, self._pca.n_components), dtype=np.float32)
        stress   = np.zeros(n_traj, dtype=np.float32)

        for i in range(n_traj):
            traj = self._loader.load(i)
            wp   = traj["world_pos"]         # [T, N, 3]
            mp   = traj["mesh_pos"]          # [N, 2]
            nt   = traj["node_type"]         # [N, 1]

            pose0        = wp[0].astype(np.float32).flatten()
            pose_pca[i]  = self._pca.transform(pose0.reshape(1, -1)).squeeze(0)
            stress[i]    = self._stress_fn(wp.astype(np.float32), mp, nt)

            if verbose and (i + 1) % 100 == 0:
                print(f"  [{i + 1}/{n_traj}] trajectories processed")

        if verbose:
            print(f"Extraction complete: {n_traj} trajectories, "
                  f"stress range=[{stress.min():.4f}, {stress.max():.4f}]")

        return pose_pca, stress, self._pca


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract cloth pose parameters from the flag_simple dataset."
    )
    p.add_argument("--data-dir",     default="data_flag",
                   help="Root directory of the flag_simple dataset (default: data_flag)")
    p.add_argument("--split",        default="train",
                   choices=["train", "valid", "test"])
    p.add_argument("--n-components", type=int, default=16,
                   help="Number of PCA components (default: 16)")
    p.add_argument("--out-dir",      default=None,
                   help="Directory to save outputs (default: <data-dir>/<split>/)")
    p.add_argument("--quiet",        action="store_true")
    return p


def main(argv: list[str] | None = None) -> None:
    args    = _build_arg_parser().parse_args(argv)
    out_dir = args.out_dir or os.path.join(args.data_dir, args.split)
    os.makedirs(out_dir, exist_ok=True)

    split_dir  = os.path.join(args.data_dir, args.split)
    index_path = os.path.join(args.data_dir, f"{args.split}_index.npz")

    loader    = ClothTrajectoryLoader(split_dir=split_dir)
    n_traj    = loader.count_trajectories(index_path)
    print(f"Found {n_traj} trajectories in: {split_dir}")

    extractor = ClothPoseExtractor(loader=loader, n_components=args.n_components)
    pose_pca, stress, pca = extractor.extract_all(n_traj, verbose=not args.quiet)

    # Save outputs
    pose_path   = os.path.join(out_dir, "cloth_pose_pca.npy")
    stress_path = os.path.join(out_dir, "cloth_stress.npy")
    pca_path    = os.path.join(out_dir, "cloth_pca.pkl")

    np.save(pose_path,   pose_pca)
    np.save(stress_path, stress)
    pca.save(pca_path)

    print(f"\nSaved:")
    print(f"  Pose PCA   → {pose_path}   shape={pose_pca.shape}")
    print(f"  Stress     → {stress_path}  shape={stress.shape}")
    print(f"  PCA model  → {pca_path}")


if __name__ == "__main__":
    main()
