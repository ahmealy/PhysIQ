"""
/results/{filename}/physics endpoint — per-frame vorticity, energy series, divergence proxy.

All heavy computation runs here so the frontend gets clean JSON.
Results are cached in-process (LRU) to avoid recomputing on every scrubber drag.

The route is a plain `def` so FastAPI dispatches it to the threadpool automatically,
keeping the event loop free during the scipy KD-tree and vorticity computation.
"""

import functools
import os
import pickle

import numpy as np
from fastapi import APIRouter, HTTPException
from scipy.spatial import cKDTree

from physics.poisson_pressure import PoissonPressureCorrector
from storage.factory import get_repository

router = APIRouter(prefix="/results")

RESULT_DIR = "result"
_K_NEIGHBORS = 7          # k-nearest (includes self → 6 actual neighbors)
_CACHE_SIZE   = 32        # number of (filename, t) pairs cached in memory

_repo = get_repository()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_pkl_physics(filename: str):
    """Load pkl and return (predicted [T,N,2], targets [T,N,2], crds [N,2])."""
    try:
        predicted, targets, crds, _meta = _repo.load(filename)
    except FileNotFoundError:
        raise HTTPException(404, "Result file not found: %s" % filename)
    return predicted, targets, crds


@functools.lru_cache(maxsize=_CACHE_SIZE)
def _build_kdtree(crds_bytes: bytes, N: int) -> cKDTree:
    """Build and cache a k-d tree for a given set of node coordinates.
    crds_bytes is the raw bytes of a float32 [N, 2] array (hashable for caching).
    """
    crds = np.frombuffer(crds_bytes, dtype=np.float32).reshape(N, 2)
    return cKDTree(crds)


def _compute_vorticity(crds: np.ndarray, vel: np.ndarray) -> np.ndarray:
    """
    Compute per-node vorticity ω = ∂vy/∂x − ∂vx/∂y using unstructured
    least-squares finite differences on the k-nearest-neighbor graph.

    Vectorized: builds batched [N, K-1, 2] arrays and solves all N systems
    simultaneously using np.linalg.lstsq over stacked matrices, avoiding the
    Python for-loop over nodes.

    Args:
        crds: [N, 2] node coordinates
        vel:  [N, 2] velocity field (vx, vy)

    Returns:
        omega: [N] vorticity values (positive = counter-clockwise rotation)
    """
    N = crds.shape[0]
    K = _K_NEIGHBORS
    crds_bytes = crds.astype(np.float32).tobytes()
    tree = _build_kdtree(crds_bytes, N)

    # Query k nearest neighbors for every node (includes the node itself at index 0)
    _, idxs = tree.query(crds, k=K)  # [N, K]
    neighbor_idxs = idxs[:, 1:]       # [N, K-1]  — exclude self

    # dr[i, j] = crds[neighbor_j_of_i] - crds[i]   → [N, K-1, 2]
    dr = crds[neighbor_idxs] - crds[:, np.newaxis, :]  # [N, K-1, 2]

    # dv[i, j] = vel[neighbor_j_of_i] - vel[i]     → [N, K-1, 2]
    dv = vel[neighbor_idxs] - vel[:, np.newaxis, :]    # [N, K-1, 2]

    # Solve N independent least-squares systems: dr[i] @ grad[i] ≈ dv[i]
    # np.linalg.lstsq doesn't batch, but np.linalg.solve can if we use
    # the normal equations: (drT dr) grad = drT dv  (2×2 solve, fast)
    drT  = dr.transpose(0, 2, 1)              # [N, 2, K-1]
    A    = drT @ dr                            # [N, 2, 2]  (normal eqs LHS)
    rhs  = drT @ dv                            # [N, 2, 2]  (drT @ [dvx, dvy])

    # Regularize degenerate nodes (collinear neighbors) to avoid singular A
    eye  = 1e-6 * np.eye(2, dtype=np.float64)
    A    = A.astype(np.float64) + eye[np.newaxis]

    # Batch solve: grad[i] = A[i]^-1 @ rhs[i], shape [N, 2, 2]
    grad = np.linalg.solve(A, rhs.astype(np.float64))  # [N, 2, 2]

    # grad[:, :, 0] → gradients of vx:  [∂vx/∂x, ∂vx/∂y]
    # grad[:, :, 1] → gradients of vy:  [∂vy/∂x, ∂vy/∂y]
    # ω = ∂vy/∂x − ∂vx/∂y = grad[:,0,1] − grad[:,1,0]
    omega = (grad[:, 0, 1] - grad[:, 1, 0]).astype(np.float32)
    return omega


def _compute_energy_series(predicted: np.ndarray, targets: np.ndarray):
    """
    Compute kinetic energy per timestep.
    E = 0.5 * Σ_nodes ||v||²  (unnormalized — relative changes are what matter)

    Returns:
        pred_e:   [T] predicted energy per timestep
        target_e: [T] ground-truth energy per timestep
    """
    pred_e   = 0.5 * np.sum(np.linalg.norm(predicted, axis=-1) ** 2, axis=1)   # [T]
    target_e = 0.5 * np.sum(np.linalg.norm(targets,   axis=-1) ** 2, axis=1)   # [T]
    return pred_e, target_e


def _compute_divergence_series(crds: np.ndarray,
                                predicted: np.ndarray,
                                targets: np.ndarray) -> tuple:
    """
    Approximate divergence proxy: mean |∂vx/∂x + ∂vy/∂y| per timestep.
    For incompressible flow this should be ~0 everywhere.

    Runs on every 10th timestep for performance, then interpolates.
    Vectorized over nodes using batched normal equations.

    Returns:
        div_pred   [T] — divergence proxy for predicted field
        div_target [T] — divergence proxy for ground-truth field
    """
    T = predicted.shape[0]
    N = crds.shape[0]
    K = _K_NEIGHBORS
    crds_bytes = crds.astype(np.float32).tobytes()
    tree = _build_kdtree(crds_bytes, N)
    _, idxs = tree.query(crds, k=K)   # [N, K]
    neighbor_idxs = idxs[:, 1:]       # [N, K-1]

    dr   = crds[neighbor_idxs] - crds[:, np.newaxis, :]  # [N, K-1, 2]
    drT  = dr.transpose(0, 2, 1)                          # [N, 2, K-1]
    A    = (drT @ dr).astype(np.float64)                   # [N, 2, 2]
    eye  = 1e-6 * np.eye(2, dtype=np.float64)
    A   += eye[np.newaxis]

    def _div_at(vel: np.ndarray) -> float:
        """Vectorized mean |∇·v| for a single timestep."""
        dv  = vel[neighbor_idxs] - vel[:, np.newaxis, :]  # [N, K-1, 2]
        rhs = (drT @ dv.astype(np.float64))                # [N, 2, 2]
        grad = np.linalg.solve(A, rhs)                     # [N, 2, 2]
        # div = ∂vx/∂x + ∂vy/∂y = grad[:,0,0] + grad[:,1,1]
        div  = np.abs(grad[:, 0, 0] + grad[:, 1, 1])
        return float(np.mean(div))

    # Sample every 10 steps for performance
    sample_ts = list(range(0, T, 10))
    div_pred_samples   = [_div_at(predicted[t]) for t in sample_ts]
    div_target_samples = [_div_at(targets[t])   for t in sample_ts]

    # Linear interpolation back to full T length
    all_ts = np.arange(T)
    div_pred   = np.interp(all_ts, sample_ts, div_pred_samples).tolist()
    div_target = np.interp(all_ts, sample_ts, div_target_samples).tolist()
    return div_pred, div_target


# ── Route ─────────────────────────────────────────────────────────────────────

@router.get("/{filename}/physics")
def get_physics(filename: str, t: int = 0):
    """
    Returns physics-derived quantities for a single timestep + full energy series.

    Quantities:
      - vorticity (ω = ∂vy/∂x − ∂vx/∂y) at every node for timestep t
      - kinetic energy series across all timesteps (lightweight scalar per step)
      - divergence proxy series (incompressibility indicator)

    Vorticity is computed via unstructured least-squares finite differences
    on the 6-nearest-neighbor graph of the mesh — no face/element connectivity needed.

    Note: First call per file is slow (~1–3s for N≈1876 × 6 neighbors).
    Subsequent calls for the same file are fast (k-d tree cached).
    Energy and divergence series are returned in full on every call
    (computed once per file, no per-timestep re-loading needed by caller).
    """
    predicted, targets, crds = _load_pkl_physics(filename)
    T, N, _ = predicted.shape

    if t < 0 or t >= T:
        raise HTTPException(400, "Timestep %d out of range (0–%d)" % (t, T - 1))

    # Vorticity at the requested timestep
    omega_pred   = _compute_vorticity(crds, predicted[t])   # [N]
    omega_target = _compute_vorticity(crds, targets[t])     # [N]

    # Full energy series (cheap — just norms, no spatial derivatives)
    pred_e, target_e = _compute_energy_series(predicted, targets)
    energy_drift = float(pred_e[-1] - target_e[-1])

    # Divergence series (moderate cost — sampled every 10 steps)
    div_pred, div_target = _compute_divergence_series(crds, predicted, targets)

    omega_all = np.concatenate([omega_pred, omega_target])
    omega_min  = float(np.min(omega_all))
    omega_max  = float(np.max(omega_all))

    return {
        "t":                    t,
        "time_seconds":         round(t * 0.01, 3),
        "num_nodes":            N,
        "timesteps":            T,

        # Per-node vorticity at timestep t (length N each)
        "vorticity_pred":       omega_pred.tolist(),
        "vorticity_target":     omega_target.tolist(),
        "omega_min":            omega_min,
        "omega_max":            omega_max,

        # Full energy series (length T each)
        "energy_pred_series":   pred_e.tolist(),
        "energy_target_series": target_e.tolist(),
        "energy_drift":         energy_drift,

        # Divergence proxy series (length T each) — ≈0 for incompressible flow
        "divergence_pred":      div_pred,
        "divergence_target":    div_target,
    }


# ── Helmholtz correction endpoint ────────────────────────────────────────────

def _build_correction_edges(crds: np.ndarray, k: int = 7) -> np.ndarray:
    """Build k-NN edges for PoissonPressureCorrector (undirected, i < j)."""
    return PoissonPressureCorrector._build_knn_edges(crds, k)


@router.get("/{filename}/physics/corrected_divergence")
def get_corrected_divergence(filename: str):
    """
    Apply Helmholtz pressure projection to the predicted velocity field and return:
    - divergence_before: RMS divergence per sampled timestep (raw GNN output)
    - divergence_after:  RMS divergence per sampled timestep (after Poisson correction)
    - divergence_reduction_pct: percentage reduction in mean divergence
    - correction_norm: mean ||v_corrected - v_pred|| per sampled timestep

    Uses sparse LU factorisation cached per mesh — first call ~0.5s, subsequent calls fast.
    Samples every 10 timesteps for performance (matches existing divergence series pattern).
    """
    predicted, targets, crds = _load_pkl_physics(filename)
    T, N, _ = predicted.shape

    crds_f64 = crds.astype(np.float64)
    corrector = PoissonPressureCorrector(crds_f64, edges=None, k_neighbors=_K_NEIGHBORS)

    sample_ts = list(range(0, T, 10))
    div_before = []
    div_after  = []
    corr_norms = []

    for t in sample_ts:
        vel = predicted[t].astype(np.float64)
        rms_before = corrector.divergence_rms(vel)
        vel_corr   = corrector.correct(vel)
        rms_after  = corrector.divergence_rms(vel_corr)
        diff_norm  = float(np.mean(np.linalg.norm(vel_corr - vel, axis=-1)))
        div_before.append(rms_before)
        div_after.append(rms_after)
        corr_norms.append(diff_norm)

    mean_before = float(np.mean(div_before))
    mean_after  = float(np.mean(div_after))
    reduction_pct = (
        100.0 * (mean_before - mean_after) / mean_before
        if mean_before > 0 else 0.0
    )

    # Interpolate back to full T length for frontend compatibility
    all_ts = np.arange(T)
    div_before_full = np.interp(all_ts, sample_ts, div_before).tolist()
    div_after_full  = np.interp(all_ts, sample_ts, div_after).tolist()
    corr_norm_full  = np.interp(all_ts, sample_ts, corr_norms).tolist()

    return {
        "filename":                filename,
        "num_nodes":               N,
        "timesteps":               T,
        "divergence_before":       div_before_full,
        "divergence_after":        div_after_full,
        "divergence_reduction_pct": round(reduction_pct, 2),
        "correction_norm":         corr_norm_full,
    }
