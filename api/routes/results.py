"""
/results/* endpoints — list, get metadata, get per-frame data, get RMSE, delete.
"""

import os
import pickle
from datetime import datetime, timezone
from typing import Optional

import matplotlib.tri as mtri
import numpy as np
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/results")

RESULT_DIR = "result"


def _list_pkl_files() -> list[str]:
    if not os.path.exists(RESULT_DIR):
        return []
    return sorted([f for f in os.listdir(RESULT_DIR) if f.endswith(".pkl")])


def _load_pkl(filename: str):
    # Guard against path traversal (e.g. "../../etc/passwd")
    safe_dir = os.path.realpath(RESULT_DIR)
    path = os.path.realpath(os.path.join(RESULT_DIR, filename))
    if not path.startswith(safe_dir + os.sep):
        raise HTTPException(400, "Invalid filename")
    if not os.path.exists(path):
        raise HTTPException(404, "Result file not found: %s" % filename)
    with open(path, "rb") as f:
        data = pickle.load(f)
    result, crds = data
    predicted = result[0]   # [T, N, 2]
    targets   = result[1]   # [T, N, 2]
    return predicted, targets, crds


def _compute_rmse(predicted: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Per-step RMSE — shape [T]."""
    T = predicted.shape[0]
    sq = np.square(predicted - targets).reshape(T, -1)
    return np.sqrt(np.mean(sq, axis=1))


def _compute_mae(predicted: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Per-step MAE — mean absolute error of velocity magnitude across nodes, shape [T]."""
    pred_mag   = np.linalg.norm(predicted, axis=-1)   # [T, N]
    target_mag = np.linalg.norm(targets,   axis=-1)   # [T, N]
    return np.mean(np.abs(pred_mag - target_mag), axis=1)  # [T]


def _get_triangles(crds: np.ndarray) -> np.ndarray:
    """Delaunay triangulation of 2D mesh coordinates. Returns [F, 3]."""
    triang = mtri.Triangulation(crds[:, 0], crds[:, 1])
    return triang.triangles


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("")
def list_results():
    files = _list_pkl_files()
    out = []
    for fname in files:
        path = os.path.join(RESULT_DIR, fname)
        stat = os.stat(path)
        # Try to infer trajectory index from filename (result0.pkl → 0)
        try:
            traj_idx = int("".join(filter(str.isdigit, fname.replace(".pkl", ""))))
        except ValueError:
            traj_idx = None
        out.append({
            "filename":          fname,
            "path":              path,
            "trajectory_index":  traj_idx,
            "created":           datetime.fromtimestamp(
                stat.st_ctime, tz=timezone.utc).isoformat(),
            "size_mb":           round(stat.st_size / 1e6, 2),
        })
    return out


@router.get("/{filename}")
def get_result(filename: str):
    """
    Returns mesh structure and RMSE curve.
    Does NOT return raw velocity arrays (too large).
    Use /results/{filename}/frame/{t} for per-frame data.
    """
    predicted, targets, crds = _load_pkl(filename)
    triangles   = _get_triangles(crds)
    per_step_rmse = _compute_rmse(predicted, targets)

    return {
        "timesteps":       int(predicted.shape[0]),
        "num_nodes":       int(predicted.shape[1]),
        "dt":              0.01,
        "crds":            crds.tolist(),
        "triangles":       triangles.tolist(),
        "per_step_rmse":   per_step_rmse.tolist(),
        "elapsed_seconds": None,   # not stored in pkl — available from rollout SSE
        "speedup":         None,
    }


@router.get("/{filename}/frame/{t}")
def get_frame(filename: str, t: int):
    """
    Returns visualization data for a single timestep.
    Called on-demand by the animation scrubber.
    """
    predicted, targets, crds = _load_pkl(filename)
    T = predicted.shape[0]

    if t < 0 or t >= T:
        raise HTTPException(400, "Timestep %d out of range (0-%d)" % (t, T - 1))

    pred_mag   = np.linalg.norm(predicted[t], axis=-1)   # [N]
    target_mag = np.linalg.norm(targets[t],   axis=-1)   # [N]
    error      = np.abs(pred_mag - target_mag)            # [N]
    rmse       = float(np.sqrt(np.mean(np.square(predicted[t] - targets[t]))))

    return {
        "t":                    t,
        "time_seconds":         round(t * 0.01, 3),
        "predicted_magnitude":  pred_mag.tolist(),
        "target_magnitude":     target_mag.tolist(),
        "error":                error.tolist(),
        "rmse":                 rmse,
    }


@router.get("/{filename}/rmse")
def get_rmse(filename: str):
    """Returns the full RMSE and MAE curves — lightweight, no mesh data."""
    predicted, targets, _ = _load_pkl(filename)
    per_step_rmse = _compute_rmse(predicted, targets)
    per_step_mae  = _compute_mae(predicted, targets)
    T = len(per_step_rmse)
    times = [round(i * 0.01, 3) for i in range(T)]

    return {
        "per_step_rmse":  per_step_rmse.tolist(),
        "per_step_mae":   per_step_mae.tolist(),
        "times":          times,
        "rmse_at_0":      float(per_step_rmse[0]),
        "rmse_at_300":    float(per_step_rmse[min(299, T - 1)]),
        "rmse_at_599":    float(per_step_rmse[min(598, T - 1)]),
        "mae_at_0":       float(per_step_mae[0]),
        "mae_at_end":     float(per_step_mae[-1]),
        "growth_ratio":   float(per_step_rmse[-1] / (per_step_rmse[0] + 1e-12)),
    }


@router.delete("/{filename}")
def delete_result(filename: str):
    safe_dir = os.path.realpath(RESULT_DIR)
    path = os.path.realpath(os.path.join(RESULT_DIR, filename))
    if not path.startswith(safe_dir + os.sep):
        raise HTTPException(400, "Invalid filename")
    if not os.path.exists(path):
        raise HTTPException(404, "File not found: %s" % filename)
    os.remove(path)
    return {"status": "deleted", "filename": filename}
