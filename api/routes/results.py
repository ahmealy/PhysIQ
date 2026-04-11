"""
/results/* endpoints — list, get metadata, get per-frame data, get RMSE, delete.
"""

import os
import pickle
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Optional

import matplotlib.tri as mtri
import numpy as np
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/results")

RESULT_DIR = "result"

# ---------------------------------------------------------------------------
# In-memory pkl cache — avoids re-deserialising the same file on every request.
# On Visualize page load the frontend fires 3 parallel requests for the same
# file (/results/f, /results/f/rmse, /results/f/frame/0).  Without caching
# each request independently reads and unpickles the full array from disk,
# which for a 600-step rollout (~100 MB pickle) takes 20-40 s per call.
#
# Cache key: (filename, mtime) so stale entries are automatically invalidated
# when the file is replaced (e.g. after a new rollout).
# ---------------------------------------------------------------------------
_PKL_CACHE_MAX = 8   # keep at most 8 files in memory (~800 MB worst case)

class _LRUCache:
    def __init__(self, maxsize: int):
        self._maxsize = maxsize
        self._data: OrderedDict = OrderedDict()   # key → value

    def get(self, key):
        if key not in self._data:
            return None
        self._data.move_to_end(key)
        return self._data[key]

    def put(self, key, value):
        if key in self._data:
            self._data.move_to_end(key)
        else:
            if len(self._data) >= self._maxsize:
                self._data.popitem(last=False)   # evict LRU
        self._data[key] = value

    def invalidate_filename(self, filename: str):
        """Remove all cache entries for a given filename (after delete)."""
        to_del = [k for k in self._data if k[0] == filename]
        for k in to_del:
            del self._data[k]

_pkl_cache = _LRUCache(_PKL_CACHE_MAX)


def _list_pkl_files() -> list[str]:
    if not os.path.exists(RESULT_DIR):
        return []
    return sorted([f for f in os.listdir(RESULT_DIR) if f.endswith(".pkl")])


def _load_pkl(filename: str):
    """
    Load and deserialise a result pkl, using an in-memory LRU cache keyed by
    (filename, mtime).  Concurrent requests for the same file share one read.
    """
    # Guard against path traversal (e.g. "../../etc/passwd")
    safe_dir = os.path.realpath(RESULT_DIR)
    path = os.path.realpath(os.path.join(RESULT_DIR, filename))
    if not path.startswith(safe_dir + os.sep):
        raise HTTPException(400, "Invalid filename")
    if not os.path.exists(path):
        raise HTTPException(404, "Result file not found: %s" % filename)

    mtime = os.path.getmtime(path)
    cache_key = (filename, mtime)

    cached = _pkl_cache.get(cache_key)
    if cached is not None:
        return cached

    with open(path, "rb") as f:
        data = pickle.load(f)
    # Support both old format (2 elements) and new format (3 elements with metadata)
    if len(data) == 2:
        result, crds = data
        meta = {}
    else:
        result, crds, meta = data
    predicted = result[0]   # [T, N, 2] or [T, N, 3]
    targets   = result[1]   # [T, N, 2] or [T, N, 3]

    # Precompute triangles once and cache alongside arrays — avoids re-running
    # Delaunay triangulation (62 ms) on every /results/{filename} request.
    triangles = _get_triangles(crds)

    parsed = (predicted, targets, crds, meta, triangles)
    _pkl_cache.put(cache_key, parsed)
    return parsed


def _confidence_label(score: Optional[float]) -> Optional[str]:
    if score is None:
        return None
    if score >= 0.7:
        return "High"
    if score >= 0.4:
        return "Medium"
    return "Low"


def _compute_rmse(predicted: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Per-step RMSE — shape [T]."""
    T = predicted.shape[0]
    sq = np.square(predicted - targets).reshape(T, -1)
    return np.sqrt(np.mean(sq, axis=1))


def _compute_mae(predicted: np.ndarray, targets: np.ndarray,
                 target_field: str = "velocity") -> np.ndarray:
    """Per-step MAE of field magnitude across nodes, shape [T]."""
    if target_field == "pressure":
        pred_mag   = predicted[:, :, 0]   # [T, N] — scalar pressure, no norm
        target_mag = targets[:, :, 0]     # [T, N]
    else:
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
    Returns mesh structure (crds, triangles) and basic metadata.
    Does NOT compute RMSE here — use /results/{filename}/rmse for that.
    Does NOT return raw velocity arrays (too large).
    Use /results/{filename}/frame/{t} for per-frame data.
    """
    predicted, targets, crds, meta, triangles = _load_pkl(filename)

    confidence_score = meta.get("confidence_score", None)
    confidence_label = _confidence_label(confidence_score)
    domain = meta.get("domain", "cylinder_flow")
    target_field = meta.get("target_field", "velocity")

    # Compute a quick single scalar RMSE for the header badge (cheap: only step 0)
    rmse_step0 = float(np.sqrt(np.mean(np.square(predicted[0] - targets[0]))))

    return {
        "timesteps":        int(predicted.shape[0]),
        "num_nodes":        int(predicted.shape[1]),
        "dt":               0.01,
        "crds":             crds.tolist(),
        "triangles":        triangles.tolist(),
        "per_step_rmse":    None,        # fetch separately via /rmse if needed
        "elapsed_seconds":  None,
        "speedup":          None,
        "confidence_score": confidence_score,
        "confidence_label": confidence_label,
        "domain":           domain,
        "target_field":     target_field,
        "rmse_step0":       rmse_step0,
    }


@router.get("/{filename}/frame/{t}")
def get_frame(filename: str, t: int):
    """
    Returns visualization data for a single timestep.
    Called on-demand by the animation scrubber.
    """
    predicted, targets, crds, meta, _triangles = _load_pkl(filename)
    target_field = meta.get("target_field", "velocity")
    T = predicted.shape[0]

    if t < 0 or t >= T:
        raise HTTPException(400, "Timestep %d out of range (0-%d)" % (t, T - 1))

    if target_field == "pressure":
        pred_mag   = predicted[t, :, 0]   # [N] — scalar pressure
        target_mag = targets[t, :, 0]     # [N]
    else:
        pred_mag   = np.linalg.norm(predicted[t], axis=-1)   # [N]
        target_mag = np.linalg.norm(targets[t],   axis=-1)   # [N]
    error = np.abs(pred_mag - target_mag)   # [N]
    rmse  = float(np.sqrt(np.mean(np.square(predicted[t] - targets[t]))))

    return {
        "t":                    t,
        "time_seconds":         round(t * 0.01, 3),
        "predicted_magnitude":  pred_mag.tolist(),
        "target_magnitude":     target_mag.tolist(),
        "error":                error.tolist(),
        "rmse":                 rmse,
        "target_field":         target_field,
    }


@router.get("/{filename}/rmse")
def get_rmse(filename: str):
    """Returns the full RMSE and MAE curves — lightweight, no mesh data."""
    predicted, targets, _, meta, _triangles = _load_pkl(filename)
    target_field = meta.get("target_field", "velocity")
    per_step_rmse = _compute_rmse(predicted, targets)
    per_step_mae  = _compute_mae(predicted, targets, target_field)
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
        "target_field":   target_field,
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
    _pkl_cache.invalidate_filename(filename)   # drop from cache
    return {"status": "deleted", "filename": filename}
