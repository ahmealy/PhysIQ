"""
/results/* endpoints — list, get metadata, get per-frame data, get RMSE, delete.
"""

import math
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
    stored_faces = meta.get("faces")
    if stored_faces is not None:
        triangles = np.asarray(stored_faces, dtype=np.int32)
    else:
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
    if target_field == "world_pos":
        # Mean per-node 3D position error per timestep
        return np.mean(np.linalg.norm(predicted - targets, axis=-1), axis=1)
    if target_field == "pressure":
        pred_mag   = predicted[:, :, 0]   # [T, N] — scalar pressure, no norm
        target_mag = targets[:, :, 0]     # [T, N]
    else:
        pred_mag   = np.linalg.norm(predicted, axis=-1)   # [T, N]
        target_mag = np.linalg.norm(targets,   axis=-1)   # [T, N]
    return np.mean(np.abs(pred_mag - target_mag), axis=1)  # [T]


def _safe_float(v: float) -> Optional[float]:
    """Return None for NaN/Inf so JSON serialisation never crashes."""
    if not math.isfinite(v):
        return None
    return float(v)


def _safe_list(arr: np.ndarray) -> list:
    """Convert numpy array to list, replacing NaN/Inf with None."""
    flat = arr.flatten().tolist()
    return [None if (isinstance(x, float) and not math.isfinite(x)) else x for x in flat]


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
    is_generate = bool(meta.get("is_generate", False))
    speedup         = meta.get("speedup", None)
    elapsed_seconds = meta.get("elapsed_seconds", None)

    # Compute a quick single scalar RMSE for the header badge (cheap: only step 0)
    rmse_step0 = _safe_float(float(np.sqrt(np.mean(np.square(predicted[0] - targets[0])))))

    return {
        "timesteps":        int(predicted.shape[0]),
        "num_nodes":        int(predicted.shape[1]),
        "dt":               0.01,
        "crds":             crds.tolist(),
        "triangles":        triangles.tolist(),
        "per_step_rmse":    None,        # fetch separately via /rmse if needed
        "elapsed_seconds":  elapsed_seconds,
        "speedup":          speedup,
        "confidence_score": confidence_score,
        "confidence_label": confidence_label,
        "domain":           domain,
        "target_field":     target_field,
        "is_generate":      is_generate,
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

    if target_field == "world_pos":
        # Cloth: colour panels by per-node displacement from rest (mesh_pos).
        # crds = mesh_pos [N, 2]; pad to 3D for distance from 3D world_pos.
        mesh_rest = np.pad(crds, ((0, 0), (0, 1)), constant_values=0.0)  # [N, 3]
        pred_mag   = np.linalg.norm(predicted[t] - mesh_rest, axis=-1)   # displacement from rest
        target_mag = np.linalg.norm(targets[t]   - mesh_rest, axis=-1)
        error      = np.linalg.norm(predicted[t] - targets[t], axis=-1)  # pred vs GT error
    elif target_field == "pressure":
        pred_mag   = predicted[t, :, 0]   # [N] — scalar pressure
        target_mag = targets[t, :, 0]     # [N]
        error      = np.abs(pred_mag - target_mag)
    else:
        # velocity (2D), world_pos (3D cloth), or any other multi-dim field
        pred_mag   = np.linalg.norm(predicted[t], axis=-1)   # [N]
        target_mag = np.linalg.norm(targets[t],   axis=-1)   # [N]
        error      = np.abs(pred_mag - target_mag)
    rmse  = float(np.sqrt(np.mean(np.square(predicted[t] - targets[t]))))

    domain = meta.get("domain", "cylinder_flow")
    resp = {
        "t":                    t,
        "time_seconds":         round(t * 0.01, 3),
        "predicted_magnitude":  _safe_list(pred_mag),
        "target_magnitude":     _safe_list(target_mag),
        "error":                _safe_list(error),
        "rmse":                 _safe_float(rmse),
        "target_field":         target_field,
    }
    if domain == "flag_simple":
        resp["world_pos_pred"]   = predicted[t].tolist()   # [N, 3]
        resp["world_pos_target"] = targets[t].tolist()     # [N, 3]
    return resp


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
        "per_step_rmse":  [_safe_float(v) for v in per_step_rmse.tolist()],
        "per_step_mae":   [_safe_float(v) for v in per_step_mae.tolist()],
        "times":          times,
        "rmse_at_0":      _safe_float(float(per_step_rmse[0])),
        "rmse_at_300":    _safe_float(float(per_step_rmse[min(299, T - 1)])),
        "rmse_at_599":    _safe_float(float(per_step_rmse[min(598, T - 1)])),
        "mae_at_0":       _safe_float(float(per_step_mae[0])),
        "mae_at_end":     _safe_float(float(per_step_mae[-1])),
        "growth_ratio":   _safe_float(float(per_step_rmse[-1] / (per_step_rmse[0] + 1e-12))),
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


@router.get("/{filename}/download")
def download_result(filename: str):
    """Download the raw pkl file."""
    from fastapi.responses import FileResponse
    safe_dir = os.path.realpath(RESULT_DIR)
    path = os.path.realpath(os.path.join(RESULT_DIR, filename))
    if not path.startswith(safe_dir + os.sep):
        raise HTTPException(400, "Invalid filename")
    if not os.path.exists(path):
        raise HTTPException(404, "File not found: %s" % filename)
    return FileResponse(path, media_type="application/octet-stream", filename=filename)


@router.get("/{filename}/cloth_physics")
def get_cloth_physics(filename: str):
    """Per-step edge stretch statistics for cloth (flag_simple) rollouts."""
    predicted, targets, crds, meta, triangles = _load_pkl(filename)
    domain = meta.get("domain", "cylinder_flow")
    if domain != "flag_simple":
        raise HTTPException(400, "cloth_physics only available for flag_simple domain")

    # Use stored faces if available, else use triangles (Delaunay fallback)
    faces = np.asarray(meta.get("faces", triangles), dtype=np.int32)  # [F, 3]

    # Build unique undirected edges from face list
    edge_set = set()
    for f in faces:
        for i, j in [(int(f[0]), int(f[1])), (int(f[1]), int(f[2])), (int(f[0]), int(f[2]))]:
            edge_set.add((min(i, j), max(i, j)))
    if len(edge_set) == 0:
        raise HTTPException(500, "No edges found in mesh")
    edges = np.array(sorted(edge_set), dtype=np.int32)  # [E, 2]

    # Rest lengths from 2D mesh_pos (crds = mesh_pos for cloth)
    # Pad to 3D with z=0 for norm computation
    mesh_pos_3d = np.pad(crds, ((0, 0), (0, 1)), constant_values=0.0)  # [N, 3]
    rest_lens = np.linalg.norm(
        mesh_pos_3d[edges[:, 0]] - mesh_pos_3d[edges[:, 1]], axis=-1
    )  # [E]

    T = predicted.shape[0]
    mean_stretches = []
    max_stretches  = []
    for t_idx in range(T):
        world_lens = np.linalg.norm(
            predicted[t_idx, edges[:, 0]] - predicted[t_idx, edges[:, 1]], axis=-1
        )  # [E]
        stretch = np.abs(world_lens / (rest_lens + 1e-8) - 1.0)
        mean_stretches.append(_safe_float(float(np.mean(stretch))))
        max_stretches.append(_safe_float(float(np.max(stretch))))

    times = [round(i * 0.01, 3) for i in range(T)]
    return {
        "per_step_mean_stretch": mean_stretches,
        "per_step_max_stretch":  max_stretches,
        "times":                 times,
    }
