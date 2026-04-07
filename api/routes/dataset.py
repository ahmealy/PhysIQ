"""
/dataset/info and /dataset/samples endpoints.
"""

import asyncio
import os
import sys

import numpy as np
from fastapi import APIRouter, HTTPException

from api.state import DOMAINS

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

router = APIRouter(prefix="/dataset")

# Cache computed samples result so repeated page loads are instant
_samples_cache: dict = {}


@router.get("/info")
def dataset_info(domain: str = "cylinder_flow", split: str = "test"):
    if domain not in DOMAINS:
        raise HTTPException(404, "Unknown domain: %s" % domain)

    cfg = DOMAINS[domain]
    if not cfg["available"]:
        raise HTTPException(400, "Domain '%s' is not available yet" % domain)

    data_dir = cfg["data_dir"]
    npz_path = os.path.join(data_dir, "%s.npz" % split)
    if not os.path.exists(npz_path):
        raise HTTPException(404, "Dataset not found: %s" % npz_path)

    meta = np.load(npz_path, allow_pickle=True)
    indices = meta["indices"]
    n_trajectories = len(indices) - 1
    velocity_shape = meta["all_velocity_shape"]
    timesteps = int(velocity_shape[1])

    return {
        "domain":                   domain,
        "split":                    split,
        "num_trajectories":         n_trajectories,
        "timesteps_per_trajectory": timesteps,
        "dt":                       cfg["dt"],
        "total_samples":            n_trajectories * (timesteps - 1),
    }


def _compute_samples(npz_path: str, dat_path: str) -> dict:
    """CPU/IO-heavy computation — always called via run_in_executor."""
    meta = np.load(npz_path, allow_pickle=True)
    indices = meta["indices"]
    n_trajectories = len(indices) - 1
    shape = tuple(meta["all_velocity_shape"])

    all_velocity = np.memmap(dat_path, dtype="float32", mode="r", shape=shape)
    vel_t0_all = np.array(all_velocity[:, 0, :])        # single contiguous memmap read
    vel_mag_all = np.linalg.norm(vel_t0_all, axis=-1)

    # Per-trajectory stats
    traj_means = np.array([
        float(vel_mag_all[int(indices[i]):int(indices[i + 1])].mean())
        for i in range(n_trajectories)
    ])
    # Node count per trajectory
    node_counts = np.array([
        int(indices[i + 1]) - int(indices[i])
        for i in range(n_trajectories)
    ])

    sample_size = min(20, n_trajectories)
    sampled = np.linspace(0, n_trajectories - 1, sample_size, dtype=int)
    sample_nodes = np.concatenate([
        vel_mag_all[int(indices[i]):int(indices[i + 1])] for i in sampled
    ])
    energy_vals = 0.5 * sample_nodes ** 2

    v_counts, v_edges = np.histogram(sample_nodes, bins=50)
    velocity_bins = [{"bin": round(float(v_edges[i]), 6), "count": int(v_counts[i])} for i in range(len(v_counts))]

    e_counts, e_edges = np.histogram(energy_vals, bins=50)
    energy_bins = [{"bin": round(float(e_edges[i]), 6), "count": int(e_counts[i])} for i in range(len(e_counts))]

    nc_counts, nc_edges = np.histogram(node_counts, bins=min(30, len(set(node_counts.tolist()))))
    node_count_bins = [{"bin": int(nc_edges[i]), "count": int(nc_counts[i])} for i in range(len(nc_counts))]

    mean_v = float(traj_means.mean())
    std_v  = float(traj_means.std()) if traj_means.std() > 0 else 1.0
    z_scores = (traj_means - mean_v) / std_v

    outliers = [
        {"trajectory": i, "mean_v": round(float(traj_means[i]), 6),
         "z_score": round(float(z_scores[i]), 3), "flag": bool(abs(z_scores[i]) > 3.0)}
        for i in range(n_trajectories)
    ]
    flagged = [o for o in outliers if o["flag"]]
    unflagged_sample = [o for o in outliers if not o["flag"]][:max(0, 50 - len(flagged))]
    outlier_table = sorted(flagged + unflagged_sample, key=lambda x: abs(x["z_score"]), reverse=True)

    return {
        "velocity_bins":   velocity_bins,
        "energy_bins":     energy_bins,
        "node_count_bins": node_count_bins,
        "outliers":        outlier_table,
        "n_trajectories":  n_trajectories,
        "total_nodes":     int(node_counts.sum()),
        "mean_nodes":      round(float(node_counts.mean()), 1),
    }


@router.get("/samples")
async def dataset_samples(domain: str = "cylinder_flow", split: str = "test"):
    """Return velocity/energy histograms and Z-score outlier table.
    Heavy numpy work runs in a thread so the event loop stays responsive."""
    cache_key = (domain, split)
    if cache_key in _samples_cache:
        return _samples_cache[cache_key]

    if domain not in DOMAINS:
        raise HTTPException(404, "Unknown domain: %s" % domain)

    cfg = DOMAINS[domain]
    if not cfg["available"]:
        raise HTTPException(400, "Domain '%s' is not available yet" % domain)

    data_dir = cfg["data_dir"]
    npz_path = os.path.join(data_dir, "%s.npz" % split)
    if not os.path.exists(npz_path):
        raise HTTPException(404, "Dataset not found: %s" % npz_path)

    dat_path = npz_path.replace(".npz", ".dat")
    if not os.path.exists(dat_path):
        raise HTTPException(404, "Velocity data file not found: %s" % dat_path)

    result = await asyncio.get_event_loop().run_in_executor(
        None, _compute_samples, npz_path, dat_path
    )
    _samples_cache[cache_key] = result
    return result


@router.post("/flag_outliers")
async def flag_outliers(domain: str = "cylinder_flow", split: str = "test"):
    """
    Write result/outlier_mask_{split}.npy — a boolean array [n_trajectories]
    where True = flagged as outlier (|z-score| > 3σ).
    Clears the samples cache so next load reflects any data changes.
    """
    cache_key = (domain, split)
    if cache_key not in _samples_cache:
        # Need to compute first
        if domain not in DOMAINS:
            raise HTTPException(404, "Unknown domain: %s" % domain)
        cfg = DOMAINS[domain]
        data_dir = cfg["data_dir"]
        npz_path = os.path.join(data_dir, "%s.npz" % split)
        dat_path = npz_path.replace(".npz", ".dat")
        result = await asyncio.get_event_loop().run_in_executor(
            None, _compute_samples, npz_path, dat_path
        )
        _samples_cache[cache_key] = result

    data = _samples_cache[cache_key]
    outliers = data["outliers"]
    n = data["n_trajectories"]

    mask = np.zeros(n, dtype=bool)
    for o in outliers:
        if o["flag"]:
            mask[o["trajectory"]] = True

    os.makedirs("result", exist_ok=True)
    mask_path = os.path.join("result", f"outlier_mask_{split}.npy")
    np.save(mask_path, mask)

    flagged_ids = [o["trajectory"] for o in outliers if o["flag"]]
    return {
        "status":      "saved",
        "path":        mask_path,
        "n_flagged":   int(mask.sum()),
        "flagged_ids": flagged_ids,
    }
