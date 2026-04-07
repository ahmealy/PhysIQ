"""
/rollout endpoint — SSE stream of autoregressive inference progress.
"""

import asyncio
import json
import os
import pickle
import sys
import time
from typing import AsyncGenerator

import numpy as np
import torch
import torch_geometric.transforms as T
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from api.state import DOMAINS, get_model
from dataset import FpcDataset
from utils.utils import NodeType

router = APIRouter()

# Cache dataset objects so we don't reload on every rollout request
_dataset_cache: dict = {}


def _get_dataset(data_dir: str, split: str) -> FpcDataset:
    key = (data_dir, split)
    if key not in _dataset_cache:
        _dataset_cache[key] = FpcDataset(data_dir, split=split)
    return _dataset_cache[key]


class RolloutRequest(BaseModel):
    domain:            str = "cylinder_flow"
    trajectory_index:  int = 0
    split:             str = "test"
    device:            str = "cuda:0"


def _run_rollout_sync(req: RolloutRequest, cfg: dict, device: str,
                      progress_callback) -> dict:
    """
    Blocking inference loop — called from a thread via run_in_executor
    so the event loop stays free during the entire rollout.

    progress_callback(step, total) is called every 20 steps and at the end.
    It puts a message onto the asyncio queue using loop.call_soon_threadsafe.
    """
    model = get_model(cfg["checkpoint"], device)
    dataset = _get_dataset(cfg["data_dir"], req.split)

    n_traj = len(dataset) // dataset.num_sampes_per_tra
    if req.trajectory_index >= n_traj:
        raise ValueError("trajectory_index %d out of range (0-%d)" % (
            req.trajectory_index, n_traj - 1))

    transformer = T.Compose([
        T.FaceToEdge(),
        T.Cartesian(norm=False),
        T.Distance(norm=False),
    ])

    n_steps = dataset.num_sampes_per_tra
    predicted_velocity = None
    boundary_mask = None
    predicteds, targets_list = [], []

    t_start = time.perf_counter()

    with torch.no_grad():
        for i in range(n_steps):
            idx = req.trajectory_index * n_steps + i
            graph = dataset[idx]
            graph = transformer(graph)
            graph = graph.to(device)

            if boundary_mask is None:
                node_type = graph.x[:, 0]
                fluid = torch.logical_or(
                    node_type == NodeType.NORMAL,
                    node_type == NodeType.OUTFLOW,
                )
                boundary_mask = torch.logical_not(fluid)

            if predicted_velocity is not None:
                graph.x[:, 1:3] = predicted_velocity.detach()

            next_v = graph.y
            predicted_velocity = model(graph, velocity_sequence_noise=None)
            predicted_velocity[boundary_mask] = next_v[boundary_mask]

            predicteds.append(predicted_velocity.detach().cpu().numpy())
            targets_list.append(next_v.detach().cpu().numpy())

            # Report progress every 20 steps
            if i % 20 == 0 or i == n_steps - 1:
                progress_callback(i + 1, n_steps)

    elapsed = time.perf_counter() - t_start
    sim_time = n_steps * cfg["dt"]
    speedup = sim_time / elapsed

    predicted_arr = np.stack(predicteds)   # [T, N, 2]
    targets_arr   = np.stack(targets_list)  # [T, N, 2]
    crds          = graph.pos.cpu().numpy() # [N, 2]

    sq_diff       = np.square(predicted_arr - targets_arr).reshape(n_steps, -1)
    per_step_rmse = np.sqrt(np.mean(sq_diff, axis=1))

    # Similarity score vs existing rollouts
    query_mean_v = float(np.linalg.norm(predicted_arr, axis=-1).mean())
    query_n      = predicted_arr.shape[1]
    query_feat   = np.array([query_mean_v, query_n / 1000.0])

    similarity_score = None
    ref_feats = []
    result_dir = "result"
    if os.path.exists(result_dir):
        for fname in os.listdir(result_dir):
            if not fname.endswith(".pkl"):
                continue
            ref_path = os.path.join(result_dir, fname)
            try:
                with open(ref_path, "rb") as f:
                    ref_data = pickle.load(f)
                ref_pred = ref_data[0][0]  # [T, N, 2]
                ref_feats.append(np.array([
                    float(np.linalg.norm(ref_pred, axis=-1).mean()),
                    ref_pred.shape[1] / 1000.0,
                ]))
            except Exception:
                pass

    if ref_feats:
        ref_arr = np.array(ref_feats)
        dists   = np.linalg.norm(ref_arr - query_feat, axis=1)
        d_min   = float(dists.min())
        d_max   = float(np.linalg.norm(
            ref_arr.max(axis=0) - ref_arr.min(axis=0))) + 1e-12
        similarity_score = float(np.clip(1.0 - d_min / d_max, -0.5, 1.0))

    # Save pkl
    os.makedirs("result", exist_ok=True)
    pkl_path = "result/result%d.pkl" % req.trajectory_index
    with open(pkl_path, "wb") as f:
        pickle.dump([[predicted_arr, targets_arr], crds], f)

    return {
        "elapsed_seconds":  round(elapsed, 3),
        "speedup":          round(speedup, 2),
        "pkl_path":         pkl_path,
        "rmse_final":       float(per_step_rmse[-1]),
        "similarity_score": round(similarity_score, 3) if similarity_score is not None else None,
    }


@router.post("/rollout")
async def run_rollout(req: RolloutRequest):
    """
    Runs autoregressive rollout and streams progress via SSE.

    The heavy GNN inference loop runs in a thread (via run_in_executor)
    so the asyncio event loop stays responsive during the entire rollout.
    Progress events are passed back to the async generator via an asyncio.Queue.
    """
    if req.domain not in DOMAINS:
        raise HTTPException(404, "Unknown domain: %s" % req.domain)

    cfg = DOMAINS[req.domain]
    if not cfg["available"]:
        raise HTTPException(400, "Domain '%s' not available" % req.domain)

    if not os.path.exists(cfg["checkpoint"]):
        raise HTTPException(404, "No checkpoint at %s. Train first." % cfg["checkpoint"])

    # Validate device
    device = req.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    # Queue bridges the sync thread and the async SSE generator
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def progress_callback(step: int, total: int):
        """Called from the worker thread — puts progress onto the async queue."""
        loop.call_soon_threadsafe(queue.put_nowait, {
            "type":  "progress",
            "step":  step,
            "total": total,
        })

    async def generate() -> AsyncGenerator[str, None]:
        try:
            # Kick off the blocking inference in a thread
            future = loop.run_in_executor(
                None,
                _run_rollout_sync,
                req, cfg, device, progress_callback,
            )

            # Drain progress events while the thread is running
            while not future.done():
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=1.0)
                    yield "data: %s\n\n" % json.dumps(msg)
                except asyncio.TimeoutError:
                    pass   # nothing in queue yet, loop again

            # Flush any remaining messages (e.g. the final progress event)
            while not queue.empty():
                msg = queue.get_nowait()
                yield "data: %s\n\n" % json.dumps(msg)

            # Get the final result (raises if the thread raised)
            result = await future
            yield "data: %s\n\n" % json.dumps({"type": "done", **result})

        except Exception as e:
            yield "data: %s\n\n" % json.dumps({
                "type":    "error",
                "message": str(e),
            })

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
