"""
/rollout endpoint — SSE stream of autoregressive inference progress.
"""

import asyncio
import json
import os
import pickle
import shlex
import subprocess
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
from api.routes.train import _load_remote_cfg, _build_ssh_prefix
from dataset import FpcDataset
from utils.utils import NodeType

router = APIRouter()

# ── In-progress rollout state (for frontend reconnect on page refresh) ────────
_rollout_state: dict = {
    "running":           False,
    "domain":            None,
    "trajectory_index":  None,
    "step":              0,
    "total":             0,
}

# Cache dataset objects so we don't reload on every rollout request
_dataset_cache: dict = {}


def _get_dataset(data_dir: str, split: str) -> FpcDataset:
    key = (data_dir, split)
    if key not in _dataset_cache:
        _dataset_cache[key] = FpcDataset(data_dir, split=split)
    return _dataset_cache[key]


def _get_cloth_dataset(data_dir: str, split: str):
    """Load and cache FlagDataset."""
    from dataset.flag_dataset import FlagDataset
    key = (data_dir, split, "cloth")
    if key not in _dataset_cache:
        _dataset_cache[key] = FlagDataset(data_dir, split=split)
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

    # Read target_field from the already-loaded model (authoritative, avoids second torch.load)
    target_field = getattr(model, "target_field", "velocity")
    field_slice = slice(1, 2) if target_field == "pressure" else slice(1, 3)

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
                graph.x[:, field_slice] = predicted_velocity.detach()

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

    # Confidence score — requires domain-scoped embedding_index built after training
    confidence_score = None
    _domain_slug = req.domain.replace("_", "")  # cylinderflow, flagsimple
    index_path = os.path.join("runs", f"embedding_index_{_domain_slug}.pkl")
    if os.path.exists(index_path):
        try:
            from confidence.index import NearestNeighborIndex
            from model.embedding import extract_embedding

            _index = NearestNeighborIndex.load(index_path)

            # Use first graph of the rollout trajectory for the embedding
            # Apply same transform used at index-build time (transformer is already in scope)
            _first_graph = dataset[req.trajectory_index * n_steps]
            if transformer is not None:
                _first_graph = transformer(_first_graph)
            _emb = extract_embedding(model, _first_graph, device=device)
            confidence_score = float(_index.query(_emb))
        except Exception:
            pass  # confidence is optional — never block the rollout

    # Save pkl
    os.makedirs("result", exist_ok=True)
    pkl_path = "result/result%d.pkl" % req.trajectory_index
    with open(pkl_path, "wb") as f:
        pickle.dump([[predicted_arr, targets_arr], crds, {
            "domain":           req.domain,
            "target_field":     target_field,
            "confidence_score": confidence_score,
        }], f)

    return {
        "elapsed_seconds":  round(elapsed, 3),
        "speedup":          round(speedup, 2),
        "pkl_path":         pkl_path,
        "rmse_final":       float(per_step_rmse[-1]),
        "similarity_score": round(similarity_score, 3) if similarity_score is not None else None,
        "confidence_score": round(confidence_score, 3) if confidence_score is not None else None,
    }


def _run_cloth_rollout_sync(req, cfg: dict, device: str, progress_callback) -> dict:
    """Cloth (flag_simple) rollout using Verlet integration."""
    from utils.utils import NodeType
    from api.state import get_model

    model = get_model(cfg["checkpoint"], device)
    dataset = _get_cloth_dataset(cfg.get("data_dir", "data_flag"), req.split)

    n_traj = dataset.n_traj
    if req.trajectory_index >= n_traj:
        raise ValueError("trajectory_index %d out of range (0-%d)" % (
            req.trajectory_index, n_traj - 1))

    n_steps = dataset.steps_per_traj[req.trajectory_index]
    predicteds, targets_list = [], []
    prev_world = None
    cur_world  = None

    t_start = time.perf_counter()

    with torch.no_grad():
        for i in range(n_steps):
            idx = int(dataset._cum_steps[req.trajectory_index]) + i
            graph = dataset[idx]
            graph = graph.to(device)

            if cur_world is not None:
                graph.world_pos = cur_world.detach()
                graph.x = torch.cat([cur_world.detach(), graph.x[:, 3:]], dim=-1)
                graph.prev_x    = prev_world.detach()

            prev_world = graph.world_pos.clone()
            next_world = model(graph)  # [N, 3]

            node_type = graph.x[:, 3].long()
            handle_mask = (node_type == NodeType.HANDLE)
            next_world[handle_mask] = graph.y[handle_mask]

            predicteds.append(next_world.detach().cpu().numpy())
            targets_list.append(graph.y.detach().cpu().numpy())
            cur_world = next_world

            if i % 20 == 0 or i == n_steps - 1:
                progress_callback(i + 1, n_steps)

    elapsed = time.perf_counter() - t_start
    sim_time = n_steps * cfg.get("dt", 0.01)
    speedup = sim_time / (elapsed + 1e-12)

    predicted_arr = np.stack(predicteds)
    targets_arr   = np.stack(targets_list)
    mesh_pos = np.asarray(dataset.mesh_pos_list[req.trajectory_index], dtype=np.float32)

    sq = np.square(predicted_arr - targets_arr).reshape(n_steps, -1)
    per_step_rmse = np.sqrt(np.mean(sq, axis=1))

    # Confidence score — requires domain-scoped embedding_index built after training
    confidence_score = None
    index_path = os.path.join("runs", "embedding_index_flagsimple.pkl")
    if os.path.exists(index_path):
        try:
            from confidence.index import NearestNeighborIndex
            from model.embedding import extract_embedding

            _index = NearestNeighborIndex.load(index_path)

            # Use first graph of the rollout trajectory for the embedding
            # Cloth (flag_simple) uses no transform — embedding index was built the same way
            _first_idx = int(dataset._cum_steps[req.trajectory_index])
            _first_graph = dataset[_first_idx]
            _emb = extract_embedding(model, _first_graph, device=device)
            confidence_score = float(_index.query(_emb))
        except Exception:
            pass  # confidence is optional — never block the rollout

    os.makedirs("result", exist_ok=True)
    pkl_path = "result/flag_result%d.pkl" % req.trajectory_index
    with open(pkl_path, "wb") as f:
        pickle.dump([[predicted_arr, targets_arr], mesh_pos, {
            "domain":           "flag_simple",
            "target_field":     "velocity",
            "confidence_score": confidence_score,
        }], f)

    return {
        "elapsed_seconds": round(elapsed, 3),
        "speedup":         round(speedup, 2),
        "pkl_path":        pkl_path,
        "rmse_final":      float(per_step_rmse[-1]),
        "similarity_score": None,
        "confidence_score": round(confidence_score, 3) if confidence_score is not None else None,
    }


@router.get("/rollout/status")
async def get_rollout_status():
    """Return current rollout state for frontend reconnect on page refresh."""
    return _rollout_state.copy()


@router.post("/rollout")
async def run_rollout(req: RolloutRequest):
    """
    Runs autoregressive rollout and streams progress via SSE.

    If a remote GPU SSH config is saved (runs/remote_gpu.json), the rollout
    is executed on the remote host via SSH using rollout_ssh.py.
    Otherwise it runs locally in a thread via run_in_executor.

    Progress events are streamed back to the browser as SSE events.
    """
    if req.domain not in DOMAINS:
        raise HTTPException(404, "Unknown domain: %s" % req.domain)

    cfg = DOMAINS[req.domain]
    if not cfg["available"]:
        raise HTTPException(400, "Domain '%s' not available" % req.domain)

    if not os.path.exists(cfg["checkpoint"]):
        raise HTTPException(404, "No checkpoint at %s. Train first." % cfg["checkpoint"])

    # Track rollout state for frontend reconnect
    _rollout_state.update({
        "running":          True,
        "domain":           req.domain,
        "trajectory_index": req.trajectory_index,
        "step":             0,
        "total":            0,
    })

    # ── Remote GPU path ───────────────────────────────────────────────────────
    remote_cfg = _load_remote_cfg()
    if remote_cfg:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        # Write rollout config JSON for rollout_ssh.py to read
        os.makedirs("runs", exist_ok=True)
        rollout_cfg_path = os.path.join(project_root, "runs", "ui_rollout_config.json")
        with open(rollout_cfg_path, "w") as f:
            json.dump({
                "domain":            req.domain,
                "trajectory_index":  req.trajectory_index,
                "split":             req.split,
                "device":            "cuda:0",  # always use GPU on remote host
            }, f)

        venv_py    = remote_cfg.get("venv_python", "/home/ahmealy/.pyenv/versions/venv_gpu/bin/python").strip()
        ssh_prefix = _build_ssh_prefix(remote_cfg)
        script_path = os.path.join(project_root, "rollout_ssh.py")
        remote_cmd  = (
            "cd %s && %s -u %s --config %s"
            % (
                shlex.quote(project_root),
                shlex.quote(venv_py),
                shlex.quote(script_path),
                shlex.quote(rollout_cfg_path),
            )
        )
        cmd = ssh_prefix + [remote_cmd]

        async def generate_ssh() -> AsyncGenerator[str, None]:
            loop = asyncio.get_running_loop()
            proc = await loop.run_in_executor(
                None,
                lambda: subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
            )
            try:
                # Read stdout line-by-line; each SSE line from rollout_ssh.py
                # is already in the "data: {...}\n" format — forward verbatim.
                while True:
                    line = await loop.run_in_executor(None, proc.stdout.readline)
                    if not line:
                        break
                    line = line.rstrip("\n")
                    if line.startswith("data: "):
                        yield line + "\n\n"
                        # Check if this was the terminal event
                        try:
                            payload = json.loads(line[6:])
                            if payload.get("type") in ("done", "error"):
                                break
                        except Exception:
                            pass
                # If process exited without a done/error event, emit error
                rc = await loop.run_in_executor(None, proc.wait)
                if rc != 0:
                    yield "data: %s\n\n" % json.dumps({
                        "type":    "error",
                        "message": "Remote rollout exited with code %d" % rc,
                    })
            except Exception as e:
                yield "data: %s\n\n" % json.dumps({"type": "error", "message": str(e)})
            finally:
                proc.stdout.close()
                proc.wait()
                _rollout_state["running"] = False

        return StreamingResponse(
            generate_ssh(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ── Local execution path ──────────────────────────────────────────────────
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
        _rollout_state.update({"running": True, "step": step, "total": total})

    async def generate() -> AsyncGenerator[str, None]:
        try:
            # Kick off the blocking inference in a thread
            run_fn = _run_cloth_rollout_sync if req.domain == 'flag_simple' else _run_rollout_sync
            future = loop.run_in_executor(
                None,
                run_fn,
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
            _rollout_state["running"] = False

        except Exception as e:
            yield "data: %s\n\n" % json.dumps({
                "type":    "error",
                "message": str(e),
            })
            _rollout_state["running"] = False

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
