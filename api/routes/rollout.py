"""
/rollout endpoint — SSE stream of autoregressive inference progress.
"""

import asyncio
import datetime
import json
import logging
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

logger = logging.getLogger(__name__)

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


def _ts() -> str:
    """Compact timestamp suffix: YYYYMMDD_HHMMSS"""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


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
    domain:             str  = "cylinder_flow"
    trajectory_index:   int  = 0
    split:              str  = "test"
    device:             str  = "cuda:0"
    checkpoint:         str  = ""    # optional override; empty string → use DOMAINS default
    poisson_correction: bool = False  # Helmholtz projection after each GNN step (CFD only)


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
    _raw_graph = dataset[req.trajectory_index * n_steps]
    faces = _raw_graph.face.numpy().T.astype(np.int32)  # [F, 3]
    predicted_velocity = None
    boundary_mask = None
    predicteds, targets_list = [], []

    # Initialise Poisson corrector once (one-time LU factorisation) before the loop
    corrector = None
    if req.poisson_correction and req.domain == "cylinder_flow":
        from physics.poisson_pressure import PoissonPressureCorrector
        crds_init = _raw_graph.pos.numpy()  # [N, 2]
        corrector = PoissonPressureCorrector(crds_init)
        logger.info("Poisson pressure corrector initialised (LU factorised)")

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

            # GT velocity at t — read BEFORE any overwrite (always from dataset)
            gt_velocity_t = graph.x[:, field_slice].detach().cpu().numpy()  # [N, d]

            if predicted_velocity is not None:
                graph.x[:, field_slice] = predicted_velocity.detach()

            # Autoregressive input velocity at t — GT at t=0, own prediction from t=1
            # This is what DeepMind writes to trajectory[t] in cfd_eval.py
            rollout_velocity_t = graph.x[:, field_slice].detach().cpu().numpy()

            predicted_velocity = model(graph, velocity_sequence_noise=None)

            # Apply Poisson pressure correction before boundary pin (CFD only, opt-in)
            if corrector is not None:
                vel_np = predicted_velocity.detach().cpu().numpy()  # [N, 2]
                vel_np = corrector.correct(vel_np)
                predicted_velocity = torch.from_numpy(vel_np).to(predicted_velocity.device)

            predicted_velocity[boundary_mask] = graph.y[boundary_mask]

            # predicteds[t] = autoregressive input velocity at t  (matches DeepMind trajectory[t])
            # targets[t]    = GT velocity at t                    (matches DeepMind inputs['velocity'][t])
            predicteds.append(rollout_velocity_t)
            targets_list.append(gt_velocity_t)

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
    logger.info("Confidence index path: %s (cwd=%s, exists=%s)", index_path, os.getcwd(), os.path.exists(index_path))
    if os.path.exists(index_path):
        # Warn if index predates the checkpoint — scores may be stale
        try:
            _ckpt = cfg.get("checkpoint", "")
            if _ckpt and os.path.exists(_ckpt):
                if os.path.getmtime(_ckpt) > os.path.getmtime(index_path) + 60:
                    logger.warning(
                        "Confidence index '%s' predates checkpoint '%s' — "
                        "scores may be inaccurate. Rebuild: "
                        "python -m confidence.build_index --checkpoint %s --output %s",
                        index_path, _ckpt, _ckpt, index_path,
                    )
        except Exception:
            pass  # never block rollout
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
            logger.info(
                "Confidence debug — emb_norm=%.4f emb_mean=%.4f "
                "train_diameter=%.4f d_min=%.4f score=%.4f",
                float((_emb ** 2).sum() ** 0.5),
                float(_emb.mean()),
                _index.train_diameter,
                float(1.0 - confidence_score) * _index.train_diameter,
                confidence_score,
            )
        except Exception as _ce:
            logger.warning("Confidence score failed: %s", _ce, exc_info=True)

    # Save pkl
    os.makedirs("result", exist_ok=True)
    pkl_path = "result/result_traj%d_%s.pkl" % (req.trajectory_index, _ts())
    with open(pkl_path, "wb") as f:
        pickle.dump([[predicted_arr, targets_arr], crds, {
            "domain":             req.domain,
            "target_field":       target_field,
            "confidence_score":   confidence_score,
            "faces":              faces,
            "speedup":            round(speedup, 2),
            "elapsed_seconds":    round(elapsed, 3),
            "poisson_correction": req.poisson_correction,
        }], f)

    return {
        "elapsed_seconds":   round(elapsed, 3),
        "speedup":           round(speedup, 2),
        "pkl_path":          pkl_path,
        "rmse_final":        float(per_step_rmse[-1]),
        "similarity_score":  round(similarity_score, 3) if similarity_score is not None else None,
        "confidence_score":  round(confidence_score, 3) if confidence_score is not None else None,
        "poisson_correction": req.poisson_correction,
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

            # Record cur_world BEFORE stepping — matches DeepMind cloth_eval.py:
            #   trajectory[t] = cur_pos (the input at step t)
            #   compared against inputs['world_pos'][t] (GT at same step t)
            # So both predicted and target are world_pos at time t.
            gt_world_t = dataset[idx].world_pos.numpy()   # GT at t, before any rollout overwrite
            predicteds.append(graph.world_pos.detach().cpu().numpy())
            targets_list.append(gt_world_t)

            prev_world = graph.world_pos.clone()
            next_world = model(graph)  # [N, 3]

            # Pin HANDLE nodes to cur_pos (fully autoregressive, matches DeepMind)
            node_type = graph.x[:, 3].long()
            handle_mask = (node_type != NodeType.NORMAL)
            next_world[handle_mask] = graph.world_pos[handle_mask]

            cur_world = next_world

            if i % 20 == 0 or i == n_steps - 1:
                progress_callback(i + 1, n_steps)

    elapsed = time.perf_counter() - t_start
    sim_time = n_steps * cfg.get("dt", 0.01)
    speedup = sim_time / (elapsed + 1e-12)

    predicted_arr = np.stack(predicteds)
    targets_arr   = np.stack(targets_list)

    # Load mesh_pos directly from the traj file — FlagDataset is on-demand and
    # does not cache mesh_pos_list as an attribute.
    _traj_path = os.path.join(dataset._split_dir, f"traj_{req.trajectory_index:05d}.npz")
    mesh_pos = np.load(_traj_path)["mesh_pos"].astype(np.float32)
    cells = np.load(_traj_path)["cells"].astype(np.int32)  # [F, 3]

    sq = np.square(predicted_arr - targets_arr).reshape(n_steps, -1)
    per_step_rmse = np.sqrt(np.mean(sq, axis=1))

    # Confidence score — requires domain-scoped embedding_index built after training
    confidence_score = None
    index_path = os.path.join("runs", "embedding_index_flagsimple.pkl")
    if os.path.exists(index_path):
        # Warn if index predates the checkpoint — scores may be stale
        try:
            _ckpt = cfg.get("checkpoint", "")
            if _ckpt and os.path.exists(_ckpt):
                if os.path.getmtime(_ckpt) > os.path.getmtime(index_path) + 60:
                    logger.warning(
                        "Confidence index '%s' predates checkpoint '%s' — "
                        "scores may be inaccurate. Rebuild: "
                        "python -m confidence.build_index --checkpoint %s --output %s",
                        index_path, _ckpt, _ckpt, index_path,
                    )
        except Exception:
            pass  # never block rollout
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
    pkl_path = "result/flag_result_traj%d_%s.pkl" % (req.trajectory_index, _ts())
    with open(pkl_path, "wb") as f:
        pickle.dump([[predicted_arr, targets_arr], mesh_pos, {
            "domain":            "flag_simple",
            "target_field":      "world_pos",
            "confidence_score":  confidence_score,
            "faces":             cells,
            "speedup":           round(speedup, 2),
            "elapsed_seconds":   round(elapsed, 3),
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


@router.get("/checkpoints")
async def list_checkpoints(domain: str = "cylinder_flow"):
    """List all .pth checkpoint files, annotated with metadata from the checkpoint."""
    import glob
    default = DOMAINS.get(domain, {}).get("checkpoint", "")
    files = sorted(glob.glob("checkpoints/*.pth"))
    result = []
    for path in files:
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            arch = ckpt.get("architecture", "gn")
            epoch = int(ckpt.get("epoch", 0))
            vloss = float(ckpt.get("valid_loss", float("nan")))
            ckpt_domain = ckpt.get("domain", "cylinder_flow")
            target_field = ckpt.get("target_field", "velocity")
            import math
            vloss_safe = round(vloss, 6) if math.isfinite(vloss) else None
            loss_str   = f"{vloss:.4f}" if math.isfinite(vloss) else "n/a"
            tf_label   = f" [{target_field}]" if ckpt_domain == "cylinder_flow" and target_field != "velocity" else ""
            result.append({
                "path":         path,
                "domain":       ckpt_domain,
                "architecture": arch,
                "target_field": target_field,
                "epoch":        epoch,
                "valid_loss":   vloss_safe,
                "is_default":   path == default,
                "label":        f"{arch.upper()}{tf_label} · ep{epoch} · loss={loss_str}",
            })
        except Exception:
            result.append({"path": path, "domain": domain, "architecture": "?",
                           "target_field": "velocity",
                           "epoch": 0, "valid_loss": None, "is_default": path == default,
                           "label": path})
    # Build arch summary: best (lowest valid_loss) physics checkpoint per domain × arch
    arch_summary: dict = {}
    for _dom in ["cylinder_flow", "flag_simple"]:
        arch_summary[_dom] = {}
        for _arch in ["gn", "tns", "sage"]:
            _phys = [
                r for r in result
                if r["domain"] == _dom
                and r["architecture"] == _arch
                and r["valid_loss"] is not None
                and not any(x in r["path"] for x in ["cvae", "surrogate"])
            ]
            arch_summary[_dom][_arch] = (
                min(_phys, key=lambda x: x["valid_loss"]) if _phys else None
            )

    return {"checkpoints": result, "arch_summary": arch_summary}


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

    # Resolve checkpoint: explicit override takes priority over domain default
    checkpoint_path = req.checkpoint if req.checkpoint else cfg["checkpoint"]
    if not os.path.exists(checkpoint_path):
        raise HTTPException(404, "No checkpoint at %s. Train first." % checkpoint_path)
    # Inject into cfg so downstream helpers (_run_rollout_sync) see the right path
    cfg = {**cfg, "checkpoint": checkpoint_path}

    # Track rollout state for frontend reconnect
    _rollout_state.update({
        "running":          True,
        "domain":           req.domain,
        "trajectory_index": req.trajectory_index,
        "step":             0,
        "total":            0,
    })

    # ── Remote GPU path ───────────────────────────────────────────────────────
    from api.routes.train import _remote_ssh_active
    remote_cfg = _load_remote_cfg()
    if _remote_ssh_active(remote_cfg):
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        # Write rollout config JSON for rollout_ssh.py to read
        os.makedirs("runs", exist_ok=True)
        rollout_cfg_path = os.path.join(project_root, "runs", "ui_rollout_config.json")
        with open(rollout_cfg_path, "w") as f:
            json.dump({
                "domain":             req.domain,
                "trajectory_index":   req.trajectory_index,
                "split":              req.split,
                "device":             "cuda:0",  # always use GPU on remote host
                "poisson_correction": req.poisson_correction,
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
            done_payload: dict = {}
            try:
                # Read stdout line-by-line; each SSE line from rollout_ssh.py
                # is already in the "data: {...}\n" format — forward verbatim.
                # Do NOT break early on "done" — drain until EOF so the remote
                # process has fully finished writing the pkl before we scp it.
                while True:
                    line = await loop.run_in_executor(None, proc.stdout.readline)
                    if not line:
                        break
                    line = line.rstrip("\n")
                    if line.startswith("data: "):
                        try:
                            payload = json.loads(line[6:])
                            if payload.get("type") == "done":
                                done_payload = payload
                                # Log confidence debug info from the remote script
                                _dbg = payload.get("_confidence_debug", "")
                                if _dbg:
                                    logger.info("[rollout_ssh confidence] %s", _dbg)
                                # Don't yield yet — copy the file first, then yield done
                                continue
                            elif payload.get("type") == "error":
                                yield line + "\n\n"
                                break
                        except Exception:
                            pass
                        yield line + "\n\n"
                    elif line.startswith("CONFIDENCE_DEBUG") or line.startswith("WARNING") or line.startswith("ERROR"):
                        logger.info("[rollout_ssh] %s", line)

                # Wait for the remote process to fully exit
                rc = await loop.run_in_executor(None, proc.wait)

                if rc != 0 and not done_payload:
                    yield "data: %s\n\n" % json.dumps({
                        "type":    "error",
                        "message": "Remote rollout exited with code %d" % rc,
                    })
                    return

                # ── copy the result file from remote if needed ────────────
                if done_payload:
                    remote_pkl = done_payload.get("pkl_path", "")
                    if remote_pkl:
                        os.makedirs("result", exist_ok=True)
                        project_root_local = os.path.dirname(
                            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                        )
                        local_pkl = os.path.join(project_root_local, remote_pkl)

                        if os.path.exists(local_pkl):
                            # Shared filesystem (NFS) — file already present locally, skip scp
                            pass
                        else:
                            # Non-shared filesystem — scp the file back
                            host     = remote_cfg.get("host", "")
                            port     = str(remote_cfg.get("port", 22))
                            user     = remote_cfg.get("user", "").strip()
                            key_file = remote_cfg.get("key_file", "")
                            scp_cmd  = ["scp", "-P", port,
                                        "-o", "StrictHostKeyChecking=no",
                                        "-o", "BatchMode=yes"]
                            if key_file:
                                scp_cmd += ["-i", key_file]
                            remote_src = ("%s@%s:%s" % (user, host, remote_pkl)
                                          if user else "%s:%s" % (host, remote_pkl))
                            scp_cmd += [remote_src, local_pkl]

                            scp_rc = await loop.run_in_executor(
                                None,
                                lambda: subprocess.run(scp_cmd, capture_output=True).returncode,
                            )
                            if scp_rc != 0:
                                yield "data: %s\n\n" % json.dumps({
                                    "type":    "error",
                                    "message": "Rollout finished on remote but scp of result failed (rc=%d). "
                                               "File is at %s on the remote host." % (scp_rc, remote_pkl),
                                })
                                return

                        # Update pkl_path to local absolute path before yielding done
                        done_payload["pkl_path"] = os.path.relpath(local_pkl)

                    yield "data: %s\n\n" % json.dumps(done_payload)

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
