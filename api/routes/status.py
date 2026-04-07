"""
/status, /checkpoint, and /events endpoints.
"""

import asyncio
import glob as _glob
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from typing import Optional

import torch
from fastapi import APIRouter, HTTPException

import api.state as state
from api.state import DOMAINS, get_orphan_pid

router = APIRouter()

_checkpoint_cache: dict = {}   # path → {info, mtime}
# Clear on import so a server restart always re-reads checkpoints with current schema


def _checkpoint_info_sync(path: str) -> Optional[dict]:
    """Synchronous checkpoint loader — call via run_in_executor to avoid blocking."""
    if not os.path.exists(path):
        return None
    stat = os.stat(path)
    mtime = stat.st_mtime
    if path in _checkpoint_cache and _checkpoint_cache[path]["mtime"] == mtime:
        return _checkpoint_cache[path]["info"]
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    # Count trainable parameters from state dict
    param_count = sum(v.numel() for v in ckpt.get("model_state_dict", {}).values())
    info = {
        "epoch":        ckpt.get("epoch"),
        "valid_loss":   ckpt.get("valid_loss"),
        "size_mb":      round(stat.st_size / 1e6, 2),
        "param_count":  param_count,
        "param_count_m": round(param_count / 1e6, 2),
        "path":         path,
        "last_modified": datetime.fromtimestamp(
            mtime, tz=timezone.utc
        ).isoformat(),
    }
    _checkpoint_cache[path] = {"mtime": mtime, "info": info}
    return info


@router.get("/status")
async def get_status():
    def _gather_status_sync() -> dict:
        """All blocking work in one executor call — keeps event loop free."""
        saved = 0
        if os.path.exists("result"):
            saved = len([f for f in os.listdir("result") if f.endswith(".pkl")])

        gpu_available = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if gpu_available else None

        # Skip checkpoint load while training is actively running to avoid
        # blocking on a file that train.py is currently writing.
        # Check both the Popen handle and any orphaned process from a previous server session
        is_running = (
            (state.train_process is not None and state.train_process.poll() is None)
            or get_orphan_pid() is not None
        )
        ckpt = None
        if not is_running:
            default_ckpt = DOMAINS["cylinder_flow"]["checkpoint"]
            ckpt = _checkpoint_info_sync(default_ckpt)

        return {
            "checkpoint_exists":     ckpt is not None,
            "checkpoint_epoch":      ckpt["epoch"]      if ckpt else None,
            "checkpoint_valid_loss": ckpt["valid_loss"]  if ckpt else None,
            "checkpoint_size_mb":    ckpt["size_mb"]    if ckpt else None,
            "gpu_available":         gpu_available,
            "gpu_name":              gpu_name,
            "training_running":      is_running,
            "saved_rollouts":        saved,
        }

    result = await asyncio.get_running_loop().run_in_executor(None, _gather_status_sync)

    # Domains are static config — no I/O needed, safe to build on event loop
    result["domains"] = {
        k: {
            "label":       v["label"],
            "description": v["description"],
            "available":   v["available"],
        }
        for k, v in DOMAINS.items()
    }
    return result


@router.get("/status/gpu")
def get_gpu_status():
    """Return current GPU memory usage and utilization."""
    gpu_available = torch.cuda.is_available()

    mem_alloc_gb = None
    mem_reserved_gb = None
    utilization = None

    if gpu_available:
        mem_alloc_gb = round(torch.cuda.memory_allocated() / 1e9, 3)
        mem_reserved_gb = round(torch.cuda.memory_reserved() / 1e9, 3)

        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0:
                utilization = int(result.stdout.strip().split("\n")[0])
        except Exception:
            pass

    return {
        "gpu_available": gpu_available,
        "mem_alloc_gb": mem_alloc_gb,
        "mem_reserved_gb": mem_reserved_gb,
        "utilization": utilization,
    }


@router.get("/checkpoint")
async def get_checkpoint(domain: str = "cylinder_flow"):
    if domain not in DOMAINS:
        raise HTTPException(404, f"Unknown domain: {domain}")
    path = DOMAINS[domain]["checkpoint"]
    info = await asyncio.get_running_loop().run_in_executor(
        None, _checkpoint_info_sync, path
    )
    if info is None:
        raise HTTPException(404, f"No checkpoint found at {path}")
    return info


def _relative_time(ts: float) -> str:
    """Human-readable relative time string from a Unix timestamp."""
    delta = datetime.now(tz=timezone.utc).timestamp() - ts
    if delta < 60:
        return "just now"
    if delta < 3600:
        return f"{int(delta // 60)}m ago"
    if delta < 86400:
        return f"{int(delta // 3600)}h ago"
    return f"{int(delta // 86400)}d ago"


@router.get("/events")
def get_events():
    """Return real system events derived from log files and checkpoint/rollout filesystem state."""
    events: list[dict] = []

    # ── 1. Training log events (last 10 epochs + errors) ─────────────────────
    log_path = "runs/train_ui.log"
    if os.path.exists(log_path):
        stat = os.stat(log_path)
        log_mtime = stat.st_mtime
        try:
            with open(log_path, "r") as f:
                lines = f.readlines()[-200:]  # tail last 200 lines
            epoch_lines = [l for l in lines if "Epoch" in l and "valid" in l.lower()]
            if epoch_lines:
                last_epoch_line = epoch_lines[-1].strip()
                events.append({
                    "type": "info",
                    "message": f"Training log: {last_epoch_line[:80]}",
                    "time": _relative_time(log_mtime),
                })
            # Detect OOM / CUDA errors in log
            error_lines = [l for l in lines if any(k in l for k in ["Error", "OOM", "CUDA error", "Traceback"])]
            if error_lines:
                last_err = error_lines[-1].strip()
                events.append({
                    "type": "error",
                    "message": f"Training error: {last_err[:80]}",
                    "time": _relative_time(log_mtime),
                })
        except Exception:
            pass

    # ── 2. Checkpoint saved events ────────────────────────────────────────────
    ckpt_dir = "checkpoints"
    if os.path.isdir(ckpt_dir):
        ckpt_files = sorted(
            _glob.glob(os.path.join(ckpt_dir, "*.pth")),
            key=os.path.getmtime, reverse=True
        )
        for ckpt_path in ckpt_files[:3]:
            mtime = os.path.getmtime(ckpt_path)
            fname = os.path.basename(ckpt_path)
            events.append({
                "type": "info",
                "message": f"Checkpoint saved: {fname}",
                "time": _relative_time(mtime),
            })

    # ── 3. Rollout saved events ───────────────────────────────────────────────
    result_dir = "result"
    if os.path.isdir(result_dir):
        pkl_files = sorted(
            _glob.glob(os.path.join(result_dir, "*.pkl")),
            key=os.path.getmtime, reverse=True
        )
        for pkl_path in pkl_files[:2]:
            mtime = os.path.getmtime(pkl_path)
            fname = os.path.basename(pkl_path)
            events.append({
                "type": "info",
                "message": f"Rollout saved: {fname}",
                "time": _relative_time(mtime),
            })

    # ── 4. GPU utilization warning (live) ────────────────────────────────────
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(",")
            if len(parts) >= 1:
                temp = int(parts[0].strip())
                if temp >= 80:
                    events.append({
                        "type": "warning",
                        "message": f"High GPU temperature: {temp}°C",
                        "time": "now",
                    })
    except Exception:
        pass

    # Sort by recency: "just now" / "Xm ago" < "Xh ago" < "Xd ago"
    def sort_key(ev: dict) -> int:
        t = ev["time"]
        if t == "now" or t == "just now":
            return 0
        m = re.match(r"(\d+)([mhd]) ago", t)
        if not m:
            return 9999
        v, unit = int(m.group(1)), m.group(2)
        return v * {"m": 1, "h": 60, "d": 1440}[unit]

    events.sort(key=sort_key)
    return events[:8]


@router.get("/pipeline")
def get_pipeline_status():
    """
    Return filesystem-probed status for each pipeline node.
    Used by the Pipeline View DAG.
    """
    import glob as _glob

    def _probe(paths: list[str]) -> bool:
        return any(
            (_glob.glob(p) if "*" in p else os.path.exists(p))
            for p in paths
        )

    def _file_details(pattern: str) -> list[dict]:
        files = _glob.glob(pattern) if "*" in pattern else ([pattern] if os.path.exists(pattern) else [])
        out = []
        for f in sorted(files)[:10]:
            stat = os.stat(f)
            out.append({
                "name":     os.path.basename(f),
                "size_mb":  round(stat.st_size / 1e6, 2),
                "modified": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            })
        return out

    # Load train config if it exists
    train_config = {}
    try:
        if os.path.exists("runs/ui_train_config.json"):
            with open("runs/ui_train_config.json") as f:
                train_config = json.load(f)
    except Exception:
        pass

    # Load checkpoint info
    ckpt_info = {}
    ckpt_path = DOMAINS["cylinder_flow"]["checkpoint"]
    if os.path.exists(ckpt_path):
        try:
            import torch
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            ckpt_info = {
                "epoch":      ckpt.get("epoch"),
                "valid_loss": ckpt.get("valid_loss"),
            }
        except Exception:
            pass

    nodes = [
        {
            "id":     "dataset",
            "label":  "Dataset",
            "done":   _probe(["data/*.npz", "data/test.npz", "data/train.npz"]),
            "files":  _file_details("data/*.npz"),
            "config": {},
        },
        {
            "id":     "preprocess",
            "label":  "Preprocess",
            "done":   _probe(["data/*.dat", "data/test.dat", "data/train.dat"]),
            "files":  _file_details("data/*.dat"),
            "config": {},
        },
        {
            "id":     "graph_build",
            "label":  "Graph Build",
            "done":   _probe(["data/*.npz"]),   # graph is built on-the-fly from npz
            "files":  [],
            "config": {"note": "Built on-the-fly from .npz at training time"},
        },
        {
            "id":     "train",
            "label":  "Train",
            "done":   _probe([DOMAINS["cylinder_flow"]["checkpoint"]]),
            "files":  _file_details("checkpoints/*.pth"),
            "config": {**train_config, **ckpt_info},
        },
        {
            "id":     "evaluate",
            "label":  "Evaluate",
            "done":   _probe(["result/*.pkl"]),
            "files":  _file_details("result/*.pkl"),
            "config": {},
        },
        {
            "id":     "predict",
            "label":  "Predict",
            "done":   _probe(["result/*.pkl"]),
            "files":  _file_details("result/*.pkl"),
            "config": {},
        },
        {
            "id":     "export",
            "label":  "Export",
            "done":   _probe(["videos/*.mp4"]),
            "files":  _file_details("videos/*.mp4"),
            "config": {},
        },
    ]

    return {"nodes": nodes}
