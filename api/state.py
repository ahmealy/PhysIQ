"""
Shared state for the FastAPI server.
Single module so all routers share the same objects without circular imports.
"""

import os
import subprocess
import threading
import time
from typing import Optional
import torch
from extensions.generative.gnn_scorer import GnnScorer

# ── Training state ────────────────────────────────────────────────────────────
train_process: Optional[subprocess.Popen] = None
# Use absolute path so the log path shown in the UI is copy-pasteable
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
train_log_path: str = os.path.join(_project_root, "runs", "train_ui.log")

# PID file — written when training starts, deleted when it ends.
# Survives uvicorn restarts so we can detect orphaned processes.
_train_pid_file: str = os.path.join(_project_root, "runs", "train_ui.pid")
# Remote PID file written by the nohup & launch on the GPU host (shared NFS).
_train_remote_pid_file: str = os.path.join(_project_root, "runs", "train_remote.pid")
# Launch-time file — stores Unix timestamp (ms) of when training was started.
# Persisted to disk so elapsed time survives server restarts.
_train_start_time_file: str = os.path.join(_project_root, "runs", "train_start_time.txt")


def save_train_pid(pid: int) -> None:
    os.makedirs(os.path.join(_project_root, "runs"), exist_ok=True)
    with open(_train_pid_file, "w") as f:
        f.write(str(pid))


def save_train_start_time(ms: int | None = None) -> None:
    """Persist the training start timestamp (ms since epoch) to disk."""
    os.makedirs(os.path.join(_project_root, "runs"), exist_ok=True)
    ts = ms if ms is not None else int(time.time() * 1000)
    with open(_train_start_time_file, "w") as f:
        f.write(str(ts))


def get_train_start_time() -> int | None:
    """Return the persisted training start timestamp in ms, or None."""
    try:
        return int(open(_train_start_time_file).read().strip())
    except (FileNotFoundError, ValueError):
        return None


def clear_train_pid() -> None:
    for path in (_train_pid_file, _train_remote_pid_file, _train_start_time_file):
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


def get_orphan_pid() -> Optional[int]:
    """Return a truthy sentinel if a training process is still running, or None.

    For local processes: checks the PID file with os.kill(0).
    For remote SSH processes: the remote PID can't be checked with os.kill
    (wrong host), so instead we check:
      1. train_remote.pid file exists (written by nohup launcher)
      2. The training log was modified within the last 30 seconds
         OR the log is actively growing (mtime recent relative to file age)
    If the log hasn't been touched for >120s and remote pid file exists,
    we assume it finished or died and clean up.
    """
    # Check remote PID file first
    if os.path.exists(_train_remote_pid_file):
        try:
            remote_pid = int(open(_train_remote_pid_file).read().strip())
            if remote_pid > 0:
                # Can't os.kill a remote PID — use log freshness instead
                log_age = (
                    time.time() - os.path.getmtime(train_log_path)
                    if os.path.exists(train_log_path) else 9999
                )
                if log_age < 120:
                    # Log updated recently — training is alive
                    return remote_pid
                # Log stale for >120s — training likely finished/died
                clear_train_pid()
                return None
        except (ValueError, OSError):
            clear_train_pid()
            return None

    # Fall back to local SSH/process PID
    if not os.path.exists(_train_pid_file):
        return None
    try:
        pid = int(open(_train_pid_file).read().strip())
        if pid > 0:
            os.kill(pid, 0)
            return pid
    except (ValueError, ProcessLookupError, PermissionError):
        pass
    clear_train_pid()
    return None

# ── Loaded model cache ────────────────────────────────────────────────────────
# Keyed by (checkpoint_path, device) so we reload only when needed
_model_cache: dict = {}
_model_cache_lock = threading.Lock()


def get_model(checkpoint_path: str, device: str):
    """
    Load and cache the correct Simulator based on checkpoint metadata.
    Thread-safe via double-checked locking.
    """
    key = (checkpoint_path, device)
    # Fast path — no lock needed if already cached
    if key in _model_cache:
        return _model_cache[key]
    with _model_cache_lock:
        # Second check inside the lock (another thread may have populated it)
        if key not in _model_cache:
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
            domain = ckpt.get("domain", "cylinder_flow")

            if domain == "flag_simple":
                from model.flag_simulator import FlagSimulator
                sim = FlagSimulator(
                    message_passing_num=15,
                    device=device,
                )
            else:
                from model.simulator import Simulator
                node_input_size = ckpt.get("node_input_size", 11)
                edge_input_size = ckpt.get("edge_input_size", 3)
                target_field = ckpt.get("target_field", "velocity")
                sim = Simulator(
                    message_passing_num=15,
                    node_input_size=node_input_size,
                    edge_input_size=edge_input_size,
                    device=device,
                    target_field=target_field,
                )

            # Filter checkpoint keys to only those that match the current model's
            # name and shape. This handles checkpoints saved with an older/deeper
            # architecture (e.g. 4-layer MLP decoder vs the current 3-layer one).
            # The decoder is not used for rollout (only encoder + processor are),
            # so skipping mismatched decoder keys is safe.
            import logging as _logging
            ckpt_sd    = ckpt["model_state_dict"]
            current_sd = sim.state_dict()
            filtered   = {
                k: v for k, v in ckpt_sd.items()
                if k in current_sd and current_sd[k].shape == v.shape
            }
            skipped = [k for k in ckpt_sd if k not in filtered]
            if skipped:
                _logging.getLogger(__name__).warning(
                    "get_model: skipping %d checkpoint keys with shape/name mismatch "
                    "(likely older deeper architecture): %s%s",
                    len(skipped), skipped[:3], " ..." if len(skipped) > 3 else "",
                )
            sim.load_state_dict(filtered, strict=False)
            sim.eval()
            _model_cache[key] = sim
    return _model_cache[key]


def clear_model_cache():
    """Force reload on next get_model() call (e.g. after training completes)."""
    _model_cache.clear()


# ── GnnScorer cache (Deep mode) ────────────────────────────────────────────
_gnn_scorer_cache: dict[tuple, "GnnScorer"] = {}
_gnn_scorer_lock  = threading.Lock()


def get_gnn_scorer(checkpoint_path: str, device: str) -> "GnnScorer":
    """Lazy-load and cache a GnnScorer. Thread-safe double-checked locking."""
    key = (checkpoint_path, device)
    if key in _gnn_scorer_cache:
        return _gnn_scorer_cache[key]
    with _gnn_scorer_lock:
        if key not in _gnn_scorer_cache:
            _gnn_scorer_cache[key] = GnnScorer(checkpoint_path, device=device)
    return _gnn_scorer_cache[key]


def clear_gnn_scorer_cache() -> None:
    """Clear the GnnScorer cache (used in tests)."""
    _gnn_scorer_cache.clear()


# ── Domain registry ───────────────────────────────────────────────────────────
DOMAINS = {
    "cylinder_flow": {
        "label":         "Cylinder Flow (CFD)",
        "description":   "2D fluid flow past a cylinder — von Kármán vortex street",
        "data_dir":      "data",
        "checkpoint":    "checkpoints/best_model.pth",
        "node_input":    11,
        "edge_input":    3,
        "mp_steps":      15,
        "dt":            0.01,
        "available":     True,
        "target_fields": ["velocity", "pressure"],
    },
    "flag_simple": {
        "label":       "Flag Simple (Cloth)",
        "description": "3D cloth simulation — deformable mesh",
        "data_dir":    "data_flag",
        "checkpoint":  "checkpoints/flag_best_model.pth",
        "node_input":  12,
        "edge_input":  7,
        "mp_steps":    15,
        "dt":          0.01,
        "available":   False,
    },
}


def _probe_flag_available() -> bool:
    """Check if flag_simple data has been parsed and is ready."""
    return os.path.exists("data_flag/train_index.npz")

# Update flag_simple availability at import time
DOMAINS["flag_simple"]["available"] = _probe_flag_available()
