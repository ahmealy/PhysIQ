"""
Shared state for the FastAPI server.
Single module so all routers share the same objects without circular imports.
"""

import os
import subprocess
import threading
from typing import Optional
import torch

# ── Training state ────────────────────────────────────────────────────────────
train_process: Optional[subprocess.Popen] = None
# Use absolute path so the log path shown in the UI is copy-pasteable
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
train_log_path: str = os.path.join(_project_root, "runs", "train_ui.log")

# PID file — written when training starts, deleted when it ends.
# Survives uvicorn restarts so we can detect orphaned processes.
_train_pid_file: str = os.path.join(_project_root, "runs", "train_ui.pid")


def save_train_pid(pid: int) -> None:
    os.makedirs("runs", exist_ok=True)
    with open(_train_pid_file, "w") as f:
        f.write(str(pid))


def clear_train_pid() -> None:
    try:
        os.remove(_train_pid_file)
    except FileNotFoundError:
        pass


def get_orphan_pid() -> Optional[int]:
    """Return PID of a training process that outlived a uvicorn restart, or None."""
    if not os.path.exists(_train_pid_file):
        return None
    try:
        pid = int(open(_train_pid_file).read().strip())
        # Check if process is actually alive
        os.kill(pid, 0)   # signal 0 = existence check, raises if gone
        return pid
    except (ValueError, ProcessLookupError, PermissionError):
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

            sim.load_state_dict(ckpt["model_state_dict"])
            sim.eval()
            _model_cache[key] = sim
    return _model_cache[key]


def clear_model_cache():
    """Force reload on next get_model() call (e.g. after training completes)."""
    _model_cache.clear()


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
