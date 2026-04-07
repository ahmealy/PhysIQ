"""
Shared state for the FastAPI server.
Single module so all routers share the same objects without circular imports.
"""

import os
import subprocess
from typing import Optional
import torch

# ── Training state ────────────────────────────────────────────────────────────
train_process: Optional[subprocess.Popen] = None
train_log_path: str = "runs/train_ui.log"

# PID file — written when training starts, deleted when it ends.
# Survives uvicorn restarts so we can detect orphaned processes.
_train_pid_file: str = "runs/train_ui.pid"


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


def get_model(checkpoint_path: str, device: str):
    """
    Load and cache the Simulator. Returns cached instance if already loaded
    from the same checkpoint and device.
    """
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from model.simulator import Simulator

    key = (checkpoint_path, device)
    if key not in _model_cache:
        sim = Simulator(
            message_passing_num=15,
            node_input_size=11,
            edge_input_size=3,
            device=device,
        )
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
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
        "label":       "Cylinder Flow (CFD)",
        "description": "2D fluid flow past a cylinder — von Kármán vortex street",
        "data_dir":    "data",
        "checkpoint":  "checkpoints/best_model.pth",
        "node_input":  11,
        "edge_input":  3,
        "mp_steps":    15,
        "dt":          0.01,
        "available":   True,
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
