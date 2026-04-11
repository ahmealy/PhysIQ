"""
/train/* endpoints — start, stop, status, SSE stream, SSH remote config.
"""

import json
import asyncio
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from typing import AsyncGenerator, Literal, Optional

import psutil
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import api.state as state

router = APIRouter(prefix="/train")


# ── Request models ────────────────────────────────────────────────────────────
class TrainConfig(BaseModel):
    domain:                   str                           = "cylinder_flow"
    target_field:             Literal["velocity", "pressure"] = "velocity"
    epochs:                   int   = 100
    batch_size:               int   = 20
    lr:                       float = 1e-4
    noise_std:                float = 0.02
    early_stopping_patience:  int   = 10
    message_passing_steps:    int   = 15
    output_size:              int   = 2
    node_input_size:          int   = 11
    edge_input_size:          int   = 3
    fresh_start:              bool  = False   # if True, delete existing checkpoint before training


class RemoteConfig(BaseModel):
    host:        str            # e.g. dvt-gpubig1.wv.mentorg.com
    port:        int   = 22
    user:        str   = ""     # empty = use current username
    venv_python: str   = "/home/ahmealy/.pyenv/versions/venv_gpu/bin/python"
    enabled:     bool  = True


# ── SSH config persistence ─────────────────────────────────────────────────────
_SSH_CFG_PATH = "runs/remote_gpu.json"

def _load_remote_cfg() -> Optional[dict]:
    if not os.path.exists(_SSH_CFG_PATH):
        return None
    try:
        with open(_SSH_CFG_PATH) as f:
            d = json.load(f)
        if not d.get("enabled") or not d.get("host"):
            return None
        return d
    except Exception:
        return None

def _save_remote_cfg(cfg: Optional[dict]) -> None:
    os.makedirs("runs", exist_ok=True)
    with open(_SSH_CFG_PATH, "w") as f:
        json.dump(cfg or {}, f, indent=2)


# ── SSH command builder ────────────────────────────────────────────────────────
def _build_ssh_prefix(cfg: dict) -> list[str]:
    """Return the ssh [...] prefix list for subprocess."""
    user = cfg.get("user", "").strip()
    host = cfg["host"].strip()
    port = int(cfg.get("port", 22))
    target = f"{user}@{host}" if user else host
    return [
        "ssh",
        "-p", str(port),
        "-o", "StrictHostKeyChecking=no",   # don't block on first-connect prompt
        "-o", "BatchMode=yes",              # fail immediately if key auth not set up
        "-o", "ConnectTimeout=10",
        target,
    ]


# ── Helpers ───────────────────────────────────────────────────────────────────
def _parse_log(log_path: str) -> list[dict]:
    """Parse epoch lines from train.py stdout.
    Format: 'Epoch N/M Train Loss: X.XXe-XX Valid Loss: X.XXe-XX'
    """
    epochs = []
    if not os.path.exists(log_path):
        return epochs
    with open(log_path, "r", errors="replace") as f:
        for line in f:
            if line.startswith("Epoch ") and "Train Loss:" in line:
                try:
                    parts = line.split()
                    ep = int(parts[1].split("/")[0])
                    tl = float(parts[4])
                    vl = float(parts[7])
                    epochs.append({"epoch": ep, "train_loss": tl, "valid_loss": vl})
                except (IndexError, ValueError):
                    continue
    return epochs


def _friendly_error(log_path: str) -> str:
    """Extract a human-readable error from a failed training log."""
    default = "Training failed to start — check that dependencies are installed."
    try:
        with open(log_path, "r", errors="replace") as f:
            lines = [l.rstrip() for l in f.readlines() if l.strip()]
        for line in reversed(lines):
            if "RuntimeError" in line or "Error:" in line or "CUDA" in line:
                msg = line.strip()
                if "can't allocate memory" in msg or "Cannot allocate memory" in msg:
                    return "Out of memory (CPU RAM). Try reducing Batch Size (e.g. 4–8) or closing other applications."
                if "CUDA out of memory" in msg:
                    return "Out of GPU memory. Try reducing Batch Size or Message Passing Steps."
                return msg
        missing = next((l for l in lines if "No module named" in l), None)
        if missing:
            return f"Missing dependency: {missing.strip()}"
    except Exception:
        pass
    return default


# ── SSH config endpoints ───────────────────────────────────────────────────────
@router.get("/remote")
def get_remote():
    """Return the current remote GPU SSH config (or empty if not set)."""
    cfg = _load_remote_cfg()
    return cfg or {"enabled": False, "host": "", "port": 22, "user": "", "venv_python": "/home/ahmealy/.pyenv/versions/venv_gpu/bin/python"}


@router.post("/remote")
def set_remote(cfg: RemoteConfig):
    """Save SSH config for remote GPU training."""
    _save_remote_cfg(cfg.model_dump())
    return {"status": "saved"}


@router.delete("/remote")
def clear_remote():
    """Disable remote GPU — revert to local execution."""
    _save_remote_cfg({"enabled": False})
    return {"status": "cleared"}


@router.post("/remote/test")
def test_remote(cfg: RemoteConfig):
    """
    Test SSH connectivity and verify the venv Python exists on the remote.
    Returns {ok: bool, message: str}.
    """
    venv_py = cfg.venv_python.strip()

    try:
        # Write a small probe script to the shared filesystem so we can run it
        # as a file rather than via -c '...' (some remote shells mangle single-quoted
        # inline code when commands are chained, causing silent empty output).
        probe_path = os.path.join("runs", "gpu_probe.py")
        os.makedirs("runs", exist_ok=True)
        with open(probe_path, "w") as fp:
            fp.write(
                "import sys\n"
                "sys.stdout.reconfigure(line_buffering=True)\n"
                "try:\n"
                "    import torch\n"
                "    if torch.cuda.is_available():\n"
                "        print('GPU:' + torch.cuda.get_device_name(0))\n"
                "    else:\n"
                "        print('no-gpu')\n"
                "except Exception as e:\n"
                "    print('probe-error:' + str(e))\n"
            )

        probe_abs = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            probe_path,
        )
        ssh_prefix = _build_ssh_prefix(cfg.model_dump())
        remote_cmd = (
            f"echo __PYTHON__; {shlex.quote(venv_py)} --version; "
            f"echo __GPU__; {shlex.quote(venv_py)} -u {shlex.quote(probe_abs)}"
        )
        result = subprocess.run(
            ssh_prefix + [remote_cmd],
            capture_output=True, text=True, timeout=20,
        )
        if result.returncode != 0:
            stderr = result.stderr.strip()
            if "Permission denied" in stderr:
                return {"ok": False, "message": "SSH key auth failed — run: ssh-copy-id -p {port} {user}@{host}".format(**cfg.model_dump())}
            if "Connection refused" in stderr or "No route" in stderr:
                return {"ok": False, "message": f"Cannot reach {cfg.host}:{cfg.port} — is the machine on and port correct?"}
            return {"ok": False, "message": stderr or "SSH command failed (exit %d)" % result.returncode}

        # Parse sections delimited by markers — immune to login-shell noise before __PYTHON__
        output = result.stdout
        python_ver, gpu_line = "unknown", ""
        section = None
        for raw_line in output.splitlines():
            line = raw_line.strip()
            if line == "__PYTHON__":
                section = "python"
                continue
            if line == "__GPU__":
                section = "gpu"
                continue
            if section == "python" and line:
                python_ver = line
                section = None  # only take first non-empty line
            elif section == "gpu" and line:
                if line.startswith("GPU:"):
                    gpu_line = line[4:]   # strip "GPU:" prefix written by probe script
                elif line not in ("no-gpu",) and not line.startswith("probe-error:"):
                    gpu_line = line       # fallback: bare GPU name
                break

        msg = f"Connected ✓  {python_ver}"
        if gpu_line:
            msg += f"  |  GPU: {gpu_line}"
        else:
            msg += "  |  No GPU / CUDA not available"
        return {"ok": True, "message": msg}

    except subprocess.TimeoutExpired:
        return {"ok": False, "message": f"Connection timed out after 15s — is {cfg.host}:{cfg.port} reachable?"}
    except Exception as e:
        return {"ok": False, "message": str(e)}


# ── Routes ────────────────────────────────────────────────────────────────────
@router.post("/start")
def train_start(config: TrainConfig):
    # Block if our Popen handle is alive
    if state.train_process is not None and state.train_process.poll() is None:
        raise HTTPException(409, "Training is already running (PID %d)" % state.train_process.pid)
    # Also block if an orphaned process from a previous server session is still alive
    orphan = state.get_orphan_pid()
    if orphan:
        raise HTTPException(409, "Training is already running (PID %d, orphaned from previous session)" % orphan)

    if config.domain not in state.DOMAINS:
        raise HTTPException(404, "Unknown domain: %s" % config.domain)

    domain_cfg = state.DOMAINS[config.domain]
    if not domain_cfg["available"]:
        raise HTTPException(400, "Domain '%s' is not available yet" % config.domain)

    # Switch all domain-scoped file paths (log, pid, heartbeat) to the new domain
    state.set_active_domain(config.domain)

    # Write config JSON for train.py to consume
    os.makedirs("runs", exist_ok=True)
    cfg_path = "runs/ui_train_config.json"

    # fresh_start: delete existing checkpoint so train.py starts from epoch 1
    if config.fresh_start:
        ckpt_path = domain_cfg["checkpoint"]
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)
        # Also clear the in-memory model cache so the old weights aren't served
        state.clear_model_cache()
    _DOMAIN_SIZES = {
        ("cylinder_flow", "velocity"):  {"output_size": 2, "node_input_size": 11, "edge_input_size": 3},
        ("cylinder_flow", "pressure"):  {"output_size": 1, "node_input_size": 10, "edge_input_size": 3},
        ("flag_simple",   "velocity"):  {"output_size": 3, "node_input_size": 12, "edge_input_size": 7},
    }
    _tf = config.target_field if config.domain == "cylinder_flow" else "velocity"
    key = (config.domain, _tf)
    if key in _DOMAIN_SIZES:
        sizes = _DOMAIN_SIZES[key]
        config.output_size     = sizes["output_size"]
        config.node_input_size = sizes["node_input_size"]
        config.edge_input_size = sizes["edge_input_size"]
    with open(cfg_path, "w") as f:
        json.dump({
            "domain":                  config.domain,
            "target_field":            _tf,
            "output_size":             config.output_size,
            "node_input_size":         config.node_input_size,
            "edge_input_size":         config.edge_input_size,
            "dataset_dir":             domain_cfg["data_dir"],
            "checkpoint_dir":          os.path.dirname(domain_cfg["checkpoint"]),
            "num_epochs":              config.epochs,
            "batch_size":              config.batch_size,
            "lr":                      config.lr,
            "noise_std":               config.noise_std,
            "early_stopping_patience": config.early_stopping_patience,
            "message_passing_num":     config.message_passing_steps,
            "log_dir":                 "runs",
        }, f)

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    log_file = open(state.train_log_path, "w")

    remote_cfg = _load_remote_cfg()
    if remote_cfg:
        # ── Remote GPU execution over SSH ─────────────────────────────────────
        venv_py    = remote_cfg.get("venv_python", "/home/ahmealy/.pyenv/versions/venv_gpu/bin/python").strip()
        cfg_abs    = os.path.join(project_root, cfg_path)
        ssh_prefix = _build_ssh_prefix(remote_cfg)
        log_abs    = state.train_log_path
        # Use domain-scoped paths from state so log/pid/heartbeat never collide
        remote_pid_file = state._train_remote_pid_file
        heartbeat_path  = state._train_heartbeat_file

        # Write a bash launcher script on shared NFS — avoids all csh quoting issues.
        # nohup detaches train.py from the SSH session so it survives server restarts.
        launcher_path = os.path.join(project_root, "runs", "train_launcher.sh")
        with open(launcher_path, "w") as lf:
            lf.write("#!/bin/bash\n")
            lf.write(f"cd {shlex.quote(project_root)}\n")
            # Launch train.py in background, capture PID
            lf.write(f"nohup {shlex.quote(venv_py)} -u train.py --config {shlex.quote(cfg_abs)}"
                     f" >> {shlex.quote(log_abs)} 2>&1 &\n")
            lf.write("TRAIN_PID=$!\n")
            lf.write(f"printf '%s' \"$TRAIN_PID\" > {shlex.quote(remote_pid_file)}\n")
            lf.write("echo REMOTE_PID:$TRAIN_PID\n")
            # Heartbeat: touch a file every 60s while train.py is alive.
            # Cloth (flag_simple) epochs can take hours, so the log isn't written for
            # a long time — heartbeat lets the UI distinguish "running" from "dead".
            lf.write(f"( while kill -0 $TRAIN_PID 2>/dev/null; do"
                     f" touch {shlex.quote(heartbeat_path)}; sleep 60; done ) &\n")
        os.chmod(launcher_path, 0o755)

        remote_cmd = f"bash {shlex.quote(launcher_path)}"
        cmd = ssh_prefix + [remote_cmd]
        execution = "remote"
    else:
        # ── Local execution ───────────────────────────────────────────────────
        venv_python = os.path.join(project_root, "venv", "bin", "python")
        python_bin  = venv_python if os.path.exists(venv_python) else sys.executable
        cmd = [python_bin, "-u", "train.py", "--config", cfg_path]
        execution = "local"

    if remote_cfg:
        # SSH exits immediately (nohup &); capture remote PID from stdout
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=log_file,
            cwd=project_root,
        )
        stdout, _ = proc.communicate(timeout=15)
        log_file.close()
        # Parse REMOTE_PID:<n> from output
        remote_pid = None
        for line in (stdout or b"").decode(errors="replace").splitlines():
            if line.startswith("REMOTE_PID:"):
                try:
                    remote_pid = int(line.split(":")[1].strip())
                except ValueError:
                    pass
        # Store remote PID in the shared PID file so orphan detection works
        pid_to_save = remote_pid or 0
        state.save_train_pid(pid_to_save)
        state.save_train_start_time()   # record launch timestamp for elapsed timer
        state.train_process = None   # SSH proc already exited
        state.clear_model_cache()
        return {"pid": remote_pid, "status": "started", "execution": execution}
    else:
        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=project_root,
        )
        log_file.close()
        state.train_process = proc
        state.save_train_pid(proc.pid)
        state.save_train_start_time()   # record launch timestamp for elapsed timer
        state.clear_model_cache()
        return {"pid": proc.pid, "status": "started", "execution": execution}


@router.post("/stop")
def train_stop():
    stopped = False
    if state.train_process is not None and state.train_process.poll() is None:
        state.train_process.send_signal(signal.SIGTERM)
        state.train_process = None
        stopped = True
    orphan = state.get_orphan_pid()
    if orphan:
        # Check if it's a remote nohup job — must kill via SSH
        remote_pid_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "runs", "train_remote.pid"
        )
        is_remote = False
        if os.path.exists(remote_pid_file):
            try:
                is_remote = int(open(remote_pid_file).read().strip()) == orphan
            except (ValueError, OSError):
                pass
        if is_remote:
            remote_cfg = _load_remote_cfg()
            if remote_cfg:
                try:
                    ssh_prefix = _build_ssh_prefix(remote_cfg)
                    subprocess.run(
                        ssh_prefix + [f"kill -TERM {orphan} 2>/dev/null || kill -KILL {orphan} 2>/dev/null"],
                        capture_output=True, timeout=10
                    )
                except Exception:
                    pass
        else:
            try:
                os.kill(orphan, signal.SIGTERM)
            except ProcessLookupError:
                pass
        stopped = True
    state.clear_train_pid()
    if not stopped:
        raise HTTPException(400, "No training process is running")
    return {"status": "stopped"}


# ── Process manager endpoints ─────────────────────────────────────────────────

def _get_train_processes() -> list[dict]:
    """
    Scan all running processes for train.py invocations using psutil.
    Also injects a synthetic entry for remote nohup jobs tracked via
    runs/train_remote.pid + log freshness.
    """
    managed_pid = None
    if state.train_process is not None and state.train_process.poll() is None:
        managed_pid = state.train_process.pid
    if managed_pid is None:
        managed_pid = state.get_orphan_pid()

    procs = []

    # ── Remote nohup job (no local psutil entry) ──────────────────────────────
    remote_pid_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "runs", "train_remote.pid"
    )
    if os.path.exists(remote_pid_file):
        try:
            remote_pid = int(open(remote_pid_file).read().strip())
            log_path   = state.train_log_path
            log_mtime  = os.path.getmtime(log_path) if os.path.exists(log_path) else 0
            log_age    = time.time() - log_mtime
            if remote_pid > 0 and log_age < 120:
                # Build elapsed from log file creation time
                log_ctime  = os.path.getctime(log_path) if os.path.exists(log_path) else time.time()
                elapsed_s  = int(time.time() - log_ctime)
                hours, rem = divmod(elapsed_s, 3600)
                mins, secs = divmod(rem, 60)
                elapsed_str = "%02d:%02d:%02d" % (hours, mins, secs)
                # Read domain from active config
                domain = "unknown"
                cfg_path = os.path.join(
                    os.path.dirname(remote_pid_file), "ui_train_config.json"
                )
                try:
                    with open(cfg_path) as f:
                        domain = json.load(f).get("domain", "unknown")
                except Exception:
                    pass
                remote_cfg = _load_remote_cfg()
                procs.append({
                    "pid":       remote_pid,
                    "status":    "running",
                    "cpu_pct":   None,   # can't measure remote CPU locally
                    "mem_mb":    None,
                    "elapsed":   elapsed_str,
                    "elapsed_s": elapsed_s,
                    "domain":    domain,
                    "device":    "remote GPU (%s)" % remote_cfg.get("host", "?") if remote_cfg else "remote GPU",
                    "managed":   True,
                    "cmdline":   "train.py (nohup, remote)",
                })
        except (ValueError, OSError):
            pass

    # ── Local psutil scan ─────────────────────────────────────────────────────
    for p in psutil.process_iter(["pid", "name", "cmdline", "status", "create_time", "cpu_percent", "memory_info"]):
        try:
            cmd = p.info["cmdline"] or []
            # Match any process running train.py (our training script)
            if not any("train.py" in arg for arg in cmd):
                continue
            # Also exclude rollout_ssh.py / rollout.py which may contain "train" in path
            if any("rollout" in arg for arg in cmd):
                continue

            create_time = p.info["create_time"]
            elapsed_s = int(time.time() - create_time)
            hours, rem = divmod(elapsed_s, 3600)
            mins, secs = divmod(rem, 60)
            elapsed_str = "%02d:%02d:%02d" % (hours, mins, secs)

            mem_mb = round(p.info["memory_info"].rss / 1024 / 1024, 1) if p.info["memory_info"] else None

            device = "local CPU"
            domain = "unknown"
            for i, arg in enumerate(cmd):
                if arg == "--config" and i + 1 < len(cmd):
                    try:
                        with open(cmd[i + 1]) as f:
                            cfg = json.load(f)
                        domain = cfg.get("domain", "unknown")
                    except Exception:
                        pass
                    break

            procs.append({
                "pid":       p.info["pid"],
                "status":    p.info["status"],
                "cpu_pct":   round(p.cpu_percent(interval=0.1), 1),
                "mem_mb":    mem_mb,
                "elapsed":   elapsed_str,
                "elapsed_s": elapsed_s,
                "domain":    domain,
                "device":    device,
                "managed":   p.info["pid"] == managed_pid,
                "cmdline":   " ".join(cmd[:6]),
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    return sorted(procs, key=lambda x: x["elapsed_s"], reverse=True)


@router.get("/processes")
async def list_processes():
    """List all running train.py processes with resource usage."""
    loop = asyncio.get_running_loop()
    procs = await loop.run_in_executor(None, _get_train_processes)
    return {"processes": procs}


@router.post("/kill/{pid}")
def kill_process(pid: int):
    """
    Kill a training process by PID.
    - Remote nohup jobs (tracked via train_remote.pid): kills via SSH SIGTERM.
    - Local processes: kills via os.kill after psutil safety check.
    """
    remote_pid_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "runs", "train_remote.pid"
    )

    # ── Remote nohup job ──────────────────────────────────────────────────────
    is_remote = False
    if os.path.exists(remote_pid_file):
        try:
            stored = int(open(remote_pid_file).read().strip())
            if stored == pid:
                is_remote = True
        except (ValueError, OSError):
            pass

    if is_remote:
        remote_cfg = _load_remote_cfg()
        if not remote_cfg:
            raise HTTPException(500, "Remote config not found — cannot SSH-kill")
        ssh_prefix = _build_ssh_prefix(remote_cfg)
        kill_cmd = ssh_prefix + [f"kill -TERM {pid} 2>/dev/null || kill -KILL {pid} 2>/dev/null; echo done"]
        try:
            result = subprocess.run(kill_cmd, capture_output=True, timeout=10)
        except Exception as e:
            raise HTTPException(500, "SSH kill failed: %s" % str(e))
        state.clear_train_pid()
        return {"status": "killed", "pid": pid, "method": "ssh"}

    # ── Local process ─────────────────────────────────────────────────────────
    try:
        p = psutil.Process(pid)
        cmd = p.cmdline()
        if not any("train.py" in arg for arg in cmd):
            raise HTTPException(403, "PID %d is not a train.py process" % pid)
    except psutil.NoSuchProcess:
        raise HTTPException(404, "Process %d not found" % pid)
    except psutil.AccessDenied:
        raise HTTPException(403, "Cannot access process %d" % pid)

    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        raise HTTPException(404, "Process %d already exited" % pid)

    if state.train_process is not None and state.train_process.pid == pid:
        state.train_process = None
    orphan = state.get_orphan_pid()
    if orphan == pid:
        state.clear_train_pid()

    return {"status": "killed", "pid": pid, "method": "local"}



@router.get("/status")
async def train_status():
    # Restore domain-scoped paths from the active config file if server restarted
    cfg_path = "runs/ui_train_config.json"
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path) as f:
                saved_domain = json.load(f).get("domain", "cylinder_flow")
            state.set_active_domain(saved_domain)
        except Exception:
            pass

    is_running = (
        (state.train_process is not None and state.train_process.poll() is None)
        or state.get_orphan_pid() is not None
    )
    if not is_running:
        state.clear_train_pid()
    epochs = await asyncio.get_running_loop().run_in_executor(
        None, _parse_log, state.train_log_path
    )
    best = min(epochs, key=lambda e: e["valid_loss"]) if epochs else None

    # Read active training config so the UI can sync domain/target/etc on reload
    active_config = None
    cfg_path = "runs/ui_train_config.json"
    if is_running and os.path.exists(cfg_path):
        try:
            with open(cfg_path) as f:
                active_config = json.load(f)
        except Exception:
            pass

    remote_cfg = _load_remote_cfg()
    device_label = "LOCAL CPU"
    if remote_cfg:
        host = remote_cfg.get("host", "remote")
        device_label = f"REMOTE GPU ({host})"
    return {
        "running":         is_running,
        "pid":             state.train_process.pid if is_running and state.train_process else None,
        "epochs":          epochs,
        "best_epoch":      best["epoch"]      if best else None,
        "best_valid_loss": best["valid_loss"] if best else None,
        "remote":          remote_cfg is not None,
        "device":          device_label,
        # Accurate training start time (ms since epoch) so the frontend shows correct elapsed.
        # Prefer the persisted start-time file written at launch; fall back to log ctime.
        "log_start_ms":    state.get_train_start_time()
                           or (int(os.path.getctime(state.train_log_path) * 1000)
                               if os.path.exists(state.train_log_path) else None),
        "log_path":        state.train_log_path,
        "active_config":   active_config,
    }


@router.get("/log")
async def train_log(tail: int = 80):
    """Return the last `tail` lines of the training log as plain text."""
    if not os.path.exists(state.train_log_path):
        return {"lines": []}
    loop = asyncio.get_running_loop()
    def _read():
        with open(state.train_log_path, "r", errors="replace") as f:
            lines = f.readlines()
        # Strip tqdm carriage-return overwrite sequences — keep last \r-separated chunk per line
        cleaned = []
        for raw in lines[-tail:]:
            # tqdm uses \r to overwrite the line; keep only the last segment
            parts = raw.split('\r')
            cleaned.append(parts[-1].rstrip('\n'))
        return cleaned
    lines = await loop.run_in_executor(None, _read)
    return {"lines": lines}


@router.get("/stream")
async def train_stream():
    """
    SSE stream — polls the training log every 5s and emits new epoch events.
    """
    async def event_generator() -> AsyncGenerator[str, None]:
        seen_epochs: set[int] = set()
        best_loss = float("inf")
        dead_polls = 0

        while True:
            epochs = await asyncio.get_running_loop().run_in_executor(
                None, _parse_log, state.train_log_path
            )
            for ep in epochs:
                if ep["epoch"] not in seen_epochs:
                    seen_epochs.add(ep["epoch"])
                    dead_polls = 0
                    yield "data: %s\n\n" % json.dumps({
                        "type":        "epoch",
                        "epoch":       ep["epoch"],
                        "train_loss":  ep["train_loss"],
                        "valid_loss":  ep["valid_loss"],
                    })
                    if ep["valid_loss"] < best_loss:
                        best_loss = ep["valid_loss"]
                        yield "data: %s\n\n" % json.dumps({
                            "type":       "best",
                            "epoch":      ep["epoch"],
                            "valid_loss": ep["valid_loss"],
                        })

            is_running = (
                (state.train_process is not None and state.train_process.poll() is None)
                or state.get_orphan_pid() is not None
            )
            if not is_running:
                if seen_epochs:
                    state.clear_train_pid()
                    yield "data: %s\n\n" % json.dumps({"type": "done", "reason": "completed"})
                    break
                else:
                    dead_polls += 1
                    if dead_polls >= 3:
                        state.clear_train_pid()
                        yield "data: %s\n\n" % json.dumps({
                            "type":    "error",
                            "message": _friendly_error(state.train_log_path),
                        })
                        break

            # Keepalive ping so the browser EventSource doesn't time out
            # during long epochs (each flag_simple epoch can take hours).
            yield ": ping\n\n"
            await asyncio.sleep(5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
