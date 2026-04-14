import sys; sys.stdout.reconfigure(line_buffering=True)  # MUST be line 1 — before all other imports
"""
generate_ssh.py — SSH-compatible generate runner.

Reads a JSON config file (runs/ui_generate_config.json produced by the FastAPI
server), runs CVAE sampling / gradient-descent generation, and prints named
SSE-format lines to stdout so the API can relay them to the browser.

Usage:
    python -u generate_ssh.py --config /abs/path/to/runs/ui_generate_config.json

stdout lines (named SSE):
    event: trajectory
    data: {"values": [...]}

    event: candidate
    data: {<CandidateResult fields>, "thumbnail_url": "/api/...", "session_id": "<uuid>"}

    event: done
    data: {"best_id": <int>, "session_id": "<uuid>"}

    event: error
    data: {"detail": "...traceback..."}

The -u flag is required to disable output buffering; sys.stdout.reconfigure
on line 1 provides a second layer of defence when -u is omitted.

NFS design: thumbnails and graphs are saved to disk (runs/thumbnails/ and
runs/graphs/) so the local API server can serve them without any scp.
The session_id is written into the config by the API server so both sides
use the same path.
"""

import argparse
import dataclasses
import json
import os
import traceback

# ── Project root ──────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))

# chdir BEFORE sys.path so every relative path used by samplers (checkpoints/,
# data/, result/) resolves from the project root, regardless of where the SSH
# session started.
os.chdir(_ROOT)
sys.path.insert(0, _ROOT)


# ── Named-SSE emitter ─────────────────────────────────────────────────────────

def _sse(event: str, data: dict) -> None:
    """Emit a named SSE event to stdout.

    The blank line after ``data:`` is the SSE event terminator — without it
    the browser's EventSource never fires the event handler.
    """
    print(f"event: {event}\ndata: {json.dumps(data)}\n", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="SSH-compatible generate runner")
    parser.add_argument("--config", required=True,
                        help="Path to ui_generate_config.json")
    args = parser.parse_args()

    try:
        with open(args.config) as f:
            cfg = json.load(f)
    except Exception as e:
        _sse("error", {"detail": "Cannot read config: %s" % e})
        sys.exit(1)

    try:
        _run(cfg)
    except Exception as e:
        _sse("error", {"detail": str(e) + "\n" + traceback.format_exc()})
        sys.exit(1)


def _run(cfg: dict) -> None:
    domain        = cfg["domain"]
    target_value  = float(cfg["target_value"])
    n_candidates  = int(cfg["n_candidates"])
    method        = cfg.get("method", "sample")
    device        = cfg.get("device", "cuda:0")
    # session_id is written by the API server into the config so the remote
    # script uses the same ID — shared NFS means graph/thumbnail paths are
    # visible on both sides without any file transfer.
    session_id    = cfg.get("session_id")

    # Import samplers and renderer from the API module (project is on sys.path).
    from api.routes.generate import _DOMAIN_SAMPLERS, ThumbnailRenderer

    if domain not in _DOMAIN_SAMPLERS:
        raise ValueError(
            "Unknown domain '%s'. Available: %s" % (domain, list(_DOMAIN_SAMPLERS))
        )

    sampler = _DOMAIN_SAMPLERS[domain]()

    # Run sampling (CVAE + surrogate / gradient descent).
    results, trajectory = sampler.sample(target_value, n_candidates, device, method)
    # results: list[tuple[CandidateResult, graph_or_None]]

    # Stream optimisation trajectory first (gradient mode only).
    if trajectory:
        _sse("trajectory", {"values": list(trajectory)})

    # ── Thumbnails via NFS ───────────────────────────────────────────────────
    # On a shared filesystem both the remote worker and the local API server
    # see the same paths, so we save PNG files to disk instead of base64-
    # encoding them.  The API relay layer looks for these files and skips the
    # thumbnail_b64 decode path entirely.
    thumb_dir = None
    if session_id:
        thumb_dir = os.path.join(_ROOT, "runs", "thumbnails", session_id)
        os.makedirs(thumb_dir, exist_ok=True)

    # ── Graph cache via NFS ──────────────────────────────────────────────────
    # Save each graph to disk so the API's /generate/rollout/{session_id}/{id}
    # endpoint can load and analyse them (Analyze button).  Uses pickle since
    # torch_geometric Data objects are not JSON-serialisable.
    graph_dir = None
    if session_id:
        graph_dir = os.path.join(_ROOT, "runs", "graphs", session_id)
        os.makedirs(graph_dir, exist_ok=True)

    # Stream candidates with rendered thumbnails.
    best_id  = 0
    best_err = float("inf")

    for c, graph in results:
        # Save graph to disk for Analyze ──────────────────────────────────────
        if graph_dir and graph is not None:
            try:
                import pickle
                graph_path = os.path.join(graph_dir, "graph_%d.pkl" % c.id)
                graph.domain = c.domain   # tag domain so rollout endpoint can branch
                with open(graph_path, "wb") as gf:
                    pickle.dump(graph, gf)
            except Exception:
                pass  # graph save is best-effort

        # Render thumbnail and save to disk ───────────────────────────────────
        thumbnail_url = None
        if thumb_dir and graph is not None:
            try:
                if domain == "cylinder_flow":
                    p   = c.params
                    png = ThumbnailRenderer.render_cfd(
                        graph,
                        cx=p.get("cx"),
                        cy=p.get("cy"),
                        r=p.get("r"),
                    )
                else:
                    png = ThumbnailRenderer.render_cloth(graph)
                thumb_path = os.path.join(thumb_dir, "cand_%d.png" % c.id)
                with open(thumb_path, "wb") as tf:
                    tf.write(png)
                thumbnail_url = "/api/generate/thumbnail/%s/%d" % (session_id, c.id)
            except Exception:
                pass  # thumbnail is best-effort; never block the candidate stream

        # Emit candidate event ────────────────────────────────────────────────
        payload                   = dataclasses.asdict(c)
        payload["thumbnail_url"]  = thumbnail_url   # direct URL — no base64
        payload["thumbnail_b64"]  = None            # always null on NFS path
        payload["session_id"]     = session_id
        _sse("candidate", payload)

        # Track best candidate (closest predicted_value to target_value).
        err = abs(c.predicted_value - target_value)
        if err < best_err:
            best_err = err
            best_id  = c.id

    # Final done event ────────────────────────────────────────────────────────
    _sse("done", {"best_id": best_id, "session_id": session_id})


if __name__ == "__main__":
    main()
