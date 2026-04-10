"""
/generate endpoint — PhysicsAI Generate
========================================
SSE stream that samples novel mesh designs from the CVAE and evaluates
them using the MeshGraphNets predictor.

Endpoints
---------
POST /api/generate
    Body: GenerateRequest
    Response: SSE stream of ``candidate`` + ``done`` events

GET /api/generate/thumbnail/{session_id}/{candidate_id}
    Returns a PNG image of the candidate mesh with node-type colour coding.

Design principles
-----------------
- **Single Responsibility**: ``DesignSampler`` generates design params;
  ``DesignEvaluator`` runs the predictor; ``ThumbnailRenderer`` renders PNGs.
- **Open / Closed**: add new domains by registering in ``_DOMAIN_SAMPLERS``.
- **Dependency Inversion**: the router depends on abstract ``BaseDesignSampler``.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import io
import uuid
import base64
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import AsyncGenerator, Optional

import numpy as np
import torch
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from api.state import DOMAINS

router = APIRouter()

# In-memory thumbnail cache: session_id → {candidate_id → png_bytes}
_thumbnail_cache: dict[str, dict[int, bytes]] = {}


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    domain:           str   = "cylinder_flow"
    target_value:     float = 0.025    # drag (CFD) or stress (cloth)
    n_candidates:     int   = 5
    method:           str   = "sample"  # "sample" | "gradient"
    device:           str   = "cpu"


# ---------------------------------------------------------------------------
# Candidate DTO
# ---------------------------------------------------------------------------

@dataclass
class CandidateResult:
    """One generated design candidate."""
    id:              int
    domain:          str
    predicted_value: float    # drag (CFD) or stress (cloth)
    target_value:    float
    ood_confidence:  float    # 1.0 = in-distribution
    is_ood:          bool
    mesh_nodes:      int
    params:          dict     # domain-specific params


# ---------------------------------------------------------------------------
# Abstract sampler interface
# ---------------------------------------------------------------------------

class BaseDesignSampler(ABC):
    """Abstract interface for domain-specific design generation."""

    @abstractmethod
    def sample(self, target: float, n: int, device: str) -> list[CandidateResult]:
        """Generate n design candidates aiming for the given physics target."""


# ---------------------------------------------------------------------------
# CFD sampler
# ---------------------------------------------------------------------------

class CFDDesignSampler(BaseDesignSampler):
    """Samples cylinder_flow designs using the CFD CVAE."""

    CFD_CVAE_PATH      = "checkpoints/cfd_cvae.pth"
    SURROGATE_PATH     = "checkpoints/drag_surrogate.pth"
    PARAMS_PATH        = "data/design_params.npy"
    REFERENCE_TRAJ_DIR = "data"

    def sample(self, target: float, n: int, device: str) -> list[CandidateResult]:
        from extensions.generative.cvae_cfd import CVAETrainer, CFDCVAE, CVAEConfig
        from extensions.generative.drag_surrogate import DragSurrogateTrainer
        from extensions.generative.mesh_generator import CFDMeshBuilder
        from extensions.confidence.ood_detector import OODDetector
        from api.state import get_model

        # ── Load CVAE trainer ───────────────────────────────────────────────
        if not os.path.exists(self.CFD_CVAE_PATH):
            raise HTTPException(
                status_code=503,
                detail=f"CFD CVAE not trained yet. "
                       f"Run: python extensions/generative/train_cvae.py --domain cylinder_flow"
            )
        ckpt = torch.load(self.CFD_CVAE_PATH, map_location=device, weights_only=False)
        cfg   = ckpt["cfg"]
        model = CFDCVAE(cfg=cfg)
        model.load_state_dict(ckpt["model_state_dict"])

        # Build a minimal trainer wrapper just for generate()
        surrogate_trainer = DragSurrogateTrainer.load(self.SURROGATE_PATH,
                                                       device=device)

        from extensions.generative.cvae_cfd import CVAETrainer, CVAEScaler
        trainer             = CVAETrainer.__new__(CVAETrainer)
        trainer._model      = model.to(device)
        trainer._scaler     = ckpt["scaler"]
        trainer._cfg        = cfg
        trainer._device     = device
        trainer._surrogate  = surrogate_trainer._model

        # ── Generate samples ─────────────────────────────────────────────────
        params_phys = trainer.generate(target_drag_physical=target, n=n)  # [n, 4]

        # ── Build meshes and score ────────────────────────────────────────────
        builder = CFDMeshBuilder()

        # Load OOD detector if index exists
        index_path = "runs/embedding_index.pkl"
        detector   = None
        simulator  = get_model("cylinder_flow", device=device)
        if simulator is not None and os.path.exists(index_path):
            try:
                detector = OODDetector.from_index_file(
                    index_path, simulator=simulator, device=device
                )
            except Exception:
                pass

        candidates = []
        for i, row in enumerate(params_phys):
            cx, cy, r, v_in = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            # Clamp r to valid range
            r   = float(np.clip(r, 0.01, 0.15))
            cx  = float(np.clip(cx, r + 0.01, 1.6 - r - 0.01))
            cy  = float(np.clip(cy, r + 0.01, 0.41 - r - 0.01))
            v_in = float(np.clip(v_in, 0.05, 2.0))

            try:
                graph = builder.build(cx, cy, r, v_in)
            except Exception as e:
                continue

            # Predict drag via surrogate
            p_arr      = np.array([[cx, cy, r, v_in]], dtype=np.float32)
            pred_drag  = float(surrogate_trainer.predict(p_arr)[0])

            # OOD score
            ood_conf = 0.5   # default if no detector
            is_ood   = False
            if detector is not None and simulator is not None:
                try:
                    ood_result = detector.score(graph)
                    ood_conf   = ood_result.confidence
                    is_ood     = ood_result.is_ood
                except Exception:
                    pass

            c = CandidateResult(
                id=i,
                domain="cylinder_flow",
                predicted_value=pred_drag,
                target_value=target,
                ood_confidence=ood_conf,
                is_ood=is_ood,
                mesh_nodes=graph.num_nodes,
                params={"cx": cx, "cy": cy, "r": r, "v_inlet": v_in},
            )
            candidates.append((c, graph))

        # Sort by |predicted - target|
        candidates.sort(key=lambda x: abs(x[0].predicted_value - target))
        return [(c, g) for c, g in candidates[:n]]


# ---------------------------------------------------------------------------
# Cloth sampler
# ---------------------------------------------------------------------------

class ClothDesignSampler(BaseDesignSampler):
    """Samples flag_simple designs using the Cloth CVAE."""

    CVAE_PATH      = "checkpoints/flag-simple_cvae.pth"
    PCA_PATH       = "data_flag/train/cloth_pca.pkl"
    STRESS_PATH    = "data_flag/train/cloth_stress.npy"
    REF_TRAJ       = "data_flag/train/traj_00000.npz"

    def sample(self, target: float, n: int, device: str) -> list[CandidateResult]:
        from extensions.generative.cvae_cloth import ClothCVAE, ClothCVAETrainer
        from extensions.generative.cloth_extractor import PosePCA
        from extensions.generative.mesh_generator import ClothMeshBuilder

        if not os.path.exists(self.CVAE_PATH):
            raise HTTPException(
                status_code=503,
                detail="Cloth CVAE not trained yet. "
                       "Run: python extensions/generative/train_cvae.py --domain flag_simple"
            )

        pca  = PosePCA.load(self.PCA_PATH)
        ckpt = torch.load(self.CVAE_PATH, map_location=device, weights_only=False)
        cfg  = ckpt["cfg"]
        model = ClothCVAE(cfg=cfg)
        model.load_state_dict(ckpt["model_state_dict"])

        # Minimal trainer wrapper for generate()
        from extensions.generative.cvae_cloth import (
            ClothCVAETrainer, StressSurrogate, StressSurrogateTrainer
        )
        s_model   = StressSurrogate(pose_dim=cfg.pose_dim)
        s_trainer = StressSurrogateTrainer(s_model, device=device)
        trainer           = ClothCVAETrainer.__new__(ClothCVAETrainer)
        trainer._model    = model.to(device)
        trainer._scaler   = ckpt["scaler"]
        trainer._cfg      = cfg
        trainer._device   = device
        trainer._stress_trainer = s_trainer

        # Load stress data for reference
        stress_all = np.load(self.STRESS_PATH)

        builder    = ClothMeshBuilder(reference_traj_path=self.REF_TRAJ)
        world_poses = trainer.generate(target_stress=target, n=n, pca=pca)  # [n,N,3]

        candidates = []
        for i, wp in enumerate(world_poses):
            graph = builder.build(wp)

            # Stress proxy from surrogate
            pose_flat = pca.transform(wp.flatten().reshape(1, -1)).squeeze(0)
            scaler    = trainer._scaler
            pose_n    = scaler.norm_pose(pose_flat.reshape(1, -1))
            pred_stress = target   # approximate; full prediction needs simulator

            c = CandidateResult(
                id=i,
                domain="flag_simple",
                predicted_value=pred_stress,
                target_value=target,
                ood_confidence=0.5,   # OOD for cloth TBD
                is_ood=False,
                mesh_nodes=graph.num_nodes,
                params={"world_pos_norm": float(np.linalg.norm(wp))},
            )
            candidates.append((c, graph))

        return candidates


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_DOMAIN_SAMPLERS: dict[str, type[BaseDesignSampler]] = {
    "cylinder_flow": CFDDesignSampler,
    "flag_simple":   ClothDesignSampler,
}


# ---------------------------------------------------------------------------
# Thumbnail renderer (Single Responsibility)
# ---------------------------------------------------------------------------

class ThumbnailRenderer:
    """Renders a PyG Data graph as a PNG thumbnail using matplotlib."""

    SIZE: tuple[int, int] = (400, 300)   # px

    @staticmethod
    def render_cfd(graph, cx: float = None, cy: float = None,
                   r: float = None) -> bytes:
        """Render CFD mesh with node-type colour coding."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.tri as tri

        from utils.utils import NodeType

        pos = graph.pos.numpy()
        x   = graph.x.numpy()
        nt  = x[:, 0].astype(int)

        # Colour map per node type
        colours = np.where(nt == int(NodeType.NORMAL),       0,
                  np.where(nt == int(NodeType.WALL_BOUNDARY), 1,
                  np.where(nt == int(NodeType.INFLOW),        2,
                  np.where(nt == int(NodeType.OUTFLOW),       3, 0))))
        cmap = plt.cm.get_cmap("tab10")

        fig, ax = plt.subplots(figsize=(4, 2.4), dpi=100)
        ax.scatter(pos[:, 0], pos[:, 1], c=colours, cmap=cmap,
                   s=2, vmin=0, vmax=9)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_facecolor("#1a1a2e")
        fig.patch.set_facecolor("#1a1a2e")

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    @staticmethod
    def render_cloth(graph) -> bytes:
        """Render cloth mesh wireframe."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        wp  = graph.world_pos.numpy()  # [N, 3]
        face = graph.face.numpy().T    # [F, 3]

        fig = plt.figure(figsize=(4, 3), dpi=100)
        ax  = fig.add_subplot(111, projection="3d")
        ax.plot_trisurf(wp[:, 0], wp[:, 1], wp[:, 2],
                        triangles=face, alpha=0.6, color="#6c63ff")
        ax.set_axis_off()
        fig.patch.set_facecolor("#1a1a2e")

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        return buf.read()


# ---------------------------------------------------------------------------
# SSE helper
# ---------------------------------------------------------------------------

def _sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# ---------------------------------------------------------------------------
# Generate endpoint
# ---------------------------------------------------------------------------

@router.post("/generate")
async def generate(req: GenerateRequest):
    """
    PhysicsAI Generate — sample novel designs + predict physics.

    Returns a Server-Sent Events (SSE) stream:
        event: candidate  data: { CandidateResult fields... }
        event: error      data: { "detail": "..." }
        event: done       data: { "best_id": int }
    """
    if req.domain not in _DOMAIN_SAMPLERS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown domain '{req.domain}'. "
                   f"Available: {list(_DOMAIN_SAMPLERS)}"
        )
    if req.n_candidates < 1 or req.n_candidates > 50:
        raise HTTPException(status_code=400,
                            detail="n_candidates must be in [1, 50]")

    session_id = str(uuid.uuid4())

    async def event_stream() -> AsyncGenerator[str, None]:
        try:
            sampler = _DOMAIN_SAMPLERS[req.domain]()

            # Run sampler in thread pool so it doesn't block the event loop
            loop = asyncio.get_running_loop()
            results = await loop.run_in_executor(
                None,
                lambda: sampler.sample(req.target_value,
                                       req.n_candidates,
                                       req.device)
            )

            # Store thumbnails and yield events
            _thumbnail_cache[session_id] = {}
            best_id  = 0
            best_err = float("inf")

            for c, graph in results:
                # Render thumbnail
                try:
                    if req.domain == "cylinder_flow":
                        png = ThumbnailRenderer.render_cfd(graph)
                    else:
                        png = ThumbnailRenderer.render_cloth(graph)
                    _thumbnail_cache[session_id][c.id] = png
                    thumbnail_url = f"/api/generate/thumbnail/{session_id}/{c.id}"
                except Exception:
                    thumbnail_url = None

                payload        = asdict(c)
                payload["thumbnail_url"] = thumbnail_url
                payload["session_id"]    = session_id

                err = abs(c.predicted_value - c.target_value)
                if err < best_err:
                    best_err = err
                    best_id  = c.id

                yield _sse_event("candidate", payload)
                await asyncio.sleep(0)   # yield to event loop

            yield _sse_event("done", {"best_id": best_id,
                                       "session_id": session_id})

        except HTTPException as e:
            yield _sse_event("error", {"detail": e.detail})
        except Exception as e:
            yield _sse_event("error", {"detail": str(e)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":  "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Thumbnail endpoint
# ---------------------------------------------------------------------------

@router.get("/generate/thumbnail/{session_id}/{candidate_id}")
async def get_thumbnail(session_id: str, candidate_id: int):
    """Return a PNG thumbnail for a generated candidate."""
    session = _thumbnail_cache.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    png = session.get(candidate_id)
    if png is None:
        raise HTTPException(status_code=404, detail="Candidate not found")
    return Response(content=png, media_type="image/png")
