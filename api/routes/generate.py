"""
/generate endpoint — MeshGraph Generate
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

POST /api/generate/rollout/{session_id}/{candidate_id}
    Runs a 50-step GNN rollout on the cached graph, saves a pkl to result/,
    and returns {"pkl_filename": "generate_<session>_<id>.pkl"}.

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
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import AsyncGenerator, Optional

import numpy as np
import torch
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from api.state import DOMAINS, get_gnn_scorer

router = APIRouter()

# In-memory thumbnail cache: session_id → {candidate_id → png_bytes}
# Bounded at _THUMBNAIL_CACHE_MAX_SESSIONS entries; oldest sessions are evicted.
_THUMBNAIL_CACHE_MAX_SESSIONS = 100
_thumbnail_cache: dict[str, dict[int, bytes]] = {}
_cache_order: list[str] = []  # insertion-order LRU tracker

# Graph cache: session_id → {candidate_id → PyG Data}  (for rollout endpoint)
_graph_cache: dict[str, dict] = {}


def _cache_session(session_id: str, thumbnails: dict[int, bytes]) -> None:
    """Insert a session into the thumbnail cache, evicting the oldest if over capacity."""
    if session_id in _thumbnail_cache:
        _cache_order.remove(session_id)
    elif len(_thumbnail_cache) >= _THUMBNAIL_CACHE_MAX_SESSIONS:
        oldest = _cache_order.pop(0)
        _thumbnail_cache.pop(oldest, None)
        _graph_cache.pop(oldest, None)
    _thumbnail_cache[session_id] = thumbnails
    _cache_order.append(session_id)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    domain:           str   = "cylinder_flow"
    target_value:     float = 0.025    # drag (CFD) or stress (cloth)
    n_candidates:     int   = 5
    method:           str   = "sample"  # "sample" | "gradient"
    device:           str   = "cpu"
    mode:             str   = "quick"   # "quick" | "deep"


# ---------------------------------------------------------------------------
# Candidate DTO
# ---------------------------------------------------------------------------

@dataclass
class CandidateResult:
    """One generated design candidate."""
    id:                   int
    domain:               str
    predicted_value:      float    # drag (CFD) or stress (cloth) from MLP surrogate
    target_value:         float
    ood_confidence:       float    # 1.0 = in-distribution; -1.0 = unavailable
    is_ood:               bool
    mesh_nodes:           int
    params:               dict     # domain-specific params
    # Deep mode fields — all None/False in quick mode
    gnn_predicted_value:  float | None = None
    score_gap:            float | None = None   # |surrogate - gnn|
    gnn_converged:        bool  | None = None   # False if 200-step cap hit
    gnn_failed:           bool         = False  # True if rollout threw


# ---------------------------------------------------------------------------
# Abstract sampler interface
# ---------------------------------------------------------------------------

class BaseDesignSampler(ABC):
    """Abstract interface for domain-specific design generation."""

    @abstractmethod
    def sample(self, target: float, n: int, device: str,
               method: str = "sample") -> list[CandidateResult]:
        """
        Generate n design candidates aiming for the given physics target.

        Args:
            target:  target physics value (drag or stress)
            n:       number of candidates to return
            device:  torch device string
            method:  "sample"   — draw n independent samples from CVAE prior
                     "gradient" — optimise a single latent code via gradient
                                  descent against a differentiable surrogate,
                                  then generate n variations around that code
        """


# ---------------------------------------------------------------------------
# CFD sampler
# ---------------------------------------------------------------------------

class CFDDesignSampler(BaseDesignSampler):
    """Samples cylinder_flow designs using the CFD CVAE."""

    CFD_CVAE_PATH  = "checkpoints/cfd_cvae.pth"
    SURROGATE_PATH = "checkpoints/drag_surrogate.pth"

    def sample(self, target: float, n: int, device: str,
               method: str = "sample", mode: str = "quick") -> tuple[list, list]:
        """
        Generate n candidates. mode='quick' uses MLP surrogate only.
        mode='deep' adds GNN adaptive rollout scoring after sampling.
        """
        results, trajectory = self._quick_sample(target, n, device, method)

        if mode == "deep":
            cfd_ckpt = DOMAINS["cylinder_flow"]["checkpoint"]
            if not os.path.exists(cfd_ckpt):
                import logging
                logging.getLogger(__name__).warning(
                    "Deep mode requested but GNN checkpoint not found at %s. "
                    "Returning quick-mode results.", cfd_ckpt
                )
                return results, trajectory

            scorer = get_gnn_scorer(cfd_ckpt, device=device)
            graphs = [g for _, g in results]
            gnn_scores = scorer.score_candidates(graphs, device=device)

            import math
            updated = []
            for (c, g), score in zip(results, gnn_scores):
                if math.isnan(score.gnn_predicted_value):
                    c.gnn_failed = True
                else:
                    c.gnn_predicted_value = score.gnn_predicted_value
                    c.score_gap           = abs(c.predicted_value - score.gnn_predicted_value)
                    c.gnn_converged       = score.converged
                updated.append((c, g))
            results = updated

        return results, trajectory

    def _quick_sample(self, target: float, n: int, device: str,
                      method: str = "sample") -> tuple[list, list]:
        import torch
        import torch.nn.functional as F
        from extensions.generative.cvae_cfd import CVAETrainer
        from extensions.generative.drag_surrogate import DragSurrogateTrainer
        from extensions.generative.mesh_generator import CFDMeshBuilder
        from extensions.confidence.ood_detector import OODDetector
        from api.state import get_model, DOMAINS

        if not os.path.exists(self.CFD_CVAE_PATH):
            raise HTTPException(
                status_code=503,
                detail="CFD CVAE not trained yet. "
                       "Run: python extensions/generative/train_cvae.py --domain cylinder_flow"
            )

        surrogate_trainer = DragSurrogateTrainer.load(self.SURROGATE_PATH, device=device)
        trainer = CVAETrainer.load(self.CFD_CVAE_PATH,
                                   surrogate=surrogate_trainer._model,
                                   device=device)

        # ── Generate params via chosen method ────────────────────────────────
        trajectory: list[float] = []

        if method == "gradient":
            params_phys, trajectory = self._gradient_sample(
                trainer, surrogate_trainer, target, n, device
            )
        else:
            params_phys = trainer.generate(target_drag_physical=target, n=n)

        # ── OOD detector (optional) ──────────────────────────────────────────
        cfd_ckpt = DOMAINS["cylinder_flow"]["checkpoint"]
        detector = None
        try:
            sim = get_model(cfd_ckpt, device=device) if os.path.exists(cfd_ckpt) else None
            if sim is not None and os.path.exists("runs/embedding_index.pkl"):
                detector = OODDetector.from_index_file(
                    "runs/embedding_index.pkl", simulator=sim, device=device
                )
        except Exception as _ood_exc:
            import logging as _log
            _log.getLogger(__name__).warning(
                "OOD detector failed to initialise (confidence will show N/A): %s",
                _ood_exc, exc_info=True
            )

        # ── Build meshes ─────────────────────────────────────────────────────
        builder    = CFDMeshBuilder()
        candidates = []

        for i, row in enumerate(params_phys):
            cx, cy, r, v_in = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            r    = float(np.clip(r,    0.01, 0.15))
            cx   = float(np.clip(cx,   r + 0.01, 1.6 - r - 0.01))
            cy   = float(np.clip(cy,   r + 0.01, 0.41 - r - 0.01))
            v_in = float(np.clip(v_in, 0.05, 2.0))

            try:
                graph = builder.build(cx, cy, r, v_in)
            except Exception:
                continue

            pred_drag = float(surrogate_trainer.predict(
                np.array([[cx, cy, r, v_in]], dtype=np.float32))[0])

            ood_conf, is_ood = -1.0, False
            if detector is not None:
                try:
                    res = detector.score(graph)
                    ood_conf, is_ood = res.confidence, res.is_ood
                except Exception:
                    pass

            candidates.append((CandidateResult(
                id=i, domain="cylinder_flow",
                predicted_value=pred_drag, target_value=target,
                ood_confidence=ood_conf, is_ood=is_ood,
                mesh_nodes=graph.num_nodes,
                params={"cx": cx, "cy": cy, "r": r, "v_inlet": v_in},
            ), graph))

        candidates.sort(key=lambda x: abs(x[0].predicted_value - target))
        return [(c, g) for c, g in candidates[:n]], trajectory

    # ── Gradient descent in CVAE latent space ───────────────────────────────

    def _gradient_sample(self, trainer, surrogate_trainer,
                         target: float, n: int, device: str):
        """
        Gradient descent in CVAE latent space using the differentiable surrogate.

        Full chain (every step differentiable w.r.t. z):
            z [16]
            → CVAE decoder  → params_norm [4]     (linear layers)
            → denorm        → params_phys [4]     (affine, differentiable)
            → surrogate norm→ params_surr [4]     (affine, differentiable)
            → DragSurrogate → drag_pred   [1]     (MLP forward, no no_grad)
            → MSE loss = (drag_pred - target)²

        After convergence the optimal z* is found.
        n diverse candidates are produced by sampling z ~ N(z*, σ²I)
        so the output shares the same design intent but varies geometrically.
        """
        import torch
        import torch.nn.functional as F

        model     = trainer._model.to(device)
        surrogate = surrogate_trainer._model.to(device)
        sc_cvae   = trainer._scaler
        sc_surr   = surrogate_trainer._scaler

        model.eval()
        surrogate.eval()

        drag_norm = float(
            (target - sc_cvae.drag_min) / (sc_cvae.drag_max - sc_cvae.drag_min + 1e-8)
        )
        target_t = torch.tensor([[drag_norm]], dtype=torch.float32, device=device)
        target_s = torch.tensor(target, dtype=torch.float32, device=device)

        # Affine constants as tensors (so autograd flows through denorm)
        p_min = torch.from_numpy(sc_cvae.param_min.astype(np.float32)).to(device)
        p_max = torch.from_numpy(sc_cvae.param_max.astype(np.float32)).to(device)
        x_min = torch.from_numpy(sc_surr.x_min.astype(np.float32)).to(device)
        x_max = torch.from_numpy(sc_surr.x_max.astype(np.float32)).to(device)
        y_min = float(sc_surr.y_min)
        y_max = float(sc_surr.y_max)

        def z_to_drag(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """z [L] → (drag_phys [scalar], params_phys [4])"""
            params_n   = model.decoder(z.unsqueeze(0), target_t)         # [1, 4]
            params_n   = torch.clamp(params_n, 0.0, 1.0)
            params_p   = params_n * (p_max - p_min) + p_min              # [1, 4] physical
            params_s   = (params_p - x_min) / (x_max - x_min + 1e-8)    # [1, 4] surrogate-norm
            drag_n_out = surrogate(params_s)                              # [1] normed drag
            drag_phys  = drag_n_out * (y_max - y_min) + y_min            # [1] physical
            return drag_phys.squeeze(), params_p.squeeze(0)

        # Multi-restart Adam optimisation
        n_restarts = 3
        n_iters    = 150
        lr         = 0.05

        best_z    = None
        best_err  = float("inf")
        best_traj: list[float] = []

        for _ in range(n_restarts):
            z     = torch.randn(trainer._cfg.latent_dim, device=device, requires_grad=True)
            optim = torch.optim.Adam([z], lr=lr)
            traj: list[float] = []

            for _ in range(n_iters):
                optim.zero_grad()
                drag_pred, _ = z_to_drag(z)
                loss = F.mse_loss(drag_pred, target_s)
                loss.backward()
                optim.step()
                traj.append(float(drag_pred.detach().item()))

            err = abs(traj[-1] - target)
            if err < best_err:
                best_err  = err
                best_z    = z.detach().clone()
                best_traj = traj

        # Sample n diverse candidates around optimal z
        noise_scale = 0.25
        results     = []
        with torch.no_grad():
            for _ in range(n):
                z_p = best_z + torch.randn_like(best_z) * noise_scale
                _, params_p = z_to_drag(z_p)
                results.append(params_p.cpu().numpy())

        return np.array(results), best_traj


# ---------------------------------------------------------------------------
# Cloth sampler
# ---------------------------------------------------------------------------

class ClothDesignSampler(BaseDesignSampler):
    """Samples flag_simple designs using the Cloth CVAE."""

    CVAE_PATH      = "checkpoints/flag-simple_cvae.pth"
    PCA_PATH       = "data_flag/train/cloth_pca.pkl"
    STRESS_PATH    = "data_flag/train/cloth_stress.npy"
    REF_TRAJ       = "data_flag/train/traj_00000.npz"

    def sample(self, target: float, n: int, device: str,
               method: str = "sample") -> tuple[list, list]:
        from extensions.generative.cvae_cloth import (
            ClothCVAETrainer, StressSurrogate, StressSurrogateTrainer
        )
        from extensions.generative.cloth_extractor import PosePCA
        from extensions.generative.mesh_generator import ClothMeshBuilder

        if not os.path.exists(self.CVAE_PATH):
            raise HTTPException(
                status_code=503,
                detail="Cloth CVAE not trained yet. "
                       "Run: python extensions/generative/train_cvae.py --domain flag_simple"
            )

        pca = PosePCA.load(self.PCA_PATH)

        import torch as _torch
        _peek    = _torch.load(self.CVAE_PATH, map_location=device, weights_only=False)
        pose_dim = (_peek.get("cfg_dict") or {}).get("pose_dim", 16)

        s_trainer = StressSurrogateTrainer(StressSurrogate(pose_dim=pose_dim), device=device)
        trainer   = ClothCVAETrainer.load(self.CVAE_PATH,
                                          stress_trainer=s_trainer, device=device)
        builder   = ClothMeshBuilder(reference_traj_path=self.REF_TRAJ)

        trajectory: list[float] = []

        if method == "gradient":
            world_poses, trajectory = self._gradient_sample(
                trainer, pca, target, n, device
            )
        else:
            world_poses = trainer.generate(target_stress=target, n=n, pca=pca)

        candidates = []
        for i, wp in enumerate(world_poses):
            graph = builder.build(wp)
            # Use the stress surrogate to get a predicted stress for this pose.
            # Recover the normalised PCA coords the trainer would have used:
            pca_coords = pca.transform(wp.reshape(1, -1))   # [1, K]
            pred_stress = float(s_trainer.predict(pca_coords.astype(np.float32))[0])
            candidates.append((CandidateResult(
                id=i, domain="flag_simple",
                predicted_value=pred_stress,
                target_value=target,
                ood_confidence=-1.0,
                is_ood=False,
                mesh_nodes=graph.num_nodes,
                params={"world_pos_norm": float(np.linalg.norm(wp))},
            ), graph))

        return candidates, trajectory

    def _gradient_sample(self, trainer, pca, target: float,
                         n: int, device: str) -> tuple[np.ndarray, list[float]]:
        """
        Gradient descent in cloth CVAE latent space via ClothInverseDesigner.

        The cloth pipeline is fully differentiable:
            z → decoder → pose_pca → PCA⁻¹ → world_pos → FlagSimulator → stress_loss

        Requires flag_best_model.pth to be present.
        Falls back to CVAE sampling if the simulator checkpoint is missing.
        """
        from api.state import DOMAINS
        sim_ckpt = DOMAINS["flag_simple"]["checkpoint"]

        if not os.path.exists(sim_ckpt):
            # Simulator checkpoint missing — log clearly and fall back to sampling.
            # Callers should surface this to the user rather than silently substituting.
            import logging
            logging.getLogger(__name__).warning(
                "Cloth gradient descent requested but simulator checkpoint not found at %s. "
                "Falling back to CVAE sampling (no gradient optimisation).", sim_ckpt
            )
            world_poses = trainer.generate(target_stress=target, n=n, pca=pca)
            return world_poses, []

        import torch
        from model.flag_simulator import FlagSimulator
        from extensions.generative.inverse_design import (
            ClothInverseDesigner, StressObjective
        )
        from utils.utils import NodeType

        # Load simulator
        sim_ckpt_data = torch.load(sim_ckpt, map_location=device, weights_only=False)
        simulator     = FlagSimulator(message_passing_num=15, device=device)
        simulator.load_state_dict(sim_ckpt_data["model_state_dict"])
        simulator.eval()

        # Build stress objective
        ref         = np.load(self.REF_TRAJ)
        N           = ref["mesh_pos"].shape[0]
        mp_3d       = np.concatenate([ref["mesh_pos"],
                                       np.zeros((N, 1), dtype=np.float32)], axis=-1)
        mesh_rest   = torch.from_numpy(mp_3d).to(device)
        nt          = ref["node_type"].squeeze(-1)
        normal_mask = torch.from_numpy(nt == int(NodeType.NORMAL)).to(device)
        objective   = StressObjective(target_stress=target,
                                      mesh_rest=mesh_rest,
                                      normal_mask=normal_mask)

        designer = ClothInverseDesigner(
            cvae_trainer=trainer, flag_simulator=simulator,
            objective=objective, reference_traj_path=self.REF_TRAJ, device=device
        )
        result = designer.optimise(
            target_stress=target, n_iters=80, lr=0.05,
            n_restarts=2, pca=pca, verbose=False
        )

        # Decode best_z into n diverse world_poses
        noise_scale = 0.2
        world_poses = []
        with torch.no_grad():
            best_z = torch.from_numpy(result.best_z).to(device)
            sc     = trainer._scaler
            target_n = float(
                (target - sc.stress_min) / (sc.stress_max - sc.stress_min + 1e-8)
            )
            target_t = torch.tensor([[target_n]], dtype=torch.float32, device=device)
            p_min    = torch.from_numpy(sc.pose_min.astype(np.float32)).to(device)
            p_max    = torch.from_numpy(sc.pose_max.astype(np.float32)).to(device)

            for _ in range(n):
                z_p      = best_z + torch.randn_like(best_z) * noise_scale
                pose_n   = trainer._model.decoder(z_p.unsqueeze(0), target_t)
                pose_p   = (pose_n * (p_max - p_min) + p_min).squeeze(0).cpu().numpy()
                world_pos = pca.inverse_transform(pose_p.reshape(1, -1)).reshape(N, 3)
                world_poses.append(world_pos.astype(np.float32))

        return world_poses, result.trajectory


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
        """Render CFD mesh with triangulated edges and node-type colour coding.

        Triangles whose centroids fall inside the cylinder are filtered out
        (FaceToEdge consumed the original face; we re-triangulate from pos and
        then apply the centroid test using the known cx/cy/r).
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.tri as mtri
        from scipy.spatial import Delaunay

        from utils.utils import NodeType

        pos = graph.pos.numpy()    # [N, 2]
        x   = graph.x.numpy()
        nt  = x[:, 0].astype(int)

        # Re-triangulate from pos (face was consumed by FaceToEdge transform)
        tri  = Delaunay(pos)
        face = tri.simplices         # [F, 3]

        # Filter triangles whose centroid falls inside the cylinder
        if cx is not None and cy is not None and r is not None:
            centroids = pos[face].mean(axis=1)    # [F, 2]
            d_cent    = np.sqrt((centroids[:, 0] - cx) ** 2 +
                                (centroids[:, 1] - cy) ** 2)
            face = face[d_cent >= r]

        # Use velocity magnitude (first two output dims) for heatmap colouring.
        # graph.x columns for cylinder_flow: [node_type, vx_t-5..vx_t, vy_t-5..vy_t]
        # Columns 1–5 = vx history, 6–10 = vy history; use the latest (col 5, 10).
        if x.shape[1] >= 11:
            vx = x[:, 5].astype(float)
            vy = x[:, 10].astype(float)
            vel_mag = np.sqrt(vx ** 2 + vy ** 2)
        else:
            vel_mag = np.zeros(len(pos))

        triang = mtri.Triangulation(pos[:, 0], pos[:, 1], face)

        fig, ax = plt.subplots(figsize=(4, 2.4), dpi=100)
        ax.set_facecolor("#0f172a")
        fig.patch.set_facecolor("#0f172a")

        # Flat-shaded heatmap — matches the Visualize/Analyze page (per-triangle average,
        # auto-normalised to the velocity range of this frame).
        vmin = float(vel_mag.min())
        vmax = float(vel_mag.max()) or 1.0   # guard against all-zero initial condition
        ax.tripcolor(triang, vel_mag, cmap="turbo", shading="flat",
                     vmin=vmin, vmax=vmax)

        ax.set_aspect("equal")
        ax.axis("off")

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
    MeshGraph Generate — sample novel designs + predict physics.

    Returns a Server-Sent Events (SSE) stream:
        event: candidate   data: { CandidateResult fields... }
        event: gnn_score   data: { id, gnn_predicted_value, score_gap, ... }
        event: warning     data: { "detail": "..." }
        event: error       data: { "detail": "..." }
        event: done        data: { "best_id": int }
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

            # ── Phase 1: quick sample (CVAE + surrogate) ──────────────────────
            # Run in thread pool to avoid blocking the event loop.
            loop = asyncio.get_running_loop()
            results, trajectory = await loop.run_in_executor(
                None,
                lambda: sampler._quick_sample(req.target_value,
                                              req.n_candidates,
                                              req.device,
                                              req.method)
                if hasattr(sampler, "_quick_sample")
                else sampler.sample(req.target_value,
                                    req.n_candidates,
                                    req.device,
                                    req.method,
                                    "quick")
            )

            # Stream optimisation trajectory first (gradient mode only)
            if trajectory:
                yield _sse_event("trajectory", {"values": trajectory})
                await asyncio.sleep(0)

            # ── Phase 2: render thumbnails + stream quick candidates ───────────
            session_thumbs: dict[int, bytes] = {}
            thumbnail_urls: dict[int, str | None] = {}
            best_id  = 0
            best_err = float("inf")

            for c, graph in results:
                try:
                    if req.domain == "cylinder_flow":
                        p = c.params
                        png = ThumbnailRenderer.render_cfd(
                            graph,
                            cx=p.get("cx"), cy=p.get("cy"), r=p.get("r"),
                        )
                    else:
                        png = ThumbnailRenderer.render_cloth(graph)
                    session_thumbs[c.id] = png
                    thumbnail_urls[c.id] = f"/api/generate/thumbnail/{session_id}/{c.id}"
                except Exception:
                    thumbnail_urls[c.id] = None

                err = abs(c.predicted_value - c.target_value)
                if err < best_err:
                    best_err = err
                    best_id  = c.id

            # Cache all thumbnails before streaming so every URL is live when
            # the browser receives the first candidate event.
            _cache_session(session_id, session_thumbs)
            # Also cache graphs for the rollout endpoint.
            # Tag each graph with its domain so the rollout endpoint can branch correctly.
            for c, graph in results:
                graph.domain = c.domain
            _graph_cache[session_id] = {c.id: graph for c, graph in results}

            for c, _ in results:
                payload                  = asdict(c)
                payload["thumbnail_url"] = thumbnail_urls[c.id]
                payload["session_id"]    = session_id
                yield _sse_event("candidate", payload)
                await asyncio.sleep(0)

            # ── Phase 3 (deep mode): GNN score each candidate one at a time ───
            # Stream a `gnn_score` event after each rollout so the UI updates
            # progressively instead of waiting for all candidates to finish.
            if req.mode == "deep" and req.domain == "cylinder_flow":
                from api.state import get_gnn_scorer, DOMAINS
                import math

                cfd_ckpt = DOMAINS["cylinder_flow"]["checkpoint"]
                if not os.path.exists(cfd_ckpt):
                    yield _sse_event("warning", {
                        "detail": "GNN checkpoint not found — results are surrogate-only."
                    })
                else:
                    scorer = get_gnn_scorer(cfd_ckpt, device=req.device)
                    scorer.simulator.eval()

                    for idx, (c, graph) in enumerate(results):
                        # Score one candidate in a thread so the event loop stays live
                        try:
                            gnn_score = await loop.run_in_executor(
                                None,
                                lambda g=graph: scorer._adaptive_rollout(g, req.device)
                            )
                            if math.isnan(gnn_score.gnn_predicted_value):
                                raise ValueError("NaN drag")
                            gnn_val  = gnn_score.gnn_predicted_value
                            gap      = abs(c.predicted_value - gnn_val)
                            converged = gnn_score.converged
                            failed   = False
                        except Exception as exc:
                            import logging
                            logging.getLogger(__name__).warning(
                                "GNN rollout failed for candidate %d: %s", idx, exc)
                            gnn_val  = None
                            gap      = None
                            converged = None
                            failed   = True

                        yield _sse_event("gnn_score", {
                            "id":                  c.id,
                            "gnn_predicted_value": gnn_val,
                            "score_gap":           gap,
                            "gnn_converged":       converged,
                            "gnn_failed":          failed,
                            "candidate_index":     idx,
                            "total_candidates":    len(results),
                        })
                        await asyncio.sleep(0)

            yield _sse_event("done", {"best_id": best_id, "session_id": session_id})

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


# ---------------------------------------------------------------------------
# Rollout endpoint — run GNN on a cached generated candidate
# ---------------------------------------------------------------------------

@router.post("/generate/rollout/{session_id}/{candidate_id}")
async def generate_rollout(session_id: str, candidate_id: int,
                           n_steps: int = 50, device: str = "cpu"):
    """
    Run a short GNN rollout on a previously generated candidate graph,
    save a pkl to result/, and return the filename so the caller can
    open /visualize?file=<filename>.

    Supports both CFD (cylinder_flow) and cloth (flag_simple) domains.
    The graph must be in the in-memory _graph_cache, which is populated
    during the /generate SSE call and lives for the session lifetime
    (up to 100 sessions, LRU-evicted).
    """
    import pickle
    import logging

    session_graphs = _graph_cache.get(session_id)
    if session_graphs is None:
        raise HTTPException(status_code=404,
                            detail="Session not found — regenerate candidates first.")
    graph = session_graphs.get(candidate_id)
    if graph is None:
        raise HTTPException(status_code=404, detail="Candidate not found in session.")

    # Determine domain from the cached graph (set during generation)
    domain = getattr(graph, "domain", "cylinder_flow")

    from api.state import get_model, DOMAINS
    loop = asyncio.get_running_loop()

    if domain == "flag_simple":
        # ── Cloth rollout ──────────────────────────────────────────────────────
        cloth_ckpt = DOMAINS["flag_simple"]["checkpoint"]
        if not os.path.exists(cloth_ckpt):
            raise HTTPException(status_code=503,
                                detail="Cloth GNN checkpoint not found — cannot run rollout.")

        def _run_cloth_rollout():
            import torch
            import torch_geometric.transforms as T
            from model.flag_simulator import FlagSimulator

            ckpt_data = torch.load(cloth_ckpt, map_location=device, weights_only=False)
            sim = FlagSimulator(message_passing_num=15, device=device)
            sd = ckpt_data["model_state_dict"]
            cur_sd = sim.state_dict()
            filtered = {k: v for k, v in sd.items()
                        if k in cur_sd and cur_sd[k].shape == v.shape}
            sim.load_state_dict(filtered, strict=False)
            sim.eval()

            g = graph.clone()
            if g.edge_index is None:
                transformer = T.Compose([
                    T.FaceToEdge(), T.Cartesian(norm=False), T.Distance(norm=False)
                ])
                g = transformer(g)
            g = g.to(device)

            # Cloth node feature layout: x[:,0:3]=world_pos, x[:,3]=node_type
            # graph.world_pos and graph.prev_x are set by ClothMeshBuilder
            if hasattr(g, "world_pos") and g.world_pos is not None:
                cur_world  = g.world_pos.clone().to(device)   # [N, 3]
                prev_world = g.prev_x.clone().to(device)      # [N, 3]
            else:
                # Fallback: extract world_pos from x columns 0:3
                cur_world  = g.x[:, :3].clone()
                prev_world = cur_world.clone()

            node_type = g.x[:, 3].long()

            predicted_frames = []
            with torch.no_grad():
                for _ in range(n_steps):
                    # Rebuild node features with current positions
                    g.x = torch.cat([cur_world, node_type.unsqueeze(-1).float()], dim=-1)
                    g.world_pos = cur_world
                    g.prev_x    = prev_world

                    next_world = sim(g)        # FlagSimulator returns next world_pos [N, 3]
                    # Note: FlagSimulator.forward() already pins handle/boundary nodes
                    # internally via Verlet + handle_mask, so no external masking needed.

                    predicted_frames.append(next_world.cpu().numpy())   # [N, 3]
                    prev_world = cur_world
                    cur_world  = next_world

            import numpy as np
            predicted_arr = np.stack(predicted_frames)    # [T, N, 3]
            targets_arr   = predicted_arr.copy()          # no ground truth — use predicted as target

            # crds = 2D mesh coordinates (UV layout), shape [N, 2]
            crds = graph.pos.cpu().numpy()                # [N, 2]

            os.makedirs("result", exist_ok=True)
            filename = f"generate_{session_id[:8]}_{candidate_id}.pkl"
            pkl_path = os.path.join("result", filename)
            with open(pkl_path, "wb") as f:
                pickle.dump(
                    [[predicted_arr, targets_arr], crds, {
                        "domain":           "flag_simple",
                        "target_field":     "world_pos",
                        "confidence_score": None,
                        "is_generate":      True,   # no ground truth — prediction only
                    }],
                    f,
                )
            logging.getLogger(__name__).info("Saved cloth generate rollout to %s", pkl_path)
            return filename

        try:
            filename = await loop.run_in_executor(None, _run_cloth_rollout)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Cloth rollout failed: {exc}")

    else:
        # ── CFD rollout (cylinder_flow) ────────────────────────────────────────
        cfd_ckpt = DOMAINS["cylinder_flow"]["checkpoint"]
        if not os.path.exists(cfd_ckpt):
            raise HTTPException(status_code=503,
                                detail="GNN checkpoint not found — cannot run rollout.")

        def _run_cfd_rollout():
            import torch
            import torch_geometric.transforms as T
            from utils.utils import NodeType

            sim = get_model(cfd_ckpt, device=device)
            sim.eval()

            g = graph.clone()
            if g.edge_index is None:
                transformer = T.Compose([
                    T.FaceToEdge(), T.Cartesian(norm=False), T.Distance(norm=False)
                ])
                g = transformer(g)
            g = g.to(device)

            node_type     = g.x[:, 0].long()
            current_vel   = g.x[:, 1:3].clone()
            original_x    = g.x.clone()
            boundary_mask = ~((node_type == int(NodeType.NORMAL)) |
                              (node_type == int(NodeType.OUTFLOW)))

            predicted_frames = []
            with torch.no_grad():
                for _ in range(n_steps):
                    g.x = original_x.clone()
                    g.x[:, 1:3] = current_vel
                    next_vel = sim(g, velocity_sequence_noise=None)
                    next_vel[boundary_mask] = current_vel[boundary_mask]
                    current_vel = next_vel
                    predicted_frames.append(current_vel.cpu().numpy())   # [N, 2]

            import numpy as np
            predicted_arr = np.stack(predicted_frames)    # [T, N, 2]
            targets_arr   = predicted_arr.copy()          # no ground truth
            crds          = graph.pos.numpy()             # [N, 2]

            os.makedirs("result", exist_ok=True)
            filename = f"generate_{session_id[:8]}_{candidate_id}.pkl"
            pkl_path = os.path.join("result", filename)
            with open(pkl_path, "wb") as f:
                pickle.dump(
                    [[predicted_arr, targets_arr], crds, {
                        "domain":           "cylinder_flow",
                        "target_field":     "velocity",
                        "confidence_score": None,
                        "is_generate":      True,   # no ground truth — prediction only
                    }],
                    f,
                )
            logging.getLogger(__name__).info("Saved generate rollout to %s", pkl_path)
            return filename

        try:
            filename = await loop.run_in_executor(None, _run_cfd_rollout)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Rollout failed: {exc}")

    return {"pkl_filename": filename}
