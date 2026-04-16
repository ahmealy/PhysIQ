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
from typing import Any, AsyncGenerator, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from api.state import DOMAINS

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


# ---------------------------------------------------------------------------
# Abstract sampler interface
# ---------------------------------------------------------------------------

class BaseDesignSampler(ABC):
    """Abstract interface for domain-specific design generation."""

    @abstractmethod
    def sample(self, target: float, n: int, device: str,
               method: str = "sample") -> tuple[list[CandidateResult], list]:
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

    def __init__(self) -> None:
        import logging
        self._mesh_lookup = None   # RealMeshLookup (nearest real mesh by geometry)
        self._simulator   = None   # GNN Simulator (for GNN-in-the-loop gradient)
        self._dataset     = None   # FpcDataset (training split, for mesh loading)
        try:
            from extensions.generative.mesh_generator import RealMeshLookup
            self._mesh_lookup = RealMeshLookup(dataset_path='data')
        except Exception as e:
            logging.getLogger(__name__).warning(
                "RealMeshLookup unavailable: %s", e
            )
        try:
            from api.state import get_model, DOMAINS
            cfd_ckpt = DOMAINS["cylinder_flow"]["checkpoint"]
            if os.path.exists(cfd_ckpt):
                self._simulator = get_model(cfd_ckpt, device='cpu')
                self._simulator.eval()
        except Exception as e:
            logging.getLogger(__name__).warning(
                "GNN simulator unavailable for gradient coupling: %s", e
            )
        try:
            from dataset import FpcDataset
            self._dataset = FpcDataset(data_root='data', split='train')
        except Exception as e:
            logging.getLogger(__name__).warning(
                "Training dataset unavailable for mesh lookup: %s", e
            )
        # Param-space OOD confidence (no mesh or GNN needed)
        try:
            from extensions.confidence.ood_detector import ParamSpaceOOD as _ParamSpaceOOD
            self._param_ood: Optional[_ParamSpaceOOD] = _ParamSpaceOOD(dataset_path='data')
            if not self._param_ood.available:
                logging.getLogger(__name__).warning(
                    "ParamSpaceOOD: design_params.npy not found — confidence will be N/A"
                )
        except Exception as _e:
            logging.getLogger(__name__).warning("ParamSpaceOOD unavailable: %s", _e)
            self._param_ood = None

    def sample(self, target: float, n: int, device: str,
               method: str = "sample") -> tuple[list, list]:
        """
        Generate n candidates using the CVAE + MLP surrogate.
        """
        import torch
        from extensions.generative.cvae_cfd import CVAETrainer
        from extensions.generative.drag_surrogate import DragSurrogateTrainer
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

        # ── Build meshes ─────────────────────────────────────────────────────
        candidates = []

        for i, row in enumerate(params_phys):
            cx, cy, r, v_in = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            r    = float(np.clip(r,    0.01, 0.15))
            cx   = float(np.clip(cx,   r + 0.01, 1.6 - r - 0.01))
            cy   = float(np.clip(cy,   r + 0.01, 0.41 - r - 0.01))
            v_in = float(np.clip(v_in, 0.05, 2.0))

            # Load real mesh for thumbnail (no gradient needed here)
            graph = None
            if (self._mesh_lookup is not None and self._mesh_lookup.available
                    and self._dataset is not None):
                try:
                    _traj_idx = self._mesh_lookup.find_nearest(cx, cy, r)
                    import torch as _torch
                    _dummy_vin = _torch.tensor([v_in], dtype=_torch.float32)
                    graph = self._mesh_lookup.load_mesh_for_trajectory(
                        _traj_idx, _dummy_vin, self._dataset, device='cpu'
                    )
                except Exception as _mesh_err:
                    import logging as _mlog
                    _mlog.getLogger(__name__).debug(
                        "Real mesh load failed for thumbnail: %s", _mesh_err
                    )
                    # graph stays None — thumbnail will be skipped

            pred_drag = float(surrogate_trainer.predict(
                np.array([[cx, cy, r, v_in]], dtype=np.float32))[0])

            ood_conf, is_ood = -1.0, False
            if self._param_ood is not None:
                try:
                    _ood_res = self._param_ood.score(cx, cy, r, v_in)
                    ood_conf, is_ood = _ood_res.confidence, _ood_res.is_ood
                except Exception:
                    pass

            candidates.append((CandidateResult(
                id=i, domain="cylinder_flow",
                predicted_value=pred_drag, target_value=target,
                ood_confidence=ood_conf, is_ood=is_ood,
                mesh_nodes=graph.num_nodes if graph is not None else 0,
                params={"cx": cx, "cy": cy, "r": r, "v_inlet": v_in},
            ), graph))

        candidates.sort(key=lambda x: abs(x[0].predicted_value - target))
        return [(c, g) for c, g in candidates[:n]], trajectory

    # ── Gradient descent in CVAE latent space ───────────────────────────────

    def _gradient_sample(self, trainer, surrogate_trainer,
                         target: float, n: int, device: str):
        """
        Gradient descent in CVAE latent space.

        When a GNN simulator and real-mesh lookup are available the chain is:

            z [L]
            → CVAE decoder  → params_phys [4]     (cx, cy, r, v_inlet)
            → RealMeshLookup.find_nearest(cx,cy,r) → trajectory index  (discrete)
            → load real OpenFOAM mesh + inject v_inlet differentiably
            → GNN rollout (K-1 detached steps + 1 differentiable step)
            → drag_proxy = mean(|vx|) over OUTFLOW nodes
            → MSE loss = (drag_proxy - target)²

        Gradient flows through:  v_inlet → GNN final step → OUTFLOW vx → loss.
        cx, cy, r have zero gradient (discrete mesh lookup) but are guided by
        the CVAE encoder's prior.

        Falls back to the DragSurrogate MLP when the simulator or mesh lookup
        is unavailable (no checkpoint / data).

        After convergence the optimal z* is found.
        n diverse candidates are produced by sampling z ~ N(z*, σ²I)
        so the output shares the same design intent but varies geometrically.
        """
        import logging
        import torch
        import torch.nn.functional as F
        from utils.utils import NodeType as _NodeType
        from extensions.generative.mesh_generator import _inject_scalar_differentiable

        _logger = logging.getLogger(__name__)

        model     = trainer._model.to(device)
        surrogate = surrogate_trainer._model.to(device)
        sc_cvae   = trainer._scaler
        sc_surr   = surrogate_trainer._scaler

        model.eval()
        surrogate.eval()

        # The GNN drag proxy (mean|vx| at OUTFLOW, units: m/s, typical 0.8–1.5)
        # lives in a completely different unit/scale from the surrogate drag proxy
        # (r·v_inlet²/(1-2r/H), units: m³/s², typical 0.01–0.15) and from the
        # physical drag coefficient Cd the user specifies.  Comparing either
        # against target_s directly via MSE drives the optimizer in the wrong
        # direction.  Additionally, cx/cy/r have zero gradient through the
        # discrete mesh lookup, so the GNN path only guides v_inlet anyway.
        # The surrogate normalises all four inputs and outputs to [0,1] and
        # back, giving the optimizer a well-scaled, fully differentiable signal.
        # → always use the surrogate path for gradient descent.
        _use_gnn = False

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

        # NodeType constants (avoids magic numbers in hot path)
        _OUTFLOW_TYPE = int(_NodeType.OUTFLOW)  # 5

        # Number of GNN rollout steps; only the last step is differentiable.
        # Keep K small (10) for speed — enough to propagate velocity downstream.
        K_ROLLOUT = 10

        def _gnn_drag(params_p: torch.Tensor) -> torch.Tensor:
            """
            Load nearest real mesh, inject v_inlet, run K-step GNN rollout,
            return drag proxy (scalar, differentiable w.r.t. v_inlet).

            params_p: [4] physical params (cx, cy, r, v_inlet) — detachable
                       except v_inlet which must keep requires_grad.
            """
            cx_val   = float(params_p[0].item())
            cy_val   = float(params_p[1].item())
            r_val    = float(params_p[2].item())
            v_in_t   = params_p[3:4]          # [1], keeps grad_fn

            traj_idx = self._mesh_lookup.find_nearest(cx_val, cy_val, r_val)
            graph    = self._mesh_lookup.load_mesh_for_trajectory(
                traj_idx, v_in_t, self._dataset, device=device
            )

            # Save masks now — sim.forward() overwrites graph.x in-place
            # with normalised node attributes, so we must capture them first.
            orig_x        = graph.x.detach().clone()       # [N, 3] original
            node_type_col = orig_x[:, 0]                   # [N]
            boundary_mask = ~(
                (node_type_col == 0) |                     # NORMAL
                (node_type_col == _OUTFLOW_TYPE)            # OUTFLOW
            )
            outflow_mask  = (node_type_col == _OUTFLOW_TYPE)
            inflow_mask   = (node_type_col == int(_NodeType.INFLOW))  # 4

            # Detached warm-up steps — advance the velocity field K-1 steps
            # without tracking gradients (saves memory and compute).
            current_x = orig_x.clone()              # [N, 3], no grad
            with torch.no_grad():
                for _ in range(K_ROLLOUT - 1):
                    graph.x = current_x
                    next_vel = self._simulator(graph, velocity_sequence_noise=None)
                    # Pin boundary nodes to their current values
                    next_vel[boundary_mask] = current_x[boundary_mask, 1:3]
                    current_x = torch.cat([
                        current_x[:, 0:1],           # node_type column
                        next_vel,                     # updated velocities [N, 2]
                    ], dim=1)                         # [N, 3]

            # Re-inject v_inlet into INFLOW nodes BEFORE the differentiable
            # final step.  This is the critical link: v_inlet → INFLOW velocity
            # → GNN message passing → OUTFLOW velocity → drag proxy → loss.
            # We splice differentiably via the shared helper.
            vx_warm = current_x[:, 1].clone()           # [N] detached warm-up vx
            if inflow_mask.any():
                vx_col = _inject_scalar_differentiable(vx_warm, inflow_mask, v_in_t)
            else:
                vx_col = vx_warm

            # Build final input x with differentiable vx for INFLOW nodes
            final_input_x = torch.cat([
                current_x[:, 0:1],                        # node_type [N, 1]
                vx_col.unsqueeze(1),                       # vx        [N, 1]
                current_x[:, 2:3],                        # vy        [N, 1]
            ], dim=1)                                      # [N, 3]

            # Final differentiable GNN step
            graph.x   = final_input_x
            final_vel = self._simulator(graph, velocity_sequence_noise=None)
            # Pin boundary nodes (detached) — INFLOW vx is already set above,
            # but the boundary pin would overwrite it with detached values.
            # Instead, use torch.where to zero out gradient at boundary nodes,
            # while keeping full gradient for OUTFLOW (and NORMAL) nodes.
            pin_vel   = final_input_x[:, 1:3].detach()    # [N, 2]
            keep_mask = ~boundary_mask                     # True for NORMAL + OUTFLOW
            final_vel = torch.where(
                keep_mask.unsqueeze(1).expand_as(final_vel),
                final_vel,
                pin_vel,
            )

            # Drag proxy: mean |vx| over OUTFLOW nodes
            if outflow_mask.any():
                drag_proxy = final_vel[outflow_mask, 0].abs().mean()
            else:
                drag_proxy = final_vel[:, 0].abs().mean()

            return drag_proxy

        def z_to_drag(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """z [L] → (drag_phys [scalar], params_phys [4])"""
            params_n   = model.decoder(z.unsqueeze(0), target_t)         # [1, 4]
            params_n   = torch.clamp(params_n, 0.0, 1.0)
            params_p   = params_n * (p_max - p_min) + p_min              # [1, 4] physical
            params_p   = params_p.squeeze(0)                              # [4]

            if _use_gnn:
                try:
                    drag_phys = _gnn_drag(params_p)
                    return drag_phys, params_p
                except (RuntimeError, ValueError, IndexError) as _gnn_err:
                    _logger.warning(
                        "GNN drag step failed (%s); falling back to surrogate.",
                        _gnn_err,
                        exc_info=True,
                    )
                    # fall through to surrogate

            # Surrogate fallback path
            params_s   = (params_p.unsqueeze(0) - x_min) / (x_max - x_min + 1e-8)
            drag_n_out = surrogate(params_s)                              # [1] normed drag
            drag_phys  = drag_n_out * (y_max - y_min) + y_min            # [1] physical
            return drag_phys.squeeze(), params_p

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

        # Sample n diverse candidates around optimal z.
        # noise_scale=0.10: at latent_dim=16, E[||noise||] = 0.10*sqrt(16)=0.40
        # which is a modest perturbation — keeps candidates close enough to
        # the optimum while still diversifying within the training distribution.
        noise_scale = 0.10
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
    PCA_PATH       = "data_flag/train/cloth_pca.npz"
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
        event: trajectory  data: { "values": [...] }
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

            # ── Phase 1: CVAE sampling / gradient optimisation ────────────────
            phase_label = (
                "Optimising in latent space…"
                if req.method == "gradient"
                else "Sampling from CVAE…"
            )
            yield _sse_event("progress", {
                "phase":   phase_label,
                "step":    1,            # "Generate design" step index
                "done":    0,
                "total":   req.n_candidates,
            })
            await asyncio.sleep(0)

            # Run in thread pool to avoid blocking the event loop.
            loop = asyncio.get_running_loop()
            results, trajectory = await loop.run_in_executor(
                None,
                lambda: sampler.sample(req.target_value,
                                       req.n_candidates,
                                       req.device,
                                       req.method)
            )

            # Stream optimisation trajectory first (gradient mode only)
            if trajectory:
                yield _sse_event("trajectory", {"values": trajectory})
                await asyncio.sleep(0)

            # ── Phase 2: render thumbnails and stream candidates one by one ───
            # We cache graphs up-front (needed by the rollout endpoint), but
            # render thumbnails and stream each candidate as soon as it's ready
            # so the UI shows progress instead of waiting for the full batch.
            _graph_cache[session_id] = {}
            for c, graph in results:
                if graph is not None:
                    graph.domain = c.domain
                    _graph_cache[session_id][c.id] = graph

            best_id  = 0
            best_err = float("inf")

            for i, (c, graph) in enumerate(results):
                yield _sse_event("progress", {
                    "phase": "Rendering meshes…",
                    "step":  3,          # "Check reliability" / final step index
                    "done":  i,
                    "total": len(results),
                })
                await asyncio.sleep(0)

                # Render thumbnail
                png: bytes | None = None
                try:
                    if req.domain == "cylinder_flow":
                        if graph is not None:
                            p   = c.params
                            png = ThumbnailRenderer.render_cfd(
                                graph,
                                cx=p.get("cx"), cy=p.get("cy"), r=p.get("r"),
                            )
                    else:
                        if graph is not None:
                            png = ThumbnailRenderer.render_cloth(graph)
                except Exception:
                    png = None

                if png is not None:
                    _cache_session(session_id, {c.id: png})
                    thumbnail_url: str | None = f"/api/generate/thumbnail/{session_id}/{c.id}"
                else:
                    thumbnail_url = None

                err = abs(c.predicted_value - c.target_value)
                if err < best_err:
                    best_err = err
                    best_id  = c.id

                payload                  = asdict(c)
                payload["thumbnail_url"] = thumbnail_url
                payload["session_id"]    = session_id
                yield _sse_event("candidate", payload)
                await asyncio.sleep(0)

            yield _sse_event("done", {"best_id": best_id, "session_id": session_id})

        except HTTPException as e:
            yield _sse_event("error", {"detail": e.detail})
        except Exception as e:
            yield _sse_event("error", {"detail": str(e)})

    async def event_stream_ssh() -> AsyncGenerator[str, None]:
        """Relay named-SSE events from generate_ssh.py running on the remote GPU via SSH.

        Uses shared NFS — thumbnails and graphs are saved to disk by generate_ssh.py
        using the same session_id, so no base64 encoding or scp needed.
        The Analyze button works because graphs are pickled to runs/graphs/{session_id}/.
        """
        import pickle
        import shlex
        import subprocess

        from api.routes.train import _build_ssh_prefix

        project_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        os.makedirs(os.path.join(project_root, "runs"), exist_ok=True)
        cfg_path = os.path.join(project_root, "runs", "ui_generate_config.json")

        # Pass the session_id so the remote script uses the same NFS paths.
        with open(cfg_path, "w") as _f:
            json.dump({
                "domain":       req.domain,
                "target_value": req.target_value,
                "n_candidates": req.n_candidates,
                "method":       req.method,
                "device":       "cuda:0",   # always GPU on remote
                "session_id":   session_id,
            }, _f)

        remote_cfg  = _load_remote_cfg()
        venv_py     = remote_cfg.get(
            "venv_python",
            "/home/ahmealy/.pyenv/versions/venv_gpu/bin/python",
        ).strip()
        ssh_prefix  = _build_ssh_prefix(remote_cfg)
        script_path = os.path.join(project_root, "generate_ssh.py")
        remote_cmd  = "cd %s && %s -u %s --config %s" % (
            shlex.quote(project_root),
            shlex.quote(venv_py),
            shlex.quote(script_path),
            shlex.quote(cfg_path),
        )
        cmd = ssh_prefix + [remote_cmd]

        loop = asyncio.get_running_loop()
        proc = await loop.run_in_executor(
            None,
            lambda: subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            ),
        )

        current_event: str | None = None
        try:
            while True:
                line = await loop.run_in_executor(None, proc.stdout.readline)
                if not line:
                    break
                line = line.rstrip("\n")

                if line.startswith("event: "):
                    current_event = line[7:]    # extract event name
                    yield line + "\n"
                elif line.startswith("data: "):
                    # NFS path: generate_ssh.py writes thumbnail URLs and graph
                    # files to disk directly using the shared filesystem.
                    # We only need to load the graphs into the API's _graph_cache
                    # once all candidates arrive (on "done"), so Analyze works.
                    if current_event == "done":
                        # Load graphs from disk into _graph_cache for Analyze.
                        try:
                            graph_dir = os.path.join(
                                project_root, "runs", "graphs", session_id
                            )
                            graph_cache: dict[int, Any] = {}
                            if os.path.isdir(graph_dir):
                                for fname in os.listdir(graph_dir):
                                    if fname.startswith("graph_") and fname.endswith(".pkl"):
                                        cid = int(fname[6:-4])
                                        with open(os.path.join(graph_dir, fname), "rb") as gf:
                                            graph_cache[cid] = pickle.load(gf)
                            if graph_cache:
                                _graph_cache[session_id] = graph_cache
                        except Exception:
                            pass  # graph loading is best-effort; Analyze will just be disabled
                    yield line + "\n"
                    current_event = None   # reset after data line
                elif not line:
                    yield "\n"   # blank line = SSE event terminator
                # silently drop other lines (SSH banner messages, etc.)

        finally:
            await loop.run_in_executor(None, proc.wait)

    # Dispatch to remote GPU via SSH if configured
    from api.routes.train import _load_remote_cfg, _remote_ssh_active
    _remote = _load_remote_cfg()
    if _remote_ssh_active(_remote):
        return StreamingResponse(
            event_stream_ssh(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

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
    """Return a PNG thumbnail for a generated candidate.

    Checks the in-memory cache first (local generation), then falls back to
    the NFS disk path written by generate_ssh.py (SSH generation).
    """
    session = _thumbnail_cache.get(session_id)
    if session is not None:
        png = session.get(candidate_id)
        if png is not None:
            return Response(content=png, media_type="image/png")

    # NFS fallback — SSH path writes PNGs to runs/thumbnails/{session_id}/cand_{id}.png
    _project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    nfs_path = os.path.join(_project_root, "runs", "thumbnails", session_id, f"cand_{candidate_id}.png")
    if os.path.exists(nfs_path):
        with open(nfs_path, "rb") as f:
            return Response(content=f.read(), media_type="image/png")

    raise HTTPException(status_code=404, detail="Thumbnail not found")


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
    graph = session_graphs.get(candidate_id) if session_graphs is not None else None

    # SSH/NFS fallback — generate_ssh.py saves graphs to disk under runs/graphs/
    if graph is None:
        _project_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        nfs_graph_path = os.path.join(
            _project_root, "runs", "graphs", session_id, f"graph_{candidate_id}.pkl"
        )
        if os.path.exists(nfs_graph_path):
            try:
                with open(nfs_graph_path, "rb") as _gf:
                    graph = pickle.load(_gf)
                # Populate in-memory cache so subsequent requests are fast
                if session_graphs is None:
                    _graph_cache[session_id] = {}
                _graph_cache[session_id][candidate_id] = graph
            except Exception as _e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to load graph from NFS cache: {_e}"
                )

    if session_graphs is None and graph is None:
        raise HTTPException(status_code=404,
                            detail="Session not found — regenerate candidates first.")
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

            _device = device
            if _device.startswith("cuda") and not torch.cuda.is_available():
                _device = "cpu"

            ckpt_data = torch.load(cloth_ckpt, map_location=_device, weights_only=False)
            sim = FlagSimulator(message_passing_num=15, device=_device)
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
            g = g.to(_device)

            # Cloth node feature layout: x[:,0:3]=world_pos, x[:,3]=node_type
            # graph.world_pos and graph.prev_x are set by ClothMeshBuilder
            if hasattr(g, "world_pos") and g.world_pos is not None:
                cur_world  = g.world_pos.clone().to(_device)   # [N, 3]
                prev_world = g.prev_x.clone().to(_device)      # [N, 3]
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

            # Fall back to CPU if the requested device isn't available locally
            # (common when the API runs on a CPU-only machine but the frontend
            # still has cuda:0 selected from a previous remote-GPU session).
            _device = device
            if _device.startswith("cuda") and not torch.cuda.is_available():
                _device = "cpu"

            sim = get_model(cfd_ckpt, device=_device)
            sim.eval()

            g = graph.clone()
            if g.edge_index is None:
                transformer = T.Compose([
                    T.FaceToEdge(), T.Cartesian(norm=False), T.Distance(norm=False)
                ])
                g = transformer(g)
            g = g.to(_device)

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
            import traceback
            logging.getLogger(__name__).error("generate_rollout CFD error:\n%s", traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Rollout failed: {exc}")

    return {"pkl_filename": filename}
