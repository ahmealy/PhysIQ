"""
Differentiable Inverse Design — Cloth Domain
=============================================
Gradient-based optimisation in the Cloth CVAE latent space.

The cloth simulation pipeline is fully differentiable:

    z [L]
    │ ClothDecoder
    ↓
    pose_pca [K]
    │ PCA inverse-transform + reshape
    ↓
    world_pos [N, 3]
    │ ClothMeshBuilder.build()     ← no non-differentiable step!
    ↓
    PyG Data
    │ FlagSimulator.forward() eval  ← MeshGraphNets (differentiable)
    ↓
    next_world_pos [N, 3]
    │ stress computation
    ↓
    stress_loss  →  ∂loss/∂z  via autograd

This module implements the optimisation loop and exposes a simple
``optimize_cloth()`` convenience function.

Design principles
-----------------
- **Single Responsibility**: ``ClothInverseDesigner`` handles optimisation;
  ``StressObjective`` computes the physics objective.
- **Open / Closed**: add new objective functions by subclassing
  ``BaseObjective`` without touching the optimiser.
- **Dependency Inversion**: the optimiser depends on the abstract
  ``BaseObjective``, not on a specific physics formula.
"""
from __future__ import annotations

import os
import sys
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from torch_geometric.data import Data   # noqa: E402


# ---------------------------------------------------------------------------
# Result DTO
# ---------------------------------------------------------------------------

@dataclass
class OptimisationResult:
    """Outcome of one inverse design run."""
    best_z:        np.ndarray   # [L] best latent code found
    best_stress:   float        # predicted stress at best_z
    target_stress: float        # the requested target
    trajectory:    list[float]  # stress per iteration
    n_iters:       int          # iterations actually run


# ---------------------------------------------------------------------------
# Objective interface (Open / Closed principle)
# ---------------------------------------------------------------------------

class BaseObjective(ABC):
    """Abstract physics objective for inverse design."""

    @abstractmethod
    def __call__(self, world_pos: torch.Tensor) -> torch.Tensor:
        """
        Compute a scalar loss from predicted next world positions.

        Args:
            world_pos: [N, 3] predicted cloth positions (from simulator)

        Returns:
            scalar loss tensor (grad-enabled)
        """


class StressObjective(BaseObjective):
    """
    Penalises deviation of mean deformation from a target stress value.

    stress = mean_N( ||world_pos_pred - mesh_rest||₂ )
    loss   = (stress - target_stress)²
    """

    def __init__(self, target_stress: float,
                 mesh_rest: torch.Tensor,
                 normal_mask: torch.Tensor) -> None:
        """
        Args:
            target_stress: desired stress value
            mesh_rest:     [N, 3] rest-configuration positions (z=0 plane)
            normal_mask:   [N] bool — True for NORMAL nodes
        """
        self._target      = target_stress
        self._mesh_rest   = mesh_rest
        self._normal_mask = normal_mask

    def __call__(self, world_pos: torch.Tensor) -> torch.Tensor:
        disp    = torch.norm(world_pos[self._normal_mask] -
                             self._mesh_rest[self._normal_mask], dim=-1)
        stress  = disp.mean()
        return F.mse_loss(stress, torch.tensor(self._target,
                                               device=world_pos.device))


# ---------------------------------------------------------------------------
# Differentiable PCA inverse-transform (torch)
# ---------------------------------------------------------------------------

class TorchPCAInverseTransform(torch.nn.Module):
    """
    Differentiable inverse PCA transform:  z_pca [K] → world_pos [N, 3]

    Registers PCA parameters as buffers so gradients flow through them.
    """

    def __init__(self, components: np.ndarray,
                 mean: np.ndarray, N: int) -> None:
        """
        Args:
            components: [K, N*3] PCA components (Vt from SVD)
            mean:       [N*3]    PCA mean
            N:          number of cloth nodes
        """
        super().__init__()
        self._N = N
        self.register_buffer("components",
                             torch.from_numpy(components.astype(np.float32)))
        self.register_buffer("mean",
                             torch.from_numpy(mean.astype(np.float32)))

    def forward(self, z_pca: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_pca: [K] or [B, K]  PCA coefficient vector(s)

        Returns:
            world_pos: [N, 3] or [B, N, 3]
        """
        flat = z_pca @ self.components + self.mean   # [N*3] or [B, N*3]
        if flat.dim() == 1:
            return flat.view(self._N, 3)
        return flat.view(-1, self._N, 3)


# ---------------------------------------------------------------------------
# Inverse designer
# ---------------------------------------------------------------------------

class ClothInverseDesigner:
    """
    Gradient-descent optimiser in cloth CVAE latent space.

    The full differentiable chain:
        z → decoder → pose_pca → PCA^{-1} → world_pos → FlagSimulator → loss

    All steps are differentiable with respect to z.
    """

    def __init__(self,
                 cvae_trainer,           # ClothCVAETrainer (loaded)
                 flag_simulator,         # FlagSimulator (loaded, eval mode)
                 objective: BaseObjective,
                 reference_traj_path: str,
                 device: str = "cpu") -> None:
        self._trainer     = cvae_trainer
        self._simulator   = flag_simulator
        self._objective   = objective
        self._ref_path    = reference_traj_path
        self._device      = device

        # Pre-build PCA inverse transform module
        from extensions.generative.cloth_extractor import PosePCA
        from torch_geometric.data import Data

        # Load PCA from cvae_trainer's config (pose_dim = K)
        pose_dim = cvae_trainer._cfg.pose_dim

        # Build differentiable PCA module using trainer's scaler and loaded PCA
        # (PCA is stored alongside the CVAE checkpoint via the extractor)
        self._pca_inv: Optional[TorchPCAInverseTransform] = None

    def _setup_pca_inv(self, pca) -> None:
        """Initialise differentiable PCA inverse transform."""
        N = pca.mean_.shape[0] // 3
        self._pca_inv = TorchPCAInverseTransform(
            components=pca.components_,
            mean=pca.mean_,
            N=N
        ).to(self._device)

    def _decode_pose(self, z: torch.Tensor,
                     target_stress_norm: float) -> torch.Tensor:
        """
        z [L] → world_pos [N, 3] (fully differentiable).
        """
        target_t  = torch.tensor([[target_stress_norm]], device=self._device)
        pose_norm = self._trainer._model.decoder(z.unsqueeze(0), target_t)  # [1, K]

        # Denormalise pose from [0,1] back to PCA coefficient space
        scaler    = self._trainer._scaler
        pose_min  = torch.from_numpy(scaler.pose_min.astype(np.float32)).to(self._device)
        pose_max  = torch.from_numpy(scaler.pose_max.astype(np.float32)).to(self._device)
        pose_phys = pose_norm * (pose_max - pose_min) + pose_min   # [1, K]

        return self._pca_inv(pose_phys.squeeze(0))  # [N, 3]

    def _build_data_from_pos(self, world_pos: torch.Tensor) -> Data:
        """Build a FlagSimulator-compatible Data from world_pos (non-differentiable scaffold)."""
        from extensions.generative.mesh_generator import ClothMeshBuilder
        ref = np.load(self._ref_path)
        mp  = ref["mesh_pos"].astype(np.float32)
        nt  = ref["node_type"].astype(np.float32)
        cells = ref["cells"].astype(np.int64)

        wp_np    = world_pos.detach().cpu().numpy()
        nt_t     = torch.from_numpy(nt).to(self._device)
        mp_t     = torch.from_numpy(mp).to(self._device)
        face_t   = torch.from_numpy(cells.T).to(self._device)

        x        = torch.cat([world_pos, nt_t], dim=-1)   # [N, 4]  ← grad flows
        graph    = Data(
            x          = x,
            prev_x     = world_pos.detach().clone(),
            pos        = mp_t,
            world_pos  = world_pos,
            face       = face_t,
            y          = torch.zeros_like(world_pos),
        )
        return graph

    def optimise(self,
                 target_stress: float,
                 n_iters: int = 100,
                 lr: float = 0.05,
                 n_restarts: int = 3,
                 pca=None,
                 verbose: bool = True) -> OptimisationResult:
        """
        Gradient descent in cloth CVAE latent space.

        Args:
            target_stress: desired stress value (physical units)
            n_iters:       gradient descent steps per restart
            lr:            learning rate
            n_restarts:    number of random initialisations (best is kept)
            pca:           PosePCA object (used for differentiable inverse PCA)
            verbose:       print progress

        Returns:
            OptimisationResult with best latent code and trajectory
        """
        if pca is not None:
            self._setup_pca_inv(pca)
        if self._pca_inv is None:
            raise ValueError("PCA inverse transform not initialised. Pass pca= arg.")

        # Normalise target stress for decoder conditioning
        scaler           = self._trainer._scaler
        target_norm      = float(
            (target_stress - scaler.stress_min) /
            (scaler.stress_max - scaler.stress_min + 1e-8)
        )
        latent_dim       = self._trainer._cfg.latent_dim
        self._trainer._model.eval()
        self._simulator.eval()

        best_z:        Optional[np.ndarray] = None
        best_stress:   float                = float("inf")
        best_loss:     float                = float("inf")
        all_trajectories: list[list[float]] = []

        for restart in range(n_restarts):
            z = torch.randn(latent_dim, device=self._device, requires_grad=True)
            optim     = torch.optim.Adam([z], lr=lr)
            traj:     list[float] = []

            for it in range(n_iters):
                optim.zero_grad()

                world_pos = self._decode_pose(z, target_norm)   # [N, 3]
                graph     = self._build_data_from_pos(world_pos)

                # FlagSimulator eval forward returns next_world_pos
                next_pos  = self._simulator(graph)               # [N, 3]
                loss      = self._objective(next_pos)

                loss.backward()
                optim.step()

                traj.append(loss.item())
                if verbose and (it + 1) % 20 == 0:
                    print(f"  Restart {restart + 1}/{n_restarts}  "
                          f"Iter {it + 1}/{n_iters}  loss={loss.item():.4e}")

            all_trajectories.append(traj)
            final_loss = traj[-1]
            if final_loss < best_loss:
                best_loss   = final_loss
                best_z      = z.detach().cpu().numpy().copy()
                # Compute stress from best z
                with torch.no_grad():
                    wp_best   = self._decode_pose(z.detach(), target_norm)
                    g_best    = self._build_data_from_pos(wp_best.detach())
                    np_best   = self._simulator(g_best)
                    disp      = torch.norm(np_best[self._objective._normal_mask] -
                                          self._objective._mesh_rest[
                                              self._objective._normal_mask], dim=-1)
                    best_stress = float(disp.mean().item())

        # Use trajectory from best restart
        best_run_idx = min(range(n_restarts),
                           key=lambda i: all_trajectories[i][-1])

        return OptimisationResult(
            best_z=best_z,
            best_stress=best_stress,
            target_stress=target_stress,
            trajectory=all_trajectories[best_run_idx],
            n_iters=n_iters,
        )


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def optimize_cloth(cvae_path: str,
                   pca_path: str,
                   simulator_ckpt_path: str,
                   reference_traj_path: str,
                   target_stress: float,
                   n_iters: int = 100,
                   n_restarts: int = 3,
                   device: str = "cpu",
                   verbose: bool = True) -> OptimisationResult:
    """
    High-level convenience function for cloth inverse design.

    Args:
        cvae_path:            path to cloth_cvae.pth
        pca_path:             path to cloth_pca.pkl
        simulator_ckpt_path:  path to flag_best_model.pth
        reference_traj_path:  any traj_XXXXX.npz for mesh topology
        target_stress:        desired deformation stress
        n_iters:              gradient descent iterations per restart
        n_restarts:           number of random restarts
        device:               torch device
        verbose:              print optimisation progress

    Returns:
        OptimisationResult
    """
    import torch
    from extensions.generative.cvae_cloth import ClothCVAE, ClothCVAETrainer, StressSurrogate, StressSurrogateTrainer
    from extensions.generative.cloth_extractor import PosePCA
    from model.flag_simulator import FlagSimulator
    from utils.utils import NodeType

    # Load PCA
    pca = PosePCA.load(pca_path)

    # Load CVAE trainer (model + scaler)
    ckpt    = torch.load(cvae_path, map_location=device, weights_only=False)
    cfg     = ckpt["cfg"]
    model   = ClothCVAE(cfg=cfg)
    model.load_state_dict(ckpt["model_state_dict"])

    # Dummy stress surrogate (not used for optimisation, only for trainer interface)
    surrogate = StressSurrogate(pose_dim=cfg.pose_dim)
    s_trainer = StressSurrogateTrainer(surrogate, device=device)

    from extensions.generative.cvae_cloth import ClothCVAETrainer, ClothCVAEScaler
    trainer           = ClothCVAETrainer.__new__(ClothCVAETrainer)
    trainer._model    = model.to(device)
    trainer._scaler   = ckpt["scaler"]
    trainer._cfg      = cfg
    trainer._device   = device
    trainer._stress_trainer = s_trainer

    # Load simulator
    sim_ckpt  = torch.load(simulator_ckpt_path, map_location=device, weights_only=False)
    simulator = FlagSimulator(message_passing_num=15, device=device)
    simulator.load_state_dict(sim_ckpt["model_state_dict"])
    simulator.eval()

    # Build objective
    ref       = np.load(reference_traj_path)
    N         = ref["mesh_pos"].shape[0]
    mp_3d     = np.concatenate([ref["mesh_pos"],
                                 np.zeros((N, 1), dtype=np.float32)], axis=-1)
    mesh_rest = torch.from_numpy(mp_3d).to(device)
    nt        = ref["node_type"].squeeze(-1)
    normal_mask = torch.from_numpy(nt == int(NodeType.NORMAL)).to(device)

    objective = StressObjective(
        target_stress=target_stress,
        mesh_rest=mesh_rest,
        normal_mask=normal_mask,
    )

    designer = ClothInverseDesigner(
        cvae_trainer=trainer,
        flag_simulator=simulator,
        objective=objective,
        reference_traj_path=reference_traj_path,
        device=device,
    )

    return designer.optimise(
        target_stress=target_stress,
        n_iters=n_iters,
        n_restarts=n_restarts,
        pca=pca,
        verbose=verbose,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Differentiable cloth inverse design via CVAE latent optimisation."
    )
    p.add_argument("--cvae",       default="checkpoints/flag-simple_cvae.pth")
    p.add_argument("--pca",        default="data_flag/train/cloth_pca.pkl")
    p.add_argument("--simulator",  default="checkpoints/flag_best_model.pth")
    p.add_argument("--ref-traj",   default="data_flag/train/traj_00000.npz")
    p.add_argument("--target",     type=float, default=1.0,
                   help="Target stress value to optimise for")
    p.add_argument("--iters",      type=int, default=100)
    p.add_argument("--restarts",   type=int, default=3)
    p.add_argument("--lr",         type=float, default=0.05)
    p.add_argument("--device",     default="cpu")
    p.add_argument("--quiet",      action="store_true")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)

    print(f"Cloth inverse design — target_stress={args.target:.4f}")
    result = optimize_cloth(
        cvae_path=args.cvae,
        pca_path=args.pca,
        simulator_ckpt_path=args.simulator,
        reference_traj_path=args.ref_traj,
        target_stress=args.target,
        n_iters=args.iters,
        n_restarts=args.restarts,
        device=args.device,
        verbose=not args.quiet,
    )

    print(f"\nOptimisation complete:")
    print(f"  Target  stress: {result.target_stress:.4f}")
    print(f"  Achieved stress: {result.best_stress:.4f}")
    print(f"  Loss trajectory: {[f'{v:.4e}' for v in result.trajectory[::20]]}")


if __name__ == "__main__":
    main()
