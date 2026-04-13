"""
Unified CVAE Training Script
==============================
Entry point for training either the CFD or Cloth CVAE.

Usage
-----
    # CFD (cylinder_flow):
    python extensions/generative/train_cvae.py --domain cylinder_flow

    # Cloth (flag_simple):
    python extensions/generative/train_cvae.py --domain flag_simple

The script auto-runs Phase 0 extraction if outputs don't exist yet.

Design principles
-----------------
- **Strategy pattern**: domain-specific logic is isolated in ``CVAEStrategy``
  subclasses; the main training loop is domain-agnostic.
- **Open / Closed**: add a new domain by subclassing ``CVAEStrategy``
  without touching the training loop.
- **Dependency Inversion**: the training loop depends on the abstract
  ``CVAEStrategy`` interface.
"""
from __future__ import annotations

import math
import os
import sys
import argparse

import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from abc import ABC, abstractmethod
from typing import Optional


# ---------------------------------------------------------------------------
# Strategy interface
# ---------------------------------------------------------------------------

class CVAEStrategy(ABC):
    """Abstract strategy for domain-specific CVAE training."""

    @abstractmethod
    def ensure_data(self) -> None:
        """Run extraction if data files don't exist yet."""

    @abstractmethod
    def load_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (design_vectors [N, D], targets [N])."""

    @abstractmethod
    def train(self, design: np.ndarray, targets: np.ndarray,
              args: argparse.Namespace) -> None:
        """Train the CVAE and save the checkpoint."""


# ---------------------------------------------------------------------------
# CFD strategy
# ---------------------------------------------------------------------------

class CFDStrategy(CVAEStrategy):
    """Strategy for cylinder_flow CFD CVAE training."""

    def __init__(self, args: argparse.Namespace) -> None:
        self._args = args

    def ensure_data(self) -> None:
        params_path    = os.path.join(self._args.data_dir, "design_params.npy")
        surrogate_path = self._args.surrogate_out

        if not os.path.exists(params_path):
            print("design_params.npy not found — running shape_extractor.py...")
            from extensions.generative.shape_extractor import main as extract_main
            extract_main([
                "--data-dir", self._args.data_dir,
                "--split",    self._args.split,
            ])

        if not os.path.exists(surrogate_path):
            print("drag_surrogate.pth not found — running drag_surrogate.py...")
            from extensions.generative.drag_surrogate import main as surrogate_main
            surrogate_main([
                "--params", params_path,
                "--out",    surrogate_path,
                "--device", self._args.device,
            ])

    def load_data(self) -> tuple[np.ndarray, np.ndarray]:
        from extensions.generative.drag_surrogate import DragProxyComputer
        params_path = os.path.join(self._args.data_dir, "design_params.npy")
        params      = np.load(params_path).astype(np.float32)
        valid       = np.isfinite(params[:, 0])
        params      = params[valid]
        drag        = DragProxyComputer()(params)
        return params, drag

    def train(self, design: np.ndarray, targets: np.ndarray,
              args: argparse.Namespace) -> None:
        from extensions.generative.drag_surrogate import (
            DragSurrogateTrainer, extract_true_drag,
        )
        from extensions.generative.cvae_cfd import CVAEConfig, CFDCVAE, CVAETrainer

        surrogate_trainer = DragSurrogateTrainer.load(args.surrogate_out,
                                                       device=args.device)

        # ── Try to enrich labels with physics-accurate drag from GNN rollouts ──
        # Only attempt this if a trained simulator checkpoint is available.
        true_drag_labels: dict[int, float] = {}
        _ckpt = getattr(args, 'simulator_ckpt', None) or 'checkpoints/best_model.pth'
        if os.path.exists(_ckpt):
            try:
                from model.simulator import Simulator
                from dataset import FpcDataset
                import torch

                _device = args.device
                _sim = Simulator(
                    message_passing_num=15,
                    node_input_size=11,
                    edge_input_size=3,
                    device=_device,
                )
                _state = torch.load(_ckpt, map_location=_device, weights_only=False)
                _sim.load_state_dict(_state['model_state_dict'])
                _sim.eval()

                _dataset = FpcDataset(args.data_dir, split=args.split)
                n_traj = len(_dataset) // _dataset.num_sampes_per_tra
                print(f"  Extracting true drag labels from {n_traj} GNN rollouts …")
                for i in range(min(n_traj, len(design))):
                    try:
                        drag = extract_true_drag(
                            _sim, _dataset, i, device=_device,
                        )
                        if not math.isnan(drag):
                            true_drag_labels[i] = drag
                    except Exception:
                        pass  # fallback to analytical for this trajectory
                print(f"  Got true drag for {len(true_drag_labels)}/{min(n_traj, len(design))} "
                      f"trajectories (rest use analytical formula)")
            except Exception:
                pass  # no simulator available — use analytical formula entirely

        cfg     = CVAEConfig(
            latent_dim=args.latent_dim,
            epochs=args.epochs,
            beta=args.beta,
            lam=args.lam,
            lr=args.lr,
        )
        model   = CFDCVAE(cfg=cfg)
        trainer = CVAETrainer(model=model, surrogate=surrogate_trainer._model,
                              cfg=cfg, device=args.device)
        trainer.fit(design, targets, verbose=not args.quiet)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        trainer.save(args.out)

        print(f"\nDemo: generating 5 samples at target_drag={targets.mean():.4f}")
        samples = trainer.generate(target_drag_physical=float(targets.mean()), n=5)
        print("  cx      cy      r    v_inlet")
        for row in samples:
            print("  " + "  ".join(f"{v:.4f}" for v in row))


# ---------------------------------------------------------------------------
# Cloth strategy
# ---------------------------------------------------------------------------

class ClothStrategy(CVAEStrategy):
    """Strategy for flag_simple cloth CVAE training."""

    def __init__(self, args: argparse.Namespace) -> None:
        self._args = args

    def ensure_data(self) -> None:
        split_dir  = os.path.join(self._args.data_dir, self._args.split)
        pose_path  = os.path.join(split_dir, "cloth_pose_pca.npy")
        if not os.path.exists(pose_path):
            print("cloth_pose_pca.npy not found — running cloth_extractor.py...")
            from extensions.generative.cloth_extractor import main as cloth_main
            cloth_main([
                "--data-dir",     self._args.data_dir,
                "--split",        self._args.split,
                "--n-components", str(self._args.pose_dim),
            ])

    def load_data(self) -> tuple[np.ndarray, np.ndarray]:
        split_dir  = os.path.join(self._args.data_dir, self._args.split)
        pose_path  = os.path.join(split_dir, "cloth_pose_pca.npy")
        stress_path = os.path.join(split_dir, "cloth_stress.npy")
        return (np.load(pose_path).astype(np.float32),
                np.load(stress_path).astype(np.float32))

    def train(self, design: np.ndarray, targets: np.ndarray,
              args: argparse.Namespace) -> None:
        from extensions.generative.cvae_cloth import (
            ClothCVAEConfig, ClothCVAE, ClothCVAETrainer,
            StressSurrogate, StressSurrogateTrainer
        )
        from extensions.generative.cloth_extractor import PosePCA

        # Prefer the portable .npz format; fall back to legacy .pkl
        _pca_npz = os.path.join(args.data_dir, args.split, "cloth_pca.npz")
        _pca_pkl = os.path.join(args.data_dir, args.split, "cloth_pca.pkl")
        pca_path = _pca_npz if os.path.exists(_pca_npz) else _pca_pkl
        pca      = PosePCA.load(pca_path)

        # Train stress surrogate
        surrogate = StressSurrogate(pose_dim=design.shape[1])
        s_trainer = StressSurrogateTrainer(surrogate, device=args.device)
        s_trainer.fit(design, targets, epochs=200, verbose=not args.quiet)

        # Train CVAE
        cfg     = ClothCVAEConfig(
            pose_dim=design.shape[1],
            latent_dim=args.latent_dim,
            epochs=args.epochs,
            beta=args.beta,
            lam=args.lam,
            lr=args.lr,
        )
        model   = ClothCVAE(cfg=cfg)
        trainer = ClothCVAETrainer(model=model, stress_trainer=s_trainer,
                                   cfg=cfg, device=args.device)
        trainer.fit(design, targets, verbose=not args.quiet)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        trainer.save(args.out)

        print(f"\nDemo: generating 3 cloth configs at target_stress={targets.mean():.4f}")
        samples = trainer.generate(target_stress=float(targets.mean()), n=3, pca=pca)
        print(f"  Generated {len(samples)} configs  world_pos shape: {samples[0].shape}")


# ---------------------------------------------------------------------------
# Registry (Open / Closed)
# ---------------------------------------------------------------------------

_STRATEGIES: dict[str, type[CVAEStrategy]] = {
    "cylinder_flow": CFDStrategy,
    "flag_simple":   ClothStrategy,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train domain-specific CVAE for inverse design."
    )
    p.add_argument("--domain",        required=True,
                   choices=list(_STRATEGIES),
                   help="Physics domain to train on")
    p.add_argument("--data-dir",      default=None,
                   help="Dataset root directory (default: 'data' for CFD, 'data_flag' for cloth)")
    p.add_argument("--split",         default="train",
                   choices=["train", "valid"])
    p.add_argument("--out",           default=None,
                   help="Output path for CVAE checkpoint "
                        "(default: checkpoints/{domain}_cvae.pth)")
    p.add_argument("--surrogate-out", default="checkpoints/drag_surrogate.pth",
                   help="Path for drag surrogate (CFD only)")
    p.add_argument("--epochs",        type=int, default=300)
    p.add_argument("--latent-dim",    type=int, default=16)
    p.add_argument("--pose-dim",      type=int, default=16,
                   help="PCA components for cloth (ignored for CFD)")
    p.add_argument("--beta",          type=float, default=1e-3,
                   help="KL weight in CVAE loss")
    p.add_argument("--lam",           type=float, default=0.5,
                   help="Physics consistency weight")
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--device",        default="cpu")
    p.add_argument("--quiet",         action="store_true")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)

    # Apply domain-specific defaults
    if args.data_dir is None:
        args.data_dir = "data" if args.domain == "cylinder_flow" else "data_flag"
    if args.out is None:
        args.out = f"checkpoints/{args.domain.replace('_', '-')}_cvae.pth"

    print(f"=== PhysicsAI Generate — CVAE Training ===")
    print(f"Domain  : {args.domain}")
    print(f"Data    : {args.data_dir}")
    print(f"Output  : {args.out}")
    print(f"Device  : {args.device}")
    print()

    strategy = _STRATEGIES[args.domain](args)

    print("Step 1/3: Ensuring data (extraction if needed)...")
    strategy.ensure_data()

    print("\nStep 2/3: Loading design vectors + targets...")
    design, targets = strategy.load_data()
    print(f"  design shape: {design.shape}  target range: [{targets.min():.4f}, {targets.max():.4f}]")

    print("\nStep 3/3: Training CVAE...")
    strategy.train(design, targets, args)

    print("\n✓ Done.")


if __name__ == "__main__":
    main()
