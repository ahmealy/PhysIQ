"""
Drag Surrogate Model
====================
A lightweight MLP that maps cylinder design parameters
``(cx, cy, r, v_inlet)`` → ``drag_proxy`` (scalar).

Drag proxy definition
---------------------
We use a blockage-corrected velocity-squared formula:

    drag_proxy = r * v_inlet² / (1 - 2r/H)

where H = channel height (0.41 m in the DeepMind dataset).
This approximates the Stokes-flow drag on a cylinder in a channel and
correlates monotonically with the true pressure-based drag coefficient.

The surrogate replaces a full MeshGraphNets rollout during CVAE training's
physics consistency loss, giving a ~1000x speedup per forward pass.

Design principles
-----------------
- **Single Responsibility**: ``DragProxyComputer`` computes analytical proxies;
  ``DragSurrogate`` is the neural network; ``DragSurrogateTrainer`` handles training.
- **Dependency Inversion**: ``DragSurrogateTrainer`` depends on the abstract
  ``BaseSurrogate`` interface, not the concrete MLP.
- **Open / Closed**: add new surrogate architectures by subclassing
  ``BaseSurrogate`` without modifying the trainer.
"""
from __future__ import annotations

import os
import sys
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Analytical drag proxy computation
# ---------------------------------------------------------------------------

class DragProxyComputer:
    """
    Computes a physics-informed drag proxy from cylinder parameters.

    Formula:  drag = r * v_inlet² / (1 - 2r / H)

    Single Responsibility: proxy computation only.
    """

    CHANNEL_HEIGHT: float = 0.41  # DeepMind cylinder_flow domain height

    def __call__(self, params: np.ndarray) -> np.ndarray:
        """
        Compute drag proxies for a batch of design parameters.

        Args:
            params: [N, 4] array with columns [cx, cy, r, v_inlet]

        Returns:
            [N] float32 array of drag proxy values
        """
        r       = params[:, 2].astype(np.float64)
        v       = params[:, 3].astype(np.float64)
        H       = self.CHANNEL_HEIGHT
        # Clamp to avoid division by zero for extreme blockage
        blockage = np.clip(2.0 * r / H, 0.0, 0.99)
        drag     = r * v ** 2 / (1.0 - blockage)
        return drag.astype(np.float32)


# ---------------------------------------------------------------------------
# Abstract surrogate interface (Dependency Inversion / Open-Closed)
# ---------------------------------------------------------------------------

@dataclass
class SurrogateConfig:
    """Hyperparameters for the surrogate MLP."""
    input_size:  int   = 4
    hidden_size: int   = 64
    n_layers:    int   = 3
    lr:          float = 1e-3
    epochs:      int   = 200
    batch_size:  int   = 64
    val_split:   float = 0.1


class BaseSurrogate(ABC, nn.Module):
    """Abstract interface for drag surrogate models."""

    @abstractmethod
    def predict(self, params: torch.Tensor) -> torch.Tensor:
        """Return drag prediction for a batch of design params [B, 4]."""


# ---------------------------------------------------------------------------
# Concrete MLP surrogate
# ---------------------------------------------------------------------------

class DragSurrogate(BaseSurrogate):
    """
    3-layer MLP: (cx, cy, r, v_inlet) → drag_proxy.

    Architecture mirrors the EncoderProcessorDecoder MLPs (2 hidden layers)
    but is intentionally small (64-D) to keep inference fast.
    """

    def __init__(self, config: Optional[SurrogateConfig] = None) -> None:
        super().__init__()
        cfg = config or SurrogateConfig()
        sizes = [cfg.input_size] + [cfg.hidden_size] * (cfg.n_layers - 1) + [1]
        layers: list[nn.Module] = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)   # [B]

    def predict(self, params: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(params)


# ---------------------------------------------------------------------------
# Normalizer (min-max, avoids scipy/sklearn dependency)
# ---------------------------------------------------------------------------

class MinMaxScaler:
    """Stateless min-max normalizer stored as torch buffers."""

    def __init__(self) -> None:
        self.x_min:    Optional[np.ndarray] = None
        self.x_max:    Optional[np.ndarray] = None
        self.y_min:    Optional[float]      = None
        self.y_max:    Optional[float]      = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.x_min = X.min(axis=0)
        self.x_max = X.max(axis=0)
        self.y_min = float(y.min())
        self.y_max = float(y.max())

    def transform_X(self, X: np.ndarray) -> np.ndarray:
        return (X - self.x_min) / (self.x_max - self.x_min + 1e-8)

    def transform_y(self, y: np.ndarray) -> np.ndarray:
        return (y - self.y_min) / (self.y_max - self.y_min + 1e-8)

    def inverse_y(self, y_norm: np.ndarray) -> np.ndarray:
        return y_norm * (self.y_max - self.y_min + 1e-8) + self.y_min


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class DragSurrogateTrainer:
    """
    Trains a ``BaseSurrogate`` on (design_params, drag_proxy) pairs.

    Depends on the abstract ``BaseSurrogate`` — any subclass can be used.
    """

    def __init__(self,
                 surrogate: BaseSurrogate,
                 config: Optional[SurrogateConfig] = None,
                 device: str = "cpu") -> None:
        self._model  = surrogate.to(device)
        self._cfg    = config or SurrogateConfig()
        self._device = device
        self._scaler = MinMaxScaler()

    def fit(self, X: np.ndarray, y: np.ndarray,
            verbose: bool = True) -> list[float]:
        """
        Train the surrogate.

        Args:
            X: [N, 4] design parameters
            y: [N]    drag proxy targets
            verbose: print loss every 20 epochs

        Returns:
            list of per-epoch training losses
        """
        self._scaler.fit(X, y)
        X_n = self._scaler.transform_X(X).astype(np.float32)
        y_n = self._scaler.transform_y(y).astype(np.float32)

        # Train / val split
        n_val = max(1, int(len(X_n) * self._cfg.val_split))
        idx   = np.random.permutation(len(X_n))
        val_idx, trn_idx = idx[:n_val], idx[n_val:]

        X_trn = torch.from_numpy(X_n[trn_idx]).to(self._device)
        y_trn = torch.from_numpy(y_n[trn_idx]).to(self._device)
        X_val = torch.from_numpy(X_n[val_idx]).to(self._device)
        y_val = torch.from_numpy(y_n[val_idx]).to(self._device)

        optim     = torch.optim.Adam(self._model.parameters(), lr=self._cfg.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=self._cfg.epochs
        )
        B         = self._cfg.batch_size
        losses    = []

        for epoch in range(1, self._cfg.epochs + 1):
            self._model.train()
            perm = torch.randperm(len(X_trn))
            epoch_loss = 0.0
            n_batches  = 0
            for i in range(0, len(X_trn), B):
                idx_b = perm[i:i + B]
                pred  = self._model(X_trn[idx_b])
                loss  = F.mse_loss(pred, y_trn[idx_b])
                optim.zero_grad()
                loss.backward()
                optim.step()
                epoch_loss += loss.item()
                n_batches  += 1
            scheduler.step()
            losses.append(epoch_loss / n_batches)

            if verbose and epoch % 20 == 0:
                self._model.eval()
                with torch.no_grad():
                    val_loss = F.mse_loss(self._model(X_val), y_val).item()
                print(f"  Epoch {epoch:4d}/{self._cfg.epochs} | "
                      f"train_loss={losses[-1]:.4e} | val_loss={val_loss:.4e}")

        return losses

    def save(self, path: str) -> None:
        """Save model weights and scaler to a single file.

        Both the scaler and config are serialised as plain dicts to avoid
        pickle module-path issues when loading from a different entry point
        (e.g. loading from ``api/`` after training via ``python drag_surrogate.py``
        which would pickle as ``__main__.SurrogateConfig``).
        """
        from dataclasses import asdict
        torch.save({
            "state_dict":  self._model.state_dict(),
            "scaler_dict": {
                "x_min": self._scaler.x_min,
                "x_max": self._scaler.x_max,
                "y_min": self._scaler.y_min,
                "y_max": self._scaler.y_max,
            },
            "config_dict": asdict(self._cfg),
        }, path)
        print(f"Saved surrogate to: {path}")

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "DragSurrogateTrainer":
        """Load a previously saved trainer (model + scaler)."""
        ckpt = torch.load(path, map_location=device, weights_only=False)
        # Restore config — handle both old (pickled dataclass) and new (dict) format
        if "config_dict" in ckpt:
            cfg = SurrogateConfig(**ckpt["config_dict"])
        else:
            cfg = ckpt["config"]   # backward compat
        model    = DragSurrogate(config=cfg)
        trainer  = cls(surrogate=model, config=cfg, device=device)
        model.load_state_dict(ckpt["state_dict"])
        # Restore scaler — handle both old (pickled object) and new (dict) format
        if "scaler_dict" in ckpt:
            sd = ckpt["scaler_dict"]
            trainer._scaler.x_min = sd["x_min"]
            trainer._scaler.x_max = sd["x_max"]
            trainer._scaler.y_min = sd["y_min"]
            trainer._scaler.y_max = sd["y_max"]
        else:
            trainer._scaler = ckpt["scaler"]   # backward compat
        return trainer

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict drag proxy for new design parameters [N, 4] → [N]."""
        X_n = self._scaler.transform_X(X).astype(np.float32)
        xt  = torch.from_numpy(X_n).to(self._device)
        self._model.eval()
        with torch.no_grad():
            y_n = self._model(xt).cpu().numpy()
        return self._scaler.inverse_y(y_n)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train a drag surrogate MLP on cylinder_flow design params."
    )
    p.add_argument("--params",     default="data/design_params.npy",
                   help="Path to design_params.npy (from shape_extractor.py)")
    p.add_argument("--out",        default="checkpoints/drag_surrogate.pth",
                   help="Where to save the trained surrogate")
    p.add_argument("--epochs",     type=int, default=200)
    p.add_argument("--hidden",     type=int, default=64)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--device",     default="cpu")
    p.add_argument("--quiet",      action="store_true")
    return p


def main(argv: list[str] | None = None) -> None:
    args   = _build_arg_parser().parse_args(argv)

    print(f"Loading design params from: {args.params}")
    params = np.load(args.params)         # [N, 4]
    valid  = np.isfinite(params).all(axis=1)   # check all 4 columns
    params = params[valid]
    print(f"Loaded {len(params)} valid param vectors.")

    # Compute analytical drag proxies
    proxy_fn   = DragProxyComputer()
    drag_proxy = proxy_fn(params)         # [N]
    print(f"Drag proxy: min={drag_proxy.min():.4f}  max={drag_proxy.max():.4f}  "
          f"mean={drag_proxy.mean():.4f}")

    cfg = SurrogateConfig(
        hidden_size=args.hidden,
        epochs=args.epochs,
        lr=args.lr,
    )
    surrogate = DragSurrogate(config=cfg)
    trainer   = DragSurrogateTrainer(surrogate=surrogate, config=cfg,
                                     device=args.device)

    print(f"\nTraining drag surrogate ({args.epochs} epochs)...")
    trainer.fit(params, drag_proxy, verbose=not args.quiet)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    trainer.save(args.out)

    # Quick sanity check
    sample_preds = trainer.predict(params[:5])
    sample_truth = drag_proxy[:5]
    print("\nSanity check (first 5 samples):")
    print(f"  {'truth':>10}  {'pred':>10}  {'err%':>8}")
    for t, p_val in zip(sample_truth, sample_preds):
        err = abs(p_val - t) / (abs(t) + 1e-8) * 100
        print(f"  {t:10.4f}  {p_val:10.4f}  {err:7.1f}%")


if __name__ == "__main__":
    main()
