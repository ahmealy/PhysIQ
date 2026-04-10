"""
CFD Conditional VAE (CVAE)
===========================
Maps cylinder design parameters ↔ a latent space, conditioned on a target
drag value.  Used for inverse design: given a desired drag, sample designs
from the learned distribution.

Architecture
------------
Encoder:   [cx, cy, r, v_inlet, drag_actual] (5-D) → FC → (μ[16], log σ[16])
Decoder:   [z[16], target_drag] (17-D)             → FC → [cx, cy, r, v_inlet] (4-D)

Loss:
    L = α · ||θ_recon - θ_gt||²         (reconstruction)
      + β · KL(q(z|θ,drag) || N(0,I))  (regularisation)
      + λ · |drag_surrogate(θ_recon) - target_drag|  (physics consistency)

Design principles
-----------------
- **Single Responsibility**: Encoder, Decoder, CVAE are separate classes.
- **Open / Closed**: extend by subclassing ``BaseCVAE`` (e.g. add flow conditioning).
- **Dependency Inversion**: CVAE physics loss depends on the abstract
  ``BaseSurrogate`` interface, not the concrete ``DragSurrogate``.
- **Liskov Substitution**: any ``BaseSurrogate`` can be injected.
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
from dataclasses import dataclass, field
from typing import Optional

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from extensions.generative.drag_surrogate import (   # noqa: E402
    BaseSurrogate, DragSurrogate, DragSurrogateTrainer,
    DragProxyComputer, SurrogateConfig, MinMaxScaler
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CVAEConfig:
    """Hyperparameters for the CFD CVAE."""
    param_dim:    int   = 4      # cx, cy, r, v_inlet
    latent_dim:   int   = 16
    hidden_size:  int   = 64
    beta:         float = 1e-3   # KL weight
    lam:          float = 0.5    # physics consistency weight
    alpha:        float = 1.0    # reconstruction weight
    lr:           float = 1e-3
    epochs:       int   = 300
    batch_size:   int   = 64
    val_split:    float = 0.1
    # Gumbel-softmax temperature (unused for CFD, here for interface parity)
    gumbel_tau:   float = 0.5


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class CVAEEncoder(nn.Module):
    """
    Encoder: (design_params [B,4], drag_actual [B,1]) → (μ[B,L], log_σ[B,L])

    Single Responsibility: feature extraction + distributional parameterisation.
    """

    def __init__(self, cfg: CVAEConfig) -> None:
        super().__init__()
        in_dim = cfg.param_dim + 1     # 4 params + 1 drag condition
        self.net = nn.Sequential(
            nn.Linear(in_dim,        cfg.hidden_size), nn.ReLU(),
            nn.Linear(cfg.hidden_size, cfg.hidden_size), nn.ReLU(),
        )
        self.fc_mu     = nn.Linear(cfg.hidden_size, cfg.latent_dim)
        self.fc_logvar = nn.Linear(cfg.hidden_size, cfg.latent_dim)

    def forward(self, params: torch.Tensor,
                drag: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            params: [B, 4]
            drag:   [B, 1]  (actual drag value)
        Returns:
            mu [B, L], logvar [B, L]
        """
        x      = torch.cat([params, drag], dim=-1)  # [B, 5]
        h      = self.net(x)
        return self.fc_mu(h), self.fc_logvar(h)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class CVAEDecoder(nn.Module):
    """
    Decoder: (z [B,L], target_drag [B,1]) → design_params [B,4]

    Outputs are in normalised [0,1] space; the trainer handles denormalisation.
    Single Responsibility: latent → design-space mapping.
    """

    def __init__(self, cfg: CVAEConfig) -> None:
        super().__init__()
        in_dim = cfg.latent_dim + 1   # z + target condition
        self.net = nn.Sequential(
            nn.Linear(in_dim,          cfg.hidden_size), nn.ReLU(),
            nn.Linear(cfg.hidden_size, cfg.hidden_size), nn.ReLU(),
            nn.Linear(cfg.hidden_size, cfg.param_dim),
        )

    def forward(self, z: torch.Tensor,
                target_drag: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z:           [B, L]
            target_drag: [B, 1]
        Returns:
            params_recon [B, 4] (normalised)
        """
        x = torch.cat([z, target_drag], dim=-1)   # [B, L+1]
        return self.net(x)                         # [B, 4]


# ---------------------------------------------------------------------------
# Abstract CVAE base
# ---------------------------------------------------------------------------

class BaseCVAE(ABC, nn.Module):
    """Abstract interface shared by CFD and Cloth CVAEs."""

    @abstractmethod
    def encode(self, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (mu, logvar)."""

    @abstractmethod
    def decode(self, *args, **kwargs) -> torch.Tensor:
        """Return reconstructed design parameters."""

    @abstractmethod
    def sample(self, target_condition: torch.Tensor,
               n: int = 1) -> torch.Tensor:
        """Sample n novel designs conditioned on target."""


# ---------------------------------------------------------------------------
# CVAE model
# ---------------------------------------------------------------------------

class CFDCVAE(BaseCVAE):
    """
    Full CFD Conditional VAE.

    Composes Encoder and Decoder following the VAE formulation.
    Exposes ``sample()`` for inference (inverse design).
    """

    def __init__(self, cfg: Optional[CVAEConfig] = None) -> None:
        super().__init__()
        self.cfg     = cfg or CVAEConfig()
        self.encoder = CVAEEncoder(self.cfg)
        self.decoder = CVAEDecoder(self.cfg)

    @staticmethod
    def reparameterise(mu: torch.Tensor,
                       logvar: torch.Tensor) -> torch.Tensor:
        """Sample z = μ + ε·exp(½ logvar), ε ~ N(0,I)."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, params: torch.Tensor,
               drag: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(params, drag)

    def decode(self, z: torch.Tensor,
               target_drag: torch.Tensor) -> torch.Tensor:
        return self.decoder(z, target_drag)

    def forward(self, params: torch.Tensor, drag: torch.Tensor,
                target_drag: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass used during training.

        Returns:
            recon [B,4], mu [B,L], logvar [B,L]
        """
        mu, logvar = self.encode(params, drag)
        z          = self.reparameterise(mu, logvar)
        recon      = self.decode(z, target_drag)
        return recon, mu, logvar

    def sample(self, target_drag: torch.Tensor, n: int = 1) -> torch.Tensor:
        """
        Sample novel design parameters for a target drag value.

        Args:
            target_drag: [1, 1] or scalar tensor — target drag (normalised)
            n:           number of samples to draw

        Returns:
            [n, 4] normalised design parameters
        """
        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            z    = torch.randn(n, self.cfg.latent_dim, device=device)
            cond = target_drag.expand(n, 1).to(device)
            return self.decode(z, cond)


# ---------------------------------------------------------------------------
# Normalizer (min-max, same as drag_surrogate.py but for design params + drag)
# ---------------------------------------------------------------------------

class CVAEScaler:
    """Min-max scaler for CVAE inputs and conditions."""

    def __init__(self) -> None:
        self.param_min:  Optional[np.ndarray] = None
        self.param_max:  Optional[np.ndarray] = None
        self.drag_min:   Optional[float]      = None
        self.drag_max:   Optional[float]      = None

    def fit(self, params: np.ndarray, drag: np.ndarray) -> None:
        self.param_min = params.min(axis=0)
        self.param_max = params.max(axis=0)
        self.drag_min  = float(drag.min())
        self.drag_max  = float(drag.max())

    def norm_params(self, p: np.ndarray) -> np.ndarray:
        return (p - self.param_min) / (self.param_max - self.param_min + 1e-8)

    def norm_drag(self, d: np.ndarray) -> np.ndarray:
        return (d - self.drag_min) / (self.drag_max - self.drag_min + 1e-8)

    def denorm_params(self, p: np.ndarray) -> np.ndarray:
        return p * (self.param_max - self.param_min) + self.param_min

    def denorm_drag(self, d: np.ndarray) -> np.ndarray:
        return d * (self.drag_max - self.drag_min) + self.drag_min


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class CVAETrainer:
    """
    Trains a ``BaseCVAE`` on (design_params, drag) pairs.

    Depends on ``BaseSurrogate`` for the physics consistency loss —
    any surrogate implementing ``predict()`` can be injected (DIP).
    """

    def __init__(self,
                 model:     BaseCVAE,
                 surrogate: BaseSurrogate,
                 cfg:       Optional[CVAEConfig] = None,
                 device:    str = "cpu") -> None:
        self._model     = model.to(device)
        self._surrogate = surrogate.to(device)
        self._cfg       = cfg or CVAEConfig()
        self._device    = device
        self._scaler    = CVAEScaler()

    # ── Loss helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """KL divergence: KL(q(z|x) || N(0,I))."""
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    def _physics_loss(self, recon_norm: torch.Tensor,
                      target_drag_norm: torch.Tensor) -> torch.Tensor:
        """
        ||drag_surrogate(denorm(recon)) - denorm(target_drag)||²

        Uses the injected surrogate (Dependency Inversion).
        """
        # Denormalise reconstructed params back to physical units
        recon_np      = recon_norm.detach().cpu().numpy()
        recon_phys    = self._scaler.denorm_params(recon_np)
        # Predict drag for each reconstructed design
        pred_drag     = self._surrogate.predict(
            torch.from_numpy(recon_phys.astype(np.float32)).to(self._device)
        )
        # Normalise predicted drag
        pred_norm     = (pred_drag - self._scaler.drag_min) / (
            self._scaler.drag_max - self._scaler.drag_min + 1e-8
        )
        target_flat   = target_drag_norm.squeeze(-1)
        return F.mse_loss(pred_norm, target_flat)

    # ── Training loop ─────────────────────────────────────────────────────────

    def fit(self, params: np.ndarray, drag: np.ndarray,
            verbose: bool = True) -> list[float]:
        """
        Train the CVAE.

        Args:
            params: [N, 4] design parameters
            drag:   [N]    drag proxy values
            verbose: print loss every 20 epochs

        Returns:
            list of per-epoch total losses
        """
        self._scaler.fit(params, drag)
        p_n = self._scaler.norm_params(params).astype(np.float32)
        d_n = self._scaler.norm_drag(drag).astype(np.float32)

        n_val  = max(1, int(len(p_n) * self._cfg.val_split))
        idx    = np.random.permutation(len(p_n))
        v_idx, t_idx = idx[:n_val], idx[n_val:]

        P_t = torch.from_numpy(p_n[t_idx]).to(self._device)
        D_t = torch.from_numpy(d_n[t_idx]).unsqueeze(-1).to(self._device)
        P_v = torch.from_numpy(p_n[v_idx]).to(self._device)
        D_v = torch.from_numpy(d_n[v_idx]).unsqueeze(-1).to(self._device)

        optim  = torch.optim.Adam(self._model.parameters(), lr=self._cfg.lr)
        sched  = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=self._cfg.epochs
        )
        B      = self._cfg.batch_size
        losses = []
        cfg    = self._cfg

        for epoch in range(1, cfg.epochs + 1):
            self._model.train()
            perm       = torch.randperm(len(P_t))
            epoch_loss = 0.0
            n_batches  = 0

            for i in range(0, len(P_t), B):
                ib      = perm[i:i + B]
                p_b     = P_t[ib]       # [B, 4]
                d_b     = D_t[ib]       # [B, 1]

                recon, mu, logvar = self._model(p_b, d_b, d_b)

                recon_loss = F.mse_loss(recon, p_b)
                kl_loss    = self._kl_loss(mu, logvar)
                phys_loss  = self._physics_loss(recon, d_b)

                loss = (cfg.alpha * recon_loss
                        + cfg.beta * kl_loss
                        + cfg.lam  * phys_loss)

                optim.zero_grad()
                loss.backward()
                optim.step()

                epoch_loss += loss.item()
                n_batches  += 1

            sched.step()
            losses.append(epoch_loss / n_batches)

            if verbose and epoch % 20 == 0:
                self._model.eval()
                with torch.no_grad():
                    recon_v, mu_v, lv_v = self._model(P_v, D_v, D_v)
                    val_recon = F.mse_loss(recon_v, P_v).item()
                    val_kl    = self._kl_loss(mu_v, lv_v).item()
                print(f"  Epoch {epoch:4d}/{cfg.epochs} | loss={losses[-1]:.4e} | "
                      f"val_recon={val_recon:.4e} | val_kl={val_kl:.4e}")

        return losses

    def save(self, path: str) -> None:
        """Save model weights + scaler.

        The scaler and config are serialised as plain dicts of numpy arrays to avoid
        pickle module-path issues when loading from a different entry point
        (e.g. loading from ``api/routes/generate.py`` after training via
        ``python cvae_cfd.py`` which pickles as ``__main__.CVAEScaler``).
        """
        from dataclasses import asdict
        torch.save({
            "model_state_dict": self._model.state_dict(),
            "scaler_dict": {
                "param_min": self._scaler.param_min,
                "param_max": self._scaler.param_max,
                "drag_min":  self._scaler.drag_min,
                "drag_max":  self._scaler.drag_max,
            },
            "cfg_dict":         asdict(self._cfg),
        }, path)
        print(f"Saved CVAE to: {path}")

    @classmethod
    def load(cls, path: str, surrogate: BaseSurrogate,
             device: str = "cpu") -> "CVAETrainer":
        """Reload a previously saved trainer."""
        ckpt    = torch.load(path, map_location=device, weights_only=False)
        # Restore config — handle both old (pickled dataclass) and new (dict) format
        if "cfg_dict" in ckpt:
            cfg = CVAEConfig(**ckpt["cfg_dict"])
        else:
            cfg = ckpt["cfg"]   # backward compat
        model   = CFDCVAE(cfg=cfg)
        model.load_state_dict(ckpt["model_state_dict"])
        trainer = cls(model=model, surrogate=surrogate, cfg=cfg, device=device)
        # Restore scaler — handle both old (pickled object) and new (dict) format
        if "scaler_dict" in ckpt:
            sd = ckpt["scaler_dict"]
            trainer._scaler.param_min = sd["param_min"]
            trainer._scaler.param_max = sd["param_max"]
            trainer._scaler.drag_min  = sd["drag_min"]
            trainer._scaler.drag_max  = sd["drag_max"]
        else:
            trainer._scaler = ckpt["scaler"]   # backward compat
        return trainer

    def generate(self, target_drag_physical: float, n: int = 10) -> np.ndarray:
        """
        Sample n designs with target drag (physical units).

        Returns [n, 4] array of (cx, cy, r, v_inlet) in physical units.
        """
        drag_n = np.array([(target_drag_physical - self._scaler.drag_min) /
                           (self._scaler.drag_max - self._scaler.drag_min + 1e-8)],
                          dtype=np.float32)
        t_drag = torch.from_numpy(drag_n).unsqueeze(0).to(self._device)

        samples_norm = self._model.sample(t_drag, n=n).cpu().numpy()   # [n, 4]
        # Clamp normalised outputs to [0,1] before denormalising
        samples_norm = np.clip(samples_norm, 0.0, 1.0)
        return self._scaler.denorm_params(samples_norm)                 # [n, 4]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train the CFD CVAE on cylinder design parameters."
    )
    p.add_argument("--params",        default="data/design_params.npy")
    p.add_argument("--surrogate",     default="checkpoints/drag_surrogate.pth")
    p.add_argument("--out",           default="checkpoints/cfd_cvae.pth")
    p.add_argument("--epochs",        type=int, default=300)
    p.add_argument("--latent-dim",    type=int, default=16)
    p.add_argument("--hidden",        type=int, default=64)
    p.add_argument("--beta",          type=float, default=1e-3)
    p.add_argument("--lam",           type=float, default=0.5)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--device",        default="cpu")
    p.add_argument("--quiet",         action="store_true")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)

    print(f"Loading design params from: {args.params}")
    params = np.load(args.params).astype(np.float32)
    valid  = np.isfinite(params[:, 0])
    params = params[valid]
    print(f"Loaded {len(params)} valid param vectors.")

    drag_fn = DragProxyComputer()
    drag    = drag_fn(params)
    print(f"Drag proxy range: [{drag.min():.4f}, {drag.max():.4f}]")

    # Load pre-trained surrogate
    print(f"Loading surrogate from: {args.surrogate}")
    surrogate_trainer = DragSurrogateTrainer.load(args.surrogate, device=args.device)
    surrogate_model   = surrogate_trainer._model

    cfg = CVAEConfig(
        latent_dim=args.latent_dim,
        hidden_size=args.hidden,
        beta=args.beta,
        lam=args.lam,
        lr=args.lr,
        epochs=args.epochs,
    )
    model   = CFDCVAE(cfg=cfg)
    trainer = CVAETrainer(model=model, surrogate=surrogate_model,
                          cfg=cfg, device=args.device)

    print(f"\nTraining CFD CVAE ({args.epochs} epochs)...")
    trainer.fit(params, drag, verbose=not args.quiet)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    trainer.save(args.out)

    # Demo generation
    print("\nGenerating 5 samples at target_drag = 0.025 (medium drag):")
    samples = trainer.generate(target_drag_physical=0.025, n=5)
    header  = "  " + "  ".join(f"{n:>10}" for n in ["cx", "cy", "r", "v_inlet"])
    print(header)
    for row in samples:
        print("  " + "  ".join(f"{v:10.4f}" for v in row))


if __name__ == "__main__":
    main()
