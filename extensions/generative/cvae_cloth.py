"""
Cloth Conditional VAE (CVAE)
=============================
Maps compressed cloth initial-pose vectors ↔ a latent space, conditioned
on a target stress value.

The pose is represented as PCA coefficients of the flattened initial
world position (from ``cloth_extractor.PosePCA``).

Architecture
------------
Encoder:   [pose_pca[K], stress_actual] (K+1) → FC → (μ[L], log σ[L])
Decoder:   [z[L], target_stress] (L+1)        → FC → [pose_pca[K]] (K)

K defaults to 16 (PCA dims), L defaults to 16 (latent dims).

Loss:
    L = α · ||pose_recon - pose_gt||²          (reconstruction)
      + β · KL(q(z|pose,stress) || N(0,I))    (regularisation)
      + λ · |stress_surrogate(pose_recon) - target_stress|  (physics)

Design notes
------------
- CVAE uses the same abstract interfaces as cvae_cfd.py.
- The stress surrogate is a small MLP trained on (pose_pca → stress).
- Differentiable inverse design (Phase 4) operates in this latent space:
  ∂stress/∂z is well-defined through the decoder.
"""
from __future__ import annotations

import os
import sys
import argparse
import pickle

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

from extensions.generative.drag_surrogate import (   # noqa: E402
    BaseSurrogate, MinMaxScaler, SurrogateConfig
)
from extensions.generative.cloth_extractor import PosePCA  # noqa: E402


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ClothCVAEConfig:
    """Hyperparameters for the Cloth CVAE."""
    pose_dim:    int   = 16     # PCA components
    latent_dim:  int   = 16
    hidden_size: int   = 128
    beta:        float = 1e-3
    lam:         float = 0.5
    alpha:       float = 1.0
    lr:          float = 1e-3
    epochs:      int   = 300
    batch_size:  int   = 64
    val_split:   float = 0.1
    # Free-bits threshold: KL per latent dim is clamped to at least this value.
    # Prevents posterior collapse on unused dimensions.
    # 0.0 = disabled (standard KL); recommended value: 0.05
    free_bits:   float = 0.0


# ---------------------------------------------------------------------------
# Stress surrogate (small MLP trained on pose_pca → stress)
# ---------------------------------------------------------------------------

class StressSurrogate(BaseSurrogate):
    """
    Tiny MLP: pose_pca [K] → stress_proxy (scalar).

    Trained on extracted pose/stress pairs before the CVAE.
    """

    def __init__(self, pose_dim: int = 16, hidden_size: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(pose_dim,  hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)   # [B]

    def predict(self, params: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(params)


class StressSurrogateTrainer:
    """Trains a StressSurrogate on pose_pca → stress data."""

    def __init__(self, model: StressSurrogate, device: str = "cpu") -> None:
        self._model  = model.to(device)
        self._device = device
        self._scaler = MinMaxScaler()

    def fit(self, X: np.ndarray, y: np.ndarray,
            epochs: int = 200, lr: float = 1e-3,
            verbose: bool = True) -> None:
        self._scaler.fit(X, y)
        Xn = self._scaler.transform_X(X).astype(np.float32)
        yn = self._scaler.transform_y(y).astype(np.float32)

        Xt = torch.from_numpy(Xn).to(self._device)
        yt = torch.from_numpy(yn).to(self._device)

        optim = torch.optim.Adam(self._model.parameters(), lr=lr)
        for ep in range(1, epochs + 1):
            self._model.train()
            perm  = torch.randperm(len(Xt))
            total = 0.0
            for i in range(0, len(Xt), 64):
                b  = perm[i:i + 64]
                p  = self._model(Xt[b])
                l  = F.mse_loss(p, yt[b])
                optim.zero_grad()
                l.backward()
                optim.step()
                total += l.item()
            if verbose and ep % 50 == 0:
                print(f"  StressSurrogate epoch {ep}/{epochs}  loss={total:.4e}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xn = self._scaler.transform_X(X).astype(np.float32)
        xt = torch.from_numpy(Xn).to(self._device)
        self._model.eval()
        with torch.no_grad():
            yn = self._model(xt).cpu().numpy()
        return self._scaler.inverse_y(yn)

    def save(self, path: str) -> None:
        torch.save({
            "state_dict":  self._model.state_dict(),
            "scaler_dict": {
                "x_min": self._scaler.x_min,
                "x_max": self._scaler.x_max,
                "y_min": self._scaler.y_min,
                "y_max": self._scaler.y_max,
            },
        }, path)

    @classmethod
    def load(cls, path: str, pose_dim: int, device: str = "cpu") -> "StressSurrogateTrainer":
        ckpt   = torch.load(path, map_location=device, weights_only=False)
        model  = StressSurrogate(pose_dim=pose_dim)
        model.load_state_dict(ckpt["state_dict"])
        t      = cls(model=model, device=device)
        # Restore scaler — handle both old (pickled object) and new (dict) format
        if "scaler_dict" in ckpt:
            sd = ckpt["scaler_dict"]
            t._scaler.x_min = sd["x_min"]
            t._scaler.x_max = sd["x_max"]
            t._scaler.y_min = sd["y_min"]
            t._scaler.y_max = sd["y_max"]
        else:
            t._scaler = ckpt["scaler"]   # backward compat
        return t


# ---------------------------------------------------------------------------
# Cloth CVAE components
# ---------------------------------------------------------------------------

class ClothEncoder(nn.Module):
    """Encoder: (pose_pca [B,K], stress [B,1]) → (μ[B,L], logvar[B,L])"""

    def __init__(self, cfg: ClothCVAEConfig) -> None:
        super().__init__()
        in_dim = cfg.pose_dim + 1
        self.net       = nn.Sequential(
            nn.Linear(in_dim,          cfg.hidden_size), nn.ReLU(),
            nn.Linear(cfg.hidden_size, cfg.hidden_size), nn.ReLU(),
        )
        self.fc_mu     = nn.Linear(cfg.hidden_size, cfg.latent_dim)
        self.fc_logvar = nn.Linear(cfg.hidden_size, cfg.latent_dim)

    def forward(self, pose: torch.Tensor,
                stress: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([pose, stress], dim=-1)
        h = self.net(x)
        return self.fc_mu(h), self.fc_logvar(h)


class ClothDecoder(nn.Module):
    """Decoder: (z [B,L], target_stress [B,1]) → pose_pca_recon [B,K]"""

    def __init__(self, cfg: ClothCVAEConfig) -> None:
        super().__init__()
        in_dim = cfg.latent_dim + 1
        self.net = nn.Sequential(
            nn.Linear(in_dim,          cfg.hidden_size), nn.ReLU(),
            nn.Linear(cfg.hidden_size, cfg.hidden_size), nn.ReLU(),
            nn.Linear(cfg.hidden_size, cfg.pose_dim),
        )

    def forward(self, z: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, target], dim=-1))   # [B, K]


class ClothCVAE(nn.Module):
    """Full Cloth CVAE model."""

    def __init__(self, cfg: Optional[ClothCVAEConfig] = None) -> None:
        super().__init__()
        self.cfg     = cfg or ClothCVAEConfig()
        self.encoder = ClothEncoder(self.cfg)
        self.decoder = ClothDecoder(self.cfg)

    @staticmethod
    def reparameterise(mu: torch.Tensor,
                       logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def forward(self, pose: torch.Tensor, stress: torch.Tensor,
                target: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(pose, stress)
        z          = self.reparameterise(mu, logvar)
        recon      = self.decoder(z, target)
        return recon, mu, logvar

    def sample(self, target_stress: torch.Tensor, n: int = 1) -> torch.Tensor:
        """Sample n pose_pca vectors conditioned on target stress."""
        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            z    = torch.randn(n, self.cfg.latent_dim, device=device)
            cond = target_stress.expand(n, 1).to(device)
            return self.decoder(z, cond)  # [n, K]


# ---------------------------------------------------------------------------
# Scaler for CVAE
# ---------------------------------------------------------------------------

class ClothCVAEScaler:
    """Min-max scaler for pose PCA coords and stress values."""

    def __init__(self) -> None:
        self.pose_min:   Optional[np.ndarray] = None
        self.pose_max:   Optional[np.ndarray] = None
        self.stress_min: Optional[float]      = None
        self.stress_max: Optional[float]      = None

    def fit(self, pose: np.ndarray, stress: np.ndarray) -> None:
        self.pose_min   = pose.min(axis=0)
        self.pose_max   = pose.max(axis=0)
        self.stress_min = float(stress.min())
        self.stress_max = float(stress.max())

    def norm_pose(self, p: np.ndarray) -> np.ndarray:
        return (p - self.pose_min) / (self.pose_max - self.pose_min + 1e-8)

    def norm_stress(self, s: np.ndarray) -> np.ndarray:
        return (s - self.stress_min) / (self.stress_max - self.stress_min + 1e-8)

    def denorm_pose(self, p: np.ndarray) -> np.ndarray:
        return p * (self.pose_max - self.pose_min + 1e-8) + self.pose_min

    def denorm_stress(self, s: np.ndarray) -> np.ndarray:
        return s * (self.stress_max - self.stress_min + 1e-8) + self.stress_min


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class ClothCVAETrainer:
    """
    Trains the Cloth CVAE.

    Depends on the abstract ``BaseSurrogate`` for physics consistency loss.
    """

    def __init__(self,
                 model:         ClothCVAE,
                 stress_trainer: StressSurrogateTrainer,
                 cfg:           Optional[ClothCVAEConfig] = None,
                 device:        str = "cpu") -> None:
        self._model           = model.to(device)
        self._stress_trainer  = stress_trainer
        self._cfg             = cfg or ClothCVAEConfig()
        self._device          = device
        self._scaler          = ClothCVAEScaler()

    def _kl_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        KL divergence with free-bits per latent dimension.

        See CVAETrainer._kl_loss for full description.
        When free_bits=0.0 this is identical to the standard formula.
        """
        free_bits = self._cfg.free_bits
        if free_bits <= 0.0:
            return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        kl_per_dim = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=0)
        return torch.sum(torch.clamp(kl_per_dim, min=free_bits))

    def _physics_loss(self, recon_norm: torch.Tensor,
                      target_stress_norm: torch.Tensor) -> torch.Tensor:
        """
        Physics consistency loss — gradient flows back through the decoder.

        The denorm is expressed as differentiable tensor arithmetic so that
        autograd can trace gradients from the stress prediction back through
        recon_norm to the CVAE decoder weights.  The StressSurrogate is called
        without no_grad so the graph stays connected.
        """
        p_min = torch.from_numpy(self._scaler.pose_min.astype(np.float32)).to(self._device)
        p_max = torch.from_numpy(self._scaler.pose_max.astype(np.float32)).to(self._device)
        recon_phys = recon_norm * (p_max - p_min) + p_min    # [B, K] grad-enabled

        # Normalise into the surrogate's input scale (uses MinMaxScaler fitted on pose)
        surr_sc = self._stress_trainer._scaler
        x_min   = torch.from_numpy(surr_sc.x_min.astype(np.float32)).to(self._device)
        x_max   = torch.from_numpy(surr_sc.x_max.astype(np.float32)).to(self._device)
        surr_in = (recon_phys - x_min) / (x_max - x_min + 1e-8)  # [B, K]

        # StressSurrogate forward — in-graph (no no_grad wrapper)
        self._stress_trainer._model.train(False)
        stress_n_out = self._stress_trainer._model(surr_in)         # [B]

        # Denorm surrogate output to physical stress
        y_min = float(surr_sc.y_min)
        y_max = float(surr_sc.y_max)
        stress_phys = stress_n_out * (y_max - y_min + 1e-8) + y_min  # [B]

        # Re-normalise back to CVAE stress scale for comparison
        s_min = float(self._scaler.stress_min)
        s_max = float(self._scaler.stress_max)
        pred_norm   = (stress_phys - s_min) / (s_max - s_min + 1e-8)
        target_flat = target_stress_norm.squeeze(-1)
        return F.mse_loss(pred_norm, target_flat)

    def fit(self, pose_pca: np.ndarray, stress: np.ndarray,
            verbose: bool = True) -> list[float]:
        """Train the Cloth CVAE."""
        self._scaler.fit(pose_pca, stress)
        pn = self._scaler.norm_pose(pose_pca).astype(np.float32)
        sn = self._scaler.norm_stress(stress).astype(np.float32)

        n_val = max(1, int(len(pn) * self._cfg.val_split))
        idx   = np.random.permutation(len(pn))
        v_i, t_i = idx[:n_val], idx[n_val:]

        Pt = torch.from_numpy(pn[t_i]).to(self._device)
        St = torch.from_numpy(sn[t_i]).unsqueeze(-1).to(self._device)
        Pv = torch.from_numpy(pn[v_i]).to(self._device)
        Sv = torch.from_numpy(sn[v_i]).unsqueeze(-1).to(self._device)

        optim  = torch.optim.Adam(self._model.parameters(), lr=self._cfg.lr)
        sched  = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=self._cfg.epochs
        )
        B      = self._cfg.batch_size
        losses = []
        cfg    = self._cfg

        for ep in range(1, cfg.epochs + 1):
            self._model.train()
            perm  = torch.randperm(len(Pt))
            eloss = 0.0
            nb    = 0
            for i in range(0, len(Pt), B):
                ib  = perm[i:i + B]
                p_b = Pt[ib]
                s_b = St[ib]

                recon, mu, lv = self._model(p_b, s_b, s_b)
                rl = F.mse_loss(recon, p_b)
                kl = self._kl_loss(mu, lv)
                if cfg.lam > 0.0:
                    ph = self._physics_loss(recon, s_b)
                    ls = cfg.alpha * rl + cfg.beta * kl + cfg.lam * ph
                else:
                    ls = cfg.alpha * rl + cfg.beta * kl

                optim.zero_grad()
                ls.backward()
                optim.step()
                eloss += ls.item()
                nb    += 1

            sched.step()
            losses.append(eloss / nb)

            if verbose and ep % 20 == 0:
                self._model.eval()
                with torch.no_grad():
                    rv, mv, lv = self._model(Pv, Sv, Sv)
                    vr = F.mse_loss(rv, Pv).item()
                    vk = self._kl_loss(mv, lv).item()
                print(f"  Epoch {ep:4d}/{cfg.epochs} | loss={losses[-1]:.4e} | "
                      f"val_recon={vr:.4e} | val_kl={vk:.4e}")

        return losses

    def save(self, path: str) -> None:
        """Save model weights + scaler.

        The scaler and config are serialised as plain dicts to avoid
        pickle module-path issues when loading from a different entry point.
        """
        from dataclasses import asdict
        torch.save({
            "model_state_dict": self._model.state_dict(),
            "scaler_dict": {
                "pose_min":   self._scaler.pose_min,
                "pose_max":   self._scaler.pose_max,
                "stress_min": self._scaler.stress_min,
                "stress_max": self._scaler.stress_max,
            },
            "cfg_dict": asdict(self._cfg),
        }, path)
        print(f"Saved Cloth CVAE to: {path}")

    @classmethod
    def load(cls, path: str, stress_trainer: "StressSurrogateTrainer",
             device: str = "cpu") -> "ClothCVAETrainer":
        """Reload a previously saved trainer."""
        ckpt    = torch.load(path, map_location=device, weights_only=False)
        # Restore config — handle both old (pickled dataclass) and new (dict) format
        if "cfg_dict" in ckpt:
            cfg = ClothCVAEConfig(**ckpt["cfg_dict"])
        else:
            cfg = ckpt["cfg"]   # backward compat
        model   = ClothCVAE(cfg=cfg)
        model.load_state_dict(ckpt["model_state_dict"])
        trainer = cls(model=model, stress_trainer=stress_trainer,
                      cfg=cfg, device=device)
        # Restore scaler — handle both old (pickled object) and new (dict) format
        if "scaler_dict" in ckpt:
            sd = ckpt["scaler_dict"]
            trainer._scaler.pose_min   = sd["pose_min"]
            trainer._scaler.pose_max   = sd["pose_max"]
            trainer._scaler.stress_min = sd["stress_min"]
            trainer._scaler.stress_max = sd["stress_max"]
        else:
            trainer._scaler = ckpt["scaler"]   # backward compat
        return trainer

    def generate(self, target_stress: float, n: int = 10,
                 pca: Optional[PosePCA] = None) -> np.ndarray:
        """
        Sample n cloth initial poses with target stress.

        Args:
            target_stress: physical stress value to target
            n:             number of samples
            pca:           if provided, inverse-transforms to world_pos [n, N, 3]

        Returns:
            if pca is None:  [n, K] normalised pose PCA coords
            else:            [n, N, 3] world position arrays
        """
        sn = np.array(
            [(target_stress - self._scaler.stress_min) /
             (self._scaler.stress_max - self._scaler.stress_min + 1e-8)],
            dtype=np.float32
        )
        t = torch.from_numpy(sn).unsqueeze(0).to(self._device)

        samples_norm = self._model.sample(t, n=n).cpu().numpy()  # [n, K]
        samples_phys = self._scaler.denorm_pose(np.clip(samples_norm, 0.0, 1.0))

        if pca is not None:
            # Inverse PCA → [n, N*3] → [n, N, 3]
            n_nodes = pca.mean_.shape[0] // 3
            world   = pca.inverse_transform(samples_phys)  # [n, N*3]
            return world.reshape(n, n_nodes, 3)

        return samples_phys  # [n, K]


# ---------------------------------------------------------------------------
# Unified training script entry point
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train the Cloth CVAE on flag_simple pose/stress pairs."
    )
    p.add_argument("--pose",    default="data_flag/train/cloth_pose_pca.npy",
                   help="Path to cloth_pose_pca.npy from cloth_extractor.py")
    p.add_argument("--stress",  default="data_flag/train/cloth_stress.npy",
                   help="Path to cloth_stress.npy from cloth_extractor.py")
    p.add_argument("--pca",     default="data_flag/train/cloth_pca.pkl",
                   help="Path to cloth_pca.pkl from cloth_extractor.py")
    p.add_argument("--out",     default="checkpoints/flag-simple_cvae.pth")
    p.add_argument("--epochs",  type=int, default=300)
    p.add_argument("--latent",  type=int, default=16)
    p.add_argument("--hidden",  type=int, default=128)
    p.add_argument("--beta",    type=float, default=1e-3)
    p.add_argument("--lam",     type=float, default=0.5)
    p.add_argument("--device",  default="cpu")
    p.add_argument("--quiet",   action="store_true")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)

    print("Loading cloth pose/stress data...")
    pose_pca = np.load(args.pose).astype(np.float32)
    stress   = np.load(args.stress).astype(np.float32)
    pca      = PosePCA.load(args.pca)
    print(f"Loaded {len(pose_pca)} samples  "
          f"pose_dim={pose_pca.shape[1]}  "
          f"stress=[{stress.min():.4f}, {stress.max():.4f}]")

    # Train stress surrogate first
    print("\nTraining stress surrogate...")
    surrogate  = StressSurrogate(pose_dim=pose_pca.shape[1], hidden_size=64)
    s_trainer  = StressSurrogateTrainer(model=surrogate, device=args.device)
    s_trainer.fit(pose_pca, stress, epochs=200, verbose=not args.quiet)

    # Train Cloth CVAE
    cfg = ClothCVAEConfig(
        pose_dim=pose_pca.shape[1],
        latent_dim=args.latent,
        hidden_size=args.hidden,
        beta=args.beta,
        lam=args.lam,
        epochs=args.epochs,
    )
    model   = ClothCVAE(cfg=cfg)
    trainer = ClothCVAETrainer(model=model, stress_trainer=s_trainer,
                               cfg=cfg, device=args.device)

    print(f"\nTraining Cloth CVAE ({args.epochs} epochs)...")
    trainer.fit(pose_pca, stress, verbose=not args.quiet)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    trainer.save(args.out)

    # Demo generation
    target = float(stress.mean())
    print(f"\nGenerating 3 cloth configs at target_stress = {target:.4f} (mean):")
    samples = trainer.generate(target_stress=target, n=3, pca=pca)
    print(f"  Output world_pos shapes: {[s.shape for s in samples]}")


if __name__ == "__main__":
    main()
