"""
Tests for free-bits KL divergence in CFD and Cloth CVAEs.

Free-bits: KL loss for each latent dimension is clamped to a minimum
threshold (free_bits). Dimensions whose KL < free_bits contribute 0
to the loss — preventing posterior collapse on unused dimensions.

Formula (per dim):
    kl_dim_i = -0.5 * mean_batch(1 + logvar_i - mu_i² - exp(logvar_i))
    kl_free_i = max(kl_dim_i, free_bits)
    kl_loss   = sum_i(kl_free_i)

Key behavioural contracts tested here:
  1. When free_bits=0 the result equals the standard KL formula.
  2. When a dimension's KL is below free_bits, it contributes exactly
     free_bits to the loss (not the raw KL value).
  3. When a dimension's KL is above free_bits, it contributes its raw KL.
  4. CVAEConfig and ClothCVAEConfig expose a free_bits field defaulting to 0.0.
  5. CVAETrainer._kl_loss and ClothCVAETrainer._kl_loss accept and use
     the free_bits value from their config.
  6. The training loop in fit() uses the free-bits KL, not the old formula.
  7. Gradient flows through the free-bits KL (no detach breaks the graph).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
import numpy as np

from extensions.generative.cvae_cfd import CVAEConfig, CVAETrainer, CFDCVAE
from extensions.generative.cvae_cloth import ClothCVAEConfig, ClothCVAETrainer, ClothCVAE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _standard_kl(mu, logvar):
    """Reference: standard KL per scalar (no free-bits)."""
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def _make_cfd_trainer(free_bits: float = 0.0):
    from extensions.generative.drag_surrogate import DragSurrogate, DragSurrogateTrainer
    cfg      = CVAEConfig(free_bits=free_bits, epochs=1, batch_size=4)
    model    = CFDCVAE(cfg=cfg)
    surrogate = DragSurrogate()
    trainer  = CVAETrainer(model=model, surrogate=surrogate, cfg=cfg)
    return trainer


def _make_cloth_trainer(free_bits: float = 0.0):
    from extensions.generative.cvae_cloth import (
        ClothCVAE, ClothCVAETrainer, StressSurrogate, StressSurrogateTrainer
    )
    cfg       = ClothCVAEConfig(free_bits=free_bits, epochs=1, batch_size=4)
    model     = ClothCVAE(cfg=cfg)
    s_model   = StressSurrogate(pose_dim=cfg.pose_dim)
    s_trainer = StressSurrogateTrainer(s_model)
    trainer   = ClothCVAETrainer(model=model, stress_trainer=s_trainer, cfg=cfg)
    return trainer


# ---------------------------------------------------------------------------
# 1. CVAEConfig exposes free_bits field defaulting to 0.0
# ---------------------------------------------------------------------------

def test_cvae_config_has_free_bits_field():
    """CVAEConfig must have a free_bits attribute defaulting to 0.0."""
    cfg = CVAEConfig()
    assert hasattr(cfg, "free_bits"), "CVAEConfig missing free_bits field"
    assert cfg.free_bits == 0.0


def test_cloth_cvae_config_has_free_bits_field():
    """ClothCVAEConfig must have a free_bits attribute defaulting to 0.0."""
    cfg = ClothCVAEConfig()
    assert hasattr(cfg, "free_bits"), "ClothCVAEConfig missing free_bits field"
    assert cfg.free_bits == 0.0


# ---------------------------------------------------------------------------
# 2. free_bits=0 → identical to standard KL
# ---------------------------------------------------------------------------

def test_cfd_kl_free_bits_zero_equals_standard_kl():
    """With free_bits=0 the CFD KL loss equals the standard formula."""
    trainer = _make_cfd_trainer(free_bits=0.0)
    torch.manual_seed(42)
    mu     = torch.randn(8, 16)
    logvar = torch.randn(8, 16)

    result   = trainer._kl_loss(mu, logvar)
    expected = _standard_kl(mu, logvar)
    assert torch.isclose(result, expected, atol=1e-6), (
        f"free_bits=0 KL={result.item():.6f} != standard KL={expected.item():.6f}"
    )


def test_cloth_kl_free_bits_zero_equals_standard_kl():
    """With free_bits=0 the Cloth KL loss equals the standard formula."""
    trainer = _make_cloth_trainer(free_bits=0.0)
    torch.manual_seed(7)
    mu     = torch.randn(8, 16)
    logvar = torch.randn(8, 16)

    result   = trainer._kl_loss(mu, logvar)
    expected = _standard_kl(mu, logvar)
    assert torch.isclose(result, expected, atol=1e-6), (
        f"free_bits=0 KL={result.item():.6f} != standard KL={expected.item():.6f}"
    )


# ---------------------------------------------------------------------------
# 3. Dimension below free_bits contributes exactly free_bits
# ---------------------------------------------------------------------------

def test_cfd_kl_clamps_low_kl_dim_to_free_bits():
    """
    A dimension with near-zero KL (mu≈0, logvar≈0 → KL≈0) must contribute
    exactly free_bits to the loss, not its raw ~0 value.
    """
    free_bits = 0.05
    trainer   = _make_cfd_trainer(free_bits=free_bits)

    # mu=0, logvar=0 → KL_dim = -0.5*(1+0-0-1) = 0  →  should be clamped to free_bits
    latent_dim = trainer._cfg.latent_dim
    mu     = torch.zeros(4, latent_dim)
    logvar = torch.zeros(4, latent_dim)

    result = trainer._kl_loss(mu, logvar)
    # Every dim is at 0 < free_bits=0.05, so each contributes free_bits
    expected = torch.tensor(float(latent_dim) * free_bits)
    assert torch.isclose(result, expected, atol=1e-5), (
        f"All-zero KL with free_bits={free_bits}: got {result.item():.6f}, "
        f"expected {expected.item():.6f} ({latent_dim} dims × {free_bits})"
    )


def test_cloth_kl_clamps_low_kl_dim_to_free_bits():
    """Same test for Cloth CVAE trainer."""
    free_bits = 0.05
    trainer   = _make_cloth_trainer(free_bits=free_bits)

    latent_dim = trainer._cfg.latent_dim
    mu     = torch.zeros(4, latent_dim)
    logvar = torch.zeros(4, latent_dim)

    result   = trainer._kl_loss(mu, logvar)
    expected = torch.tensor(float(latent_dim) * free_bits)
    assert torch.isclose(result, expected, atol=1e-5), (
        f"Cloth all-zero KL with free_bits={free_bits}: got {result.item():.6f}, "
        f"expected {expected.item():.6f}"
    )


# ---------------------------------------------------------------------------
# 4. Dimension above free_bits contributes its raw KL (not free_bits)
# ---------------------------------------------------------------------------

def test_cfd_kl_passes_through_high_kl_dim():
    """
    A dimension with KL >> free_bits must contribute its raw KL value,
    not the free_bits floor.

    The free-bits formula sums over dims (not means), so the expected value
    when all dims are identical is:  latent_dim × (per-dim KL mean over batch).
    """
    free_bits  = 0.05
    trainer    = _make_cfd_trainer(free_bits=free_bits)
    latent_dim = trainer._cfg.latent_dim

    # Large mu → large KL >> free_bits  (mu=3, logvar=0 → KL_dim = 0.5*(9+0-1) = 4.0)
    mu     = torch.full((4, latent_dim), 3.0)
    logvar = torch.zeros(4, latent_dim)

    result_fb = trainer._kl_loss(mu, logvar)

    # per-dim KL mean over batch: -0.5*(1 + 0 - 9 - 1) = 4.5
    per_dim_kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=0)
    # free-bits sums across dims; all dims >> free_bits so clamp has no effect
    expected = per_dim_kl.sum()

    assert torch.isclose(result_fb, expected, atol=1e-5), (
        f"High-KL dims: free_bits result {result_fb.item():.6f} "
        f"!= sum-of-per-dim KL {expected.item():.6f}"
    )


# ---------------------------------------------------------------------------
# 5. Gradient flows through free-bits KL
# ---------------------------------------------------------------------------

def test_cfd_kl_gradient_flows():
    """Gradient must be non-zero through the free-bits KL loss."""
    trainer = _make_cfd_trainer(free_bits=0.05)
    mu      = torch.randn(4, 16, requires_grad=True)
    logvar  = torch.randn(4, 16, requires_grad=True)

    loss = trainer._kl_loss(mu, logvar)
    loss.backward()

    assert mu.grad is not None, "No gradient on mu"
    assert logvar.grad is not None, "No gradient on logvar"
    assert mu.grad.abs().sum() > 0, "mu gradient is all zeros"
    assert logvar.grad.abs().sum() > 0, "logvar gradient is all zeros"


# ---------------------------------------------------------------------------
# 6. Free-bits reduces loss compared to standard when KL < threshold
# ---------------------------------------------------------------------------

def test_free_bits_loss_ge_standard_kl_when_kl_near_zero():
    """
    When raw KL is near zero, free-bits loss must be >= standard KL
    (because we clamp from below at free_bits).
    """
    trainer_fb  = _make_cfd_trainer(free_bits=0.05)
    trainer_std = _make_cfd_trainer(free_bits=0.0)

    mu     = torch.zeros(4, 16)
    logvar = torch.zeros(4, 16)

    loss_fb  = trainer_fb._kl_loss(mu, logvar)
    loss_std = trainer_std._kl_loss(mu, logvar)

    assert loss_fb >= loss_std, (
        f"Free-bits loss {loss_fb.item():.4f} should be >= "
        f"standard loss {loss_std.item():.4f} when KL is near zero"
    )


# ---------------------------------------------------------------------------
# 7. fit() runs without error with free_bits set (smoke test)
# ---------------------------------------------------------------------------

def test_cfd_trainer_fit_with_free_bits_smoke():
    """CVAETrainer.fit() completes one epoch when free_bits=0.05."""
    trainer = _make_cfd_trainer(free_bits=0.05)
    np.random.seed(0)
    params = np.random.rand(20, 4).astype(np.float32)
    drag   = np.random.rand(20).astype(np.float32) * 0.1

    losses = trainer.fit(params, drag, verbose=False)
    assert len(losses) == 1
    assert np.isfinite(losses[0]), f"Loss is not finite: {losses[0]}"


def test_cloth_trainer_fit_with_free_bits_smoke():
    """ClothCVAETrainer.fit() completes one epoch when free_bits=0.05.

    Uses lam=0.0 to disable the physics (stress-surrogate) loss term so the
    test does not require a pre-fitted StressSurrogate scaler.
    """
    from extensions.generative.cvae_cloth import (
        ClothCVAE, ClothCVAETrainer, StressSurrogate, StressSurrogateTrainer
    )
    cfg       = ClothCVAEConfig(free_bits=0.05, epochs=1, batch_size=4, lam=0.0)
    model     = ClothCVAE(cfg=cfg)
    s_model   = StressSurrogate(pose_dim=cfg.pose_dim)
    s_trainer = StressSurrogateTrainer(s_model)
    trainer   = ClothCVAETrainer(model=model, stress_trainer=s_trainer, cfg=cfg)

    np.random.seed(1)
    pose   = np.random.rand(20, 16).astype(np.float32)
    stress = np.random.rand(20).astype(np.float32) * 2.0

    losses = trainer.fit(pose, stress, verbose=False)
    assert len(losses) == 1
    assert np.isfinite(losses[0]), f"Loss is not finite: {losses[0]}"
