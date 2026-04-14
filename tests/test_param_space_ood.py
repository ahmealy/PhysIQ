"""Tests for ParamSpaceOOD in extensions/confidence/ood_detector.py"""
import numpy as np
import pytest
from pathlib import Path

from extensions.confidence.ood_detector import ParamSpaceOOD


def _make_params_npy(tmp_path: Path, n: int = 20, seed: int = 42) -> Path:
    """Create a fake design_params.npy with n rows of [cx, cy, r, v_inlet]."""
    rng = np.random.RandomState(seed)
    params = np.column_stack([
        rng.uniform(0.15, 0.50, n),   # cx
        rng.uniform(0.10, 0.30, n),   # cy
        rng.uniform(0.02, 0.08, n),   # r
        rng.uniform(0.14, 1.14, n),   # v_inlet
    ]).astype(np.float32)
    npy_path = tmp_path / "design_params.npy"
    np.save(npy_path, params)
    return npy_path, params


class TestParamSpaceOOD:

    def test_available_false_when_no_data_file(self, tmp_path):
        ood = ParamSpaceOOD(dataset_path=str(tmp_path))
        assert not ood.available

    def test_score_returns_minus_one_when_unavailable(self, tmp_path):
        ood = ParamSpaceOOD(dataset_path=str(tmp_path))
        result = ood.score(0.2, 0.2, 0.05, 1.0)
        assert result.confidence == -1.0
        assert result.is_ood is False

    def test_available_true_with_data_file(self, tmp_path):
        _make_params_npy(tmp_path)
        ood = ParamSpaceOOD(dataset_path=str(tmp_path))
        assert ood.available

    def test_score_in_distribution(self, tmp_path):
        """Query identical to a training point → confidence close to 1."""
        _, params = _make_params_npy(tmp_path)
        ood = ParamSpaceOOD(dataset_path=str(tmp_path))
        # Query exactly the first training point
        cx, cy, r, v = params[0]
        result = ood.score(float(cx), float(cy), float(r), float(v))
        # Distance is 0 → confidence = 1.0
        assert result.confidence > 0.9, f"Expected high confidence for training point, got {result.confidence}"
        assert result.is_ood is False

    def test_score_ood_point(self, tmp_path):
        """Query far outside all training ranges → confidence == 0."""
        _make_params_npy(tmp_path)
        ood = ParamSpaceOOD(dataset_path=str(tmp_path))
        # v_inlet=100.0 is far outside training range [0.14, 1.14]
        result = ood.score(0.2, 0.2, 0.05, 100.0)
        assert result.confidence == 0.0
        assert result.is_ood is True

    def test_confidence_in_valid_range(self, tmp_path):
        """All confidence scores must be in [0, 1]."""
        _, params = _make_params_npy(tmp_path, n=50)
        ood = ParamSpaceOOD(dataset_path=str(tmp_path))
        rng = np.random.RandomState(0)
        for _ in range(20):
            # Mix in-distribution and slightly-out queries
            cx = rng.uniform(0.1, 0.6)
            cy = rng.uniform(0.05, 0.35)
            r  = rng.uniform(0.01, 0.10)
            v  = rng.uniform(0.0, 2.0)
            result = ood.score(cx, cy, r, v)
            assert 0.0 <= result.confidence <= 1.0, \
                f"Confidence {result.confidence} out of [0,1] for ({cx},{cy},{r},{v})"

    def test_train_diameter_is_95th_percentile(self, tmp_path):
        """train_diameter should equal 95th percentile of NN distances in normalised space."""
        _, params = _make_params_npy(tmp_path, n=30)
        ood = ParamSpaceOOD(dataset_path=str(tmp_path))

        # Recompute manually
        p_min = params.min(axis=0).astype(np.float64)
        p_max = params.max(axis=0).astype(np.float64)
        scale = p_max - p_min
        scale[scale < 1e-12] = 1.0
        norm = (params.astype(np.float64) - p_min) / scale

        from scipy.spatial import KDTree
        tree = KDTree(norm)
        dists, _ = tree.query(norm, k=2)
        expected = float(np.percentile(dists[:, 1], 95))

        assert abs(ood.train_diameter - expected) < 1e-6, \
            f"train_diameter {ood.train_diameter} != expected {expected}"

    def test_is_ood_flag(self, tmp_path):
        """is_ood=True for distant points; is_ood=False for training points."""
        _, params = _make_params_npy(tmp_path)
        ood = ParamSpaceOOD(dataset_path=str(tmp_path))

        # Training point → in-distribution
        cx, cy, r, v = params[0]
        assert ood.score(float(cx), float(cy), float(r), float(v)).is_ood is False

        # Far point → OOD
        assert ood.score(0.2, 0.2, 0.05, 100.0).is_ood is True

    def test_uses_all_four_params(self, tmp_path):
        """Changing only v_inlet should change the confidence score."""
        _, params = _make_params_npy(tmp_path, n=50)
        ood = ParamSpaceOOD(dataset_path=str(tmp_path))

        cx, cy, r = 0.25, 0.20, 0.05
        score_low_v  = ood.score(cx, cy, r, 0.14)   # near training min v_inlet
        score_high_v = ood.score(cx, cy, r, 50.0)   # far outside training v_inlet

        assert score_low_v.confidence != score_high_v.confidence, \
            "Confidence should differ when v_inlet changes — 4th param must be used"
        assert score_high_v.confidence < score_low_v.confidence, \
            "Far v_inlet should have lower confidence than in-range v_inlet"
