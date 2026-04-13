import numpy as np
import pytest


def _random_embeddings(n: int, dim: int = 128, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((n, dim), dtype=np.float64).astype(np.float32)


def test_build_sets_backend():
    """After build(), backend is 'scipy', 'cpp', or 'faiss' depending on what's installed."""
    from confidence.index import NearestNeighborIndex
    idx = NearestNeighborIndex()
    idx.build(_random_embeddings(50))
    assert idx.backend in ("scipy", "cpp", "faiss")


def test_build_sets_train_diameter():
    """train_diameter is positive after build."""
    from confidence.index import NearestNeighborIndex
    idx = NearestNeighborIndex()
    idx.build(_random_embeddings(50))
    assert idx.train_diameter > 0.0


def test_query_in_range():
    """Score is in [0, 1] for an embedding drawn from training distribution."""
    from confidence.index import NearestNeighborIndex
    embeddings = _random_embeddings(100)
    idx = NearestNeighborIndex()
    idx.build(embeddings)
    score = idx.query(embeddings[0])
    assert 0.0 <= score <= 1.0, f"score {score} out of [0, 1]"


def test_query_ood_score_low():
    """An embedding 10x far from training data scores 0."""
    from confidence.index import NearestNeighborIndex
    embeddings = _random_embeddings(100)  # values in [0, 1)
    idx = NearestNeighborIndex()
    idx.build(embeddings)
    far_embedding = np.full(128, 100.0, dtype=np.float32)
    score = idx.query(far_embedding)
    assert score == 0.0, f"OOD score should be 0, got {score}"


def test_save_and_load(tmp_path):
    """save() then load() rebuilds index with same train_diameter."""
    from confidence.index import NearestNeighborIndex
    embeddings = _random_embeddings(50)
    idx = NearestNeighborIndex()
    idx.build(embeddings)
    diameter_before = idx.train_diameter

    path = str(tmp_path / "test_index.pkl")
    idx.save(path)

    idx2 = NearestNeighborIndex.load(path)
    assert abs(idx2.train_diameter - diameter_before) < 1e-6
    score1 = idx.query(embeddings[5])
    score2 = idx2.query(embeddings[5])
    assert abs(score1 - score2) < 1e-5


def test_diameter_is_95th_percentile():
    """train_diameter equals scipy-computed 95th percentile NN distance."""
    from confidence.index import NearestNeighborIndex
    from scipy.spatial import KDTree
    embeddings = _random_embeddings(100)
    idx = NearestNeighborIndex()
    idx.build(embeddings)

    tree = KDTree(embeddings)
    dists, _ = tree.query(embeddings, k=2)
    expected_diameter = float(np.percentile(dists[:, 1], 95))
    assert abs(idx.train_diameter - expected_diameter) < 1e-5
