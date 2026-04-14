"""
OOD Detector
=============
Wraps the existing ``NearestNeighborIndex`` (128-dim KDTree) to provide
a clean, domain-agnostic interface for out-of-distribution detection
on generated mesh designs.

Usage
-----
    from extensions.confidence.ood_detector import OODDetector

    detector = OODDetector.from_index_file("runs/embedding_index.pkl",
                                            simulator=simulator,
                                            device="cpu")
    result = detector.score(generated_graph)
    if result.is_ood:
        print("Warning: design is OOD — predictor uncertainty is high")

Design principles
-----------------
- **Single Responsibility**: ``EmbeddingExtractor`` extracts embeddings;
  ``OODDetector`` decides OOD status; ``OODResult`` carries the result.
- **Open / Closed**: swap the scoring backend by subclassing
  ``BaseOODScorer``.
- **Dependency Inversion**: ``OODDetector`` depends on the abstract
  ``BaseOODScorer``, not ``NearestNeighborIndex`` directly.
"""
from __future__ import annotations

import os
import sys
import pickle
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from torch_geometric.data import Data  # noqa: E402


# ---------------------------------------------------------------------------
# Result DTO
# ---------------------------------------------------------------------------

@dataclass
class OODResult:
    """Result of an OOD detection check."""
    confidence: float   # in [0, 1]: 1 = in-distribution, 0 = OOD
    is_ood:     bool    # True if confidence < threshold
    threshold:  float   # the threshold used
    embedding:  np.ndarray  # [128] encoder embedding


# ---------------------------------------------------------------------------
# Param-space OOD (4-D design parameter KDTree — no mesh/GNN needed)
# ---------------------------------------------------------------------------

class ParamSpaceOOD:
    """
    OOD confidence based on 4-D design parameter space.

    Scores a generated design (cx, cy, r, v_inlet) by its normalised L2 distance
    to the nearest training design in data/design_params.npy. No mesh or GNN needed.

    This is the correct OOD tool for the generate pipeline: it asks
    "are these design params within the training distribution?" rather than
    "does this mesh embed similarly to training meshes?" (which is always true
    if we use RealMeshLookup, and always false if we use synthetic meshes).

    Formula mirrors NearestNeighborIndex exactly:
        confidence = clip(1 - d_min / (train_diameter + 1e-12), 0, 1)
    where train_diameter = 95th percentile of leave-one-out NN distances in training set.
    """

    DEFAULT_THRESHOLD: float = 0.3

    def __init__(self, dataset_path: str = 'data', ood_threshold: float = DEFAULT_THRESHOLD):
        self._threshold = ood_threshold
        self._tree = None
        self._p_min = None
        self._scale = None
        self.train_diameter: float = 1.0

        npy_path = os.path.join(dataset_path, 'design_params.npy')
        if not os.path.exists(npy_path):
            return  # available == False; score() returns -1.0

        try:
            params = np.load(npy_path).astype(np.float64)  # [N, 4]: cx,cy,r,v_inlet
            if params.ndim != 2 or params.shape[1] < 4:
                return

            # Per-column min/max normalisation to [0, 1]
            p_min = params.min(axis=0)   # [4]
            p_max = params.max(axis=0)   # [4]
            scale = p_max - p_min
            scale[scale < 1e-12] = 1.0   # degenerate column guard
            self._p_min = p_min
            self._scale = scale

            norm_params = (params - p_min) / scale  # [N, 4]

            # Build KDTree
            from scipy.spatial import KDTree
            self._tree = KDTree(norm_params)

            # train_diameter = 95th percentile of leave-one-out NN distances
            # Identical formula to NearestNeighborIndex.build() in confidence/index.py
            dists, _ = self._tree.query(norm_params, k=2)  # k=2: skip self (dist=0)
            self.train_diameter = float(np.percentile(dists[:, 1], 95))
        except Exception:
            self._tree = None  # graceful degradation

    @property
    def available(self) -> bool:
        """True if design_params.npy was loaded successfully."""
        return self._tree is not None

    def score(self, cx: float, cy: float, r: float, v_inlet: float) -> 'OODResult':
        """
        Score a generated design by its distance to training params.

        Returns OODResult with confidence in [0, 1] (1 = in-distribution),
        or confidence=-1.0 if design_params.npy was not available.
        """
        if self._tree is None:
            return OODResult(
                confidence=-1.0,
                is_ood=False,
                threshold=self._threshold,
                embedding=np.zeros(4, dtype=np.float32),
            )

        query = np.array([cx, cy, r, v_inlet], dtype=np.float64)
        q_norm = (query - self._p_min) / self._scale  # normalise to [0,1]^4

        # Distance to nearest training param
        d_min, _ = self._tree.query(q_norm.reshape(1, -1), k=1)
        d_min = float(d_min[0])

        # Identical formula to NearestNeighborIndex.query() in confidence/index.py
        confidence = float(np.clip(1.0 - d_min / (self.train_diameter + 1e-12), 0.0, 1.0))

        return OODResult(
            confidence=confidence,
            is_ood=confidence < self._threshold,
            threshold=self._threshold,
            embedding=q_norm.astype(np.float32),  # repurpose embedding field for normalised params
        )


# ---------------------------------------------------------------------------
# Abstract scorer interface (Dependency Inversion / Open-Closed)
# ---------------------------------------------------------------------------

class BaseOODScorer(ABC):
    """Abstract interface for confidence scoring from an embedding."""

    @abstractmethod
    def score(self, embedding: np.ndarray) -> float:
        """Return a confidence score in [0, 1].  1 = in-distribution."""


class KDTreeScorer(BaseOODScorer):
    """
    Uses the pre-built ``NearestNeighborIndex`` KDTree to score an embedding.

    Wraps the existing confidence system without duplicating it.
    """

    def __init__(self, index) -> None:
        """
        Args:
            index: a ``NearestNeighborIndex`` instance with build() already called.
        """
        self._index = index

    def score(self, embedding: np.ndarray) -> float:
        return self._index.query(embedding)


# ---------------------------------------------------------------------------
# OOD Detector (main public API)
# ---------------------------------------------------------------------------

class OODDetector:
    """
    Out-of-distribution detector for generated mesh designs.

    Combines embedding extraction and confidence scoring.  Supports both
    CFD (Simulator) and cloth (FlagSimulator) domains transparently
    via the updated ``model.embedding.extract_embedding``.
    """

    DEFAULT_THRESHOLD: float = 0.3   # designs below this are flagged OOD

    def __init__(self,
                 scorer:    BaseOODScorer,
                 simulator,
                 device:    str = "cpu",
                 threshold: float = DEFAULT_THRESHOLD) -> None:
        self._scorer    = scorer
        self._simulator = simulator
        self._device    = device
        self._threshold = threshold

    def score(self, graph: Data) -> OODResult:
        """
        Compute OOD status for a single generated graph.

        Args:
            graph: PyG Data object (CFD: pre-transformed;
                                   cloth: raw with world_pos + face)

        Returns:
            OODResult with confidence, is_ood flag, and embedding
        """
        from model.embedding import extract_embedding   # lazy to avoid circular
        emb        = extract_embedding(self._simulator, graph, self._device)
        confidence = self._scorer.score(emb)
        return OODResult(
            confidence=confidence,
            is_ood=(confidence < self._threshold),
            threshold=self._threshold,
            embedding=emb,
        )

    def batch_score(self, graphs: list[Data]) -> list[OODResult]:
        """Score a list of graphs."""
        return [self.score(g) for g in graphs]

    @classmethod
    def from_index_file(cls, index_path: str, simulator,
                        device: str = "cpu",
                        threshold: float = DEFAULT_THRESHOLD) -> "OODDetector":
        """
        Convenience constructor: load a saved NearestNeighborIndex and
        create the full detector.

        Args:
            index_path: path to ``embedding_index.pkl`` built by train.py
            simulator:  Simulator or FlagSimulator instance
            device:     torch device string
            threshold:  OOD threshold (confidence < threshold → is_ood)
        """
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"Embedding index not found: {index_path}\n"
                "Run training (train.py) with build_confidence_index=true first."
            )
        from confidence.index import NearestNeighborIndex   # existing class
        index  = NearestNeighborIndex.load(index_path)
        scorer = KDTreeScorer(index=index)
        return cls(scorer=scorer, simulator=simulator,
                   device=device, threshold=threshold)
