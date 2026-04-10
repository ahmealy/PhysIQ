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
