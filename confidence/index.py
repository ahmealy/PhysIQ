"""
NearestNeighborIndex — k-d tree nearest-neighbor index over trajectory embeddings.

Default backend: scipy.spatial.KDTree — always available.
Optional backend: C++ KDTree via pybind11 — compile once with:
    cd confidence && pip install pybind11 && cmake . && make

Confidence score formula:
    score = clip(1 - d_min / train_diameter, 0, 1)
    where train_diameter = 95th percentile of NN distances within training set.
"""
import pickle
import numpy as np
from scipy.spatial import KDTree


class NearestNeighborIndex:
    """
    Nearest-neighbor index over [N, 128] trajectory embeddings.
    """

    def __init__(self):
        self.backend: str = "scipy"
        self.embeddings: np.ndarray = np.empty((0, 128), dtype=np.float32)
        self.train_diameter: float = 1.0
        self._scipy_tree: KDTree = None
        self._cpp_tree = None
        self._faiss_index = None

    def build(self, embeddings: np.ndarray) -> None:
        """Build the index from training embeddings [N_train, dim].

        Backend priority (fastest available wins):
          1. FAISS IndexFlatL2  — exact, highly optimised  (pip install faiss-cpu)
          2. C++ KDTree         — exact, pybind11           (cmake && make)
          3. scipy KDTree       — exact, always available   (fallback)
        """
        self.embeddings = embeddings.astype(np.float32)

        # Always build scipy tree (used for train_diameter computation + fallback)
        self._scipy_tree = KDTree(self.embeddings)
        self.backend = "scipy"

        # Try C++ backend (opt-in, compiled once)
        self._cpp_tree = None
        try:
            import os, sys
            confidence_dir = os.path.dirname(os.path.abspath(__file__))
            if confidence_dir not in sys.path:
                sys.path.insert(0, confidence_dir)
            from _kdtree import KDTree as CppKDTree  # type: ignore
            self._cpp_tree = CppKDTree(self.embeddings)
            self.backend = "cpp"
        except ImportError:
            pass

        # Try FAISS (fastest — overrides C++ if available)
        self._faiss_index = None
        try:
            import faiss  # type: ignore
            n, dim = self.embeddings.shape
            idx = faiss.IndexFlatL2(dim)
            idx.add(self.embeddings)
            self._faiss_index = idx
            self.backend = "faiss"
        except ImportError:
            pass

        # Compute train_diameter = 95th percentile of NN distances (k=2 to skip self)
        dists, _ = self._scipy_tree.query(self.embeddings, k=2)
        self.train_diameter = float(np.percentile(dists[:, 1], 95))

    def query(self, embedding: np.ndarray) -> float:
        """
        Returns confidence score in [0, 1].
        score = clip(1 - d_min / train_diameter, 0, 1)
        """
        if self._scipy_tree is None:
            raise RuntimeError("Call build() before query()")
        q = embedding.reshape(1, -1).astype(np.float32)

        if self._faiss_index is not None:
            dists_sq, _ = self._faiss_index.search(q, 1)
            d_min = float(np.sqrt(max(dists_sq[0, 0], 0.0)))
        elif self._cpp_tree is not None:
            d_min = float(self._cpp_tree.query(q, k=1)[0])
        else:
            dist, _ = self._scipy_tree.query(q, k=1)
            d_min = float(np.asarray(dist).flat[0])

        return float(np.clip(1.0 - d_min / (self.train_diameter + 1e-12), 0.0, 1.0))

    def save(self, path: str) -> None:
        """Serialize index to pickle file."""
        with open(path, "wb") as f:
            pickle.dump({
                "embeddings":     self.embeddings,
                "train_diameter": self.train_diameter,
            }, f)

    @classmethod
    def load(cls, path: str) -> "NearestNeighborIndex":
        """Load from pickle file and rebuild trees."""
        with open(path, "rb") as f:
            d = pickle.load(f)
        obj = cls()
        obj.build(d["embeddings"])  # rebuilds trees from embeddings
        # Restore stored train_diameter (may differ from recomputed if manually set)
        obj.train_diameter = float(d["train_diameter"])
        return obj
