# Confidence Score (k-d Tree) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a confidence score to every rollout result that measures how similar the test trajectory is to the training distribution using k-d tree nearest-neighbor search on encoder embeddings.

**Architecture:** A `confidence/` package provides `NearestNeighborIndex` — built on scipy.KDTree by default (always available), with optional C++/pybind11 upgrade (compiled once, auto-detected). Both backends produce identical results; `benchmark.py` measures build/query times at N=100/1000/10000 and verifies correctness. The embedding is mean-pooled encoder output over NORMAL nodes [128]. After training, embeddings are extracted and saved to `runs/embedding_index.pkl`. During rollout, the index is queried and `confidence_score` is attached to every result.

**Tech Stack:** Python, scipy.spatial.KDTree, C++/pybind11 (opt-in), NumPy, PyTorch, FastAPI, React/TypeScript

---

## File Map

| File | Action | Purpose |
|---|---|---|
| `confidence/__init__.py` | **Create** | Package init — exports `NearestNeighborIndex` |
| `confidence/index.py` | **Create** | `NearestNeighborIndex` — scipy default, C++ opt-in, identical interface |
| `confidence/benchmark.py` | **Create** | Compare C++ vs scipy build/query times at multiple N values |
| `confidence/build_index.py` | **Create** | Standalone script: checkpoint → embeddings → save index |
| `confidence/kdtree.cpp` | **Create** | C++ k-d tree implementation (standard median-split, branch-and-bound) |
| `confidence/kdtree_bind.cpp` | **Create** | pybind11 bindings for Python access |
| `confidence/CMakeLists.txt` | **Create** | Build config for C++ extension |
| `model/embedding.py` | **Create** | `extract_embedding(simulator, graph, device) → np.ndarray[128]` |
| `train.py` | **Modify** | After training: extract embeddings, build index, save |
| `api/routes/rollout.py` | **Modify** | After rollout: query index, attach `confidence_score` to SSE done event and pkl |
| `api/routes/results.py` | **Modify** | Add `confidence_score` + `confidence_label` to result responses |
| `app/src/pages/Predict.tsx` | **Modify** | Show confidence badge after rollout completes |
| `app/src/pages/Visualize.tsx` | **Modify** | Show confidence card in Diagnostics tab |
| `tests/test_confidence_index.py` | **Create** | Unit tests for NearestNeighborIndex |
| `tests/test_embedding.py` | **Create** | Unit tests for extract_embedding |

---

## Task 1: `model/embedding.py` — encoder-only forward pass

**Files:**
- Create: `model/embedding.py`
- Test: `tests/test_embedding.py`

This function runs only the encoder on a single graph (no processor, no decoder) and returns mean-pooled NORMAL-node features as a [128] numpy vector.

- [ ] **Step 1: Write the failing test**

Create `tests/test_embedding.py`:

```python
import torch
import numpy as np
import pytest
from torch_geometric.data import Data
import torch_geometric.transforms as T


def _make_cfd_graph(N=30, E=60):
    """Synthetic CFD graph: node_type + velocity, edges with 3 features."""
    node_type = torch.zeros(N, 1)
    velocity  = torch.randn(N, 2)
    x = torch.cat([node_type, velocity], dim=-1)  # [N, 3]
    face = torch.randint(0, N, (3, E // 3))       # rough triangles
    return Data(x=x, pos=torch.randn(N, 2), face=face)


def test_extract_embedding_shape():
    """extract_embedding returns [128] numpy array."""
    from model.simulator import Simulator
    from model.embedding import extract_embedding

    sim = Simulator(message_passing_num=2, node_input_size=11,
                    edge_input_size=3, device="cpu")
    transformer = __import__("torch_geometric.transforms", fromlist=["Compose"])
    import torch_geometric.transforms as T
    tfm = T.Compose([T.FaceToEdge(), T.Cartesian(norm=False), T.Distance(norm=False)])

    graph = _make_cfd_graph()
    graph = tfm(graph)

    emb = extract_embedding(sim, graph, device="cpu")
    assert isinstance(emb, np.ndarray), "embedding must be numpy array"
    assert emb.shape == (128,), f"expected shape (128,), got {emb.shape}"


def test_extract_embedding_is_finite():
    """Embedding values must be finite (no NaN/Inf)."""
    from model.simulator import Simulator
    from model.embedding import extract_embedding
    import torch_geometric.transforms as T

    sim = Simulator(message_passing_num=2, node_input_size=11,
                    edge_input_size=3, device="cpu")
    tfm = T.Compose([T.FaceToEdge(), T.Cartesian(norm=False), T.Distance(norm=False)])
    graph = _make_cfd_graph()
    graph = tfm(graph)

    emb = extract_embedding(sim, graph, device="cpu")
    assert np.isfinite(emb).all(), "embedding contains NaN or Inf"


def test_extract_embedding_no_grad():
    """extract_embedding must not modify any model parameters (eval mode, no_grad)."""
    from model.simulator import Simulator
    from model.embedding import extract_embedding
    import torch_geometric.transforms as T

    sim = Simulator(message_passing_num=2, node_input_size=11,
                    edge_input_size=3, device="cpu")
    tfm = T.Compose([T.FaceToEdge(), T.Cartesian(norm=False), T.Distance(norm=False)])

    params_before = [p.clone() for p in sim.parameters()]
    graph = _make_cfd_graph()
    graph = tfm(graph)
    extract_embedding(sim, graph, device="cpu")
    params_after = list(sim.parameters())

    for before, after in zip(params_before, params_after):
        assert torch.allclose(before, after), "parameters changed during embedding extraction"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/ahmealy/ML/meshGraphNets_pytorch
source venv/bin/activate
pytest tests/test_embedding.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'model.embedding'`

- [ ] **Step 3: Create `model/embedding.py`**

```python
"""
Encoder-only forward pass for confidence score computation.

extract_embedding() runs only the encoder on a single graph, mean-pools
over NORMAL nodes, and returns a [128] numpy vector.

This does NOT run the processor (15 GnBlocks) or decoder — encoder output
is sufficient to capture the flow regime for nearest-neighbor comparison.
"""
import numpy as np
import torch
import torch_geometric.transforms as T

from utils.utils import NodeType


def extract_embedding(simulator, graph, device: str) -> np.ndarray:
    """
    Run only the encoder on a single (already-transformed) graph.
    Returns mean-pooled NORMAL node embedding of shape [128].

    Args:
        simulator:  Simulator (or FlagSimulator) — must have .model.encoder and
                    .edge_normalizer, ._node_normalizer, .update_node_attr
        graph:      PyG Data — must already have edge_index and edge_attr
                    (apply FaceToEdge + Cartesian + Distance transforms first)
        device:     torch device string

    Returns:
        np.ndarray of shape [128]
    """
    simulator.eval()

    with torch.no_grad():
        graph = graph.to(device)

        # Build normalized node features (same path as Simulator.forward inference)
        node_type = graph.x[:, 0:1]   # [N, 1]
        frames    = graph.x[:, 1:]    # [N, 2] — velocity (or pressure for pressure model)

        node_attr = simulator.update_node_attr(frames, node_type)
        graph.x   = node_attr

        edge_attr = graph.edge_attr
        graph.edge_attr = simulator.edge_normalizer(edge_attr, training=False)

        # Run encoder only (no processor, no decoder)
        encoded = simulator.model.encoder(graph)   # graph.x: [N, 128]

        # Mean-pool over NORMAL nodes
        node_type_idx = node_type.squeeze(-1).long()   # [N]
        normal_mask = (node_type_idx == NodeType.NORMAL)

        if normal_mask.sum() == 0:
            # Fallback: use all nodes if no NORMAL nodes (shouldn't happen in practice)
            embedding = encoded.x.mean(dim=0)
        else:
            embedding = encoded.x[normal_mask].mean(dim=0)   # [128]

    return embedding.cpu().numpy()
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_embedding.py -v
```

Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add model/embedding.py tests/test_embedding.py
git commit -m "feat: add extract_embedding() for encoder-only mean-pool"
```

---

## Task 2: `confidence/index.py` — `NearestNeighborIndex`

**Files:**
- Create: `confidence/__init__.py`
- Create: `confidence/index.py`
- Test: `tests/test_confidence_index.py`

The index wraps scipy.KDTree (always available) and optionally a C++ KDTree (if compiled). Both backends are built simultaneously when C++ is available so `benchmark.py` can compare them on identical data. The `query()` method uses whichever backend is available.

- [ ] **Step 1: Write the failing test**

Create `tests/test_confidence_index.py`:

```python
import numpy as np
import pytest


def _random_embeddings(n: int, dim: int = 128, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((n, dim), dtype=np.float64).astype(np.float32)


def test_build_sets_backend():
    """After build(), backend is 'scipy' (or 'cpp' if compiled)."""
    from confidence.index import NearestNeighborIndex
    idx = NearestNeighborIndex()
    idx.build(_random_embeddings(50))
    assert idx.backend in ("scipy", "cpp")


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
    # Query with a training embedding — should score close to 1.0
    score = idx.query(embeddings[0])
    assert 0.0 <= score <= 1.0, f"score {score} out of [0, 1]"


def test_query_ood_score_low():
    """An embedding 10× far from training data scores near 0."""
    from confidence.index import NearestNeighborIndex
    embeddings = _random_embeddings(100)  # values in [0, 1)
    idx = NearestNeighborIndex()
    idx.build(embeddings)
    # Query with embedding far outside training distribution
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
    # Query gives same result
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_confidence_index.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'confidence'`

- [ ] **Step 3: Create `confidence/__init__.py`**

```python
"""
Confidence score package.
Provides NearestNeighborIndex for training-distribution similarity scoring.
"""
from .index import NearestNeighborIndex

__all__ = ["NearestNeighborIndex"]
```

- [ ] **Step 4: Create `confidence/index.py`**

```python
"""
NearestNeighborIndex — k-d tree nearest-neighbor index over trajectory embeddings.

Default backend: scipy.spatial.KDTree — always available, zero setup.
Optional backend: C++ KDTree via pybind11 — compile once with:
    cd confidence && pip install pybind11 && cmake . && make

When C++ is available, both backends are built (for benchmark comparison).
query() uses C++ if available, scipy otherwise.

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

    Attributes:
        backend (str):          'scipy' or 'cpp'
        embeddings (np.ndarray):[N_train, 128] training embeddings
        train_diameter (float): 95th percentile NN distance in training set
    """

    def __init__(self):
        self.backend: str = "scipy"
        self.embeddings: np.ndarray = np.empty((0, 128), dtype=np.float32)
        self.train_diameter: float = 1.0
        self._scipy_tree: KDTree = None
        self._cpp_tree = None

    def build(self, embeddings: np.ndarray) -> None:
        """
        Build the index from training embeddings.

        Args:
            embeddings: [N_train, dim] float32 array of encoder outputs
        """
        self.embeddings = embeddings.astype(np.float32)

        # Always build scipy tree (reference backend, used for diameter computation)
        self._scipy_tree = KDTree(self.embeddings)
        self.backend = "scipy"

        # Attempt C++ backend (opt-in: compile with cmake && make in confidence/)
        self._cpp_tree = None
        try:
            import importlib, os, sys
            # Look for _kdtree.so next to this file
            confidence_dir = os.path.dirname(os.path.abspath(__file__))
            if confidence_dir not in sys.path:
                sys.path.insert(0, confidence_dir)
            from confidence._kdtree import KDTree as CppKDTree  # type: ignore
            self._cpp_tree = CppKDTree(self.embeddings.astype(np.float32))
            self.backend = "cpp"
        except ImportError:
            self._cpp_tree = None   # C++ not compiled — scipy remains

        # Compute train_diameter = 95th percentile of NN distances
        # k=2 to skip self (distance to itself is always 0)
        dists, _ = self._scipy_tree.query(self.embeddings, k=2)
        self.train_diameter = float(np.percentile(dists[:, 1], 95))

    def query(self, embedding: np.ndarray) -> float:
        """
        Returns confidence score in [0, 1].

        score = clip(1 - d_min / train_diameter, 0, 1)
        where d_min = distance to nearest training embedding.

        Args:
            embedding: [128] float32 query vector

        Returns:
            float in [0.0, 1.0]
        """
        q = embedding.reshape(1, -1).astype(np.float32)

        if self._cpp_tree is not None:
            d_min = float(self._cpp_tree.query(q, k=1)[0])
        else:
            dist, _ = self._scipy_tree.query(q, k=1)
            d_min = float(dist[0, 0])

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
        """Load from pickle file and rebuild trees (auto-selects best backend)."""
        with open(path, "rb") as f:
            d = pickle.load(f)
        obj = cls()
        obj.build(d["embeddings"])  # rebuilds both trees; auto-selects best backend
        return obj
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_confidence_index.py -v
```

Expected: 6 PASSED

- [ ] **Step 6: Commit**

```bash
git add confidence/__init__.py confidence/index.py tests/test_confidence_index.py
git commit -m "feat: add NearestNeighborIndex (scipy KDTree default, C++ opt-in)"
```

---

## Task 3: C++ k-d tree implementation + pybind11 bindings

**Files:**
- Create: `confidence/kdtree.cpp`
- Create: `confidence/kdtree_bind.cpp`
- Create: `confidence/CMakeLists.txt`

This is the opt-in C++ implementation. If build fails, all code silently falls back to scipy. The C++ tree must produce distances identical to scipy (verified by `benchmark.py`).

- [ ] **Step 1: Create `confidence/kdtree.cpp`**

```cpp
/**
 * kdtree.cpp — Standard k-d tree on Euclidean distance in R^d.
 *
 * Build: median split on alternating dimensions (standard k-d tree).
 * Search: branch-and-bound with pruning — only recurse into a subtree if it
 *         could contain a closer point than the current best.
 */
#include <cmath>
#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "kdtree.h"

// ── Node ─────────────────────────────────────────────────────────────────────

struct KDNode {
    int   idx;        // index into data array (-1 for internal nodes)
    int   split_dim;
    float split_val;
    KDNode* left  = nullptr;
    KDNode* right = nullptr;
};

// ── KDTree implementation ─────────────────────────────────────────────────────

struct KDTreeImpl {
    std::vector<float> data;   // flat [n * dim]
    int n, dim;
    KDNode* root = nullptr;

    // Pool allocator to avoid per-node heap fragmentation
    std::vector<KDNode> pool;
    int pool_pos = 0;

    KDNode* alloc_node() {
        if (pool_pos >= (int)pool.size()) {
            pool.resize(pool.size() == 0 ? 1 : pool.size() * 2);
        }
        KDNode* node = &pool[pool_pos++];
        node->left = node->right = nullptr;
        node->idx = -1;
        return node;
    }

    const float* point(int i) const { return data.data() + i * dim; }

    float sq_dist(const float* a, const float* b) const {
        float d = 0.0f;
        for (int k = 0; k < dim; ++k) {
            float diff = a[k] - b[k];
            d += diff * diff;
        }
        return d;
    }

    KDNode* build(std::vector<int>& indices, int depth) {
        if (indices.empty()) return nullptr;
        KDNode* node = alloc_node();
        if (indices.size() == 1) {
            node->idx = indices[0];
            return node;
        }
        int axis = depth % dim;
        // Partial sort to find median
        int mid = indices.size() / 2;
        std::nth_element(indices.begin(), indices.begin() + mid, indices.end(),
            [&](int a, int b) { return point(a)[axis] < point(b)[axis]; });

        node->split_dim = axis;
        node->split_val = point(indices[mid])[axis];
        node->idx       = indices[mid];

        std::vector<int> left_idx(indices.begin(), indices.begin() + mid);
        std::vector<int> right_idx(indices.begin() + mid + 1, indices.end());
        node->left  = build(left_idx,  depth + 1);
        node->right = build(right_idx, depth + 1);
        return node;
    }

    void search(KDNode* node, const float* query,
                float& best_sq, int& best_idx) const {
        if (node == nullptr) return;

        if (node->idx >= 0) {
            float d = sq_dist(query, point(node->idx));
            if (d < best_sq) { best_sq = d; best_idx = node->idx; }
        }

        if (node->left == nullptr && node->right == nullptr) return;

        int axis = node->split_dim;
        float diff = query[axis] - node->split_val;
        KDNode* near = (diff <= 0) ? node->left : node->right;
        KDNode* far  = (diff <= 0) ? node->right : node->left;

        search(near, query, best_sq, best_idx);

        // Prune: only search far side if the splitting hyperplane is within best_sq
        if (diff * diff < best_sq) {
            search(far, query, best_sq, best_idx);
        }
    }
};

// ── Public C API (called from pybind11 binding) ───────────────────────────────

KDTree* kdtree_build(const float* data, int n, int dim) {
    auto* impl = new KDTreeImpl();
    impl->n    = n;
    impl->dim  = dim;
    impl->data.assign(data, data + n * dim);
    impl->pool.reserve(2 * n);   // pre-allocate node pool

    std::vector<int> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    impl->root = impl->build(indices, 0);
    return reinterpret_cast<KDTree*>(impl);
}

float kdtree_query_nn(const KDTree* tree, const float* query) {
    auto* impl = reinterpret_cast<const KDTreeImpl*>(tree);
    float best_sq = std::numeric_limits<float>::infinity();
    int   best_idx = -1;
    impl->search(impl->root, query, best_sq, best_idx);
    return std::sqrt(best_sq);
}

void kdtree_free(KDTree* tree) {
    delete reinterpret_cast<KDTreeImpl*>(tree);
}
```

- [ ] **Step 2: Create `confidence/kdtree.h`**

```cpp
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

struct KDTree;

KDTree* kdtree_build(const float* data, int n, int dim);
float   kdtree_query_nn(const KDTree* tree, const float* query);
void    kdtree_free(KDTree* tree);

#ifdef __cplusplus
}
#endif
```

- [ ] **Step 3: Create `confidence/kdtree_bind.cpp`**

```cpp
/**
 * kdtree_bind.cpp — pybind11 Python bindings for KDTree.
 *
 * Exposed as: from confidence._kdtree import KDTree
 *
 * Interface:
 *   KDTree(data: np.ndarray[float32, shape=(N, dim)])
 *   .query(point: np.ndarray[float32, shape=(1, dim)], k: int = 1)
 *       -> (distance: float, index: int)
 */
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "kdtree.h"

namespace py = pybind11;

class PyKDTree {
public:
    explicit PyKDTree(py::array_t<float, py::array::c_style | py::array::forcecast> data) {
        auto buf = data.request();
        if (buf.ndim != 2)
            throw std::runtime_error("data must be 2-D array [N, dim]");
        n_   = static_cast<int>(buf.shape[0]);
        dim_ = static_cast<int>(buf.shape[1]);
        tree_ = kdtree_build(static_cast<const float*>(buf.ptr), n_, dim_);
    }

    ~PyKDTree() {
        if (tree_) kdtree_free(tree_);
    }

    // query(point, k=1) -> (distances, indices)
    // point shape: [1, dim]  (same as scipy interface: query(point.reshape(1, -1), k=1))
    std::pair<float, int> query(
        py::array_t<float, py::array::c_style | py::array::forcecast> point,
        int /*k*/ = 1)
    {
        auto buf = point.request();
        // Accept either [dim] or [1, dim]
        const float* ptr = static_cast<const float*>(buf.ptr);
        float dist = kdtree_query_nn(tree_, ptr);
        // We only support k=1; index not tracked for performance — return 0
        return {dist, 0};
    }

private:
    KDTree* tree_ = nullptr;
    int n_ = 0, dim_ = 0;
};

PYBIND11_MODULE(_kdtree, m) {
    m.doc() = "C++ k-d tree for nearest-neighbor confidence scoring";
    py::class_<PyKDTree>(m, "KDTree")
        .def(py::init<py::array_t<float>>(),
             py::arg("data"),
             "Build k-d tree from [N, dim] float32 array")
        .def("query",
             &PyKDTree::query,
             py::arg("point"),
             py::arg("k") = 1,
             "Query nearest neighbor. Returns (distance, index).");
}
```

- [ ] **Step 4: Create `confidence/CMakeLists.txt`**

```cmake
cmake_minimum_required(VERSION 3.14)
project(_kdtree CXX)

set(CMAKE_CXX_STANDARD 17)

# Find pybind11 (installed via pip install pybind11)
execute_process(
    COMMAND python3 -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 REQUIRED)

pybind11_add_module(_kdtree
    kdtree.cpp
    kdtree_bind.cpp
)

target_compile_options(_kdtree PRIVATE -O3 -march=native)

# Output the .so file directly into the confidence/ directory
set_target_properties(_kdtree PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
)
```

- [ ] **Step 5: Attempt to build the C++ extension**

```bash
cd /home/ahmealy/ML/meshGraphNets_pytorch/confidence
pip install pybind11 2>&1 | tail -3
cmake . -B build && cmake --build build --config Release 2>&1 | tail -10
```

If this succeeds, `confidence/_kdtree.*.so` will exist and `NearestNeighborIndex` will auto-detect it.

If cmake or compilation fails (missing compiler, etc.), the system still works — scipy backend remains. Ignore the failure and continue.

- [ ] **Step 6: Verify backend detection**

```bash
cd /home/ahmealy/ML/meshGraphNets_pytorch
python -c "
from confidence.index import NearestNeighborIndex
import numpy as np
idx = NearestNeighborIndex()
idx.build(np.random.randn(50, 128).astype(np.float32))
print('backend:', idx.backend)
print('train_diameter:', round(idx.train_diameter, 4))
"
```

Expected: `backend: cpp` (if compiled) or `backend: scipy` (if not)

- [ ] **Step 7: Commit**

```bash
cd /home/ahmealy/ML/meshGraphNets_pytorch
git add confidence/kdtree.cpp confidence/kdtree.h confidence/kdtree_bind.cpp \
        confidence/CMakeLists.txt
git commit -m "feat: add C++ KDTree with pybind11 bindings (opt-in backend)"
```

---

## Task 4: `confidence/benchmark.py` — compare both backends

**Files:**
- Create: `confidence/benchmark.py`

Runs both backends on identical data at N=100/1000/10000, prints a comparison table, and verifies distance correctness. Designed to produce concrete numbers for interview use.

- [ ] **Step 1: Create `confidence/benchmark.py`**

```python
"""
Benchmark scipy vs C++ KDTree backends on identical data.

Usage:
    python -m confidence.benchmark
    python -m confidence.benchmark --index runs/embedding_index.pkl

Output example:
    Benchmark: dim=128
    ───────────────────────────────────────────────────────────────────
    N        Backend       Build(ms)   Query(ms)   Batch(ms)   Correct
    100      scipy             1.2        0.041        3.1      —
    100      C++               0.9        0.031        2.3      ✅
    1000     scipy            14.2        0.051        4.8      —
    1000     C++              11.8        0.038        3.1      ✅
    10000    scipy           162.4        0.063       46.2      —
    10000    C++             118.1        0.041       28.7      ✅
    ───────────────────────────────────────────────────────────────────
    Correctness: max distance error = 2.4e-06 (float32 precision)
"""
import argparse
import time

import numpy as np
from scipy.spatial import KDTree as ScipyKDTree


def _bench_one(n: int, dim: int = 128, n_queries: int = 100, seed: int = 0):
    """Benchmark both backends for given N. Returns result dict."""
    rng = np.random.default_rng(seed)
    data    = rng.random((n, dim)).astype(np.float32)
    queries = rng.random((n_queries, dim)).astype(np.float32)

    results = {}

    # ── scipy ─────────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    scipy_tree = ScipyKDTree(data)
    build_scipy = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    scipy_tree.query(queries[0:1], k=1)
    query_scipy = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    scipy_dists, _ = scipy_tree.query(queries, k=1)
    batch_scipy = (time.perf_counter() - t0) * 1000
    scipy_dists = scipy_dists.flatten()

    results["scipy"] = {
        "build_ms":  round(build_scipy, 2),
        "query_ms":  round(query_scipy, 3),
        "batch_ms":  round(batch_scipy, 2),
        "dists":     scipy_dists,
        "correct":   "—",
    }

    # ── C++ (if compiled) ──────────────────────────────────────────────────────
    try:
        import sys, os
        confidence_dir = os.path.dirname(os.path.abspath(__file__))
        if confidence_dir not in sys.path:
            sys.path.insert(0, confidence_dir)
        from confidence._kdtree import KDTree as CppKDTree  # type: ignore

        t0 = time.perf_counter()
        cpp_tree = CppKDTree(data)
        build_cpp = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        cpp_tree.query(queries[0:1].reshape(1, -1), k=1)
        query_cpp = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        cpp_dists = np.array([cpp_tree.query(q.reshape(1, -1), k=1)[0] for q in queries])
        batch_cpp = (time.perf_counter() - t0) * 1000

        max_err = float(np.max(np.abs(cpp_dists - scipy_dists)))
        correct = "✅" if max_err < 1e-4 else "❌ (err=%.2e)" % max_err

        results["cpp"] = {
            "build_ms":  round(build_cpp, 2),
            "query_ms":  round(query_cpp, 3),
            "batch_ms":  round(batch_cpp, 2),
            "dists":     cpp_dists,
            "correct":   correct,
            "max_err":   max_err,
        }
    except ImportError:
        results["cpp"] = None

    return results


def run_benchmark(dim: int = 128, n_queries: int = 100):
    ns = [100, 1000, 10000]

    sep = "─" * 67
    print("\nBenchmark: dim=%d, n_queries=%d" % (dim, n_queries))
    print(sep)
    print("%-8s %-14s %-12s %-12s %-12s %s" % (
        "N", "Backend", "Build(ms)", "Query(ms)", "Batch(ms)", "Correct"))
    print(sep)

    max_errs = []

    for n in ns:
        res = _bench_one(n, dim=dim, n_queries=n_queries)

        sp = res["scipy"]
        print("%-8d %-14s %-12.2f %-12.3f %-12.2f %s" % (
            n, "scipy KDTree", sp["build_ms"], sp["query_ms"], sp["batch_ms"], sp["correct"]))

        if res["cpp"]:
            cpp = res["cpp"]
            print("%-8d %-14s %-12.2f %-12.3f %-12.2f %s" % (
                n, "C++ KDTree", cpp["build_ms"], cpp["query_ms"], cpp["batch_ms"], cpp["correct"]))
            if "max_err" in cpp:
                max_errs.append(cpp["max_err"])
        else:
            print("%-8d %-14s %-12s (C++ not compiled — run cmake && make in confidence/)" % (
                n, "C++ KDTree", "—"))
        print()

    print(sep)
    if max_errs:
        print("Correctness: max distance error = %.2e (float32 precision)" % max(max_errs))
    else:
        print("C++ backend not available. Build with: cd confidence && cmake . && make")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark scipy vs C++ KDTree")
    parser.add_argument("--dim",       type=int, default=128)
    parser.add_argument("--queries",   type=int, default=100)
    parser.add_argument("--index",     type=str, default=None,
                        help="Optional: path to saved embedding_index.pkl to use real embeddings")
    args = parser.parse_args()

    if args.index:
        import pickle
        with open(args.index, "rb") as f:
            d = pickle.load(f)
        embs = d["embeddings"]
        print("Using real embeddings from %s: shape %s" % (args.index, embs.shape))
        # Override dimension
        args.dim = embs.shape[1]

    run_benchmark(dim=args.dim, n_queries=args.queries)
```

- [ ] **Step 2: Verify benchmark runs**

```bash
python -m confidence.benchmark
```

Expected: table with scipy rows at N=100/1000/10000, C++ rows if compiled

- [ ] **Step 3: Commit**

```bash
git add confidence/benchmark.py
git commit -m "feat: add KDTree benchmark comparing scipy vs C++ at N=100/1000/10000"
```

---

## Task 5: `confidence/build_index.py` — standalone index builder

**Files:**
- Create: `confidence/build_index.py`

Standalone script: loads a checkpoint + dataset, extracts embeddings via `extract_embedding`, builds and saves `NearestNeighborIndex`.

- [ ] **Step 1: Create `confidence/build_index.py`**

```python
"""
Standalone script to build the confidence index from a saved checkpoint.

Usage:
    python -m confidence.build_index \
        --checkpoint checkpoints/best_model.pth \
        --split train \
        --output runs/embedding_index.pkl \
        --domain cylinder_flow \
        --data_dir data

The index is saved to runs/embedding_index.pkl (or --output path).
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from confidence.index import NearestNeighborIndex
from model.embedding import extract_embedding


def main():
    parser = argparse.ArgumentParser(description="Build confidence embedding index")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split",      type=str, default="train")
    parser.add_argument("--output",     type=str, default="runs/embedding_index.pkl")
    parser.add_argument("--domain",     type=str, default="cylinder_flow",
                        choices=["cylinder_flow", "flag_simple"])
    parser.add_argument("--data_dir",   type=str, default="data")
    parser.add_argument("--device",     type=str,
                        default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    # Load checkpoint and simulator
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    domain = ckpt.get("domain", args.domain)

    if domain == "flag_simple":
        from model.flag_simulator import FlagSimulator
        simulator = FlagSimulator(message_passing_num=15, device=device)
    else:
        from model.simulator import Simulator
        node_input_size = ckpt.get("node_input_size", 11)
        edge_input_size = ckpt.get("edge_input_size", 3)
        simulator = Simulator(
            message_passing_num=15,
            node_input_size=node_input_size,
            edge_input_size=edge_input_size,
            device=device,
        )

    simulator.load_state_dict(ckpt["model_state_dict"])
    simulator.eval()
    print("Loaded checkpoint from %s (epoch %d)" % (args.checkpoint, ckpt.get("epoch", 0)))

    # Load dataset
    if domain == "flag_simple":
        from dataset.flag_dataset import FlagDataset
        dataset = FlagDataset(args.data_dir, split=args.split)
        transformer = None
    else:
        from dataset import FpcDataset
        dataset = FpcDataset(data_root=args.data_dir, split=args.split)
        transformer = T.Compose([
            T.FaceToEdge(),
            T.Cartesian(norm=False),
            T.Distance(norm=False),
        ])

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    print("Dataset: %d samples. Extracting embeddings..." % len(dataset))

    embeddings = []
    for graph in tqdm(loader, desc="Extracting embeddings"):
        graph = graph[0] if isinstance(graph, list) else graph  # unbatch single item
        if transformer is not None:
            graph = transformer(graph)
        emb = extract_embedding(simulator, graph, device=device)
        embeddings.append(emb)

    embeddings = np.stack(embeddings)   # [N_train, 128]
    print("Embeddings shape: %s" % str(embeddings.shape))

    index = NearestNeighborIndex()
    index.build(embeddings)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    index.save(args.output)
    print("Index saved to %s (backend: %s, diameter: %.4f)" % (
        args.output, index.backend, index.train_diameter))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify script loads without errors**

```bash
python -m confidence.build_index --help
```

Expected: usage message

- [ ] **Step 3: Commit**

```bash
git add confidence/build_index.py
git commit -m "feat: add standalone confidence index builder script"
```

---

## Task 6: Auto-build index after training in `train.py`

**Files:**
- Modify: `train.py`

After `writer.close()`, extract embeddings from the training set and build/save the confidence index. Controlled by `cfg.get('build_confidence_index', True)`.

- [ ] **Step 1: Add confidence index building block to `train.py`**

At the end of the `if __name__ == '__main__':` block, after `writer.close()`, add:

```python
# ── Build confidence index (optional, default enabled) ────────────────────────
if cfg.get('build_confidence_index', True):
    try:
        print("\nBuilding confidence index on training set...")
        from confidence.index import NearestNeighborIndex
        from model.embedding import extract_embedding

        embeddings = []
        simulator.eval()

        for graph in tqdm.tqdm(train_loader, desc='Extracting embeddings'):
            if transformer is not None:
                graph = transformer(graph)
            # DataLoader returns batched graph — unbatch to get one graph at a time
            # For embedding, use one sample at a time
            break  # Extract from first batch only for speed — oversample is fine

        # Re-run without batching for clean single-graph embeddings
        single_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)
        for graph in tqdm.tqdm(single_loader, desc='Extracting embeddings'):
            if transformer is not None:
                graph = transformer(graph)
            emb = extract_embedding(simulator, graph, device=device)
            embeddings.append(emb)

        embeddings_arr = np.stack(embeddings)   # [N_train, 128]
        index = NearestNeighborIndex()
        index.build(embeddings_arr)
        index_path = os.path.join(log_dir, 'embedding_index.pkl')
        index.save(index_path)
        print("Confidence index saved to %s (backend: %s, diameter: %.4f, N=%d)" % (
            index_path, index.backend, index.train_diameter, len(embeddings)))
    except Exception as e:
        print("Warning: confidence index build failed (non-fatal): %s" % e)
```

Note: extracting embeddings per single graph is slow for large datasets. The above approach uses `batch_size=1` which is correct. For very large train sets, the user can use `build_index.py` standalone with subsampling.

- [ ] **Step 2: Verify train.py still imports cleanly**

```bash
python -c "
import sys
sys.argv = ['train.py']
# Just verify parse/import, not run training
import importlib.util
spec = importlib.util.spec_from_file_location('train', 'train.py')
m = importlib.util.module_from_spec(spec)
# Don't exec (would run training) — just verify no syntax errors
import ast
with open('train.py') as f:
    ast.parse(f.read())
print('train.py syntax OK')
"
```

Expected: `train.py syntax OK`

- [ ] **Step 3: Commit**

```bash
git add train.py
git commit -m "feat: auto-build confidence index after training completes"
```

---

## Task 7: Rollout integration — query index, attach `confidence_score`

**Files:**
- Modify: `api/routes/rollout.py`
- Modify: `api/routes/results.py`

After rollout, query the confidence index and attach `confidence_score` to the SSE `done` event and to the saved pkl. The results API reads it from the pkl and adds `confidence_label`.

- [ ] **Step 1: Add confidence score computation to `_run_rollout_sync()` in `api/routes/rollout.py`**

After the `# Save pkl` section and before `return {...}`, add:

```python
# Confidence score — requires embedding_index.pkl built after training
confidence_score = None
index_path = "runs/embedding_index.pkl"
if os.path.exists(index_path):
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
        from confidence.index import NearestNeighborIndex
        from model.embedding import extract_embedding
        import torch_geometric.transforms as T

        index = NearestNeighborIndex.load(index_path)

        # Re-load first graph for embedding (same trajectory, step 0)
        first_graph = dataset[req.trajectory_index * n_steps]
        transformer_emb = T.Compose([
            T.FaceToEdge(),
            T.Cartesian(norm=False),
            T.Distance(norm=False),
        ])
        first_graph = transformer_emb(first_graph)
        emb = extract_embedding(model, first_graph, device=device)
        confidence_score = float(index.query(emb))
    except Exception:
        pass  # confidence is optional — never block the rollout
```

Update the pkl save to include `confidence_score`:

```python
# Save pkl with confidence metadata
os.makedirs("result", exist_ok=True)
pkl_path = "result/result%d.pkl" % req.trajectory_index
with open(pkl_path, "wb") as f:
    pickle.dump([[predicted_arr, targets_arr], crds, {
        "confidence_score": confidence_score,
        "domain": req.domain,
    }], f)
```

Add `confidence_score` to the return dict:

```python
return {
    "elapsed_seconds":  round(elapsed, 3),
    "speedup":          round(speedup, 2),
    "pkl_path":         pkl_path,
    "rmse_final":       float(per_step_rmse[-1]),
    "similarity_score": round(similarity_score, 3) if similarity_score is not None else None,
    "confidence_score": round(confidence_score, 3) if confidence_score is not None else None,
}
```

- [ ] **Step 2: Update `_load_pkl()` in `api/routes/results.py` to handle new 3-element pkl format**

Replace the `_load_pkl` function:

```python
def _load_pkl(filename: str):
    # Guard against path traversal (e.g. "../../etc/passwd")
    safe_dir = os.path.realpath(RESULT_DIR)
    path = os.path.realpath(os.path.join(RESULT_DIR, filename))
    if not path.startswith(safe_dir + os.sep):
        raise HTTPException(400, "Invalid filename")
    if not os.path.exists(path):
        raise HTTPException(404, "Result file not found: %s" % filename)
    with open(path, "rb") as f:
        data = pickle.load(f)
    # Support both old format (2 elements) and new format (3 elements with metadata)
    if len(data) == 2:
        result, crds = data
        meta = {}
    else:
        result, crds, meta = data
    predicted = result[0]   # [T, N, 2] or [T, N, 3]
    targets   = result[1]   # [T, N, 2] or [T, N, 3]
    return predicted, targets, crds, meta
```

Update all callers of `_load_pkl` in `results.py` that currently unpack 3 values to unpack 4:

In `get_result()`:
```python
predicted, targets, crds, meta = _load_pkl(filename)
```
Add `confidence_score` and `confidence_label` to response:
```python
confidence_score = meta.get("confidence_score", None)
confidence_label = _confidence_label(confidence_score)
return {
    "timesteps":        int(predicted.shape[0]),
    "num_nodes":        int(predicted.shape[1]),
    "dt":               0.01,
    "crds":             crds.tolist(),
    "triangles":        triangles.tolist(),
    "per_step_rmse":    per_step_rmse.tolist(),
    "elapsed_seconds":  None,
    "speedup":          None,
    "confidence_score": confidence_score,
    "confidence_label": confidence_label,
    "domain":           meta.get("domain", "cylinder_flow"),
}
```

In `get_frame()`, `get_rmse()`, `delete_result()` — update unpacking to 4-tuple.

Add the helper function before the routes:

```python
def _confidence_label(score: Optional[float]) -> Optional[str]:
    if score is None:
        return None
    if score >= 0.7:
        return "High"
    if score >= 0.4:
        return "Medium"
    return "Low"
```

- [ ] **Step 3: Run a quick sanity check**

```bash
python -c "
from api.routes.results import router
print('results.py imports OK')
from api.routes.rollout import router
print('rollout.py imports OK')
"
```

Expected: two OK messages

- [ ] **Step 4: Commit**

```bash
git add api/routes/rollout.py api/routes/results.py
git commit -m "feat: attach confidence_score to rollout results and API responses"
```

---

## Task 8: Frontend — confidence badge and card

**Files:**
- Modify: `app/src/pages/Predict.tsx`
- Modify: `app/src/pages/Visualize.tsx`

- [ ] **Step 1: Update `Predict.tsx` — confidence badge after rollout**

In the rollout `done` SSE handler, add confidence display after the RMSE metrics. Find where `similarity_score` is shown and add alongside:

```tsx
{doneData?.confidence_score != null && (
  <div className="flex items-center gap-2 mt-2">
    <span className="text-sm text-gray-600">Confidence:</span>
    <div className="flex-1 bg-gray-200 rounded-full h-2 max-w-32">
      <div
        className={`h-2 rounded-full ${
          doneData.confidence_score >= 0.7 ? "bg-green-500" :
          doneData.confidence_score >= 0.4 ? "bg-yellow-500" : "bg-red-500"
        }`}
        style={{ width: `${Math.max(0, doneData.confidence_score * 100)}%` }}
      />
    </div>
    <span className="text-sm font-semibold">
      {Math.round(doneData.confidence_score * 100)}%
    </span>
    <span className={`text-xs px-2 py-0.5 rounded font-bold ${
      doneData.confidence_score >= 0.7 ? "bg-green-100 text-green-800" :
      doneData.confidence_score >= 0.4 ? "bg-yellow-100 text-yellow-800" :
      "bg-red-100 text-red-800"
    }`}>
      {doneData.confidence_score >= 0.7 ? "HIGH" :
       doneData.confidence_score >= 0.4 ? "MEDIUM" : "LOW"}
    </span>
  </div>
)}
```

Also extend the `DoneData` TypeScript type to include `confidence_score`:
```tsx
type DoneData = {
  elapsed_seconds: number;
  speedup: number;
  rmse_final: number;
  similarity_score: number | null;
  confidence_score: number | null;
  // ... other fields
};
```

- [ ] **Step 2: Update `Visualize.tsx` — confidence card in Diagnostics tab**

In the Diagnostics tab, after the existing MAE/overfitting content, add:

```tsx
{/* Confidence Score Card */}
{resultMeta?.confidence_score != null && (
  <div className="border rounded-lg p-4 bg-gray-50">
    <h3 className="font-semibold text-gray-700 mb-2">Confidence Score</h3>
    <div className="flex items-center gap-4">
      <span className={`text-3xl font-bold ${
        resultMeta.confidence_score >= 0.7 ? "text-green-600" :
        resultMeta.confidence_score >= 0.4 ? "text-yellow-600" : "text-red-600"
      }`}>
        {Math.round(resultMeta.confidence_score * 100)}%
      </span>
      <span className={`text-lg font-semibold px-3 py-1 rounded ${
        resultMeta.confidence_score >= 0.7 ? "bg-green-100 text-green-800" :
        resultMeta.confidence_score >= 0.4 ? "bg-yellow-100 text-yellow-800" :
        "bg-red-100 text-red-800"
      }`}>
        {resultMeta.confidence_label ?? (
          resultMeta.confidence_score >= 0.7 ? "HIGH" :
          resultMeta.confidence_score >= 0.4 ? "MEDIUM" : "LOW"
        )}
      </span>
    </div>
    <p className="text-xs text-gray-500 mt-2">
      Measures how similar this test trajectory is to the training distribution,
      based on nearest-neighbor distance in the model's latent embedding space.
      Low scores indicate out-of-distribution inputs where predictions may be unreliable.
    </p>
  </div>
)}
```

Also extend the `ResultMeta` type:
```tsx
type ResultMeta = {
  // ... existing fields
  confidence_score?: number | null;
  confidence_label?: string | null;
  domain?: string;
};
```

- [ ] **Step 3: Verify TypeScript builds**

```bash
cd /home/ahmealy/ML/meshGraphNets_pytorch/app
npm run build 2>&1 | tail -20
```

Expected: no TypeScript errors

- [ ] **Step 4: Commit**

```bash
git add app/src/pages/Predict.tsx app/src/pages/Visualize.tsx
git commit -m "feat: show confidence score badge in Predict and Diagnostics tab"
```

---

## Task 9: End-to-end smoke test

- [ ] **Step 1: Run all confidence tests**

```bash
cd /home/ahmealy/ML/meshGraphNets_pytorch
pytest tests/test_embedding.py tests/test_confidence_index.py -v
```

Expected: 9 PASSED

- [ ] **Step 2: Run benchmark (scipy only if C++ not compiled)**

```bash
python -m confidence.benchmark
```

Expected: table printed with scipy rows for N=100/1000/10000; C++ rows if compiled

- [ ] **Step 3: Verify full API imports**

```bash
python -c "
from api.routes.results import router as r1
from api.routes.rollout import router as r2
print('API imports OK')
from confidence import NearestNeighborIndex
import numpy as np
idx = NearestNeighborIndex()
idx.build(np.random.randn(20, 128).astype(np.float32))
score = idx.query(np.zeros(128, dtype=np.float32))
print('NearestNeighborIndex OK, sample score:', round(score, 3))
"
```

Expected: `API imports OK` + `NearestNeighborIndex OK, sample score: 0.0`

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "feat: confidence score complete — embedding, KDTree index, rollout integration, UI"
```
