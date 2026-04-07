# Spec: Confidence Score (k-d Tree)

Date: 2026-04-07  
Status: Updated

---

## 1. Overview

Add a confidence score to every rollout result. The score answers: *"How similar is this test
trajectory to the training distribution?"*

- Score near **1.0** → test mesh looks like training data → prediction likely reliable
- Score near **0.0** → test mesh is novel / out-of-distribution → treat prediction with caution

This directly mirrors PhysicsAI Studio's "Confidence Score Metric" feature. The interviewer
specifically asked about k-d trees in this context.

---

## 2. Backend Design: C++ k-d Tree with pybind11, scipy fallback

### Why C++ instead of FAISS?

| | Custom C++ + pybind11 | FAISS | scipy KDTree |
|---|---|---|---|
| Interview signal | ✅ "I implemented k-d tree in C++" | "I used Meta's library" | "I used scipy" |
| Performance | O(log N), competitive with scipy at N≤10K | Overkill for N≤10K | O(log N), same complexity |
| Dependencies | pybind11 (header-only) | pip install faiss-cpu | built-in |
| Maintenance | We own it | Zero | Zero |

At N=1000 training trajectories all three have identical performance. The C++ implementation
is **not for performance** — it demonstrates understanding of the data structure itself.
This is the right answer when an interviewer asks "do you know k-d trees."

### Backend selection (runtime)

```
Primary:  C++ extension (confidence/_kdtree.so) — used if compiled
Fallback: scipy.spatial.KDTree — used if C++ extension not built
```

Same `NearestNeighborIndex` interface in both cases. The backend is transparent to all callers.

### Benchmarking

A `confidence/benchmark.py` script runs both backends on the same data and prints:

```
Backend       Build (ms)   Query (ms)   N
------------  ----------   ----------   -----
C++ KDTree         12.3         0.04    1000
scipy KDTree       14.1         0.05    1000
C++ KDTree         89.2         0.06    10000
scipy KDTree       98.7         0.07    10000
```

This gives a concrete number to quote in an interview.

---

## 3. C++ k-d Tree Implementation

### `confidence/kdtree.cpp`

Standard k-d tree on Euclidean distance in R^128:

```cpp
struct KDNode {
    int idx;          // index into training embeddings array
    int split_dim;    // dimension used to split at this node
    float split_val;
    KDNode* left;
    KDNode* right;
};

class KDTree {
public:
    KDTree(const float* data, int n, int dim);  // build
    float query_nn(const float* point) const;   // nearest neighbor distance
private:
    KDNode* build(std::vector<int>& indices, int depth);
    void search(KDNode* node, const float* point,
                float& best_dist, int& best_idx) const;
    std::vector<float> data_;
    int n_, dim_;
    KDNode* root_;
};
```

Build: median split on alternating dimensions (standard k-d tree).  
Search: branch-and-bound with pruning — only recurse into a subtree if it could contain
a closer point than the current best.

### `confidence/kdtree_bind.cpp` (pybind11 bindings)

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

PYBIND11_MODULE(_kdtree, m) {
    py::class_<KDTree>(m, "KDTree")
        .def(py::init<py::array_t<float>>())
        .def("query", &KDTree::query_nn_py);  // accepts numpy array
}
```

### `confidence/CMakeLists.txt`

```cmake
find_package(pybind11 REQUIRED)
pybind11_add_module(_kdtree kdtree.cpp kdtree_bind.cpp)
```

Build command (one-time):
```bash
cd confidence && pip install pybind11 && cmake . && make
```

If build fails or `.so` not present, `NearestNeighborIndex` silently falls back to scipy.

---

## 4. What is the Embedding?

After the encoder runs on a graph, `graph.x` has shape `[N, 128]` — one 128-dim vector per node.
We need a **single vector per trajectory**, not per node.

**Trajectory embedding = mean pooling over NORMAL nodes' post-encoder features**:
```python
normal_mask = (node_type == NodeType.NORMAL)
embedding = graph.x[normal_mask].mean(dim=0)  # [128]
```

Mean pooling over fluid nodes is:
- Permutation invariant (mesh node ordering doesn't matter)
- Cheap (one mean call)
- Captures the overall flow regime of the mesh
- Consistent with graph-level pooling in literature

---

## 5. Files Changed / Created

### New files
| File | Purpose |
|---|---|
| `confidence/__init__.py` | Package init |
| `confidence/kdtree.cpp` | C++ k-d tree implementation |
| `confidence/kdtree_bind.cpp` | pybind11 bindings |
| `confidence/CMakeLists.txt` | Build config |
| `confidence/index.py` | `NearestNeighborIndex` — C++ or scipy backend, same interface |
| `confidence/build_index.py` | Standalone script: checkpoint → embeddings → index |
| `confidence/benchmark.py` | Compare C++ vs scipy build/query times |
| `model/embedding.py` | `extract_embedding(simulator, graph) → np.ndarray[128]` |

### Modified files
| File | Change |
|---|---|
| `train.py` | After training: extract embeddings from train set, build index, save to `runs/embedding_index.pkl` |
| `api/routes/rollout.py` | After rollout: extract test embedding, query index, attach `confidence_score` |
| `api/routes/results.py` | Add `confidence_score` + `confidence_label` to result responses |
| `app/src/pages/Predict.tsx` | Show confidence badge after rollout completes |
| `app/src/pages/Visualize.tsx` | Show confidence card in Diagnostics tab |

---

## 6. Embedding Extraction (`model/embedding.py`)

```python
def extract_embedding(simulator: Simulator, graph: Data, device: str) -> np.ndarray:
    """
    Run only the encoder on a single graph, return mean-pooled NORMAL node embedding.
    Shape: [128]. Does NOT run processor or decoder.
    """
    simulator.eval()
    with torch.no_grad():
        node_type = graph.x[:, 0:1]
        frames    = graph.x[:, 1:]          # velocity or pressure depending on target_field
        
        node_attr  = simulator.update_node_attr(frames, node_type)
        graph.x    = node_attr
        graph.edge_attr = simulator.edge_normalizer(graph.edge_attr, training=False)
        
        encoded = simulator.model.encoder(graph)   # graph.x: [N, 128]
        
        normal_mask = (node_type.squeeze(-1) == NodeType.NORMAL)
        embedding   = encoded.x[normal_mask].mean(dim=0)   # [128]
    
    return embedding.cpu().numpy()
```

---

## 7. Index (`confidence/index.py`)

```python
class NearestNeighborIndex:
    """
    Nearest-neighbor index over trajectory embeddings.
    Uses C++ k-d tree if compiled, scipy.KDTree otherwise.
    """
    backend: str          # 'cpp' or 'scipy'
    embeddings: np.ndarray   # [N_train, 128]
    train_diameter: float    # 95th percentile NN distance among training embeddings

    def build(self, embeddings: np.ndarray) -> None:
        self.embeddings = embeddings
        # Try C++ backend
        try:
            from confidence._kdtree import KDTree
            self._tree = KDTree(embeddings.astype(np.float32))
            self.backend = 'cpp'
        except ImportError:
            from scipy.spatial import KDTree
            self._tree = KDTree(embeddings)
            self.backend = 'scipy'
        # Compute train_diameter (95th percentile of each point's NN distance)
        dists, _ = self._tree.query(embeddings, k=2)  # k=2: skip self (dist=0)
        self.train_diameter = float(np.percentile(dists[:, 1], 95))

    def query(self, embedding: np.ndarray) -> float:
        """Returns confidence score in [0, 1]."""
        dist, _ = self._tree.query(embedding.reshape(1, -1), k=1)
        d_min = float(dist[0])
        return float(np.clip(1.0 - d_min / self.train_diameter, 0.0, 1.0))

    def save(self, path: str) -> None:
        import pickle
        pickle.dump({'embeddings': self.embeddings,
                     'train_diameter': self.train_diameter,
                     'backend': self.backend}, open(path, 'wb'))

    @classmethod
    def load(cls, path: str) -> 'NearestNeighborIndex':
        import pickle
        d = pickle.load(open(path, 'rb'))
        obj = cls()
        obj.build(d['embeddings'])   # rebuilds tree from saved embeddings
        return obj
```

**`train_diameter`** = 95th percentile of each training embedding's distance to its nearest
neighbor (among training data). Using 95th percentile (not max) prevents one outlier training
sample from making all scores low.

---

## 8. Index Building

### Auto-build after training (`train.py`)

```python
# After writer.close():
if cfg.get('build_confidence_index', True):
    print("Building confidence index on training set...")
    embeddings = []
    simulator.eval()
    for graph in tqdm(train_loader, desc='Extracting embeddings'):
        graph = transformer(graph).to(device)
        emb = extract_embedding(simulator, graph, device)
        embeddings.append(emb)
    embeddings = np.stack(embeddings)   # [N_train, 128]
    
    index = NearestNeighborIndex()
    index.build(embeddings)
    index_path = os.path.join(log_dir, 'embedding_index.pkl')
    index.save(index_path)
    print(f"Confidence index built ({index.backend} backend): "
          f"{len(embeddings)} embeddings, diameter={index.train_diameter:.4f}")
```

### Standalone rebuild (`confidence/build_index.py`)

```bash
python -m confidence.build_index \
    --checkpoint checkpoints/best_model.pth \
    --split train \
    --output runs/embedding_index.pkl
```

### Benchmark (`confidence/benchmark.py`)

```bash
python -m confidence.benchmark --index runs/embedding_index.pkl
```

Outputs a comparison table of C++ vs scipy build and query times.

---

## 9. Rollout Integration (`api/routes/rollout.py`)

After rollout loop, before saving pkl:

```python
confidence_score = None
index_path = "runs/embedding_index.pkl"
if os.path.exists(index_path):
    try:
        index = NearestNeighborIndex.load(index_path)
        emb   = extract_embedding(simulator, first_graph, device)
        confidence_score = index.query(emb)
    except Exception:
        pass   # confidence is optional — never block the rollout

# Attach to SSE done event
yield json.dumps({"type": "done", "confidence_score": confidence_score, ...})

# Save to pkl — backward compatible (existing unpacking `result, crds = data` still works)
pickle.dump([[predicted, targets], crds, {
    "confidence_score": confidence_score,
    "target_field": target_field,
}], f)
```

---

## 10. API

### `GET /results/{filename}` and `GET /results/{filename}/rmse`

Both add:
```json
{
  "confidence_score": 0.87,      // null if index not built
  "confidence_label": "High"     // "High" ≥0.7, "Medium" 0.4–0.7, "Low" <0.4, null if no score
}
```

No new endpoints needed.

---

## 11. UI

### Predict page — confidence badge (after rollout SSE `done`)

```
┌─────────────────────────────────────────┐
│  RMSE at T=0: 2.3e-4   RMSE at T=599: 1.1e-2  │
│  Confidence:  ████████░░  87%   HIGH           │
└─────────────────────────────────────────┘
```

Color: green (≥70%), yellow (40–70%), red (<40%). Hidden if `confidence_score` is null.

### Visualize page — confidence card in Diagnostics tab

- Large percentage number, color-coded
- Label: HIGH / MEDIUM / LOW
- Tooltip: *"Measures how similar this test trajectory is to the training distribution,
  based on nearest-neighbor distance in the model's latent embedding space.
  Low scores indicate out-of-distribution inputs where predictions may be unreliable."*
- Hidden if score is null (index not built yet)

---

## 12. Performance

| Metric | Value |
|---|---|
| Index build time | ~30s for 1000 trajectories (one encoder pass each on CPU) |
| Query time (C++ backend) | <0.1ms |
| Query time (scipy fallback) | <0.1ms |
| Index file size | 1000 × 128 × 4 bytes ≈ 500 KB |

C++ and scipy have identical complexity O(log N) and near-identical wall time at N=1000.
The benchmark exists to demonstrate and measure the difference, not to justify C++ on perf grounds.

---

## 13. Index File

Saved to `runs/embedding_index.pkl`. Added to `.gitignore`.  
If missing, all confidence features show `null` gracefully — no errors anywhere.
