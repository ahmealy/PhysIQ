# Spec: Confidence Score (k-d Tree / FAISS)

Date: 2026-04-07  
Status: Draft

---

## 1. Overview

Add a confidence score to every rollout result. The score answers: *"How similar is this test trajectory to the training distribution?"*

- Score near **1.0** → test mesh looks like training data → prediction likely reliable
- Score near **0.0** → test mesh is novel / out-of-distribution → treat prediction with caution

This directly mirrors PhysicsAI Studio's "Confidence Score Metric" feature. The interviewer specifically asked about k-d trees in this context.

---

## 2. Approach

### Chosen: scipy KDTree (N ≤ training trajectories) with FAISS fallback

**Option A — scipy.spatial.KDTree**: Pure Python/NumPy, no extra deps, O(log N) query.
Good for cylinder_flow (~1000 training trajectories). Simple, interpretable.

**Option B — FAISS HNSW**: C++ index, sub-linear at N > 100K, GPU-acceleratable.
Overkill for current dataset sizes but future-proof.

**Decision**: Implement scipy KDTree by default. Add FAISS as an optional backend when `faiss` is importable. Same interface either way — the backend is an implementation detail behind a `NearestNeighborIndex` abstraction.

**No C++ offload needed**: scipy KDTree is already backed by Cython/C. FAISS is C++ with Python bindings. Neither requires us to write C++.

---

## 3. What is the Embedding?

After the encoder runs on a graph, `graph.x` has shape `[N, 128]` — one 128-dim vector per node.
We need a **single vector per trajectory**, not per node.

**Trajectory embedding = mean pooling over all NORMAL nodes' post-encoder features**:
```python
normal_mask = (node_type == NodeType.NORMAL)
embedding = graph.x[normal_mask].mean(dim=0)  # [128]
```

Mean pooling over fluid nodes is:
- Permutation invariant (mesh node ordering doesn't matter)
- Cheap (one mean call)
- Captures the overall flow regime of the mesh
- Consistent with how graph-level representations are built in literature

---

## 4. Files Changed / Created

### New files
| File | Purpose |
|---|---|
| `model/embedding.py` | `extract_embedding(simulator, graph) → np.ndarray[128]` — hooks into encoder, returns mean-pooled node embedding |
| `confidence/index.py` | `NearestNeighborIndex` — build/save/load/query abstraction over scipy KDTree or FAISS |
| `confidence/build_index.py` | Standalone script: load checkpoint + training data → extract embeddings → save index |

### Modified files
| File | Change |
|---|---|
| `train.py` | After training completes: auto-run embedding extraction on training set, save index to `runs/embedding_index.pkl` |
| `api/routes/rollout.py` | After rollout completes: extract embedding of test trajectory, query index, attach `confidence_score` to done event and save to result pkl |
| `api/routes/results.py` | Add `confidence_score` field to `GET /results/{filename}` and `GET /results/{filename}/rmse` responses |
| `app/src/pages/Predict.tsx` | Show confidence score badge after rollout completes |
| `app/src/pages/Visualize.tsx` | Show confidence score card in the metrics panel |

---

## 5. Embedding Extraction

### `model/embedding.py`

```python
def extract_embedding(simulator: Simulator, graph: Data, device: str) -> np.ndarray:
    """
    Run the encoder on a single graph and return the mean-pooled
    node embedding over NORMAL nodes. Shape: [128].
    
    Does NOT run the full forward pass — stops after encoder.
    """
    simulator.eval()
    with torch.no_grad():
        # Build normalized node + edge features (same as inference forward pass)
        node_attr = simulator.update_node_attr(frames, node_type)
        graph.x = node_attr
        graph.edge_attr = simulator.edge_normalizer(graph.edge_attr, training=False)
        
        # Run encoder only (not processor or decoder)
        encoded = simulator.model.encoder(graph)  # graph.x: [N, 128]
        
        # Mean pool over NORMAL nodes
        normal_mask = (node_type.squeeze(-1) == NodeType.NORMAL)
        embedding = encoded.x[normal_mask].mean(dim=0)  # [128]
    
    return embedding.cpu().numpy()
```

---

## 6. Index

### `confidence/index.py`

```python
class NearestNeighborIndex:
    backend: str  # 'scipy' or 'faiss'
    embeddings: np.ndarray  # [N_train, 128]
    train_diameter: float   # 95th percentile of pairwise distances (computed at build time)
    
    def build(self, embeddings: np.ndarray) -> None: ...
    def query(self, embedding: np.ndarray) -> float:
        """Returns confidence score in [0, 1]."""
        d_min = distance to nearest training embedding
        return float(np.clip(1.0 - d_min / self.train_diameter, 0.0, 1.0))
    
    def save(self, path: str) -> None: ...  # pickle
    
    @classmethod
    def load(cls, path: str) -> 'NearestNeighborIndex': ...
```

**`train_diameter`** = 95th percentile of distances from each training embedding to its nearest neighbor. Using 95th percentile (not max) makes the score robust to outlier training samples.

**Backend selection**:
```python
try:
    import faiss
    backend = 'faiss'
except ImportError:
    backend = 'scipy'
```

---

## 7. Index Building

### During training (`train.py`)
After `writer.close()` at the end of training:
```python
if cfg.get('build_confidence_index', True):
    print("Building confidence index on training set...")
    embeddings = extract_embeddings_for_split(simulator, train_dataset, device)
    index = NearestNeighborIndex()
    index.build(embeddings)
    index.save(os.path.join(log_dir, 'embedding_index.pkl'))
    print(f"Index built: {len(embeddings)} training embeddings")
```

### Standalone rebuild (`confidence/build_index.py`)
For when you want to rebuild the index without retraining:
```bash
python -m confidence.build_index --checkpoint checkpoints/best_model.pth --split train
```
Saves to `runs/embedding_index.pkl` by default.

---

## 8. Rollout Integration

In `api/routes/rollout.py`, after the rollout loop completes:

```python
# Load index if available
index_path = "runs/embedding_index.pkl"
confidence_score = None
if os.path.exists(index_path):
    index = NearestNeighborIndex.load(index_path)
    emb = extract_embedding(simulator, first_graph, device)
    confidence_score = index.query(emb)

# Attach to done SSE event
yield json.dumps({
    "type": "done",
    "confidence_score": confidence_score,
    ...
})

# Save to pkl alongside velocities
pickle.dump([[predicted, targets], crds, {"confidence_score": confidence_score}], f)
```

The pkl format gains an optional third element (metadata dict) — backward compatible (existing code that does `result, crds = data` still works; new code does `result, crds, meta = data` with `meta = {}` fallback).

---

## 9. API

### `GET /results/{filename}`
Adds:
```json
{
  "confidence_score": 0.87,   // null if index not built yet
  "confidence_label": "High"  // "High" ≥0.7, "Medium" 0.4-0.7, "Low" <0.4
}
```

### `GET /results/{filename}/rmse`
Same two fields added.

No new endpoints needed.

---

## 10. UI

### Predict page (`Predict.tsx`)
After rollout SSE `done` event, show a score badge next to the RMSE summary:
```
Confidence: ██████████ 87%  HIGH
```
Color: green ≥70%, yellow 40-70%, red <40%. Hidden if `confidence_score` is null (index not built).

### Visualize page (`Visualize.tsx`)
In the Diagnostics tab metrics panel, add a `ConfidenceCard`:
- Large score number (e.g. "87%")
- Color-coded label
- Tooltip: "How similar this trajectory is to the training distribution. Low scores mean the model may be unreliable for this input."
- Hidden if score is null

---

## 11. Index File Location

`runs/embedding_index.pkl` — alongside training logs and PID files. Added to `.gitignore`.
If the file doesn't exist, all confidence features gracefully show `null` — no errors.

---

## 12. Performance

- **Index build time**: ~30s for 1000 training trajectories (one forward pass per trajectory on CPU)
- **Query time**: <1ms (scipy KDTree log N lookup)
- **Memory**: 1000 × 128 × 4 bytes ≈ 500 KB — negligible
- **FAISS**: Only needed if N > 100K trajectories — not applicable to current datasets
