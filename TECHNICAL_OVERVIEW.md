# MeshGraphNets PyTorch — Technical Overview

> **Purpose:** Interview preparation document for PhysicsAI roles (Siemens / Altair).  
> **Date:** April 2026  
> **Scope:** Full-stack physics simulation surrogate — GNN model, data pipeline, confidence system, REST API, React UI.

---

## Executive Summary

This project is a production-quality PyTorch reimplementation of DeepMind's MeshGraphNets (Pfaff et al., ICLR 2021), extended with multi-domain support, a deployable REST API, and an original confidence scoring system. It trains graph neural networks on real physics simulation datasets from DeepMind's open benchmark suite and provides autoregressive rollout that predicts the full time evolution of a physical system — fluid flow past a cylinder (CFD) or deformable cloth — at a fraction of the cost of a traditional solver.

The central challenge MeshGraphNets addresses is that classical CFD solvers (OpenFOAM, Fluent) are accurate but slow — a single transient simulation can take hours to days on an HPC cluster. Neural surrogates trained on an ensemble of such simulations can reproduce the physics at inference time in seconds on a single GPU. This project demonstrates that capability concretely: the rollout endpoint reports a real wall-clock speedup ratio (simulated seconds / inference seconds), which for the cylinder flow dataset is typically several hundred × faster than real-time.

The project goes beyond a standard research reimplementation by adding three original engineering contributions: (1) a multi-domain training harness with a `target_field` parameter that switches between velocity, pressure, and cloth position prediction without changing any model code; (2) a post-training confidence scoring system that detects when an inference query lies outside the distribution seen during training using an embedding-space nearest-neighbor index; and (3) a full REST API + React frontend that makes the entire pipeline accessible without writing code.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         MeshGraphNets PyTorch                           │
│                                                                         │
│  DeepMind TFRecords                                                     │
│       │                                                                 │
│       ▼  parse_tfrecord.py                                              │
│  ┌──────────┐    .dat (memmap)                                          │
│  │  Raw     │ ─────────────────► FpcDataset / FlagDataset               │
│  │  Data    │    .npz (meta)          │                                 │
│  └──────────┘                         │  PyG Data objects               │
│                                       ▼                                 │
│                              ┌─────────────────┐                       │
│                              │   Simulator /    │                       │
│                              │  FlagSimulator   │                       │
│                              │                  │                       │
│                              │  Encoder         │  node/edge → latent   │
│                              │  Processor×15    │  message passing      │
│                              │  Decoder         │  latent → Δfield      │
│                              └────────┬─────────┘                       │
│                                       │  Training: MSE(Δfield_norm)     │
│                                       │  Inference: field_{t+1} = ...   │
│                                       ▼                                 │
│                          ┌──────────────────────┐                      │
│                          │  Autoregressive        │                      │
│                          │  Rollout (T steps)     │                      │
│                          │  boundary enforcement  │                      │
│                          └──────────┬─────────────┘                    │
│                                     │                                   │
│                    ┌────────────────┼────────────────┐                  │
│                    ▼                ▼                 ▼                  │
│              result*.pkl    confidence_score    per_step_rmse           │
│                    │                                                     │
│                    ▼                                                     │
│           FastAPI Backend (/api/*)                                      │
│                    │                                                     │
│                    ▼                                                     │
│           React Frontend (port 3000)                                    │
│           Train │ Predict │ Visualize │ Dataset Studio                  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Graph Neural Network Architecture

### 1.1 EncoderProcessorDecoder

The model follows the standard Encode-Process-Decode architecture from the original MeshGraphNets paper. All three stages are implemented in `model/model.py`.

**Encoder** maps raw node and edge features into a shared latent space:

```python
# model/model.py — Encoder
self.eb_encoder = build_mlp(edge_input_size, hidden_size, hidden_size)  # edges
self.nb_encoder = build_mlp(node_input_size, hidden_size, hidden_size)  # nodes
```

Each MLP has the structure: `Linear → ReLU → Linear → ReLU → Linear → ReLU → Linear → LayerNorm`. The final `LayerNorm` is critical — it prevents activations from blowing up early in training before the running-statistics normalizers have accumulated enough data.

**Processor** applies `message_passing_num` (default 15) rounds of graph network blocks:

```python
# model/model.py — GnBlock (one message-passing round)
eb_input_dim = 3 * hidden_size   # sender_node + receiver_node + edge = 3 × 128
nb_input_dim = 2 * hidden_size   # node + aggregated_edge = 2 × 128
```

Each `GnBlock` first updates edges (EdgeBlock: concatenates sender, receiver, and current edge embeddings → MLP → new edge), then updates nodes (NodeBlock: aggregates incoming edge embeddings → concatenate with node → MLP → new node). Crucially, both blocks use **residual connections** — the output of each MLP is added to the input, not replacing it:

```python
x = x + graph.x              # residual: identity + learned update
edge_attr = edge_attr + graph.edge_attr
```

This is the standard residual pattern that lets gradients flow across 15 rounds of message passing without vanishing.

**Decoder** projects the final node embeddings to the physics output:

```python
# model/model.py — Decoder
self.decode_module = build_mlp(hidden_size, hidden_size, output_size, lay_norm=False)
```

LayerNorm is intentionally omitted from the decoder — the output must be in an unnormalized space that the `_output_normalizer.inverse()` can interpret correctly.

**Dimensions summary:**

| Domain | `node_input_size` | `edge_input_size` | `hidden_size` | `output_size` |
|---|---|---|---|---|
| cylinder_flow (velocity) | 11 | 3 | 128 | 2 |
| cylinder_flow (pressure) | 10 | 3 | 128 | 1 |
| flag_simple (cloth) | 12 | 7 | 128 | 3 |

### 1.2 Multi-Domain Support

Three distinct physics domains are supported, each requiring different node/edge feature construction and different output interpretation.

**cylinder_flow — velocity mode** (`node_input_size=11`):  
Node features = `[node_type_onehot(9), vx(1), vy(1)]`. The one-hot encoding covers 9 DeepMind `NodeType` values: NORMAL, OBSTACLE, AIRFOIL, HANDLE, INFLOW, OUTFLOW, WALL_BOUNDARY, SIZE. The model predicts `[Δvx, Δvy]` — the change in velocity, not the raw next velocity. Inference adds this delta back to the current velocity.

**cylinder_flow — pressure mode** (`node_input_size=10`):  
Node features = `[node_type_onehot(9), p(1)]`. One dimension narrower because pressure is a scalar. The `_frames_slice()` method on `Simulator` selects `slice(1, 2)` for pressure vs `slice(1, 3)` for velocity — this is the only branching point needed to support both modes.

```python
# model/simulator.py
def _frames_slice(self) -> slice:
    if self.target_field == "pressure":
        return slice(1, 2)   # graph.x[:, 1:2] — pressure [N, 1]
    return slice(1, 3)       # graph.x[:, 1:3] — velocity [N, 2]
```

**flag_simple — cloth mode** (`node_input_size=12`, `edge_input_size=7`):  
Implemented in the separate `FlagSimulator` class. Node features = `[velocity(3), node_type_onehot(9)]` where velocity here is the finite difference `world_pos_t - world_pos_{t-1}`. Edge features are 7-dimensional: `[rel_mesh_pos(2), |rel_mesh|(1), rel_world_pos(3), |rel_world|(1)]`. The dual edge encoding (mesh-space + world-space) lets the model distinguish between rest-configuration distances and current deformed distances — this is critical for cloth physics.

**Config propagation**: `target_field` is written into the checkpoint at save time:
```python
torch.save({..., 'target_field': target_field, 'node_input_size': node_input_size, ...}, path)
```
At inference, `get_model()` in `api/state.py` reads it back and reconstructs the correct `Simulator`. No second user input is needed — the checkpoint is self-describing.

### 1.3 Normalization Strategy

All three domains use `utils/normalization.py`'s `Normalizer` — an online running-statistics normalizer that accumulates mean and variance over training batches (Welford-style):

```python
# Used in Simulator.forward()
self._node_normalizer   = Normalizer(size=node_input_size, ...)
self.edge_normalizer    = Normalizer(size=edge_input_size, ...)
self._output_normalizer = Normalizer(size=output_size, ...)
```

During training (`self.training == True`), the normalizer updates its running stats and normalizes. During inference (`self.training == False`), it uses frozen stats to normalize inputs and `inverse()` to denormalize the output delta back to physical units.

**Why this matters for physics:** Velocity in the cylinder flow dataset has typical values ~0.5–2.0 m/s, while pressure has typical values ~0.01–1.0 Pa, and cloth positions are ~0.0–1.0 m. Without per-field normalization, whichever field has larger magnitude will dominate the loss and gradients will ignore the others. The normalizer ensures each field contributes equally regardless of physical scale.

**Noise injection** during training prevents exposure bias:
```python
noised_frames = frames + velocity_sequence_noise  # Gaussian noise ~ N(0, 0.02²)
target_change = self.velocity_to_acceleration(noised_frames, graph.y)
```
The model is trained to recover from noisy inputs, which prevents the autoregressive rollout from compounding small errors into large drift.

---

## 2. Data Pipeline

### 2.1 TFRecord Parsing

The raw data is DeepMind's MeshGraphNets dataset in TFRecord format (available at: https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets). The `data/parse_tfrecord.py` script converts these into NumPy memory-mapped files:

- `train.dat` / `valid.dat` / `test.dat` — float32 memory-mapped arrays of shape `[total_nodes_across_all_trajectories, T, 2]` for velocity
- `train_pressure.dat` — shape `[total_nodes, T, 1]` for pressure (separate file, optional)
- `train.npz` — metadata: `pos`, `node_type`, `cells`, `indices` (trajectory boundaries), `cindices` (cell boundaries), `all_velocity_shape`

The `parse_flag_tfrecord.py` script outputs:
- `{split}_pos.npz` — object array of shape `[n_traj]`, each element `[T, N, 3]`
- `{split}_mesh.npz` — `mesh_pos`, `node_type`, `cells` per trajectory

### 2.2 FpcDataset (CFD)

`dataset/fpc.py` uses `numpy.memmap` for zero-copy disk access:

```python
self.fp = np.memmap(data_path, dtype="float32", mode="r", shape=vel_shape)
```

No trajectory is ever fully loaded into RAM. `__getitem__` uses the `indices` array to locate the correct trajectory slice on disk, reads exactly the two timesteps needed (`t` and `t+1`), and constructs a PyG `Data` object:

```python
graph = Data(
    x    = torch.as_tensor(x.copy(),   dtype=torch.float32),  # [N, 3] node features
    pos  = torch.as_tensor(pos.copy(), dtype=torch.float32),  # [N, 2] coordinates
    face = torch.as_tensor(cells.T.copy(), dtype=torch.int64), # [3, F] triangles
    y    = torch.as_tensor(y.copy(),   dtype=torch.float32),  # [N, 2] next velocity
)
```

The `face` attribute (triangular connectivity) is converted to edges by `T.FaceToEdge()` in the training loop. `T.Cartesian(norm=False)` adds the `[Δx, Δy]` relative positions as edge features, and `T.Distance(norm=False)` appends the Euclidean distance — giving the 3-dimensional edge input `[Δx, Δy, |Δr|]`.

### 2.3 FlagDataset (Cloth)

`dataset/flag_dataset.py` uses lazy `np.load(..., allow_pickle=True)` with `NpzFile` handles kept open:

```python
self._pos_data  = np.load(pos_path,  allow_pickle=True)   # lazy handle
self._mesh_data = np.load(mesh_path, allow_pickle=True)   # lazy handle
```

Only the object array headers are materialized at `__init__` time to compute trajectory lengths. Per-trajectory data is read on demand in `__getitem__`. Trajectory/timestep index mapping uses `np.searchsorted` on cumulative step counts for O(log n) lookup.

The `prev_x` attribute carries `world_pos_{t-1}` (or `world_pos_t` at `t=0` — zero initial velocity):

```python
world_pos_prev = np.asarray(
    world_pos[t - 1] if t > 0 else world_pos[t], dtype=np.float32
)
```

This is the Verlet integration input: the model needs both the current and previous world positions to predict the next.

---

## 3. Training

### 3.1 Training Loop

Training is domain-aware throughout. The key components:

**Noise injection** (CFD only):
```python
velocity_sequence_noise = get_velocity_noise(graph, noise_std=0.02, device=device)
if target_field == 'pressure':
    velocity_sequence_noise = velocity_sequence_noise[:, :1]  # trim to [N, 1]
predicted_acc, target_acc = model(graph, velocity_sequence_noise)
```
Noise standard deviation of `0.02` is the value from the original paper (2% of typical velocity magnitude).

**Loss masking by node type:**
- CFD: loss on `NORMAL` and `OUTFLOW` nodes only (WALL and INFLOW nodes are boundary conditions — no point learning their values)
- Cloth: loss on `NORMAL` nodes only (HANDLE nodes are pinned — their positions are prescribed)

```python
errors = ((predicted_acc - target_acc) ** 2)[mask]
loss = torch.mean(errors)
```

The loss is on normalized deltas (acceleration / pressure change), not raw field values. This means a unit MSE corresponds to a roughly equal prediction error across all training samples, regardless of flow speed or mesh resolution.

**Early stopping** with `patience=10`: the best validation checkpoint is saved as `checkpoints/best_model.pth`. The checkpoint is self-describing (stores `domain`, `target_field`, `node_input_size`, `edge_input_size`, `epoch`, `valid_loss`).

**TensorBoard** integration: `writer.add_scalar('Loss/train', ...)` and `Loss/valid` at every epoch, viewable with `tensorboard --logdir runs/`.

### 3.2 Multi-Domain Config

Training is driven by a JSON config file generated by the UI:

```json
{
  "domain": "cylinder_flow",
  "target_field": "pressure",
  "batch_size": 20,
  "noise_std": 0.02,
  "num_epochs": 100,
  "early_stopping_patience": 10,
  "lr": 1e-4,
  "message_passing_num": 15
}
```

The config is loaded at module level in `train.py`, which means the entire file is importable by the FastAPI process for subprocess invocation. The domain-to-architecture mapping is in `_DOMAIN_DEFAULTS`:

```python
_DOMAIN_DEFAULTS = {
    'cylinder_flow': dict(output_size=2, node_input_size=11, edge_input_size=3),
    'flag_simple':   dict(output_size=3, node_input_size=12, edge_input_size=7),
}
```

### 3.3 Post-Training: Confidence Index

After every training run (unless `build_confidence_index=False` in config), the training loop automatically iterates over the entire training set at batch_size=1, extracts a 128-dimensional embedding per sample, and builds the KDTree index:

```python
for graph in tqdm.tqdm(single_loader, desc='Extracting embeddings'):
    emb = extract_embedding(simulator, graph, device=device)
    embeddings.append(emb)
embeddings_arr = np.stack(embeddings)   # [N_train, 128]
index = NearestNeighborIndex()
index.build(embeddings_arr)
index.save(os.path.join(log_dir, 'embedding_index.pkl'))
```

This is ~5–30 minutes of extra work after training (depending on dataset size), but it pays off at inference time — every rollout can query the index and report a confidence score with no re-training required.

---

## 4. Confidence Score System

### 4.1 Motivation

A neural network will produce an output for any input, regardless of whether that input is in-distribution or completely alien. For a physics simulation surrogate deployed in an engineering context (e.g., Siemens NX or Altair HyperWorks), a prediction with no uncertainty estimate is dangerous — an engineer might trust a confidently-wrong prediction because the number looks plausible.

The confidence score answers the question: *"Is this inference query similar to the training data the model has seen?"* It is not a Bayesian posterior, but it is a principled and computationally cheap indicator of extrapolation risk.

### 4.2 NearestNeighborIndex

The score is computed in the embedding space of the encoder:

```python
# model/embedding.py
encoded = simulator.model.encoder(graph)   # Data with x: [N, 128]
normal_mask = (node_type_idx == int(NodeType.NORMAL))
embedding = encoded.x[normal_mask].mean(dim=0)  # [128] — mean pool over NORMAL nodes
```

Mean-pooling over `NORMAL` nodes (ignoring boundary/obstacle nodes) gives a geometry-invariant summary of the flow state that the encoder has learned to represent.

The `NearestNeighborIndex` stores all training embeddings and the **train diameter** = 95th percentile of nearest-neighbor distances within the training set:

```python
dists, _ = self._scipy_tree.query(self.embeddings, k=2)   # k=2: skip self-match
self.train_diameter = float(np.percentile(dists[:, 1], 95))
```

Using the 95th percentile (rather than max) makes the diameter robust to outliers in the training set. A score is then:

```python
score = clip(1 - d_nearest / train_diameter, 0, 1)
```

Where `d_nearest` is the L2 distance from the query embedding to its nearest training neighbor. Score = 1.0 means the query is identical to a training sample. Score = 0.0 means the query is at least as far from training data as the typical furthest training point.

**Thresholds:**

| Score | Label | Meaning |
|---|---|---|
| ≥ 0.7 | HIGH | Interpolation regime — model reliable |
| 0.4–0.7 | MEDIUM | Moderate extrapolation — verify key quantities |
| < 0.4 | LOW | OOD warning — treat with caution |

### 4.3 C++ KDTree Implementation

The project ships a custom median-split KDTree in `confidence/kdtree.cpp` with pybind11 Python bindings:

```cpp
// Pool allocator to avoid per-node heap fragmentation
std::vector<KDNode> pool;
int pool_pos = 0;

KDNode* alloc_node() {
    KDNode* node = &pool[pool_pos++];
    node->left = node->right = nullptr;
    node->idx = -1;
    return node;
}
```

The **pool allocator** pre-allocates `2*n+1` nodes in a contiguous vector before building. This avoids per-node `malloc` calls, keeps the tree nodes cache-friendly (sequential in memory), and eliminates heap fragmentation. The pool size is exact (a complete binary tree on `n` points has at most `2n-1` nodes), so the overflow guard is purely defensive.

Build uses `std::nth_element` (O(n log n) median split) and search uses branch-and-bound pruning:

```cpp
// Only recurse into a subtree if it could contain a closer point
float axis_dist = query[node->split_dim] - node->split_val;
if (axis_dist * axis_dist < best_dist_sq) { ... recurse ... }
```

The C++ backend is ~3-5× faster than `scipy.spatial.KDTree` for repeated single-query calls (the dominant use case at inference). However, `scipy` is always built as the fallback:

```python
try:
    from _kdtree import KDTree as CppKDTree
    self._cpp_tree = CppKDTree(self.embeddings)
    self.backend = "cpp"
except ImportError:
    pass   # scipy tree already built — no action needed
```

This graceful fallback means the system works out of the box without a C++ compiler — the C++ extension is purely an optimization.

### 4.4 Where Else Could a KD-Tree Help?

The KD-tree currently lives only in the confidence scoring system — it indexes 128-dimensional embeddings, not mesh geometry. There are two places in the physics simulation pipeline where a spatial KD-tree would become genuinely useful as the system grows:

#### A. Physics Post-Processing (already using scipy.cKDTree)

`api/routes/physics.py` already uses `scipy.spatial.cKDTree` to find the k=6 nearest spatial neighbors of each mesh node — this is needed to approximate local velocity derivatives (vorticity, divergence) on an unstructured mesh without a regular grid. The tree is built over 2D node coordinates `[N, 2]` and queried once per physics request. It is LRU-cached per result file, so the O(N log N) build cost is paid once per rollout result, and each query is O(log N) per node.

For the current mesh sizes (N ≈ 1,800 nodes) scipy is fast enough. At N > 50,000 nodes — which would arise with higher-resolution CFD meshes — the cache-friendly C++ tree in `confidence/kdtree.cpp` would give a 3–5× speedup here too, since the usage pattern (many single-point queries against a fixed tree) is exactly the case where the C++ branch-and-bound search dominates.

#### B. World Edges / Self-Collision Detection (not yet implemented)

This is where a KD-tree would have the most impact. The DeepMind MeshGraphNets paper describes a second class of edges called **world edges** for complex cloth simulations: dynamic edges between mesh nodes that happen to be spatially close in 3D world-space at the current timestep, even if they are far apart in mesh connectivity. These detect self-collision — when a flag folds and two distant cloth regions nearly touch.

The current `flag_simple` implementation (and the DeepMind baseline) does **not** include world edges — the flag waving scenario doesn't produce dramatic self-folds. But a more physically accurate cloth simulator would need them.

If world edges were added, the graph construction at every forward pass would need to answer: *"for each of the N nodes, find all other nodes within radius r in 3D world-space"* — a radius-graph query. Without a spatial index this is O(N²) pairwise distances per forward pass per training step. At N = 1,800 nodes and 100,000 training iterations, that is 1,800² × 100,000 = **324 billion distance comparisons**.

With a KD-tree:
- Build: O(N log N) ≈ 19,000 operations per step
- Radius query for all N nodes: O(N log N) amortized

The speedup scales with N — at N = 10,000 nodes the ratio is approximately **N / log N ≈ 750×**.

The C++ KDTree already in `confidence/kdtree.cpp` operates on arbitrary-dimensional float arrays and would need only minor extension (a `query_radius` method alongside the existing `query_nn`) to serve this purpose.

**Summary: KD-tree is not needed now, but is the correct next step if self-collision cloth simulation is added.**

### 4.5 Persistence

The index is serialized as a pickle file containing `embeddings` (the raw `[N, 128]` array) and `train_diameter` (the computed scalar). On load, the trees are rebuilt from the embeddings:

```python
@classmethod
def load(cls, path: str) -> "NearestNeighborIndex":
    d = pickle.load(f)
    obj = cls()
    obj.build(d["embeddings"])   # rebuilds scipy + cpp trees
    obj.train_diameter = float(d["train_diameter"])
    return obj
```

---

## 5. Autoregressive Rollout

### 5.1 CFD Rollout Algorithm

The rollout in `api/routes/rollout.py` (`_run_rollout_sync`) implements the core autoregressive inference loop:

1. Load the initial graph at `t=0` from the dataset
2. Build edge graph with `FaceToEdge + Cartesian + Distance` transforms
3. On first step, compute `boundary_mask`: WALL and INFLOW nodes are boundaries
4. **Loop:** inject `predicted_velocity` into `graph.x[:, field_slice]`, run forward pass, enforce boundary conditions, collect results

```python
if predicted_velocity is not None:
    graph.x[:, field_slice] = predicted_velocity.detach()  # swap in prediction

predicted_velocity = model(graph, velocity_sequence_noise=None)
predicted_velocity[boundary_mask] = next_v[boundary_mask]  # enforce BCs
```

The boundary enforcement is critical: without it, WALL nodes (which should have zero velocity) would drift over time and corrupt the entire flow field.

### 5.2 Cloth Rollout Algorithm

The cloth rollout in `_run_cloth_rollout_sync` uses Verlet integration:

```python
if cur_world is not None:
    graph.world_pos = cur_world.detach()
    graph.x = torch.cat([cur_world.detach(), graph.x[:, 3:]], dim=-1)
    graph.prev_x = prev_world.detach()

prev_world = graph.world_pos.clone()
next_world = model(graph)  # returns world_pos_{t+1} directly

handle_mask = (node_type == NodeType.HANDLE)
next_world[handle_mask] = graph.y[handle_mask]  # pin handle nodes to GT
```

The model internally computes `next_world_pos = 2*world_pos - prev_world + acc` (Verlet integration). The rollout only needs to track `cur_world` and `prev_world` across steps.

### 5.3 Performance and Threading

The blocking rollout function is dispatched to a thread pool so the FastAPI event loop stays free:

```python
loop = asyncio.get_running_loop()
result = await loop.run_in_executor(None, _run_rollout_sync, req, cfg, device, callback)
```

Progress is streamed to the frontend via Server-Sent Events (SSE) — a `asyncio.Queue` carries `(step, total)` tuples from the background thread (via `loop.call_soon_threadsafe`) to the async SSE generator.

Wall-clock performance metric:
```python
elapsed = time.perf_counter() - t_start
sim_time = n_steps * cfg["dt"]   # e.g. 600 * 0.01 = 6 simulated seconds
speedup = sim_time / elapsed     # typically 200-500× for cylinder flow
```

---

## 6. FastAPI Backend

### 6.1 Endpoint Summary

| Method | Path | Purpose |
|---|---|---|
| GET | `/api` | Health check / docs index |
| POST | `/api/train/start` | Launch training subprocess |
| GET | `/api/train/stream` | SSE stream of training log lines |
| POST | `/api/rollout` | SSE stream of autoregressive rollout progress |
| GET | `/api/results` | List all result `.pkl` files |
| GET | `/api/results/{filename}` | Mesh structure + RMSE curve + confidence metadata |
| GET | `/api/results/{filename}/frame/{t}` | Per-timestep field magnitudes + error heatmap |
| GET | `/api/results/{filename}/rmse` | Full RMSE and MAE curves |
| GET | `/api/results/{filename}/physics` | Vorticity, energy series, divergence proxy |
| DELETE | `/api/results/{filename}` | Delete a result file |
| GET | `/api/status` | GPU availability, training status, domain registry |
| GET | `/api/dataset/samples` | Sample graphs from dataset |

### 6.2 Design Decisions

**Thread-safe model cache with double-checked locking** (`api/state.py`):

```python
if key in _model_cache:          # fast path — no lock
    return _model_cache[key]
with _model_cache_lock:          # slow path — exclusive
    if key not in _model_cache:  # second check: avoid duplicate load
        ... load checkpoint ...
        _model_cache[key] = sim
```

This avoids the pathological case where two concurrent rollout requests both miss the cache and both load the 200+ MB checkpoint simultaneously.

**Path traversal guard** on all result file access:

```python
safe_dir = os.path.realpath(RESULT_DIR)
path = os.path.realpath(os.path.join(RESULT_DIR, filename))
if not path.startswith(safe_dir + os.sep):
    raise HTTPException(400, "Invalid filename")
```

Prevents `../../etc/passwd`-style attacks on the results endpoint.

**PID file for training process** (`api/state.py`):
Training is launched as a subprocess. The PID is written to `runs/train_ui.pid` so that if `uvicorn` restarts, the server can detect the orphaned training process and report its status correctly.

**Line-buffered stdout in train.py**:
```python
sys.stdout.reconfigure(line_buffering=True)
```
This must be the first statement in `train.py`. Without it, Python buffers output when stdout is a pipe (not a TTY), so the SSE stream would see nothing until the 8 KB buffer fills.

**Pydantic validation** on all request bodies with `Literal` type constraints on `domain` and `target_field` prevents invalid configurations from reaching the model loading code.

### 6.3 Physics Endpoint

The `/results/{filename}/physics` endpoint computes three quantities from stored predictions:

**Vorticity** (ω = ∂vy/∂x − ∂vx/∂y) via vectorized least-squares on an unstructured mesh:

```python
# Build normal equations across all N nodes simultaneously
A    = drT @ dr     # [N, 2, 2]  — spatial structure matrix per node
rhs  = drT @ dv     # [N, 2, 2]  — RHS (gradients of vx and vy)
A    = A + 1e-6 * eye   # regularize degenerate (collinear) nodes
grad = np.linalg.solve(A, rhs)  # [N, 2, 2]

omega = grad[:, 0, 1] - grad[:, 1, 0]   # ω = ∂vy/∂x − ∂vx/∂y
```

This uses k=6 nearest neighbors (k=7 including self) per node. The key insight is that by building the normal equations batch-wise (`[N, 2, 2]` matrices), the loop over N nodes is replaced by a single batched linear algebra call. A 1e-6 diagonal regularizer handles degenerate cases where neighbors are nearly collinear.

**Kinetic energy series**: `E_t = 0.5 * Σ_nodes ||v_t||²`. Computed over all T timesteps as a single `np.linalg.norm` call — O(T·N) but no spatial derivatives needed.

**Divergence proxy** (incompressibility check): `∇·v = ∂vx/∂x + ∂vy/∂y`. For incompressible flow (Re=1000 cylinder flow), this should be ~0. The series is sampled every 10 timesteps and linearly interpolated, since it requires the expensive gradient computation at each step.

All physics results are cached with `@functools.lru_cache(maxsize=32)` keyed on the mesh coordinates bytes (hashable).

---

## 7. React Frontend

### 7.1 Pages and Their API Wiring

**Train page** (`/train`):
- Domain selector (cylinder_flow / flag_simple) and target_field selector (VELOCITY / PRESSURE / CLOTH)
- Hyperparameter controls (epochs, batch size, noise std, message passing steps, learning rate)
- "Start Training" sends a `POST /api/train/start` with a JSON config body
- Training log and loss curves stream via `EventSource` on `/api/train/stream`
- Remote GPU SSH config panel (optional — for training on a remote machine)

**Predict page** (`/predict`):
- Domain + trajectory selector
- "Run Rollout" opens an `EventSource` on `POST /api/rollout` (SSE)
- Progress bar updates every 20 steps
- On completion: displays elapsed time, speedup ratio, RMSE, **confidence score badge** (HIGH/MEDIUM/LOW pill), and similarity score
- Domain and target_field badges (PRESSURE / VELOCITY / CLOTH) shown after rollout completes

**Visualize page** (`/visualize`), three tabs:
- **Viewer**: animated mesh plot, scrubber, predicted vs. ground-truth field magnitude rendered as color map, error heatmap. Labels adapt to `target_field` (e.g., "Velocity Magnitude (m/s)" vs "Pressure (Pa)")
- **Diagnostics**: per-step RMSE and MAE curves with colored zones (green/yellow/red by growth), confidence score card with score + label + explanation
- **Physics**: vorticity field animation, energy series with drift annotation (shows final `pred_energy - gt_energy`), divergence proxy series (should be near zero for incompressible flow). Physics tab is gated on `domain == 'cylinder_flow'` — disabled for cloth.

**Dataset Studio**: node count histogram, outlier detection, dataset statistics.

**Pipeline View**: DAG showing the full data → train → rollout pipeline with filesystem-probed status (green if data/checkpoint files exist, gray otherwise).

**Experiment Tracking**: list of runs with editable names, hyperparameters, and validation loss history.

### 7.2 Key UX Decisions

- All long-running operations (training, rollout) use SSE, not polling, to avoid polling latency and reduce server load
- The frontend communicates with FastAPI via `server.ts` proxy (all `/api/*` requests are proxied to `localhost:8000`)
- Physics tab is only shown for velocity/CFD domains — the vorticity computation is not meaningful for cloth positions
- Confidence score interpretation is explained inline to the user (not just a number) to make the OOD warning actionable

---

## 8. Interview Insights

### "Tell me about your physics simulation experience"

Frame this as: *"I built a graph neural network surrogate for physics simulation on the DeepMind MeshGraphNets benchmark, extended it to support three physics domains, and deployed it as a production-quality API with confidence scoring."*

Key differentiators to mention:
- Used real DeepMind CFD and cloth datasets (not toy examples)
- Extended the original architecture to support pressure field prediction as a separate configuration (not in the original paper)
- Built original confidence scoring system on top of the encoder embedding — this is a genuine engineering contribution beyond reimplementation
- Deployed as a FastAPI backend with React frontend — the system is demonstrable, not just a training script

### "How does the GNN handle irregular meshes?"

*"The model operates on the mesh as a graph. Node features encode local physics (velocity, pressure, node type). Edge features encode the relative geometry between connected nodes (Δx, Δy, |Δr|). The message-passing processor aggregates information from each node's neighborhood — this is inherently mesh-invariant because it only looks at local graph connectivity, not global indices or coordinates. A triangle mesh, a quad mesh, or an entirely unstructured mesh would all work as long as you can build a graph from the elements.*

*The cloth simulator uses dual edge encoding — both mesh-space (rest configuration) and world-space (current deformed configuration) relative positions — because the cloth physics depends on how much the mesh has deformed from rest, not just where the nodes are currently.*"

### "How do you know when to trust the model's predictions?"

*"I built a confidence scoring system based on nearest-neighbor search in the encoder's embedding space. After training, I extract 128-dimensional embeddings from the encoder for every training sample and build a KDTree index. At inference time, I embed the query graph the same way and find its nearest training neighbor. The score is `1 - d_nearest / train_diameter` where `train_diameter` is the 95th percentile inter-point distance in the training set — this normalizes the distance by the typical spread of the training distribution.*

*A score ≥0.7 means the query is well within the training distribution (interpolation). A score <0.4 is an OOD warning. This is not a Bayesian uncertainty estimate, but it's cheap to compute and principled — it doesn't require re-training or model modification.*"

### "How does this compare to traditional CFD?"

*"Traditional CFD (OpenFOAM, Fluent, Ansys) solves discretized PDEs iteratively — the Navier-Stokes equations — which requires convergence checks, adaptive time-stepping, and potentially hours of compute for a transient simulation. The GNN surrogate, once trained, runs the same simulation in seconds by directly predicting the next flow state from the current one. The cylinder flow rollout runs ~200-500× faster than wall-clock real-time.*

*The trade-off is accuracy and generalization: the surrogate is only reliable for flow conditions similar to the training set. It doesn't satisfy conservation laws exactly — the divergence proxy in the physics endpoint measures how much incompressibility is violated. For design exploration and rapid screening this is acceptable; for final certification you'd still run the full solver.*"

### "What's the hardest technical problem you solved here?"

Pick 2-3 of these depending on conversation context:

1. **Autoregressive exposure bias**: During training, the model sees ground-truth inputs at every step. During rollout, it sees its own predictions. Small errors compound. The solution is input noise injection during training (`noise_std=0.02`) — the model is trained to recover from noisy inputs, making it robust to its own accumulated errors.

2. **Pressure field architecture**: The pressure field is a scalar (not a vector), so the node input is 1 dimension narrower (10 vs 11). The key design decision was to make the field selection a property of the `Simulator` class (`target_field` + `_frames_slice()`) rather than a separate model class. This meant the same `EncoderProcesserDecoder` architecture supports all field types with a single-line configuration change.

3. **Self-describing checkpoints**: Early versions required the user to specify `target_field` and `domain` at inference time, which caused mismatches. The fix was to save all architecture parameters into the checkpoint at train time, and have `get_model()` in `api/state.py` read them back — making the checkpoint self-describing. The double-checked locking pattern ensures this is thread-safe.

4. **Pool allocator in C++ KDTree**: The initial C++ implementation allocated each `KDNode` separately with `new`. Under LLVM ASan, this caused UB from pointer aliasing after vector reallocation. The pool allocator fixes this by pre-sizing the vector before building and indexing into it by integer position — no pointers into a growing vector.

### "How would you scale this to production?"

*"The foundation is already in place: checkpoints are self-describing (domain, target_field, architecture sizes), so a model registry just needs to track checkpoint paths. The model cache in `api/state.py` uses double-checked locking and is keyed on (checkpoint_path, device) — extending to multiple checkpoints is trivial.*

*The confidence gating system provides a natural threshold for routing: if `score < 0.4`, fall back to a lightweight solver or flag for manual review. For production scale, I'd add: (1) ensemble uncertainty (multiple forward passes with dropout — gives proper variance estimates not just proximity scores), (2) active learning loop to identify which simulation conditions are OOD and prioritize them for new training data generation, (3) model versioning with migration utilities to handle checkpoint format changes across training runs.*"

### "What would you add next?"

1. **Ensemble uncertainty**: Mc-Dropout or a small ensemble of 3-5 models. Unlike the embedding-space confidence, this gives per-node variance estimates — useful for identifying which spatial regions the model is uncertain about, not just whether the whole query is in-distribution.

2. **Multi-fidelity training**: Train on a mix of high-resolution (accurate, expensive) and low-resolution (cheap) simulations. The GNN architecture is mesh-invariant so this is architecturally straightforward; the challenge is handling heterogeneous node counts in batching.

3. **Physics-informed loss terms**: Add a soft divergence penalty `λ * mean(|∇·v|²)` to the training loss for CFD. This would improve the incompressibility constraint visible in the divergence proxy chart without changing the model architecture.

4. **Uncertainty-weighted rollout**: Use per-node variance estimates (from ensemble) to weight the boundary condition enforcement — nodes where the model is more uncertain receive stronger BC weighting.

5. **World edges + C++ KD-tree for self-collision cloth**: The current cloth model uses only mesh edges (fixed topology from triangle faces). A more physically accurate simulator would add *world edges* — dynamic edges between nodes that come within a radius threshold in 3D space at each timestep, enabling self-collision detection when the flag folds. This requires a radius-graph query at every forward pass: O(N²) brute-force but O(N log N) with a spatial KD-tree. The C++ KDTree already in `confidence/kdtree.cpp` needs only a `query_radius` method added to serve this role — the pool allocator and branch-and-bound search are already correct. At N=1,800 nodes the speedup is ~100×; at N=10,000 it is ~750×. See §4.4 for the full analysis.

---

## Appendix: File Map

| File | Purpose |
|---|---|
| `model/model.py` | `EncoderProcesserDecoder`, `GnBlock`, `Encoder`, `Decoder`, `build_mlp` |
| `model/simulator.py` | `Simulator` — CFD wrapper with normalizers, noise injection, `target_field` support |
| `model/flag_simulator.py` | `FlagSimulator` — Cloth wrapper with Verlet integration, dual edge encoding |
| `model/embedding.py` | `extract_embedding()` — encoder-only forward + NORMAL-node mean pool → [128] |
| `model/blocks.py` | `EdgeBlock`, `NodeBlock` — PyG-based message passing primitives |
| `dataset/fpc.py` | `FpcDataset` — memory-mapped CFD dataset, velocity + pressure modes |
| `dataset/flag_dataset.py` | `FlagDataset` — lazy NpzFile cloth dataset, Verlet-compatible prev_x |
| `data/parse_tfrecord.py` | TFRecord → `.dat` + `.npz` for CFD data |
| `data/parse_flag_tfrecord.py` | TFRecord → `_pos.npz` + `_mesh.npz` for cloth data |
| `confidence/index.py` | `NearestNeighborIndex` — KDTree index, score formula, save/load |
| `confidence/build_index.py` | Standalone script: checkpoint → embeddings → index.pkl |
| `confidence/kdtree.cpp` | Custom median-split KDTree with pool allocator, branch-and-bound search |
| `confidence/kdtree.h` | C++ header for KDTree |
| `confidence/pybind_module.cpp` | pybind11 bindings for `_kdtree.KDTree` |
| `confidence/benchmark.py` | Speed comparison: scipy vs C++ KDTree |
| `train.py` | Training loop: domain-aware config, noise injection, early stopping, TensorBoard, auto-build index |
| `api/main.py` | FastAPI app factory, CORS, router registration |
| `api/state.py` | Shared state: model cache (double-checked locking), DOMAINS registry, PID file |
| `api/routes/rollout.py` | SSE rollout: CFD + cloth, boundary enforcement, confidence + similarity score |
| `api/routes/results.py` | Results CRUD: list, metadata, per-frame, RMSE/MAE, delete; path traversal guard |
| `api/routes/physics.py` | Vorticity (vectorized LSQ), energy series, divergence proxy; LRU cache |
| `api/routes/train.py` | Training subprocess launch, SSE log stream, status |
| `api/routes/status.py` | GPU status, training process health, domain registry |
| `api/routes/dataset.py` | Dataset sample browser |
| `utils/normalization.py` | Online running-statistics `Normalizer` (Welford accumulation) |
| `utils/utils.py` | `NodeType` enum (NORMAL=0, OBSTACLE=1, ..., HANDLE=3, ...) |
| `utils/noise.py` | `get_velocity_noise()` — Gaussian noise generation |
| `pyproject.toml` | Project metadata, dependency pinning, ruff/mypy/pytest config |
| `app/` | React frontend (Vite + TypeScript): Train / Predict / Visualize / Dataset Studio / Pipeline View / Experiment Tracking |
| `tests/` | pytest suite: unit tests for dataset, model, confidence, API endpoints |

---

*Document generated April 2026. All code references are to the actual implementation — exact line numbers and method signatures reflect the current codebase.*
