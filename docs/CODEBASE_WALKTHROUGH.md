---
tags: [physicsai, reference, codebase, deep-dive]
created: 2026-04-10
aliases: [walkthrough, code-guide, codebase-map]
---

# Codebase Walkthrough
### Every file, every function — from high-level flow down to individual lines

## Quick summary

- Starts with the big picture (what runs what), then drills into each file
- Covers both domains: CFD (`cylinder_flow`) and cloth (`flag_simple`)
- Explains every function's inputs, outputs, and the one thing it does
- No assumed knowledge beyond basic Python and PyTorch

---

## Part 1 — The big picture: what calls what

The project has three separate entry points depending on what you want to do:

```
python train.py        ← train a new model
python rollout.py      ← run inference on a trained model
uvicorn api.main:app   ← start the web API (used by the React UI)
```

Each one reaches down into the same shared modules:

```
train.py / rollout.py / api/
    │
    ├── dataset/fpc.py              ← CFD data loading
    ├── dataset/flag_dataset.py     ← Cloth data loading
    │
    ├── model/simulator.py          ← CFD predictor (wraps GNN)
    ├── model/flag_simulator.py     ← Cloth predictor (wraps GNN)
    │     └── model/model.py        ← EncoderProcessorDecoder GNN
    │           └── model/blocks.py ← EdgeBlock, NodeBlock (atomic ops)
    │
    ├── utils/normalization.py      ← Normalizer (running mean/std)
    ├── utils/noise.py              ← Training noise injection
    ├── utils/utils.py              ← NodeType enum
    │
    ├── model/embedding.py          ← OOD embedding extraction
    ├── confidence/index.py         ← KDTree nearest-neighbour index
    │
    └── extensions/generative/      ← CVAE inverse design system
          ├── cvae_cfd.py
          ├── cvae_cloth.py
          ├── drag_surrogate.py
          ├── mesh_generator.py
          └── inverse_design.py
```

---

## Part 2 — Bottom layer: `model/blocks.py`

This is the atomic unit of the GNN. Everything else is built on top of these two classes.

### `EdgeBlock`

**What it does:** Updates every edge's representation using information from both its endpoint nodes.

```python
class EdgeBlock(nn.Module):
    def __init__(self, custom_func: nn.Module):
        self.net = custom_func   # an MLP passed in from outside
```

**`forward(graph) → Data`:**

```
Input:  graph.x         [N, hidden]   — current node representations
        graph.edge_index [2, E]       — pairs of (sender_idx, receiver_idx)
        graph.edge_attr  [E, hidden]  — current edge representations

Step 1: gather sender and receiver node features for every edge
        senders_attr   = graph.x[edge_index[0]]   # [E, hidden]
        receivers_attr = graph.x[edge_index[1]]   # [E, hidden]

Step 2: concatenate all three → [E, 3*hidden]
        collected = cat([senders_attr, receivers_attr, edge_attr], dim=1)

Step 3: run MLP → new edge representation
        new_edge_attr = self.net(collected)        # [E, hidden]

Output: Data with same x and edge_index, but updated edge_attr
```

The key operation is **index gather**: `graph.x[senders_idx]` picks the row corresponding to each edge's sender node. This is just Python indexing — no magic.

---

### `NodeBlock`

**What it does:** Updates every node's representation by aggregating the messages from all its incoming edges, then running an MLP.

**`forward(graph) → Data`:**

```
Input:  graph.x         [N, hidden]  — current node representations
        graph.edge_index [2, E]
        graph.edge_attr  [E, hidden]  — already updated by EdgeBlock

Step 1: sum all incoming edge messages per node
        receivers_idx = edge_index[1]              # [E] — which node receives each edge
        agg = scatter(edge_attr, receivers_idx,    # [N, hidden]
                      dim=0, dim_size=N, reduce='sum')
        # agg[i] = sum of edge_attr for all edges pointing INTO node i

Step 2: concatenate node's own features with aggregated messages
        collected = cat([graph.x, agg], dim=-1)   # [N, 2*hidden]

Step 3: run MLP → new node representation
        new_x = self.net(collected)               # [N, hidden]

Output: Data with updated x, same edge_attr and edge_index
```

`scatter` is from `torch_geometric.utils`. It's equivalent to:
```python
agg = torch.zeros(N, hidden)
for e in range(E):
    agg[receivers_idx[e]] += edge_attr[e]
```
but vectorised on GPU.

---

## Part 3 — GNN architecture: `model/model.py`

Four classes that stack on top of `EdgeBlock` and `NodeBlock`.

### `build_mlp(in_size, hidden_size, out_size, lay_norm=True)`

A helper function. Builds:
```
Linear(in → hidden) → ReLU
Linear(hidden → hidden) → ReLU
Linear(hidden → out)
[optional: LayerNorm(out)]
```
This is the MLP used everywhere: inside EdgeBlock, NodeBlock, Encoder, and Decoder. LayerNorm is included by default everywhere **except** the Decoder (because the Decoder's output is the actual prediction — normalising it would destroy the scale).

---

### `Encoder`

**What it does:** Takes the raw input graph (with physical features) and maps it into the latent 128-dimensional space.

```
Input:  graph.x         [N, node_input_size]   e.g. [N, 11] for CFD
        graph.edge_attr  [E, edge_input_size]   e.g. [E, 3]  for CFD

Two separate MLPs — nodes and edges are encoded independently:
        node_  = nb_encoder(graph.x)          [N, 128]
        edge_  = eb_encoder(graph.edge_attr)   [E, 128]

Output: Data with x=[N,128], edge_attr=[E,128]
```

After this step, all domain-specific information (velocities, positions, node types) has been compressed into 128-d vectors. The Processor doesn't know what domain it's in — it just works on 128-d latent vectors.

---

### `GnBlock` (one round of message passing)

**What it does:** One complete encode → process → residual step. Used 15 times in a row.

```python
def forward(self, graph):
    x_old         = graph.x.clone()          # save residual
    edge_attr_old = graph.edge_attr.clone()

    graph = self.eb_module(graph)    # EdgeBlock: update edges
    graph = self.nb_module(graph)    # NodeBlock: update nodes

    graph.x        = x_old + graph.x           # residual: new = old + delta
    graph.edge_attr = edge_attr_old + graph.edge_attr

    return graph
```

The `.clone()` before and `+` after is the **residual connection**. Each block only learns the *change* (delta) to apply to the current representation. After 15 blocks, the representation has been refined 15 times.

---

### `Decoder`

**What it does:** Maps final node latent representations to the physical output.

```
Input:  graph.x  [N, 128]   — final latent node representations

Output: [N, output_size]
        output_size=2  → velocity (vx, vy)
        output_size=1  → pressure
        output_size=3  → cloth position (x, y, z)
```

One MLP with `lay_norm=False` — no LayerNorm because the output scale carries information.

---

### `EncoderProcesserDecoder`

The top-level GNN class. Wires everything together:

```python
def forward(self, graph):
    graph = self.encoder(graph)               # raw features → 128-d latent
    for block in self.processer_list:         # 15× GnBlock (or TNSBlock / SAGEBlock)
        graph = block(graph)
    return self.decoder(graph)                # 128-d latent → physical output
```

This is what `Simulator` and `FlagSimulator` both wrap. It receives a normalised graph and returns a normalised prediction. It knows nothing about physics — that's handled by the wrappers.

---

### `TNSBlock` *(added — Transformer-style message passing)*

A drop-in replacement for `GnBlock` in the Processor. Uses multi-head self-attention over edge embeddings instead of a plain MLP.

```
For each edge e with embedding h_e  [E, 128]:
    Q = W_Q · h_e,  K = W_K · h_e,  V = W_V · h_e    [E, 128 each]
    scores = softmax(Q · K^T / √d_k)                  — attention over neighbourhood
    attended = scores · V                              [E, 128]
    new_edge_attr = LayerNorm(h_e + attended)          — residual + norm
```

Node update then follows the same `NodeBlock` scatter-sum as `GnBlock`.

**Training notes:**
- Requires a clipped learning rate of **3e-5** (vs 1e-3 for GnBlock) — attention weights are sensitive to large gradient steps.
- Gradient norm clipping of **1.0** must be applied; without it TNSBlock training diverges.
- Select via `architecture='tns'` in `Simulator.__init__`.

---

### `SAGEBlock` *(added — GraphSAGE-style aggregation)*

Another drop-in replacement for `GnBlock`. Uses **mean aggregation** (not sum) and concatenates the node's own features with the neighbourhood mean before the linear layer, following the GraphSAGE formulation.

```
agg[i] = mean over neighbours j of h_j              [N, 128]
h_new[i] = Linear(cat([h_i, agg[i]]))               [N, 128]
```

Mean aggregation is more robust to high-degree nodes (obstacle surfaces in CFD where many edges converge) than sum aggregation.

- Select via `architecture='sage'` in `Simulator.__init__`.

---

## Part 4 — CFD wrapper: `model/simulator.py`

`Simulator` is the public interface for CFD training and inference. It wraps `EncoderProcesserDecoder` and handles everything domain-specific: feature construction, normalisation, noise, and denormalisation.

### Constructor

```python
Simulator(
    message_passing_num=15,
    node_input_size=11,    # 2 (velocity) + 9 (one-hot node type)
    edge_input_size=3,     # Δx + Δy + distance
    device='cuda:0',
    target_field='velocity',  # or 'pressure'
    architecture='gn'         # 'gn' | 'tns' | 'sage'  ← added
)
```

The `architecture` parameter selects which block type populates the 15-step Processor:
- `'gn'` → `GnBlock` (original, default)
- `'tns'` → `TNSBlock` (transformer attention)
- `'sage'` → `SAGEBlock` (mean-agg GraphSAGE)

The chosen architecture name is **stored in the checkpoint** alongside domain and target_field. On `load_checkpoint`, a mismatch raises a stale-checkpoint error so a `gn` checkpoint is never silently loaded into a `tns` model.

Creates three `Normalizer` instances:
- `_node_normalizer` — normalises node features before encoder
- `edge_normalizer` — normalises edge features before encoder
- `_output_normalizer` — normalises targets during training, inverts during inference

---

### `update_node_attr(frames, types) → [N, node_input_size]`

Constructs normalised node features from raw inputs:

```
frames:  [N, 2] velocity  OR  [N, 1] pressure
types:   [N, 1] integer node type (0–8)

Step 1: squeeze type to [N], convert to one-hot [N, 9]
Step 2: cat([frames, one_hot], dim=-1)   → [N, 11] or [N, 10]
Step 3: pass through _node_normalizer    → zero-mean, unit-std
```

---

### `forward(graph, velocity_sequence_noise)`

The logic splits on `self.training`:

**Training mode:**
```
1. Add noise to interior fluid nodes (via velocity_sequence_noise)
2. Build normalised node features from noised input
3. Normalise edge features
4. Run EncoderProcesserDecoder → predicted_change_norm  [N, output_size]
5. Compute target_change = graph.y - noised_frames
6. Normalise target_change → target_change_norm
7. Return (predicted_change_norm, target_change_norm)
   ← caller computes MSE between these two
```

**Inference mode (eval):**
```
1. Build normalised node features from clean input
2. Normalise edge features
3. Run EncoderProcesserDecoder → predicted_change_norm
4. Denormalise: delta = _output_normalizer.inverse(predicted_change_norm)
5. next_value = current_frames + delta
6. Return next_value   [N, 2] velocity or [N, 1] pressure
```

---

### `_frames_slice()`

Small utility — returns which columns of `graph.x` hold the physical field:
```python
'velocity' → slice(1, 3)   # graph.x[:, 1:3]
'pressure' → slice(1, 2)   # graph.x[:, 1:2]
```
Column 0 is always the node type integer.

---

## Part 5 — Cloth wrapper: `model/flag_simulator.py`

Same role as `Simulator` but for 3D cloth dynamics. Key differences:

| | Simulator (CFD) | FlagSimulator (Cloth) |
|---|---|---|
| Domain | 2D fluid | 3D cloth |
| node_input_size | 11 | 12 |
| edge_input_size | 3 | 7 |
| output_size | 2 or 1 | 3 |
| Integration | first-order (v + Δv) | Verlet (2p - p_prev + acc) |
| Boundary condition | INFLOW/WALL pinning | HANDLE node pinning |
| Edge features | Δx, Δy, dist | rel_world[3], \|world\|, rel_mesh[2], \|mesh\| |

---

### `_build_graph(graph) → Data`

Cloth graphs arrive with `face [3, F]` (triangles) but the GNN needs `edge_index [2, E]`. This method:

1. Calls `FaceToEdge` to convert faces → directed edges
2. Builds cloth-specific edge features:
   ```
   rel_world  = world_pos[senders] - world_pos[receivers]   [E, 3]
   world_norm = ||rel_world||                                [E, 1]
   rel_mesh   = mesh_pos[senders]  - mesh_pos[receivers]    [E, 2]
   mesh_norm  = ||rel_mesh||                                 [E, 1]
   edge_attr  = cat([rel_world, world_norm, rel_mesh, mesh_norm])  [E, 7]
   ```
   `world_pos` is the current 3D position. `mesh_pos` is the fixed 2D rest-pose coordinate. The difference encodes stretch relative to rest length.

---

### `_build_node_features(graph, node_type_col) → [N, 12]`

```
velocity  = world_pos - prev_world   [N, 3]   ← finite-difference velocity
one_hot   = F.one_hot(node_type, 9)  [N, 9]
node_feats = cat([velocity, one_hot]) [N, 12]
```

Note: velocity here is estimated from two consecutive world positions, not stored directly.

---

### `forward(graph)` — training

```
target_acc = graph.y - 2*world_pos + prev_world   ← Verlet acceleration
```
This is the second-order finite difference: given three consecutive positions (prev, current, next), acceleration = next − 2·current + prev.

### `forward(graph)` — inference (Verlet integration)

```
acc           = _output_normalizer.inverse(predicted_acc_norm)
next_world    = 2*world_pos - prev_world + acc     ← Verlet update

# Pin HANDLE nodes (cloth corners) — boundary condition
handle_mask   = (node_type != NORMAL)              [N, 1] bool
next_world    = torch.where(handle_mask, world_pos, next_world)
                # if HANDLE: keep current position
                # if NORMAL: use predicted next position
```

`torch.where(condition, a, b)` returns `a` where condition is True, `b` otherwise. This is how the cloth corners are pinned to the pole — their predicted motion is discarded and replaced with their current position.

---

## Part 6 — Utilities: `utils/`

### `utils/normalization.py` — `Normalizer`

A running-statistics normaliser stored as an `nn.Module`.

**Key methods:**

| Method | What it does |
|---|---|
| `forward(data, accumulate=True)` | Normalises data; if `accumulate=True` and still under 1M updates, updates running stats |
| `inverse(data)` | Denormalises: `data * std + mean` |
| `_accumulate(data)` | Adds to `_acc_sum` and `_acc_sum_squared` |
| `_mean()` | Returns `_acc_sum / _acc_count` |
| `_std_with_epsilon()` | Returns `sqrt(E[x²] - E[x]²)`, clamped to avoid NaN |

All four running buffers (`_acc_count`, `_num_accumulations`, `_acc_sum`, `_acc_sum_squared`) are registered with `register_buffer` — saved in checkpoint, moved to GPU automatically.

> [!warning] Known performance bug
> `if self._num_accumulations < self._max_accumulations` compares a GPU tensor to a Python int, forcing a CUDA→CPU synchronisation on every batch. Fix: cache a Python bool once the limit is hit.

---

### `utils/noise.py` — `get_velocity_noise`

```python
def get_velocity_noise(graph, noise_std, device):
    velocity = graph.x[:, 1:3]                          # [N, 2]
    noise    = torch.normal(0.0, noise_std, size=velocity.shape)
    mask     = graph.x[:, 0] != NodeType.NORMAL         # boundary nodes
    noise[mask] = 0                                     # zero out boundary nodes
    return noise
```

Only NORMAL (interior fluid) nodes get noise. Boundary nodes always have prescribed values so they must not be perturbed.

---

### `utils/utils.py` — `NodeType`

An `IntEnum` mapping names to integers:

```python
class NodeType(IntEnum):
    NORMAL        = 0
    OBSTACLE      = 1
    AIRFOIL       = 2
    HANDLE        = 3
    INFLOW        = 4
    OUTFLOW       = 5
    WALL_BOUNDARY = 6
    SIZE          = 9
```

Used throughout for masking (loss computation, noise injection, boundary pinning).

---

## Part 7 — Data loading: `dataset/`

### `dataset/fpc.py` — `FpcDataset`

CFD dataset. Loads from `.dat` (memmap) + `_meta.npz` files.

**Constructor:**
```python
FpcDataset(data_root='data', split='train', target_field='velocity')
```
- Opens `split.dat` as `np.memmap` — never reads the whole file
- Loads `split_meta.npz` — contains index arrays, face arrays, node types

**`__getitem__(idx)`:**
```
1. Look up trajectory index and timestep from flat index
   traj_idx, t = divmod(idx, T-1)
2. Slice memmap: positions[traj_start:traj_end, t:t+2]
   → [N, 2, 2]  (N nodes, 2 timesteps, 2 coords)
3. Compute velocity: pos[:, 1] - pos[:, 0]   [N, 2]
4. Target:           pos[:, 2] - pos[:, 1]   [N, 2]
5. Build PyG Data(x, pos, y, face)
```

---

### `dataset/flag_dataset.py` — `FlagDataset`

Cloth dataset. Same structure but stores 3D world positions.

**`__getitem__(idx)`:**
```
Returns Data with:
  pos        [N, 2]   — mesh_pos (2D rest coordinates, fixed)
  world_pos  [N, 3]   — current 3D position
  prev_x     [N, 3]   — previous 3D position (for Verlet)
  x          [N, 4]   — [world_pos[0..2], node_type]
  y          [N, 3]   — next 3D position (target)
  face       [3, F]   — triangle connectivity
```

---

## Part 8 — Training entry point: `train.py`

This is the script you run to train a model. It's structured as a flat script (not a class) that runs top-to-bottom when executed.

### Startup (lines 1–110)

```
1. Reconfigure stdout to line-buffered (so SSE streaming sees print output)
2. Parse --config argument (JSON file from UI) or use _defaults
3. Determine domain ('cylinder_flow' or 'flag_simple')
4. Set derived sizes (node_input_size, edge_input_size, output_size) from domain
5. Create device ('cuda:0' or 'cpu')
6. Instantiate simulator (Simulator or FlagSimulator based on domain)
7. Create Adam optimizer
8. Create TensorBoard SummaryWriter
```

For CFD, also creates the transform pipeline:
```python
transformer = T.Compose([
    T.FaceToEdge(),        # face [3,F] → edge_index [2,E]
    T.Cartesian(norm=False), # adds edge_attr Δx, Δy
    T.Distance(norm=False)   # adds edge_attr distance
])
```
For cloth, `transformer = None` because `FlagSimulator._build_graph()` handles it internally.

---

### `load_checkpoint(path, model, optimizer, device)`

Loads a saved checkpoint. Validates that the checkpoint's `domain` and `target_field` match the current run — raises an error if not (prevents accidentally loading a cloth checkpoint for a CFD run).

Returns `(start_epoch, best_valid_loss)`.

---

### `train_one_epoch(model, dataloader, optimizer, transformer, device, noise_std, domain, target_field)`

One full pass over the training data:

```
for each batch:
    1. Apply transformer (CFD only — converts faces → edges + edge features)
    2. Move graph to device
    
    CFD path:
        3. Generate noise (NORMAL nodes only)
        4. model(graph, noise) → (predicted_norm, target_norm)
        5. Mask to NORMAL + OUTFLOW nodes
        6. loss = mean((predicted - target)²)
    
    Cloth path:
        3. model(graph) → (predicted_acc_norm, target_acc_norm)
        4. Mask to NORMAL nodes only
        5. loss = mean(sum_over_xyz((predicted - target)²))
        ← different formula: sum over xyz first, then mean over nodes
    
    7. optimizer.zero_grad()
    8. loss.backward()
    9. optimizer.step()
```

---

### `evaluate(model, dataloader, transformer, device, domain)`

Validation pass. Runs `model.eval()` and `torch.no_grad()`. Returns RMSE (root mean squared error) averaged over all batches.

---

### Main training loop (lines 208–299)

```
1. Create datasets and DataLoaders
2. Move model to device
3. Load checkpoint if it exists
4. For each epoch:
     a. train_one_epoch → train_loss
     b. evaluate        → valid_loss
     c. Log to TensorBoard
     d. If best valid_loss so far: save checkpoint (with domain, target_field, sizes)
     e. Early stopping: if no improvement for patience epochs, stop
5. After training: build confidence index
     - extract_embedding() for every training sample
     - NearestNeighborIndex.build(embeddings)
     - Save index to runs/embedding_index.pkl
```

---

## Part 9 — OOD confidence: `model/embedding.py`

### `extract_embedding(simulator, graph, device) → np.ndarray [128]`

Public function. Dispatches based on simulator type:
- `isinstance(simulator, FlagSimulator)` → `_extract_embedding_cloth()`
- else → `_extract_embedding_cfd()`

### `_extract_embedding_cfd(simulator, graph, device)`

```
1. Normalise node and edge features exactly as simulator.forward() does
2. Run only the encoder: simulator.model.encoder(graph)  → [N, 128]
3. Find NORMAL nodes (node_type == 0)
4. Mean-pool over NORMAL nodes → [128]
5. Return as numpy array
```

### `_extract_embedding_cloth(simulator, graph, device)`

```
1. Call simulator._build_graph() to build edge features
2. Call simulator._build_node_features() to build node features
3. Normalise exactly as FlagSimulator.forward() does
4. Run encoder → [N, 128]
5. Mean-pool over NORMAL nodes → [128]
```

Both functions run under `torch.no_grad()` and `simulator.eval()` — no gradient tracking, no normalizer updates.

### Changes to `_extract_embedding_cfd` *(updated)*

- **`CFD_WARMUP_FRAMES = 5`** — module-level constant. The encoder is run on both frame 0 **and** frame 5 of each trajectory.
- Returns a **256-dim dual-frame embedding**: `cat([embed_frame0, embed_frame5])`. This gives the index information about the transient startup phase, not just the initial state.
- Pooling is now over **all nodes** (not just NORMAL nodes). Obstacle and boundary nodes contribute to the embedding, making different cylinder geometries more distinguishable.
- `extract_embedding()` dispatcher: detects CFD by checking `not isinstance(simulator, FlagSimulator)` and routes to the dual-frame path; cloth continues to use the 128-dim single-frame path.

---

## Part 10 — Confidence index: `confidence/index.py`

### `NearestNeighborIndex`

Stores training embeddings and answers "how similar is this new embedding to the training set?"

**`build(embeddings: np.ndarray)`**
```
embeddings: [N_train, 128]
1. Try to build C++ KDTree (via pybind11) — fast
2. Fall back to scipy.spatial.KDTree if C++ unavailable
3. Store train_diameter = max pairwise distance in training set
   (used to normalise the confidence score)
```

**`query(embedding: np.ndarray) → (confidence: float, is_ood: bool)`**
```
1. Find nearest training embedding: dist, idx = tree.query(embedding, k=1)
2. confidence = 1.0 - (dist / train_diameter)
   → 1.0 = identical to a training point
   → 0.0 = as far as the two farthest training points
   → negative = farther than any training pair (very OOD)
3. is_ood = confidence < OOD_THRESHOLD (default 0.5)
```

**`save(path)` / `load(path)`** — pickle the index to disk.

### Changes to `NearestNeighborIndex` *(updated)*

- **`train_diameter`** is now the **95th percentile of 5-NN distances** in the training set (not the max pairwise distance). The old max was skewed by outlier embeddings; the 95th-percentile of local 5-NN distances is a more stable proxy for "how spread out is the training distribution."
- **SHA-256 checkpoint hash** — the index now stores a 16-character prefix of the SHA-256 hash of the checkpoint file used to build it.
- **`IndexStaleError`** — raised when `NearestNeighborIndex.load(path, expected_checkpoint="<hash>")` finds a hash mismatch. This catches the case where the model was retrained but the index was not rebuilt.
- **`load(path, expected_checkpoint="")`** — new optional parameter. Pass the checkpoint path; the loader computes its hash and compares against the stored value.

---

## Part 11 — The full training flow, end to end

```
python train.py --config config.json
        │
        ├── parse config, set domain + sizes
        ├── instantiate Simulator / FlagSimulator
        │       └── creates EncoderProcesserDecoder
        │               └── creates Encoder, 15× GnBlock, Decoder
        │                       └── each GnBlock has EdgeBlock + NodeBlock
        │                               └── each holds a build_mlp() MLP
        │
        ├── load dataset (FpcDataset or FlagDataset)
        │       └── opens .dat memmap, loads _meta.npz
        │
        ├── training loop:
        │   for each batch:
        │       if CFD: transformer(graph)   → edge_index + edge_attr
        │       get_velocity_noise(graph)    → noise [N, 2]
        │       simulator(graph, noise)      → (pred_norm, target_norm)
        │           └── update_node_attr()  → normalised node features
        │           └── edge_normalizer()   → normalised edge features
        │           └── model(graph)        → EncoderProcesserDecoder forward
        │               └── Encoder         → [N, 128], [E, 128]
        │               └── 15× GnBlock     → refine representations
        │                   └── EdgeBlock   → gather + cat + MLP
        │                   └── NodeBlock   → scatter_sum + cat + MLP
        │               └── Decoder         → [N, output_size]
        │           └── _output_normalizer  → normalised target
        │       compute loss, backward, step
        │
        └── after training:
            extract_embedding() for all training samples
            NearestNeighborIndex.build()
            save index to disk
```

---

## Part 12 — The extensions: `extensions/generative/`

These files implement PhysicsAI Generate — the inverse design system.

| File | What it does |
|---|---|
| `shape_extractor.py` | Reads 1,000 CFD trajectories, fits circles to WALL_BOUNDARY nodes → extracts (cx, cy, r, v_inlet) |
| `drag_surrogate.py` | 3-layer MLP trained on (cx, cy, r, v_inlet) → drag proxy. ~1000× faster than full simulator rollout |
| `cvae_cfd.py` | CFD CVAE: encoder [params, drag] → (μ[16], σ[16]); decoder [z, target_drag] → params |
| `cvae_cloth.py` | Cloth CVAE: GNN encoder → (μ[32], σ[32]); decoder → Gumbel-softmax → soft HANDLE mask |
| `mesh_generator.py` | `params_to_graph(cx,cy,r,v_in)` → PyG Data (Delaunay triangulation); `mask_to_graph(mask)` → cloth Data |
| `inverse_design.py` | Cloth gradient descent: `z ← z - α·∂stress_loss/∂z`, 50–100 steps |
| `train_cvae.py` | Training script for CVAEs |

### Changes to `cvae_cfd.py` *(updated)*

- **`free_bits = 0.05`** — KL free-bits regularisation. The KL term is clamped to a minimum of 0.05 nats per dimension. This prevents **posterior collapse** (the pathology where the encoder ignores its input and outputs the prior, making the latent space useless).
- **Latin Hypercube Sampling (LHS)** replaces random Normal sampling for the latent `z` at generation time. LHS tiles the latent space more evenly, so 10 generated candidates cover more of the design space than 10 independent random draws.
- A **`scipy` guard** at module level catches import failure and raises a clear message, since `scipy.stats.qmc` is needed for LHS.

### Changes to `cvae_cloth.py` *(updated)*

- Same `free_bits = 0.05` and LHS changes as `cvae_cfd.py`.

### Changes to `mesh_generator.py` *(updated)*

- **`RealMeshLookup`** — new class. Builds a KDTree over the parameter vectors `(cx, cy, r, v_inlet)` of all 1,000 training meshes. `snap(params)` returns the nearest real training mesh's parameters, guaranteeing the generated design corresponds to an actual mesh in the dataset (in-distribution guarantee). Used by `inverse_design.py` before every gradient rollout.

### Changes to `inverse_design.py` *(updated)*

- **K=5 BPTT through GNN** — backpropagation is now unrolled through **5 rollout steps** (was 1). Gradients flow through 5 sequential GNN forward passes, giving the optimiser information about how the design affects the medium-term flow trajectory, not just the first step.
- **`RealMeshLookup.snap()`** is called on the decoded `params` before the gradient rollout, ensuring the GNN always sees a valid in-distribution mesh.

### New: `extensions/confidence/ood_detector.py` — `ParamSpaceOOD` *(added)*

Complements the embedding-space `OODDetector` with a **parameter-space OOD check** for the Generate page.

```python
class ParamSpaceOOD:
    def __init__(self, train_params: np.ndarray):
        # train_params: [N_train, 4] — (cx, cy, r, v_inlet) for each training mesh
        self.tree = KDTree(train_params)
        self.threshold = ...   # 95th-percentile 5-NN distance

    def score(self, params: np.ndarray) -> float:
        # returns confidence in [0, 1]; <0.5 → OOD
```

Used on the Generate page to flag candidates whose design parameters are far from the training distribution, independently of what the GNN embedding says.

---

## Part 12b — Physics correction: `physics/poisson_pressure.py`

### `PoissonPressureCorrector`

Enforces the **incompressibility constraint** (∇·u = 0) on the GNN's predicted velocity field by solving a Poisson pressure-correction equation on the mesh graph. Sparse LU factorisation happens once per mesh; thereafter every rollout step is an O(N) triangular solve.

**`_build_knn_edges(pos) → (row, col)`**

Builds a sparse connectivity graph for the pressure Laplacian. For each node, connects to its *k*-nearest neighbours (default k=6) using a KDTree over node positions. Returns COO-format row/col index arrays.

**`_build_laplacian(pos, edges) → scipy.sparse.csc_matrix`**

Assembles the weighted Laplacian matrix:

```
w_ij = 1 / ||pos_i - pos_j||²    (inverse-distance-squared weights)
L_ii = -∑_j w_ij
L_ij = w_ij   for i ≠ j
```

Then applies the **Dirichlet boundary condition** at node 0 (pressure reference): row 0 and column 0 are zeroed and the diagonal set to 1, fixing p[0] = 0. Returns a CSC sparse matrix (column-sparse format, optimal for LU factorisation).

The sparse LU is computed once (`scipy.sparse.linalg.splu`) and stored as `self._lu`. This is the "factor once" step — subsequent calls to `correct()` only do forward/back substitution.

**`_compute_divergence(vel) → np.ndarray [N]`**

Computes ∇·u at each node from the velocity field `vel [N, 2]`.

Uses **batched 2×2 normal equations**: for each node *i* and its k neighbours, fits a local linear velocity field by solving the least-squares system:

```
[Δx_ij, Δy_ij] · [∂u/∂x, ∂u/∂y]^T = Δu_ij    for each neighbour j
```

via the 2×2 normal equations `(AᵀA)⁻¹Aᵀb`. The divergence is then `∂u/∂x + ∂v/∂y`. All nodes are processed in a single batched matrix operation — no Python loop over nodes.

**`_compute_gradient(phi) → np.ndarray [N, 2]`**

Same local least-squares approach as `_compute_divergence` but recovers the gradient `[∂φ/∂x, ∂φ/∂y]` of a scalar field `phi [N]`.

**`correct(vel) → np.ndarray [N, 2]`**

The main per-step call. Applies one pressure-correction step:

```
1. div_u   = _compute_divergence(vel)       # how much each node violates ∇·u=0
2. phi     = self._lu.solve(div_u)          # solve L·φ = ∇·u  (O(N) LU back-solve)
3. grad_phi = _compute_gradient(phi)        # ∇φ
4. return vel - grad_phi                    # project onto divergence-free subspace
```

Step 2 reuses the pre-factored LU — this is why the cost is O(N) after the first call.

**`correct_series(vel_series) → np.ndarray [T, N, 2]`**

Applies `correct()` to every frame in a rollout `vel_series [T, N, 2]` and returns the corrected series. Called from `rollout.py` when `--poisson_correction` is set.

**`divergence_rms(vel) → float`**

Diagnostic: returns the RMS divergence of `vel`. Use this to verify how much correction was applied. Typical values: ~0.01–0.05 before correction, ~1e-6 after.

**Performance summary:**
- `_build_laplacian` + `splu` factorisation: ~200 ms for N=1,800 nodes (done once).
- `correct()` per frame: ~0.3 ms (LU back-solve only).
- For a 600-step rollout: factor once, solve 600×. Total overhead: ~380 ms vs ~120 s for the rollout itself.

---

## Part 12c — SSH dispatch: `rollout_ssh.py` and `generate_ssh.py`

### `rollout_ssh.py` — SSH dispatch for rollout

Used when the API server runs on a CPU machine and GPU inference should run on a remote node. The API route calls this module instead of running `rollout.py` directly.

**Flow:**
1. Write the rollout config to a temporary JSON file.
2. SSH into the remote GPU node and launch `rollout.py` with the config path and `--poisson_correction` flag (if set).
3. Stream `stdout` from the remote process back to the browser as **Server-Sent Events (SSE)** — each line of output becomes an SSE `data:` field.
4. After the rollout completes, `scp` the result `.pkl` file back to the local `runs/` directory.
5. The `poisson_correction` flag is read from the config JSON and forwarded to the remote `rollout.py` invocation.

### `generate_ssh.py` — SSH dispatch for generate

Same pattern as `rollout_ssh.py` but for the CVAE inverse design pipeline.

**Named SSE events:**
- `event: candidate` — fired each time a new design candidate is ready (streams the candidate card data as JSON in the `data:` field).
- `event: done` — fired once when all candidates have been generated.

This allows the browser to progressively render candidate cards as they arrive, rather than waiting for the full batch.

---

## Part 12d — Storage layer: `storage/`

Pluggable result storage with three backends and a common Protocol.

### `storage/protocols.py` — `ResultRepository`

```python
@runtime_checkable
class ResultRepository(Protocol):
    def save(self, key: str, data: dict) -> None: ...
    def load(self, key: str) -> dict: ...
    def load_timestep(self, key: str, t: int) -> np.ndarray: ...
    def list(self) -> list[str]: ...
    def exists(self, key: str) -> bool: ...
    def delete(self, key: str) -> None: ...
    def get_path(self, key: str) -> Path: ...
```

`@runtime_checkable` means `isinstance(repo, ResultRepository)` works at runtime — useful for API-level validation.

`load_timestep(key, t)` is the key operation for the UI: it loads a **single frame** without deserialising the entire file. HDF5 and Zarr backends make this O(1); the PKL backend falls back to loading everything.

---

### `storage/pkl_repository.py` — `PklResultRepository`

Legacy pickle-based backend. Each rollout is stored as `runs/<key>.pkl`. Kept for backward compatibility with existing saved rollouts.

`load_timestep` loads and deserialises the full file, then indexes into it — no partial-read optimisation.

---

### `storage/hdf5_repository.py` — `HDF5ResultRepository`

HDF5-based backend using `h5py`.

- **Compression**: `gzip` level 4.
- **Chunk shape**: `(1, N, D)` — one timestep at a time. This means `load_timestep(key, t)` reads exactly one chunk from disk, making random-access frame fetching O(1) regardless of rollout length.
- Each rollout is stored as `runs/<key>.h5` with datasets `velocity [T, N, 2]`, `pressure [T, N, 1]`, `node_type [N]`, and a `metadata` JSON string attribute.

---

### `storage/factory.py` — `StorageFactory`

```python
class StorageFactory:
    @staticmethod
    def create() -> ResultRepository:
        config = json.load(open('runs/storage_config.json'))
        if config['backend'] == 'hdf5':
            return HDF5ResultRepository(config.get('root', 'runs'))
        elif config['backend'] == 'zarr':
            return ZarrArchive(config.get('root', 'runs'))
        else:
            return PklResultRepository(config.get('root', 'runs'))
```

Reads `runs/storage_config.json` to choose the backend. If the file doesn't exist, defaults to `PklResultRepository`. All API routes call `StorageFactory.create()` to get the active backend — swapping storage requires only a config change.

---

### `storage/zarr_archive.py` — `ZarrArchive`

Zarr-based cloud-native backend.

- **Compression**: Blosc/LZ4 (faster than gzip, similar ratio).
- **Sentinel files**: each completed rollout creates a `<key>.zarr.ok` file. If `.ok` is absent, the archive is treated as incomplete/corrupt.
- **Chunking**: same `(1, N, D)` scheme as HDF5 for efficient partial reads.
- Compatible with cloud object stores (S3, GCS) via `fsspec` — just change the `root` path to `s3://bucket/prefix`.

---

## Part 12e — Ingest pipeline: `ingest/`

Converts raw solver output into the `.dat` / `_meta.npz` format consumed by `FpcDataset` and `FlagDataset`. Designed to support multiple solver formats via the `SolverAdapter` Protocol.

### `ingest/protocols.py` — `SolverAdapter`

```python
class SolverAdapter(Protocol):
    def list_splits(self) -> list[str]: ...       # ['train', 'valid', 'test']
    def load_split(self, split: str) -> ...: ...  # yields trajectory dicts
    def source_path(self) -> Path: ...            # where raw data lives
    def name(self) -> str: ...                    # 'tfrecord' | 'openfoam' | ...
```

---

### `ingest/adapters/tfrecord.py` — `TFRecordAdapter`

Wraps the existing `.dat` memmap files produced by `parse_tfrecord.py`. Implements `SolverAdapter` so that already-converted data can pass through the ingest pipeline for re-normalisation or format conversion without re-parsing TFRecords.

### `ingest/adapters/openfoam.py` — `OpenFOAMAdapter`

Stub. All methods raise `NotImplementedError`. Reserved for future OpenFOAM case directory ingestion.

---

### `ingest/pipeline.py` — `IngestPipeline`

**`run(adapter: SolverAdapter) → None`**

Executes 5 sequential stages:

| Stage | File | What it does |
|---|---|---|
| 1. Harvest | `ingest/stages/harvest.py` | Calls `adapter.load_split()` for each split; collects raw trajectory dicts |
| 2. Validate | `ingest/stages/validate.py` | Checks shapes, dtypes, node-count consistency, no NaN/Inf values |
| 3. Normalise | `ingest/stages/normalise.py` | Computes dataset-level mean/std; applies to all fields |
| 4. Write | `ingest/stages/write.py` | Writes `.dat` memmap files and `_meta.npz` index files |
| 5. Index | `ingest/stages/index.py` | Builds the flat `(traj_idx, timestep)` index used by `__getitem__` |

Each stage is a standalone module with a single `run(context)` function where `context` is a shared dict passed through the pipeline. Stages are composable — you can run a subset by constructing the pipeline manually.

---

## Part 12f — Migration and maintenance scripts: `scripts/`

### `scripts/migrate_pkl_to_hdf5.py`

CLI tool to convert legacy `.pkl` rollout files to the HDF5 backend.

```
python scripts/migrate_pkl_to_hdf5.py [--dry-run] [--delete-pkl]
```

- **`--dry-run`** — lists all `.pkl` files that would be migrated without writing anything. Safe to run at any time.
- **`--delete-pkl`** — after successful HDF5 write and verification (re-loads and compares checksums), deletes the source `.pkl`. Without this flag, `.pkl` files are preserved.
- Processes files one at a time and logs progress to stdout (compatible with SSE streaming if called via API).

### `scripts/regenerate_dat.py`

Scaffold for rebuilding `.dat` memmap files from a Zarr archive. Useful if the memmap index is corrupted or a new split needs to be extracted.

Currently implements the read-from-Zarr path; the write-to-dat path raises `NotImplementedError` pending the Zarr archive schema being finalised.

---

## Part 12g — Result retention: `result/retention.py`

```
python -m result.retention --keep 10 [--dry-run]
```

Enforces a maximum number of saved rollout results. Deletes the oldest results (by `mtime`) beyond the `--keep` limit, using whichever storage backend `StorageFactory.create()` returns.

- **`--dry-run`** — prints which files would be deleted without deleting them.
- Without `--dry-run`, calls `repository.delete(key)` for each expired result.
- Intended to be run as a cron job or post-rollout hook to prevent unbounded disk usage.

---

## Part 12h — Container infrastructure: Docker

### `Dockerfile.api`

Multi-stage build producing a minimal CPU-capable API image:

- **Base**: `python:3.12-slim`
- **PyTorch**: `2.1.0+cpu` wheel (no CUDA — GPU inference is dispatched via SSH to a separate node)
- Installs `requirements.txt`, copies source, sets `CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]`

### `Dockerfile.frontend`

Two-stage build:

- **Stage 1** (`node:20-alpine`): `npm ci && npm run build` — produces the static React bundle in `/app/dist`
- **Stage 2** (`nginx:alpine`): copies `/app/dist` into `/usr/share/nginx/html` and drops in `docker/nginx.conf`

### `docker/nginx.conf`

Nginx configuration tuned for **SSE (Server-Sent Events)**:

```nginx
proxy_buffering            off;          # disable response buffering — required for SSE
chunked_transfer_encoding  on;           # allow chunked encoding for streaming
proxy_read_timeout         3600s;        # hold connection open for up to 1 hour
                                         # (rollout + generate can take minutes)
proxy_pass                 http://api:8000;
```

Without `proxy_buffering off`, Nginx would buffer the entire SSE stream and the browser would see nothing until the rollout finished.

### `docker-compose.yml`

Three services:

| Service | Image | Profile | Description |
|---|---|---|---|
| `api` | `Dockerfile.api` | *(always)* | FastAPI backend on port 8000 |
| `frontend` | `Dockerfile.frontend` | *(always)* | Nginx serving React bundle on port 80 |
| `frontend-dev` | `node:20-alpine` | `dev` | Vite dev server with HMR on port 5173 |

The `frontend-dev` service is only started when `--profile dev` is passed:

```
docker compose --profile dev up
```

This lets developers run `docker compose up` for a production-like stack, or add `--profile dev` for live-reload during UI development.

---

## Part 13 — API layer: `api/`

| File | What it does |
|---|---|
| `api/state.py` | Module-level model cache; `get_model(domain, device)` loads checkpoint once |
| `api/routes/train.py` | `POST /api/train/start` — launches `train.py` as subprocess; streams stdout |
| `api/routes/rollout.py` | `POST /api/rollout` — runs inference; streams frames as SSE events |
| `api/routes/generate.py` | `POST /api/generate` — CVAE inverse design; streams candidate cards as SSE |
| `api/routes/results.py` | `GET /api/results/{filename}` — serves saved rollout .pkl files |
| `api/routes/dataset.py` | `GET /api/dataset/samples` — returns sample graphs for Dataset Studio UI |
| `api/routes/status.py` | `GET /api/status/gpu` — GPU memory info |

The generate route uses a **Strategy pattern**:
```python
_DOMAIN_SAMPLERS = {
    'cylinder_flow': CFDDesignSampler,
    'flag_simple':   ClothDesignSampler,
}
sampler = _DOMAIN_SAMPLERS[req.domain]()
results = await loop.run_in_executor(None, lambda: sampler.sample(...))
```
ML inference runs in a thread pool (`run_in_executor`) so it doesn't block the async event loop.

### Route map (updated)

| Method | Path | Handler file | Description |
|---|---|---|---|
| POST | `/api/train/start` | `routes/train.py` | Launch `train.py` subprocess; stream stdout |
| POST | `/api/rollout` | `routes/rollout.py` | Run 600-step inference; stream SSE frames |
| GET | `/api/rollout/status` | `routes/rollout.py` | Return rollout progress state dict |
| GET | `/api/checkpoints` | `routes/rollout.py` | Return `arch_summary` per architecture |
| POST | `/api/generate` | `routes/generate.py` | CVAE inverse design; stream SSE candidates |
| GET | `/api/results/{filename}` | `routes/results.py` | Serve saved rollout `.pkl` files |
| GET | `/api/dataset/samples` | `routes/dataset.py` | Sample graphs for Dataset Studio UI |
| GET | `/api/dataset/mesh_preview` | `routes/dataset.py` | Mesh geometry preview for UI |
| GET | `/api/dataset/node_type_counts` | `routes/dataset.py` | Per-node-type counts for UI |
| GET | `/api/status/gpu` | `routes/status.py` | GPU memory info |
| GET/POST | `/api/storage/*` | `routes/storage.py` | Storage backend CRUD (new) |

### Changes to `api/routes/rollout.py` *(updated)*

- **`RolloutRequest.poisson_correction: bool = False`** — new field. When `True`, `PoissonPressureCorrector` is instantiated from `physics/poisson_pressure.py` and wired **into the rollout loop** (not applied post-hoc). The corrector is factored once on the first frame (Laplacian sparse LU), then applied at every step.
- **`GET /api/checkpoints`** returns an `arch_summary` dict:
  ```json
  {
    "gn":   {"best_epoch": 42, "best_val_loss": 0.0031, "path": "runs/best_gn.pt"},
    "tns":  {"best_epoch": 38, "best_val_loss": 0.0028, "path": "runs/best_tns.pt"},
    "sage": {"best_epoch": 45, "best_val_loss": 0.0033, "path": "runs/best_sage.pt"}
  }
  ```
- **`GET /api/rollout/status`** returns the current rollout progress as a state dict (step, total_steps, domain, etc.).
- **`IndexStaleError` handling** — if the confidence index hash doesn't match the loaded checkpoint, the rollout response includes a `"staleness_warning"` key instead of raising an HTTP error.
- `"poisson_correction"` key is saved into the rollout PKL metadata.

### Changes to `api/routes/generate.py` *(updated)*

- **SSH dispatch branch** — if `runs/ssh_config.json` exists, the generate route writes a temporary config JSON, calls `generate_ssh.py` on the remote GPU via SSH, and streams the SSE response back to the browser. Named SSE events are used: `event: candidate` for each design card, `event: done` when complete.

### Changes to `api/routes/dataset.py` *(updated)*

- **`GET /api/dataset/mesh_preview`** — returns mesh node positions and edge connectivity for rendering in the Dataset Studio UI.
- **`GET /api/dataset/node_type_counts`** — returns a dict mapping node type name → count for the requested split.

---

## Part 14 — File index (every Python file, one line each)

| File | One-line description |
|---|---|
| `train.py` | Entry point: load data, train model, save checkpoint, build confidence index |
| `rollout.py` | Entry point: load checkpoint, run 600-step autoregressive inference, save result; `--poisson_correction` flag wires in Poisson pressure correction per step |
| `rollout_ssh.py` | SSH dispatch for rollout: writes config JSON, SSH-launches `rollout.py` on remote GPU, streams SSE back, scp results |
| `generate_ssh.py` | SSH dispatch for generate: same SSH pattern; uses named SSE events (`event: candidate`, `event: done`) |
| `render_results.py` | Load result file, render velocity contour frames with matplotlib, encode MP4 |
| `parse_tfrecord.py` | One-time conversion: TFRecord → .dat memmap + _meta.npz |
| `data/parse_flag_tfrecord.py` | One-time conversion for cloth TFRecords |
| `model/blocks.py` | `EdgeBlock`, `NodeBlock` — atomic GNN operations |
| `model/model.py` | `Encoder`, `GnBlock`, `TNSBlock`, `SAGEBlock`, `Decoder`, `EncoderProcesserDecoder` |
| `model/simulator.py` | `Simulator` — CFD wrapper; `architecture='gn'/'tns'/'sage'` param; arch stored in checkpoint |
| `model/flag_simulator.py` | `FlagSimulator` — cloth wrapper with Verlet integration, HANDLE pinning |
| `model/embedding.py` | `extract_embedding()` — 256-dim dual-frame CFD or 128-dim single-frame cloth; all-node pooling for CFD |
| `utils/normalization.py` | `Normalizer` — online running-statistics normaliser |
| `utils/noise.py` | `get_velocity_noise()` — Gaussian noise on NORMAL nodes only |
| `utils/utils.py` | `NodeType` enum |
| `dataset/fpc.py` | `FpcDataset` — CFD lazy memmap loader |
| `dataset/flag_dataset.py` | `FlagDataset` — cloth lazy loader |
| `confidence/index.py` | `NearestNeighborIndex` — KDTree OOD detector; SHA-256 checkpoint hash; `IndexStaleError`; 95th-pct 5-NN train_diameter |
| `confidence/benchmark.py` | Benchmarks KDTree backends |
| `confidence/build_index.py` | CLI to build confidence index; CFD uses dual-frame per trajectory; stores checkpoint_hash |
| `physics/poisson_pressure.py` | `PoissonPressureCorrector`: sparse LU Laplacian (inverse-dist-squared, Dirichlet BC node 0), `_compute_divergence` (batched 2×2 normal eqs), `correct(vel)`, `correct_series`, `divergence_rms` |
| `extensions/generative/shape_extractor.py` | Extract (cx,cy,r,v_inlet) from CFD trajectories |
| `extensions/generative/drag_surrogate.py` | Fast MLP: params → drag proxy |
| `extensions/generative/cvae_cfd.py` | CFD CVAE; free_bits=0.05; LHS latent sampling; scipy guard |
| `extensions/generative/cvae_cloth.py` | Cloth CVAE; free_bits=0.05; LHS latent sampling |
| `extensions/generative/mesh_generator.py` | CVAE params → PyG Data; `RealMeshLookup` snaps to nearest training mesh |
| `extensions/generative/inverse_design.py` | Cloth latent gradient descent; K=5 BPTT through GNN; uses `RealMeshLookup` |
| `extensions/generative/train_cvae.py` | CVAE training entry point |
| `extensions/confidence/ood_detector.py` | `OODDetector` (embedding-space) + `ParamSpaceOOD` (design-parameter KDTree, used on Generate page) |
| `storage/protocols.py` | `ResultRepository` Protocol (`@runtime_checkable`): save, load, load_timestep, list, exists, delete, get_path |
| `storage/pkl_repository.py` | `PklResultRepository` — legacy pickle files |
| `storage/hdf5_repository.py` | `HDF5ResultRepository` — gzip-4 compressed, chunks=(1,N,D) for O(1) frame reads |
| `storage/factory.py` | `StorageFactory.create()` — reads `runs/storage_config.json`, returns PKL/HDF5/Zarr backend |
| `storage/zarr_archive.py` | `ZarrArchive` — Blosc/LZ4 compression, `.zarr.ok` sentinels, cloud-native chunked storage |
| `ingest/protocols.py` | `SolverAdapter` Protocol: list_splits, load_split, source_path, name |
| `ingest/adapters/tfrecord.py` | `TFRecordAdapter` — wraps existing `.dat` memmap files |
| `ingest/adapters/openfoam.py` | `OpenFOAMAdapter` stub — all methods raise NotImplementedError |
| `ingest/pipeline.py` | `IngestPipeline.run()` — 5 stages: harvest → validate → normalise → write → index |
| `ingest/stages/harvest.py` | Stage 1: collect raw trajectory dicts from adapter |
| `ingest/stages/validate.py` | Stage 2: shape/dtype/NaN checks |
| `ingest/stages/normalise.py` | Stage 3: dataset-level mean/std normalisation |
| `ingest/stages/write.py` | Stage 4: write `.dat` memmap + `_meta.npz` |
| `ingest/stages/index.py` | Stage 5: build flat (traj_idx, timestep) index |
| `scripts/migrate_pkl_to_hdf5.py` | Migration tool: `--dry-run` lists candidates; `--delete-pkl` removes originals after verified HDF5 write |
| `scripts/regenerate_dat.py` | Scaffold for rebuilding `.dat` from Zarr archive |
| `result/retention.py` | `python -m result.retention --keep 10 [--dry-run]` — deletes oldest results beyond keep limit |
| `api/main.py` | FastAPI app, router registration |
| `api/state.py` | Model cache: `get_model(domain, device)` |
| `api/routes/train.py` | Train start/status/stop endpoints |
| `api/routes/rollout.py` | Rollout SSE endpoint; `/checkpoints` arch_summary; `/status` progress dict; Poisson correction; IndexStaleError warning |
| `api/routes/generate.py` | Generate SSE endpoint; SSH dispatch branch |
| `api/routes/results.py` | Results file serving |
| `api/routes/dataset.py` | Dataset samples, mesh_preview, node_type_counts endpoints |
| `api/routes/status.py` | GPU status endpoint |
| `train_ddp.py` | Multi-GPU DDP training entry point |
| `Dockerfile.api` | Python 3.12-slim, CPU PyTorch 2.1.0 |
| `Dockerfile.frontend` | Multi-stage: node:20-alpine build → nginx:alpine serve |
| `docker/nginx.conf` | SSE support: proxy_buffering off, chunked_transfer_encoding on, proxy_read_timeout 3600s |
| `docker-compose.yml` | api + frontend + frontend-dev (--profile dev) |

---

## See also

- [[SUBSYSTEM_PREDICTOR]] — deep-dive into the GNN architecture and physics
- [[SUBSYSTEM_DATA]] — deep-dive into data loading and the memmap pipeline
- [[SUBSYSTEM_GENERATOR]] — deep-dive into the CVAE inverse design system
- [[SUBSYSTEM_CONFIDENCE]] — deep-dive into OOD detection and embeddings
- [[SUBSYSTEM_API]] — deep-dive into the FastAPI backend and SSE streaming
- [[SUBSYSTEM_STORAGE]] — deep-dive into the pluggable storage backends (PKL/HDF5/Zarr)
- [[SUBSYSTEM_INGEST]] — deep-dive into the ingest pipeline and SolverAdapter protocol
- [[SUBSYSTEM_PHYSICS]] — deep-dive into the Poisson pressure corrector
- [[PIPELINE_DAG]] — tensor shapes at every stage of the pipeline
