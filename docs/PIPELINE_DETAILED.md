# MeshGraphNets Pipeline — Detailed Code Walkthrough

> Every stage documented with the **actual code from this repo**, ASCII flowcharts, shape annotations, and billion-scale upgrade notes. Use as a narration companion while stepping through the UI.

---

## Stage 1 — Data Ingestion: TFRecord → NumPy

### What format does the raw data arrive in?

DeepMind's open-source physics datasets are distributed as **TFRecords** — a sequential binary format used by TensorFlow. Each record encodes one simulation trajectory as a serialised `tf.Example` with byte-string fields. A companion `meta.json` describes field names, dtypes, and shapes.

```
data/
├── meta.json          ← field schema: names, dtypes, shapes, static vs dynamic
├── train.tfrecord     ← N_train trajectories, binary
├── valid.tfrecord
└── test.tfrecord
```

**meta.json structure (cylinder_flow):**
```json
{
  "field_names": ["cells","mesh_pos","node_type","pressure","velocity"],
  "features": {
    "velocity": { "dtype": "float32", "shape": [-1, 1881, 2], "type": "dynamic" },
    "mesh_pos": { "dtype": "float32", "shape": [1,    1881, 2], "type": "static"  },
    "node_type":{ "dtype": "int32",   "shape": [1,    1881, 1], "type": "static"  },
    "cells":    { "dtype": "int32",   "shape": [1,    3520, 3], "type": "static"  }
  },
  "trajectory_length": 600
}
```

`static` fields are stored once and tiled to match trajectory length. `dynamic` fields vary per timestep.

### Flowchart — CFD TFRecord → memmap + npz

```
 TFRecord (binary, sequential)
         │
         ▼
  tf.data.TFRecordDataset          # one record = one trajectory
         │
         ▼
  _parse(proto, meta)
  ┌─────────────────────────────────────────────────────────┐
  │ for each field in meta["features"]:                     │
  │   data = tf.io.decode_raw(features[key], dtype)        │
  │   data = tf.reshape(data, field["shape"])              │
  │   if static: tile to [T, N, D]                        │
  │   if dynamic: keep as-is                              │
  └─────────────────────────────────────────────────────────┘
         │
         ▼
  Pass 1: count total nodes shape0=ΣN, shape1=max(T)
         │
         ▼
  Allocate memmap:
    fp = np.memmap("train.dat", dtype="float32",
                   mode="w+", shape=(shape0, shape1, 2))
    fp_pressure = np.memmap("train_pressure.dat", dtype="float32",
                            mode="w+", shape=(shape0, shape1, 1))
         │
         ▼
  Pass 2: write each trajectory
  ┌─────────────────────────────────────────────────────────┐
  │ for each trajectory d:                                  │
  │   velocity  = d["velocity"].numpy()     # [T, N, 2]   │
  │   velocity  = velocity.transpose(1,0,2) # → [N, T, 2] │
  │   fp[write_shift : write_shift+N] = velocity           │
  │                                                        │
  │   pressure  = d["pressure"].numpy()     # [T, N, 1]   │
  │   pressure  = pressure.transpose(1,0,2) # → [N, T, 1] │
  │   fp_pressure[write_shift : write_shift+N] = pressure  │
  │                                                        │
  │   collect: pos[N,2], node_type[N,1], cells[F,3]       │
  │   write_shift += N                                     │
  └─────────────────────────────────────────────────────────┘
         │
         ▼
  Build cumulative node index:
    indices = np.cumsum([N₀, N₁, N₂, ...])
    indices = np.insert(indices, 0, 0)   # [0, N₀, N₀+N₁, ...]
         │
         ▼
  np.savez_compressed("train.npz",
    pos=all_pos,           # [total_nodes, 2]
    node_type=...,         # [total_nodes, 1]
    cells=...,             # [total_cells, 3]
    indices=indices,       # [n_traj+1]   ← trajectory boundaries
    cindices=cindices,     # [n_traj+1]   ← cell boundaries
    all_velocity_shape=(shape0, shape1, 2)
  )
```

### The actual code — parse_tfrecord.py

```python
# Pass 1: count total nodes
shape0, shape1 = 0, 0
for d in ds:
    velocity = d['velocity'].numpy()
    velocity = velocity.transpose(1, 0, 2)   # [T,N,2] → [N,T,2]
    N, T, D = velocity.shape
    shape0 += N
    shape1 = max(shape1, T)

# Allocate one big memory-mapped file — never loads fully into RAM
fp = np.memmap("train.dat", dtype='float32', mode='w+',
               shape=(shape0, shape1, 2))

# Pass 2: write each trajectory contiguously
write_shift = 0
for d in ds:
    velocity = d['velocity'].numpy().transpose(1, 0, 2)  # [N, T, 2]
    fp[write_shift : write_shift + velocity.shape[0]] = velocity
    fp.flush()
    write_shift += velocity.shape[0]

# Companion index — tells you where each trajectory lives in the memmap
indices = np.cumsum([pos.shape[0] for pos in all_pos])
indices = np.insert(indices, 0, 0)  # e.g. [0, 1881, 3762, ...]
np.savez_compressed("train.npz",
    pos=all_pos,                              # mesh node positions
    node_type=all_node_type,                  # 0=NORMAL,4=BOUNDARY,etc.
    cells=all_cells,                          # triangular face connectivity
    indices=indices,                          # node offset per trajectory
    all_velocity_shape=(shape0, shape1, 2)    # memmap shape descriptor
)
```

**Why memmap?** `np.memmap` creates a view backed by the OS page cache. Reading `fp[0:1881, :, :]` for trajectory 0 triggers one page-fault range — the OS loads only that region from disk. Nothing is copied into Python heap. For 600 GB of training data, this is the only way to avoid OOM.

---

### Flowchart — Cloth TFRecord → per-trajectory npz

```
 data_flag/train.tfrecord
         │
         ▼
  Load meta.json (flag_simple schema)
  Validate required fields: world_pos, mesh_pos, node_type, cells
         │
         ▼
  for idx, d in enumerate(ds):
  ┌────────────────────────────────────────────────────────────┐
  │ world_pos = d["world_pos"].numpy()   # [T, N, 3]          │
  │ mesh_pos  = d["mesh_pos"].numpy()[0] # [N, 2]  (static)   │
  │ node_type = d["node_type"].numpy()[0]# [N, 1]  (static)   │
  │ cells     = d["cells"].numpy()[0]    # [F, 3]  (static)   │
  │                                                            │
  │ np.savez_compressed(                                       │
  │   f"data_flag/train/traj_{idx:05d}.npz",                  │
  │   world_pos=world_pos.astype(np.float32),  # [T, N, 3]   │
  │   mesh_pos =mesh_pos.astype(np.float32),   # [N, 2]      │
  │   node_type=node_type.astype(np.int32),    # [N, 1]      │
  │   cells    =cells.astype(np.int32),        # [F, 3]      │
  │ )                                                          │
  └────────────────────────────────────────────────────────────┘
         │
         ▼
  np.savez_compressed("data_flag/train_index.npz",
    n_traj        = 1000,
    steps_per_traj= [250, 250, 250, ...]   # T per trajectory
  )
```

**Why per-file for cloth, not memmap?**  
Cloth trajectories have variable `N` (node count) — different cloth resolutions. A flat memmap requires a fixed stride. Since `N` varies, you'd waste huge padding or need a ragged index. Per-file npz with an index sidecar is simpler and still fast — numpy's zip decompression of a 2 MB file is ~10ms.

---

## Stage 2 — How the Stored Data Is Read at Training Time

### CFD — FpcDataset (`dataset/fpc.py`)

```
__getitem__(index):
    index = 1847

    tra_index        = 1847 // 599  = 3     ← trajectory 3
    tra_sample_index = 1847 %  599  = 50    ← timestep 50 within trajectory

    # Lookup node range from index array
    tra_start = indices[3]   = 5643   ← node 5643 in the flat memmap
    tra_end   = indices[4]   = 7524   ← node 7524 (1881 nodes per traj)

    pos       = meta["pos"][5643:7524]        # [1881, 2]  mesh positions
    node_type = meta["node_type"][5643:7524]  # [1881, 1]
    cells     = meta["cells"][c_start:c_end]  # [3520, 3]

    # One page-fault range from disk — zero copy
    velocity_t   = fp[5643:7524, 50]    # [1881, 2]  velocity at t=50
    velocity_tp1 = fp[5643:7524, 51]    # [1881, 2]  velocity at t=51  (target)

    x = np.concatenate([node_type, velocity_t], axis=-1)  # [1881, 3]
    y = velocity_tp1                                        # [1881, 2]

    return Data(x=x, pos=pos, face=cells.T, y=y)
```

**Access pattern:** Index lookup → memmap slice → concatenate → PyG `Data`. No disk seek other than the OS page cache lookup. The `DataLoader` with `num_workers=2` runs this in parallel background processes.

### PyG Transform pipeline (CFD only)

```python
transformer = T.Compose([
    T.FaceToEdge(),       # cells[F,3] → edge_index[2,E] (bidirectional)
    T.Cartesian(norm=False),   # edge_attr = Δpos = pos[src] - pos[dst]  [E, 2]
    T.Distance(norm=False),    # edge_attr = cat([Δpos, ‖Δpos‖])         [E, 3]
])
```

**FaceToEdge detail:**  
A triangular face `(a, b, c)` becomes 6 directed edges: `a→b, b→a, b→c, c→b, a→c, c→a`. This gives the graph bidirectionality for message passing.

After the transform, a CFD graph has:
```
graph.x          [N=1881, 3]   node_type(1) + velocity(2)
graph.pos        [N=1881, 2]   mesh coordinates
graph.edge_index [2,    E]     directed edge pairs
graph.edge_attr  [E,    3]     Δpos(2) + ‖Δpos‖(1)
graph.y          [N=1881, 2]   ground-truth velocity at t+1
```

---

## Stage 3 — Graph Construction & Feature Engineering

### How the Mesh Becomes a Graph

```
Triangular mesh (cells array)         Graph
                                       
  0 ─── 1 ─── 2                    0 ──→ 1 ──→ 2
  │   / │   / │        →           ↑ ↘ ↑ ↘ ↑  
  │  /  │  /  │                    │  ↓ │  ↓ │
  3 ─── 4 ─── 5                    3 ──→ 4 ──→ 5

 cells[i] = [a, b, c]              FaceToEdge:
 (triangles, counter-clockwise)    a→b, b→a, a→c, c→a, b→c, c→b
```

Every triangular face produces 6 directed edges. So a mesh with `F=3520` faces produces `E ≈ 21120` edges (minus boundary deduplication).

### Node Features

**CFD (Simulator):**
```python
# graph.x[:, 0:1] — raw node_type int (0=NORMAL, 1=OBSTACLE, 4=BOUNDARY, ...)
# Converted inside simulator.forward():

node_type = graph.x[:, 0:1]              # [N, 1]
frames    = graph.x[:, 1:3]             # [N, 2]  velocity

one_hot = F.one_hot(node_type.squeeze().long(), num_classes=9)  # [N, 9]
node_feats = torch.cat([frames, one_hot], dim=-1)               # [N, 11]
node_feats = node_normalizer(node_feats)                        # [N, 11] normalised
```

**Cloth (FlagSimulator):**
```python
# Node features encode VELOCITY (difference), not absolute position
velocity  = world_pos_t - world_pos_{t-1}              # [N, 3]  ← Verlet velocity
one_hot   = F.one_hot(node_type, num_classes=9)         # [N, 9]
node_feats = torch.cat([velocity, one_hot], dim=-1)     # [N, 12]
```

> **Key insight — why velocity not position?**  
> Using absolute position would couple the model to the coordinate system — it would fail on meshes shifted or rotated from the training distribution. Relative displacement is translation-invariant.

### Edge Features

**CFD — 3 features:**
```python
# PyTorch Geometric T.Cartesian + T.Distance apply this automatically:
Δpos      = pos[src] - pos[dst]     # [E, 2]  relative displacement
norm      = ‖Δpos‖                  # [E, 1]  displacement magnitude
edge_attr = cat([Δpos, norm])       # [E, 3]
```

**Cloth — 7 features (two coordinate systems):**
```python
# FlagSimulator._build_graph():
rel_world  = world_pos[src] - world_pos[dst]   # [E, 3]  3D world-space
world_norm = ‖rel_world‖                        # [E, 1]
rel_mesh   = mesh_pos[src]  - mesh_pos[dst]    # [E, 2]  2D rest-configuration
mesh_norm  = ‖rel_mesh‖                         # [E, 1]
edge_attr  = cat([rel_world, world_norm, rel_mesh, mesh_norm])  # [E, 7]
```

**Why two coordinate systems for cloth?**  
`rel_world` tells the model how the cloth is currently deformed in 3D space. `rel_mesh` tells it the rest configuration — how far apart nodes are when the cloth is flat. The ratio `‖rel_world‖ / ‖rel_mesh‖` implicitly encodes stretch, which is the primary driver of elastic restoring force.

---

## Stage 4 — The GNN Model

### Architecture Overview

```
Input Graph                    Latent Graph                    Output
                                                               
 node_feats [N, 11]           latent_nodes [N, 128]          predicted_Δ [N, 2]
 edge_attr  [E,  3]   ──→     latent_edges [E, 128]   ──→
 edge_index [2,  E]           edge_index   [2,  E]

         ENCODER                  PROCESSOR (×15)             DECODER
```

### Encoder

```python
class Encoder(nn.Module):
    def __init__(self, edge_input_size=3, node_input_size=11, hidden_size=128):
        self.nb_encoder = build_mlp(node_input_size, 128, 128)  # MLP with LayerNorm
        self.eb_encoder = build_mlp(edge_input_size, 128, 128)

    def forward(self, graph):
        node_ = self.nb_encoder(graph.x)          # [N, 11] → [N, 128]
        edge_ = self.eb_encoder(graph.edge_attr)  # [E,  3] → [E, 128]
        return Data(x=node_, edge_attr=edge_, edge_index=graph.edge_index)
```

`build_mlp` is a 3-layer MLP with ReLU + LayerNorm:
```python
def build_mlp(in_size, hidden_size, out_size, lay_norm=True):
    module = nn.Sequential(
        nn.Linear(in_size,  hidden_size), nn.ReLU(),
        nn.Linear(hidden_size, hidden_size), nn.ReLU(),
        nn.Linear(hidden_size, out_size)
    )
    if lay_norm:
        return nn.Sequential(module, nn.LayerNorm(out_size))
    return module
```

### One GnBlock (Message Passing Round)

```
                    ┌─────────────────────────────────────────────────────┐
                    │  GnBlock (one of 15 identical blocks)               │
                    │                                                      │
  latent_node[N,128]│                EdgeBlock:                           │
  latent_edge[E,128]│  e'ᵢⱼ = MLP( hᵢ ∥ hⱼ ∥ eᵢⱼ )   [E, 384→128]     │
                    │       ↑ sender_i, receiver_j, current_edge         │
                    │                                                      │
                    │  NodeBlock:                                          │
                    │  aggregated_e = Σⱼ e'ᵢⱼ  (sum over incoming edges) │
                    │  h'ᵢ = MLP( hᵢ ∥ aggregated_eᵢ )  [N, 256→128]   │
                    │                                                      │
                    │  Residual:                                           │
                    │  hᵢ += h'ᵢ   ,   eᵢⱼ += e'ᵢⱼ                      │
                    └─────────────────────────────────────────────────────┘
```

```python
class EdgeBlock(nn.Module):
    def forward(self, graph):
        senders   = graph.x[graph.edge_index[0]]   # [E, 128]
        receivers = graph.x[graph.edge_index[1]]   # [E, 128]
        collected = torch.cat([senders, receivers, graph.edge_attr], dim=1)  # [E, 384]
        new_edge  = self.net(collected)             # [E, 128]
        return Data(x=graph.x, edge_attr=new_edge, edge_index=graph.edge_index)

class NodeBlock(nn.Module):
    def forward(self, graph):
        # Sum incoming updated edge messages at each node
        agg = scatter(graph.edge_attr, graph.edge_index[1],
                      dim=0, dim_size=graph.num_nodes, reduce='sum')  # [N, 128]
        collected = torch.cat([graph.x, agg], dim=-1)   # [N, 256]
        new_node  = self.net(collected)                  # [N, 128]
        return Data(x=new_node, edge_attr=graph.edge_attr, edge_index=graph.edge_index)

class GnBlock(nn.Module):
    def forward(self, graph):
        x_old, e_old = graph.x.clone(), graph.edge_attr.clone()
        graph = self.eb_module(graph)   # update edges
        graph = self.nb_module(graph)   # update nodes
        graph.x        = x_old + graph.x          # residual
        graph.edge_attr = e_old + graph.edge_attr  # residual
        return graph
```

After 15 rounds each node has "seen" information from nodes up to 15 hops away — enough to span most cylinder-flow meshes (diameter ~30 hops).

### Decoder

```python
class Decoder(nn.Module):
    def __init__(self, hidden_size=128, output_size=2):
        self.decode_module = build_mlp(128, 128, output_size, lay_norm=False)

    def forward(self, graph):
        return self.decode_module(graph.x)   # [N, 128] → [N, 2]
# Output is normalised Δvelocity; inverse-transform gives next_velocity
```

### Full forward pass — shapes annotated

```
Input:
  graph.x         [1881, 11]   node features (normalised)
  graph.edge_attr [21120, 3]   edge features (normalised)
  graph.edge_index[2, 21120]   directed edge pairs

Encoder:
  latent_nodes    [1881, 128]
  latent_edges    [21120, 128]

Processor (15 rounds):
  latent_nodes    [1881, 128]   ← updated each round, residual connection
  latent_edges    [21120, 128]

Decoder:
  output          [1881, 2]     normalised Δvelocity

Inverse normalise → Δvelocity → v_{t+1} = v_t + Δv
```

---

## Stage 5 — Online Normaliser

```python
class Normalizer(nn.Module):
    """Welford-style online normaliser stored as nn.Module buffers.
    Travels with the checkpoint — no separate stats file needed."""

    def __init__(self, size, max_accumulations=10**6):
        # Registered as buffers → saved in state_dict, moved with .to(device)
        self.register_buffer('_acc_count',       torch.tensor(0.0))
        self.register_buffer('_acc_sum',         torch.zeros(1, size))
        self.register_buffer('_acc_sum_squared', torch.zeros(1, size))

    def forward(self, data, accumulate=True):
        if accumulate and self.training:
            # Welford accumulation: sum and sum-of-squares
            self._acc_sum         += data.sum(dim=0, keepdim=True)
            self._acc_sum_squared += (data**2).sum(dim=0, keepdim=True)
            self._acc_count       += data.shape[0]
        return (data - self._mean()) / self._std_with_epsilon()

    def inverse(self, z):
        return z * self._std_with_epsilon() + self._mean()

    def _mean(self):
        return self._acc_sum / self._acc_count.clamp(min=1)

    def _std_with_epsilon(self):
        var = self._acc_sum_squared / self._acc_count.clamp(min=1) - self._mean()**2
        return torch.sqrt(var.clamp(min=0)).clamp(min=1e-8)
```

**Why store in the model instead of pre-computing?**  
1. The normaliser stats are computed incrementally as batches arrive — no pre-scan of the full dataset required.  
2. `state_dict()` saves them automatically. Load a checkpoint and normalisation is restored exactly.  
3. Works correctly for distributed training (each worker accumulates independently; stats converge by the time they saturate).

---

## Stage 6 — Training Loop

```
for epoch in range(start_epoch, num_epochs + 1):

    ┌── train_one_epoch ──────────────────────────────────────────────┐
    │ for graph in DataLoader(train_dataset, batch_size=20):          │
    │                                                                  │
    │   1. Apply transforms (CFD only):                               │
    │      FaceToEdge → Cartesian → Distance                          │
    │      graph.edge_attr now [E, 3]                                 │
    │                                                                  │
    │   2. Add noise to current field (CFD):                          │
    │      noise = σ * randn_like(velocity)                           │
    │      noised_v = velocity + noise   ← key training trick         │
    │                                                                  │
    │   3. Forward pass:                                               │
    │      predicted_Δ_norm, target_Δ_norm = model(graph, noise)     │
    │                                                                  │
    │   4. Mask to fluid nodes only:                                  │
    │      mask = (node_type == NORMAL) | (node_type == OUTFLOW)      │
    │      errors = (predicted_Δ - target_Δ)²[mask]                  │
    │                                                                  │
    │   5. Loss + backward + grad clip + step:                        │
    │      loss = errors.mean()                                       │
    │      loss.backward()                                            │
    │      torch.nn.utils.clip_grad_norm_(model.params, max_norm=1)  │
    │      optimizer.step()                                           │
    └──────────────────────────────────────────────────────────────────┘

    validate → if val_loss < best: save checkpoint

    checkpoint = {
        'epoch':                epoch,
        'model_state_dict':     simulator.state_dict(),  # includes normalizers!
        'optimizer_state_dict': optimizer.state_dict(),
        'valid_loss':           valid_loss,
        'domain':               domain,
        'target_field':         target_field,
    }
```

**Why noise injection?**  
At rollout time, step `t+1`'s input is the model's own prediction — not clean ground truth. The prediction has small errors. Without noise training, the model is never exposed to these errors and diverges exponentially over hundreds of steps. Adding σ=0.02 noise during training teaches the model to be robust to its own drift.

**Why mask boundary nodes from loss?**  
Boundary nodes are Dirichlet-constrained — their velocity is fixed by the boundary condition. The model doesn't need to predict them; including them in the loss would dilute gradients from the physically interesting fluid interior nodes.

---

## Stage 7 — Autoregressive Rollout (Inference)

### CFD Rollout

```
State at t=0: graph.x = [node_type | v₀]   (ground-truth seed)

for t in range(0, T-1):
    ┌── one step ──────────────────────────────────────────────────────┐
    │ if t > 0:                                                         │
    │   graph.x[:, 1:3] = predicted_v          # swap in own output   │
    │                                                                   │
    │ Apply transforms (FaceToEdge + Cartesian + Distance)             │
    │ model.eval(); predicted_v = model(graph, noise=None)             │
    │   → Encoder: [N,11]→[N,128], [E,3]→[E,128]                     │
    │   → Processor ×15                                                │
    │   → Decoder: [N,128]→[N,2] (normalised Δv)                     │
    │   → inverse_normalise(Δv) + v_t  → v_{t+1}                     │
    │                                                                   │
    │ Pin boundary nodes: predicted_v[boundary_mask] = gt_v[boundary] │
    │                                                                   │
    │ predicteds.append(predicted_v.cpu().numpy())                    │
    └───────────────────────────────────────────────────────────────────┘

result = [np.stack(predicteds), np.stack(targets)]   # [T, N, 2] each
pickle.dump([result, crds], open("result/result0.pkl", "wb"))
```

### Cloth Rollout (Verlet Integration)

```
State at t=0: world_pos₀ = ground truth, prev_world = world_pos₀

for t in range(T):
    ┌── one step ──────────────────────────────────────────────────────┐
    │ if t > 0:                                                         │
    │   graph.world_pos = cur_world   # use model's last prediction    │
    │   graph.prev_x    = prev_world  # one step back                  │
    │                                                                   │
    │ FlagSimulator.forward():                                          │
    │   velocity = world_pos_t - world_pos_{t-1}        # [N, 3]      │
    │   edge_attr = [rel_world|world_norm|rel_mesh|mesh_norm] [E, 7]  │
    │   predicted_acc_norm = GNN(node_feats, edge_attr)  # [N, 3]     │
    │   acc = inverse_normalise(predicted_acc_norm)      # [N, 3]     │
    │                                                                   │
    │   Verlet:                                                         │
    │   next_world = 2·world_pos_t - world_pos_{t-1} + acc            │
    │                                                                   │
    │ Pin HANDLE nodes (corners) back to ground truth                  │
    │ handle_mask = (node_type != NORMAL)                              │
    │ next_world[handle_mask] = gt_next_world[handle_mask]             │
    │                                                                   │
    │ prev_world = world_pos_t                                         │
    │ cur_world  = next_world                                          │
    └───────────────────────────────────────────────────────────────────┘
```

**Why Verlet for cloth but Euler for CFD?**  
Cloth is a second-order system — the flag accelerates (spring restoring force). Verlet is energy-conserving for conservative forces: `x_{t+1} = 2x_t - x_{t-1} + a·dt²`. CFD is first-order (Navier-Stokes in velocity form) — predicting Δv and integrating with Euler is sufficient.

---

## Stage 8 — Confidence / OOD Detection (k-d Tree)

```
TRAINING TIME:
  After training completes:
  1. Run inference on ALL training trajectories
  2. Extract latent embedding per trajectory:
       emb = mean(latent_nodes [N, 128])  → [128]
  3. Collect embeddings_train [N_train, 128]
  4. Build KDTree(embeddings_train)
  5. Compute train_diameter = 95th percentile of 1-NN distances within training set
  6. Save: pickle({"embeddings": ..., "train_diameter": ...})

INFERENCE TIME:
  For a new test graph:
  1. Extract embedding [128]
  2. query = embedding.reshape(1, -1)
  3. d_min = tree.query(query, k=1)[0]   # distance to nearest training point
  4. score = clip(1 - d_min / train_diameter, 0, 1)
     → score ≈ 1.0: well within training distribution
     → score ≈ 0.0: very different from anything seen in training
```

**The actual index code:**
```python
class NearestNeighborIndex:
    def build(self, embeddings):
        self._scipy_tree = KDTree(embeddings)
        # Try compiled C++ backend (pybind11) for 2-4× faster queries
        try:
            from _kdtree import KDTree as CppKDTree
            self._cpp_tree = CppKDTree(embeddings)
            self.backend = "cpp"
        except ImportError:
            self.backend = "scipy"
        # 95th percentile of nearest-neighbor distances within training set
        dists, _ = self._scipy_tree.query(embeddings, k=2)  # k=2 skips self
        self.train_diameter = float(np.percentile(dists[:, 1], 95))

    def query(self, embedding):
        d_min = self._cpp_tree.query(embedding, k=1)[0] if self._cpp_tree \
                else self._scipy_tree.query(embedding, k=1)[0]
        return float(np.clip(1.0 - d_min / (self.train_diameter + 1e-12), 0, 1))
```

---

## About the STL / Point-Cloud Approach Mentioned in Your Question

The approach you referenced — "constructing graphs directly from STL files, generating point clouds on the surface and connecting k-NN" — is a different variant popularised by NVIDIA PhysicsNeMo's DoMINO and X-MeshGraphNet:

**What that approach does:**
1. Load STL (surface tessellation language) — a triangulated surface mesh used in CAD
2. Sample `N` points on the surface using Poisson disk or uniform area-weighted sampling
3. Build a k-NN graph by querying a k-d tree: for each point, connect to its 16 nearest neighbours
4. Run MeshGraphNet on this k-NN graph

**What our system does instead:**
- We receive pre-meshed simulation data from DeepMind's simulator — the mesh topology is embedded in the TFRecord. We use the simulation mesh topology directly (faces → edges via `FaceToEdge`), not a recomputed k-NN graph.
- Our edges carry displacement vectors in **simulation mesh space** (CFD: Δpos 2D, cloth: two coordinate systems). k-NN edges would only carry Euclidean distances, losing the rest-configuration encoding.

**When the STL/k-NN approach is better:**
- Inference on new geometry at deployment time — no meshing software needed, just an STL
- The PhysicsNeMo workflow: upload CAD → sample surface → k-NN graph → GNN → aerodynamic predictions
- This is what makes it "simulator-free at inference": no OpenFOAM or Fluent license needed to mesh a new design

**Trade-off:**
| Approach | Ours (simulation mesh) | STL + k-NN |
|---|---|---|
| Topology source | Simulation mesh (physics-informed) | Geometric proximity only |
| Accuracy | Higher (mesh captures flow features) | Slightly lower |
| Deployment | Needs meshed input | Works from raw CAD |
| Edge features | Mesh-space + world-space | Euclidean distance only |

---

## Quick Reference — Tensor Shapes Throughout the Pipeline

```
Stage          Object              Shape            Notes
─────────────────────────────────────────────────────────────────────────
TFRecord       velocity            [T, N, 2]        T=600, N≈1881 per traj
memmap write   velocity (stored)   [N, T, 2]        transposed for node-first access
memmap read    velocity slice      [N_traj, T, 2]   one memmap range
FpcDataset     graph.x             [N, 3]           node_type(1) + vel(2)
               graph.edge_attr     [E, 3]           after transform: Δpos(2)+‖Δ‖(1)
               graph.y             [N, 2]           target velocity

Encoder out    latent_nodes        [N, 128]
               latent_edges        [E, 128]
Per GnBlock    EdgeBlock input     [E, 384]         sender+receiver+edge
               NodeBlock input     [N, 256]         node+aggregated_edges
Decoder out    predicted_Δ_norm   [N, 2]           normalised
Simulator out  next_velocity       [N, 2]           after inverse normalise + add

Rollout pkl    predicted           [T, N, 2]        stacked over all timesteps
               targets             [T, N, 2]
Confidence     embedding           [128]            mean-pooled latent nodes
               train_diameter      scalar           95th percentile NN dist
               confidence_score    scalar ∈ [0,1]
```
