# Spec: flag_simple Domain

Date: 2026-04-07  
Status: Draft

---

## 1. Overview

Add `flag_simple` (3D cloth simulation) as a second domain alongside `cylinder_flow`.
Follows DeepMind's cloth model exactly — ported to PyTorch/PyG the same way cylinder_flow was.

The cloth model uses **Verlet integration** (position-based, second-order) instead of cylinder_flow's velocity update.
All domain differences are isolated: a new dataset class, a new parse script, config-driven model construction, and a new rollout function. The core `EncoderProcesserDecoder` architecture is unchanged except `output_size` becomes a constructor parameter.

---

## 2. Architecture Verification

### Confirmed match with DeepMind cloth_model.py

| Component | DeepMind | Our implementation |
|---|---|---|
| Node features | `(world_pos - prev_world_pos)[3] + one_hot(node_type,9)[9]` = **12** | computed dynamically from data |
| Edge features | `rel_mesh_pos[2] + norm[1] + rel_world_pos[3] + norm[1]` = **7** | computed dynamically from data |
| Decoder output | **3** (3D acceleration) | `output_size` param |
| Integration | `pos_next = 2*pos_cur - pos_prev + acc` (Verlet) | new `FlagSimulator` class |
| Loss target | `target_acc = target_world_pos - 2*cur_world_pos + prev_world_pos` | matches DeepMind exactly |
| Loss mask | `NodeType.NORMAL` only | same as cylinder_flow mask logic |
| Noise | `0.003` on `world_pos` | config-driven |
| Batch size | `1` | config-driven |
| History | `prev|world_pos` required at every step | stored in dataset item |

### No hardcoded sizes
`node_input_size` and `edge_input_size` are computed at dataset construction time and passed to `Simulator` via config. They are NOT hardcoded in model code.

---

## 3. Files Changed / Created

### New files
| File | Purpose |
|---|---|
| `data/parse_flag_tfrecord.py` | Parse flag_simple TFRecords → `.npz` + `.dat` files |
| `dataset/flag_dataset.py` | `FlagDataset(PyG Dataset)` — loads parsed flag data, returns per-timestep graph |
| `model/flag_simulator.py` | `FlagSimulator` — wraps `EncoderProcesserDecoder`, handles Verlet integration, cloth-specific normalizers |

### Modified files
| File | Change |
|---|---|
| `model/model.py` | `Decoder output_size=2` → constructor param `output_size`; `EncoderProcesserDecoder` passes it through |
| `train.py` | Domain-aware: reads `output_size`, `node_input_size`, `edge_input_size`, `noise_field`, `history` from config — instantiates correct `Simulator` subclass |
| `rollout.py` | Domain-aware rollout: CFD uses velocity update, cloth uses Verlet integration |
| `api/state.py` | `flag_simple` entry: set `available: True`, add correct `node_input`, `edge_input`, `output_size` fields |
| `api/routes/train.py` | Pass `output_size` in train config JSON |

### UI changes
| File | Change |
|---|---|
| `app/src/pages/Train.tsx` | Domain selector already exists — enable flag_simple option when data present |
| `app/src/pages/Predict.tsx` | Domain selector for rollout — pass domain to API |
| `app/src/pages/Visualize.tsx` | 3D→2D projection for cloth mesh (project world_pos XY plane); show position error instead of velocity error |
| `app/src/pages/Dashboard.tsx` | Show flag_simple status in domain cards |

---

## 4. Dataset

### TFRecord fields (from DeepMind meta.json)
| Field | Type | Shape | Used |
|---|---|---|---|
| `world_pos` | dynamic | `[T, N, 3]` | ✅ current + target positions |
| `mesh_pos` | static | `[1, N, 2]` | ✅ rest-configuration 2D coords |
| `node_type` | static | `[1, N, 1]` | ✅ one-hot encoding |
| `cells` | static | `[1, F, 3]` | ✅ triangle connectivity |

No pressure field in cloth domain.

### `parse_flag_tfrecord.py`
Reads `data_flag/train.tfrecord`, `valid.tfrecord`, `test.tfrecord`.
Saves per-trajectory:
- `data_flag/train_pos.npz` — `world_pos [T, N, 3]` per trajectory
- `data_flag/train_mesh.npz` — `mesh_pos [N, 2]`, `node_type [N, 1]`, `cells [F, 3]` (static, one per trajectory)
- Same for valid/test

### `FlagDataset.__getitem__(index)`
Each index maps to one timestep pair `(t, t+1)` within a trajectory.
Returns a PyG `Data` object with:
```
graph.x      = [N, 4]  — concat(world_pos_t[3], node_type[1])  (raw, before feature construction)
graph.prev_x = [N, 3]  — world_pos_{t-1}  (needed for Verlet at t=0: use world_pos_t as prev)
graph.pos    = [N, 2]  — mesh_pos (2D rest configuration)
graph.world_pos = [N, 3]  — world_pos_t (current 3D position)
graph.face   = [3, F]  — triangle connectivity
graph.y      = [N, 3]  — world_pos_{t+1} (target)
```

---

## 5. Model

### `model/model.py` change
```python
# Before
class Decoder(nn.Module):
    def __init__(self, hidden_size=128, output_size=2):

class EncoderProcesserDecoder(nn.Module):
    def __init__(self, ..., hidden_size=128):
        ...
        self.decoder = Decoder(hidden_size=hidden_size, output_size=2)  # hardcoded

# After
class EncoderProcesserDecoder(nn.Module):
    def __init__(self, ..., hidden_size=128, output_size=2):  # param added
        ...
        self.decoder = Decoder(hidden_size=hidden_size, output_size=output_size)
```

### `FlagSimulator` (new, `model/flag_simulator.py`)
Mirrors `Simulator` but:
- `node_input_size = 12` (computed, not hardcoded — verified at construction from dataset)
- `edge_input_size = 7` (computed, not hardcoded)
- `output_size = 3`
- `_output_normalizer` size = 3
- `_node_normalizer` size = 12
- `_edge_normalizer` size = 7

**`update_node_attr`** for cloth:
```python
velocity = world_pos - prev_world_pos        # [N, 3]
one_hot = F.one_hot(node_type, num_classes=9)  # [N, 9]
node_feats = cat([velocity, one_hot], dim=-1)  # [N, 12]
```

**`build_graph`** constructs edges with both mesh-space and world-space features:
```python
rel_mesh = mesh_pos[senders] - mesh_pos[receivers]    # [E, 2]
mesh_norm = norm(rel_mesh, keepdim=True)               # [E, 1]
rel_world = world_pos[senders] - world_pos[receivers]  # [E, 3]
world_norm = norm(rel_world, keepdim=True)             # [E, 1]
edge_attr = cat([rel_mesh, mesh_norm, rel_world, world_norm], dim=-1)  # [E, 7]
```
Note: `FaceToEdge + Cartesian + Distance` transforms are NOT used for cloth — edge features are built explicitly inside `FlagSimulator.forward()` since they depend on both mesh_pos and world_pos.

**Forward pass (training)**:
```python
target_acc = target_world_pos - 2*world_pos + prev_world_pos  # [N, 3] Verlet target
predicted_acc_norm = model(graph)
target_acc_norm = output_normalizer(target_acc, training=True)
# loss on NodeType.NORMAL nodes only
```

**Forward pass (inference)**:
```python
predicted_acc_norm = model(graph)
acc = output_normalizer.inverse(predicted_acc_norm)
next_world_pos = 2*world_pos - prev_world_pos + acc  # Verlet integration
```

---

## 6. Training

`train.py` reads domain from config JSON. If domain is `flag_simple`:
- Instantiates `FlagSimulator` instead of `Simulator`
- Uses `FlagDataset` instead of `FpcDataset`
- Loss = MSE on normalized acceleration, masked to `NodeType.NORMAL`
- Noise applied to `world_pos`: `world_pos_noised = world_pos + noise * 0.003`
- No `T.FaceToEdge` / `T.Cartesian` / `T.Distance` transforms (edge features built inside simulator)

Train config JSON additions:
```json
{
  "domain": "flag_simple",
  "noise_std": 0.003,
  "batch_size": 1,
  "output_size": 3,
  "node_input_size": 12,
  "edge_input_size": 7
}
```
These are not hardcoded — they come from `api/state.py` DOMAINS config when the UI starts training.

---

## 7. Rollout

`rollout.py` dispatches on domain:
```python
if domain == 'flag_simple':
    _rollout_cloth(model, dataset, rollout_index, device)
else:
    _rollout_cfd(model, dataset, rollout_index, device)
```

`_rollout_cloth`:
- Keeps `prev_world_pos` and `cur_world_pos` state
- At each step: build graph → forward (inference) → get `next_world_pos`
- Saves `result_flag{N}.pkl`: `[[predicted_pos[T,N,3], target_pos[T,N,3]], mesh_pos[N,2]]`
- Error metric: position MSE (not velocity RMSE)

---

## 8. Visualization

Cloth is 3D but the existing mesh viewer is 2D. Solution: project `world_pos` onto XY plane for visualization, using mesh_pos as the "rest" reference layout and animating the projected deformation.

UI changes:
- `Visualize.tsx`: detect domain from result metadata, project `world_pos[:, :2]` when domain is cloth
- Error colormap: position error `|pred_pos - target_pos|` per node
- Label: "Position Error (m)" instead of "Velocity Error"

---

## 9. UI Domain Selector

Add a domain selector in the Training page and Predict page. Implementation:
- Training page already has domain selector — `flag_simple` becomes enabled when `data_flag/` exists
- Predict page: domain dropdown → determines which result pkl format and which simulator to load
- The API already has `domain` param on `/api/train/start` and `/api/rollout/start`

---

## 10. Download Instructions

`flag_simple` is not auto-downloaded. User runs:
```bash
bash meshgraphnets/download_dataset.sh flag_simple data_flag
```
(Script to be downloaded from DeepMind repo or documented in README.)
The `api/state.py` `available` flag is set by checking if `data_flag/` exists at startup.

---

## 11. What Does NOT Change

- `EncoderProcesserDecoder` internal architecture (GnBlocks, latent size, residuals) — identical
- `Normalizer` utility — same class, different sizes
- Checkpoint format — same 4 keys, different model weights
- FastAPI endpoint signatures — same, domain passed as param
- `NodeType` enum — same 9 values
