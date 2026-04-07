# Spec: Pressure Field (cylinder_flow)

Date: 2026-04-07  
Status: Updated

---

## 1. Overview

The `cylinder_flow` TFRecords contain a `pressure` field `[T=600, N, 1]` that is currently
parsed by TensorFlow but silently discarded. DeepMind's own implementation also ignores it.

This spec adds pressure as a **user-selectable target field** — a completely separate,
independent surrogate model trained on pressure instead of velocity. The user chooses
**once at training time** which field to predict. Two separate training runs produce two
separate checkpoints; everything downstream reads the mode from checkpoint metadata.

This mirrors how PhysicsAI Studio works: separate surrogate models per quantity of interest.

---

## 2. Design Principle: Two Independent Single-Task Models

| `target_field` | Input node features | Output | Node input size | Output size |
|---|---|---|---|---|
| `velocity` (default) | v[2] + node_type[9] | v[2] | **11** — exact DeepMind | 2 |
| `pressure` | p[1] + node_type[9] | p[1] | **10** | 1 |

**Why not multi-task (predict both simultaneously)?**
- Multi-task requires balancing two loss scales — needs careful λ tuning
- Feeding predicted pressure back into velocity prediction introduces cross-field error accumulation
- Two clean single-task models are easier to evaluate and explain

**Why not "pressure as input, velocity as output"?**
- At rollout time, ground-truth pressure is not available — you'd need to feed predicted pressure
  back in, which couples the two fields and accumulates error
- If GT pressure is used instead, it leaks future information — artificially inflates accuracy
- This design is not scientifically clean; removed from spec

**Pressure-only model integration:**
Same as velocity — `p_{t+1} = p_t + Δp` where `Δp` is the model's predicted (denormalized) output.
Autoregressive rollout feeds predicted pressure back. Error accumulates in pressure only —
no cross-field coupling.

---

## 3. Confirmed TFRecord Field

From `data/meta.json`:
```json
"pressure": {
  "dtype": "float32",
  "shape": [-1, 1],
  "type": "dynamic"
}
```
Shape per trajectory: `[T=600, N, 1]` — one scalar pressure per node per timestep.
Fully readable by existing `_parse()` in `parse_tfrecord.py` — never extracted or saved.

---

## 4. Files Changed / Created

### Modified files
| File | Change |
|---|---|
| `data/parse_tfrecord.py` | Extract `pressure`, save `{split}_pressure.dat` alongside velocity |
| `dataset/fpc.py` | Load pressure `.dat`; when `target_field=pressure`, swap pressure into `graph.x` and `graph.y` |
| `model/model.py` | `output_size` already made a constructor param (flag_simple spec) — no further change |
| `model/simulator.py` | `target_field`-aware: build node features from pressure or velocity; single output head always |
| `train.py` | Read `target_field` from config; single loss, same structure regardless of field |
| `api/state.py` | Add `target_field` to DOMAINS config for cylinder_flow |
| `api/routes/rollout.py` | Domain-aware rollout: use `target_field` from checkpoint metadata |
| `api/routes/results.py` | Return `target_field` in result metadata; pressure RMSE when applicable |
| `app/src/pages/Train.tsx` | "Target field" dropdown: Velocity / Pressure |
| `app/src/pages/Predict.tsx` | Show field label from checkpoint metadata |
| `app/src/pages/Visualize.tsx` | Mesh viewer field selector; pressure time series in Physics tab |

---

## 5. Re-parsing

`parse_tfrecord.py` gains pressure extraction — idempotent (skips if `.dat` already exists):

```python
pressure = d['pressure'].numpy()   # [T, N, 1]
np.save(open(f'data/{split}_pressure.dat', 'wb'), pressure)
```

Same `.dat` format as velocity. Re-parsing takes ~2 min. No re-download needed.

---

## 6. Dataset (`dataset/fpc.py`)

`FpcDataset` gains a `target_field` constructor param (`'velocity'` or `'pressure'`).

**When `target_field='velocity'`** (default, unchanged):
```python
graph.x = [node_type[1], velocity_x[1], velocity_y[1]]   # [N, 3] raw
graph.y = velocity_{t+1}                                   # [N, 2]
```
`node_input_size = 11` after feature construction in Simulator.

**When `target_field='pressure'`**:
```python
graph.x = [node_type[1], pressure_t[1]]   # [N, 2] raw
graph.y = pressure_{t+1}                   # [N, 1]
```
`node_input_size = 10` after feature construction in Simulator.

Backward compatibility: if `{split}_pressure.dat` doesn't exist and `target_field='pressure'`
is requested, raise a clear error: *"Pressure data not found — re-run parse_tfrecord.py"*.

---

## 7. Model

### `Simulator.update_node_attr` — field-aware

```python
def update_node_attr(self, frames: Tensor, types: Tensor) -> Tensor:
    # frames: [N, 2] if velocity, [N, 1] if pressure
    one_hot = F.one_hot(types.squeeze(-1).long(), num_classes=9)  # [N, 9]
    node_feats = cat([frames, one_hot], dim=-1)  # [N, 11] or [N, 10]
    return self._node_normalizer(node_feats, self.training)
```

`frames` carries either velocity or pressure — the rest of the code is identical.
`node_input_size` is passed in from config, not hardcoded.

### `Simulator.forward` — same structure for both fields

Training:
```python
target_change = graph.y - frames        # acceleration (vel) or pressure change
target_norm   = output_normalizer(target_change, training=True)
predicted_norm = model(graph)
return predicted_norm, target_norm
```

Inference:
```python
predicted_norm = model(graph)
delta = output_normalizer.inverse(predicted_norm)
next_value = frames + delta             # v_{t+1} or p_{t+1}
return next_value
```

`output_size`: 2 for velocity, 1 for pressure — passed from config.

### No architecture change
`EncoderProcesserDecoder` is unchanged beyond the `output_size` param already added for
flag_simple. The GNN processes whatever normalized node features it receives — it doesn't
know or care whether they represent velocity or pressure.

---

## 8. Training Config

```json
{
  "domain": "cylinder_flow",
  "target_field": "velocity",   // or "pressure"
  "node_input_size": 11,        // 10 for pressure
  "output_size": 2,             // 1 for pressure
  "checkpoint_dir": "checkpoints"
}
```

`node_input_size` and `output_size` are derived automatically from `target_field` in
`api/routes/train.py` — user only picks the field, the sizes are computed server-side.

Checkpoint metadata saves `target_field` so Predict and Visualize know which model they loaded:
```python
torch.save({
    'epoch': ...,
    'model_state_dict': ...,
    'optimizer_state_dict': ...,
    'valid_loss': ...,
    'target_field': cfg['target_field'],   # new
    'node_input_size': ...,                # new
    'output_size': ...,                    # new
}, path)
```

---

## 9. Rollout

`api/routes/rollout.py` reads `target_field` from checkpoint metadata:

```python
target_field = ckpt.get('target_field', 'velocity')

# Swap field: velocity uses graph.x[:,1:3], pressure uses graph.x[:,1:2]
if target_field == 'velocity':
    graph.x[:, 1:3] = predicted.detach()
else:
    graph.x[:, 1:2] = predicted.detach()
```

Result pkl gains metadata:
```python
pickle.dump([[predicted, targets], crds, {
    "target_field": target_field,
    "confidence_score": confidence_score,
}], f)
```

---

## 10. API

### `GET /results/{filename}`
Adds `target_field` to response — frontend uses this to label axes correctly.

### `GET /results/{filename}/rmse`
Returns same structure regardless of field — `per_step_rmse`, `mae_at_0`, etc.
Axis label is determined by `target_field` in the response.

### `GET /results/{filename}/frame/{t}`
Returns `predicted_magnitude` and `target_magnitude` — for pressure these are scalar
values directly (no `norm()` needed since pressure is already scalar).

---

## 11. UI

### Training page (`Train.tsx`)
Add "Target field" dropdown below domain selector:
- **Velocity** (default) — *"Predict fluid velocity field (DeepMind baseline)"*
- **Pressure** — *"Predict pressure field (requires re-parsed data)"*

Grayed out with tooltip if pressure `.dat` files not present.

### Predict page (`Predict.tsx`)
Show checkpoint's `target_field` as a badge: `VELOCITY MODEL` or `PRESSURE MODEL`.
No user choice here — determined by which checkpoint is loaded.

### Visualize page (`Visualize.tsx`)
Mesh viewer field selector adapts to `target_field`:
- Velocity model: toggle between **Velocity magnitude** / **Velocity error**
- Pressure model: toggle between **Pressure** / **Pressure error**

Physics tab pressure time series: shown only when `target_field=pressure`.
All axis labels, units, and chart titles adapt to field.

---

## 12. Backward Compatibility

- Existing velocity checkpoints: `target_field` missing → defaults to `'velocity'` → fully compatible
- Existing result pkls: metadata dict missing → defaults to `target_field='velocity'` → fully compatible
- No forced re-parsing unless user selects pressure mode
