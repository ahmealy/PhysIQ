# Spec: Pressure Field (cylinder_flow)

Date: 2026-04-07  
Status: Draft

---

## 1. Overview

The `cylinder_flow` TFRecords contain a `pressure` field `[T=600, N, 1]` that is currently parsed by TensorFlow but then silently discarded. This spec adds:

1. **Re-parse**: Extract pressure from TFRecords and save alongside velocity
2. **Node feature**: Add pressure as an additional input feature to the model
3. **Second output head**: Predict pressure in addition to velocity
4. **Visualization**: Show pressure field in the mesh viewer alongside velocity

---

## 2. Confirmed TFRecord Field

From `data/meta.json`:
```json
"pressure": {
  "dtype": "float32",
  "shape": [-1, 1],
  "type": "dynamic"
}
```
Shape per trajectory: `[T=600, N, 1]` — one scalar pressure value per node per timestep.

This field is fully readable by the existing `_parse()` function in `parse_tfrecord.py` — it's just never extracted or saved.

---

## 3. Two Separable Sub-Features

Pressure touches two independent things:

**A — Pressure as input feature**: Add `pressure_t` to node features at each timestep. The model gets richer physics signal (pressure drives flow) → likely improves velocity prediction accuracy. Node input size: `11 → 12`.

**B — Pressure as output head**: Add a second decoder head predicting `pressure_{t+1}`. This is a multi-task learning setup — the model predicts both velocity and pressure simultaneously.

These are separable. Sub-feature A is simpler and lower risk. Sub-feature B requires careful multi-task loss balancing.

**Decision**: Implement both, but A is the priority. B is additive on top.

---

## 4. Files Changed / Created

### Modified files
| File | Change |
|---|---|
| `data/parse_tfrecord.py` | Extract `pressure` field, save to `train_pressure.dat`, `valid_pressure.dat`, `test_pressure.dat` |
| `dataset/fpc.py` | Load pressure `.dat`, include `pressure_t` in `graph.x`, include `pressure_{t+1}` in `graph.pressure_y` |
| `model/simulator.py` | `update_node_attr`: add pressure to node features; add `pressure_head` decoder; forward returns pressure prediction too |
| `model/model.py` | `EncoderProcesserDecoder`: add optional `pressure_output_size` param; instantiate second `Decoder` for pressure |
| `train.py` | Multi-task loss: `loss = loss_vel + lambda_p * loss_pressure`; `lambda_p = 0.1` default |
| `api/routes/rollout.py` | Save pressure predictions to pkl alongside velocity |
| `api/routes/results.py` | New `GET /results/{filename}/frame/{t}` returns pressure field; new field in rmse response |
| `app/src/pages/Visualize.tsx` | Pressure colormap toggle in mesh viewer; pressure time series chart |

---

## 5. Re-parsing

### `parse_tfrecord.py` changes

Add pressure extraction in the main loop alongside velocity:

```python
pressure = d['pressure'].numpy()   # [T, N, 1]

# Save as .dat alongside velocity
with open(f'data/{split}_pressure.dat', 'wb') as f:
    np.save(f, pressure)  # shape stored in file
```

The `.dat` format already used for velocity works identically for pressure — same `np.save` / `np.load` pattern.

**Re-parsing required**: Users who already have parsed data need to re-run `parse_tfrecord.py`. The script checks if pressure `.dat` already exists and skips if so (idempotent).

---

## 6. Dataset

### `FpcDataset.__getitem__` changes

```python
# Current graph.x: [N, 3] = [node_type[1], velocity_x[1], velocity_y[1]]
# New graph.x:     [N, 4] = [node_type[1], velocity_x[1], velocity_y[1], pressure[1]]

pressure_t = self.pressure_data[traj_idx][t]   # [N, 1]
x = np.concatenate([node_type, tra_velocity, pressure_t], axis=-1)  # [N, 4]

graph.pressure_y = pressure_{t+1}  # [N, 1] — target pressure
```

Backward compatibility: if pressure `.dat` doesn't exist (user hasn't re-parsed), `FpcDataset` falls back to `node_input_size=11` (no pressure). Controlled by a `use_pressure` flag in config, default `True` if file exists.

---

## 7. Model

### Node features (updated)

In `Simulator.update_node_attr`:
```python
# Before: velocity[2] + one_hot[9] = 11
# After:  velocity[2] + one_hot[9] + pressure[1] = 12

pressure = graph.x[:, 3:4]   # [N, 1]
node_feats = cat([frames, one_hot, pressure], dim=-1)   # [N, 12]
```

`_node_normalizer` size: `11 → 12`.

### Second output head (pressure)

In `EncoderProcesserDecoder`:
```python
class EncoderProcesserDecoder(nn.Module):
    def __init__(self, ..., output_size=2, pressure_output_size=0):
        ...
        self.decoder = Decoder(hidden_size, output_size)
        self.pressure_decoder = (
            Decoder(hidden_size, pressure_output_size)
            if pressure_output_size > 0 else None
        )
    
    def forward(self, graph):
        graph = self.encoder(graph)
        for block in self.processer_list:
            graph = block(graph)
        vel = self.decoder(graph)
        pres = self.pressure_decoder(graph) if self.pressure_decoder else None
        return vel, pres
```

The pressure decoder is a **separate MLP from the same latent** — same architecture as velocity decoder, output size 1. Both decoders have no LayerNorm (same as DeepMind).

### Multi-task loss in `train.py`

```python
pred_vel_norm, pred_pres_norm = model(graph, noise)

# Velocity loss (existing)
loss_vel = mse(pred_vel_norm[fluid_mask], target_vel_norm[fluid_mask])

# Pressure loss (new)
if pred_pres_norm is not None:
    target_pres_norm = pressure_normalizer(graph.pressure_y, training=True)
    loss_pres = mse(pred_pres_norm[fluid_mask], target_pres_norm[fluid_mask])
    loss = loss_vel + cfg.get('lambda_pressure', 0.1) * loss_pres
else:
    loss = loss_vel
```

`lambda_pressure = 0.1` — pressure is a secondary task; velocity accuracy is the primary goal.

---

## 8. Rollout

In `_rollout_cfd`, after each step save the pressure prediction:

```python
pred_pressure = pressure_normalizer.inverse(pred_pres_norm)  # [N, 1]
pressures.append(pred_pressure.cpu().numpy())
```

Saved in pkl as additional field:
```python
pickle.dump([[predicted_vel, targets_vel], crds, {
    "predicted_pressure": np.stack(pressures),   # [T, N, 1]
    "target_pressure":    np.stack(target_pres),  # [T, N, 1]
    "confidence_score":   confidence_score,        # from confidence spec
}], f)
```

---

## 9. API

### `GET /results/{filename}/frame/{t}`
Adds:
```json
{
  "predicted_pressure": [...],    // [N] scalar per node
  "target_pressure":    [...]     // [N] scalar per node
}
```

### `GET /results/{filename}/rmse`
Adds:
```json
{
  "per_step_pressure_rmse": [...],
  "pressure_rmse_at_0":     0.0012,
  "pressure_rmse_at_end":   0.0089
}
```

---

## 10. Visualization

### Mesh viewer toggle
Add a field selector in the Visualize page mesh viewer:
- **Velocity magnitude** (existing, default)
- **Pressure** (new)
- **Velocity error** (existing)
- **Pressure error** (new)

When "Pressure" is selected, the colormap uses the `predicted_pressure` values instead of velocity magnitude. Same triangulation, same colormap scale logic.

### Pressure time series
In the Physics tab (currently shows kinetic energy + divergence), add:
- Mean pressure over time (predicted vs ground truth)
- Pressure RMSE over time

---

## 11. Config / Backward Compatibility

`use_pressure` flag in train config JSON (default: `True` if pressure `.dat` exists, `False` otherwise).

If `use_pressure=False`:
- `node_input_size = 11` (no change to existing behavior)
- `pressure_output_size = 0` (no second head)
- Existing checkpoints load fine

If `use_pressure=True`:
- `node_input_size = 12`
- `pressure_output_size = 1`
- Existing checkpoints are **incompatible** (different input size) — must retrain

The UI shows a "Pressure field" toggle in Training config. Default: on if pressure data is available.

---

## 12. Re-parse Required

Users must re-run `parse_tfrecord.py` to extract pressure. The script:
1. Checks for existing `train_pressure.dat` — skips if present
2. Re-reads TFRecords to extract pressure
3. Saves pressure `.dat` files

Since the TFRecords are still on disk, re-parsing is fast (~2 min). No re-download needed.
