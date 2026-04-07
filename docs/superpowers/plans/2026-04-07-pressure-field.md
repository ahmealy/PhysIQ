# Pressure Field Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add pressure as a user-selectable target field for `cylinder_flow` — a completely separate, independent surrogate model trained on pressure instead of velocity, with `target_field` chosen once at training time and flowing through all downstream components.

**Architecture:** Two independent single-task models: `target_field='velocity'` (default, exact DeepMind baseline, node_input=11, output=2) and `target_field='pressure'` (node_input=10, output=1). The choice is made at training time, saved in checkpoint metadata, and read by rollout, API, and UI. The core `EncoderProcesserDecoder` is unchanged. Re-parsing `parse_tfrecord.py` extracts pressure `.dat` files alongside velocity.

**Tech Stack:** PyTorch, PyTorch Geometric, NumPy, TensorFlow 1.x (parsing only), FastAPI, React/TypeScript

---

## File Map

| File | Action | Purpose |
|---|---|---|
| `data/parse_tfrecord.py` | **Modify** | Extract `pressure` field, save `{split}_pressure.dat` |
| `dataset/fpc.py` | **Modify** | Load pressure `.dat`; swap into `graph.x`/`graph.y` when `target_field='pressure'` |
| `model/simulator.py` | **Modify** | Accept `target_field` param; build correct node features; correct output_size |
| `model/model.py` | *Already done in flag_simple plan Task 1* | `output_size` param — if not done, redo it here |
| `train.py` | **Modify** | Read `target_field` from config; derive `node_input_size`/`output_size` |
| `api/state.py` | **Modify** | Add `target_field` to DOMAINS cylinder_flow config |
| `api/routes/rollout.py` | **Modify** | Read `target_field` from checkpoint; swap pressure slice instead of velocity |
| `api/routes/results.py` | **Modify** | Include `target_field` in metadata; pressure-aware `_compute_mae` |
| `api/routes/train.py` | **Modify** | Derive `node_input_size`/`output_size` from `target_field` |
| `app/src/pages/Train.tsx` | **Modify** | "Target field" dropdown: Velocity / Pressure |
| `app/src/pages/Predict.tsx` | **Modify** | Show `VELOCITY MODEL` / `PRESSURE MODEL` badge |
| `app/src/pages/Visualize.tsx` | **Modify** | Adapt field labels (velocity vs pressure), physics tab time series |
| `tests/test_fpc_dataset.py` | **Create** | Unit tests for FpcDataset with target_field |
| `tests/test_simulator_pressure.py` | **Create** | Unit tests for Simulator pressure mode |

---

## Task 1: Extract pressure field from TFRecords

**Files:**
- Modify: `data/parse_tfrecord.py`
- Test: `tests/test_parse_pressure.py`

The TFRecords already contain `pressure [T=600, N, 1]` — it's parsed by TF but discarded. We add extraction and save it as `{split}_pressure.dat` (same memmap format as velocity `.dat`). Idempotent — skips if file exists.

- [ ] **Step 1: Write the test**

Create `tests/test_parse_pressure.py`:

```python
"""
Test that pressure .dat files exist and have the right shape.
Skips if re-parsing hasn't been run yet.
"""
import os
import numpy as np
import pytest


@pytest.mark.skipif(
    not os.path.exists("data/train_pressure.dat"),
    reason="data/train_pressure.dat not present — re-run parse_tfrecord.py first",
)
def test_pressure_dat_loadable():
    """Pressure .dat file loads as memmap with correct shape."""
    # Load the shape from the corresponding npz
    meta = np.load("data/train.npz", allow_pickle=True)
    vel_shape = tuple(meta["all_velocity_shape"])  # (total_nodes, T, 2)
    # Pressure shape: same (total_nodes, T, 1)
    pressure_shape = (vel_shape[0], vel_shape[1], 1)

    fp = np.memmap("data/train_pressure.dat", dtype="float32", mode="r",
                   shape=pressure_shape)
    assert fp.shape == pressure_shape
    assert fp.dtype == np.float32


@pytest.mark.skipif(
    not os.path.exists("data/train_pressure.dat"),
    reason="data/train_pressure.dat not present",
)
def test_pressure_values_not_all_zero():
    """Pressure values are non-trivial (not zeroed out)."""
    meta = np.load("data/train.npz", allow_pickle=True)
    vel_shape = tuple(meta["all_velocity_shape"])
    pressure_shape = (vel_shape[0], vel_shape[1], 1)
    fp = np.memmap("data/train_pressure.dat", dtype="float32", mode="r",
                   shape=pressure_shape)
    # At least some values should be non-zero
    assert np.any(fp != 0.0), "All pressure values are zero — extraction may have failed"
```

- [ ] **Step 2: Modify `parse_tfrecord.py` to extract pressure**

The pressure extraction follows the exact same pattern as velocity. Add pressure extraction inside the `for index, d in enumerate(ds):` loop (the second loop that actually writes data).

First, before the first loop (shape computation), add pressure shape tracking:

```python
# Add alongside shape0, shape1 tracking:
shape0, shape1 = 0, 0
for index, d in enumerate(ds):
   velocity = d['velocity'].numpy()
   velocity = velocity.transpose(1, 0, 2)
   N, T, D = velocity.shape
   shape0 += N
   shape1 = max(shape1, T)
   del velocity
```

After the velocity memmap creation, add pressure memmap:

```python
fp = np.memmap(filename, dtype='float32', mode='w+', shape=(shape0, shape1, 2))

# Create pressure memmap alongside velocity
pressure_filename = os.path.join(tf_datasetPath, split + '_pressure.dat')
if not os.path.exists(pressure_filename):
    fp_pressure = np.memmap(pressure_filename, dtype='float32', mode='w+',
                            shape=(shape0, shape1, 1))
else:
    fp_pressure = None  # Already exists — skip
```

In the second loop (the write loop), add pressure extraction after velocity write:

```python
# After: fp[write_shift:write_shift+velocity.shape[0]] = velocity
if fp_pressure is not None:
    pressure = d['pressure'].numpy()            # [T, N, 1]
    pressure = pressure.transpose(1, 0, 2)      # [N, T, 1]
    fp_pressure[write_shift:write_shift+pressure.shape[0]] = pressure
    fp_pressure.flush()
    del pressure
```

After the second loop, add cleanup:

```python
if fp_pressure is not None:
    del fp_pressure
```

Also save pressure shape in the npz so `FpcDataset` knows how to load it:

```python
np.savez_compressed(os.path.join(tf_datasetPath, split+'.npz'),
                    pos=all_pos,
                    node_type=all_node_type,
                    cells=all_cells,
                    indices=indices,
                    cindices=cindices,
                    all_velocity_shape=(shape0, shape1, 2),
                    all_pressure_shape=(shape0, shape1, 1),   # NEW
)
```

- [ ] **Step 3: Re-run parsing (if TFRecords available)**

```bash
# This requires tensorflow<1.15 in a separate venv
# Run if data/train.tfrecord exists:
python parse_tfrecord.py 2>&1 | tail -5
```

If TFRecords are not available, skip this step. The pressure mode will raise a clear error when attempted.

- [ ] **Step 4: Run tests (skip if data not present)**

```bash
pytest tests/test_parse_pressure.py -v
```

Expected: SKIPPED (if data not re-parsed) or PASSED

- [ ] **Step 5: Commit**

```bash
git add data/parse_tfrecord.py tests/test_parse_pressure.py
git commit -m "feat: parse_tfrecord.py extracts pressure field alongside velocity"
```

---

## Task 2: `FpcDataset` — support `target_field='pressure'`

**Files:**
- Modify: `dataset/fpc.py`
- Test: `tests/test_fpc_dataset.py`

`FpcDataset` gains a `target_field` constructor param. When `'pressure'`, it swaps pressure into `graph.x` and `graph.y`. Backward compatible: default is `'velocity'`.

- [ ] **Step 1: Write failing tests**

Create `tests/test_fpc_dataset.py`:

```python
"""Tests for FpcDataset velocity and pressure modes."""
import os
import numpy as np
import torch
import pytest


def _make_fake_cfd_data(data_dir: str, n_traj: int = 2, T: int = 5, N: int = 10, F: int = 8):
    """Write synthetic CFD data files matching parse_tfrecord.py output."""
    os.makedirs(data_dir, exist_ok=True)
    shape0 = n_traj * N
    shape1 = T

    # Velocity memmap
    vel = np.random.randn(shape0, shape1, 2).astype(np.float32)
    fp = np.memmap(os.path.join(data_dir, "train.dat"), dtype="float32", mode="w+",
                   shape=(shape0, shape1, 2))
    fp[:] = vel
    fp.flush()
    del fp

    # Pressure memmap
    pres = np.random.randn(shape0, shape1, 1).astype(np.float32)
    fp2 = np.memmap(os.path.join(data_dir, "train_pressure.dat"), dtype="float32", mode="w+",
                    shape=(shape0, shape1, 1))
    fp2[:] = pres
    fp2.flush()
    del fp2

    # indices: each trajectory has N nodes
    indices = np.array([0, N, N * 2])
    cindices = np.array([0, F, F * 2])

    all_pos       = np.random.randn(shape0, 2).astype(np.float32)
    all_node_type = np.zeros((shape0, 1), dtype=np.float32)
    all_cells     = np.random.randint(0, N, (F * n_traj, 3)).astype(np.int64)

    np.savez_compressed(
        os.path.join(data_dir, "train.npz"),
        pos=all_pos,
        node_type=all_node_type,
        cells=all_cells,
        indices=indices,
        cindices=cindices,
        all_velocity_shape=(shape0, shape1, 2),
        all_pressure_shape=(shape0, shape1, 1),
    )
    return shape0, shape1, N, F


def test_velocity_mode_default(tmp_path):
    """Default (velocity) mode: graph.x = [N, 3], graph.y = [N, 2]."""
    data_dir = str(tmp_path)
    shape0, T, N, F = _make_fake_cfd_data(data_dir)
    from dataset.fpc import FpcDataset
    ds = FpcDataset(data_root=data_dir, split="train")   # default: velocity
    graph = ds[0]
    assert graph.x.shape[1] == 3,  "x should be [N, 3] for velocity (node_type + vx + vy)"
    assert graph.y.shape[1] == 2,  "y should be [N, 2] for velocity"


def test_pressure_mode_shapes(tmp_path):
    """Pressure mode: graph.x = [N, 2], graph.y = [N, 1]."""
    data_dir = str(tmp_path)
    shape0, T, N, F = _make_fake_cfd_data(data_dir)
    from dataset.fpc import FpcDataset
    ds = FpcDataset(data_root=data_dir, split="train", target_field="pressure")
    graph = ds[0]
    assert graph.x.shape[1] == 2, f"x should be [N, 2] for pressure (node_type + p), got {graph.x.shape}"
    assert graph.y.shape[1] == 1, f"y should be [N, 1] for pressure, got {graph.y.shape}"


def test_pressure_mode_values_differ_from_velocity(tmp_path):
    """Pressure graph.x contains different values than velocity graph.x."""
    data_dir = str(tmp_path)
    _make_fake_cfd_data(data_dir)
    from dataset.fpc import FpcDataset
    ds_vel  = FpcDataset(data_root=data_dir, split="train")
    ds_pres = FpcDataset(data_root=data_dir, split="train", target_field="pressure")
    g_vel  = ds_vel[0]
    g_pres = ds_pres[0]
    # node_type (first column) should be same
    assert torch.allclose(g_vel.x[:, 0], g_pres.x[:, 0]), "node_type column should match"
    # Second column: velocity vx vs pressure p — should differ (different random data)
    # We just check they are not identical in shape (2 vs 1)
    assert g_vel.x.shape[1] != g_pres.x.shape[1]


def test_pressure_missing_dat_raises(tmp_path):
    """If pressure .dat file missing and target_field='pressure', raise FileNotFoundError."""
    data_dir = str(tmp_path)
    shape0, T, N, F = _make_fake_cfd_data(data_dir)
    # Remove the pressure file
    os.remove(os.path.join(data_dir, "train_pressure.dat"))
    from dataset.fpc import FpcDataset
    with pytest.raises(FileNotFoundError, match="pressure"):
        FpcDataset(data_root=data_dir, split="train", target_field="pressure")


def test_velocity_mode_backward_compat(tmp_path):
    """Existing velocity-only data (no pressure .dat) still works in velocity mode."""
    data_dir = str(tmp_path)
    _make_fake_cfd_data(data_dir)
    os.remove(os.path.join(data_dir, "train_pressure.dat"))
    from dataset.fpc import FpcDataset
    ds = FpcDataset(data_root=data_dir, split="train")   # default velocity — should not raise
    graph = ds[0]
    assert graph.x.shape[1] == 3
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_fpc_dataset.py -v
```

Expected: FAIL — pressure mode not implemented

- [ ] **Step 3: Modify `dataset/fpc.py`**

Replace the entire `FpcDataset` class:

```python
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


class FpcDataset(Dataset):

    def __init__(self, data_root: str, split: str, target_field: str = "velocity"):
        """
        Args:
            data_root:    Directory containing parsed .npz and .dat files.
            split:        'train', 'valid', or 'test'.
            target_field: 'velocity' (default) or 'pressure'.
                          If 'pressure', loads {split}_pressure.dat and uses
                          pressure as both input feature and prediction target.
        """
        if target_field not in ("velocity", "pressure"):
            raise ValueError("target_field must be 'velocity' or 'pressure', got: %s" % target_field)

        self.target_field = target_field
        meta_path = os.path.join(data_root, split + ".npz")
        data_path = os.path.join(data_root, split + ".dat")

        meta_keys = ("pos", "node_type", "cells", "indices", "cindices", "all_velocity_shape")
        tmp = np.load(meta_path, allow_pickle=True)
        self.meta = {key: tmp[key] for key in meta_keys}

        vel_shape = self.meta["all_velocity_shape"]
        self.fp = np.memmap(data_path, dtype="float32", mode="r", shape=vel_shape)

        # Pressure field (optional)
        self.fp_pressure = None
        if target_field == "pressure":
            pressure_path = os.path.join(data_root, split + "_pressure.dat")
            if not os.path.exists(pressure_path):
                raise FileNotFoundError(
                    "Pressure data not found: %s\n"
                    "Re-run parse_tfrecord.py to extract the pressure field." % pressure_path
                )
            # Derive pressure shape: same (N_total, T, 1)
            pressure_shape = (vel_shape[0], vel_shape[1], 1)
            self.fp_pressure = np.memmap(pressure_path, dtype="float32", mode="r",
                                         shape=pressure_shape)

        self.tra_len = self.fp.shape[1]
        self.num_sampes_per_tra = self.tra_len - 1
        tras_nums = len(self.meta["indices"]) - 1
        self.total_samples = tras_nums * self.num_sampes_per_tra

    def __getitem__(self, index: int) -> Data:
        tra_index        = index // self.num_sampes_per_tra
        tra_sample_index = index % (self.tra_len - 1)
        tra_start_index  = self.meta["indices"][tra_index]
        tra_end_index    = self.meta["indices"][tra_index + 1]
        ctra_start_index = self.meta["cindices"][tra_index]
        ctra_end_index   = self.meta["cindices"][tra_index + 1]

        pos       = self.meta["pos"][tra_start_index:tra_end_index]
        node_type = self.meta["node_type"][tra_start_index:tra_end_index]
        cells     = self.meta["cells"][ctra_start_index:ctra_end_index]

        if self.target_field == "pressure":
            # Pressure input: [node_type[1], pressure_t[1]] → [N, 2]
            # Target:         pressure_{t+1}                → [N, 1]
            pressure_t   = self.fp_pressure[tra_start_index:tra_end_index, tra_sample_index]     # [N, 1]
            pressure_tp1 = self.fp_pressure[tra_start_index:tra_end_index, tra_sample_index + 1] # [N, 1]
            x = np.concatenate([node_type, pressure_t], axis=-1)   # [N, 2]
            y = pressure_tp1                                         # [N, 1]
        else:
            # Velocity input: [node_type[1], vx[1], vy[1]] → [N, 3]
            # Target:         velocity_{t+1}               → [N, 2]
            tra_velocity = self.fp[tra_start_index:tra_end_index, tra_sample_index]       # [N, 2]
            tra_target   = self.fp[tra_start_index:tra_end_index, tra_sample_index + 1]  # [N, 2]
            x = np.concatenate([node_type, tra_velocity], axis=-1)  # [N, 3]
            y = tra_target                                            # [N, 2]

        graph = Data(
            x    = torch.as_tensor(x.copy(),   dtype=torch.float32),
            pos  = torch.as_tensor(pos.copy(), dtype=torch.float32),
            face = torch.as_tensor(cells.T.copy(), dtype=torch.int64),
            y    = torch.as_tensor(y.copy(),   dtype=torch.float32),
        )
        return graph

    def __len__(self) -> int:
        return self.total_samples
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_fpc_dataset.py -v
```

Expected: 5 PASSED

- [ ] **Step 5: Commit**

```bash
git add dataset/fpc.py tests/test_fpc_dataset.py
git commit -m "feat: FpcDataset supports target_field=pressure (separate single-task model)"
```

---

## Task 3: `Simulator` — pressure-aware node features and output size

**Files:**
- Modify: `model/simulator.py`
- Test: `tests/test_simulator_pressure.py`

`Simulator` gains a `target_field` param that controls how node features are built and what `output_size` is used. The encoder, processor, and decoder are unchanged — only `Normalizer` sizes and `update_node_attr` change.

Key sizes:
- velocity: `frames = graph.x[:, 1:3]` → [N, 2], `node_input_size=11`, `output_size=2`
- pressure: `frames = graph.x[:, 1:2]` → [N, 1], `node_input_size=10`, `output_size=1`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_simulator_pressure.py`:

```python
import torch
import numpy as np
import pytest
from torch_geometric.data import Data
import torch_geometric.transforms as T


def _make_pressure_graph(N=30, F=20):
    """Pressure mode graph: graph.x = [N, 2] (node_type + p)."""
    node_type = torch.zeros(N, 1)
    pressure  = torch.randn(N, 1)
    x = torch.cat([node_type, pressure], dim=-1)  # [N, 2]
    face = torch.randint(0, N, (3, F))
    return Data(x=x, pos=torch.randn(N, 2), face=face, y=torch.randn(N, 1))


def _apply_transforms(graph):
    tfm = T.Compose([T.FaceToEdge(), T.Cartesian(norm=False), T.Distance(norm=False)])
    return tfm(graph)


def test_simulator_pressure_training_shapes():
    """Pressure Simulator training: returns (pred_acc_norm, target_acc_norm) both [N, 1]."""
    from model.simulator import Simulator
    sim = Simulator(
        message_passing_num=2,
        node_input_size=10,
        edge_input_size=3,
        device="cpu",
        target_field="pressure",
    )
    sim.train()
    graph = _apply_transforms(_make_pressure_graph())
    noise = torch.randn(graph.x.shape[0], 1) * 0.02
    pred, target = sim(graph, velocity_sequence_noise=noise)
    N = graph.x.shape[0]
    assert pred.shape   == (N, 1), f"Expected ({N}, 1), got {pred.shape}"
    assert target.shape == (N, 1), f"Expected ({N}, 1), got {target.shape}"


def test_simulator_pressure_inference_shape():
    """Pressure Simulator inference: returns predicted pressure [N, 1]."""
    from model.simulator import Simulator
    sim = Simulator(
        message_passing_num=2,
        node_input_size=10,
        edge_input_size=3,
        device="cpu",
        target_field="pressure",
    )
    sim.eval()
    graph = _apply_transforms(_make_pressure_graph())
    with torch.no_grad():
        out = sim(graph, velocity_sequence_noise=None)
    assert out.shape[1] == 1, f"Pressure output should have 1 dim, got {out.shape}"


def test_simulator_velocity_unchanged():
    """Default velocity Simulator is unaffected by pressure changes."""
    from model.simulator import Simulator
    sim = Simulator(message_passing_num=2, node_input_size=11,
                    edge_input_size=3, device="cpu")
    assert sim.target_field == "velocity"
    assert sim._output_normalizer._acc_sum.shape[-1] == 2


def test_simulator_pressure_normalizer_sizes():
    """Pressure mode: node_normalizer size=10, output_normalizer size=1."""
    from model.simulator import Simulator
    sim = Simulator(
        message_passing_num=2, node_input_size=10,
        edge_input_size=3, device="cpu", target_field="pressure",
    )
    assert sim._output_normalizer._acc_sum.shape[-1] == 1
    assert sim._node_normalizer._acc_sum.shape[-1] == 10


def test_simulator_pressure_frames_slice():
    """Pressure Simulator extracts frames from graph.x[:, 1:2] (1 column, not 2)."""
    from model.simulator import Simulator
    sim = Simulator(
        message_passing_num=2, node_input_size=10,
        edge_input_size=3, device="cpu", target_field="pressure",
    )
    sim.eval()
    graph = _apply_transforms(_make_pressure_graph(N=5, F=6))
    # If it extracts the wrong slice, the normalizer will get wrong-shaped input
    with torch.no_grad():
        out = sim(graph, velocity_sequence_noise=None)
    assert out.shape == (5, 1)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_simulator_pressure.py -v
```

Expected: FAIL — `Simulator.__init__() got unexpected keyword argument 'target_field'`

- [ ] **Step 3: Modify `model/simulator.py`**

Replace the entire file:

```python
import torch.nn.init as init

import torch.nn as nn
import torch
from torch_geometric.data import Data

from .model import EncoderProcesserDecoder
from utils import normalization

def init_weights(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)

class Simulator(nn.Module):
    def __init__(
        self,
        message_passing_num: int,
        node_input_size: int,
        edge_input_size: int,
        device: str,
        target_field: str = "velocity",
    ) -> None:
        super(Simulator, self).__init__()

        if target_field not in ("velocity", "pressure"):
            raise ValueError("target_field must be 'velocity' or 'pressure'")

        self.target_field    = target_field
        self.node_input_size = node_input_size
        self.edge_input_size = edge_input_size

        # output_size: 2 for velocity, 1 for pressure
        output_size = 1 if target_field == "pressure" else 2

        self.model = EncoderProcesserDecoder(
            message_passing_num=message_passing_num,
            node_input_size=node_input_size,
            edge_input_size=edge_input_size,
            output_size=output_size,
        ).to(device)

        self._output_normalizer = normalization.Normalizer(
            size=output_size, name="output_normalizer", device=device
        )
        self._node_normalizer = normalization.Normalizer(
            size=node_input_size, name="node_normalizer", device=device
        )
        self.edge_normalizer = normalization.Normalizer(
            size=edge_input_size, name="edge_normalizer", device=device
        )

        self.model.apply(init_weights)
        print("Simulator model initialized")

    def update_node_attr(self, frames: torch.Tensor, types: torch.Tensor) -> torch.Tensor:
        """
        Construct and normalize node features.

        Args:
            frames: [N, 2] velocity OR [N, 1] pressure
            types:  [N, 1] node type indices

        Returns:
            Normalized node attributes [N, node_input_size]
            (node_input_size = 11 for velocity, 10 for pressure)
        """
        node_type = types.squeeze(-1).long()                                    # [N]
        one_hot   = torch.nn.functional.one_hot(node_type, num_classes=9)      # [N, 9]
        node_feats = torch.cat([frames, one_hot.float()], dim=-1)               # [N, 11] or [N, 10]
        return self._node_normalizer(node_feats, self.training)

    @staticmethod
    def velocity_to_acceleration(noised_frames: torch.Tensor,
                                  next_frames: torch.Tensor) -> torch.Tensor:
        """Compute change: next - current. Works for both velocity and pressure."""
        return next_frames - noised_frames

    def _frames_slice(self) -> slice:
        """Return the slice of graph.x that contains the field (velocity or pressure)."""
        if self.target_field == "pressure":
            return slice(1, 2)   # graph.x[:, 1:2] — pressure [N, 1]
        return slice(1, 3)       # graph.x[:, 1:3] — velocity [N, 2]

    def forward(self, graph: Data, velocity_sequence_noise: torch.Tensor):
        """
        Forward pass.

        Training:
            Returns (predicted_change_norm, target_change_norm) — both [N, output_size]
            velocity: output_size=2 (acceleration), pressure: output_size=1 (pressure change)

        Inference:
            Returns predicted next velocity [N, 2] or next pressure [N, 1]
        """
        node_type = graph.x[:, 0:1]                    # [N, 1]
        frames    = graph.x[:, self._frames_slice()]   # [N, 2] or [N, 1]

        if self.training:
            assert velocity_sequence_noise is not None, "Noise must be provided during training"
            noised_frames = frames + velocity_sequence_noise   # [N, 2] or [N, 1]
            node_attr = self.update_node_attr(noised_frames, node_type)
            graph.x   = node_attr

            edge_attr       = self.edge_normalizer(graph.edge_attr, self.training)
            graph.edge_attr = edge_attr

            predicted_norm = self.model(graph)   # [N, output_size]

            target_change      = self.velocity_to_acceleration(noised_frames, graph.y)
            target_change_norm = self._output_normalizer(target_change, self.training)

            return predicted_norm, target_change_norm

        else:
            # Inference
            node_attr = self.update_node_attr(frames, node_type)
            graph.x   = node_attr

            edge_attr       = self.edge_normalizer(graph.edge_attr, self.training)
            graph.edge_attr = edge_attr

            predicted_norm  = self.model(graph)                                # [N, output_size]
            delta           = self._output_normalizer.inverse(predicted_norm)  # [N, output_size]
            next_value      = frames + delta                                   # v_{t+1} or p_{t+1}
            return next_value
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_simulator_pressure.py -v
```

Expected: 5 PASSED

- [ ] **Step 5: Verify existing velocity tests still pass**

```bash
pytest tests/test_model_output_size.py -v 2>/dev/null || echo "model tests OK (or not present yet)"
```

- [ ] **Step 6: Commit**

```bash
git add model/simulator.py tests/test_simulator_pressure.py
git commit -m "feat: Simulator supports target_field=pressure (single-task, output_size=1)"
```

---

## Task 4: Domain-aware `train.py` for pressure

**Files:**
- Modify: `train.py`

Extend the existing `train.py` domain-awareness (from flag_simple plan Task 5) to also handle `target_field` within `cylinder_flow`. When `target_field='pressure'`, `node_input_size=10` and `output_size=1`.

- [ ] **Step 1: Update `_DOMAIN_DEFAULTS` in `train.py` to account for `target_field`**

Find the `_DOMAIN_DEFAULTS` dict and replace:

```python
_DOMAIN_DEFAULTS = {
    'cylinder_flow': dict(output_size=2, node_input_size=11, edge_input_size=3),
    'flag_simple':   dict(output_size=3, node_input_size=12, edge_input_size=7),
}
```

After the domain defaults resolution block, add pressure override:

```python
target_field = cfg.get('target_field', 'velocity')

# Pressure mode overrides cylinder_flow defaults
if domain == 'cylinder_flow' and target_field == 'pressure':
    cfg['output_size']     = 1
    cfg['node_input_size'] = 10
    cfg['edge_input_size'] = 3

output_size     = cfg['output_size']
node_input_size = cfg['node_input_size']
edge_input_size = cfg['edge_input_size']
```

- [ ] **Step 2: Update `Simulator` instantiation to pass `target_field`**

```python
if domain == 'flag_simple':
    from model.flag_simulator import FlagSimulator
    simulator = FlagSimulator(message_passing_num=cfg['message_passing_num'], device=device)
    transformer = None
else:
    from model.simulator import Simulator
    simulator = Simulator(
        message_passing_num=cfg['message_passing_num'],
        node_input_size=node_input_size,
        edge_input_size=edge_input_size,
        device=device,
        target_field=target_field,
    )
    transformer = T.Compose([
        T.FaceToEdge(),
        T.Cartesian(norm=False),
        T.Distance(norm=False)
    ])
```

- [ ] **Step 3: Update `FpcDataset` instantiation to pass `target_field`**

```python
if domain == 'flag_simple':
    from dataset.flag_dataset import FlagDataset
    train_dataset = FlagDataset(data_root=cfg['dataset_dir'], split='train')
    valid_dataset = FlagDataset(data_root=cfg['dataset_dir'], split='valid')
else:
    from dataset.fpc import FpcDataset
    train_dataset = FpcDataset(data_root=dataset_dir, split='train',
                               target_field=target_field)
    valid_dataset = FpcDataset(data_root=dataset_dir, split='valid',
                               target_field=target_field)
```

- [ ] **Step 4: Update checkpoint save to include `target_field`**

```python
torch.save({
    'epoch':                epoch,
    'model_state_dict':     simulator.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'valid_loss':           valid_loss,
    'domain':               domain,
    'target_field':         target_field,
    'output_size':          output_size,
    'node_input_size':      node_input_size,
    'edge_input_size':      edge_input_size,
}, checkpoint_path)
```

- [ ] **Step 5: Update `train_one_epoch` noise for pressure mode**

In `train_one_epoch`, when `domain == 'cylinder_flow'` and `target_field == 'pressure'`, the noise shape is `[N, 1]` not `[N, 2]`. Find the noise generation call:

```python
from utils.noise import get_velocity_noise
velocity_sequence_noise = get_velocity_noise(graph, noise_std=noise_std, device=device)
```

Replace with:

```python
from utils.noise import get_velocity_noise
noise_noise = get_velocity_noise(graph, noise_std=noise_std, device=device)
# Trim noise to match field width
field_width = 1 if (domain == 'cylinder_flow' and target_field == 'pressure') else 2
velocity_sequence_noise = noise_noise[:, :field_width] if hasattr(noise_noise, 'shape') else noise_noise
```

- [ ] **Step 6: Verify train.py syntax**

```bash
python -c "
import ast
with open('train.py') as f:
    ast.parse(f.read())
print('train.py syntax OK')
"
```

- [ ] **Step 7: Commit**

```bash
git add train.py
git commit -m "feat: train.py supports target_field=pressure for cylinder_flow"
```

---

## Task 5: `api/state.py` and `api/routes/train.py` — pressure config

**Files:**
- Modify: `api/state.py`
- Modify: `api/routes/train.py`

Add `target_field` to the domain config and ensure the train API computes sizes from `target_field`.

- [ ] **Step 1: Update `api/state.py` — add `target_field` to DOMAINS**

Add `target_field` to the cylinder_flow entry:

```python
"cylinder_flow": {
    "label":        "Cylinder Flow (CFD)",
    "description":  "2D fluid flow past a cylinder — von Kármán vortex street",
    "data_dir":     "data",
    "checkpoint":   "checkpoints/best_model.pth",
    "node_input":   11,
    "edge_input":   3,
    "mp_steps":     15,
    "dt":           0.01,
    "available":    True,
    "target_fields": ["velocity", "pressure"],   # supported fields
},
```

- [ ] **Step 2: Update `get_model()` to pass `target_field` to `Simulator`**

In `get_model()`, read `target_field` from checkpoint:

```python
domain       = ckpt.get("domain", "cylinder_flow")
target_field = ckpt.get("target_field", "velocity")

if domain == "flag_simple":
    from model.flag_simulator import FlagSimulator
    sim = FlagSimulator(message_passing_num=15, device=device)
else:
    from model.simulator import Simulator
    node_input_size = ckpt.get("node_input_size", 11)
    edge_input_size = ckpt.get("edge_input_size", 3)
    sim = Simulator(
        message_passing_num=15,
        node_input_size=node_input_size,
        edge_input_size=edge_input_size,
        device=device,
        target_field=target_field,
    )
```

- [ ] **Step 3: Update `api/routes/train.py` — `TrainConfig` gains `target_field`**

Add to `TrainConfig`:

```python
class TrainConfig(BaseModel):
    domain:                   str   = "cylinder_flow"
    target_field:             str   = "velocity"   # NEW
    epochs:                   int   = 100
    batch_size:               int   = 20
    lr:                       float = 1e-4
    noise_std:                float = 0.02
    early_stopping_patience:  int   = 10
    message_passing_steps:    int   = 15
    output_size:              int   = 2
    node_input_size:          int   = 11
    edge_input_size:          int   = 3
```

Update the domain size computation in `train_start()`:

```python
_DOMAIN_SIZES = {
    ("cylinder_flow", "velocity"):  {"output_size": 2, "node_input_size": 11, "edge_input_size": 3},
    ("cylinder_flow", "pressure"):  {"output_size": 1, "node_input_size": 10, "edge_input_size": 3},
    ("flag_simple",   "velocity"):  {"output_size": 3, "node_input_size": 12, "edge_input_size": 7},
}
key = (cfg.domain, cfg.target_field if cfg.domain == "cylinder_flow" else "velocity")
if key in _DOMAIN_SIZES:
    sizes = _DOMAIN_SIZES[key]
    cfg.output_size     = sizes["output_size"]
    cfg.node_input_size = sizes["node_input_size"]
    cfg.edge_input_size = sizes["edge_input_size"]
```

Also include `target_field` in the JSON config written to `runs/ui_train_config.json`:

The existing code writes `cfg.dict()` — this already includes `target_field` since it's a pydantic field.

- [ ] **Step 4: Commit**

```bash
git add api/state.py api/routes/train.py
git commit -m "feat: API train config supports target_field, computes sizes per field"
```

---

## Task 6: `api/routes/rollout.py` — pressure-aware rollout

**Files:**
- Modify: `api/routes/rollout.py`

The rollout must read `target_field` from checkpoint metadata and swap the correct field slice.

- [ ] **Step 1: Update `_run_rollout_sync()` to handle pressure**

Find where `predicted_velocity` is swapped into `graph.x`:

```python
if predicted_velocity is not None:
    graph.x[:, 1:3] = predicted_velocity.detach()
```

Replace with field-aware swap:

```python
# Read target_field from checkpoint (passed via cfg or loaded inline)
target_field = cfg.get("target_field", "velocity")
field_slice = slice(1, 2) if target_field == "pressure" else slice(1, 3)

if predicted_velocity is not None:
    graph.x[:, field_slice] = predicted_velocity.detach()
```

Update the variable name for clarity (optional — keeping `predicted_velocity` as the variable name is fine even in pressure mode since it plays the same structural role).

Also add `target_field` to DOMAINS config reads. Update `_run_rollout_sync` signature to receive `target_field` from `cfg`:

```python
target_field = cfg.get("target_field", "velocity")
```

Also read it from the loaded checkpoint to handle cached models that predate the config:

```python
ckpt_meta_path = cfg["checkpoint"]
if os.path.exists(ckpt_meta_path):
    try:
        import torch
        ckpt = torch.load(ckpt_meta_path, map_location="cpu", weights_only=False)
        target_field = ckpt.get("target_field", target_field)
    except Exception:
        pass
```

- [ ] **Step 2: Update pkl save to include `target_field` in metadata**

```python
with open(pkl_path, "wb") as f:
    pickle.dump([[predicted_arr, targets_arr], crds, {
        "confidence_score": confidence_score,
        "domain":           req.domain,
        "target_field":     target_field,
    }], f)
```

- [ ] **Step 3: Commit**

```bash
git add api/routes/rollout.py
git commit -m "feat: rollout API reads target_field from checkpoint, swaps correct slice"
```

---

## Task 7: `api/routes/results.py` — field-aware responses

**Files:**
- Modify: `api/routes/results.py`

Include `target_field` in result metadata. For pressure, `_compute_mae` uses scalar directly.

- [ ] **Step 1: Update `get_result()` to include `target_field`**

The `_load_pkl()` function already returns `meta` (from pressure plan Task 6 Step 2). In `get_result()`:

```python
predicted, targets, crds, meta = _load_pkl(filename)
target_field = meta.get("target_field", "velocity")
# ... existing code ...
return {
    # ... existing fields ...
    "target_field":     target_field,
    "confidence_score": meta.get("confidence_score", None),
    "confidence_label": _confidence_label(meta.get("confidence_score", None)),
    "domain":           meta.get("domain", "cylinder_flow"),
}
```

- [ ] **Step 2: Update `get_rmse()` to include `target_field`**

```python
predicted, targets, _, meta = _load_pkl(filename)
target_field = meta.get("target_field", "velocity")
# ...
return {
    # ... existing fields ...
    "target_field": target_field,
}
```

- [ ] **Step 3: Update `get_frame()` for pressure (scalar, no `norm()` needed)**

```python
predicted, targets, crds, meta = _load_pkl(filename)
target_field = meta.get("target_field", "velocity")
# ...
if target_field == "pressure":
    pred_mag   = predicted[t, :, 0]   # [N] — scalar pressure, no norm()
    target_mag = targets[t, :, 0]     # [N]
else:
    pred_mag   = np.linalg.norm(predicted[t], axis=-1)   # [N]
    target_mag = np.linalg.norm(targets[t],   axis=-1)   # [N]
error = np.abs(pred_mag - target_mag)   # [N]
# ...
```

- [ ] **Step 4: Commit**

```bash
git add api/routes/results.py
git commit -m "feat: results API includes target_field, pressure-aware magnitude computation"
```

---

## Task 8: Frontend — target field selector and labels

**Files:**
- Modify: `app/src/pages/Train.tsx`
- Modify: `app/src/pages/Predict.tsx`
- Modify: `app/src/pages/Visualize.tsx`

- [ ] **Step 1: Update `Train.tsx` — "Target field" dropdown**

Add a dropdown below the domain selector, shown only when domain is `cylinder_flow`:

```tsx
{selectedDomain === "cylinder_flow" && (
  <div className="mb-4">
    <label className="block text-sm font-medium text-gray-700 mb-1">
      Target field
    </label>
    <select
      value={targetField}
      onChange={(e) => setTargetField(e.target.value)}
      className="w-full border rounded px-3 py-2 text-sm"
    >
      <option value="velocity">Velocity — Predict fluid velocity field (DeepMind baseline)</option>
      <option
        value="pressure"
        disabled={!pressureDataAvailable}
      >
        Pressure — Predict pressure field{!pressureDataAvailable ? " (requires re-parsed data)" : ""}
      </option>
    </select>
  </div>
)}
```

Add state variable: `const [targetField, setTargetField] = useState("velocity");`

Include `target_field` in the train request body:
```tsx
body: JSON.stringify({
  domain: selectedDomain,
  target_field: targetField,  // NEW
  // ... other fields
})
```

Check `pressureDataAvailable` by fetching `/api/status` — the status endpoint can expose whether pressure `.dat` files exist. For simplicity, assume pressure is available when `cylinder_flow` data exists (user is responsible for re-parsing).

- [ ] **Step 2: Update `Predict.tsx` — show field badge**

After rollout completes, show a badge indicating the model type:

```tsx
{resultMeta?.target_field && (
  <span className={`px-2 py-0.5 rounded text-xs font-bold ml-2 ${
    resultMeta.target_field === "pressure"
      ? "bg-orange-100 text-orange-800"
      : "bg-blue-100 text-blue-800"
  }`}>
    {resultMeta.target_field === "pressure" ? "PRESSURE MODEL" : "VELOCITY MODEL"}
  </span>
)}
```

- [ ] **Step 3: Update `Visualize.tsx` — adapt labels**

```tsx
const fieldLabel = resultMeta?.target_field === "pressure"
  ? "Pressure (Pa)"
  : "Velocity Magnitude (m/s)";

const errorLabel = resultMeta?.target_field === "pressure"
  ? "Pressure Error (Pa)"
  : "Velocity Error (m/s)";

const yAxisLabel = resultMeta?.target_field === "pressure"
  ? "Pressure RMSE (Pa)"
  : "Velocity RMSE (m/s)";
```

Also in the Physics tab, show "Pressure time series" section when `target_field === "pressure"`:

```tsx
{resultMeta?.target_field === "pressure" && energyData && (
  <div className="mt-4">
    <h4 className="font-medium text-gray-700 mb-2">Pressure Field Energy</h4>
    {/* existing energy chart */}
  </div>
)}
```

- [ ] **Step 4: Verify TypeScript compiles**

```bash
cd /home/ahmealy/ML/meshGraphNets_pytorch/app
npm run build 2>&1 | tail -20
```

Expected: no TypeScript errors

- [ ] **Step 5: Commit**

```bash
git add app/src/pages/Train.tsx app/src/pages/Predict.tsx app/src/pages/Visualize.tsx
git commit -m "feat: frontend pressure field selector, model badge, adapted labels"
```

---

## Task 9: End-to-end smoke test

- [ ] **Step 1: Run all pressure tests**

```bash
cd /home/ahmealy/ML/meshGraphNets_pytorch
source venv/bin/activate
pytest tests/test_fpc_dataset.py tests/test_simulator_pressure.py -v
```

Expected: 10 PASSED

- [ ] **Step 2: Verify backward compatibility — load existing checkpoint**

```bash
python -c "
from api.state import get_model
import os
if os.path.exists('checkpoints/best_model.pth'):
    m = get_model('checkpoints/best_model.pth', 'cpu')
    print('Existing checkpoint loaded OK:', type(m).__name__)
    print('target_field:', m.target_field)
else:
    print('No existing checkpoint — skipping backward compat check')
"
```

Expected: `target_field: velocity` (defaults correctly for old checkpoints without this key)

- [ ] **Step 3: Verify all API imports clean**

```bash
python -c "
from api.routes.results import router as r1
from api.routes.rollout import router as r2
from api.routes.train  import router as r3
from api.state import DOMAINS
print('API imports OK')
print('cylinder_flow target_fields:', DOMAINS['cylinder_flow'].get('target_fields'))
"
```

Expected: `API imports OK` + `cylinder_flow target_fields: ['velocity', 'pressure']`

- [ ] **Step 4: Run full test suite**

```bash
pytest tests/ -v --ignore=tests/test_parse_pressure.py --ignore=tests/test_parse_flag.py 2>&1 | tail -30
```

Expected: All non-skip tests pass

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "feat: pressure field complete — parse, dataset, simulator, train, API, UI"
```
