# flag_simple Domain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `flag_simple` (3D cloth simulation) as a second domain alongside `cylinder_flow`, following DeepMind's cloth_model.py faithfully ported to PyTorch/PyG.

**Architecture:** A new `FlagDataset` loads parsed cloth TFRecords; a new `FlagSimulator` wraps the existing `EncoderProcesserDecoder` (unchanged except `output_size` becomes a constructor param) and handles Verlet integration; `train.py` and `rollout.py` dispatch on `domain` from config. The core GNN processes whatever normalized node features it receives — it does not know or care whether they represent CFD velocity or cloth position.

**Tech Stack:** PyTorch, PyTorch Geometric, NumPy, TensorFlow 1.x (parsing only), FastAPI, React/TypeScript

---

## File Map

| File | Action | Purpose |
|---|---|---|
| `model/model.py` | **Modify** line 94 | `output_size=2` hardcoded → `output_size` constructor param |
| `data/parse_flag_tfrecord.py` | **Create** | Parse flag_simple TFRecords → per-split `.npz` + `.dat` files |
| `dataset/flag_dataset.py` | **Create** | `FlagDataset` — loads parsed cloth data, returns per-timestep PyG graph |
| `model/flag_simulator.py` | **Create** | `FlagSimulator` — cloth-specific forward pass with Verlet integration |
| `train.py` | **Modify** | Domain-aware: reads `domain`, `output_size`, `node_input_size`, `edge_input_size` from config |
| `rollout.py` | **Modify** | Domain-aware: dispatches `_rollout_cloth` or `_rollout_cfd` based on domain |
| `api/state.py` | **Modify** | `get_model()` reads `output_size` from checkpoint metadata; `flag_simple` `available` probed at startup |
| `api/routes/rollout.py` | **Modify** | Cloth rollout: Verlet integration, saves `[T,N,3]` position arrays |
| `api/routes/train.py` | **Modify** | Pass `output_size`, `node_input_size`, `edge_input_size` in train config JSON |
| `app/src/pages/Train.tsx` | **Modify** | Enable `flag_simple` option when `data_flag/` exists |
| `app/src/pages/Predict.tsx` | **Modify** | Domain badge shows `CLOTH MODEL` or `CFD MODEL` |
| `app/src/pages/Visualize.tsx` | **Modify** | Project `world_pos[:, :2]` for cloth; "Position Error (m)" label |
| `app/src/pages/Dashboard.tsx` | **Modify** | Show flag_simple domain card with availability status |
| `tests/test_flag_simulator.py` | **Create** | Unit tests for FlagSimulator and FlagDataset |

---

## Task 1: Parameterize `output_size` in `EncoderProcesserDecoder`

**Files:**
- Modify: `model/model.py:81-94`
- Test: `tests/test_model_output_size.py`

This is a one-line change required by all subsequent tasks. The current `Decoder(output_size=2)` is hardcoded; it must be a parameter so cloth (output=3) and pressure (output=1) can reuse the same architecture.

- [ ] **Step 1: Write the failing test**

Create `tests/test_model_output_size.py`:

```python
import pytest
import torch
from torch_geometric.data import Data
from model.model import EncoderProcesserDecoder


def test_output_size_param_velocity():
    """EncoderProcesserDecoder with output_size=2 produces [N, 2] output."""
    model = EncoderProcesserDecoder(
        message_passing_num=2,
        node_input_size=11,
        edge_input_size=3,
        output_size=2,
    )
    N, E = 10, 20
    graph = Data(
        x=torch.randn(N, 11),
        edge_attr=torch.randn(E, 3),
        edge_index=torch.randint(0, N, (2, E)),
    )
    out = model(graph)
    assert out.shape == (N, 2)


def test_output_size_param_cloth():
    """EncoderProcesserDecoder with output_size=3 produces [N, 3] output."""
    model = EncoderProcesserDecoder(
        message_passing_num=2,
        node_input_size=12,
        edge_input_size=7,
        output_size=3,
    )
    N, E = 10, 20
    graph = Data(
        x=torch.randn(N, 12),
        edge_attr=torch.randn(E, 7),
        edge_index=torch.randint(0, N, (2, E)),
    )
    out = model(graph)
    assert out.shape == (N, 3)


def test_output_size_param_pressure():
    """EncoderProcesserDecoder with output_size=1 produces [N, 1] output."""
    model = EncoderProcesserDecoder(
        message_passing_num=2,
        node_input_size=10,
        edge_input_size=3,
        output_size=1,
    )
    N, E = 10, 20
    graph = Data(
        x=torch.randn(N, 10),
        edge_attr=torch.randn(E, 3),
        edge_index=torch.randint(0, N, (2, E)),
    )
    out = model(graph)
    assert out.shape == (N, 1)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/ahmealy/ML/meshGraphNets_pytorch
source venv/bin/activate
pytest tests/test_model_output_size.py -v
```

Expected: FAIL — `EncoderProcesserDecoder.__init__() got an unexpected keyword argument 'output_size'`

- [ ] **Step 3: Implement the change in `model/model.py`**

Replace lines 81–94 (the `EncoderProcesserDecoder.__init__`):

```python
class EncoderProcesserDecoder(nn.Module):

    def __init__(self, message_passing_num, node_input_size, edge_input_size,
                 hidden_size=128, output_size=2):

        super(EncoderProcesserDecoder, self).__init__()

        self.encoder = Encoder(edge_input_size=edge_input_size,
                               node_input_size=node_input_size,
                               hidden_size=hidden_size)

        processer_list = []
        for _ in range(message_passing_num):
            processer_list.append(GnBlock(hidden_size=hidden_size))
        self.processer_list = nn.ModuleList(processer_list)

        self.decoder = Decoder(hidden_size=hidden_size, output_size=output_size)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_model_output_size.py -v
```

Expected: 3 PASSED

- [ ] **Step 5: Verify existing `Simulator` still works (output_size defaults to 2)**

```bash
python -c "
from model.simulator import Simulator
s = Simulator(message_passing_num=2, node_input_size=11, edge_input_size=3, device='cpu')
print('Simulator OK, model decoder out:', s.model.decoder.decode_module[-1].out_features)
"
```

Expected output: `Simulator model initialized` + `Simulator OK, model decoder out: 2`

- [ ] **Step 6: Commit**

```bash
git add model/model.py tests/test_model_output_size.py
git commit -m "feat: parameterize output_size in EncoderProcesserDecoder"
```

---

## Task 2: Parse flag_simple TFRecords

**Files:**
- Create: `data/parse_flag_tfrecord.py`
- Test: `tests/test_parse_flag.py`

This requires `tensorflow<1.15` (same as `parse_tfrecord.py`) and the `data_flag/` directory with downloaded TFRecords. The test validates the output file structure without needing TFRecords by mocking the output.

Note: `flag_simple` TFRecords are NOT auto-downloaded. User must run:
```bash
bash meshgraphnets/download_dataset.sh flag_simple data_flag
```
before running this script.

- [ ] **Step 1: Create `data/parse_flag_tfrecord.py`**

```python
# -*- encoding: utf-8 -*-
"""
Parse flag_simple TFRecords into numpy arrays.

Usage:
    python data/parse_flag_tfrecord.py

Requires: tensorflow<1.15, data_flag/{train,valid,test}.tfrecord

Outputs per split:
    data_flag/{split}_pos.npz   — world_pos [T, N, 3] per trajectory, stacked
    data_flag/{split}_mesh.npz  — mesh_pos [N, 2], node_type [N, 1], cells [F, 3]
                                   (per trajectory, stored as ragged lists)
"""
import functools
import json
import os

import numpy as np
import tensorflow as tf
from packaging import version

if version.parse(tf.__version__) >= version.parse("1.15"):
    raise RuntimeError(
        f"TensorFlow {tf.__version__} found. This script requires tensorflow<1.15.\n"
        "Install in a separate env: pip install 'tensorflow<1.15'"
    )

DATA_DIR = "data_flag"


def _parse(proto, meta):
    """Parses a trajectory from tf.Example — same pattern as parse_tfrecord.py."""
    feature_lists = {k: tf.io.VarLenFeature(tf.string) for k in meta["field_names"]}
    features = tf.io.parse_single_example(proto, feature_lists)
    out = {}
    for key, field in meta["features"].items():
        data = tf.io.decode_raw(features[key].values, getattr(tf, field["dtype"]))
        data = tf.reshape(data, field["shape"])
        if field["type"] == "static":
            data = tf.tile(data, [meta["trajectory_length"], 1, 1])
        elif field["type"] == "dynamic_varlen":
            length = tf.io.decode_raw(features["length_" + key].values, tf.int32)
            length = tf.reshape(length, [-1])
            data = tf.RaggedTensor.from_row_lengths(data, row_lengths=length)
        elif field["type"] != "dynamic":
            raise ValueError("invalid data format: %s" % field["type"])
        out[key] = data
    return out


def load_dataset(split):
    with open(os.path.join(DATA_DIR, "meta.json")) as fp:
        meta = json.loads(fp.read())
    ds = tf.data.TFRecordDataset(os.path.join(DATA_DIR, split + ".tfrecord"))
    ds = ds.map(functools.partial(_parse, meta=meta), num_parallel_calls=1)
    ds = ds.prefetch(1)
    return ds


def parse_split(split: str):
    """Parse one split and write output files. Idempotent (skips if already exists)."""
    pos_path  = os.path.join(DATA_DIR, f"{split}_pos.npz")
    mesh_path = os.path.join(DATA_DIR, f"{split}_mesh.npz")

    if os.path.exists(pos_path) and os.path.exists(mesh_path):
        print(f"[{split}] Output files already exist, skipping.")
        return

    print(f"[{split}] Parsing...")
    ds = load_dataset(split)

    all_world_pos  = []  # list of [T, N, 3] per trajectory
    all_mesh_pos   = []  # list of [N, 2] per trajectory
    all_node_type  = []  # list of [N, 1] per trajectory
    all_cells      = []  # list of [F, 3] per trajectory

    for idx, d in enumerate(ds):
        world_pos  = d["world_pos"].numpy()   # [T, N, 3]
        mesh_pos   = d["mesh_pos"].numpy()[0]  # [N, 2] — static, use step 0
        node_type  = d["node_type"].numpy()[0] # [N, 1] — static, use step 0
        cells      = d["cells"].numpy()[0]     # [F, 3] — static, use step 0

        all_world_pos.append(world_pos)
        all_mesh_pos.append(mesh_pos)
        all_node_type.append(node_type)
        all_cells.append(cells)

        if idx % 10 == 0:
            print(f"  [{split}] Parsed {idx} trajectories...")

    print(f"[{split}] {len(all_world_pos)} trajectories parsed.")

    # Save world_pos as ragged (different N per trajectory possible)
    # Use object dtype arrays for ragged storage
    np.savez_compressed(
        pos_path,
        world_pos=np.array(all_world_pos, dtype=object),  # [n_traj] of [T, N, 3]
    )
    np.savez_compressed(
        mesh_path,
        mesh_pos=np.array(all_mesh_pos, dtype=object),    # [n_traj] of [N, 2]
        node_type=np.array(all_node_type, dtype=object),  # [n_traj] of [N, 1]
        cells=np.array(all_cells, dtype=object),          # [n_traj] of [F, 3]
    )
    print(f"[{split}] Saved to {pos_path} and {mesh_path}")


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)

    tf.enable_resource_variables()   # type: ignore
    tf.enable_eager_execution()      # type: ignore

    for split in ["train", "valid", "test"]:
        parse_split(split)

    print("Done. Re-run train.py with domain=flag_simple to use the parsed data.")
```

- [ ] **Step 2: Write test for parse output validation**

Create `tests/test_parse_flag.py`:

```python
"""
Test that parse_flag_tfrecord output files have the expected structure.
This test does NOT parse TFRecords (no TF needed) — it validates pre-existing output files.
Skip if data_flag/ doesn't exist.
"""
import os
import numpy as np
import pytest

DATA_DIR = "data_flag"


@pytest.mark.skipif(
    not os.path.exists(os.path.join(DATA_DIR, "train_pos.npz")),
    reason="data_flag/train_pos.npz not present — run parse_flag_tfrecord.py first",
)
def test_train_pos_structure():
    data = np.load(os.path.join(DATA_DIR, "train_pos.npz"), allow_pickle=True)
    world_pos = data["world_pos"]
    # Object array of trajectories
    assert world_pos.dtype == object
    assert len(world_pos) > 0
    # First trajectory: [T, N, 3]
    first = world_pos[0]
    assert first.ndim == 3
    assert first.shape[2] == 3   # 3D positions


@pytest.mark.skipif(
    not os.path.exists(os.path.join(DATA_DIR, "train_mesh.npz")),
    reason="data_flag/train_mesh.npz not present — run parse_flag_tfrecord.py first",
)
def test_train_mesh_structure():
    data = np.load(os.path.join(DATA_DIR, "train_mesh.npz"), allow_pickle=True)
    mesh_pos  = data["mesh_pos"]
    node_type = data["node_type"]
    cells     = data["cells"]

    assert mesh_pos.dtype == object
    first_mesh = mesh_pos[0]
    assert first_mesh.ndim == 2
    assert first_mesh.shape[1] == 2   # 2D rest coords

    first_cells = cells[0]
    assert first_cells.ndim == 2
    assert first_cells.shape[1] == 3  # triangles
```

- [ ] **Step 3: Run tests (both skip if data not present, that's fine)**

```bash
pytest tests/test_parse_flag.py -v
```

Expected: both SKIPPED (no data_flag yet) or PASSED (if data present)

- [ ] **Step 4: Commit**

```bash
git add data/parse_flag_tfrecord.py tests/test_parse_flag.py
git commit -m "feat: add parse_flag_tfrecord.py for flag_simple cloth data"
```

---

## Task 3: `FlagDataset` — load parsed cloth data

**Files:**
- Create: `dataset/flag_dataset.py`
- Test: `tests/test_flag_dataset.py`

The dataset returns a PyG `Data` object per timestep with the graph structure described in the spec. We test it with synthetic numpy arrays (no TFRecord dependency).

- [ ] **Step 1: Write the failing test**

Create `tests/test_flag_dataset.py`:

```python
import os
import numpy as np
import torch
import pytest
from torch_geometric.data import Data


def _make_fake_data(data_dir: str, n_traj: int = 2, T: int = 10, N: int = 15, F: int = 20):
    """Write synthetic flag data files matching the parse_flag_tfrecord.py output format."""
    os.makedirs(data_dir, exist_ok=True)
    world_pos = np.array([
        np.random.randn(T, N, 3).astype(np.float32) for _ in range(n_traj)
    ], dtype=object)
    np.savez_compressed(os.path.join(data_dir, "train_pos.npz"), world_pos=world_pos)

    mesh_pos  = np.array([np.random.randn(N, 2).astype(np.float32)  for _ in range(n_traj)], dtype=object)
    node_type = np.array([np.zeros((N, 1), dtype=np.int32)           for _ in range(n_traj)], dtype=object)
    cells     = np.array([np.random.randint(0, N, (F, 3)).astype(np.int64) for _ in range(n_traj)], dtype=object)
    np.savez_compressed(os.path.join(data_dir, "train_mesh.npz"),
                        mesh_pos=mesh_pos, node_type=node_type, cells=cells)
    return n_traj, T, N, F


def test_flag_dataset_length(tmp_path):
    data_dir = str(tmp_path)
    n_traj, T, N, F = _make_fake_data(data_dir)
    from dataset.flag_dataset import FlagDataset
    ds = FlagDataset(data_dir, split="train")
    # Each trajectory has T-1 timestep pairs
    assert len(ds) == n_traj * (T - 1)


def test_flag_dataset_item_shapes(tmp_path):
    data_dir = str(tmp_path)
    n_traj, T, N, F = _make_fake_data(data_dir)
    from dataset.flag_dataset import FlagDataset
    ds = FlagDataset(data_dir, split="train")
    graph = ds[0]
    assert isinstance(graph, Data)
    # graph.x: [N, 4] — concat(world_pos_t[3], node_type[1])
    assert graph.x.shape == (N, 4)
    # graph.prev_x: [N, 3] — world_pos_{t-1}
    assert graph.prev_x.shape == (N, 3)
    # graph.pos: [N, 2] — mesh_pos (2D rest configuration)
    assert graph.pos.shape == (N, 2)
    # graph.world_pos: [N, 3] — world_pos_t
    assert graph.world_pos.shape == (N, 3)
    # graph.face: [3, F] — triangle connectivity
    assert graph.face.shape == (3, F)
    # graph.y: [N, 3] — world_pos_{t+1} (target)
    assert graph.y.shape == (N, 3)


def test_flag_dataset_first_step_prev_equals_cur(tmp_path):
    """At t=0, prev_world_pos should equal world_pos_t (no history available)."""
    data_dir = str(tmp_path)
    _make_fake_data(data_dir)
    from dataset.flag_dataset import FlagDataset
    ds = FlagDataset(data_dir, split="train")
    graph = ds[0]
    # t=0: prev_x should equal world_pos
    assert torch.allclose(graph.prev_x, graph.world_pos)


def test_flag_dataset_second_step_has_history(tmp_path):
    """At t=1, prev_world_pos should equal world_pos at t=0."""
    data_dir = str(tmp_path)
    n_traj, T, N, F = _make_fake_data(data_dir, T=5)
    from dataset.flag_dataset import FlagDataset
    ds = FlagDataset(data_dir, split="train")
    graph_t0 = ds[0]   # t=0
    graph_t1 = ds[1]   # t=1
    # t=1: prev_x == world_pos at t=0
    assert torch.allclose(graph_t1.prev_x, graph_t0.world_pos)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_flag_dataset.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'dataset.flag_dataset'`

- [ ] **Step 3: Create `dataset/flag_dataset.py`**

```python
"""
FlagDataset — loads parsed flag_simple cloth simulation data.

Data files produced by data/parse_flag_tfrecord.py:
    {data_root}/{split}_pos.npz   — world_pos per trajectory (object array of [T, N, 3])
    {data_root}/{split}_mesh.npz  — mesh_pos, node_type, cells per trajectory

Returns a PyG Data object per (trajectory, timestep) pair with:
    graph.x         [N, 4]  — concat(world_pos_t[3], node_type[1])
    graph.prev_x    [N, 3]  — world_pos_{t-1}  (= world_pos_t at t=0)
    graph.pos       [N, 2]  — mesh_pos (2D rest configuration)
    graph.world_pos [N, 3]  — world_pos_t
    graph.face      [3, F]  — triangle connectivity (int64)
    graph.y         [N, 3]  — world_pos_{t+1} (regression target)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


class FlagDataset(Dataset):

    def __init__(self, data_root: str, split: str):
        pos_path  = os.path.join(data_root, f"{split}_pos.npz")
        mesh_path = os.path.join(data_root, f"{split}_mesh.npz")

        if not os.path.exists(pos_path):
            raise FileNotFoundError(
                f"Flag position data not found: {pos_path}\n"
                "Re-run: python data/parse_flag_tfrecord.py"
            )

        pos_data  = np.load(pos_path,  allow_pickle=True)
        mesh_data = np.load(mesh_path, allow_pickle=True)

        self.world_pos_list  = pos_data["world_pos"]      # [n_traj] of [T, N, 3]
        self.mesh_pos_list   = mesh_data["mesh_pos"]      # [n_traj] of [N, 2]
        self.node_type_list  = mesh_data["node_type"]     # [n_traj] of [N, 1]
        self.cells_list      = mesh_data["cells"]         # [n_traj] of [F, 3]

        self.n_traj = len(self.world_pos_list)
        # Each trajectory has T-1 timestep pairs (t, t+1)
        self.steps_per_traj = [arr.shape[0] - 1 for arr in self.world_pos_list]
        self.total_samples = sum(self.steps_per_traj)

        # Cumulative step counts for index mapping
        self._cum_steps = np.cumsum([0] + self.steps_per_traj)

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, index: int) -> Data:
        # Find which trajectory and which timestep
        traj_idx = int(np.searchsorted(self._cum_steps[1:], index, side="right"))
        t = index - self._cum_steps[traj_idx]  # local timestep within trajectory

        world_pos  = self.world_pos_list[traj_idx]    # [T, N, 3]
        mesh_pos   = self.mesh_pos_list[traj_idx]     # [N, 2]
        node_type  = self.node_type_list[traj_idx]    # [N, 1]
        cells      = self.cells_list[traj_idx]        # [F, 3]

        world_pos_t   = world_pos[t]        # [N, 3]
        world_pos_tp1 = world_pos[t + 1]    # [N, 3]
        # At t=0 there is no previous frame — use current as previous (zero velocity)
        world_pos_prev = world_pos[t - 1] if t > 0 else world_pos[t]  # [N, 3]

        # Node features: concat(world_pos_t, node_type) → [N, 4]
        x = np.concatenate([world_pos_t, node_type.astype(np.float32)], axis=-1)

        graph = Data(
            x          = torch.as_tensor(x.copy(),                  dtype=torch.float32),
            prev_x     = torch.as_tensor(world_pos_prev.copy(),     dtype=torch.float32),
            pos        = torch.as_tensor(mesh_pos.copy(),            dtype=torch.float32),
            world_pos  = torch.as_tensor(world_pos_t.copy(),        dtype=torch.float32),
            face       = torch.as_tensor(cells.T.copy(),             dtype=torch.int64),
            y          = torch.as_tensor(world_pos_tp1.copy(),      dtype=torch.float32),
        )
        return graph
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_flag_dataset.py -v
```

Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add dataset/flag_dataset.py tests/test_flag_dataset.py
git commit -m "feat: add FlagDataset for flag_simple cloth simulation"
```

---

## Task 4: `FlagSimulator` — cloth-specific forward pass

**Files:**
- Create: `model/flag_simulator.py`
- Test: `tests/test_flag_simulator.py`

`FlagSimulator` wraps `EncoderProcesserDecoder` with cloth-specific edge feature construction (mesh-space + world-space), Verlet integration at inference, and normalized acceleration target at training. This is the most complex new file.

Key formulas (from DeepMind cloth_model.py):
- Node features: `velocity = world_pos - prev_world_pos` → concat one_hot → [N, 12]
- Edge features: `[rel_mesh[2], |rel_mesh|[1], rel_world[3], |rel_world|[1]]` → [E, 7]
- Training target: `acc = world_pos_next - 2*world_pos + world_pos_prev` — [N, 3]
- Inference update: `world_pos_next = 2*world_pos - world_pos_prev + acc`

- [ ] **Step 1: Write the failing test**

Create `tests/test_flag_simulator.py`:

```python
import torch
import numpy as np
import pytest
from torch_geometric.data import Data


def _make_cloth_graph(N=20, F=30):
    """Synthetic cloth graph matching FlagDataset output format."""
    world_pos  = torch.randn(N, 3)
    prev_world = torch.randn(N, 3)
    mesh_pos   = torch.randn(N, 2)
    node_type  = torch.zeros(N, 1)  # all NORMAL

    # x: concat(world_pos, node_type)
    x = torch.cat([world_pos, node_type], dim=-1)  # [N, 4]

    # Triangular faces [3, F]
    face = torch.randint(0, N, (3, F))
    # Target: next world_pos
    y = torch.randn(N, 3)

    return Data(
        x=x,
        prev_x=prev_world,
        pos=mesh_pos,
        world_pos=world_pos,
        face=face,
        y=y,
    )


def test_flag_simulator_training_shapes():
    """Training forward pass returns (predicted_acc_norm, target_acc_norm) both [N, 3]."""
    from model.flag_simulator import FlagSimulator
    sim = FlagSimulator(message_passing_num=2, device="cpu")
    sim.train()

    graph = _make_cloth_graph(N=20, F=30)

    predicted, target = sim(graph)
    N = 20
    assert predicted.shape == (N, 3), f"Expected ({N}, 3), got {predicted.shape}"
    assert target.shape    == (N, 3), f"Expected ({N}, 3), got {target.shape}"


def test_flag_simulator_inference_shape():
    """Inference forward pass returns next world_pos [N, 3]."""
    from model.flag_simulator import FlagSimulator
    sim = FlagSimulator(message_passing_num=2, device="cpu")
    sim.eval()

    graph = _make_cloth_graph(N=20, F=30)
    with torch.no_grad():
        next_pos = sim(graph)
    assert next_pos.shape == (20, 3)


def test_flag_simulator_verlet_integration():
    """Inference: next_pos = 2*cur - prev + acc (Verlet)."""
    from model.flag_simulator import FlagSimulator
    sim = FlagSimulator(message_passing_num=2, device="cpu")
    sim.eval()

    graph = _make_cloth_graph(N=5, F=6)
    world_pos  = graph.world_pos.clone()
    prev_world = graph.prev_x.clone()

    with torch.no_grad():
        next_pos = sim(graph)

    # Manually compute what acc should be
    # next_pos = 2*world_pos - prev_world + acc  =>  acc = next_pos - 2*world + prev
    acc_implied = next_pos - 2 * world_pos + prev_world
    # acc must be finite (not NaN/Inf)
    assert torch.isfinite(acc_implied).all()
    assert torch.isfinite(next_pos).all()


def test_flag_simulator_node_features_size():
    """Node input size is exactly 12 (vel[3] + one_hot[9])."""
    from model.flag_simulator import FlagSimulator
    sim = FlagSimulator(message_passing_num=2, device="cpu")
    assert sim.node_input_size == 12


def test_flag_simulator_edge_features_size():
    """Edge input size is exactly 7."""
    from model.flag_simulator import FlagSimulator
    sim = FlagSimulator(message_passing_num=2, device="cpu")
    assert sim.edge_input_size == 7
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_flag_simulator.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'model.flag_simulator'`

- [ ] **Step 3: Create `model/flag_simulator.py`**

```python
"""
FlagSimulator — cloth physics simulator using Verlet integration.

Matches DeepMind cloth_model.py:
    node_input_size = 12  (velocity[3] + one_hot(node_type, 9)[9])
    edge_input_size = 7   (rel_mesh[2] + |rel_mesh|[1] + rel_world[3] + |rel_world|[1])
    output_size     = 3   (3D acceleration)

Training target: acc = world_pos_next - 2*world_pos + world_pos_prev   (Verlet)
Inference:       world_pos_next = 2*world_pos - world_pos_prev + acc
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.transforms import FaceToEdge

from .model import EncoderProcesserDecoder
from utils import normalization
from utils.utils import NodeType


def _init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class FlagSimulator(nn.Module):
    """
    Cloth simulator: wraps EncoderProcesserDecoder with Verlet integration.

    Node input:  12 = (world_pos_t - world_pos_{t-1})[3] + one_hot(node_type, 9)[9]
    Edge input:   7 = rel_mesh[2] + |rel_mesh|[1] + rel_world[3] + |rel_world|[1]
    Output:       3 = predicted acceleration in 3D
    """
    node_input_size: int = 12
    edge_input_size: int = 7
    output_size:     int = 3

    def __init__(self, message_passing_num: int = 15, device: str = "cpu") -> None:
        super(FlagSimulator, self).__init__()

        self.model = EncoderProcesserDecoder(
            message_passing_num=message_passing_num,
            node_input_size=self.node_input_size,
            edge_input_size=self.edge_input_size,
            output_size=self.output_size,
        ).to(device)

        self._output_normalizer = normalization.Normalizer(
            size=self.output_size, name="flag_output_normalizer", device=device
        )
        self._node_normalizer = normalization.Normalizer(
            size=self.node_input_size, name="flag_node_normalizer", device=device
        )
        self._edge_normalizer = normalization.Normalizer(
            size=self.edge_input_size, name="flag_edge_normalizer", device=device
        )

        self.model.apply(_init_weights)
        self._face_to_edge = FaceToEdge(remove_faces=False)
        print("FlagSimulator initialized")

    def _build_graph(self, graph: Data) -> Data:
        """
        Convert face-based graph to edge-based and build cloth edge features.

        Edge features [E, 7]:
            rel_mesh[2]   — relative 2D mesh-space position (sender - receiver)
            |rel_mesh|[1] — norm of rel_mesh
            rel_world[3]  — relative 3D world-space position
            |rel_world|[1]— norm of rel_world
        """
        # Apply FaceToEdge transform to get edge_index
        graph = self._face_to_edge(graph)
        edge_index = graph.edge_index  # [2, E]
        senders, receivers = edge_index[0], edge_index[1]

        mesh_pos  = graph.pos         # [N, 2]
        world_pos = graph.world_pos   # [N, 3]

        rel_mesh  = mesh_pos[senders]  - mesh_pos[receivers]    # [E, 2]
        mesh_norm = torch.norm(rel_mesh,  dim=-1, keepdim=True)  # [E, 1]
        rel_world = world_pos[senders] - world_pos[receivers]   # [E, 3]
        world_norm = torch.norm(rel_world, dim=-1, keepdim=True) # [E, 1]

        edge_attr = torch.cat([rel_mesh, mesh_norm, rel_world, world_norm], dim=-1)  # [E, 7]
        graph.edge_attr = edge_attr
        return graph

    def _build_node_features(self, graph: Data) -> torch.Tensor:
        """
        Node features [N, 12]:
            velocity[3]  = world_pos_t - world_pos_{t-1}
            one_hot[9]   = one_hot(node_type, num_classes=9)
        """
        world_pos  = graph.world_pos   # [N, 3] — current position (from graph.x[:, :3])
        prev_world = graph.prev_x      # [N, 3] — previous position

        velocity  = world_pos - prev_world                          # [N, 3]
        node_type = graph.x[:, 3:4].squeeze(-1).long()             # [N]
        one_hot   = F.one_hot(node_type, num_classes=9).float()    # [N, 9]
        node_feats = torch.cat([velocity, one_hot], dim=-1)         # [N, 12]
        return node_feats

    def forward(self, graph: Data):
        """
        Training:
            Returns (predicted_acc_norm, target_acc_norm) — both [N, 3].
            Loss should be MSE on NodeType.NORMAL nodes only.

        Inference (model.eval()):
            Returns next_world_pos [N, 3] via Verlet integration.
        """
        graph = self._build_graph(graph)

        world_pos  = graph.world_pos    # [N, 3]
        prev_world = graph.prev_x       # [N, 3]

        node_feats = self._build_node_features(graph)                             # [N, 12]
        graph.x    = self._node_normalizer(node_feats, self.training)             # [N, 12]
        graph.edge_attr = self._edge_normalizer(graph.edge_attr, self.training)   # [E, 7]

        predicted_acc_norm = self.model(graph)   # [N, 3]

        if self.training:
            # target_acc = world_pos_next - 2*world_pos + world_pos_prev  (Verlet)
            target_world = graph.y                                         # [N, 3]
            target_acc   = target_world - 2.0 * world_pos + prev_world    # [N, 3]
            target_acc_norm = self._output_normalizer(target_acc, self.training)
            return predicted_acc_norm, target_acc_norm
        else:
            # Denormalize and apply Verlet integration
            acc = self._output_normalizer.inverse(predicted_acc_norm)      # [N, 3]
            next_world_pos = 2.0 * world_pos - prev_world + acc            # [N, 3]
            return next_world_pos
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_flag_simulator.py -v
```

Expected: 5 PASSED

- [ ] **Step 5: Commit**

```bash
git add model/flag_simulator.py tests/test_flag_simulator.py
git commit -m "feat: add FlagSimulator with Verlet integration for cloth physics"
```

---

## Task 5: Domain-aware `train.py`

**Files:**
- Modify: `train.py`
- Test: none (integration — tested by running `python train.py --config ...`)

`train.py` must instantiate `FlagSimulator` or `Simulator` based on `cfg['domain']`, use `FlagDataset` or `FpcDataset` accordingly, apply correct loss mask (cloth uses NORMAL only, CFD uses NORMAL|OUTFLOW), and save `domain`/`node_input_size`/`output_size` in checkpoint metadata.

- [ ] **Step 1: Read the current train.py structure**

Current train.py uses hardcoded `node_input_size=11`, `edge_input_size=3`, and always `FpcDataset`. We must:
1. Add config keys `domain`, `output_size`, `node_input_size`, `edge_input_size`
2. Dispatch dataset and simulator based on domain
3. Adjust loss mask for cloth (NORMAL only vs CFD NORMAL|OUTFLOW)
4. Save domain metadata to checkpoint

- [ ] **Step 2: Update `train.py` defaults and imports**

Replace the `_defaults` dict and imports section (lines 1–20) with:

```python
import sys
# Force line-buffered stdout so every print() flushes immediately to the log file.
sys.stdout.reconfigure(line_buffering=True)

import torch
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
import numpy as np
import os
import json
import argparse
import tqdm
from torch.utils.tensorboard.writer import SummaryWriter

# ── Config: defaults, overridable via --config JSON ───────────────────────────
_defaults = dict(
    domain                   = 'cylinder_flow',
    dataset_dir              = 'data',
    batch_size               = 20,
    noise_std                = 2e-2,
    num_epochs               = 100,
    early_stopping_patience  = 10,
    lr                       = 1e-4,
    message_passing_num      = 15,
    checkpoint_dir           = 'checkpoints',
    log_dir                  = 'runs',
    # Derived from domain if not provided:
    output_size              = None,   # 2 = velocity, 1 = pressure, 3 = cloth
    node_input_size          = None,   # 11 = CFD velocity, 10 = CFD pressure, 12 = cloth
    edge_input_size          = None,   # 3 = CFD, 7 = cloth
)
```

- [ ] **Step 3: Update simulator and dataset instantiation in `train.py`**

Replace the `simulator = Simulator(...)` block and `transformer` block with:

```python
# ── Domain defaults ────────────────────────────────────────────────────────────
_DOMAIN_DEFAULTS = {
    'cylinder_flow': dict(output_size=2, node_input_size=11, edge_input_size=3),
    'flag_simple':   dict(output_size=3, node_input_size=12, edge_input_size=7),
}
domain = cfg['domain']
if domain not in _DOMAIN_DEFAULTS:
    raise ValueError("Unknown domain: %s. Valid: cylinder_flow, flag_simple" % domain)

# Fill in derived sizes if not explicitly provided
for key, val in _DOMAIN_DEFAULTS[domain].items():
    if cfg.get(key) is None:
        cfg[key] = val

output_size     = cfg['output_size']
node_input_size = cfg['node_input_size']
edge_input_size = cfg['edge_input_size']

# ── Simulator ─────────────────────────────────────────────────────────────────
if domain == 'flag_simple':
    from model.flag_simulator import FlagSimulator
    simulator = FlagSimulator(
        message_passing_num=cfg['message_passing_num'],
        device=device
    )
    transformer = None  # FlagSimulator builds edges internally
else:
    from model.simulator import Simulator
    simulator = Simulator(
        message_passing_num=cfg['message_passing_num'],
        node_input_size=node_input_size,
        edge_input_size=edge_input_size,
        device=device
    )
    transformer = T.Compose([
        T.FaceToEdge(),
        T.Cartesian(norm=False),
        T.Distance(norm=False)
    ])
```

- [ ] **Step 4: Update dataset loading in `if __name__ == '__main__':` block**

```python
if domain == 'flag_simple':
    from dataset.flag_dataset import FlagDataset
    train_dataset = FlagDataset(data_root=cfg['dataset_dir'], split='train')
    valid_dataset = FlagDataset(data_root=cfg['dataset_dir'], split='valid')
else:
    from dataset import FpcDataset
    train_dataset = FpcDataset(data_root=dataset_dir, split='train')
    valid_dataset = FpcDataset(data_root=dataset_dir, split='valid')
```

- [ ] **Step 5: Update `train_one_epoch` and `evaluate` to be domain-aware**

Replace both functions:

```python
def train_one_epoch(model, dataloader, optimizer, transformer, device, noise_std, domain):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for graph in tqdm.tqdm(dataloader):
        if transformer is not None:
            graph = transformer(graph)
        graph = graph.to(device)

        if domain == 'flag_simple':
            predicted_acc, target_acc = model(graph)
            # Cloth: loss on NORMAL nodes only (no OUTFLOW in cloth domain)
            from utils.utils import NodeType
            node_type = graph.x[:, 3].long()  # node_type is 4th feature in cloth
            mask = (node_type == NodeType.NORMAL)
        else:
            from utils.noise import get_velocity_noise
            from utils.utils import NodeType
            velocity_sequence_noise = get_velocity_noise(graph, noise_std=noise_std, device=device)
            predicted_acc, target_acc = model(graph, velocity_sequence_noise)
            node_type = graph.x[:, 0]
            mask = torch.logical_or(node_type == NodeType.NORMAL, node_type == NodeType.OUTFLOW)

        errors = ((predicted_acc - target_acc) ** 2)[mask]
        loss = torch.mean(errors)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def evaluate(model, dataloader, transformer, device, domain):
    model.eval()
    losses = []

    with torch.no_grad():
        for graph in dataloader:
            if transformer is not None:
                graph = transformer(graph)
            graph = graph.to(device)

            if domain == 'flag_simple':
                from utils.utils import NodeType
                # In eval mode, FlagSimulator returns next_world_pos
                next_world_pos = model(graph)
                node_type = graph.x[:, 3].long()
                mask = (node_type == NodeType.NORMAL)
                # Position RMSE
                errors = ((next_world_pos - graph.y) ** 2)[mask]
            else:
                from utils.utils import NodeType
                predicted_velocity = model(graph, None)
                node_type = graph.x[:, 0]
                mask = torch.logical_or(node_type == NodeType.NORMAL, node_type == NodeType.OUTFLOW)
                errors = ((predicted_velocity - graph.y) ** 2)[mask]

            loss = torch.sqrt(torch.mean(errors))
            losses.append(loss.item())

    return np.mean(losses)
```

- [ ] **Step 6: Update the checkpoint save to include domain metadata**

Replace the `torch.save(...)` call inside the training loop:

```python
torch.save({
    'epoch':                epoch,
    'model_state_dict':     simulator.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'valid_loss':           valid_loss,
    'domain':               domain,
    'output_size':          output_size,
    'node_input_size':      node_input_size,
    'edge_input_size':      edge_input_size,
}, checkpoint_path)
```

- [ ] **Step 7: Update function call sites** (train_one_epoch and evaluate calls)

```python
train_loss = train_one_epoch(simulator, train_loader, optimizer, transformer, device, noise_std, domain)
valid_loss = evaluate(simulator, valid_loader, transformer, device, domain)
```

- [ ] **Step 8: Verify default cylinder_flow training still parses**

```bash
python -c "
import sys
sys.argv = ['train.py']
# Just check imports and config parsing without running training
import train
print('domain:', train.domain)
print('node_input_size:', train.node_input_size)
print('edge_input_size:', train.edge_input_size)
"
```

Expected:
```
domain: cylinder_flow
node_input_size: 11
edge_input_size: 3
```

- [ ] **Step 9: Commit**

```bash
git add train.py
git commit -m "feat: make train.py domain-aware (cylinder_flow + flag_simple)"
```

---

## Task 6: Domain-aware `rollout.py`

**Files:**
- Modify: `rollout.py`

The command-line rollout script needs a `--domain` flag and must dispatch the cloth rollout (Verlet, saves world_pos [T,N,3]) vs CFD rollout (velocity update, saves vel [T,N,2]).

- [ ] **Step 1: Add `--domain` argument and cloth rollout function to `rollout.py`**

Add at the top (after existing imports):

```python
from model.flag_simulator import FlagSimulator
from dataset.flag_dataset import FlagDataset
```

Add a new function after the existing `rollout()` function:

```python
@torch.no_grad()
def rollout_cloth(model: FlagSimulator, dataset: FlagDataset,
                  rollout_index: int = 0, device: str = "cpu"):
    """
    Autoregressive rollout for cloth (flag_simple).
    Uses Verlet integration: next_pos = 2*cur - prev + acc.
    """
    steps_per_traj = dataset.steps_per_traj[rollout_index]
    predicteds = []
    targets    = []
    prev_world = None
    cur_world  = None

    t_start = time.perf_counter()

    for i in tqdm(range(steps_per_traj), desc="Rollout cloth trajectory %d" % rollout_index):
        # Compute global index for this trajectory's step i
        cum = dataset._cum_steps
        idx = cum[rollout_index] + i
        graph = dataset[idx]
        graph = graph.to(device)

        if cur_world is not None:
            # Swap in our predictions for autoregressive rollout
            graph.world_pos = cur_world.detach()
            graph.x[:, :3]  = cur_world.detach()
            graph.prev_x    = prev_world.detach()

        prev_world = graph.world_pos.clone()
        next_world = model(graph)   # [N, 3]

        # Pin HANDLE nodes to ground truth
        from utils.utils import NodeType
        node_type = graph.x[:, 3].long()
        handle_mask = (node_type == NodeType.HANDLE)
        next_world[handle_mask] = graph.y[handle_mask]

        predicteds.append(next_world.detach().cpu().numpy())
        targets.append(graph.y.detach().cpu().numpy())
        cur_world = next_world

    elapsed = time.perf_counter() - t_start
    predicted_arr = np.stack(predicteds)   # [T, N, 3]
    targets_arr   = np.stack(targets)       # [T, N, 3]

    # Position RMSE
    sq = np.square(predicted_arr - targets_arr).reshape(steps_per_traj, -1)
    per_step_rmse = np.sqrt(np.mean(sq, axis=1))
    for step in range(0, steps_per_traj, 50):
        print("rollout position rmse @ step %d: %.2e" % (step, per_step_rmse[step]))

    mesh_pos = dataset.mesh_pos_list[rollout_index]  # [N, 2] rest coords
    os.makedirs("result", exist_ok=True)
    pkl_path = "result/flag_result%d.pkl" % rollout_index
    with open(pkl_path, "wb") as f:
        pickle.dump([[predicted_arr, targets_arr], mesh_pos,
                     {"domain": "flag_simple"}], f)
    print("Result saved to %s" % pkl_path)
    return [predicted_arr, targets_arr], mesh_pos, elapsed
```

- [ ] **Step 2: Update `if __name__ == '__main__':` to dispatch on domain**

Add `--domain` arg and dispatch:

```python
parser.add_argument('--domain', type=str, default='cylinder_flow',
                    choices=['cylinder_flow', 'flag_simple'])
args = parser.parse_args()

device = 'cuda:%d' % args.gpu if torch.cuda.is_available() else 'cpu'
torch.cuda.set_device(args.gpu) if torch.cuda.is_available() else None

if args.domain == 'flag_simple':
    model = FlagSimulator(message_passing_num=15, device=device)
    ckpt = torch.load(args.model_dir, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    dataset = FlagDataset('data_flag', split=args.test_split)
    for i in range(args.rollout_num):
        result, mesh_pos, elapsed = rollout_cloth(model, dataset, rollout_index=i, device=device)
else:
    # existing CFD path — unchanged
    ...
```

- [ ] **Step 3: Verify rollout.py parses without error**

```bash
python -c "import rollout; print('rollout.py OK')"
```

Expected: `rollout.py OK`

- [ ] **Step 4: Commit**

```bash
git add rollout.py
git commit -m "feat: add domain-aware rollout with cloth Verlet integration"
```

---

## Task 7: Update `api/state.py` — cloth-aware `get_model`

**Files:**
- Modify: `api/state.py`

`get_model()` currently hardcodes `node_input_size=11, edge_input_size=3`. It must read these from checkpoint metadata and instantiate `FlagSimulator` when `domain=='flag_simple'`. Also set `available=True` for `flag_simple` when `data_flag/` exists.

- [ ] **Step 1: Update `get_model()` in `api/state.py`**

Replace the current `get_model()` function:

```python
def get_model(checkpoint_path: str, device: str):
    """
    Load and cache the correct Simulator based on checkpoint metadata.
    Reads domain/node_input_size/output_size from checkpoint to select simulator class.
    """
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    key = (checkpoint_path, device)
    if key not in _model_cache:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        domain = ckpt.get("domain", "cylinder_flow")

        if domain == "flag_simple":
            from model.flag_simulator import FlagSimulator
            sim = FlagSimulator(
                message_passing_num=15,
                device=device,
            )
        else:
            from model.simulator import Simulator
            node_input_size = ckpt.get("node_input_size", 11)
            edge_input_size = ckpt.get("edge_input_size", 3)
            sim = Simulator(
                message_passing_num=15,
                node_input_size=node_input_size,
                edge_input_size=edge_input_size,
                device=device,
            )

        sim.load_state_dict(ckpt["model_state_dict"])
        sim.eval()
        _model_cache[key] = sim
    return _model_cache[key]
```

- [ ] **Step 2: Add startup check for `flag_simple` data availability**

At the bottom of `api/state.py`, add a dynamic availability check:

```python
def _probe_flag_available() -> bool:
    """Check if flag_simple data has been parsed and is ready."""
    return os.path.exists("data_flag/train_pos.npz")

# Update flag_simple availability at import time
DOMAINS["flag_simple"]["available"] = _probe_flag_available()
```

- [ ] **Step 3: Verify import works**

```bash
python -c "from api.state import DOMAINS, get_model; print(DOMAINS['flag_simple']['available'])"
```

Expected: `False` (no data_flag yet) or `True` if data present

- [ ] **Step 4: Commit**

```bash
git add api/state.py
git commit -m "feat: api/state.py — cloth-aware get_model, dynamic flag_simple availability"
```

---

## Task 8: Update `api/routes/rollout.py` — cloth rollout endpoint

**Files:**
- Modify: `api/routes/rollout.py`

The API rollout must handle cloth: different dataset type, Verlet integration, save `[T,N,3]` positions with domain metadata.

- [ ] **Step 1: Add cloth dataset cache and import to `api/routes/rollout.py`**

At the top of the file, add:

```python
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
```

Add to the imports:

```python
from api.state import DOMAINS, get_model
```

Add a `_get_cloth_dataset` function:

```python
def _get_cloth_dataset(data_dir: str, split: str):
    from dataset.flag_dataset import FlagDataset
    key = (data_dir, split, "cloth")
    if key not in _dataset_cache:
        _dataset_cache[key] = FlagDataset(data_dir, split=split)
    return _dataset_cache[key]
```

- [ ] **Step 2: Add `_run_cloth_rollout_sync()` function**

```python
def _run_cloth_rollout_sync(req: RolloutRequest, cfg: dict, device: str,
                             progress_callback) -> dict:
    """Cloth (flag_simple) rollout using Verlet integration."""
    from utils.utils import NodeType

    model = get_model(cfg["checkpoint"], device)
    dataset = _get_cloth_dataset(cfg["data_dir"], req.split)

    n_traj = dataset.n_traj
    if req.trajectory_index >= n_traj:
        raise ValueError("trajectory_index %d out of range (0-%d)" % (
            req.trajectory_index, n_traj - 1))

    n_steps = dataset.steps_per_traj[req.trajectory_index]
    predicteds, targets_list = [], []
    prev_world = None
    cur_world  = None

    t_start = time.perf_counter()

    with torch.no_grad():
        for i in range(n_steps):
            cum = dataset._cum_steps
            idx = cum[req.trajectory_index] + i
            graph = dataset[idx]
            graph = graph.to(device)

            if cur_world is not None:
                graph.world_pos = cur_world.detach()
                graph.x[:, :3]  = cur_world.detach()
                graph.prev_x    = prev_world.detach()

            prev_world = graph.world_pos.clone()
            next_world = model(graph)  # [N, 3]

            # Pin HANDLE nodes to ground truth
            node_type = graph.x[:, 3].long()
            handle_mask = (node_type == NodeType.HANDLE)
            next_world[handle_mask] = graph.y[handle_mask]

            predicteds.append(next_world.detach().cpu().numpy())
            targets_list.append(graph.y.detach().cpu().numpy())
            cur_world = next_world

            if i % 20 == 0 or i == n_steps - 1:
                progress_callback(i + 1, n_steps)

    elapsed = time.perf_counter() - t_start
    sim_time = n_steps * cfg["dt"]
    speedup = sim_time / elapsed

    predicted_arr = np.stack(predicteds)  # [T, N, 3]
    targets_arr   = np.stack(targets_list)  # [T, N, 3]
    mesh_pos = dataset.mesh_pos_list[req.trajectory_index]  # [N, 2]

    sq = np.square(predicted_arr - targets_arr).reshape(n_steps, -1)
    per_step_rmse = np.sqrt(np.mean(sq, axis=1))

    os.makedirs("result", exist_ok=True)
    pkl_path = "result/flag_result%d.pkl" % req.trajectory_index
    with open(pkl_path, "wb") as f:
        pickle.dump([[predicted_arr, targets_arr], mesh_pos,
                     {"domain": "flag_simple"}], f)

    return {
        "elapsed_seconds": round(elapsed, 3),
        "speedup":         round(speedup, 2),
        "pkl_path":        pkl_path,
        "rmse_final":      float(per_step_rmse[-1]),
        "similarity_score": None,
    }
```

- [ ] **Step 3: Update `run_rollout()` to dispatch cloth vs CFD**

In the `run_rollout` endpoint, replace the `_run_rollout_sync` call with:

```python
run_fn = _run_cloth_rollout_sync if req.domain == "flag_simple" else _run_rollout_sync
future = loop.run_in_executor(None, run_fn, req, cfg, device, progress_callback)
```

- [ ] **Step 4: Verify the API imports cleanly**

```bash
python -c "from api.routes.rollout import router; print('rollout router OK')"
```

Expected: `rollout router OK`

- [ ] **Step 5: Commit**

```bash
git add api/routes/rollout.py
git commit -m "feat: api rollout endpoint handles flag_simple cloth domain"
```

---

## Task 9: Update `api/routes/train.py` — pass cloth config

**Files:**
- Modify: `api/routes/train.py`

The train endpoint must include `domain`, `output_size`, `node_input_size`, `edge_input_size` in the JSON config written to `runs/ui_train_config.json`.

- [ ] **Step 1: Update `TrainConfig` pydantic model in `api/routes/train.py`**

Add fields to `TrainConfig`:

```python
class TrainConfig(BaseModel):
    domain:                   str   = "cylinder_flow"
    epochs:                   int   = 100
    batch_size:               int   = 20
    lr:                       float = 1e-4
    noise_std:                float = 0.02
    early_stopping_patience:  int   = 10
    message_passing_steps:    int   = 15
    # Derived from domain server-side — not required from UI
    output_size:              int   = 2
    node_input_size:          int   = 11
    edge_input_size:          int   = 3
```

- [ ] **Step 2: In `train_start()`, compute domain-derived sizes before writing config**

Find the section that writes the config JSON and add before it:

```python
_DOMAIN_SIZES = {
    "cylinder_flow": {"output_size": 2, "node_input_size": 11, "edge_input_size": 3},
    "flag_simple":   {"output_size": 3, "node_input_size": 12, "edge_input_size": 7},
}
if cfg.domain in _DOMAIN_SIZES:
    sizes = _DOMAIN_SIZES[cfg.domain]
    cfg.output_size     = sizes["output_size"]
    cfg.node_input_size = sizes["node_input_size"]
    cfg.edge_input_size = sizes["edge_input_size"]
```

- [ ] **Step 3: Commit**

```bash
git add api/routes/train.py
git commit -m "feat: train API endpoint passes domain sizes to train.py config"
```

---

## Task 10: Frontend — enable flag_simple domain

**Files:**
- Modify: `app/src/pages/Train.tsx`
- Modify: `app/src/pages/Predict.tsx`
- Modify: `app/src/pages/Visualize.tsx`
- Modify: `app/src/pages/Dashboard.tsx`

- [ ] **Step 1: Update `Train.tsx` — domain selector shows flag_simple when available**

Find the domain selector section in `Train.tsx`. Add a `flag_simple` option that is disabled when the domain `available` field is false:

```tsx
// In the domain selector, check DOMAINS from /status endpoint
// The existing selector already fetches domain status — add flag_simple option:
<option value="flag_simple" disabled={!flagAvailable}>
  Flag Simple (Cloth) {!flagAvailable ? "— data not found" : ""}
</option>
```

The `flagAvailable` value comes from the `/status` endpoint `domains.flag_simple.available`.

- [ ] **Step 2: Update `Predict.tsx` — show domain badge**

After the rollout completes (in the SSE `done` handler), show a badge indicating the model type:

```tsx
{domain && (
  <span className={`px-2 py-0.5 rounded text-xs font-bold ${
    domain === "flag_simple"
      ? "bg-purple-100 text-purple-800"
      : "bg-blue-100 text-blue-800"
  }`}>
    {domain === "flag_simple" ? "CLOTH MODEL" : "CFD MODEL"}
  </span>
)}
```

- [ ] **Step 3: Update `Visualize.tsx` — handle cloth result pkls**

In the frame data handler, check `domain` from the result metadata. For cloth, the `predicted_magnitude` is position magnitude (not velocity), so just update the label:

```tsx
const fieldLabel = resultMeta?.domain === "flag_simple"
  ? "Position Magnitude (m)"
  : "Velocity Magnitude (m/s)";

const errorLabel = resultMeta?.domain === "flag_simple"
  ? "Position Error (m)"
  : "Velocity Error (m/s)";
```

- [ ] **Step 4: Verify TypeScript compiles**

```bash
cd /home/ahmealy/ML/meshGraphNets_pytorch/app
npm run build 2>&1 | tail -20
```

Expected: no TypeScript errors

- [ ] **Step 5: Commit**

```bash
git add app/src/pages/Train.tsx app/src/pages/Predict.tsx \
        app/src/pages/Visualize.tsx app/src/pages/Dashboard.tsx
git commit -m "feat: frontend flag_simple domain support — selector, badge, field labels"
```

---

## Task 11: Self-test and integration smoke test

- [ ] **Step 1: Run all tests**

```bash
cd /home/ahmealy/ML/meshGraphNets_pytorch
source venv/bin/activate
pytest tests/test_model_output_size.py tests/test_flag_dataset.py tests/test_flag_simulator.py -v
```

Expected: 12 PASSED total

- [ ] **Step 2: Verify default cylinder_flow train config still works**

```bash
python -c "
import json, sys
sys.argv = ['train.py', '--config', '/dev/stdin']
# Write minimal CFD config to /tmp
with open('/tmp/test_cfd.json', 'w') as f:
    json.dump({'domain': 'cylinder_flow', 'num_epochs': 1}, f)
import importlib.util
spec = importlib.util.spec_from_file_location('train', 'train.py')
# Just check the config loads
import train
print('domain:', train.domain, 'OK')
print('node_input_size:', train.node_input_size, 'OK')
"
```

- [ ] **Step 3: Verify API starts cleanly**

```bash
python -c "
from api.main import app
from api.state import DOMAINS, get_model
print('DOMAINS:', list(DOMAINS.keys()))
print('flag_simple available:', DOMAINS['flag_simple']['available'])
"
```

Expected: `DOMAINS: ['cylinder_flow', 'flag_simple']`

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "feat: flag_simple cloth domain complete — dataset, simulator, train, rollout, API, UI"
```
