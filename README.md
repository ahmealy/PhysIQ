# PhysIQ

**AI-powered physics simulation and inverse design — predict, visualise, and optimise engineering meshes in your browser.**

PhysIQ is a full-stack platform built on Graph Neural Networks (GNNs) that replaces slow finite-element solvers with fast neural surrogates. Run a fluid simulation in seconds instead of hours, then use the built-in inverse design engine to find the shape that hits your performance target.

---

## What it does

| Capability | Description |
|---|---|
| **Predict** | Run autoregressive GNN rollouts on CFD (cylinder flow) or cloth (flag) meshes — 10–100× faster than traditional solvers |
| **Generate** | Give a drag or stress target; the AI proposes candidate designs via CVAE sampling or gradient-descent optimisation |
| **Train** | Fine-tune the GNN on your own data, with live loss curves, remote GPU support, and architecture selection (GN / TNS / SAGE) |
| **Visualise** | Animated mesh viewer, per-step RMSE, physics diagnostics, 3D cloth viewer |
| **Dataset Studio** | Field distributions, outlier detection, node type breakdown, mesh quality metrics |
| **Training Similarity** | Latent-space KDTree score — tells you how close a new mesh is to the training distribution before you trust the prediction |

---

## Quick start

### 1. Install dependencies

```bash
# Python backend
pip install -r requirements.txt

# Frontend
cd app && npm install
```

### 2. Get the data

```bash
# CFD — cylinder flow (DeepMind dataset)
aria2c -x 8 https://storage.googleapis.com/dm-meshgraphnets/cylinder_flow/train.tfrecord -d data
aria2c -x 8 https://storage.googleapis.com/dm-meshgraphnets/cylinder_flow/test.tfrecord  -d data

# Parse to PyTorch memmap format (writes .dat.ok sentinel files on completion)
python parse_tfrecord.py

# To also parse pressure fields (needed for pressure-target training):
python parse_tfrecord.py  # creates train_pressure.dat, valid_pressure.dat, test_pressure.dat
```

### 3. Train

```bash
python train.py
# or via the UI — go to /train and click Start Training
```

### 4. Launch the UI

```bash
# Terminal 1 — backend
source venv/bin/activate
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 — frontend
cd app && npm run dev   # → http://localhost:5173
```

API docs at `http://localhost:8000/docs`.

---

## Remote GPU

Training and inference can be offloaded to a remote GPU over SSH. Configure it in **Train → Remote GPU**, or create the config manually:

```json
// runs/remote_gpu.json
{
  "host": "your-gpu-machine",
  "port": 22,
  "user": "you",
  "venv_python": "/path/to/venv/bin/python",
  "enabled": true
}
```

Requirements: SSH key auth, shared filesystem (NFS or same path on both machines), dependencies installed in the remote venv.

---

## Supported domains

| Domain | Physics | Input | Output |
|---|---|---|---|
| `cylinder_flow` | Incompressible CFD | Mesh + cylinder position + inlet velocity | Velocity / pressure field over time |
| `flag_simple` | Cloth dynamics | Mesh + handle positions | World position over time |

---

## Architecture

```
Browser (Vite dev server :5173)
  └── /api/*  →  FastAPI backend (:8000)
        ├── /train        — training + live log streaming
        ├── /rollout      — autoregressive inference
        ├── /generate     — CVAE + gradient-descent inverse design
        ├── /dataset      — statistics + mesh quality
        ├── /checkpoint   — model info
        └── /status       — GPU, training state, events
```

The GNN follows the encoder–processor–decoder pattern from [MeshGraphNets (Pfaff et al., ICLR 2021)](https://arxiv.org/abs/2010.03409), extended with:
- **TNS** (Transformer-based) and **SAGE** (GraphSAGE) processor variants selectable per training run
- CVAE-based inverse design with Latin Hypercube sampling and gradient-descent optimisation
- Compiled C++ KDTree for latent-space similarity scoring
- Gradient clipping for TNS/SAGE to prevent attention-layer explosion (GN unchanged)
- LRU model cache (max 3) to avoid repeated checkpoint deserialization

---

## Results

| Cylinder flow — trajectory 0 | Cylinder flow — trajectory 1 |
|---|---|
| ![Demo 0](videos/0.gif) | ![Demo 1](videos/1.gif) |

---

## Data pipeline

### Parse & train (first time setup)

```bash
# Parse TFRecord → .dat memmap (writes sentinel files on success)
python parse_tfrecord.py

# Optional: track data files with DVC
dvc add data/train.dat data/valid.dat data/test.dat
```

### Storage backends

Results are stored as `.pkl` by default. Switch to compressed HDF5 with partial-timestep reads:

```bash
# Create config
mkdir -p runs
echo '{"result_backend": "hdf5"}' > runs/storage_config.json

# Migrate existing PKL files
python scripts/migrate_pkl_to_hdf5.py --dry-run   # preview
python scripts/migrate_pkl_to_hdf5.py             # migrate
```

### Result retention

```bash
# Keep only the 10 most recent results
python -m result.retention --keep 10

# Preview without deleting
python -m result.retention --keep 10 --dry-run
```

### Ingest pipeline

For programmatic data ingestion (e.g. integrating a new solver):

```python
from ingest import IngestPipeline
from ingest.adapters.tfrecord import TFRecordAdapter

adapter  = TFRecordAdapter("data/", domain="cylinder_flow")
pipeline = IngestPipeline(adapter, out_dir="data/")
results  = pipeline.run()   # harvest → validate → normalise → write → index
```

---

## Project layout

```
api/            FastAPI routes (train, rollout, generate, dataset, status)
app/            React + Tailwind frontend
model/          GNN architecture (encoder, processor, decoder)
confidence/     Latent-space KDTree similarity index (C++ + pybind11)
storage/        Repository Pattern — PKL, HDF5, Zarr backends + StorageFactory
ingest/         Ingest pipeline — SolverAdapter Protocol, composable stages
result/         retention.py — result pruning CLI
scripts/        migrate_pkl_to_hdf5.py, regenerate_dat.py
train.py        Training entry point
rollout.py      Inference entry point
generate_ssh.py Remote GPU generate dispatch
tests/          pytest test suite (~60+ tests)
```
