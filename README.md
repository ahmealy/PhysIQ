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
| **Visualise** | Animated mesh viewer, per-step RMSE, physics diagnostics (vorticity, energy conservation, divergence), 3D cloth viewer |
| **Dataset Studio** | Field distributions, outlier detection, node type breakdown, mesh quality metrics |
| **Training Similarity** | Latent-space KDTree score — tells you how close a new mesh is to the training distribution before you trust the prediction |

---

## Demo

[![PhysIQ Demo](https://img.youtube.com/vi/Ke9_Fz0Nj8M/maxresdefault.jpg)](https://www.youtube.com/watch?v=Ke9_Fz0Nj8M)

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

#### CFD — cylinder flow

```bash
# Download DeepMind TFRecord files
aria2c -x 8 https://storage.googleapis.com/dm-meshgraphnets/cylinder_flow/train.tfrecord -d data
aria2c -x 8 https://storage.googleapis.com/dm-meshgraphnets/cylinder_flow/valid.tfrecord -d data
aria2c -x 8 https://storage.googleapis.com/dm-meshgraphnets/cylinder_flow/test.tfrecord  -d data

# Parse to PyTorch memmap format
# Creates: data/{split}.dat (velocity), data/{split}_pressure.dat, data/{split}.npz (indices)
# Writes .dat.ok sentinel files on success
python parse_tfrecord.py
```

#### Cloth — flag simple

```bash
# Download DeepMind TFRecord files
aria2c -x 8 https://storage.googleapis.com/dm-meshgraphnets/flag_simple/train.tfrecord -d data_flag/raw
aria2c -x 8 https://storage.googleapis.com/dm-meshgraphnets/flag_simple/valid.tfrecord -d data_flag/raw
aria2c -x 8 https://storage.googleapis.com/dm-meshgraphnets/flag_simple/test.tfrecord  -d data_flag/raw

# Parse to per-trajectory .npz files
# Creates: data_flag/{split}/traj_NNNNN.npz  (1,000 train / 100 valid / 100 test)
python parse_flag_tfrecord.py
```

Both datasets have **1,000 training / 100 validation / 100 test trajectories**, matching the original DeepMind MeshGraphNets paper splits.

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

## Docker

The quickest way to run PhysIQ without setting up Python/Node dependencies manually.

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) + [Docker Compose](https://docs.docker.com/compose/)
- Data already parsed (`.dat` / `.npz` files in `data/` and `data_flag/`) — parsing requires TensorFlow, do this on the host first

### Production (API + frontend via nginx)

```bash
docker compose up --build
```

- Frontend → http://localhost:80
- API docs → http://localhost:8000/docs

### Development (hot reload)

```bash
# API with hot reload + dev frontend (Vite HMR)
docker compose --profile dev up --build
```

- Frontend (Vite) → http://localhost:5173
- API → http://localhost:8000

### Volumes

Data, checkpoints, and results are mounted from the host — they persist across container restarts and are accessible from both host and container:

| Host path | Container path | Purpose |
|---|---|---|
| `./data` | `/app/data` | CFD dataset (.dat memmap files) |
| `./data_flag` | `/app/data_flag` | Cloth dataset (.npz per trajectory) |
| `./checkpoints` | `/app/checkpoints` | Trained model checkpoints |
| `./result` | `/app/result` | Rollout result PKL/HDF5 files |
| `./runs` | `/app/runs` | Training logs, configs, embedding index |

### GPU note

The Docker containers use **CPU-only PyTorch**. GPU training/inference uses the existing SSH remote GPU feature — configure it in Train → Remote GPU as usual. The API container handles data management and serves results; the heavy compute stays on your GPU machine.

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

- **Three processor variants** selectable per training run:
  - `gn` — Graph Network (default): custom EdgeBlock + NodeBlock MLPs, full edge feature updates every layer
  - `tns` — Graph Transformer: `TransformerConv` with 4-head scaled dot-product attention, edge features in Key + Value
  - `sage` — GraphSAGE: mean-aggregation `SAGEConv`, degree-normalised, L2-normalised output
- **Gradient clipping** (`max_norm=1.0`) and **reduced LR cap** (`3e-5`) for TNS/SAGE to prevent attention-layer divergence (GN unchanged at `1e-4`)
- **CVAE-based inverse design** with Latin Hypercube Sampling and gradient-descent optimisation in latent space
- **Compiled C++ KDTree** (pybind11, pool allocator) for latent-space similarity scoring with FAISS → C++ → scipy fallback chain
- **Optional Poisson pressure correction**: sparse LU factorisation of graph Laplacian, factorised once per rollout, `lu.solve` called at every timestep to enforce incompressibility
- **LRU model cache** (max 3) to avoid repeated checkpoint deserialisation

---

## Physics diagnostics

After a rollout, the following diagnostics are computed and shown alongside the animation:

| Diagnostic | Formula | What it tells you |
|---|---|---|
| **Vorticity** | `ω = ∂vy/∂x − ∂vx/∂y` | Fluid rotation per node — shows Kármán vortex street behind cylinder |
| **Kinetic energy** | `E(t) = 0.5 · Σ ‖vᵢ‖²` | Total energy over time — energy drift reveals GNN dissipation/blow-up |
| **Divergence proxy** | `mean |∂vx/∂x + ∂vy/∂y|` | Incompressibility violation — should be ≈ 0 for physical flow |

All three use k-NN (k=6) least-squares gradient estimation on the unstructured mesh — no structured grid required.

---

## Data pipeline

### Parse & train (first time setup)

```bash
# CFD: TFRecord → .dat memmap + .npz indices
python parse_tfrecord.py

# Cloth: TFRecord → per-trajectory .npz files
python parse_flag_tfrecord.py

# Data files are tracked by DVC (checksums in .dvc pointer files)
# If a remote is configured:
dvc pull   # fetch data from remote
dvc push   # upload data to remote
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
api/                  FastAPI routes (train, rollout, generate, dataset, status)
app/                  React + Tailwind frontend
model/                GNN architecture (encoder, processor, decoder variants)
dataset/              Dataset classes — FpcDataset (CFD memmap), FlagDataset (cloth .npz)
extensions/
  generative/         CVAE (CFD + cloth), drag surrogate, inverse design, mesh generator
  confidence/         OOD detector, ParamSpaceOOD, NearestNeighborIndex
physics/              Poisson pressure correction (sparse LU, k-NN gradient estimation)
confidence/           Compiled C++ KDTree extension (pybind11) + Python index builder
storage/              Repository Pattern — PKL, HDF5, Zarr backends + StorageFactory
ingest/               Ingest pipeline — SolverAdapter Protocol, composable stages
result/               retention.py — result pruning CLI
scripts/              migrate_pkl_to_hdf5.py, regenerate_dat.py
docs/technical/       21 technical deep-dive notes + interview walkthrough
train.py              CFD/cloth training entry point
train_ddp.py          Distributed Data Parallel training (multi-GPU)
rollout.py            Inference entry point
generate_ssh.py       Remote GPU generate dispatch
tests/                pytest test suite (~60+ tests)
```

---

## Technical notes

In-depth documentation lives in [`docs/technical/`](docs/technical/00_INDEX.md), covering:

- Data pipeline: TFRecord → memmap → GNN (CFD and cloth separately)
- GNN architecture: encoder–processor–decoder, all three variants, message passing mechanics
- Inverse design: CVAE architecture, LHS sampling, gradient backprop paths (CFD vs cloth)
- Confidence scoring: KDTree, train_diameter, OOD detection
- Poisson correction, equivariance analysis, design decisions and tradeoffs
