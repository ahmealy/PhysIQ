# 🌊 Learning Mesh-Based Simulation with Graph Networks  
### *Fast, Adaptive, and Physics-Informed Neural Simulators for Complex Fluid Dynamics*

This repository provides a **PyTorch + PyG (PyTorch Geometric)** implementation of **MeshGraphNets**—a powerful graph neural network framework for learning mesh-based physical simulations. We focus on the **flow around a circular cylinder** problem, reproducing and extending the groundbreaking work from DeepMind.

> 🔬 **Original Paper**:  
> [**Learning Mesh-Based Simulation with Graph Networks**](https://arxiv.org/abs/2010.03409)  
> *Tobias Pfaff, Meire Fortunato, Alvaro Sanchez-Gonzalez, Peter W. Battaglia*  
> **ICLR 2021**
---

## ✨ Why This Project?

- **Physics-aware learning**: Leverages mesh structure to respect geometric and physical priors.
- **High performance**: Runs **10–100× faster** than traditional solvers while maintaining fidelity.
- **Extensible**: Built on PyTorch Geometric—easy to adapt to new PDEs, materials, or domains.
---

## 🛠️ Requirements

Install dependencies via:

```bash
pip install -r requirements.txt
```

> 💡 **Note**: TensorFlow < 1.15.0 is required only for parsing the original TFRecord datasets.

A `pyproject.toml` is also provided for tool configs (ruff, mypy, pytest) and project metadata.

---

## 🚀 Quick Start

### 1. Download the Dataset
We use DeepMind’s `cylinder_flow` dataset:

```bash
aria2c -x 8 -s 8 https://storage.googleapis.com/dm-meshgraphnets/cylinder_flow/train.tfrecord -d data
aria2c -x 8 -s 8 https://storage.googleapis.com/dm-meshgraphnets/cylinder_flow/valid.tfrecord -d data
aria2c -x 8 -s 8 https://storage.googleapis.com/dm-meshgraphnets/cylinder_flow/test.tfrecord -d data
```

### 2. Parse TFRecords
Convert to PyTorch-friendly format:

```bash
python parse_tfrecord.py
```
> Output saved in `./data/`.

### 3. Train the Model
```bash
python train.py
```

The UI writes a JSON config for `train.py`. You can also pass one manually:
```bash
python train.py --config runs/my_config.json
```

Config keys (all optional, defaults shown):
```json
{
  "num_epochs": 100,
  "batch_size": 20,
  "lr": 1e-4,
  "noise_std": 0.02,
  "early_stopping_patience": 10,
  "message_passing_num": 15
}
```

FOR MULTI-GPU TRAINING:

```bash
export NGPUS=2 # set as your machine's available GPUs
torchrun --nproc_per_node=$NGPUS train_ddp.py --dataset_dir data
```

### 4. Run Rollouts & Visualize
Generate long-horizon predictions and render videos:

```bash
python rollout.py          # saves results to ./result/
python render_results.py   # generates videos in ./videos/
```

## 🌐 Web UI

The project includes a **React + FastAPI** full-stack dashboard for training, inference, and result visualization.

### Architecture

```
Browser (localhost:5173)
  └── Vite dev server  →  proxy /api/*  →  FastAPI backend (port 8000)
```

### Startup

**Terminal 1 — FastAPI backend:**
```bash
# From the repo root — activate the venv first
source venv/bin/activate
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

> Use `python -m uvicorn` (not just `uvicorn`) to ensure it runs inside the venv
> where all dependencies (torch, fastapi, scipy, etc.) are installed.
> Full interactive API docs at: http://localhost:8000/docs

**Terminal 2 — React frontend:**
```bash
cd app
npm install
npm run dev   # opens http://localhost:5173
```

All `/api/*` calls from the browser are proxied by Vite directly to FastAPI at port 8000.

### Available Pages

| Page | Route | Description |
|------|-------|-------------|
| Home / Dashboard | `/` | Status overview, GPU info, quick-start |
| Training | `/train` | Configure + monitor training with live loss curves |
| Predict | `/predict` | Run autoregressive rollouts, GPU performance panel |
| Visualize | `/visualize?file=X` | 3-panel mesh viewer + RMSE + Diagnostics + Physics tabs |
| Pipeline | `/pipeline` | End-to-end workflow DAG with live status |
| Experiments | `/experiments` | Compare RMSE curves across saved rollouts |
| Dataset Studio | `/dataset` | Dataset statistics, node count histogram, outlier flagging |

---

### Remote GPU Training

To offload training to a remote GPU machine over SSH, configure it in the UI under **Training → Remote GPU**, or save a config manually:

```bash
cat > runs/remote_gpu.json <<EOF
{
  "host": "your-gpu-machine.example.com",
  "port": 22,
  "user": "ahmealy",
  "venv_python": "/home/ahmealy/.pyenv/versions/venv_gpu/bin/python",
  "enabled": true
}
EOF
```

Requirements:
- SSH key auth set up (`ssh-copy-id user@host`)
- Shared filesystem — the repo root must be accessible at the same path on both machines (e.g. NFS mount)
- The remote venv must have all dependencies installed

When enabled, **Start Training** in the UI will run `train.py` on the remote machine. Logs stream back over SSH in real time.



### Results on DeepMind’s `cylinder_flow`:
| Demo 0 | Demo 1 |
|------------|--------------|
| ![Demo 0](videos/0.gif) | ![Demo 1](videos/1.gif) |

### Results on **our own CFD-generated data** (new geometries & conditions):
| Demo 2 | Demo 3 |
|------------|--------------|
| ![Demo 2](videos/2.gif) | ![Demo 3](videos/3.gif) |

> ✅ The model generalizes well—even to unseen flow regimes and mesh configurations!

---

> ⭐ **If you find this project useful, please consider starring the repo!**  
> Your support helps us keep improving open-source scientific ML tools.
