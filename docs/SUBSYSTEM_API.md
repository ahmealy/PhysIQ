---
tags: [physicsai, subsystem, deep-dive]
created: 2026-04-10
aliases: [api, backend, fastapi]
---

# Subsystem Deep-Dive: API & Backend
### PhysicsAI — technical reference from the ground up

## Quick summary

- FastAPI + SSE streaming: candidates arrive in the browser one-by-one as they're generated
- Heavy ML inference runs in a thread pool — never blocks the async event loop
- Domain samplers use the Strategy pattern: add new domains without touching the route
- Thumbnails rendered with Matplotlib (Agg backend) into PNG bytes, cached in-memory
- Model checkpoints are loaded once and cached as module-level singletons

---

## 1. What this subsystem does

The backend is a **FastAPI** application that sits between the React frontend and every Python subsystem (predictor, generator, confidence scorer). It exposes HTTP endpoints for training, rollout, results, dataset inspection, and design generation. The Generate endpoint in particular orchestrates the full inverse-design pipeline on every request — loading CVAE checkpoints, sampling designs, building meshes, predicting physics, scoring OOD confidence, rendering thumbnails, and streaming results back to the browser as they arrive via Server-Sent Events (SSE).

---

## 2. Technology choices

| Layer | Choice | Why |
|---|---|---|
| Web framework | **FastAPI** | Async-native, Pydantic validation, automatic OpenAPI docs |
| Streaming | **SSE (Server-Sent Events)** | One-directional server→browser push; simpler than WebSockets for fire-and-forget streams |
| Heavy compute | **Thread pool executor** | PyTorch and NumPy are not async-native; running them on the event loop would block all other requests |
| Thumbnails | **Matplotlib (Agg backend)** | No display server needed; produces PNG bytes directly into a BytesIO buffer |
| In-memory cache | **Python dict** | Thumbnails live for the duration of the session; no need for Redis at this scale |
| Model state | **Module-level singleton** | `api/state.py` caches loaded models; avoids reloading 50MB checkpoints per request |

---

## 3. Route map

```
POST /api/generate                              ← stream of SSE candidate events
GET  /api/generate/thumbnail/{sid}/{cid}        ← PNG image for one candidate
POST /api/train/start                           ← launch training subprocess
GET  /api/train/status                          ← poll training progress
POST /api/train/stop                            ← kill training process
POST /api/rollout                               ← run inference, stream SSE frames
GET  /api/rollout/status/{run_id}               ← returns rollout progress state dict (_rollout_state)
GET  /api/results/{filename}                    ← fetch saved rollout result
GET  /api/results/{filename}/timestep/{t}       ← efficient partial read via HDF5 load_timestep
GET  /api/checkpoints                           ← arch_summary: {gn: {best_epoch, val_loss, path}, tns: ..., sage: ...}
GET  /api/dataset/samples                       ← sample graphs for Dataset Studio
GET  /api/dataset/mesh_preview                  ← mesh coordinates for Dataset Studio preview
GET  /api/dataset/info                          ← node_type_counts, trajectory counts, stats
GET  /api/status/gpu                            ← GPU memory info
```

`GET /api/checkpoints` is used by the Predict page architecture selector buttons to populate available checkpoint metadata per architecture variant (GraphNet / TNS / GraphSAGE).

`GET /api/results/{filename}/timestep/{t}` calls `HDF5ResultRepository.load_timestep(key, t)` which reads exactly one `(1, N, D)` chunk — the entire result file is never loaded into memory. Critical for the Visualize page's frame scrubber.

---

## 4. The Generate endpoint in depth

This is the most complex endpoint in the system. Every request triggers the full pipeline.

### Request

```python
class GenerateRequest(BaseModel):
    domain:       str   = "cylinder_flow"   # or "flag_simple"
    target_value: float = 0.025             # drag or stress
    n_candidates: int   = 5
    method:       str   = "sample"          # "sample" | "gradient"
    device:       str   = "cpu"
```

### Response: SSE stream

Instead of waiting for all candidates before returning, the endpoint streams each candidate as it's ready:

```
event: candidate
data: {"id": 0, "predicted_value": 0.0241, "target_value": 0.025,
       "ood_confidence": 0.87, "is_ood": false, "mesh_nodes": 912,
       "params": {"cx": 0.51, "cy": 0.20, "r": 0.082, "v_inlet": 0.43},
       "thumbnail_url": "/api/generate/thumbnail/abc123/0"}

event: candidate
data: { ... candidate 1 ... }

event: done
data: {"best_id": 0, "session_id": "abc123"}
```

The frontend opens a `fetch()` with a streaming body reader, splits on `\n\n`, and parses each chunk as an SSE event. Cards appear in the UI one by one as they arrive.

### Why SSE instead of WebSockets?

SSE is unidirectional (server → client only). The generate flow is exactly that: the browser sends one request, the server sends many events. WebSockets are bidirectional and require more complex connection management. SSE is just HTTP with `Content-Type: text/event-stream` — it works through proxies and doesn't need special server support.

### The async/thread split

FastAPI runs an asyncio event loop. PyTorch and NumPy are **not** async — they block the thread they run on. If you ran `sampler.sample()` directly in the async handler, it would block the entire event loop (no other requests could be served) for the duration of inference.

The fix:

```python
loop = asyncio.get_running_loop()
results = await loop.run_in_executor(
    None,   # use default thread pool
    lambda: sampler.sample(req.target_value, req.n_candidates, req.device)
)
```

`run_in_executor` moves the blocking call to a worker thread. The event loop is free to serve other requests while the thread runs. When the thread finishes, `await` resumes.

After the thread returns, we're back on the event loop and can `yield` SSE events safely:

```python
for c, graph in results:
    png = ThumbnailRenderer.render_cfd(graph)
    yield _sse_event("candidate", payload)
    await asyncio.sleep(0)   # yield to event loop between events
```

The `await asyncio.sleep(0)` is important — it gives the event loop a chance to actually flush the SSE event to the client's TCP buffer before processing the next candidate.

---

## 4b. Poisson Pressure Correction in Rollout

`RolloutRequest` now includes an optional `poisson_correction` flag:

```python
class RolloutRequest(BaseModel):
    ...
    poisson_correction: bool = False
```

When `True`, a `PoissonPressureCorrector` is created from the initial mesh
coordinates before the rollout loop begins. After each GNN step — and **before**
the boundary pin that overwrites INFLOW/HANDLE nodes — the corrector is applied:

```
GNN step t → raw_state
        │  PoissonPressureCorrector.correct(vel)   ← if poisson_correction=True
        ↓
corrected_state
        │  boundary pin (INFLOW nodes overwritten with v_inlet)
        ↓
state_t+1
```

The corrector solves a Poisson equation on the mesh to enforce
∇·u = 0 (incompressibility), removing the spurious divergence that
accumulates over long rollouts. It is applied every step so errors
do not compound.

---

## 5. The Strategy pattern in the sampler

The generate route uses a **registry** of domain samplers:

```python
_DOMAIN_SAMPLERS: dict[str, type[BaseDesignSampler]] = {
    "cylinder_flow": CFDDesignSampler,
    "flag_simple":   ClothDesignSampler,
}
```

`BaseDesignSampler` is an ABC with one method:
```python
class BaseDesignSampler(ABC):
    @abstractmethod
    def sample(self, target: float, n: int, device: str) -> list[CandidateResult]:
        ...
```

The route handler calls `_DOMAIN_SAMPLERS[req.domain]().sample(...)` — it never knows which concrete class it's using. Adding a new domain is one line in the registry dict and one new class.

### CFD sampler flow

```
1. Load CVAE checkpoint (checkpoints/cfd_cvae.pth)
2. Load surrogate checkpoint (checkpoints/drag_surrogate.pth)
3. Call trainer.generate(target_drag_physical=target, n=n)  → [n, 4] params
4. For each row (cx, cy, r, v_inlet):
   a. Clamp to valid physical bounds
   b. CFDMeshBuilder.build(cx, cy, r, v_inlet) → PyG Data
   c. surrogate_trainer.predict([[cx,cy,r,v_in]]) → pred_drag
   d. OODDetector.score(graph) → ood_confidence, is_ood
   e. Package as CandidateResult
5. Sort by |predicted - target|, return top n
```

### Why not run MeshGraphNets for the drag prediction?

The surrogate (3-layer MLP) is ~1000× faster than a full MeshGraphNets rollout (which requires building a mesh, running 15 message passing steps, extracting the velocity field, and computing drag from it). For generating 10 candidates, the surrogate costs microseconds; MeshGraphNets would cost ~1 second per candidate on CPU.

The tradeoff: surrogate drag is an approximation (analytical formula fit, not true CFD). It's accurate enough for ranking candidates but shouldn't be reported as a calibrated drag coefficient.

---

## 6. Thumbnail rendering

The `ThumbnailRenderer` renders each generated mesh as a PNG using Matplotlib. This runs synchronously inside the thread pool (same thread as the sampler).

**CFD thumbnail:**
- Scatter plot of node positions, coloured by node type (NORMAL=grey, WALL_BOUNDARY=orange, INFLOW=blue, OUTFLOW=red)
- Dark background (#1a1a2e) to match the UI theme
- 400×300px at 100 DPI

**Cloth thumbnail:**
- 3D surface plot of the initial cloth world positions using `plot_trisurf`
- Purple/violet colour to match the Generate page accent

Both render to a `BytesIO` buffer and return raw PNG bytes. The bytes are cached in `_thumbnail_cache[session_id][candidate_id]` (a plain Python dict). The `/api/generate/thumbnail/{sid}/{cid}` endpoint serves them directly as `image/png` responses.

**Lifetime:** Thumbnails exist until the process restarts. No eviction policy. This is a known limitation (see tradeoffs).

---

## 7. Model state management (`api/state.py`)

Loading a 50MB PyTorch checkpoint takes ~300ms. If every request loaded fresh from disk, the API would be unusably slow.

`api/state.py` maintains a module-level dict:
```python
_models: dict[tuple, nn.Module] = {}
```

`get_model(domain, checkpoint_path, architecture)` checks this dict first. The cache key is the triple `(domain, checkpoint_path, architecture)` — allowing different architecture variants (GraphNet `"gn"`, TNS `"tns"`, GraphSAGE `"sage"`) to be cached independently without evicting each other. If the model isn't loaded yet, it loads from the checkpoint path and stores it. Subsequent calls return the cached instance in microseconds.

A threading `Lock` protects the dict from simultaneous load attempts when multiple requests arrive at startup.

### IndexStaleError handling

If the confidence index is stale (e.g. new training data was added after the index was built), `IndexStaleError` is raised during rollout. The rollout route catches this and adds a `rebuild_warning` field to the rollout response:

```json
{
  "frames": [...],
  "rebuild_warning": "Confidence index is stale — rebuild recommended (python -m confidence.index build)"
}
```

The frontend displays this as a non-blocking banner. The rollout result is still returned.

---

## 8. SSH Dispatch

Some routes can offload computation to a remote GPU machine via SSH. This is the
primary deployment pattern when the API container runs on a CPU-only host (e.g.
a Docker container without NVIDIA Container Runtime).

### Rollout SSH (`api/routes/rollout.py` + `rollout_ssh.py`)

```
POST /api/rollout
        │ runs/ssh_config.json exists?
        │  Yes ──► write rollout config → SSH launch rollout_ssh.py on GPU machine
        │                               → rollout_ssh.py runs inference
        │                               → applies Poisson correction if enabled
        │                               → streams unnamed SSE events (frame updates) back
        │  No  ──► run inference locally
        ↓
SSE stream to browser
```

### Generate SSH (`api/routes/generate.py` + `generate_ssh.py`)

Same pattern. `generate_ssh.py` uses **named SSE events**:

```
event: candidate
data: {...}

event: done
data: {"best_id": 2}
```

Named events are required because the generate stream carries multiple event
types that need separate frontend handlers. The rollout stream has only one event
type, so unnamed `data:` events are sufficient there.

**Why SSH instead of a container with GPU access?**
Docker containers require NVIDIA Container Runtime and a matching CUDA driver on
the host. SSH dispatch avoids this complexity: the GPU machine runs a plain Python
process, and the Docker container only needs `openssh-client` (already in
`Dockerfile.api`).

---

## 9. Docker and nginx

### Container images

| Image | Base | Final size |
|---|---|---|
| `Dockerfile.api` | `python:3.12-slim` | ~1.1 GB (CPU PyTorch 2.1.0 + deps) |
| `Dockerfile.frontend` | Multi-stage: `node:20-alpine` → `nginx:alpine` | ~22 MB final (vs ~490 MB build stage) |

`Dockerfile.api` includes `openssh-client` so SSH dispatch works inside the
container without any extra setup.

`Dockerfile.frontend` is a **multi-stage build**: the `node:20-alpine` stage runs
`npm run build` and produces a `dist/` folder; the `nginx:alpine` stage copies
only `dist/` and the nginx config. The Node runtime (~490 MB) is discarded,
giving a 22 MB production image.

### `docker/nginx.conf` — SSE-critical settings

```nginx
proxy_buffering           off;
chunked_transfer_encoding on;
```

Without `proxy_buffering off`, nginx accumulates the entire SSE response body
before forwarding it to the browser — the user sees nothing until the job
completes. With buffering disabled, each `yield _sse_event(...)` reaches the
browser as soon as it is written.

### `docker-compose.yml`

```yaml
services:
  api:
    ports: ["8000:8000"]
    volumes:
      - ./data:/app/data
      - ./checkpoints:/app/checkpoints
      - ./result:/app/result
      - ./runs:/app/runs
  frontend:
    ports: ["80:80"]
  frontend-dev:
    ports: ["5173:5173"]
    profiles: ["dev"]   # only started with --profile dev
```

The four host volumes (`data/`, `checkpoints/`, `result/`, `runs/`) are mounted
into the API container so data and checkpoints persist across container restarts
and rebuilds. The `frontend-dev` service (Vite dev server with HMR) is gated
behind `--profile dev` so it does not start in production.

---

## 10. Error handling

The SSE stream catches two categories of errors:

```python
except HTTPException as e:
    yield _sse_event("error", {"detail": e.detail})
except Exception as e:
    yield _sse_event("error", {"detail": str(e)})
```

`HTTPException` (e.g. "CVAE not trained yet") carries a user-readable message. Generic `Exception` is caught so the stream doesn't silently close — the frontend receives an error event and shows an alert.

Validation errors (bad domain, n_candidates out of range) are raised as `HTTPException(400)` before the SSE stream starts, so they come back as a normal HTTP error response (not an SSE event).

---

## 11. Code snippet: the full SSE generator

```python
@router.post("/generate")
async def generate(req: GenerateRequest):
    session_id = str(uuid.uuid4())

    async def event_stream() -> AsyncGenerator[str, None]:
        try:
            sampler = _DOMAIN_SAMPLERS[req.domain]()
            loop    = asyncio.get_running_loop()

            # Heavy compute → thread pool, not event loop
            results = await loop.run_in_executor(
                None,
                lambda: sampler.sample(req.target_value,
                                       req.n_candidates,
                                       req.device)
            )

            _thumbnail_cache[session_id] = {}
            best_id, best_err = 0, float("inf")

            for c, graph in results:
                try:
                    png = ThumbnailRenderer.render_cfd(graph)
                    _thumbnail_cache[session_id][c.id] = png
                    thumbnail_url = f"/api/generate/thumbnail/{session_id}/{c.id}"
                except Exception:
                    thumbnail_url = None

                payload = asdict(c)
                payload["thumbnail_url"] = thumbnail_url
                payload["session_id"]    = session_id

                if abs(c.predicted_value - c.target_value) < best_err:
                    best_err = abs(c.predicted_value - c.target_value)
                    best_id  = c.id

                yield _sse_event("candidate", payload)
                await asyncio.sleep(0)   # flush to client

            yield _sse_event("done", {"best_id": best_id})

        except HTTPException as e:
            yield _sse_event("error", {"detail": e.detail})
        except Exception as e:
            yield _sse_event("error", {"detail": str(e)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
```

`X-Accel-Buffering: no` tells nginx (if used as a reverse proxy) not to buffer the response — critical for SSE to work through a proxy.

---

## 12. Tradeoffs

| Tradeoff | Detail |
|---|---|
| **All candidates computed before streaming** | The thread pool runs the full sampler batch before returning — candidates don't actually stream one-by-one; they all appear at once when the executor finishes. True streaming would require the sampler to yield candidates incrementally. |
| **In-memory thumbnail cache** | No eviction. Long-running servers accumulate PNG bytes indefinitely. Fix: LRU cache with TTL, or store thumbnails to disk with a cleanup task. |
| **Surrogate drag ≠ true CFD drag** | Predicted drag values on candidate cards are from the surrogate MLP, not MeshGraphNets. The predictor is loaded for OOD scoring but not used for the drag prediction shown to the user. |
| **No auth** | Any client can hit `/api/generate` and trigger expensive ML inference. Fine for local use; needs API keys or rate limiting for production. |
| **Single device** | The device is per-request from the client. Multiple concurrent requests on `cpu` are fine; `cuda:0` would cause CUDA OOM if two requests run simultaneously. No GPU request queue. |
| **Model loading is lazy** | First request to `/api/generate` after server start triggers checkpoint loading (~300ms). Subsequent requests are fast. A warmup endpoint would eliminate cold-start latency. |
| **Pickle for thumbnails is not portable** | The thumbnail cache dict is in-process. If you run multiple uvicorn workers, each has its own cache. A request for `/thumbnail/{sid}/{cid}` will 404 if it hits a different worker. Fix: shared Redis or on-disk storage. |

---

## 13. Potential enhancements

| Enhancement | Benefit |
|---|---|
| **True streaming sampler** | Yield candidates one at a time as they're generated instead of waiting for all n. Cards appear progressively in the UI. Requires sampler to be a generator. |
| **Celery task queue** | Move long-running inference to Celery workers; API returns a task_id immediately; client polls for results. Enables horizontal scaling. |
| **GPU request queue** | Serialize GPU inference requests through an `asyncio.Queue` to prevent CUDA OOM when multiple users hit the endpoint simultaneously. |
| **Warmup on startup** | `@app.on_event("startup")` loads all checkpoints into `_models` at server start. Eliminates cold-start latency. |
| **LRU thumbnail cache** | Replace the plain dict with `functools.lru_cache` or a TTL dict to bound memory usage. |
| **Run MeshGraphNets rollout per candidate** | Replace the surrogate drag estimate with a full simulator rollout for more accurate physics predictions. Would need the streaming sampler to keep latency acceptable. |
| **OpenAPI docs for SSE** | FastAPI doesn't natively document SSE events in its OpenAPI spec. Add custom schema annotations or use AsyncAPI for the streaming contract. |

## See also

- [[SUBSYSTEM_GENERATOR]] — the API calls the CVAE sampler to produce candidate designs on every `/api/generate` request
- [[SUBSYSTEM_CONFIDENCE]] — the OOD detector is invoked per candidate inside the generate route
- [[SUBSYSTEM_PREDICTOR]] — the simulator is loaded via `api/state.py` and used for rollout inference and OOD embedding extraction
