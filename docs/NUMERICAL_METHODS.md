# Numerical Methods Roadmap

## Status

| Method | Status | File |
|---|---|---|
| Pressure Poisson Correction | ✅ Implemented | `physics/poisson_pressure.py` |
| Laplacian Mesh Smoothing | 📋 Planned | — |
| Implicit FVM Baseline Solver | 📋 Planned | — |

---

## 1. Pressure Poisson Correction ✅

### Motivation

GNN simulators predict velocity fields step-by-step but have no built-in mechanism to
enforce the incompressibility constraint ∇·v = 0. Even small per-step divergence
errors accumulate over a long rollout, causing unphysical mass sources/sinks and
eventually destabilising the prediction. The "Divergence Proxy" visible in the
Analyze tab quantifies this violation — typical raw GNN output sits at 0.01–0.1 RMS.

### Method

The Helmholtz–Hodge decomposition guarantees that any vector field v* can be written as:

    v* = v_div_free + ∇φ

where v_div_free has zero divergence. We find φ (the "pressure correction") by
solving the Poisson equation:

    ∇²φ = ∇·v*   (with Dirichlet BC: φ=0 at one boundary node)

and then correct:

    v_corrected = v* − ∇φ

On an unstructured mesh the Laplacian ∇² is approximated by the graph Laplacian L
assembled from edge connectivity, and both divergence and gradient are estimated by
local k-NN least-squares finite differences (the same vectorised approach used in the
existing vorticity and divergence-proxy computations in `api/routes/physics.py`).

### LU Decomposition Role

The mesh topology is fixed for a given result file, so L is constant across all
timesteps. We factorise L once with `scipy.sparse.linalg.splu` (supernodal LU,
O(N log N)) and cache the `SuperLU` object inside the corrector. Each subsequent
timestep only requires a triangular solve — O(N) — making the per-step cost
negligible for N ≈ 1 876 (typical cylinder mesh).

    First call  ≈ 0.3–0.5 s  (builds k-NN tree + LU factorisation)
    Per-step    ≈ 1–5 ms     (divergence eval + LU solve + gradient eval)

A small diagonal regularisation ε·I (default ε = 1e-6) is added to guarantee
non-singularity even for disconnected or near-degenerate meshes.

### API

**Endpoint**

```
GET /results/{filename}/physics/corrected_divergence
```

Returns JSON:

```json
{
  "divergence_before":        [T floats],
  "divergence_after":         [T floats],
  "divergence_reduction_pct": 73.4,
  "correction_norm":          [T floats]
}
```

**Python usage**

```python
from physics.poisson_pressure import PoissonPressureCorrector

corrector = PoissonPressureCorrector(crds)          # crds: [N, 2]
v_corrected = corrector.correct(v_predicted)         # [N, 2]
v_series    = corrector.correct_series(v_series)     # [T, N, 2]
rms         = corrector.divergence_rms(v_corrected)  # float
```

### Value

- **Divergence proxy** in the Analyze tab drops significantly after correction —
  visible validation that the projection works.
- **Long-rollout stability** improves because divergence errors no longer accumulate.
- **Energy conservation** is better preserved; the corrected field stays closer to
  the physical attractor.

---

## 2. Laplacian Mesh Smoothing (Planned)

### Motivation

The CVAE generator (`extensions/generative/cvae_cfd.py`) occasionally produces
mesh node configurations with locally irregular spacing — clustered nodes or
near-degenerate triangles. The GNN was trained on DeepMind's carefully meshed
datasets, so irregular inputs from the CVAE reduce prediction quality, partially
explaining why generated designs sometimes have unrealistic drag estimates.

### Method

Given an irregular mesh with coordinates X, we want a smoothed version X_smooth
that is close to X but has more uniform node spacing. This is the discrete analogue
of minimising the Dirichlet energy:

    min_{X_smooth}  X_smooth^T L X_smooth   subject to  ||X_smooth − X||² ≤ δ

The closed-form solution is:

    (L + λI) X_smooth = λ X

which is a sparse linear system with the same graph Laplacian L used in the
Poisson corrector. LU-factorise once per mesh topology, then solve in O(N) per
generated sample.

### Implementation plan

- **File**: `physics/mesh_smoothing.py`
- **Class**: `LaplacianSmoother(crds, edges, smoothing_strength=0.3)`
- **Method**: `smooth(crds) → crds_smooth`
- **Wire into**: `extensions/generative/generate.py` after CVAE `.sample()`,
  before passing coordinates to the GNN scorer
- **LU reuse**: factorise once per generation batch (mesh topology fixed within batch)

### Expected value

Better GNN inputs from Generate → more realistic drag predictions → more diverse
and higher-quality design candidates surfaced to the user.

---

## 3. Implicit FVM Baseline Solver (Planned)

### Motivation

The project claims a "speedup badge" (GNN vs. classical CFD) but the comparison is
estimated, not measured against a real solver running on the same hardware. Without
a concrete baseline the claim is hard to defend in a technical interview or paper.

### Method

A minimal implicit Euler finite-volume solver on the same unstructured mesh:

1. **Momentum step**: implicit diffusion + explicit advection
   `(M/dt − ν·L) u^{n+1} = M/dt · u^n − A(u^n) · u^n`
2. **Pressure-Poisson step**: `L p = (1/dt) ∇·u^{n+1}`
3. **Projection**: `u_corrected = u^{n+1} − dt · ∇p`

Each timestep requires two sparse LU solves (momentum + pressure). Re-factorise
the momentum matrix every 10 steps (when the advection term changes significantly),
reuse the pressure Laplacian factorisation from step (2) (identical to the Poisson
corrector above).

### Implementation plan

- **File**: `physics/fvm_baseline.py`
- **CLI**: `python physics/fvm_baseline.py --traj 0 --steps 100 --dt 0.01`
- **Wire into**: new `/results/{filename}/baseline` API endpoint for side-by-side
  comparison with the GNN rollout in the Analyze tab
- **Output**: velocity and pressure fields at each step, JSON-serialisable

### Expected value

- Defensible, measured speedup numbers (wall-clock GNN vs. FVM on the same mesh)
- Accuracy crossover analysis: at what rollout length does GNN error exceed FVM?
- Potential for a hybrid GNN+FVM corrector: run GNN fast, apply FVM correction
  every K steps to keep trajectories on the physical attractor.
