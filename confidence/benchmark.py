"""
Benchmark scipy KDTree vs C++ KDTree vs FAISS on identical data.

Usage:
    python -m confidence.benchmark
    python -m confidence.benchmark --index runs/embedding_index_cylinderflow.pkl
    python -m confidence.benchmark --n 5000 --queries 1000

Output example:
    Benchmark: dim=128, n_queries=100
    ───────────────────────────────────────────────────────────────────────────
    N        Backend         Build(ms)    Query(ms)    Batch(ms)    Correct
    100      scipy KDTree        1.2         0.041         3.1      —
    100      C++ KDTree          0.9         0.031         2.3      ✅
    100      FAISS IVF           0.4         0.012         0.8      ✅
    ...
    ───────────────────────────────────────────────────────────────────────────
    Correctness: max distance error = 2.4e-06  (float32 rounding, exact for small N)
"""
import argparse
import time
import os
import sys

import numpy as np
from scipy.spatial import KDTree as ScipyKDTree


# ── FAISS helper ──────────────────────────────────────────────────────────────
def _build_faiss(data: np.ndarray):
    """Return (index, build_ms).  Uses Flat (exact) for N<=10k, IVF for larger."""
    import faiss
    n, dim = data.shape
    data = np.ascontiguousarray(data, dtype=np.float32)
    t0 = time.perf_counter()
    if n <= 10_000:
        # IndexFlatL2 — exact, no training needed
        index = faiss.IndexFlatL2(dim)
        index.add(data)
    else:
        # IVF with sqrt(N) centroids — approximate but very fast
        nlist = max(1, int(np.sqrt(n)))
        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
        index.train(data)
        index.add(data)
        index.nprobe = max(1, nlist // 10)
    build_ms = (time.perf_counter() - t0) * 1000
    return index, build_ms


def _query_faiss(index, queries: np.ndarray, k: int = 1):
    """Return distances array [Q] (L2, not squared — we sqrt to match KDTree)."""
    import faiss
    q = np.ascontiguousarray(queries, dtype=np.float32)
    dists_sq, _ = index.search(q, k)
    return np.sqrt(np.maximum(dists_sq[:, 0], 0.0))


# ── per-N benchmark ───────────────────────────────────────────────────────────
def _bench_one(n: int, dim: int = 128, n_queries: int = 100, seed: int = 0,
               real_data: np.ndarray = None) -> dict:
    rng = np.random.default_rng(seed)
    if real_data is not None:
        idx  = rng.choice(len(real_data), size=min(n, len(real_data)), replace=False)
        data = real_data[idx].astype(np.float32)
        n    = len(data)
        qidx    = rng.choice(len(real_data), size=n_queries, replace=True)
        queries = real_data[qidx].astype(np.float32)
    else:
        data    = rng.random((n, dim)).astype(np.float32)
        queries = rng.random((n_queries, dim)).astype(np.float32)

    results = {}

    # ── scipy ─────────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    scipy_tree = ScipyKDTree(data)
    build_scipy = (time.perf_counter() - t0) * 1000

    # Warm up (avoid import / JIT overhead in timing)
    scipy_tree.query(queries[0:1], k=1)

    t0 = time.perf_counter()
    scipy_tree.query(queries[0:1], k=1)
    query_scipy = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    scipy_dists, _ = scipy_tree.query(queries, k=1)
    batch_scipy = (time.perf_counter() - t0) * 1000
    scipy_dists = scipy_dists.flatten().astype(np.float32)

    results["scipy"] = {
        "build_ms": round(build_scipy, 2),
        "query_ms": round(query_scipy, 3),
        "batch_ms": round(batch_scipy, 2),
        "dists":    scipy_dists,
        "correct":  "—  (reference)",
    }

    # ── C++ KDTree (opt-in, pybind11) ─────────────────────────────────────────
    try:
        confidence_dir = os.path.dirname(os.path.abspath(__file__))
        if confidence_dir not in sys.path:
            sys.path.insert(0, confidence_dir)
        from _kdtree import KDTree as CppKDTree  # type: ignore

        t0 = time.perf_counter()
        cpp_tree = CppKDTree(data)
        build_cpp = (time.perf_counter() - t0) * 1000

        cpp_tree.query(queries[0:1].reshape(1, -1), k=1)  # warm up

        t0 = time.perf_counter()
        cpp_tree.query(queries[0:1].reshape(1, -1), k=1)
        query_cpp = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        cpp_dists = np.array([cpp_tree.query(q.reshape(1, -1), k=1)[0] for q in queries],
                             dtype=np.float32)
        batch_cpp = (time.perf_counter() - t0) * 1000

        max_err = float(np.max(np.abs(cpp_dists - scipy_dists)))
        correct = "✅  (err=%.2e)" % max_err if max_err < 1e-3 else "❌  (err=%.2e)" % max_err

        results["cpp"] = {
            "build_ms": round(build_cpp, 2),
            "query_ms": round(query_cpp, 3),
            "batch_ms": round(batch_cpp, 2),
            "dists":    cpp_dists,
            "correct":  correct,
            "max_err":  max_err,
        }
    except ImportError:
        results["cpp"] = None

    # ── FAISS ─────────────────────────────────────────────────────────────────
    try:
        faiss_index, build_faiss = _build_faiss(data)

        _query_faiss(faiss_index, queries[0:1])  # warm up

        t0 = time.perf_counter()
        _query_faiss(faiss_index, queries[0:1])
        query_faiss = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        faiss_dists = _query_faiss(faiss_index, queries)
        batch_faiss = (time.perf_counter() - t0) * 1000

        max_err = float(np.max(np.abs(faiss_dists - scipy_dists)))
        # FAISS IVF is approximate for large N — relax tolerance
        tol = 0.01 if n > 10_000 else 1e-3
        correct = "✅  (err=%.2e)" % max_err if max_err < tol else "⚠️   (err=%.2e, approx)" % max_err
        label = "FAISS-Flat" if n <= 10_000 else "FAISS-IVF"

        results["faiss"] = {
            "build_ms": round(build_faiss, 2),
            "query_ms": round(query_faiss, 3),
            "batch_ms": round(batch_faiss, 2),
            "dists":    faiss_dists,
            "correct":  correct,
            "max_err":  max_err,
            "label":    label,
        }
    except ImportError:
        results["faiss"] = None

    return n, results


# ── table printer ─────────────────────────────────────────────────────────────
def run_benchmark(ns: list, dim: int = 128, n_queries: int = 100,
                  real_data: np.ndarray = None):
    sep = "─" * 76
    print("\nBenchmark: dim=%d, n_queries=%d" % (dim, n_queries))
    print(sep)
    print("%-9s %-16s %-13s %-13s %-13s %s" % (
        "N", "Backend", "Build(ms)", "Query(ms)", "Batch(ms)", "Correct"))
    print(sep)

    for n in ns:
        actual_n, res = _bench_one(n, dim=dim, n_queries=n_queries,
                                   real_data=real_data)

        sp = res["scipy"]
        print("%-9d %-16s %-13.2f %-13.3f %-13.2f %s" % (
            actual_n, "scipy KDTree",
            sp["build_ms"], sp["query_ms"], sp["batch_ms"], sp["correct"]))

        if res.get("cpp"):
            cpp = res["cpp"]
            print("%-9d %-16s %-13.2f %-13.3f %-13.2f %s" % (
                actual_n, "C++ KDTree",
                cpp["build_ms"], cpp["query_ms"], cpp["batch_ms"], cpp["correct"]))
        else:
            print("%-9d %-16s %-13s (not compiled — cd confidence && cmake . && make)" % (
                actual_n, "C++ KDTree", "—"))

        if res.get("faiss"):
            f = res["faiss"]
            print("%-9d %-16s %-13.2f %-13.3f %-13.2f %s" % (
                actual_n, f["label"],
                f["build_ms"], f["query_ms"], f["batch_ms"], f["correct"]))
        else:
            print("%-9d %-16s %-13s (not installed — pip install faiss-cpu)" % (
                actual_n, "FAISS", "—"))

        print()

    print(sep)
    print("Query(ms) = single query latency (warmed up)")
    print("Batch(ms) = %d queries total" % n_queries)
    print("FAISS-Flat = exact (N<=10k) | FAISS-IVF = approximate (N>10k)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark scipy vs C++ KDTree vs FAISS")
    parser.add_argument("--dim",     type=int, default=128)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--n",       type=int, default=None,
                        help="Single N to benchmark (instead of default sweep)")
    parser.add_argument("--index",   type=str, default=None,
                        help="Path to saved embedding_index*.pkl (uses real embeddings)")
    args = parser.parse_args()

    real_data = None
    if args.index:
        import pickle
        with open(args.index, "rb") as f:
            d = pickle.load(f)
        real_data = d["embeddings"]
        args.dim  = real_data.shape[1]
        print("Real embeddings: %s  shape=%s" % (args.index, real_data.shape))

    ns = [args.n] if args.n else [100, 1_000, 5_000, 10_000, 50_000]
    run_benchmark(ns=ns, dim=args.dim, n_queries=args.queries, real_data=real_data)
