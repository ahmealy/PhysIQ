"""
Benchmark scipy vs C++ KDTree backends on identical data.

Usage:
    python -m confidence.benchmark
    python -m confidence.benchmark --index runs/embedding_index.pkl

Output example:
    Benchmark: dim=128
    ───────────────────────────────────────────────────────────────────
    N        Backend       Build(ms)   Query(ms)   Batch(ms)   Correct
    100      scipy             1.2        0.041        3.1      —
    100      C++               0.9        0.031        2.3      ✅
    1000     scipy            14.2        0.051        4.8      —
    1000     C++              11.8        0.038        3.1      ✅
    10000    scipy           162.4        0.063       46.2      —
    10000    C++             118.1        0.041       28.7      ✅
    ───────────────────────────────────────────────────────────────────
    Correctness: max distance error = 2.4e-06 (float32 precision)
"""
import argparse
import time

import numpy as np
from scipy.spatial import KDTree as ScipyKDTree


def _bench_one(n: int, dim: int = 128, n_queries: int = 100, seed: int = 0,
               real_data: np.ndarray = None):
    """Benchmark both backends for given N. Returns result dict."""
    rng = np.random.default_rng(seed)
    if real_data is not None:
        # Use a subsample of real embeddings; generate synthetic queries from same dist
        idx = rng.choice(len(real_data), size=min(n, len(real_data)), replace=False)
        data = real_data[idx].astype(np.float32)
        n = len(data)
        queries = real_data[rng.choice(len(real_data), size=n_queries, replace=True)].astype(np.float32)
    else:
        data    = rng.random((n, dim)).astype(np.float32)
        queries = rng.random((n_queries, dim)).astype(np.float32)

    results = {}

    # ── scipy ─────────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    scipy_tree = ScipyKDTree(data)
    build_scipy = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    scipy_tree.query(queries[0:1], k=1)
    query_scipy = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    scipy_dists, _ = scipy_tree.query(queries, k=1)
    batch_scipy = (time.perf_counter() - t0) * 1000
    scipy_dists = scipy_dists.flatten()

    results["scipy"] = {
        "build_ms":  round(build_scipy, 2),
        "query_ms":  round(query_scipy, 3),
        "batch_ms":  round(batch_scipy, 2),
        "dists":     scipy_dists,
        "correct":   "—",
    }

    # ── C++ (if compiled) ──────────────────────────────────────────────────────
    try:
        import sys, os
        confidence_dir = os.path.dirname(os.path.abspath(__file__))
        if confidence_dir not in sys.path:
            sys.path.insert(0, confidence_dir)
        from _kdtree import KDTree as CppKDTree  # type: ignore

        t0 = time.perf_counter()
        cpp_tree = CppKDTree(data)
        build_cpp = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        cpp_tree.query(queries[0:1].reshape(1, -1), k=1)
        query_cpp = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        cpp_dists = np.array([cpp_tree.query(q.reshape(1, -1), k=1)[0] for q in queries])
        batch_cpp = (time.perf_counter() - t0) * 1000

        max_err = float(np.max(np.abs(cpp_dists - scipy_dists)))
        correct = "✅" if max_err < 1e-4 else "❌ (err=%.2e)" % max_err

        results["cpp"] = {
            "build_ms":  round(build_cpp, 2),
            "query_ms":  round(query_cpp, 3),
            "batch_ms":  round(batch_cpp, 2),
            "dists":     cpp_dists,
            "correct":   correct,
            "max_err":   max_err,
        }
    except ImportError:
        results["cpp"] = None

    return results


def run_benchmark(dim: int = 128, n_queries: int = 100, real_data: np.ndarray = None):
    ns = [100, 1000, 10000]

    sep = "─" * 67
    print("\nBenchmark: dim=%d, n_queries=%d" % (dim, n_queries))
    print(sep)
    print("%-8s %-14s %-12s %-12s %-12s %s" % (
        "N", "Backend", "Build(ms)", "Query(ms)", "Batch(ms)", "Correct"))
    print(sep)

    max_errs = []

    for n in ns:
        res = _bench_one(n, dim=dim, n_queries=n_queries, real_data=real_data)

        sp = res["scipy"]
        print("%-8d %-14s %-12.2f %-12.3f %-12.2f %s" % (
            n, "scipy KDTree", sp["build_ms"], sp["query_ms"], sp["batch_ms"], sp["correct"]))

        if res["cpp"]:
            cpp = res["cpp"]
            print("%-8d %-14s %-12.2f %-12.3f %-12.2f %s" % (
                n, "C++ KDTree", cpp["build_ms"], cpp["query_ms"], cpp["batch_ms"], cpp["correct"]))
            if "max_err" in cpp:
                max_errs.append(cpp["max_err"])
        else:
            print("%-8d %-14s %-12s (C++ not compiled — run cmake && make in confidence/)" % (
                n, "C++ KDTree", "—"))
        print()

    print(sep)
    if max_errs:
        print("Correctness: max distance error = %.2e (float32 precision)" % max(max_errs))
    else:
        print("C++ backend not available. Build with: cd confidence && cmake . && make")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark scipy vs C++ KDTree")
    parser.add_argument("--dim",       type=int, default=128)
    parser.add_argument("--queries",   type=int, default=100)
    parser.add_argument("--index",     type=str, default=None,
                        help="Optional: path to saved embedding_index.pkl to use real embeddings")
    args = parser.parse_args()

    if args.index:
        import pickle
        with open(args.index, "rb") as f:
            d = pickle.load(f)
        embs = d["embeddings"]
        print("Using real embeddings from %s: shape %s" % (args.index, embs.shape))
        # Override dimension and pass real data to benchmark
        args.dim = embs.shape[1]
        run_benchmark(dim=args.dim, n_queries=args.queries, real_data=embs)
    else:
        run_benchmark(dim=args.dim, n_queries=args.queries)
