"""
Performance comparison script for JIT-compiled eigensolvers.
Compares Safe, Subspace, and Stable variants in non-masked modes.
"""

import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp

# Add parent directory to path to import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.jax.generalized_eigensolver import (
    safe_generalized_eigh,
    subspace_generalized_eigh
)
from src.jax.generalized_eigensolver_stable import stable_generalized_eigh
from src.jax.generalized_eigensolver_pyscfad import stable_eigh_gen_pyscfad


# Enable double precision
jax.config.update("jax_enable_x64", True)


def benchmark(f, *args, n_iter=20):
    """JIT-compiles and benchmarks the given function."""
    # Warmup and JIT
    jit_f = jax.jit(f)
    out = jit_f(*args)
    jax.block_until_ready(out)

    # Benchmark
    t0 = time.perf_counter()
    for _ in range(n_iter):
        out = jit_f(*args)
        jax.block_until_ready(out)
    t1 = time.perf_counter()

    return (t1 - t0) / n_iter * 1000  # Return in ms


def run_performance_test(n=100):
    """Runs a performance comparison for a matrix of size n."""
    print(f"Benchmarking Matrix Size: {n}")
    print("-" * 50)

    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)

    # Active matrices
    f_small = jax.random.normal(k1, (n, n))
    f_small = f_small @ f_small.T
    s_small = jax.random.normal(k2, (n, n))
    s_small = s_small @ s_small.T + jnp.eye(n)

    non_masked_solvers = [
        ("Safe (NM)", safe_generalized_eigh),
        ("Subspace (NM)", subspace_generalized_eigh),
        ("Stable (NM)", stable_generalized_eigh),
        ("Pyscfad (NM)", stable_eigh_gen_pyscfad),
    ]

    results = []

    # Non-masked benchmarks
    for name, solver in non_masked_solvers:
        # Note: Non-masked solvers take (A, B)
        ms = benchmark(solver, f_small, s_small)
        results.append((name, ms))
        print(f"{name:15}: {ms:8.4f} ms")

    print("-" * 50)

    # Find overall winner
    winner = min(results, key=lambda x: x[1])
    print(f"Overall Fastest: {winner[0]} ({winner[1]:.4f} ms)")


if __name__ == "__main__":
    # Test for a few different sizes
    for n in [32, 64, 128, 256, 512, 1024, 2048]:
        run_performance_test(n)
        print()
