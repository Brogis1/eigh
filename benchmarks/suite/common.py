"""Shared utilities for the CPU benchmarking suite: solver registry,
matrix generators, timing, accuracy metrics."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.jax.generalized_eigensolver import (
    safe_generalized_eigh,
    subspace_generalized_eigh,
    generalized_eigh,
    jax_eig,
)
from src.jax.generalized_eigensolver_stable import stable_generalized_eigh
from src.jax.generalized_eigensolver_pyscfad import stable_eigh_gen_pyscfad

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


SOLVERS = {
    "jax_eig": jax_eig,
    "generalized": generalized_eigh,
    "safe": safe_generalized_eigh,
    "subspace": subspace_generalized_eigh,
    "stable": stable_generalized_eigh,
    "pyscfad": stable_eigh_gen_pyscfad,
}


@dataclass
class TimingResult:
    name: str
    n: int
    mean_ms: float
    std_ms: float
    min_ms: float
    samples: int


def well_conditioned(key, n, dtype=jnp.float64):
    """SPD B with small eigenvalue spread, symmetric A."""
    k1, k2 = jax.random.split(key)
    A = jax.random.normal(k1, (n, n), dtype=dtype)
    A = (A + A.T) * 0.5
    B = jax.random.normal(k2, (n, n), dtype=dtype)
    B = B @ B.T + jnp.eye(n, dtype=dtype)
    return A, B


def ill_conditioned(key, n, cond=1e8, dtype=jnp.float64):
    """SPD B with a target condition number (log-spaced spectrum)."""
    k1, k2 = jax.random.split(key)
    A = jax.random.normal(k1, (n, n), dtype=dtype)
    A = (A + A.T) * 0.5
    Q, _ = jnp.linalg.qr(jax.random.normal(k2, (n, n), dtype=dtype))
    spectrum = jnp.logspace(0, -jnp.log10(cond), n, dtype=dtype)
    B = (Q * spectrum) @ Q.T
    B = (B + B.T) * 0.5
    return A, B


def degenerate_spectrum(key, n, gap=1e-8, dtype=jnp.float64):
    """A with clustered eigenvalues (pairwise near-degenerate), B well-conditioned."""
    k1, k2, k3 = jax.random.split(key, 3)
    Q, _ = jnp.linalg.qr(jax.random.normal(k1, (n, n), dtype=dtype))
    base = jax.random.normal(k2, (n // 2 + 1,), dtype=dtype)
    evals = jnp.repeat(base, 2)[:n]
    perturb = jnp.arange(n, dtype=dtype) * gap
    evals = evals + perturb
    A = (Q * evals) @ Q.T
    A = (A + A.T) * 0.5
    B = jax.random.normal(k3, (n, n), dtype=dtype)
    B = B @ B.T + jnp.eye(n, dtype=dtype)
    return A, B


def ensure_ready(*arrays):
    for a in arrays:
        jax.block_until_ready(a)


def benchmark(fn, args, n_iter=30, warmup=3):
    """JIT-compile fn, do warmup, time n_iter calls. Returns TimingResult fields."""
    jit_fn = jax.jit(fn)
    for _ in range(warmup):
        out = jit_fn(*args)
        ensure_ready(out)

    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        out = jit_fn(*args)
        ensure_ready(out)
        times.append((time.perf_counter() - t0) * 1000.0)

    times = np.array(times)
    return {
        "mean_ms": float(times.mean()),
        "std_ms": float(times.std(ddof=1)) if len(times) > 1 else 0.0,
        "min_ms": float(times.min()),
        "samples": len(times),
    }


def reference_solution(A, B):
    """Ground truth via scipy on float64."""
    A64 = np.asarray(A, dtype=np.float64)
    B64 = np.asarray(B, dtype=np.float64)
    w, V = scipy.linalg.eigh(A64, B64)
    idx = np.argsort(w)
    return w[idx], V[:, idx]


def eigenvalue_error(w_test, w_ref):
    w_test = np.sort(np.asarray(w_test))
    w_ref = np.sort(np.asarray(w_ref))
    return float(np.max(np.abs(w_test - w_ref)))


def subspace_error(V_test, V_ref, k=None):
    """Angle between subspaces spanned by first k columns. k=None uses all."""
    V_test = np.asarray(V_test)
    V_ref = np.asarray(V_ref)
    if k is None:
        k = V_ref.shape[1]
    Q_test, _ = np.linalg.qr(V_test[:, :k])
    Q_ref, _ = np.linalg.qr(V_ref[:, :k])
    s = np.linalg.svd(Q_test.T @ Q_ref, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    return float(np.max(np.arcsin(np.sqrt(np.clip(1.0 - s**2, 0.0, 1.0)))))


def residual(A, B, w, V):
    """||A V - B V diag(w)|| / ||A||."""
    A = np.asarray(A)
    B = np.asarray(B)
    w = np.asarray(w)
    V = np.asarray(V)
    r = A @ V - B @ V * w[None, :]
    return float(np.linalg.norm(r) / (np.linalg.norm(A) + 1e-30))
