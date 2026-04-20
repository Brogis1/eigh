"""JAX-based implementations of generalized eigenvalue solvers.

Experimental solvers, we have a collection of them because they strongly influence
training stability and we can choose the best one eventually.

This module provides various differentiable and stable generalized eigenvalue solvers
designed to handle degeneracies and ill-conditioned matrices common in DFT calculations.
The implementation aims for high numerical stability in the reverse-mode gradients,
ensuring robust optimization of machine-learned exchange-correlation functionals.

There are many eigenssolvers here:
1. standard_eig: standard scipy.linalg.eigh (CPU only)
2. jax_eig: simple JAX-based generalized eigensolver without stabilization
3. generalized_eigh: robust generalized eigensolver using standard JAX eigh
4. safe_generalized_eigh: robust generalized eigensolver using JAX custom VJP
5. subspace_generalized_eigh: robust generalized eigensolver using subspace projection

If you get NaNs in the gradients, try using safe_generalized_eigh or subspace_generalized_eigh.
The perturbation_strength parameter controls the amount of perturbation to eigenvalues
to lift degeneracies (tested on periodic systems with degeneracies).
Try different values of perturbation_strength to find the best value for your problem.


safe_generalized_eigh or subspace_generalized_eigh are most stable here but
training with safe_generalized_eigh can stop when degeneracies are present.

See generalized_eigh_stable.py for a different implementation using [2].

References:
    - [1] Kasim et al., "Learning the Exchange-Correlation Functional
      from Nature with Fully Differentiable Density Functional Theory",
      PhysRevLett.127.126403
      https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.127.126403
    - [2] Colburn, S., Majumdar, A. Inverse design and flexible parameterization
      of meta-optics using algorithmic differentiation.
      https://doi.org/10.1038/s42005-021-00568-6
    - [3] JAX Issue #2748: Differentiable eigh with degeneracies.
      https://github.com/jax-ml/jax/issues/2748
    - [4] JAX Issue #5461: Stable generalized eigh.
      https://github.com/jax-ml/jax/issues/5461
"""

from __future__ import annotations

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp
import scipy.linalg
from jaxtyping import Array, Float


def is_debug_enabled() -> bool:
    """Check if debug mode is enabled for eigensolver internals."""
    return False


# ------------------------------------------------------------
# Standard eigensolvers
# ------------------------------------------------------------


def standard_eig(
    fock: Float[Array, "n n"], overlap: Float[Array, "n n"]
) -> tuple[Float[Array, "n"], Float[Array, "n n"]]:
    """Non-differentiable generalized eigensolver using scipy (CPU only).

    Args:
      fock: The Fock matrix of shape (n, n).
      overlap: The overlap matrix of shape (n, n).

    Returns:
      A tuple (evals, evecs) containing:
        - evals: Sorted eigenvalues of shape (n,).
        - evecs: Sorted eigenvectors of shape (n, n).
    """
    e, c = scipy.linalg.eigh(fock, overlap)
    idx = e.argsort()
    return e[idx], c[:, idx]


@jax.jit
def jax_eig(
    A: Float[Array, "n n"], B: Float[Array, "n n"]
) -> tuple[Float[Array, "n"], Float[Array, "n n"]]:
    """Simple JAX-based generalized eigensolver without stabilization.

    Args:
      A: Real-symmetric matrix of shape (n, n).
      B: Real-symmetric SPD matrix of shape (n, n).

    Returns:
      A tuple (eigenvalues, eigenvectors) where eigenvalues has shape (n,)
      and eigenvectors has shape (n, n).
    """
    L = jnp.linalg.cholesky(B)
    L_inv = jnp.linalg.inv(L)
    C = L_inv @ A @ L_inv.T
    eigenvalues, eigenvectors_transformed = jnp.linalg.eigh(C)
    eigenvectors_original = L_inv.T @ eigenvectors_transformed
    return eigenvalues, eigenvectors_original


# ------------------------------------------------------------
# Robust eigensolvers
# ------------------------------------------------------------


@partial(jax.jit, static_argnames=("eps", "scale", "dtype"))
def generalized_eigh(
    A: Float[Array, "... n n"],
    B: Float[Array, "... n n"],
    *,
    eps: float = 1.0e-12,
    scale: bool = False,
    dtype: Any = jnp.float64,
) -> tuple[Float[Array, "... n"], Float[Array, "... n n"]]:
    """Robust generalized symmetric eigensolver using standard JAX eigh.

    Note: This version uses jnp.linalg.eigh, which may be unstable in reverse
    mode for degenerate eigenvalues. Use safe_generalized_eigh for training.

    Args:
      A: Batch of real-symmetric matrices of shape (..., n, n).
      B: Batch of SPD matrices of shape (..., n, n).
      eps: Minimal eigenvalue shift for B.
      scale: Whether to apply diagonal scaling.
      dtype: Working precision.

    Returns:
      A tuple (evals, evecs) containing:
        - evals: Eigenvalues of shape (..., n).
        - evecs: Eigenvectors of shape (..., n, n).
    """
    A = (A + A.T.conj()) * 0.5
    B = (B + B.T.conj()) * 0.5
    A = A.astype(dtype)
    B = B.astype(dtype)

    if scale:
        s = jnp.sqrt(jnp.diag(B))
        S_inv = 1.0 / s
        A = (S_inv[:, None] * A) * S_inv[None, :]
        B = (S_inv[:, None] * B) * S_inv[None, :]

    l_min = jnp.min(jnp.linalg.eigvalsh(B))
    shift = jnp.where(l_min < eps, eps - l_min, 0.0)
    B = B + shift * jnp.eye(B.shape[-1], dtype=dtype)

    L = jnp.linalg.cholesky(B)
    Y = jsp.solve_triangular(L, A, lower=True, trans="N")
    C = jsp.solve_triangular(L, Y.T, lower=True, trans="N").T
    C = (C + C.T.conj()) * 0.5

    w, U = jnp.linalg.eigh(C)
    V = jsp.solve_triangular(L.T, U, lower=False, trans="N")

    return w, V


# -------------------------------------------------------------
# Even more robust eigensolvers
# -------------------------------------------------------------


@jax.custom_vjp
def degen_eigh(
    A: Float[Array, "... n n"]
) -> tuple[Float[Array, "... n"], Float[Array, "... n n"]]:
    """Symmetric eigendecomposition that handles degenerate cases in reverse mode.

    Args:
      A: A real-symmetric or Hermitian matrix of shape (..., n, n).

    Returns:
      A tuple (w, v) where:
        - w: Eigenvalues in ascending order of shape (..., n).
        - v: Corresponding eigenvectors in columns of shape (..., n, n).
    """
    eival, eivec = jnp.linalg.eigh(A)
    return eival, eivec


def degen_eigh_fwd(
    A: Float[Array, "... n n"]
) -> tuple[
    tuple[Float[Array, "... n"], Float[Array, "... n n"]],
    tuple[Float[Array, "... n"], Float[Array, "... n n"]],
]:
    """Forward pass for degen_eigh.

    Args:
      A: Input matrix of shape (..., n, n).

    Returns:
      A tuple (output, residuals) where:
        - output: (evals, evecs)
        - residuals: (evals, evecs)
    """
    eival, eivec = degen_eigh(A)
    return (eival, eivec), (eival, eivec)


def degen_eigh_bwd(
    res: tuple[Float[Array, "... n"], Float[Array, "... n n"]],
    grads: tuple[Float[Array, "... n"], Float[Array, "... n n"]],
) -> tuple[Float[Array, "... n n"]]:
    """Backward pass for degen_eigh handling degeneracies via F-matrix regularization.

    For degenerate eigenvalues, the standard gradient of eigenvectors is
    ill-defined. We apply a thresholding approach suggested by Kasim et al. to
    provide stable gradients when the loss function depends only on the space
    spanned by the degenerate eigenvectors (common in DFT).

    Args:
      res: Saved values from the forward pass (eiva, eivec).
      grads: Gradients w.r.t. the outputs (grad_eival, grad_eivec).

    Returns:
      A tuple containing the gradient of the loss w.r.t. the input matrix A.
    """
    eival, eivec = res
    grad_eival, grad_eivec = grads
    in_debug_mode = is_debug_enabled()

    # Numerical threshold for detecting degeneracies
    min_threshold = jnp.finfo(eival.dtype).eps**0.6
    eivect = jnp.transpose(eivec, axes=(-2, -1))

    # Initialize gradient matrix
    result = jnp.zeros_like(eivec)

    if grad_eivec is not None:
        # Calculate F matrix: F_ij = 1 / (λ_i - λ_j)
        F = jnp.expand_dims(eival, -2) - jnp.expand_dims(eival, -1)
        idx = jnp.abs(F) < min_threshold
        # Set degenerate terms to infinity to effectively zero them out after reciprocal
        F = jnp.where(idx, jnp.inf, F)

        if in_debug_mode:
            # Check if gradient requirements for degenerate subspaces are met
            xtg = eivect @ grad_eivec
            diff_xtg = (xtg - jnp.transpose(xtg, axes=(-2, -1)))[idx]
            req_satisfied = jnp.allclose(diff_xtg, jnp.zeros_like(diff_xtg))
            if not req_satisfied:
                print("Warning: Degeneracy requirements not fully met in backward pass.")

        F = jnp.power(F, -1)
        F = F * jnp.matmul(eivect, grad_eivec)
        result = jnp.matmul(eivec, jnp.matmul(F, eivect))

    if grad_eival is not None:
        # Contribution from eigenvalues: v @ diag(grad_λ) @ v.T
        result = result + jnp.matmul(eivec, jnp.expand_dims(grad_eival, -1) * eivect)

    # Gradient must be symmetric for symmetric input
    result = (result + jnp.transpose(result, axes=(-2, -1))) * 0.5

    return (result,)


degen_eigh.defvjp(degen_eigh_fwd, degen_eigh_bwd)


@partial(jax.jit, static_argnames=("eps", "scale", "dtype"))
def safe_generalized_eigh(
    A: Float[Array, "... n n"],
    B: Float[Array, "... n n"],
    *,
    eps: float = 1.0e-12,
    scale: bool = False,
    dtype: Any = jnp.float64,
) -> tuple[Float[Array, "... n"], Float[Array, "... n n"]]:
    """Differentiable generalized eigensolver (A v = λ B v) for degenerate cases.

    Uses Cholesky decomposition of B followed by degen_eigh for differentiation.

    Args:
      A: Real-symmetric or Hermitian batch of matrices of shape (..., n, n).
      B: Real-symmetric or Hermitian batch of matrices of shape (..., n, n).
        Must be SPD.
      eps: Lower bound for eigenvalues of B to ensure stability.
      scale: Whether to apply Jacobi (diagonal) scaling to improve conditioning.
      dtype: Numerical precision to use for the calculation.

    Returns:
      A tuple (evals, evecs) where evals contains eigenvalues and evecs contains
      eigenvectors in columns.
    """
    A = (A + A.T.conj()) * 0.5
    B = (B + B.T.conj()) * 0.5
    A = A.astype(dtype)
    B = B.astype(dtype)

    if scale:
        s = jnp.sqrt(jnp.diag(B))
        S_inv = 1.0 / s
        A = (S_inv[:, None] * A) * S_inv[None, :]
        B = (S_inv[:, None] * B) * S_inv[None, :]

    # Ensure B is Symmetric Positive Definite (SPD)
    lambda_min = jnp.min(jnp.linalg.eigvalsh(B))
    shift = jnp.where(lambda_min < eps, eps - lambda_min, 0.0)
    B = B + shift * jnp.eye(B.shape[-1], dtype=dtype)

    # Cholesky decomposition: B = L @ L.T
    L = jnp.linalg.cholesky(B)

    # Transform to standard eigenvalue problem: L^-1 @ A @ L^-T
    Y = jsp.solve_triangular(L, A, lower=True, trans="N")
    C = jsp.solve_triangular(L, Y.T, lower=True, trans="N").T
    C = (C + C.T.conj()) * 0.5

    # Solve the transformed problem using degenerate-aware solver
    w, U = degen_eigh(C)

    # Back-transform eigenvectors: v = L^-T @ U
    V = jsp.solve_triangular(L.T, U, lower=False, trans="N")

    return w, V


# -------------------------------------------------------------
# Subspace approach, use perturbation strength to avoid singularity
# -------------------------------------------------------------


@jax.custom_vjp
def subspace_eigh(
    A: Float[Array, "... n n"], grad_eps: float = 0.05
) -> tuple[Float[Array, "... n"], Float[Array, "... n n"]]:
    """Eigendecomposition with subspace-based gradients to avoid singularity.

    Designed for DFT where loss depends on the density matrix (occupied subspace)
    rather than individual eigenvectors.

    Args:
      A: Batch of real-symmetric matrices of shape (..., n, n).
      grad_eps: Smoothing parameter for Lorentzian regularization.

    Returns:
      A tuple (evals, evecs) containing eigenvalues and eigenvectors.
    """
    eival, eivec = jnp.linalg.eigh(A)
    return eival, eivec


def subspace_eigh_fwd(
    A: Float[Array, "... n n"], grad_eps: float
) -> tuple[
    tuple[Float[Array, "... n"], Float[Array, "... n n"]],
    tuple[Float[Array, "... n n"], Float[Array, "... n"], Float[Array, "... n n"], float],
]:
    """Forward pass for subspace_eigh.

    Args:
      A: Input matrix of shape (..., n, n).
      grad_eps: Smoothing parameter.

    Returns:
      A tuple (output, residuals).
    """
    eival, eivec = subspace_eigh(A, grad_eps)
    return (eival, eivec), (A, eival, eivec, grad_eps)


def subspace_eigh_bwd(
    res: tuple[Float[Array, "... n n"], Float[Array, "... n"], Float[Array, "... n n"], float],
    grads: tuple[Float[Array, "... n"], Float[Array, "... n n"]],
) -> tuple[Float[Array, "... n n"], None]:
    """Backward pass using Lorentzian regularization for subspace stability.

    Uses a Lorentzian broadening approach to prevent gradient explosions near
    degenerate points. This is particularly effective when the physical
    observable is invariant to rotations within the degenerate subspace.

    Args:
      res: Saved values (A, eival, eivec, grad_eps).
      grads: Output gradients (grad_eival, grad_eivec).

    Returns:
      A tuple containing the gradient w.r.t. input A and None for grad_eps.
    """
    A, eival, eivec, grad_eps = res
    grad_eival, grad_eivec = grads

    result = jnp.zeros_like(A)
    eivect = jnp.transpose(eivec, axes=(-2, -1))

    # Contribution from eigenvalues
    if grad_eival is not None:
        result = result + jnp.matmul(eivec, jnp.expand_dims(grad_eival, -1) * eivect)

    # Contribution from eigenvectors using Lorentzian broadening
    if grad_eivec is not None:
        # Skew-symmetric part of eigenvector gradients
        VtG = jnp.matmul(eivect, grad_eivec)
        skew = (VtG - jnp.transpose(VtG, axes=(-2, -1))) * 0.5

        # Lorentzian regularization: F / (F^2 + eps^2)
        F = jnp.expand_dims(eival, -2) - jnp.expand_dims(eival, -1)
        F_reg = F / (F**2 + grad_eps**2)

        # Zero out diagonal to avoid self-interaction
        n = eival.shape[-1]
        F_reg = F_reg * (1.0 - jnp.eye(n, dtype=eival.dtype))

        # Clipping as a secondary safety measure
        max_grad = 1.0 / (2.0 * grad_eps)
        F_reg = jnp.clip(F_reg, -max_grad, max_grad)

        skew_scaled = skew * F_reg
        result = result + jnp.matmul(eivec, jnp.matmul(skew_scaled, eivect))

    result = (result + jnp.transpose(result, axes=(-2, -1))) * 0.5

    return (result, None)


subspace_eigh.defvjp(subspace_eigh_fwd, subspace_eigh_bwd)


@partial(jax.jit, static_argnames=("eps", "scale", "dtype"))
def subspace_generalized_eigh(
    A: Float[Array, "... n n"],
    B: Float[Array, "... n n"],
    *,
    eps: float = 1.0e-12,
    scale: bool = False,
    perturbation_strength: float = 1.0e-10,
    grad_eps: float = 0.05,
    dtype: Any = jnp.float64,
) -> tuple[Float[Array, "... n"], Float[Array, "... n n"]]:
    """Robust subspace-based generalized eigensolver for DFT training.

    Combines symmetry-breaking perturbations (for stable ordering) with
    subspace-based Lorentzian regularization for gradient stability.

    Args:
      A: Real-symmetric batch of shape (..., n, n).
      B: Real-symmetric SPD batch of shape (..., n, n).
      eps: Stability shift for B.
      scale: Whether to apply diagonal scaling.
      perturbation_strength: Strength of orbit-index dependent perturbation.
      grad_eps: Lorentzian width for gradients.
      dtype: Working precision.

    Returns:
      A tuple (evals, evecs) containing eigenvalues of shape (..., n) and
      eigenvectors of shape (..., n, n).
    """
    A = (A + A.T.conj()) * 0.5
    B = (B + B.T.conj()) * 0.5
    A = A.astype(dtype)
    B = B.astype(dtype)

    if scale:
        s = jnp.sqrt(jnp.diag(B))
        S_inv = 1.0 / s
        A = (S_inv[:, None] * A) * S_inv[None, :]
        B = (S_inv[:, None] * B) * S_inv[None, :]

    lambda_min = jnp.min(jnp.linalg.eigvalsh(B))
    shift = jnp.where(lambda_min < eps, eps - lambda_min, 0.0)
    B = B + shift * jnp.eye(B.shape[-1], dtype=dtype)

    # Lift degeneracy in forward pass to ensure deterministic ordering
    n = A.shape[-1]
    perturbation = jnp.diag(jnp.arange(n, dtype=dtype) * perturbation_strength)
    A = A + perturbation

    L = jnp.linalg.cholesky(B)
    Y = jsp.solve_triangular(L, A, lower=True, trans="N")
    C = jsp.solve_triangular(L, Y.T, lower=True, trans="N").T
    C = (C + C.T.conj()) * 0.5

    # Diagonalize with stable subspace gradients
    w, U = subspace_eigh(C, grad_eps=grad_eps)

    V = jsp.solve_triangular(L.T, U, lower=False, trans="N")

    return w, V


# -----------------------------------------
# Solver based on broadening and shifting
# -----------------------------------------


if __name__ == "__main__":

    jax.config.update("jax_enable_x64", True)

    print("\n=== Testing gradient behavior with different eigensolvers ===\n")

    A_test = jnp.array([[4.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=jnp.float64)
    B_test = jnp.array([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=jnp.float64)

    def test_eigensolver(name, solver_fn, *args):
        print(f"\n--- Testing {name} ---")

        def get_eigenvector_component(*solver_args, i, j):
            return solver_fn(*solver_args)[1][i, j]

        grad_fn = jax.grad(get_eigenvector_component, argnums=0)

        # Print current values
        try:
            _, V_res = solver_fn(*args)
            print(f"Eigenvectors:\n{V_res}")

            # Note: This might still NaN for individual components if not using subspace methods
            dV = jnp.array([[grad_fn(*args, i=i, j=j) for j in range(3)] for i in range(3)])
            print(f"Gradients of eigenvectors:\n{dV}")

            if jnp.any(jnp.isnan(dV)):
                print(f"Result: {name} produced NaNs as expected.")
            else:
                print(f"Result: {name} successfully computed gradients without NaNs.")
        except Exception as e:
            print(f"Result: {name} failed with error: {e}")

    test_eigensolver("Standard linalg.eigh", lambda x: (jnp.linalg.eigh(x)), A_test)
    test_eigensolver("Degenerate-aware eigh", degen_eigh, A_test)
    test_eigensolver("Subspace-aware eigh", subspace_eigh, A_test)
    test_eigensolver("Generalized eigh", generalized_eigh, A_test, B_test)
    test_eigensolver("Safe Generalized eigh", safe_generalized_eigh, A_test, B_test)
    test_eigensolver("Subspace Generalized eigh", subspace_generalized_eigh, A_test, B_test)

    print("\n=== Generalized Eigensolve Verification ===")
    A_val = jnp.array([[1.0, 2.0], [2.0, 3.0]])
    B_val = jnp.array([[2.0, 0.5], [0.5, 1.0]])

    scipy_w, _ = standard_eig(A_val, B_val)
    jax_w, v_jax = jax_eig(A_val, B_val)
    print(f"Scipy w: {scipy_w}")
    print(f"JAX w:   {jax_w}")

    for i in range(len(jax_w)):
        res = jnp.linalg.norm(A_val @ v_jax[:, i] - jax_w[i] * B_val @ v_jax[:, i])
        print(f"Residual {i}: {res:.2e}")
