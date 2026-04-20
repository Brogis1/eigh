"""Stable wrappers for the eigh package with Lorentzian broadening.

This module wraps the eigh package's differentiable eigensolvers with custom
VJPs that use Lorentzian broadening for numerical stability with degenerate
eigenvalues.

Based on: Colburn, S., & Majumdar, A. Communications Physics 4, 54 (2021).
https://doi.org/10.1038/s42005-021-00568-6

We do not use the Cholesky here as in
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import custom_vjp
from jaxtyping import Array, Float
from eigh import eigh, eigh_gen

# Default parameters for degeneracy tolerance and broadening
DEGENERACY_TOLERANCE = 1e-6
BROADENING = 1e-10


def _compute_stable_F_matrix(evals: Float[Array, "n"]) -> Float[Array, "n n"]:
    """Compute the F matrix with Lorentzian broadening for stable gradients.

    Args:
        evals: Eigenvalues of shape (n,).

    Returns:
        F matrix of shape (n, n) with stabilized inverse differences.
    """
    eval_diff = evals.reshape((1, -1)) - evals.reshape((-1, 1))

    mask_degen = (jnp.abs(eval_diff) < DEGENERACY_TOLERANCE).astype(jnp.float64)
    mask_non_degen = 1.0 - mask_degen

    # Regular gap: 1/(e_j - e_i) for non-degenerate
    regular_gap = jnp.where(
        mask_non_degen > 0.5,
        1.0 / jnp.where(jnp.abs(eval_diff) > 1e-12, eval_diff, 1.0),
        0.0,
    )

    # Lorentzian broadened gap: (e_j - e_i)/((e_j - e_i)^2 + eps) for degenerate
    broadened_gap = eval_diff / (eval_diff * eval_diff + BROADENING)

    F = 0.5 * (mask_non_degen * regular_gap + mask_degen * broadened_gap)
    F = F.at[jnp.diag_indices_from(F)].set(0.0)

    return F


@custom_vjp
def stable_eigh_pyscfad(
    A: Float[Array, "n n"],
) -> tuple[Float[Array, "n"], Float[Array, "n n"]]:
    """Computes eigenvalues and eigenvectors using eigh with stable gradients.

    Args:
        A: A real symmetric matrix of shape (n, n).

    Returns:
        A tuple (evals, evecs) where:
            - evals: Eigenvalues in ascending order of shape (n,).
            - evecs: Eigenvectors of shape (n, n).
    """
    return eigh(A)


def _stable_eigh_pyscfad_fwd(
    A: Float[Array, "n n"],
) -> tuple[
    tuple[Float[Array, "n"], Float[Array, "n n"]],
    tuple[Float[Array, "n"], Float[Array, "n n"]],
]:
    """Forward pass for stable_eigh_pyscfad."""
    evals, evecs = eigh(A)
    return (evals, evecs), (evals, evecs)


def _stable_eigh_pyscfad_rev(
    res: tuple[Float[Array, "n"], Float[Array, "n n"]],
    g: tuple[Float[Array, "n"], Float[Array, "n n"]],
) -> tuple[Float[Array, "n n"]]:
    """Backward pass with Lorentzian broadening for stability."""
    evals, evecs = res
    grad_evals, grad_evecs = g

    grad_evals_diag = jnp.diag(grad_evals)
    evecs_trans = evecs.T

    F = _compute_stable_F_matrix(evals)

    grad = (
        jnp.linalg.inv(evecs_trans)
        @ (0.5 * grad_evals_diag + F * (evecs_trans @ grad_evecs))
        @ evecs_trans
    )

    return (grad + grad.T,)


stable_eigh_pyscfad.defvjp(_stable_eigh_pyscfad_fwd, _stable_eigh_pyscfad_rev)


@custom_vjp
def stable_eigh_gen_pyscfad(
    A: Float[Array, "n n"], B: Float[Array, "n n"]
) -> tuple[Float[Array, "n"], Float[Array, "n n"]]:
    """Solves generalized eigenvalue problem AC = BCE using eigh_gen with stable gradients.

    Args:
        A: A real symmetric matrix of shape (n, n).
        B: A real symmetric positive-definite matrix of shape (n, n).

    Returns:
        A tuple (evals, evecs) where:
            - evals: Generalized eigenvalues of shape (n,).
            - evecs: Generalized eigenvectors of shape (n, n).
    """
    return eigh_gen(A, B)


def _stable_eigh_gen_pyscfad_fwd(
    A: Float[Array, "n n"], B: Float[Array, "n n"]
) -> tuple[
    tuple[Float[Array, "n"], Float[Array, "n n"]],
    tuple[Float[Array, "n"], Float[Array, "n n"], Float[Array, "n n"]],
]:
    """Forward pass for stable_eigh_gen_pyscfad."""
    evals, evecs = eigh_gen(A, B)
    return (evals, evecs), (evals, evecs, B)


def _stable_eigh_gen_pyscfad_rev(
    res: tuple[Float[Array, "n"], Float[Array, "n n"], Float[Array, "n n"]],
    g: tuple[Float[Array, "n"], Float[Array, "n n"]],
) -> tuple[Float[Array, "n n"], Float[Array, "n n"]]:
    """Backward pass for generalized eigenproblem with Lorentzian broadening.

    For generalized eigenproblem AC = BCE, the gradient formulas are:
        grad_A = C @ (diag(grad_evals) + F * (C^T B grad_C)) @ C^T B
        grad_B = -C @ (diag(evals * grad_evals) + F * (C^T B grad_C) * evals) @ C^T B
    """
    evals, evecs, B = res
    grad_evals, grad_evecs = g

    F = _compute_stable_F_matrix(evals)

    # C^T B
    CtB = evecs.T @ B

    # C^T B grad_C
    CtB_grad_C = CtB @ grad_evecs

    # Common term: diag(grad_evals) + F * (C^T B grad_C)
    middle = jnp.diag(grad_evals) + F * CtB_grad_C

    # grad_A = C @ middle @ C^T B
    grad_A = evecs @ middle @ CtB

    # grad_B = -C @ (diag(evals) @ middle + middle @ diag(evals)) / 2 @ C^T B
    # Simplified: grad_B involves eigenvalues weighting
    evals_diag = jnp.diag(evals)
    middle_B = jnp.diag(grad_evals * evals) + F * CtB_grad_C * evals.reshape(1, -1)
    grad_B = -evecs @ middle_B @ CtB

    # Symmetrize gradients
    grad_A = 0.5 * (grad_A + grad_A.T)
    grad_B = 0.5 * (grad_B + grad_B.T)

    return (grad_A, grad_B)


stable_eigh_gen_pyscfad.defvjp(_stable_eigh_gen_pyscfad_fwd, _stable_eigh_gen_pyscfad_rev)


def stable_fock_solver_pyscfad(
    fock: Float[Array, "spin orbitals orbitals"], overlap: Float[Array, "orbitals orbitals"]
) -> tuple[Float[Array, "spin orbitals"], Float[Array, "spin orbitals orbitals"]]:
    """Computes eigenenergies and MO coefficients for spin-polarized Fock matrices.

    Args:
        fock: Spin-polarized Fock matrices of shape (spin, orbitals, orbitals).
        overlap: Overlap matrix of shape (orbitals, orbitals).

    Returns:
        A tuple (mo_energies, mo_coeffs) where:
            - mo_energies: Molecular orbital energies of shape (spin, orbitals).
            - mo_coeffs: Molecular orbital coefficients of shape (spin, orbitals, orbitals).
    """
    mo_energies_up, mo_coeffs_up = stable_eigh_gen_pyscfad(fock[0], overlap)
    mo_energies_dn, mo_coeffs_dn = stable_eigh_gen_pyscfad(fock[1], overlap)
    return jnp.stack((mo_energies_up, mo_energies_dn), axis=0), jnp.stack(
        (mo_coeffs_up, mo_coeffs_dn), axis=0
    )


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)

    # 1. Test standard eigensolver
    print("Testing stable_eigh_pyscfad...")
    A = jnp.array([[2.0, 1.0], [1.0, 2.0]])
    evals, evecs = stable_eigh_pyscfad(A)
    print(f"Eigenvalues: {evals}")
    assert jnp.allclose(A @ evecs, evecs @ jnp.diag(evals))

    # 2. Test generalized eigensolver
    print("\nTesting stable_eigh_gen_pyscfad...")
    B = jnp.array([[1.0, 0.5], [0.5, 1.0]])
    evals_gen, evecs_gen = stable_eigh_gen_pyscfad(A, B)
    print(f"General Eigenvalues: {evals_gen}")
    assert jnp.allclose(A @ evecs_gen, B @ evecs_gen @ jnp.diag(evals_gen))

    # 3. Test Fock solver
    print("\nTesting stable_fock_solver_pyscfad...")
    fock = jnp.stack([A, A * 1.1], axis=0)
    mo_energies, mo_coeffs = stable_fock_solver_pyscfad(fock, B)
    print(f"MO Energies shape: {mo_energies.shape}")
    assert mo_energies.shape == (2, 2)

    # 4. Test gradients
    print("\nTesting gradients...")

    def sum_eigenvalues(mat):
        ev, _ = stable_eigh_pyscfad(mat)
        return jnp.sum(ev)

    grad_non_degen = jax.grad(sum_eigenvalues)(A)
    print(f"Gradient at non-degenerate matrix:\n{grad_non_degen}")
    assert not jnp.any(jnp.isnan(grad_non_degen))

    # Test degenerate case
    print("Testing degenerate gradient (Identity matrix)...")
    A_degen = jnp.eye(2)
    grad_degen = jax.grad(sum_eigenvalues)(A_degen)
    print(f"Gradient at identity:\n{grad_degen}")
    assert not jnp.any(jnp.isnan(grad_degen))

    # Test generalized eigensolver gradient
    print("\nTesting generalized eigensolver gradient...")

    def sum_gen_eigenvalues(mat_A, mat_B):
        ev, _ = stable_eigh_gen_pyscfad(mat_A, mat_B)
        return jnp.sum(ev)

    grad_A, grad_B = jax.grad(sum_gen_eigenvalues, argnums=(0, 1))(A, B)
    print(f"Gradient w.r.t. A:\n{grad_A}")
    print(f"Gradient w.r.t. B:\n{grad_B}")
    assert not jnp.any(jnp.isnan(grad_A))
    assert not jnp.any(jnp.isnan(grad_B))

    # Test with larger degenerate matrix
    print("\nTesting large degenerate matrix...")
    key = jax.random.PRNGKey(0)
    Q, _ = jnp.linalg.qr(jax.random.normal(key, (10, 10)))
    A_large_degen = Q @ jnp.diag(jnp.array([1.0] * 5 + [2.0] * 5)) @ Q.T
    grad_large_degen = jax.grad(sum_eigenvalues)(A_large_degen)
    print(f"Gradient at large degenerate matrix (no NaNs): {not jnp.any(jnp.isnan(grad_large_degen))}")
    assert not jnp.any(jnp.isnan(grad_large_degen))

    print("\nAll tests passed!")