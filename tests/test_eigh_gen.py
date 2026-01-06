# Copyright 2021-2025 Xing Zhang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Tests specifically for the eigh_gen function.
"""

import os
import sys
import pytest
import numpy as np
import jax
from jax import numpy as jnp

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)

# Add parent directory to path to import eigh
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..',
                                'src', 'python'))

try:
    from eigh import eigh_gen
except ImportError as e:
    print(f"Failed to import eigh_gen: {e}")
    sys.exit(1)


def generate_pos_def(n, rng, dtype=jnp.float64):
    """Generate a positive definite matrix."""
    a = rng.standard_normal((n, n)).astype(dtype)
    if jnp.issubdtype(dtype, jnp.complexfloating):
        a = a + 1j * rng.standard_normal((n, n)).astype(dtype)
    return a @ a.T.conj() + n * jnp.eye(n, dtype=dtype)


def generate_hermitian(n, rng, dtype=jnp.float64):
    """Generate a symmetric/hermitian matrix."""
    a = rng.standard_normal((n, n)).astype(dtype)
    if jnp.issubdtype(dtype, jnp.complexfloating):
        a = a + 1j * rng.standard_normal((n, n)).astype(dtype)
    return (a + a.T.conj()) / 2


def test_eigh_gen_itype_1():
    """Test itype=1: A @ v = B @ v @ diag(w)"""
    print("\nTest: eigh_gen itype=1")
    rng = np.random.default_rng(42)
    n = 3
    a = generate_hermitian(n, rng)
    b = generate_pos_def(n, rng)

    w, v = eigh_gen(a, b, itype=1)

    # Equation: A @ V = B @ V @ W
    lhs = a @ v
    rhs = b @ v @ jnp.diag(w)
    residual = jnp.max(jnp.abs(lhs - rhs))
    assert residual < 1e-12, f"itype=1 failed: {residual}"


def test_eigh_gen_itype_2():
    """Test itype=2: A @ B @ v = v @ diag(w)"""
    print("\nTest: eigh_gen itype=2")
    rng = np.random.default_rng(43)
    n = 3
    a = generate_hermitian(n, rng)
    b = generate_pos_def(n, rng)

    w, v = eigh_gen(a, b, itype=2)

    # Equation: A @ B @ V = V @ W
    lhs = a @ b @ v
    rhs = v @ jnp.diag(w)
    residual = jnp.max(jnp.abs(lhs - rhs))
    assert residual < 1e-12, f"itype=2 failed: {residual}"


def test_eigh_gen_itype_3():
    """Test itype=3: B @ A @ v = v @ diag(w)"""
    print("\nTest: eigh_gen itype=3")
    rng = np.random.default_rng(44)
    n = 3
    a = generate_hermitian(n, rng)
    b = generate_pos_def(n, rng)

    w, v = eigh_gen(a, b, itype=3)

    # Equation: B @ A @ V = V @ W
    lhs = b @ a @ v
    rhs = v @ jnp.diag(w)
    residual = jnp.max(jnp.abs(lhs - rhs))
    assert residual < 1e-12, f"itype=3 failed: {residual}"


def test_eigh_gen_lower_flag():
    """Test that eigh_gen currently averages triangles (symmetrize behavior)."""
    print("\nTest: eigh_gen lower vs upper (current averaging behavior)")
    # Non-symmetric matrix
    a = jnp.array([[1.0, 0.5],
                   [2.0, 1.0]])
    b = jnp.eye(2)

    # Current behavior in _core.py: a = symmetrize(a)
    # symmetrize(a) = [[1.0, 1.25], [1.25, 1.0]]
    # w = [1+1.25, 1-1.25] = [2.25, -0.25]
    expected = jnp.array([-0.25, 2.25])

    w_low, _ = eigh_gen(a, b, lower=True)
    assert jnp.allclose(jnp.sort(w_low), expected)

    w_up, _ = eigh_gen(a, b, lower=False)
    assert jnp.allclose(jnp.sort(w_up), expected)


def test_eigh_gen_jit_vmap():
    """Test compatibility with jax.jit and jax.vmap."""
    print("\nTest: eigh_gen with JIT and vmap")

    @jax.jit
    @jax.vmap
    def batched_eigh_gen(a_batch, b_batch):
        return eigh_gen(a_batch, b_batch, itype=1)

    rng = np.random.default_rng(45)
    batch_size = 4
    n = 3
    a_batch = jnp.stack([generate_hermitian(n, rng)
                         for _ in range(batch_size)])
    b_batch = jnp.stack([generate_pos_def(n, rng)
                         for _ in range(batch_size)])

    w, v = batched_eigh_gen(a_batch, b_batch)

    assert w.shape == (batch_size, n)
    assert v.shape == (batch_size, n, n)

    # Check one batch element
    lhs = a_batch[0] @ v[0]
    rhs = b_batch[0] @ v[0] @ jnp.diag(w[0])
    assert jnp.max(jnp.abs(lhs - rhs)) < 1e-12


def test_eigh_gen_gradients():
    """Test gradients for itype=1."""
    print("\nTest: eigh_gen gradients (itype=1)")

    def loss(a, b):
        w, _ = eigh_gen(a, b, itype=1)
        return jnp.sum(w**2)

    rng = np.random.default_rng(46)
    n = 3
    a = generate_hermitian(n, rng)
    b = generate_pos_def(n, rng)

    # Autodiff
    grad_a, grad_b = jax.grad(loss, argnums=(0, 1))(a, b)

    # Finite differences
    eps = 1e-6
    da = generate_hermitian(n, rng)
    db = generate_hermitian(n, rng)

    f0 = loss(a, b)
    f_a = loss(a + eps * da, b)
    f_b = loss(a, b + eps * db)

    df_a_expected = jnp.sum(grad_a * da)
    df_a_actual = (f_a - f0) / eps

    df_b_expected = jnp.sum(grad_b * db)
    df_b_actual = (f_b - f0) / eps

    assert jnp.abs(df_a_expected - df_a_actual) < 1e-5
    assert jnp.abs(df_b_expected - df_b_actual) < 1e-5


def test_eigh_gen_itype_grad_error():
    """Verify that itype != 1 raises NotImplementedError for gradients."""
    print("\nTest: eigh_gen unimplemented gradients (itype=2,3)")
    a = jnp.eye(2)
    b = jnp.eye(2)

    def loss(a, b, itype):
        w, _ = eigh_gen(a, b, itype=itype)
        return jnp.sum(w)

    with pytest.raises(NotImplementedError, match="itype=2 is not implemented"):
        jax.grad(lambda x: loss(x, b, 2))(a)

    with pytest.raises(NotImplementedError, match="itype=3 is not implemented"):
        jax.grad(lambda x: loss(x, b, 3))(a)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
