# Handling Degenerate Eigenvalues

## The Problem

When eigenvalues are **degenerate** (repeated), the eigenvectors are **not unique**. Any linear combination of eigenvectors in the degenerate subspace is a valid eigenvector.

### Example

For a matrix with eigenvalue λ = 2 with multiplicity 3:
```
A @ v₁ = 2 * v₁
A @ v₂ = 2 * v₂
A @ v₃ = 2 * v₃
```

Then **any** linear combination is also valid:
```
A @ (α*v₁ + β*v₂ + γ*v₃) = 2 * (α*v₁ + β*v₂ + γ*v₃)
```

## Impact on Gradients

### Ill-Defined: Individual Eigenvalue Gradients

The gradient of an individual eigenvalue with respect to the matrix is **ill-defined** when that eigenvalue is degenerate:

```python
def smallest_eigenvalue(A):
    w, _ = eigh(A)
    return w.min()  # Gradient ill-defined if w.min() is degenerate

grad = jax.grad(smallest_eigenvalue)(A)  # May give incorrect results
```

**Why?** Small perturbations can cause discontinuous changes in which eigenvector is returned, leading to unstable gradients.

### Well-Defined: Symmetric Functions

Symmetric functions of eigenvalues (functions invariant to permutation) have well-defined gradients:

```python
def trace(A):
    w, _ = eigh(A)
    return w.sum()  # Sum is symmetric

def variance(A):
    w, _ = eigh(A)
    return jnp.var(w)  # Variance is symmetric

def sum_of_squares(A):
    w, _ = eigh(A)
    return (w ** 2).sum()  # Symmetric function
```

## The `deg_thresh` Parameter

The `deg_thresh` parameter controls how close eigenvalues must be to be considered degenerate:

```python
w, v = eigh(A, deg_thresh=1e-9)  # Default threshold
```

### How It Works

In the JVP (forward-mode differentiation) rule, the implementation computes:

```python
eji = w[..., np.newaxis, :] - w[..., np.newaxis]  # eigenvalue differences
Fmat = 1 / eji  # Problematic when eji ≈ 0

# Solution: mask near-zero differences
Fmat = reciprocal(
    jnp.where(abs(eji) > deg_thresh, eji, np.inf)
)
```

When `|λᵢ - λⱼ| < deg_thresh`, the algorithm treats them as degenerate and handles them specially.

### Choosing `deg_thresh`

- **Too small** (e.g., 1e-15): Numerical instability, large gradients
- **Too large** (e.g., 1e-3): May incorrectly treat distinct eigenvalues as degenerate
- **Default** (1e-9): Good balance for most cases

```python
# For nearly degenerate systems
w, v = eigh(A, deg_thresh=1e-6)  # More robust

# For well-separated eigenvalues
w, v = eigh(A, deg_thresh=1e-12)  # Higher precision
```

## Testing Degenerate Cases

### What to Test

**DO test:**
- Forward pass (eigenvalues and eigenvectors)
- Eigenvalue equation: `A @ v = λ * v`
- Gradients of symmetric functions (trace, variance, etc.)
- Generalized problems: `A @ v = B @ v @ λ`

**DON'T test:**
- Gradients of individual degenerate eigenvalues
- Exact eigenvector matching (not unique)

### Example Test

```python
def test_degenerate():
    # Create matrix with degeneracy: eigenvalues [1, 1, 2, 2]
    D = jnp.diag(jnp.array([1.0, 1.0, 2.0, 2.0]))
    Q, _ = np.linalg.qr(np.random.randn(4, 4))
    A = Q @ D @ Q.T

    # Test eigenvalue equation
    w, v = eigh(A)
    residual = jnp.max(jnp.abs(A @ v - v @ jnp.diag(w)))
    assert residual < 1e-10

    # Test gradient of trace
    def trace(A):
        w, _ = eigh(A)
        return w.sum()

    grad_trace = jax.grad(trace)(A)
    # Verify: d/dA[tr(A)] = I
    assert jnp.allclose(grad_trace, jnp.eye(4), atol=1e-6)

    # DON'T test gradient of min eigenvalue
    # def min_eig(A):
    #     w, _ = eigh(A)
    #     return w.min()  # Ill-defined if min is degenerate
```

## Mathematical Background

### Perturbation Theory

For a simple (non-degenerate) eigenvalue λ, first-order perturbation gives:

```
dλ = v^H @ dA @ v
```

where v is the corresponding eigenvector.

For **degenerate** eigenvalues, this formula breaks down because v is not unique. The correct approach requires **degenerate perturbation theory**, which the implementation approximates using the `deg_thresh` masking.

### Frobenius Formula

The implementation uses a generalized Frobenius formula:

```
dV = V @ (F ⊙ (V^H @ dA @ V))
```

where:
- `F[i,j] = 1/(λᵢ - λⱼ)` for non-degenerate pairs
- `F[i,j] = 0` (or masked) when `|λᵢ - λⱼ| < deg_thresh`

This prevents division by zero while maintaining correctness for symmetric functions.

## Practical Recommendations

1. **Use symmetric functions** for loss functions involving eigenvalues
2. **Tune `deg_thresh`** based on your eigenvalue spacing
3. **Check conditioning**: If eigenvalues are very close, consider:
   - Increasing `deg_thresh`
   - Regularizing your problem
   - Using alternative formulations

4. **Validate gradients** with finite differences for symmetric functions only

## See Also

- Original test: [tests/test_eigh.py](tests/test_eigh.py) - `test_degenerate_eigenvalues()`
- Implementation: [src/python/eigh.py](src/python/eigh.py) - `_eigh_gen_jvp_rule()`
- Paper: Magnus, J. R. (1985). "On differentiating eigenvalues and eigenvectors"
