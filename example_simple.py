from eigh import eigh, eigh_gen
import jax.numpy as jnp
import jax

print(jax.devices())

# Standard eigenvalue problem
A = jnp.array([[2., 1.], [1., 2.]])
w, v = eigh(A)
# w, v = eigh_gen(A)

# With gradients
grad = jax.grad(lambda A: eigh(A)[0].sum())(A)
print("Eigenvalues:", w)
print("Eigenvectors:", v)
print("Gradient computed:", grad.shape)
