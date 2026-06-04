from eigh import eigh, eigh_gen
import jax.numpy as jnp
import jax

print(jax.devices())

# Standard eigenvalue problem
A = jnp.array([[2., 1.], [1., 2.]])
B = jnp.array([[1., 1], [0.5, 1.]])
w1, v1 = eigh(A)
w2, v2 = eigh_gen(A, B)

# With gradients
grad1 = jax.grad(lambda A: eigh(A)[0].sum())(A)
grad2 = jax.grad(lambda A: eigh_gen(A, B)[0].sum())(A)
print("Eigenvalues:", w1, w2)
print("Eigenvectors:", v1, v2)1
print("Gradients computed:", grad1.shape, grad2.shape)
