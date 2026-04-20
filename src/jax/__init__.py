"""
JAX-based eigenvalue solvers with CPU and GPU backend support.
"""

from .generalized_eigensolver import (
    safe_generalized_eigh,
    subspace_generalized_eigh
)
from .generalized_eigensolver_stable import stable_generalized_eigh
from .generalized_eigensolver_pyscfad import stable_eigh_gen_pyscfad

__all__ = [
    "safe_generalized_eigh",
    "subspace_generalized_eigh", 
    "stable_generalized_eigh",
    "stable_eigh_gen_pyscfad"
]