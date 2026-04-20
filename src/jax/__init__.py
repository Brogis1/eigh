"""
JAX-based eigenvalue solvers with CPU and GPU backend support.
"""

from .generalized_eigensolver import (
    safe_generalized_eigh,
    subspace_generalized_eigh,
)
from .generalized_eigensolver_stable import stable_generalized_eigh

__all__ = [
    "safe_generalized_eigh",
    "subspace_generalized_eigh",
    "stable_generalized_eigh",
    "stable_eigh_gen_pyscfad",
]


def __getattr__(name):
    if name in ("stable_eigh_pyscfad", "stable_eigh_gen_pyscfad"):
        from .generalized_eigensolver_pyscfad import (
            stable_eigh_pyscfad,
            stable_eigh_gen_pyscfad,
        )
        return {
            "stable_eigh_pyscfad": stable_eigh_pyscfad,
            "stable_eigh_gen_pyscfad": stable_eigh_gen_pyscfad,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")