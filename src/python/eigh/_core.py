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
Standalone differentiable eigh (eigenvalue decomposition) implementation.
Extracted from pyscfad: https://github.com/fishjojo/pyscfad
"""

from functools import partial
import warnings
import numpy as np

try:
    from jax import ffi
except ImportError:
    from jax.extend import ffi

import jax
from jax import lax
from jax import numpy as jnp
from jax._src import ad_util
from jax._src import api
from jax._src import dispatch
from jax._src import dtypes
from jax._src.core import Primitive, ShapedArray
from jax._src.interpreters import ad
from jax._src.interpreters import batching
from jax._src.interpreters import mlir
from jax._src.lax import lax as lax_internal
from jax._src.lax.linalg import (
    _H,
    symmetrize,
    _nan_like_hlo,
    _broadcasting_select_hlo,
)
from jax._src.numpy import ufuncs
from jax._src.numpy.util import promote_dtypes_inexact

# Import the compiled C++ extensions
# Try relative import first (installed package), then absolute (development)
try:
    from eigh.eigh_lapack import (initialize as lapack_initialize,
                                  registrations as lapack_registrations)
    _lapack_available = True
except ImportError:
    try:
        from eigh_lapack import (initialize as lapack_initialize,
                                 registrations as lapack_registrations)
        _lapack_available = True
    except ImportError:
        _lapack_available = False
        warnings.warn("LAPACK backend not available. CPU execution will fail.")

try:
    from eigh.eigh_cuda import registrations as cuda_registrations
    _cuda_available = True
except ImportError:
    try:
        from eigh_cuda import registrations as cuda_registrations
        _cuda_available = True
    except ImportError:
        _cuda_available = False
        warnings.warn("CUDA backend not available. GPU execution will fail.")


# Register FFI targets
if _lapack_available:
    for _name, _value in lapack_registrations().items():
        ffi.register_ffi_target(
            _name,
            _value,
            platform="cpu",
            api_version=(1 if _name.endswith("_ffi") else 0),
        )

if _cuda_available:
    for _name, _value in cuda_registrations().items():
        ffi.register_ffi_target(
            _name,
            _value,
            platform="gpu",
            api_version=1,
        )


LAPACK_DTYPE_PREFIX = {
    np.float32: "s",
    np.float64: "d",
    np.complex64: "c",
    np.complex128: "z",
}


def prepare_lapack_call(fn_base, dtype):
    """Prepare LAPACK function name based on dtype."""
    if _lapack_available:
        lapack_initialize()
    try:
        prefix = (LAPACK_DTYPE_PREFIX.get(dtype, None) or
                  LAPACK_DTYPE_PREFIX[dtype.type])
        return f"eigh_lapack_{prefix}{fn_base}"
    except KeyError:
        raise NotImplementedError(f"Unsupported dtype {dtype}")


def eigh_gen(a, b, *, lower=True, itype=1, deg_thresh=1e-9):
    """
    Solve generalized eigenvalue problem with differentiation support.

    Solves: a @ v = b @ v @ diag(w)

    Parameters
    ----------
    a : array_like
        Hermitian/symmetric matrix (shape [..., n, n])
    b : array_like
        Hermitian/symmetric positive definite matrix (shape [..., n, n])
    lower : bool, optional
        Whether to use lower triangle (default: True)
    itype : int, optional
        Type of generalized eigenvalue problem (default: 1)
        - 1: a @ v = b @ v @ diag(w)
        - 2: a @ b @ v = v @ diag(w)
        - 3: b @ a @ v = v @ diag(w)
    deg_thresh : float, optional
        Threshold for degenerate eigenvalues in differentiation (default: 1e-9)

    Returns
    -------
    w : ndarray
        Eigenvalues (shape [..., n])
    v : ndarray
        Eigenvectors (shape [..., n, n])
    """
    a = symmetrize(a)
    b = symmetrize(b)
    w, v = eigh_gen_p.bind(a, b, lower=lower, itype=itype, deg_thresh=deg_thresh)
    return w, v


def eigh(a, b=None, *, lower=True, eigvals_only=False, type=1, deg_thresh=1e-9):
    """
    Solve eigenvalue problem with scipy-compatible interface.

    Parameters
    ----------
    a : array_like
        Hermitian/symmetric matrix (shape [..., n, n])
    b : array_like, optional
        Hermitian/symmetric positive definite matrix for generalized problem.
        If None, solves standard eigenvalue problem.
    lower : bool, optional
        Whether to use lower triangle (default: True)
    eigvals_only : bool, optional
        Whether to return only eigenvalues (default: False)
    type : int, optional
        Type of generalized eigenvalue problem (default: 1)
    deg_thresh : float, optional
        Threshold for degenerate eigenvalues in differentiation (default: 1e-9)

    Returns
    -------
    w : ndarray
        Eigenvalues
    v : ndarray, optional
        Eigenvectors (only if eigvals_only=False)
    """
    if b is None:
        b = jnp.zeros_like(a) + jnp.eye(a.shape[-1])

    a, b = promote_dtypes_inexact(jnp.asarray(a), jnp.asarray(b))
    w, v = eigh_gen(a, b, lower=lower, itype=type, deg_thresh=deg_thresh)

    if eigvals_only:
        return w
    else:
        return w, v


# JAX primitive implementation
def _eigh_gen_impl(a, b, *, lower, itype, deg_thresh):
    w, v = dispatch.apply_primitive(
        eigh_gen_p,
        a, b,
        lower=lower,
        itype=itype,
        deg_thresh=deg_thresh
    )
    return w, v


def _eigh_gen_abstract_eval(a, b, *, lower, itype, deg_thresh):
    if isinstance(a, ShapedArray):
        if a.ndim < 2 or a.shape[-2] != a.shape[-1]:
            raise ValueError(
                "Argument 'a' to eigh must have shape [..., n, n], "
                f"but got shape {a.shape}"
            )

        batch_dims = a.shape[:-2]
        n = a.shape[-1]
        v = a.update(shape=batch_dims + (n, n))
        w = a.update(
            shape=batch_dims + (n,),
            dtype=lax_internal._complex_basetype(a.dtype)
        )
    else:
        w, v = a, a
    return w, v


def _eigh_gen_jvp_rule(primals, tangents, *, lower, itype, deg_thresh):
    """Forward-mode automatic differentiation rule."""
    if itype != 1:
        raise NotImplementedError(f"JVP for itype={itype} is not implemented.")

    a, b = primals
    n = a.shape[-1]
    at, bt = tangents

    w_real, v = eigh_gen_p.bind(
        symmetrize(a),
        symmetrize(b),
        lower=lower,
        itype=itype,
        deg_thresh=deg_thresh
    )

    w = w_real.astype(a.dtype)
    eji = w[..., np.newaxis, :] - w[..., np.newaxis]
    Fmat = ufuncs.reciprocal(
        jnp.where(ufuncs.absolute(eji) > deg_thresh, eji, np.inf)
    )

    dot = partial(
        lax.dot if a.ndim == 2 else lax.batch_matmul,
        precision=lax.Precision.HIGHEST
    )

    if type(at) is ad_util.Zero:
        vt_at_v = lax.zeros_like_array(a)
    else:
        vt_at_v = dot(_H(v), dot(at, v))

    if type(bt) is not ad_util.Zero:
        if a.ndim == 2:
            w_diag = jnp.diag(w)
        else:
            batch_dims = a.shape[:-2]
            fn = jnp.diag
            for _ in batch_dims:
                fn = api.vmap(fn)
            w_diag = fn(w)
        vt_bt_v = dot(_H(v), dot(bt, v))
        vt_bt_v_w = dot(vt_bt_v, w_diag)
        vt_at_v -= vt_bt_v_w

    dw = ufuncs.real(jnp.diagonal(vt_at_v, axis1=-2, axis2=-1))

    F_vt_at_v = ufuncs.multiply(Fmat, vt_at_v)
    if type(bt) is not ad_util.Zero:
        bmask = jnp.where(ufuncs.absolute(eji) > deg_thresh, jnp.zeros_like(a), 1)
        F_vt_at_v -= ufuncs.multiply(bmask, vt_bt_v) * 0.5

    dv = dot(v, F_vt_at_v)
    return (w_real, v), (dw, dv)


def _eigh_gen_cpu_gpu_lowering(
    ctx, a, b, *, lower, itype, deg_thresh, target_name_prefix
):
    """Platform-specific lowering to LAPACK or cuSOLVER."""
    del deg_thresh
    a_aval, b_aval = ctx.avals_in
    w_aval, v_aval = ctx.avals_out
    batch_dims = a_aval.shape[:-2]

    if target_name_prefix == "cpu":
        dtype = a_aval.dtype
        prefix = "he" if dtypes.issubdtype(dtype, np.complexfloating) else "sy"
        target_name = prepare_lapack_call(f"{prefix}gvd_ffi", dtype)

        kwargs = {
            "itype": np.int32(itype),
            "jobz": np.uint8(ord("V")),
            "uplo": np.uint8(ord("L" if lower else "U")),
        }
    elif target_name_prefix == "cuda":
        target_name = "eigh_cusolver_sygvd_ffi"
        kwargs = {
            "itype": np.int32(itype),
            "lower": lower,
        }
    else:
        raise NotImplementedError(f"Platform {target_name_prefix} is not supported.")

    info_aval = ShapedArray(batch_dims, np.int32)

    layout = _column_major_matrix_layout(len(a_aval.shape))

    nb = len(batch_dims)
    operand_layouts = [
        layout,  # a
        layout,  # b
    ]
    result_layouts = [
        layout,  # v
        layout,  # b_out
        tuple(range(nb, -1, -1)),  # w
        tuple(range(nb - 1, -1, -1)),  # info
    ]

    rule = ffi.ffi_lowering(
        target_name,
        operand_layouts=operand_layouts,
        result_layouts=result_layouts,
        operand_output_aliases={0: 0, 1: 1}
    )

    sub_ctx = ctx.replace(
        avals_in=ctx.avals_in,
        avals_out=[v_aval, v_aval, w_aval, info_aval]
    )

    v, _, w, info = rule(sub_ctx, a, b, **kwargs)

    zeros = mlir.full_like_aval(ctx, 0, info_aval)
    ok = mlir.compare_hlo(info, zeros, "EQ", "SIGNED")
    w = _replace_not_ok_with_nan(ctx, batch_dims, ok, w, w_aval)
    v = _replace_not_ok_with_nan(ctx, batch_dims, ok, v, v_aval)
    return [w, v]


def _eigh_gen_batching_rule(batched_args, batch_dims, *, lower, itype, deg_thresh):
    """Batching rule for vmap support."""
    a, b = batched_args
    bd_a, bd_b = batch_dims
    size = next(
        t.shape[i] for t, i in zip(batched_args, batch_dims)
        if i is not None
    )
    a = batching.bdim_at_front(a, bd_a, size)
    b = batching.bdim_at_front(b, bd_b, size)
    return eigh_gen_p.bind(
        a, b,
        lower=lower,
        itype=itype,
        deg_thresh=deg_thresh
    ), (0, 0)


def _replace_not_ok_with_nan(ctx, batch_dims, ok, x, x_aval):
    """Replace failed computations with NaN."""
    num_bcast_dims = len(x_aval.shape) - len(batch_dims)
    select_aval = ShapedArray(batch_dims + (1,) * num_bcast_dims, np.bool_)
    return _broadcasting_select_hlo(
        ctx,
        mlir.broadcast_in_dim(
            ctx, ok, select_aval,
            broadcast_dimensions=range(len(batch_dims))
        ),
        select_aval,
        x, x_aval, _nan_like_hlo(ctx, x_aval), x_aval
    )


def _column_major_matrix_layout(dim):
    """Generate column-major layout for BLAS/LAPACK compatibility."""
    return (dim - 2, dim - 1) + tuple(range(dim - 3, -1, -1))


# Register JAX primitive
eigh_gen_p = Primitive('eigh_gen')
eigh_gen_p.multiple_results = True
eigh_gen_p.def_impl(_eigh_gen_impl)
eigh_gen_p.def_abstract_eval(_eigh_gen_abstract_eval)
ad.primitive_jvps[eigh_gen_p] = _eigh_gen_jvp_rule
batching.primitive_batchers[eigh_gen_p] = _eigh_gen_batching_rule

if _lapack_available:
    mlir.register_lowering(
        eigh_gen_p,
        partial(_eigh_gen_cpu_gpu_lowering, target_name_prefix="cpu"),
        platform="cpu"
    )

if _cuda_available:
    mlir.register_lowering(
        eigh_gen_p,
        partial(_eigh_gen_cpu_gpu_lowering, target_name_prefix="cuda"),
        platform="gpu"
    )


__all__ = ['eigh', 'eigh_gen']
