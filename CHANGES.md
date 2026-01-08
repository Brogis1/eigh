# Changes Made

Renaming to prevent conflicts with JAX's standard `eigh`.

| File | Old Name | New Name |
| :--- | :--- | :--- |
| `lapack.cc` | `lapack_ssygvd` | `eigh_lapack_ssygvd` |
| | `lapack_dsygvd` | `eigh_lapack_dsygvd` |
| | `lapack_chegvd` | `eigh_lapack_chegvd` |
| | `lapack_zhegvd` | `eigh_lapack_zhegvd` |
| | `lapack_*_ffi` variants | `eigh_lapack_*_ffi` |
| `solver.cc` | `cusolver_sygvd_ffi` | `eigh_cusolver_sygvd_ffi` |
| `_core.py` | Updated `prepare_lapack_call` and CUDA target name | - |

The FFI target names are just string identifiers used to look up the registered function pointers.
The flow is:

1. C++ registers function pointer with name `"eigh_lapack_ssygvd_ffi"`
2. Python calls `ffi.register_ffi_target("eigh_lapack_ssygvd_ffi", ...)`
3. During JIT lowering, Python requests target `"eigh_lapack_ssygvd_ffi"`
4. XLA looks up the function pointer by that name

As long as both sides agree on the name (which they now do), everything works. The actual kernel code is unchanged.

# To Apply

You need to rebuild the package since the C++ files changed:

```bash
pip install -e .
```

After rebuilding, both JAX's standard `eigh` and this package's `eigh` can coexist without conflicts.