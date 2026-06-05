"""Smoke test for the compiled CUDA extension in the built wheel.

Run by cibuildwheel's test step (CIBW_TEST_COMMAND) for the eigh-cudaXXX wheels.
CI has no GPU, so we only assert that the compiled `eigh_cuda` extension DLOPENS
and exposes its FFI registration symbols — proof that the kernel compiled and its
CUDA runtime dependencies resolve.

We load the .so by file path rather than `import eigh`, because the eigh package
__init__ imports jax and registers a "gpu" platform FFI target, which can require
a live GPU backend that the CI runner does not have. Real device execution is
verified on actual GPU hardware before trusting a release.
"""
import importlib.util
import os
import sys


def find_eigh_cuda_so():
    # Locate the installed `eigh` package directory without importing it.
    for entry in sys.path:
        candidate_dir = os.path.join(entry, "eigh")
        if not os.path.isdir(candidate_dir):
            continue
        for name in os.listdir(candidate_dir):
            if name.startswith("eigh_cuda") and name.endswith(".so"):
                return os.path.join(candidate_dir, name)
    return None


def main():
    so_path = find_eigh_cuda_so()
    if so_path is None:
        print("FAIL: eigh_cuda*.so not found in any installed eigh package", file=sys.stderr)
        return 1

    spec = importlib.util.spec_from_file_location("eigh_cuda", so_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # this is the dlopen — fails if CUDA deps unresolved

    assert hasattr(module, "registrations"), "eigh_cuda has no registrations()"
    keys = list(module.registrations().keys())
    print(f"eigh_cuda dlopen OK ({so_path}): {keys}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
