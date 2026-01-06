# Build System Fixes

## Issue: XLA Headers Not Found

### Problem
```
fatal error: 'xla/ffi/api/c_api.h' file not found
```

### Root Cause
The CMakeLists.txt was looking for XLA headers in the wrong location:
- **Incorrect**: `jax/_src/lib/xla/ffi/api/c_api.h`
- **Correct**: `jaxlib/include/xla/ffi/api/c_api.h`

XLA headers are provided by **jaxlib**, not jax.

### Solution

Updated `CMakeLists.txt` to correctly locate XLA headers:

```cmake
# XLA headers (from jaxlib)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import jaxlib; import os; print(os.path.join(jaxlib.__path__[0], 'include'))"
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE XLA_INCLUDE_DIR
    RESULT_VARIABLE XLA_RESULT
)

if(NOT XLA_RESULT EQUAL 0 OR NOT EXISTS "${XLA_INCLUDE_DIR}")
    message(FATAL_ERROR "Could not find XLA headers. Make sure jaxlib is installed: pip install jaxlib")
endif()

message(STATUS "XLA headers found at: ${XLA_INCLUDE_DIR}")
```

This will correctly find headers at:
```
/path/to/.venv/lib/python3.11/site-packages/jaxlib/include/xla/ffi/api/c_api.h
```

## Issue: Python Virtual Environment

### Problem
Build system wasn't detecting the `.venv` virtual environment in the parent directory.

### Solution

1. **Updated `build.sh`** to auto-detect virtual environment:

```bash
# Detect Python (prefer .venv if it exists)
if [ -f "../.venv/bin/python" ]; then
    PYTHON="../.venv/bin/python"
    echo "Using virtual environment Python"
elif [ -f ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
    echo "Using virtual environment Python"
elif command -v python &> /dev/null; then
    PYTHON="python"
else
    echo "Error: Python not found"
    exit 1
fi
```

2. **Pass Python to CMake**:

```bash
cmake -DPython_EXECUTABLE=$PYTHON ..
```

This ensures CMake uses the correct Python interpreter with jaxlib installed.

## Verification

After fixes, you can verify:

```bash
# Check XLA headers exist
../.venv/bin/python -c "import jaxlib; import os; print(os.path.join(jaxlib.__path__[0], 'include'))"

# Expected output:
# /path/to/.venv/lib/python3.11/site-packages/jaxlib/include

# Verify c_api.h exists
ls ../.venv/lib/python3.11/site-packages/jaxlib/include/xla/ffi/api/c_api.h

# Expected: file exists
```

## Building Now Works

```bash
cd eigh_standalone
./build.sh
```

The script will:
1. Detect `.venv` Python automatically
2. Find XLA headers from jaxlib
3. Configure and build successfully
4. Output modules to `src/python/`

## Manual Build

If you prefer manual control:

```bash
cd eigh_standalone
mkdir build && cd build

# Use your venv Python explicitly
cmake -DPython_EXECUTABLE=../.venv/bin/python ..
make
```

## Requirements

Make sure these are installed in your virtual environment:

```bash
../.venv/bin/python -m pip install jax jaxlib nanobind numpy
```

The key package is **jaxlib** which provides the XLA headers.
