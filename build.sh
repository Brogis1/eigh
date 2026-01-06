#!/bin/bash
# Build script for eigh_standalone

set -e

echo "==================================="
echo "Building eigh_standalone"
echo "==================================="

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

echo "Python: $($PYTHON --version)"

# Check CMake
if ! command -v cmake &> /dev/null; then
    echo "Error: CMake not found. Please install CMake 3.18+"
    exit 1
fi

echo "CMake: $(cmake --version | head -n1)"

# Check CUDA (optional)
if command -v nvcc &> /dev/null; then
    echo "CUDA: $(nvcc --version | grep release)"
    CUDA_AVAILABLE=1
else
    echo "CUDA: Not found (GPU support disabled)"
    CUDA_AVAILABLE=0
fi

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."
$PYTHON -m pip install -r requirements.txt

# Create build directory
echo ""
echo "Creating build directory..."
rm -rf build
mkdir build
cd build

# Configure
echo ""
echo "Configuring CMake..."
if [ $CUDA_AVAILABLE -eq 1 ]; then
    cmake -DPython_EXECUTABLE=$PYTHON ..
else
    cmake -DPython_EXECUTABLE=$PYTHON -DCMAKE_DISABLE_FIND_PACKAGE_CUDA=TRUE ..
fi

# Build
echo ""
echo "Building..."
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Check output
echo ""
echo "==================================="
echo "Build complete!"
echo "==================================="
echo ""
echo "Built modules:"
ls -lh ../src/python/*.so 2>/dev/null || ls -lh ../src/python/*.dylib 2>/dev/null || echo "No modules found"

echo ""
echo "To test, run:"
echo "  python tests/test_eigh.py"
