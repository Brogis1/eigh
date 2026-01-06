#!/bin/bash
# Source this script to set up GPU environment
# Usage: source setup_gpu_env.sh

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# CUDA 12.8.1 paths
CUDA_ROOT=/softs/nvidia/sdk/12.8.1

# Add wheel-bundled NVIDIA library paths (for JAX CUDA plugin)
VENV_NVIDIA="${SCRIPT_DIR}/.venv/lib/python3.10/site-packages/nvidia"
if [ -d "${VENV_NVIDIA}" ]; then
    # Add all nvidia package lib directories to LD_LIBRARY_PATH
    for pkg in cusparse cublas cufft curand cusolver cudnn nvjitlink cuda_runtime cuda_nvrtc; do
        if [ -d "${VENV_NVIDIA}/${pkg}/lib" ]; then
            export LD_LIBRARY_PATH="${VENV_NVIDIA}/${pkg}/lib:$LD_LIBRARY_PATH"
        fi
    done
fi

# Add all CUDA library paths (system CUDA)
export LD_LIBRARY_PATH="${CUDA_ROOT}/lib64:${CUDA_ROOT}/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="${CUDA_ROOT}/nvvm/lib64:$LD_LIBRARY_PATH"

# Set CUDA paths for various tools
export CUDA_HOME="${CUDA_ROOT}"
export CUDA_PATH="${CUDA_ROOT}"

# XLA configuration
# Use wheel-bundled CUDA data if available, otherwise system CUDA
if [ -d "${VENV_NVIDIA}/cuda_nvcc/nvvm" ]; then
    export XLA_FLAGS="--xla_gpu_cuda_data_dir=${VENV_NVIDIA}/cuda_nvcc"
else
    export XLA_FLAGS="--xla_gpu_cuda_data_dir=${CUDA_ROOT}"
fi

# Add CUDA binaries to PATH
export PATH="${CUDA_ROOT}/bin:$PATH"

# Preload NVIDIA libraries to resolve dependencies for JAX CUDA plugin
# This is needed because wheel-bundled libraries have interdependencies
if [ -d "${VENV_NVIDIA}" ]; then
    PRELOAD_LIBS=""
    for lib in libnvJitLink.so.12 libcublas.so.12 libcublasLt.so.12; do
        for pkg_dir in nvjitlink cublas; do
            if [ -f "${VENV_NVIDIA}/${pkg_dir}/lib/${lib}" ]; then
                PRELOAD_LIBS="${VENV_NVIDIA}/${pkg_dir}/lib/${lib}:${PRELOAD_LIBS}"
                break
            fi
        done
    done
    if [ -n "${PRELOAD_LIBS}" ]; then
        export LD_PRELOAD="${PRELOAD_LIBS}${LD_PRELOAD}"
    fi
fi

echo "GPU environment configured:"
echo "  CUDA_ROOT: ${CUDA_ROOT}"
echo "  Wheel NVIDIA libs: ${VENV_NVIDIA}"
echo "  LD_LIBRARY_PATH includes wheel libraries and ${CUDA_ROOT}/lib64"
if [ -n "${LD_PRELOAD}" ]; then
    echo "  LD_PRELOAD: ${LD_PRELOAD:0:100}..."
fi
echo "  XLA_FLAGS: ${XLA_FLAGS}"
