#!/bin/bash
# Simplified GPU environment setup for system CUDA
# Usage: source setup_gpu_env_clean.sh

# CUDA 12.8.1 paths
CUDA_ROOT=/softs/nvidia/sdk/12.8.1

# Add system CUDA library paths
export LD_LIBRARY_PATH="${CUDA_ROOT}/lib64:${CUDA_ROOT}/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="${CUDA_ROOT}/nvvm/lib64:$LD_LIBRARY_PATH"

# Set CUDA paths for tools
export CUDA_HOME="${CUDA_ROOT}"
export CUDA_PATH="${CUDA_ROOT}"

# XLA configuration
export XLA_FLAGS="--xla_gpu_cuda_data_dir=${CUDA_ROOT}"

# Add CUDA binaries to PATH
export PATH="${CUDA_ROOT}/bin:$PATH"

echo "GPU environment configured (clean - using system CUDA):"
echo "  CUDA_ROOT: ${CUDA_ROOT}"
echo "  XLA_FLAGS: ${XLA_FLAGS}"
