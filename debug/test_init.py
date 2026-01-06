#!/usr/bin/env python3
"""Test that JAX can see GPU devices."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'python'))

import jax
import jax.numpy as jnp

print("=" * 60)
print("JAX Device Test")
print("=" * 60)

# Get available devices
devices = jax.devices()
print(f"\nAvailable devices: {devices}")
print(f"Default device: {devices[0]}")
print(f"Device platform: {devices[0].platform}")

# Check if GPU is available
has_gpu = any(d.platform == 'gpu' for d in devices)
print(f"\nGPU available: {has_gpu}")

if has_gpu:
    print("\n✓ GPU detected successfully!")
    # Test a simple computation on GPU
    x = jnp.array([1.0, 2.0, 3.0])
    y = x * 2
    print(f"Test computation: {x} * 2 = {y}")
    print(f"Computation device: {y.device()}")
else:
    print("\n✗ No GPU detected. Running on CPU.")
    print("\nMake sure you:")
    print("  1. Load CUDA module: module load nvidia/cuda-sdk/12.8.1")
    print("  2. Set LD_LIBRARY_PATH: export LD_LIBRARY_PATH=/softs/nvidia/sdk/12.8.1/lib64:$LD_LIBRARY_PATH")
    print("  3. Set XLA_FLAGS: export XLA_FLAGS='--xla_gpu_cuda_data_dir=/softs/nvidia/sdk/12.8.1'")
    print("\nOr simply run: ./run_gpu.sh")

print("=" * 60)
