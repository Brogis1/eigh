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
Performance comparison between JAX eigh and our implementation.
"""

import sys
import os
import time
import pytest

sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), '..', 'src', 'python')
)

import numpy as np
import jax
from jax import numpy as jnp
from jax import scipy as jsp

jax.config.update("jax_enable_x64", True)

try:
    from eigh import eigh
except ImportError as e:
    print(f"✗ Failed to import eigh module: {e}")
    sys.exit(1)


def test_jit_performance():
    """Compare performance of JAX eigh vs our implementation."""
    print("\nTest: JIT performance comparison")
    @jax.jit
    def jax_eigh_jit(a):
        return jsp.linalg.eigh(a)

    @jax.jit
    def our_eigh_jit(a):
        return eigh(a)

    sizes = [10, 50, 100, 200]
    n_iterations = 100

    for n in sizes:
        np.random.seed(42)
        a_np = np.random.randn(n, n).astype(np.float64)
        a_np = (a_np + a_np.T) / 2
        a = jnp.array(a_np)

        w_jax, _ = jax_eigh_jit(a)
        w_jax.block_until_ready()
        w_ours, _ = our_eigh_jit(a)
        w_ours.block_until_ready()

        start = time.time()
        for _ in range(n_iterations):
            w_jax, _ = jax_eigh_jit(a)
            w_jax.block_until_ready()
        time_jax = (time.time() - start) / n_iterations

        start = time.time()
        for _ in range(n_iterations):
            w_ours, _ = our_eigh_jit(a)
            w_ours.block_until_ready()
        time_ours = (time.time() - start) / n_iterations

        speedup = time_jax / time_ours if time_ours > 0 else float('inf')
        print(
            f"n={n:3d}: JAX={time_jax*1000:.3f} ms, "
            f"Ours={time_ours*1000:.3f} ms, "
            f"Speedup={speedup:.2f}x"
        )

    print("Performance comparison test passed")


if __name__ == "__main__":
    # Run all tests verbosely
    pytest.main([__file__, "-v"])
