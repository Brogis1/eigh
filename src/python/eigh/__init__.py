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
Differentiable eigenvalue decomposition with JAX (CPU/GPU).

This package provides:
- eigh: Scipy-compatible interface for eigenvalue decomposition
- eigh_gen: Lower-level generalized eigenvalue problem solver

Features:
- Automatic differentiation via JAX
- CPU backend using LAPACK
- GPU backend using cuSOLVER
- Batching support via vmap
"""

from eigh._core import eigh, eigh_gen

__version__ = "0.1.0"
__all__ = ["eigh", "eigh_gen"]
