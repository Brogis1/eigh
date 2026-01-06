#!/bin/bash
# Run eigh example on GPU
# Usage: ./run_gpu.sh [script.py]

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Source the GPU environment setup
source "${SCRIPT_DIR}/setup_gpu_env.sh"

# Run the script
SCRIPT=${1:-example.py}
"${SCRIPT_DIR}/.venv/bin/python" "$SCRIPT"