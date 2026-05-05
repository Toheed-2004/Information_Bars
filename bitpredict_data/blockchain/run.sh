#!/bin/bash

# 1. Configuration
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"

# 2. UNIQUE LOCKING
# This creates /tmp/data_blockchain.lock
LOCK_FILE="/tmp/data_blockchain.lock"

exec 8>"$LOCK_FILE"
if ! flock -n 8; then
    # If the script is already running, exit silently
    exit 0
fi

# 3. Environment Setup
export PYTHONPATH="/home/neurog/Documents/work"

if [ -f "/home/neurog/miniconda3/etc/profile.d/conda.sh" ]; then
    source "/home/neurog/miniconda3/etc/profile.d/conda.sh"
    conda activate bitpredict
fi


# 4. Execution
cd "$PROJECT_ROOT"
python main.py

# Note: The lock is automatically released when the script exits