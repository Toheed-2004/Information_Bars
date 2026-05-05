#!/bin/bash

# 1. Configuration
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
ARG=$1  # This captures 'init', 'update', or 'resample'

# Verify an argument was passed
if [ -z "$ARG" ]; then
    echo "Usage: ./init.sh [init|update|resample]"
    exit 1
fi

# 2. UNIQUE LOCKING
# This creates /tmp/binance_init.lock, /tmp/binance_update.lock, etc.
LOCK_FILE="/tmp/binance_$ARG.lock"

exec 8>"$LOCK_FILE"
if ! flock -n 8; then
    # If 'update' is running, another 'update' will stop here.
    # But 'resample' can still proceed because its lock file is different.
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
python main.py "$ARG"