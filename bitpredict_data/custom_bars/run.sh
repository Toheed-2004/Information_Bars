#!/bin/bash

# 1. Configuration
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
ARG=$1  # This captures 'init', 'update', or 'stats'

# Verify an argument was passed
if [ -z "$ARG" ]; then
    echo "Usage: ./init.sh [init|update|stats]"
    exit 1
fi

# Validate argument
if [[ "$ARG" != "init" && "$ARG" != "update" && "$ARG" != "stats" ]]; then
    echo "Error: Invalid argument '$ARG'"
    echo "Usage: ./bars_.sh [init|update|stats]"
    exit 1
fi

# 2. UNIQUE LOCKING - Different lock file PER COMMAND
# This creates /tmp/bars_init.lock, /tmp/bars_update.lock, etc.
LOCK_FILE="/tmp/bars_${ARG}.lock"

exec 8>"$LOCK_FILE"
if ! flock -n 8; then
    echo "Another '$ARG' process is already running. Exiting."
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