#!/bin/bash
set -e

echo "[INFO] Starting JLab Standalone Agent"

# Default values from environment or use defaults
AGENT_TYPE="${JLAB_AGENT_TYPE:-KerasTD3-v0}"
ENV_TYPE="${JLAB_ENV_TYPE:-Pendulum-v1}"
LOGDIR="${JLAB_LOGDIR:-}"

echo "[INFO] Agent Type: $AGENT_TYPE"
echo "[INFO] Environment: $ENV_TYPE"

# Initialize a git repo in /tmp/jlab if it doesn't exist
if [ ! -d "/tmp/jlab/.git" ]; then
    echo "[INFO] Initializing git repository for jlab_opt_control..."
    cd /tmp/jlab
    git init
    git config user.email "docker@smocs.local"
    git config user.name "SMOCS Docker"
    git add -A
    git commit -m "Docker installation" || true
    echo "[INFO] Git repository initialized"
else
    echo "[INFO] Git repository already exists"
fi

# The run_continuous.py script is in the drivers subdirectory of jlab_opt_control
SCRIPT_PATH=$(python3 -c "import jlab_opt_control; import os; print(os.path.join(os.path.dirname(jlab_opt_control.__file__), 'drivers', 'run_continuous.py'))")

if [ ! -f "$SCRIPT_PATH" ]; then
    echo "[ERROR] Could not find run_continuous.py in SciOptControlToolkit installation"
    echo "[ERROR] Searched at: $SCRIPT_PATH"
    exit 1
fi

echo "[INFO] Found run_continuous.py at: $SCRIPT_PATH"

# Change to /tmp/jlab directory so git commands work
cd /tmp/jlab

# Build the command
CMD="python3 -O $SCRIPT_PATH --agent $AGENT_TYPE --env $ENV_TYPE"

# Add custom logdir if specified
if [ ! -z "$LOGDIR" ]; then
    echo "[INFO] Using custom log directory: $LOGDIR"
    CMD="$CMD --logdir $LOGDIR"
fi

echo "[INFO] Running command: $CMD"
echo "[INFO] Working directory: $(pwd)"
exec $CMD