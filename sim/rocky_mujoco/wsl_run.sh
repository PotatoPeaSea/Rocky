#!/usr/bin/env bash
# Run a sim script inside WSL2 with the GPU venv (~/rocky, see wsl_setup.sh). From Windows:
#   wsl -d Ubuntu -- bash /mnt/c/Users/smart/Documents/Pentapod/rocky/sim/rocky_mujoco/wsl_run.sh train_mjx.py --smoke
# Output (minus XLA's autotuner noise) is also written to runs/wsl_last.log.
cd "$(dirname "$0")"
export PATH="$HOME/rocky/bin:$PATH"
export XLA_PYTHON_CLIENT_PREALLOCATE=false   # Windows apps hold part of the 12 GB: allocate on demand
export TF_CPP_MIN_LOG_LEVEL=3                # WSL driver version string makes XLA log harmless errors
export MUJOCO_GL=egl
mkdir -p runs
python -u "$@" 2>&1 | grep --line-buffered -v -E '^E[0-9]{4} |^ *(parameter_|ROOT |\}$|Computation:)' | tee runs/wsl_last.log
exit "${PIPESTATUS[0]}"
