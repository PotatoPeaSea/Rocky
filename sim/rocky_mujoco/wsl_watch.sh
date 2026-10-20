#!/usr/bin/env bash
# Live training window, run inside WSL (drawn on the Windows desktop by WSLg). From Windows:
#   wsl -d Ubuntu -- bash /mnt/c/Users/smart/Documents/Pentapod/rocky/sim/rocky_mujoco/wsl_watch.sh runs/mjx_300m
#   ... wsl_watch.sh runs/mjx_300m --record runs/mjx_300m/dashboard/now.mp4
# Runs in WSL rather than Windows Python so it uses the VM's spare RAM when Windows is short on memory.
cd "$(dirname "$0")"
export PATH="$HOME/rocky/bin:$PATH"
exec python watch_training.py "$@"
