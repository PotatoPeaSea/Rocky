#!/usr/bin/env bash
# One-time GPU setup inside WSL2 Ubuntu (no sudo needed). From Windows:
#   wsl -d Ubuntu -- bash /mnt/c/Users/smart/Documents/Pentapod/rocky/sim/rocky_mujoco/wsl_setup.sh
# Needs only the normal Windows NVIDIA driver (no CUDA toolkit). Creates the venv ~/rocky.
#
# Why this way:
#  - Ubuntu 26.04 ships Python 3.14; the pinned JAX / Brax stack is tested on 3.12, so uv provides 3.12.
#  - pip, not `uv pip`, installs the packages: uv rejects some NVIDIA CUDA wheels ("Invalid Wheel-Version")
#    and an interrupted attempt left a truncated libnccl.so.2 that crashed JAX with SIGBUS.
#  - mediapy needs an ffmpeg on PATH for the rollout video: link the one imageio-ffmpeg ships.
set -euo pipefail
command -v ~/.local/bin/uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -x ~/rocky/bin/python ] || ~/.local/bin/uv venv --python 3.12 ~/rocky
~/.local/bin/uv pip install -q --python ~/rocky/bin/python pip
rm -rf ~/rocky/lib/python3.12/site-packages/~*            # leftovers of interrupted installs
~/rocky/bin/python -m pip install -q "jax[cuda12]==0.9.2" "mujoco==3.16.0" "mujoco-mjx==3.16.0" \
    "brax==0.14.2" "playground==0.2.0" "mujoco-warp==3.16.0" mediapy imageio-ffmpeg
ln -sf "$(~/rocky/bin/python -c 'import imageio_ffmpeg as f; print(f.get_ffmpeg_exe())')" ~/rocky/bin/ffmpeg
XLA_PYTHON_CLIENT_PREALLOCATE=false ~/rocky/bin/python -c "import jax; print('JAX', jax.__version__, jax.devices())" 2>/dev/null
