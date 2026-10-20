#!/usr/bin/env bash
# Records the training dashboard (newest clip + reward curve) to MP4 when a train_mjx.py run passes
# each threshold, and once more when it finishes (params.pkl written). Runs inside WSL. From Windows:
#   wsl -d Ubuntu -- bash /mnt/c/Users/smart/Documents/Pentapod/rocky/sim/rocky_mujoco/wsl_record_dashboard.sh runs/mjx_300m 100 200
# Thresholds are in millions of env steps. Output: <run>/dashboard/dashboard_<step>M.mp4
cd "$(dirname "$0")"
run="$1"; shift
export PATH="$HOME/rocky/bin:$PATH"
mkdir -p "$run/dashboard"
last_step() { tail -n 1 "$run/progress.csv" 2>/dev/null | cut -d, -f1; }
record() {
    sleep 60                                   # the clip for this checkpoint is written ~30 s after the eval
    python watch_training.py "$run" --record "$run/dashboard/dashboard_$1.mp4" && echo "$(date +%T) recorded $1"
}
for m in "$@"; do
    until s=$(last_step); [ -n "$s" ] && [ "$s" != step ] && [ "${s%.*}" -ge $((m * 1000000)) ]; do sleep 20; done
    record "${m}M"
done
until [ -f "$run/params.pkl" ]; do sleep 20; done
sleep 30                                       # final clip comes after params.pkl
record final
