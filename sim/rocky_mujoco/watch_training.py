"""Live window for a train_mjx.py run: eval-reward curve + the newest progress video, looping.

    python watch_training.py                  # runs/mjx_big
    python watch_training.py runs/mjx_smoke   # any run folder
    python watch_training.py --record media/training_now.mp4   # save one loop of the window as an MP4

Polls the run folder every 2 s (progress.csv and videos/ are rewritten at every eval), so it can be
opened, closed and reopened while training runs. Press left / right to step through older clips,
End to go back to following the newest.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")    # OpenBLAS reserves big per-thread buffers; tiny job here
import argparse
import csv
import glob
import json
import os
import time

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

HERE = os.path.dirname(os.path.abspath(__file__))


def read_progress(run):
    try:
        with open(os.path.join(run, "progress.csv"), newline="") as f:
            rows = list(csv.DictReader(f))
        return [{k: float(v) for k, v in r.items()} for r in rows]
    except (OSError, ValueError):
        return []                       # file is being rewritten: try again next poll


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", default=os.path.join(HERE, "runs", "mjx_big"))
    ap.add_argument("--record", help="write one loop of the newest clip, with the curve, to this MP4 and exit")
    a = ap.parse_args()
    if a.record:
        import matplotlib
        matplotlib.use("Agg")
    run = os.path.abspath(a.run)
    total = None
    try:
        total = json.load(open(os.path.join(run, "config.json")))["args"]["num_timesteps"]
    except (OSError, KeyError, ValueError):
        pass

    fig = plt.figure(figsize=(11, 8.2))
    fig.canvas.manager.set_window_title(f"Rocky training - {os.path.basename(run)}")
    ax_v = fig.add_axes([0.03, 0.36, 0.94, 0.62])
    ax_r = fig.add_axes([0.08, 0.06, 0.88, 0.24])
    ax_v.axis("off")
    img = ax_v.imshow(np.zeros((360, 640, 3), np.uint8))
    line, = ax_r.plot([], [], "o-", color="#1e8449")
    ax_r.set_xlabel("env steps (M)")
    ax_r.set_ylabel("eval reward")
    ax_r.grid(alpha=0.3)
    status = fig.text(0.08, 0.31, "", fontsize=10)

    st = {"videos": [], "idx": None, "frames": None, "k": 0, "poll": 0.0, "follow": True}

    def load(i):
        try:
            p = st["videos"][i]
            st["frames"] = iio.imread(p)
            st["idx"], st["k"] = i, 0
            st["loaded"] = (p, os.path.getmtime(p))
        except Exception:               # still being written
            pass

    def poll():
        vids = sorted(glob.glob(os.path.join(run, "videos", "step_*.mp4")))
        st["videos"] = vids
        if st["idx"] is not None and st["idx"] >= len(vids):     # run folder moved or cleaned
            st["idx"], st["frames"], st["follow"] = None, None, True
        # follow the newest clip; retried every poll until it loads (a clip < 3 s old may still be written)
        if st["follow"] and vids and st.get("loaded") != (vids[-1], os.path.getmtime(vids[-1]))                 and time.time() - os.path.getmtime(vids[-1]) > 3:
            load(len(vids) - 1)
        rows = read_progress(run)
        if rows:
            x = [r["step"] / 1e6 for r in rows]
            y = [r["eval_reward"] for r in rows]
            line.set_data(x, y)
            ax_r.set_xlim(0, max((total or 0) / 1e6, x[-1] * 1.05, 1))
            ax_r.set_ylim(min(0, min(y)), max(y) * 1.15 + 1)
            last = rows[-1]
            msg = f"step {last['step'] / 1e6:.1f} M" + (f" / {total / 1e6:.0f} M" if total else "")
            msg += f"   eval reward {last['eval_reward']:.1f}   elapsed {last['time_s'] / 60:.0f} min"
            if total and len(rows) > 2 and last["step"] > 0:
                rate = (last["step"] - rows[1]["step"]) / max(last["time_s"] - rows[1]["time_s"], 1)
                msg += f"   ETA {(total - last['step']) / rate / 60:.0f} min"
            status.set_text(msg)
        if st["idx"] is not None:
            ax_v.set_title(f"clip {st['idx'] + 1}/{len(st['videos'])}: {os.path.basename(st['videos'][st['idx']])}"
                           + ("" if st["follow"] else "   (browsing: End = follow newest)"), fontsize=9)

    def on_key(e):
        if not st["videos"]:
            return
        i = st["idx"] or 0
        if e.key == "left":
            st["follow"] = False
            load(max(0, i - 1))
        elif e.key == "right":
            st["follow"] = False
            load(min(len(st["videos"]) - 1, i + 1))
        elif e.key == "end":
            st["follow"] = True
            load(len(st["videos"]) - 1)

    def tick(_):
        if time.time() - st["poll"] > 2.0:
            st["poll"] = time.time()
            poll()
        if st["frames"] is not None:
            img.set_data(st["frames"][st["k"] % len(st["frames"])])
            st["k"] += 1
        return img, line, status

    if a.record:
        poll()
        if st["frames"] is None:
            raise SystemExit("no video in the run folder yet")
        import imageio
        os.makedirs(os.path.dirname(os.path.abspath(a.record)), exist_ok=True)
        n = len(st["frames"])
        with imageio.get_writer(a.record, fps=25, quality=7, macro_block_size=8) as w:   # streamed: low RAM
            for _ in range(n):
                tick(None)
                fig.canvas.draw()
                w.append_data(np.asarray(fig.canvas.buffer_rgba())[..., :3])
        print("wrote", a.record, n, "frames")
        return None

    fig.canvas.mpl_connect("key_press_event", on_key)
    anim = FuncAnimation(fig, tick, interval=40, cache_frame_data=False)    # 25 fps, like the clips
    plt.show()
    return anim


if __name__ == "__main__":
    main()
