# Rocky training log

Each step of policy improvement: what changed, how it was trained, how it scores, and videos.
All GPU runs are on the RTX 3060 under WSL2 (see the README's GPU section). The progress videos of each
run are in `runs/<name>/videos/` (not in git). The clips linked here are copied into `media/training/`.

- **Scoring** is done with `eval_mjx.py`: a deterministic policy, no domain randomisation, all rollouts in one GPU batch.
  - **Flat:** velocity-tracking error and fall rate for 7 fixed commands, 256 envs each, over 10 s.
  - **Terrain:** success rate per terrain type and level, 128 envs each, commanded 10 cm/s straight ahead for 15 s. Success means no fall and at least 70 % of the commanded distance (1.05 m).
- **Every clip** follows the same script, so clips from different checkpoints compare directly.
  - Flat clips: 3 s forward, 3 s sideways, 3 s turning.
  - Terrain clips: 5 cm rocks (level 9), then 4.3 cm stairs (level 6), then 3.3 cm blocks (level 5).

---

## Step 0: CPU sanity runs (2026-10-20)

`train_ppo.py` (SB3 PPO) on the Windows CPU, 8 envs, 300 k steps. These only check that the pipeline works; neither policy walks yet.

| Run | Time | Eval vs "just stand" |
|---|---|---|
| IMU observation (default) | 1.7 min | 23.70 vs 24.33 |
| `--estimator ekf` (Kalman-filter velocity + height in the observation) | 2.7 min | 24.25 vs 24.33 |

## Step 1: flat-ground walking on the GPU: `runs/mjx_300m`

```
train_mjx.py --num-timesteps 300000000 --num-envs 4096 --impl warp --est-obs --num-evals 15 --name mjx_300m
```

- **Setup:** MuJoCo Warp backend at ~83 k env steps/s. The policy also sees the Kalman-filter body velocity and height (`--est-obs`), with the filter's measured error added. Only the feet collide with the ground.
- **Training:** 300 M steps in 42 min.
- **Eval reward (step: reward):** 0: 7.9 · 21 M: 40.9 · 43 M: 53.7 · 107 M: 58.9 · 214 M: 60.6 · 300 M: 61.3.
  - Most of the gain comes in the first 50 M steps; after that it's refinement.
- **An earlier attempt died at 43 M steps.** I had started it as a background job with a 10-minute limit, and the limit killed it. Runs now go in their own terminal tab.

Flat evaluation (`runs/mjx_300m/eval.json`). Solver at MuJoCo's default 100 iterations; terrain runs use 10.

| Command | Velocity error | Yaw-rate error | Falls |
|---|---|---|---|
| forward 8 cm/s | 2.0 mm/s | 0.009 rad/s | 0 % |
| back 8 cm/s | 3.8 mm/s | 0.011 rad/s | 0 % |
| sideways ±8 cm/s | 1.6 / 4.6 mm/s | 0.006 / 0.026 rad/s | 0 % |
| fast 12 cm/s | 3.5 mm/s | 0.008 rad/s | 0 % |
| turn 0.4 rad/s | 2.0 mm/s | 0.009 rad/s | 0 % |
| stand | 0.2 mm/s | 0.002 rad/s | 0 % |

Videos:
- [step 0 (untrained)](media/training/flat_step000M.mp4)
- [21 M](media/training/flat_step021M.mp4)
- [107 M](media/training/flat_step107M.mp4)
- [300 M final](media/training/flat_step300M_final.mp4)
- [dashboard at the end](media/training/flat_dashboard_final.mp4)

### Baseline: the flat policy on the terrain course

`eval_mjx.py runs/mjx_300m --terrain` (`runs/mjx_300m/eval_terrain.json`). Each cell is success % / falls %.

| Terrain | L0 | L3 | L6 | L9 |
|---|---|---|---|---|
| rocks (1 → 5 cm) | 100 / 0 | 98 / 0 | 70 / 0 | 18 / 0 |
| slope up, walking down (0 → 15°) | 100 / 0 | 99 / 0 | 91 / 0 | 82 / 0 |
| slope down, climbing out | 100 / 0 | 86 / 0 | 6 / 0 | 0 / 0 |
| stairs, walking down (1 → 6 cm risers) | 88 / 0 | 25 / 0 | 0 / 0 | 0 / 0 |
| stairs, climbing out | 66 / 0 | 0 / 0 | 0 / 0 | 0 / 11 |
| blocks (0 → 6 cm) | 100 / 0 | 69 / 0 | 52 / 0 | 12 / 0 |

- Walking blind on flat-ground training, Rocky handles small rocks and gentle downhill. It cannot climb, which needs lifting feet onto higher ground, and it mostly just stalls (few falls) when it can't progress.
- **Measurement bug fixed along the way:** a first version of this table showed blocks failing 100 % even on flat level-0 tiles. Warp shares one contact buffer across all envs in a batch; when it overflowed, the last envs (the blocks column) lost their ground contacts and fell through. The eval and training budgets are now 64 contacts per env.

## Step 2: blind rough-terrain policy with a curriculum: `runs/mjx_terrain` (running)

```
train_mjx.py --num-timesteps 200000000 --num-envs 4096 --terrain curriculum --impl warp --est-obs --num-evals 20 --name mjx_terrain
```

**Terrain** (`rocky_terrain.py`, overview in `media/terrain_overview.png`): one 24 × 40 m heightfield made of 4 m tiles.
- 6 terrain types × 10 difficulty levels: rocks, both pyramids (slope up and down), both stair pyramids (1 → 6 cm risers, 30 cm treads), and blocks.
- Each tile has a flat 0.6 m spawn pad in the middle.

**Curriculum** (per env):
- Envs start at levels 0–2.
- After each episode, an env moves up a level if it covered at least 70 % of the commanded distance without falling, and down if it covered under 35 % or fell.
- It then respawns on a random terrain type at its new level.
- Envs that clear level 9 get a random level, so the policy keeps practising the easy tiles.

**Collisions:**
- The base and leg links are now capsules and cylinders that collide with the terrain; the Warp backend can't do heightfield contact with boxes or meshes.
- Plate, dome or camera touching the ground ends the episode. Shin, link and belly contacts are penalised.

**Observations and reward:**
- The policy is still blind: the same observation as the flat run.
- The critic additionally sees a 7 × 7 grid of terrain heights around the robot.
- Reward changes from flat: no body-height penalty, a smaller tilt penalty, and the collision penalties above.

**Engineering issues hit and fixed before this run:**

| Problem | Fix |
|---|---|
| Warp has no heightfield collisions on the JAX backend | Terrain runs use the Warp backend only |
| Videos took ~6 min each: every frame drew the whole 1.5 M-cell heightfield | Render-only copy of the model with just the three showcase tiles: 61 s per clip |
| The solver hit MuJoCo's default 100-iteration cap on the terrain | Capped at 10 iterations / 20 line-search iterations, as GPU sims usually do |
| Warp's overflow warnings are GPU printfs, some per env per step | Switched off |
| Contact buffer overflow (above) | 64 contacts per env |
| GPU memory overflowed at 4096 envs: the solver keeps (constraint rows)² floats per env, and the eval env had been sized like the training batch | Rows capped at 160 per env; eval env sized for its 128 envs. The run now uses 10.8 of 12 GB at ~15 k env steps/s |

Videos:
- [step 0 (untrained, on the three showcase tiles)](media/training/terrain_step000M.mp4)
- More to be added as checkpoints arrive.
