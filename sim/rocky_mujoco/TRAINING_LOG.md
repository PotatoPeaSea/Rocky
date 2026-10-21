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

## Step 2: blind rough-terrain policy with a curriculum: `runs/mjx_terrain2`

```
train_mjx.py --num-timesteps 120000000 --num-envs 4096 --terrain curriculum --impl warp --est-obs --num-evals 12 --name mjx_terrain2
```

120 M steps in 1 h 53 min, at ~17 k env steps/s using 10.8 of 12 GB.

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
| GPU memory overflowed at 4096 envs: the solver keeps (constraint rows)² floats per env, and the eval env had been sized like the training batch | Rows capped at 160 per env; eval env sized for its 128 envs |
| **First 200 M-step attempt (`runs/mjx_terrain_diverged`) diverged at 202 M steps.** Training reward had plateaued at ~57 from 100 M. Then one simulated robot blew up, and its NaN observations went into the shared observation normaliser and the loss: every policy and value weight became NaN, and Brax crashed. Only the latest checkpoint was kept, so the last good one (192 M) had already been overwritten | NaN guards: an env whose state is non-finite or whose joint speeds exceed 100 rad/s ends and resets, and observations and rewards are scrubbed of NaN/inf. Every checkpoint is now kept (`checkpoints/params_step_*.pkl`), and a non-finite one never replaces `params_latest.pkl`. Re-ran for 120 M steps, since the plateau starts at ~100 M; no NaN warnings |

Eval reward (step: reward): 0: 8.2 · 11 M: 26.8 · 22 M: 41.9 · 44 M: 53.4 · 66 M: 56.2 · 99 M: 57.8 · 121 M: 57.6.

### Result: blind terrain policy vs the flat policy

`eval_mjx.py runs/mjx_terrain2 --terrain` (`runs/mjx_terrain2/eval_terrain.json`). Each cell is success %, flat policy → terrain policy; terrain-policy falls are in brackets when not 0.

| Terrain | L0 | L3 | L6 | L9 |
|---|---|---|---|---|
| rocks (1 → 5 cm) | 100 → 100 | 98 → 100 | 70 → **100** | 18 → **98** |
| slope up, walking down (0 → 15°) | 100 → 100 | 99 → 100 | 91 → 100 | 82 → 67 (1 % falls) |
| slope down, climbing out | 100 → 100 | 86 → 100 | 6 → **62** | 0 → 0 |
| stairs, walking down (1 → 6 cm risers) | 88 → 100 | 25 → **100** | 0 → **48** | 0 → 3 (2 % falls) |
| stairs, climbing out (1 → 6 cm risers) | 66 → **100** | 0 → 0 | 0 → 1 | 0 → 0 (9 % falls) |
| blocks (0 → 6 cm) | 100 → 100 | 69 → **100** | 52 → 73 | 12 → 30 |

**What improved:**
- **Rocks are solved:** 98 % success on 5 cm rocks (level 9), up from 18 %.
- **Downhill:** walking down 2.7 cm stairs (level 3) goes from 25 % to 100 %, and down 4.3 cm stairs from 0 % to 48 %.
- **Climbing out of slope pits:** up to 10° (level 6), 62 % from 6 %.
- **Blocks:** 100 % up to 2 cm, and 73 % at 3.3 cm.
- **It almost never falls:** when the terrain policy can't make progress, it stalls rather than tipping over.

**What it still can't do: climb stairs.**
- It manages 1 cm risers (level 0) but 0 % at 2.7 cm (level 3), where it gets about 0.6 m before stalling.
- This is a known limit of blind policies. A foot that hits a riser feels like any other obstacle, and the policy has no way to know a step is coming or to lift its feet higher in advance.
- The likely fixes are to give the policy the terrain ahead (step 2 of the perception plan: the OAK-D height map, through a teacher-student setup), or to add a reward for swing-foot clearance.

**Regressions and oddities:**
- **Steep downhill:** 15° downhill (level 9) dropped from 82 % to 67 %. The new policy walks more cautiously, which costs distance on the steepest slope.
- **Stair-pit numbers:** the 9 % falls in the level-9 stair pit, and its mean distance of 3.6 m, come from a few robots tumbling down the 30 cm deep pyramid.

**Caveats:**
- These numbers are in the GPU simulation: only the feet and the capsule approximations collide, with at most one contact per capsule. There's no encoder quantisation, latency or backlash, and the solver is capped at 10 iterations.
- The CPU env (`rocky_env.py`, with the real EKF and camera in the loop) can't load these GPU policies yet, so they haven't been checked there.

Videos:

| | |
|---|---|
| [step 0, untrained](media/training/terrain_step000M.mp4) | eval reward 8.2 |
| [22 M](media/training/terrain_step022M.mp4) | eval reward 41.9 |
| [55 M](media/training/terrain_step055M.mp4) | eval reward 55.3, near the plateau |
| [120 M, final](media/training/terrain_step120M_final.mp4) | eval reward 57.6. The middle segment is climbing 4.3 cm stairs (level 6), where the eval gives 1 % success |
| [dashboard at 40 M](media/training/terrain_dashboard_040M.mp4), [dashboard at the end](media/training/terrain_dashboard_final.mp4) | clip and reward curve together |

## Step 3: teacher policy that sees the terrain heights: `runs/mjx_teacher`

```
train_mjx.py --num-timesteps 120000000 --num-envs 4096 --terrain curriculum --impl warp --est-obs --scan true --num-evals 12 --name mjx_teacher
```

120 M steps in 1 h 56 min. Same terrain, curriculum, rewards and collisions as step 2. The differences:

- **Height scan in the observation (`--scan true`):** a 13 × 13 grid of points 6 cm apart (72 × 72 cm around the body, turning with the body's heading). Each point gives the base's height above the exact terrain there. This is "perfect perception": the teacher of step 4.
- **Commands are camera-first:** the walking direction is within ±45° of body −y, the side the OAK-D faces. The camera can only map ground it can see, so the student in step 4 must walk that way, and the teacher is trained the same way for a like-for-like comparison.

Eval reward (step: reward): 0: 8.7 · 11 M: 27.7 · 22 M: 44.6 · 33 M: 50.7 · 44 M: 52.7 · 55 M: 54.6 · 77 M: 55.3 · 99 M: 55.7 · 121 M: 55.2.
The reward plateaus where the blind run's did. Most of the course is terrain the blind policy already handled, so the reward can't show the difference; the per-terrain eval below does.

### Result: blind vs teacher, both walking camera-first

`eval_mjx.py <run> --terrain --dir=-y` (`eval_terrain_negy.json` in each run). Unlike the step-2 table, both policies walk towards body −y here, so the tiles and directions are identical. Success %, blind → teacher; teacher falls in brackets when not 0.

| Terrain | L0 | L3 | L6 | L9 |
|---|---|---|---|---|
| rocks (1 → 5 cm) | 100 → 100 | 100 → 100 | 100 → 100 | 99 → 99 |
| slope up, walking down (0 → 15°) | 100 → 100 | 100 → 100 | 100 → 100 | 100 → 99 |
| slope down, climbing out | 100 → 100 | 100 → 100 | 69 → **100** | 12 → **99** |
| stairs, walking down (1 → 6 cm risers) | 100 → 100 | 100 → 100 | 73 → **96** | 17 → 29 (5 % falls) |
| stairs, climbing out (1 → 6 cm risers) | 100 → 100 | 2 → **43** | 2 → 9 | 1 → 1 |
| blocks (0 → 6 cm) | 100 → 100 | 100 → 100 | 91 → 89 | 46 → **74** |

**What the height scan buys:**
- **First real stair climbing:** 43 % up 2.7 cm risers (level 3), where the blind policy gets 2 %.
- **Climbing out of slope pits:** 99 % at 15° (level 9), up from 12 %.
- **Walking down stairs:** 96 % on 4.3 cm risers (level 6), up from 73 %.
- **High blocks:** 74 % at 6 cm (level 9), up from 46 %.

**Still not solved:** climbing 4.3 cm and 6 cm risers (levels 6 and 9): 9 % and 1 %. The risers are 20–30 % of the leg's reach, and the policy has to lift each foot over them. Ideas for later: a swing-foot clearance reward, more stair tiles in the curriculum, a longer run.

Videos:

| | |
|---|---|
| [step 0, untrained](media/training/teacher_step000M.mp4) | |
| [22 M](media/training/teacher_step022M.mp4) | eval reward 44.6 |
| [55 M](media/training/teacher_step055M.mp4) | eval reward 54.6 |
| [120 M, final](media/training/teacher_step120M_final.mp4) | same three showcase tiles as step 2 (5 cm rocks, 4.3 cm stairs, 3.3 cm blocks), walking camera-first |
| [dashboard at 40 M](media/training/teacher_dashboard_040M.mp4), [dashboard at the end](media/training/teacher_dashboard_final.mp4) | |

## Step 4: student policy on the simulated OAK-D height map: `runs/mjx_student`

```
distill_mjx.py --teacher runs/mjx_teacher --name mjx_student --num-steps 60000000 --num-envs 1024 --unroll 40
```

60 M steps in 70 min at ~14 k env steps/s.

**What the student sees (`scan=camera`):** the same 13 × 13 grid, but as the camera would map it.
- **Only seen cells:** a cell counts as known only if it was inside the OAK-D's field of view (73 × 58°, 0.2–3 m) from one of the last 40 camera poses (one every 5 steps, so the last 4 s).
- **Unseen cells:** read 0, and a second 169-value mask says which cells are known.
- **Errors:** known cells get noise growing with range (4 mm + 4 mm per metre) and a per-episode offset (1 cm in height, 1 cm sideways), standing in for drift of the estimator's pose.
- At spawn only the camera side of the grid is known; the rest fills in as the robot walks.

**Training (DAgger distillation):** no reward is involved.
- The student drives the robot. The teacher, which sees the exact heights, labels every visited state with the action it would take.
- The student is fitted to those labels (squared error on the action).
- For the first 15 % of training the teacher drives a falling share of the envs, so the student sees good behaviour before its own mistakes.
- Domain randomisation and the terrain curriculum are on, as in training.

**Fixed along the way:** the first attempt (2048 envs) ran out of GPU memory after 25 iterations, because another process on the PC was holding ~5.8 GB of the GPU. Re-run with 1024 envs × 40 steps per iteration (the same 41 k samples per update).

Imitation error (mean squared action difference; actions are in [−1, 1]): 0.12 at the start → 0.003 by 6 M steps, and 0.0015–0.003 to the end. Reward on the student's own rollouts was 54–58 per episode from 6 M steps on, the teacher's level.

### Result: blind vs teacher vs student, all walking camera-first

`eval_mjx.py <run> --terrain --dir=-y`. Success %, blind / teacher / **student**.

| Terrain | L0 | L3 | L6 | L9 |
|---|---|---|---|---|
| rocks (1 → 5 cm) | 100 / 100 / **100** | 100 / 100 / **100** | 100 / 100 / **100** | 99 / 99 / **100** |
| slope up, walking down (0 → 15°) | 100 / 100 / **100** | 100 / 100 / **100** | 100 / 100 / **100** | 100 / 99 / **98** |
| slope down, climbing out | 100 / 100 / **100** | 100 / 100 / **100** | 69 / 100 / **100** | 12 / 99 / **91** |
| stairs, walking down (1 → 6 cm risers) | 100 / 100 / **100** | 100 / 100 / **100** | 73 / 96 / **100** | 17 / 29 / **55** (2 % falls) |
| stairs, climbing out (1 → 6 cm risers) | 100 / 100 / **100** | 2 / 43 / **33** | 2 / 9 / **12** | 1 / 1 / **0** |
| blocks (0 → 6 cm) | 100 / 100 / **100** | 100 / 100 / **100** | 91 / 89 / **86** | 46 / 74 / **73** |

**What carries over from the teacher:**
- **Stair climbing with real perception:** 33 % up 2.7 cm risers (level 3), against 2 % blind. The teacher, with exact heights, gets 43 %.
- **Slopes and blocks:** climbing out of the 15° pit drops only from 99 % to 91 %. 6 cm blocks hold at 73 %.
- **Walking down stairs:** the student beats the teacher on the steepest descent (55 % vs 29 %).

**Where it falls short:** climbing 4.3 cm and 6 cm risers stays unsolved (12 % and 0 %), as it was for the teacher. The student can only be as good as its teacher here, so that work goes on the teacher side.

**The robot side of this map:** `rocky_vision.ElevationMap` (validated with `elevation_demo.py`, `media/elevation_map.png`) builds the same grid and mask from real depth frames and the estimator's pose. In that test its error (mean 2–4 mm, 95th percentile up to 24 mm on stair edges) is smaller than the noise the student was trained on (mean 9 mm, 95th percentile 21 mm).

**Caveats:**
- This is all in simulation, with the camera map simulated as noisy truth rather than rendered depth.
- On the real robot `ElevationMap` replaces it, so its delay (the Pi's frame rate) and real stereo artefacts haven't been tested in the loop yet.

Videos:

| | |
|---|---|
| [blind vs teacher vs student on the same stair pit](media/training/compare_stairs_L3.mp4) | `compare_stairs.py`, level 3 (2.7 cm risers), same start and heading, side view. Of 8 headings, blind gets out 0, teacher 5, student 5 (a small sample; the 128-run eval gives 2 / 43 / 33 %). The shown robots rise 4 / 7 / 7 cm over 0.9 / 1.3 / 1.3 m, i.e. they scrape up two or three 2.7 cm steps. **From the side view this is hard to see, and it is not convincing stair climbing** (see the gait problems below) |
| [student, step 0](media/training/student_step000M.mp4) | random student weights |
| [student, 18 M](media/training/student_step018M.mp4) | |
| [student, 60 M final](media/training/student_step060M_final.mp4) | the showcase tiles, camera-first |
| [dashboard at 40 M](media/training/student_dashboard_040M.mp4), [dashboard at the end](media/training/student_dashboard_final.mp4) | for the student the curve is reward on its own rollouts |

### Problem found by watching the videos: the gait is poor, and "climbing" is scraping

Prompted by the user watching `compare_stairs_L3.mp4` ("it's not climbing the stairs at all, the way it's walking is really weird and looks inefficient and slow"). Measured from the same rollouts (foot positions vs the terrain under them, 15 s at 10 cm/s):

| | blind | teacher | student |
|---|---|---|---|
| speed (commanded 10 cm/s) | 7.6 cm/s | 9.4 cm/s | 9.5 cm/s |
| stride length | 2.5 cm | 2.6 cm | 2.5 cm |
| steps per leg in 15 s (legs 1-5) | 43 64 48 28 44 | 57 14 12 116 72 | 70 6 10 130 72 |
| median foot height in swing | 1.3 cm | 1.6 cm | 1.6 cm |
| joint direction reversals per joint | 27 /s | 23 /s | 24 /s |

- **Shuffling:** 2.5 cm strides at 3-4 steps a second instead of long, regular steps.
- **Lopsided:** in the height-scan policies two legs hardly step (6-14 times in 15 s) and are dragged, while leg 4 paddles at ~8 Hz.
- **Jitter:** the joints reverse direction ~24 times a second, a tremor rather than a gait.
- **Feet barely lift:** the median swing height is 1.6 cm, below a 2.7 cm riser, so the robot gets up steps by scraping and pushing rather than stepping onto them. That is also why 4.3-6 cm risers fail.
- **Slow by design, too:** commands only go up to 12 cm/s. The servos (5.5 rad/s no-load) allow more.

**Cause:** the reward only asks for velocity tracking, staying upright and alive; the gait-shaping terms are tiny (`action_rate` -0.01, `torques` -0.002, `feet_air_time` +1.0 with a 0.2 s threshold). Nothing asks for regular, symmetric, high, smooth steps, so PPO found a jittery shuffle that tracks the velocity.

## Step 5: a real gait on flat ground: `runs/gait1` → `runs/gait4`

Four 100 M-step flat runs (≈ 20 min each), each checked with **`gait_check.py`**. For one deterministic rollout per command (no randomisation), it measures speed, stride, step rate per leg, how evenly the legs share the work, swing-foot height, joint jitter and slip of planted feet. It also draws a foot-contact diagram and records a close-up side video that follows the robot.

```
train_mjx.py --num-timesteps 100000000 --num-envs 4096 --impl warp --est-obs --num-evals 8 --name gait4
python gait_check.py runs/gait4
```

**What was added to the env** (`rocky_mjx_env.py`; the old behaviour is `--no-gait`; runs made before this have no `gait` entry in their config and load without it):
- **Gait clock:** a 1.5 Hz phase in the observation. Each leg has a 40 % swing window, in the order 1, 3, 5, 2, 4, so two non-adjacent legs swing at a time and three are always down. With a zero command, all feet are meant to stay down.
- **New reward terms:**
  - `gait_contact`: feet on the ground or in the air when the schedule says so;
  - `feet_clearance`: swing feet follow a 3.5 cm half-sine (5 cm on terrain), and stance feet stay on the ground;
  - `foot_slip`: penalty on planted feet moving;
  - `action_acc`: penalty on the second difference of the action (high-frequency jitter);
  - `dof_acc`: penalty on joint acceleration.
- **Commands:** up to 25 cm/s (was 12) and 0.8 rad/s.
- **Velocity tracking in two terms:** a loose one (σ² = 0.01) for a gradient from standing, and a sharp one (σ² = 0.001) for actually reaching the speed.
- `train_mjx.py` writes each reward term's contribution to `progress.csv` (`r_*` columns) and takes `--set key=value` config overrides.

| Run | What changed | What happened |
|---|---|---|
| old `mjx_300m` | — | Skates: legs 1-3 never lift, leg 4 paddles at 4-11 Hz, feet rise 6 mm and slide as fast as the body moves, joints reverse 13-23 times/s |
| `gait1` | gait clock + terms above, slip −10, schedule weight 1 | Regular stepping on 4 legs, but **leg 4 is held in the air the whole time**: it cost only ~2 reward per episode and saved slip and acceleration penalties. Reaches 14 of 20 cm/s |
| `gait2` | stance feet must stay down, schedule weight 3, sharp velocity tracking only, slip −30 | **Marches in place** (0.2-0.9 cm/s). The sharp tracking term gives no gradient from standing still, and the schedule pays well for stepping on the spot |
| `gait3` | loose + sharp tracking, schedule weight 1.5 | Moves, but slowly (4-6 cm/s at 72 M). The slip term also charged a foot touching down at swing speed, so long fast steps were expensive |
| **`gait4`** | slip counted only for feet planted this step and the last, back to −10 | **Walks properly** (table below) |

`gait_check.py` results, old flat policy → `gait4`:

| Command | speed | stride | step rate, legs 1-5 (Hz) | swing height median / p90 | joint reversals | planted-foot slip |
|---|---|---|---|---|---|---|
| forward 10 cm/s | 10.0 → 10.0 | 10.0* → 6.6 cm | 0, 0, 0, 4.5, 0.5 → 1.5 ×5 | 0.6 / 0.7 → 1.9 / 2.7 cm | 13 → 6 /s | 10 → 5 cm/s |
| forward 20 cm/s | 19.4 → 20.6 | 9.7* → 13.8 cm | 0, 0.25, 0, 5.0, 4.75 → 1.5 ×5 | 0.7 / 1.0 → 2.2 / 3.4 cm | 18 → 6 /s | 17 → 9 cm/s |
| camera-first 15 cm/s | 14.8 → 15.3 | 3.5* → 10.2 cm | 8.8, 0.1, 0, 10.6, 1.4 → 1.5 ×5 | 0.5 / 0.7 → 2.0 / 2.9 cm | 23 → 5 /s | 15 → 6 cm/s |
| sideways 15 cm/s | 14.7 → 15.5 | 6.1* → 10.3 cm | 0.6, 0.4, 0.1, 11, 0 → 1.5 ×5 | 0.5 / 0.7 → 2.0 / 3.0 cm | 17 → 6 /s | 16 → 7 cm/s |
| stand | still → still | | | | 3.5 → 1.5 /s | |

\* the old policy's "stride" is speed / mean step rate; with most legs not stepping it doesn't mean much. The old policy was also trained only up to 12 cm/s, so 15-20 cm/s is outside its range.

**Still not perfect:** planted feet slip 5-9 cm/s, about half the old policy's. Feet lift 2-3.4 cm on flat ground, below the 3.5 cm target.

Videos (close-ups follow the robot from the side; every command, one after another):

| | |
|---|---|
| [old flat policy, close-up](media/training/gait0_old_flat.mp4), [its foot contacts](media/training/gait0_old_flat_contacts.png) | the skating |
| [gait1](media/training/gait1.mp4), [contacts](media/training/gait1_contacts.png) | regular steps, leg 4 held up |
| [gait2 at 43 M](media/training/gait2_43M_marching.mp4) | marching in place |
| [gait3 at 72 M](media/training/gait3_72M_slow.mp4) | slow, short steps |
| [**gait4**](media/training/gait4.mp4), [**contacts**](media/training/gait4_contacts.png) | the wave gait: every leg once per cycle, two non-adjacent legs up at a time |
| [gait4 training: step 0](media/training/gait4_step000M.mp4), [29 M](media/training/gait4_step029M.mp4), [100 M](media/training/gait4_step100M_final.mp4), [dashboard](media/training/gait4_dashboard_final.mp4) | tracking-camera clips, 10 and 20 cm/s forward, sideways, turning |
