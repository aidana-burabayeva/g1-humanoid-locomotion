# Experiment: 10 cm stairs (`model_33800`)

The goal was a single 283-input G1 policy that crosses both regular and
inverted 10 cm stairs without a terrain label or a policy switch. This is an
experimental extension. The release checkpoint `model_28996` and its
manifest are unchanged.

## Test geometry

**Full crossing.** The robot starts 1 m before a flat-to-stairs boundary.
The run ends when it reaches the flat landing after an 8 m stair tile. Each
geometry contains an ascent and a descent. A run succeeds only if the robot
reaches the landing within 60 s without falling or leaving its terrain row.
The command is `(0.5, 0, 0)` m/s with the same position-based yaw correction
used on the release route. Terrain row 9 gives an actual 10 cm step. The
pinned mjlab pyramid generator builds four 10 cm rises to a 40 cm central
platform and four 10 cm drops, with 40 cm step depth. The inverted variant
traverses the same height changes in the opposite order. These are synthetic
box stairs; physical stairs are outside this test.

**Fixed-10 cm route.** The same flat → rough → slope → stairs → flat layout
as the release route, at row 4, with the step height fixed at 10 cm. The
original row-4 route has steps of about 2.22 cm.

Tasks: `G1-Stairs10-FullCrossing-Deploy`, `G1-StairsInv10-FullCrossing-Deploy`,
and `G1-FinalRoute-StairsFixed10-Deploy`, defined in
[`src/g1_locomotion/tasks/stairs10.py`](../src/g1_locomotion/tasks/stairs10.py).
Evaluators: [`stairs_eval.py`](../src/g1_locomotion/stairs_eval.py) and
[`mixed_route_eval.py`](../src/g1_locomotion/mixed_route_eval.py).

## Acceptance criteria (fixed before the runs)

- At least 17/20 full crossings for each geometry on each seed.
- On the fixed-10 cm route: at least 17/20 completions per seed, forward-speed
  MAE at most 0.15 m/s, a success-rate spread of at most 15 percentage
  points, and at most two falls within 1 s of any terrain boundary.
- The candidate must also keep the release route limits on the original
  row-4 route and pass the nine required flat-ground commands.
- Held-out promotion test: the same gates on seeds 4 and 5. Seed 3 was used
  in training; it is reported separately and must also reach 17/20, but it
  is not evidence of generalization. The checkpoint is not reselected after
  this test.

## Training

`model_33800` comes from three bounded fine-tunes of the 5 cm stair policy
(iteration 30550). Each fine-tune used 512 environments and changed one
thing at a time:

1. Stair height range raised from 0–5 cm to 0–10 cm (2000 iterations,
   seed 2). At seed 0, checkpoint 32000 crossed 10 cm regular and inverted
   stairs 9/20 and 11/20 times, but 20/20 and 19/20 at about 7.8 cm. This
   sharp difficulty boundary motivated step 2.
2. From 32000, minimum stair height raised to 7.5 cm (1500 iterations,
   seed 3). Checkpoint 33499 cleared all GPU stair tests but missed the
   original-route gate on seed 1 twice (16/20). All misses were timeouts
   0.4–1.5 m before the finish.
3. From 33499, the `track_linear_velocity` reward weight raised from 2.0 to
   2.5 (600 iterations, seed 3). Checkpoint 34000 failed the flat standing
   test (1.198 m drift against a 0.5 m limit). Checkpoint 33800 passed all
   selection gates and became the candidate.

All three tasks are registered in
[`stairs10.py`](../src/g1_locomotion/tasks/stairs10.py); the `train`
service starts step 3 by default.

## Results

Selection seeds, container run with 20 robots × 60 s
([verdict and raw results](results/stairs10_selection/)):

| Scenario | Seed 0 | Seed 1 | Seed 2 |
| --- | ---: | ---: | ---: |
| Regular 10 cm stairs, full crossing | 19 | 19 | 19 |
| Inverted 10 cm stairs, full crossing | 18 | 19 | 19 |
| Original row-4 route, ~2.22 cm steps | 19 | 20 | 20 |
| Row-4 route, fixed 10 cm steps | 19 | 18 | 18 |

On the fixed-10 cm route the forward-speed MAE was 0.0690/0.0691/0.0669 m/s.
All 60 robots reached the stair section; there were 1/2/1 falls, 0/0/1
timeouts, and no departures from the row. All nine required flat commands
passed without falls.

A rerun of the same selection protocol with this repository's image
([verdict and raw results](results/stairs10_selection_rerun/)) gave regular
19/20/19, inverted 20/20/20, original route 20/20/20, and a passing flat
protocol, but the fixed-10 cm route dropped to 15/16/19 (seed-0 repeat
16/20), so that rerun is a FAIL. The route with 10 cm steps is the weak
point in both runs. A parity check ruled out configuration drift: task,
checkpoint, mjlab commit and dependency versions match, and the task and
evaluation code differ only in import paths. Repeating seed 0 three more
times in each image ([raw results](results/stairs10_seed0_repeats/)) gave
19/17/17 in the development image and 17/18/19 in this repository's image,
so the spread is GPU run-to-run non-determinism. Pooled over all 16 route
runs on seeds 0–5 the success rate is 280/320 = 87.5 % (95 % Wilson
interval 83–91 %).

Held-out test ([verdict and raw results](results/stairs10_heldout/)):

| Scenario | Seed 3 (training) | Seed 4 (held out) | Seed 5 (held out) |
| --- | ---: | ---: | ---: |
| Regular 10 cm stairs | 17 | 17 | 19 |
| Inverted 10 cm stairs | 18 | 19 | 17 |
| Route with fixed 10 cm steps | 19 | **16** | 17 |

**The checkpoint failed the held-out gate.** On the seed-4 route, three
robots fell on the stair tile and one timed out. All 20 reached the stairs
and none left the row. The forward-speed MAE was 0.0665 m/s, below the
limit. Seed 5 had three falls and an MAE of 0.0659 m/s. The held-out spread
was 5 percentage points. The only check that failed was the seed-4 success
count; tracking error and seam falls were within limits.

Across both held-out seeds the route succeeded in 33 of 40 runs (82.5 %;
95 % Wilson interval 68–91 %). If the true success rate were 85 %, a ≥17/20
gate would pass on all three seeds of one scenario only about 27 % of the
time, so the selection-seed pass partly reflects selection.
`model_33800` remains experimental.

## Diagnostics with the release checkpoint

- On inverted 10 cm stairs, `model_28996` sees the 10 cm rise in its
  187-point height scan shortly before falling. Torso pitch reaches 62° and
  foot contact is lost at 3.98 s; the matching 5 cm rollout stays upright
  for 12 s. This points to a locomotion-capability limit rather than missing
  terrain information, but the exact failure mechanism is unproven.
- `model_28996` completed 20/20 full 5 cm crossings for each geometry at
  seed 0. This is a control for the protocol, not evidence on 10 cm stairs.

## Reproduce

```bash
docker compose run --rm eval-stairs10                                        # selection seeds 0-2
docker compose run --rm eval-stairs10 /app/scripts/eval_stairs10_heldout.py  # seeds 3-5
```

Both commands verify the `model_33800` SHA-256, the network dimensions, and
the pinned environment, then write a `verdict.json` under `outputs/`. GPU
results are not bit-for-bit deterministic, so counts can differ by a run or
two between repetitions.
