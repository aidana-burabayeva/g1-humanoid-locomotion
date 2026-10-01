# Engineering notes

This document records the decisions that affect the released simulation
policy. Measurements use MuJoCo Lab 1.6.0 at
[`c2e1e06`](https://github.com/mujocolab/mjlab/tree/c2e1e06400e309b6897a536693fe5d2aa772c5b2)
unless stated otherwise. The accepted checkpoint is `model_28996.pt`; its
hash and evaluation configuration are in the [release manifest](../weights/manifest.json).
The policy has not been validated on a physical G1.

## Policy interface and training progression

The actor receives 283 values: 96 proprioceptive, previous-action, and
command values, followed by a 187-value local height scan. It produces 29
joint-position actions. The critic has additional privileged observations.
The [policy interface](model-contract.md) specifies the ordering and
compatibility boundary. The route controller uses position and heading to
adjust only the yaw-rate command; it does not select a policy or send a
terrain label.

The initial flat-policy comparison held the training budget at 147.5 million
environment steps. Adding a height scan to the 99-value flat actor had a
smaller effect than removing `base_lin_vel` from the actor: at the same
budget, the 283-value policy missed forward and yaw-tracking limits despite
zero falls in the flat protocol. Its learning curves were still rising at
the budget limit. This comparison used one training seed and does not
isolate the eventual performance of either interface. Subsequent training
kept `base_lin_vel` out of the actor so the final interface could be
assembled from available proprioception, command, and height-scan inputs.

Training directly on the stock rough-terrain task produced a largely
stationary policy and little terrain-curriculum progression. The
[`terrain_levels_vel` implementation](https://github.com/mujocolab/mjlab/blob/c2e1e06400e309b6897a536693fe5d2aa772c5b2/src/mjlab/envs/mdp/curriculums.py)
uses commanded travel in its promotion logic; the rough task also has a
[`command_vel` curriculum](https://github.com/mujocolab/mjlab/blob/c2e1e06400e309b6897a536693fe5d2aa772c5b2/src/mjlab/tasks/velocity/velocity_env_cfg.py)
that expands the command range during training. Removing `command_vel` in
a resumed comparison raised mean terrain level from 0.0030 to 0.0205 over
the measured window, still near zero on a 0–9 scale. That single-seed
comparison also reset environment state at resume; it supports a possible
interaction but does not establish `command_vel` as the root cause. Starting
rough-terrain training from a walking flat checkpoint raised mean level to
approximately 5/9. The stronger evidence is therefore that a policy must
travel before the distance-based curriculum can advance it.

## Preserving command tracking

A rough-terrain checkpoint learned to survive but lost yaw-command tracking:
at a commanded ±0.5 rad/s, measured angular-speed MAE was approximately
0.49–0.50 rad/s on flat ground. Mixing flat and rough terrain preserved more
tracking, but the negative-yaw command remained outside the 0.15 rad/s
limit. A mirror-loss-only run and a higher yaw-tracking reward weight
improved the evaluated policy. These comparisons include training noise and
different intermediate checkpoints; they describe observed outcomes, not a
general guarantee that the same weights work for another robot.

The first symmetry run enabled both data augmentation and mirror loss. Its
reward dropped from 57.73 to 29.17 and its logged yaw-tracking reward from
0.876 to 0.363 between iterations 9100 and 9948. Physical checks of the
observation and action mirror did not identify a mapping error. In the
[RSL-RL 5.5.1 symmetry implementation](https://github.com/leggedrobotics/rsl_rl/blob/v5.5.1/rsl_rl/extensions/symmetry.py),
`augment_batch` repeats the original `old_actions_log_prob` for mirrored
samples. For an asymmetric behavior policy, that value need not equal the
old-policy log probability of the mirrored action under the mirrored
observation. The measured log-probability difference was about −4.3 ± 2.9
in that run. This is a plausible mechanism for the observed PPO instability;
the project did not prove it is the only cause. Mirror loss without data
augmentation avoided the collapse in the tested run, and the accepted
policy uses that configuration.

## Terrain and evaluation boundaries

The project configuration keeps random-roughness amplitude at 0.02–0.06 m
independently of terrain row. Its row number must therefore not be read as
a roughness-amplitude scale. Slope and step height do change with row: the
accepted row-4 route has an approximately 10° slope and 2.22 cm steps.
The final policy was trained on flat ground, random roughness, and slopes;
the step-specific fine-tune was stopped without accepting a new checkpoint.

The isolated 5 cm step tests terminate at the starting tile edge, after
about 10 seconds on average. Their 20/20 result means tile exits without
falls, not 60 seconds of step climbing. The continuous row-4 route was
tested separately and completed 19/20, 18/20, and 17/20 runs in the
container on seeds 0, 1, and 2. The [verdict and raw JSON](results/release_route_row4/)
record timeouts, falls, and departures separately. Fixed-seed reruns have
produced different completion counts, so the table is a measured result,
not a deterministic promise.

At actual 10 cm step height, the inverted isolated test produced 20/20
falls at seed 0 and 18/20 at seed 1. The row-9 continuous route with 5 cm
steps completed 16/20; its four falls occurred on the steeper slope before
the steps. These observations bound the current policy's demonstrated
capability and motivate separate work on higher steps and external-simulator
validation.

## Training lineage and reproduction

Each checkpoint was produced by resuming the previous one on a new task.
The terrain steps used 512 environments:

| Step | Task | Iterations | Training seed |
| --- | --- | --- | ---: |
| Flat ground, actor without `base_lin_vel` (resumed runs, 2048 environments) | flat deploy tasks | → 8998 | 1 |
| Flat and rough terrain, yaw-tracking weight 4 | `G1-Mix-Deploy-NoCmdCurriculum-SymLoss-YawW4` | 8998 → 18997 | 2 |
| Flat, rough, and slopes (release) | `G1-Mix3-Deploy-NoCmdCurriculum-SymLoss-YawW4` | 18997 → **28996** | 2 |
| Adds 0–5 cm stairs | `G1-Mix4-Stairs5-Deploy-NoCmdCurriculum-SymLoss-YawW4` | 28996 → 30550 | 2 |
| 0–10 cm stairs | `G1-Mix4-Stairs10-Deploy-NoCmdCurriculum-SymLoss-YawW4` | 30550 → 32549 (32000 kept) | 2 |
| 7.5–10 cm stairs | `G1-Mix4-Stairs7p5to10-Deploy-NoCmdCurriculum-SymLoss-YawW4` | 32000 → 33499 | 3 |
| Linear-velocity tracking weight 2.5 | `G1-Mix4-Stairs7p5to10-TrackLinW2p5-Deploy-SymLoss-YawW4` | 33499 → 34098 (**33800** selected) | 3 |

Before each transfer to a new terrain mix, the observation normalizer of
the starting checkpoint was adjusted. Its height-scan statistics were reset,
with a reduced pseudo-count, because the scan variance learned on flat
ground (σ ≈ 0.013) would otherwise amplify terrain scans by a factor of
40–50. The tool that performed this step is not included here. All tasks
above are registered by this repository and start inside the image. Training
the chain end to end and reproducing either checkpoint has not been
verified.

Mirror loss uses the observation and action mirror map in
[`src/g1_locomotion/mirror.py`](../src/g1_locomotion/mirror.py). It covers
left-right joint swaps with sign flips, the mirrored command, and the
height-scan grid reflected across the robot's sagittal plane (y → −y).
