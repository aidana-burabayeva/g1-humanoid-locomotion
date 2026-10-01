# Policy interface

The contract applies to `model_28996.pt` in the mjlab task
`G1-FinalRoute-Stairs5-Deploy-NoCmdCurriculum-SymLoss-YawW4`. The checkpoint
hash, task, dependency versions, and acceptance configuration are recorded in
[`weights/manifest.json`](../weights/manifest.json). This is a simulation
interface, not a G1 hardware controller interface.

## Observations

The actor consumes 283 values in mjlab observation-term order:

| Term | Values |
| --- | ---: |
| `base_ang_vel` | 3 |
| `projected_gravity` | 3 |
| `joint_pos` | 29 |
| `joint_vel` | 29 |
| `actions` | 29 |
| `command` | 3 |
| `height_scan` | 187 |

The first 96 values contain proprioception, the previous action, and the
velocity command. The remaining 187 values form the local height scan.
`base_lin_vel` is not an actor input. The training critic consumes 298 values
and is not used during evaluation.

## Actions and compatibility

The actor produces 29 outputs. mjlab applies them through
`JointPositionActionCfg`, with joint ordering, per-joint scaling, and default
position offsets defined by the pinned mjlab G1 configuration. The actor's
observation-normalizer statistics are stored in the checkpoint.

Changing joint order, observation order, height-scan geometry, normalization,
action scale, or offsets changes the policy interface. The pinned mjlab commit
and project source hash identify the implementation used for this evaluation.
[`scripts/eval_route.py`](../scripts/eval_route.py) rejects a checkpoint or
environment that fails its hash, version, or dimension checks.

These checks do not establish numerical equivalence with another simulator
or a physical robot. Such a runtime must independently verify the complete
observation and action mapping against mjlab before using this policy.

## ONNX export

[`weights/model_28996.onnx`](../weights/model_28996.onnx) was exported by
mjlab when the release checkpoint was saved;
[`weights/model_33800.onnx`](../weights/model_33800.onnx) was exported with
[`scripts/export_onnx.py`](../scripts/export_onnx.py), which uses the same
mjlab exporter. Both take one input, `obs` with shape `[1, 283]`, and return
`actions` with shape `[1, 29]` (opset 18). The actor's observation
normalizer is part of the graph, so the input is the raw, unnormalized
observation vector. mjlab attaches metadata to the model: joint names,
observation term names, and command names, plus action scale, default joint
positions, and PD gains rounded to three decimals.

The exact layout, including term offsets, joint order, unrounded action
scales, and default positions, is in
[`src/g1_locomotion/sim2sim/obs_spec.yaml`](../src/g1_locomotion/sim2sim/obs_spec.yaml)
and [`deploy_cfg.json`](../src/g1_locomotion/sim2sim/deploy_cfg.json). The
latter was generated from the built mjlab environment by
[`scripts/sim2sim_reference.py`](../scripts/sim2sim_reference.py). The
`onnx-parity` service checks the ONNX model against the torch actor and
against `obs_spec.yaml`. The `sim2sim` service builds the same 283 inputs
from raw MuJoCo state in Unitree's G1 model. In that model the height scan
over a flat plane is the pelvis height above the floor for all 187 points.
