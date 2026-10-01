# Container usage

All tools run from one image defined in [`docker/Dockerfile`](../docker/Dockerfile).
The Compose services in [`compose.yaml`](../compose.yaml) share that image
and differ only in their default command.

## Host prerequisites

- Linux x86-64, an NVIDIA GPU, and a compatible driver.
- Docker Engine with Compose and the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).
  Compose reserves one GPU through Docker's
  [GPU device reservation](https://docs.docker.com/compose/how-tos/gpu-support/).
- About 11 GB of free disk space for the image, plus space for outputs.

The host does not need Python, mjlab, or the weights outside this repository.

## Mounts and outputs

| Container path | Host path | Mode |
| --- | --- | --- |
| `/app` | Baked into the image: `src/`, `scripts/`, `tests/`, `third_party/`, `weights/manifest.json`, `uv.lock` | read-only root filesystem |
| `/model` | `./weights` | read-only |
| `/outputs` | `./outputs` (ignored by Git) | read-write |
| `/tmp` | tmpfs | read-write, discarded on exit |

Containers run as `${HOST_UID:-1000}:${HOST_GID:-1000}`. If your user ID is
not 1000, export `HOST_UID` and `HOST_GID` before running Compose so that
`outputs/` stays writable.

Every service uses `python` as its entrypoint. `docker compose run --rm
<service>` runs the default command. `docker compose run --rm <service>
<script> <args...>` runs any script in the image with the same mounts.

## Services

```bash
docker compose build eval-route
docker compose run --rm eval-route                      # release acceptance
docker compose run --rm eval-flat                       # flat command protocol
docker compose run --rm eval-stairs10                   # 10 cm stair protocol, seeds 0-2
docker compose run --rm eval-stairs10 /app/scripts/eval_stairs10_heldout.py   # seeds 3-5
docker compose run --rm eval-transitions                # flat -> stairs seam
docker compose run --rm render                          # HUD video
docker compose run --rm onnx-parity                     # torch vs ONNX Runtime
docker compose run --rm sim2sim                         # unitree_mujoco, 60 s
docker compose --profile train run --rm train           # training
```

| Service | Writes | Notes |
| --- | --- | --- |
| `eval-route` | `outputs/acceptance-<UTC>/seed{0,1,2}.{json,log}`, `verdict.json` | Exits nonzero if an acceptance check fails. `--preflight-only` checks hashes and versions only. |
| `eval-flat` | `outputs/flat_protocol_28996/<scenario>.json`, `stage1_meta.json` | Prints a per-command table. The overall result is `PASS` only if every required scenario is present and passes. |
| `eval-stairs10` | `outputs/acceptance-<UTC>/` | Regular and inverted 10 cm stairs, the original route, the fixed-10 cm route, and the flat protocol for `model_33800`. Runs for about an hour. |
| `eval-transitions` | `outputs/transitions_flat_28996.json` | Use `--first slope` for the slope-to-stairs seam. |
| `render` | `outputs/video/*.mp4` and a control-value JSON | Modes: `commands`, `route`, `stairs10`, `compare`. CPU physics, EGL rendering, ffmpeg/libx264 encoding. |
| `onnx-parity` | `outputs/onnx_parity_28996.json` | Runs on CPU. Also checks the actor input layout against [`obs_spec.yaml`](../src/g1_locomotion/sim2sim/obs_spec.yaml). |
| `sim2sim` | `outputs/sim2sim_28996_flat.json` | Plain MuJoCo on CPU with ONNX Runtime; no mjlab or torch in the control loop. |
| `train` | `outputs/train/g1_velocity/<timestamp>/` | TensorBoard logging. Defaults to the 10 cm stair fine-tune task with 512 environments. |

## Smoke tests

These commands check startup and output generation in about a minute each.
They do not measure performance.

```bash
# Release acceptance: 2 robots x 2 s, seed 0
docker compose run --rm eval-route /app/scripts/eval_route.py --smoke

# Flat protocol: three short scenarios
docker compose run --rm eval-flat /app/scripts/eval_flat_protocol.py /model/model_28996.pt \
  --task G1-Flat-Deploy-NoCmdCurriculum-SymLoss --num-envs 2 --duration 2 \
  --brake-hold 1 --brake-after 1 --out /outputs/smoke_flat --scenarios stand fwd_05 brake

# 10 cm stairs (model_33800): one full-crossing evaluator and the fixed-10 cm route evaluator
docker compose run --rm eval-stairs10 /app/src/g1_locomotion/stairs_eval.py /model/model_33800.pt \
  --task G1-StairsInv10-FullCrossing-Deploy --row 9 --num-envs 2 --duration 3 --out /outputs/smoke_stairs10.json
docker compose run --rm eval-stairs10 /app/src/g1_locomotion/mixed_route_eval.py /model/model_33800.pt \
  --row 4 --num-envs 2 --duration 3 --out /outputs/smoke_mixed10.json

# Terrain seams
docker compose run --rm eval-transitions /app/scripts/eval_transitions.py /model/model_28996.pt \
  --first flat --rows 4 --num-envs 2 --duration 3 --out /outputs/smoke_transitions.json
docker compose run --rm eval-transitions /app/src/g1_locomotion/transition_eval.py /model/model_28996.pt \
  --rows 4 --directions f2r --num-envs 2 --duration 3 --out /outputs/smoke_transition_rough

# Video: 2.5 s clip written to the container's /tmp (discarded on exit)
docker compose run --rm --entrypoint bash render -c \
  'python /app/scripts/render_video.py commands --checkpoint /model/model_28996.pt --hud --plots \
     --sequence "fwd 0.5:0.5,0,0:2.5" --hold-end 0 --out /tmp/smoke.mp4 --log /tmp/smoke.json \
   && ffmpeg -hide_banner -i /tmp/smoke.mp4 2>&1 | grep Duration'

# Training: three PPO iterations with 64 environments
docker compose --profile train run --rm train /app/scripts/train.py \
  G1-Mix4-Stairs7p5to10-TrackLinW2p5-Deploy-SymLoss-YawW4 \
  --env.scene.num-envs 64 --agent.max-iterations 3 --agent.logger tensorboard --log-root /outputs/train_smoke

# Unit test: observation history under partial resets
docker compose run --rm eval-route /app/tests/test_partial_reset.py
```

## Other tools

| Command | Purpose |
| --- | --- |
| `python /app/src/g1_locomotion/harness.py CKPT --task T --difficulty 0.44 --tile-edge stop --commands 0.5,0,0 --out DIR` | Isolated-surface evaluation with fixed commands |
| `python /app/scripts/openloop_drift.py CKPT --out DIR` | Lateral and heading drift under a constant forward command without steering |
| `python /app/scripts/export_onnx.py CKPT --out /outputs/policy.onnx` | ONNX export with mjlab's exporter and metadata |
| `python /app/scripts/sim2sim_reference.py --checkpoint CKPT --onnx ONNX --out DIR` | Writes `deploy_cfg.json` (joint order, gains, scales from the built mjlab environment) and checks the sim-to-sim observation builder against mjlab's observation manager |

## Image provenance

[`pyproject.toml`](../pyproject.toml) and [`uv.lock`](../uv.lock) pin the
Python environment (mjlab at commit `c2e1e06`, MuJoCo 3.11.0, MuJoCo Warp
3.11.0, torch 2.14.0, rsl-rl-lib 5.5.1). The Dockerfile pins the CUDA base
image and the uv image by digest and installs the environment with
`uv sync --locked`, following the
[uv Docker guide](https://docs.astral.sh/uv/guides/integration/docker/).
On top of the locked environment, the image adds:

- `onnxruntime==1.30.0` and its dependency `flatbuffers==25.12.19`, used only
  by the ONNX parity and sim-to-sim tools;
- an `ffmpeg` link to the static binary that ships with the locked
  `imageio-ffmpeg` package (the build checks for libx264);
- `fonts-dejavu-core` for the video overlay.

Ubuntu packages from `apt` are not version-pinned, so a rebuild may produce
a different image. Publish the image by registry digest if an exact image is
required.

[`weights/manifest.json`](../weights/manifest.json) records the release
checkpoint hash, iteration, network dimensions, dependency versions, the
mjlab commit, the `uv.lock` hash, the acceptance settings, and
`source_sha256`. That last value is a hash over the evaluation sources
(`harness.py`, `route_eval.py`, `scripts/eval_route.py`, and
`src/g1_locomotion/tasks/`). Editing any of these files makes the acceptance
runner refuse to start until the manifest is updated deliberately.

The result files in [`results/`](results/) for 2026-09-28 and 2026-10-01 come
from an earlier image built from the same lock file and the same evaluation
logic. Their `source_sha256` values and `experiment_sources` paths refer to
the source layout used at that time. The `release_route_row4_rerun` and
`flat_protocol_28996_rerun` directories come from this repository's image.

## Troubleshooting

- `GPU unavailable inside container`: check `docker run --rm --gpus all
  nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi` and the NVIDIA Container
  Toolkit configuration.
- `runtime source SHA-256 mismatch`: the evaluation sources differ from the
  manifest. Rebuild the image from an unmodified checkout.
- `Permission denied` under `/outputs`: set `HOST_UID`/`HOST_GID` to your IDs.
- Training needs a W&B login unless `--agent.logger tensorboard` is passed;
  the `train` service passes it by default.
