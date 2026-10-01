"""Render an mjlab policy rollout video under given commands (CPU-only).

Why a separate file instead of mjlab/scripts/play.py:
mjlab 1.5.3's play.py can write video (--video), but the VideoRecorder's
recording trigger is per-step/episode with a fixed frame count, while the
command goes through the standard command_manager (resampled on a timer and
not held constant across forward/turn/brake segments). We need a SEQUENCE of
different constant commands over given intervals, synchronized with frame
rendering — closer to harness.py's rollout(), which does the same thing but
writes metrics instead of frames. That logic is reused here via import
(build/_fix_command), not copied.

Rendering is kept on CPU so it does not compete with GPU training jobs that
may run concurrently on the same machine. Hence:
  - CUDA_VISIBLE_DEVICES="" — policy and physics (torch) run on CPU;
  - MUJOCO_GL=osmesa — offscreen MuJoCo rendering via the Mesa software
    rasterizer (OSMesa), without touching the GPU/EGL. Verified separately:
    the same process with MUJOCO_GL=egl uses libEGL_nvidia.so (a hardware
    path on the GPU), which must be avoided here. osmesa renders the same
    frames without touching the nvidia driver (confirmed manually: rendering
    works, and `nvidia-smi` shows no associated process during rendering).
  Both variables must be set BEFORE importing mujoco/mjlab (otherwise the GL
  backend and CUDA context are already fixed), so this is done at the very
  top of the file, before the other imports.

The policy/environment is built via build() from harness.py — the same path
as in the harness (load_env_cfg(play=True); deterministic terrain is not
needed for flat; the command is fixed via _fix_command the same way as in
rollout()).
"""

import os
from pathlib import Path as _Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MUJOCO_GL", "osmesa")

# libOSMesa.so is not installed via system apt (no root required this way),
# so a locally unpacked .deb (libosmesa6) is used; its .so path is added to
# LD_LIBRARY_PATH BEFORE importing mujoco/OpenGL, otherwise ctypes can't find
# the library and GL_context.py fails with 'NoneType' object has no
# attribute 'glGetError'. Override via the OSMESA_LIB_DIR env var if the
# library is available somewhere else.
_osmesa_dir = os.environ.get(
  "OSMESA_LIB_DIR",
  os.path.join(
    os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
    "g1-locomotion", "osmesa", "usr", "lib", "x86_64-linux-gnu",
  ),
)
if _Path(_osmesa_dir).is_dir():
  os.environ["LD_LIBRARY_PATH"] = (
    _osmesa_dir + ":" + os.environ.get("LD_LIBRARY_PATH", "")
  )

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from g1_locomotion.harness import build, _fix_command  # noqa: E402


def parse_commands(spec: str) -> list[tuple[float, float, float, float]]:
  """"vx,vy,wz:sec;vx,vy,wz:sec;..." -> [(vx, vy, wz, sec), ...]."""
  segments = []
  for chunk in spec.split(";"):
    chunk = chunk.strip()
    if not chunk:
      continue
    cmd_part, sec_part = chunk.split(":")
    vx, vy, wz = (float(x) for x in cmd_part.split(","))
    segments.append((vx, vy, wz, float(sec_part)))
  return segments


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--checkpoint", required=True)
  ap.add_argument("--task", default="Mjlab-Velocity-Flat-Unitree-G1")
  ap.add_argument("--commands", required=True,
                  help='"vx,vy,wz:sec;vx,vy,wz:sec;..."')
  ap.add_argument("--out", required=True)
  ap.add_argument("--fps", type=int, default=30)
  ap.add_argument("--width", type=int, default=640)
  ap.add_argument("--height", type=int, default=480)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--command-name", default="twist")
  ap.add_argument("--frame-png", default=None,
                  help="save one frame (last of the segment) as a PNG here")
  args = ap.parse_args()

  segments = parse_commands(args.commands)
  if not segments:
    raise ValueError("empty command list")

  torch.manual_seed(args.seed)
  np.random.seed(args.seed)
  device = "cpu"

  print(f"[render] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r} "
        f"MUJOCO_GL={os.environ.get('MUJOCO_GL')!r} device={device}")

  env, policy, _ = build(args.task, args.checkpoint, num_envs=1, level=0,
                         device=device, seed=args.seed,
                         command_name=args.command_name)

  base = env.unwrapped
  base.cfg.viewer.width = args.width
  base.cfg.viewer.height = args.height
  base.render_mode = "rgb_array"
  from mjlab.viewer.offscreen_renderer import OffscreenRenderer
  renderer = OffscreenRenderer(
    model=base.sim.mj_model,
    cfg=base.cfg.viewer,
    scene=base.scene,
    sim_model=base.sim.model,
    expanded_fields=base.sim.expanded_fields,
  )
  renderer.initialize()
  base._offline_renderer = renderer

  term = base.command_manager._terms[args.command_name]

  vx0, vy0, wz0 = segments[0][:3]
  _fix_command(term, (vx0, vy0, wz0))
  obs, _ = env.reset()
  _fix_command(term, (vx0, vy0, wz0))

  frames = []
  step_dt = base.step_dt
  frame_interval = 1.0 / args.fps
  t_render_acc = 0.0
  last_frame = None

  for seg_idx, (vx, vy, wz, sec) in enumerate(segments):
    _fix_command(term, (vx, vy, wz))
    n_steps = max(1, round(sec / step_dt))
    print(f"[render] segment {seg_idx}: cmd=({vx:.2f},{vy:.2f},{wz:.2f}) "
          f"{sec:.1f}s -> {n_steps} steps")
    for _ in range(n_steps):
      _fix_command(term, (vx, vy, wz))
      with torch.no_grad():
        actions = policy(obs)
        obs, _, dones, _ = env.step(actions)
      t_render_acc += step_dt
      while t_render_acc >= frame_interval:
        frame = base.render()
        frames.append(frame[0] if frame.ndim == 4 else frame)
        last_frame = frames[-1]
        t_render_acc -= frame_interval
      if bool(dones[0]):
        # The robot fell / the episode ended earlier than planned — stop
        # recording honestly instead of substituting a reset frame: the
        # video must show EXACTLY the requested scenario, not a mid-restart.
        print(f"[render] WARNING: episode ended (done) during segment "
              f"{seg_idx}, stopping early")
        break
    else:
      continue
    break

  renderer.close()

  if not frames:
    raise RuntimeError("no frames were captured — rendering failed")

  out_path = Path(args.out)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  import mediapy as media
  media.write_video(str(out_path), frames, fps=args.fps)
  duration = len(frames) / args.fps
  print(f"[render] saved: {out_path} ({len(frames)} frames, "
        f"{duration:.2f} s @ {args.fps} fps, {args.width}x{args.height})")

  if args.frame_png:
    png_path = Path(args.frame_png)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    media.write_image(str(png_path), last_frame)
    print(f"[render] saved check frame: {png_path}")


if __name__ == "__main__":
  main()
