#!/usr/bin/env python3
"""Evaluate the flat->stairs or slope->stairs seam with one policy.

The strip tasks have four columns of the first terrain followed by four
columns of the same 5 cm stair preset. The robot starts on the last column
before the seam. This reuses the tested transition rollout and reset logic
(src/g1_locomotion/transition_eval.py).
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from g1_locomotion import transition_eval as transition  # noqa: E402

TASKS = {
  "flat": "G1-FlatStairs5Strip-Deploy-NoCmdCurriculum-SymLoss-YawW4",
  "slope": "G1-SlopeStairs5Strip-Deploy-NoCmdCurriculum-SymLoss-YawW4",
}
EXPECTED_MJLAB_VERSION = "1.6.0"


def layout(base, first: str) -> dict:
  terrain = base.scene.terrain
  tg = terrain.cfg.terrain_generator
  keys = list(tg.sub_terrains)
  expected_first = "flat" if first == "flat" else "hf_pyramid_slope"
  if (len(keys) != 8 or
      any(not key.startswith(expected_first + "_") for key in keys[:4]) or
      any(not key.startswith("pyramid_stairs_") for key in keys[4:])):
    raise RuntimeError(f"unexpected strip layout: {keys}")
  origins = terrain.terrain_origins
  if not tg.curriculum or origins.shape[1] != 8:
    raise RuntimeError("expected one ordered column per sub-terrain")
  centers = origins[0, :, 1].tolist()
  width = float(tg.size[1])
  if any(abs(centers[i + 1] - centers[i] - width) > 1e-4 for i in range(7)):
    raise RuntimeError(f"non-contiguous columns: {centers}")
  return {
    "keys": keys, "first_type": first, "second_type": "pyramid_stairs",
    "boundary_y": (centers[3] + centers[4]) / 2,
    "tile_size": list(tg.size), "centers_y": centers,
    "num_rows": origins.shape[0], "border_width": float(tg.border_width),
  }


def set_start(base, lay: dict, row: int, offset: float,
              xy_jitter: tuple[float, float], yaw_jitter: float) -> dict:
  width = lay["tile_size"][1]
  if not 0 < offset - xy_jitter[1] and offset + xy_jitter[1] < width:
    raise ValueError("start must remain in the last pre-seam tile")
  col = 3
  start_y = lay["boundary_y"] - offset
  dy = start_y - lay["centers_y"][col]
  jx, jy = xy_jitter
  pose = dict(base.event_manager.get_term_cfg("reset_base").params["pose_range"])
  pose.update({"x": (-jx, jx), "y": (dy - jy, dy + jy),
               "yaw": (math.pi / 2 - yaw_jitter, math.pi / 2 + yaw_jitter)})
  base.event_manager.get_term_cfg("reset_base").params["pose_range"] = pose
  base.scene.terrain.terrain_types[:] = col
  return {"start_col": col, "start_type": lay["first_type"],
          "target_type": lay["second_type"], "row": row, "sign_y": 1.0,
          "start_y_nominal": start_y,
          "pose_range": {key: list(value) for key, value in pose.items()}}


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument("checkpoint")
  ap.add_argument("--first", choices=tuple(TASKS), required=True)
  ap.add_argument("--rows", type=int, nargs="+", default=[0, 4, 9])
  ap.add_argument("--num-envs", type=int, default=20)
  ap.add_argument("--duration", type=float, default=60.0)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--cmd", default="0.5,0,0")
  ap.add_argument("--start-offset", type=float, default=2.5)
  ap.add_argument("--xy-jitter", type=float, nargs=2, default=[0.5, 0.25])
  ap.add_argument("--yaw-jitter", type=float, default=0.1)
  ap.add_argument("--out", required=True)
  args = ap.parse_args()

  installed_mjlab = importlib.metadata.version("mjlab")
  if installed_mjlab != EXPECTED_MJLAB_VERSION:
    ap.error(f"expected mjlab {EXPECTED_MJLAB_VERSION}, got {installed_mjlab}; "
             "use the pinned container environment")

  cmd = [float(x) for x in args.cmd.split(",")]
  if len(cmd) != 3:
    ap.error("--cmd needs vx,vy,wz")
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  task = TASKS[args.first]
  env, policy, _ = transition.H.build(
    task, args.checkpoint, args.num_envs, args.rows[0], device, seed=args.seed
  )
  base = env.unwrapped
  lay = layout(base, args.first)
  tg = base.scene.terrain.cfg.terrain_generator
  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  meta = {"kind": "run_meta", "protocol": "stage4_transition_eval",
          "mjlab_version": installed_mjlab,
          "python_executable": sys.executable,
          "task": task, "checkpoint": str(Path(args.checkpoint).resolve()),
          "first_type": args.first, "second_type": "pyramid_stairs",
          "rows": args.rows, "command": cmd, "num_envs": args.num_envs,
          "duration_s": args.duration, "seed": args.seed, "device": device,
          "layout": lay, "started": dt.datetime.now().astimezone().isoformat()}
  for row in args.rows:
    start = set_start(base, lay, row, args.start_offset,
                      tuple(args.xy_jitter), args.yaw_jitter)
    result = transition.run_episode_batch(
      env, policy, cmd, args.duration, row, lay, start
    )
    result.update({"kind": "transition_result", "task": task,
                   "checkpoint": meta["checkpoint"], "terrain_row": row,
                   "difficulty": transition.H.difficulty_of_row(
                     tg.num_rows, row, tg.difficulty_range
                   ), "start": start, "seed": args.seed})
    (out / f"row{row}.json").write_text(json.dumps(result, indent=2))
    print(f"[stage4-seam] {args.first}->stairs row{row}: "
          f"crossed={result['crossed']}/{args.num_envs}, "
          f"passed_no_fall={result['passed_no_fall']}/{args.num_envs}, "
          f"fell_after={result['fell_after_cross']}, "
          f"left_row={result['left_row']}")
  meta["finished"] = dt.datetime.now().astimezone().isoformat()
  (out / "run_meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
  main()
