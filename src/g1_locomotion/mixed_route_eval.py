#!/usr/bin/env python3
"""Evaluate one policy on flat, rough, slope, fixed 10 cm stairs, and flat."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import g1_locomotion.tasks  # noqa: E402,F401
from g1_locomotion import harness  # noqa: E402
from g1_locomotion import route_eval as route  # noqa: E402
from g1_locomotion.tasks import stairs10 as stairs10_tasks  # noqa: E402
from g1_locomotion.stairs_eval import EXPECTED_MJLAB_COMMIT, mjlab_commit  # noqa: E402

stairs10_tasks.register()
route.EXPECTED_LAYOUTS[stairs10_tasks.MIXED_ROUTE_10] = [
  "flat", "random_rough", "hf_pyramid_slope", "pyramid_stairs", "flat_landing"
]
route.SECTION_LABELS[stairs10_tasks.MIXED_ROUTE_10] = [
  "flat", "random_rough", "slope", "stairs"
]


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("checkpoint", type=Path)
  parser.add_argument("--row", type=int, default=4)
  parser.add_argument("--num-envs", type=int, default=20)
  parser.add_argument("--duration", type=float, default=60.0)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
  parser.add_argument("--out", type=Path, required=True)
  args = parser.parse_args()

  version = importlib.metadata.version("mjlab")
  if version != route.EXPECTED_MJLAB_VERSION:
    parser.error(f"expected mjlab {route.EXPECTED_MJLAB_VERSION}, got {version}")
  commit = mjlab_commit()
  if commit != EXPECTED_MJLAB_COMMIT:
    parser.error(f"expected mjlab commit {EXPECTED_MJLAB_COMMIT}, got {commit}")

  torch.manual_seed(args.seed)
  np.random.seed(args.seed)
  device = ("cuda:0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
  task = stairs10_tasks.MIXED_ROUTE_10
  env, policy, row = harness.build(
    task, str(args.checkpoint), args.num_envs, args.row, device, seed=args.seed
  )
  layout = route.route_layout(env.unwrapped, row, task)
  stair_cfg = env.unwrapped.scene.terrain.cfg.terrain_generator.sub_terrains[
    "pyramid_stairs"
  ]
  if tuple(stair_cfg.step_height_range) != (0.10, 0.10):
    raise RuntimeError("mixed route must contain fixed 10 cm steps")
  start = route.configure_start(env.unwrapped, layout, 1.0, 0.15, 0.05)
  started_at = dt.datetime.now().astimezone().isoformat(timespec="seconds")
  result = route.evaluate(
    env, policy, row, layout, (0.5, 0.0, 0.0), args.duration,
    "cross_track", 4.0, 1.5, 0.5, 0.0,
  )
  result.update({
    "kind": "mixed_10cm_route_result",
    "task": task,
    "checkpoint": str(args.checkpoint.resolve()),
    "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
    "mjlab_version": version,
    "mjlab_commit": commit,
    "python_executable": sys.executable,
    "device": device,
    "seed": args.seed,
    "start": start,
    "layout": layout,
    "step_height_m": 0.10,
    "started": started_at,
    "finished": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
  })
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2))
  print(f"passed={result['passed']}/{args.num_envs} fell={result['fell']} "
        f"left_row={result['left_row']} timed_out={result['not_finished']}")


if __name__ == "__main__":
  main()
