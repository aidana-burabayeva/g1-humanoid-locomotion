#!/usr/bin/env python3
"""Evaluate a complete flat-to-stairs-to-flat crossing at a fixed row."""

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

EXPECTED_MJLAB_COMMIT = "c2e1e06400e309b6897a536693fe5d2aa772c5b2"


def mjlab_commit() -> str:
  direct_url = importlib.metadata.distribution("mjlab").read_text("direct_url.json")
  if direct_url is None:
    raise RuntimeError("mjlab installation has no VCS provenance")
  return json.loads(direct_url).get("vcs_info", {}).get("commit_id", "")


def layout_for(base, row: int) -> dict:
  generator = base.scene.terrain.cfg.terrain_generator
  assert generator is not None
  keys = list(generator.sub_terrains)
  if keys != ["flat", "stairs", "flat_landing"]:
    raise RuntimeError(f"unexpected terrain layout: {keys}")
  centers = base.scene.terrain.terrain_origins[row, :, 1].tolist()
  width = float(generator.size[1])
  if len(centers) != 3 or any(
    abs(centers[i + 1] - centers[i] - width) > 1e-4 for i in range(2)
  ):
    raise RuntimeError(f"noncontiguous terrain tiles: {centers}")
  return {
    "keys": keys,
    "sections": ["flat", "stairs"],
    "centers_y": centers,
    "seams_y": [(centers[i] + centers[i + 1]) / 2 for i in range(2)],
    "tile_size_m": list(generator.size),
    "terrain_row": row,
    "difficulty": harness.difficulty_of_row(
      generator.num_rows, row, generator.difficulty_range
    ),
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("checkpoint", type=Path)
  parser.add_argument("--task", choices=(
    stairs10_tasks.REGULAR_ROUTE,
    stairs10_tasks.INVERTED_ROUTE,
    stairs10_tasks.REGULAR_ROUTE_5,
    stairs10_tasks.INVERTED_ROUTE_5,
  ), required=True)
  parser.add_argument("--row", type=int, default=9)
  parser.add_argument("--num-envs", type=int, default=20)
  parser.add_argument("--duration", type=float, default=60.0)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
  parser.add_argument("--out", type=Path, required=True)
  args = parser.parse_args()
  stairs10_tasks.register()

  version = importlib.metadata.version("mjlab")
  if version != route.EXPECTED_MJLAB_VERSION:
    parser.error(f"expected mjlab {route.EXPECTED_MJLAB_VERSION}, got {version}")
  commit = mjlab_commit()
  if commit != EXPECTED_MJLAB_COMMIT:
    parser.error(f"expected mjlab commit {EXPECTED_MJLAB_COMMIT}, got {commit}")
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)
  device = ("cuda:0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
  env, policy, row = harness.build(
    args.task, str(args.checkpoint), args.num_envs, args.row, device, seed=args.seed
  )
  layout = layout_for(env.unwrapped, row)
  stair_cfg = env.unwrapped.scene.terrain.cfg.terrain_generator.sub_terrains["stairs"]
  low, high = stair_cfg.step_height_range
  start = route.configure_start(env.unwrapped, layout, 1.0, 0.15, 0.05)
  started_at = dt.datetime.now().astimezone().isoformat(timespec="seconds")
  result = route.evaluate(
    env, policy, row, layout, (0.5, 0.0, 0.0), args.duration,
    "cross_track", 4.0, 1.5, 0.5, 0.0,
  )
  result.update({
    "kind": "stair_full_crossing_result",
    "task": args.task,
    "checkpoint": str(args.checkpoint.resolve()),
    "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
    "mjlab_version": version,
    "mjlab_commit": commit,
    "python_executable": sys.executable,
    "device": device,
    "seed": args.seed,
    "start": start,
    "layout": layout,
    "step_height_m": low + layout["difficulty"] * (high - low),
    "started": started_at,
    "finished": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
  })
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2))
  print(f"passed={result['passed']}/{args.num_envs} fell={result['fell']} "
        f"left_row={result['left_row']} timed_out={result['not_finished']}")


if __name__ == "__main__":
  main()
