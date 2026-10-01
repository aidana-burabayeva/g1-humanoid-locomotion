#!/usr/bin/env python3
"""Evaluate the 10 cm candidate on training-seen and held-out seeds."""

from __future__ import annotations

import datetime as dt
import json
import os
import sys
from pathlib import Path

import torch

from eval_stairs10 import (
  MIXED,
  MODEL,
  MODEL_SHA256,
  OUTPUT,
  ROOT,
  STAIRS,
  compact,
  route_gate,
  route_result,
  run,
  sha256,
)

SEEDS = (3, 4, 5)
HELD_OUT = (4, 5)


def main() -> int:
  from eval_route import check_environment

  manifest = json.loads((ROOT / "weights" / "manifest.json").read_text())
  dependencies = check_environment(manifest)
  if sha256(MODEL) != MODEL_SHA256:
    raise ValueError("checkpoint SHA-256 mismatch")
  checkpoint = torch.load(MODEL, map_location="cpu", weights_only=True)
  actor = checkpoint["actor_state_dict"]
  if (checkpoint["iter"] != 33800 or actor["mlp.0.weight"].shape[1] != 283 or
      actor["mlp.6.weight"].shape[0] != 29):
    raise ValueError("checkpoint contract mismatch")
  if not torch.cuda.is_available():
    raise RuntimeError("GPU unavailable in container")

  output = OUTPUT / f"heldout-{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}"
  output.mkdir(parents=True, exist_ok=False)
  verdict = {
    "kind": "stairs10_candidate_heldout", "model_sha256": MODEL_SHA256,
    "image_id": os.environ.get("G1_IMAGE_ID"),
    "mjlab_commit": manifest["mjlab_commit"], "dependencies": dependencies,
    "seed_roles": {"3": "training_seed", "4": "held_out", "5": "held_out"},
    "source_sha256": manifest["source_sha256"],
    "experiment_sources": {path.relative_to(ROOT).as_posix(): sha256(path) for path in
                           (Path(__file__), STAIRS, MIXED,
                            ROOT / "src" / "g1_locomotion" / "tasks" / "stairs10.py")},
    "results": {}, "checks": {},
  }
  try:
    for prefix, script, task, row in (
      ("regular10", STAIRS, "G1-Stairs10-FullCrossing-Deploy", 9),
      ("inverted10", STAIRS, "G1-StairsInv10-FullCrossing-Deploy", 9),
      ("mixed10", MIXED, "G1-FinalRoute-StairsFixed10-Deploy", 4),
    ):
      series = []
      for seed in SEEDS:
        name = f"{prefix}_seed{seed}"
        args = ["--row", str(row), "--num-envs", "20", "--duration", "60",
                "--seed", str(seed), "--out", str(output / f"{name}.json")]
        if script == STAIRS:
          args.extend(["--task", task])
        run(script, args, name, output)
        series.append(route_result(output, name, seed, task, row))
      verdict["results"][prefix] = {
        str(seed): compact(result) for seed, result in zip(SEEDS, series)
      }
      heldout_results = [result for seed, result in zip(SEEDS, series)
                         if seed in HELD_OUT]
      verdict["checks"][prefix] = (
        all(result["passed"] >= 17 for result in heldout_results)
        if prefix != "mixed10" else route_gate(heldout_results)
      )
      verdict["checks"][f"{prefix}_training_seed3"] = series[0]["passed"] >= 17

    verdict["overall"] = "PASS" if all(verdict["checks"].values()) else "FAIL"
  except Exception as exc:
    verdict["overall"] = "ERROR"
    verdict["error"] = f"{type(exc).__name__}: {exc}"
  (output / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
  print(f"[stairs-heldout] {verdict['overall']}: {output}", flush=True)
  return 0 if verdict["overall"] == "PASS" else 1


if __name__ == "__main__":
  raise SystemExit(main())
