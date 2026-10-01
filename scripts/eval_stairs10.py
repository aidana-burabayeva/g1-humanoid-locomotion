#!/usr/bin/env python3
"""Run the 10 cm stair acceptance protocol in the pinned simulation image."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
MODEL = Path("/model/model_33800.pt")
OUTPUT = Path("/outputs")
MODEL_SHA256 = "ea7a1f2b3c083e87388f6c3391598d7efbb2096e574395fafa95b8d667068269"
STAIRS = ROOT / "src" / "g1_locomotion" / "stairs_eval.py"
MIXED = ROOT / "src" / "g1_locomotion" / "mixed_route_eval.py"
ORIGINAL = ROOT / "src" / "g1_locomotion" / "route_eval.py"
FLAT = ROOT / "scripts" / "eval_flat_protocol.py"


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def run(script: Path, args: list[str], name: str, output: Path) -> None:
  command = [sys.executable, str(script), str(MODEL), *args]
  with (output / f"{name}.log").open("w") as log:
    subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                   check=True)
  print(f"[stairs-container] {name} complete", flush=True)


def route_result(output: Path, name: str, seed: int, expected_task: str,
                 expected_row: int) -> dict:
  result = json.loads((output / f"{name}.json").read_text())
  if (result["task"] != expected_task or result["seed"] != seed or
      result["n_envs"] != 20 or result["layout"]["terrain_row"] != expected_row or
      result["checkpoint"] != str(MODEL.resolve()) or
      result["duration_s"] != 60.0):
    raise ValueError(f"invalid metadata: {name}")
  if result.get("checkpoint_sha256", MODEL_SHA256) != MODEL_SHA256:
    raise ValueError(f"checkpoint hash mismatch in {name}")
  return result


def route_gate(results: list[dict]) -> bool:
  passed = [result["passed"] for result in results]
  return (all(count >= 17 for count in passed) and
          max(passed) - min(passed) <= 3 and
          all(math.isfinite(result["overall_mae"]["vx"]) and
              result["overall_mae"]["vx"] <= 0.15 for result in results) and
          all(seam["fell_within_1s_of_crossing"] <= 2
              for result in results for seam in result["seam_metrics"].values()))


def compact(result: dict) -> dict:
  return {**{key: result[key] for key in
             ("passed", "fell", "left_row", "not_finished")},
          "mae_vx_m_s": result["overall_mae"]["vx"]}


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
  if not all(torch.isfinite(value).all() for value in actor.values()
             if isinstance(value, torch.Tensor)):
    raise ValueError("non-finite actor weights")
  if not torch.cuda.is_available():
    raise RuntimeError("GPU unavailable in container")

  output = OUTPUT / f"acceptance-{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}"
  output.mkdir(parents=True, exist_ok=False)
  verdict = {
    "kind": "stairs10_container_acceptance", "model_sha256": MODEL_SHA256,
    "image_id": os.environ.get("G1_IMAGE_ID"),
    "source_sha256": manifest["source_sha256"], "dependencies": dependencies,
    "experiment_sources": {path.relative_to(ROOT).as_posix(): sha256(path) for path in
                           (Path(__file__), STAIRS, MIXED,
                            ROOT / "src" / "g1_locomotion" / "tasks" / "stairs10.py",
                            FLAT)},
    "python_executable": sys.executable, "results": {}, "checks": {},
  }
  try:
    for prefix, script, task, row in (
      ("regular10", STAIRS, "G1-Stairs10-FullCrossing-Deploy", 9),
      ("inverted10", STAIRS, "G1-StairsInv10-FullCrossing-Deploy", 9),
      ("original", ORIGINAL,
       "G1-FinalRoute-Stairs5-Deploy-NoCmdCurriculum-SymLoss-YawW4", 4),
      ("mixed10", MIXED, "G1-FinalRoute-StairsFixed10-Deploy", 4),
    ):
      series = []
      for seed in range(3):
        name = f"{prefix}_seed{seed}"
        args = ["--row", str(row), "--num-envs", "20", "--duration", "60",
                "--seed", str(seed), "--out", str(output / f"{name}.json")]
        if prefix in ("regular10", "inverted10"):
          args.extend(["--task", task])
        elif prefix == "original":
          args.extend(["--task", task, "--steer", "cross_track"])
        run(script, args, name, output)
        series.append(route_result(output, name, seed, task, row))
      verdict["results"][prefix] = [compact(result) for result in series]
      verdict["checks"][prefix] = (
        all(result["passed"] >= 17 for result in series)
        if prefix in ("regular10", "inverted10") else route_gate(series)
      )

    flat_dir = output / "flat"
    run(FLAT, ["--task", "G1-Flat-Deploy-NoCmdCurriculum-SymLoss",
               "--num-envs", "20", "--duration", "60", "--seed", "0",
               "--out", str(flat_dir), "--scenarios", "stand", "fwd_05",
               "fwd_10", "yaw_pos_05", "yaw_neg_05", "fwd_yaw_05",
               "left_04", "right_04", "brake"], "flat_protocol", output)
    flat_meta = json.loads((flat_dir / "stage1_meta.json").read_text())
    verdict["results"]["flat"] = flat_meta["overall"]
    verdict["checks"]["flat"] = (flat_meta["overall"] == "PASS" and
                                  not flat_meta["plan_required_missing"])

    for prefix, script, task, row in (
      ("regular10", STAIRS, "G1-Stairs10-FullCrossing-Deploy", 9),
      ("mixed10", MIXED, "G1-FinalRoute-StairsFixed10-Deploy", 4),
    ):
      first = verdict["results"][prefix][0]["passed"]
      if first <= 17:
        name = f"{prefix}_seed0_repeat"
        args = ["--row", str(row), "--num-envs", "20", "--duration", "60",
                "--seed", "0", "--out", str(output / f"{name}.json")]
        if prefix == "regular10":
          args.extend(["--task", task])
        run(script, args, name, output)
        repeated = route_result(output, name, 0, task, row)
        verdict["results"][name] = compact(repeated)
        verdict["checks"][name] = repeated["passed"] >= 17

    verdict["overall"] = "PASS" if all(verdict["checks"].values()) else "FAIL"
  except Exception as exc:
    verdict["overall"] = "ERROR"
    verdict["error"] = f"{type(exc).__name__}: {exc}"
  (output / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
  print(f"[stairs-container] {verdict['overall']}: {output}", flush=True)
  return 0 if verdict["overall"] == "PASS" else 1


if __name__ == "__main__":
  raise SystemExit(main())
