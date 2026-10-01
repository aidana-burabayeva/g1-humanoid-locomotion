#!/usr/bin/env python3
"""Verify the released policy and run the frozen mjlab acceptance protocol."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "weights" / "manifest.json"


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def source_sha256() -> str:
  paths = [
    ROOT / "src" / "g1_locomotion" / "harness.py",
    ROOT / "src" / "g1_locomotion" / "route_eval.py",
    ROOT / "scripts" / "eval_route.py",
    *(ROOT / "src" / "g1_locomotion" / "tasks").rglob("*.py"),
  ]
  digest = hashlib.sha256()
  for path in sorted(paths):
    if not path.is_file():
      raise FileNotFoundError(path)
    digest.update(path.relative_to(ROOT).as_posix().encode())
    digest.update(b"\0")
    digest.update(bytes.fromhex(sha256_file(path)))
  return digest.hexdigest()


def check_checkpoint(path: Path, manifest: dict) -> None:
  if not path.is_file():
    raise FileNotFoundError(f"missing checkpoint: {path}")
  actual = sha256_file(path)
  expected = manifest["checkpoint_sha256"]
  if actual != expected:
    raise ValueError(f"checkpoint SHA-256 mismatch: {actual} != {expected}")
  checkpoint = torch.load(path, map_location="cpu", weights_only=True)
  if checkpoint.get("iter") != manifest["checkpoint_iteration"]:
    raise ValueError(f"checkpoint iteration: {checkpoint.get('iter')}")
  actor = checkpoint["actor_state_dict"]
  critic = checkpoint["critic_state_dict"]
  dimensions = manifest["dimensions"]
  shapes = {
    "actor_input": actor["mlp.0.weight"].shape[1],
    "actor_output": actor["mlp.6.weight"].shape[0],
    "actor_normalizer": actor["obs_normalizer._mean"].numel(),
    "critic_input": critic["mlp.0.weight"].shape[1],
    "critic_normalizer": critic["obs_normalizer._mean"].numel(),
  }
  for key, expected_dim in dimensions.items():
    if shapes[key] != expected_dim:
      raise ValueError(f"{key}: checkpoint {shapes[key]} != manifest {expected_dim}")
  if not all(torch.isfinite(value).all() for value in actor.values()
             if isinstance(value, torch.Tensor)):
    raise ValueError("actor checkpoint contains non-finite values")


def check_environment(manifest: dict) -> dict:
  expected = manifest["dependencies"]
  actual = {name: importlib.metadata.version(name) for name in expected}
  if actual != expected:
    raise ValueError(f"dependency mismatch: {actual} != {expected}")
  distribution = importlib.metadata.distribution("mjlab")
  direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
  commit = direct_url.get("vcs_info", {}).get("commit_id")
  if commit != manifest["mjlab_commit"]:
    raise ValueError(f"mjlab source commit: {commit} != {manifest['mjlab_commit']}")
  actual_source = source_sha256()
  if actual_source != manifest["source_sha256"]:
    raise ValueError(f"runtime source SHA-256 mismatch: {actual_source}")
  actual_lock = sha256_file(ROOT / "uv.lock")
  if actual_lock != manifest["lock_sha256"]:
    raise ValueError(f"uv.lock SHA-256 mismatch: {actual_lock}")
  return actual


def run_seed(manifest: dict, model: Path, seed: int, output: Path,
             num_envs: int, duration_s: float) -> dict:
  result_path = output / f"seed{seed}.json"
  log_path = output / f"seed{seed}.log"
  command = [
    sys.executable, str(ROOT / "src" / "g1_locomotion" / "route_eval.py"), str(model),
    "--task", manifest["task"], "--row", str(manifest["terrain_row"]),
    "--num-envs", str(num_envs), "--duration", str(duration_s),
    "--seed", str(seed), "--steer", manifest["steer"],
    "--cmd", ",".join(map(str, manifest["command"])), "--out", str(result_path),
  ]
  with log_path.open("w") as stream:
    subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                   check=True)
  result = json.loads(result_path.read_text())
  if (result["seed"] != seed or result["task"] != manifest["task"] or
      result["layout"]["terrain_row"] != manifest["terrain_row"] or
      result["n_envs"] != num_envs or
      result["checkpoint"] != str(model.resolve())):
    raise ValueError(f"result metadata mismatch for seed {seed}")
  return result


def assess(results: list[dict], manifest: dict) -> dict:
  threshold = manifest["acceptance"]
  counts = [result["passed"] for result in results]
  spread = (max(counts) - min(counts)) / manifest["num_envs"]
  checks = {
    "minimum_passes_each_seed": all(
      count >= threshold["minimum_passes"] for count in counts
    ),
    "overall_vx_mae": all(
      math.isfinite(result["overall_mae"]["vx"]) and
      result["overall_mae"]["vx"] <= threshold["maximum_vx_mae_m_s"]
      for result in results
    ),
    "success_spread": spread <= threshold["maximum_success_spread"],
    "seam_falls": all(
      seam["fell_within_1s_of_crossing"] <= threshold["maximum_seam_falls"]
      for result in results for seam in result["seam_metrics"].values()
    ),
  }
  return {"accepted": all(checks.values()), "checks": checks,
          "passed_per_seed": counts, "success_spread": spread,
          "falls_per_seed": [result["fell"] for result in results],
          "timeouts_per_seed": [result["not_finished"] for result in results],
          "left_row_per_seed": [result["left_row"] for result in results],
          "vx_mae_per_seed": [result["overall_mae"]["vx"] for result in results]}


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
  parser.add_argument("--model", type=Path, default=Path("/model/model_28996.pt"))
  parser.add_argument("--output-root", type=Path, default=Path("/outputs"))
  parser.add_argument("--preflight-only", action="store_true")
  parser.add_argument("--smoke", action="store_true",
                      help="2 env x 2 s startup test, not an acceptance run")
  args = parser.parse_args()
  manifest = json.loads(args.manifest.read_text())
  versions = check_environment(manifest)
  check_checkpoint(args.model, manifest)
  print("[release] provenance and checkpoint contract OK", flush=True)
  if args.preflight_only:
    return 0
  if not torch.cuda.is_available():
    raise RuntimeError("GPU unavailable inside container; check NVIDIA runtime")

  timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
  kind = "smoke" if args.smoke else "acceptance"
  output = args.output_root / f"{kind}-{timestamp}"
  output.mkdir(parents=True, exist_ok=False)
  seeds = [manifest["seeds"][0]] if args.smoke else manifest["seeds"]
  num_envs = 2 if args.smoke else manifest["num_envs"]
  duration_s = 2.0 if args.smoke else manifest["duration_s"]
  results = [run_seed(manifest, args.model, seed, output, num_envs, duration_s)
             for seed in seeds]
  verdict = {"kind": kind, "manifest": manifest, "dependencies": versions,
             "model_sha256": manifest["checkpoint_sha256"],
             "source_sha256": manifest["source_sha256"],
             "result_files": [f"seed{seed}.json" for seed in seeds]}
  if args.smoke:
    verdict["startup_ok"] = True
  else:
    verdict.update(assess(results, manifest))
  temp_path = output / "verdict.json.tmp"
  temp_path.write_text(json.dumps(verdict, ensure_ascii=False, indent=2))
  temp_path.replace(output / "verdict.json")
  print(f"[release] {kind}: {output} — "
        f"{'PASS' if verdict.get('accepted', True) else 'FAIL'}", flush=True)
  return 0 if verdict.get("accepted", True) else 1


if __name__ == "__main__":
  raise SystemExit(main())
