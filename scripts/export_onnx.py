#!/usr/bin/env python3
"""Export a checkpoint's actor to ONNX with mjlab's exporter and metadata.

This follows the export performed by mjlab's ``VelocityOnPolicyRunner.save``
(opset 18, legacy exporter, base metadata attached). Verify the result with
``scripts/onnx_parity.py`` before using it.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import g1_locomotion.tasks  # noqa: E402,F401


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("checkpoint")
  ap.add_argument("--task", default="G1-Flat-Deploy-NoCmdCurriculum-SymLoss")
  ap.add_argument("--out", required=True, help="output .onnx path")
  ap.add_argument("--device", default="cpu")
  args = ap.parse_args()

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

  env_cfg = load_env_cfg(args.task, play=True)
  env_cfg.scene.num_envs = 1
  agent_cfg = load_rl_cfg(args.task)
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device=args.device),
                           clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=args.device)
  runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True,
              map_location=args.device)

  out = Path(args.out)
  runner.export_policy_to_onnx(str(out.parent), out.name)
  attach_metadata_to_onnx(str(out), get_base_metadata(env.unwrapped, "local"))
  env.close()
  digest = hashlib.sha256(out.read_bytes()).hexdigest()
  print(f"[export] {out} sha256={digest}", flush=True)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
