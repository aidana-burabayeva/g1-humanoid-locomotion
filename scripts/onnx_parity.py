"""Parity smoke test: exported .onnx against the torch policy from a checkpoint.

Why this lives here (sim2sim): this is the first check of the deploy path. Anything
related to carrying the policy from training into inference outside the env
lives in this script.

What is ACTUALLY checked:
  1. .onnx metadata: dimensions, term order, scales, default poses,
     observation history.
  2. Input composition: sum of active actor term dimensions == .onnx input.
  3. Parity: on every step of the real env, the torch policy and onnxruntime
     receive the EXACT SAME observation vector (groups concatenated in the
     order of model.obs_groups, exactly as inside MLPModel), and the actions
     are compared.

Tolerance. Both sides are float32, the same network (the onnx is exported from
this same model), but different GEMM backends (ATen/MKL vs onnxruntime). For an
MLP made of several Linear+ELU layers, the accumulated relative difference is on
the order of the float32 epsilon (1.2e-7) per layer; with ~4 layers and
|activation| ~ O(1..10), a reasonable threshold is 1e-5 absolute difference. It
is not tuned to the result: if the discrepancy is larger, it means DIFFERENT
weights (onnx from a different checkpoint) or different preprocessing, not
arithmetic noise.

Flags for 283-input policies (283-dim actor, g1_locomotion tasks): `--tol`
(tolerance, set before the run), `--neg-checkpoint` (negative control: a
different checkpoint from the same run against the same .onnx), `--obs-spec`
(check against obs_spec.yaml: terms/offsets, joint order, action_scale,
default_joint_pos), `--cmd` (pin the command via the harness's _fix_command),
`--json-out`. Without these flags, behavior is unchanged.
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

TOL = 1e-5  # default; for the 283-dim policy the tolerance is set via --tol


def check_obs_spec(spec_path: Path, terms, dims, meta: dict, in_dim: int) -> bool:
  """Check obs_spec.yaml against the built env and the .onnx metadata (item 2)."""
  import yaml
  spec = yaml.safe_load(spec_path.read_text())
  ok = True
  env_layout, off = [], 0
  for name, d in zip(terms, dims):
    n = int(np.prod(d))
    env_layout.append((name, off, n))
    off += n
  spec_layout = [(t["name"], int(t["offset"]), int(t["dim"])) for t in spec["terms"]]
  print(f"[spec] {spec_path}")
  for (en, eo, ed), (sn, so, sd) in zip(env_layout, spec_layout):
    good = (en, eo, ed) == (sn, so, sd)
    ok &= good
    print(f"[spec] {'OK ' if good else 'BAD'} env {en}@{eo}+{ed}  spec {sn}@{so}+{sd}")
  if len(env_layout) != len(spec_layout):
    ok = False
    print(f"[spec] BAD term count: env {len(env_layout)}, spec {len(spec_layout)}")
  good = int(spec["policy"]["actor_input_dim"]) == in_dim == off
  ok &= good
  print(f"[spec] {'OK ' if good else 'BAD'} actor_input_dim spec="
        f"{spec['policy']['actor_input_dim']} onnx={in_dim} env={off}")
  mj = meta.get("joint_names", "").split(",")
  good = mj == list(spec["joint_order"])
  ok &= good
  print(f"[spec] {'OK ' if good else 'BAD'} joint_order == metadata joint_names")
  for key in ("action_scale", "default_joint_pos"):
    mv = np.array([float(x) for x in meta.get(key, "").split(",")])
    sv = np.array(spec["action"][key], dtype=float)
    d = float(np.abs(mv - sv).max()) if mv.shape == sv.shape else float("inf")
    good = d <= 1e-3  # metadata rounded to 3 digits, spec to 4
    ok &= good
    print(f"[spec] {'OK ' if good else 'BAD'} {key}: max|spec - meta| = {d:.1e}")
  ms = meta.get("observation_terms_scale", "").split(",")
  hs = [t for t in spec["terms"] if t["name"] == "height_scan"]
  if hs and "height_scan" in terms:
    i = list(terms).index("height_scan")
    good = abs(float(ms[i]) - float(hs[0]["scale"])) < 1e-6
    ok &= good
    print(f"[spec] {'OK ' if good else 'BAD'} height_scan scale meta={ms[i]} spec={hs[0]['scale']}")
  print(f"[spec] obs_spec check RESULT: {'OK' if ok else 'MISMATCH'}")
  return ok


def dump_metadata(onnx_path: Path) -> dict:
  m = onnx.load(str(onnx_path))
  meta = {e.key: e.value for e in m.metadata_props}
  print(f"[onnx] {onnx_path}")
  for i in m.graph.input:
    shape = [d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim]
    print(f"[onnx] input  {i.name}: {shape}")
  for o in m.graph.output:
    shape = [d.dim_value or d.dim_param for d in o.type.tensor_type.shape.dim]
    print(f"[onnx] output {o.name}: {shape}")
  print(f"[onnx] opset={m.opset_import[0].version} producer={m.producer_name}")
  for k, v in meta.items():
    print(f"[meta] {k} = {v}")
  return meta


def build(task_id: str, checkpoint: str, num_envs: int, device: str, seed: int,
          neg_checkpoint: str | None = None):
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)
  env_cfg.scene.num_envs = num_envs
  env_cfg.seed = seed
  if getattr(env_cfg, "curriculum", None):
    env_cfg.curriculum = {}
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(checkpoint, load_cfg={"actor": True}, strict=True,
              map_location=device)
  policy = runner.get_inference_policy(device=device)
  model = runner.alg.get_policy()
  neg_policy = None
  if neg_checkpoint is not None:
    runner2 = runner_cls(env, asdict(load_rl_cfg(task_id)), device=device)
    runner2.load(neg_checkpoint, load_cfg={"actor": True}, strict=True,
                 map_location=device)
    neg_policy = runner2.get_inference_policy(device=device)
  if neg_checkpoint is None:
    return env, policy, model
  return env, policy, model, neg_policy


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--task", default="Mjlab-Velocity-Flat-Unitree-G1")
  ap.add_argument("--checkpoint", required=True)
  ap.add_argument("--onnx", default=None,
                  help="default: auto-export next to the checkpoint")
  ap.add_argument("--num-envs", type=int, default=2)
  ap.add_argument("--steps", type=int, default=300)
  ap.add_argument("--device", default="cpu")
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--tol", type=float, default=TOL,
                  help="max|Δ| tolerance, set BEFORE the run")
  ap.add_argument("--neg-checkpoint", default=None,
                  help="negative control: torch policy from a different checkpoint "
                       "against the same .onnx — the discrepancy should be >> tol")
  ap.add_argument("--obs-spec", default=None, help="check against obs_spec.yaml")
  ap.add_argument("--cmd", default=None,
                  help="pin the command vx,vy,wz (otherwise regular play)")
  ap.add_argument("--json-out", default=None)
  args = ap.parse_args()
  tol = args.tol

  ckpt = Path(args.checkpoint).resolve()
  onnx_path = (Path(args.onnx).resolve() if args.onnx
               else ckpt.parent / f"{ckpt.parent.name}.onnx")
  if not onnx_path.exists():
    sys.exit(f"[FAIL] no .onnx next to the checkpoint: {onnx_path}")

  meta = dump_metadata(onnx_path)

  # Custom tasks (G1-*-Deploy-*) are registered by importing the g1_locomotion.tasks package.
  sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
  try:
    import g1_locomotion.tasks  # noqa: F401
  except Exception as exc:  # pragma: no cover
    print(f"[parity] custom tasks not registered ({exc!r})")

  torch.manual_seed(args.seed)
  np.random.seed(args.seed)
  built = build(args.task, str(ckpt), args.num_envs, args.device, args.seed,
                neg_checkpoint=args.neg_checkpoint)
  env, policy, model = built[:3]
  neg_policy = built[3] if len(built) > 3 else None
  fix_cmd = None
  if args.cmd is not None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from g1_locomotion.harness import _fix_command
    cmd = [float(x) for x in args.cmd.split(",")]
    term = env.unwrapped.command_manager._terms["twist"]
    fix_cmd = lambda: _fix_command(term, cmd)  # noqa: E731
    fix_cmd()
    env.reset()
    fix_cmd()
  base = env.unwrapped
  om = base.observation_manager

  # --- 2. input composition: numerically ---
  groups = list(model.obs_groups)
  print(f"[obs] model.obs_groups = {groups}")
  obs_dim = int(model.obs_dim)
  print(f"[obs] model.obs_dim = {obs_dim}")
  terms = om.active_terms["actor"]
  dims = om.group_obs_term_dim["actor"]
  total = 0
  for name, d in zip(terms, dims):
    n = int(np.prod(d))
    total += n
    print(f"[obs] term {name:20s} dim={tuple(d)} -> {n} (offset {total - n})")
  print(f"[obs] sum of actor terms = {total}")

  sess = ort.InferenceSession(str(onnx_path),
                              providers=["CPUExecutionProvider"])
  in_name = sess.get_inputs()[0].name
  in_dim = sess.get_inputs()[0].shape[-1]
  print(f"[obs] onnx input = {in_dim}")
  ok_dim = (in_dim == total == obs_dim)
  print(f"[obs] dimension match: {'OK' if ok_dim else 'MISMATCH'}")

  # --- 4. term order from metadata vs. observation_manager ---
  meta_names = meta.get("observation_names", "").split(",")
  print(f"[order] onnx metadata: {meta_names}")
  print(f"[order] observation_manager.active_terms['actor']: {terms}")
  print(f"[order] order matches: {meta_names == list(terms)}")
  print("[order] NOTE: metadata only gives term NAMES in order; "
        "term dimensions and their offsets are NOT recorded in the .onnx.")
  ok_spec = True
  if args.obs_spec:
    ok_spec = check_obs_spec(Path(args.obs_spec), terms, dims, meta, in_dim)

  # --- 3. parity ---
  # get_observations() returns a TensorDict; unpacking `obs, _ = ...`
  # would silently give back a slice of the first env (batch_size=[num_envs]).
  obs = env.get_observations()
  diffs = []
  neg_diffs = []
  first_bad = None
  obs_absmax = []
  with torch.no_grad():
    for step_i in range(args.steps):
      flat = torch.cat([obs[g] for g in groups], dim=-1)
      if step_i == 0:
        print(f"[parity] obs[{groups[0]}].shape="
              f"{tuple(obs[groups[0]].shape)}")
      assert flat.shape == (args.num_envs, obs_dim), flat.shape
      x = flat.reshape(-1, obs_dim).cpu().numpy().astype(np.float32)
      a_torch = policy(obs)
      a_onnx = np.concatenate(
        [sess.run(None, {in_name: x[i:i + 1]})[0] for i in range(x.shape[0])],
        axis=0)
      d = np.abs(a_torch.cpu().numpy() - a_onnx)
      diffs.append(d)
      if neg_policy is not None:
        neg_diffs.append(np.abs(neg_policy(obs).cpu().numpy() - a_onnx))
      obs_absmax.append(float(np.abs(x).max()))
      if first_bad is None and d.max() > tol:
        first_bad = (step_i, float(d.max()))
      if fix_cmd is not None:
        fix_cmd()
      obs, _, dones, _ = env.step(a_torch)
      if fix_cmd is not None:
        fix_cmd()

  D = np.concatenate([d.reshape(-1) for d in diffs])
  per_step_max = np.array([d.max() for d in diffs])
  print("\n=== PARITY ===")
  print(f"steps={args.steps} envs={args.num_envs} comparisons={D.size}")
  print(f"max|Δ| = {D.max():.3e}")
  print(f"mean   = {D.mean():.3e}")
  print(f"p95    = {np.percentile(D, 95):.3e}")
  print(f"p99    = {np.percentile(D, 99):.3e}")
  print(f"tol    = {tol:.1e}")
  neg_ok = True
  N = None
  if neg_diffs:
    N = np.concatenate([d.reshape(-1) for d in neg_diffs])
    ratio = float(N.max() / max(D.max(), 1e-30))
    neg_ok = N.max() > 100 * tol
    print(f"[neg] {args.neg_checkpoint}")
    print(f"[neg] max|Δ| = {N.max():.3e}, mean = {N.mean():.3e}, "
          f"ratio to main max = {ratio:.1e} -> "
          f"{'control worked' if neg_ok else 'CONTROL DOES NOT DISCRIMINATE'}")
  if first_bad is not None:
    s, v = first_bad
    print(f"[localize] first exceedance: step {s}, Δ={v:.3e}")
    half = len(per_step_max) // 2
    print(f"[localize] max|Δ| first half={per_step_max[:half].max():.3e} "
          f"second half={per_step_max[half:].max():.3e} (growing?)")
    r = np.corrcoef(per_step_max, np.array(obs_absmax))[0, 1]
    print(f"[localize] corr(max|Δ|, max|obs|) = {r:.3f}")
  verdict = ok_dim and D.max() <= tol and neg_ok and ok_spec
  print(f"\nVERDICT: {'PASS' if verdict else 'FAIL'}")
  if args.json_out:
    Path(args.json_out).write_text(json.dumps({
      "onnx": str(onnx_path), "checkpoint": str(ckpt), "task": args.task,
      "steps": args.steps, "num_envs": args.num_envs, "cmd": args.cmd,
      "tol": tol, "comparisons": int(D.size), "max_abs_diff": float(D.max()),
      "mean_abs_diff": float(D.mean()), "p99_abs_diff": float(np.percentile(D, 99)),
      "neg_checkpoint": args.neg_checkpoint,
      "neg_max_abs_diff": None if N is None else float(N.max()),
      "neg_mean_abs_diff": None if N is None else float(N.mean()),
      "dims_ok": bool(ok_dim), "obs_spec_ok": bool(ok_spec),
      "verdict": "PASS" if verdict else "FAIL", "metadata": meta,
    }, indent=2, ensure_ascii=False))
  return 0 if verdict else 1


if __name__ == "__main__":
  raise SystemExit(main())
