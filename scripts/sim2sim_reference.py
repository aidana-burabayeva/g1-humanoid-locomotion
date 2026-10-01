"""Reference for the deploy runner: exact numbers + obs-builder check.

1. From the built mjlab env (not from the config and not from rounded .onnx
   metadata), the following are read out: joint order, default_joint_pos,
   action_scale, actuator kp/kd (gainprm/biasprm, as in mjlab exporter_utils),
   torque limits, armature, physics step, decimation, start height, fell_over
   threshold. -> deploy_cfg.json (read by sim2sim_unitree_mujoco.py).
2. The policy (torch) is run in the same env with a fixed command; on each
   step the 283-dim input is built INDEPENDENTLY by g1_locomotion.sim2sim.obs283
   .build_obs from the raw MuJoCo state (free-joint qpos/qvel, joints, last
   action, command, pelvis z, plane z=0) and compared against the
   ObservationManager vector; plus onnxruntime on the built vector against the
   torch action.
   Observation noise is disabled in play (enable_corruption=False); the
   constant encoder bias (the encoder_bias startup event) is added to the
   build explicitly and separately — on the real robot this is a genuine
   encoder calibration error.

Run (CPU): CUDA_VISIBLE_DEVICES="" python scripts/sim2sim_reference.py \
  --checkpoint ... --out <output_dir>
  (onnxruntime is installed in the container image)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from g1_locomotion import harness as H  # noqa: E402
from g1_locomotion.sim2sim import obs283 as G  # noqa: E402

TERMS = [("base_ang_vel", 0, 3), ("projected_gravity", 3, 3), ("joint_pos", 6, 29),
         ("joint_vel", 35, 29), ("actions", 64, 29), ("command", 93, 3),
         ("height_scan", 96, 187)]


def t2n(x):
  return x.detach().cpu().numpy().astype(np.float64)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--task", default="G1-Flat-Deploy-NoCmdCurriculum-SymLoss")
  ap.add_argument("--checkpoint", required=True)
  ap.add_argument("--onnx", default=None)
  ap.add_argument("--cmd", default="0.5,0,0")
  ap.add_argument("--num-envs", type=int, default=2)
  ap.add_argument("--steps", type=int, default=300)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--tol-obs", type=float, default=1e-5,
                  help="obs-builder tolerance (float32 from the same numbers)")
  ap.add_argument("--tol-act", type=float, default=1e-4)
  ap.add_argument("--out", required=True)
  args = ap.parse_args()

  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  ckpt = Path(args.checkpoint).resolve()
  onnx_path = Path(args.onnx) if args.onnx else ckpt.parent / f"{ckpt.parent.name}.onnx"
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)
  cmd = [float(x) for x in args.cmd.split(",")]

  env, policy, _ = H.build(args.task, str(ckpt), args.num_envs, 0, "cpu", seed=args.seed)
  base = env.unwrapped
  robot = base.scene["robot"]
  m = base.sim.mj_model
  jn = list(robot.joint_names)
  act = base.action_manager.get_term("joint_pos")
  assert list(act.target_names) == jn, "action order != joint order"

  # --- 1. exact numbers ---
  jid = {m.joint(f"robot/{n}").id: n for n in jn}
  by_joint = {}
  for a in range(m.nu):
    j = int(m.actuator_trnid[a, 0])
    if j in jid:
      by_joint[jid[j]] = a
  aid = [by_joint[n] for n in jn]
  dof = [int(m.jnt_dofadr[m.joint(f"robot/{n}").id]) for n in jn]
  fell_cfg = base.termination_manager.get_term_cfg("fell_over")
  cfg = {
    "source": "mjlab env (built env)", "task": args.task, "checkpoint": str(ckpt),
    "joint_names": jn,
    "default_joint_pos": t2n(robot.data.default_joint_pos[0]).tolist(),
    "action_scale": t2n(act._scale[0]).tolist(),
    "kp": m.actuator_gainprm[aid, 0].astype(float).tolist(),
    "kd": (-m.actuator_biasprm[aid, 2]).astype(float).tolist(),
    "effort_limit": m.actuator_forcerange[aid, 1].astype(float).tolist(),
    "armature": m.dof_armature[dof].astype(float).tolist(),
    "joint_damping_passive": m.dof_damping[dof].astype(float).tolist(),
    "joint_frictionloss": m.dof_frictionloss[dof].astype(float).tolist(),
    "physics_dt": float(m.opt.timestep), "decimation": int(base.cfg.decimation),
    "integrator": int(m.opt.integrator),
    "init_root_height": float(robot.data.default_root_state[0, 2]),
    "fell_over_limit_angle_rad": float(fell_cfg.params["limit_angle"]),
    "encoder_bias_env0": t2n(robot.data.encoder_bias[0]).tolist(),
  }
  (out / "deploy_cfg.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
  dc = G.DeployCfg(out / "deploy_cfg.json")

  import onnx
  import onnxruntime as ort
  meta = {e.key: e.value for e in onnx.load(str(onnx_path)).metadata_props}
  bad_meta = G.check_against_metadata(dc, meta)
  sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
  in_name = sess.get_inputs()[0].name

  # --- 2. obs-builder check against ObservationManager ---
  pelvis = m.body("robot/pelvis").id
  fj = int(m.body_jntadr[pelvis])
  assert m.jnt_type[fj] == 0, "pelvis has no free joint"
  qadr, vadr = int(m.jnt_qposadr[fj]), int(m.jnt_dofadr[fj])
  term = base.command_manager._terms["twist"]
  H._fix_command(term, cmd)
  obs, _ = env.reset()
  H._fix_command(term, cmd)

  per_term = {n: 0.0 for n, _, _ in TERMS}
  per_term_nobias = {n: 0.0 for n, _, _ in TERMS}
  act_diff = 0.0
  scan_spread = 0.0
  n_cmp = 0
  with torch.no_grad():
    for _ in range(args.steps):
      ref = t2n(obs["actor"])
      qpos = t2n(base.sim.data.qpos)
      qvel = t2n(base.sim.data.qvel)
      xpos = t2n(base.sim.data.xpos)
      q = t2n(robot.data.joint_pos)
      dq = t2n(robot.data.joint_vel)
      bias = t2n(robot.data.encoder_bias)
      last = t2n(base.action_manager.action)
      cmdv = t2n(term.command)
      a_torch = policy(obs)
      for e in range(args.num_envs):
        quat = qpos[e, qadr + 3:qadr + 7]
        pz = xpos[e, pelvis]
        scan = G.height_scan_plane(pz, G.yaw_of(quat))
        o = G.build_obs(dc, quat, qvel[e, vadr + 3:vadr + 6], q[e] + bias[e],
                        dq[e], last[e], cmdv[e], scan)
        o_nb = G.build_obs(dc, quat, qvel[e, vadr + 3:vadr + 6], q[e],
                           dq[e], last[e], cmdv[e], scan)
        for n, off, dim in TERMS:
          per_term[n] = max(per_term[n], float(np.abs(o[off:off + dim] - ref[e, off:off + dim]).max()))
          per_term_nobias[n] = max(per_term_nobias[n],
                                   float(np.abs(o_nb[off:off + dim] - ref[e, off:off + dim]).max()))
        scan_spread = max(scan_spread, float(np.ptp(ref[e, 96:283])))
        a_onnx = sess.run(None, {in_name: o[None, :]})[0][0]
        act_diff = max(act_diff, float(np.abs(a_onnx - t2n(a_torch)[e]).max()))
        n_cmp += 1
      H._fix_command(term, cmd)
      obs, _, dones, _ = env.step(a_torch)
      H._fix_command(term, cmd)
      if dones.any():
        ids = dones.bool().nonzero(as_tuple=False).squeeze(-1)
        from tensordict import TensorDict
        o2, _ = H.reset_done_envs(base, ids, obs)
        obs = TensorDict(o2, batch_size=[args.num_envs])

  max_obs = max(per_term.values())
  res = {
    "task": args.task, "checkpoint": str(ckpt), "onnx": str(onnx_path),
    "steps": args.steps, "num_envs": args.num_envs, "comparisons": n_cmp, "cmd": cmd,
    "metadata_vs_exact_mismatch": bad_meta,
    "obs_builder_max_abs_diff_per_term": per_term,
    "obs_builder_max_abs_diff": max_obs,
    "obs_builder_max_abs_diff_per_term_without_encoder_bias": per_term_nobias,
    "encoder_bias_abs_max": float(np.abs(np.array(cfg["encoder_bias_env0"])).max()),
    "height_scan_ptp_on_flat_max": scan_spread,
    "onnx_on_built_obs_vs_torch_max_abs_diff": act_diff,
    "tol_obs": args.tol_obs, "tol_act": args.tol_act,
    "verdict": "PASS" if (not bad_meta and max_obs <= args.tol_obs
                          and act_diff <= args.tol_act) else "FAIL",
  }
  (out / "obs_builder_check.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
  print(json.dumps(res, indent=2, ensure_ascii=False))
  return 0 if res["verdict"] == "PASS" else 1


if __name__ == "__main__":
  raise SystemExit(main())
