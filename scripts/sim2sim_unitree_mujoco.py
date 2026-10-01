"""Sim2sim: the 283-dim ONNX policy in the unitree_mujoco G1 model (plain MuJoCo, CPU).

Pipeline same as on the real robot, without mjlab and without torch:
  MuJoCo state -> obs283.build_obs (obs_spec.yaml layout, checked against
  ObservationManager in sim2sim_reference.py) -> onnxruntime ->
  q_target = a*action_scale + q_default -> PD on every physics step
  tau = kp(q_target - q) - kd*dq, clipped to the ctrlrange of the unitree
  model's motors.

Model: third_party/unitree_mujoco/g1/scene_29dof.xml (torque motors, plane
z=0, its own armature/damping/frictionloss — that is exactly what makes this
a sim2sim check).
The unitree_mujoco DDS bridge (simulate_python/unitree_mujoco.py +
unitree_sdk2py_bridge.py) is NOT used: it is needed for exchange via
unitree_sdk2 LowCmd/LowState, while the goal here is to check the policy's
input/output and the physics of someone else's model; the loop is the same
as in unitree_rl_gym's deploy_mujoco.py.

Input sources: gyro and quaternion — the unitree model's imu_gyro / imu_quat
sensors (imu site at the base of the pelvis); joints — qpos/qvel by name;
height_scan — plane z=0 (all 187 rays = pelvis z; in mjlab on a plane it is
the same, verified: spread 0.0). Pelvis z is taken from the simulator (on
the real robot — from state estimation/a height map). base_lin_vel does NOT
go into the input.

Metrics: fall (tilt angle > the mjlab fell_over threshold, or pelvis z < 0.3 m),
distance traveled (sum of pelvis xy increments), MAE of vx in body frame
(R^T · qvel[0:3]; for a free joint, qvel[0:3] is the world-frame linear
velocity), mean vx/vy/wz, drift at 10/20 m of travel as in
scripts/openloop_drift.py.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from g1_locomotion.sim2sim import obs283 as G  # noqa: E402

DEFAULT_XML = HERE.parent / "third_party" / "unitree_mujoco" / "g1" / "scene_29dof.xml"


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--onnx", required=True)
  ap.add_argument("--deploy-cfg", required=True, help="deploy_cfg.json from scripts/sim2sim_reference.py")
  ap.add_argument("--xml", default=str(DEFAULT_XML))
  ap.add_argument("--cmd", default="0.5,0,0")
  ap.add_argument("--duration", type=float, default=60.0)
  ap.add_argument("--sim-dt", type=float, default=None,
                  help="physics step; default matches mjlab (deploy_cfg physics_dt)")
  ap.add_argument("--joint-model", choices=("unitree", "mjlab"), default="unitree",
                  help="unitree — use the model's own armature/damping/frictionloss; "
                       "mjlab — substitute armature from mjlab, damping=frictionloss=0 "
                       "(diagnostic, if the unitree variant doesn't work)")
  ap.add_argument("--scan", choices=("plane", "zero"), default="plane",
                  help="plane — raycast to the plane z=0 (as in mjlab); zero — an all-zero "
                       "row (control: how much the policy depends on the scan)")
  ap.add_argument("--marks", type=float, nargs="+", default=[10.0, 20.0])
  ap.add_argument("--out", required=True)
  args = ap.parse_args()

  import onnx
  import onnxruntime as ort

  dc = G.DeployCfg(Path(args.deploy_cfg))
  meta = {e.key: e.value for e in onnx.load(args.onnx).metadata_props}
  bad = G.check_against_metadata(dc, meta)
  if bad:
    sys.exit(f"[FAIL] deploy_cfg does not match the .onnx metadata: {bad}")
  sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
  in_name = sess.get_inputs()[0].name
  assert sess.get_inputs()[0].shape[-1] == G.OBS_DIM

  m = mujoco.MjModel.from_xml_path(args.xml)
  d = mujoco.MjData(m)
  sim_dt = args.sim_dt or dc.physics_dt
  m.opt.timestep = sim_dt
  control_dt = dc.physics_dt * dc.decimation
  dec = int(round(control_dt / sim_dt))
  assert abs(dec * sim_dt - control_dt) < 1e-9, "control_dt is not a multiple of sim_dt"

  qadr = np.array([m.jnt_qposadr[m.joint(n).id] for n in dc.joint_names])
  vadr = np.array([m.jnt_dofadr[m.joint(n).id] for n in dc.joint_names])
  act_of = {}
  for a in range(m.nu):
    act_of[m.joint(int(m.actuator_trnid[a, 0])).name] = a
  aid = np.array([act_of[n] for n in dc.joint_names])
  assert m.actuator_gaintype[aid].max() == 0 and m.actuator_biastype[aid].max() == 0, \
    "expected torque motors"
  ctrl_lim = m.actuator_ctrlrange[aid].copy()
  if args.joint_model == "mjlab":
    m.dof_armature[vadr] = np.array(dc.raw["armature"])
    m.dof_damping[vadr] = 0.0
    m.dof_frictionloss[vadr] = 0.0
  model_diff = {
    "armature_unitree": m.dof_armature[vadr].tolist() if args.joint_model == "unitree" else None,
    "armature_mjlab": dc.raw["armature"],
    "effort_unitree_ctrlrange": ctrl_lim[:, 1].tolist(),
    "effort_mjlab": dc.effort.tolist(),
  }

  pelvis = m.body("pelvis").id
  fj = int(m.body_jntadr[pelvis])
  assert m.jnt_type[fj] == mujoco.mjtJoint.mjJNT_FREE
  fq, fv = int(m.jnt_qposadr[fj]), int(m.jnt_dofadr[fj])
  s_quat = m.sensor("imu_quat")
  s_gyro = m.sensor("imu_gyro")
  sq = slice(s_quat.adr[0], s_quat.adr[0] + 4)
  sg = slice(s_gyro.adr[0], s_gyro.adr[0] + 3)

  mujoco.mj_resetData(m, d)
  d.qpos[fq:fq + 3] = [0.0, 0.0, dc.init_height]
  d.qpos[fq + 3:fq + 7] = [1.0, 0.0, 0.0, 0.0]
  d.qpos[qadr] = dc.default_q
  d.qvel[:] = 0.0
  mujoco.mj_forward(m, d)

  cmd = np.array([float(x) for x in args.cmd.split(",")])
  last_action = np.zeros(G.N_J)
  q_target = dc.default_q.copy()
  n_ctrl = int(round(args.duration / control_dt))
  xy0 = d.qpos[fq:fq + 2].copy()
  yaw0 = G.yaw_of(d.qpos[fq + 3:fq + 7])
  prev_xy, prev_yaw, yaw_acc = xy0.copy(), yaw0, 0.0
  path = 0.0
  fell_t = None
  errs, vxs, vys, wzs = [], [], [], []
  marks = {mk: None for mk in args.marks}
  sat = 0
  cos_lim = math.cos(dc.fall_angle)
  obs_absmax = 0.0

  for k in range(n_ctrl):
    quat = d.sensordata[sq].copy()
    gyro = d.sensordata[sg].copy()
    q = d.qpos[qadr].copy()
    dq = d.qvel[vadr].copy()
    scan = (G.height_scan_plane(d.xpos[pelvis], G.yaw_of(quat)) if args.scan == "plane"
            else np.zeros(G.SCAN_NX * G.SCAN_NY))
    obs = G.build_obs(dc, quat, gyro, q, dq, last_action, cmd, scan)
    obs_absmax = max(obs_absmax, float(np.abs(obs).max()))
    a = sess.run(None, {in_name: obs[None, :]})[0][0].astype(np.float64)
    last_action = a
    q_target = a * dc.action_scale + dc.default_q
    for _ in range(dec):
      tau = dc.kp * (q_target - d.qpos[qadr]) - dc.kd * d.qvel[vadr]
      clipped = np.clip(tau, ctrl_lim[:, 0], ctrl_lim[:, 1])
      sat += int((clipped != tau).any())
      d.ctrl[aid] = clipped
      mujoco.mj_step(m, d)

    t = (k + 1) * control_dt
    qb = d.qpos[fq + 3:fq + 7]
    v_b = G.quat_rotate_inverse_wxyz(qb, d.qvel[fv:fv + 3])
    w_b = d.qvel[fv + 3:fv + 6]
    errs.append(abs(v_b[0] - cmd[0]))
    vxs.append(v_b[0]); vys.append(v_b[1]); wzs.append(w_b[2])
    xy = d.qpos[fq:fq + 2].copy()
    path += float(np.linalg.norm(xy - prev_xy))
    yaw = G.yaw_of(qb)
    yaw_acc += (yaw - prev_yaw + math.pi) % (2 * math.pi) - math.pi
    prev_xy, prev_yaw = xy, yaw
    dxy = xy - xy0
    lat = -math.sin(yaw0) * dxy[0] + math.cos(yaw0) * dxy[1]
    for mk in marks:
      if marks[mk] is None and path >= mk:
        marks[mk] = {"t_s": t, "lateral_y_m": lat, "yaw_drift_deg": math.degrees(yaw_acc)}
    g_b = G.quat_rotate_inverse_wxyz(qb, np.array([0.0, 0.0, -1.0]))
    if -g_b[2] < cos_lim or d.xpos[pelvis][2] < 0.3:
      fell_t = t
      break

  res = {
    "xml": args.xml, "onnx": args.onnx, "joint_model": args.joint_model, "scan": args.scan,
    "sim_dt": sim_dt, "decimation": dec, "control_dt": control_dt,
    "cmd": cmd.tolist(), "duration_s": args.duration,
    "fell": fell_t is not None, "fell_time_s": fell_t,
    "sim_time_s": (k + 1) * control_dt,
    "path_m": path, "displacement_m": float(np.linalg.norm(d.qpos[fq:fq + 2] - xy0)),
    "mae_vx": float(np.mean(errs)), "mean_vx": float(np.mean(vxs)),
    "mean_vx_after_2s": float(np.mean(vxs[int(2 / control_dt):])) if len(vxs) > 100 else None,
    "mean_vy": float(np.mean(vys)), "mean_wz": float(np.mean(wzs)),
    "final_lateral_y_m": lat, "final_yaw_drift_deg": math.degrees(yaw_acc),
    "marks": {f"{mk:g}m": v for mk, v in marks.items()},
    "torque_saturated_physics_steps": sat,
    "obs_absmax": obs_absmax, "model_diff": model_diff,
  }
  Path(args.out).parent.mkdir(parents=True, exist_ok=True)
  Path(args.out).write_text(json.dumps(res, indent=2, ensure_ascii=False))
  print(json.dumps({k_: v for k_, v in res.items() if k_ != "model_diff"}, indent=2,
                   ensure_ascii=False))


if __name__ == "__main__":
  main()
