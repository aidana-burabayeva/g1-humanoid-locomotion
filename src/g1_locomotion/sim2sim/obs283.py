"""Build the 283-dim actor input (obs_spec.yaml) from MuJoCo state — shared by both runners.

The same code builds the vector both in the check against mjlab
(scripts/sim2sim_reference.py: the vector from mjlab state is compared with
what the ObservationManager builds) and in the run against the unitree_mujoco
model (scripts/sim2sim_unitree_mujoco.py). So a layout bug is caught by the
first one, instead of being discovered as the robot falling over in the second.

Layout (offsets checked by `onnx_parity.py --obs-spec` against the built env):
  0   base_ang_vel      3   gyro in pelvis axes, rad/s (free joint: qvel[3:6] — body axes)
  3   projected_gravity 3   R(q)^T · (0,0,-1)
  6   joint_pos         29  q - q_default (joint_order order)
  35  joint_vel         29  dq
  64  actions           29  previous policy output (raw, before scale/offset)
  93  command           3   vx, vy, wz
  96  height_scan       187 (pelvis_z - hit_z) * 0.2; 17 (x) x 11 (y) grid,
                            inner loop over x, outer over y; rays are vertical
                            (ray_alignment=yaw). In mjlab the rays only hit
                            geom group 0 (terrain), the robot (group 3) does not
                            interfere — so on a flat plane all 187 values equal
                            pelvis_z*0.2.
Action: q_target = a * action_scale + q_default.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OBS_DIM = 283
N_J = 29
SCAN_NX, SCAN_NY, SCAN_RES = 17, 11, 0.1
SCAN_SCALE = 0.2
SCAN_MISS = 5.0


def quat_rotate_inverse_wxyz(q: np.ndarray, v: np.ndarray) -> np.ndarray:
  """R(q)^T v for a wxyz quaternion (MuJoCo/mjlab)."""
  w, x, y, z = q
  R = np.array([
    [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
    [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
    [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
  ])
  return R.T @ v


def yaw_of(q: np.ndarray) -> float:
  w, x, y, z = q
  return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def scan_offsets_xy() -> np.ndarray:
  """Ray offsets in pelvis-heading axes, ordered as in mjlab's GridPattern."""
  xs = -0.8 + SCAN_RES * np.arange(SCAN_NX)
  ys = -0.5 + SCAN_RES * np.arange(SCAN_NY)
  return np.array([(x, y) for y in ys for x in xs])  # outer y, inner x


def height_scan_plane(pelvis_pos: np.ndarray, yaw: float,
                      plane_z: float = 0.0) -> np.ndarray:
  """Scan over a horizontal plane z=plane_z (no scale applied)."""
  n = SCAN_NX * SCAN_NY
  h = np.full(n, float(pelvis_pos[2]) - plane_z)
  # Hit points are not needed for a plane, but check that the ray can
  # actually reach it (max_distance), otherwise mjlab would give miss_value.
  h[h > SCAN_MISS] = SCAN_MISS
  return h


class DeployCfg:
  """Exact deploy-time numbers (from the mjlab env, see scripts/sim2sim_reference.py)."""

  def __init__(self, path: Path):
    d = json.loads(Path(path).read_text())
    self.raw = d
    self.joint_names: list[str] = d["joint_names"]
    self.default_q = np.array(d["default_joint_pos"], dtype=np.float64)
    self.action_scale = np.array(d["action_scale"], dtype=np.float64)
    self.kp = np.array(d["kp"], dtype=np.float64)
    self.kd = np.array(d["kd"], dtype=np.float64)
    self.effort = np.array(d["effort_limit"], dtype=np.float64)
    self.physics_dt = float(d["physics_dt"])
    self.decimation = int(d["decimation"])
    self.init_height = float(d["init_root_height"])
    self.fall_angle = float(d["fell_over_limit_angle_rad"])
    assert len(self.joint_names) == N_J


def build_obs(cfg: DeployCfg, quat_wxyz, ang_vel_b, q, dq, last_action, cmd,
              scan_heights) -> np.ndarray:
  """283-dim vector (float32). All arrays are in cfg.joint_names order."""
  o = np.empty(OBS_DIM, dtype=np.float64)
  o[0:3] = ang_vel_b
  o[3:6] = quat_rotate_inverse_wxyz(np.asarray(quat_wxyz, float),
                                    np.array([0.0, 0.0, -1.0]))
  o[6:35] = np.asarray(q) - cfg.default_q
  o[35:64] = dq
  o[64:93] = last_action
  o[93:96] = cmd
  o[96:283] = np.asarray(scan_heights) * SCAN_SCALE
  return o.astype(np.float32)


def check_against_metadata(cfg: DeployCfg, meta: dict, tol: float = 1e-3) -> list[str]:
  """Check exact numbers against the rounded .onnx metadata (3 digits)."""
  bad = []
  if meta.get("joint_names", "").split(",") != cfg.joint_names:
    bad.append("joint_names")
  for key, val in (("default_joint_pos", cfg.default_q), ("action_scale", cfg.action_scale),
                   ("joint_stiffness", cfg.kp), ("joint_damping", cfg.kd)):
    mv = np.array([float(x) for x in meta[key].split(",")])
    if mv.shape != val.shape or np.abs(mv - val).max() > tol:
      bad.append(key)
  if meta.get("observation_names", "").split(",") != [
      "base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions",
      "command", "height_scan"]:
    bad.append("observation_names")
  return bad
