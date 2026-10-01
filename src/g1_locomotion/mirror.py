"""Mirror G1 observations and actions across the sagittal plane."""

from __future__ import annotations

from typing import Any

import torch


JOINT_NAMES: tuple[str, ...] = (
  "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
  "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
  "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
  "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
  "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
  "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
  "left_wrist_pitch_joint", "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
  "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)


ACTOR_LAYOUT: tuple[tuple[str, int], ...] = (
  ("base_ang_vel", 3), ("projected_gravity", 3), ("joint_pos", 29),
  ("joint_vel", 29), ("actions", 29), ("command", 3), ("height_scan", 187),
)
CRITIC_LAYOUT: tuple[tuple[str, int], ...] = (
  ("base_lin_vel", 3), ("base_ang_vel", 3), ("projected_gravity", 3),
  ("joint_pos", 29), ("joint_vel", 29), ("actions", 29), ("command", 3),
  ("height_scan", 187), ("foot_height", 2), ("foot_air_time", 2),
  ("foot_contact", 2), ("foot_contact_forces", 6),
)
SCAN_SIZE = (1.6, 1.0)
SCAN_RESOLUTION = 0.1


CALLS = {"obs": 0, "actions": 0}


def joint_mirror(names=JOINT_NAMES) -> tuple[list[int], list[float]]:
  """Return joint index and sign mappings for sagittal mirroring."""
  idx = {n: i for i, n in enumerate(names)}
  perm, sign = [], []
  for n in names:
    if n.startswith("left_"):
      p = "right_" + n[len("left_"):]
    elif n.startswith("right_"):
      p = "left_" + n[len("right_"):]
    else:
      p = n
    perm.append(idx[p])
    sign.append(-1.0 if ("_roll" in n or "_yaw" in n) else 1.0)
  return perm, sign


def scan_mirror(size=SCAN_SIZE, resolution=SCAN_RESOLUTION, pattern=None) -> list[int]:
  """Mirror height-scan samples across the lateral axis."""
  if pattern is None:
    from mjlab.sensor import GridPatternCfg
    pattern = GridPatternCfg(size=size, resolution=resolution)
  offs, _ = pattern.generate_rays(None, "cpu")
  xy = offs[:, :2].double()
  target = xy * torch.tensor([1.0, -1.0], dtype=torch.float64)
  d = torch.cdist(target, xy)
  perm = d.argmin(dim=1)
  assert float(d.min(dim=1).values.max()) < 1e-4, "height scan is not symmetric about y"
  assert sorted(perm.tolist()) == list(range(len(perm))), "scan mapping is not a permutation"
  return perm.tolist()


def _term_ops(name: str, dim: int, jperm, jsign, sperm) -> tuple[list[int], list[float]]:
  if name in ("base_lin_vel", "projected_gravity"):
    return [0, 1, 2], [1.0, -1.0, 1.0]
  if name == "base_ang_vel":
    return [0, 1, 2], [-1.0, 1.0, -1.0]
  if name == "command":
    return [0, 1, 2], [1.0, -1.0, -1.0]
  if name in ("joint_pos", "joint_vel", "actions"):
    return list(jperm), list(jsign)
  if name == "height_scan":
    return list(sperm), [1.0] * len(sperm)
  if name in ("foot_height", "foot_air_time", "foot_contact"):
    return [1, 0], [1.0, 1.0]
  if name == "foot_contact_forces":
    return [3, 4, 5, 0, 1, 2], [1.0, -1.0, 1.0, 1.0, -1.0, 1.0]
  raise KeyError(f"g1_mirror: no mirror rule for term '{name}' ({dim})")


def build_group_ops(layout, jperm=None, jsign=None, sperm=None) -> tuple[torch.Tensor, torch.Tensor]:
  """Build mirror mappings for one observation group."""
  if jperm is None:
    jperm, jsign = joint_mirror()
  if sperm is None:
    sperm = scan_mirror()
  perm, sign, off = [], [], 0
  for name, dim in layout:
    p, s = _term_ops(name, dim, jperm, jsign, sperm)
    base = len(p)
    assert dim % base == 0, f"{name}: dimension {dim} is not divisible by {base}"
    for h in range(dim // base):
      o = off + h * base
      perm += [o + i for i in p]
      sign += s
    off += dim
  return torch.tensor(perm, dtype=torch.long), torch.tensor(sign)


class G1Mirror:
  """Apply G1 observation and action mirror mappings."""

  def __init__(self, env: Any = None):
    jnames, pattern, layouts = JOINT_NAMES, None, {"actor": ACTOR_LAYOUT, "critic": CRITIC_LAYOUT}
    base = getattr(env, "unwrapped", env)
    if base is not None and hasattr(base, "observation_manager"):
      om = base.observation_manager
      layouts = {
        g: tuple((n, int(d[-1])) for n, d in zip(om.active_terms[g], om.group_obs_term_dim[g]))
        for g in om.active_terms
      }
      jnames = tuple(base.scene["robot"].joint_names)
      try:
        sensor_cfg = base.scene["terrain_scan"].cfg
        pattern = sensor_cfg.pattern
      except KeyError:
        pattern = None
    jperm, jsign = joint_mirror(jnames)
    sperm = scan_mirror(pattern=pattern)
    self.layouts = layouts
    self.ops = {g: build_group_ops(lay, jperm, jsign, sperm) for g, lay in layouts.items()}
    self.act_perm = torch.tensor(jperm, dtype=torch.long)
    self.act_sign = torch.tensor(jsign)

  @staticmethod
  def _apply(x: torch.Tensor, perm: torch.Tensor, sign: torch.Tensor) -> torch.Tensor:
    assert x.shape[-1] == perm.numel(), f"size {x.shape[-1]} does not match layout {perm.numel()}"
    return x[..., perm.to(x.device)] * sign.to(device=x.device, dtype=x.dtype)

  def mirror_obs_group(self, group: str, x: torch.Tensor) -> torch.Tensor:
    perm, sign = self.ops[group]
    return self._apply(x, perm, sign)

  def mirror_actions(self, a: torch.Tensor) -> torch.Tensor:
    return self._apply(a, self.act_perm, self.act_sign)

  def mirror_obs(self, obs):
    """Mirror observation groups without changing their structure."""
    out = obs.clone() if hasattr(obs, "clone") and not isinstance(obs, dict) else dict(obs)
    for g in list(obs.keys()):
      if g not in self.ops:
        raise KeyError(f"g1_mirror: unknown observation group '{g}'")
      out[g] = self.mirror_obs_group(g, obs[g])
    return out


_CACHE: dict[int, G1Mirror] = {}


def get_mirror(env: Any = None) -> G1Mirror:
  key = id(getattr(env, "unwrapped", env)) if env is not None else 0
  m = _CACHE.get(key)
  if m is None:
    m = _CACHE[key] = G1Mirror(env)
  return m


def g1_symmetry_augmentation(env=None, obs=None, actions=None):
  """Generate mirrored samples for rsl-rl symmetry loss."""
  m = get_mirror(env)
  obs_aug = None
  if obs is not None:
    CALLS["obs"] += 1
    mo = m.mirror_obs(obs)
    obs_aug = torch.cat([obs, mo], dim=0)
  act_aug = None
  if actions is not None:
    CALLS["actions"] += 1
    act_aug = torch.cat([actions, m.mirror_actions(actions)], dim=0)
  return obs_aug, act_aug
