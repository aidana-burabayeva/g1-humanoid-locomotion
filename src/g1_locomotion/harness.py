"""Evaluate mjlab policies under fixed commands and terrain conditions."""

import argparse
import importlib.metadata
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
  import g1_locomotion.tasks  # noqa: F401
except Exception as exc:  # pragma: no cover
  print(f"[eval] custom tasks unavailable ({exc!r}); "
        "only built-in mjlab tasks are registered")

DEFAULT_COMMANDS = ["0.3,0,0", "0.5,0,0", "0.8,0,0", "0,0.3,0", "0,0,0.5", "0.5,0,0.5"]


def configure_terrain(env_cfg, level: int, seed: int) -> None:
  """Fix terrain generation and disable terrain randomization on reset."""
  events = getattr(env_cfg, "events", None)
  if events is not None and "randomize_terrain" in events:
    events.pop("randomize_terrain")
    print("[eval] disabled randomize_terrain on reset")

  terrain = getattr(env_cfg.scene, "terrain", None)
  if terrain is None:
    return
  tg = getattr(terrain, "terrain_generator", None)
  if tg is None:
    print("[eval] flat terrain has no terrain level")
    return
  # Curriculum mode maps each terrain row to a fixed difficulty.
  tg.curriculum = True
  tg.seed = seed
  terrain.max_init_terrain_level = max(0, min(level, tg.num_rows - 1))
  print(f"[eval] terrain generator: curriculum=True, num_rows={tg.num_rows}, seed={seed}")


def pin_terrain_level(base_env, level: int) -> int | None:
  """Assign every environment to the requested terrain row."""
  terrain = getattr(base_env.scene, "terrain", None)
  if terrain is None or getattr(terrain, "terrain_origins", None) is None:
    return None
  num_rows = terrain.terrain_origins.shape[0]
  row = max(0, min(level, num_rows - 1))
  if row != level:
    print(f"[eval] terrain row {level} outside 0..{num_rows - 1}; using {row}")
  terrain.terrain_levels[:] = row
  terrain.env_origins[:] = terrain.terrain_origins[
    terrain.terrain_levels, terrain.terrain_types
  ]
  return row


def restore_training_grid(env_cfg, task_id: str) -> None:
  """Use training terrain dimensions during evaluation."""
  terrain = getattr(env_cfg.scene, "terrain", None)
  tg = getattr(terrain, "terrain_generator", None) if terrain is not None else None
  if tg is None:
    return
  from mjlab.tasks.registry import load_env_cfg

  train_cfg = load_env_cfg(task_id, play=False)
  train_terrain = getattr(train_cfg.scene, "terrain", None)
  train_tg = (getattr(train_terrain, "terrain_generator", None)
              if train_terrain is not None else None)
  if train_tg is None:
    print("[eval] training configuration has no terrain generator; grid unchanged")
    return
  changed = {}
  for field in ("size", "border_width", "num_rows", "num_cols",
                "difficulty_range"):
    old, new = getattr(tg, field, None), getattr(train_tg, field, None)
    if old != new:
      changed[field] = (old, new)
    setattr(tg, field, new)
  if changed:
    print(f"[eval] restored training terrain grid: {changed}")


def difficulty_of_row(num_rows: int, row: int,
                      difficulty_range=(0.0, 1.0)) -> float:
  """Return the difficulty assigned to a terrain row."""
  lower, upper = difficulty_range
  t = row / max(num_rows - 1, 1)
  return float(lower + (upper - lower) * t)


def row_from_difficulty(num_rows: int, difficulty: float) -> int:
  """Return the closest row for a target terrain difficulty."""
  return int(round(max(0.0, min(1.0, difficulty)) * max(num_rows - 1, 1)))


def restore_bounds_termination(env_cfg) -> None:
  """Restore terrain-boundary termination in the play task."""
  terrain = getattr(env_cfg.scene, "terrain", None)
  tg = getattr(terrain, "terrain_generator", None) if terrain is not None else None
  if tg is None:
    return
  if "out_of_terrain_bounds" in env_cfg.terminations:
    return
  from mjlab.managers.termination_manager import TerminationTermCfg
  from mjlab.tasks.velocity import mdp as velocity_mdp

  env_cfg.terminations["out_of_terrain_bounds"] = TerminationTermCfg(
    func=velocity_mdp.out_of_terrain_bounds, time_out=True
  )
  print("[eval] restored out_of_terrain_bounds termination (time_out)")


def _history_buffers(obs_manager):
  """Iterate over observation history and delay buffers."""
  for group, terms in obs_manager._group_obs_term_history_buffer.items():
    for name, buf in terms.items():
      yield ("history", group, name, buf)
  for group, terms in obs_manager._group_obs_term_delay_buffer.items():
    for name, delay in terms.items():
      yield ("delay", group, name, delay._buffer)


def snapshot_obs_history(obs_manager, keep_ids: torch.Tensor):
  """Save observation history for environments that remain active."""
  state = []
  for kind, group, name, buf in _history_buffers(obs_manager):
    if not buf.is_initialized:
      continue
    state.append({
      "buf": buf,

      "chrono": buf.buffer[keep_ids].clone(),
      "pushes": buf._num_pushes[keep_ids].clone(),
    })

  delays = []
  for group, terms in obs_manager._group_obs_term_delay_buffer.items():
    for name, delay in terms.items():
      delays.append({
        "delay": delay,
        "lags": delay._current_lags[keep_ids].clone(),
        "steps": delay._step_count[keep_ids].clone(),
      })
  return {"buffers": state, "delays": delays, "keep_ids": keep_ids}


def restore_obs_history(obs_manager, state) -> None:
  """Restore active environments after a partial reset."""
  keep = state["keep_ids"]
  for rec in state["buffers"]:
    buf = rec["buf"]
    chrono = rec["chrono"]
    max_len = buf._max_len
    # Restore chronology relative to the buffer's current pointer.
    start = (buf._pointer + 1) % max_len
    for k in range(max_len):
      slot = (start + k) % max_len
      buf._buffer[slot, keep] = chrono[:, k]
    buf._num_pushes[keep] = rec["pushes"]
  for rec in state["delays"]:
    rec["delay"]._current_lags[keep] = rec["lags"]
    rec["delay"]._step_count[keep] = rec["steps"]
  obs_manager._obs_buffer = None


def _merge_obs(prev_obs, new_obs, done_ids, num_envs: int):
  """Merge reset observations with unchanged active observations."""
  merged = {}
  for group, value in new_obs.items():

    prev = (prev_obs[group]
            if prev_obs is not None and group in prev_obs else None)
    if isinstance(value, dict):
      sub = {}
      for name, tensor in value.items():
        out = tensor.clone()
        if prev is not None:
          keep = torch.ones(num_envs, dtype=torch.bool, device=tensor.device)
          keep[done_ids] = False
          out[keep] = prev[name][keep]
        sub[name] = out
      merged[group] = sub
    else:
      out = value.clone()
      if prev is not None:
        keep = torch.ones(num_envs, dtype=torch.bool, device=value.device)
        keep[done_ids] = False
        out[keep] = prev[keep]
      merged[group] = out
  return merged


def reset_done_envs(base, done_ids: torch.Tensor, prev_obs):
  """Reset completed environments without advancing active history."""
  n = base.num_envs
  keep_mask = torch.ones(n, dtype=torch.bool, device=base.device)
  keep_mask[done_ids] = False
  keep_ids = keep_mask.nonzero(as_tuple=False).squeeze(-1)

  om = base.observation_manager
  state = snapshot_obs_history(om, keep_ids) if keep_ids.numel() > 0 else None
  obs_new, extras = base.reset(env_ids=done_ids)
  if state is not None:
    restore_obs_history(om, state)
    obs_new = _merge_obs(prev_obs, obs_new, done_ids, n)
    om._obs_buffer = obs_new
    base.obs_buf = obs_new
  return obs_new, extras


def assert_level_held(base_env, row: int | None) -> None:
  """Verify all environments remain on the selected terrain row."""
  if row is None:
    return
  terrain = base_env.scene.terrain
  levels = terrain.terrain_levels
  if not bool((levels == row).all()):
    uniq = torch.unique(levels).tolist()
    raise RuntimeError(
      f"[eval] terrain row changed: expected {row}, found {uniq}"
    )


def build(task_id: str, checkpoint: str, num_envs: int, level: int, device: str,
          seed: int = 0, command_name: str = "twist"):
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)

  env_cfg.scene.num_envs = num_envs
  env_cfg.seed = seed


  if getattr(env_cfg, "curriculum", None):
    env_cfg.curriculum = {}


  # The play grid must match training before the row is selected.
  restore_training_grid(env_cfg, task_id)
  configure_terrain(env_cfg, level, seed)
  restore_bounds_termination(env_cfg)


  # Keep terminal states available for outcome classification.
  env_cfg.auto_reset = False


  cmd_cfg = env_cfg.commands.get(command_name)
  if cmd_cfg is not None:
    cmd_cfg.resampling_time_range = (1.0e9, 1.0e9)
    cmd_cfg.heading_command = False
    if hasattr(cmd_cfg, "ranges") and hasattr(cmd_cfg.ranges, "heading"):
      cmd_cfg.ranges.heading = None
    for field in ("rel_standing_envs", "rel_heading_envs", "rel_world_envs",
                  "rel_forward_envs", "init_velocity_prob"):
      if hasattr(cmd_cfg, field):
        setattr(cmd_cfg, field, 0.0)


  try:
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  except ValueError as exc:
    # Expand contact capacity only when MuJoCo Warp reports the bound.
    m = re.search(r"nconmax must be >= (\d+)", str(exc))
    if m is None:
      raise
    need = int(m.group(1)) * 2
    print(f"[eval] nconmax {env_cfg.sim.nconmax} too small; increased to {need} "
          f"as requested by mujoco_warp")
    env_cfg.sim.nconmax = need
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  row = pin_terrain_level(env.unwrapped, level)
  if row is not None:
    num_rows = env.unwrapped.scene.terrain.terrain_origins.shape[0]
    print(f"[eval] pinned terrain row {row} of {num_rows - 1}")

  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(checkpoint, load_cfg={"actor": True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)
  return env, policy, row


def _fix_command(term, cmd) -> None:
  """Hold the same command through reset and rollout."""
  vx, vy, wz = (float(x) for x in cmd)
  ranges = getattr(term.cfg, "ranges", None)
  if ranges is not None:
    ranges.lin_vel_x = (vx, vx)
    ranges.lin_vel_y = (vy, vy)
    ranges.ang_vel_z = (wz, wz)
  term.vel_command_b[:, 0] = vx
  term.vel_command_b[:, 1] = vy
  term.vel_command_b[:, 2] = wz
  term.is_standing_env[:] = False


def _signed_diagnostics(out: dict, hist_vx, hist_vy, hist_wz, hist_alive,
                        hist_cmd) -> None:
  """Summarize signed velocity-tracking error."""
  if not hist_alive:
    return
  mask = torch.stack(hist_alive).cpu().numpy()
  cmd_arr = np.asarray(hist_cmd, dtype=np.float64)
  series = {
    "vx": (torch.stack(hist_vx).cpu().numpy().astype(np.float64), cmd_arr[:, 0]),
    "vy": (torch.stack(hist_vy).cpu().numpy().astype(np.float64), cmd_arr[:, 1]),
    "wz": (torch.stack(hist_wz).cpu().numpy().astype(np.float64), cmd_arr[:, 2]),
  }
  n_alive = mask.sum(axis=0)
  out["signed_diag_note"] = (
    "mean_*_body is signed body-frame velocity; "
    "mean_signed_error_* is actual minus commanded velocity; "
    "p05/p95 are per-step signed-error percentiles. "
    "Only active steps contribute to these statistics.")
  out["signed_diag_steps_per_ep"] = [int(x) for x in n_alive.tolist()]
  for key, (val, cmd_ts) in series.items():
    err = val - cmd_ts[:, None]
    val_m = np.where(mask, val, np.nan)
    err_m = np.where(mask, err, np.nan)
    empty = n_alive == 0
    with np.errstate(invalid="ignore"):
      mean_v = np.where(empty, np.nan, np.nanmean(np.where(empty, 0.0, val_m), axis=0))
      mean_e = np.where(empty, np.nan, np.nanmean(np.where(empty, 0.0, err_m), axis=0))
      p05 = np.where(empty, np.nan,
                     np.nanpercentile(np.where(empty, 0.0, err_m), 5, axis=0))
      p95 = np.where(empty, np.nan,
                     np.nanpercentile(np.where(empty, 0.0, err_m), 95, axis=0))

    def _j(a):
      return [None if np.isnan(x) else float(x) for x in np.asarray(a).tolist()]

    def _agg(prefix, a):
      good = np.asarray(a)[~np.isnan(np.asarray(a))]
      if good.size == 0:
        out[prefix] = out[f"{prefix}_std"] = None
        out[f"{prefix}_min"] = out[f"{prefix}_max"] = None
        return
      out[prefix] = float(good.mean())
      out[f"{prefix}_std"] = float(good.std())
      out[f"{prefix}_min"] = float(good.min())
      out[f"{prefix}_max"] = float(good.max())

    body = f"mean_{key}_body"
    out[f"{body}_per_ep"] = _j(mean_v)
    _agg(body, mean_v)
    out[f"mean_signed_error_{key}_per_ep"] = _j(mean_e)
    _agg(f"mean_signed_error_{key}", mean_e)
    out[f"p05_signed_error_{key}_per_ep"] = _j(p05)
    _agg(f"p05_signed_error_{key}", p05)
    out[f"p95_signed_error_{key}_per_ep"] = _j(p95)
    _agg(f"p95_signed_error_{key}", p95)


def rollout(env, policy, cmd, duration_s: float, level_row: int | None = None,
            command_name: str = "twist", tile_edge: str = "mark",
            switch_at_s: float | None = None, cmd_after=None,
            stop_speed: float = 0.10, stop_yaw: float = 0.15,
            stop_hold_s: float = 0.5):
  """Evaluate one fixed policy and collect per-environment metrics."""
  base = env.unwrapped
  device = base.device
  n = base.num_envs
  cmd_t = torch.tensor([float(x) for x in cmd], dtype=torch.float32,
                       device=device).repeat(n, 1)

  steps = int(duration_s / base.step_dt)
  term = base.command_manager._terms[command_name]


  has_switch = switch_at_s is not None
  cmd2 = [0.0, 0.0, 0.0] if cmd_after is None else [float(x) for x in cmd_after]
  switch_step = int(switch_at_s / base.step_dt) if has_switch else steps + 1


  cmd_step = max(switch_step - 1, 0) if has_switch else steps + 1
  cur_cmd = list(float(x) for x in cmd)


  terrain_cfg = getattr(base.scene.terrain, "cfg", None) if base.scene.terrain else None
  tg = getattr(terrain_cfg, "terrain_generator", None) if terrain_cfg else None
  tile_half = None
  if tg is not None:
    tile_half = torch.tensor([tg.size[0] / 2.0, tg.size[1] / 2.0],
                             dtype=torch.float32, device=device)


  _fix_command(term, cmd)
  pin_terrain_level(base, level_row if level_row is not None else 0)
  obs, _ = env.reset()
  _fix_command(term, cmd)
  assert_level_held(base, level_row)

  robot = base.scene["robot"]
  start_xy = robot.data.root_link_pos_w[:, :2].clone()
  prev_xy = start_xy.clone()

  alive = torch.ones(n, dtype=torch.bool, device=device)
  ep_len = torch.zeros(n, device=device)
  path = torch.zeros(n, device=device)
  disp = torch.zeros(n, device=device)
  fell = torch.zeros(n, dtype=torch.bool, device=device)
  left_terrain = torch.zeros(n, dtype=torch.bool, device=device)
  left_own_tile = torch.zeros(n, dtype=torch.bool, device=device)
  time_on_tile = torch.zeros(n, device=device)
  err_vx = torch.zeros(n, device=device)
  err_vy = torch.zeros(n, device=device)
  err_wz = torch.zeros(n, device=device)
  power = torch.zeros(n, device=device)
  counted = torch.zeros(n, device=device)


  hist_vx, hist_vy, hist_wz, hist_alive, hist_cmd = [], [], [], [], []


  err2_vx = torch.zeros(n, device=device)
  err2_vy = torch.zeros(n, device=device)
  err2_wz = torch.zeros(n, device=device)
  counted2 = torch.zeros(n, device=device)
  path_after = torch.zeros(n, device=device)
  xy_at_switch = torch.zeros(n, 2, device=device)
  stop_time = torch.full((n,), float("nan"), device=device)
  stop_path = torch.full((n,), float("nan"), device=device)
  stop_disp = torch.full((n,), float("nan"), device=device)
  quiet_for = torch.zeros(n, device=device)
  stopped = torch.zeros(n, dtype=torch.bool, device=device)
  t_since_switch = 0.0

  for step_i in range(steps):
    if has_switch and step_i == cmd_step:


      cur_cmd = list(cmd2)
    if has_switch and step_i == switch_step:


      cmd_t[:, 0], cmd_t[:, 1], cmd_t[:, 2] = cmd2[0], cmd2[1], cmd2[2]
      xy_at_switch = robot.data.root_link_pos_w[:, :2].clone()
      t_since_switch = 0.0
    _fix_command(term, cur_cmd)


    with torch.no_grad():
      actions = policy(obs)
      obs, _, dones, _ = env.step(actions)


    lin = robot.data.root_link_lin_vel_b
    ang = robot.data.root_link_ang_vel_b
    alive_f = alive.float()
    err_vx += alive_f * (lin[:, 0] - cmd_t[:, 0]).abs()
    err_vy += alive_f * (lin[:, 1] - cmd_t[:, 1]).abs()
    err_wz += alive_f * (ang[:, 2] - cmd_t[:, 2]).abs()


    tau = robot.data.qfrc_actuator
    qd = robot.data.joint_vel
    power += alive_f * (tau * qd).abs().sum(dim=1)
    counted += alive_f
    ep_len += alive_f * base.step_dt


    hist_vx.append(lin[:, 0].detach().clone())
    hist_vy.append(lin[:, 1].detach().clone())
    hist_wz.append(ang[:, 2].detach().clone())
    hist_alive.append(alive.clone())
    hist_cmd.append((float(cmd_t[0, 0]), float(cmd_t[0, 1]), float(cmd_t[0, 2])))

    xy = robot.data.root_link_pos_w[:, :2]
    step_len = torch.nan_to_num(torch.norm(xy - prev_xy, dim=1), nan=0.0)
    path += alive_f * step_len


    if has_switch and step_i >= switch_step:
      t_since_switch += base.step_dt
      err2_vx += alive_f * (lin[:, 0] - cmd_t[:, 0]).abs()
      err2_vy += alive_f * (lin[:, 1] - cmd_t[:, 1]).abs()
      err2_wz += alive_f * (ang[:, 2] - cmd_t[:, 2]).abs()
      counted2 += alive_f
      path_after += alive_f * step_len
      quiet = ((torch.norm(lin[:, :2], dim=1) < stop_speed)
               & (ang[:, 2].abs() < stop_yaw))
      quiet_for = torch.where(quiet & alive, quiet_for + base.step_dt,
                              torch.zeros_like(quiet_for))
      just = alive & ~stopped & (quiet_for >= stop_hold_s)
      if just.any():

        stop_time = torch.where(just, torch.full_like(stop_time,
                                                      t_since_switch - stop_hold_s),
                                stop_time)
        stop_path = torch.where(just, path_after, stop_path)
        stop_disp = torch.where(just, torch.norm(xy - xy_at_switch, dim=1),
                                stop_disp)
        stopped |= just
    disp = torch.where(alive, torch.norm(xy - start_xy, dim=1), disp)
    prev_xy = xy.clone()

    if tile_half is not None and tile_edge != "ignore":
      off = (xy - base.scene.env_origins[:, :2]).abs()
      off_tile = (off[:, 0] > tile_half[0]) | (off[:, 1] > tile_half[1])
      left_own_tile |= alive & off_tile
      time_on_tile += alive_f * (~off_tile).float() * base.step_dt
    else:
      off_tile = torch.zeros(n, dtype=torch.bool, device=device)

    tm = base.termination_manager

    oob = (tm.get_term("out_of_terrain_bounds")
           if "out_of_terrain_bounds" in tm.active_terms
           else torch.zeros_like(dones, dtype=torch.bool))
    tilt = (tm.get_term("fell_over") if "fell_over" in tm.active_terms
            else torch.zeros_like(dones, dtype=torch.bool))
    newly_done = dones.bool() & alive
    fell |= newly_done & tilt
    left_terrain |= newly_done & oob
    alive &= ~newly_done
    if tile_edge == "stop":


      alive &= ~off_tile


    done_ids = dones.bool().nonzero(as_tuple=False).squeeze(-1)
    if done_ids.numel() > 0:
      pin_terrain_level(base, level_row if level_row is not None else 0)


      obs_d, _ = reset_done_envs(base, done_ids, obs)


      obs = TensorDict(obs_d, batch_size=[n])
      _fix_command(term, cur_cmd)
      prev_xy = robot.data.root_link_pos_w[:, :2].clone()

    if not alive.any():
      break

  counted = counted.clamp(min=1.0)
  path_km = path.clamp(min=1e-6)
  ep_vx = (err_vx / counted)
  ep_vy = (err_vy / counted)
  ep_wz = (err_wz / counted)
  c2 = counted2.clamp(min=1.0)
  out = {
    "n_envs": int(base.num_envs),
    "cmd": [float(x) for x in cmd],
    "survival_rate": float((~fell).float().mean()),
    "fell": int(fell.sum()),
    "left_terrain": int(left_terrain.sum()),


    "left_own_tile": (int(left_own_tile.sum()) if tile_half is not None
                      and tile_edge != "ignore" else None),
    "tile_edge_mode": tile_edge if tile_half is not None else "n/a",
    "time_on_own_tile_s_mean": (float(time_on_tile.mean())
                                if tile_half is not None
                                and tile_edge != "ignore" else None),
    "duration_s": float(duration_s),
    "episode_len_s_mean": float(ep_len.mean()),
    "distance_m_mean": float(path.mean()),
    "displacement_m_mean": float(disp.mean()),
    "mae_vx": float((err_vx / counted).mean()),
    "mae_vy": float((err_vy / counted).mean()),
    "mae_wz": float((err_wz / counted).mean()),
    "falls_per_100m": float(fell.float().sum() / (path_km.sum() / 100.0)),

    "mech_power_w_mean": float((power / counted).mean()),
    "energy_per_s": float((power / counted).mean()),
  }


  out["mae_vx_per_ep"] = [float(x) for x in ep_vx.tolist()]
  out["mae_vy_per_ep"] = [float(x) for x in ep_vy.tolist()]
  out["mae_wz_per_ep"] = [float(x) for x in ep_wz.tolist()]
  out["fell_per_ep"] = [bool(x) for x in fell.tolist()]
  out["episode_len_s_per_ep"] = [float(x) for x in ep_len.tolist()]
  out["distance_m_per_ep"] = [float(x) for x in path.tolist()]
  out["displacement_m_per_ep"] = [float(x) for x in disp.tolist()]
  out["mae_vx_std"] = float(ep_vx.std(unbiased=False))
  out["mae_vy_std"] = float(ep_vy.std(unbiased=False))
  out["mae_wz_std"] = float(ep_wz.std(unbiased=False))
  out["mae_vx_max"] = float(ep_vx.max())
  out["mae_vy_max"] = float(ep_vy.max())
  out["mae_wz_max"] = float(ep_wz.max())


  _signed_diagnostics(out, hist_vx, hist_vy, hist_wz, hist_alive, hist_cmd)


  out["scenario"] = "transition" if has_switch else "hold"
  if has_switch:
    out["switch_at_s"] = float(switch_at_s)
    out["cmd_after"] = list(cmd2)
    out["stop_criterion"] = {
      "speed_lt": stop_speed, "yaw_rate_lt": stop_yaw, "hold_s": stop_hold_s,
      "note": "body speed < speed_lt and |wz| < yaw_rate_lt for hold_s seconds",
    }
    out["mae_vx_after"] = float((err2_vx / c2).mean())
    out["mae_vy_after"] = float((err2_vy / c2).mean())
    out["mae_wz_after"] = float((err2_wz / c2).mean())
    out["mae_vx_after_per_ep"] = [float(x) for x in (err2_vx / c2).tolist()]
    out["mae_vy_after_per_ep"] = [float(x) for x in (err2_vy / c2).tolist()]
    out["mae_wz_after_per_ep"] = [float(x) for x in (err2_wz / c2).tolist()]


    out["stopped_count"] = int(stopped.sum())
    out["stop_window_s"] = float(duration_s - float(switch_at_s))
    out["stop_time_s_per_ep"] = [None if np.isnan(x) else float(x)
                                 for x in stop_time.tolist()]
    out["stop_confirm_path_m_per_ep"] = [None if np.isnan(x) else float(x)
                                         for x in stop_path.tolist()]
    out["stop_confirm_disp_m_per_ep"] = [None if np.isnan(x) else float(x)
                                         for x in stop_disp.tolist()]
    out["path_after_switch_m_per_ep"] = [float(x) for x in path_after.tolist()]
    fin = stopped.nonzero(as_tuple=False).squeeze(-1)
    out["stop_time_s_mean"] = (float(stop_time[fin].mean()) if fin.numel() else None)
    out["stop_confirm_path_m_mean"] = (float(stop_path[fin].mean())
                                       if fin.numel() else None)
    out["stop_confirm_disp_m_mean"] = (float(stop_disp[fin].mean())
                                       if fin.numel() else None)
    out["path_after_switch_m_mean"] = float(path_after.mean())
    out["stop_time_note"] = (
      "stop_time_s marks the start of the stable window; "
      "stop_confirm_path_m and stop_confirm_disp_m measure motion until "
      "the stop is confirmed")
    if int(stopped.sum()) < int(base.num_envs):
      out["stop_not_registered_count"] = int(base.num_envs) - int(stopped.sum())
      out["stop_not_registered_note"] = (
        f"stop not confirmed within the observation window "
        f"({out['stop_window_s']:.1f} s) in "
        f"{out['stop_not_registered_count']}/{base.num_envs} episodes "
        "(the criterion did not hold within the window)"
        "")
  return out


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("checkpoint")
  ap.add_argument("--task", default="Mjlab-Velocity-Flat-Unitree-G1")
  ap.add_argument("--num-envs", type=int, default=20)
  ap.add_argument("--level", type=int, default=0,
                  help="terrain row in the training grid (0..num_rows-1)")
  ap.add_argument("--difficulty", type=float, default=None,
                  help="physical terrain difficulty 0..1; overrides --level")
  ap.add_argument("--tile-edge", choices=("mark", "stop", "ignore"),
                  default="mark",
                  help="how to handle leaving the assigned tile: mark, "
                       "stop, or ignore")
  ap.add_argument("--duration", type=float, default=60.0)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--commands", nargs="*", default=DEFAULT_COMMANDS)
  ap.add_argument("--out", default="results_mjlab")
  args = ap.parse_args()


  torch.manual_seed(args.seed)
  np.random.seed(args.seed)
  device = "cuda:0" if torch.cuda.is_available() else "cpu"

  level = args.level
  if args.difficulty is not None:

    from mjlab.tasks.registry import load_env_cfg
    train_cfg = load_env_cfg(args.task, play=False)
    t = getattr(train_cfg.scene, "terrain", None)
    tg = getattr(t, "terrain_generator", None) if t is not None else None
    num_rows = getattr(tg, "num_rows", 1) if tg is not None else 1
    level = row_from_difficulty(num_rows, args.difficulty)
    print(f"[eval] difficulty {args.difficulty} maps to row {level} of {num_rows}")

  env, policy, row = build(args.task, args.checkpoint, args.num_envs,
                           level, device, seed=args.seed)


  tg_built = None
  if env.unwrapped.scene.terrain is not None:
    tg_built = getattr(env.unwrapped.scene.terrain.cfg, "terrain_generator", None)
  difficulty = (difficulty_of_row(tg_built.num_rows, row, tg_built.difficulty_range)
                if tg_built is not None and row is not None else None)
  num_rows_built = getattr(tg_built, "num_rows", None) if tg_built else None

  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  results = []
  for c in args.commands:
    cmd = [float(x) for x in c.split(",")]
    res = rollout(env, policy, cmd, args.duration, level_row=row,
                  tile_edge=args.tile_edge)
    res["task"] = args.task
    res["level"] = level
    res["terrain_row"] = row
    res["num_rows"] = num_rows_built
    res["difficulty"] = difficulty
    res["seed"] = args.seed
    res["checkpoint"] = str(Path(args.checkpoint).resolve())
    res["duration_s"] = args.duration
    results.append(res)
    (out / f"{c.replace(',', '_')}.json").write_text(json.dumps(res, indent=2))
    print(res)


  (out / "run_meta.json").write_text(json.dumps({
    "kind": "run_meta",
    "mjlab_version": importlib.metadata.version("mjlab"),
    "python_executable": sys.executable,
    "task": args.task, "level": level, "terrain_row": row,
    "num_rows": num_rows_built, "difficulty": difficulty,
    "tile_edge": args.tile_edge,
    "seed": args.seed, "num_envs": args.num_envs, "duration_s": args.duration,
    "checkpoint": str(Path(args.checkpoint).resolve()),
    "commands": list(args.commands),
  }, indent=2))

  print()
  print("| Command | Survived | Path, m | Displacement, m | MAE vx | MAE vy | MAE wz | Falls/100 m |")
  print("|---|---|---|---|---|---|---|---|")
  for r in results:
    c = r["cmd"]
    print(f"| {c[0]:.1f}, {c[1]:.1f}, {c[2]:.1f} | {r['survival_rate']*100:.0f}% | "
          f"{r['distance_m_mean']:.1f} | {r['displacement_m_mean']:.1f} | "
          f"{r['mae_vx']:.3f} | {r['mae_vy']:.3f} | "
          f"{r['mae_wz']:.3f} | {r['falls_per_100m']:.1f} |")


if __name__ == "__main__":
  main()
