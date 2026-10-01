#!/usr/bin/env python3
"""Evaluate a fixed policy across a continuous multi-terrain route.

Cross-track steering adjusts only the yaw-rate command; the policy receives
no terrain label.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from g1_locomotion import harness  # noqa: E402

TASK = "G1-Mix3-Deploy-NoCmdCurriculum-SymLoss-YawW4"
FINAL_TASK = "G1-FinalRoute-Stairs5-Deploy-NoCmdCurriculum-SymLoss-YawW4"
FINAL10_TASK = "G1-FinalRoute-Stairs10-Deploy-NoCmdCurriculum-SymLoss-YawW4"
EXPECTED_MJLAB_VERSION = "1.6.0"
EXPECTED_LAYOUTS = {
  TASK: ["flat", "random_rough", "hf_pyramid_slope", "hf_pyramid_slope_inv"],
  FINAL_TASK: ["flat", "random_rough", "hf_pyramid_slope",
               "pyramid_stairs", "flat_landing"],
  FINAL10_TASK: ["flat", "random_rough", "hf_pyramid_slope",
                 "pyramid_stairs", "flat_landing"],
}
SECTION_LABELS = {
  TASK: ["flat", "random_rough", "slope"],
  FINAL_TASK: ["flat", "random_rough", "slope", "stairs"],
  FINAL10_TASK: ["flat", "random_rough", "slope", "stairs"],
}


def yaw_from_quat_wxyz(q: torch.Tensor) -> torch.Tensor:
  w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
  return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap(a: torch.Tensor) -> torch.Tensor:
  return torch.remainder(a + math.pi, 2.0 * math.pi) - math.pi


def route_layout(base, row: int, task: str) -> dict:
  terrain = base.scene.terrain
  cfg = terrain.cfg.terrain_generator
  keys = list(cfg.sub_terrains)
  expected = EXPECTED_LAYOUTS.get(task)
  if expected is None or keys != expected:
    raise RuntimeError(f"route requires {expected}; task has {keys}")
  if not cfg.curriculum or terrain.terrain_origins.shape[1] != len(expected):
    raise RuntimeError("route requires one ordered column per sub-terrain")
  centers = terrain.terrain_origins[row, :, 1].tolist()
  width = float(cfg.size[1])
  if any(abs(centers[i + 1] - centers[i] - width) > 1e-4
         for i in range(len(expected) - 1)):
    raise RuntimeError(f"terrain columns are not contiguous: {centers}")
  seams = [(centers[i] + centers[i + 1]) / 2
           for i in range(len(expected) - 1)]
  return {"keys": keys, "sections": SECTION_LABELS[task],
          "centers_y": centers, "seams_y": seams,
          "tile_size_m": list(cfg.size), "terrain_row": row,
          "difficulty": harness.difficulty_of_row(cfg.num_rows, row, cfg.difficulty_range)}


def configure_start(base, layout: dict, offset: float, xy_jitter: float,
                    yaw_jitter: float) -> dict:
  width = layout["tile_size_m"][1]
  if not 0 < offset - xy_jitter and offset + xy_jitter < width:
    raise ValueError("start offset and jitter must keep all robots on flat tile")
  start_y = layout["seams_y"][0] - offset
  dy = start_y - layout["centers_y"][0]
  term = base.event_manager.get_term_cfg("reset_base")
  pose = dict(term.params["pose_range"])
  pose["x"] = (-xy_jitter, xy_jitter)
  pose["y"] = (dy - xy_jitter, dy + xy_jitter)
  pose["yaw"] = (math.pi / 2 - yaw_jitter, math.pi / 2 + yaw_jitter)
  term.params["pose_range"] = pose
  base.scene.terrain.terrain_types[:] = 0
  return {"start_y_nominal": start_y, "offset_m": offset,
          "xy_jitter_m": xy_jitter, "yaw_jitter_rad": yaw_jitter,
          "pose_range": {key: list(value) for key, value in pose.items()}}


def evaluate(env, policy, row: int, layout: dict, command: tuple[float, float, float],
             duration_s: float, steer: str, lookahead_m: float,
             yaw_gain: float, yaw_limit: float, turn_offset_m: float) -> dict:
  base = env.unwrapped
  n = base.num_envs
  device = base.device
  cmd = torch.tensor(command, dtype=torch.float32, device=device)
  term = base.command_manager._terms["twist"]
  harness._fix_command(term, command)
  harness.pin_terrain_level(base, row)
  obs, _ = env.reset()
  harness._fix_command(term, command)
  harness.assert_level_held(base, row)

  robot = base.scene["robot"]
  starts = robot.data.root_link_pos_w[:, :2].clone()
  prev = starts.clone()
  row_x = float(base.scene.terrain.terrain_origins[row, 0, 0])
  half_x = layout["tile_size_m"][0] / 2
  seams = layout["seams_y"]
  seam0, finish_y = seams[0], seams[-1]
  sections = layout["sections"]
  n_sections = len(sections)
  section_edges = torch.tensor(seams[:-1], device=device)
  if steer == "cross_track_turns" and n_sections != 4:
    raise ValueError("turns-at-seams protocol requires the four-section final route")
  turn_targets = torch.tensor((0.0, 1.0, -1.0, 0.0), device=device)
  if not bool((starts[:, 1] < seam0).all()):
    raise RuntimeError("some robots spawned beyond the flat/rough seam")
  if not bool(((starts[:, 0] - row_x).abs() < half_x).all()):
    raise RuntimeError("some robots spawned outside the requested terrain row")

  active = torch.ones(n, dtype=torch.bool, device=device)
  reached = {section: torch.zeros_like(active) for section in sections[1:]}
  passed = torch.zeros_like(active)
  fell = torch.zeros_like(active)
  left_row = torch.zeros_like(active)
  other_done = torch.zeros_like(active)
  length = torch.zeros(n, device=device)
  path = torch.zeros(n, device=device)
  elapsed = torch.zeros(n, device=device)
  section_steps = torch.zeros((n_sections, n), device=device)
  section_error = torch.zeros((n_sections, 3, n), device=device)
  section_x_sum = torch.zeros((n_sections, n), device=device)
  overall_steps = torch.zeros(n, device=device)
  overall_error = torch.zeros((3, n), device=device)
  last_y = starts[:, 1].clone()
  y_history = []
  error_history = []
  live_history = []
  fall_history = []
  x_abs_max = torch.zeros(n, device=device)
  yaw_command_abs_sum = torch.zeros(n, device=device)
  term_steps = base.termination_manager

  for _ in range(math.ceil(duration_s / base.step_dt)):
    harness._fix_command(term, command)
    cmd_now = cmd.expand(n, -1).clone()
    if steer in ("cross_track", "cross_track_turns"):
      xy_before = robot.data.root_link_pos_w[:, :2]
      yaw = yaw_from_quat_wxyz(robot.data.root_link_quat_w)
      target_x = torch.full_like(yaw, row_x)
      if steer == "cross_track_turns":
        current_section = torch.bucketize(
          xy_before[:, 1].contiguous(), section_edges
        )
        target_x += turn_targets[current_section] * turn_offset_m
      desired_yaw = torch.atan2(
        torch.full_like(yaw, lookahead_m), target_x - xy_before[:, 0]
      )
      yaw_rate = (cmd[2] + yaw_gain * wrap(desired_yaw - yaw)).clamp(
        -yaw_limit, yaw_limit
      )
      term.vel_command_b[:, 2] = yaw_rate
      cmd_now[:, 2] = yaw_rate
    with torch.no_grad():
      action = policy(obs)
      obs, _, dones, _ = env.step(action)
    live = active.clone()
    xy = robot.data.root_link_pos_w[:, :2]
    length += live.float() * base.step_dt
    path += live.float() * torch.nan_to_num(torch.norm(xy - prev, dim=1), nan=0.0)
    prev = xy.clone()
    y = xy[:, 1]
    last_y = torch.where(live, y, last_y)
    x_abs_max = torch.maximum(x_abs_max, (xy[:, 0] - row_x).abs() * live.float())
    yaw_command_abs_sum += cmd_now[:, 2].abs() * live.float()
    for section_i, section in enumerate(sections[1:], start=1):
      reached[section] |= live & (y >= seams[section_i - 1])

    segment = torch.bucketize(y.contiguous(), section_edges)
    lin = robot.data.root_link_lin_vel_b
    ang = robot.data.root_link_ang_vel_b
    err = torch.stack(((lin[:, 0] - cmd_now[:, 0]).abs(),
                       (lin[:, 1] - cmd_now[:, 1]).abs(),
                       (ang[:, 2] - cmd_now[:, 2]).abs()))
    overall_steps += live.float()
    overall_error += err * live.float()
    for s in range(n_sections):
      mask = live & (segment == s)
      section_steps[s] += mask.float()
      section_error[s] += err * mask.float()
      section_x_sum[s] += (xy[:, 0] - row_x) * mask.float()

    tilted = (term_steps.get_term("fell_over")
              if "fell_over" in term_steps.active_terms else torch.zeros_like(dones))
    just_done = live & dones.bool()
    fell |= just_done & tilted.bool()
    other_done |= just_done & ~tilted.bool()
    y_history.append(y.detach().clone())
    error_history.append(err.detach().clone())
    live_history.append(live.detach().clone())
    fall_history.append((just_done & tilted.bool()).detach().clone())
    out_of_row = live & ((xy[:, 0] - row_x).abs() > half_x)
    left_row |= out_of_row
    just_passed = live & (y >= finish_y) & ~out_of_row & ~just_done
    passed |= just_passed
    elapsed = torch.where(just_passed, length, elapsed)
    active &= ~(just_done | out_of_row | just_passed)

    done_ids = dones.bool().nonzero(as_tuple=False).squeeze(-1)
    if done_ids.numel():
      harness.pin_terrain_level(base, row)
      new_obs, _ = harness.reset_done_envs(base, done_ids, obs)
      obs = TensorDict(new_obs, batch_size=[n])
      harness._fix_command(term, command)
      prev = robot.data.root_link_pos_w[:, :2].clone()
    if not bool(active.any()):
      break

  # section_error is [section, axis, env]; align the denominator to sections.
  mae = torch.where(section_steps.unsqueeze(1) > 0,
                    section_error / section_steps.unsqueeze(1).clamp(min=1),
                    torch.full_like(section_error, float("nan")))
  def mean_valid(tensor):
    valid = tensor[torch.isfinite(tensor)]
    return float(valid.mean()) if valid.numel() else None

  overall_mae = overall_error / overall_steps.unsqueeze(0).clamp(min=1)
  hist_y = torch.stack(y_history)
  hist_err = torch.stack(error_history)
  hist_live = torch.stack(live_history)
  hist_fall = torch.stack(fall_history)
  window_steps = math.ceil(1.0 / base.step_dt)
  seam_metrics = {}
  for seam_i, seam_y in enumerate(seams[:-1]):
    name = f"{sections[seam_i]}_to_{sections[seam_i + 1]}"
    crossed_steps = hist_live & (hist_y >= seam_y)
    crossed = crossed_steps.any(dim=0)
    first_steps = torch.argmax(crossed_steps.int(), dim=0)
    per_ep_mae = torch.full((3, n), float("nan"), device=device)
    fall_in_time_window = torch.zeros(n, dtype=torch.bool, device=device)
    for env_i in crossed.nonzero(as_tuple=False).flatten().tolist():
      step = int(first_steps[env_i])
      lo = max(0, step - window_steps)
      hi = min(hist_y.shape[0], step + window_steps + 1)
      valid = hist_live[lo:hi, env_i]
      if bool(valid.any()):
        per_ep_mae[:, env_i] = hist_err[lo:hi, :, env_i][valid].mean(dim=0)
      fall_in_time_window[env_i] = hist_fall[lo:hi, env_i].any()
    fall_within_1m = (hist_fall & ((hist_y - seam_y).abs() <= 1.0)).any(dim=0)
    seam_metrics[name] = {
      "crossed": int(crossed.sum()),
      "fell_within_1s_of_crossing": int(fall_in_time_window.sum()),
      "fell_within_1m_of_seam": int(fall_within_1m.sum()),
      "window_mae": {axis: mean_valid(per_ep_mae[axis_i])
                     for axis_i, axis in enumerate(("vx", "vy", "wz"))},
    }

  return {
    "n_envs": n, "command": list(command), "duration_s": duration_s,
    "steer": steer, "lookahead_m": lookahead_m,
    "yaw_gain": yaw_gain, "yaw_limit_rad_s": yaw_limit,
    "turn_offset_m": turn_offset_m,
    "passed": int(passed.sum()), "fell": int(fell.sum()),
    "left_row": int(left_row.sum()), "other_termination": int(other_done.sum()),
    "reached_sections": {key: int(value.sum()) for key, value in reached.items()},
    "reached_rough": int(reached.get("random_rough", torch.zeros_like(active)).sum()),
    "reached_slope": int(reached.get("slope", torch.zeros_like(active)).sum()),
    "reached_stairs": int(reached.get("stairs", torch.zeros_like(active)).sum()),
    "not_finished": int((~passed & ~fell & ~left_row & ~other_done).sum()),
    "passed_per_ep": passed.tolist(), "fell_per_ep": fell.tolist(),
    "left_row_per_ep": left_row.tolist(),
    "episode_len_s_per_ep": length.tolist(),
    "path_m_per_ep": path.tolist(),
    "final_y_per_ep": last_y.tolist(),
    "max_abs_cross_track_m_per_ep": x_abs_max.tolist(),
    "mean_abs_yaw_command_rad_s_per_ep":
      (yaw_command_abs_sum / (length / base.step_dt).clamp(min=1)).tolist(),
    "time_to_finish_s_per_ep": [float(x) if ok else None
                                for x, ok in zip(elapsed.tolist(), passed.tolist())],
    "path_m_mean": float(path.mean()),
    "overall_mae": {axis: mean_valid(overall_mae[axis_i])
                    for axis_i, axis in enumerate(("vx", "vy", "wz"))},
    "seam_metrics": seam_metrics,
    "mae_by_section": {
      section: {axis: mean_valid(mae[section_i, axis_i])
                for axis_i, axis in enumerate(("vx", "vy", "wz"))}
      for section_i, section in enumerate(sections)
    },
    "section_steps_per_ep": {section: section_steps[i].tolist()
                             for i, section in enumerate(sections)},
    "mean_cross_track_by_section_m": {
      section: mean_valid(torch.where(
        section_steps[i] > 0,
        section_x_sum[i] / section_steps[i].clamp(min=1),
        torch.full_like(section_steps[i], float("nan")),
      ))
      for i, section in enumerate(sections)
    },
  }


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument("checkpoint")
  ap.add_argument("--task", default=TASK)
  ap.add_argument("--row", type=int, default=4)
  ap.add_argument("--num-envs", type=int, default=20)
  ap.add_argument("--duration", type=float, default=60.0)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--start-offset", type=float, default=1.0)
  ap.add_argument("--xy-jitter", type=float, default=0.15)
  ap.add_argument("--yaw-jitter", type=float, default=0.05)
  ap.add_argument("--cmd", default="0.5,0,0")
  ap.add_argument("--steer", choices=("none", "cross_track", "cross_track_turns"),
                  default="none")
  ap.add_argument("--lookahead", type=float, default=4.0)
  ap.add_argument("--yaw-gain", type=float, default=1.5)
  ap.add_argument("--yaw-limit", type=float, default=0.5)
  ap.add_argument("--turn-offset", type=float, default=1.0)
  ap.add_argument("--out", required=True)
  args = ap.parse_args()

  installed_mjlab = importlib.metadata.version("mjlab")
  if installed_mjlab != EXPECTED_MJLAB_VERSION:
    ap.error(f"expected mjlab {EXPECTED_MJLAB_VERSION}, got {installed_mjlab}; "
             "use the pinned container environment")

  torch.manual_seed(args.seed)
  np.random.seed(args.seed)
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  cmd = tuple(float(x) for x in args.cmd.split(","))
  if len(cmd) != 3:
    ap.error("--cmd must have vx,vy,wz")
  if args.lookahead <= 0 or args.yaw_gain <= 0 or args.yaw_limit <= 0:
    ap.error("steering parameters must be positive")
  if args.turn_offset < 0 or args.turn_offset >= 2.0:
    ap.error("--turn-offset must be in [0, 2) m to remain inside the 8 m row")
  started = dt.datetime.now().astimezone().isoformat(timespec="seconds")
  env, policy, row = harness.build(args.task, args.checkpoint, args.num_envs,
                                    args.row, device, seed=args.seed)
  layout = route_layout(env.unwrapped, row, args.task)
  start = configure_start(env.unwrapped, layout, args.start_offset,
                          args.xy_jitter, args.yaw_jitter)
  result = evaluate(env, policy, row, layout, cmd, args.duration,
                    args.steer, args.lookahead, args.yaw_gain, args.yaw_limit,
                    args.turn_offset)
  result.update({"kind": "route_result", "task": args.task,
                 "checkpoint": str(Path(args.checkpoint).resolve()),
                 "mjlab_version": installed_mjlab,
                 "python_executable": sys.executable,
                 "seed": args.seed, "device": device, "layout": layout,
                 "start": start, "started": started,
                 "finished": dt.datetime.now().astimezone().isoformat(timespec="seconds")})
  output = Path(args.out)
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(json.dumps(result, indent=2, ensure_ascii=False))
  print(f"[route-eval] passed={result['passed']}/{args.num_envs}, "
        f"fell={result['fell']}, left_row={result['left_row']}, "
        f"not_finished={result['not_finished']}; {output}")


if __name__ == "__main__":
  main()
