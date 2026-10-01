#!/usr/bin/env python3
"""Route-junction transition eval: flat -> random_rough and back.

Each episode starts on a tile of one type, `--start-offset` meters from the
boundary with a tile of the other type in the SAME row, and drives a
"forward" command across the boundary. Metrics: fall before / after crossing
the boundary, failure to reach the boundary, path length after crossing, and
MAE vx/vy/wz before and after.

Layout (verified in code, not assumed). mjlab c2e1e06 in curriculum mode
gives one column per `sub_terrains` key
(terrains/terrain_generator.py: `_num_cols = len(sub_terrains)`); row
(difficulty) runs along x, column (type) along y (`_get_sub_terrain_position`:
rel_x = row*size[0], rel_y = col*size[1], grid centered at 0). So a
flat|random_rough boundary within one row only occurs along y, and the robot
is placed with heading ±90° (along ±y). The default task is
`G1-FlatRoughStrip-Deploy-NoCmdCurriculum-SymLoss-YawW4`
(src/g1_locomotion/tasks/stage2_gap_eval_tasks.py): 4 flat columns, then 4
random_rough columns, boundary at y=0; 32 m of the same type beyond the
boundary.

The harness (src/g1_locomotion/harness.py) is unchanged: build(),
pin_terrain_level(), assert_level_held(), reset_done_envs(), _fix_command()
are imported from it as-is (curriculum manager removed, row pinned,
randomize_terrain removed, out_of_terrain_bounds returned as time_out,
auto_reset=False, partial reset without corrupting neighbor history).

The only intervention in the environment beyond build(): parameters of the
`reset_base` reset event (pose_range) — start at a given point with a given
heading — and `terrain_types` of all envs set to the start column.

Episode = one environment (after termination, an env stops accumulating
metrics). Crossing position is `root_link_pos_w` (pelvis).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

ROOT = Path(__file__).resolve().parents[2]
EXP3 = ROOT / "src"
sys.path.insert(0, str(EXP3))

from g1_locomotion import harness as H  # noqa: E402  (also registers the tasks)

DEFAULT_TASK = "G1-FlatRoughStrip-Deploy-NoCmdCurriculum-SymLoss-YawW4"


def _type_of(key: str) -> str:
  for t in ("random_rough", "flat"):
    if key.startswith(t):
      return t
  return key


def layout(base) -> dict:
  """Flat|random_rough boundary from the generator and from terrain_origins, cross-checked."""
  terrain = base.scene.terrain
  tg = terrain.cfg.terrain_generator
  keys = list(tg.sub_terrains.keys())
  types = [_type_of(k) for k in keys]
  origins = terrain.terrain_origins            # (rows, cols, 3)
  n_rows, n_cols = origins.shape[:2]
  if n_cols != len(keys):
    raise RuntimeError(f"columns {n_cols} != sub_terrains keys {len(keys)}")
  switches = [i for i in range(1, n_cols) if types[i] != types[i - 1]]
  if len(switches) != 1 or types[switches[0] - 1] != "flat" \
     or types[switches[0]] != "random_rough":
    raise RuntimeError(f"expected layout flat..|random_rough.., got {types}")
  nf = switches[0]
  sx, sy = float(tg.size[0]), float(tg.size[1])
  y_b = -n_cols * sy / 2.0 + nf * sy
  # Cross-check against actual tile centers (origin = corner + size/2).
  y_b_orig = float((origins[0, nf - 1, 1] + origins[0, nf, 1]) / 2.0)
  if abs(y_b - y_b_orig) > 1e-4:
    raise RuntimeError(f"boundary from formula {y_b} != from terrain_origins {y_b_orig}")
  return {"keys": keys, "types": types, "n_rows": n_rows, "n_cols": n_cols,
          "flat_cols": list(range(nf)), "rough_cols": list(range(nf, n_cols)),
          "boundary_y": y_b, "tile_size": [sx, sy],
          "border_width": float(tg.border_width),
          "difficulty_range": list(tg.difficulty_range)}


def set_start(base, lay: dict, direction: str, offset: float,
              xy_jitter: tuple[float, float], yaw_jitter: float) -> dict:
  """Start column, point, and heading. Returns a start description for JSON."""
  y_b = lay["boundary_y"]
  if direction == "f2r":
    col, sign, yaw = lay["flat_cols"][-1], +1.0, math.pi / 2
  elif direction == "r2f":
    col, sign, yaw = lay["rough_cols"][0], -1.0, -math.pi / 2
  else:
    raise ValueError(direction)
  terrain = base.scene.terrain
  col_center_y = float(terrain.terrain_origins[0, col, 1])
  start_y = y_b - sign * offset
  dy = start_y - col_center_y
  jx, jy = xy_jitter
  term = base.event_manager.get_term_cfg("reset_base")
  pr = dict(term.params["pose_range"])
  pr["x"] = (-jx, jx)
  pr["y"] = (dy - jy, dy + jy)
  pr["yaw"] = (yaw - yaw_jitter, yaw + yaw_jitter)
  term.params["pose_range"] = pr
  terrain.terrain_types[:] = col
  return {"direction": direction, "start_col": col,
          "start_type": lay["types"][col], "sign_y": sign,
          "start_y_nominal": start_y, "offset_from_boundary_m": offset,
          "yaw_nominal_rad": yaw, "pose_range": {k: list(v) for k, v in pr.items()}}


def run_episode_batch(env, policy, cmd, duration_s, row, lay, start, cmd_name="twist"):
  base = env.unwrapped
  dev = base.device
  n = base.num_envs
  steps = int(duration_s / base.step_dt)
  term = base.command_manager._terms[cmd_name]
  cmd_t = torch.tensor(cmd, dtype=torch.float32, device=dev)
  y_b = lay["boundary_y"]
  sign = start["sign_y"]
  half_x = lay["tile_size"][0] / 2.0

  H._fix_command(term, cmd)
  H.pin_terrain_level(base, row)
  obs, _ = env.reset()
  H._fix_command(term, cmd)
  H.assert_level_held(base, row)

  robot = base.scene["robot"]
  row_center_x = float(base.scene.terrain.terrain_origins[row, 0, 0])
  xy0 = robot.data.root_link_pos_w[:, :2].clone()
  prev_xy = xy0.clone()

  z = lambda: torch.zeros(n, device=dev)  # noqa: E731
  alive = torch.ones(n, dtype=torch.bool, device=dev)
  crossed = torch.zeros(n, dtype=torch.bool, device=dev)
  fell = torch.zeros(n, dtype=torch.bool, device=dev)
  fell_after = torch.zeros(n, dtype=torch.bool, device=dev)
  oob = torch.zeros(n, dtype=torch.bool, device=dev)
  other_term = torch.zeros(n, dtype=torch.bool, device=dev)
  left_row = torch.zeros(n, dtype=torch.bool, device=dev)
  left_row_before_cross = torch.zeros(n, dtype=torch.bool, device=dev)
  recrossed = torch.zeros(n, dtype=torch.bool, device=dev)
  t_cross = torch.full((n,), float("nan"), device=dev)
  t_fall = torch.full((n,), float("nan"), device=dev)
  depth_at_fall = torch.full((n,), float("nan"), device=dev)
  max_depth = torch.full((n,), -1e9, device=dev)
  ep_len = z()
  path_before, path_after = z(), z()
  e_b = {k: z() for k in ("vx", "vy", "wz")}
  e_a = {k: z() for k in ("vx", "vy", "wz")}
  c_b, c_a = z(), z()
  on_row_after = z()
  t = 0.0

  for _ in range(steps):
    H._fix_command(term, cmd)
    with torch.no_grad():
      actions = policy(obs)
      obs, _, dones, _ = env.step(actions)
    t += base.step_dt
    lin = robot.data.root_link_lin_vel_b
    ang = robot.data.root_link_ang_vel_b
    xy = robot.data.root_link_pos_w[:, :2]
    depth = (xy[:, 1] - y_b) * sign            # >0 means past the boundary
    af = alive.float()

    # The step where the pelvis is first past the boundary already counts as "after".
    new_cross = alive & ~crossed & (depth > 0)
    t_cross = torch.where(new_cross, torch.full_like(t_cross, t), t_cross)
    crossed |= new_cross
    recrossed |= alive & crossed & (depth < 0)
    max_depth = torch.where(alive, torch.maximum(max_depth, depth), max_depth)

    step_len = torch.nan_to_num(torch.norm(xy - prev_xy, dim=1), nan=0.0)
    before_f = af * (~crossed).float()
    after_f = af * crossed.float()
    path_before += before_f * step_len
    path_after += after_f * step_len
    errs = {"vx": (lin[:, 0] - cmd_t[0]).abs(), "vy": (lin[:, 1] - cmd_t[1]).abs(),
            "wz": (ang[:, 2] - cmd_t[2]).abs()}
    for k, v in errs.items():
      e_b[k] += before_f * v
      e_a[k] += after_f * v
    c_b += before_f
    c_a += after_f
    ep_len += af * base.step_dt

    off_row = (xy[:, 0] - row_center_x).abs() > half_x
    left_row |= alive & off_row
    left_row_before_cross |= alive & off_row & ~crossed
    on_row_after += after_f * (~off_row).float() * base.step_dt

    tm = base.termination_manager
    tilt = tm.get_term("fell_over") if "fell_over" in tm.active_terms \
      else torch.zeros_like(dones, dtype=torch.bool)
    oob_t = tm.get_term("out_of_terrain_bounds") \
      if "out_of_terrain_bounds" in tm.active_terms \
      else torch.zeros_like(dones, dtype=torch.bool)
    newly = dones.bool() & alive
    nf_ = newly & tilt
    fell |= nf_
    fell_after |= nf_ & crossed
    t_fall = torch.where(nf_, torch.full_like(t_fall, t), t_fall)
    depth_at_fall = torch.where(nf_, depth, depth_at_fall)
    oob |= newly & oob_t & ~tilt
    other_term |= newly & ~tilt & ~oob_t
    alive &= ~newly
    prev_xy = xy.clone()

    done_ids = dones.bool().nonzero(as_tuple=False).squeeze(-1)
    if done_ids.numel() > 0:
      H.pin_terrain_level(base, row)
      obs_d, _ = H.reset_done_envs(base, done_ids, obs)
      obs = TensorDict(obs_d, batch_size=[n])
      H._fix_command(term, cmd)
      prev_xy = robot.data.root_link_pos_w[:, :2].clone()
    if not alive.any():
      break

  fell_before = fell & ~fell_after
  passed = crossed & ~fell
  not_reached = ~crossed & ~fell

  def L(x):
    return [None if (isinstance(v, float) and math.isnan(v)) else v
            for v in x.tolist()]

  def mae(e, c):
    per = torch.where(c > 0, e / c.clamp(min=1.0), torch.full_like(e, float("nan")))
    good = per[~torch.isnan(per)]
    return (float(good.mean()) if good.numel() else None), L(per)

  out = {
    "n_episodes": n, "cmd": list(cmd), "duration_s": duration_s,
    "passed_no_fall": int(passed.sum()),
    "fell_before_cross": int(fell_before.sum()),
    "fell_after_cross": int(fell_after.sum()),
    "not_reached_boundary": int(not_reached.sum()),
    "fell_total": int(fell.sum()),
    "crossed": int(crossed.sum()),
    "left_terrain_timeout": int(oob.sum()),
    "other_termination": int(other_term.sum()),
    "recrossed_back": int(recrossed.sum()),
    "left_row": int(left_row.sum()),
    "left_row_before_cross": int(left_row_before_cross.sum()),
    "time_to_cross_s_mean": (float(t_cross[crossed].mean()) if crossed.any() else None),
    "path_before_m_mean": float(path_before.mean()),
    "path_after_m_mean": float(path_after[crossed].mean()) if crossed.any() else None,
    "max_depth_after_m_mean": float(max_depth[crossed].mean()) if crossed.any() else None,
    "time_on_row_after_s_mean": float(on_row_after[crossed].mean()) if crossed.any() else None,
    "episode_len_s_mean": float(ep_len.mean()),
  }
  for k in ("vx", "vy", "wz"):
    out[f"mae_{k}_before"], out[f"mae_{k}_before_per_ep"] = mae(e_b[k], c_b)
    out[f"mae_{k}_after"], out[f"mae_{k}_after_per_ep"] = mae(e_a[k], c_a)
  out.update({
    "passed_per_ep": L(passed), "fell_per_ep": L(fell),
    "fell_after_cross_per_ep": L(fell_after), "crossed_per_ep": L(crossed),
    "time_to_cross_s_per_ep": L(t_cross), "time_fall_s_per_ep": L(t_fall),
    "depth_at_fall_m_per_ep": L(depth_at_fall),
    "path_before_m_per_ep": L(path_before), "path_after_m_per_ep": L(path_after),
    "max_depth_m_per_ep": L(max_depth), "episode_len_s_per_ep": L(ep_len),
    "start_xy_per_ep": xy0.tolist(),
  })
  return out


def fmt(x, p=3):
  return "—" if x is None else f"{x:.{p}f}"


def write_summary(out_dir: Path) -> None:
  runs = [json.loads(p.read_text()) for p in sorted(out_dir.glob("row*_*.json"))]
  if not runs:
    return
  meta = json.loads((out_dir / "run_meta.json").read_text())
  lay = meta["layout"]
  lines = [
    "# Route junction flat <-> random_rough transition eval",
    "",
    f"- Checkpoint: `{meta['checkpoint']}`",
    f"- Task: `{meta['task']}`; script: `src/g1_locomotion/transition_eval.py`; "
    f"device {meta['device']}; seed {meta['seed']}",
    f"- Episodes per cell {meta['num_envs']}, {meta['duration_s']:.0f} s, "
    f"command {meta['cmd']} (body frame), start {meta['start_offset_m']} m from the boundary",
    f"- Layout along y (columns, +y direction): {lay['types']}; boundary y={lay['boundary_y']:.2f}; "
    f"tile size {lay['tile_size']} m; field border {lay['border_width']} m",
    f"- Started/finished: {meta['started']} / {meta.get('finished', '')}",
    "",
    "Per-episode outcomes are mutually exclusive: passed = crossed the "
    "boundary and did not fall for the whole episode; fell before / after = "
    "fall relative to the pelvis's first crossing of the boundary; "
    "not reached = did not cross and did not fall.",
    "",
    "| Row (diff.) | Direction | Passed w/o fall | Fell before | Fell after | Not reached | "
    "t crossing, s | Path after, m | Depth after, m | MAE vx before | MAE vx after | "
    "MAE wz before | MAE wz after | Left row | Recrossed back |",
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
  ]
  for r in runs:
    n = r["n_episodes"]
    lines.append(
      f"| {r['terrain_row']} ({r['difficulty']:.3f}) | {r['start']['direction']} "
      f"({r['start']['start_type']}→) | {r['passed_no_fall']}/{n} | "
      f"{r['fell_before_cross']}/{n} | {r['fell_after_cross']}/{n} | "
      f"{r['not_reached_boundary']}/{n} | {fmt(r['time_to_cross_s_mean'], 1)} | "
      f"{fmt(r['path_after_m_mean'], 1)} | {fmt(r['max_depth_after_m_mean'], 1)} | "
      f"{fmt(r['mae_vx_before'])} | {fmt(r['mae_vx_after'])} | "
      f"{fmt(r['mae_wz_before'])} | {fmt(r['mae_wz_after'])} | "
      f"{r['left_row']}/{n} | {r['recrossed_back']}/{n} |")
  lines += ["", "No pass/fail verdict is rendered here; this is a raw metrics report.", ""]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main():
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument("checkpoint")
  ap.add_argument("--task", default=DEFAULT_TASK)
  ap.add_argument("--rows", type=int, nargs="+", default=[4, 9],
                  help="training grid rows (0..num_rows-1)")
  ap.add_argument("--directions", nargs="+", default=["f2r", "r2f"],
                  choices=["f2r", "r2f"])
  ap.add_argument("--cmd", default="0.5,0,0")
  ap.add_argument("--num-envs", type=int, default=20)
  ap.add_argument("--duration", type=float, default=60.0)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--start-offset", type=float, default=2.5,
                  help="start distance (pelvis) from the boundary, m")
  ap.add_argument("--xy-jitter", type=float, nargs=2, default=[0.5, 0.25],
                  help="start jitter half-width along x and y, m")
  ap.add_argument("--yaw-jitter", type=float, default=0.1,
                  help="heading jitter half-width, rad")
  ap.add_argument("--out", required=True)
  ap.add_argument("--summary-only", action="store_true")
  args = ap.parse_args()

  out = Path(args.out)
  if args.summary_only:
    write_summary(out)
    return
  out.mkdir(parents=True, exist_ok=True)
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  cmd = [float(x) for x in args.cmd.split(",")]
  started = _dt.datetime.now().astimezone().isoformat(timespec="seconds")

  env, policy, _ = H.build(args.task, args.checkpoint, args.num_envs,
                           args.rows[0], device, seed=args.seed)
  base = env.unwrapped
  lay = layout(base)
  tg = base.scene.terrain.cfg.terrain_generator
  print(f"[transition] layout: {lay['types']}, boundary y={lay['boundary_y']}")

  meta = {"kind": "run_meta", "protocol": "transition_eval", "task": args.task,
          "checkpoint": str(Path(args.checkpoint).resolve()), "device": device,
          "seed": args.seed, "num_envs": args.num_envs, "duration_s": args.duration,
          "cmd": cmd, "rows": args.rows, "directions": args.directions,
          "start_offset_m": args.start_offset, "xy_jitter_m": args.xy_jitter,
          "yaw_jitter_rad": args.yaw_jitter, "layout": lay, "started": started}
  (out / "run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

  for row in args.rows:
    for d in args.directions:
      start = set_start(base, lay, d, args.start_offset, tuple(args.xy_jitter),
                        args.yaw_jitter)
      res = run_episode_batch(env, policy, cmd, args.duration, row, lay, start)
      res.update({"task": args.task, "terrain_row": row,
                  "difficulty": H.difficulty_of_row(tg.num_rows, row, tg.difficulty_range),
                  "start": start, "seed": args.seed,
                  "checkpoint": meta["checkpoint"]})
      (out / f"row{row}_{d}.json").write_text(json.dumps(res, indent=2))
      print(f"[transition] row {row} {d}: passed {res['passed_no_fall']}/{res['n_episodes']}, "
            f"fell before {res['fell_before_cross']}, after {res['fell_after_cross']}, "
            f"not reached {res['not_reached_boundary']}; path after "
            f"{fmt(res['path_after_m_mean'], 1)} m; MAE vx before/after "
            f"{fmt(res['mae_vx_before'])}/{fmt(res['mae_vx_after'])}")

  meta["finished"] = _dt.datetime.now().astimezone().isoformat(timespec="seconds")
  (out / "run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
  write_summary(out)


if __name__ == "__main__":
  main()
