#!/usr/bin/env python3
"""Open-loop drift: lateral drift and heading drift under a "straight" command.

The command (vx, 0, 0) is held for the whole episode with no position
feedback — exactly like an operator pressing "forward" without steering
corrections. This measures where the robot drifts to.

Episode = one environment: from start until it covers `--path-max` meters of
PATH (pelvis trajectory length, sum of xy increments) or `--max-time`
seconds, whichever comes first; a fall (any non-time_out termination) ends
the episode.

For each episode, at path marks `--marks` (default 10 and 20 m):
  * lateral offset y — projection of the pelvis offset from the start onto
    the axis perpendicular to the INITIAL heading (start frame: x forward,
    y left); start heading comes from the standard reset_base, hence the
    start frame;
  * heading drift — unwrapped (accumulated step-by-step with increments
    wrapped into (-pi, pi]) body yaw minus the initial yaw; heading from the
    `root_link_quat_w` quaternion (wxyz).
A mark is taken at the first step where path >= the mark. Summary by
magnitude: median and p90 among episodes that reached the mark (the reached
count is reported).

The harness (src/g1_locomotion/harness.py) is unchanged: build(),
row_from_difficulty(), pin_terrain_level(), assert_level_held(),
reset_done_envs(), _fix_command() are imported from it as-is (curriculum
removed, row pinned, randomize_terrain removed, out_of_terrain_bounds
returned as time_out, auto_reset=False, partial reset without corrupting
neighbor history). The loop pattern follows src/g1_locomotion/transition_eval.py.
There is no intervention in the environment beyond build().

Device: cuda if visible; CPU via CUDA_VISIBLE_DEVICES="".
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

ROOT = Path(__file__).resolve().parents[1]
EXP3 = ROOT / "src"
sys.path.insert(0, str(EXP3))

from g1_locomotion import harness as H  # noqa: E402  (also registers the tasks)

DEFAULT_TASK = "G1-Flat-Deploy-NoCmdCurriculum-SymLoss"


def yaw_from_quat_wxyz(q: torch.Tensor) -> torch.Tensor:
  w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
  return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap(a: torch.Tensor) -> torch.Tensor:
  return torch.remainder(a + math.pi, 2.0 * math.pi) - math.pi


def run(env, policy, cmd, max_time_s, path_max, marks, row, cmd_name="twist"):
  base = env.unwrapped
  dev = base.device
  n = base.num_envs
  steps = int(round(max_time_s / base.step_dt))
  term = base.command_manager._terms[cmd_name]

  H._fix_command(term, cmd)
  if row is not None:
    H.pin_terrain_level(base, row)
  obs, _ = env.reset()
  H._fix_command(term, cmd)
  H.assert_level_held(base, row)

  robot = base.scene["robot"]
  xy0 = robot.data.root_link_pos_w[:, :2].clone()
  yaw0 = yaw_from_quat_wxyz(robot.data.root_link_quat_w)
  c0, s0 = torch.cos(yaw0), torch.sin(yaw0)
  prev_xy = xy0.clone()
  prev_yaw = yaw0.clone()
  yaw_acc = torch.zeros(n, device=dev)           # unwrapped heading drift

  nan = lambda: torch.full((n,), float("nan"), device=dev)  # noqa: E731
  active = torch.ones(n, dtype=torch.bool, device=dev)       # episode running
  fell = torch.zeros(n, dtype=torch.bool, device=dev)
  other_end = torch.zeros(n, dtype=torch.bool, device=dev)   # time_out/exit
  done_path = torch.zeros(n, dtype=torch.bool, device=dev)
  path = torch.zeros(n, device=dev)
  t_end = nan()
  at = {m: {"y": nan(), "yaw": nan(), "x": nan(), "t": nan()} for m in marks}
  final = {"y": nan(), "yaw": nan(), "x": nan()}
  vx_err = torch.zeros(n, device=dev)
  cnt = torch.zeros(n, device=dev)
  t = 0.0

  for _ in range(steps):
    H._fix_command(term, cmd)
    with torch.no_grad():
      actions = policy(obs)
      obs, _, dones, _ = env.step(actions)
    t += base.step_dt
    xy = robot.data.root_link_pos_w[:, :2]
    yaw = yaw_from_quat_wxyz(robot.data.root_link_quat_w)
    af = active.float()
    step_len = torch.nan_to_num(torch.norm(xy - prev_xy, dim=1), nan=0.0)
    path += af * step_len
    yaw_acc += af * wrap(yaw - prev_yaw)
    d = xy - xy0
    lon = c0 * d[:, 0] + s0 * d[:, 1]
    lat = -s0 * d[:, 0] + c0 * d[:, 1]
    vx_err += af * (robot.data.root_link_lin_vel_b[:, 0] - float(cmd[0])).abs()
    cnt += af

    for m in marks:
      hit = active & torch.isnan(at[m]["y"]) & (path >= m)
      for k, v in (("y", lat), ("yaw", yaw_acc), ("x", lon),
                   ("t", torch.full_like(lat, t))):
        at[m][k] = torch.where(hit, v, at[m][k])

    tm = base.termination_manager
    tilt = tm.get_term("fell_over") if "fell_over" in tm.active_terms \
      else torch.zeros_like(dones, dtype=torch.bool)
    newly = dones.bool() & active
    fell |= newly & tilt
    other_end |= newly & ~tilt
    reached = active & ~newly & (path >= path_max)
    done_path |= reached
    ended = newly | reached
    for k, v in (("y", lat), ("yaw", yaw_acc), ("x", lon)):
      final[k] = torch.where(ended, v, final[k])
    t_end = torch.where(ended, torch.full_like(t_end, t), t_end)
    active &= ~ended
    prev_xy = xy.clone()
    prev_yaw = yaw.clone()

    done_ids = dones.bool().nonzero(as_tuple=False).squeeze(-1)
    if done_ids.numel() > 0:
      if row is not None:
        H.pin_terrain_level(base, row)
      obs_d, _ = H.reset_done_envs(base, done_ids, obs)
      obs = TensorDict(obs_d, batch_size=[n])
      H._fix_command(term, cmd)
      prev_xy = robot.data.root_link_pos_w[:, :2].clone()
      prev_yaw = yaw_from_quat_wxyz(robot.data.root_link_quat_w)
    if not active.any():
      break

  # Episodes that neither reached the mark nor fell by the end of time: final = last state.
  for k, v in (("y", lat), ("yaw", yaw_acc), ("x", lon)):
    final[k] = torch.where(active, v, final[k])
  t_end = torch.where(active, torch.full_like(t_end, t), t_end)

  def L(x):
    return [None if (isinstance(v, float) and math.isnan(v)) else v
            for v in x.tolist()]

  def stats(x):
    a = x[~torch.isnan(x)].abs().cpu().numpy()
    if a.size == 0:
      return {"n": 0, "median": None, "p90": None, "max": None}
    return {"n": int(a.size), "median": float(np.median(a)),
            "p90": float(np.percentile(a, 90)), "max": float(a.max())}

  out = {"n_episodes": n, "cmd": list(cmd), "max_time_s": max_time_s,
         "path_max_m": path_max, "marks_m": list(marks),
         "fell": int(fell.sum()), "other_termination": int(other_end.sum()),
         "reached_path_max": int(done_path.sum()),
         "timeout_before_path_max": int((~done_path & ~fell & ~other_end).sum()),
         "mae_vx_mean": float((vx_err / cnt.clamp(min=1)).mean()),
         "summary": {}, "per_ep": {}}
  for m in marks:
    key = f"{m:g}m"
    out["summary"][key] = {
      "abs_lateral_y_m": stats(at[m]["y"]),
      "abs_yaw_drift_rad": stats(at[m]["yaw"]),
      "abs_yaw_drift_deg": {k: (None if v is None else
                                (v if k == "n" else math.degrees(v)))
                            for k, v in stats(at[m]["yaw"]).items()},
      "time_s_median": (float(torch.nanmedian(at[m]["t"]))
                        if (~torch.isnan(at[m]["t"])).any() else None),
      "signed_y_mean": (float(at[m]["y"][~torch.isnan(at[m]["y"])].mean())
                        if (~torch.isnan(at[m]["y"])).any() else None),
      "signed_yaw_mean_rad": (float(at[m]["yaw"][~torch.isnan(at[m]["yaw"])].mean())
                              if (~torch.isnan(at[m]["yaw"])).any() else None),
    }
    out["per_ep"][key] = {k: L(v) for k, v in at[m].items()}
  out["per_ep"]["final"] = {k: L(v) for k, v in final.items()}
  out["per_ep"]["path_m"] = L(path)
  out["per_ep"]["t_end_s"] = L(t_end)
  out["per_ep"]["fell"] = L(fell)
  out["per_ep"]["start_yaw_rad"] = L(yaw0)
  return out


def fmt(x, p=3):
  return "—" if x is None else f"{x:.{p}f}"


def write_summary(root: Path) -> None:
  runs = []
  for p in sorted(root.glob("*/result.json")):
    runs.append((p.parent.name, json.loads(p.read_text())))
  if not runs:
    return
  lines = [
    f"# Open-loop drift: {root.name}",
    "",
    "Script `scripts/openloop_drift.py`. The command is held for the whole "
    "episode with no steering correction. y is the lateral pelvis offset in "
    "the START frame (perpendicular to the initial heading); heading is the "
    "unwrapped body yaw minus the initial yaw. Marks are by pelvis PATH "
    "length. Stats are by magnitude, among episodes that reached the mark (n).",
    "",
    "| Run | Task | Row (diff.) | Command | Episodes | Fell | Reached 20 m | "
    "n@10 | \\|y\\|@10 med / p90, m | \\|heading\\|@10 med / p90, deg | "
    "n@20 | \\|y\\|@20 med / p90, m | \\|heading\\|@20 med / p90, deg | t@20 med, s | MAE vx |",
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
  ]
  for name, r in runs:
    s10, s20 = r["summary"].get("10m", {}), r["summary"].get("20m", {})

    def pair(s, k):
      d = s.get(k, {})
      return f"{fmt(d.get('median'), 2)} / {fmt(d.get('p90'), 2)}"
    diff = r.get("difficulty")
    rowtxt = "flat" if r.get("terrain_row") is None else \
      f"{r['terrain_row']} ({diff:.3f})"
    lines.append(
      f"| {name} | `{r['task']}` | {rowtxt} | {tuple(r['cmd'])} | {r['n_episodes']} | "
      f"{r['fell']} | {r['reached_path_max']} | "
      f"{s10.get('abs_lateral_y_m', {}).get('n', 0)} | {pair(s10, 'abs_lateral_y_m')} | "
      f"{pair(s10, 'abs_yaw_drift_deg')} | "
      f"{s20.get('abs_lateral_y_m', {}).get('n', 0)} | {pair(s20, 'abs_lateral_y_m')} | "
      f"{pair(s20, 'abs_yaw_drift_deg')} | {fmt(s20.get('time_s_median'), 1)} | "
      f"{fmt(r['mae_vx_mean'])} |")
  lines += ["", "Signed means (systematic drift: y>0 = left, heading>0 = "
            "counterclockwise):", ""]
  for name, r in runs:
    for m in ("10m", "20m"):
      s = r["summary"].get(m, {})
      lines.append(f"- {name} @{m}: <y>={fmt(s.get('signed_y_mean'), 3)} m, "
                   f"<heading>={fmt(None if s.get('signed_yaw_mean_rad') is None else math.degrees(s['signed_yaw_mean_rad']), 2)} deg")
  meta = [json.loads((root / n / "run_meta.json").read_text()) for n, _ in runs]
  lines += ["", "Runs:", ""]
  for m in meta:
    lines.append(f"- `{m['task']}`: checkpoint `{m['checkpoint']}`, device "
                 f"{m['device']}, seed {m['seed']}, {m['started']} - {m.get('finished', '')}")
  lines += ["", "No acceptance threshold is set here; this is a raw metrics report.", ""]
  (root / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main():
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument("checkpoint", nargs="?")
  ap.add_argument("--task", default=DEFAULT_TASK)
  ap.add_argument("--difficulty", type=float, default=None,
                  help="difficulty 0..1 -> training grid row (as in the harness)")
  ap.add_argument("--level", type=int, default=0)
  ap.add_argument("--cmd", default="0.5,0,0")
  ap.add_argument("--num-envs", type=int, default=20)
  ap.add_argument("--max-time", type=float, default=60.0)
  ap.add_argument("--path-max", type=float, default=20.0)
  ap.add_argument("--marks", type=float, nargs="+", default=[10.0, 20.0])
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--out", required=True, help="run directory (.../<run>/<name>)")
  ap.add_argument("--summary-only", action="store_true",
                  help="only rebuild SUMMARY.md in the parent of --out")
  args = ap.parse_args()

  out = Path(args.out)
  if args.summary_only:
    write_summary(out.parent)
    return
  if not args.checkpoint:
    ap.error("checkpoint is required")
  out.mkdir(parents=True, exist_ok=True)
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  cmd = [float(x) for x in args.cmd.split(",")]
  started = _dt.datetime.now().astimezone().isoformat(timespec="seconds")

  level = args.level
  if args.difficulty is not None:
    from mjlab.tasks.registry import load_env_cfg
    tcfg = load_env_cfg(args.task, play=False)
    tr = getattr(tcfg.scene, "terrain", None)
    tg = getattr(tr, "terrain_generator", None) if tr is not None else None
    level = H.row_from_difficulty(getattr(tg, "num_rows", 1) if tg else 1,
                                  args.difficulty)
    print(f"[drift] difficulty {args.difficulty} -> row {level}")

  env, policy, row = H.build(args.task, args.checkpoint, args.num_envs, level,
                             device, seed=args.seed)
  base = env.unwrapped
  tg = (getattr(base.scene.terrain.cfg, "terrain_generator", None)
        if base.scene.terrain is not None else None)
  difficulty = (H.difficulty_of_row(tg.num_rows, row, tg.difficulty_range)
                if tg is not None and row is not None else None)
  meta = {"kind": "run_meta", "protocol": "openloop_drift", "task": args.task,
          "checkpoint": str(Path(args.checkpoint).resolve()), "device": device,
          "seed": args.seed, "num_envs": args.num_envs, "max_time_s": args.max_time,
          "path_max_m": args.path_max, "marks_m": args.marks, "cmd": cmd,
          "terrain_row": row, "difficulty": difficulty, "started": started}
  (out / "run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

  res = run(env, policy, cmd, args.max_time, args.path_max, args.marks, row)
  res.update({"task": args.task, "terrain_row": row, "difficulty": difficulty,
              "seed": args.seed, "checkpoint": meta["checkpoint"]})
  (out / "result.json").write_text(json.dumps(res, indent=2))
  meta["finished"] = _dt.datetime.now().astimezone().isoformat(timespec="seconds")
  (out / "run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
  for k, s in res["summary"].items():
    print(f"[drift] @{k}: |y| {s['abs_lateral_y_m']}, |heading| deg {s['abs_yaw_drift_deg']}")
  print(f"[drift] fell {res['fell']}/{res['n_episodes']}, reached "
        f"{args.path_max:g} m {res['reached_path_max']}")
  write_summary(out.parent)


if __name__ == "__main__":
  main()
