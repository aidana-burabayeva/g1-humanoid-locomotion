"""Flat-ground command-tracking protocol: each command is checked SEPARATELY.

Why a separate file instead of a mode flag in `src/g1_locomotion/harness.py`.
`harness.py` is the matrix harness "policy x terrain x level"; its `main()`
writes a per-command results directory consumed by other tools.
The flat protocol is a different object: a fixed set of commands,
fixed thresholds, a PASS/FAIL verdict, and a separate scenario type
(transition). Mixing them into one `main()` would mean changing the results
directory format both loaders depend on. Hence this separate runner, which
IMPORTS `build()`/`rollout()` from `harness.py` (one implementation of the
measurement on both paths, so there is no room for divergence). Lives in
`scripts/`, as the flat-ground evaluation protocol.

THRESHOLDS -- a project requirement set IN ADVANCE. This is not a measured
result and is not tuned to match one:

    MAE vx <= 0.10 m/s, MAE vy <= 0.10 m/s, MAE wz <= 0.15 rad/s,
    0 falls out of N for EACH command,
    displacement <= 0.5 m for the STANDING scenario (worst episode).

Protocol rules baked into the code:
* each command gets its own result and its own verdict; averaging good and
  bad commands into one overall score is impossible by construction (the
  script never prints an "average MAE across all commands" summary);
* MAE is computed in the BODY FRAME -- `root_link_lin_vel_b` /
  `root_link_ang_vel_b` (`harness.rollout`), not in the world frame;
* N episodes per command = N environments: one environment is one episode;
  after termination an environment stops accumulating metrics;
* besides the mean MAE, PER-EPISODE values are saved (`mae_*_per_ep`) -- so
  the spread is visible, not just the mean.

TWO DIFFERENT SCENARIO TYPES that must not be confused:
* `stand`  -- the (0,0,0) command from the start of the episode. This checks
              STANDING. It is not braking: there is no transition, so
              nothing to brake from.
* `brake`  -- a transition: `hold_s` seconds of forward motion, then a zero
              command. Metrics: time to the START of stopping and distance
              to CONFIRMED stop. The stop criterion is defined explicitly in
              `rollout()` and remains DIAGNOSTIC -- the protocol does not impose a
              separate pass threshold on braking.

EACH SCENARIO HAS ITS OWN DURATION, named precisely: hold/stand scenarios run
for `--duration` seconds, while the braking scenario runs for
`--brake-hold + --brake-after` (10 + 10 = 20 s by default, not 60 s).

FILLING IN MISSING COMMANDS. `--scenarios` only computes what is listed;
previously computed JSON files in `--out` are kept and included in the
`stage1_meta.json` summary. The summary is marked `INCOMPLETE` if results for
the required set of scenarios are missing.

Run (smoke test, CPU):
    CUDA_VISIBLE_DEVICES="" python scripts/eval_flat_protocol.py <ckpt> \
        --num-envs 2 --duration 3 --scenarios fwd_05 stand brake
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_TERRAIN = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_TERRAIN))

from g1_locomotion.harness import build, rollout  # noqa: E402

# --- FLAT PROTOCOL THRESHOLDS. Not to be changed based on run results. ---
THRESHOLDS = {"mae_vx": 0.10, "mae_vy": 0.10, "mae_wz": 0.15, "max_falls": 0,
              # Standing: displacement over 60 s at (0,0,0) from the start of
              # the episode <= 0.5 m, 0 falls out of 20. This requirement
              # existed before but was not part of the verdict. Applies ONLY
              # to the standing scenario (kind == "stand"), per episode
              # (worst episode).
              "stand_max_disp_m": 0.5}

# --- Command set. Defined EXPLICITLY, both positive AND negative directions. ---
# name -> (scenario type, command, command after switch)
SCENARIOS: dict[str, dict] = {
  # --- required command set for the flat protocol ---
  "stand":      {"kind": "stand", "cmd": (0.0, 0.0, 0.0)},        # standing
  "fwd_05":     {"kind": "hold",  "cmd": (0.5, 0.0, 0.0)},        # forward
  "fwd_10":     {"kind": "hold",  "cmd": (1.0, 0.0, 0.0)},        # forward
  "yaw_pos_05": {"kind": "hold",  "cmd": (0.0, 0.0, 0.5)},        # turn
  "yaw_neg_05": {"kind": "hold",  "cmd": (0.0, 0.0, -0.5)},       # turn
  "fwd_yaw_05": {"kind": "hold",  "cmd": (0.5, 0.0, 0.5)},        # combined
  "left_04":    {"kind": "hold",  "cmd": (0.0, 0.4, 0.0)},        # lateral
  "right_04":   {"kind": "hold",  "cmd": (0.0, -0.4, 0.0)},       # lateral
  "brake":      {"kind": "brake", "cmd": (0.5, 0.0, 0.0), "cmd_after": (0.0, 0.0, 0.0)},
  # --- beyond the required set: a finer amplitude grid. Not required by the
  # protocol verdict, but useful for diagnostics; do not remove. ---
  "fwd_03":     {"kind": "hold",  "cmd": (0.3, 0.0, 0.0)},
  "fwd_08":     {"kind": "hold",  "cmd": (0.8, 0.0, 0.0)},
  "back_03":    {"kind": "hold",  "cmd": (-0.3, 0.0, 0.0)},
  "back_05":    {"kind": "hold",  "cmd": (-0.5, 0.0, 0.0)},
  "left_03":    {"kind": "hold",  "cmd": (0.0, 0.3, 0.0)},
  "right_03":   {"kind": "hold",  "cmd": (0.0, -0.3, 0.0)},
}
# Required set of scenarios for the flat protocol -- so that a pass
# cannot be declared from an incomplete run.
PLAN_REQUIRED = ["stand", "fwd_05", "fwd_10", "yaw_pos_05", "yaw_neg_05",
                 "fwd_yaw_05", "left_04", "right_04", "brake"]
DEFAULT_ORDER = list(SCENARIOS)


def verdict(res: dict) -> dict:
  """PASS/FAIL per command, against FIXED thresholds.

  For the transition scenario, MAE is taken from the phase AFTER the switch:
  mixing it with the motion phase would compare different commands.
  """
  if res.get("scenario") == "transition":
    vx, vy, wz = res["mae_vx_after"], res["mae_vy_after"], res["mae_wz_after"]
    src = "after switch"
  else:
    vx, vy, wz = res["mae_vx"], res["mae_vy"], res["mae_wz"]
    src = "whole episode"
  checks = {
    "mae_vx": (vx, vx <= THRESHOLDS["mae_vx"]),
    "mae_vy": (vy, vy <= THRESHOLDS["mae_vy"]),
    "mae_wz": (wz, wz <= THRESHOLDS["mae_wz"]),
    "falls":  (res["fell"], res["fell"] <= THRESHOLDS["max_falls"]),
  }
  # Displacement limit -- requirement for STANDING. Checked by the WORST
  # episode: averaging over 20 episodes would hide one episode that drifted
  # off.
  if res.get("scenario_kind") == "stand":
    per_disp = res.get("displacement_m_per_ep")
    d = (max(per_disp) if per_disp
         else float(res.get("displacement_m_mean", 0.0)))
    checks["displacement_m_max"] = (d, d <= THRESHOLDS["stand_max_disp_m"])
  # --- PER-EPISODE DIAGNOSTICS (informational, NOT part of the verdict) ----
  # The PASS/FAIL criterion remains the MEAN MAE -- that is how it was fixed,
  # and it is neither tightened nor loosened here. But the mean over 20
  # episodes hides the spread: `fwd_10` with MAE 0.0910 passes the 0.10
  # threshold on average, and that does NOT mean every one of the 20 episodes
  # stayed within threshold. So alongside it we print how many episodes
  # exceeded the threshold and what the worst one was.
  suffix = "_after_per_ep" if res.get("scenario") == "transition" else "_per_ep"
  diag = {"note": "for reference, NOT part of the verdict: the PASS/FAIL "
                  "criterion is the mean MAE; this shows the per-episode "
                  "spread against the same threshold",
          "source": src, "n_ep": int(res.get("n_envs", 0))}
  for axis in ("vx", "vy", "wz"):
    per = res.get(f"mae_{axis}{suffix}")
    thr = THRESHOLDS[f"mae_{axis}"]
    if not per:
      diag[f"n_ep_over_{axis}"] = None
      diag[f"mae_{axis}_max"] = None
      continue
    diag[f"n_ep_over_{axis}"] = int(sum(1 for x in per if x > thr))
    diag[f"mae_{axis}_max"] = float(max(per))
    diag[f"mae_{axis}_min"] = float(min(per))
  return {
    "mae_source": src,
    "checks": {k: {"value": float(v), "pass": bool(p)} for k, (v, p) in checks.items()},
    "verdict": "PASS" if all(p for _, p in checks.values()) else "FAIL",
    "failed": [k for k, (_, p) in checks.items() if not p],
    "per_episode_diagnostic": diag,
  }


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("checkpoint")
  ap.add_argument("--task", default="Mjlab-Velocity-Flat-Unitree-G1")
  ap.add_argument("--num-envs", type=int, default=20,
                  help="episodes per command (one environment = one episode)")
  ap.add_argument("--duration", type=float, default=60.0,
                  help="seconds per episode for hold/stand scenarios; the "
                       "braking scenario lasts brake_hold + brake_after and "
                       "is NOT controlled by this flag")
  ap.add_argument("--brake-hold", type=float, default=10.0,
                  help="seconds of forward motion before switching to zero command")
  ap.add_argument("--brake-after", type=float, default=10.0,
                  help="seconds of observation after the switch")
  ap.add_argument("--level", type=int, default=0)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--scenarios", nargs="*", default=DEFAULT_ORDER,
                  help="which scenarios to compute in this run. Previously "
                       "computed JSON files in --out are not overwritten if "
                       "a scenario is not selected: the summary is built "
                       "from all files in the directory")
  ap.add_argument("--out", default="results_stage1")
  args = ap.parse_args()

  unknown = [s for s in args.scenarios if s not in SCENARIOS]
  if unknown:
    raise SystemExit(f"unknown scenarios: {unknown}; available: {DEFAULT_ORDER}")

  torch.manual_seed(args.seed)
  np.random.seed(args.seed)
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  print(f"[stage1] device={device} episodes/command={args.num_envs} "
        f"duration={args.duration}s")

  env, policy, row = build(args.task, args.checkpoint, args.num_envs,
                           args.level, device, seed=args.seed)

  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  results = []
  for name in args.scenarios:
    sc = SCENARIOS[name]
    if sc["kind"] == "brake":
      dur = args.brake_hold + args.brake_after
      res = rollout(env, policy, sc["cmd"], dur, level_row=row,
                    switch_at_s=args.brake_hold, cmd_after=sc["cmd_after"])
    else:
      res = rollout(env, policy, sc["cmd"], args.duration, level_row=row)
    res["name"] = name
    # Duration is named EXACTLY: for the braking scenario it equals
    # brake_hold + brake_after, not --duration.
    res["duration_s"] = float(dur if sc["kind"] == "brake" else args.duration)
    res["duration_note"] = (
      f"{args.brake_hold:.1f} s motion + {args.brake_after:.1f} s after "
      f"switch = {res['duration_s']:.1f} s"
      if sc["kind"] == "brake" else f"{res['duration_s']:.1f} s per episode")
    res["scenario_kind"] = sc["kind"]
    res["task"] = args.task
    res["seed"] = args.seed
    res["checkpoint"] = str(Path(args.checkpoint).resolve())
    res["thresholds"] = dict(THRESHOLDS)
    res["result"] = verdict(res)
    results.append(res)
    (out / f"{name}.json").write_text(json.dumps(res, indent=2))
    print(f"[stage1] {name:11s} {res['result']['verdict']:4s} "
          f"{res['duration_s']:.1f}s "
          f"vx={res['mae_vx']:.3f} vy={res['mae_vy']:.3f} wz={res['mae_wz']:.3f} "
          f"falls={res['fell']}/{res['n_envs']}")
    d = res["result"]["per_episode_diagnostic"]
    print(f"[stage1] {'':11s} for reference (does not affect verdict): "
          f"episodes above threshold vx={d['n_ep_over_vx']}/{d['n_ep']} "
          f"vy={d['n_ep_over_vy']}/{d['n_ep']} "
          f"wz={d['n_ep_over_wz']}/{d['n_ep']}; worst episode "
          f"vx={d['mae_vx_max']:.3f} vy={d['mae_vy_max']:.3f} "
          f"wz={d['mae_wz_max']:.3f}")
    if res.get("mean_signed_error_vy") is not None:
      # IMPORTANT about scope: signed numbers are computed over the WHOLE
      # episode. For the transition scenario that is both phases together
      # (motion + braking), but the verdict is based on the phase AFTER the
      # switch -- they cannot be compared directly, so the phase is named
      # explicitly.
      ph = ("whole episode, BOTH phases" if res.get("scenario") == "transition"
            else "whole episode")
      print(f"[stage1] {'':11s} sign ({ph}): "
            f"vx_body={res['mean_vx_body']:+.3f} "
            f"(error {res['mean_signed_error_vx']:+.3f}) "
            f"vy_body={res['mean_vy_body']:+.3f} "
            f"(error {res['mean_signed_error_vy']:+.3f}) "
            f"wz_body={res['mean_wz_body']:+.3f} "
            f"(error {res['mean_signed_error_wz']:+.3f})")

  # The summary is built from ALL scenario files in the directory, not just
  # the ones computed in this run: --scenarios exists precisely so missing
  # commands can be filled in without recomputing ones already done.
  by_name = {}
  for f in sorted(out.glob("*.json")):
    try:
      d = json.loads(f.read_text())
    except Exception:
      continue
    if d.get("kind") == "run_meta" or "result" not in d:
      continue
    by_name[d.get("name", f.stem)] = d
  for r in results:
    by_name[r["name"]] = r
  missing = [n for n in PLAN_REQUIRED if n not in by_name]
  all_results = list(by_name.values())

  (out / "stage1_meta.json").write_text(json.dumps({
    "kind": "run_meta",  # internal marker file: result loaders filter it out
    "protocol": "stage1",
    "thresholds": dict(THRESHOLDS),
    "task": args.task, "seed": args.seed, "num_envs": args.num_envs,
    # Each scenario has its own duration; there is no single "60s for everything".
    "duration_s_hold": args.duration,
    "duration_s_brake": args.brake_hold + args.brake_after,
    "duration_s_by_scenario": {n: d.get("duration_s") for n, d in
                               sorted(by_name.items())},
    "brake_hold_s": args.brake_hold,
    "brake_after_s": args.brake_after, "terrain_row": row,
    "checkpoint": str(Path(args.checkpoint).resolve()),
    "scenarios_this_run": list(args.scenarios),
    "scenarios_known": sorted(by_name),
    "plan_required": list(PLAN_REQUIRED),
    "plan_required_missing": missing,
    "overall": ("INCOMPLETE" if missing else
                "PASS" if all(r["result"]["verdict"] == "PASS"
                              for r in all_results) else "FAIL"),
  }, indent=2))

  print()
  print(f"Thresholds (set in advance): MAE vx<={THRESHOLDS['mae_vx']}, "
        f"vy<={THRESHOLDS['mae_vy']}, wz<={THRESHOLDS['mae_wz']}, "
        f"falls<={THRESHOLDS['max_falls']}, standing displacement<="
        f"{THRESHOLDS['stand_max_disp_m']} m")
  print()
  print("| Scenario | Command vx,vy,wz | Dur., s | Ep. | Falls | MAE vx "
        "| MAE vy | MAE wz | vx spread (min..max) | Max disp., m | Verdict |")
  print("|---|---|---|---|---|---|---|---|---|---|---|")
  for r in results:
    c = r["cmd"]
    # The spread is taken from the same phase the verdict is based on.
    per = (r["mae_vx_after_per_ep"] if r.get("scenario") == "transition"
           else r["mae_vx_per_ep"])
    v = r["result"]
    ch = v["checks"]
    def mark(k):
      return f"{ch[k]['value']:.3f}" + ("" if ch[k]["pass"] else " ✗")
    dm = (mark("displacement_m_max") if "displacement_m_max" in ch else "—")
    print(f"| {r['name']} | {c[0]:+.1f}, {c[1]:+.1f}, {c[2]:+.1f} | "
          f"{r['duration_s']:.1f} | "
          f"{r['n_envs']} | {r['fell']} | {mark('mae_vx')} | {mark('mae_vy')} | "
          f"{mark('mae_wz')} | {min(per):.3f}..{max(per):.3f} | {dm} | "
          f"{v['verdict']} |")

  print()
  print("REFERENCE per-episode table. NOT part of the verdict: PASS/FAIL "
        "is decided above by MEAN MAE; the thresholds are the same and did "
        "not change.")
  print("The \"ep. > threshold\" column answers the question the mean hides: "
        "how many of the N individual episodes missed the threshold.")
  print("The signed columns distinguish overshoot from undershoot, which "
        "absolute MAE cannot: error = actual - command, >0 overshoot, "
        "<0 undershoot.")
  print("| Scenario | Ep. | ep. > thresh vx | vy | wz | worst vx | vy | wz "
        "| vx actual | error vx | vy actual | error vy | wz actual | error wz |")
  print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
  for r in results:
    d = r["result"]["per_episode_diagnostic"]
    g = lambda k: "—" if r.get(k) is None else f"{r[k]:+.3f}"  # noqa: E731
    print(f"| {r['name']} | {d['n_ep']} | {d['n_ep_over_vx']} | "
          f"{d['n_ep_over_vy']} | {d['n_ep_over_wz']} | "
          f"{d['mae_vx_max']:.3f} | {d['mae_vy_max']:.3f} | "
          f"{d['mae_wz_max']:.3f} | "
          f"{g('mean_vx_body')} | {g('mean_signed_error_vx')} | "
          f"{g('mean_vy_body')} | {g('mean_signed_error_vy')} | "
          f"{g('mean_wz_body')} | {g('mean_signed_error_wz')} |")

  br = [r for r in results if r.get("scenario") == "transition"]
  if br:
    print()
    print("TRANSITION scenario (motion -> zero command). "
          "This is NOT the same as standing from a zero command.")
    print("Time to stop refers to the START of the quiet window; path and "
          "displacement refer to its END (up to stop confirmation).")
    print("| Scenario | Dur., s | Switch, s | Observation window, s "
          "| Stop registered | Time to stop onset, s "
          "| Path to stop confirmation, m "
          "| Displacement to stop confirmation, m |")
    print("|---|---|---|---|---|---|---|---|")
    for r in br:
      f = lambda x: "—" if x is None else f"{x:.2f}"  # noqa: E731
      print(f"| {r['name']} | {r['duration_s']:.1f} | {r['switch_at_s']:.1f} | "
            f"{r['stop_window_s']:.1f} | "
            f"{r['stopped_count']}/{r['n_envs']} | {f(r['stop_time_s_mean'])} | "
            f"{f(r['stop_confirm_path_m_mean'])} | "
            f"{f(r['stop_confirm_disp_m_mean'])} |")
      if r.get("stop_not_registered_count"):
        print(f"  {r['name']}: stop not registered within the observation "
              f"window ({r['stop_window_s']:.1f} s) for "
              f"{r['stop_not_registered_count']}/{r['n_envs']} episodes -- "
              "this means \"the criterion did not trigger within the "
              "window\", not \"did not stop\".")

  print()
  fails = [n for n, r in sorted(by_name.items())
           if r["result"]["verdict"] == "FAIL"]
  if missing:
    print(f"RESULT: INCOMPLETE -- missing results for required scenarios: "
          f"{', '.join(missing)}")
  else:
    print("RESULT: " + ("PASS -- all commands passed" if not fails
                      else f"FAIL for commands: {', '.join(fails)}"))
  print(f"(summary over {len(by_name)} scenarios in directory {out}; "
        f"computed in this run: {len(results)})")
  print("An overall averaged score across commands is intentionally not "
        "computed: the protocol requires a separate verdict for each command.")


if __name__ == "__main__":
  main()
