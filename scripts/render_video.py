#!/usr/bin/env python3
"""Engineering videos: policy rollouts with a numeric HUD and tracking plots.

Why a separate file instead of more flags in render_policy.py:
render_policy.main() is one fixed loop over constant command segments. Route
videos must run the unmodified route evaluator (route_eval.evaluate),
which owns its own step loop and the position-based yaw controller. That loop
is driven here through a thin env proxy whose step() records the same tensors
the evaluator reads and renders a frame, so steering, reset and outcome logic
are not copied. render_policy.py is not modified by this tool; this module
imports it for the GL/library setup and for parse_commands/build/_fix_command.

Every HUD value is read from the simulation after the env step that produced
it: command from the command buffer (term.vel_command_b, as set right before
the step), velocities from root_link_lin_vel_b / root_link_ang_vel_b (body
frame, the same tensors the harness uses for MAE), attitude from
root_link_quat_w, terrain type and physical parameter from the terrain
generator configuration and the pinned row, not from a scenario label.

Rendering: MuJoCo offscreen via EGL by default (OSMesa is ~30x slower);
physics and policy run on CPU (deterministic, identical initial state for
the same seed). Frames are piped straight into ffmpeg (libx264); no image
sequences are written to disk.

Examples (inside the container, from /app):
  python scripts/render_video.py commands --checkpoint CKPT --hud --plots \
      --out /outputs/flat_commands_28996.mp4
  python scripts/render_video.py route --checkpoint CKPT --hud --plots \
      --out /outputs/route_28996.mp4
  python scripts/render_video.py stairs10 --checkpoint CKPT --hud --plots \
      --out /outputs/stairs10_33800.mp4
  python scripts/render_video.py compare --checkpoint CKPT_A --checkpoint-b CKPT_B \
      --hud --plots --out /outputs/inverted10_before_after.mp4
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")
if os.environ["MUJOCO_GL"] == "egl":
  # NVIDIA EGL needs a visible device; physics still runs on --device cpu.
  os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import argparse
import json
import math
import shutil
import subprocess
import sys
import zlib
from collections import deque
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from g1_locomotion import render_policy  # noqa: E402  (GL/library environment setup)
from g1_locomotion import harness  # noqa: E402
from g1_locomotion import route_eval as route  # noqa: E402
# Registers the 10 cm tasks and their route layouts exactly as the evaluator.
from g1_locomotion import mixed_route_eval as evaluate_mixed_route  # noqa: E402,F401
from g1_locomotion import stairs_eval as evaluate_stairs  # noqa: E402
from g1_locomotion.tasks import stairs10 as stairs10_tasks  # noqa: E402

FLAT_TASK = "G1-Flat-Deploy-NoCmdCurriculum-SymLoss"
MIN_FREE_BYTES = 500 * 1024 * 1024
FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
FLAT_SEQUENCE = (
  "stand:0,0,0:4.5;fwd 0.5:0.5,0,0:4.5;fwd 1.0:1.0,0,0:4.5;"
  "yaw +0.5:0,0,0.5:4.5;yaw -0.5:0,0,-0.5:4.5;lateral +0.4:0,0.4,0:4.5;"
  "lateral -0.4:0,-0.4,0:4.5;brake:0,0,0:4.5"
)
ROUTE_STEER = dict(steer="cross_track", lookahead_m=4.0, yaw_gain=1.5,
                   yaw_limit=0.5, turn_offset_m=1.0)
PLANNER_NOTE = "wz from planner (row hold)"
EXPERIMENTAL_NOTE = ("experimental checkpoint — held-out route success "
                     "33/40 (82.5 %)")


# ---------------------------------------------------------------- terrain --

def terrain_label(base, row: int | None, xy: np.ndarray) -> str:
  """Type and physical parameter of the tile under the robot."""
  terrain = base.scene.terrain
  gen = getattr(getattr(terrain, "cfg", None), "terrain_generator", None)
  if gen is None or getattr(terrain, "terrain_origins", None) is None:
    return "flat (plane)"
  origins = terrain.terrain_origins.detach().cpu().numpy()
  r = 0 if row is None else row
  cy = origins[r, :, 1]
  col = int(np.argmin(np.abs(cy - xy[1])))
  half_y, half_x = gen.size[1] / 2, gen.size[0] / 2
  if abs(cy[col] - xy[1]) > half_y + 1e-6 or abs(origins[r, col, 0] - xy[0]) > half_x:
    return "border"
  cfg = list(gen.sub_terrains.values())[col]
  d = harness.difficulty_of_row(gen.num_rows, r, gen.difficulty_range)
  name = type(cfg).__name__
  if name == "BoxFlatTerrainCfg":
    return "flat"
  if name == "HfRandomUniformTerrainCfg":
    scale = d if getattr(cfg, "scale_with_difficulty", False) else 1.0
    lo, hi = cfg.noise_range[0] * scale, cfg.noise_range[1] * scale
    return f"rough {lo:.2f}–{hi:.2f} m"
  if name == "HfPyramidSlopedTerrainCfg":
    s = cfg.slope_range[0] + d * (cfg.slope_range[1] - cfg.slope_range[0])
    inv = " inv" if getattr(cfg, "inverted", False) else ""
    return f"slope{inv} {s:.2f} ({math.degrees(math.atan(s)):.0f}°)"
  if name in ("BoxPyramidStairsTerrainCfg", "BoxInvertedPyramidStairsTerrainCfg"):
    h = cfg.step_height_range[0] + d * (cfg.step_height_range[1]
                                        - cfg.step_height_range[0])
    kind = "inv stairs" if "Inverted" in name else "stairs"
    cm = h * 100
    txt = f"{cm:.0f}" if abs(cm - round(cm)) < 0.05 else f"{cm:.1f}"
    return f"{kind} {txt} cm"
  return name


def roll_pitch_deg(q: torch.Tensor) -> tuple[float, float]:
  w, x, y, z = (float(v) for v in q)
  roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
  pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
  return math.degrees(roll), math.degrees(pitch)


# ------------------------------------------------------------- instrument --

class Instrument:
  """Per-step measurements for env 0, computed from the harness tensors."""

  def __init__(self, base, row, route_geom=None, mae_window_s=2.0,
               plot_window_s=5.0):
    self.base, self.row, self.geom = base, row, route_geom
    self.dt = base.step_dt
    self.robot = base.scene["robot"]
    self.win = deque(maxlen=max(1, round(mae_window_s / self.dt)))
    self.plot = deque(maxlen=max(2, round(plot_window_s / self.dt)) + 1)
    self.step = 0
    self.t = 0.0
    self.dist = 0.0
    self.err_sum = np.zeros(3)
    self.status = "OK"
    self.finished = False
    self.cmd = np.zeros(3)
    self.prev_xy = None
    self.state = {}

  def on_reset(self):
    self.prev_xy = self.robot.data.root_link_pos_w[:, :2].clone()

  def before_step(self, term):
    self.cmd = term.vel_command_b[0].detach().cpu().double().numpy().copy()

  def after_step(self, dones):
    d = self.robot.data
    lin = d.root_link_lin_vel_b[0].detach().cpu().double().numpy()
    ang = d.root_link_ang_vel_b[0].detach().cpu().double().numpy()
    xy_t = d.root_link_pos_w[:, :2]
    step_len = float(torch.nan_to_num(torch.norm(xy_t - self.prev_xy, dim=1),
                                      nan=0.0)[0])
    self.prev_xy = xy_t.clone()
    self.step += 1
    self.t += self.dt
    self.dist += step_len
    err = np.abs(np.array([lin[0] - self.cmd[0], lin[1] - self.cmd[1],
                           ang[2] - self.cmd[2]]))
    self.err_sum += err
    self.win.append(err)
    self.plot.append((self.t, self.cmd[0], lin[0], self.cmd[2], ang[2]))
    done = bool(dones[0])
    tm = self.base.termination_manager
    fell = done and "fell_over" in tm.active_terms and bool(tm.get_term("fell_over")[0])
    xy = xy_t[0].detach().cpu().numpy()
    if self.status == "OK":
      if fell:
        self.status = "FELL"
      elif self.geom is not None and abs(xy[0] - self.geom["row_x"]) > self.geom["half_x"]:
        self.status = "LEFT ROW"
      elif done:
        self.status = "TERMINATED"
      elif self.geom is not None and xy[1] >= self.geom["finish_y"]:
        self.finished = True
    roll, pitch = roll_pitch_deg(d.root_link_quat_w[0].detach().cpu())
    w = np.mean(np.stack(self.win), axis=0)
    self.state = {
      "step": self.step, "t": self.t, "dist": self.dist,
      "cmd": self.cmd.tolist(), "act": [lin[0], lin[1], ang[2]],
      "mae_win_vx": float(w[0]), "mae_win_wz": float(w[2]),
      "mae_win_s": len(self.win) * self.dt,
      "roll": roll, "pitch": pitch,
      "terrain": terrain_label(self.base, self.row, xy),
      "status": self.status, "finished": self.finished,
      "xy": xy.tolist(),
      "mae_cum": (self.err_sum / self.step).tolist(),
    }
    return self.state


# ----------------------------------------------------------------- drawing --

def font(size, bold=True):
  name = "DejaVuSansMono-Bold.ttf" if bold else "DejaVuSansMono.ttf"
  return ImageFont.truetype(str(FONT_DIR / name), size)


STATUS_COLOR = {"OK": (80, 220, 100), "FELL": (255, 70, 70),
                "LEFT ROW": (255, 170, 40), "TERMINATED": (255, 170, 40)}


def hud_lines(s):
  c, a = s["cmd"], s["act"]
  return [
    f"t {s['t']:6.2f} s   dist {s['dist']:6.2f} m",
    f"cmd  vx {c[0]:+.2f}  vy {c[1]:+.2f}  wz {c[2]:+.2f}",
    f"act  vx {a[0]:+.2f}  vy {a[1]:+.2f}  wz {a[2]:+.2f}",
    f"MAE vx {s['mae_win_vx']:.3f}  wz {s['mae_win_wz']:.3f}  "
    f"({s['mae_win_s']:.1f} s)",
    f"roll {s['roll']:+5.1f}°  pitch {s['pitch']:+5.1f}°",
    f"terrain  {s['terrain']}",
  ]


def draw_hud(img, s, header, notes, banner=None, scale=1.0):
  draw = ImageDraw.Draw(img, "RGBA")
  fs = max(11, int(19 * scale))
  f, fsmall = font(fs), font(max(10, int(15 * scale)), bold=False)
  lines = hud_lines(s)
  status = s["status"] + ("  finish reached" if s["finished"] and s["status"] == "OK" else "")
  pad, lh = int(8 * scale), int(fs * 1.3)
  width = int(max(draw.textlength(x, font=f) for x in lines + [status])) + 2 * pad
  height = lh * (len(lines) + 1) + 2 * pad
  draw.rectangle([6, 6, 6 + width, 6 + height], fill=(0, 0, 0, 160))
  y = 6 + pad
  for line in lines:
    draw.text((6 + pad, y), line, font=f, fill=(235, 235, 235))
    y += lh
  draw.text((6 + pad, y), status, font=f, fill=STATUS_COLOR.get(s["status"], (255, 255, 255)))
  # Header (checkpoint, scenario) and persistent notes, top right.
  y = 8
  for i, text in enumerate([header] + list(notes)):
    ff = f if i == 0 else fsmall
    tw = draw.textlength(text, font=ff)
    x = img.width - tw - 12
    draw.rectangle([x - 6, y - 3, img.width - 6, y + ff.size + 5], fill=(0, 0, 0, 160))
    color = (255, 210, 90) if text == EXPERIMENTAL_NOTE else (235, 235, 235)
    draw.text((x, y), text, font=ff, fill=color)
    y += ff.size + 10
  if banner:
    fb = font(max(14, int(30 * scale)))
    tw = draw.textlength(banner, font=fb)
    x, yb = (img.width - tw) / 2, img.height - fb.size - int(24 * scale)
    draw.rectangle([x - 14, yb - 8, x + tw + 14, yb + fb.size + 10], fill=(20, 60, 140, 200))
    draw.text((x, yb), banner, font=fb, fill=(255, 255, 255))


def draw_plots(width, height, s, history, window_s=5.0, scale=1.0):
  img = Image.new("RGB", (width, height), (24, 24, 28))
  draw = ImageDraw.Draw(img)
  f = font(max(10, int(14 * scale)), bold=False)
  hist = np.array(history) if history else np.zeros((0, 5))
  t_now = s["t"] if s else 0.0
  panels = (("vx [m/s]  cmd vs act", 1, 2, (-0.6, 1.3)),
            ("wz [rad/s]  cmd vs act", 3, 4, (-1.0, 1.0)))
  pw = width // 2
  for k, (title, ci, ai, (lo, hi)) in enumerate(panels):
    x0, x1 = k * pw + int(44 * scale), (k + 1) * pw - int(10 * scale)
    y0, y1 = int(22 * scale), height - int(20 * scale)
    draw.rectangle([x0, y0, x1, y1], outline=(90, 90, 90))
    label = title + f"   last {window_s:.0f} s   white=cmd  cyan=act"
    if draw.textlength(label, font=f) > x1 - x0:
      label = title.split()[0] + f" {title.split()[1]}  {window_s:.0f} s  white=cmd cyan=act"
    draw.text((x0, 3), label, font=f, fill=(220, 220, 220))
    for v in (lo, 0.0, hi):
      yy = y1 - (v - lo) / (hi - lo) * (y1 - y0)
      draw.line([x0, yy, x1, yy], fill=(60, 60, 60) if v else (110, 110, 110))
      draw.text((k * pw + 2, yy - 7), f"{v:+.1f}", font=f, fill=(160, 160, 160))
    if len(hist) < 2:
      continue
    def pts(col):
      out = []
      for row in hist:
        xx = x1 - (t_now - row[0]) / window_s * (x1 - x0)
        v = min(max(row[col], lo), hi)
        out.append((xx, y1 - (v - lo) / (hi - lo) * (y1 - y0)))
      return out
    draw.line(pts(ci), fill=(240, 240, 240), width=max(1, int(2 * scale)))
    draw.line(pts(ai), fill=(60, 210, 240), width=max(1, int(2 * scale)))
  return img


def add_scan_spheres(scene, hit_pos, hit_ok):
  """Height-scan hits (terrain_scan.hit_pos_w) as user spheres in mjvScene.

  Colour encodes hit height relative to the median of the 187 hits:
  blue = 10 cm or more below, green = level, red = 10 cm or more above.
  """
  import mujoco
  eye = np.eye(3).flatten()
  ref = float(np.median(hit_pos[hit_ok, 2])) if hit_ok.any() else 0.0
  for p, ok in zip(hit_pos, hit_ok):
    if not ok or scene.ngeom >= scene.maxgeom:
      continue
    u = float(np.clip((p[2] - ref) / 0.10, -1.0, 1.0))
    rgba = np.array([max(u, 0.0), 1.0 - abs(u), max(-u, 0.0), 0.95], dtype=np.float32)
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([0.02, 0, 0]), p.astype(np.float64), eye, rgba)
    scene.ngeom += 1


# ------------------------------------------------------------------ output --

class VideoWriter:
  def __init__(self, path: Path, width: int, height: int, fps: int, crf: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(path.parent).free
    if free < MIN_FREE_BYTES:
      raise SystemExit(f"[hud] refusing to write {path}: only {free / 2**20:.0f} MB free")
    print(f"[hud] free disk {free / 2**20:.0f} MB; writing {path}")
    self.path, self.w, self.h, self.n = path, width, height, 0
    self.proc = subprocess.Popen(
      ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
       "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
       "-preset", "slow", "-crf", str(crf), "-pix_fmt", "yuv420p",
       "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)

  def write(self, frame: np.ndarray):
    assert frame.shape == (self.h, self.w, 3), frame.shape
    self.proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
    self.n += 1

  def close(self):
    self.proc.stdin.close()
    if self.proc.wait() != 0:
      raise RuntimeError(f"ffmpeg failed for {self.path}")


def save_png(frame: np.ndarray, path: Path, limit=200_000):
  img = Image.fromarray(frame)
  for w in (img.width, 960, 800, 640):
    im = img if w == img.width else img.resize((w, round(img.height * w / img.width)),
                                                Image.LANCZOS)
    im.quantize(colors=256, method=Image.Quantize.MEDIANCUT).save(path, optimize=True)
    if path.stat().st_size < limit:
      break
  print(f"[hud] frame png {path} ({path.stat().st_size} B)")


# ------------------------------------------------------------- the camera --

class Camera:
  """EGL/OSMesa offscreen renderer for env 0 plus HUD/plot composition."""

  def __init__(self, base, width, height, args, plot_h):
    from mjlab.viewer.offscreen_renderer import OffscreenRenderer
    self.base, self.args = base, args
    self.width, self.height = width, height
    self.plot_h = plot_h if args.plots else 0
    self.view_h = height - self.plot_h
    v = base.cfg.viewer
    v.width, v.height = width, self.view_h
    v.distance, v.elevation, v.azimuth = args.cam_distance, args.cam_elevation, args.cam_azimuth
    self.r = OffscreenRenderer(model=base.sim.mj_model, cfg=v, scene=base.scene,
                               sim_model=base.sim.model,
                               expanded_fields=base.sim.expanded_fields)
    self.r.initialize()
    self.scale = width / 1280
    if args.scan_viz:
      # Replace mjlab's uniform hit markers with the height-coded spheres.
      base.scene["terrain_scan"].cfg.debug_vis = False

  def frame(self, inst, header, notes, banner=None):
    b = self.base
    self.r.update(b.sim.data, debug_vis_callback=b.update_visualizers)
    if self.args.scan_viz:
      data = b.scene["terrain_scan"].data
      add_scan_spheres(self.r.renderer.scene,
                       data.hit_pos_w[0].detach().cpu().numpy(),
                       (data.distances[0] >= 0).detach().cpu().numpy())
    view = Image.fromarray(self.r.render())
    s = inst.state
    if self.args.hud and s:
      draw_hud(view, s, header, notes, banner, self.scale)
    if not self.plot_h:
      return np.asarray(view)
    out = Image.new("RGB", (self.width, self.height))
    out.paste(view, (0, 0))
    out.paste(draw_plots(self.width, self.plot_h, s, list(inst.plot), scale=self.scale),
              (0, self.view_h))
    return np.asarray(out)

  def close(self):
    self.r.close()


class RecordingEnv:
  """Env proxy: same reset/step, plus measurement and frame capture."""

  def __init__(self, env, term, inst, on_frame, fps):
    self._env, self._term, self._inst = env, term, inst
    self._on_frame, self._interval, self._acc = on_frame, 1.0 / fps, 0.0
    self.first_reset_state = None

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def unwrapped(self):
    return self._env.unwrapped

  def reset(self, *a, **k):
    out = self._env.reset(*a, **k)
    self._inst.on_reset()
    if self.first_reset_state is None:
      d = self._env.unwrapped.sim.data
      self.first_reset_state = {
        "qpos": d.qpos[0].detach().cpu().double().numpy().tolist(),
        "qvel": d.qvel[0].detach().cpu().double().numpy().tolist(),
      }
    return out

  def step(self, action):
    self._inst.before_step(self._term)
    out = self._env.step(action)
    self._inst.after_step(out[2])
    self._acc += self._inst.dt
    while self._acc >= self._interval - 1e-9:
      self._on_frame(self._inst)
      self._acc -= self._interval
    return out


# ----------------------------------------------------------------- runners --

def make_env(task, ckpt, row, args):
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)
  env, policy, row = harness.build(task, str(ckpt), 1, row, args.device, seed=args.seed)
  return env, policy, row


def ckpt_tag(path):
  name = Path(path).stem
  return name


class FrameLog:
  """Collects per-frame HUD values for the control log."""

  def __init__(self):
    self.frames = []

  def add(self, s, extra=None):
    rec = {k: s[k] for k in ("step", "t", "dist", "cmd", "act", "mae_win_vx",
                             "mae_win_wz", "roll", "pitch", "terrain", "status", "xy")}
    rec["hud_text"] = hud_lines(s)
    if extra:
      rec.update(extra)
    self.frames.append(rec)


def raw_tensors(base):
  d = base.scene["robot"].data
  term = base.command_manager._terms["twist"]
  return {
    "root_link_lin_vel_b": d.root_link_lin_vel_b[0].tolist(),
    "root_link_ang_vel_b": d.root_link_ang_vel_b[0].tolist(),
    "root_link_quat_w": d.root_link_quat_w[0].tolist(),
    "root_link_pos_w": d.root_link_pos_w[0].tolist(),
    "vel_command_b": term.vel_command_b[0].tolist(),
  }


def run_route_segment(task, ckpt, row, layout_fn, args, sink, header, notes,
                      png_hook=None, panel=None):
  """Run route_eval.evaluate (cross_track) for one env, rendering frames."""
  env, policy, row = make_env(task, ckpt, row, args)
  base = env.unwrapped
  layout = layout_fn(base, row)
  start = route.configure_start(base, layout, 1.0, 0.15, 0.05)
  geom = {"row_x": float(base.scene.terrain.terrain_origins[row, 0, 0]),
          "half_x": layout["tile_size_m"][0] / 2, "finish_y": layout["seams_y"][-1]}
  inst = Instrument(base, row, geom)
  w, h = panel or (args.width, args.height)
  cam = Camera(base, w, h, args, args.plot_height)
  log = FrameLog()
  first_raw, last = {}, {}

  def on_frame(i):
    frame = cam.frame(i, header, notes)
    if not log.frames:
      first_raw.update(raw_tensors(base))
    log.add(i.state)
    last["raw"] = raw_tensors(base)
    last["frame"] = frame
    if png_hook:
      png_hook(i.state, frame)
    sink(frame)

  proxy = RecordingEnv(env, base.command_manager._terms["twist"], inst, on_frame, args.fps)
  result = route.evaluate(proxy, policy, row, layout, (0.5, 0.0, 0.0), args.duration,
                          **ROUTE_STEER)
  for _ in range(round(args.hold_end * args.fps)):
    sink(last["frame"])
  cam.close()
  env.close()
  checks = route_checks(inst, result)
  return {"task": task, "checkpoint": str(ckpt), "row": row, "seed": args.seed,
          "device": args.device, "start": start,
          "initial_state_qpos": proxy.first_reset_state["qpos"],
          "result_summary": {k: result[k] for k in (
            "passed", "fell", "left_row", "not_finished", "path_m_per_ep",
            "episode_len_s_per_ep", "overall_mae", "final_y_per_ep")},
          "checks": checks,
          "first_frame": dict(log.frames[0], raw=first_raw),
          "last_frame": dict(log.frames[-1], raw=last["raw"]),
          "n_frames": len(log.frames)}


def route_checks(inst, result):
  """HUD stream vs the evaluator's own aggregates on the same steps."""
  s = inst.state
  overall = result["overall_mae"]
  pairs = {
    "dist_vs_path_m": (s["dist"], result["path_m_per_ep"][0]),
    "t_vs_episode_len_s": (s["t"], result["episode_len_s_per_ep"][0]),
    "cum_mae_vx_vs_overall": (s["mae_cum"][0], overall["vx"]),
    "cum_mae_vy_vs_overall": (s["mae_cum"][1], overall["vy"]),
    "cum_mae_wz_vs_overall": (s["mae_cum"][2], overall["wz"]),
    "y_vs_final_y": (s["xy"][1], result["final_y_per_ep"][0]),
  }
  out = {}
  for k, (a, b) in pairs.items():
    # The evaluator sums time and path in float32 over thousands of steps;
    # its rounding drift (~1e-5 per 100 steps) sets these two tolerances.
    tol = 2e-3 if k in ("t_vs_episode_len_s", "dist_vs_path_m") else 1e-4
    out[k] = {"hud": a, "harness": b, "abs_diff": abs(a - b), "tol": tol,
              "ok": abs(a - b) < tol}
  hud_status = s["status"]
  expected = ("FELL" if result["fell"] else "LEFT ROW" if result["left_row"]
              else "OK")
  out["status"] = {"hud": hud_status, "harness_passed": result["passed"],
                   "harness_fell": result["fell"], "harness_left_row": result["left_row"],
                   "ok": hud_status == expected and
                   (bool(result["passed"]) == (s["finished"] and hud_status == "OK"))}
  out["all_ok"] = all(v["ok"] for v in out.values() if isinstance(v, dict))
  return out


def write_log(path, payload):
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=float))
  print(f"[hud] control log {path}")


def cmd_commands(args):
  segs = []
  for chunk in args.sequence.split(";"):
    label, spec, sec = chunk.split(":")
    vx, vy, wz = render_policy.parse_commands(f"{spec}:{sec}")[0][:3]
    segs.append((label, (vx, vy, wz), float(sec)))
  env, policy, row = make_env(args.task, args.checkpoint, 0, args)
  base = env.unwrapped
  term = base.command_manager._terms["twist"]
  inst = Instrument(base, None)
  cam = Camera(base, args.width, args.height, args, args.plot_height)
  writer = VideoWriter(Path(args.out), args.width, args.height, args.fps, args.crf)
  header = f"{ckpt_tag(args.checkpoint)} (release)  flat command sequence"
  log, first_raw, last = FrameLog(), {}, {}
  state = {"label": segs[0][0], "since": 0.0, "png": None}

  def on_frame(i):
    banner = (f"→ {state['label']}" if i.t - state["since"] < 1.5 else None)
    frame = cam.frame(i, header, [f"segment: {state['label']}"], banner)
    if not log.frames:
      first_raw.update(raw_tensors(base))
    log.add(i.state, {"segment": state["label"]})
    last["raw"], last["frame"] = raw_tensors(base), frame
    if state["png"] is None and state["label"] == "yaw +0.5" and i.t - state["since"] > 2.0:
      state["png"] = frame
    writer.write(frame)

  proxy = RecordingEnv(env, term, inst, on_frame, args.fps)
  render_policy._fix_command(term, segs[0][1])
  obs, _ = proxy.reset()
  render_policy._fix_command(term, segs[0][1])
  seg_mae = []
  for label, cmd, sec in segs:
    state["label"], state["since"] = label, inst.t
    e0, n0 = inst.err_sum.copy(), inst.step
    for _ in range(max(1, round(sec / base.step_dt))):
      render_policy._fix_command(term, cmd)
      with torch.no_grad():
        obs, _, dones, _ = proxy.step(policy(obs))
      if bool(dones[0]):
        break
    seg_mae.append({"segment": label, "cmd": list(cmd),
                    "mae_vx_vy_wz": ((inst.err_sum - e0) / max(1, inst.step - n0)).tolist()})
    if bool(dones[0]):
      print(f"[hud] episode ended during {label}: {inst.status}")
      break
  for _ in range(round(args.hold_end * args.fps)):
    writer.write(last["frame"])
  writer.close()
  if args.frame_png and state["png"] is not None:
    save_png(state["png"], Path(args.frame_png))
  cam.close()
  verify = verify_against_rollout(env, policy, term, args)
  env.close()
  write_log(Path(args.log), {
    "video": args.out, "task": args.task, "checkpoint": str(args.checkpoint),
    "seed": args.seed, "device": args.device, "status": inst.status,
    "segments": seg_mae, "first_frame": dict(log.frames[0], raw=first_raw),
    "last_frame": dict(log.frames[-1], raw=last["raw"]), "n_frames": writer.n,
    "rollout_crosscheck": verify})


def verify_against_rollout(env, policy, term, args):
  """Run harness.rollout() through the same instrument; compare aggregates."""
  out = []
  for cmd in ((0.5, 0.0, 0.0), (0.0, 0.0, 0.5)):
    base = env.unwrapped
    inst = Instrument(base, None)
    proxy = RecordingEnv(env, term, inst, lambda i: None, args.fps)
    res = harness.rollout(proxy, policy, cmd, 6.0, level_row=None)
    s = inst.state
    pairs = {"mae_vx": (s["mae_cum"][0], res["mae_vx_per_ep"][0]),
             "mae_vy": (s["mae_cum"][1], res["mae_vy_per_ep"][0]),
             "mae_wz": (s["mae_cum"][2], res["mae_wz_per_ep"][0]),
             "distance_m": (s["dist"], res["distance_m_per_ep"][0]),
             "t_s": (s["t"], res["episode_len_s_per_ep"][0])}
    rec = {k: {"hud": a, "harness": b, "abs_diff": abs(a - b),
               "ok": abs(a - b) < (2e-3 if k in ("t_s", "distance_m") else 1e-4)}
           for k, (a, b) in pairs.items()}
    rec["cmd"] = list(cmd)
    rec["all_ok"] = all(v["ok"] for v in rec.values() if isinstance(v, dict))
    out.append(rec)
  return out


def layout_route(base, row):
  return route.route_layout(base, row, route.FINAL_TASK)


def layout_mixed10(base, row):
  layout = route.route_layout(base, row, stairs10_tasks.MIXED_ROUTE_10)
  stair = base.scene.terrain.cfg.terrain_generator.sub_terrains["pyramid_stairs"]
  if tuple(stair.step_height_range) != (0.10, 0.10):
    raise RuntimeError("mixed route must contain fixed 10 cm steps")
  return layout


def cmd_route(args):
  writer = VideoWriter(Path(args.out), args.width, args.height, args.fps, args.crf)
  png = {}

  def hook(s, frame):
    if "f" not in png and s["terrain"].startswith("stairs"):
      png["f"] = frame
  rec = run_route_segment(route.FINAL_TASK, args.checkpoint, 4, layout_route, args,
                          writer.write,
                          f"{ckpt_tag(args.checkpoint)} (release)  route row 4",
                          [PLANNER_NOTE, "flat → rough → slope → stairs → flat"],
                          png_hook=hook)
  writer.close()
  if args.frame_png and "f" in png:
    save_png(png["f"], Path(args.frame_png))
  write_log(Path(args.log), {"video": args.out, "segments": [rec], "n_frames": writer.n})


def title_card(width, height, lines, frames, sink):
  img = Image.new("RGB", (width, height), (16, 16, 20))
  draw = ImageDraw.Draw(img)
  f = font(int(34 * width / 1280))
  y = height / 2 - len(lines) * f.size * 0.75
  for line in lines:
    tw = draw.textlength(line, font=f)
    draw.text(((width - tw) / 2, y), line, font=f, fill=(235, 235, 235))
    y += f.size * 1.5
  arr = np.asarray(img)
  for _ in range(frames):
    sink(arr)


def cmd_stairs10(args):
  writer = VideoWriter(Path(args.out), args.width, args.height, args.fps, args.crf)
  tag = ckpt_tag(args.checkpoint)
  plan = [
    ("regular 10 cm stairs — full crossing", stairs10_tasks.REGULAR_ROUTE, 9,
     evaluate_stairs.layout_for),
    ("inverted 10 cm stairs — full crossing", stairs10_tasks.INVERTED_ROUTE, 9,
     evaluate_stairs.layout_for),
    ("route row 4 with fixed 10 cm stairs", stairs10_tasks.MIXED_ROUTE_10, 4,
     layout_mixed10),
  ]
  recs, png = [], {}
  for k, (title, task, row, lay) in enumerate(plan, 1):
    title_card(args.width, args.height,
               [f"{k}/3  {title}", f"{tag}", EXPERIMENTAL_NOTE], args.fps, writer.write)

    def hook(s, frame, k=k):
      if k == 2 and "f" not in png and s["terrain"].startswith("inv stairs") and s["t"] > 6:
        png["f"] = frame
    recs.append(run_route_segment(task, args.checkpoint, row, lay, args, writer.write,
                                  f"{tag} (experimental)  {k}/3 {title}",
                                  [EXPERIMENTAL_NOTE, PLANNER_NOTE], png_hook=hook))
    recs[-1]["title"] = title
  writer.close()
  if args.frame_png and "f" in png:
    save_png(png["f"], Path(args.frame_png))
  write_log(Path(args.log), {"video": args.out, "segments": recs, "n_frames": writer.n})


def cmd_compare(args):
  pw, ph = args.width // 2, args.height
  panels, recs = [], []
  for ckpt, note in ((args.checkpoint, "release"), (args.checkpoint_b, "experimental")):
    frames = []
    sink = lambda f, frames=frames: frames.append(zlib.compress(f.tobytes(), 1))
    tag = ckpt_tag(ckpt)
    rec = run_route_segment(stairs10_tasks.INVERTED_ROUTE, ckpt, 9, evaluate_stairs.layout_for,
                            args, sink, f"{tag} ({note})",
                            ["inverted 10 cm stairs", f"seed {args.seed}"], panel=(pw, ph))
    rec["label"] = f"{tag} ({note})"
    panels.append(frames)
    recs.append(rec)
  same_init = recs[0]["initial_state_qpos"] == recs[1]["initial_state_qpos"]
  writer = VideoWriter(Path(args.out), pw * 2, ph, args.fps, args.crf)
  n = max(len(p) for p in panels)
  png_idx = min(len(panels[0]) - 1 + args.fps, n - 1)
  for i in range(n):
    parts = [np.frombuffer(zlib.decompress(p[min(i, len(p) - 1)]), np.uint8).reshape(ph, pw, 3)
             for p in panels]
    frame = np.concatenate(parts, axis=1)
    frame[:, pw - 1:pw + 1] = 255
    writer.write(frame)
    if args.frame_png and i == png_idx:
      save_png(frame, Path(args.frame_png))
  writer.close()
  outcome = {r["label"]: ("passed" if r["result_summary"]["passed"] else
                          "fell" if r["result_summary"]["fell"] else
                          "left_row" if r["result_summary"]["left_row"] else "timeout")
             for r in recs}
  for r in recs:
    r.pop("initial_state_qpos")
  write_log(Path(args.log), {"video": args.out, "seed": args.seed,
                             "identical_initial_state": same_init,
                             "outcome": outcome, "segments": recs, "n_frames": writer.n})
  print(f"[hud] outcome seed {args.seed}: {outcome}; identical initial state: {same_init}")


def main():
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument("mode", choices=("commands", "route", "stairs10", "compare"))
  ap.add_argument("--checkpoint", required=True, type=Path)
  ap.add_argument("--checkpoint-b", type=Path, help="compare: right panel")
  ap.add_argument("--task", default=FLAT_TASK, help="commands mode task")
  ap.add_argument("--sequence", default=FLAT_SEQUENCE,
                  help='commands: "label:vx,vy,wz:sec;..."')
  ap.add_argument("--out", required=True)
  ap.add_argument("--log", help="control-value JSON (default: outputs/video_hud/<name>.json)")
  ap.add_argument("--frame-png")
  ap.add_argument("--hud", action="store_true")
  ap.add_argument("--plots", action="store_true")
  ap.add_argument("--scan-viz", action="store_true")
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--device", default="cpu")
  ap.add_argument("--duration", type=float, default=60.0)
  ap.add_argument("--width", type=int, default=1280)
  ap.add_argument("--height", type=int, default=720)
  ap.add_argument("--plot-height", type=int, default=170)
  ap.add_argument("--fps", type=int, default=30)
  ap.add_argument("--crf", type=int, default=26)
  ap.add_argument("--hold-end", type=float, default=1.5)
  ap.add_argument("--cam-distance", type=float, default=3.6)
  ap.add_argument("--cam-elevation", type=float, default=-20.0)
  ap.add_argument("--cam-azimuth", type=float, default=145.0)
  args = ap.parse_args()
  if args.log is None:
    args.log = str(ROOT / "outputs" / "video_hud" / (Path(args.out).stem + ".json"))
  if args.mode == "compare" and args.checkpoint_b is None:
    ap.error("compare needs --checkpoint-b")
  print(f"[hud] MUJOCO_GL={os.environ['MUJOCO_GL']} device={args.device}")
  {"commands": cmd_commands, "route": cmd_route, "stairs10": cmd_stairs10,
   "compare": cmd_compare}[args.mode](args)


if __name__ == "__main__":
  main()
