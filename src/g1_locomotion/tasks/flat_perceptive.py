"""Keep height-scan observations in the G1 flat task."""

from __future__ import annotations

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_rough_env_cfg
from mjlab.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

TASK_ID = "G1-Flat-Perceptive"


EXPECTED_SCAN = 187
EXPECTED_ACTOR_DIM = 286


def flat_perceptive_env_cfg(play: bool = False):
  """Keep height scans in the flat G1 task."""
  cfg = unitree_g1_rough_env_cfg(play=play)


  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None


  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None


  cfg.terminations.pop("out_of_terrain_bounds", None)


  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-1.5, 2.0)
    twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)

  _assert_scan_present(cfg)
  return cfg


def _assert_scan_present(cfg) -> None:
  """Verify the scan sensor and observation terms."""
  names = {s.name for s in (cfg.scene.sensors or ())}
  assert "terrain_scan" in names, f"terrain_scan sensor missing; found {names}"
  for group in ("actor", "critic"):
    terms = cfg.observations[group].terms
    assert "height_scan" in terms, f"height_scan term missing from {group}"


def register() -> None:
  register_mjlab_task(
    task_id=TASK_ID,
    env_cfg=flat_perceptive_env_cfg(),
    play_env_cfg=flat_perceptive_env_cfg(play=True),
    rl_cfg=unitree_g1_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
  )
