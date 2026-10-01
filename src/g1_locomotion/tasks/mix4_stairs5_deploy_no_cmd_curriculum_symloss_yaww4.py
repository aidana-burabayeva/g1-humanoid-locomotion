"""Add 5 cm stairs to the mixed-terrain G1 training task."""

from __future__ import annotations

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
from mjlab.terrains.config import (
  flat,
  hf_pyramid_slope,
  hf_pyramid_slope_inv,
  pyramid_stairs,
  pyramid_stairs_inv,
  random_rough,
)

from .flat_deploy_no_cmd_curriculum_symloss import symloss_runner_cfg
from .mix3_deploy_no_cmd_curriculum_symloss_yaww4 import (
  MIX_NOISE_RANGE,
  SLOPE_RANGE,
  mix3_deploy_no_cmd_curriculum_symloss_yaww4_env_cfg,
)

TASK_ID = "G1-Mix4-Stairs5-Deploy-NoCmdCurriculum-SymLoss-YawW4"
STAIR_RANGE = (0.0, 0.05)
STAIR_WIDTH = 0.40


def mix4_sub_terrains() -> dict:
  return {
    "flat": flat(proportion=0.25),
    "random_rough": random_rough(
      proportion=0.20, noise_range=MIX_NOISE_RANGE
    ),
    "hf_pyramid_slope": hf_pyramid_slope(
      proportion=0.15, slope_range=SLOPE_RANGE
    ),
    "hf_pyramid_slope_inv": hf_pyramid_slope_inv(
      proportion=0.10, slope_range=SLOPE_RANGE
    ),
    "pyramid_stairs": pyramid_stairs(
      proportion=0.20, step_height_range=STAIR_RANGE, step_width=STAIR_WIDTH
    ),
    "pyramid_stairs_inv": pyramid_stairs_inv(
      proportion=0.10, step_height_range=STAIR_RANGE, step_width=STAIR_WIDTH
    ),
  }


def mix4_stairs5_deploy_no_cmd_curriculum_symloss_yaww4_env_cfg(
  play: bool = False,
):
  cfg = mix3_deploy_no_cmd_curriculum_symloss_yaww4_env_cfg(play=play)
  tg = cfg.scene.terrain.terrain_generator
  assert tg is not None
  tg.sub_terrains = mix4_sub_terrains()
  return cfg


def register() -> None:
  register_mjlab_task(
    task_id=TASK_ID,
    env_cfg=mix4_stairs5_deploy_no_cmd_curriculum_symloss_yaww4_env_cfg(),
    play_env_cfg=mix4_stairs5_deploy_no_cmd_curriculum_symloss_yaww4_env_cfg(
      play=True
    ),
    rl_cfg=symloss_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
  )
