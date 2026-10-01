"""Register a G1 task with flat, rough, and sloped terrain."""

from __future__ import annotations

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
from mjlab.terrains.config import (
  flat,
  hf_pyramid_slope,
  hf_pyramid_slope_inv,
  random_rough,
)

from .flat_deploy_no_cmd_curriculum_symloss import symloss_runner_cfg
from .mix_deploy_no_cmd_curriculum_symloss_yaww4 import (
  mix_deploy_no_cmd_curriculum_symloss_yaww4_env_cfg,
)

TASK_ID = "G1-Mix3-Deploy-NoCmdCurriculum-SymLoss-YawW4"


SLOPE_RANGE = (0.0, 0.4)

MIX_NOISE_RANGE = (0.02, 0.06)


def mix3_sub_terrains() -> dict:
  return {
    "flat": flat(proportion=0.3),
    "random_rough": random_rough(proportion=0.3, noise_range=MIX_NOISE_RANGE),
    "hf_pyramid_slope": hf_pyramid_slope(
      proportion=0.2, slope_range=SLOPE_RANGE
    ),
    "hf_pyramid_slope_inv": hf_pyramid_slope_inv(
      proportion=0.2, slope_range=SLOPE_RANGE
    ),
  }


def mix3_deploy_no_cmd_curriculum_symloss_yaww4_env_cfg(play: bool = False):
  cfg = mix_deploy_no_cmd_curriculum_symloss_yaww4_env_cfg(play=play)
  tg = cfg.scene.terrain.terrain_generator
  assert tg is not None, "expected a rough terrain generator"
  tg.sub_terrains = mix3_sub_terrains()
  return cfg


def register() -> None:
  register_mjlab_task(
    task_id=TASK_ID,
    env_cfg=mix3_deploy_no_cmd_curriculum_symloss_yaww4_env_cfg(),
    play_env_cfg=mix3_deploy_no_cmd_curriculum_symloss_yaww4_env_cfg(play=True),
    rl_cfg=symloss_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
  )
