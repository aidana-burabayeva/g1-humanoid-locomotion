"""Register a mixed flat and rough G1 task."""

from __future__ import annotations

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
from mjlab.terrains.config import flat, random_rough

from .deploy_no_cmd_curriculum import rough_deploy_no_cmd_curriculum_env_cfg

TASK_ID = "G1-Mix-Deploy-NoCmdCurriculum"

MIX_NOISE_RANGE = (0.02, 0.06)


def mix_sub_terrains() -> dict:
  return {
    "flat": flat(proportion=0.5),
    "random_rough": random_rough(proportion=0.5, noise_range=MIX_NOISE_RANGE),
  }


def mix_deploy_no_cmd_curriculum_env_cfg(play: bool = False):
  cfg = rough_deploy_no_cmd_curriculum_env_cfg(play=play)
  tg = cfg.scene.terrain.terrain_generator
  assert tg is not None, "expected a rough terrain generator"
  tg.sub_terrains = mix_sub_terrains()
  return cfg


def register() -> None:
  register_mjlab_task(
    task_id=TASK_ID,
    env_cfg=mix_deploy_no_cmd_curriculum_env_cfg(),
    play_env_cfg=mix_deploy_no_cmd_curriculum_env_cfg(play=True),
    rl_cfg=unitree_g1_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
  )
