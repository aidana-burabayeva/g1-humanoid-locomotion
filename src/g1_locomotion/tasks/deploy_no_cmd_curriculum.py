"""Disable velocity-command curriculum for the G1 rough task."""

from __future__ import annotations

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .deploy_obs import rough_deploy_env_cfg

TASK_ID = "G1-Rough-Deploy-NoCmdCurriculum"


def rough_deploy_no_cmd_curriculum_env_cfg(play: bool = False):
  cfg = rough_deploy_env_cfg(play=play)

  cfg.curriculum.pop("command_vel", None)
  return cfg


def register() -> None:
  register_mjlab_task(
    task_id=TASK_ID,
    env_cfg=rough_deploy_no_cmd_curriculum_env_cfg(),
    play_env_cfg=rough_deploy_no_cmd_curriculum_env_cfg(play=True),
    rl_cfg=unitree_g1_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
  )
