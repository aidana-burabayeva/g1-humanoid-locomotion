"""Disable velocity-command curriculum for the G1 flat task."""

from __future__ import annotations

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .deploy_obs import flat_deploy_env_cfg

TASK_ID = "G1-Flat-Deploy-NoCmdCurriculum"


def flat_deploy_no_cmd_curriculum_env_cfg(play: bool = False):
  cfg = flat_deploy_env_cfg(play=play)

  cfg.curriculum.pop("command_vel", None)
  return cfg


def register() -> None:
  register_mjlab_task(
    task_id=TASK_ID,
    env_cfg=flat_deploy_no_cmd_curriculum_env_cfg(),
    play_env_cfg=flat_deploy_no_cmd_curriculum_env_cfg(play=True),
    rl_cfg=unitree_g1_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
  )
