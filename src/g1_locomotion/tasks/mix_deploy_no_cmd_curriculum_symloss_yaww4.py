"""Configure mixed-terrain symmetry loss and yaw weighting."""

from __future__ import annotations

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .flat_deploy_no_cmd_curriculum_symloss import symloss_runner_cfg
from .mix_deploy_no_cmd_curriculum import mix_deploy_no_cmd_curriculum_env_cfg

TASK_ID = "G1-Mix-Deploy-NoCmdCurriculum-SymLoss-YawW4"


def mix_deploy_no_cmd_curriculum_symloss_yaww4_env_cfg(play: bool = False):
  cfg = mix_deploy_no_cmd_curriculum_env_cfg(play=play)
  cfg.rewards["track_angular_velocity"].weight = 4.0
  return cfg


def register() -> None:
  register_mjlab_task(
    task_id=TASK_ID,
    env_cfg=mix_deploy_no_cmd_curriculum_symloss_yaww4_env_cfg(),
    play_env_cfg=mix_deploy_no_cmd_curriculum_symloss_yaww4_env_cfg(play=True),
    rl_cfg=symloss_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
  )
