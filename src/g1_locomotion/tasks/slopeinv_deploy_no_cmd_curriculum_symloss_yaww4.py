"""Register an isolated reverse-slope G1 task."""

from __future__ import annotations

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
from mjlab.terrains.config import hf_pyramid_slope_inv

from .flat_deploy_no_cmd_curriculum_symloss import symloss_runner_cfg
from .mix3_deploy_no_cmd_curriculum_symloss_yaww4 import (
  SLOPE_RANGE,
  mix3_deploy_no_cmd_curriculum_symloss_yaww4_env_cfg,
)

TASK_ID = "G1-SlopeInv-Deploy-NoCmdCurriculum-SymLoss-YawW4"

SLOPE_COLS = 10


def slope_inv_only_sub_terrains(n_cols: int = SLOPE_COLS) -> dict:
  p = 1.0 / n_cols
  return {
    f"hf_pyramid_slope_inv_{i}": hf_pyramid_slope_inv(
      proportion=p, slope_range=SLOPE_RANGE
    )
    for i in range(n_cols)
  }


def slopeinv_deploy_no_cmd_curriculum_symloss_yaww4_env_cfg(play: bool = False):
  cfg = mix3_deploy_no_cmd_curriculum_symloss_yaww4_env_cfg(play=play)
  tg = cfg.scene.terrain.terrain_generator
  assert tg is not None, "expected a rough terrain generator"
  tg.sub_terrains = slope_inv_only_sub_terrains()
  tg.curriculum = True
  tg.num_cols = len(tg.sub_terrains)
  return cfg


def register() -> None:
  register_mjlab_task(
    task_id=TASK_ID,
    env_cfg=slopeinv_deploy_no_cmd_curriculum_symloss_yaww4_env_cfg(),
    play_env_cfg=slopeinv_deploy_no_cmd_curriculum_symloss_yaww4_env_cfg(play=True),
    rl_cfg=symloss_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
  )
