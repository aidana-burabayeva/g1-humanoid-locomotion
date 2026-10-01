"""Register isolated rough terrain and flat-to-rough route tasks."""

from __future__ import annotations

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
from mjlab.terrains.config import flat, random_rough

from .flat_deploy_no_cmd_curriculum_symloss import symloss_runner_cfg
from .mix_deploy_no_cmd_curriculum import MIX_NOISE_RANGE
from .mix_deploy_no_cmd_curriculum_symloss_yaww4 import (
  mix_deploy_no_cmd_curriculum_symloss_yaww4_env_cfg,
)

RANDOM_ROUGH_TASK_ID = "G1-RandomRough-Deploy-NoCmdCurriculum-SymLoss-YawW4"
STRIP_TASK_ID = "G1-FlatRoughStrip-Deploy-NoCmdCurriculum-SymLoss-YawW4"

RANDOM_ROUGH_COLS = 10
STRIP_FLAT_COLS = 4
STRIP_ROUGH_COLS = 4


def random_rough_only_sub_terrains(n_cols: int = RANDOM_ROUGH_COLS) -> dict:
  p = 1.0 / n_cols
  return {f"random_rough_{i}": random_rough(proportion=p, noise_range=MIX_NOISE_RANGE)
          for i in range(n_cols)}


def strip_sub_terrains(n_flat: int = STRIP_FLAT_COLS,
                       n_rough: int = STRIP_ROUGH_COLS) -> dict:
  """Place flat and rough terrain in adjacent columns."""
  p = 1.0 / (n_flat + n_rough)
  sub = {f"flat_{i}": flat(proportion=p) for i in range(n_flat)}
  sub.update({f"random_rough_{i}": random_rough(proportion=p, noise_range=MIX_NOISE_RANGE)
              for i in range(n_rough)})
  return sub


def _with_sub_terrains(sub_terrains: dict, play: bool):
  cfg = mix_deploy_no_cmd_curriculum_symloss_yaww4_env_cfg(play=play)
  tg = cfg.scene.terrain.terrain_generator
  assert tg is not None, "expected a terrain generator"
  tg.sub_terrains = sub_terrains
  tg.curriculum = True
  tg.num_cols = len(sub_terrains)
  return cfg


def random_rough_only_env_cfg(play: bool = False):
  return _with_sub_terrains(random_rough_only_sub_terrains(), play)


def strip_env_cfg(play: bool = False):
  return _with_sub_terrains(strip_sub_terrains(), play)


def register() -> None:
  register_mjlab_task(
    task_id=RANDOM_ROUGH_TASK_ID,
    env_cfg=random_rough_only_env_cfg(),
    play_env_cfg=random_rough_only_env_cfg(play=True),
    rl_cfg=symloss_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
  )
  register_mjlab_task(
    task_id=STRIP_TASK_ID,
    env_cfg=strip_env_cfg(),
    play_env_cfg=strip_env_cfg(play=True),
    rl_cfg=symloss_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
  )
