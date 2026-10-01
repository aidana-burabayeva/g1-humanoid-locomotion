"""Register isolated stair, terrain-boundary, and continuous-route tasks."""

from __future__ import annotations

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
from mjlab.terrains.config import (
  flat,
  hf_pyramid_slope,
  pyramid_stairs,
  pyramid_stairs_inv,
  random_rough,
)

from .flat_deploy_no_cmd_curriculum_symloss import symloss_runner_cfg
from .mix4_stairs5_deploy_no_cmd_curriculum_symloss_yaww4 import (
  STAIR_RANGE,
  STAIR_WIDTH,
  mix4_stairs5_deploy_no_cmd_curriculum_symloss_yaww4_env_cfg,
)
from .mix3_deploy_no_cmd_curriculum_symloss_yaww4 import SLOPE_RANGE
from .mix3_deploy_no_cmd_curriculum_symloss_yaww4 import MIX_NOISE_RANGE

STAIRS_TASK = "G1-Stairs5-Deploy-NoCmdCurriculum-SymLoss-YawW4"
STAIRS_INV_TASK = "G1-StairsInv5-Deploy-NoCmdCurriculum-SymLoss-YawW4"
STAIRS10_TASK = "G1-Stairs10-Deploy-NoCmdCurriculum-SymLoss-YawW4"
STAIRS_INV10_TASK = "G1-StairsInv10-Deploy-NoCmdCurriculum-SymLoss-YawW4"
FLAT_STAIRS_TASK = "G1-FlatStairs5Strip-Deploy-NoCmdCurriculum-SymLoss-YawW4"
SLOPE_STAIRS_TASK = "G1-SlopeStairs5Strip-Deploy-NoCmdCurriculum-SymLoss-YawW4"
FINAL_ROUTE_TASK = "G1-FinalRoute-Stairs5-Deploy-NoCmdCurriculum-SymLoss-YawW4"
FINAL_ROUTE10_TASK = "G1-FinalRoute-Stairs10-Deploy-NoCmdCurriculum-SymLoss-YawW4"
STAIR_RANGE_10 = (0.0, 0.10)


def _env_cfg(factory, play: bool, height_range=STAIR_RANGE):
  cfg = mix4_stairs5_deploy_no_cmd_curriculum_symloss_yaww4_env_cfg(play=play)
  tg = cfg.scene.terrain.terrain_generator
  assert tg is not None
  tg.sub_terrains = {
    f"stairs_{i}": factory(
      proportion=0.1,
      step_height_range=height_range,
      step_width=STAIR_WIDTH,
    )
    for i in range(10)
  }
  tg.curriculum = True
  tg.num_cols = len(tg.sub_terrains)
  return cfg


def _strip_env_cfg(first: str, play: bool):
  cfg = mix4_stairs5_deploy_no_cmd_curriculum_symloss_yaww4_env_cfg(play=play)
  tg = cfg.scene.terrain.terrain_generator
  assert tg is not None
  if first == "flat":
    before_factory, before_kwargs = flat, {}
  elif first == "hf_pyramid_slope":
    before_factory, before_kwargs = hf_pyramid_slope, {"slope_range": SLOPE_RANGE}
  else:
    raise ValueError(first)
  tg.sub_terrains = {
    **{f"{first}_{i}": before_factory(proportion=0.125, **before_kwargs)
       for i in range(4)},
    **{f"pyramid_stairs_{i}": pyramid_stairs(
      proportion=0.125,
      step_height_range=STAIR_RANGE,
      step_width=STAIR_WIDTH,
    ) for i in range(4)},
  }
  tg.curriculum = True
  tg.num_cols = len(tg.sub_terrains)
  return cfg


def _final_route_env_cfg(play: bool, height_range=STAIR_RANGE):
  cfg = mix4_stairs5_deploy_no_cmd_curriculum_symloss_yaww4_env_cfg(play=play)
  tg = cfg.scene.terrain.terrain_generator
  assert tg is not None
  tg.sub_terrains = {
    "flat": flat(proportion=0.2),
    "random_rough": random_rough(
      proportion=0.2, noise_range=MIX_NOISE_RANGE
    ),
    "hf_pyramid_slope": hf_pyramid_slope(
      proportion=0.2, slope_range=SLOPE_RANGE
    ),
    "pyramid_stairs": pyramid_stairs(
      proportion=0.2, step_height_range=height_range, step_width=STAIR_WIDTH
    ),
    "flat_landing": flat(proportion=0.2),
  }
  tg.curriculum = True
  tg.num_cols = len(tg.sub_terrains)
  return cfg


def register() -> None:
  for task_id, factory, height_range in (
    (STAIRS_TASK, pyramid_stairs, STAIR_RANGE),
    (STAIRS_INV_TASK, pyramid_stairs_inv, STAIR_RANGE),
    (STAIRS10_TASK, pyramid_stairs, STAIR_RANGE_10),
    (STAIRS_INV10_TASK, pyramid_stairs_inv, STAIR_RANGE_10),
  ):
    register_mjlab_task(
      task_id=task_id,
      env_cfg=_env_cfg(factory, play=False, height_range=height_range),
      play_env_cfg=_env_cfg(factory, play=True, height_range=height_range),
      rl_cfg=symloss_runner_cfg(),
      runner_cls=VelocityOnPolicyRunner,
    )
  for task_id, first in (
    (FLAT_STAIRS_TASK, "flat"),
    (SLOPE_STAIRS_TASK, "hf_pyramid_slope"),
  ):
    register_mjlab_task(
      task_id=task_id,
      env_cfg=_strip_env_cfg(first, play=False),
      play_env_cfg=_strip_env_cfg(first, play=True),
      rl_cfg=symloss_runner_cfg(),
      runner_cls=VelocityOnPolicyRunner,
    )
  for task_id, height_range in (
    (FINAL_ROUTE_TASK, STAIR_RANGE),
    (FINAL_ROUTE10_TASK, STAIR_RANGE_10),
  ):
    register_mjlab_task(
      task_id=task_id,
      env_cfg=_final_route_env_cfg(play=False, height_range=height_range),
      play_env_cfg=_final_route_env_cfg(play=True, height_range=height_range),
      rl_cfg=symloss_runner_cfg(),
      runner_cls=VelocityOnPolicyRunner,
    )
