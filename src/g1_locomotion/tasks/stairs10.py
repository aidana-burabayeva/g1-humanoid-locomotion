"""Register 10 cm stair training and full-crossing evaluation tasks."""

from __future__ import annotations

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
from mjlab.terrains.config import flat, pyramid_stairs, pyramid_stairs_inv

from g1_locomotion.tasks.flat_deploy_no_cmd_curriculum_symloss import symloss_runner_cfg
from g1_locomotion.tasks.stage4_stairs5_eval_tasks import _final_route_env_cfg
from g1_locomotion.tasks.mix4_stairs5_deploy_no_cmd_curriculum_symloss_yaww4 import (
  STAIR_WIDTH,
  mix4_stairs5_deploy_no_cmd_curriculum_symloss_yaww4_env_cfg,
)

TRAIN_TASK = "G1-Mix4-Stairs10-Deploy-NoCmdCurriculum-SymLoss-YawW4"
FOCUSED_TRAIN_TASK = "G1-Mix4-Stairs7p5to10-Deploy-NoCmdCurriculum-SymLoss-YawW4"
TRACKING_TRAIN_TASK = "G1-Mix4-Stairs7p5to10-TrackLinW2p5-Deploy-SymLoss-YawW4"
REGULAR_ROUTE = "G1-Stairs10-FullCrossing-Deploy"
INVERTED_ROUTE = "G1-StairsInv10-FullCrossing-Deploy"
REGULAR_ROUTE_5 = "G1-Stairs5-FullCrossing-Deploy"
INVERTED_ROUTE_5 = "G1-StairsInv5-FullCrossing-Deploy"
MIXED_ROUTE_10 = "G1-FinalRoute-StairsFixed10-Deploy"
HEIGHT_RANGE = (0.0, 0.10)
FOCUSED_HEIGHT_RANGE = (0.075, 0.10)
TRACK_LINEAR_WEIGHT = 2.5


def training_env_cfg(play: bool = False):
  cfg = mix4_stairs5_deploy_no_cmd_curriculum_symloss_yaww4_env_cfg(play=play)
  generator = cfg.scene.terrain.terrain_generator
  assert generator is not None
  generator.sub_terrains["pyramid_stairs"] = pyramid_stairs(
    proportion=0.20, step_height_range=HEIGHT_RANGE, step_width=STAIR_WIDTH
  )
  generator.sub_terrains["pyramid_stairs_inv"] = pyramid_stairs_inv(
    proportion=0.10, step_height_range=HEIGHT_RANGE, step_width=STAIR_WIDTH
  )
  return cfg


def focused_training_env_cfg(play: bool = False):
  cfg = training_env_cfg(play=play)
  generator = cfg.scene.terrain.terrain_generator
  assert generator is not None
  generator.sub_terrains["pyramid_stairs"] = pyramid_stairs(
    proportion=0.20, step_height_range=FOCUSED_HEIGHT_RANGE,
    step_width=STAIR_WIDTH
  )
  generator.sub_terrains["pyramid_stairs_inv"] = pyramid_stairs_inv(
    proportion=0.10, step_height_range=FOCUSED_HEIGHT_RANGE,
    step_width=STAIR_WIDTH
  )
  return cfg


def tracking_training_env_cfg(play: bool = False):
  cfg = focused_training_env_cfg(play=play)
  cfg.rewards["track_linear_velocity"].weight = TRACK_LINEAR_WEIGHT
  return cfg


def crossing_env_cfg(inverted: bool, play: bool = False,
                     height_range: tuple[float, float] = HEIGHT_RANGE):
  cfg = training_env_cfg(play=play)
  generator = cfg.scene.terrain.terrain_generator
  assert generator is not None
  stairs = pyramid_stairs_inv if inverted else pyramid_stairs
  generator.sub_terrains = {
    "flat": flat(proportion=1 / 3),
    "stairs": stairs(
      proportion=1 / 3, step_height_range=height_range, step_width=STAIR_WIDTH
    ),
    "flat_landing": flat(proportion=1 / 3),
  }
  generator.curriculum = True
  generator.num_cols = 3
  return cfg


def mixed_route_env_cfg(play: bool = False):
  return _final_route_env_cfg(play=play, height_range=(0.10, 0.10))


def register() -> None:
  for task_id, factory in (
    (TRAIN_TASK, training_env_cfg),
    (FOCUSED_TRAIN_TASK, focused_training_env_cfg),
    (TRACKING_TRAIN_TASK, tracking_training_env_cfg),
    (REGULAR_ROUTE, lambda play=False: crossing_env_cfg(False, play)),
    (INVERTED_ROUTE, lambda play=False: crossing_env_cfg(True, play)),
    (REGULAR_ROUTE_5,
     lambda play=False: crossing_env_cfg(False, play, (0.0, 0.05))),
    (INVERTED_ROUTE_5,
     lambda play=False: crossing_env_cfg(True, play, (0.0, 0.05))),
    (MIXED_ROUTE_10, mixed_route_env_cfg),
  ):
    register_mjlab_task(
      task_id=task_id,
      env_cfg=factory(),
      play_env_cfg=factory(play=True),
      rl_cfg=symloss_runner_cfg(),
      runner_cls=VelocityOnPolicyRunner,
    )
