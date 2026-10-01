"""Remove unavailable linear velocity from actor observations."""

from __future__ import annotations

import copy

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_rough_env_cfg
from mjlab.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .flat_perceptive import flat_perceptive_env_cfg

ROUGH_TASK_ID = "G1-Rough-Deploy"
FLAT_TASK_ID = "G1-Flat-Deploy"


UNMEASURED_TERMS = ("base_lin_vel",)


def detach_actor_terms(cfg) -> None:
  """Copy actor terms so critic settings remain independent."""
  actor = cfg.observations["actor"]
  for name, term in list(actor.terms.items()):
    actor.terms[name] = copy.deepcopy(term)


def drop_unmeasured(cfg) -> list[str]:
  """Remove actor terms unavailable to the deployed policy."""
  detach_actor_terms(cfg)
  actor = cfg.observations["actor"]
  removed = []
  for term in UNMEASURED_TERMS:
    if term in actor.terms:
      del actor.terms[term]
      removed.append(term)

  critic = cfg.observations["critic"]
  for term in removed:
    assert term in critic.terms, f"{term} missing from critic observations"
  return removed


def rough_deploy_env_cfg(play: bool = False):
  """Remove base linear velocity from the rough-task actor."""
  cfg = unitree_g1_rough_env_cfg(play=play)
  removed = drop_unmeasured(cfg)
  if removed:
    print(f"[g1_locomotion] {ROUGH_TASK_ID}: removed actor terms {removed}")
  return cfg


def flat_deploy_env_cfg(play: bool = False):
  """Remove base linear velocity from the flat-task actor."""
  cfg = flat_perceptive_env_cfg(play=play)
  removed = drop_unmeasured(cfg)
  if removed:
    print(f"[g1_locomotion] {FLAT_TASK_ID}: removed actor terms {removed}")
  return cfg


def register() -> None:
  register_mjlab_task(
    task_id=ROUGH_TASK_ID,
    env_cfg=rough_deploy_env_cfg(),
    play_env_cfg=rough_deploy_env_cfg(play=True),
    rl_cfg=unitree_g1_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
  )
  register_mjlab_task(
    task_id=FLAT_TASK_ID,
    env_cfg=flat_deploy_env_cfg(),
    play_env_cfg=flat_deploy_env_cfg(play=True),
    rl_cfg=unitree_g1_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
  )
