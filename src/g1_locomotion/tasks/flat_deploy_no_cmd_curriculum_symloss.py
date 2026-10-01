"""Configure symmetry loss for the G1 flat task."""

from __future__ import annotations

from dataclasses import dataclass, field

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .flat_deploy_no_cmd_curriculum import flat_deploy_no_cmd_curriculum_env_cfg
from .flat_deploy_no_cmd_curriculum_sym import (
  AUG_FUNC,
  RslRlOnPolicyRunnerSymCfg,
  RslRlPpoSymAlgorithmCfg,
  SymmetryCfg,
)
from mjlab.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg

TASK_ID = "G1-Flat-Deploy-NoCmdCurriculum-SymLoss"


@dataclass
class SymLossCfg(SymmetryCfg):
  use_data_augmentation: bool = False
  use_mirror_loss: bool = True
  mirror_loss_coeff: float = 0.5
  data_augmentation_func: str = AUG_FUNC


def symloss_runner_cfg() -> RslRlOnPolicyRunnerSymCfg:
  """Attach symmetry loss to the G1 PPO runner."""
  base = unitree_g1_ppo_runner_cfg()
  kw = {k: getattr(base, k) for k in base.__dataclass_fields__}
  alg_kw = {k: getattr(base.algorithm, k) for k in base.algorithm.__dataclass_fields__}
  kw["algorithm"] = RslRlPpoSymAlgorithmCfg(**alg_kw, symmetry_cfg=SymLossCfg())
  return RslRlOnPolicyRunnerSymCfg(**kw)


def register() -> None:
  register_mjlab_task(
    task_id=TASK_ID,
    env_cfg=flat_deploy_no_cmd_curriculum_env_cfg(),
    play_env_cfg=flat_deploy_no_cmd_curriculum_env_cfg(play=True),
    rl_cfg=symloss_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
  )
