"""Test: partial reset must not corrupt observation history of other environments.

Checked against the REAL `mjlab.managers.observation_manager.ObservationManager`
version 1.5.3, but without physics: the observation is a counter, one per
environment. Physics is not needed for this check because the defect is purely
a buffering issue.

Reproducible defect: `ManagerBasedRlEnv.reset(env_ids=...)`
(`envs/manager_based_rl_env.py:374`) ends with
`observation_manager.compute(update_history=True)`, while `CircularBuffer.append`
(`utils/buffers/circular_buffer.py:190`) writes a frame for ALL environments at
once. So every partial reset adds an extra frame -- without a physics step --
to environments that have not terminated.

Run: python tests/test_partial_reset.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mjlab.managers.observation_manager import (  # noqa: E402
    ObservationGroupCfg,
    ObservationManager,
    ObservationTermCfg,
)

from g1_locomotion.harness import restore_obs_history, snapshot_obs_history  # noqa: E402


class FakeEnv:
  """The minimum ManagerBase asks for: number of environments and device."""

  def __init__(self, num_envs):
    self.num_envs = num_envs
    self.device = "cpu"


def make_manager(num_envs, history_length, delay=0):
  clock = {"t": 0.0}

  def tick(env):
    # Environment i returns 100*i + t: each environment's history is visible
    # separately.
    base = torch.arange(num_envs, dtype=torch.float32) * 100.0
    return (base + clock["t"]).unsqueeze(-1)

  term = ObservationTermCfg(func=tick)
  term.delay_min_lag = 0
  term.delay_max_lag = delay
  cfg = {"actor": ObservationGroupCfg(terms={"tick": term},
                                      history_length=history_length,
                                      flatten_history_dim=True)}
  return ObservationManager(cfg, FakeEnv(num_envs)), clock


def hist(om):
  return om.compute()["actor"].tolist()


def partial_reset(om, done_ids, num_envs, fixed=True):
  """Same as what mjlab does on `reset(env_ids=done_ids)`.

  `fixed=True` -- with the wrapper from `harness.py`.
  """
  keep = torch.ones(num_envs, dtype=torch.bool)
  keep[done_ids] = False
  keep_ids = keep.nonzero(as_tuple=False).squeeze(-1)
  state = snapshot_obs_history(om, keep_ids) if fixed and keep_ids.numel() else None
  om.reset(env_ids=done_ids)
  om.compute(update_history=True)          # <- line 374 of manager_based_rl_env
  if state is not None:
    restore_obs_history(om, state)


def check(name, got, expected):
  ok = got == expected
  print(f"  [{'OK ' if ok else 'FAIL'}] {name}\n        got      {got}\n"
        f"        expected {expected}")
  return ok


def scenario_two_envs(fixed):
  om, clock = make_manager(2, 3)
  for t in (0.0, 1.0, 2.0):
    clock["t"] = t
    om.compute(update_history=True)
  before = hist(om)
  partial_reset(om, torch.tensor([0]), 2, fixed=fixed)
  after = hist(om)
  print(f"  before reset: {before}")
  print(f"  after reset:  {after}")
  return check("environment 1's history is unchanged", after[1], before[1])


def scenario_repeated(fixed):
  om, clock = make_manager(2, 3)
  for t in (0.0, 1.0, 2.0):
    clock["t"] = t
    om.compute(update_history=True)
  before1 = hist(om)[1]
  for _ in range(5):
    partial_reset(om, torch.tensor([0]), 2, fixed=fixed)
  return check("5 resets in a row: environment 1 intact", hist(om)[1], before1)


def scenario_staggered(fixed):
  """Three environments, terminating at different times."""
  om, clock = make_manager(3, 3)
  for t in (0.0, 1.0, 2.0):
    clock["t"] = t
    om.compute(update_history=True)
  partial_reset(om, torch.tensor([0]), 3, fixed=fixed)
  snap2 = hist(om)[2]
  clock["t"] = 3.0
  om.compute(update_history=True)          # a real step
  expect2 = hist(om)[2]
  partial_reset(om, torch.tensor([1]), 3, fixed=fixed)
  ok = check("environment 2 survived two different resets", hist(om)[2], expect2)
  del snap2
  return ok


def scenario_reset_env_backfilled():
  """A reset environment must get a fresh frame throughout its history."""
  om, clock = make_manager(2, 3)
  for t in (0.0, 1.0, 2.0):
    clock["t"] = t
    om.compute(update_history=True)
  partial_reset(om, torch.tensor([0]), 2, fixed=True)
  return check("reset environment 0 is backfilled with the current frame",
               hist(om)[0], [2.0, 2.0, 2.0])


def scenario_history_one():
  om, clock = make_manager(2, 1)
  for t in (0.0, 1.0):
    clock["t"] = t
    om.compute(update_history=True)
  before = hist(om)[1]
  partial_reset(om, torch.tensor([0]), 2, fixed=True)
  return check("history_length=1: environment 1 intact", hist(om)[1], before)


def scenario_with_delay():
  om, clock = make_manager(2, 3, delay=2)
  for t in (0.0, 1.0, 2.0, 3.0):
    clock["t"] = t
    om.compute(update_history=True)
  before = hist(om)[1]
  partial_reset(om, torch.tensor([0]), 2, fixed=True)
  return check("with delay buffer: environment 1 intact", hist(om)[1], before)


def scenario_all_envs_done():
  om, clock = make_manager(2, 3)
  for t in (0.0, 1.0, 2.0):
    clock["t"] = t
    om.compute(update_history=True)
  partial_reset(om, torch.tensor([0, 1]), 2, fixed=True)
  return check("reset of all environments: both backfilled",
               hist(om), [[2.0, 2.0, 2.0], [102.0, 102.0, 102.0]])


def main():
  print("=== BEFORE fix (stock base.reset(env_ids=...)) ===")
  broken = []
  print(" scenario: two environments, only the first terminated")
  broken.append(scenario_two_envs(fixed=False))
  broken.append(scenario_repeated(fixed=False))
  broken.append(scenario_staggered(fixed=False))

  print()
  print("=== AFTER fix (snapshot_obs_history/restore_obs_history) ===")
  ok = []
  print(" scenario: two environments, only the first terminated")
  ok.append(scenario_two_envs(fixed=True))
  ok.append(scenario_repeated(fixed=True))
  ok.append(scenario_staggered(fixed=True))
  ok.append(scenario_reset_env_backfilled())
  ok.append(scenario_history_one())
  ok.append(scenario_with_delay())
  ok.append(scenario_all_envs_done())

  print()
  print(f"before fix: passed {sum(broken)}/{len(broken)} (expected to fail)")
  print(f"after fix: passed {sum(ok)}/{len(ok)}")
  return 0 if all(ok) and not any(broken) else 1


if __name__ == "__main__":
  raise SystemExit(main())
