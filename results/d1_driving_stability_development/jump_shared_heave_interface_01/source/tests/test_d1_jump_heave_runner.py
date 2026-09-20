"""New seed consumption and physical-prefix evidence; no integration permitted."""

import copy
import json

import mujoco
import numpy as np
import pytest

from scripts.d1_jump_heave_env import D1JumpHeaveEnv
from scripts.evaluate_d1_jump_heave import compare_physical_prefix
from scripts.run_d1_jump_heave_study import CurriculumLedger, NativeCounter


@pytest.mark.parametrize("mode,base", [("smoke",629980000),("formal",620020000)])
def test_new_ledger_passes_actual_new_episode_seeds(mode, base, monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail("seed qualification must not integrate")
    for name in ("mj_step","mj_step1","mj_step2"):
        monkeypatch.setattr(mujoco,name,forbidden)
    env = D1JumpHeaveEnv()
    observed = []
    original = env._inner.reset
    def reset_spy(*, seed=None, options=None):
        observed.append(seed)
        return original(seed=seed, options=options)
    monkeypatch.setattr(env._inner,"reset",reset_spy)
    ledger = CurriculumLedger(env, mode, tmp_path, NativeCounter(env.plant,3200))
    try:
        ledger.reset(seed=999)
        ledger.total_transitions = 600
        ledger.episode_finished = True  # simulated ledger boundary, no physical transitions
        env._inner._active = False  # corresponding completed inner episode boundary
        ledger.reset(seed=888)
        assert observed == [base,base+1]
        boundary = json.loads((tmp_path/"boundaries/episode_00001_reset.json").read_text())
        assert boundary["actual_seed"] == base+1 and boundary["incoming_seed"] == 888
        assert boundary["metadata"]["policy_action_size"] == 1
        assert env.plant.data.time == 0.
        assert env.jump_episode_spec.request_tick == (None if mode == "smoke" else 200)
    finally:
        ledger.close()


def pair_fixture():
    rows = [{"action":[0.], "reward":1., "info": {
        "policy_action":[0.], "applied_action":[0.]*8,
        "heave_action":{"active":False,"applied_physical_action_8":[0.]*8}},
        "controller":{"torque_nm":[0.]*16}} for _ in range(3)]
    return {"arrays":{"qpos":np.zeros((4,23)),"observation":np.zeros((4,95),dtype=np.float32)},
            "trace":rows,"native":[{"ctrl_nm":[0.]*16} for _ in range(15)]}


def test_prefix_includes_request_observation_but_allows_unused_policy_action():
    zero = pair_fixture()
    policy = copy.deepcopy(zero)
    for row in policy["trace"]:
        row["action"]=[.9]
        row["info"]["policy_action"]=[.9]
    assert compare_physical_prefix(zero,policy,{"request_tick":2})["passed"]
    policy["arrays"]["observation"][2,90]=.5
    assert not compare_physical_prefix(zero,policy,{"request_tick":2})["passed"]


def test_prefix_checks_physical_reward_and_native_inputs():
    zero = pair_fixture()
    policy = copy.deepcopy(zero)
    policy["trace"][1]["reward"]=0.
    assert not compare_physical_prefix(zero,policy,{"request_tick":2})["passed"]
    policy = copy.deepcopy(zero)
    policy["native"][9]["ctrl_nm"][0]=1.
    assert not compare_physical_prefix(zero,policy,{"request_tick":2})["passed"]
