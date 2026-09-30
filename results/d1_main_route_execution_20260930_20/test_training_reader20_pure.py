"""Saved-data reader gate fixtures; no model, engine or simulation imports."""
import copy
import numpy as np
import pytest

from read_training20 import _coverage, _check_numeric


def _episode(source, *, active=220, load=30, endpoint=None):
    slot = (source-8) % 8
    terrain = {3:"bumps",4:"rough",5:"ramp",7:"ramp"}.get(slot,"flat")
    return {"source_episode_index":source,"terrain":terrain,
            "completed_controls":1000,"effective_ticks":active,
            "positive_active_intervals":load if terrain in ("bumps","rough","ramp") else 0,
            "endpoint_sign_ticks":endpoint or {}}


def test_coverage_requires_actual_effective_and_positive_load():
    episodes = [_episode(8+8*cycle+slot,
                         endpoint={"positive":110,"negative":110}
                         if slot == 2 and cycle in (0,2) else None)
                for cycle in range(3) for slot in range(8)]
    assert _coverage(episodes,"B",1)["coverage_valid"] is True
    no_bump_force = copy.deepcopy(episodes)
    no_bump_force[3]["positive_active_intervals"] = 0
    assert _coverage(no_bump_force,"B",1)["coverage_valid"] is False
    no_permission = copy.deepcopy(episodes)
    no_permission[0]["effective_ticks"] = 0
    assert _coverage(no_permission,"B",1)["coverage_valid"] is False


def test_stage2_reports_observed_coverage_without_requiring_old_cycle_ids():
    episodes = [_episode(40+8*cycle+slot) for cycle in range(3) for slot in range(8)]
    result = _coverage(episodes,"B",2)
    assert result["applicability"] == "stage2_observed_exposure_only"
    assert result["coverage_valid"] is True


def test_gaussian_and_numeric_must_match_actual_full_record():
    raw = np.zeros((1,16),dtype=np.float64)
    raw[0,0] = 1.3
    clipped = np.clip(raw,-1.,1.)
    observation = np.zeros((1,99),dtype=np.float32)
    row = {"control_index":0,"episode_index":0,"episode_tick":0,
           "input_observation99":observation[0].tolist(),
           "policy_input_action":clipped[0].tolist(),"reward":1.,
           "terminated":False,"truncated":False,
           "info":{"policy_clipped_action":clipped[0].tolist(),
                   "applied_action":clipped[0].tolist(),
                   "metrics":{"body_com_vx_mps":.2,"body_yaw_rate_rps":.01}}}
    numeric = {"control_index":np.array([0]),"episode_index":np.array([0]),
               "episode_tick":np.array([0]),"input_observation99":observation,
               "policy_env_input16":clipped,"policy_clipped16":clipped,
               "effective_action16":clipped,"reward":np.array([1.]),
               "terminated":np.array([False]),"truncated":np.array([False]),
               "body_com_vx_mps":np.array([.2]),
               "body_yaw_rate_rps":np.array([.01])}
    gaussian = {"control_index":np.array([0]),"episode_index":np.array([0]),
                "episode_tick":np.array([0]),"raw_gaussian_action16":raw,
                "clipped_action16":clipped,"effective_action16":clipped}
    _check_numeric(row,numeric,gaussian,0)
    tampered = copy.deepcopy(gaussian)
    tampered["raw_gaussian_action16"][0,0] = -.2
    with pytest.raises(ValueError):
        _check_numeric(row,numeric,tampered,0)
