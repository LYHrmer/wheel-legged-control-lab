"""Small saved-data C22 geometry-bridge fixtures; no full finite replay."""
from copy import deepcopy

from geometry21 import compare_reset_to_template21
from geometry_runtime22 import load_sealed_templates22
from recipes22 import SPEC, make_schedule


def test_only_four_preregistered_yaw_segments_changed():
    expected = {
        'dev_yaw_left':((430,530,.3),(530,730,-.3),(730,830,.3)),
        'dev_yaw_right':((430,530,-.3),(530,730,.3),(730,830,-.3)),
        'final_yaw_left':((440,540,.28),(540,740,-.28),(740,840,.28)),
        'final_yaw_right':((440,540,-.28),(540,740,.28),(740,840,-.28)),
    }
    for case,segments in expected.items():
        commands = make_schedule(case).raw_commands
        assert len(commands)==1600
        for start,end,yaw in segments:
            assert all(commands[i].yaw_rate_rps==yaw for i in range(start,end))
        active = {i for start,end,_ in segments for i in range(start,end)}
        assert all(commands[i].yaw_rate_rps==0. for i in range(1600) if i not in active)


def test_sealed_template_bridge_accepts_addresses_and_rejects_geom_tamper():
    templates,controls,inputs = load_sealed_templates22(SPEC)
    assert set(templates)==set(controls)=={'flat','bumps','rough','ramp'}
    assert len(inputs)==13
    manifest,initial = templates['flat']
    actual = deepcopy(manifest)
    actual['model_identity']['model_address'] += 123456
    actual['model_identity']['data_address'] += 123456
    same = compare_reset_to_template21(manifest,actual,initial,initial,
                                       controls['flat'],controls['flat'])
    assert same['passed'] is True
    actual['robot_collision_geoms'][0]['world_center_m'][0] += 1e-6
    changed = compare_reset_to_template21(manifest,actual,initial,initial,
                                          controls['flat'],controls['flat'])
    assert changed['passed'] is False


def test_sealed_template_bridge_rejects_initial_byte_change():
    templates,controls,_ = load_sealed_templates22(SPEC)
    manifest,initial = templates['flat']
    actual = {key:value.copy() for key,value in initial.items()}
    actual['qvel'][0] += 1e-6
    result = compare_reset_to_template21(manifest,manifest,initial,actual,
                                         controls['flat'],controls['flat'])
    assert result['passed'] is False
