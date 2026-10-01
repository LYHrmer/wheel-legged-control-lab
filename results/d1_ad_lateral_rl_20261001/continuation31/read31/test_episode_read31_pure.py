"""Small saved-reader boundary/schema examples, no model or dynamics imports."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

W = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(W/'continuation26/root26'))
import reader26  # noqa: F401 -- establishes the cold forbidden-import guard
for path in (W/'continuation24', W/'continuation27/read27', W/'continuation30/read30'):
    sys.path.insert(0, str(path))
from episode_read31 import translate_boundary31, canonical_check, COMPONENTS


def test_nonzero_global_offset_keeps_original_compiler_CCD_and_history():
    boundary = dict(C_state=dict(control_attempts=15875, control_returns=15875,
        construction_attempts=2, construction_returns=2, ccd_attempts=94567, ccd_returns=94567),
        python=dict(control_attempted=3175, control_completed=3175,
            native_attempted=15875, native_returned=15875, clock_advanced_substeps=15875,
            segments=[dict(name='actual_previous', completed=1572)]))
    original = copy.deepcopy(boundary)
    local = translate_boundary31(boundary, 1572, 3175)
    assert local['python']['control_completed'] == 1603
    assert local['python']['native_returned'] == 8015
    assert local['C_state']['control_returns'] == 8015
    assert local['C_state']['construction_returns'] == 2
    assert local['C_state']['ccd_returns'] == 94567
    assert local['python']['segments'] == original['python']['segments']
    assert local['python']['clock_advanced_substeps'] == 15875
    assert boundary == original
    boundary['python']['native_returned'] -= 1
    with pytest.raises(ValueError, match='global ledger'):
        translate_boundary31(boundary, 1572, 3175)


def test_canonical_alias_cannot_replace_physical_cost_or_duration():
    physical = {key: float(index)/10 for index, key in enumerate(COMPONENTS.values())}
    physical['elapsed_s'] = .37
    saved = dict(physical, control_range=[200, 237], elapsed_controls=37,
        reward_weights_defined=False, weighted_scalar_reward=None,
        torque_cost_is_energy=False, source_component_schema='d1-c30-macros-v1')
    canonical_check(saved, physical, [200, 237])
    wrong = copy.deepcopy(saved)
    wrong['torque_normalized_square_integral_s'] += .05
    with pytest.raises(ValueError, match='canonical component'):
        canonical_check(wrong, physical, [200, 237])
    wrong = copy.deepcopy(saved)
    wrong['elapsed_controls'] = 36
    with pytest.raises(ValueError, match='canonical component'):
        canonical_check(wrong, physical, [200, 237])
