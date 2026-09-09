"""Diagnostic overrides are local and produce measured, not assumed, outcomes."""

import numpy as np
import pytest

from scripts import diagnose_d1_heightfield_delay as diagnostic
from scripts import probe_d1_delay_parity as probe


@pytest.mark.parametrize("variant", ("leg_pd_quarter", "attitude_feedback_quarter"))
def test_feedback_override_completes_delayed_flat_stand_without_changing_defaults(variant):
    original_build = probe.build_case
    rows, result = diagnostic.run_variant(variant)
    assert probe.build_case is original_build
    assert result["completed"] and result["executed_steps"] == 400
    assert result["max_joint_velocity_rated_fraction"] < 1.0
    assert all(row["command_was_applied"] for row in rows)
    _, loop = probe.build_case(heightfield=True)
    controller = loop.controller.controller
    np.testing.assert_array_equal(controller.leg_kp, np.tile((80, 80, 80, 0), 4))
    assert controller.roll_pitch_kp == 180 and controller.roll_pitch_kd == 24


def test_diagnostic_rejects_unknown_variant():
    with pytest.raises(ValueError, match="unknown"):
        diagnostic.run_variant("make_everything_better")
