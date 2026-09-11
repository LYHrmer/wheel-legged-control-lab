import importlib.util
import math
from pathlib import Path

import pytest

script = Path(__file__).resolve().parents[1] / "scripts/delay_metrics.py"
if not script.is_file():
    script = Path(__file__).resolve().with_name("delay_metrics.py")
spec = importlib.util.spec_from_file_location("delay_metrics_under_test", script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
delay_steps, summarize = module.delay_steps, module.summarize


def row(time_s):
    return {"time_s": time_s, "velocity_error_mps": 2.0, "yaw_rate_error_rps": 3.0,
                "height_error_m": 4.0, "roll_error_rad": 3.0, "pitch_error_rad": 4.0,
                "measurement_age_s": 0.03, "torque_input_clipped_fraction": 0.25}


def test_delay_units_are_physical_time():
    assert [delay_steps(ms, "measurement") for ms in (0, 10, 20, 30)] == [0, 1, 2, 3]
    assert [delay_steps(ms, "actuator") for ms in (0, 10, 20, 30)] == [0, 5, 10, 15]


@pytest.mark.parametrize("value,channel", [(True, "actuator"), (-1, "actuator"), (3, "actuator"), (2, "measurement"), (10.0, "measurement"), (10, "action")])
def test_reject_invalid_delay(value, channel):
    with pytest.raises((TypeError, ValueError)):
        delay_steps(value, channel)


@pytest.mark.parametrize("value", [True, 10.0, "10"])
def test_delay_wrong_type_raises_type_error(value):
    with pytest.raises(TypeError):
        delay_steps(value, "measurement")


@pytest.mark.parametrize("case", ["duration", "metric", "reason", "row"])
def test_summary_wrong_type_raises_type_error(case):
    rows, duration, reason = [row(0.01)], 1.0, "time_limit"
    if case == "duration":
        duration = "1"
    elif case == "metric":
        rows[0]["velocity_error_mps"] = "2"
    elif case == "reason":
        reason = None
    else:
        rows = [None]
    with pytest.raises(TypeError):
        summarize(rows, duration, reason)


def test_failed_and_short_time_limit_cannot_report_full_horizon():
    for reason in ("fall_or_body_contact", "time_limit"):
        result = summarize([row(0.01), row(0.02)], 60.0, reason)
        assert result["full_horizon_tracking"] is None
        assert not result["completed"]


def test_completed_metrics_use_executed_rows():
    result = summarize([row(0.01), row(0.02)], 0.02, "time_limit")
    assert result["completed"]
    assert result["full_horizon_tracking"]["attitude_rmse_rad"] == 5.0
    assert result["full_horizon_tracking"]["velocity_rmse_mps"] == 2.0


@pytest.mark.parametrize("rows,duration", [([], 1), ([row(0.02), row(0.01)], 1), ([row(0.01)], 0), ([row(math.nan)], 1), ([row(2)], 1)])
def test_invalid_trajectory_is_rejected(rows, duration):
    with pytest.raises(ValueError):
        summarize(rows, duration, "time_limit")
