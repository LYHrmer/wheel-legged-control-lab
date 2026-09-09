"""The delay parity probe must cross the real loop and record what it claims."""

import csv
import hashlib
import json
import sys
import tarfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
from gymnasium.utils import seeding

from wheel_legged_control.d1.actuator_channel import ActuatorChannelConfig
from wheel_legged_control.d1.control_loop import D1ControlLoop, D1MotionCommand
from wheel_legged_control.d1.locomotion_env import (
    LOCOMOTION_SAFE_HALF_SIZE_M,
    LOCOMOTION_SPAWN_POSITION_M,
)
from wheel_legged_control.d1.locomotion_terrain import D1LocomotionTerrainConfig
from wheel_legged_control.d1.model import JOINT_TORQUE_LIMIT
from wheel_legged_control.d1.wheel_leg_controller import D1WheelLegControlConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import probe_d1_delay_parity as probe

REQUIRED_COLUMNS = (
    "time_s",
    "x_m",
    "y_m",
    "z_m",
    "roll_rad",
    "pitch_rad",
    "yaw_rad",
    "joint_velocity_rated_fraction",
    "truth_wheel_ground_contacts",
    "world_command_roll_rad",
    "world_command_pitch_rad",
    "mechanical_activity_w",
    "termination_reason",
)


def test_short_rollout_crosses_the_real_loop_with_a_two_tick_measurement_age():
    rows, summary = probe.run_case("plane_bare", duration_s=0.2)

    assert summary["executed_steps"] == summary["expected_steps"] == len(rows) == 20
    assert summary["control_clock_consistent"] and summary["all_values_finite"]
    for index, row in enumerate(rows):
        assert row["tick_completed"] == 1
        assert row["control_time_s"] == pytest.approx(0.01 * (index + 1), abs=1e-12)
        assert row["time_s"] == pytest.approx(row["control_time_s"], abs=1e-12)
        assert row["measurement_age_s"] == pytest.approx(
            0.01 * min(index + 1, probe.SENSOR_DELAY_STEPS), abs=1e-12
        )
    # Genuine physics, not a replayed table: the plant carries real activity.
    assert rows[-1]["mechanical_activity_w"] > 0
    assert summary["max_joint_velocity_rated_fraction"] > 0
    assert rows[-1]["z_m"] != probe.SPAWN_POSITION_M[2]
    assert summary["min_truth_wheel_ground_contacts"] > 0


def test_recorded_world_command_attitude_equals_the_prepared_decision():
    plant, loop = probe.build_case(heightfield=False)
    loop.reset(seed=probe.MEASUREMENT_SEED, base_position=probe.SPAWN_POSITION_M)
    for step in range(3):
        ground, state = loop.provider.ground_reference(), loop.provider.read()
        decision = loop.prepare(D1MotionCommand())
        row = probe.observe(plant, step, ground, state)
        assert row["world_command_roll_rad"] == pytest.approx(decision.world_command.roll_rad)
        assert row["world_command_pitch_rad"] == pytest.approx(decision.world_command.pitch_rad)
        assert row["sensor_ground_height_m"] == pytest.approx(decision.ground.height_m)
        assert decision.world_command.base_height_m - row["sensor_ground_height_m"] == (
            pytest.approx(D1MotionCommand().clearance_m)
        )
        loop.step(np.zeros(8))


@pytest.mark.parametrize("noise", [True, False])
def test_both_noise_settings_run_and_report_their_own_configuration(noise):
    rows, summary = probe.run_case("plane_set_domain", duration_s=0.1, noise=noise)
    recorded = probe.protocol(["plane_set_domain"], 0.1, noise, probe.MEASUREMENT_SEED)
    standard = {
        "gyro_std_rad_s": 0.002,
        "accelerometer_std_m_s2": 0.03,
        "encoder_position_std_rad": 0.0005,
        "encoder_velocity_std_rad_s": 0.005,
    }

    assert summary["sensor_noise"] is noise and summary["completed"]
    assert recorded["sensor_noise_enabled"] is noise
    assert len(rows) == 10 and summary["all_values_finite"]
    for name, value in standard.items():
        assert recorded["provider"]["sensor_noise"][name] == (value if noise else 0.0)


def test_protocol_parameters_match_the_locomotion_task_setup():
    recorded = probe.protocol(list(probe.CASES), probe.DURATION_S, True, probe.MEASUREMENT_SEED)

    assert tuple(recorded["spawn_position_m"]) == LOCOMOTION_SPAWN_POSITION_M
    assert tuple(recorded["safe_half_size_m"]) == LOCOMOTION_SAFE_HALF_SIZE_M
    assert recorded["controller"] == asdict(D1WheelLegControlConfig(0.55, 1.5, 4.0))
    assert recorded["provider"]["sensor_delay_steps"] == 2
    assert recorded["provider"]["kind"] == "imu_encoder_fusion"
    assert recorded["terrain"] == asdict(D1LocomotionTerrainConfig())
    assert recorded["control_schema"] == D1ControlLoop.schema
    assert recorded["duration_s"] == 4.0 and recorded["action"] == "zeros(8) every tick"
    assert recorded["command"] == {"kind": "D1MotionCommand()", **asdict(D1MotionCommand())}
    # Nominal passthrough on the same 16 axes and torque limits as the task.
    nominal = asdict(ActuatorChannelConfig(torque_limit_nm=tuple(JOINT_TORQUE_LIMIT)))
    assert recorded["actuator_channel"] == nominal
    assert recorded["actuator_channel"]["delay_steps"] == 0
    assert recorded["actuator_channel"]["time_constant_s"] == 0.0
    assert recorded["actuator_channel"]["gain"] == 1.0
    assert recorded["actuator_channel"]["n_axes"] == 16
    assert recorded["actuator_channel"]["physics_dt_s"] == 0.002
    assert np.array_equal(recorded["actuator_channel"]["torque_limit_nm"], JOINT_TORQUE_LIMIT)
    # The measurement seed is the one the full env draws from episode seed 17.
    rng, _ = seeding.np_random(17)
    assert int(rng.integers(0, 2**31, 2)[1]) == recorded["measurement_seed"]
    assert probe.CASES == {
        "plane_bare": (False, False),
        "plane_set_domain": (False, True),
        "heightfield_bare": (True, False),
        "heightfield_set_domain": (True, True),
    }


def test_existing_output_directory_is_never_overwritten(tmp_path, monkeypatch):
    output = tmp_path / "results"
    output.mkdir()
    monkeypatch.setattr(
        sys,
        "argv",
        ["probe", "--output", str(output), "--cases", "plane_bare", "--duration", "0.05"],
    )

    with pytest.raises(SystemExit) as failure:
        probe.main()

    assert failure.value.code == 2
    assert list(output.iterdir()) == []


def test_cli_records_rows_summary_manifest_and_source_archive(tmp_path, monkeypatch):
    output = tmp_path / "results"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "probe",
            "--output",
            str(output),
            "--cases",
            "plane_bare",
            "heightfield_bare",
            "--duration",
            "0.05",
        ],
    )

    probe.main()

    summary = json.loads((output / "summary.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    recorded = json.loads((output / "protocol.json").read_text())
    assert [case["case"] for case in summary["cases"]] == ["plane_bare", "heightfield_bare"]
    for case in summary["cases"]:
        rows = list(csv.DictReader((output / f"{case['case']}.csv").open()))
        assert len(rows) == case["executed_steps"] == 5
        assert set(REQUIRED_COLUMNS) <= set(rows[0])
        assert {row["termination_reason"] for row in rows} == {""}
        assert case["completed"] and case["first_failure_time_s"] is None
        assert case["wall_clock_s"] > 0
    for name, digest in manifest.items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    assert set(manifest) == {
        "protocol.json",
        "source.tar.gz",
        "summary.json",
        "plane_bare.csv",
        "heightfield_bare.csv",
    }
    with tarfile.open(output / "source.tar.gz") as archive:
        assert set(archive.getnames()) == set(probe.SOURCE_FILES)
    for path, digest in recorded["source_sha256"].items():
        assert hashlib.sha256((probe.ROOT / path).read_bytes()).hexdigest() == digest


@pytest.mark.parametrize("case", list(probe.CASES))
def test_full_duration_case_reports_a_consistent_measured_outcome(case):
    rows, summary = probe.run_case(case)

    assert summary["all_values_finite"] and summary["control_clock_consistent"]
    assert summary["expected_steps"] == 400 and 0 < summary["executed_steps"] <= 400
    assert summary["measurement_age_values_s"] == [0.01, 0.02]
    assert len(rows) == summary["recorded_rows"]
    assert summary["executed_steps"] == sum(r["tick_completed"] for r in rows)
    # Geometry causality is not asserted: either outcome must stay self-consistent.
    if summary["completed"]:
        assert summary["executed_steps"] == 400
        assert summary["termination_reason"] == "duration_reached"
        assert summary["first_failure_time_s"] is None
    else:
        assert summary["termination_reason"] == rows[-1]["termination_reason"] != ""
        assert summary["first_failure_time_s"] == rows[-1]["time_s"] <= 4.0
        assert summary["first_failure_step"] == rows[-1]["step"] == len(rows)
        assert summary["executed_steps"] < 400
    assert summary["max_abs_truth_ground_height_m"] <= probe.FLAT_HEIGHT_TOLERANCE_M
    assert summary["max_truth_ground_slope_rad"] <= probe.FLAT_SLOPE_TOLERANCE_RAD


@pytest.mark.parametrize("duration", (0.0, -1.0, 0.015, float("nan"), float("inf")))
def test_invalid_or_fractional_duration_is_rejected(duration):
    with pytest.raises(ValueError, match="control ticks"):
        probe.run_case("plane_bare", duration_s=duration)


def test_poststep_rows_keep_the_actual_prestep_attitude_reference(monkeypatch):
    original_build = probe.build_case
    commands = []

    def build(**kwargs):
        plant, loop = original_build(**kwargs)
        prepare = loop.prepare

        def record(command):
            decision = prepare(command)
            commands.append(decision.world_command)
            return decision

        loop.prepare = record
        return plant, loop

    monkeypatch.setattr(probe, "build_case", build)
    rows, _ = probe.run_case("plane_bare", duration_s=0.03)
    for row, command in zip(rows, commands, strict=True):
        assert row["command_was_applied"] == 1
        assert row["world_command_roll_rad"] == command.roll_rad
        assert row["world_command_pitch_rad"] == command.pitch_rad
