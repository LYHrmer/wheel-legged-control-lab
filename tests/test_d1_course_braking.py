"""Course release behavior against an unchanged real MuJoCo legacy control path."""
import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.d1_course_controls import CourseKeyboardCommands
from scripts.d1_course_side_step import CourseSideStepDrive
from scripts.d1_fast_side_step import FastSideStepController
from wheel_legged_control.d1.control_loop import D1MotionCommand
from wheel_legged_control.d1.interactive import D1InteractiveSimulation


class MemoryController:
    def __init__(self):
        self._distance_m = 1.0
        self._distance_reference_m = 1.3
        self._prepared_control = None
        self.untouched = object()

    @property
    def control_memory(self):
        return SimpleNamespace(distance_m=self._distance_m,
                               distance_reference_m=self._distance_reference_m)


def test_actual_command_edge_is_consumed_once_and_preserves_other_memory():
    from scripts.d1_course_braking import CourseBrakeReference

    brake, controller = CourseBrakeReference(), MemoryController()
    assert not brake.prepare(controller, 0.0, eligible=True)["brake_reference_reanchored"]
    brake.observe_applied(0.0, eligible=True)
    brake.prepare(controller, .4, eligible=True)
    brake.observe_applied(0.0, eligible=True)  # A guard rejected the requested motion.
    assert not brake.prepare(controller, 0.0, eligible=True)["brake_reference_reanchored"]
    brake.observe_applied(.4, eligible=True)
    before = vars(controller).copy()
    info = brake.prepare(controller, 0.0, eligible=True)
    assert info["brake_reference_reanchored"]
    assert info["brake_reference_previous_error_m"] == pytest.approx(.3)
    assert info["brake_reference_new_error_m"] == 0.0
    assert info["brake_reference_event_count"] == 1
    assert controller._distance_reference_m == 1.0
    for key, value in before.items():
        if key != "_distance_reference_m":
            assert getattr(controller, key) == value
    assert not brake.prepare(controller, 0.0, eligible=True)["brake_reference_reanchored"]
    brake.observe_applied(0.0, eligible=True)
    assert not brake.prepare(controller, 0.0, eligible=True)["brake_reference_reanchored"]


@pytest.mark.parametrize("operation", ["prepare", "observe", "reset"])
def test_ownership_gap_and_reset_discard_pending_edge(operation):
    from scripts.d1_course_braking import CourseBrakeReference

    brake, controller = CourseBrakeReference(), MemoryController()
    brake.observe_applied(-.2, eligible=True)
    if operation == "prepare":
        brake.prepare(controller, 0.0, eligible=False)
    elif operation == "observe":
        brake.observe_applied(0.0, eligible=False)
    else:
        brake.reset()
    assert not brake.prepare(controller, 0.0, eligible=True)["brake_reference_reanchored"]
    assert controller._distance_reference_m == 1.3


@pytest.mark.parametrize("value,error", [(True, TypeError), ("0", TypeError),
                                        (np.nan, ValueError), (np.inf, ValueError)])
def test_invalid_command_cannot_change_memory(value, error):
    from scripts.d1_course_braking import CourseBrakeReference

    brake, controller = CourseBrakeReference(), MemoryController()
    brake.observe_applied(.2, eligible=True)
    with pytest.raises(error):
        brake.prepare(controller, value, eligible=True)
    assert controller._distance_reference_m == 1.3
    assert brake.prepare(controller, 0.0, eligible=True)["brake_reference_reanchored"]


def test_prepared_proposal_is_rejected_without_discarding_it():
    from scripts.d1_course_braking import CourseBrakeReference

    brake, controller = CourseBrakeReference(), MemoryController()
    proposal = controller._prepared_control = object()
    brake.observe_applied(.2, eligible=True)
    with pytest.raises(RuntimeError):
        brake.prepare(controller, 0.0, eligible=True)
    assert controller._prepared_control is proposal
    assert controller._distance_reference_m == 1.3
    controller._prepared_control = None
    assert brake.prepare(controller, 0.0, eligible=True)["brake_reference_reanchored"]


def test_invalid_reference_does_not_consume_a_valid_release_edge():
    from scripts.d1_course_braking import CourseBrakeReference

    brake, controller = CourseBrakeReference(), MemoryController()
    brake.observe_applied(.2, eligible=True)
    controller._distance_reference_m = np.inf
    with pytest.raises(ValueError):
        brake.prepare(controller, 0.0, eligible=True)
    assert np.isinf(controller._distance_reference_m)
    controller._distance_reference_m = 1.3
    assert brake.prepare(controller, 0.0, eligible=True)["brake_reference_reanchored"]


@pytest.mark.parametrize("moving_key,release_keys,focused", [
    ("W", set(), True), ("S", set(), True), ("W", {ord("X")}, True),
    ("W", set(), False),
])
def test_release_cancel_and_focus_loss_share_one_real_braking_edge(moving_key,
                                                                  release_keys, focused):
    simulation = D1InteractiveSimulation(arena="flat")
    drive = CourseSideStepDrive(simulation, brake_reference_mode="release_reanchor_experimental")
    commands = CourseKeyboardCommands(clock=lambda: float(simulation.plant.data.time))
    commands.update_pressed({ord(moving_key)})
    drive.step(commands(0.0), 0)
    commands.update_pressed(release_keys, focused=focused)
    _, _, info = drive.step(commands(float(simulation.plant.data.time)), 0)
    assert info["brake_reference_reanchored"]
    assert info["brake_reference_event_count"] == 1
    _, _, info = drive.step(D1MotionCommand(), 0)
    assert not info["brake_reference_reanchored"]
    assert info["brake_reference_event_count"] == 1
    simulation.reset("start")
    drive.reset()
    _, _, info = drive.step(D1MotionCommand(), 0)
    assert not info["brake_reference_reanchored"]
    assert info["brake_reference_event_count"] == 0


def _physical_release(arena, zone, gear, release, *, adapter, brake_reference_mode="legacy"):
    simulation = D1InteractiveSimulation(arena=arena, baseline="lqr", state_mode="oracle")
    simulation.reset(zone)
    plant, teleop = simulation.plant, simulation.teleop
    commands = CourseKeyboardCommands(clock=lambda: float(plant.data.time))
    commands.reset(float(teleop._state.base_rpy[2]), teleop._state.base_position[:2])
    teleop.controller.low_level.wheel_velocity_gain = commands.wheel_velocity_gain
    drive = CourseSideStepDrive(simulation, side_controller_factory=FastSideStepController,
                               brake_reference_mode=brake_reference_mode)
    states, torques, events = [plant.data.qpos.copy()], [], []
    for tick in range(round((release+4)/.01)):
        state = teleop._state
        commands.set_state(yaw_rad=float(state.base_rpy[2]), position_xy=state.base_position[:2],
                           forward_velocity_mps=float(state.base_linear_velocity_body[0]),
                           yaw_rate_rps=float(state.base_angular_velocity_body[2]),
                           fallen=bool(state.has_fallen()), simulation_time_s=float(plant.data.time))
        keys = set()
        if gear >= 2 and tick == 100:
            keys.add(340)
        if gear >= 3 and tick == 102:
            keys.add(344)
        if 200 <= tick < round(release/.01):
            keys.add(ord("W"))
        commands.update_pressed(keys)
        requested = commands(float(plant.data.time))
        if adapter:
            _, torque, info = drive.step(requested, 0)
            if info.get("brake_reference_reanchored"):
                events.append((tick, info))
        else:
            teleop.forward_velocity_mps = requested.forward_velocity_mps
            teleop.yaw_rate_rps = requested.yaw_rate_rps
            teleop.base_height_m = requested.clearance_m
            torque, _ = teleop.compute()
            plant.step(torque)
            teleop.update_state()
        states.append(plant.data.qpos.copy())
        torques.append(torque.copy())
        assert not plant.has_fallen()
        assert np.all(np.abs(torque) <= plant.actuator_torque_limit_nm)
    return np.asarray(states), np.asarray(torques), events


@pytest.mark.parametrize("arena,zone,gear,release,ratio", [
    ("flat", "start", 3, 10., .65), ("course", "ramp", 2, 14., .60),
])
def test_release_reduces_real_drift_without_changing_pre_release_physics(arena, zone, gear,
                                                                      release, ratio):
    original, original_torque, _ = _physical_release(arena, zone, gear, release, adapter=False)
    default, default_torque, default_events = _physical_release(arena, zone, gear, release, adapter=True)
    np.testing.assert_array_equal(default, original)
    np.testing.assert_array_equal(default_torque, original_torque)
    assert default_events == []
    candidate, candidate_torque, events = _physical_release(
        arena, zone, gear, release, adapter=True, brake_reference_mode="release_reanchor_experimental"
    )
    cut = round(release/.01)
    np.testing.assert_array_equal(original[:cut+1], candidate[:cut+1])
    np.testing.assert_array_equal(original_torque[:cut], candidate_torque[:cut])
    original_drift = original[-1, 0]-original[cut, 0]
    candidate_drift = candidate[-1, 0]-candidate[cut, 0]
    assert abs(candidate_drift) < ratio*abs(original_drift)
    assert np.mean(np.abs(np.diff(candidate[-101:, 0])/.01)) < .05
    assert len(events) == 1 and events[0][0] == cut
    assert events[0][1]["brake_reference_event_count"] == 1


def test_pending_jump_and_side_request_do_not_create_release_edges():
    simulation = D1InteractiveSimulation(arena="course")
    for _ in range(200):
        simulation.step()
    drive = CourseSideStepDrive(simulation, brake_reference_mode="release_reanchor_experimental")
    drive.step(D1MotionCommand(.1), 0)
    _, _, info = drive.step(D1MotionCommand(), 1)
    assert not info["brake_reference_reanchored"]
    assert info["brake_reference_event_count"] == 0
    # Explicit reset is the supported way to end this independent side scenario.
    simulation.reset("start")
    drive.reset()
    drive.step(D1MotionCommand(.1), 0)
    simulation.teleop._jump_requested = True
    _, _, info = drive.step(D1MotionCommand(), 0)
    assert not info["brake_reference_reanchored"]
    assert info["brake_reference_event_count"] == 0


@pytest.mark.parametrize("mode,profile", [("legacy", "legacy_retained_reference"),
    ("release_reanchor_experimental", "legacy_release_reference_edge_v1")])
def test_cli_defaults_and_explicit_experiment_are_recorded_truthfully(tmp_path, mode, profile):
    from scripts.run_d1_course_drive import main

    output = tmp_path / mode
    arguments = ["--output", str(output), "--seconds", ".01", "--headless"]
    if mode != "legacy":
        arguments += ["--brake-reference-mode", mode]
    assert main(arguments) == 0
    protocol = json.loads((output / "protocol.json").read_text())
    assert protocol["brake_reference_mode"] == mode
    assert protocol["braking_profile"] == profile
    assert protocol["schema"] == "d1-course-side-stepping-v4"
    assert "scripts/d1_course_braking.py" in protocol["source_sha256"]
    with (output / "telemetry.csv").open() as stream:
        row = next(csv.DictReader(stream))
    assert row["brake_reference_mode"] == mode and row["braking_profile"] == profile
    assert row["brake_reference_event_count"] == "0"
