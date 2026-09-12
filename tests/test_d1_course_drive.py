import csv
import hashlib
import json
from contextlib import nullcontext
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from scripts.d1_keyboard_commands import HeldKeyboardCommands
from scripts.run_d1_course_drive import apply_course_events, main, run
from wheel_legged_control.d1.interactive import D1InteractiveSimulation
from wheel_legged_control.d1.terrain import D1_COURSE_SPAWNS


def key(char, action=1):
    return {"type": "key", "key": ord(char), "action": action}


def test_event_core_handles_fifo_reset_and_space_jump_and_ignores_repeat():
    calls = []
    simulation = SimpleNamespace(
        reset=lambda zone: calls.append(("reset", zone)),
        teleop=SimpleNamespace(handle_key=lambda value: calls.append(("jump", value))),
    )
    viewer = SimpleNamespace(
        events=[key("3"), key("J"), key(" "), key("J", 2), key("4"), key("J", 0)],
        commands=SimpleNamespace(update_pressed=lambda value: calls.append(("held", value))),
        reset_camera=lambda: calls.append(("camera",)),
    )

    def reset(zone, index):
        calls.append(("segment", zone, index))
        if zone == "ramp":
            viewer.events.append(key("6"))

    cursor = apply_course_events(simulation, viewer, 0, reset)
    assert cursor == 6
    assert calls == [
        ("reset", "ramp"),
        ("held", set()),
        ("camera",),
        ("segment", "ramp", 0),
        ("jump", 32),
        ("jump", 32),
        ("reset", "stairs"),
        ("held", set()),
        ("camera",),
        ("segment", "stairs", 4),
    ]
    cursor = apply_course_events(simulation, viewer, cursor, reset)
    assert calls[-1] == ("segment", "jump", 6)
    before = calls.copy()
    assert apply_course_events(simulation, viewer, cursor, reset) == cursor
    assert calls == before
    for invalid in (True, -1, 20, 1.0):
        with pytest.raises((TypeError, ValueError)):
            apply_course_events(simulation, viewer, invalid, reset)
    assert calls == before


class FakeViewer:
    def __init__(self, model, data, commands, *, terrain_label):
        self.model, self.data, self.commands = model, data, commands
        self.events, self.frame = [], 0
        self.cam = SimpleNamespace(lookat=np.zeros(3))
        self.closed = False

    def poll(self, seconds):
        self.frame += 1
        if self.frame == 2:
            self.events.append(key("3"))
        elif self.frame == 3:
            self.events.extend([key("J"), key(" ")])
        elif self.frame == 4:
            self.events.append(key("4"))
        self.commands.update_pressed({ord("W")} if self.frame != 3 else {32})

    def reset_camera(self):
        pass

    def is_running(self):
        return True

    def lock(self):
        return nullcontext()

    def set_status(self, command, seconds):
        self.last_command = command

    def sync(self):
        # Exercise the renderer's derived-state refresh only on its own copy.
        mujoco.mj_forward(self.model, self.data)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.closed = True


def test_real_microsteps_record_integrated_states_and_discontinuous_resets(tmp_path):
    simulation = D1InteractiveSimulation()
    counter = [0.0]

    def clock():
        counter[0] += 0.01
        return counter[0]

    output = tmp_path / "drive"
    result = run(
        simulation,
        output,
        seconds=0.05,
        commands=HeldKeyboardCommands(clock=clock),
        viewer_factory=FakeViewer,
        realtime=False,
    )
    assert result["steps"] == 5 and result["segments"] == 3
    assert result["source_unchanged"]
    segments = json.loads((output / "segments.json").read_text())
    assert [s["zone"] for s in segments] == ["start", "ramp", "stairs"]
    rows = list(csv.DictReader((output / "telemetry.csv").open()))
    assert rows[2]["requested_forward_mps"] == "0.0"  # Space clears driving.
    assert {row["rl_mode"] for row in rows} == {"unavailable"}
    with np.load(output / "states.npz", allow_pickle=False) as state:
        assert len(state["qpos"]) == result["steps"] + result["segments"]
        np.testing.assert_array_equal(state["qpos"][-1], simulation.plant.data.qpos)
        np.testing.assert_array_equal(state["qvel"][-1], simulation.plant.data.qvel)
        for segment in segments:
            index = segment["first_state_index"]
            assert state["segment_time_s"][index] == 0.0
            np.testing.assert_allclose(
                state["qpos"][index, :3], D1_COURSE_SPAWNS[segment["zone"]].position_m
            )
        for row in rows:
            before, after = int(row["state_before_index"]), int(row["state_after_index"])
            assert (
                state["segment_id"][before] == state["segment_id"][after] == int(row["segment_id"])
            )
            assert state["segment_time_s"][after] - state["segment_time_s"][
                before
            ] == pytest.approx(0.01)
    manifest = json.loads((output / "manifest.json").read_text())
    for name, digest in manifest.items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    protocol = json.loads((output / "protocol.json").read_text())
    assert protocol["sampling_mode"] == "legacy_mixed"
    assert protocol["state_mode"] == "oracle" and protocol["policy"] == "none"
    assert protocol["terrain_geom_count"] > 80
    assert "scripts/run_d1_course_drive.py" in protocol["source_sha256"]


def test_headless_runner_leaves_legacy_physics_identical_to_direct_steps(tmp_path):
    actual, reference = D1InteractiveSimulation(), D1InteractiveSimulation()
    reference.teleop.controller.low_level.wheel_velocity_gain = 0.55
    reference.reset("rough")
    for _ in range(3):
        reference.step()
    run(
        actual,
        tmp_path / "headless",
        seconds=0.03,
        zone="rough",
        viewer_factory=None,
        realtime=False,
    )
    np.testing.assert_array_equal(actual.plant.data.qpos, reference.plant.data.qpos)
    np.testing.assert_array_equal(actual.plant.data.qvel, reference.plant.data.qvel)


def test_space_press_reaches_existing_guard_after_settle():
    simulation = D1InteractiveSimulation()
    for _ in range(100):
        simulation.step()
    viewer = SimpleNamespace(events=[key(" ")])
    cursor = apply_course_events(simulation, viewer, 0, lambda *_: None)
    assert simulation.step().jump_phase == "crouch"
    viewer.events.append(key("J"))
    apply_course_events(simulation, viewer, cursor, lambda *_: None)
    assert simulation.step().jump_phase == "crouch"


def test_existing_output_rejected_before_reset_and_cli_has_no_policy_option(tmp_path):
    simulation = D1InteractiveSimulation()
    simulation.step()
    before = simulation.plant.data.qpos.copy()
    with pytest.raises(FileExistsError):
        run(simulation, tmp_path, zone="stairs", viewer_factory=None)
    np.testing.assert_array_equal(simulation.plant.data.qpos, before)
    with pytest.raises(SystemExit):
        main(["--output", str(tmp_path / "unused"), "--policy", "checkpoint.zip"])
