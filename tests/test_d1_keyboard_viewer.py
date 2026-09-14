from types import SimpleNamespace

import pytest

from scripts.d1_course_controls import CourseKeyboardCommands
from scripts.d1_keyboard_viewer import KeyboardViewer


def make_viewer():
    states = {ord("W"): 1}
    focus = [True]
    samples = []
    viewer = KeyboardViewer.__new__(KeyboardViewer)
    viewer.glfw = SimpleNamespace(
        PRESS=1,
        KEY_SPACE=32,
        KEY_ESCAPE=256,
        KEY_C=ord("C"),
        FOCUSED=1,
        MOUSE_BUTTON_LEFT=0,
        get_mouse_button=lambda *_: 1,
        poll_events=lambda: None,
        get_window_attrib=lambda *_: focus[0],
        get_key=lambda _, key: states.get(key, 0),
    )
    viewer.commands = SimpleNamespace(
        update_pressed=lambda keys, focused=True: samples.append((keys, focused))
    )
    viewer.window = object()
    viewer.cam = SimpleNamespace(distance=4.5, azimuth=135.0, elevation=-28.0)
    viewer.events = []
    viewer._clock_start = 0.0
    viewer._simulation_time = 0.0
    viewer._focused = True
    viewer._mouse = None
    return viewer, states, focus, samples


def test_poll_reads_held_key_without_repeated_key_events_and_clears_focus():
    viewer, states, focus, samples = make_viewer()
    for tick in range(201):
        viewer.poll(tick * 0.01)
    assert samples == [({ord("W")}, True)] * 201
    # A stale key state in GLFW cannot keep driving an unfocused window.
    focus[0] = False
    viewer.poll(2.01)
    assert samples[-1] == (set(), False)
    focus[0] = True
    states.clear()
    viewer.poll(2.02)
    assert samples[-1] == (set(), True)


def test_driving_key_callback_does_not_run_native_render_or_camera_shortcuts():
    viewer, _, _, samples = make_viewer()
    before = vars(viewer.cam).copy()
    for key in "WASDRF":
        viewer._on_key(viewer.window, ord(key), 0, 1, 0)
        viewer._on_key(viewer.window, ord(key), 0, 0, 0)
    assert vars(viewer.cam) == before
    assert samples == []  # Driving samples come from poll, not event frequency.
    assert len(viewer.events) == 12
    viewer._on_key(viewer.window, 256, 0, 1, 0)
    assert samples[-1] == ({256}, True)


def test_mouse_cannot_put_follow_camera_below_ground_or_inside_body():
    viewer, _, _, _ = make_viewer()
    viewer._on_mouse(viewer.window, 0, 0)
    viewer._on_mouse(viewer.window, 100, -10000)
    assert viewer.cam.elevation == -8.0
    viewer._on_mouse(viewer.window, 100, 10000)
    assert viewer.cam.elevation == -80.0
    viewer._on_scroll(viewer.window, 0, 50)
    assert viewer.cam.distance == 1.5
    viewer._on_scroll(viewer.window, 0, -50)
    assert viewer.cam.distance == 15.0
    viewer._on_key(viewer.window, ord("C"), 0, 1, 0)
    assert viewer.cam.distance == pytest.approx(4.5)
    assert viewer.cam.elevation == pytest.approx(-28.0)


def test_focus_callback_stops_without_waiting_for_next_render():
    viewer, _, _, samples = make_viewer()
    viewer._on_focus(viewer.window, False)
    assert samples == [(set(), False)]
    assert viewer.events[-1]["type"] == "focus"
    assert viewer.events[-1]["focused"] is False


def test_shift_quick_taps_survive_one_poll_and_do_not_repeat_or_double_count():
    viewer, states, _, _ = make_viewer()
    viewer.commands = CourseKeyboardCommands(clock=lambda: 0.0)
    states.clear()
    # A complete tap can arrive during a slow rendered frame.
    for action in (1, 0):
        viewer._on_key(viewer.window, 340, 0, action, 0)
    viewer.poll(0.0)
    assert viewer.commands.gear == 2
    for _ in range(5):
        viewer.poll(0.0)
    assert viewer.commands.gear == 2
    # Several complete taps in one event batch retain their count.
    for _ in range(2):
        for action in (1, 0):
            viewer._on_key(viewer.window, 344, 0, action, 0)
    viewer.poll(0.0)
    assert viewer.commands.gear == 1
    states[340] = 1
    viewer._on_key(viewer.window, 340, 0, 1, 0)
    viewer.poll(0.0)
    for _ in range(5):
        viewer._on_key(viewer.window, 340, 0, 2, 0)
        viewer.poll(0.0)
    assert viewer.commands.gear == 2
    # Overlapping left/right Shift is one press until both are released.
    states[344] = 1
    viewer._on_key(viewer.window, 344, 0, 1, 0)
    viewer.poll(0.0)
    del states[340]
    viewer._on_key(viewer.window, 340, 0, 0, 0)
    viewer.poll(0.0)
    assert viewer.commands.gear == 2
    states.clear()
    viewer._on_key(viewer.window, 344, 0, 0, 0)
    viewer.poll(0.0)
    viewer._on_key(viewer.window, 344, 0, 1, 0)
    viewer.poll(0.0)
    assert viewer.commands.gear == 3


def test_shift_events_cancel_on_focus_loss_and_reset_does_not_shift_again():
    viewer, states, focus, _ = make_viewer()
    viewer.commands = CourseKeyboardCommands(clock=lambda: 0.0)
    states.clear()
    viewer._on_key(viewer.window, 340, 0, 1, 0)
    viewer._on_focus(viewer.window, False)
    focus[0] = False
    viewer.poll(0.0)
    assert viewer.commands.gear == 1
    focus[0] = True
    viewer._on_focus(viewer.window, True)
    viewer._on_key(viewer.window, 344, 0, 1, 0)
    states[344] = 1
    viewer.poll(0.0)
    assert viewer.commands.gear == 2
    viewer.commands.reset(0.0, (0.0, 0.0))
    viewer.poll(0.0)
    assert viewer.commands.gear == 1
