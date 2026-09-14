"""Validate Shift through real GLFW on an owned private Xvfb display."""
import argparse
import ctypes as C
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path('/home/lyh/wheel-legged-control-lab')
WORK = Path('/home/lyh/wheel-legged-control-lab-work/recovery-20260912')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--child', action='store_true')
    args = parser.parse_args()
    if not args.child:
        launcher = load('private_x11', WORK/'gui_isolation/isolated_x11.py')
        command = ['rtk', 'proxy', sys.executable, '-B', str(Path(__file__).resolve()),
                   '--output', str(args.output), '--child']
        return launcher.run(command, args.output)
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(ROOT))
    from scripts.d1_course_controls import CourseKeyboardCommands
    from scripts.d1_keyboard_viewer import KeyboardViewer
    from wheel_legged_control.d1.interactive import D1InteractiveSimulation
    helpers = load('input_helpers', WORK/'verify_held_gui.py')
    x11 = helpers.module.X11()
    x11.x.XSendEvent.argtypes = [C.c_void_p, C.c_ulong, C.c_int, C.c_long, C.c_void_p]
    x11.x.XSendEvent.restype = C.c_int
    simulation = D1InteractiveSimulation()
    commands = CourseKeyboardCommands(clock=lambda: 0.0)
    checks = []
    with KeyboardViewer(simulation.plant.model, simulation.plant.data,
                        commands, render_quality='low') as viewer:
        window = viewer.glfw.get_x11_window(viewer.window)
        x11.x.XSetInputFocus(x11.display, window, 2, 0)
        x11.x.XSync(x11.display, 0)
        viewer.poll(0)
        timestamp = 1000

        def send(name, down):
            nonlocal timestamp
            timestamp += 10
            event = helpers.XKeyEvent()
            event.type = 2 if down else 3
            event.display, event.window, event.root = x11.display, window, x11.root
            event.time = timestamp
            event.keycode = x11.x.XKeysymToKeycode(
                x11.display, x11.x.XStringToKeysym(name.encode()))
            event.same_screen = 1
            storage = C.create_string_buffer(24*C.sizeof(C.c_long))
            C.memmove(storage, C.byref(event), C.sizeof(event))
            if not x11.x.XSendEvent(x11.display, window, 0, 1 if down else 2, storage):
                raise RuntimeError('owned-window XSendEvent failed')
            x11.x.XSync(x11.display, 0)

        def check(name, expected):
            viewer.poll(0)
            checks.append({'case': name, 'gear': commands.gear,
                           'expected': expected, 'passed': commands.gear == expected})
            assert commands.gear == expected, checks[-1]

        send('Shift_L', True)
        send('Shift_L', False)
        check('press_release_in_one_poll', 2)
        send('Shift_R', True)
        check('right_shift_press', 3)
        for _ in range(4):
            viewer.poll(0)
        check('held_shift_does_not_cycle', 3)
        send('Shift_R', True)
        check('repeated_press_does_not_cycle', 3)
        send('Shift_L', True)
        check('both_shifts_overlap_is_one_press', 3)
        send('Shift_R', False)
        send('Shift_L', False)
        check('release_keeps_gear', 3)
        for _ in range(2):
            send('Shift_L', True)
            send('Shift_L', False)
        check('two_taps_in_one_poll', 2)
        send('Shift_L', True)
        check('held_before_reset', 3)
        commands.reset(0, (0, 0))
        check('reset_with_shift_still_held', 1)
        send('Shift_L', False)
        check('release_after_reset', 1)
        events = list(viewer.events)
    report = {'status': 'passed', 'display': os.environ.get('DISPLAY'),
              'desktop_used': False, 'physics_stepped': False,
              'input_transport': 'XSendEvent to own GLFW window on private Xvfb',
              'checks': checks, 'events': events,
              'probe_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
