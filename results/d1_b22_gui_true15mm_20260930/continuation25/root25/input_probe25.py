"""One finite real GLFW/XTest focus probe; zero models and physics."""
import argparse
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import sys
import time


class NoEngine(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'mujoco', 'torch', 'stable_baselines3', 'gymnasium',
                                       'wheel_legged_control', 'engine_binding'}:
            raise RuntimeError('forbidden model/engine import: '+fullname)
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.meta_path.insert(0, NoEngine())
    import glfw
    from x11_events23 import EventDriver
    session = json.loads(args.session.read_text())
    assert session['control_limit'] == 0 and not any(session['model_limits'].values())
    assert not any(name.split('.')[0] in ('mujoco', 'torch', 'stable_baselines3') for name in sys.modules)
    ready = {'pid': os.getpid(), 'session_sha256': hashlib.sha256(args.session.read_bytes()).hexdigest(),
             'monotonic_s': time.monotonic()}
    (args.output/'preflight_ready.json').write_text(json.dumps(ready))
    assert glfw.init()
    window = None
    driver = None
    callbacks = []
    injections = []
    try:
        window = glfw.create_window(320, 200, 'B22 input-only probe', None, None)
        assert window
        glfw.set_key_callback(window, lambda _w, key, scan, action, mods:
                              callbacks.append({'type': 'key', 'key': key, 'action': action,
                                                'wall_ns': time.monotonic_ns()}))
        glfw.set_window_focus_callback(window, lambda _w, focused:
                                       callbacks.append({'type': 'focus', 'focused': bool(focused),
                                                         'wall_ns': time.monotonic_ns()}))
        driver = EventDriver(args.output, [])
        driver.initialize({'x11_window': int(glfw.get_x11_window(window))})
        # Wall-time landmarks correspond to the proposed focus fragment. No controls are simulated.
        events = [(0.20, 450, 'key_down'), (0.40, 470, 'focus_lost'),
                  (0.70, 500, 'focus_return'), (0.75, 505, 'key_down'),
                  (0.80, 510, 'key_up'), (0.90, 520, 'key_down'), (1.60, 590, 'key_up')]
        start = time.monotonic()
        index = 0
        next_poll = start
        while time.monotonic()-start < 2.0:
            now = time.monotonic()
            while index < len(events) and now-start >= events[index][0]:
                _, label, operation = events[index]
                if operation.startswith('key_'):
                    driver.key('w', operation == 'key_down')
                else:
                    target = driver.helper if operation == 'focus_lost' else driver.window
                    driver.x.XSetInputFocus(driver.display, target, 1, 0)
                driver.x.XSync(driver.display, 0)
                injections.append({'landmark': label, 'operation': operation, 'wall_ns': time.monotonic_ns()})
                index += 1
            if now >= next_poll:
                glfw.poll_events()
                next_poll = now+0.08
            time.sleep(0.001)
        glfw.poll_events()
        assert index == len(events)
        (args.output/'input_only_receipt.json').write_text(json.dumps({
            'schema': 'd1-c25-real-input-only-probe-v1', 'injections': injections, 'callbacks': callbacks,
            'elapsed_s': time.monotonic()-start, 'model_loads': 0, 'controls': 0,
            'normal_native': 0, 'compiler_native': 0, 'render_calls': 0,
            'host_compiler_reservation_unused': 2,
            'physical_human_keyboard_tested': False}, indent=2)+'\n')
    finally:
        if driver is not None:
            driver.close()
        if window is not None:
            glfw.destroy_window(window)
        glfw.terminate()


if __name__ == '__main__':
    main()
