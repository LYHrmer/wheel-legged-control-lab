"""Send press/release events only to our child window; never global fake keys."""
import argparse
import ctypes as C
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/home/lyh/wheel-legged-control-lab')
spec = importlib.util.spec_from_file_location('old_window_helpers', ROOT / 'results/d1_budget_demo/gui_check.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class XKeyEvent(C.Structure):
    _fields_ = [('type', C.c_int), ('serial', C.c_ulong), ('send_event', C.c_int),
                ('display', C.c_void_p), ('window', C.c_ulong), ('root', C.c_ulong),
                ('subwindow', C.c_ulong), ('time', C.c_ulong), ('x', C.c_int),
                ('y', C.c_int), ('x_root', C.c_int), ('y_root', C.c_int),
                ('state', C.c_uint), ('keycode', C.c_uint), ('same_screen', C.c_int)]


class XClientMessageEvent(C.Structure):
    _fields_ = [('type', C.c_int), ('serial', C.c_ulong), ('send_event', C.c_int),
                ('display', C.c_void_p), ('window', C.c_ulong),
                ('message_type', C.c_ulong), ('format', C.c_int), ('data', C.c_long*5)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--course', action='store_true')
    parser.add_argument('--heading', action='store_true')
    args = parser.parse_args()
    if args.heading:
        args.course = True
    args.output.mkdir(parents=True, exist_ok=False)
    x = module.X11()
    x.x.XSendEvent.argtypes = [C.c_void_p, C.c_ulong, C.c_int, C.c_long, C.c_void_p]
    x.x.XSendEvent.restype = C.c_int
    command = ['rtk', 'proxy', sys.executable, '-B', str(ROOT/'scripts/run_d1_locomotion.py'),
               '--keyboard', '--source', 'sensor', '--baseline', 'wheel_leg',
               '--action-mode', 'independent8', '--seed', '1017', '--seconds', '30',
               '--wheel-kp', '0.55', '--wheel-ki', '1.5', '--yaw-feedback-gain', '4',
               '--leg-feedback-scale', '1', '--attitude-feedback-scale', '0.25',
               '--output', str(args.output/'rollout')]
    if args.course:
        command = ['rtk', 'proxy', sys.executable, '-B', str(ROOT/'scripts/run_d1_course_drive.py'),
                   '--zone', 'start' if args.heading else 'rough', '--seconds', '60', '--output', str(args.output/'rollout')]
    events = []
    environment = os.environ.copy()
    environment.update(PYTHONPATH=str(ROOT/'src')+':'+str(ROOT/'.local-deps'),
                       PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1',
                       OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    with (args.output/'process.log').open('x') as log:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
        window = None
        try:
            deadline = time.monotonic() + 40
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError('child stopped before opening window; see process.log')
                owned = module.descendants(process.pid)
                for candidate in x.property(x.root, '_NET_CLIENT_LIST'):
                    pid = x.property(candidate, '_NET_WM_PID')
                    title = x.property(candidate, '_NET_WM_NAME')
                    if pid and pid[0] in owned and 'D1 driving' in str(title):
                        window, expected_pid = candidate, pid[0]
                        break
                if window:
                    break
                time.sleep(.1)
            if not window:
                raise RuntimeError('no unique owned driving window')
            activate = XClientMessageEvent()
            activate.type, activate.send_event, activate.format = 33, 1, 32
            activate.display, activate.window = x.display, window
            activate.message_type = x.x.XInternAtom(x.display, b'_NET_ACTIVE_WINDOW', 0)
            activate.data[0] = 2
            storage = C.create_string_buffer(24*C.sizeof(C.c_long))
            C.memmove(storage, C.byref(activate), C.sizeof(activate))
            x.x.XSendEvent(x.display, x.root, 0, (1<<20)|(1<<19), storage)
            x.x.XSync(x.display, 0)
            time.sleep(.5)
            x.x.XSetInputFocus(x.display, window, 2, 0)
            x.x.XSync(x.display, 0)
            time.sleep(.25)

            def key(name, down):
                if x.property(window, '_NET_WM_PID') != [expected_pid]:
                    raise RuntimeError('owned window changed')
                event = XKeyEvent()
                event.type = 2 if down else 3
                event.display, event.window, event.root = x.display, window, x.root
                event.time = int(time.monotonic()*1000) & 0xFFFFFFFF
                event.keycode = x.x.XKeysymToKeycode(x.display, x.x.XStringToKeysym(name.encode()))
                event.same_screen = 1
                storage = C.create_string_buffer(24*C.sizeof(C.c_long))
                C.memmove(storage, C.byref(event), C.sizeof(event))
                if not x.x.XSendEvent(x.display, window, 0, 1 if down else 2, storage):
                    raise RuntimeError('XSendEvent failed')
                x.x.XSync(x.display, 0)
                events.append({'key':name, 'down':down, 'monotonic_s':time.monotonic()})

            def screenshot(name):
                subprocess.run(['rtk', 'proxy', 'ffmpeg', '-nostdin', '-loglevel', 'error',
                                '-f', 'x11grab', '-window_id', str(window), '-i', os.environ['DISPLAY'],
                                '-frames:v', '1', str(args.output/name)], check=True, timeout=15)

            if args.heading:
                screenshot('before.png')
                key('w', True)
                time.sleep(1.5)
                key('w', False)
                time.sleep(.3)
                key('s', True)
                time.sleep(1.0)
                key('s', False)
                time.sleep(.4)
                key('q', True)
                time.sleep(1.0)
                key('q', False)
                time.sleep(6)
                screenshot('left_heading.png')
                key('e', True)
                time.sleep(1.0)
                key('e', False)
                time.sleep(6)
                screenshot('right_heading.png')
                key('r', True)
                time.sleep(.08)
                key('r', False)
                time.sleep(1.5)
                screenshot('reset_upright.png')
                key('space', True)
                time.sleep(.08)
                key('space', False)
                time.sleep(1.8)
                key('t', True)
                time.sleep(.4)
                key('t', False)
                key('g', True)
                time.sleep(.4)
                key('g', False)
                key('x', True)
                time.sleep(.08)
                key('x', False)
                screenshot('after.png')
            else:
                screenshot('before.png')
                key('w', True)
                time.sleep(2.0)
                screenshot('holding_w.png')
                key('a', True)
                time.sleep(.8)
                key('w', False)
                time.sleep(.4)
                key('a', False)
                time.sleep(.5)
                key('r', True)
                time.sleep(.6)
                key('r', False)
                key('f', True)
                time.sleep(.6)
                key('f', False)
                key('s', True)
                time.sleep(.7)
                key('space', True)
                time.sleep(.3)
                key('space', False)
                key('s', False)
                screenshot('after.png')
                if args.course:
                    for digit, zone in [('3','ramp'), ('4','stairs'), ('5','bumps'), ('6','jump'), ('1','start')]:
                        key(digit, True)
                        time.sleep(.08)
                        key(digit, False)
                        time.sleep(.6)
                        screenshot(zone+'.png')
                    time.sleep(1)
                    key('j', True)
                    time.sleep(.08)
                    key('j', False)
                    time.sleep(.2)
                    screenshot('jump_request.png')
                    time.sleep(2)
            key('Escape', True)
            key('Escape', False)
            returncode = process.wait(timeout=45)
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=30)
            x.x.XCloseDisplay(x.display)
    rollout = args.output/'rollout'
    with (args.output/'events_sent.json').open('x') as f:
        json.dump(events, f, indent=2)
        f.write('\n')
    summary = json.loads((rollout/'summary.json').read_text())
    protocol = json.loads((rollout/'protocol.json').read_text())
    with (rollout/'telemetry.csv').open() as f:
        rows = list(csv.DictReader(f))
    if args.heading:
        segments = json.loads((rollout/'segments.json').read_text())
        assert returncode == 0 and summary['source_unchanged']
        assert summary['stop_reason'] == 'keyboard_escape'
        assert protocol['schema'] == 'd1-course-heading-driving-v2'
        assert len(segments) == 2 and all(s['zone'] == 'start' for s in segments)
        assert {'crouch','thrust','flight','landing'} <= {row['jump_phase'] for row in rows}
        assert max(float(r['operator_forward_mps']) for r in rows) > .25
        assert min(float(r['operator_forward_mps']) for r in rows) < -.15
        assert max(float(r['actual_heading_rad']) for r in rows) > .2
        assert max(float(r['heading_target_rad']) for r in rows) > .45
        post = [r for r in rows if r['segment_id'] == '1']
        assert post[0]['operator_forward_mps'] == '0.0'
        assert abs(float(post[0]['heading_target_rad'])) < 1e-6
        result = {'status':'passed','automatic_test_not_human_acceptance':True,
                  'summary':summary,'events_sent':events,'source_sha256':protocol['source_sha256']}
        with (args.output/'report.json').open('x') as f:
            json.dump(result, f, indent=2)
            f.write('\n')
        print(json.dumps({'status':'passed','steps':len(rows),'segments':len(segments)}))
        return
    if args.course:
        segments = json.loads((rollout/'segments.json').read_text())
        phases = {row['jump_phase'] for row in rows}
        assert returncode == 0 and summary['stop_reason'] == 'keyboard_escape'
        assert summary['source_unchanged'] and protocol['policy'] == 'none'
        assert [segment['zone'] for segment in segments] == ['rough','ramp','stairs','bumps','jump','start']
        assert {'crouch','thrust','flight','landing'} <= phases, phases
        assert max(float(row['requested_forward_mps']) for row in rows) > .25
        assert any(float(row['requested_forward_mps']) < -.05 for row in rows)
        assert 'scripts/d1_terrain_display.py' in protocol['source_sha256']
        result = {'status':'passed','automatic_test_not_human_acceptance':True,
                  'summary':summary,'events_sent':events,'source_sha256':protocol['source_sha256']}
        with (args.output/'report.json').open('x') as f:
            json.dump(result, f, indent=2)
            f.write('\n')
        print(json.dumps({'status':'passed','steps':len(rows),'segments':len(segments),'jump_phases':sorted(phases)}))
        return
    vx = [float(r['command_vx_mps']) for r in rows]
    yaw = [float(r['command_yaw_rps']) for r in rows]
    height = [float(r['command_clearance_m']) for r in rows]
    assert returncode == 1 and summary['stop_reason'] == 'keyboard_escape'
    assert protocol['keyboard_input'] == 'held_keys_glfw_v1'
    assert protocol['episode']['terrain']['layout'] != 'flat'
    assert summary['source_unchanged']
    received = [json.loads(line) for line in (rollout/'keyboard_events.jsonl').read_text().splitlines()]
    w_press = [e for e in received if e.get('key') == 87 and e.get('action') == 1]
    w_release = [e for e in received if e.get('key') == 87 and e.get('action') == 0]
    assert len(w_press) == len(w_release) == 1, 'test requires one press and one release'
    assert w_release[0]['wall_time_s'] - w_press[0]['wall_time_s'] >= 2.0
    held = [float(row['command_vx_mps']) for row in rows
            if w_press[0]['simulation_time_s'] + .05 <= float(row['decision_time_s'])
            < w_release[0]['simulation_time_s']]
    assert len(held) >= 20 and min(held) > 0 and held[-1] > .25, 'held key expired or did not accelerate'
    assert any(v < -.05 for v in vx), 'reverse missing'
    assert any(v > .1 and y > .1 for v,y in zip(vx,yaw)), 'combined keys missing'
    assert any(v == 0 and y > .1 for v,y in zip(vx,yaw)), 'release of one axis lost other axis'
    assert max(height) > .46 and height[-1] < max(height), 'held height up/down missing'
    result = {'status':'passed', 'input_method':'XSendEvent to owned child window, no global key injection',
              'automatic_test_not_human_acceptance':True, 'summary':summary,
              'events_sent':events, 'source_sha256':protocol['source_sha256']}
    with (args.output/'report.json').open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps({'status':'passed', 'steps':len(rows), 'events':len(events)}))


if __name__ == '__main__':
    main()
