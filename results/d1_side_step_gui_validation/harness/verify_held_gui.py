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
    parser.add_argument('--side', action='store_true')
    parser.add_argument('--render-quality', choices=('normal', 'low'), default='normal')
    parser.add_argument('--isolated', action='store_true')
    parser.add_argument('--isolated-child', action='store_true')
    args = parser.parse_args()
    if args.isolated:
        launcher_path = Path(__file__).parent/'gui_isolation/isolated_x11.py'
        launcher_spec = importlib.util.spec_from_file_location('isolated_x11', launcher_path)
        launcher = importlib.util.module_from_spec(launcher_spec)
        launcher_spec.loader.exec_module(launcher)
        child_args = [a for a in sys.argv[1:] if a != '--isolated']
        command = ['rtk', 'proxy', sys.executable, '-B', str(Path(__file__).resolve()),
                   *child_args, '--isolated-child']
        raise SystemExit(launcher.run(command, args.output))
    if args.heading or args.side:
        args.course = True
    args.output.mkdir(parents=True, exist_ok=False)
    x = module.X11()
    original_property = x.property

    def existing_property(window, name):
        # A fresh X server has no WM atoms until a client creates them.
        if not x.x.XInternAtom(x.display, name.encode(), 1):
            return []
        return original_property(window, name)

    x.property = existing_property
    x.x.XSendEvent.argtypes = [C.c_void_p, C.c_ulong, C.c_int, C.c_long, C.c_void_p]
    x.x.XSendEvent.restype = C.c_int
    x.x.XQueryTree.argtypes = [C.c_void_p, C.c_ulong, C.POINTER(C.c_ulong),
                              C.POINTER(C.c_ulong), C.POINTER(C.POINTER(C.c_ulong)),
                              C.POINTER(C.c_uint)]
    x.x.XQueryTree.restype = C.c_int

    def candidates():
        managed = x.property(x.root, '_NET_CLIENT_LIST')
        if managed or not args.isolated_child:
            return managed
        root, parent, count = C.c_ulong(), C.c_ulong(), C.c_uint()
        children = C.POINTER(C.c_ulong)()
        if not x.x.XQueryTree(x.display, x.root, C.byref(root), C.byref(parent),
                             C.byref(children), C.byref(count)):
            return []
        try:
            return list(children[:count.value])
        finally:
            if children:
                x.x.XFree(children)
    command = ['rtk', 'proxy', sys.executable, '-B', str(ROOT/'scripts/run_d1_locomotion.py'),
               '--keyboard', '--source', 'sensor', '--baseline', 'wheel_leg',
               '--action-mode', 'independent8', '--seed', '1017', '--seconds', '30',
               '--wheel-kp', '0.55', '--wheel-ki', '1.5', '--yaw-feedback-gain', '4',
               '--leg-feedback-scale', '1', '--attitude-feedback-scale', '0.25',
               '--output', str(args.output/'rollout')]
    if args.course:
        command = ['rtk', 'proxy', sys.executable, '-B', str(ROOT/'scripts/run_d1_course_drive.py'),
                   '--zone', 'start' if args.heading or args.side else 'rough', '--seconds', '240' if args.side else '60', '--output', str(args.output/'rollout')]
        if args.render_quality != 'normal':
            command.extend(['--render-quality', args.render_quality])
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
                for candidate in candidates():
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
                with (args.output/'events_sent.jsonl').open('a') as stream:
                    stream.write(json.dumps(events[-1])+'\n')

            def screenshot(name):
                subprocess.run(['rtk', 'proxy', 'ffmpeg', '-nostdin', '-loglevel', 'error',
                                '-f', 'x11grab', '-window_id', str(window), '-i', os.environ['DISPLAY'],
                                '-frames:v', '1', str(args.output/name)], check=True, timeout=15)

            if args.side:
                telemetry = args.output/'rollout/telemetry.csv'
                handoff_limit = 180. if args.isolated_child else 20.
                def rows_now():
                    try:
                        with telemetry.open() as stream:
                            return [r for r in csv.DictReader(stream)
                                    if r.get('total_time_s') and r.get('applied_controller')]
                    except FileNotFoundError:
                        return []
                def wait_rows(predicate, limit=None, fail_on_cancel=False):
                    if limit is None:
                        limit = 900. if args.isolated_child else 180.
                    until = time.monotonic()+limit
                    while time.monotonic() < until:
                        if process.poll() is not None:
                            raise RuntimeError('child stopped during side-step trial')
                        rows = rows_now()
                        if rows and fail_on_cancel and rows[-1]['side_failure']:
                            raise RuntimeError('side step interrupted: '+rows[-1]['side_failure']+'; focus='+rows[-1].get('input_focused','unknown'))
                        if rows and predicate(rows):
                            return rows
                        time.sleep(.15)
                    raise RuntimeError('side-step phase timeout; raw trajectory retained')
                def wait_sim(seconds):
                    first_time = float(wait_rows(bool)[-1]['total_time_s'])
                    return wait_rows(lambda rows: float(rows[-1]['total_time_s'])
                                     >= first_time + seconds)
                def press(name):
                    key(name, True)
                    time.sleep(.08)
                    key(name, False)
                wait_sim(1.5)
                screenshot('before.png')
                for name, direction in [('a', 1), ('d', -1)]:
                    first = len(rows_now())
                    key(name, True)
                    wait_rows(lambda rows: any(r['side_phase']=='swing' for r in rows[first:]))
                    screenshot(name+'_swing.png')
                    wait_rows(lambda rows: any(r['side_success']=='True' and r['side_direction']==str(direction) for r in rows[first:]), fail_on_cancel=True)
                    key(name, False)
                    wait_rows(lambda rows: rows[-1]['applied_controller']=='legacy', handoff_limit)
                    wait_sim(.6)
                    screenshot(name+'_complete.png')
                    press('r')
                    wait_sim(1.5)
                first = len(rows_now())
                key('a', True)
                wait_rows(lambda rows: any(r['side_phase']=='swing' for r in rows[first:]))
                press('space')
                key('a', False)
                wait_rows(lambda rows: rows[-1]['applied_controller']=='legacy', handoff_limit)
                screenshot('cancel_landed.png')
                press('r')
                wait_sim(1.5)
                first = len(rows_now())
                key('a', True)
                wait_rows(lambda rows: any(r['side_phase']=='swing' for r in rows[first:]))
                # Real X11 focus transition exercises GLFW's cancellation path.
                x.x.XSetInputFocus(x.display, x.root, 2, 0)
                x.x.XSync(x.display, 0)
                wait_rows(lambda rows: rows[-1]['applied_controller']=='legacy'
                          and rows[-1]['input_focused']=='False', handoff_limit)
                screenshot('focus_cancel_landed.png')
                x.x.XSetInputFocus(x.display, window, 2, 0)
                x.x.XSync(x.display, 0)
                time.sleep(.2)
                key('a', False)
                press('r')
                wait_sim(1.5)
                press('space')
                wait_sim(1.8)
                heading_before = float(rows_now()[-1]['heading_target_rad'])
                key('q', True)
                wait_rows(lambda rows: float(rows[-1]['heading_target_rad'])
                          >= heading_before + .45)
                key('q', False)
                wait_sim(6.)
                screenshot('after.png')
            elif args.heading:
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
        except BaseException as error:
            with (args.output/'failure.json').open('x') as stream:
                json.dump({'type':type(error).__name__,'message':str(error),'automatic_test_not_human_acceptance':True},stream,indent=2)
            raise
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
    if args.side:
        import numpy as np
        segments = json.loads((rollout/'segments.json').read_text())
        assert returncode == 0 and summary['source_unchanged']
        assert summary['stop_reason'] == 'keyboard_escape'
        assert protocol['schema'] == 'd1-course-side-stepping-v3'
        assert len(segments) == 5
        measurements = []
        with np.load(rollout/'states.npz', allow_pickle=False) as states:
            assert len(states['applied_torque_nm']) == len(rows)
            assert np.isfinite(states['applied_torque_nm']).all()
            for segment, direction in [(0, 1), (1, -1)]:
                selected = [r for r in rows if int(r['segment_id'])==segment]
                entering = next(r for r in selected if r['applied_controller'] in ('d1_side_step','d1_fast_side_step'))
                done = next(r for r in selected if r['side_success']=='True')
                before = states['qpos'][int(entering['state_before_index'])]
                after = states['qpos'][int(done['state_after_index'])]
                assert direction*(after[1]-before[1]) > .02
                assert abs(after[0]-before[0]) < .03
                measurements.append({'segment':segment,'direction':direction,'delta_xy_m':(after[:2]-before[:2]).tolist()})
            assert states['qpos'][:,2].min() > .30
        assert any(r['side_failure']=='cancelled' for r in rows)
        with (rollout/'keyboard_events.jsonl').open() as stream:
            assert any(json.loads(line)['type']=='jump_blocked' for line in stream)
        assert {'crouch','thrust','flight','landing'} <= {r['jump_phase'] for r in rows}
        assert any(r['segment_id']=='3' and r['input_focused']=='False'
                   and r['side_failure']=='cancelled' for r in rows)
        assert max(float(r['actual_heading_rad']) for r in rows if r['segment_id']=='4') > .2
        result = {'status':'passed','automatic_test_not_human_acceptance':True,
                  'display_mode':'isolated_x11' if args.isolated_child else 'desktop_x11',
                  'summary':summary,'measurements':measurements,'events_sent':events,
                  'source_sha256':protocol['source_sha256']}
        with (args.output/'report.json').open('x') as stream:
            json.dump(result,stream,indent=2)
        print(json.dumps({'status':'passed','steps':len(rows),'segments':len(segments),'measurements':measurements}))
        return
    if args.heading:
        segments = json.loads((rollout/'segments.json').read_text())
        assert returncode == 0 and summary['source_unchanged']
        assert summary['stop_reason'] == 'keyboard_escape'
        assert protocol['schema'] in ('d1-course-heading-driving-v2','d1-course-side-stepping-v3')
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
