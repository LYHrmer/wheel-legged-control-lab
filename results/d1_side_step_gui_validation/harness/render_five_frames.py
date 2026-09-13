"""Bounded render-only benchmark on one owned display; no physics steps or input."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = Path('/home/lyh/wheel-legged-control-lab')
MODEL = HERE.parent / 'side_gui_check_06/rollout/model.mjb'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def measure(output, quality):
    sys.path[:0] = [str(REPO / 'scripts'), str(REPO / 'src'), str(REPO / '.local-deps')]
    import mujoco
    from d1_keyboard_viewer import KeyboardViewer
    class Commands:
        def update_pressed(self, *args, **kwargs):
            pass
    model = mujoco.MjModel.from_binary_path(str(MODEL))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    initial_qpos = data.qpos.copy()
    initial_time = data.time
    start = time.perf_counter()
    with KeyboardViewer(model, data, Commands(), terrain_label='Course: flat', render_quality=quality) as viewer:
        init_s = time.perf_counter() - start
        gl = ctypes.CDLL('libGL.so.1')
        gl.glGetString.argtypes = [ctypes.c_uint]
        gl.glGetString.restype = ctypes.c_char_p
        gl.glFinish.argtypes = []
        renderer = gl.glGetString(0x1F01).decode()
        version = gl.glGetString(0x1F02).decode()
        frame_seconds = []
        for _ in range(5):
            viewer._last_render = -float('inf')
            start = time.perf_counter()
            viewer.sync()
            gl.glFinish()
            frame_seconds.append(time.perf_counter() - start)
        status = Path('/proc/self/status').read_text().splitlines()
        thread_count = int(next(line.split()[1] for line in status if line.startswith('Threads:')))
        assert data.time == initial_time and (data.qpos == initial_qpos).all()
        report = dict(status='passed', renderer=renderer, gl_version=version, render_quality=quality,
            lp_num_threads=os.environ.get('LP_NUM_THREADS'), process_threads=thread_count,
            init_s=init_s, frame_seconds=frame_seconds, frame_mean_s=sum(frame_seconds)/5,
            shadow_size=int(model.vis.quality.shadowsize), offsamples=int(model.vis.quality.offsamples),
            scene_geoms=int(viewer.scene.ngeom), framebuffer=list(viewer.glfw.get_framebuffer_size(viewer.window)),
            simulation_time_unchanged=True, qpos_unchanged=True, physics_steps=0)
    output.write_text(json.dumps(report, indent=2) + '\n')


def compare(output):
    from isolated_x11 import _stop_owned
    output.mkdir(parents=True, exist_ok=False)
    reports = []
    for label, threads in [('normal', None), ('low', None)]:
        env = os.environ.copy()
        if threads is None:
            env.pop('LP_NUM_THREADS', None)
        else:
            env['LP_NUM_THREADS'] = threads
        target = output / ('quality_' + label + '.json')
        command = ['rtk', 'proxy', sys.executable, '-B', str(Path(__file__).resolve()),
                   '--measure', '--quality', label, '--output', str(target)]
        start = time.perf_counter()
        with (output / ('quality_' + label + '.log')).open('x') as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                returncode = process.wait(timeout=25)
                report = json.loads(target.read_text()) if target.exists() else {'status': 'failed', 'exit_code': returncode}
            except subprocess.TimeoutExpired:
                report = {'status': 'timeout', 'limit_s': 25}
            finally:
                _stop_owned(process)
        report.update(label=label, subprocess_wall_s=time.perf_counter()-start)
        reports.append(report)
        (output / 'partial_results.json').write_text(json.dumps(reports, indent=2) + '\n')
    result = dict(status='completed', exact_model=str(MODEL), model_sha256=sha256(MODEL),
        viewer_sha256=sha256(REPO/'scripts/d1_keyboard_viewer.py'),
        cases=reports, isolated_display=os.environ.get('DISPLAY'), parent_desktop_touched=False,
        note='Same serialized course model; five full sync frames with glFinish; no physics or keyboard input. Only explicit render quality differs between children; normal and low share the same viewer source.')
    (output / 'benchmark.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--measure', action='store_true')
    parser.add_argument('--quality', choices=('normal', 'low'), default='normal')
    parser.add_argument('--child', action='store_true')
    args = parser.parse_args()
    if args.measure:
        measure(args.output, args.quality)
    elif args.child:
        compare(args.output)
    else:
        from isolated_x11 import run
        raise SystemExit(run(['rtk', 'proxy', sys.executable, '-B', str(Path(__file__).resolve()),
                              '--child', '--output', str(args.output)], args.output))
