"""Run a child on an owned Xvfb display; never connect to the user's display.

run(command, outputdir) returns the child's exit code. outputdir is reserved
for the child; X11 logs/receipt use a new sibling named outputdir.name + '.x11'.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import time

HERE = Path(__file__).resolve().parent
XVFB = HERE / 'rootfs/usr/bin/Xvfb'


def _stop_owned(process):
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


def run(command: list[str], outputdir: Path) -> int:
    if not command or not all(isinstance(arg, str) for arg in command):
        raise ValueError('command must be a non-empty list of strings')
    outputdir = Path(outputdir).absolute()
    if outputdir.exists():
        raise FileExistsError('child output must be new: ' + str(outputdir))
    logdir = outputdir.with_name(outputdir.name + '.x11')
    logdir.mkdir(mode=0o700, parents=True, exist_ok=False)
    auth = logdir / 'Xauthority'
    auth.touch(mode=0o600, exist_ok=False)
    server = child = None
    report = {'status': 'starting', 'started_at': datetime.now(timezone.utc).isoformat(),
              'child_output': str(outputdir), 'logs': str(logdir),
              'parent_display_unchanged': os.environ.get('DISPLAY'),
              'auth_cookie_logged': False, 'command': command}
    previous_handlers = {}

    def interrupted(number, frame):
        raise KeyboardInterrupt('isolated X11 received signal ' + str(number))

    try:
        for number in (signal.SIGTERM, signal.SIGHUP):
            previous_handlers[number] = signal.signal(number, interrupted)
        if not XVFB.is_file():
            raise FileNotFoundError('local Xvfb package not extracted')
        candidates = list(range(100, 200))
        secrets.SystemRandom().shuffle(candidates)
        number = next((n for n in candidates if not Path('/tmp/.X11-unix/X' + str(n)).exists()
                       and not Path('/tmp/.X' + str(n) + '-lock').exists()), None)
        if number is None:
            raise RuntimeError('no unused isolated display in :100..:199')
        display = ':' + str(number)
        report['display'] = display
        # Supply the cookie through stdin, never argv, logs, or the receipt.
        cookie = secrets.token_hex(16)
        configured = subprocess.run(['rtk', 'proxy', 'xauth', '-f', str(auth)],
            input='add ' + display + ' MIT-MAGIC-COOKIE-1 ' + cookie + '\n',
            text=True, capture_output=True, timeout=10)
        if configured.returncode:
            raise RuntimeError('private xauth setup failed, exit=' + str(configured.returncode))
        auth.chmod(0o600)
        environment = os.environ.copy()
        environment.update(DISPLAY=display, XAUTHORITY=str(auth), LIBGL_ALWAYS_SOFTWARE='1')
        environment.pop('WAYLAND_DISPLAY', None)
        server_command = ['rtk', 'proxy', str(XVFB), display, '-screen', '0', '1600x1000x24',
                          '-nolisten', 'tcp', '-auth', str(auth), '-noreset', '+extension', 'GLX']
        with (logdir / 'xvfb.log').open('x') as server_log, (logdir / 'child.log').open('x') as child_log:
            server = subprocess.Popen(server_command, env=environment, stdin=subprocess.DEVNULL,
                stdout=server_log, stderr=subprocess.STDOUT, start_new_session=True)
            report['server_pid'] = server.pid
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError('owned Xvfb stopped during startup; see xvfb.log')
                ready = subprocess.run(['rtk', 'proxy', 'xdpyinfo', '-display', display], env=environment,
                    capture_output=True, text=True, timeout=3)
                if ready.returncode == 0:
                    (logdir / 'xdpyinfo.txt').write_text(ready.stdout)
                    break
                time.sleep(.1)
            else:
                raise RuntimeError('owned Xvfb did not become ready within 20 seconds')
            report['status'] = 'child_running'
            child = subprocess.Popen(command, env=environment, stdin=subprocess.DEVNULL,
                stdout=child_log, stderr=subprocess.STDOUT, start_new_session=True)
            report['child_pid'] = child.pid
            while child.poll() is None:
                if server.poll() is not None:
                    raise RuntimeError('owned Xvfb stopped while child was running')
                time.sleep(.1)
            report['child_returncode'] = child.returncode
            report['status'] = 'passed' if child.returncode == 0 else 'child_failed'
            return child.returncode
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        _stop_owned(child)
        _stop_owned(server)
        auth.unlink(missing_ok=True)
        report['auth_removed'] = not auth.exists()
        report['owned_server_stopped'] = server is None or server.poll() is not None
        report['finished_at'] = datetime.now(timezone.utc).isoformat()
        (logdir / 'launcher_receipt.json').write_text(json.dumps(report, indent=2) + '\n')
        for number, handler in previous_handlers.items():
            signal.signal(number, handler)


if __name__ == '__main__':
    import argparse
    import shutil
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe-output', type=Path, required=True)
    arguments = parser.parse_args()
    probe = ['rtk', 'proxy', 'glxinfo', '-B'] if shutil.which('glxinfo') else ['rtk', 'proxy', 'xdpyinfo']
    raise SystemExit(run(probe, arguments.probe_output))
