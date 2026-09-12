"""Exercise only this process's MuJoCo window, and capture its drawable only."""
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def descendants(pid):
    found = {pid}
    for child in Path(f"/proc/{pid}/task/{pid}/children").read_text().split():
        try:
            found.update(descendants(int(child)))
        except FileNotFoundError:
            pass
    return found


class X11:
    def __init__(self):
        self.x = C.CDLL("libX11.so.6")
        self.xt = C.CDLL("libXtst.so.6")
        signatures = {
            "XOpenDisplay": ([C.c_char_p], C.c_void_p),
            "XDefaultRootWindow": ([C.c_void_p], C.c_ulong),
            "XInternAtom": ([C.c_void_p, C.c_char_p, C.c_int], C.c_ulong),
            "XGetWindowProperty": ([C.c_void_p, C.c_ulong, C.c_ulong,
                C.c_long, C.c_long, C.c_int, C.c_ulong, C.POINTER(C.c_ulong),
                C.POINTER(C.c_int), C.POINTER(C.c_ulong), C.POINTER(C.c_ulong),
                C.POINTER(C.POINTER(C.c_ubyte))], C.c_int),
            "XFree": ([C.c_void_p], C.c_int),
            "XSetInputFocus": ([C.c_void_p, C.c_ulong, C.c_int, C.c_ulong], C.c_int),
            "XGetInputFocus": ([C.c_void_p, C.POINTER(C.c_ulong), C.POINTER(C.c_int)], C.c_int),
            "XStringToKeysym": ([C.c_char_p], C.c_ulong),
            "XKeysymToKeycode": ([C.c_void_p, C.c_ulong], C.c_ubyte),
            "XGrabServer": ([C.c_void_p], C.c_int),
            "XUngrabServer": ([C.c_void_p], C.c_int),
            "XSync": ([C.c_void_p, C.c_int], C.c_int),
            "XCloseDisplay": ([C.c_void_p], C.c_int),
        }
        for name, (args, result) in signatures.items():
            getattr(self.x, name).argtypes = args
            getattr(self.x, name).restype = result
        self.xt.XTestFakeKeyEvent.argtypes = [C.c_void_p, C.c_uint, C.c_int, C.c_ulong]
        self.xt.XTestFakeKeyEvent.restype = C.c_int
        self.display = self.x.XOpenDisplay(os.environ.get("DISPLAY", ":0").encode())
        if not self.display:
            raise RuntimeError("Cannot connect to authorized X11 display")
        self.root = self.x.XDefaultRootWindow(self.display)

    def property(self, window, name):
        atom = self.x.XInternAtom(self.display, name.encode(), 1)
        actual, fmt, count, remaining = C.c_ulong(), C.c_int(), C.c_ulong(), C.c_ulong()
        data = C.POINTER(C.c_ubyte)()
        status = self.x.XGetWindowProperty(self.display, window, atom, 0, 4096, 0, 0,
            C.byref(actual), C.byref(fmt), C.byref(count), C.byref(remaining), C.byref(data))
        if status or not data:
            return []
        try:
            if fmt.value == 32:
                return list(C.cast(data, C.POINTER(C.c_ulong))[:count.value])
            if fmt.value == 8:
                return C.string_at(data, count.value).decode(errors="replace")
            return []
        finally:
            self.x.XFree(data)

    def find_owned(self, launch_pid):
        pids = descendants(launch_pid)
        owned = []
        for window in self.property(self.root, "_NET_CLIENT_LIST"):
            pid = self.property(window, "_NET_WM_PID")
            if pid and pid[0] in pids:
                name = self.property(window, "_NET_WM_NAME") or self.property(window, "WM_NAME")
                if "MuJoCo" in str(name):
                    owned.append((window, pid[0], name))
        if len(owned) > 1:
            raise RuntimeError("Multiple owned MuJoCo windows; refusing ambiguous targeting")
        return owned[0] if owned else None

    def key(self, window, expected_pid, key):
        if self.property(window, "_NET_WM_PID") != [expected_pid]:
            raise RuntimeError("Owned window PID changed; no key sent")
        # Server grab makes focus validation and paired key events atomic w.r.t.
        # other X11 clients. No event is sent to a user-selected active window.
        self.x.XGrabServer(self.display)
        try:
            self.x.XSetInputFocus(self.display, window, 2, 0)
            focus, revert = C.c_ulong(), C.c_int()
            self.x.XGetInputFocus(self.display, C.byref(focus), C.byref(revert))
            if focus.value != window:
                raise RuntimeError("Owned window did not receive focus; no key sent")
            code = self.x.XKeysymToKeycode(self.display, self.x.XStringToKeysym(key.encode()))
            if not code:
                raise RuntimeError(f"Unknown keysym {key}")
            if not self.xt.XTestFakeKeyEvent(self.display, code, 1, 0):
                raise RuntimeError("XTest key down failed")
            if not self.xt.XTestFakeKeyEvent(self.display, code, 0, 0):
                raise RuntimeError("XTest key up failed")
            self.x.XSync(self.display, 0)
        finally:
            self.x.XUngrabServer(self.display)
            self.x.XSync(self.display, 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir()
    x11 = X11()
    command = ["rtk", "proxy", "env", "PYTHONDONTWRITEBYTECODE=1",
        f"PYTHONPATH={args.root / 'src'}:{args.root / '.local-deps'}",
        "OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1", "MKL_NUM_THREADS=1",
        "python3", str(args.root / "scripts/run_d1_locomotion.py"),
        "--keyboard", "--source", "sensor", "--seed", "1017", "--seconds", "30",
        "--baseline", "wheel_leg", "--action-mode", "independent8",
        "--wheel-kp", "0.55", "--wheel-ki", "1.5", "--yaw-feedback-gain", "4",
        "--leg-feedback-scale", "1", "--attitude-feedback-scale", "0.25",
        "--output", str(args.output / "rollout")]
    events = [(0.5, "w"), (0.75, "w"), (1.0, "a"), (1.25, "r"),
        (2.3, "r"), (2.55, "f"), (3.0, "w"), (3.3, "d"), (3.6, "space"),
        (4.0, "s"), (4.3, "a"), (4.6, "space"), (5.0, "r"), (5.3, "f")]
    report = {"kind": "automated_real_gui_integration", "human_manual_acceptance": False,
        "capture_scope": "owned MuJoCo client window drawable only; no desktop capture",
        "command": command, "planned_events": events, "events": [],
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    process = capture = None
    with (args.output / "viewer.log").open("x") as log, (args.output / "capture.log").open("x") as caplog:
        try:
            process = subprocess.Popen(command, cwd=args.root, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
            deadline = time.monotonic() + 40
            owned = None
            while time.monotonic() < deadline and process.poll() is None:
                owned = x11.find_owned(process.pid)
                if owned:
                    break
                time.sleep(0.1)
            if owned is None:
                raise RuntimeError("No uniquely owned MuJoCo window appeared")
            window, pid, name = owned
            report.update(window_id=hex(window), window_pid=pid, window_name=name)
            subprocess.run(["rtk", "proxy", "xwd", "-silent", "-id", hex(window),
                "-out", str(args.output / "window.xwd")], check=True)
            capcommand = ["rtk", "proxy", "ffmpeg", "-hide_banner", "-loglevel", "warning",
                "-n", "-f", "x11grab", "-window_id", str(window), "-draw_mouse", "0",
                "-framerate", "20", "-i", os.environ.get("DISPLAY", ":0"), "-an",
                "-threads", "1", "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p", "-t", "6.6",
                str(args.output / "window.mp4")]
            report["capture_command"] = capcommand
            capture = subprocess.Popen(capcommand, stdout=caplog, stderr=subprocess.STDOUT,
                start_new_session=True)
            start = time.monotonic()
            report["event_origin_monotonic_s"] = start
            for planned, key in events:
                time.sleep(max(0, start + planned - time.monotonic()))
                if process.poll() is not None:
                    raise RuntimeError("Viewer process exited before planned event")
                x11.key(window, pid, key)
                report["events"].append({"planned_wall_s": planned,
                    "actual_wall_s": time.monotonic() - start, "key": key,
                    "target_window": hex(window), "target_pid": pid})
            time.sleep(max(0, start + 6.6 - time.monotonic()))
            report["capture_exit_code"] = capture.wait(timeout=20)
            if report["capture_exit_code"] != 0:
                raise RuntimeError("Window capture did not close normally")
            x11.key(window, pid, "Escape")
            report["events"].append({"actual_wall_s": time.monotonic() - start,
                "key": "Escape", "target_window": hex(window), "target_pid": pid})
            report["viewer_exit_code"] = process.wait(timeout=15)
            report["status"] = "captured; telemetry verification required"
        except Exception as exc:
            report.update(status="failed; partial evidence retained", error=repr(exc))
            raise
        finally:
            for child in (capture, process):
                if child is not None and child.poll() is None:
                    os.killpg(child.pid, signal.SIGINT)
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGTERM)
                        child.wait(timeout=10)
            x11.x.XCloseDisplay(x11.display)
            save(args.output / "events.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
