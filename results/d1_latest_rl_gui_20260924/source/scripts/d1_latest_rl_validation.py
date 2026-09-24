"""Fixed 07 integration input; no model, physics, or training calls.

GUI validation sends X11 events only to its owned GLFW window on a private
Xvfb display. Headless validation exercises the same pure command callbacks.
Neither path claims that synthetic input is a human usability assessment.
"""

from __future__ import annotations

import ctypes as ct
import os
from typing import Any


class _XKeyEvent(ct.Structure):
    _fields_ = [
        ("type", ct.c_int), ("serial", ct.c_ulong), ("send_event", ct.c_int),
        ("display", ct.c_void_p), ("window", ct.c_ulong), ("root", ct.c_ulong),
        ("subwindow", ct.c_ulong), ("time", ct.c_ulong),
        ("x", ct.c_int), ("y", ct.c_int), ("x_root", ct.c_int),
        ("y_root", ct.c_int), ("state", ct.c_uint), ("keycode", ct.c_uint),
        ("same_screen", ct.c_int),
    ]


class _OwnedX11:
    def __init__(self, viewer: Any) -> None:
        display_name = os.environ.get("DISPLAY", "")
        number = display_name.removeprefix(":").split(".")[0]
        if not display_name.startswith(":") or not number.isdigit() or not 100 <= int(number) < 200:
            raise RuntimeError("07 synthetic GUI input requires private Xvfb :100..:199")
        self.lib = ct.CDLL("libX11.so.6")
        signatures = {
            "XOpenDisplay": ([ct.c_char_p], ct.c_void_p),
            "XDefaultRootWindow": ([ct.c_void_p], ct.c_ulong),
            "XSetInputFocus": ([ct.c_void_p, ct.c_ulong, ct.c_int, ct.c_ulong], ct.c_int),
            "XSync": ([ct.c_void_p, ct.c_int], ct.c_int),
            "XStringToKeysym": ([ct.c_char_p], ct.c_ulong),
            "XKeysymToKeycode": ([ct.c_void_p, ct.c_ulong], ct.c_ubyte),
            "XSendEvent": ([ct.c_void_p, ct.c_ulong, ct.c_int, ct.c_long, ct.c_void_p], ct.c_int),
            "XCloseDisplay": ([ct.c_void_p], ct.c_int),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(self.lib, name)
            function.argtypes, function.restype = arguments, result
        self.display = self.lib.XOpenDisplay(display_name.encode())
        if not self.display:
            raise RuntimeError("cannot open owned validation X11 display")
        self.window = int(viewer.glfw.get_x11_window(viewer.window))
        self.root = self.lib.XDefaultRootWindow(self.display)
        self.timestamp = 1000

    def focus(self, active: bool) -> None:
        self.lib.XSetInputFocus(self.display, self.window if active else self.root, 2, 0)
        self.lib.XSync(self.display, 0)

    def key(self, name: str, pressed: bool) -> None:
        self.timestamp += 10
        event = _XKeyEvent()
        event.type = 2 if pressed else 3
        event.display, event.window, event.root = self.display, self.window, self.root
        event.time, event.same_screen = self.timestamp, 1
        event.keycode = self.lib.XKeysymToKeycode(
            self.display, self.lib.XStringToKeysym(name.encode()))
        if not event.keycode:
            raise RuntimeError("validation key has no X11 keycode: " + name)
        storage = ct.create_string_buffer(24 * ct.sizeof(ct.c_long))
        ct.memmove(storage, ct.byref(event), ct.sizeof(event))
        if not self.lib.XSendEvent(self.display, self.window, 0,
                                  1 if pressed else 2, storage):
            raise RuntimeError("owned-window key event rejected")
        self.lib.XSync(self.display, 0)

    def close(self) -> None:
        if self.display:
            self.lib.XCloseDisplay(self.display)
            self.display = None


# Entries are polling ticks. Prepared commands change at the next tick.
EVENTS = {
    (0, 0): (("focus", True), ("tap", "1")),
    (0, 50): (("down", "W"),),
    (0, 400): (("tap", "2"),),
    (0, 800): (("tap", "space"),),
    (0, 950): (("up", "W"),),
    (0, 1000): (("tap", "R"),),
    (1, 0): (("up", "W"), ("focus", True)),
    (1, 10): (("down", "W"),),
    (1, 200): (("tap", "Escape"),),
}


class ValidationDriver:
    """Deterministic new input sequence over two segments of 1000 + 200 ticks."""

    def __init__(self, profile: str, commands: Any, viewer: Any = None) -> None:
        if profile not in ("gui_policy_box", "headless_zero_plane"):
            raise ValueError("unknown 07 validation profile")
        if (viewer is not None) != (profile == "gui_policy_box"):
            raise ValueError("profile and actual window presence disagree")
        self.commands, self.viewer, self.profile = commands, viewer, profile
        self.x11 = None if viewer is None else _OwnedX11(viewer)
        self.held: set[int] = set()
        self.focused = True
        self.delivered: set[tuple[int, int]] = set()
        self.events: list[dict[str, Any]] = []
        self.checks: list[dict[str, Any]] = []

    @staticmethod
    def _code(name: str) -> int:
        return {"space": 32, "Escape": 256}.get(name, ord(name) if len(name) == 1 else -1)

    def _key(self, name: str, pressed: bool) -> None:
        code = self._code(name)
        if pressed:
            self.held.add(code)
        else:
            self.held.discard(code)
        if self.x11 is not None:
            self.x11.key(name, pressed)
        else:
            self.commands.handle_key_event(code, 1 if pressed else 0)

    def before_poll(self, segment_index: int, tick: int) -> None:
        coordinate = (segment_index, tick)
        if coordinate not in self.delivered:
            for operation, value in EVENTS.get(coordinate, ()):
                if operation == "focus":
                    self.focused = bool(value)
                    if self.x11 is not None:
                        self.x11.focus(self.focused)
                else:
                    self._key(str(value), operation != "up")
                    if operation == "tap":
                        self._key(str(value), False)
                self.events.append({"segment": segment_index, "poll_tick": tick,
                                    "operation": operation, "value": value})
            self.delivered.add(coordinate)
        if self.viewer is None:
            self.commands.update_pressed(set(self.held) if self.focused else set(),
                                         focused=self.focused)

    def after_poll(self, segment_index: int, tick: int) -> None:
        if (segment_index, tick) not in EVENTS:
            return
        snapshot = self.commands.snapshot()
        expected = 0.0
        if segment_index == 0 and 50 <= tick < 800:
            expected = .20 if tick < 400 else .25
        elif segment_index == 1 and 10 <= tick < 200:
            expected = .20
        passed = snapshot["raw_forward_mps"] == expected
        if (segment_index, tick) == (0, 1000):
            passed = passed and snapshot["reset_pending"]
        if (segment_index, tick) == (1, 200):
            passed = passed and snapshot["stopped"]
        row = {"segment": segment_index, "poll_tick": tick,
               "expected_requested_speed_mps": expected, "snapshot": snapshot,
               "passed": bool(passed)}
        self.checks.append(row)
        if not passed:
            raise RuntimeError("07 actual input check failed: " + str(row))

    def report(self) -> dict[str, Any]:
        return {"profile": self.profile,
                "transport": "owned-window XSendEvent" if self.x11 else "pure input callbacks",
                "events": self.events, "checks": self.checks,
                "human_usability_assessed": False,
                "complete": set(EVENTS) <= self.delivered,
                "passed": len(self.checks) == len(EVENTS)
                and all(row["passed"] for row in self.checks)}

    def close(self) -> None:
        if self.x11 is not None:
            self.x11.close()
