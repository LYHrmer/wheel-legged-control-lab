"""Pure, bounded keyboard intent for the latest straight-driving RL viewer.

GLFW integer codes are used without importing GLFW. This module issues no
physics command: the session's prepared-decision callback consumes its sampled
forward speed at the next control tick.
"""

from __future__ import annotations

from typing import Any


class LatestRLCommands:
    """Forward at one of two fixed speeds, stop, reset, or exit."""

    KEY_W = ord("W")
    KEY_1 = ord("1")
    KEY_2 = ord("2")
    KEY_R = ord("R")
    KEY_X = ord("X")
    KEY_C = ord("C")
    KEY_SPACE = 32
    KEY_ESCAPE = 256
    RELEASE = 0
    PRESS = 1
    REPEAT = 2
    key_codes = frozenset({KEY_W, KEY_1, KEY_2, KEY_R, KEY_X, KEY_C,
                           KEY_SPACE, KEY_ESCAPE})
    speeds_mps = (0.20, 0.25)

    def __init__(self) -> None:
        self._gear = 1
        self._held: set[int] = set()
        self._event_down: set[int] = set()
        self._focused = False
        self._blocked_until_w_release = False
        self._reset_pending = False
        self._stopped = False

    @property
    def gear(self) -> int:
        return self._gear

    @property
    def selected_speed_mps(self) -> float:
        return self.speeds_mps[self._gear - 1]

    @property
    def raw_forward_mps(self) -> float:
        if (not self._focused or self._stopped or self._blocked_until_w_release
                or self.KEY_W not in self._held):
            return 0.0
        return self.selected_speed_mps

    @property
    def stopped(self) -> bool:
        return self._stopped

    def select_initial_speed(self, speed_mps: float) -> None:
        """Apply the launcher's fixed initial gear before the first key poll."""
        if self._focused or self._held or self._event_down:
            raise RuntimeError("initial speed must be selected before keyboard input")
        if type(speed_mps) is not float or speed_mps not in self.speeds_mps:
            raise ValueError("initial speed must be exactly 0.20 or 0.25 m/s")
        self._gear = self.speeds_mps.index(speed_mps) + 1

    def handle_key_event(self, key: int, action: int) -> None:
        """Capture short press edges; system key repeats never duplicate them."""
        if type(key) is not int or type(action) is not int:
            raise TypeError("key and action must be integer GLFW codes")
        if action not in (self.RELEASE, self.PRESS, self.REPEAT):
            raise ValueError("unsupported GLFW key action")
        if key not in self.key_codes:
            return
        if action == self.RELEASE:
            self._event_down.discard(key)
            return
        if action != self.PRESS or key in self._event_down:
            return
        self._event_down.add(key)
        if key in (self.KEY_1, self.KEY_2):
            self._gear = 1 if key == self.KEY_1 else 2
        elif key == self.KEY_R:
            self._reset_pending = True
        elif key in (self.KEY_SPACE, self.KEY_X):
            self._blocked_until_w_release = True
        elif key == self.KEY_ESCAPE:
            self._stopped = True

    def update_pressed(self, keys: set[int], focused: bool = True) -> None:
        """Sample held W only from the focused window, never from stale events."""
        if type(keys) is not set or any(type(key) is not int for key in keys):
            raise TypeError("pressed keys must be a set of integer GLFW codes")
        if type(focused) is not bool:
            raise TypeError("focused must be bool")
        if not keys <= self.key_codes:
            raise ValueError("pressed set contains unsupported keys")
        if not focused:
            self._held.clear()
            self._event_down.clear()
            self._focused = False
            self._blocked_until_w_release = True
            self._reset_pending = False
            return
        self._held = set(keys)
        self._focused = True
        if self.KEY_W not in keys:
            self._blocked_until_w_release = False
        if self.KEY_ESCAPE in keys:
            self._stopped = True

    def consume_reset_request(self) -> bool:
        """Return a single pending R request; the runner owns physical reset."""
        result = self._reset_pending
        self._reset_pending = False
        return result

    def reset_for_new_segment(self) -> None:
        """Start stopped in first gear; require actual W release before drive."""
        self._gear = 1
        self._reset_pending = False
        self._blocked_until_w_release = True

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema": "d1-latest-rl-straight-keyboard-intent-v1",
            "gear": self._gear,
            "selected_speed_mps": self.selected_speed_mps,
            "held_forward": self.KEY_W in self._held,
            "focused": self._focused,
            "blocked_until_w_release": self._blocked_until_w_release,
            "reset_pending": self._reset_pending,
            "stopped": self._stopped,
            "raw_forward_mps": self.raw_forward_mps,
        }
