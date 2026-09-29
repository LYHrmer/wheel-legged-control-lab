"""Pure scripted input restoration for the bounded 08-Q compact probe.

Only the public LatestRLCommands handle_key_event/update_pressed/snapshot API is
used. A viewer poll may sample no physical keys and clear the command's stop
block while the scripted W remains held. In that case a public X edge restores
the already requested stop. This is a harness state restoration, not a new
operator event. No engine, GLFW, keyboard transport, or pacing is imported.
"""

from __future__ import annotations

from typing import Any, ClassVar

SOURCE = "scripted_logical_compact08"
SCHEMA = "d1-full-drive-perf08q-logical-input-v1"


class LogicalProfileQDriver:
    KEY_W = ord("W")
    KEY_2 = ord("2")
    KEY_X = ord("X")
    KEY_R = ord("R")
    KEY_ESCAPE = 256
    RELEASE = 0
    PRESS = 1
    TERMINAL_TICK = 400
    SCHEDULE: ClassVar[dict[int, tuple[tuple[str, int], ...]]] = {
        0: (("release", KEY_W),),
        50: (("press", KEY_W),),
        275: (("tap", KEY_2),),
        350: (("tap", KEY_X),),
        375: (("release", KEY_W),),
    }

    def __init__(self, commands: Any) -> None:
        self.commands = commands
        self.held: set[int] = set()
        self.stop_latched = False
        self._pending: tuple[int, int] | None = None
        self.events: list[dict[str, Any]] = []
        self.reassertions: list[dict[str, Any]] = []
        self.checks: list[dict[str, Any]] = []

    def reset_for_new_segment(self) -> None:
        """Call after the command's own public reset_for_new_segment()."""
        self.held.clear()
        self.stop_latched = False
        self._pending = None

    @staticmethod
    def expected_requested_mps(tick: int) -> float:
        if tick < 50 or tick >= 350:
            return 0.0
        return 0.20 if tick < 275 else 0.25

    def before_poll(self, segment_index: int, tick: int) -> None:
        if (segment_index not in (0, 1) or not 0 <= tick <= self.TERMINAL_TICK
                or self._pending is not None):
            raise RuntimeError("08-Q logical poll order is invalid")
        self._pending = (segment_index, tick)

    def after_poll(self, segment_index: int, tick: int) -> None:
        if self._pending != (segment_index, tick):
            raise RuntimeError("08-Q injection did not follow its paired poll")
        self._pending = None
        physical_poll_snapshot = self.commands.snapshot()
        operations = (("tap", self.KEY_R if segment_index == 0 else self.KEY_ESCAPE),) \
            if tick == self.TERMINAL_TICK else self.SCHEDULE.get(tick, ())
        for operation, key in operations:
            if operation == "press":
                self.commands.handle_key_event(key, self.PRESS)
                self.held.add(key)
            elif operation == "release":
                self.commands.handle_key_event(key, self.RELEASE)
                self.held.discard(key)
                if key == self.KEY_W:
                    self.stop_latched = False
            else:
                self.commands.handle_key_event(key, self.PRESS)
                self.commands.handle_key_event(key, self.RELEASE)
                if key == self.KEY_X:
                    self.stop_latched = True
            self.events.append({"segment": segment_index, "poll_tick": tick,
                                "operation": operation, "key_code": key,
                                "source": SOURCE})

        # Physical focus/empty samples are not the scripted W-release event.
        # When no stop is latched, an empty focused sample clears a stale focus
        # block before restoring the logical held W. This only affects the
        # scripted probe, never the manual viewer's command handling.
        if self.KEY_W in self.held and not self.stop_latched:
            self.commands.update_pressed(self.held - {self.KEY_W}, focused=True)
        if (self.KEY_W in self.held and self.stop_latched
                and not self.commands.snapshot()["blocked_until_w_release"]):
            # A real viewer poll cleared the existing X stop while W is still
            # logically held. Reassert the same stop via the public key API.
            self.commands.handle_key_event(self.KEY_X, self.PRESS)
            self.commands.handle_key_event(self.KEY_X, self.RELEASE)
            self.reassertions.append({
                "segment": segment_index, "poll_tick": tick,
                "reason": "physical_poll_cleared_X_block_before_logical_W_release",
                "physical_poll_snapshot": physical_poll_snapshot,
                "source": SOURCE, "additional_operator_event": False,
            })
        self.commands.update_pressed(set(self.held), focused=True)
        snapshot = self.commands.snapshot()
        expected = self.expected_requested_mps(tick)
        passed = snapshot["raw_forward_mps"] == expected
        if tick == self.TERMINAL_TICK:
            passed = passed and (snapshot["reset_pending"] if segment_index == 0
                                 else snapshot["stopped"])
        row = {"segment": segment_index, "poll_tick": tick,
               "expected_requested_forward_mps": expected,
               "physical_poll_snapshot": physical_poll_snapshot,
               "held_forward_injected": self.KEY_W in self.held,
               "stop_request_latched": self.stop_latched,
               "operations": [operation for operation, _ in operations],
               "snapshot": snapshot, "passed": bool(passed)}
        self.checks.append(row)
        if not passed:
            raise RuntimeError("08-Q logical input check failed: " + str(row))

    def report(self) -> dict[str, Any]:
        expected_polls = 2 * (self.TERMINAL_TICK + 1)
        return {"schema": SCHEMA, "source": SOURCE,
                "transport": "pure published command callbacks",
                "x11_or_xsendevent_used": False,
                "extra_pacing_or_sleep_added": False,
                "events": self.events, "reassertions": self.reassertions,
                "checks": self.checks, "polls_checked": len(self.checks),
                "all_polls_checked": len(self.checks) == expected_polls,
                "passed": bool(self.checks) and all(row["passed"] for row in self.checks),
                "human_usability_assessed": False,
                "reassertions_are_not_operator_events": True,
                "input_timing": "poll(k) changes prepared command at k+1"}


# The compact worker uses this spelling; keep the short name for pure callers.
LogicalProfileDriverQ08 = LogicalProfileQDriver
