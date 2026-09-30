"""Pure logical script and fenced owner-to-snapshot copy edges."""
from __future__ import annotations

import math
import threading
import time
from typing import Any

from gui13_contract import HORIZON, command_speed


class LogicalScript:
    """Synthetic event script; it is not a claim about keyboard latency."""

    def __init__(self, main_ident: int) -> None:
        if type(main_ident) is not int or main_ident <= 0:
            raise ValueError("GUI13 main thread identity must be positive")
        self.main_ident = main_ident
        self.acks: list[dict] = []
        self._last_tick = -1
        self._owner_ident: int | None = None
        self._lock = threading.Lock()
        self._pending: dict | None = None

    def submit(self, prepared_tick: int, events: tuple[str, ...]) -> None:
        if threading.get_ident() != self.main_ident:
            raise RuntimeError("GUI13 logical request must come from main thread")
        expected = {175: ("W_PRESS",), 425: ("W_RELEASE", "S_PRESS"),
                    430: ("S_RELEASE",)}
        if expected.get(prepared_tick) != events:
            raise ValueError("GUI13 logical request differs from fixed script")
        with self._lock:
            if self._pending is not None:
                raise RuntimeError("GUI13 previous logical request was not consumed")
            self._pending = {"prepared_tick": prepared_tick, "events": events,
                             "request_wall_ns": time.monotonic_ns(),
                             "source": "main_thread_logical_script"}

    def latest_ack(self) -> dict | None:
        with self._lock:
            return None if not self.acks else dict(self.acks[-1])

    def __call__(self, tick: int, time_s: float):
        from full_drive_command_08 import FullDriveCommand

        ident = threading.get_ident()
        if ident == self.main_ident:
            raise RuntimeError("GUI13 logical command cannot run on main thread")
        if self._owner_ident is None:
            self._owner_ident = ident
        elif self._owner_ident != ident:
            raise RuntimeError("GUI13 logical command owner changed")
        if (type(tick) is not int or tick != self._last_tick + 1
                or not 0 <= tick < HORIZON
                or not math.isclose(time_s, tick * 0.01, rel_tol=0.0, abs_tol=1e-10)):
            raise RuntimeError("GUI13 logical script tick/time/order differs")
        self._last_tick = tick
        events = {175: ("W_PRESS",), 425: ("W_RELEASE", "S_PRESS"),
                  430: ("S_RELEASE",)}.get(tick)
        speed = command_speed(tick)
        if events:
            with self._lock:
                request = self._pending
                if request is None or request["prepared_tick"] != tick or request["events"] != events:
                    raise RuntimeError("GUI13 prepared command lacks main-thread logical request")
                self._pending = None
                self.acks.append({**request, "prior_completed_controls": tick,
                                  "prior_completed_tick": tick - 1,
                                  "prepared_raw_vx_mps": speed,
                                  "ack_wall_ns": time.monotonic_ns(),
                                  "source": "logical_script_not_hardware_keyboard"})
        return FullDriveCommand(speed, 0.0, 0.0, 0.455, False)


def owner_runtime_type(stop_event: threading.Event):
    from run_world_upright_gui_12 import _make_thread_owned_runtime

    base = _make_thread_owned_runtime(stop_event)

    class OwnerRuntime(base):
        def _owner(self, entry: str) -> None:
            actual = threading.get_ident()
            if actual == self.owner_ident:
                return
            with self._violation_lock:
                self.thread_violations.append({
                    "entry": entry, "expected": self.owner_ident,
                    "actual": actual, "wall_ns": time.monotonic_ns()})
            # A foreign thread only latches a thread-safe event. The owner
            # records the fatal state at the next complete tick boundary.
            stop_event.set()
            raise RuntimeError("non-owner engine runtime call: " + entry)

    return OwnerRuntime


def mailbox_type():
    from latest_frame_mailbox_12 import LatestFrameMailbox

    class GuardedMailbox(LatestFrameMailbox):
        def is_writing_buffer(self, buffer: Any) -> bool:
            with self._lock:
                return bool(self._producer_ident == threading.get_ident()
                            and any(slot.buffer is buffer and slot.state == "WRITING"
                                    for slot in self._slots))

    return GuardedMailbox


def pause_snapshot_satisfied(expected_index: int, last_display_index: int,
                             lease: Any | None) -> bool:
    """A consumed pause frame is acceptable only if display already has it."""
    if type(expected_index) is not int or expected_index < 0:
        raise ValueError("pause expected index must be nonnegative")
    if lease is None:
        return last_display_index == expected_index
    return lease.metadata["control_index"] == expected_index


def fence_copy_data(mj: Any, *, plant: Any, slots: tuple[Any, ...],
                    display: Any, mailbox: Any, owner_ident: int,
                    main_ident: int, stop_event: threading.Event):
    original = mj.mj_copyData
    if len({id(item) for item in (*slots, display, plant.data, plant.measurement_data)}) != 6:
        raise RuntimeError("GUI13 copy buffers are not six distinct MjData objects")
    edges = {"live_to_measurement": 0, "measurement_to_writing": 0,
             "reading_to_display": 0, "rejected": 0}

    def checked(destination, model, source):
        ident = threading.get_ident()
        if model is not plant.model:
            edge = None
        elif (ident == owner_ident and destination is plant.measurement_data
              and source is plant.data):
            edge = "live_to_measurement"
        elif (ident == owner_ident and source is plant.measurement_data
              and mailbox.is_writing_buffer(destination)):
            edge = "measurement_to_writing"
        elif (ident == main_ident and destination is display
              and mailbox.is_reading_buffer(source)):
            edge = "reading_to_display"
        else:
            edge = None
        if edge is None:
            edges["rejected"] += 1
            stop_event.set()
            raise RuntimeError("GUI13 mj_copyData crossed exclusive snapshot fence")
        edges[edge] += 1
        return original(destination, model, source)

    mj.mj_copyData = checked
    return original, edges
