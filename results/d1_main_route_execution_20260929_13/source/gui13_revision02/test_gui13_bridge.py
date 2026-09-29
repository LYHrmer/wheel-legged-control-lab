"""Pure GUI13 ownership tests; no engine, model, or renderer import."""

from __future__ import annotations

import queue
import sys
import threading
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

W = Path(__file__).resolve().parents[2]
for folder in (Path(__file__).resolve().parent, W / "course_impl08", W / "gui12"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import gui13_bridge as bridge

EVENTS = {175: ("W_PRESS",), 425: ("W_RELEASE", "S_PRESS"),
          430: ("S_RELEASE",)}


class LogicalHandshakeTests(unittest.TestCase):
    def test_three_events_require_main_request_then_owner_ack_in_order(self) -> None:
        script = bridge.LogicalScript(threading.get_ident())
        requests: queue.Queue[int] = queue.Queue()
        acknowledgements: queue.Queue[tuple[int, dict, float]] = queue.Queue()
        errors: queue.Queue[BaseException] = queue.Queue()
        permits = {tick: threading.Event() for tick in EVENTS}

        def owner() -> None:
            try:
                for tick in range(431):
                    if tick in EVENTS:
                        requests.put(tick)
                        if not permits[tick].wait(timeout=3):
                            raise TimeoutError("main did not supply logical request")
                    command = script(tick, tick * 0.01)
                    if tick in EVENTS:
                        acknowledgements.put((tick, script.latest_ack(),
                                              command.forward_velocity_mps))
            except Exception as error:  # noqa: BLE001 - report worker failure on test thread
                errors.put(error)

        thread = threading.Thread(target=owner)
        thread.start()
        try:
            for tick, events in EVENTS.items():
                self.assertEqual(requests.get(timeout=3), tick)
                before = script.latest_ack()
                script.submit(tick, events)
                self.assertEqual(script.latest_ack(), before)
                permits[tick].set()
                actual_tick, ack, speed = acknowledgements.get(timeout=3)
                self.assertEqual(actual_tick, tick)
                self.assertEqual(ack["prepared_tick"], tick)
                self.assertEqual(ack["prior_completed_controls"], tick)
                self.assertEqual(ack["prior_completed_tick"], tick - 1)
                self.assertEqual(ack["events"], events)
                self.assertEqual(speed, 0.6 if tick == 175 else 0.0)
                self.assertLessEqual(ack["request_wall_ns"], ack["ack_wall_ns"])
            detached = script.latest_ack()
            detached["prepared_tick"] = -1
            self.assertEqual(script.latest_ack()["prepared_tick"], 430)
        finally:
            for permit in permits.values():
                permit.set()
            thread.join(timeout=3)
        self.assertFalse(thread.is_alive())
        self.assertTrue(errors.empty(), list(errors.queue))

    def test_owner_cannot_submit_its_own_request_or_ack_without_main(self) -> None:
        script = bridge.LogicalScript(threading.get_ident())
        with self.assertRaisesRegex(RuntimeError, "cannot run on main"):
            script(0, 0.0)
        results: queue.Queue[tuple[str, str]] = queue.Queue()

        def owner() -> None:
            for tick in range(175):
                script(tick, tick * 0.01)
            for label, call in (
                ("submit", lambda: script.submit(175, EVENTS[175])),
                ("consume", lambda: script(175, 1.75)),
            ):
                try:
                    call()
                except RuntimeError as error:
                    results.put((label, str(error)))

        thread = threading.Thread(target=owner)
        thread.start()
        thread.join(timeout=3)
        self.assertFalse(thread.is_alive())
        self.assertIn("main thread", results.get_nowait()[1])
        self.assertIn("lacks main-thread logical request", results.get_nowait()[1])
        self.assertIsNone(script.latest_ack())


class OwnerAndSnapshotTests(unittest.TestCase):
    def test_wrong_thread_latches_stop_without_mutating_native_phase(self) -> None:
        stop = threading.Event()

        class FakeRuntime:
            def __init__(self) -> None:
                self.owner_ident = threading.get_ident()
                self._violation_lock = threading.Lock()
                self.thread_violations: list[dict] = []
                self._fatal = None
                self.native_phase = 7

        fake_module = types.ModuleType("run_world_upright_gui_12")
        fake_module._make_thread_owned_runtime = lambda _stop: FakeRuntime
        with patch.dict(sys.modules, {"run_world_upright_gui_12": fake_module}):
            runtime_type = bridge.owner_runtime_type(stop)
        runtime = runtime_type()
        runtime._owner("control_step")
        errors: queue.Queue[str] = queue.Queue()

        def foreign() -> None:
            try:
                runtime._owner("control_step")
            except RuntimeError as error:
                errors.put(str(error))

        thread = threading.Thread(target=foreign)
        thread.start()
        thread.join(timeout=3)
        self.assertFalse(thread.is_alive())
        self.assertIn("non-owner", errors.get_nowait())
        self.assertTrue(stop.is_set())
        self.assertEqual(runtime.native_phase, 7)
        self.assertIsNone(runtime._fatal)
        self.assertEqual([item["entry"] for item in runtime.thread_violations],
                         ["control_step"])

    def test_headless_pause_reuses_only_the_exact_displayed_frame(self) -> None:
        self.assertTrue(bridge.pause_snapshot_satisfied(42, 42, None))
        self.assertFalse(bridge.pause_snapshot_satisfied(42, 41, None))
        matching = SimpleNamespace(metadata={"control_index": 42})
        stale = SimpleNamespace(metadata={"control_index": 41})
        self.assertTrue(bridge.pause_snapshot_satisfied(42, 41, matching))
        self.assertFalse(bridge.pause_snapshot_satisfied(42, 42, stale))

    def test_copy_fence_requires_current_mailbox_lease_and_matching_slot(self) -> None:
        slots = (object(), object(), object())
        display = object()
        plant = SimpleNamespace(model=object(), data=object(), measurement_data=object())
        mailbox = bridge.mailbox_type()(slots)
        copies: list[tuple[object, object]] = []
        mj = SimpleNamespace(mj_copyData=lambda destination, _model, source:
                             copies.append((destination, source)))
        stop = threading.Event()
        ready = threading.Event()
        proceed = threading.Event()
        owner_ident: list[int] = []
        owner_errors: queue.Queue[BaseException] = queue.Queue()

        def owner() -> None:
            owner_ident.append(threading.get_ident())
            ready.set()
            if not proceed.wait(timeout=3):
                owner_errors.put(TimeoutError("fence was not installed"))
                return
            try:
                mj.mj_copyData(plant.measurement_data, plant.model, plant.data)
                mailbox.publish(
                    lambda destination: mj.mj_copyData(destination, plant.model,
                                                       plant.measurement_data),
                    {"episode_id": 1, "control_index": 42, "sim_time_s": 0.42,
                     "source_wall_ns": 1},
                )
            except Exception as error:  # noqa: BLE001 - report worker failure on test thread
                owner_errors.put(error)

        thread = threading.Thread(target=owner)
        thread.start()
        self.assertTrue(ready.wait(timeout=3))
        _original, edges = bridge.fence_copy_data(
            mj, plant=plant, slots=slots, display=display, mailbox=mailbox,
            owner_ident=owner_ident[0], main_ident=threading.get_ident(), stop_event=stop)
        proceed.set()
        thread.join(timeout=3)
        self.assertFalse(thread.is_alive())
        self.assertTrue(owner_errors.empty(), list(owner_errors.queue))
        self.assertEqual(edges["live_to_measurement"], 1)
        self.assertEqual(edges["measurement_to_writing"], 1)

        lease = mailbox.acquire_latest()
        self.assertIsNotNone(lease)
        mj.mj_copyData(display, plant.model, lease.data)
        self.assertEqual(edges["reading_to_display"], 1)
        other_slot = next(slot for slot in slots if slot is not lease.data)
        before = len(copies)
        with self.assertRaisesRegex(RuntimeError, "exclusive snapshot fence"):
            mj.mj_copyData(display, plant.model, other_slot)
        lease.release()
        with self.assertRaisesRegex(RuntimeError, "exclusive snapshot fence"):
            mj.mj_copyData(display, plant.model, lease.data)
        self.assertEqual(len(copies), before)
        self.assertEqual(edges["rejected"], 2)
        self.assertTrue(stop.is_set())


if __name__ == "__main__":
    unittest.main()
