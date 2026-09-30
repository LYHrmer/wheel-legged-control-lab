"""Zero-model, zero-physics owner close tests for the C26 absolute join seam."""

import threading
import time

import pytest

from run_gui26 import join_owner_until_close


def test_long_durable_close_finishes_before_serialization():
    closed = threading.Event()

    def owner():
        time.sleep(8.2)
        closed.set()

    thread = threading.Thread(target=owner, daemon=False)
    preflight_start = time.monotonic()
    thread.start()
    join = join_owner_until_close(thread, preflight_start + 10.0)
    serialized_after_join = closed.is_set() and not thread.is_alive()
    assert serialized_after_join
    assert join['owner_alive_after'] is False
    assert join['join_end_monotonic_s'] <= join['deadline_monotonic_s']
    assert join['join_end_monotonic_s'] - join['join_start_monotonic_s'] > 8.0


def test_short_absolute_deadline_refuses_live_owner_and_no_serialization():
    release = threading.Event()
    thread = threading.Thread(target=lambda: release.wait(timeout=2), daemon=False)
    thread.start()
    serialized = False
    try:
        with pytest.raises(TimeoutError, match='absolute host deadline'):
            join_owner_until_close(thread, time.monotonic() + .03)
        assert thread.is_alive()
        assert not serialized
    finally:
        release.set()
        thread.join(timeout=2)
    assert not thread.is_alive()
