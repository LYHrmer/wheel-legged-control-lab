"""Four engine-free regressions for the 08-Q scripted input repair."""

from __future__ import annotations

import importlib.util
from pathlib import Path

WORK = Path(__file__).resolve().parent
REPO = Path("/home/lyh/wheel-legged-control-lab")


def load_exact(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


driver_module = load_exact("profile_driver_q08_pure", WORK / "profile_driver_q08.py")
controls = load_exact("published_latest_rl_controls_q08_pure",
                      REPO / "scripts/d1_latest_rl_controls.py")


def pair():
    command = controls.LatestRLCommands()
    return command, driver_module.LogicalProfileQDriver(command)


def poll(command, driver, segment: int, tick: int, physical: str = "none") -> None:
    driver.before_poll(segment, tick)
    if physical == "focused_empty":
        command.update_pressed(set(), focused=True)
    elif physical == "unfocused":
        command.update_pressed(set(), focused=False)
    elif physical == "physical_W":
        command.update_pressed({ord("W")}, focused=True)
    elif physical != "none":
        raise AssertionError("unknown fake physical poll")
    driver.after_poll(segment, tick)


def expected_prepared() -> list[float]:
    return [0.0] * 175 + [.20] * 101 + [.25] * 75 + [0.0] * 49


def test_headless_two_boundaries_and_one_tick_prepared_latency():
    command, driver = pair()
    observed = []
    for segment in (0, 1):
        if segment:
            command.reset_for_new_segment()
            driver.reset_for_new_segment()
        prepared = []
        for tick in range(401):
            if tick < 400:
                prepared.append(command.raw_forward_mps if tick >= 175 else 0.0)
            poll(command, driver, segment, tick)
        observed.append(prepared)
        if segment == 0:
            assert command.consume_reset_request() is True
            assert command.consume_reset_request() is False
        else:
            assert command.stopped is True
    assert observed == [expected_prepared(), expected_prepared()]
    assert len(driver.events) == 12
    assert driver.reassertions == []
    assert driver.report()["polls_checked"] == 802
    assert driver.report()["all_polls_checked"] is True
    assert driver.report()["passed"] is True


def test_focused_empty_physical_poll_reasserts_x_until_logical_release():
    command, driver = pair()
    prepared = []
    for tick in range(376):
        if tick < 375:
            prepared.append(command.raw_forward_mps if tick >= 175 else 0.0)
        poll(command, driver, 0, tick, physical="focused_empty")
        if 350 <= tick < 375:
            assert driver.stop_latched is True
            assert command.snapshot()["blocked_until_w_release"] is True
            assert command.raw_forward_mps == 0.0
    assert prepared == expected_prepared()[:375]
    assert [row["poll_tick"] for row in driver.reassertions] == list(range(351, 375))
    assert all(row["additional_operator_event"] is False for row in driver.reassertions)
    assert driver.stop_latched is False  # tick 375 is the real scripted W release
    assert command.snapshot()["blocked_until_w_release"] is False


def test_unfocused_and_physical_w_samples_preserve_scripted_timeline():
    command, driver = pair()
    for tick in range(401):
        physical = "unfocused" if tick % 2 == 0 else "physical_W"
        poll(command, driver, 0, tick, physical=physical)
        assert command.raw_forward_mps == driver.expected_requested_mps(tick)
    assert command.consume_reset_request() is True
    assert driver.report()["passed"] is True
    assert driver.report()["x11_or_xsendevent_used"] is False
    assert all(row["source"] == driver_module.SOURCE for row in driver.reassertions)


def test_mixed_polls_reassert_only_after_actual_block_clear_without_new_event():
    command, driver = pair()
    for tick in range(401):
        physical = {351: "focused_empty", 352: "unfocused",
                    353: "physical_W", 354: "focused_empty"}.get(tick, "none")
        poll(command, driver, 0, tick, physical=physical)
        if 350 <= tick < 375:
            assert command.raw_forward_mps == 0.0
    assert [row["poll_tick"] for row in driver.reassertions] == [351, 354]
    assert [row["poll_tick"] for row in driver.events] == [0, 50, 275, 350, 375, 400]
    assert all(row["additional_operator_event"] is False for row in driver.reassertions)
    assert command.consume_reset_request() is True
