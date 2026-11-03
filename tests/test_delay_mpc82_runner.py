"""Admission, queue and archived-metric contracts without physical rollouts."""

from __future__ import annotations

import csv
import hashlib
import json

import pytest

from scripts.delay_mpc82 import (
    COMPARE_COLUMNS,
    CORE_COLUMNS,
    SETTINGS,
    RecordedController,
    assess,
    check_fifo_transition,
    exact_baseline_replay,
    metrics_from_csv,
    nominal_identity_checks,
    percentile95,
    verify_hashes,
    write_json,
)


def test_fifo_delays_whole_two_force_control_without_retiming():
    check_fifo_transition(
        [[0.0, 100.0], [2.0, 110.0]], [4.0, 120.0], [0.0, 100.0], [[2.0, 110.0], [4.0, 120.0]]
    )
    check_fifo_transition([], [4.0, 120.0], [4.0, 120.0], [])
    with pytest.raises(ValueError, match="transition"):
        check_fifo_transition(
            [[0.0, 100.0], [2.0, 110.0]], [4.0, 120.0], [4.0, 120.0], [[0.0, 100.0], [2.0, 110.0]]
        )
    with pytest.raises(ValueError, match="length"):
        check_fifo_transition([[0.0, 100.0]], [4.0, 120.0], [0.0, 100.0], [[4.0, 120.0]])


def write_csv(path, rows):
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CORE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def test_scalar_metric_formula_matches_frozen_numpy_evaluator(tmp_path):
    np = pytest.importorskip("numpy")
    pytest.importorskip("mujoco")
    pytest.importorskip("gymnasium")
    pytest.importorskip("matplotlib")
    from wheel_legged_control.experiments import Rollout, compute_metrics

    rows = []
    for i in range(4):
        rows.append(
            dict(
                zip(
                    CORE_COLUMNS,
                    [
                        0.02 * (i + 1),
                        0.0,
                        0.01 * (i - 1),
                        0.01 * i,
                        0.1 * i,
                        0.0,
                        0.0,
                        -2.0 + i,
                        100.0 + i,
                        0.25,
                        0.015,
                        0.0,
                        1.0 - i * 0.1,
                        0.2 + i * 0.01,
                    ],
                    strict=True,
                )
            )
        )
    path = tmp_path / "trace.csv"
    write_csv(path, rows)
    solves = [0.1, *[r["next_solve_ms"] for r in rows]]
    (metric, saved) = metrics_from_csv(path, solves)
    states = np.array(
        [
            [r[k] for k in ("x", "pitch", "extension", "vx", "pitch_rate", "extension_rate")]
            for r in rows
        ]
    )
    controls = np.array([[r["wheel_force"], r["leg_force"]] for r in rows])
    commands = np.array([[r["target_vx"], r["target_extension"]] for r in rows])
    rollout = Rollout(
        "baseline",
        "nominal",
        np.arange(4) * 0.02,
        states,
        controls,
        commands,
        np.zeros(4),
        np.array([r["reward"] for r in rows]),
        False,
        np.array(solves),
    )
    expected = compute_metrics(rollout)
    for key, value in expected.items():
        if key in metric:
            assert metric[key] == pytest.approx(value, abs=1e-12, rel=1e-12)
    assert saved == rows
    assert percentile95([1.0, 2.0, 3.0]) == pytest.approx(2.9)


def test_baseline_replay_ignores_only_solver_clock(tmp_path):
    rows = [
        dict(
            zip(
                CORE_COLUMNS,
                [
                    0.02 * (i + 1),
                    0.0,
                    0.01,
                    0.02,
                    0.1,
                    0.0,
                    0.0,
                    1.0,
                    100.0,
                    0.1,
                    0.02,
                    0.0,
                    1.0,
                    0.3,
                ],
                strict=True,
            )
        )
        for i in range(400)
    ]
    old = tmp_path / "old.csv"
    write_csv(old, rows)
    changed = [dict(r, next_solve_ms=10.0) for r in rows]
    assert exact_baseline_replay(changed, old)
    changed[2]["leg_force"] += 1e-09
    assert not exact_baseline_replay(changed, old)
    assert not exact_baseline_replay(rows[:-1], old)


def cohort():
    return [
        {
            "seed": s,
            "scenario": scenario,
            "profile": "benchmark",
            "arm": arm,
            "completed": True,
            "height_rmse_mm": 100.0 if arm == "baseline" else 89.0,
            "velocity_rmse_mps": 1.0,
            "pitch_rmse_deg": 1.0,
            "max_abs_pitch_deg": 1.0,
            "normalized_effort_rms": 1.0,
            "wheel_force_rms_n": 1.0,
            "leg_force_rms_n": 1.0,
            "solve_p95_ms": 1.0,
            "metric_crosscheck": {"passed": True},
            "baseline49_replay_exact": arm == "baseline",
            "queue_checks_completed": 400,
            "nominal_model_preserved": True,
        }
        for s in SETTINGS["preflight_seeds"]
        for scenario in SETTINGS["scenarios"]
        for arm in ("baseline", "candidate")
    ]


def nominal_receipt():
    return {
        "by_seed": [{"seed": seed, "exact": True} for seed in SETTINGS["preflight_seeds"]],
        "all_exact": True,
        "comparison_columns": list(COMPARE_COLUMNS),
        "required_rows_per_episode": 400,
    }


def test_all_three_concurrent_nominal_pairs_compare_every_core_value(tmp_path):
    rows = [dict.fromkeys(CORE_COLUMNS, 0.0) for _ in range(400)]
    for seed in SETTINGS["preflight_seeds"]:
        for arm in ("baseline", "candidate"):
            directory = tmp_path / f"{seed}_nominal_benchmark_{arm}"
            directory.mkdir()
            values = [dict(row, next_solve_ms=1.0 if arm == "candidate" else 2.0) for row in rows]
            if seed == SETTINGS["preflight_seeds"][1] and arm == "candidate":
                values[399]["reward"] = 1e-12
            write_csv(directory / "trajectory.csv", values)
    result = nominal_identity_checks(tmp_path)
    assert [row["exact"] for row in result["by_seed"]] == [True, False, True]
    assert result["all_exact"] is False
    assert result["comparison_columns"] == list(COMPARE_COLUMNS)
    assert result["excluded_core_columns"] == ["next_solve_ms"]
    assert all(
        len(row["baseline_sha256"]) == len(row["candidate_sha256"]) == 64
        for row in result["by_seed"]
    )


def test_original_height_effort_solver_and_complete_grid_gates_remain():
    pytest.importorskip("numpy")
    rows = cohort()
    identity = nominal_receipt()
    good = assess(rows, identity)
    assert good["accepted"] and good["full_status"] == "NOT_RUN"
    for changed in (rows[:-1], rows + [rows[0]], [dict(rows[0], completed=False), *rows[1:]]):
        assert not assess(changed, identity)["accepted"]
    for field, bad in [
        ("height_rmse_mm", 91.0),
        ("normalized_effort_rms", 1.06),
        ("solve_p95_ms", 21.0),
    ]:
        changed = [dict(r, **{field: bad}) if r["arm"] == "candidate" else r for r in rows]
        result = assess(changed, identity)
        assert not result["accepted"] and result["meaning"].startswith("REJECTED")
    changed = [dict(rows[0], baseline49_replay_exact=False), *rows[1:]]
    assert not assess(changed, identity)["accepted"]
    bad_identity = dict(
        identity, by_seed=[dict(identity["by_seed"][0], exact=False), *identity["by_seed"][1:]]
    )
    duplicate_identity = dict(identity, by_seed=[identity["by_seed"][0]] * 3)
    for receipt in (None, bad_identity, duplicate_identity, dict(identity, all_exact=False)):
        rejected = assess(rows, receipt)
        assert not rejected["accepted"]
        assert not rejected["integrity_gates"]["all_three_nominal_candidate_baseline_exact"]
        assert rejected["performance_gates"] == good["performance_gates"]


def test_source_fingerprints_and_exclusive_failure_evidence(tmp_path):
    source = tmp_path / "frozen.py"
    source.write_text("frozen\n")
    hashes = {"frozen.py": hashlib.sha256(source.read_bytes()).hexdigest()}
    verify_hashes(tmp_path, hashes)
    source.write_text("changed\n")
    with pytest.raises(ValueError, match="changed"):
        verify_hashes(tmp_path, hashes)
    path = tmp_path / "report.json"
    payload = {"status": "INCOMPLETE", "steps": 2, "physics_steps": 20, "failure": "injected"}
    write_json(path, payload)
    with pytest.raises(FileExistsError):
        write_json(path, {"status": "COMPLETE"})
    assert json.loads(path.read_text()) == payload


def test_recording_wrapper_never_requests_additional_compute_or_sensor_draw():

    class Fake:
        calls = 0
        last_solve_ms = 0.2

        def reset(self, state):
            self.calls = 0

        def compute(self, state, command):
            self.calls += 1
            return [state[0] + command, 100.0]

    original = Fake()
    wrapper = RecordedController(original)
    wrapper.reset([0.0, 0.0])
    state = [2.0, 3.0]
    assert wrapper.compute(state, 0.5) == [2.5, 100.0]
    state[0] = 999.0
    assert wrapper.last_measured_state == [2.0, 3.0]
    assert original.calls == wrapper.compute_calls == 1 and wrapper.last_solve_ms == 0.2


def test_nonfinite_partial_preserves_raw_prefix_and_both_report_levels(tmp_path):
    from scripts.delay_mpc82 import persist_episode_receipt, write_terminal_report

    episode = tmp_path / "episode"
    episode.mkdir()
    raw = episode / "trajectory.csv"
    raw.write_text("time_s,extension\n0.02,nan\n")
    before = raw.read_bytes()
    receipt = {
        "completed": False,
        "steps": 1,
        "physics_steps": 10,
        "failure": "original nonfinite state",
        "partial_final_state": [0.0, float("nan"), float("inf")],
        "partial_actual_control": [float("-inf"), 100.0],
    }
    persist_episode_receipt(episode, receipt)
    saved = json.loads((episode / "metrics.json").read_text())
    assert saved == receipt and saved["status"] == "INCOMPLETE" and (saved["completed"] is False)
    assert saved["partial_final_state"] == [0.0, "NaN", "+Infinity"]
    assert saved["partial_actual_control"] == ["-Infinity", 100.0]
    assert len(saved["nonfinite_paths"]) == 3
    assert saved["failure"] == "original nonfinite state" and raw.read_bytes() == before
    report = write_terminal_report(
        tmp_path,
        {
            "status": "INCOMPLETE",
            "episodes": [receipt],
            "decision": {"accepted": False},
            "failure": "original nonfinite state",
        },
        1 << 20,
    )
    assert report["status"] == "INCOMPLETE"
    assert json.loads((tmp_path / "report.json").read_text())["episodes"][0]["nonfinite_paths"]
    assert raw.read_bytes() == before


@pytest.mark.parametrize("kind", ["serialization", "size"])
def test_final_report_fallback_is_small_exclusive_and_retains_original_failure(tmp_path, kind):
    from scripts.delay_mpc82 import write_terminal_report

    raw = tmp_path / "trajectory.csv"
    raw.write_text("preserved prefix\n")
    before = raw.read_bytes()
    report = {
        "status": "COMPLETE",
        "verification_after": "COMPLETE",
        "decision": {"accepted": True},
        "failure": "original failure",
        "episodes": [],
        "control_steps": 2,
        "integration_steps": 20,
    }
    report["unsupported_or_large"] = object() if kind == "serialization" else "x" * 5000
    cap = 1200
    result = write_terminal_report(tmp_path, report, cap)
    saved = json.loads((tmp_path / "report.json").read_text())
    assert (
        result == saved
        and saved["status"] == "INCOMPLETE"
        and (saved["decision"]["accepted"] is False)
    )
    assert saved["failure"] == "original failure" and "finalization_failure" in saved
    assert sum(p.stat().st_size for p in tmp_path.iterdir()) <= cap
    assert raw.read_bytes() == before
    with pytest.raises(FileExistsError):
        write_terminal_report(tmp_path, report, cap)


def test_nested_nonfinite_cannot_remain_a_complete_acceptance(tmp_path):
    from scripts.delay_mpc82 import write_terminal_report

    result = write_terminal_report(
        tmp_path,
        {
            "status": "COMPLETE",
            "decision": {"accepted": True},
            "episodes": [{"partial": float("nan")}],
        },
        1024,
    )
    assert result["status"] == "INCOMPLETE" and result["decision"]["accepted"] is False
    assert result["nonfinite_paths"] == ['$["episodes"][0]["partial"]']
