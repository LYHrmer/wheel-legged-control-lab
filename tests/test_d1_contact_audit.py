from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from wheel_legged_control.d1.contact_audit import (
    AUDIT_ARTIFACTS,
    AUDIT_METRICS,
    PREREGISTERED_HELDOUT_SEEDS,
    build_parser,
    run_contact_allocation_audit,
)
from wheel_legged_control.d1.experiments import D1Rollout


def _fake_rollout(seed: int, contact_allocation: str) -> D1Rollout:
    constrained = contact_allocation == "constrained"
    rollout = D1Rollout(
        controller="D1 LQR+VMC",
        scenario="randomized",
        time_s=np.asarray([0.0, 0.01]),
        states=np.zeros((2, 6), dtype=np.float64),
        forward_velocities_mps=np.zeros(2, dtype=np.float64),
        torques=np.zeros((2, 16), dtype=np.float64),
        commands=np.asarray([[0.0, 0.32], [0.0, 0.32]], dtype=np.float64),
        pushes=np.zeros((2, 3), dtype=np.float64),
        rewards=np.ones(2, dtype=np.float64),
        wheel_contacts=np.asarray([4, 3]),
        undesired_contacts=np.zeros(2, dtype=np.bool_),
        solve_times_ms=np.asarray([0.1, 0.2]),
        terminated=False,
        evaluation_seed=seed,
        domain={
            "base_mass_scale": 0.9 + seed * 1e-6,
            "damping_scale": 1.1,
            "friction_scale": 0.8,
            "actuator_strength_scale": 0.95,
        },
        contact_allocation=contact_allocation,
        allocation_statuses=np.asarray(
            ["converged", "converged"] if constrained else ["legacy", "legacy"],
            dtype="U32",
        ),
        allocation_wrench_tracking_statuses=np.asarray(["tracked", "tracked"]),
        allocation_solve_times_ms=np.asarray(
            [0.1, 0.2] if constrained else [0.0, 0.0]
        ),
        allocation_constraint_violations=np.zeros(2),
        allocation_force_error_norms_n=np.asarray([1.0, 2.0] if constrained else [3.0, 4.0]),
        allocation_moment_error_norms_nm=np.asarray(
            [0.1, 0.2] if constrained else [0.3, 0.4]
        ),
        contact_force_model_error_norms_n=np.asarray(
            [4.0, 4.0] if constrained else [5.0, 5.0]
        ),
        contact_moment_model_error_norms_nm=np.asarray(
            [0.4, 0.4] if constrained else [0.5, 0.5]
        ),
        contact_force_tracking_error_norms_n=np.asarray(
            [3.0, 4.0] if constrained else [6.0, 8.0]
        ),
        contact_moment_tracking_error_norms_nm=np.asarray(
            [0.3, 0.4] if constrained else [0.6, 0.8]
        ),
        contact_wrench_physics_samples=np.asarray([5, 5]),
        initial_state_fingerprint=f"state-{seed}",
        initial_command_fingerprint=f"command-{seed}",
        push_schedule_fingerprint=f"push-{seed}",
    )
    # Repeat the two numerical samples into a completed six-second episode.
    for descriptor in fields(rollout):
        values = getattr(rollout, descriptor.name)
        if isinstance(values, np.ndarray) and values.shape[0] == 2:
            repeats = (300,) + (1,) * (values.ndim - 1)
            setattr(rollout, descriptor.name, np.tile(values, repeats))
    rollout.time_s = np.arange(600) * rollout.control_dt_s
    return rollout


def _runner(calls: list[dict[str, Any]]):
    def run(
        baseline: str,
        scenario: str,
        seed: int,
        policy: object | None = None,
        **kwargs: Any,
    ) -> D1Rollout:
        del policy
        calls.append({"baseline": baseline, "scenario": scenario, "seed": seed, **kwargs})
        return _fake_rollout(seed, str(kwargs["contact_allocation"]))

    return run


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_audit_writes_matched_evidence_and_honest_exploratory_result(
    tmp_path: Path,
) -> None:
    calls: list[dict[str, Any]] = []

    records = run_contact_allocation_audit(
        tmp_path,
        seed=41,
        episodes=2,
        rollout_runner=_runner(calls),
        run_metadata={"test_fixture": "deterministic"},
    )

    assert len(records) == 4
    assert {(call["seed"], call["contact_allocation"]) for call in calls} == {
        (41, "legacy"),
        (41, "constrained"),
        (42, "legacy"),
        (42, "constrained"),
    }
    assert all(call["baseline"] == "lqr" for call in calls)
    assert all(call["scenario"] == "randomized" for call in calls)
    assert all(call["episode_options"]["sensor_noise"] == 0.0 for call in calls)

    assert {path.name for path in tmp_path.iterdir()} == set(AUDIT_ARTIFACTS)
    config = _json(tmp_path / "evaluation_config.json")
    assert config["protocol"]["episodes"] == 2
    assert config["protocol"]["exploratory"] is True
    assert config["protocol"]["seed_pool"] == "development"
    assert config["protocol"]["preregistered_held_out_seeds"] == list(
        PREREGISTERED_HELDOUT_SEEDS
    )
    allocator_config = config["protocol"]["constrained_allocator"]
    assert allocator_config["slsqp_ftol"] == 1e-10
    assert allocator_config["candidate_constraint_violation_ratio_tolerance"] == 1e-7
    assert allocator_config["force_tracking_absolute_tolerance_n"] == 1.0
    assert allocator_config["moment_tracking_absolute_tolerance_nm"] == 0.5
    assert allocator_config["wrench_tracking_relative_tolerance"] == 5e-3
    assert config["promotion_gate"]["formal_min_episodes"] == 30
    assert config["promotion_gate"]["max_solver_degraded_ratio"] == 0.01

    manifest = _json(tmp_path / "contact_allocation_manifest.json")
    assert manifest["status"] == "complete"
    assert manifest["validation"]["matched_input_fingerprints"] is True
    assert manifest["validation"]["episode_rows"] == 4
    assert manifest["promotion"]["passed"] is False
    assert "preregistered" in " ".join(manifest["promotion"]["failed_checks"]).lower()
    assert "30" in " ".join(manifest["promotion"]["failed_checks"])
    for name, expected_digest in manifest["artifact_sha256"].items():
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == expected_digest

    with (tmp_path / "contact_allocation_summary.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        summary = {row["metric"]: row for row in csv.DictReader(handle)}
    assert set(summary) == set(AUDIT_METRICS)
    assert float(summary["allocation_force_error_rms_n"]["legacy_mean"]) == pytest.approx(
        np.sqrt(12.5)
    )
    assert float(summary["allocation_moment_error_rms_nm"]["constrained_mean"]) == (
        pytest.approx(np.sqrt(0.025))
    )
    assert float(summary["partial_contact_ratio"]["legacy_mean"]) == pytest.approx(0.5)
    assert summary["allocation_constraint_violation_max"]["unit"] == "ratio"
    assert AUDIT_METRICS["allocation_constraint_violation_max"][1] == "ratio"

    report = (tmp_path / "contact_allocation_audit.md").read_text(encoding="utf-8")
    assert "Exploratory" in report
    assert "not eligible" in report
    assert "development seed pool" in report
    assert "MuJoCo solver contact wrench" in report
    assert "N and Nm" in report
    assert "only changed condition" not in report
    assert "not an allocator-only causal ablation" in report
    assert "identified sagittal" in report
    assert "outer-loop LQR weight R" in report
    assert (tmp_path / "contact_allocation_audit.png").stat().st_size > 0


@pytest.mark.parametrize(
    "mismatch",
    ["initial_state_fingerprint", "initial_command_fingerprint", "push_schedule_fingerprint", "domain"],
)
def test_audit_rejects_every_unmatched_input_fingerprint(
    tmp_path: Path,
    mismatch: str,
) -> None:
    stale_manifest = tmp_path / "contact_allocation_manifest.json"
    stale_manifest.write_text("stale", encoding="utf-8")

    def mismatched_runner(
        baseline: str,
        scenario: str,
        seed: int,
        policy: object | None = None,
        **kwargs: Any,
    ) -> D1Rollout:
        del baseline, scenario, policy
        rollout = _fake_rollout(seed, str(kwargs["contact_allocation"]))
        if rollout.contact_allocation == "constrained":
            if mismatch == "domain":
                rollout.domain["friction_scale"] += 0.01
            else:
                setattr(rollout, mismatch, "unmatched")
        return rollout

    with pytest.raises(ValueError, match=mismatch.replace("_fingerprint", "")):
        run_contact_allocation_audit(
            tmp_path,
            seed=7,
            episodes=1,
            rollout_runner=mismatched_runner,
            run_metadata={"test_fixture": "mismatch"},
        )

    assert not stale_manifest.exists()


def test_formal_run_can_pass_only_the_preregistered_held_out_gate(tmp_path: Path) -> None:
    manifest_path = tmp_path / "contact_allocation_manifest.json"

    run_contact_allocation_audit(
        tmp_path,
        seed=PREREGISTERED_HELDOUT_SEEDS[0],
        episodes=30,
        rollout_runner=_runner([]),
        run_metadata={"test_fixture": "formal"},
    )

    manifest = _json(manifest_path)
    assert manifest["protocol"]["exploratory"] is False
    assert manifest["protocol"]["seed_pool"] == "held_out"
    assert manifest["protocol"]["evaluation_seeds"] == list(PREREGISTERED_HELDOUT_SEEDS)
    assert manifest["promotion"]["passed"] is True
    assert manifest["promotion"]["failed_checks"] == []


@pytest.mark.parametrize(
    ("attribute", "replacement"),
    [
        ("controller", "D1 MPC+VMC"),
        ("scenario", "nominal"),
        ("sensor_noise_scale", 1.0),
        ("allocation_statuses", np.empty(0, dtype=str)),
        ("allocation_statuses", np.full(600, "unknown")),
        ("allocation_wrench_tracking_statuses", np.empty(0, dtype=str)),
        ("contact_force_tracking_error_norms_n", np.empty(0)),
        ("allocation_constraint_violations", np.full(600, np.nan)),
        ("allocation_solve_times_ms", np.full(600, np.nan)),
        ("allocation_solve_times_ms", np.full(600, -1.0)),
        ("time_s", np.asarray([0.0, 0.01])),
    ],
)
def test_audit_rejects_missing_invalid_or_mislabeled_evidence(
    tmp_path: Path,
    attribute: str,
    replacement: Any,
) -> None:
    def invalid_runner(
        baseline: str,
        scenario: str,
        seed: int,
        policy: object | None = None,
        **kwargs: Any,
    ) -> D1Rollout:
        del baseline, scenario, policy
        rollout = _fake_rollout(seed, str(kwargs["contact_allocation"]))
        if rollout.contact_allocation == "constrained" and seed == 122:
            setattr(rollout, attribute, replacement)
        return rollout

    with pytest.raises(ValueError):
        run_contact_allocation_audit(
            tmp_path,
            seed=121,
            episodes=30,
            rollout_runner=invalid_runner,
            run_metadata={"test_fixture": "invalid-evidence"},
        )
    assert not (tmp_path / "contact_allocation_manifest.json").exists()


def test_formal_gate_does_not_hide_feasible_nonconvergence_as_success(
    tmp_path: Path,
) -> None:
    def degraded_runner(
        baseline: str,
        scenario: str,
        seed: int,
        policy: object | None = None,
        **kwargs: Any,
    ) -> D1Rollout:
        del baseline, scenario, policy
        rollout = _fake_rollout(seed, str(kwargs["contact_allocation"]))
        if rollout.contact_allocation == "constrained":
            rollout.allocation_statuses[:] = "feasible_nonconverged"
        return rollout

    run_contact_allocation_audit(
        tmp_path,
        seed=PREREGISTERED_HELDOUT_SEEDS[0],
        episodes=30,
        rollout_runner=degraded_runner,
        run_metadata={"test_fixture": "solver-degraded"},
    )

    promotion = _json(tmp_path / "contact_allocation_manifest.json")["promotion"]
    assert promotion["passed"] is False
    assert any(
        check.startswith("solver_degraded_rate:")
        for check in promotion["failed_checks"]
    )


def test_seed_21_can_never_be_marked_formal_or_pass_even_with_30_episodes(
    tmp_path: Path,
) -> None:
    """Seed 21 (the CLI's exploratory default) stays a development run.

    Running it with the formal episode count must not smuggle it past the
    preregistered held-out-seed gate: only seeds 121..150 are eligible for
    formal/pass status.
    """

    manifest_path = tmp_path / "contact_allocation_manifest.json"

    run_contact_allocation_audit(
        tmp_path,
        seed=21,
        episodes=30,
        rollout_runner=_runner([]),
        run_metadata={"test_fixture": "seed-21-full-episode-count"},
    )

    manifest = _json(manifest_path)
    assert manifest["protocol"]["seed_pool"] == "development"
    assert manifest["protocol"]["exploratory"] is True
    assert manifest["promotion"]["passed"] is False
    assert any(
        "preregistered" in check for check in manifest["promotion"]["failed_checks"]
    )

    report = (tmp_path / "contact_allocation_audit.md").read_text(encoding="utf-8")
    assert "Formal" not in report.splitlines()[2]
    assert "development seed pool" in report


def test_cli_defaults_to_a_small_exploratory_run() -> None:
    args = build_parser().parse_args([])

    assert args.episodes == 3
    assert args.seed == 21
    assert args.output == Path("results/d1_contact_allocation")
