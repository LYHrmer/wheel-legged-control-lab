"""Matched-seed audit for the D1 legacy and constrained contact allocators."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from collections.abc import Callable, Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t

from ..provenance import capture_git_provenance
from .contact_allocation import (
    D1_CANDIDATE_CONSTRAINT_VIOLATION_TOL,
    D1_CONSTRAINED_FRICTION_COEFFICIENT,
    D1_FORCE_TRACKING_ABS_TOL_N,
    D1_MOMENT_TRACKING_ABS_TOL_NM,
    D1_NORMAL_FORCE_LIMIT_WEIGHT_FRACTION,
    D1_SLSQP_FTOL,
    D1_SLSQP_MAX_ITERATIONS,
    D1_WRENCH_CHARACTERISTIC_LENGTH_M,
    D1_WRENCH_TRACKING_REL_TOL,
)
from .experiments import D1Rollout, compute_d1_metrics, run_d1_rollout

plt.switch_backend("Agg")

AUDIT_ARTIFACTS = (
    "evaluation_config.json",
    "contact_allocation_episodes.csv",
    "contact_allocation_summary.csv",
    "contact_allocation_audit.md",
    "contact_allocation_audit.png",
    "contact_allocation_manifest.json",
)

# Force and moment errors deliberately remain separate: adding N and Nm would
# produce a number with no physical meaning.
AUDIT_METRICS: dict[str, tuple[str, str, str]] = {
    "success": ("Episode success", "ratio", "higher"),
    "velocity_rmse_mps": ("Velocity RMSE", "m/s", "lower"),
    "pitch_rmse_deg": ("Pitch RMSE", "deg", "lower"),
    "max_abs_pitch_deg": ("Maximum absolute pitch", "deg", "lower"),
    "height_rmse_mm": ("Height RMSE", "mm", "lower"),
    "normalized_torque_rms": ("Normalized torque RMS", "ratio", "lower"),
    "torque_saturation_ratio": ("Torque saturation", "ratio", "lower"),
    "four_wheel_contact_ratio": ("Four-wheel contact", "ratio", "higher"),
    "partial_contact_ratio": ("Partial wheel contact", "ratio", "lower"),
    "undesired_contact_steps": ("Undesired-contact steps", "steps", "lower"),
    "mean_abs_mechanical_power_w": ("Mean absolute mechanical power", "W", "lower"),
    "allocation_solve_p99_ms": ("Allocation solve-time P99", "ms", "lower"),
    "allocation_constraint_violation_max": (
        "Allocation maximum constraint violation",
        "ratio",
        "lower",
    ),
    "allocation_force_error_rms_n": ("Requested-force residual RMS", "N", "lower"),
    "allocation_moment_error_rms_nm": (
        "Requested-moment residual RMS",
        "Nm",
        "lower",
    ),
    "contact_force_model_error_rms_n": (
        "Allocated-to-physics force discrepancy RMS",
        "N",
        "lower",
    ),
    "contact_moment_model_error_rms_nm": (
        "Allocated-to-physics moment discrepancy RMS",
        "Nm",
        "lower",
    ),
    "contact_force_tracking_error_rms_n": (
        "Requested-to-physics force error RMS",
        "N",
        "lower",
    ),
    "contact_moment_tracking_error_rms_nm": (
        "Requested-to-physics moment error RMS",
        "Nm",
        "lower",
    ),
    "allocation_converged_ratio": ("Allocation solver converged", "ratio", "higher"),
    "allocation_feasible_nonconverged_ratio": (
        "Allocation feasible but nonconverged",
        "ratio",
        "lower",
    ),
    "allocation_fallback_ratio": ("Allocation fallback", "ratio", "lower"),
    "allocation_wrench_limited_ratio": (
        "Allocation wrench limited",
        "ratio",
        "lower",
    ),
    "allocation_no_contact_ratio": ("Allocation no-contact", "ratio", "lower"),
}

PROMOTION_GATE: dict[str, float | int | bool] = {
    "formal_min_episodes": 30,
    "success_noninferiority_margin": 0.0,
    "min_force_tracking_reduction_ratio": 0.10,
    "min_moment_tracking_reduction_ratio": 0.10,
    "max_constraint_violation": 1e-6,
    "max_solver_degraded_ratio": 0.01,
    "max_fallback_ratio": 0.01,
    "max_solve_time_fraction_of_control_period": 1.0,
    "required_physics_samples_per_control_step": 5,
}

# Formal promotion is preregistered against a fixed, held-out seed pool so the
# gate cannot be satisfied by seed-shopping. The default CLI seed (21, 3
# episodes) and any other seed range are the "development" pool: useful for
# iterating on the allocator, but never eligible for formal/pass status, even
# if run with the formal episode count.
PREREGISTERED_HELDOUT_SEEDS: tuple[int, ...] = tuple(
    range(121, 121 + int(PROMOTION_GATE["formal_min_episodes"]))
)

RolloutRunner = Callable[..., D1Rollout]
AUDIT_CONTROL_DT_S = 0.01
AUDIT_EPISODE_SECONDS = 6.0


def _capture_run_metadata() -> dict[str, Any]:
    dependencies: dict[str, str | None] = {}
    for package in ("numpy", "scipy", "mujoco", "gymnasium", "matplotlib"):
        try:
            dependencies[package] = version(package)
        except PackageNotFoundError:
            dependencies[package] = None
    return {
        "source": capture_git_provenance(Path(__file__).resolve().parent),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "dependencies": dependencies,
        },
    }


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_text_atomically(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_atomically(path: Path, payload: Mapping[str, Any]) -> None:
    _write_text_atomically(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _write_csv_atomically(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _prepare_output(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for filename in AUDIT_ARTIFACTS:
        (output / filename).unlink(missing_ok=True)
        (output / f".{filename}.tmp").unlink(missing_ok=True)


def _validate_rollout(rollout: D1Rollout, mode: str) -> None:
    expected_conditions = {
        "controller": "D1 LQR+VMC",
        "scenario": "randomized",
        "state_estimation_mode": "oracle",
        "latency_compensation": "none",
        "action_delay_steps": 0,
        "state_delay_steps": 0,
        "sensor_noise_scale": 0.0,
        "control_dt_s": AUDIT_CONTROL_DT_S,
    }
    for attribute, expected in expected_conditions.items():
        if getattr(rollout, attribute) != expected:
            raise ValueError(f"{mode} audit condition {attribute} must equal {expected!r}")

    steps = len(rollout.time_s)
    full_steps = round(AUDIT_EPISODE_SECONDS / AUDIT_CONTROL_DT_S)
    if not 0 < steps <= full_steps or (not rollout.terminated and steps != full_steps):
        raise ValueError(f"{mode} completed episode must contain {full_steps} control steps")
    expected_time = np.arange(steps) * AUDIT_CONTROL_DT_S
    if not np.allclose(rollout.time_s, expected_time, atol=1e-12, rtol=0.0):
        raise ValueError(f"{mode} time_s must match the control-step sequence")

    # Metrics have compatibility defaults for old rollouts. A new audit must
    # never interpret missing measurements as perfect tracking or zero runtime.
    diagnostics = (
        "allocation_solve_times_ms",
        "allocation_constraint_violations",
        "allocation_force_error_norms_n",
        "allocation_moment_error_norms_nm",
        "contact_force_model_error_norms_n",
        "contact_moment_model_error_norms_nm",
        "contact_force_tracking_error_norms_n",
        "contact_moment_tracking_error_norms_nm",
        "contact_wrench_physics_samples",
    )
    for attribute in diagnostics:
        values = np.asarray(getattr(rollout, attribute))
        if values.shape != (steps,) or not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError(f"{mode} {attribute} must be complete, finite and non-negative")
    permitted_statuses = (
        {"legacy"}
        if mode == "legacy"
        else {"converged", "feasible_nonconverged", "fallback", "no_contact"}
    )
    for attribute, permitted in (
        ("allocation_statuses", permitted_statuses),
        ("allocation_wrench_tracking_statuses", {"tracked", "limited"}),
    ):
        values = np.asarray(getattr(rollout, attribute))
        if values.shape != (steps,) or not set(values) <= permitted:
            raise ValueError(f"{mode} {attribute} contains missing or invalid statuses")


def _validate_pair(
    seed: int,
    legacy: D1Rollout,
    constrained: D1Rollout,
) -> dict[str, Any]:
    for mode, rollout in (("legacy", legacy), ("constrained", constrained)):
        if rollout.evaluation_seed != seed:
            raise ValueError(
                f"evaluation seed differs for {mode}: expected {seed}, got {rollout.evaluation_seed}"
            )
        if rollout.contact_allocation != mode:
            raise ValueError(
                f"contact allocation label differs: expected {mode}, got {rollout.contact_allocation}"
            )
        _validate_rollout(rollout, mode)

    fingerprints = (
        ("initial_state", "initial_state_fingerprint"),
        ("initial_command", "initial_command_fingerprint"),
        ("push_schedule", "push_schedule_fingerprint"),
    )
    evidence: dict[str, Any] = {"evaluation_seed": seed}
    for display_name, attribute in fingerprints:
        legacy_value = str(getattr(legacy, attribute))
        constrained_value = str(getattr(constrained, attribute))
        if not legacy_value or not constrained_value:
            raise ValueError(f"{display_name} fingerprint is missing")
        if legacy_value != constrained_value:
            raise ValueError(f"{display_name} fingerprint differs within seed {seed}")
        evidence[attribute] = legacy_value

    if legacy.domain != constrained.domain:
        raise ValueError(f"domain randomization differs within seed {seed}")
    evidence["domain"] = dict(legacy.domain)

    paired_conditions = (
        "state_estimation_mode",
        "latency_compensation",
        "action_delay_steps",
        "state_delay_steps",
        "sensor_noise_scale",
        "state_estimator_seed",
        "control_dt_s",
    )
    for attribute in paired_conditions:
        if getattr(legacy, attribute) != getattr(constrained, attribute):
            raise ValueError(f"paired condition {attribute} differs within seed {seed}")
    return evidence


def _episode_record(pair_index: int, rollout: D1Rollout) -> dict[str, Any]:
    record: dict[str, Any] = {
        "pair_index": pair_index,
        **compute_d1_metrics(rollout),
    }
    contacts = np.asarray(rollout.wheel_contacts)
    record["partial_contact_ratio"] = (
        float(np.mean((contacts > 0) & (contacts < 4))) if contacts.size else 0.0
    )
    for metric in AUDIT_METRICS:
        if not np.isfinite(float(record[metric])):
            raise ValueError(f"audit metric {metric} must be finite")
    return record


def _paired_interval(deltas: np.ndarray) -> tuple[float | None, float | None]:
    if deltas.size < 2:
        return None, None
    mean = float(np.mean(deltas))
    standard_error = float(np.std(deltas, ddof=1) / np.sqrt(deltas.size))
    half_width = float(student_t.ppf(0.975, deltas.size - 1) * standard_error)
    return mean - half_width, mean + half_width


def _summarize(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    modes = {
        mode: {
            int(record["evaluation_seed"]): record
            for record in records
            if record["contact_allocation"] == mode
        }
        for mode in ("legacy", "constrained")
    }
    seeds = sorted(modes["legacy"])
    rows: list[dict[str, Any]] = []
    for metric, (label, unit, direction) in AUDIT_METRICS.items():
        legacy = np.asarray([float(modes["legacy"][seed][metric]) for seed in seeds])
        constrained = np.asarray(
            [float(modes["constrained"][seed][metric]) for seed in seeds]
        )
        deltas = constrained - legacy
        low, high = _paired_interval(deltas)
        rows.append(
            {
                "metric": metric,
                "label": label,
                "unit": unit,
                "preferred": direction,
                "pair_count": len(seeds),
                "legacy_mean": float(np.mean(legacy)),
                "constrained_mean": float(np.mean(constrained)),
                "paired_delta": float(np.mean(deltas)),
                "paired_delta_ci95_low": low,
                "paired_delta_ci95_high": high,
            }
        )
    return rows


def _reduction_ratio(reference: float, candidate: float) -> float:
    if reference == 0.0:
        return 0.0 if candidate == 0.0 else -float("inf")
    return (reference - candidate) / abs(reference)


def _evaluate_promotion(
    *,
    episodes: int,
    seed_pool: str,
    records: Sequence[Mapping[str, Any]],
    summary_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    summary = {str(row["metric"]): row for row in summary_rows}
    legacy_success = float(summary["success"]["legacy_mean"])
    constrained_success = float(summary["success"]["constrained_mean"])
    force_reduction = _reduction_ratio(
        float(summary["contact_force_tracking_error_rms_n"]["legacy_mean"]),
        float(summary["contact_force_tracking_error_rms_n"]["constrained_mean"]),
    )
    moment_reduction = _reduction_ratio(
        float(summary["contact_moment_tracking_error_rms_nm"]["legacy_mean"]),
        float(summary["contact_moment_tracking_error_rms_nm"]["constrained_mean"]),
    )
    constrained_records = [
        record for record in records if record["contact_allocation"] == "constrained"
    ]
    max_violation = max(
        float(record["allocation_constraint_violation_max"])
        for record in constrained_records
    )
    fallback_ratio = float(summary["allocation_fallback_ratio"]["constrained_mean"])
    max_solver_degraded_ratio = max(
        float(record["allocation_feasible_nonconverged_ratio"])
        + float(record["allocation_fallback_ratio"])
        for record in constrained_records
    )
    max_solve_p99_ms = max(
        float(record["allocation_solve_p99_ms"]) for record in constrained_records
    )
    min_control_period_ms = min(
        float(record["episode_duration_s"]) * 1e3 / max(1, int(record["episode_steps"]))
        for record in constrained_records
    )
    min_physics_samples = min(
        int(record["contact_wrench_physics_samples_min"])
        for record in constrained_records
    )
    max_physics_samples = max(
        int(record["contact_wrench_physics_samples_max"])
        for record in constrained_records
    )
    required_physics_samples = int(
        PROMOTION_GATE["required_physics_samples_per_control_step"]
    )

    checks = {
        "preregistered_seed_pool": {
            "passed": seed_pool == "held_out",
            "observed": seed_pool,
            "criterion": (
                "evaluation_seeds must equal the preregistered held-out seeds "
                f"{PREREGISTERED_HELDOUT_SEEDS[0]}..{PREREGISTERED_HELDOUT_SEEDS[-1]}"
            ),
        },
        "formal_sample_size": {
            "passed": episodes >= int(PROMOTION_GATE["formal_min_episodes"]),
            "observed": episodes,
            "criterion": f">= {PROMOTION_GATE['formal_min_episodes']} matched episodes",
        },
        "success_noninferiority": {
            "passed": constrained_success
            >= legacy_success - float(PROMOTION_GATE["success_noninferiority_margin"]),
            "observed": constrained_success - legacy_success,
            "criterion": "constrained minus legacy success >= 0",
        },
        "force_tracking_reduction": {
            "passed": force_reduction
            >= float(PROMOTION_GATE["min_force_tracking_reduction_ratio"]),
            "observed": force_reduction,
            "criterion": ">= 10% reduction in physics-side force error",
        },
        "moment_tracking_reduction": {
            "passed": moment_reduction
            >= float(PROMOTION_GATE["min_moment_tracking_reduction_ratio"]),
            "observed": moment_reduction,
            "criterion": ">= 10% reduction in physics-side moment error",
        },
        "constraint_feasibility": {
            "passed": max_violation <= float(PROMOTION_GATE["max_constraint_violation"]),
            "observed": max_violation,
            "criterion": f"<= {PROMOTION_GATE['max_constraint_violation']}",
        },
        "fallback_rate": {
            "passed": fallback_ratio <= float(PROMOTION_GATE["max_fallback_ratio"]),
            "observed": fallback_ratio,
            "criterion": f"<= {PROMOTION_GATE['max_fallback_ratio']}",
        },
        "solver_degraded_rate": {
            "passed": max_solver_degraded_ratio
            <= float(PROMOTION_GATE["max_solver_degraded_ratio"]),
            "observed": max_solver_degraded_ratio,
            "criterion": (
                "per-episode feasible-nonconverged plus fallback ratio "
                f"<= {PROMOTION_GATE['max_solver_degraded_ratio']}"
            ),
        },
        "real_time_budget": {
            "passed": max_solve_p99_ms
            <= min_control_period_ms
            * float(PROMOTION_GATE["max_solve_time_fraction_of_control_period"]),
            "observed_ms": max_solve_p99_ms,
            "control_period_ms": min_control_period_ms,
            "criterion": "episode P99 allocation time <= one control period",
        },
        "physics_wrench_available": {
            "passed": (
                min_physics_samples == required_physics_samples
                and max_physics_samples == required_physics_samples
            ),
            "observed_min_samples_per_control_step": min_physics_samples,
            "observed_max_samples_per_control_step": max_physics_samples,
            "criterion": (
                f"exactly {required_physics_samples} MuJoCo physics contact-wrench "
                "samples per control step"
            ),
        },
    }
    failed_checks = [
        f"{name}: {check['criterion']}"
        for name, check in checks.items()
        if not bool(check["passed"])
    ]
    return {
        "passed": not failed_checks,
        "failed_checks": failed_checks,
        "checks": checks,
    }


def _write_report(
    path: Path,
    *,
    protocol: Mapping[str, Any],
    summary_rows: Sequence[Mapping[str, Any]],
    promotion: Mapping[str, Any],
) -> None:
    exploratory = bool(protocol["exploratory"])
    seed_pool = str(protocol["seed_pool"])
    if exploratory:
        if seed_pool == "held_out":
            status = (
                "Exploratory — not eligible for promotion: the formal protocol requires "
                f"{PROMOTION_GATE['formal_min_episodes']} matched episodes."
            )
        else:
            status = (
                "Exploratory (development seed pool) — not eligible for promotion: formal "
                "promotion is locked to the preregistered held-out seeds "
                f"{PREREGISTERED_HELDOUT_SEEDS[0]}..{PREREGISTERED_HELDOUT_SEEDS[-1]} with "
                f"{PROMOTION_GATE['formal_min_episodes']} matched episodes. This run used the "
                f"development seed pool starting at seed {protocol['first_seed']} and can never "
                "be marked formal or passing, regardless of episode count."
            )
    elif promotion["passed"]:
        status = "Formal — the preregistered simulation promotion gate passed."
    else:
        status = "Formal — the preregistered simulation promotion gate did not pass."
    lines = [
        "# D1 contact-allocation matched-seed audit",
        "",
        status,
        "",
        (
            "Each seed replays the same initial state, command, randomized domain, and push "
            "schedule. The treatment is the full mode-specific control path, not the allocator "
            "in isolation: the contact allocator, the allocator-matched identified sagittal "
            "linear model used by the outer-loop controller, and the mode-specific fixed "
            "outer-loop LQR weight R are all changed together between legacy and constrained. "
            "This is not an allocator-only causal ablation — differences between the two modes "
            "reflect this paired allocator/model/R control-path change, not the allocator alone."
        ),
        "",
        (
            "The physics-side errors use the MuJoCo solver contact wrench, averaged over "
            "physics substeps. Force and moment are reported separately in N and Nm; they are "
            "never combined into a dimensionally invalid score."
        ),
        "",
        (
            "Solver outcome and wrench tracking are independent diagnostics. Feasible "
            "nonconverged candidates remain visible and count toward the solver-degraded "
            "gate. Wrench tracking uses tolerances of 1 N + 0.5% of requested force norm "
            "and 0.5 Nm + 0.5% of requested moment norm."
        ),
        "",
        "| Metric | Unit | Legacy mean | Constrained mean | Paired delta | Delta 95% t CI |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in summary_rows:
        low = row["paired_delta_ci95_low"]
        high = row["paired_delta_ci95_high"]
        interval = "unavailable" if low is None else f"[{float(low):+.4g}, {float(high):+.4g}]"
        lines.append(
            "| {label} | {unit} | {legacy:.6g} | {constrained:.6g} | {delta:+.6g} | {ci} |".format(
                label=row["label"],
                unit=row["unit"],
                legacy=float(row["legacy_mean"]),
                constrained=float(row["constrained_mean"]),
                delta=float(row["paired_delta"]),
                ci=interval,
            )
        )
    lines.extend(
        [
            "",
            "## Promotion checks",
            "",
            "| Check | Result | Criterion |",
            "|---|---:|---|",
        ]
    )
    for name, check in promotion["checks"].items():
        lines.append(
            f"| {name.replace('_', ' ')} | {'pass' if check['passed'] else 'fail'} | "
            f"{check['criterion']} |"
        )
    lines.extend(
        [
            "",
            (
                "Passing this gate supports promoting the constrained allocator as the default "
                "simulation baseline. It is not hardware validation and does not establish a "
                "full whole-body controller."
            ),
            "",
        ]
    )
    _write_text_atomically(path, "\n".join(lines))


def _write_plot(path: Path, summary_rows: Sequence[Mapping[str, Any]]) -> None:
    summary = {str(row["metric"]): row for row in summary_rows}
    colors = ("#6B7280", "#0F766E")
    figure, axes = plt.subplots(2, 2, figsize=(10.2, 7.2), constrained_layout=True)
    plot_labels = {
        "success": "Episode\nsuccess",
        "four_wheel_contact_ratio": "Four-wheel\ncontact",
        "partial_contact_ratio": "Partial\ncontact",
        "allocation_force_error_rms_n": "Request →\nallocation",
        "allocation_moment_error_rms_nm": "Request →\nallocation",
        "contact_force_model_error_rms_n": "Allocation →\nphysics",
        "contact_moment_model_error_rms_nm": "Allocation →\nphysics",
        "contact_force_tracking_error_rms_n": "Request →\nphysics",
        "contact_moment_tracking_error_rms_nm": "Request →\nphysics",
        "allocation_solve_p99_ms": "Mean of episode P99",
    }

    def paired_bars(axis: Any, metrics: Sequence[str], title: str, unit: str) -> None:
        positions = np.arange(len(metrics), dtype=np.float64)
        width = 0.36
        legacy = [float(summary[metric]["legacy_mean"]) for metric in metrics]
        constrained = [float(summary[metric]["constrained_mean"]) for metric in metrics]
        axis.bar(positions - width / 2, legacy, width, label="Legacy", color=colors[0])
        axis.bar(
            positions + width / 2,
            constrained,
            width,
            label="Constrained",
            color=colors[1],
        )
        axis.set_xticks(positions, [plot_labels[metric] for metric in metrics])
        axis.set_ylabel(unit)
        axis.set_title(title, loc="left", fontweight="bold")
        axis.grid(axis="y", alpha=0.22)

    paired_bars(
        axes[0, 0],
        ("success", "four_wheel_contact_ratio", "partial_contact_ratio"),
        "Contact and survival",
        "ratio",
    )
    paired_bars(
        axes[0, 1],
        (
            "allocation_force_error_rms_n",
            "contact_force_model_error_rms_n",
            "contact_force_tracking_error_rms_n",
        ),
        "Force error RMS",
        "N",
    )
    paired_bars(
        axes[1, 0],
        (
            "allocation_moment_error_rms_nm",
            "contact_moment_model_error_rms_nm",
            "contact_moment_tracking_error_rms_nm",
        ),
        "Moment error RMS",
        "Nm",
    )
    paired_bars(
        axes[1, 1],
        ("allocation_solve_p99_ms",),
        "Allocator runtime",
        "ms",
    )
    axes[0, 0].set_ylim(0.0, 1.2)
    axes[0, 0].legend(frameon=False)
    axes[1, 1].axhline(
        AUDIT_CONTROL_DT_S * 1e3, color="#B45309", linestyle="--", label="Control period"
    )
    axes[1, 1].legend(frameon=False)
    count = int(summary["success"]["pair_count"])
    figure.suptitle(f"D1 contact allocation · {count} matched episodes · means", fontweight="bold")
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        figure.savefig(temporary, format="png", dpi=180)
        temporary.replace(path)
    finally:
        plt.close(figure)
        temporary.unlink(missing_ok=True)


def run_contact_allocation_audit(
    output: Path,
    *,
    seed: int,
    episodes: int,
    rollout_runner: RolloutRunner = run_d1_rollout,
    run_metadata: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Run and publish a paired legacy-versus-constrained simulation audit."""

    if episodes <= 0:
        raise ValueError("contact-allocation audit requires at least one episode")
    metadata = (
        json.loads(json.dumps(run_metadata))
        if run_metadata is not None
        else _capture_run_metadata()
    )
    _prepare_output(output)

    evaluation_seeds = list(range(seed, seed + episodes))
    seed_pool = (
        "held_out" if evaluation_seeds == list(PREREGISTERED_HELDOUT_SEEDS) else "development"
    )
    options = {
        "action_delay_steps": 0,
        "state_delay_steps": 0,
        "sensor_noise": 0.0,
    }
    records: list[dict[str, Any]] = []
    pair_evidence: list[dict[str, Any]] = []
    for pair_index, evaluation_seed in enumerate(evaluation_seeds):
        rollouts: dict[str, D1Rollout] = {}
        for mode in ("legacy", "constrained"):
            rollouts[mode] = rollout_runner(
                "lqr",
                "randomized",
                evaluation_seed,
                None,
                capture=False,
                state_mode="oracle",
                latency_compensation="none",
                contact_allocation=mode,
                episode_options=dict(options),
            )
        pair_evidence.append(
            _validate_pair(evaluation_seed, rollouts["legacy"], rollouts["constrained"])
        )
        records.extend(
            _episode_record(pair_index, rollouts[mode])
            for mode in ("legacy", "constrained")
        )

    expected_rows = episodes * 2
    if len(records) != expected_rows:
        raise ValueError(
            f"episode record count differs: expected {expected_rows}, got {len(records)}"
        )
    summary_rows = _summarize(records)
    promotion = _evaluate_promotion(
        episodes=episodes,
        seed_pool=seed_pool,
        records=records,
        summary_rows=summary_rows,
    )
    protocol = {
        "baseline": "lqr",
        "scenario": "randomized",
        "control_dt_s": AUDIT_CONTROL_DT_S,
        "episode_seconds": AUDIT_EPISODE_SECONDS,
        "contact_allocation_modes": ["legacy", "constrained"],
        "state_mode": "oracle",
        "latency_compensation": "none",
        "episode_options": options,
        "first_seed": seed,
        "evaluation_seeds": evaluation_seeds,
        "episodes": episodes,
        "seed_pool": seed_pool,
        "preregistered_held_out_seeds": list(PREREGISTERED_HELDOUT_SEEDS),
        "exploratory": seed_pool != "held_out",
        "pairing": "same-process, same-seed replay",
        "contact_wrench_measurement": "MuJoCo physics-substep mean",
        "constrained_allocator": {
            "friction_coefficient": D1_CONSTRAINED_FRICTION_COEFFICIENT,
            "normal_force_limit_weight_fraction": (
                D1_NORMAL_FORCE_LIMIT_WEIGHT_FRACTION
            ),
            "wrench_characteristic_length_m": D1_WRENCH_CHARACTERISTIC_LENGTH_M,
            "slsqp_max_iterations": D1_SLSQP_MAX_ITERATIONS,
            "slsqp_ftol": D1_SLSQP_FTOL,
            "candidate_constraint_violation_ratio_tolerance": (
                D1_CANDIDATE_CONSTRAINT_VIOLATION_TOL
            ),
            "force_tracking_absolute_tolerance_n": D1_FORCE_TRACKING_ABS_TOL_N,
            "moment_tracking_absolute_tolerance_nm": (
                D1_MOMENT_TRACKING_ABS_TOL_NM
            ),
            "wrench_tracking_relative_tolerance": D1_WRENCH_TRACKING_REL_TOL,
        },
    }
    identity = json.dumps(
        {"protocol": protocol, "provenance": metadata},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    run_id = f"d1-contact-allocation-{hashlib.sha256(identity).hexdigest()[:16]}"
    config = {
        "schema": "d1-contact-allocation-audit-config-v1",
        "run_id": run_id,
        "protocol": protocol,
        "promotion_gate": PROMOTION_GATE,
        "provenance": metadata,
    }

    _write_json_atomically(output / "evaluation_config.json", config)
    _write_csv_atomically(
        output / "contact_allocation_episodes.csv",
        records,
        list(records[0]),
    )
    _write_csv_atomically(
        output / "contact_allocation_summary.csv",
        summary_rows,
        list(summary_rows[0]),
    )
    _write_report(
        output / "contact_allocation_audit.md",
        protocol=protocol,
        summary_rows=summary_rows,
        promotion=promotion,
    )
    _write_plot(output / "contact_allocation_audit.png", summary_rows)

    artifact_names = [name for name in AUDIT_ARTIFACTS if not name.endswith("manifest.json")]
    missing = [name for name in artifact_names if not (output / name).is_file()]
    if missing:
        raise ValueError(f"contact-allocation audit artifacts are missing: {missing}")
    manifest = {
        "schema": "d1-contact-allocation-audit-v1",
        "run_id": run_id,
        "status": "complete",
        "completed": True,
        "protocol": protocol,
        "provenance": metadata,
        "validation": {
            "matched_input_fingerprints": True,
            "matched_conditions": True,
            "matched_pairs": episodes,
            "episode_rows": len(records),
            "pairs": pair_evidence,
        },
        "promotion": promotion,
        "artifact_sha256": {
            name: _file_sha256(output / name) for name in artifact_names
        },
    }
    _write_json_atomically(output / "contact_allocation_manifest.json", manifest)
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit D1 legacy and constrained contact allocation on matched seeds."
    )
    parser.add_argument("--output", type=Path, default=Path("results/d1_contact_allocation"))
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--episodes", type=int, default=3)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    records = run_contact_allocation_audit(
        args.output,
        seed=args.seed,
        episodes=args.episodes,
    )
    print(f"wrote {len(records)} episode records to {args.output}")
    print((args.output / "contact_allocation_audit.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
