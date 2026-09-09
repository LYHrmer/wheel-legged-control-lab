"""Development-only, single-variable probes of moving terrain references.

The default variant is exactly the v2 controller. Other variants change either
attitude-reference bandwidth or vertical feedforward, never both. They are
process-local counterfactuals, not new policy-compatible control schemas.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from wheel_legged_control.d1.hierarchical import D1_VERTICAL_FEEDFORWARD_LIMIT_N
from wheel_legged_control.d1.terrain_tracking_env import D1TerrainTrackingEnv

if __package__:
    from . import run_d1_terrain_curriculum as runner
else:
    import run_d1_terrain_curriculum as runner

INFO_FIELDS = runner.INFO_FIELDS
ROOT = runner.ROOT
TRACKING_INFO_FIELDS = runner.TRACKING_INFO_FIELDS
TRACKING_QUALITY_CRITERIA = runner.TRACKING_QUALITY_CRITERIA
_sha256 = runner._sha256
_source_hashes = runner._source_hashes
_write_csv = runner._write_csv
_write_json = runner._write_json
evaluation_cases = runner.evaluation_cases
summarize_episode = runner.summarize_episode

VARIANTS = (
    "baseline",
    "vertical_feedforward",
    "attitude_tau_20ms",
    "attitude_tau_50ms",
    "attitude_tau_100ms",
)


def moving_height_velocity(ground_pitch_rad: float, velocity_x_world_mps: float) -> float:
    """Chain rule for a static, y-extruded collision heightfield, in SI units."""
    return -math.tan(ground_pitch_rad) * velocity_x_world_mps


class ReferenceProbeEnv(D1TerrainTrackingEnv):
    """Temporary control probe; never used to train or load a policy."""

    def __init__(self, variant: str = "baseline", **kwargs: Any) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"unknown reference probe: {variant}")
        self.variant = variant
        self._filter_state: np.ndarray | None = None
        self._filter_step = -1
        self.requested_feedforward_n = 0.0
        self.requested_height_velocity_mps = 0.0
        super().__init__(**kwargs)
        compute = self.controller.compute

        def probe_compute(command, state, residual_force_n=None, **extra):
            if self.variant == "vertical_feedforward":
                self.requested_height_velocity_mps = moving_height_velocity(
                    self._ground_reference().pitch_rad, float(state.base_linear_velocity_world[0])
                )
                self.requested_feedforward_n = (
                    self.controller.low_level.height_kd * self.requested_height_velocity_mps
                )
                extra["vertical_feedforward_force_n"] = self.requested_feedforward_n
            return compute(command, state, residual_force_n, **extra)

        self.controller.compute = probe_compute

    def reset(self, **kwargs):
        self._filter_state = None
        self._filter_step = -1
        self.requested_feedforward_n = 0.0
        self.requested_height_velocity_mps = 0.0
        return super().reset(**kwargs)

    def _command_at_step(self):
        command = super()._command_at_step()
        if not self.variant.startswith("attitude_tau_"):
            return command
        tau_s = float(self.variant.removeprefix("attitude_tau_").removesuffix("ms")) / 1000
        target = np.asarray((command.roll_rad, command.pitch_rad))
        if self._filter_state is None:
            self._filter_state = target.copy()
        elif self._filter_step != self._step_count:
            alpha = -math.expm1(-self.plant.control_dt / tau_s)
            self._filter_state += alpha * (target - self._filter_state)
        self._filter_step = self._step_count
        return replace(
            command, roll_rad=float(self._filter_state[0]), pitch_rad=float(self._filter_state[1])
        )


def runtime_hashes() -> dict[str, str]:
    """Hash loaded project Python plus robot assets, not unrelated parallel edits."""
    loaded = {
        str(Path(module.__file__).resolve())
        for module in list(sys.modules.values())
        if getattr(module, "__file__", None)
    }
    result = {
        relative: digest
        for relative, digest in _source_hashes().items()
        if not relative.endswith(".py") or str((ROOT / relative).resolve()) in loaded
    }
    result[str(Path(__file__).resolve().relative_to(ROOT))] = _sha256(Path(__file__))
    return result


def probe_case(case: dict[str, Any], variant: str, seconds: float = 4.0):
    env = ReferenceProbeEnv(
        variant=variant, episode_seconds=seconds, training_mode="flat", randomize=False
    )
    rows = []
    try:
        _, reset_info = env.reset(
            seed=case["environment_seed"],
            options={key: case[key] for key in ("terrain", "velocity_mps", "height_m")},
        )
        for step in range(env.max_steps):
            command = env._command
            _, reward, terminated, truncated, info = env.step(np.zeros(2))
            row = {
                "step": step + 1,
                "time_s": (step + 1) * env.plant.control_dt,
                **{key: info[key] for key in (*INFO_FIELDS, *TRACKING_INFO_FIELDS)},
                "initial_position_x_m": reset_info["position_x_m"],
                "policy_enabled": 0,
                "policy_gated": 0,
                "reward": float(reward),
                "terminated": int(terminated),
                "truncated": int(truncated),
                "termination_reason": info["termination_reason"],
                "control_pitch_target_rad": command.pitch_rad,
                "control_roll_target_rad": command.roll_rad,
                "height_velocity_reference_mps": env.requested_height_velocity_mps,
                "requested_vertical_feedforward_n": env.requested_feedforward_n,
                "applied_vertical_feedforward_n": float(
                    np.clip(
                        env.requested_feedforward_n,
                        -D1_VERTICAL_FEEDFORWARD_LIMIT_N,
                        D1_VERTICAL_FEEDFORWARD_LIMIT_N,
                    )
                ),
                "body_vertical_velocity_mps": float(env.plant.base_velocity()[0][2]),
                "longitudinal_force_n": env.controller.last_longitudinal_force_n,
                **{f"reward_{key}": value for key, value in info["reward_terms"].items()},
            }
            rows.append(row)
            if terminated or truncated:
                break
    finally:
        env.close()
    metrics = summarize_episode(
        rows,
        quality_criteria=TRACKING_QUALITY_CRITERIA,
        requested_duration_s=seconds,
        target_velocity_mps=case["velocity_mps"],
    )
    error = np.asarray([row["velocity_error_mps"] for row in rows[-100:]])
    bias, variance = float(error.mean()), float(error.var())
    metrics.update(
        variant=variant,
        case_id=case["case_id"],
        tail_velocity_bias_mps=bias,
        tail_velocity_std_mps=math.sqrt(variance),
        decomposition_error=float(abs(np.mean(error**2) - (bias**2 + variance))),
        tail_error_type="bias_dominant" if bias**2 >= variance else "oscillation_dominant",
    )
    return rows, metrics


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--case-indices", nargs="+", type=int, default=list(range(24)))
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--require-quality",
        action="store_true",
        help="red-capable assertion on the original tracking symptom",
    )
    args = parser.parse_args(argv)
    if not math.isfinite(args.seconds) or args.seconds < 4 or args.repeats < 1:
        parser.error("seconds must be finite and >=4; repeats must be positive")
    if any(index not in range(24) for index in args.case_indices):
        parser.error("case indices must be in [0,23]")
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    sources = runtime_hashes()
    cases = [evaluation_cases("development", "tracking-v2")[i] for i in args.case_indices]
    args.output.mkdir(parents=True, exist_ok=False)
    _write_json(
        args.output / "protocol.json",
        {
            "scope": "development-only zero-residual reference dynamics, no policy selection",
            "variants": args.variants,
            "cases": cases,
            "seconds": args.seconds,
            "repeats": args.repeats,
            "quality_criteria": TRACKING_QUALITY_CRITERIA,
            "vertical_formula": "kd * (-tan(ground_pitch)) * vx_world, existing +/-500N feedforward clip",
            "attitude_filter": "exact discrete first-order lowpass of roll/pitch command; all gains and raw reward targets unchanged",
            "source_sha256": sources,
        },
    )
    records = []
    for variant in args.variants:
        for case in cases:
            for repeat in range(args.repeats):
                rows, metrics = probe_case(case, variant, args.seconds)
                metrics["repeat"] = repeat
                name = f"{variant}__{case['case_id']}__repeat_{repeat}.csv"
                _write_csv(args.output / name, rows, tuple(rows[0]))
                records.append(metrics)
                print(
                    f"{variant} {case['case_id']} tail={metrics['tail_velocity_rmse_mps']:.5f} pass={metrics['quality_success']}",
                    flush=True,
                )
    _write_csv(args.output / "metrics.csv", records, tuple(records[0]))
    unchanged = all(_sha256(ROOT / key) == value for key, value in sources.items())
    _write_json(
        args.output / "summary.json",
        {"episodes": len(records), "source_unchanged_during_run": unchanged, "metrics": records},
    )
    _write_json(
        args.output / "manifest.json",
        {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(args.output.iterdir())
            if p.is_file()
        },
    )
    if not unchanged:
        raise SystemExit("loaded runtime source changed during probe")
    if args.require_quality and not all(row["quality_success"] == 1 for row in records):
        raise SystemExit("terrain tracking quality failed")


if __name__ == "__main__":
    main()
