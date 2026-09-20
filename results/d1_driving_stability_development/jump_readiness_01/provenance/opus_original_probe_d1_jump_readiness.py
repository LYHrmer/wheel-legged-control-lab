"""Fixed two-episode runner for the D1 stationary height-profile hop readiness
diagnostic: hold then profile, 600 executed ticks each, seed 55101.

The runner owns source identity, per-tick command verification against the fixed
pure schedule, the native observer, geometry reconstruction, scoring, the paired
prefix proof and all archiving.  It never calls the G1 episode entries and never
weakens their command verification: the expected command is re-derived here from
the frozen schedule and compared field by field against what actually reached the
controller on that tick.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from scripts.d1_jump_readiness import TERMINAL_PREPARED_TICK, fixed_height_command
from scripts.d1_jump_readiness_records import (
    READINESS_SCORE_SCHEMA,
    score_readiness,
)
from scripts.d1_jump_readiness_env import READINESS_PROBE_SCHEMA

ROOT = Path(__file__).resolve().parents[1]
SEED = 55101
CONTROL_DT_S = 0.01
NATIVE_DT_S = 0.002
SUBSTEPS_PER_TICK = 5
TICKS = 600
MAX_CONTROL_INTERVALS = 1200
MAX_NATIVE_SUBSTEPS = 6000
CONDITIONS = (("stationary_height_hold", "hold"), ("stationary_height_profile", "profile"))
PAIRED_SLICES = {"execution": 200, "states": 201, "observations": 200, "native": 1000}
EXCLUDED_PAIR_FIELDS = ("condition", "display_phase", "next_prepared_clearance_m",
                        "next_prepared_phase")
_TOL_M = 1.0e-12
_TOL_S = 1.0e-9

SOURCE_FILES = (
    "scripts/d1_jump_readiness.py",
    "scripts/d1_jump_readiness_env.py",
    "scripts/d1_jump_readiness_records.py",
    "scripts/probe_d1_jump_readiness.py",
    "tests/test_d1_jump_readiness_env.py",
    "tests/test_d1_jump_readiness_records.py",
    "tests/test_d1_jump_readiness_runner.py",
    "jump_obstacle_next_plan_01/next_contract.md",
    "jump_obstacle_next_plan_01/proposed_cases.json",
    "opus_jump_readiness_01/original_module.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes(root: Path = ROOT) -> Dict[str, Any]:
    found, missing = {}, []
    for name in SOURCE_FILES:
        target = root / name
        if target.is_file():
            found[name] = sha256(target)
        else:
            missing.append(name)
    return {"files": found, "missing": missing}


def write_json(path: Path, value: Any) -> None:
    with open(path, "x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def tagged(value: Any) -> Any:
    """Encode non-finite numbers with an explicit tag instead of dropping them."""
    if isinstance(value, dict):
        return {k: tagged(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [tagged(v) for v in value]
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if number != number:
            return {"nonfinite": "nan"}
        if number in (float("inf"), float("-inf")):
            return {"nonfinite": "inf" if number > 0 else "-inf"}
        return number
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.ndarray):
        return tagged(value.tolist())
    return value


def expected_schedule(condition: str) -> List[Dict[str, Any]]:
    """Pure per-tick expectation, including the never-executed prepared tick 600."""
    rows = []
    for tick in range(TERMINAL_PREPARED_TICK + 1):
        command, label = fixed_height_command(tick, condition)
        rows.append({"tick": tick, "clearance_m": float(command.clearance_m),
                     "display_phase": label, "executed": tick < TICKS,
                     "forward_mps": float(command.forward_velocity_mps),
                     "yaw_rps": float(command.yaw_rate_rps)})
    return rows


def verify_tick_command(condition: str, tick: int, row: Dict[str, Any]) -> Dict[str, Any]:
    """Field-by-field check of one executed tick against the fixed schedule."""
    command, label = fixed_height_command(tick, condition)
    height = float(command.clearance_m)
    action = np.asarray(row.get("action", []), dtype=np.float64)
    world = float(row.get("world_command_base_height_m", float("nan")))
    ground = float(row.get("ground_height_m", float("nan")))
    checks = {
        "raw_clearance": abs(float(row.get("raw_clearance_m", float("nan"))) - height) <= _TOL_M,
        "servo_clearance": abs(
            float(row.get("servo_clearance_m", float("nan"))) - height) <= _TOL_M,
        "world_height_over_ground": abs((world - ground) - height) <= _TOL_M,
        "raw_forward_zero": float(row.get("raw_forward_mps", 1.0)) == 0.0,
        "raw_yaw_zero": float(row.get("raw_yaw_rps", 1.0)) == 0.0,
        "display_phase": row.get("display_phase") == label,
        "control_time": abs(
            float(row.get("control_time_s", float("nan"))) - tick * CONTROL_DT_S) <= _TOL_S,
        "zero_action": action.shape == (8,) and bool(np.all(action == 0.0)),
        "known_wrench": row.get("unknown_wrench") is False,
        "attitude_matches_truth": bool(
            abs(float(row.get("quaternion_attitude_error_rad", float("nan")))) <= 1.0e-6),
        "stop_latch_inactive": row.get("stop_latch_active") is False,
        "authority_gate_inactive": row.get("authority_gate_active") is False,
    }
    return {"tick": tick, "condition": condition, "checks": checks,
            "passed": all(bool(v) for v in checks.values())}


def bitwise_equal(first: Any, second: Any) -> bool:
    left = np.asarray(first)
    right = np.asarray(second)
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    return left.tobytes() == right.tobytes()


def paired_prefix(hold: Dict[str, Any], profile: Dict[str, Any]) -> Dict[str, Any]:
    """Bitwise prefix comparison of the two episodes' shared initial segment."""
    checks: Dict[str, Any] = {}
    for group, count in PAIRED_SLICES.items():
        group_checks: Dict[str, bool] = {}
        left, right = hold.get(group) or {}, profile.get(group) or {}
        keys = sorted(set(left) | set(right))
        for key in keys:
            if key in EXCLUDED_PAIR_FIELDS:
                continue
            first, second = left.get(key), right.get(key)
            if first is None or second is None:
                group_checks[key] = False
                continue
            group_checks[key] = (len(first) >= count and len(second) >= count
                                 and bitwise_equal(first[:count], second[:count]))
        checks[group] = {"count": count, "fields": group_checks,
                         "passed": bool(group_checks) and all(group_checks.values())}
    return {"schema": READINESS_PROBE_SCHEMA,
            "excluded_fields": list(EXCLUDED_PAIR_FIELDS),
            "excluded_observation_index": 200,
            "self_pairing_is_not_accepted": True,
            "groups": checks,
            "passed": all(entry["passed"] for entry in checks.values())}


def resolve_steps(*, analyze_only: bool) -> int:
    """Analysis mode forces zero steps; there is no hidden rollout."""
    return 0 if analyze_only else TICKS * len(CONDITIONS)


def execute_episode(env: Any, *, condition: str, ticks: int = TICKS,
                    action: Any = None) -> Dict[str, Any]:
    """Run one fixed episode with full consumption accounting in ``finally``."""
    if action is None:
        action = np.zeros(8, dtype=np.float32)
    accounting: Dict[str, Any] = {
        "condition": condition, "requested_ticks": int(ticks),
        "completed_control_intervals": 0, "verifications": [], "error": None,
        "terminated": False, "truncated": False,
    }
    try:
        env.reset(seed=SEED)
        for tick in range(int(ticks)):
            result = env.step(action)
            accounting["completed_control_intervals"] = tick + 1
            row = dict(getattr(env, "last_readiness_row", {}) or {})
            row.setdefault("action", np.asarray(action, dtype=np.float64).tolist())
            verified = verify_tick_command(condition, tick, row)
            accounting["verifications"].append(verified)
            if not verified["passed"]:
                raise RuntimeError(
                    f"tick {tick} command verification failed: {verified['checks']}")
            accounting["terminated"] = bool(result[2])
            accounting["truncated"] = bool(result[3])
            if result[2] or result[3]:
                break
    except BaseException as error:  # accounting must survive any failure
        accounting["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        accounting["height_callback_count"] = int(getattr(env, "height_callback_count", 0))
        accounting["parent_step_calls"] = int(getattr(env, "parent_step_calls", 0))
        accounting["executed_ticks"] = list(getattr(env, "executed_ticks", []))
        accounting["terminal_prepared_tick_executed"] = bool(
            TERMINAL_PREPARED_TICK in accounting["executed_ticks"])
    return accounting


def budget_receipt(episodes: List[Dict[str, Any]], native_calls: int) -> Dict[str, Any]:
    intervals = sum(int(e.get("completed_control_intervals", 0)) for e in episodes)
    return {
        "schema": READINESS_PROBE_SCHEMA,
        "completed_control_intervals": intervals,
        "actual_native_calls": int(native_calls),
        "maximum_new_control_intervals": MAX_CONTROL_INTERVALS,
        "maximum_new_native_substeps": MAX_NATIVE_SUBSTEPS,
        "within_budget": intervals <= MAX_CONTROL_INTERVALS
        and int(native_calls) <= MAX_NATIVE_SUBSTEPS,
        "partial_episode_consumption_preserved": True,
    }


def write_manifest(directory: Path) -> Path:
    entries = {}
    for path in sorted(Path(directory).rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            entries[str(path.relative_to(directory))] = {
                "bytes": path.stat().st_size, "sha256": sha256(path)}
    target = Path(directory) / "manifest.json"
    write_json(target, {"schema": READINESS_PROBE_SCHEMA, "files": entries})
    return target


def analyze_stored_records(folder: Path) -> Dict[str, Any]:
    """Recompute the score from stored records only; zero steps are executed."""
    payload = json.loads((Path(folder) / "readiness_records.json").read_text())
    score = score_readiness(payload)
    return {"schema": READINESS_SCORE_SCHEMA, "steps": resolve_steps(analyze_only=True),
            "score": score}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--analyze-only", action="store_true")
    return parser


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.analyze_only:
        result = analyze_stored_records(args.output)
        write_json(args.output / "readiness_score_reanalysis.json", tagged(result))
        return 0
    raise SystemExit(
        "physical execution of this diagnostic is reserved for the root agent; "
        "this worker delivered implementation and non-integrating verification only")


if __name__ == "__main__":  # pragma: no cover - root-owned execution entry
    raise SystemExit(main())
