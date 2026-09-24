"""One frozen, prepaid process: 65,536 PPO controls and eight held-out trials."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

CONTRACT_SHA256 = "500a00e22f86600b986e7d537e8d891c09888fb33e5f11a8a2a774deae7f10e0"
DSO_SHA256 = "3ec7ec9a6a130b9e1fa153c800aab97b59d4692c24971027f02de42957f8fa9c"
TRAIN_CONTROL = 65536
EVAL_CONTROL = 9600
NORMAL_STEPS = 375680
CONSTRUCTION_STEPS = 5


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _check_freeze(args: argparse.Namespace) -> dict[str, Any]:
    frozen = json.loads(args.freeze.read_text(encoding="utf-8"))
    expected = {
        "contract_sha256": CONTRACT_SHA256,
        "astra_review_decision": "GO",
        "reserved_training_control": TRAIN_CONTROL,
        "reserved_evaluation_control": EVAL_CONTROL,
        "reserved_normal_steps": NORMAL_STEPS,
        "reserved_construction_steps": CONSTRUCTION_STEPS,
        "process_limit": 1,
        "output_directory": str(args.output.resolve()),
    }
    if any(type(frozen.get(key)) is not type(value) or frozen[key] != value
           for key, value in expected.items()):
        raise ValueError("root freeze protocol or reserved budget mismatch")
    if _sha256(args.contract) != CONTRACT_SHA256 or _sha256(args.library) != DSO_SHA256:
        raise ValueError("reviewed contract or DSO identity changed")
    if frozen.get("python_argv") != [str(Path(sys.argv[0]).resolve()), *sys.argv[1:]]:
        raise ValueError("actual execution argv differs from frozen launch")
    files = frozen.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("root freeze files are absent")
    for name, entry in files.items():
        path = Path(name)
        if (not path.is_absolute() or not isinstance(entry, dict)
                or path.stat().st_size != entry.get("bytes")
                or _sha256(path) != entry.get("sha256")):
            raise ValueError(f"frozen source identity mismatch: {name}")
    if str(args.library.resolve()) not in files or str(args.contract.resolve()) not in files:
        raise ValueError("DSO and contract must be present in the root freeze")
    return frozen


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.output.is_dir():
        raise ValueError("outer launcher must precreate the sole output directory")
    freeze = _check_freeze(args)

    # C1 qualification showed that all extension GOTs must be present before
    # EngineGuard.binding_proof. These imports occur only in this prepaid process.
    import mujoco
    import mujoco._functions
    import mujoco._rollout

    from wheel_legged_control.d1.wheel_leg_controller import _nominal_geometry

    cache_before = _nominal_geometry.cache_info()._asdict()
    if cache_before != {"hits": 0, "misses": 0, "maxsize": 1, "currsize": 0}:
        raise RuntimeError("nominal geometry model cache was not cold before construction")

    from scripts.d1_rolling_engine_runtime import RollingEngineRuntime
    from scripts.d1_rolling_native_monitor import NativeGeometryGuard
    from scripts.evaluate_d1_rolling_residual import run_evaluation
    from scripts.train_d1_rolling_residual_ppo import run_training, training_protocol

    _write_json(args.output / "protocol.json", training_protocol(
        library=args.library, contract_sha256=CONTRACT_SHA256, freeze_path=args.freeze,
        freeze=freeze,
    ))
    failure: dict[str, str] | None = None
    training: dict[str, Any] | None = None
    evaluation: dict[str, Any] | None = None
    runtime: RollingEngineRuntime | None = None
    guard: NativeGeometryGuard | None = None
    warnings: list[str] = []

    def warning(message: str) -> None:
        warnings.append(str(message))
        raise RuntimeError("MuJoCo warning: " + str(message))

    previous_warning = mujoco.get_mju_user_warning()
    try:
        mujoco.set_mju_user_warning(warning)
        with RollingEngineRuntime(args.library, control_limit=TRAIN_CONTROL + EVAL_CONTROL,
                                  construction_limit=CONSTRUCTION_STEPS) as runtime:
            with NativeGeometryGuard(runtime, args.output) as guard:
                training, box_env, loaded_model = run_training(
                    runtime, args.output / "training", dso_sha256=DSO_SHA256,
                    training_protocol_sha256=_sha256(args.output / "protocol.json"),
                )
                if not training["passed"]:
                    raise RuntimeError("the single bounded training run did not complete")
                evaluation = run_evaluation(
                    runtime, box_env, loaded_model, args.output / "evaluation",
                    checkpoint_sha256=training["model_sha256"],
                )
                box_env.close()
            runtime.unbind()
    except BaseException as exc:  # noqa: BLE001 -- preserve the sole prepaid run
        failure = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        mujoco.set_mju_user_warning(previous_warning)
    state = None if runtime is None else runtime.state()
    ledger = None if runtime is None else runtime.ledger_receipt()
    native = None if guard is None else guard.report()
    cache_after = _nominal_geometry.cache_info()._asdict()
    cache_verified = cache_after == {"hits": 3, "misses": 1, "maxsize": 1, "currsize": 1}
    sources_unchanged = all(
        Path(name).is_file() and Path(name).stat().st_size == entry["bytes"]
        and _sha256(Path(name)) == entry["sha256"]
        for name, entry in freeze["files"].items()
    )
    construction_records = [] if ledger is None else ledger["construction_records"]
    compiler_deltas = [
        (row["after"]["construction_attempts"] - row["before"]["construction_attempts"],
         row["after"]["construction_returns"] - row["before"]["construction_returns"])
        for row in construction_records
    ]
    recorded_control = (None if training is None or evaluation is None
                        else training["physical_control_transitions"] + sum(
                            row["score"]["metrics"]["completed_control_intervals"]
                            for row in evaluation["cases"]
                        ))
    actual_counts_consistent = bool(
        state is not None and ledger is not None and native is not None
        and recorded_control is not None
        and state["construction_attempts"] == state["construction_returns"] == CONSTRUCTION_STEPS
        and state["control_attempts"] == state["control_returns"] == 5 * recorded_control
        and state["ccd_attempts"] == state["ccd_returns"] > 0
        and state["native_ccd_caller_verified"]
        and state["native_construction_caller_verified"]
        and state["phase"] == 0 and state["target_model"] == state["target_data"] == 0
        and state["violations"] == 0
        and ledger["control_attempted"] == ledger["control_completed"] == recorded_control
        and ledger["native_attempted"] == ledger["native_returned"]
        == ledger["clock_advanced_substeps"] == 5 * recorded_control
        and ledger["native_failed"] == ledger["forbidden_entries"] == 0
        and ledger["fatal_latched"] is False
        and native["checked_native_returns"] == native["passed_native_returns"] == 5 * recorded_control
        and native["failure"] is None
        and compiler_deltas == [(3, 3), (2, 2)]
        and cache_verified
    )
    full_counts = bool(recorded_control == TRAIN_CONTROL + EVAL_CONTROL
                       and ledger is not None and ledger["native_returned"] == NORMAL_STEPS)
    passed = (failure is None and not warnings and actual_counts_consistent
              and training is not None and training["passed"]
              and evaluation is not None and evaluation["qualified"]
              and sources_unchanged)
    _write_json(args.output / "study_receipt.json", {
        "passed": bool(passed), "failure": failure,
        "training": training, "evaluation": evaluation,
        "engine_binding": None if runtime is None else runtime.proof,
        "engine_state": state, "ledger": ledger, "native_monitor": native,
        "warnings": warnings, "cache_before": cache_before, "cache_after": cache_after,
        "cache_verified": cache_verified, "compiler_deltas": compiler_deltas,
        "recorded_control_intervals": recorded_control,
        "actual_counts_consistent": actual_counts_consistent, "full_counts": full_counts,
        "sources_unchanged": sources_unchanged,
        "no_retries_or_additional_engine_processes": True,
    })
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
