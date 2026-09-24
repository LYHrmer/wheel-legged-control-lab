"""One prepaid, fixed four-case evaluation of transferred body-speed feedback.

The root launcher owns the sole physical process and full budget reservation.
This worker never trains, repeats a case, or changes any original evidence.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

CONTRACT_SHA256 = "13343fc40b726c0a8db6f17bbd475d44d4c303f0c5b7a5a2be2f0257b1af92e7"
PREVIOUS_FREEZE_SHA256 = "3de560e0a46e456b72b30661ec497ec1a1cc497abf49fcf33ab46553a8bc12b9"
PREVIOUS_MANIFEST_SHA256 = "dc547afa31d65b099a6de93d50d796eb1f7956a2f9ac1dc26806c70d4712912c"
PREVIOUS_CLOSURE_SHA256 = "61a52581b043dc32f804fcb67a215427084a85664c6ec0b16d2ba4673f779a4f"
DSO_SHA256 = "3ec7ec9a6a130b9e1fa153c800aab97b59d4692c24971027f02de42957f8fa9c"
ORIGINAL_PROTOCOL_SHA256 = "494eb82ca80a842c190bbbff9bc75fa10c2e7832789a2d9bce0785acc3684733"
MODEL_SHA256 = "4f59d795afbdec256ec17bb7256ea05a3ddd04561e9ba52beffb1de375196b2e"
METADATA_SHA256 = "44b199ea13989412d274955a9fd85d6e97e58ad708a0c7e3ab7a4085ae3daaf3"
PROBE_SHA256 = "2cb674f54c8c46d5555e5d3580810a003be8d50fa5a7c1a5526dd0f027bb6660"
BASELINE_MAP_SHA256 = "55946f0d4c593db5e328b84021b5cbc8f299ce9b56c27e82ced71ae64e7b3dff"
EVAL_LIMIT = 4800
NORMAL_LIMIT = 24000
CONSTRUCTION_LIMIT = 5
CASE_LIMIT = 1200
CASE_ORDER = (
    ("speed200_plane_final_policy", .2, "plane"),
    ("speed200_box_final_policy", .2, "box"),
    ("speed250_plane_final_policy", .25, "plane"),
    ("speed250_box_final_policy", .25, "box"),
)
PAIR_FIELDS = ("qpos", "qvel", "ctrl", "qacc_warmstart", "observation")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _save(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _load(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise TypeError(f"expected JSON object: {path}")
    return result


def _check_freeze(args: argparse.Namespace) -> dict[str, Any]:
    freeze = _load(args.freeze)
    expected = {
        "schema": "body-common-p-evaluation-freeze-v1",
        "astra_decision": "GO", "process_limit": 1,
        "training_control_limit": 0, "evaluation_control_limit": EVAL_LIMIT,
        "normal_native_limit": NORMAL_LIMIT,
        "compiler_native_limit": CONSTRUCTION_LIMIT, "wallclock_limit_s": 600,
        "contract_path": str(args.contract.resolve()),
        "contract_sha256": CONTRACT_SHA256,
        "original_run": str(args.original_run.resolve()),
        "drive_baseline_directory": str((Path(__file__).resolve().parent.parent /
                                          "stability_20260924_drive_damping02/run_01").resolve()),
        "output_directory": str(args.output.resolve()),
        "checkpoint_sha256": MODEL_SHA256,
        "original_training_protocol_sha256": ORIGINAL_PROTOCOL_SHA256,
        "old_original_run_02_qualification": False,
        "old_final_native_C_counts": None,
    }
    for key, want in expected.items():
        if type(freeze.get(key)) is not type(want) or freeze[key] != want:
            raise ValueError(f"body-P freeze field differs: {key}")
    if freeze.get("python_argv") != [str(Path(sys.argv[0]).resolve()), *sys.argv[1:]]:
        raise ValueError("actual Python argv differs from the prepaid freeze")
    if _sha256(args.contract) != CONTRACT_SHA256 or _sha256(args.library) != DSO_SHA256:
        raise ValueError("reviewed contract or isolated engine identity differs")
    files = freeze.get("files")
    if not isinstance(files, dict) or len(files) < 662 + 77:
        raise ValueError("frozen source/archive list is incomplete")
    for name, row in files.items():
        path = Path(name)
        if (not path.is_absolute() or not path.is_file()
                or path.stat().st_size != row.get("bytes")
                or _sha256(path) != row.get("sha256")):
            raise ValueError(f"frozen input identity differs: {name}")
    work = Path(__file__).resolve().parent
    previous = work.parent / "stability_20260924_drive_damping02"
    frozen_numerics = work.parent / "stability_20260924_drive_damping01"
    for critical in (
        Path(__file__).resolve(), args.library.resolve(), args.contract.resolve(),
        *(work / name for name in ("body_speed_math_06.py", "body_speed_controller_06.py",
                                    "body_speed_validator_06.py", "body_speed_transfer_env_06.py")),
        previous / "drive_fresh_lifecycle_05.py",
        *(frozen_numerics / name for name in ("drive_damping_math_04.py",
                                               "drive_damping_controller_04.py",
                                               "drive_damping_validator_04.py")),
    ):
        if str(critical) not in files:
            raise ValueError(f"critical implementation absent from freeze: {critical}")
    old_freeze = previous / "root_execution_freeze_01.json"
    old_manifest = previous / "run_01_archive_manifest_01.json"
    old_closure = previous / "run_01_boundary_closure_01.json"
    for path, digest in ((old_freeze, PREVIOUS_FREEZE_SHA256),
                         (old_manifest, PREVIOUS_MANIFEST_SHA256),
                         (old_closure, PREVIOUS_CLOSURE_SHA256)):
        if _sha256(path) != digest or files.get(str(path), {}).get("sha256") != digest:
            raise ValueError(f"closed 05 boundary differs: {path}")
    old_inputs = _load(old_freeze)["files"]
    if len(old_inputs) != 662 or any(files.get(name) != row
                                     for name, row in old_inputs.items()):
        raise ValueError("frozen 05 input identity differs")
    old_outputs = _load(old_manifest)["files"]
    if len(old_outputs) != 77 or any(
        files.get(str(previous / "run_01" / relative)) != row
        for relative, row in old_outputs.items()
    ):
        raise ValueError("77 closed 05 output files differ")
    old_receipt = _load(previous / "run_01/drive_evaluation_receipt.json")
    if (old_receipt.get("actual_completed_controls") != 4800
            or old_receipt.get("execution_valid") is not True
            or old_receipt.get("candidate_qualified") is not False
            or _load(old_closure).get("reservation_closed") is not True):
        raise ValueError("05 valid but unqualified boundary misrepresented")
    baseline = Path(freeze["baseline_case_map"])
    if _sha256(baseline) != BASELINE_MAP_SHA256 or str(baseline) not in files:
        raise ValueError("fixed eight-case baseline map differs")
    for relative, digest in (
        ("training/final_checkpoint/model.zip", MODEL_SHA256),
        ("training/final_checkpoint/model.metadata.json", METADATA_SHA256),
        ("training/final_checkpoint/reload_probe.npz", PROBE_SHA256),
        ("protocol.json", ORIGINAL_PROTOCOL_SHA256),
    ):
        path = args.original_run / relative
        if _sha256(path) != digest or files.get(str(path.resolve()), {}).get("sha256") != digest:
            raise ValueError(f"original trained artifact differs: {relative}")
    return freeze


def _check_module_origins() -> dict[str, str]:
    """Pin unchanged 04/05 dependencies and the new 06 seam before construction."""
    import body_speed_controller_06
    import body_speed_math_06
    import body_speed_transfer_env_06
    import body_speed_validator_06
    import drive_damping_controller_04
    import drive_damping_math_04
    import drive_damping_validator_04
    import drive_fresh_lifecycle_05

    work = Path(__file__).resolve().parent
    old = work.parent / "stability_20260924_drive_damping01"
    previous = work.parent / "stability_20260924_drive_damping02"
    expected = {
        "body_speed_controller_06": work / "body_speed_controller_06.py",
        "body_speed_math_06": work / "body_speed_math_06.py",
        "body_speed_transfer_env_06": work / "body_speed_transfer_env_06.py",
        "body_speed_validator_06": work / "body_speed_validator_06.py",
        "drive_damping_controller_04": old / "drive_damping_controller_04.py",
        "drive_damping_math_04": old / "drive_damping_math_04.py",
        "drive_damping_validator_04": old / "drive_damping_validator_04.py",
        "drive_fresh_lifecycle_05": previous / "drive_fresh_lifecycle_05.py",
    }
    modules = (body_speed_controller_06, body_speed_math_06,
               body_speed_transfer_env_06, body_speed_validator_06,
               drive_damping_controller_04, drive_damping_math_04,
               drive_damping_validator_04, drive_fresh_lifecycle_05)
    for module in modules:
        if Path(module.__file__).resolve() != expected[module.__name__]:
            raise RuntimeError(f"06 source origin differs: {module.__name__}")
    return {key: str(value) for key, value in expected.items()}


def _reuse_reviewed_import_repair(args: argparse.Namespace, freeze: dict[str, Any]) -> None:
    """Run the already executed 03 bootstrap helper; no 03 engine main is called."""
    source = Path(__file__).resolve().parent.parent / (
        "stability_20260923_rl01/run_evaluation_continuation_03.py"
    )
    if freeze["files"].get(str(source), {}).get("sha256") != _sha256(source):
        raise RuntimeError("reviewed import repair source absent from freeze")
    spec = importlib.util.spec_from_file_location("frozen_continuation_import_repair", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("reviewed import repair module unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._check_and_restore_import_environment(args, freeze)


def _baseline_pair(row: dict[str, Any], freeze: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    folder = Path(row["source_dir"])
    path = folder / "states.npz"
    if freeze["files"].get(str(path), {}).get("sha256") != _sha256(path):
        raise ValueError(f"baseline state archive not frozen: {path}")
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key][0].copy() for key in PAIR_FIELDS}


def _frozen_baseline_score(row: dict[str, Any], freeze: dict[str, Any]) -> dict[str, Any]:
    path = Path(row["source_dir"]) / "score.json"
    if freeze["files"].get(str(path), {}).get("sha256") != _sha256(path):
        raise ValueError(f"historical score is not frozen: {path}")
    score = _load(path)
    if score.get("record_valid") is not True:
        raise ValueError(f"historical case record is invalid: {path}")
    return score


def _boundary(output: Path, name: str, stage: str, runtime: Any, monitor: Any,
              *, spec: Any, baseline: dict[str, Any], error: Any = None) -> None:
    _save(output / "evaluation" / f"{name}.{stage}.json", {
        "schema": "body-common-p-case-boundary-v1", "case": name, "stage": stage,
        "spec": spec.as_dict(), "actor": "final_policy", "baseline_source": baseline["source_dir"],
        "baseline_states_sha256": baseline["states_sha256"],
        "checkpoint_sha256": MODEL_SHA256,
        "binding_proof_source": "runtime_initial.json",
        "binding_proof_sha256": _sha256(output / "runtime_initial.json"),
        "C_state": runtime.state(), "python_ledger": runtime.ledger_receipt(),
        "native_monitor_checked": monitor.checked,
        "native_monitor_passed": monitor.passed, "error": error,
    })


def _validate_case(folder: Path, env: Any, spec: Any) -> dict[str, Any]:
    import numpy as np
    from body_speed_validator_06 import validate_body_precontrol_archive, validate_body_trace

    from scripts.d1_rolling_residual_task import raw_command_at_tick
    from scripts.d1_single_step_records import jsonable

    rows = []
    with gzip.open(folder / "trace.jsonl.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            rows.append(json.loads(line))
    with gzip.open(folder / "endpoints.jsonl.gz", "rt", encoding="utf-8") as stream:
        endpoints = [json.loads(line) for line in stream]
    with gzip.open(folder / "native.jsonl.gz", "rt", encoding="utf-8") as stream:
        native = [json.loads(line) for line in stream]
    before = jsonable(env.pre_control_states)
    _save(folder / "body_precontrol_states.json", before)
    if len(rows) != len(before):
        raise RuntimeError("pre-control tap and archived trace control counts differ")
    schedule = [raw_command_at_tick(spec, tick)["forward_velocity_mps"]
                for tick in range(len(rows))]
    stage = validate_body_trace(rows, expected_raw_schedule=schedule,
                                pre_control_states=before, native_rows=native)
    plant = env.unwrapped.plant
    with np.load(folder / "states.npz", allow_pickle=False) as saved:
        archive = validate_body_precontrol_archive(
            before, saved_qpos=saved["qpos"], saved_qvel=saved["qvel"],
            joint_qpos_addresses=plant.qpos_addresses,
            joint_dof_addresses=plant.dof_addresses,
            base_body_ipos_local_m=plant.model.body_ipos[plant.base_body_id],
            endpoints=endpoints,
        )
    receipt = {"schema": "body-common-p-case-stage-validation-v1",
               "stage": stage, "precontrol_archive": archive,
               "actual_compiled_joint_qpos_addresses": plant.qpos_addresses.tolist(),
               "actual_compiled_joint_dof_addresses": plant.dof_addresses.tolist(),
               "actual_compiled_base_body_ipos_local_m":
                   np.asarray(plant.model.body_ipos[plant.base_body_id]).tolist()}
    _save(folder / "body_stage_validation.json", receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--original-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.output.is_dir():
        raise ValueError("root launcher must precreate the exclusive output directory")
    freeze = _check_freeze(args)
    _save(args.output / "preflight_receipt.json", {
        "schema": "body-common-p-preflight-v1", "freeze_sha256": _sha256(args.freeze),
        "contract_sha256": CONTRACT_SHA256, "frozen_input_files": len(freeze["files"]),
        "training_controls": 0, "four_fixed_cases_only": True,
    })
    _reuse_reviewed_import_repair(args, freeze)

    # Everything below imports only inside the root-prepaid physical process.
    import mujoco
    import numpy as np
    from body_speed_transfer_env_06 import (
        BodyCommonPTransferredEnv,
        install_body_controller,
    )

    from scripts.d1_rolling_engine_runtime import RollingEngineRuntime
    from scripts.d1_rolling_native_monitor import NativeGeometryGuard
    from scripts.d1_rolling_residual_checkpoint import load_final_checkpoint
    from scripts.d1_rolling_residual_env import D1RollingResidualEnv
    from scripts.d1_rolling_residual_task import RollingEpisodeSpec
    from scripts.evaluate_d1_rolling_residual import _case
    from wheel_legged_control.d1.wheel_leg_controller import _nominal_geometry

    source_origins = _check_module_origins()
    _save(args.output / "source_origin_receipt.json", {
        "schema": "body-06-module-origins-v1", "sources": source_origins,
        "unchanged_04_leg_numerics": True,
        "unchanged_05_fresh_lifecycle": True,
        "checked_before_model_construction": True,
    })

    evaluation_dir = args.output / "evaluation"
    evaluation_dir.mkdir(exist_ok=False)
    mapped = _load(Path(freeze["baseline_case_map"]))
    all_baseline_rows = {row["case"]: row for row in mapped["cases"]}
    baseline_rows = {row["case"]: row for row in mapped["cases"]
                     if row["actor"] == "final_policy"}
    if tuple(baseline_rows) != tuple(item[0] for item in CASE_ORDER):
        raise ValueError("fixed old final-policy comparison order differs")
    baseline_scores = {
        name: _frozen_baseline_score(row, freeze)
        for name, row in all_baseline_rows.items()
    }
    warnings: list[str] = []
    runtime = monitor = None
    box_env = plane_env = None
    rows: list[dict[str, Any]] = []
    failure = None
    initial = None
    reload_delta = None
    cache_box = cache_plane = cache_transfer = None

    def warning(message: str) -> None:
        warnings.append(str(message))
        raise RuntimeError("MuJoCo warning: " + str(message))

    previous_warning = mujoco.get_mju_user_warning()
    try:
        mujoco.set_mju_user_warning(warning)
        with RollingEngineRuntime(args.library, control_limit=EVAL_LIMIT,
                                  construction_limit=CONSTRUCTION_LIMIT) as runtime:
            initial = runtime.state()
            _save(args.output / "runtime_initial.json", {
                "schema": "body-common-p-runtime-initial-v1",
                "binding_proof": runtime.proof,
                "initial_C_state_after_arm": initial,
                "initial_python_ledger": runtime.ledger_receipt(),
                "before_any_model_construction": True,
            })
            with NativeGeometryGuard(runtime, args.output) as monitor:
                original_box = runtime.construct(
                    lambda: D1RollingResidualEnv(RollingEpisodeSpec(.2, 175, 975, True)))
                box_env = original_box
                cache_box = _nominal_geometry.cache_info()._asdict()
                C_box = runtime.state()
                if (C_box["construction_attempts"] != 3
                        or C_box["construction_returns"] != 3
                        or cache_box != {"hits": 1, "misses": 1, "maxsize": 1, "currsize": 1}):
                    raise RuntimeError("original box construction/cache differs")
                original_plane = runtime.construct(
                    lambda: D1RollingResidualEnv(RollingEpisodeSpec(.25, 175, 975, False)))
                plane_env = original_plane
                cache_plane = _nominal_geometry.cache_info()._asdict()
                C_plane = runtime.state()
                if (C_plane["construction_attempts"] != 5
                        or C_plane["construction_returns"] != 5
                        or cache_plane != {"hits": 3, "misses": 1,
                                           "maxsize": 1, "currsize": 1}):
                    raise RuntimeError("original plane construction/cache differs")
                checkpoint = args.original_run / "training/final_checkpoint"
                native_before = runtime.ledger_receipt()["native_returned"]
                loaded = load_final_checkpoint(
                    checkpoint / "model.zip", checkpoint / "model.metadata.json", original_box,
                    expected_dso_sha256=DSO_SHA256,
                    expected_training_protocol_sha256=ORIGINAL_PROTOCOL_SHA256,
                )
                reload_delta = runtime.ledger_receipt()["native_returned"] - native_before
                if reload_delta != 0 or runtime.state() != C_plane:
                    raise RuntimeError("strict old checkpoint load touched native dynamics")
                _save(args.output / "strict_original_checkpoint_reload.json", {
                    "schema": "body-common-p-strict-old-loader-v1",
                    "model_sha256": MODEL_SHA256, "metadata_sha256": METADATA_SHA256,
                    "probe_sha256": PROBE_SHA256, "old_controller_schema_at_load":
                    original_box.unwrapped._controller.controller.control_schema,
                    "native_return_delta": reload_delta,
                    "controller_transfer_not_yet_performed": True,
                })
                box_transfer = install_body_controller(original_box)
                plane_transfer = install_body_controller(original_plane)
                cache_transfer = _nominal_geometry.cache_info()._asdict()
                if (cache_transfer != {"hits": 5, "misses": 1,
                                      "maxsize": 1, "currsize": 1}
                        or runtime.state() != C_plane
                        or runtime.ledger_receipt()["native_returned"] != native_before):
                    raise RuntimeError("controller transfer changed cache/dynamics")
                box_env = BodyCommonPTransferredEnv(original_box, box_transfer)
                plane_env = BodyCommonPTransferredEnv(original_plane, plane_transfer)
                _save(args.output / "controller_law_transfer.json", {
                    "schema": "d1-rolling-body-common-p-controller-transfer-v1",
                    "checkpoint_sha256": MODEL_SHA256, "box": box_transfer,
                    "plane": plane_transfer, "cache_after_box": cache_box,
                    "cache_after_plane": cache_plane,
                    "cache_after_both_transfers": cache_transfer,
                    "C_state_unchanged_after_checkpoint_and_transfer": True,
                    "new_training_or_optimizer_step": False,
                })
                for name, speed, terrain in CASE_ORDER:
                    spec = RollingEpisodeSpec(speed, 175, 975, terrain == "box")
                    env = box_env if terrain == "box" else plane_env
                    baseline = dict(baseline_rows[name])
                    baseline_path = Path(baseline["source_dir"]) / "states.npz"
                    baseline["states_sha256"] = _sha256(baseline_path)
                    paired = _baseline_pair(baseline, freeze)
                    previous_path = (Path(freeze["drive_baseline_directory"]) /
                                     "evaluation" / name / "states.npz")
                    previous = {"source_dir": str(previous_path.parent),
                                "states_sha256": _sha256(previous_path)}
                    previous_pair = _baseline_pair(previous, freeze)
                    if not all(np.array_equal(previous_pair[key], paired[key])
                               for key in PAIR_FIELDS):
                        raise RuntimeError("closed 05 initial state differs from historical case")
                    _boundary(args.output, name, "before", runtime, monitor,
                              spec=spec, baseline=baseline)
                    case_error = None
                    try:
                        score, paired_actual = _case(
                            runtime, env, loaded, evaluation_dir / name,
                            spec=spec, actor="final_policy", checkpoint_sha256=MODEL_SHA256,
                            paired_initial=paired,
                        )
                        if not all(np.array_equal(paired_actual[key], paired[key])
                                   for key in PAIR_FIELDS):
                            raise RuntimeError("baseline initial state pairing differs")
                        if not all(np.array_equal(paired_actual[key], previous_pair[key])
                                   for key in PAIR_FIELDS):
                            raise RuntimeError("closed 05 initial state pairing differs")
                        stage_validation = _validate_case(evaluation_dir / name, env, spec)
                    except BaseException as exc:  # preserve physical boundary
                        case_error = {"type": type(exc).__name__, "message": str(exc)}
                        raise
                    finally:
                        runtime.unbind()
                        _boundary(args.output, name, "after", runtime, monitor,
                                  spec=spec, baseline=baseline, error=case_error)
                    row = {"case": name, "speed_mps": speed, "terrain": terrain,
                           "actor": "final_policy", "score": score,
                           "body_stage_validation": stage_validation,
                           "paired_historical_initial_exact": True,
                           "paired_05_initial_exact": True,
                           "baseline_source": baseline["source_dir"],
                           "baseline_states_sha256": baseline["states_sha256"],
                           "previous_05_source": previous["source_dir"],
                           "previous_05_states_sha256": previous["states_sha256"],
                           "previous_05_score": _load(previous_path.parent / "score.json"),
                           "historical_zero_score":
                               baseline_scores[name.replace("_final_policy", "_zero")],
                           "historical_policy_score": baseline_scores[name],
                           "controller_law_transfer": True}
                    rows.append(row)
                    _save(evaluation_dir / f"{name}.case.json", row)
                    print(json.dumps({"event": "body_common_p_case_complete", "case": name,
                                      "task_passed": bool(score["task_passed"]),
                                      "completed_controls": score["metrics"]["completed_control_intervals"],
                                      "cumulative_controls": sum(item["score"]["metrics"]
                                                                 ["completed_control_intervals"]
                                                                 for item in rows)}), flush=True)
    except BaseException as exc:  # noqa: BLE001 -- entire reservation is consumed
        failure = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        mujoco.set_mju_user_warning(previous_warning)
        if runtime is not None:
            runtime.unbind()
        for env in (plane_env, box_env):
            if env is not None:
                env.close()

    C_state = None if runtime is None else runtime.state()
    ledger = None if runtime is None else runtime.ledger_receipt()
    native = None if monitor is None else monitor.report()
    scored_controls = sum(row["score"]["metrics"]["completed_control_intervals"]
                          for row in rows)
    actual_controls = None if ledger is None else ledger["control_completed"]
    segments = [] if ledger is None else ledger["segments"]
    segment_valid = bool(len(rows) == 4 and len(segments) == 4 and all(
        segment["name"] == row["case"] and segment["limit"] == CASE_LIMIT
        and segment["attempted"] == segment["completed"]
        == row["score"]["metrics"]["completed_control_intervals"]
        and segment["native_attempted"] == segment["native_returned"]
        == 5 * segment["completed"]
        for segment, row in zip(segments, rows, strict=True)
    ))
    counts_consistent = bool(
        failure is None and not warnings and C_state is not None and ledger is not None
        and native is not None and initial is not None
        and initial["construction_attempts"] == initial["construction_returns"] == 0
        and initial["control_attempts"] == initial["control_returns"] == 0
        and initial["ccd_attempts"] == initial["ccd_returns"] == 0
        and C_state["construction_attempts"] == C_state["construction_returns"] == 5
        and actual_controls == scored_controls
        and C_state["control_attempts"] == C_state["control_returns"] == 5 * actual_controls
        and C_state["ccd_attempts"] == C_state["ccd_returns"] > 0
        and C_state["native_ccd_caller_verified"]
        and C_state["native_construction_caller_verified"]
        and C_state["phase"] == 0 and C_state["target_model"] == C_state["target_data"] == 0
        and C_state["violations"] == 0
        and ledger["control_attempted"] == ledger["control_completed"] == actual_controls
        and ledger["native_attempted"] == ledger["native_returned"]
        == ledger["clock_advanced_substeps"] == 5 * actual_controls
        and ledger["native_failed"] == ledger["forbidden_entries"] == 0
        and ledger["fatal_latched"] is False
        and native["checked_native_returns"] == native["passed_native_returns"]
        == 5 * actual_controls and native["failure"] is None
        and segment_valid and reload_delta == 0
        and cache_box == {"hits": 1, "misses": 1, "maxsize": 1, "currsize": 1}
        and cache_plane == {"hits": 3, "misses": 1, "maxsize": 1, "currsize": 1}
        and cache_transfer == {"hits": 5, "misses": 1, "maxsize": 1, "currsize": 1}
    )
    sources_unchanged = all(Path(name).is_file() and Path(name).stat().st_size == entry["bytes"]
                            and _sha256(Path(name)) == entry["sha256"]
                            for name, entry in freeze["files"].items())
    execution_valid = bool(counts_consistent and sources_unchanged
                           and all(row["score"]["record_valid"]
                                   and row["body_stage_validation"]["stage"]["record_valid"]
                                   and row["body_stage_validation"]["precontrol_archive"]["record_valid"]
                                   for row in rows))
    candidate_qualified = bool(execution_valid and all(
        row["score"]["task_passed"]
        and row["score"]["metrics"]["completed_control_intervals"] == CASE_LIMIT
        for row in rows
    ))
    speed_deltas: dict[str, float | None] = {}
    if len(rows) == 4:
        for terrain in ("plane", "box"):
            values = {row["speed_mps"]: row["score"]["metrics"]
                      ["speed_window_mean_body_vx_mps"] for row in rows
                      if row["terrain"] == terrain}
            speed_deltas[terrain] = (None if values[.2] is None or values[.25] is None
                                     else float(values[.25] - values[.2]))
        candidate_qualified &= all(value is not None and value >= .03
                                    for value in speed_deltas.values())
    _save(args.output / "body_evaluation_receipt.json", {
        "schema": "body-common-p-four-case-evaluation-receipt-v1",
        "execution_valid": execution_valid, "candidate_qualified": candidate_qualified,
        "failure": failure, "warnings": warnings, "cases": rows,
        "actual_completed_controls": actual_controls,
        "sum_scored_case_controls": scored_controls,
        "actual_normal_native_returns": None if ledger is None else ledger["native_returned"],
        "actual_C_control_returns": None if C_state is None else C_state["control_returns"],
        "training_controls": 0, "checkpoint_sha256": MODEL_SHA256,
        "old_run_02_final_C_counts": None, "old_run_02_qualified": False,
        "C_state": C_state, "python_ledger": ledger, "native_monitor": native,
        "segment_counts_valid": segment_valid, "counts_consistent": counts_consistent,
        "source_hashes_unchanged": sources_unchanged,
        "cache_after_box": cache_box, "cache_after_plane": cache_plane,
        "cache_after_transfer": cache_transfer,
        "checkpoint_reload_native_delta": reload_delta,
        "policy_speed_mean_delta_mps": speed_deltas,
        "causal_rl_benefit_without_new_law_zero_control_claimed": False,
        "no_training_retry_padding_or_extra_cases": True,
    })
    return 0 if execution_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
