"""One prepaid five-case continuation of the interrupted rolling evaluation.

This worker never trains and never writes to the interrupted run. The root
launcher exclusively reserves its engine budget before starting this process.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

CONTRACT_SHA256 = "6be5ebc4557f124db74c7f6441ae624e70eb066f6efc94923a4269f0160dc194"
DSO_SHA256 = "3ec7ec9a6a130b9e1fa153c800aab97b59d4692c24971027f02de42957f8fa9c"
ORIGINAL_PROTOCOL_SHA256 = "494eb82ca80a842c190bbbff9bc75fa10c2e7832789a2d9bce0785acc3684733"
MODEL_SHA256 = "4f59d795afbdec256ec17bb7256ea05a3ddd04561e9ba52beffb1de375196b2e"
METADATA_SHA256 = "44b199ea13989412d274955a9fd85d6e97e58ad708a0c7e3ab7a4085ae3daaf3"
PROBE_SHA256 = "2cb674f54c8c46d5555e5d3580810a003be8d50fa5a7c1a5526dd0f027bb6660"
OLD_FREEZE_SHA256 = "5e541f030fd59b38a328e88a71e5ae0c58d6fda27cf7aeed7324a762aa5f2ac2"
OLD_ARCHIVE_SHA256 = "38ad4b520aadc7d1bc126f1ff4312f2fc976fbb0e4d11e49e6b961069c661075"
OLD_BOUNDARY_SHA256 = "2ff4dff045a0155177e4b2f95161928eea1ece2da0cc7809a9f9a474599d1a71"
OLD_READBACK_SHA256 = "94ad99fd9408c613c85e58943f1a60fc6fcdd311060b3a418edf305dfdba8d17"
EVAL_CONTROLS = 6000
NORMAL_STEPS = 30000
CONSTRUCTION_STEPS = 5
CASE_CONTROLS = 1200
CASE_ORDER = (
    ("speed200_box_final_policy", .2, "box", "final_policy"),
    ("speed250_plane_zero", .25, "plane", "zero"),
    ("speed250_plane_final_policy", .25, "plane", "final_policy"),
    ("speed250_box_zero", .25, "box", "zero"),
    ("speed250_box_final_policy", .25, "box", "final_policy"),
)
OLD_COMPLETE_ORDER = (
    "speed200_plane_zero", "speed200_plane_final_policy", "speed200_box_zero"
)
PAIR_FIELDS = ("qpos", "qvel", "ctrl", "qacc_warmstart", "observation")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _save(path: Path, value: dict[str, Any]) -> None:
    """Exclusive durable receipt; a missing receipt remains a real missing event."""
    with path.open("x", encoding="utf-8") as target:
        json.dump(value, target, indent=2, sort_keys=True, allow_nan=False)
        target.write("\n")
        target.flush()
        os.fsync(target.fileno())


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _check_freeze(args: argparse.Namespace) -> dict[str, Any]:
    frozen = _load(args.freeze)
    expected = {
        "schema": "rolling-evaluation-continuation-freeze-v1",
        "astra_decision": "GO",
        "process_limit": 1,
        "training_control_limit": 0,
        "evaluation_control_limit": EVAL_CONTROLS,
        "normal_native_limit": NORMAL_STEPS,
        "compiler_native_limit": CONSTRUCTION_STEPS,
        "wallclock_limit_s": 600,
        "contract_path": str(args.contract.resolve()),
        "contract_sha256": CONTRACT_SHA256,
        "original_run": str(args.original_run.resolve()),
        "output_directory": str(args.output.resolve()),
        "original_freeze_sha256": OLD_FREEZE_SHA256,
        "original_archive_manifest_sha256": OLD_ARCHIVE_SHA256,
        "original_training_protocol_sha256": ORIGINAL_PROTOCOL_SHA256,
        "checkpoint_sha256": MODEL_SHA256,
        "old_original_run_02_qualification": False,
        "old_final_native_C_counts": None,
    }
    for key, want in expected.items():
        if type(frozen.get(key)) is not type(want) or frozen[key] != want:
            raise ValueError(f"continuation freeze field differs: {key}")
    if frozen.get("python_argv") != [str(Path(sys.argv[0]).resolve()), *sys.argv[1:]]:
        raise ValueError("actual Python argv differs from the root freeze")
    if (_sha256(args.contract) != CONTRACT_SHA256
            or _sha256(args.library) != DSO_SHA256):
        raise ValueError("reviewed contract or isolated engine differs")
    files = frozen.get("files")
    if not isinstance(files, dict) or len(files) < 457 + 55:
        raise ValueError("frozen source and archived-record manifest is incomplete")
    for name, entry in files.items():
        path = Path(name)
        if (not path.is_absolute() or not isinstance(entry, dict)
                or not path.is_file() or path.stat().st_size != entry.get("bytes")
                or _sha256(path) != entry.get("sha256")):
            raise ValueError(f"frozen input identity mismatch: {name}")
    for path in (Path(__file__).resolve(), args.library.resolve(), args.contract.resolve()):
        if str(path) not in files:
            raise ValueError(f"critical input absent from freeze: {path}")
    old_freeze = Path(frozen["original_freeze_path"])
    archive = Path(frozen["original_archive_manifest_path"])
    boundary = Path(frozen["prior_boundary_closure"])
    if (_sha256(old_freeze) != OLD_FREEZE_SHA256
            or _sha256(archive) != OLD_ARCHIVE_SHA256
            or _sha256(boundary) != OLD_BOUNDARY_SHA256):
        raise ValueError("original freeze/archive/interruption boundary differs")
    original_inputs = _load(old_freeze)["files"]
    archive_files = _load(archive)["files"]
    if len(original_inputs) != 457 or len(archive_files) != 55:
        raise ValueError("original frozen source or retained archive count changed")
    for name, row in original_inputs.items():
        if files.get(name) != row:
            raise ValueError(f"original frozen source entry changed: {name}")
    for relative, row in archive_files.items():
        path = args.original_run / relative
        if path.resolve().is_relative_to(args.original_run.resolve()) is False:
            raise ValueError("archive manifest contains a path outside original run")
        if files.get(str(path.resolve())) != row:
            raise ValueError(f"original archived file not frozen: {relative}")
    actual_old = {
        path.relative_to(args.original_run).as_posix()
        for path in args.original_run.rglob("*") if path.is_file()
    }
    if actual_old != set(archive_files):
        raise ValueError("original archived file set changed")
    checkpoint = args.original_run / "training/final_checkpoint"
    for name, digest in (
        ("model.zip", MODEL_SHA256), ("model.metadata.json", METADATA_SHA256),
        ("reload_probe.npz", PROBE_SHA256),
    ):
        if _sha256(checkpoint / name) != digest:
            raise ValueError(f"original final checkpoint changed: {name}")
    if _sha256(args.original_run / "protocol.json") != ORIGINAL_PROTOCOL_SHA256:
        raise ValueError("original training protocol changed")
    if _load(boundary).get("final_native_C_counts") is not None:
        raise ValueError("interrupted run unexpectedly claims final native C counts")
    readback = boundary.parent / "root_run_02_interrupted_readback_01.json"
    if _sha256(readback) != OLD_READBACK_SHA256:
        raise ValueError("interrupted offline readback changed")
    return frozen


def _check_and_restore_import_environment(args: argparse.Namespace,
                                          freeze: dict[str, Any]) -> dict[str, Any]:
    """Repeat the reviewed OpenCV loader fix before creating EngineGuard."""
    from scripts import bootstrap_d1_rolling_residual_study as bootstrap

    stage = "startup"
    startup = {
        "ld_library_path_absent": "LD_LIBRARY_PATH" not in os.environ,
        "cv2_absent": "cv2" not in sys.modules,
        "ld_preload": os.environ.get("LD_PRELOAD"),
        "ld_bind_now": os.environ.get("LD_BIND_NOW"),
        "python_version": list(sys.version_info[:3]),
        "pythonpath": os.environ.get("PYTHONPATH"),
    }
    observed = None
    try:
        if sys.version_info[:2] != (3, 10):
            raise RuntimeError("continuation requires frozen Python 3.10")
        if not startup["ld_library_path_absent"] or not startup["cv2_absent"]:
            raise RuntimeError("OpenCV/import environment was already mutated")
        if (startup["ld_preload"] != str(args.library.resolve())
                or startup["ld_bind_now"] != "1"):
            raise RuntimeError("fixed DSO preload and eager ELF binding are required")
        parts = (startup["pythonpath"] or "").split(":")
        if not parts or parts[0] != "/home/lyh/.local/lib/python3.10/site-packages":
            raise RuntimeError("installed MuJoCo site-packages must be first on PYTHONPATH")
        for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            if os.environ.get(key) != "1":
                raise RuntimeError(f"{key} must equal one")
        stage = "fixed_loader_hashes"
        cv2_files = bootstrap._required_hashes(freeze)
        stage = "original_dependency_graph"
        # isort: off -- fixed import order is the observed OpenCV mutation
        import mujoco
        import mujoco._functions
        import mujoco._rollout

        from wheel_legged_control.d1.wheel_leg_controller import _nominal_geometry

        cache_before = _nominal_geometry.cache_info()._asdict()
        if cache_before != {"hits": 0, "misses": 0, "maxsize": 1, "currsize": 0}:
            raise RuntimeError("nominal geometry cache was not cold")
        from scripts import d1_rolling_engine_runtime as runtime_module
        from scripts import d1_rolling_native_monitor as monitor_module
        from scripts import evaluate_d1_rolling_residual as eval_module
        from scripts import train_d1_rolling_residual_ppo as train_module
        cv2 = __import__("cv2")  # exactly last, after the original dependency graph
        # isort: on

        torch = train_module.torch

        if not all((mujoco, runtime_module, monitor_module, eval_module, train_module)):
            raise RuntimeError("frozen original dependency graph is incomplete")
        stage = "module_origins_and_cache"
        if (Path(mujoco.__file__).resolve().parent
                != Path("/home/lyh/.local/lib/python3.10/site-packages/mujoco")
                or Path(cv2.__file__) != bootstrap.CV2_DIR / "__init__.py"
                or Path(cv2._native.__file__) != bootstrap.CV2_DIR / "cv2.abi3.so"
                or _nominal_geometry.cache_info()._asdict() != cache_before
                or (bootstrap.CV2_DIR / "config-3.10.py").exists()):
            raise RuntimeError("frozen import origins or cold model cache differ")
        observed = os.environ.get("LD_LIBRARY_PATH")
        if observed != bootstrap.EXPECTED_MUTATION:
            raise RuntimeError("OpenCV LD_LIBRARY_PATH mutation differs")
        for name, row in cv2_files.items():
            path = Path(name)
            if path.stat().st_size != row["bytes"] or _sha256(path) != row["sha256"]:
                raise RuntimeError(f"OpenCV loader file changed during import: {name}")
        stage = "restore_environment"
        os.environ.pop("LD_LIBRARY_PATH")
        if ("LD_LIBRARY_PATH" in os.environ
                or os.environ.get("LD_PRELOAD") != startup["ld_preload"]
                or os.environ.get("LD_BIND_NOW") != "1"):
            raise RuntimeError("isolated engine environment not restored")
        torch.set_num_threads(1)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
            raise RuntimeError("PyTorch thread counts differ from one")
        receipt = {
            "schema": "rolling-eval-continuation-import-repair-v1",
            "status": "restored_before_engine_guard",
            "startup": startup,
            "cv2_files": cv2_files,
            "cv2_python_file": cv2.__file__,
            "cv2_native_file": cv2._native.__file__,
            "cv2_version_observed": getattr(cv2, "__version__", None),
            "observed_ld_library_path_after_import": observed,
            "restored_ld_library_path_key_absent": True,
            "nominal_geometry_cache_before": cache_before,
            "nominal_geometry_cache_after": _nominal_geometry.cache_info()._asdict(),
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "bootstrap_explicit_engine_guard_instantiation": False,
            "bootstrap_explicit_model_or_data_construction": False,
            "bootstrap_explicit_integrator_call": False,
            "native_zero_requires_runtime_initial_proof": True,
        }
        _save(args.output / "environment_mutation_receipt.json", receipt)
        return receipt
    except BaseException as exc:
        was_mutated = os.environ.get("LD_LIBRARY_PATH")
        os.environ.pop("LD_LIBRARY_PATH", None)
        _save(args.output / "environment_bootstrap_failure.json", {
            "schema": "rolling-eval-continuation-import-repair-v1",
            "status": "failed_before_engine_guard", "stage": stage,
            "failure": {"type": type(exc).__name__, "message": str(exc)},
            "startup": startup, "observed_ld_library_path_after_import": was_mutated,
            "restored_ld_library_path_key_absent": "LD_LIBRARY_PATH" not in os.environ,
        })
        raise


def _load_initial_pair(original_run: Path) -> dict[str, Any]:
    import numpy as np

    path = original_run / "evaluation/speed200_box_zero/states.npz"
    with np.load(path, allow_pickle=False) as states:
        initial = {key: states[key][0].copy() for key in PAIR_FIELDS}
        if any(states[key].shape[0] != CASE_CONTROLS + 1 for key in PAIR_FIELDS):
            raise ValueError("original box-zero state archive is incomplete")
    return initial


def _compare_prefix(original_run: Path, new_case: Path) -> dict[str, Any]:
    """Compare saved complete rows; never interpret the interrupted tail as final."""
    original = original_run / "evaluation/speed200_box_final_policy"
    targets = (("endpoints.jsonl.gz", 578), ("trace.jsonl.gz", 577),
               ("native.jsonl.gz", 2888))
    rows: list[dict[str, Any]] = []
    for name, count in targets:
        mismatch: dict[str, Any] | None = None
        compared = 0
        try:
            with gzip.open(original / name, "rt", encoding="utf-8") as prior, gzip.open(
                new_case / name, "rt", encoding="utf-8"
            ) as current:
                for index in range(count):
                    old_line, new_line = prior.readline(), current.readline()
                    if not old_line or not new_line:
                        mismatch = {"index": index, "kind": "missing complete row",
                                    "old_present": bool(old_line), "new_present": bool(new_line)}
                        break
                    old_row, new_row = json.loads(old_line), json.loads(new_line)
                    compared += 1
                    if old_row != new_row:
                        keys = sorted(set(old_row) | set(new_row))
                        field = next((key for key in keys if old_row.get(key) != new_row.get(key)),
                                     "<whole_row>")
                        mismatch = {
                            "index": index, "kind": "saved row value differs", "first_field": field,
                            "old_line_sha256": hashlib.sha256(old_line.encode()).hexdigest(),
                            "new_line_sha256": hashlib.sha256(new_line.encode()).hexdigest(),
                            "old_value_preview": repr(old_row.get(field))[:240],
                            "new_value_preview": repr(new_row.get(field))[:240],
                        }
                        break
        except (EOFError, OSError, ValueError, json.JSONDecodeError) as exc:
            mismatch = {"index": compared, "kind": "prefix read failed",
                        "error_type": type(exc).__name__, "error": str(exc)}
        rows.append({"file": name, "required_complete_rows": count,
                     "compared_rows": compared, "match": mismatch is None, "first_mismatch": mismatch})
    return {"schema": "interrupted-prefix-comparison-v1", "files": rows,
            "all_saved_complete_prefix_rows_match": all(row["match"] for row in rows),
            "diagnostic_only_no_result_selection": True}


def _source_hashes_unchanged(freeze: dict[str, Any]) -> bool:
    return all(Path(name).is_file() and Path(name).stat().st_size == row["bytes"]
               and _sha256(Path(name)) == row["sha256"]
               for name, row in freeze["files"].items())


def _case_receipt(output: Path, name: str, stage: str, *, runtime: Any,
                  monitor: Any, spec: Any, checkpoint: str,
                  error: dict[str, str] | None = None) -> dict[str, Any]:
    row = {
        "schema": "rolling-eval-continuation-case-boundary-v1",
        "case": name, "stage": stage, "spec": spec.as_dict(),
        "checkpoint_sha256": checkpoint,
        "C_state": runtime.state(), "python_ledger": runtime.ledger_receipt(),
        "native_monitor_checked": monitor.checked,
        "native_monitor_passed": monitor.passed,
        "binding_proof_sha256": _sha256(output / "runtime_initial.json"),
        "error": error,
    }
    _save(output / "evaluation" / f"{name}.{stage}.json", row)
    return row


def _stitch(original_run: Path, fresh: list[dict[str, Any]],
            prefix: dict[str, Any] | None) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for name in OLD_COMPLETE_ORDER:
        row = _load(original_run / "evaluation" / f"{name}.case.json")
        if row.get("case") != name or not row["score"]["record_valid"]:
            raise ValueError(f"original complete case record invalid: {name}")
        rows.append({**row, "source_process": "interrupted_run_02_complete_record"})
    rows.extend(fresh)
    expected = (*OLD_COMPLETE_ORDER, *(item[0] for item in CASE_ORDER))
    if tuple(row["case"] for row in rows) != expected:
        raise RuntimeError("stitched eight-case order differs from frozen protocol")
    policy_rows = [row for row in rows if row["actor"] == "final_policy"]
    speed_deltas: dict[str, float | None] = {}
    for terrain in ("plane", "box"):
        means = {row["speed_mps"]: row["score"]["metrics"]["speed_window_mean_body_vx_mps"]
                 for row in policy_rows if row["terrain"] == terrain}
        speed_deltas[terrain] = (None if means[.2] is None or means[.25] is None
                                 else float(means[.25] - means[.2]))
    four_policy_passed = len(policy_rows) == 4 and all(
        row["score"]["task_passed"] for row in policy_rows
    )
    speed_gate_passed = all(value is not None and value >= .03
                            for value in speed_deltas.values())
    return {
        "schema": "interrupted-rolling-study-eight-case-stitch-v1",
        "cases": rows, "source_processes": [row["source_process"] for row in rows],
        "original_run_02_study_qualified": False,
        "original_final_C_counts": None,
        "original_training_completed_from_saved_records": True,
        "original_three_cases_record_valid": True,
        "old_partial_fourth_case": "run_02/evaluation/speed200_box_final_policy",
        "old_partial_fourth_case_counted_as_complete": False,
        "interrupted_prefix_comparison": prefix,
        "policy_speed_mean_delta_mps": speed_deltas,
        "four_policy_task_gates_passed": four_policy_passed,
        "policy_speed_increase_gate_passed": speed_gate_passed,
        "aggregate_four_policy_task_and_speed_gates_passed": four_policy_passed and speed_gate_passed,
        "uninterrupted_global_accounting": False,
        "causal_rl_improvement_claimed": False,
        "scope": "one checkpoint, eight fixed simulation cases across two process boundaries",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--original-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.output.is_dir():
        raise ValueError("root launcher must precreate the new exclusive output directory")
    freeze = _check_freeze(args)
    _save(args.output / "preflight_receipt.json", {
        "schema": "rolling-eval-continuation-preflight-v1",
        "freeze_sha256": _sha256(args.freeze), "contract_sha256": CONTRACT_SHA256,
        "frozen_input_files": len(freeze["files"]),
        "original_archived_files": 55, "original_frozen_sources": 457,
        "old_training_repeated": False, "old_partial_case_reused_as_result": False,
    })
    _check_and_restore_import_environment(args, freeze)

    # All imports below occur only in the new root-prepaid engine process.
    import mujoco

    from scripts.d1_rolling_engine_runtime import RollingEngineRuntime
    from scripts.d1_rolling_native_monitor import NativeGeometryGuard
    from scripts.d1_rolling_residual_checkpoint import load_final_checkpoint
    from scripts.d1_rolling_residual_env import D1RollingResidualEnv
    from scripts.d1_rolling_residual_task import RollingEpisodeSpec
    from scripts.evaluate_d1_rolling_residual import _case
    from wheel_legged_control.d1.wheel_leg_controller import _nominal_geometry

    evaluation_dir = args.output / "evaluation"
    evaluation_dir.mkdir(exist_ok=False)
    warnings: list[str] = []
    runtime: RollingEngineRuntime | None = None
    monitor: NativeGeometryGuard | None = None
    box_env: D1RollingResidualEnv | None = None
    plane_env: D1RollingResidualEnv | None = None
    fresh_rows: list[dict[str, Any]] = []
    prefix: dict[str, Any] | None = None
    failure: dict[str, str] | None = None
    initial_state: dict[str, Any] | None = None
    cache_after_box = cache_after_plane = None
    reload_native_delta: int | None = None

    def warning(message: str) -> None:
        warnings.append(str(message))
        raise RuntimeError("MuJoCo warning: " + str(message))

    previous_warning = mujoco.get_mju_user_warning()
    try:
        mujoco.set_mju_user_warning(warning)
        with RollingEngineRuntime(
            args.library, control_limit=EVAL_CONTROLS,
            construction_limit=CONSTRUCTION_STEPS,
        ) as runtime:
            initial_state = runtime.state()
            _save(args.output / "runtime_initial.json", {
                "schema": "rolling-eval-continuation-runtime-initial-v1",
                "binding_proof": runtime.proof,
                "initial_C_state_after_arm": initial_state,
                "initial_python_ledger": runtime.ledger_receipt(),
                "before_any_model_construction": True,
            })
            with NativeGeometryGuard(runtime, args.output) as monitor:
                box_env = runtime.construct(
                    lambda: D1RollingResidualEnv(RollingEpisodeSpec(.2, 175, 975, True))
                )
                cache_after_box = _nominal_geometry.cache_info()._asdict()
                box_state = runtime.state()
                if (box_state["construction_attempts"] != 3
                        or box_state["construction_returns"] != 3
                        or cache_after_box != {"hits": 1, "misses": 1,
                                               "maxsize": 1, "currsize": 1}):
                    raise RuntimeError("first box construction/cache count differs from 3/1+1")
                plane_env = runtime.construct(
                    lambda: D1RollingResidualEnv(RollingEpisodeSpec(.25, 175, 975, False))
                )
                cache_after_plane = _nominal_geometry.cache_info()._asdict()
                plane_state = runtime.state()
                if (plane_state["construction_attempts"] != CONSTRUCTION_STEPS
                        or plane_state["construction_returns"] != CONSTRUCTION_STEPS
                        or cache_after_plane != {"hits": 3, "misses": 1,
                                                 "maxsize": 1, "currsize": 1}):
                    raise RuntimeError("second plane construction/cache count differs from 2/3+1")
                _save(args.output / "construction_receipt.json", {
                    "box_C_state_after_construction": box_state,
                    "plane_C_state_after_construction": plane_state,
                    "construction_deltas": [(3, 3), (2, 2)],
                    "cache_after_box": cache_after_box,
                    "cache_after_plane": cache_after_plane,
                })
                checkpoint = args.original_run / "training/final_checkpoint"
                before_reload = runtime.ledger_receipt()["native_returned"]
                loaded = load_final_checkpoint(
                    checkpoint / "model.zip", checkpoint / "model.metadata.json", box_env,
                    expected_dso_sha256=DSO_SHA256,
                    expected_training_protocol_sha256=ORIGINAL_PROTOCOL_SHA256,
                )
                reload_native_delta = runtime.ledger_receipt()["native_returned"] - before_reload
                if reload_native_delta != 0:
                    raise RuntimeError("checkpoint reload consumed a physical native step")
                _save(args.output / "checkpoint_reload_receipt.json", {
                    "model_sha256": MODEL_SHA256,
                    "metadata_sha256": METADATA_SHA256,
                    "probe_sha256": PROBE_SHA256,
                    "original_training_protocol_sha256": ORIGINAL_PROTOCOL_SHA256,
                    "native_return_delta": reload_native_delta,
                    "same_trained_model_no_new_checkpoint": True,
                })

                paired: dict[str, Any] | None = _load_initial_pair(args.original_run)
                for name, speed, terrain, actor in CASE_ORDER:
                    spec = RollingEpisodeSpec(speed, 175, 975, terrain == "box")
                    env = box_env if terrain == "box" else plane_env
                    _case_receipt(args.output, name, "before", runtime=runtime,
                                  monitor=monitor, spec=spec, checkpoint=MODEL_SHA256)
                    case_error: dict[str, str] | None = None
                    try:
                        score, initial = _case(
                            runtime, env, loaded, evaluation_dir / name,
                            spec=spec, actor=actor, checkpoint_sha256=MODEL_SHA256,
                            paired_initial=paired if actor == "final_policy" else None,
                        )
                    except BaseException as exc:
                        case_error = {"type": type(exc).__name__, "message": str(exc)}
                        raise
                    finally:
                        runtime.unbind()
                        _case_receipt(args.output, name, "after", runtime=runtime,
                                      monitor=monitor, spec=spec, checkpoint=MODEL_SHA256,
                                      error=case_error)
                    if actor == "zero":
                        paired = initial
                    row = {"case": name, "speed_mps": speed, "terrain": terrain,
                           "actor": actor, "score": score,
                           "paired_initial_exact": actor == "final_policy",
                           "source_process": "new_run_03_continuation"}
                    fresh_rows.append(row)
                    _save(evaluation_dir / f"{name}.case.json", row)
                    print(json.dumps({
                        "event": "continuation_case_complete", "case": name,
                        "task_passed": bool(score["task_passed"]),
                        "completed_controls": score["metrics"]["completed_control_intervals"],
                        "cumulative_completed_controls": sum(
                            item["score"]["metrics"]["completed_control_intervals"]
                            for item in fresh_rows
                        ),
                    }), flush=True)
                    if name == CASE_ORDER[0][0]:
                        prefix = _compare_prefix(args.original_run, evaluation_dir / name)
                        _save(args.output / "interrupted_prefix_comparison.json", prefix)
    except BaseException as exc:  # noqa: BLE001 -- preserve the prepaid physical boundary
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
    actual_controls = sum(row["score"]["metrics"]["completed_control_intervals"]
                          for row in fresh_rows)
    segments = [] if ledger is None else ledger["segments"]
    segment_counts_valid = bool(
        len(fresh_rows) == len(CASE_ORDER) and len(segments) == len(CASE_ORDER)
        and all(segment["name"] == row["case"] and segment["limit"] == CASE_CONTROLS
                and segment["attempted"] == segment["completed"]
                == row["score"]["metrics"]["completed_control_intervals"]
                and segment["native_attempted"] == segment["native_returned"]
                == 5 * segment["completed"]
                for segment, row in zip(segments, fresh_rows, strict=True))
    )
    counts_consistent = bool(
        failure is None and not warnings and C_state is not None and ledger is not None
        and native is not None and initial_state is not None
        and initial_state["construction_attempts"] == initial_state["construction_returns"] == 0
        and initial_state["control_attempts"] == initial_state["control_returns"] == 0
        and initial_state["ccd_attempts"] == initial_state["ccd_returns"] == 0
        and C_state["construction_attempts"] == C_state["construction_returns"] == 5
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
        and segment_counts_valid and reload_native_delta == 0
        and cache_after_box == {"hits": 1, "misses": 1, "maxsize": 1, "currsize": 1}
        and cache_after_plane == {"hits": 3, "misses": 1, "maxsize": 1, "currsize": 1}
    )
    unchanged = _source_hashes_unchanged(freeze)
    execution_valid = bool(counts_consistent and unchanged
                           and all(row["score"]["record_valid"] for row in fresh_rows))
    stitched = None
    if len(fresh_rows) == len(CASE_ORDER):
        try:
            stitched = _stitch(args.original_run, fresh_rows, prefix)
            stitched["new_five_case_execution_valid"] = execution_valid
            stitched["recovered_eight_case_functional_validation"] = bool(
                execution_valid and stitched["aggregate_four_policy_task_and_speed_gates_passed"]
            )
            _save(args.output / "eight_case_stitch.json", stitched)
        except BaseException as exc:  # noqa: BLE001 -- keep failure evidence
            failure = {"type": type(exc).__name__, "message": str(exc)}
            execution_valid = False
    _save(args.output / "continuation_receipt.json", {
        "schema": "rolling-eval-continuation-receipt-v1",
        "new_five_case_execution_valid": execution_valid,
        "failure": failure, "warnings": warnings,
        "original_run_02_study_qualified": False,
        "original_final_C_counts": None,
        "original_training_completed_from_saved_records": True,
        "original_three_cases_record_valid": True,
        "old_partial_fourth_case_counted_as_complete": False,
        "new_training_controls": 0,
        "fresh_completed_case_count": len(fresh_rows),
        "actual_new_evaluation_controls": actual_controls,
        "C_state": C_state, "python_ledger": ledger,
        "native_monitor": native, "segment_counts_valid": segment_counts_valid,
        "actual_counts_consistent": counts_consistent,
        "cache_after_box": cache_after_box, "cache_after_plane": cache_after_plane,
        "checkpoint_reload_native_delta": reload_native_delta,
        "sources_and_old_archive_unchanged": unchanged,
        "interrupted_prefix_comparison": prefix,
        "aggregate_four_policy_task_and_speed_gates_passed": (
            False if stitched is None else stitched["aggregate_four_policy_task_and_speed_gates_passed"]
        ),
        "recovered_eight_case_functional_validation": (
            False if stitched is None else stitched["recovered_eight_case_functional_validation"]
        ),
        "uninterrupted_global_accounting": False,
        "no_retraining_retries_or_padding": True,
    })
    return 0 if execution_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
