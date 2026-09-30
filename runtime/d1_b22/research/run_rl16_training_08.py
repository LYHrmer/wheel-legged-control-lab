"""One prepaid 08-R PPO training process and twelve matched heldout episodes.

The root launcher alone reserves, starts and monitors this process. Importing
this file is stdlib-only. Import repair, engine binding, PPO and model creation
occur only inside ``main`` after the exact session/source preflight.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import signal
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from scripts import run_d1_latest_rl as common

SCHEMA = "d1-course-t-rl16-training-heldout-08-v1"
CONTRACT_SHA = "475273fd4517263606af7abd07008cdd6811f9992e6affacf6eb758a9ccbddd7"
TRAIN_CONTROLS = 262144
HELDOUT_CONTROLS = 1600
HELDOUT_CASES = (
    "flat_0p6", "flat_1p6", "flat_1p2_yaw", "bumps_0p4", "rough_0p35", "ramp_0p35",
)
PAIR_FIELDS = ("qpos", "qvel", "ctrl", "qacc_warmstart", "observation")
CONTROL_RETURN_OFFSET_CLOSED_E = 777003


def _preflight(session: dict[str, Any], path: Path) -> Path:
    expected = {
        "schema": SCHEMA,
        "phase": "T",
        "control_limit": 281344,
        "normal_native_limit": 1406720,
        "compiler_native_limit": 2,
        "wallclock_limit_s": 7200,
        "segments": [TRAIN_CONTROLS, *([HELDOUT_CONTROLS] * 12)],
        "seed": 88401,
        "contract_sha256": CONTRACT_SHA,
        "retry_permitted": False,
    }
    if any(session.get(key) != value for key, value in expected.items()):
        raise RuntimeError("T identity or reserved budget differs from reviewed contract")
    if (session.get("training_control_limit") != TRAIN_CONTROLS
            or session.get("heldout_control_limit") != 12 * HELDOUT_CONTROLS
            or common._absolute(session, "session_path") != path):
        raise RuntimeError("T training/heldout split or session identity differs")
    if session.get("argv") != sys.argv or sys.argv[1:] != ["--session", str(path)]:
        raise RuntimeError("T worker argv differs from the unique launcher session")
    output = common._absolute(session, "output_directory")
    if not output.is_dir():
        raise RuntimeError("T output must be exclusively precreated by root launcher")
    if (type(session.get("run_id")) is not str or not session["run_id"]
            or type(session.get("wall_clock_utc")) is not str):
        raise RuntimeError("T run identity and UTC launcher clock are missing")
    sources = session.get("source_hashes")
    if not isinstance(sources, dict) or not sources:
        raise RuntimeError("T frozen source closure is missing")
    for name, record in sources.items():
        source = Path(name)
        if (not source.is_absolute() or source.is_symlink() or not source.is_file()
                or not isinstance(record, dict)
                or source.stat().st_size != record.get("bytes")
                or common._sha256(source) != record.get("sha256")):
            raise RuntimeError("T frozen input differs: " + name)
    critical = (
        Path(__file__).resolve(), common._absolute(session, "library"),
        common._absolute(session, "continuation_import_repair"),
        common._absolute(session, "preregistered_scoring_path"),
        common._absolute(session, "contract_path"),
        common._absolute(session, "binding_module"),
    )
    if any(str(source) not in sources for source in critical):
        raise RuntimeError("T critical worker/library/scoring input absent from freeze")
    for field in ("contract_documents", "dependency_hashes"):
        rows = session.get(field)
        if (not isinstance(rows, dict) or not rows
                or any(sources.get(name) != identity for name, identity in rows.items())):
            raise RuntimeError("T " + field + " is not a subset of the frozen source closure")
    if session["preregistered_scoring_path"] not in session["contract_documents"]:
        raise RuntimeError("preregistered scoring addendum is absent from contract documents")
    if (common._sha256(common._absolute(session, "contract_path")) != CONTRACT_SHA
            or common._sha256(common._absolute(session, "library")) != common.DSO_SHA256):
        raise RuntimeError("T contract or isolated DSO identity differs")
    for key, value in session["runtime_environment"].items():
        if os.environ.get(key) != value:
            raise RuntimeError("T environment differs: " + key)
    if ("LD_LIBRARY_PATH" in os.environ or os.environ.get("LD_BIND_NOW") != "1"
            or os.environ.get("LD_PRELOAD") != session["library"]):
        raise RuntimeError("T isolated engine environment is invalid")
    return output


def _boundary(runtime: Any) -> dict[str, Any]:
    return common._jsonable({
        "C_state": runtime.state(), "python": runtime.ledger_receipt(),
    })


def _control_step_caller_verified(
    state: dict[str, Any], proof: dict[str, Any],
    sources: dict[str, Any], binding: Any,
) -> bool:
    """Bind the actual first Python control call to the closed E return site."""
    functions = str(binding.FUNCTIONS)
    caller = state.get("first_control_step_caller_dladdr")
    return bool(
        proof.get("passed") is True
        and isinstance(caller, dict)
        and caller.get("path") == functions
        and caller.get("offset") == CONTROL_RETURN_OFFSET_CLOSED_E
        and sources.get(functions, {}).get("sha256")
        == binding.ELF_SHA256[binding.FUNCTIONS]
        and proof.get("elf_hashes", {}).get(functions)
        == binding.ELF_SHA256[binding.FUNCTIONS]
        and any(
            slot.get("elf") == functions and slot.get("symbol") == "mj_step"
            and slot.get("passed") is True
            for slot in proof.get("jump_slots", ())
        )
    )


def _save_arrays(path: Path, **arrays: Any) -> None:
    import numpy as np

    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())


def _write_gzip_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("xb") as raw, gzip.GzipFile(
        fileobj=raw, mode="wb", mtime=0,
    ) as compressed:
        for row in rows:
            compressed.write((json.dumps(common._jsonable(row), sort_keys=True,
                                         allow_nan=False) + "\n").encode())
    with path.open("rb") as persisted:
        os.fsync(persisted.fileno())


def _sha_bytes(path: Path) -> dict[str, Any]:
    return {"sha256": common._sha256(path), "bytes": path.stat().st_size}


def _audited_learning_origins(session: dict[str, Any]) -> dict[str, Any]:
    """Bind loaded sources, including PyTorch's two documented module aliases."""
    prefixes = ("torch", "stable_baselines3", "numpy", "gymnasium")
    origins: dict[str, str] = {}
    aliases: dict[str, dict[str, Any]] = {}
    torch_aliases = {
        "torch.ops": ("torch._ops", "ops", "_Ops", "_ops.py"),
        "torch.classes": ("torch._classes", "classes", "_Classes", "_classes.py"),
    }
    for name, module in tuple(sys.modules.items()):
        if name in prefixes or name.startswith(tuple(prefix + "." for prefix in prefixes)):
            source = getattr(module, "__file__", None)
            if source is None:
                continue
            if name in torch_aliases:
                definition_name, attribute, class_name, synthetic_file = torch_aliases[name]
                root = sys.modules.get("torch")
                definition = sys.modules.get(definition_name)
                if (
                    root is None or definition is None
                    or source != synthetic_file
                    or module is not root.__dict__.get(attribute)
                    or module is not definition.__dict__.get(attribute)
                    or type(module) is not definition.__dict__.get(class_name)
                    or getattr(module, "__name__", None) != name
                ):
                    raise RuntimeError("PyTorch synthetic module alias identity differs: " + name)
                defining_file = getattr(definition, "__file__", None)
                if not isinstance(defining_file, str) or not Path(defining_file).is_absolute():
                    raise RuntimeError("PyTorch alias definition lacks an absolute source")
                origin = Path(defining_file).resolve(strict=True)
                aliases[name] = {
                    "synthetic_module_alias": True,
                    "synthetic_file": source,
                    "defining_module": definition_name,
                    "defining_class": class_name,
                    "identity_verified": True,
                    "defining_source": str(origin),
                }
            else:
                if not isinstance(source, str) or not Path(source).is_absolute():
                    raise RuntimeError("loaded learning dependency has relative origin: " + name)
                origin = Path(source).resolve(strict=True)
            if str(origin) not in session["source_hashes"]:
                raise RuntimeError("loaded learning dependency has unfrozen origin: " + name)
            origins[name] = str(origin)
    if not all(name in origins for name in prefixes):
        raise RuntimeError("one of torch/SB3/NumPy/Gym was not actually loaded")
    return {"module_origins": origins, "synthetic_module_aliases": aliases}


def _require_unchanged_reset(before: dict[str, Any], after: dict[str, Any]) -> None:
    for key in ("construction_attempts", "construction_returns", "control_attempts",
                "control_returns"):
        if before["C_state"][key] != after["C_state"][key]:
            raise RuntimeError("reset changed a counted construction/control call")
    for key in ("control_attempted", "control_completed", "native_attempted",
                "native_returned"):
        if before["python"][key] != after["python"][key]:
            raise RuntimeError("reset changed a Python control/native count")


def _capture_state(env: Any, observation: Any) -> dict[str, Any]:
    import numpy as np

    data = env.plant.data
    row = {name: np.asarray(getattr(data, name)).copy() for name in PAIR_FIELDS[:-1]}
    row["observation"] = np.asarray(observation, dtype=np.float32).copy()
    row["time"] = float(data.time)
    for name in PAIR_FIELDS:
        if not np.isfinite(row[name]).all():
            raise RuntimeError("nonfinite heldout initial/endpoint state: " + name)
    return row


def _numeric_actor_action(actor: str, model: Any, observation: Any):
    import numpy as np

    if actor == "zero":
        return np.zeros(16, dtype=np.float32), False
    action, state = model.predict(observation, deterministic=True)
    raw = np.asarray(action)
    if (state is not None or raw.shape != (16,) or raw.dtype.kind not in "fiu"
            or not np.isfinite(raw).all() or np.any(np.abs(raw) > 1.0)):
        raise RuntimeError("heldout final policy returned an invalid 16D action")
    return raw.astype(np.float32, copy=True), True


def _metadata(session: dict[str, Any], env: Any, runtime: Any,
              curriculum: Any, *, train_boundary: dict[str, Any]) -> dict[str, Any]:
    from full_drive_schedule_08 import FLAT_SPAWN_M
    from rl16_curriculum_08 import CURRICULUM_SCHEMA

    episode = env.episode_metadata
    return {
        "run_id": session["run_id"],
        "wall_clock_utc": session["wall_clock_utc"],
        "contract_documents": dict(session["contract_documents"]),
        "source_files": dict(session["source_hashes"]),
        "engine_elf": {
            "path": session["library"], "sha256": common.DSO_SHA256,
        },
        "dependency_hashes": dict(session["dependency_hashes"]),
        "task_schema": episode["task_schema"],
        "reward_schema": episode["reward_schema"],
        "observation_schema": episode["observation_schema"],
        "controller_schema": episode["controller_schema"],
        "control_loop_schema": episode["control_loop_schema"],
        "control_dt_s": episode["control_dt_s"],
        "physics_dt_s": episode["native_dt_s"] if "native_dt_s" in episode else episode["normal_dt_s"],
        "physics_steps_per_control": int(env.plant.physics_steps),
        "curriculum": {
            "schema": CURRICULUM_SCHEMA,
            "level_reached": curriculum.level,
            "closed_episodes": len(curriculum.closed_episodes),
            "completed_controls": curriculum.completed_controls,
            "numeric_record_schema": "d1-course-rl16-1024-control-numeric-block-v1",
            "events_sha256": common._sha256(
                curriculum.output_dir / "curriculum_events.ndjson"
            ),
            "blocks_manifest_sha256": common._sha256(
                curriculum.output_dir / "training_blocks_manifest.json"
            ),
        },
        "command_seed": curriculum.schedule_seed,
        "measurement_seed_stream": curriculum.measurement_seed_stream_seed,
        "spawn_position_m": FLAT_SPAWN_M,
        "qualified_caps": asdict(curriculum.caps),
        "budget_ledger": train_boundary,
        "residual_permission_semantics": (
            "effective16=0 whenever consumed raw motion request is zero; "
            "the separate test_actor E0 authority exception is forbidden in training"
        ),
        "baseline_credit_note": (
            "speed capability and terrain effects require matched zero/final heldout; "
            "controller or schedule effects are not credited to RL"
        ),
    }


def _heldout_case(
    runtime: Any, guard: Any, env: Any, policy: Any, schedule: Any,
    actor: str, folder: Path, *, pair_initial: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """One genuine 1600-cap episode; early valid task failure remains a result."""
    import numpy as np

    if actor not in ("zero", "final_policy"):
        raise ValueError("heldout actor is not one of the frozen matched pair")
    folder.mkdir(exist_ok=False)
    name = f"heldout_{schedule.case_id}_{actor}"
    runtime.start_segment(name, HELDOUT_CONTROLS)
    runtime.bind(env)
    guard.start_segment(env.plant, folder, name, HELDOUT_CONTROLS, "heldout")
    common._save(folder / "boundary_before.json", _boundary(runtime))
    common._save(folder / "geometry_manifest.json", env.plant.collision_terrain_metadata)
    common._save(folder / "schedule.json", {
        "case_id": schedule.case_id, "actor": actor,
        "seed": schedule.seed, "terrain": schedule.terrain,
        "spawn_position_m": schedule.spawn_position_m,
        "raw_command_sha256": schedule.command_sha256,
        "raw_commands": [asdict(command) for command in schedule.raw_commands],
    })
    states: dict[str, list[Any]] = {key: [] for key in (*PAIR_FIELDS, "time")}
    rows: list[dict[str, Any]] = []
    row_files: list[dict[str, Any]] = []
    completed = 0
    predictions = 0
    done = False
    terminated = False
    truncated = False
    final_info: dict[str, Any] | None = None
    failure = None
    stop_reason = "control_cap"
    initial: dict[str, Any] | None = None
    started = time.monotonic()

    def snapshot(observation: Any) -> None:
        item = _capture_state(env, observation)
        for field, values in states.items():
            values.append(item[field])

    def flush_rows() -> None:
        if not rows:
            return
        index = len(row_files)
        path = folder / f"control_records_{index:04d}.jsonl.gz"
        _write_gzip_rows(path, rows)
        row_files.append({"file": path.name, "rows": len(rows), **_sha_bytes(path)})
        rows.clear()

    try:
        before_reset = _boundary(runtime)
        observation, reset_info = env.reset(
            seed=schedule.seed,
            options={
                "command_source": schedule.command_source,
                "spawn_position_m": schedule.spawn_position_m,
            },
        )
        after_reset = _boundary(runtime)
        _require_unchanged_reset(before_reset, after_reset)
        initial = _capture_state(env, observation)
        if pair_initial is not None:
            if any(
                initial[field].shape != pair_initial[field].shape
                or initial[field].dtype != pair_initial[field].dtype
                or initial[field].tobytes(order="C")
                != pair_initial[field].tobytes(order="C")
                for field in PAIR_FIELDS
            ):
                raise RuntimeError("matched zero/final heldout initial state differs")
            if (int(env.plant.model._address), int(env.plant.data._address)) != pair_initial["identity"]:
                raise RuntimeError("matched pair changed the compiled model/data identity")
        initial["identity"] = (int(env.plant.model._address), int(env.plant.data._address))
        common._save(folder / "reset_receipt.json", {
            "before": before_reset, "after": after_reset,
            "seed": schedule.seed, "episode_metadata": reset_info["episode_metadata"],
            "model_address": initial["identity"][0],
            "data_address": initial["identity"][1],
            "exact_initial_pair": pair_initial is not None,
            "paired_with": None if pair_initial is None else "zero",
        })
        _save_arrays(
            folder / "initial_state.npz",
            **{key: initial[key] for key in PAIR_FIELDS},
        )
        snapshot(observation)
        for tick in range(HELDOUT_CONTROLS):
            input_observation = np.asarray(observation, dtype=np.float32).copy()
            action, predicted = _numeric_actor_action(actor, policy, observation)
            predictions += int(predicted)
            started_control = time.perf_counter_ns()
            observation, reward, terminated, truncated, info = runtime.control_step(env, action)
            elapsed_ns = time.perf_counter_ns() - started_control
            completed += 1
            final_info = info
            traces = env.plant.last_control_interval_actuator_traces
            if len(traces) != 5 or info["native_interval_summary"]["native_returns"] != 5:
                raise RuntimeError("heldout control lacks five actual native/actuator traces")
            calc = info["controller_record"]["calculation"]
            if (not np.array_equal(calc["safe_torque_nm"], env.last_transition.requested_torque_nm)
                    or not np.array_equal(info["applied_action"], calc["applied_action"])):
                raise RuntimeError("heldout controller-to-physical torque/action link differs")
            rows.append({
                "tick": tick, "actor": actor,
                "policy_predict_called": predicted,
                "policy_input_action": action,
                "input_observation99": input_observation,
                "reward": float(reward),
                "terminated": terminated, "truncated": truncated,
                "control_wall_ns": elapsed_ns,
                "info": info,
                "native_actuator_traces": [asdict(trace) for trace in traces],
            })
            snapshot(observation)
            if len(rows) == 200:
                flush_rows()
            if terminated or truncated:
                done = True
                stop_reason = info["terminal_reason"]
                break
            if (tick + 1) % 200 == 0:
                print(name, tick + 1, "COM vx", info["metrics"]["body_com_vx_mps"], flush=True)
        if not done and completed == HELDOUT_CONTROLS:
            raise RuntimeError("heldout cap exhausted without a genuine terminal transition")
    except BaseException as error:
        failure = {"type": type(error).__name__, "message": str(error)}
        stop_reason = "exception"
        raise
    finally:
        runtime.unbind()
        guard_segment = guard.finish_segment()
        flush_rows()
        if states["time"]:
            _save_arrays(
                folder / "states.npz",
                **{key: np.asarray(values) for key, values in states.items()},
            )
        receipt = {
            "schema": "d1-course-rl16-heldout-episode-record-v1",
            "case_id": schedule.case_id, "actor": actor, "seed": schedule.seed,
            "terrain": schedule.terrain,
            "raw_command_sha256": schedule.command_sha256,
            "control_cap": HELDOUT_CONTROLS,
            "completed_controls": completed,
            "policy_predictions": predictions,
            "termination_recorded": done,
            "terminated": terminated, "truncated": truncated,
            "stop_reason": stop_reason, "failure": failure,
            "record_valid": failure is None and done and len(states["time"]) == completed + 1
            and guard_segment["native_attempted"] == 5 * completed
            and guard_segment["native_returned"] == 5 * completed,
            "full_horizon_without_task_termination": (
                failure is None and done and not terminated and truncated
            ),
            "initial_state_pair_exact": None if pair_initial is None else failure is None,
            "controller_record_blocks": row_files,
            "native_segment": guard_segment,
            "last_info_metrics": None if final_info is None else final_info["metrics"],
            "boundary_after": _boundary(runtime),
            "elapsed_wall_s": time.monotonic() - started,
            "full_physical_qualification_pending_independent_scoring": True,
        }
        common._save(folder / "case_receipt.json", receipt)
    if initial is None:
        raise RuntimeError("heldout case lacks a recorded initial state")
    return receipt, initial


def _train(
    session: dict[str, Any], runtime: Any, guard: Any, env: Any, output: Path,
) -> tuple[Any, Any, dict[str, Any], dict[str, Any], Any]:
    """One learn call, durable bounded rollout progress, one final model."""
    import numpy as np
    from rl16_curriculum_08 import RL16CurriculumWrapper
    from rl16_learning_08 import (
        build_audited_ppo,
        load_and_verify_final,
        make_rollout_callback,
        save_failure_checkpoint,
        save_final_and_verify,
        training_receipt,
    )

    expected_directory = Path(__file__).resolve().parent / "course_impl08"
    for name in ("rl16_curriculum_08", "rl16_learning_08"):
        origin = Path(sys.modules[name].__file__).resolve()
        if origin.parent != expected_directory or str(origin) not in session["source_hashes"]:
            raise RuntimeError("T curriculum/learning module has an unfrozen origin: " + name)

    folder = output / "training"
    folder.mkdir(exist_ok=False)
    common._save(folder / "learning_module_origins.json", {
        name: str(Path(sys.modules[name].__file__).resolve())
        for name in ("rl16_curriculum_08", "rl16_learning_08")
    })
    runtime.start_segment("training_262144", TRAIN_CONTROLS)
    runtime.bind(env)
    guard.start_segment(env.plant, folder, "training_262144", TRAIN_CONTROLS, "train")
    common._save(folder / "boundary_before.json", _boundary(runtime))
    curriculum = model = audit = None
    failure = None
    manifest = None
    loaded_model = None
    training_result = None
    completed_at_close = None
    started = time.monotonic()
    try:
        from full_drive_command_08 import QualifiedCommandCaps

        caps = QualifiedCommandCaps(1.6, 0.0, 0.0, 0.3, False)
        curriculum = RL16CurriculumWrapper(
            env, runtime, folder, caps=caps, schedule_seed=88401,
            measurement_seed=88402, total_controls=TRAIN_CONTROLS,
        )
        model, audit = build_audited_ppo(curriculum)
        common._save(folder / "ppo_construction.json", audit.construction)
        common._save(
            folder / "learning_dependency_origins_before_physics.json",
            _audited_learning_origins(session),
        )

        def rollout_progress(summary: dict[str, Any]) -> None:
            index = summary["rollout_index"]
            if (index != len(audit.rollouts) - 1
                    or curriculum.completed_controls != (index + 1) * 1024
                    or curriculum.gaussian_records != curriculum.completed_controls):
                raise RuntimeError("rollout/curriculum/progress count differs")
            common._save(folder / f"progress_rollout_{index:04d}.json", {
                "rollout": summary,
                "completed_controls": curriculum.completed_controls,
                "gaussian_records": curriculum.gaussian_records,
                "curriculum_level": curriculum.level,
                "closed_episodes": len(curriculum.closed_episodes),
                "zero_control_postbudget_resets": curriculum._postbudget_reset_count,
                "audit_transitions": audit.transitions,
                "audit_rollouts": audit.rollout_count,
                "train_calls_completed_before_pending_update": audit.train_calls,
                "optimizer_steps_before_pending_update": audit.optimizer_steps,
                "last_completed_update": audit.updates[-1] if audit.updates else None,
                "native_attempted": guard.attempted,
                "native_returned": guard.returned,
                "native_checked": guard.checked,
                "native_monitor_summary": {
                    "contacts_checked": guard.contacts_checked,
                    "nonwheel_ground_contacts": guard.nonwheel_contacts,
                    "max_abs_roll_deg": guard.max_abs_roll_deg,
                    "max_abs_pitch_deg": guard.max_abs_pitch_deg,
                    "failure": guard.failure,
                },
                "durable_block_files": {
                    "numeric": list(curriculum._numeric.files),
                    "gaussian": list(curriculum._gaussian.files),
                    "native": list(guard._segment["native_files"]),
                    "native_arrays": list(guard._segment["train_array_files"]),
                },
                "boundary": _boundary(runtime),
                "timing_semantics": (
                    "on_rollout_end fires after 1024 real transitions and before "
                    "the current rollout's PPO.train update"
                ),
            })
            if (index + 1) % 8 == 0:
                print(
                    "T rollout", index + 1, "controls", curriculum.completed_controls,
                    "updates so far", audit.train_calls, flush=True,
                )

        callback = make_rollout_callback(
            audit,
            on_transition=curriculum.record_transition,
            on_rollout=rollout_progress,
        )
        model.learn(total_timesteps=TRAIN_CONTROLS, callback=callback,
                    reset_num_timesteps=True)
        blocks = curriculum.finalize_training()
        if (blocks["completed_controls"] != TRAIN_CONTROLS
                or blocks["gaussian_records"] != TRAIN_CONTROLS
                or guard.returned != 5 * TRAIN_CONTROLS):
            raise RuntimeError("T learned control/native/action record count differs")
        training_result = training_receipt(model, audit)
        common._save(folder / "learning_receipt.json", training_result)
        if training_result["qualified_for_final_checkpoint"] is not True:
            raise RuntimeError("real PPO count/hash audit rejected the final checkpoint")
        if sum(row["gate_zeroed_steps"] for row in audit.rollouts) >= TRAIN_CONTROLS:
            raise RuntimeError("all stochastic residuals were gated to zero")

        first_block = folder / "training_numeric_blocks" / "controls_0000.npz"
        with np.load(first_block, allow_pickle=False) as recorded:
            probes = np.ascontiguousarray(
                recorded["input_observation99"][:32], dtype=np.float32,
            )
            indices = np.asarray(recorded["control_index"][:32])
        if (probes.shape != (32, 99) or not np.isfinite(probes).all()
                or not np.array_equal(indices, np.arange(32))):
            raise RuntimeError("first32 probe is not the first actual training observation")
        common._save(folder / "final_probe_origin.json", {
            "source_file": str(first_block),
            "source_file_sha256": common._sha256(first_block),
            "field": "input_observation99", "rows": [0, 32],
            "additional_physical_probe_calls": 0,
        })
        train_boundary = _boundary(runtime)
        checkpoint_folder = output / "final_checkpoint"
        manifest = save_final_and_verify(
            model, str(checkpoint_folder),
            _metadata(session, env, runtime, curriculum,
                      train_boundary=train_boundary),
            probes, audit=audit, receipt=training_result,
        )
        common._save(output / "final_checkpoint_manifest.json", manifest)
        loaded_model, loader_report = load_and_verify_final(
            str(checkpoint_folder), manifest, return_model=True,
        )
        common._save(output / "final_checkpoint_reload.json", loader_report)
        if loader_report["matches_training_final_policy_state"] is not True:
            raise RuntimeError("strictly reloaded heldout policy has another state hash")
        completed_at_close = curriculum.completed_controls
    except BaseException as error:
        failure = {"type": type(error).__name__, "message": str(error)}
        if curriculum is not None and not curriculum._finalized:
            try:
                curriculum.seal_partial("train_or_checkpoint_failure")
            except Exception as seal_error:  # noqa: BLE001 - preserve primary training failure
                failure["partial_seal_error"] = repr(seal_error)
        if model is not None and audit is not None:
            try:
                failed_receipt = training_receipt(model, audit)
                common._save(folder / "learning_failure_receipt.json", failed_receipt)
                if failed_receipt["status"] != "complete":
                    common._save(
                        folder / "failure_checkpoint_manifest.json",
                        save_failure_checkpoint(
                            model, str(output / "failure_checkpoint"), failed_receipt,
                        ),
                    )
            except Exception as save_error:  # noqa: BLE001 - preserve primary training failure
                failure["failure_checkpoint_error"] = repr(save_error)
        raise
    finally:
        runtime.unbind()
        guard_segment = guard.finish_segment()
        common._save(folder / "training_segment_receipt.json", {
            "failure": failure,
            "completed_controls": runtime.control_completed,
            "curriculum_completed_controls": None if curriculum is None else curriculum.completed_controls,
            "gaussian_records": None if curriculum is None else curriculum.gaussian_records,
            "closed_episodes": None if curriculum is None else len(curriculum.closed_episodes),
            "level_reached": None if curriculum is None else curriculum.level,
            "numeric_training_complete": completed_at_close == TRAIN_CONTROLS,
            "native_segment": guard_segment,
            "boundary_after": _boundary(runtime),
            "elapsed_wall_s": time.monotonic() - started,
        })
    if loaded_model is None or manifest is None or training_result is None or curriculum is None:
        raise RuntimeError("full T learning returned without a strict final model")
    return loaded_model, manifest, training_result, guard_segment, curriculum


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    args = parser.parse_args()
    session_path = args.session.resolve(strict=True)
    session = json.loads(session_path.read_text(encoding="utf-8"))
    if not isinstance(session, dict):
        raise TypeError("T session must be a JSON object")
    output = _preflight(session, session_path)
    common._save(output / "worker_preflight.json", {
        "schema": SCHEMA,
        "before_engine_or_learning_import": True,
        "frozen_input_count": len(session["source_hashes"]),
        "preregistered_scoring_path": session["preregistered_scoring_path"],
    })
    common._load_import_repair(session, output)
    if ("LD_LIBRARY_PATH" in os.environ or os.environ.get("LD_BIND_NOW") != "1"
            or os.environ.get("LD_PRELOAD") != session["library"]):
        raise RuntimeError("T import repair did not restore the isolated engine environment")

    import engine_binding
    import mujoco
    from course_native_guard_08 import CourseNativeGuard
    from full_drive_command_08 import FullDriveCommand, QualifiedCommandCaps
    from full_drive_env_08 import FullDriveCourseEnv
    from full_drive_schedule_08 import HELDOUT_CASES as REGISTERED_CASES
    from full_drive_schedule_08 import heldout_schedule
    from scripts.d1_rolling_engine_runtime import RollingEngineRuntime
    from wheel_legged_control.d1.wheel_leg_controller import _nominal_geometry

    expected_modules = Path(__file__).resolve().parent / "course_impl08"
    origins = {}
    if Path(engine_binding.__file__).resolve() != common._absolute(session, "binding_module"):
        raise RuntimeError("T engine binding helper has another frozen origin")
    origins["engine_binding"] = str(Path(engine_binding.__file__).resolve())
    for name, module in tuple(sys.modules.items()):
        if name.endswith("_08") and getattr(module, "__file__", None):
            origin = Path(module.__file__).resolve()
            if (origin.parent != expected_modules
                    or str(origin) not in session["source_hashes"]):
                raise RuntimeError("T course/learning source has another origin: " + name)
            origins[name] = str(origin)
    common._save(output / "module_origins.json", origins)
    if tuple(name for name, *_ in REGISTERED_CASES) != HELDOUT_CASES:
        raise RuntimeError("heldout schedule order differs from preregistration")
    cold = _nominal_geometry.cache_info()._asdict()
    if cold["misses"] != 0 or cold["currsize"] != 0:
        raise RuntimeError("T nominal geometry cache was not cold before construction")
    common._save(output / "nominal_cache_cold.json", cold)

    warnings: list[str] = []

    def warning(message: str) -> None:
        warnings.append(str(message))
        raise RuntimeError("MuJoCo warning: " + str(message))

    def terminated(_sig: int, _frame: Any) -> None:
        raise common.TerminationRequested("T interrupted or 7200s wall limit reached")

    old_warning = mujoco.get_mju_user_warning()
    old_term = signal.signal(signal.SIGTERM, terminated)
    runtime = guard = env = None
    training_result = checkpoint_manifest = None
    training_guard_segment = None
    cases: list[dict[str, Any]] = []
    failure = None
    started = time.monotonic()
    try:
        mujoco.set_mju_user_warning(warning)
        with RollingEngineRuntime(
            Path(session["library"]), control_limit=281344, construction_limit=2,
        ) as runtime:
            common._save(output / "runtime_initial.json", {
                "proof": runtime.proof, **_boundary(runtime),
            })
            with CourseNativeGuard(runtime) as guard:
                caps = QualifiedCommandCaps(1.6, 0.0, 0.0, 0.3, False)
                env = runtime.construct(lambda: FullDriveCourseEnv(
                    caps=caps,
                    command_source=lambda _tick, _time: FullDriveCommand(),
                    spawn_position_m=(-8.0, -4.7, 0.455),
                    max_steps=1600, mode="train",
                ))
                constructed = runtime.state()
                cache = _nominal_geometry.cache_info()._asdict()
                if (constructed["construction_attempts"] != 2
                        or constructed["construction_returns"] != 2
                        or cache["misses"] != 1 or cache["currsize"] != 1
                        or env.pure_test_components):
                    raise RuntimeError("T construction was not course+nominal cold2")
                common._save(output / "construction_receipt.json", {
                    "C_state": constructed,
                    "nominal_cache": cache,
                    "course_plant_model_address": int(env.plant.model._address),
                    "course_plant_data_address": int(env.plant.data._address),
                    "compiled_geometry": env.plant.collision_terrain_metadata,
                    "action_schema": env.controller.action_schema,
                    "control_schema": env.controller.control_schema,
                    "actual_geometry_binding": {
                        "base_body_id": int(env.plant.base_body_id),
                        "base_body_ipos": env.plant.model.body_ipos[
                            env.plant.base_body_id
                        ].copy(),
                        "wheel_index_by_body_id": dict(env.plant._wheel_index_by_body_id),
                        "joint_ids": env.plant.joint_ids.copy(),
                        "joint_qpos_addresses": env.plant.qpos_addresses.copy(),
                        "joint_dof_addresses": env.plant.dof_addresses.copy(),
                        "actuator_ids": env.plant.actuator_ids.copy(),
                        "actuator_trntype": env.plant.model.actuator_trntype.copy(),
                        "actuator_trnid": env.plant.model.actuator_trnid.copy(),
                        "actuator_gear": env.plant.model.actuator_gear.copy(),
                        "body_mass": env.plant.model.body_mass.copy(),
                        "nominal_total_mass_kg": env.plant.nominal_total_mass_kg,
                        "geom_bodyid": env.plant.model.geom_bodyid.copy(),
                        "geom_type": env.plant.model.geom_type.copy(),
                        "geom_size": env.plant.model.geom_size.copy(),
                        "geom_contype": env.plant.model.geom_contype.copy(),
                        "geom_conaffinity": env.plant.model.geom_conaffinity.copy(),
                    },
                })
                env.set_native_interval_reader(
                    guard.interval_summary, begin_interval=guard.begin_interval,
                )
                (loaded_policy, checkpoint_manifest, training_result,
                 training_guard_segment, curriculum) = _train(
                    session, runtime, guard, env, output,
                )
                env.mode = "eval"
                env.controller.mode = "eval"
                common._save(
                    output / "learning_dependency_origins_before_heldout.json",
                    _audited_learning_origins(session),
                )
                common._save(output / "heldout_mode_transition.json", {
                    "one_actual_model_data_pair": [
                        int(env.plant.model._address), int(env.plant.data._address),
                    ],
                    "from": "train", "to": "eval",
                    "numerical_controller_unchanged": True,
                    "boundary": _boundary(runtime),
                })
                for name in HELDOUT_CASES:
                    schedule = heldout_schedule(name)
                    pair_initial = None
                    for actor in ("zero", "final_policy"):
                        folder = output / "heldout" / f"{name}_{actor}"
                        folder.parent.mkdir(exist_ok=True)
                        receipt, initial = _heldout_case(
                            runtime, guard, env, loaded_policy, schedule, actor,
                            folder, pair_initial=pair_initial,
                        )
                        cases.append(receipt)
                        common._save(
                            output / f"heldout_progress_{len(cases):02d}.json",
                            {
                                "case": name, "actor": actor,
                                "completed_cases": len(cases),
                                "actual_training_controls": curriculum.completed_controls,
                                "case_receipt": str(folder / "case_receipt.json"),
                                "record_valid": receipt["record_valid"],
                                "boundary": _boundary(runtime),
                            },
                        )
                        if not receipt["record_valid"]:
                            raise RuntimeError("heldout record invalid; no later case may run")
                        if actor == "zero":
                            pair_initial = initial
                    if pair_initial is None:
                        raise RuntimeError("heldout zero/policy pair lacks first initial state")
                if (len(cases) != 12 or warnings
                        or runtime.control_completed
                        != TRAIN_CONTROLS + sum(row["completed_controls"] for row in cases)
                        or runtime.control_attempted != runtime.control_completed
                        or runtime.returned != 5 * runtime.control_completed
                        or runtime.attempted != runtime.returned
                        or runtime.failed != 0
                        or runtime.advanced_substeps != runtime.returned
                        or guard.returned != runtime.returned
                        or guard.attempted != guard.returned
                        or guard.checked != guard.returned
                        or guard.failure is not None
                        or runtime.state()["control_returns"] != 5 * runtime.control_completed
                        or runtime.state()["control_attempts"] != 5 * runtime.control_completed
                        or runtime.state()["construction_attempts"] != 2
                        or runtime.state()["construction_returns"] != 2
                        or runtime.state()["native_construction_caller_verified"] is not True
                        or not _control_step_caller_verified(
                            runtime.state(), runtime.proof, session["source_hashes"],
                            engine_binding,
                        )
                        or runtime.state()["ccd_attempts"] != runtime.state()["ccd_returns"]
                        or (runtime.state()["ccd_attempts"] > 0
                            and runtime.state()["native_ccd_caller_verified"] is not True)
                        or runtime.state()["violations"] != 0
                        or runtime.ledger_receipt()["forbidden_entries"] != 0
                        or _nominal_geometry.cache_info().misses != 1):
                    raise RuntimeError("T final actual C/Python/heldout counts differ")
    except BaseException as error:
        failure = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        signal.signal(signal.SIGTERM, old_term)
        mujoco.set_mju_user_warning(old_warning)
        if runtime is not None:
            runtime.unbind()
        final_state = None if runtime is None else runtime.state()
        control_caller_verified = (
            False if final_state is None else _control_step_caller_verified(
                final_state, runtime.proof, session["source_hashes"], engine_binding,
            )
        )
        complete = (
            failure is None and training_result is not None
            and training_result["qualified_for_final_checkpoint"] is True
            and checkpoint_manifest is not None
            and len(cases) == 12 and all(row["record_valid"] for row in cases)
            and not warnings
            and control_caller_verified
            and final_state is not None
            and final_state["phase"] == 0
            and final_state["target_model"] == 0
            and final_state["target_data"] == 0
        )
        common._save(output / "worker_receipt.json", {
            "schema": SCHEMA,
            "failure": failure, "warnings": warnings,
            "execution_complete": complete,
            "training_receipt": None if training_result is None
            else str(output / "training" / "learning_receipt.json"),
            "training_guard_segment": training_guard_segment,
            "final_checkpoint_manifest": None if checkpoint_manifest is None
            else str(output / "final_checkpoint_manifest.json"),
            "heldout_cases": [
                {key: row[key] for key in (
                    "case_id", "actor", "seed", "terrain", "completed_controls",
                    "policy_predictions", "record_valid", "terminated", "truncated",
                    "stop_reason",
                )} for row in cases
            ],
            "control_step_caller_verified_against_closed_E_callsite": control_caller_verified,
            "C_final": final_state,
            "python": None if runtime is None else runtime.ledger_receipt(),
            "native_guard": None if guard is None else guard.report(),
            "elapsed_wall_s": time.monotonic() - started,
            "retry_permitted": False,
            "full_physical_qualification_pending_independent_scoring": True,
            "RL_contribution_pending_matched_pair_scoring": True,
        })
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
