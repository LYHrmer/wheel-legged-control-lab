"""Source-only 11 short-episode training phase for a future frozen worker.

The root launcher supplies the reviewed session, one counted runtime, one
already-constructed world-upright course env, native guard, budget, and frozen
episode plan. This module does not choose a curriculum or a heldout case table.
Importing it creates no model and performs no physical call.
"""

from __future__ import annotations

import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from scripts import run_d1_latest_rl as common


def _training_metadata(
    session: dict[str, Any], env: Any, runtime: Any, curriculum: Any,
    *, boundary: dict[str, Any],
) -> dict[str, Any]:
    """Bind the actual world task and the complete, selected short plan."""
    from rl16_learning_11 import WORLD_UPRIGHT_REFERENCE_DEFINITION
    from short_curriculum_11 import SHORT_BLOCK_SCHEMA, SHORT_CURRICULUM_SCHEMA
    from world_upright_course_11 import (
        WORLD_UPRIGHT_REFERENCE_SCHEMA,
        WORLD_UPRIGHT_REWARD_SCHEMA,
        WORLD_UPRIGHT_TASK_SCHEMA,
    )

    episode = env.episode_metadata
    if (episode.get("task_schema") != WORLD_UPRIGHT_TASK_SCHEMA
            or episode.get("reward_schema") != WORLD_UPRIGHT_REWARD_SCHEMA
            or episode.get("reference_schema") != WORLD_UPRIGHT_REFERENCE_SCHEMA
            or session.get("reference_definition") != WORLD_UPRIGHT_REFERENCE_DEFINITION):
        raise RuntimeError("training metadata differs from the actual world-upright task")
    return {
        "run_id": session["run_id"],
        "wall_clock_utc": session["wall_clock_utc"],
        "contract_documents": dict(session["contract_documents"]),
        "source_files": dict(session["source_hashes"]),
        "engine_elf": {"path": session["library"], "sha256": common.DSO_SHA256},
        "dependency_hashes": dict(session["dependency_hashes"]),
        "task_schema": episode["task_schema"],
        "reward_schema": episode["reward_schema"],
        "reference_schema": episode["reference_schema"],
        "reference_definition": session["reference_definition"],
        "observation_schema": episode["observation_schema"],
        "controller_schema": episode["controller_schema"],
        "control_loop_schema": episode["control_loop_schema"],
        "control_dt_s": episode["control_dt_s"],
        "physics_dt_s": episode.get("native_dt_s", episode.get("normal_dt_s")),
        "physics_steps_per_control": int(env.plant.physics_steps),
        "curriculum": {
            "schema": SHORT_CURRICULUM_SCHEMA,
            "record_schema": SHORT_BLOCK_SCHEMA,
            "plan_schema": curriculum.plan.schema,
            "plan_source_sha256": curriculum.plan.source_sha256,
            "closed_episodes": len(curriculum.closed_episodes),
            "completed_controls": curriculum.completed_controls,
            "events_sha256": common._sha256(curriculum.output_dir / "curriculum_events.ndjson"),
            "blocks_manifest_sha256": common._sha256(
                curriculum.output_dir / "training_blocks_manifest.json"
            ),
        },
        "command_seed": session["command_seed"],
        "measurement_seed_stream": curriculum.measurement_seed,
        "spawn_position_m": episode["spawn_position_m"],
        "qualified_caps": asdict(curriculum.caps),
        "budget_ledger": boundary,
        "residual_permission_semantics": (
            "effective16=0 whenever consumed raw motion request is zero; "
            "the separate test_actor E0 authority exception is forbidden in training"
        ),
        "baseline_credit_note": (
            "world-upright zero baseline and matched zero/final heldout are separate evidence; "
            "learning counts alone do not establish task conversion or RL contribution"
        ),
    }


def train_short_phase(
    session: dict[str, Any], runtime: Any, guard: Any, env: Any, output: Path,
    *, budget: Any, plan: Any, caps: Any, ppo_seed: int, measurement_seed: int,
) -> tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any], Any]:
    """Run one prepaid learn call, preserve failure prefix, and strictly reload.

    The future entrypoint must preflight every source hash and reserve the whole
    training and heldout budget before calling this function. No retries are
    possible here; an exception seals the numeric prefix and failure model.
    """
    import numpy as np
    from budget_spec_11 import BudgetSpec
    from rl16_learning_11 import (
        WORLD_UPRIGHT_REFERENCE_DEFINITION,
        build_audited_ppo,
        load_and_verify_final,
        make_rollout_callback,
        save_failure_checkpoint,
        save_final_and_verify,
        training_receipt,
    )
    from short_curriculum_11 import ShortCurriculumWrapper
    from short_episode_plan_11 import FrozenShortPlan
    from world_upright_course_11 import WorldUprightCourseEnv

    if (not isinstance(budget, BudgetSpec) or not isinstance(plan, FrozenShortPlan)
            or type(env) is not WorldUprightCourseEnv
            or budget.as_dict() != session.get("budget_spec")
            or budget.canonical_sha256() != session.get("budget_spec_sha256")
            or plan.schema != session.get("short_plan_schema")
            or plan.source_sha256 != session.get("short_plan_source_sha256")
            or session.get("reference_definition") != WORLD_UPRIGHT_REFERENCE_DEFINITION
            or type(ppo_seed) is not int or ppo_seed != session.get("ppo_seed")
            or type(measurement_seed) is not int
            or measurement_seed != session.get("measurement_seed")
            or type(session.get("command_seed")) is not int):
        raise ValueError("11 short training inputs differ from the explicit session")
    for name in ("rl16_learning_11", "short_curriculum_11", "short_episode_plan_11",
                 "world_upright_course_11"):
        source = Path(sys.modules[name].__file__).resolve()
        if str(source) not in session["source_hashes"]:
            raise RuntimeError("11 training source has an unfrozen origin: " + name)
    folder = output / "training"
    folder.mkdir(exist_ok=False)
    common._save(folder / "learning_module_origins.json", {
        name: str(Path(sys.modules[name].__file__).resolve())
        for name in ("rl16_learning_11", "short_curriculum_11",
                     "short_episode_plan_11", "world_upright_course_11")
    })
    segment_name = f"world_upright_short_training_{budget.total_controls}"
    runtime.start_segment(segment_name, budget.total_controls)
    runtime.bind(env)
    guard.start_segment(env.plant, folder, segment_name, budget.total_controls, "train")
    from run_rl16_training_08 import _audited_learning_origins, _boundary

    common._save(folder / "boundary_before.json", _boundary(runtime))
    curriculum = model = audit = None
    failure = manifest = loaded_model = training_result = completed_at_close = None
    started = time.monotonic()
    try:
        curriculum = ShortCurriculumWrapper(
            env, runtime, folder, budget=budget, plan=plan, caps=caps,
            measurement_seed=measurement_seed,
        )
        model, audit = build_audited_ppo(curriculum, budget=budget, ppo_seed=ppo_seed)
        common._save(folder / "ppo_construction.json", audit.construction)
        common._save(
            folder / "learning_dependency_origins_before_physics.json",
            _audited_learning_origins(session),
        )

        def rollout_progress(summary: dict[str, Any]) -> None:
            index = summary["rollout_index"]
            if (index != len(audit.rollouts) - 1
                    or curriculum.completed_controls != (index + 1) * budget.n_steps
                    or curriculum.gaussian_records != curriculum.completed_controls):
                raise RuntimeError("11 rollout/curriculum/Gaussian count differs")
            common._save(folder / f"progress_rollout_{index:04d}.json", {
                "rollout": summary,
                "completed_controls": curriculum.completed_controls,
                "gaussian_records": curriculum.gaussian_records,
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
                    "on_rollout_end follows actual transitions and precedes "
                    "this rollout's PPO.train update"
                ),
            })

        callback = make_rollout_callback(
            audit, on_transition=curriculum.record_transition,
            on_rollout=rollout_progress,
        )
        model.learn(total_timesteps=budget.total_controls, callback=callback,
                    reset_num_timesteps=True)
        blocks = curriculum.finalize_training()
        if (blocks["completed_controls"] != budget.total_controls
                or blocks["gaussian_records"] != budget.total_controls
                or guard.returned != 5 * budget.total_controls):
            raise RuntimeError("11 learned control/native/action count differs")
        training_result = training_receipt(model, audit)
        common._save(folder / "learning_receipt.json", training_result)
        if training_result["qualified_for_final_checkpoint"] is not True:
            raise RuntimeError("11 real PPO count/hash audit rejected final checkpoint")
        if sum(row["gate_zeroed_steps"] for row in audit.rollouts) >= budget.total_controls:
            raise RuntimeError("all stochastic residuals were gated to zero")
        first_block = folder / "training_numeric_blocks" / "controls_0000.npz"
        with np.load(first_block, allow_pickle=False) as recorded:
            probes = np.ascontiguousarray(recorded["input_observation99"][:32],
                                          dtype=np.float32)
            indices = np.asarray(recorded["control_index"][:32])
        if (probes.shape != (32, 99) or not np.isfinite(probes).all()
                or not np.array_equal(indices, np.arange(32))):
            raise RuntimeError("first32 probe differs from actual training inputs")
        common._save(folder / "final_probe_origin.json", {
            "source_file": str(first_block),
            "source_file_sha256": common._sha256(first_block),
            "field": "input_observation99", "rows": [0, 32],
            "additional_physical_probe_calls": 0,
        })
        boundary = _boundary(runtime)
        manifest = save_final_and_verify(
            model, str(output / "final_checkpoint"),
            _training_metadata(session, env, runtime, curriculum, boundary=boundary),
            probes, audit=audit, receipt=training_result,
        )
        common._save(output / "final_checkpoint_manifest.json", manifest)
        loaded_model, report = load_and_verify_final(
            str(output / "final_checkpoint"), manifest, return_model=True,
        )
        common._save(output / "final_checkpoint_reload.json", report)
        if report["matches_training_final_policy_state"] is not True:
            raise RuntimeError("strict reload produced another final policy state")
        completed_at_close = curriculum.completed_controls
    except BaseException as error:
        failure = {"type": type(error).__name__, "message": str(error)}
        if curriculum is not None and not curriculum._finalized:
            try:
                curriculum.seal_partial("train_or_checkpoint_failure")
            except Exception as seal_error:  # noqa: BLE001 - retain primary failure
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
            except Exception as save_error:  # noqa: BLE001 - retain primary failure
                failure["failure_checkpoint_error"] = repr(save_error)
        raise
    finally:
        runtime.unbind()
        segment = guard.finish_segment()
        common._save(folder / "training_segment_receipt.json", {
            "failure": failure,
            "completed_controls": runtime.control_completed,
            "curriculum_completed_controls": None if curriculum is None
            else curriculum.completed_controls,
            "gaussian_records": None if curriculum is None else curriculum.gaussian_records,
            "closed_episodes": None if curriculum is None else len(curriculum.closed_episodes),
            "numeric_training_complete": completed_at_close == budget.total_controls,
            "native_segment": segment,
            "boundary_after": _boundary(runtime),
            "elapsed_wall_s": time.monotonic() - started,
        })
    if (loaded_model is None or manifest is None or training_result is None
            or curriculum is None):
        raise RuntimeError("11 short training returned without a strict final model")
    return loaded_model, manifest, training_result, segment, curriculum


__all__ = ("train_short_phase",)
