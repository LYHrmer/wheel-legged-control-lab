"""C20 one-final save with strict frozen11 reload/probe and KL receipt binding.
"""
from __future__ import annotations
import os
import json
from typing import Any
import numpy as np
from rl16_learning_11 import (
    WHEEL_ACTION_SCALE_RAD_S,
    ACTION_SCHEMA,
    ACTION_SIZE,
    FINAL_CHECKPOINT_SCHEMA,
    LEARNING_SCHEMA,
    MAX_PROBE_OBSERVATIONS,
    METADATA_FILE,
    MODEL_FILE,
    OBSERVATION_ABS_BOUND,
    OBSERVATION_SIZE,
    PROBE_ACTION_FILE,
    PROBE_OBS_FILE,
    _float32_batch,
    _fsync_dir,
    _fsync_file,
    _hash_bytes,
    _hash_file,
    _module_metadata_block,
    _validate_caller_metadata,
    _write_exclusive,
    load_and_verify_final,
)
from learning20 import (LearningAudit20, ClipAudit20, STAGE20_LEARNING_SCHEMA,
                        training_receipt20)

def save_final_and_verify(model: Any, folder: str, metadata: dict[str, Any],
                          probe_observations: Any, *, audit: LearningAudit20,
                          clip_audit: ClipAudit20,
                          receipt: dict[str, Any] | None = None) -> dict[str, Any]:
    """Save the one preselected final checkpoint and verify it by reloading.

    ``folder`` must not exist. Nothing is written unless ``training_receipt``
    reports ``complete``. Returns the trusted manifest; root must record it in
    its own ledger, because the on-disk sidecar is never its own trust root.
    """
    if not isinstance(audit, LearningAudit20) or not isinstance(clip_audit, ClipAudit20):
        raise TypeError("save_final_and_verify needs the C20 learning and clip audits")
    verified_receipt = training_receipt20(model, audit, clip_audit)
    if receipt is not None and receipt != verified_receipt:
        raise ValueError("provided training receipt differs from the actual audited run")
    final_receipt = verified_receipt
    if final_receipt.get("status") != "complete":
        raise RuntimeError(
            "refusing to save a final checkpoint: "
            f"{final_receipt.get('violations')}"
        )
    if (final_receipt.get("stage20_learning_schema") != STAGE20_LEARNING_SCHEMA
            or final_receipt.get("course_arm") != audit.arm
            or final_receipt.get("target_kl") != audit.target_kl
            or final_receipt["actual"]["optimizer_steps"] > 512):
        raise RuntimeError("C20 KL/Adam receipt differs before final save")
    caller_metadata = _validate_caller_metadata(metadata)
    if (caller_metadata.get("stage20_parentage", {}).get("arm") != audit.arm
            or caller_metadata.get("stage20_course_arm") != audit.arm
            or caller_metadata.get("stage20_stage") != audit.construction["stage"]):
        raise ValueError("C20 parentage/course/stage metadata differs")
    observations = _float32_batch(probe_observations, OBSERVATION_SIZE,
                                  "probe_observations", MAX_PROBE_OBSERVATIONS)
    if np.any(np.abs(observations) > np.float32(OBSERVATION_ABS_BOUND)):
        raise ValueError("probe observations must already lie in the +/-5 box")

    actions, state = model.predict(observations, deterministic=True)
    actions = np.ascontiguousarray(actions, dtype=np.float32)
    if (state is not None or actions.shape != (observations.shape[0], ACTION_SIZE)
            or not np.isfinite(actions).all()):
        raise RuntimeError("deterministic probe prediction has the wrong shape/state")

    os.mkdir(folder, 0o755)
    model_path = os.path.join(folder, MODEL_FILE)
    model.save(model_path, exclude=["learning_audit"])
    if not os.path.isfile(model_path):
        raise RuntimeError("SB3 did not write the expected final_model.zip")
    _fsync_file(model_path)
    _write_exclusive(os.path.join(folder, PROBE_OBS_FILE), observations.tobytes(order="C"))
    _write_exclusive(os.path.join(folder, PROBE_ACTION_FILE), actions.tobytes(order="C"))

    document = dict(caller_metadata)
    document.update(_module_metadata_block(
        model, observations.shape, actions.shape, audit,
    ))
    document["training_receipt"] = final_receipt
    document["stage20_clip_summary"] = {
        "schema": clip_audit.as_dict()["schema"],
        "course_arm": clip_audit.arm,
        "actual_optimizer_steps": clip_audit.optimizer_returned,
        "actual_evaluated_minibatches": len(clip_audit.evaluated_batches),
        "normal_kl_stop_train_calls": sum(row["normal_kl_stop"]
                                          for row in clip_audit.train_records),
    }
    files: dict[str, Any] = {}
    for name in (MODEL_FILE, PROBE_OBS_FILE, PROBE_ACTION_FILE):
        digest, size = _hash_file(os.path.join(folder, name))
        files[name] = {"sha256": digest, "bytes": size}
    document["files"] = files
    payload = json.dumps(document, sort_keys=True, indent=2, allow_nan=False).encode("utf-8")
    _write_exclusive(os.path.join(folder, METADATA_FILE), payload)
    _fsync_dir(folder)

    manifest = {
        "loader_schema": FINAL_CHECKPOINT_SCHEMA,
        "budget_spec": audit.budget.as_dict(),
        "budget_spec_sha256": audit.budget.canonical_sha256(),
        "folder": os.path.abspath(folder),
        "metadata_sha256": _hash_bytes(payload),
        "files": files,
        "stage20_learning_schema": STAGE20_LEARNING_SCHEMA,
        "stage20_course_arm": audit.arm,
        "stage20_stage": audit.construction["stage"],
        "actual_optimizer_steps": final_receipt["actual"]["optimizer_steps"],
        "trusted_fields": {
            "loader_schema": FINAL_CHECKPOINT_SCHEMA,
            "learning_schema": LEARNING_SCHEMA,
            "budget_spec": audit.budget.as_dict(),
            "budget_spec_sha256": audit.budget.canonical_sha256(),
            "observation_size": OBSERVATION_SIZE,
            "action_size": ACTION_SIZE,
            "action_schema": ACTION_SCHEMA,
            "observation_schema": caller_metadata["observation_schema"],
            "task_schema": caller_metadata["task_schema"],
            "reward_schema": caller_metadata["reward_schema"],
            "reference_schema": caller_metadata["reference_schema"],
            "reference_definition": caller_metadata["reference_definition"],
            "controller_schema": caller_metadata["controller_schema"],
            "control_loop_schema": caller_metadata["control_loop_schema"],
            "run_id": caller_metadata["run_id"],
            "ppo_seed": audit.ppo_seed,
            "wheel_action_scale_rad_s": float(WHEEL_ACTION_SCALE_RAD_S),
            "probe_observation_shape": list(observations.shape),
            "probe_action_shape": list(actions.shape),
        },
    }
    manifest["reload_verification"] = load_and_verify_final(
        folder, manifest,
    )
    json.dumps(manifest, sort_keys=True, allow_nan=False)
    return manifest
