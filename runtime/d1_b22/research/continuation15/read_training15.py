"""NumPy-only independent readback of the paired stage15 training arms."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


PARENT_POLICY = "c20a8841dcf2174c2db584edca951adaafa12688b026172669ef0e07ad955212"
PARENT_ADAM = "bd4c898d33f4a121489f393dfe0f083737305f76fb139ddef74db2bb94878bf9"
PARENT_ZIP = "6cf2db80be7b990efc8be40eff307e58351eae839970b0c78ce5c9b5193f8e70"


def need(ok, message):
    if not ok:
        raise ValueError(message)


def near(actual, expected, message, *, rtol=3e-5, atol=1e-8):
    need(np.allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=False), message)


def identity(path):
    h = hashlib.sha256()
    size = 0
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            size += len(block)
            h.update(block)
    return {"bytes": size, "sha256": h.hexdigest()}


def read_json(path):
    return json.loads(Path(path).read_text())


def committed(path):
    """Check archive13's adjacent manifest and this payload, without imports."""
    path = Path(path)
    sidecar = path.with_name(path.name + ".manifest.json")
    manifest = read_json(sidecar)
    need(manifest.get("schema") == "d1-archive-transaction-13-v1" and
         manifest.get("payloads"), f"uncommitted payload: {path}")
    for item in manifest["payloads"]:
        member = sidecar.parent / item["file"]
        need(member.name == item["file"] and identity(member) == {
            "bytes": item["bytes"], "sha256": item["sha256"]},
            f"archive manifest member differs: {member}")
    need(path.name in {item["file"] for item in manifest["payloads"]},
         f"payload absent from archive manifest: {path}")


def arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def gradient_digest(named):
    digest = hashlib.sha256()
    for name, value in sorted(named.items()):
        value = np.ascontiguousarray(value)
        need(np.isfinite(value).all(), f"nonfinite raw gradient: {name}")
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(repr(value.shape).encode("ascii") + b"\0")
        digest.update(value.tobytes())
    return digest.hexdigest()


def vector_norm(named):
    return math.sqrt(sum(float(np.sum(value.astype(np.float64) ** 2))
                         for value in named.values()))


def check_clip_event(event, arm):
    step = event["step"]
    need(step == event["global_step"] and event["rollout"] == step // 16 and
         event["epoch"] == step % 16 // 4 and event["minibatch"] == step % 4,
         f"clip step identity differs: {step}")
    pa, pc = event["pre_actor_l2"], event["pre_critic_l2"]
    qa, qc = event["post_actor_l2"], event["post_critic_l2"]
    need(all(math.isfinite(x) and x >= 0 for x in (pa, pc, qa, qc)),
         f"nonfinite clip norm: {step}")
    near(event["pre_joint_l2"], math.hypot(pa, pc), f"pre joint norm: {step}")
    near(event["post_joint_l2"], math.hypot(qa, qc), f"post joint norm: {step}")
    global_alpha = min(1., .5 / (math.hypot(pa, pc) + 1e-6))
    actor_alpha = global_alpha if arm == "global" else min(1., .5 / (pa + 1e-6))
    critic_alpha = global_alpha if arm == "global" else min(1., .5 / (pc + 1e-6))
    near(event["expected_global_alpha"], global_alpha, f"global alpha: {step}")
    near(event["hypothetical_global_alpha_on_same_pregradient"], global_alpha,
         f"hypothetical global alpha: {step}")
    near(event["expected_actor_alpha"], actor_alpha, f"actor alpha: {step}")
    near(event["expected_critic_alpha"], critic_alpha, f"critic alpha: {step}")
    near(qa, pa * actor_alpha, f"actual actor clip: {step}")
    near(qc, pc * critic_alpha, f"actual critic clip: {step}")
    near(event["actual_actor_scale"], qa / pa if pa else 1., f"actor ratio: {step}")
    near(event["actual_critic_scale"], qc / pc if pc else 1., f"critic ratio: {step}")
    need(qa <= .50001 and qc <= .50001 and
         event["post_joint_l2"] <= (.50001 if arm == "global" else math.sqrt(2) * .50001),
         f"clip cap differs: {step}")
    need(event["optimizer_attempted"] is True and event["optimizer_returned"] is True
         and event["missing_gradients"] == event["nonfinite_gradients"] == 0,
         f"optimizer/finite ledger differs: {step}")
    for group in ("actor", "critic"):
        need(math.isfinite(event[f"adam_{group}_delta_over_parameter_norm"]) and
             event[f"adam_{group}_delta_over_parameter_norm"] >= 0 and
             math.isfinite(event[f"adam_{group}_parameter_delta_l2"]) and
             event[f"adam_{group}_parameter_delta_l2"] >= 0,
             f"Adam parameter delta invalid: {step}:{group}")
    near(event["adam_joint_parameter_delta_l2"],
         math.hypot(event["adam_actor_parameter_delta_l2"],
                    event["adam_critic_parameter_delta_l2"]), f"Adam joint delta: {step}")
    for phase in ("pre", "post"):
        need(all(isinstance(event[f"{phase}_gradient_sha256"][group], str) and
                 len(event[f"{phase}_gradient_sha256"][group]) == 64 and
                 set(event[f"{phase}_gradient_sha256"][group]) <= set("0123456789abcdef")
                 for group in ("actor", "critic")), f"gradient hash missing: {step}")
    need(event["full_gradient_saved"] == (step % 16 in (0, 15)),
         f"fixed gradient sample choice differs: {step}")


def check_gradient_samples(folder, clip):
    records = clip["gradient_sample_records"]
    expected_steps = [step for step in range(256) if step % 16 in (0, 15)]
    need(len(records) == len(expected_steps) == 32, "full gradient sample count differs")
    for step, record in zip(expected_steps, records):
        path = Path(record["file"])
        need(path.resolve() == (folder / "gradients" /
             f"gradient_step_{step:04d}.npz").resolve(), f"gradient file path differs: {step}")
        need(record["identity"] == identity(path), f"gradient file identity differs: {step}")
        committed(path)
        raw = arrays(path)
        event = clip["events"][step]
        need(record["ids"] == {"global_step": step, "rollout": step // 16,
               "epoch": step % 16 // 4, "minibatch": step % 4},
             f"gradient file step ids differ: {step}")
        for phase in ("pre", "post"):
            for group in ("actor", "critic"):
                prefix = f"{phase}::{group}::"
                named = {key[len(prefix):]: value for key, value in raw.items()
                         if key.startswith(prefix)}
                need(set(named) == set(clip[f"{group}_parameter_names"]),
                     f"raw gradient parameter set differs: {step}:{phase}:{group}")
                near(vector_norm(named), event[f"{phase}_{group}_l2"],
                     f"raw gradient norm differs: {step}:{phase}:{group}")
                need(gradient_digest(named) == event[f"{phase}_gradient_sha256"][group],
                     f"raw gradient hash differs: {step}:{phase}:{group}")
        need(len(raw) == 26, f"raw gradient tensor count differs: {step}")


def find_session(folder, expected):
    candidates = list(folder.glob("*session*.json")) + list(folder.parent.glob("*session*.json"))
    candidates += list(folder.glob("*reservation*.json")) + list(folder.parent.glob("*reservation*.json"))
    for path in candidates:
        if identity(path) == expected:
            return path, read_json(path)
    raise ValueError("frozen session JSON matching worker receipt was not found")


def check_final(folder, learned, parentage):
    manifest_path = folder / "final_checkpoint_manifest.json"
    committed(manifest_path)
    manifest = read_json(manifest_path)
    checkpoint = folder / "final_checkpoint"
    metadata_path = checkpoint / "final_metadata.json"
    metadata = read_json(metadata_path)
    need(identity(metadata_path)["sha256"] == manifest["metadata_sha256"],
         "final metadata identity differs")
    need(metadata["training_receipt"] == learned, "checkpoint learning receipt differs")
    need(metadata["stage15_parentage"] == parentage, "checkpoint parentage differs")
    for name, expected in manifest["files"].items():
        need(identity(checkpoint / name) == expected, f"final checkpoint file differs: {name}")
    reload = manifest["reload_verification"]
    need(reload["matches_training_final_policy_state"] is True and
         reload["probe_actions_byte_exact"] is True and
         reload["verified_without_engine_or_reset"] is True and
         reload["loaded_policy_state_sha256"] == learned["final_hashes"]["policy_state"],
         "strict reload proof differs")
    return {"manifest_identity": identity(manifest_path),
            "model_identity": identity(checkpoint / "final_model.zip"),
            "reload_verification": manifest["reload_verification"]}


def validate_episode_timeline(numeric):
    episode = numeric["episode_index"]
    tick = numeric["episode_tick"]
    done = numeric["terminated"] | numeric["truncated"]
    n = len(episode)
    need(n > 0 and all(len(numeric[key]) == n for key in
                         ("episode_tick", "terminated", "truncated")),
         "first block episode fields differ in length")
    need(int(episode[0]) == 0 and int(tick[0]) == 0 and
         np.array_equal(numeric["control_index"], np.arange(n)),
         "first block does not start at control/episode/tick zero")
    starts = {0: 0}
    for i in range(1, n):
        if episode[i] == episode[i - 1]:
            need(tick[i] == tick[i - 1] + 1 and not done[i - 1],
                 f"episode continuity differs at first-block row {i}")
        else:
            need(episode[i] == episode[i - 1] + 1 and tick[i] == 0 and done[i - 1],
                 f"episode reset/done boundary differs at first-block row {i}")
            starts[int(episode[i])] = i
    return starts


def check_worker_counts(worker):
    c, python, native = worker["C_final"], worker["python"], worker["native_guard"]
    need(worker["control_step_caller_verified"] is True and
         c["construction_attempts"] == c["construction_returns"] == 2 and
         c["control_attempts"] == c["control_returns"] == 81920 and
         c["violations"] == 0 and c["phase"] == c["target_model"] ==
         c["target_data"] == 0,
         "native C construction/control ledger differs")
    need(python["control_attempted"] == python["control_completed"] == 16384 and
         python["native_attempted"] == python["native_returned"] ==
         python["clock_advanced_substeps"] == 81920 and
         python["native_failed"] == python["forbidden_entries"] == 0 and
         len(python["segments"]) == 1 and
         python["segments"][0]["attempted"] ==
         python["segments"][0]["completed"] == 16384 and
         python["segments"][0]["native_attempted"] ==
         python["segments"][0]["native_returned"] == 81920,
         "Python control/native ledger differs")
    need(native["native_attempted"] == native["native_returned"] ==
         native["native_checked"] == 81920 and native["failure"] is None and
         native["record_mode"] == "compact_train" and
         len(native["segments"]) == 1 and
         native["segments"][0]["mode"] == "train" and
         native["segments"][0]["failure"] is None and
         native["segments"][0]["partial_native_interval"] is False and
         native["segments"][0]["native_attempted"] ==
         native["segments"][0]["native_returned"] == 81920,
         "native guard ledger differs")
    calls = worker["model_calls"]
    counts = calls["counts"]
    exact = {"load": (2, None), "torch_load": (6, None),
             "save": (1, None), "learn": (1, None),
             "train": (16, None), "forward": (16384, 16384),
             "evaluate_actions": (256, 65536), "predict": (3, 96),
             "backward": (256, None)}
    for key, (number, rows) in exact.items():
        record = counts[key]
        need(record["attempted"] == record["returned"] == number and
             (rows is None or record["rows_attempted"] ==
              record["rows_returned"] == rows),
             f"model call count differs: {key}")
    value_record = counts["predict_values"]
    need(16 <= value_record["attempted"] == value_record["returned"] <= 32 and
         value_record["rows_attempted"] == value_record["rows_returned"] ==
         value_record["returned"], "value-only bootstrap model calls differ")
    need(calls["actor_rows"] == 82016 and
         calls["critic_rows"] == 81920 + value_record["returned"],
         "model actor/critic row ledger differs")
    need(counts["predict"]["phases"]["probe"]["returned"] == 3,
         "three fixed 32-row probes differ")


def check_first_block(folder):
    training = folder / "training"
    manifest_path = training / "training_blocks_manifest.json"
    committed(manifest_path)
    manifest = read_json(manifest_path)
    need(manifest["complete"] is True and manifest["completed_controls"] == 16384 and
         manifest["gaussian_records"] == 16384 and
         manifest["pending_gaussian_control_index"] is None,
         "training block manifest is not complete")
    data = {}
    for kind, subdir in (("numeric", "training_numeric_blocks"),
                         ("gaussian", "training_gaussian_blocks")):
        blocks = manifest[f"{kind}_blocks"]
        need(len(blocks) == 16 and all(b["rows"] == 1024 for b in blocks),
             f"{kind} block counts differ")
        for i, block in enumerate(blocks):
            need(block["first_control_index"] == i * 1024 and
                 block["last_control_index"] == i * 1024 + 1023,
                 f"{kind} block index range differs: {i}")
            path = training / subdir / block["file"]
            need(identity(path) == {"sha256": block["sha256"], "bytes": block["bytes"]},
                 f"{kind} block identity differs: {i}")
            committed(path)
        data[kind] = arrays(training / subdir / blocks[0]["file"])
        need(np.array_equal(data[kind]["control_index"], np.arange(1024)),
             f"{kind} first control indices differ")
    episode_starts = validate_episode_timeline(data["numeric"])
    episode_records = []
    for episode_id, first_row in episode_starts.items():
        reset_path = training / f"training_episode_{episode_id:06d}_reset.json"
        schedule_path = training / f"training_episode_{episode_id:06d}_schedule.json"
        committed(reset_path)
        committed(schedule_path)
        reset, schedule = read_json(reset_path), read_json(schedule_path)
        metadata = reset["actual_episode_metadata"]
        need(reset["episode_index"] == schedule["episode_index"] ==
             schedule["choice"]["episode_index"] == episode_id and
             schedule["episode_reset_seed"] == reset["episode_reset_seed"] and
             schedule["choice"]["terrain"] == schedule["terrain"] and
             schedule["start_completed_global_controls"] == first_row and
             metadata["episode_index"] == episode_id and
             metadata["measurement_seed"] == reset["provider_measurement_seed"] and
             metadata["geometry_ground_source"] ==
             "unaltered_compiled_course_vertical_hit",
             f"first-block episode {episode_id} reset/schedule geometry or seed differs")
        episode_records.append({"episode_index": episode_id, "first_control_index": first_row,
                                "reset": reset, "schedule": schedule})
    return {"numeric": data["numeric"], "gaussian": data["gaussian"],
            "episodes": episode_records,
            "first_block_hashes": {kind: manifest[f"{kind}_blocks"][0]["sha256"]
                                   for kind in ("numeric", "gaussian")}}


def verify_arm(folder, arm):
    report = {"arm": arm, "complete": False, "errors": []}
    try:
        worker = read_json(folder / "worker_receipt.json")
        need(worker["arm"] == arm and worker["execution_complete"] is True and
             worker["failure"] is None and worker["cleanup_errors"] == [] and
             worker["warnings"] == [] and worker["archive_failed"] is False,
             "worker did not close successfully")
        host = read_json(folder / "host_receipt.json")
        need(host["arm"] == arm and host["exit_code"] == 0 and
             host["failure"] is None and host["cleanup_errors"] == [] and
             host["changed_sources"] == [] and host["postcheck_complete"] is True and
             host["no_live_owned_processes"] is True and
             host["retry_permitted"] is False,
             "host source/process closeout differs")
        session_path, session = find_session(folder, worker["session_identity"])
        need(session["arm"] == arm and Path(session["output_directory"]).resolve() == folder.resolve()
             and session["retry_permitted"] is False,
             "session binding differs")
        need(0 <= host["elapsed_s"] <= session["hard_s"] and
             session["control_limit"] == 16384 and
             session["normal_native_limit"] == 81920 and
             session["compiler_native_limit"] == 2,
             "host elapsed time or session physical budget differs")
        plan_path = Path(session["plan_path"])
        need(identity(plan_path) == session["plan_identity"] and
             read_json(plan_path)["schema"] == "d1-groupclip-plan-15-v1",
             "frozen source plan differs")
        for filename in ("learning15.py", "train15.py", "worker15.py", "recipes15.py"):
            path = folder.parent / filename
            need(session["source_hashes"].get(str(path.resolve())) == identity(path),
                 f"frozen small source differs: {filename}")
        parentage_path = folder / "parentage.json"
        clip_path = folder / "clip_audit.json"
        learning_path = folder / "training" / "learning_receipt.json"
        for path in (parentage_path, clip_path, learning_path):
            committed(path)
        parentage, clip, learned = (read_json(p) for p in
                                    (parentage_path, clip_path, learning_path))
        need(parentage["parent_model_identity"]["sha256"] == PARENT_ZIP and
             parentage["parent_hashes"]["policy_state"] == PARENT_POLICY and
             parentage["parent_hashes"]["optimizer_state"] == PARENT_ADAM and
             parentage["parent_optimizer_steps"] == 1024 and
             parentage["parent_num_timesteps"] == 65536 and
             parentage["parent_n_updates"] == 256 and
             parentage["stage_seed"] == 151001 and parentage["arm"] == arm and
             parentage["actor_reinitialized"] is False and
             parentage["optimizer_reinitialized"] is False,
             "warm-start parentage differs")
        need(learned["status"] == "complete" and
             learned["initial_hashes"] == parentage["parent_hashes"] and
             learned["actual"] == {"num_timesteps": 16384, "train_calls": 16,
                                   "epochs": 64, "optimizer_steps": 256,
                                   "rollouts": 16, "transitions": 16384} and
             learned["qualified_for_final_checkpoint"] is True,
             "learning audit incomplete or parent hash differs")
        need(clip["arm"] == arm and clip["clip_attempted"] == clip["clip_returned"] ==
             clip["optimizer_attempted"] == clip["optimizer_returned"] == 256 and
             clip["native_clip_attempted"] == clip["native_clip_returned"] ==
             (256 if arm == "global" else 512) and len(clip["events"]) == 256,
             "clip/optimizer counts differ")
        for step, event in enumerate(clip["events"]):
            need(event["step"] == step, f"clip event skipped: {step}")
            check_clip_event(event, arm)
        check_gradient_samples(folder, clip)
        final = check_final(folder, learned, parentage)
        first = check_first_block(folder)
        check_worker_counts(worker)
        need(len(learned["updates"]) == 16 and
             all(u["optimizer_steps"] == 16 for u in learned["updates"]),
             "rollout update record count differs")
        report.update(complete=True, errors=[], session_identity=identity(session_path),
                      parentage=parentage, final=final,
                      clip_summary={"clip_attempted": 256,
                                    "native_clip": clip["native_clip_returned"],
                                    "actor_delta_mean": float(np.mean([
                                        e["adam_actor_parameter_delta_l2"] for e in clip["events"]])),
                                    "actor_delta_median": float(np.median([
                                        e["adam_actor_parameter_delta_l2"] for e in clip["events"]])),
                                    "critic_delta_mean": float(np.mean([
                                        e["adam_critic_parameter_delta_l2"] for e in clip["events"]])),
                                    "critic_delta_median": float(np.median([
                                        e["adam_critic_parameter_delta_l2"] for e in clip["events"]])),
                                    "actual_actor_scale_mean": float(np.mean([
                                        e["actual_actor_scale"] for e in clip["events"]])),
                                    "actual_actor_scale_median": float(np.median([
                                        e["actual_actor_scale"] for e in clip["events"]])),
                                    "actual_critic_scale_mean": float(np.mean([
                                        e["actual_critic_scale"] for e in clip["events"]])),
                                    "actual_critic_scale_median": float(np.median([
                                        e["actual_critic_scale"] for e in clip["events"]])),
                                    "hypothetical_global_alpha_mean": float(np.mean([
                                        e["hypothetical_global_alpha_on_same_pregradient"]
                                        for e in clip["events"]])),
                                    "hypothetical_global_alpha_median": float(np.median([
                                        e["hypothetical_global_alpha_on_same_pregradient"]
                                        for e in clip["events"]])),
                                    "actor_delta_relative_mean": float(np.mean([
                                        e["adam_actor_delta_over_parameter_norm"]
                                        for e in clip["events"]])),
                                    "critic_delta_relative_mean": float(np.mean([
                                        e["adam_critic_delta_over_parameter_norm"]
                                        for e in clip["events"]]))},
                      updates=[{key: u.get(key) for key in (
                          "update_index", "train/approx_kl", "train/value_loss",
                          "train/explained_variance", "train/clip_fraction", "train/std")}
                          for u in learned["updates"]],
                      first_rollout={"numeric_sha256": first["first_block_hashes"]["numeric"],
                                     "gaussian_sha256": first["first_block_hashes"]["gaussian"],
                                     "episode_boundaries": [
                                         {"episode_index": row["episode_index"],
                                          "first_control_index": row["first_control_index"]}
                                         for row in first["episodes"]],
                                     "episode_reset_schedule_sha256": hashlib.sha256(json.dumps(
                                         first["episodes"], sort_keys=True).encode()).hexdigest()})
        return report, first
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        report["errors"].append(f"{type(error).__name__}: {error}")
        return report, None


def compare_first_rollout(global_first, grouped_first):
    need(global_first is not None and grouped_first is not None, "first rollout unavailable")
    differences = []
    for kind in ("numeric", "gaussian"):
        left, right = global_first[kind], grouped_first[kind]
        need(set(left) == set(right), f"first {kind} array keys differ")
        for name in left:
            if (left[name].shape != right[name].shape or left[name].dtype != right[name].dtype
                    or left[name].tobytes() != right[name].tobytes()):
                differences.append(f"{kind}:{name}")
    if global_first["episodes"] != grouped_first["episodes"]:
        differences.append("episode_reset_schedule_geometry_or_seed")
    return {"byte_exact": not differences, "differences": differences,
            "numeric_keys": len(global_first["numeric"]),
            "gaussian_keys": len(global_first["gaussian"])}


def read_pair(global_folder, grouped_folder):
    global_report, global_first = verify_arm(global_folder, "global")
    grouped_report, grouped_first = verify_arm(grouped_folder, "grouped")
    paired = None
    if global_first is not None and grouped_first is not None:
        try:
            paired = compare_first_rollout(global_first, grouped_first)
        except (ValueError, KeyError, TypeError) as error:
            paired = {"byte_exact": False, "differences": [str(error)]}
    complete = (global_report["complete"] and grouped_report["complete"] and
                paired is not None and paired["byte_exact"])
    return {"schema": "d1-groupclip-training-readback-15-v1", "complete": bool(complete),
            "engineering_passed": bool(complete),
            "global": global_report, "grouped": grouped_report, "first_rollout_pair": paired,
            "model_loads": 0, "physics_controls": 0,
            "boundary": "one paired seed and actual saved training gradients; no model replay or new physics"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--global", dest="global_folder", type=Path, required=True)
    parser.add_argument("--grouped", dest="grouped_folder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = read_pair(args.global_folder.resolve(), args.grouped_folder.resolve())
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"complete": result["complete"],
                      "global_complete": result["global"]["complete"],
                      "grouped_complete": result["grouped"]["complete"]}))
