"""Independent, bounded readback of every C22 training control and PPO update.

Only saved arrays, JSON and source files are read. No policy or physics module is
imported; in particular, this reader does not reconstruct a live environment.
"""
from __future__ import annotations

import argparse
import gzip
import importlib.abc
import json
import math
import os
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

FORBIDDEN = {"mujoco", "glfw", "torch", "stable_baselines3", "gym", "gymnasium",
             "engine_binding", "wheel_legged_control"}
class _NoExecution(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.partition(".")[0] in FORBIDDEN:
            raise RuntimeError("training readback forbids model/physics import: " + fullname)
if any(name.partition(".")[0] in FORBIDDEN for name in sys.modules):
    raise RuntimeError("model/physics imported before training reader")
sys.meta_path.insert(0, _NoExecution())
sys.dont_write_bytecode = True

import numpy as np
from read_eval22 import (arrays, check_binding, check_contact, close, document,
                         equal, identity, plain_document, require, _committed,
                         _body_com_velocity, _ground_hit, _roll_pitch_deg,
                         _terrain_reference, _yaw_from_qpos, _geometry_templates21)
from geometry21 import qualify_reset21, compare_reset_to_template21
from reuse22 import CONTRACT_ID
from verify_control18 import check_controller
from recipes22 import SPEC, T, select_episode
from read_training15 import gradient_digest, vector_norm
from verify_course_e_08_03 import FAMILIES, LEGS, TORQUE_LIMIT, family

SCHEMA = "d1-stage22-full-training-independent-readback-v1"
N = 32768
REWARD_NAMES = ("tracking_vx","tracking_vy","tracking_yaw","height","attitude",
                "actual_torque","action_change","leg_speed","termination")
STAGE_FIELDS = ("consumed_joint_position_rad","consumed_joint_velocity_rad_s",
                "consumed_nominal_joint_target_rad","consumed_nominal_wheel_speed_rad_s",
                "consumed_support_torque_nm","consumed_base_rotation_world_from_body",
                "consumed_foot_jacobian_world","consumed_wheel_omega_rad_s",
                "geometric_joint_target_rad","rate_limited_joint_target_rad",
                "position_limited_joint_target_rad","joint_target_rad",
                "wheel_speed_target_rad_s","wheel_speed_error_rad_s",
                "wheel_integral_before_nm","wheel_integral_candidate_nm",
                "wheel_integral_after_nm","leg_pd_nm","base_wheel_torque_nm",
                "base_request_torque_nm","projected_jx","relative_forward_mps",
                "leg_damping_delta_torque_nm","common_wheel_delta_torque_nm",
                "final_request_torque_nm","safe_torque_nm")


def _block_rows(folder, blocks, stem, *, committed):
    expected = 0
    for index, block in enumerate(blocks):
        name = f"{stem}_{index:04d}.jsonl.gz"
        require(block["file"] == name and block["rows"] > 0,
                f"{stem} block ordering differs")
        path = folder / name
        if committed:
            require(block["archive_manifest"] == name + ".manifest.json",
                    f"{stem} archive manifest label differs")
            _committed(path)
        require(identity(path) == {k: block[k] for k in ("sha256", "bytes")},
                f"{stem} block digest differs")
        count = 0
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                count += 1
                yield json.loads(line)
        require(count == block["rows"], f"{stem} block row count differs")
        expected += count


def _npz_blocks(folder, blocks, stem):
    for i, block in enumerate(blocks):
        name = f"{stem}_{i:04d}.npz"
        path = folder / name
        require(block["file"] == name and block["rows"] == 1024
                and block["first_control_index"] == 1024*i
                and block["last_control_index"] == 1024*i+1023
                and identity(path) == {k: block[k] for k in ("sha256", "bytes")},
                f"{stem} numeric block identity differs")
        data = arrays(path)
        require(all(value.shape[0] == 1024 and np.isfinite(value).all()
                    for value in data.values())
                and np.array_equal(data["control_index"], np.arange(1024*i, 1024*(i+1))),
                f"{stem} numeric shape/index differs")
        yield data


def arrays_uncommitted(path):
    require(path.is_file() and not path.is_symlink(), "numeric block missing")
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def _take(iterator, n, label):
    out = []
    for _ in range(n):
        try:
            out.append(next(iterator))
        except StopIteration:
            raise ValueError(label + " ended early") from None
    return out


def _done(iterator, label):
    require(next(iterator, None) is None, label + " has trailing rows")


def _check_episode_start(folder, episode, source, arm, offset, previous_end):
    recipe = select_episode(source-offset, arm, source_offset=offset)
    schedule = document(folder / f"training_episode_{episode:06d}_schedule.json")
    reset = document(folder / f"training_episode_{episode:06d}_reset.json")
    control = document(folder / f"training_episode_{episode:06d}_control_reset.json")
    initial = arrays(folder / f"training_episode_{episode:06d}_initial_state.npz")
    require(schedule["episode_index"] == episode
            and schedule["choice"]["source_episode_index"] == source
            and schedule["raw_command_sha256"] == recipe.command_sha256
            and schedule["raw_commands"] == [asdict(x) for x in recipe.raw_commands]
            and schedule["terrain"] == recipe.terrain
            and reset["episode_index"] == episode
            and reset["episode_reset_seed"] == schedule["episode_reset_seed"],
            "training recipe/reset identity differs")
    state = control
    require(state["variant"] == "combined"
            and equal(state["wheel_integral_nm"], np.zeros(4))
            and state["wheel_common_reference_z_rad_s"] == 0.
            and state["previous_servo_forward_mps"] is None
            and state["stop_latched"] is False
            and state["servo_last_tick"] == 0
            and state["provider_sequence"] == 0,
            "training C18 controller reset state differs")
    require(set(initial) == {"qpos","qvel","ctrl","qacc_warmstart","observation","time"},
            "training episode initial five-state archive incomplete")
    return recipe, schedule, reset, state, initial


def _check_episode_geometry22(folder,episode,source,recipe,state,initial,sources):
    """Recompute one actual reset's pure path and match the sealed C21 template."""
    templates,controls,template_inputs = _geometry_templates21()
    require(all(sources.get(path) == expected for path,expected in template_inputs.items()),
            "training sealed geometry templates are absent from the source GO")
    stem = folder/f"training_episode_{episode:06d}_geometry21"
    actual = document(Path(str(stem)+"_reset_manifest.json"))
    require(actual["model_identity"]["model_address"] > 0
            and actual["model_identity"]["data_address"] > 0,
            "training actual compiled reset identity is missing")
    receipt,path = qualify_reset21(recipe,actual,initial,kind="train",source_index=source)
    expected_manifest,expected_initial = templates[recipe.terrain]
    bridge = compare_reset_to_template21(expected_manifest,actual,expected_initial,
        initial,controls[recipe.terrain],state)
    saved = arrays(Path(str(stem)+"_nominal_path.npz"))
    require(receipt["passed"] is True and bridge["passed"] is True
            and document(Path(str(stem)+"_receipt.json")) == receipt
            and document(Path(str(stem)+"_reset_homotopy.json")) == bridge
            and set(saved) == {"path_xy"}
            and saved["path_xy"].dtype == path.dtype
            and saved["path_xy"].shape == path.shape
            and saved["path_xy"].tobytes() == path.tobytes(),
            "training actual reset geometry, path, or complete control homotopy differs")
    return {"episode_index":episode,"source_episode_index":source,"terrain":recipe.terrain,
        "qualified":True,"pure_servo_intervals":len(recipe.raw_commands),
        "actual_geometry_identity":identity(Path(str(stem)+"_reset_manifest.json")),
        "observed_new_reset_bridge_verified":True,"robot_or_model_calls":0}


def _check_native(row, five, calc, binding, geometry, native_start):
    info = row["info"]
    traces = row["native_actuator_traces"]
    interval = info["native_interval_summary"]
    tick = row["episode_tick"]
    require(len(five) == len(traces) == interval["native_returns"] == 5
            and close(interval["start_time_s"], tick*.01)
            and close(interval["end_time_s"], (tick+1)*.01),
            "training control lacks five native returns")
    nonwheel = 0
    families = Counter()
    positive = set()
    pre, post = row["pre_state"], row["post_state"]
    for substep, (native, trace) in enumerate(zip(five, traces)):
        require(native["native_index"] == native_start+substep
                and close(native["start_time_s"], tick*.01+substep*.002)
                and close(native["end_time_s"], tick*.01+(substep+1)*.002)
                and native["contact_count"] == len(native["contacts"]),
                "training native index/clock differs")
        for key in ("qpos", "qvel", "qacc_warmstart"):
            expected_before = pre[key] if substep == 0 else five[substep-1]["after"][key]
            require(equal(native["before"][key], expected_before),
                    "training native integrator before-state differs")
            if substep == 4:
                require(equal(native["after"][key], post[key]),
                        "training native integrator endpoint differs")
        require(equal(native["before"]["ctrl"], trace["applied_nm"])
                and equal(native["after"]["ctrl"], trace["applied_nm"])
                and close(native["after"]["actuator_force"], trace["applied_nm"])
                and all(close(trace[name], calc["computed"]["safe_torque_nm"])
                        for name in ("requested_nm", "limited_nm", "delayed_nm", "applied_nm")),
                "training native actuator torque chain differs")
        roll, pitch = _roll_pitch_deg(native["after"]["qpos"])
        require(close((roll, pitch), (native["roll_deg"], native["pitch_deg"])),
                "training native posture differs")
        per_native = Counter()
        candidates = 0
        for contact in native["contacts"]:
            loaded, family = check_contact(contact, geometry, binding)
            if family is not None:
                candidate = contact["robot_wheel_index"] is None
                require(contact["nonwheel_ground_candidate"] is candidate
                        and contact["nonwheel_ground_contact"] is candidate
                        and contact["nonwheel_active_solver_contact"] is (
                            candidate and contact["efc_address"] >= 0),
                        "training contact solver classification differs")
            candidates += bool(contact["nonwheel_ground_contact"])
            if loaded:
                per_native[family] += 1
                positive.add(family)
        require(candidates == native["nonwheel_contact_count"]
                and dict(per_native) == native["terrain_family_positive_wheel_load"],
                "training native contact/force summary differs")
        nonwheel += candidates
        families.update(per_native)
    require(nonwheel == interval["nonwheel_contact_count"]
            and dict(families) == interval["terrain_family_positive_wheel_load"],
            "training five-native interval summary differs")
    return positive


def _check_numeric(row, numeric, gaussian, i):
    info = row["info"]
    keys = ("control_index", "episode_index", "episode_tick")
    require(all(int(numeric[key][i]) == row[key] for key in keys)
            and int(gaussian["control_index"][i]) == row["control_index"]
            and int(gaussian["episode_index"][i]) == row["episode_index"]
            and int(gaussian["episode_tick"][i]) == row["episode_tick"]
            and equal(numeric["input_observation99"][i], row["input_observation99"])
            and equal(numeric["policy_env_input16"][i], row["policy_input_action"])
            and equal(gaussian["clipped_action16"][i], row["policy_input_action"])
            and equal(numeric["policy_clipped16"][i], info["policy_clipped_action"])
            and equal(numeric["effective_action16"][i], info["applied_action"])
            and equal(gaussian["effective_action16"][i], info["applied_action"])
            and equal(np.clip(gaussian["raw_gaussian_action16"][i], -1., 1.),
                      gaussian["clipped_action16"][i])
            and close(numeric["reward"][i], row["reward"])
            and bool(numeric["terminated"][i]) is row["terminated"]
            and bool(numeric["truncated"][i]) is row["truncated"]
            and close(numeric["body_com_vx_mps"][i], info["metrics"]["body_com_vx_mps"])
            and close(numeric["body_yaw_rate_rps"][i], info["metrics"]["body_yaw_rate_rps"]),
            "training numeric/Gaussian/full-control linkage differs")


def _check_numeric_reward(row, numeric, i, calc, binding, previous_action):
    info = row["info"]
    terms = info["reward_terms"]
    command = info["consumed_command"]
    metrics = info["metrics"]
    traces = row["native_actuator_traces"]
    action = np.asarray(info["applied_action"],dtype=np.float64)
    actual = np.asarray([trace["applied_nm"] for trace in traces],dtype=np.float64)
    leg_qvel = np.asarray(row["post_state"]["qvel"])[binding["dof_addresses"]][list(LEGS)]
    expected = {
        "tracking_vx":2*math.exp(-((metrics["body_com_vx_mps"]-command["forward_velocity_mps"])/.25)**2),
        "tracking_vy":math.exp(-((metrics["body_com_vy_mps"]-command["lateral_velocity_mps"])/.12)**2),
        "tracking_yaw":math.exp(-((metrics["body_yaw_rate_rps"]-command["yaw_rate_rps"])/.4)**2),
        "height": -.2*((metrics["clearance_m"]-command["clearance_m"])/.08)**2,
        "attitude": -.2*((metrics["task_roll_error_rad"]/.2)**2
                         +(metrics["task_pitch_error_rad"]/.2)**2),
        "actual_torque": -.02*float(np.mean((actual/TORQUE_LIMIT)**2)),
        "action_change": -.01*float(np.mean((action-previous_action)**2)),
        "leg_speed": -.005*float(np.mean((leg_qvel/20.)**2)),
        "termination": -10. if row["terminated"] else 0.,
    }
    require(set(terms) == set(REWARD_NAMES)
            and all(close(terms[k],expected[k],atol=1e-8) for k in REWARD_NAMES)
            and close(row["reward"],sum(expected.values()),atol=1e-8)
            and close(numeric["reward_terms9"][i],[terms[k] for k in REWARD_NAMES],atol=1e-8)
            and close(numeric["raw_command_vx_vy_yaw_clearance"][i],
                      [info["raw_operator_command"][k] for k in
                       ("forward_velocity_mps","lateral_velocity_mps","yaw_rate_rps","clearance_m")])
            and close(numeric["servo_command_vx_vy_yaw_clearance"][i],
                      [command[k] for k in
                       ("forward_velocity_mps","lateral_velocity_mps","yaw_rate_rps","clearance_m")])
            and all(close(numeric["stage_"+k][i],calc[k]) for k in STAGE_FIELDS)
            and all(close(numeric["actuator_delayed5x16"][i,j],trace["delayed_nm"])
                    and close(numeric["actuator_applied5x16"][i,j],trace["applied_nm"])
                    for j,trace in enumerate(traces))
            and close(numeric["native_nonwheel_contacts"][i],
                      info["native_interval_summary"]["nonwheel_contact_count"])
            and all(close(numeric[key][i],metrics[field],atol=1e-8) for key,field in (
                ("body_com_vx_mps","body_com_vx_mps"),("body_com_vy_mps","body_com_vy_mps"),
                ("body_yaw_rate_rps","body_yaw_rate_rps"),("base_x_m","x_m"),
                ("base_y_m","y_m"),("clearance_m","clearance_m"),
                ("geometry_relative_roll_rad","relative_roll_rad"),
                ("geometry_relative_pitch_rad","relative_pitch_rad"),
                ("task_roll_error_rad","task_roll_error_rad"),
                ("task_pitch_error_rad","task_pitch_error_rad"))),
            "training independent reward or full numeric controller/actuator join differs")
    return action


def _coverage(episodes, arm, stage):
    spec = SPEC["coverage"]
    slots = {slot: [] for slot in range(8)}
    endpoint = Counter()
    terrain_load = Counter()
    for e in episodes:
        if e["source_episode_index"] < 8:
            continue
        slot = (e["source_episode_index"]-8) % 8
        if e["completed_controls"] == 1000 and e["effective_ticks"] >= 200:
            slots[slot].append(e)
            if e["positive_active_intervals"]:
                terrain_load[e["terrain"]] += 1
        if arm == "B" and slot == 2 and (e["source_episode_index"]-8)//8 in (0, 2):
            endpoint.update(e["endpoint_sign_ticks"])
    slot_counts = {str(k): len(v) for k, v in slots.items()}
    ok = all(n >= spec["postwarmup_completed_1000_tick_episodes_per_slot_min"]
             for n in slot_counts.values())
    details = {"postwarmup_completed_effective_episodes_by_slot": slot_counts,
               "active_loaded_episodes_by_terrain": dict(terrain_load),
               "endpoint_ticks_by_yaw_sign": dict(endpoint)}
    ok &= terrain_load["rough"] >= 1 and terrain_load["ramp"] >= 1
    if arm == "B" and stage == 1:
        indexed = {e["source_episode_index"]: e for e in episodes}
        yaw_sources = [8+8*cycle+2 for cycle in (0, 2)]
        bump_sources = [8+8*cycle+3 for cycle in (0, 1, 2)]
        ok &= all(indexed.get(s, {}).get("completed_controls") == 1000 for s in yaw_sources)
        ok &= all(indexed.get(s, {}).get("completed_controls") == 1000
                  and indexed[s]["positive_active_intervals"] >= 25 for s in bump_sources)
        ok &= all(endpoint[sign] >= spec["B_endpoint_ticks_each_yaw_sign_min"]
                  for sign in ("positive", "negative"))
        details["required_yaw_sources"] = yaw_sources
        details["required_bump_sources"] = bump_sources
    details["coverage_valid"] = bool(ok)
    details["applicability"] = "fixed_final_claim_exposure_gate_not_final_execution_gate"
    return details


def _clip_and_learning(run, arm, stage, worker, blocks, truncated_episodes):
    clip = document(run / "clip_audit.json")
    learned = document(run / "training" / "learning_receipt.json")
    parent = document(run / "parentage.json")
    manifest = document(run / "final_checkpoint_manifest.json")
    metadata_path = run / "final_checkpoint" / "final_metadata.json"
    metadata = plain_document(metadata_path)
    actual = learned["actual"]
    events, batches, trains = clip["events"], clip["evaluated_batches"], clip["train_records"]
    require(clip["schema"] == "d1-course20-grouped-clip-actual-batches-v1"
            and clip["arm"] == learned["course_arm"] == arm
            and len(trains) == len(learned["updates"]) == 32
            and 32 <= len(events) <= len(batches) <= 512
            and learned["status"] == "complete"
            and learned["qualified_for_final_checkpoint"] is True
            and learned["violations"] == []
            and learned["target_kl"] == .03
            and learned["normal_kl_threshold"] == .045
            and learned["hard_kl_threshold"] == .30
            and actual["num_timesteps"] == actual["transitions"] == 32768
            and actual["rollouts"] == actual["train_calls"] == 32
            and actual["optimizer_steps"] == len(events)
            and actual["evaluated_minibatches"] == len(batches)
            and actual["epochs"] == sum(t["entered_epochs"] for t in trains)
            and actual["kl_early_stop_train_calls"] == sum(t["normal_kl_stop"] for t in trains),
            "C20 complete actual PPO/KL receipt differs")
    expected_batch = 0
    expected_step = 0
    prior_epochs = 0
    selected_steps = []
    for rollout, train in enumerate(trains):
        n_eval, n_opt = train["evaluated_minibatches"], train["optimizer_steps"]
        entered = train["entered_epochs"]
        require(train["rollout"] == rollout and 1 <= entered <= 4
                and 1 <= n_opt <= n_eval <= 16
                and n_eval == n_opt + int(train["normal_kl_stop"])
                and train["ending_n_updates"] == prior_epochs+entered,
                "actual PPO train/epoch counts differ")
        first_step = expected_step
        for local in range(n_eval):
            batch = batches[expected_batch]
            ids = (batch["rollout"], batch["epoch"], batch["minibatch"])
            require(ids == (rollout,local//4,local%4)
                    and math.isfinite(batch["approx_kl"])
                    and batch["approx_kl"] <= .30
                    and batch["hard_stop"] is False,
                    "actual evaluated minibatch identity/KL differs")
            rejected = local == n_eval-1 and train["normal_kl_stop"]
            require(batch["normal_kl_stop"] is (batch["approx_kl"] > .045)
                    and batch["normal_kl_stop"] is bool(rejected)
                    and batch["optimized"] is (not rejected)
                    and batch["optimizer_step"] == (None if rejected else expected_step),
                    "actual rejected minibatch/backward decision differs")
            if not rejected:
                event = events[expected_step]
                require(event["step"] == event["global_step"] == expected_step
                        and event["arm"] == arm
                        and (event["rollout"], event["epoch"], event["minibatch"]) == ids
                        and math.isclose(event["evaluated_approx_kl"], batch["approx_kl"],
                                         rel_tol=0, abs_tol=1e-9)
                        and event["optimizer_attempted"] is True
                        and event["optimizer_returned"] is True
                        and event["missing_gradients"] == event["nonfinite_gradients"] == 0,
                        "actual Adam step lacks accepted evaluated batch")
                pa, pc = event["pre_actor_l2"], event["pre_critic_l2"]
                qa, qc = event["post_actor_l2"], event["post_critic_l2"]
                require(all(math.isfinite(x) and x >= 0 for x in (pa,pc,qa,qc))
                        and np.isclose(qa, pa*min(1., .5/(pa+1e-6)), rtol=3e-5, atol=1e-5)
                        and np.isclose(qc, pc*min(1., .5/(pc+1e-6)), rtol=3e-5, atol=1e-5)
                        and qa <= .50001 and qc <= .50001
                        and math.isclose(event["adam_joint_parameter_delta_l2"],
                                         math.hypot(event["adam_actor_parameter_delta_l2"],
                                                    event["adam_critic_parameter_delta_l2"]),
                                         rel_tol=3e-5, abs_tol=1e-8),
                        "actual grouped clip/Adam delta differs")
                expected_step += 1
            expected_batch += 1
        require(entered == math.ceil(n_eval/4)
                and (train["normal_kl_stop"] or n_eval == n_opt == 16),
                "actual PPO epoch/minibatch enumeration differs")
        selected_steps.extend([first_step, expected_step-1] if n_opt > 1 else [first_step])
        require(expected_step-first_step == n_opt
                and learned["updates"][rollout]["optimizer_steps"] == n_opt
                and learned["updates"][rollout]["evaluated_minibatches"] == n_eval
                and learned["updates"][rollout]["n_updates_delta"] == entered,
                "learning receipt/clip actual update differs")
        prior_epochs += entered
    require(expected_batch == len(batches) and expected_step == len(events),
            "actual batch or optimizer ledger has trailing rows")
    samples = clip["gradient_sample_records"]
    require(len(samples) == len(selected_steps) and
            [s["ids"]["global_step"] for s in samples] == selected_steps,
            "first/last actual PPO gradient samples differ")
    for sample in samples:
        step = sample["ids"]["global_step"]
        path = run / "gradients" / f"gradient_step_{step:04d}.npz"
        require(Path(sample["file"]).resolve() == path.resolve()
                and identity(path) == sample["identity"],
                "raw gradient archive identity differs")
        raw = arrays(path)
        event = events[step]
        require(event["full_gradient_saved"] is True, "selected gradient event not marked saved")
        for phase in ("pre", "post"):
            for group in ("actor", "critic"):
                prefix = f"{phase}::{group}::"
                named = {key[len(prefix):]: value for key,value in raw.items()
                         if key.startswith(prefix)}
                require(set(named) == set(clip[group+"_parameter_names"])
                        and gradient_digest(named) == event[f"{phase}_gradient_sha256"][group]
                        and np.isclose(vector_norm(named), event[f"{phase}_{group}_l2"],
                                       rtol=3e-5, atol=1e-8),
                        "actual raw PPO gradient differs")
    require(all(e["full_gradient_saved"] is (i in selected_steps)
                for i,e in enumerate(events)), "unselected full gradient flag differs")
    counts = worker["model_calls"]["counts"]
    for name, number in {"load":2,"torch_load":6,"save":1,"learn":1,
                         "train":32,"forward":N,"evaluate_actions":len(batches),
                         "backward":len(events),"predict":3}.items():
        row = counts.get(name,{})
        require(row.get("attempted",0) == row.get("returned",0) == number,
                "actual model API count differs: " + name)
    require(counts["evaluate_actions"]["rows_returned"] == 256*len(batches)
            and counts["backward"]["rows_returned"] == 0
            and counts["predict_values"]["attempted"]
            == counts["predict_values"]["returned"] == 32+truncated_episodes
            and counts["predict_values"]["rows_returned"] == 32+truncated_episodes
            and worker["model_calls"]["actor_rows"] == N+256*len(batches)+96
            and worker["model_calls"]["critic_rows"]
            == N+256*len(batches)+32+truncated_episodes
            and all(row["rows_attempted"] == row["rows_returned"]
                    for row in counts.values()),
            "actual evaluated/backward model rows differ")
    require(metadata.get("execution_contract_id") == CONTRACT_ID
            and metadata.get("numerical_protocol_schema") == "C20"
            and metadata["training_receipt"] == learned
            and metadata["stage20_parentage"] == parent
            and metadata["stage20_course_arm"] == arm
            and metadata["stage20_stage"] == stage
            and identity(metadata_path)["sha256"] == manifest["metadata_sha256"]
            and manifest["actual_optimizer_steps"] == len(events)
            and manifest["stage20_course_arm"] == arm
            and manifest["stage20_stage"] == stage,
            "actual final checkpoint metadata/learning closure differs")
    for name, expected in manifest["files"].items():
        require(identity(run/"final_checkpoint"/name) == expected,
                "final checkpoint file identity differs")
    reload = manifest["reload_verification"]
    require(reload["matches_training_final_policy_state"] is True
            and reload["probe_actions_byte_exact"] is True
            and reload["verified_without_engine_or_reset"] is True
            and reload["loaded_policy_state_sha256"] == learned["final_hashes"]["policy_state"]
            and learned["initial_hashes"] == parent["parent_hashes"],
            "actual parent/strict reload policy hash differs")
    return {"actual":actual, "normal_kl_stop_train_calls":actual["kl_early_stop_train_calls"],
            "gradient_sample_count":len(samples), "checkpoint_model_identity":
            identity(run/"final_checkpoint"/"final_model.zip"),
            "first_1024_numeric_path":str(run/"training"/"training_numeric_blocks"/"controls_0000.npz"),
            "first_1024_gaussian_path":str(run/"training"/"training_gaussian_blocks"/"gaussian_0000.npz")}


def _native_stream(folder, segment):
    names = segment["native_files"]
    require(names == [f"native_block_{i:04d}.jsonl.gz" for i in range(len(names))]
            and segment["archive_block_manifests"]
            == [name+".manifest.json" for name in names],
            "training native block ordering/transaction labels differ")
    for name in names:
        path = folder/name
        _committed(path)
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                yield json.loads(line)


def _geometry(construction):
    binding = check_binding(construction["geometry_binding"])
    manifest = construction["compiled_geometry"]
    rows = manifest["world_collision_geoms"]
    require(len(rows) == 92 and manifest["control_dt_s"] == .01
            and manifest["native_dt_s"] == .002,
            "training actual compiled geometry dimensions differ")
    require(Counter(family(row["name"]) for row in rows) == FAMILIES,
            "training compiled terrain family counts differ")
    out = {}
    kinds = {"plane":set(),"box":set()}
    for row in rows:
        gid = row["geom_id"]
        require(type(gid) is int and gid not in out
                and 0 <= gid < len(binding["geom_bodyid"])
                and row["body_id"] == int(binding["geom_bodyid"][gid]) == 0
                and close(binding["geom_size"][gid], row["size_m"])
                and (binding["geom_contype"][gid] != 0
                     or binding["geom_conaffinity"][gid] != 0)
                and row["type"] == ("plane" if row["name"] == "floor" else "box")
                and close(np.asarray(row["quaternion_wxyz"]) @
                          np.asarray(row["quaternion_wxyz"]),1.,atol=1e-12),
                "training compiled ground geometry differs")
        kinds[row["type"]].add(int(binding["geom_type"][gid]))
        out[gid] = row
    require(len(kinds["plane"]) == len(kinds["box"]) == 1
            and kinds["plane"] != kinds["box"],
            "training compiled plane/box primitive types differ")
    return binding, out


def _worker_closure(run, session, worker, host):
    sources = session["source_hashes"]
    plan_path = Path(session["plan_path"])
    plan = plain_document(plan_path)
    key = session["reservation_key"]
    require(plan["schema"] == "d1-coverage-source-go-20-v1"
            and plan["status"] == "GO" and plan["retry_permitted"] is False
            and key in plan["arms"]
            and sources == {**plan["inputs"],str(plan_path):identity(plan_path)}
            and Path(session["output_directory"]) ==
                Path(__file__).resolve().parent/plan["arms"][key]["output_directory"]
            and all(session[field] == value for field,value in plan["arms"][key].items()
                    if field != "output_directory"),
            "training session differs from frozen source GO arm reservation")
    require(plan.get("execution_contract_id") == CONTRACT_ID
            and worker.get("execution_contract_id") == CONTRACT_ID
            and session["schema"] == "d1-coverage-session-20-v1"
            and session["mode"] == "train"
            and session["controller_variant"] == "combined"
            and session["retry_permitted"] is False
            and session["normal_native_limit"] == 5*N
            and session["compiler_native_limit"] == 2
            and worker["schema"] == "d1-coverage-worker-20-v1"
            and worker["arm"] == session["arm"]
            and worker["session_identity"] == identity(run/"session.json")
            and worker["execution_complete"] is True
            and worker["failure"] is None and worker["cleanup_errors"] == []
            and worker["warnings"] == [] and worker["archive_failed"] is False
            and worker["control_step_caller_verified"] is True
            and host["arm"] == session["arm"]
            and host["reservation_key"] == session["reservation_key"]
            and host["exit_code"] == 0 and host["failure"] is None
            and host["cleanup_errors"] == [] and host["changed_sources"] == []
            and host["postcheck_complete"] is True
            and host["no_live_owned_processes"] is True
            and host["fully_reserved_budget_closed"] is True
            and host["retry_permitted"] is False
            and host["elapsed_s"] <= session["hard_s"],
            "training worker/host/session closure differs")
    require(len(sources) >= 100 and session["plan_identity"]
            == identity(Path(session["plan_path"]))
            and session["contract_sha256"] == identity(Path(session["contract_path"]))["sha256"],
            "training C22 source GO identity incomplete")
    for name, expected in sources.items():
        require(identity(Path(name)) == expected, "training frozen source drift: "+name)
    here = Path(__file__).resolve().parent
    required = [here/name for name in ("read_training22.py","read_eval22.py","recipes22.py",
        "geometry_runtime22.py","reuse22.py","curriculum22.py","train22.py","spec22.json")]
    required += [here.parent/"continuation21"/name for name in
                 ("geometry21.py","state21.py","learning21.py","checkpoint21.py")]
    require(all(str(path) in sources for path in required), "training critical C22/immutable C21 source omitted")
    c, py, native = worker["C_final"], worker["python"], worker["native_guard"]
    require(c["construction_attempts"] == c["construction_returns"] == 2
            and c["control_attempts"] == c["control_returns"] == 5*N
            and c["violations"] == 0
            and c["phase"] == c["target_model"] == c["target_data"] == 0
            and py["control_attempted"] == py["control_completed"] == N
            and py["native_attempted"] == py["native_returned"] == 5*N
            and py["native_failed"] == py["forbidden_entries"] == 0
            and native["native_attempted"] == native["native_returned"]
            == native["native_checked"] == 5*N
            and native["failure"] is None
            and len(native["segments"]) == 1,
            "training C/Python/native physical counters differ")
    return sources


def _assert_clean():
    require("LD_PRELOAD" not in os.environ and "LD_LIBRARY_PATH" not in os.environ,
            "training reader loader injection environment is present")
    require(not any(name.partition(".")[0] in FORBIDDEN for name in sys.modules),
            "training reader imported a model/physics module")
    maps = Path("/proc/self/maps").read_text(encoding="utf-8").lower()
    require("libmujoco" not in maps and "libepa01_engine" not in maps,
            "training reader mapped model/physics library")


def verify_training(run: Path, output: Path) -> dict:
    _assert_clean()
    run, output = Path(run).resolve(strict=True), Path(output).resolve()
    require(run.is_dir() and not run.is_symlink() and not output.exists(),
            "training run/output path invalid")
    session = plain_document(run/"session.json")
    worker = plain_document(run/"worker_receipt.json")
    host = plain_document(run/"host_receipt.json")
    arm, stage = session["arm"], session["stage"]
    require(arm == "B" and stage == 1
            and session.get("execution_contract_id") == CONTRACT_ID
            and session["source_episode_offset"] == 0
            and session["control_limit"] == N,
            "training arm/stage/budget differs")
    sources = _worker_closure(run, session, worker, host)
    folder = run/"training"
    blocks = document(folder/"training_blocks_manifest.json")
    segment = document(folder/"training_segment_receipt.json")
    require(blocks["complete"] is True
            and blocks["completed_controls"] == blocks["gaussian_records"]
            == blocks["runtime_control_completed"] == N
            and blocks["pending_gaussian_control_index"] is None
            and blocks["env_mode"] == "train"
            and blocks["archive_force_sampling_mode"] == "heldout/full_contact"
            and blocks["source_episode_offset"] == session["source_episode_offset"]
            and len(blocks["full_control_blocks20"]) == math.ceil(N/200)
            and len(blocks["numeric_blocks"]) == len(blocks["gaussian_blocks"]) == 32
            and sum(b["rows"] for b in blocks["full_control_blocks20"]) == N
            and segment["mode"] == "heldout"
            and segment["native_attempted"] == segment["native_returned"] == 5*N
            and segment["force_sampling_performed"] is True
            and segment["full_contact_qualification_recorded"] is True
            and segment["record_valid"] is True
            and segment["failure"] is None and segment["archive_failure"] is None
            and segment["partial_native_interval"] is False
            and worker["native_guard"]["segments"] == [segment],
            "complete C22 training archives/native-force mode differ")
    construction = document(run/"construction_receipt.json")
    binding, geometry = _geometry(construction)
    full = iter(_block_rows(folder, blocks["full_control_blocks20"],
                            "full_control_records", committed=True))
    native = iter(_native_stream(folder, segment))
    numeric_iter = iter(_npz_blocks(folder/"training_numeric_blocks",
                                    blocks["numeric_blocks"], "controls"))
    gaussian_iter = iter(_npz_blocks(folder/"training_gaussian_blocks",
                                     blocks["gaussian_blocks"], "gaussian"))
    episodes = []
    geometry_proofs = []
    active = None
    prior_post = None
    servo_vx = servo_yaw = 0.0
    prior_servo = None
    stop_latched = False
    integral = np.zeros(4)
    z = 0.0
    previous_action = np.zeros(16)
    truncated_episodes = 0
    numeric0 = gaussian0 = None
    for block_index in range(32):
        numeric_block, gaussian_block = next(numeric_iter), next(gaussian_iter)
        if block_index == 0:
            numeric0, gaussian0 = numeric_block, gaussian_block
        for within in range(1024):
            index = 1024*block_index+within
            row = next(full)
            require(row["control_index"] == index
                    and row["source_episode_index"]
                    == session["source_episode_offset"]+row["episode_index"],
                    "training global/local/source control identity differs")
            tick = row["episode_tick"]
            if tick == 0:
                if active is not None:
                    require(active["closed"] is True, "training reset before real done")
                    episodes.append(active)
                epi = row["episode_index"]
                require(epi == len(episodes), "training episode index skipped")
                recipe, schedule, reset, state, initial = _check_episode_start(
                    folder, epi, row["source_episode_index"], arm,
                    session["source_episode_offset"], prior_post)
                geometry_proofs.append(_check_episode_geometry22(folder,epi,row["source_episode_index"],
                    recipe,state,initial,sources))
                active = {"episode_index":epi,"source_episode_index":row["source_episode_index"],
                          "terrain":recipe.terrain,"completed_controls":0,
                          "effective_ticks":0,"positive_active_intervals":0,
                          "endpoint_sign_ticks":Counter(),"closed":False}
                servo_vx = servo_yaw = 0.0
                prior_servo = None
                stop_latched = False
                integral = np.zeros(4)
                z = 0.0
                previous_action = np.zeros(16)
            require(active is not None and active["closed"] is False
                    and tick == active["completed_controls"] and tick < 1000
                    and row["terrain"] == recipe.terrain,
                    "training episode control ordering differs")
            _check_numeric(row, numeric_block, gaussian_block, within)
            info = row["info"]
            raw = asdict(recipe.raw_commands[tick])
            servo_vx += max(-.005, min(.005, raw["forward_velocity_mps"]-servo_vx))
            servo_yaw += max(-.006, min(.006, raw["yaw_rate_rps"]-servo_yaw))
            if servo_vx != 0.:
                stop_after = False
            elif prior_servo is not None and prior_servo != 0.:
                stop_after = True
            else:
                stop_after = stop_latched
            adapter = info["controller_record"]
            require(info["raw_operator_command"] == raw
                    and close(info["consumed_command"]["forward_velocity_mps"],servo_vx)
                    and close(info["consumed_command"]["yaw_rate_rps"],servo_yaw)
                    and info["consumed_command"]["lateral_velocity_mps"] == 0.
                    and info["consumed_command"]["clearance_m"] == .455
                    and info["consumed_command"]["jump_requested"] is False
                    and info["completed_control_intervals"] == tick+1
                    and close(row["reward"],sum(info["reward_terms"].values()))
                    and info["reference_roll_rad"] == info["reference_pitch_rad"] == 0.
                    and adapter["mode"] == "train"
                    and adapter["controller_variant"] == "combined"
                    and adapter["test_actor_probe"] is False
                    and adapter["previous_servo_forward_mps"] == prior_servo
                    and adapter["stop_latched_before"] is stop_latched
                    and adapter["stop_latched_after"] is stop_after,
                    "training raw/servo/reward/stop chain differs")
            prior_servo, stop_latched = servo_vx, stop_after
            pre, post = row["pre_state"], row["post_state"]
            if tick == 0:
                require(all(equal(pre[key], initial[key]) for key in initial),
                        "training reset initial five-state archive differs from actual first control")
            require(equal(pre["observation"], row["input_observation99"])
                    and close(pre["time"], tick*.01)
                    and close(post["time"], (tick+1)*.01)
                    and (prior_post is None or tick == 0
                         or all(equal(pre[key],prior_post[key]) for key in
                                ("qpos","qvel","ctrl","qacc_warmstart","observation","time"))),
                    "training pre/post state or observation continuity differs")
            body = float(_body_com_velocity(pre["qpos"],pre["qvel"],binding)[0])
            calc = check_controller(row,index,pre_qpos=pre["qpos"],pre_qvel=pre["qvel"],
                                    pre_body_forward_mps=body,servo_yaw_rps=servo_yaw)
            control = calc["calculation"]
            previous_action = _check_numeric_reward(row,numeric_block,within,
                                                      control,binding,previous_action)
            require(equal(control["wheel_integral_before_nm"],integral)
                    and close(control["wheel_common_reference_z_before_rad_s"],z)
                    and close(adapter["wheel_common_reference_z_before_rad_s"],z)
                    and equal(control["consumed_joint_position_rad"],
                              np.asarray(pre["qpos"])[binding["qpos_addresses"]])
                    and equal(control["consumed_joint_velocity_rad_s"],
                              np.asarray(pre["qvel"])[binding["dof_addresses"]]),
                    "training actual C18 joint/PI/z state differs")
            integral = np.asarray(control["wheel_integral_after_nm"],dtype=np.float64)
            z = float(control["wheel_common_reference_z_after_rad_s"])
            require(close(adapter["wheel_common_reference_z_after_rad_s"],z),
                    "training C18 adapter z differs")
            five = _take(native,5,"native training archive")
            positive = _check_native(row,five,calc,binding,geometry,5*index)
            metrics = info["metrics"]
            vx, vy = _body_com_velocity(post["qpos"],post["qvel"],binding)[:2]
            height, normal, gid = _ground_hit(geometry,post["qpos"][0],post["qpos"][1])
            roll,pitch = _roll_pitch_deg(post["qpos"])
            rel_roll,rel_pitch = _terrain_reference(normal,_yaw_from_qpos(post["qpos"]))
            require(close(vx,metrics["body_com_vx_mps"],atol=1e-6)
                    and close(vy,metrics["body_com_vy_mps"],atol=1e-6)
                    and close(post["qvel"][5],metrics["body_yaw_rate_rps"],atol=1e-6)
                    and close(post["qpos"][:2],(metrics["x_m"],metrics["y_m"]))
                    and close(post["qpos"][2]-height,metrics["clearance_m"],atol=1e-6)
                    and metrics["ground_geom_id"] == gid
                    and close(math.radians(roll)-rel_roll,metrics["relative_roll_rad"],atol=1e-6)
                    and close(math.radians(pitch)-rel_pitch,metrics["relative_pitch_rad"],atol=1e-6)
                    and close(math.radians(roll),metrics["task_roll_error_rad"],atol=1e-6)
                    and close(math.radians(pitch),metrics["task_pitch_error_rad"],atol=1e-6),
                    "training actual COM/ground/post metrics differ")
            enabled = adapter["action_gate_enabled"] and abs(servo_vx)>0.
            active["effective_ticks"] += int(enabled)
            loaded = enabled and ("bump" if recipe.terrain == "bumps" else
                                  recipe.terrain) in positive
            active["positive_active_intervals"] += int(loaded)
            if (enabled and row["source_episode_index"] >= 8
                    and (row["source_episode_index"]-8)%8 == 2
                    and abs(servo_vx-1.2)<=.005 and abs(servo_yaw)>=.28):
                active["endpoint_sign_ticks"]["positive" if servo_yaw>0 else "negative"] += 1
            active["completed_controls"] += 1
            require(type(row["terminated"]) is bool and type(row["truncated"]) is bool
                    and not (row["terminated"] and row["truncated"]),
                    "training terminal flags invalid")
            if row["terminated"] or row["truncated"]:
                require(info["terminal_reason"] is not None, "training done lacks reason")
                active["closed"] = True
                truncated_episodes += int(row["truncated"])
            prior_post = post
    _done(full,"full training control archive")
    _done(native,"native training archive")
    _done(numeric_iter,"numeric block archive")
    _done(gaussian_iter,"Gaussian block archive")
    if active is not None:
        episodes.append(active)
    closed = sum(e["closed"] for e in episodes)
    require(closed == blocks["closed_episodes"]
            and len(episodes) == closed + int(not episodes[-1]["closed"])
            and ((blocks["partial_episode"] is None) is episodes[-1]["closed"])
            and blocks["zero_control_postbudget_resets"] in (0,1),
            "training episode close/extra reset differs")
    learning = _clip_and_learning(run,arm,stage,worker,blocks,truncated_episodes)
    coverage = _coverage(episodes,arm,stage)
    result = {"schema":SCHEMA,"protocol_schema":"d1-stage20-full-training-independent-readback-v1",
              "execution_contract_id":CONTRACT_ID,"run":str(run),"arm":arm,"stage":stage,
              "geometry_reset_proofs":geometry_proofs,"geometry_resets_verified":len(geometry_proofs),
              "pure_nominal_servo_intervals_verified":sum(row["pure_servo_intervals"] for row in geometry_proofs),
              "zero_control_postbudget_resets":blocks["zero_control_postbudget_resets"],
              "training_valid":True,"archive_valid":True,"learning_valid":True,
              "coverage_valid":coverage["coverage_valid"],"coverage":coverage,
              "complete_controls":N,"five_native_per_control_verified":True,
              "next_source_episode_index":session["source_episode_offset"]+len(episodes),
              "episodes_started":len(episodes),"source_closure_verified":True,
              "actual_learning":learning,"first_1024_numeric_identity":
              identity(Path(learning["first_1024_numeric_path"])),
              "first_1024_gaussian_identity":
              identity(Path(learning["first_1024_gaussian_path"])),
              "reader_model_calls":0,"reader_physics_calls":0}
    with output.open("x",encoding="utf-8") as stream:
        json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write("\n")
    _assert_clean()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    result = verify_training(args.run,args.output)
    print(json.dumps({"training_valid":result["training_valid"],
                      "coverage_valid":result["coverage_valid"],
                      "controls":result["complete_controls"]}))


if __name__ == "__main__":
    main()
