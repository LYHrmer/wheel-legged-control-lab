"""Final shared-heave evaluation with exact hold and pre-request regressions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.d1_jump_heave_env import D1JumpHeaveEnv, load_heave_policy
from scripts.d1_jump_ppo_scoring import score_jump_episode
from scripts.d1_jump_ppo_task import JumpEpisodeSpec
from scripts.probe_d1_jump_readiness import (
    check_frozen,
    check_hashes,
    collect_episode,
    tagged,
    write_json,
    write_manifest,
)


def compare_physical_prefix(zero, policy, case):
    """Allow unused scalar policy provenance to differ; physical prefix must not."""
    count = 600 if case["request_tick"] is None else int(case["request_tick"])
    checks = {}
    for key in zero["arrays"]:
        left = np.asarray(zero["arrays"][key][:count+1])
        right = np.asarray(policy["arrays"][key][:count+1])
        checks["states."+key] = left.shape == right.shape and left.dtype == right.dtype and left.tobytes() == right.tobytes()
    def canonical(value):
        return json.dumps(tagged(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    def physical_row(row):
        result = dict(row)
        result.pop("action")
        info = dict(result["info"])
        mapping = info.pop("heave_action")
        assert not mapping["active"]
        assert mapping["applied_physical_action_8"] == [0.]*8
        info.pop("policy_action")
        result["info"] = info
        return result
    checks["trace_rows"] = len(zero["trace"]) >= count and len(policy["trace"]) >= count
    if checks["trace_rows"]:
        checks["trace_rows"] = all(canonical(physical_row(a)) == canonical(physical_row(b))
                                   for a,b in zip(zero["trace"][:count],policy["trace"][:count],strict=True))
    checks["native_rows"] = len(zero["native"]) >= 5*count and len(policy["native"]) >= 5*count
    if checks["native_rows"]:
        checks["native_rows"] = canonical(zero["native"][:5*count]) == canonical(policy["native"][:5*count])
    return {"passed": all(checks.values()), "controls_compared": count,
            "states_and_observations_compared": count+1, "native_compared":5*count,
            "checks": checks, "allowed_differences":"unused scalar action and mapping provenance only"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    args = parser.parse_args(argv)
    preflight = json.loads(args.preflight.read_text())
    inputs = preflight["input_sha256"]
    if not all(preflight.get(k) is True for k in ("passed", "root_execution_authorized", "readiness_records_valid", "interface_gate_passed")):
        raise ValueError("evaluation requires reviewed root preflight")
    if not check_hashes(inputs) or not check_frozen():
        raise ValueError("frozen source or protocol changed")
    study_protocol = json.loads(Path(preflight["evaluation_protocol"]).read_text())
    protocol = {"schema": "d1-jump-ppo-independent-evaluation-v1", **study_protocol["evaluation"]}
    formal = Path(preflight["formal_directory"])
    training = json.loads((formal / "receipt.json").read_text())
    identity = json.loads((formal / "source_after.json").read_text())
    if (not training["passed"] or not identity["passed"]
            or training["successful_training_transitions"] != 131072
            or training["train_calls"] != 1024 or training["optimization_epochs"] != 4096):
        raise ValueError("only the completed final formal run may be evaluated")
    checkpoint = formal / "checkpoint_131072"
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    scores, receipts, initial_pairs = {}, {}, {}
    regression_pairs = {}
    controls = native = 0
    model = None
    stop = False
    zero_hold_equivalence = None
    for case in protocol["cases"]:
        initial = None
        zero_data = None
        for condition in protocol["conditions"]:
            name = f"{case['name']}__{condition}"
            folder = args.output / name
            folder.mkdir()
            spec = JumpEpisodeSpec(case["request_tick"], case["net_clearance_m"], case["friction_scale"])
            env = D1JumpHeaveEnv(episode_spec=spec)
            def action_function(obs, current_env=env):
                nonlocal model
                if model is None:
                    # collect_episode has already performed the sole case reset; no extra reset/step.
                    model = load_heave_policy(checkpoint / "model.zip", checkpoint / "model.metadata.json", current_env)
                    if model.num_timesteps != 131072:
                        raise ValueError("evaluation checkpoint timestep mismatch")
                return model.predict(obs, deterministic=True)[0]
            data = collect_episode(env, folder, seed=case["seed"],
                                   action_function=(lambda obs: np.zeros(1, dtype=np.float32)) if condition == "zero" else action_function)
            controls += data["completed_control_intervals"]
            native += data["native_receipt"]["attempted_native_calls"]
            score = score_jump_episode(data["episode"], case, protocol)
            write_json(folder / "physical_score.json", score)
            scores[name] = score
            receipts[name] = {key: data[key] for key in (
                "completed_control_intervals", "native_receipt", "error", "terminated", "truncated")}
            arrays = data["arrays"]
            if not all(len(arrays[key]) for key in ("qpos", "qvel", "observation")):
                stop = True
                break
            first = {key: np.array(arrays[key][0], copy=True) for key in ("qpos", "qvel", "observation")}
            if condition == "zero":
                initial = first
                zero_data = data
                if case["name"] == "no_jump_hold":
                    reference = np.load(Path(preflight["readiness_directory"]) / "stationary_height_hold/states.npz")
                    zero_hold_equivalence = {key: np.asarray(arrays[key]).tobytes() == reference[key].tobytes()
                                             for key in ("qpos", "qvel")}
                    zero_hold_equivalence["parent85"] = (
                        np.asarray(arrays["observation"])[:, :85].tobytes() == reference["observation"].tobytes())
            else:
                initial_pairs[case["name"]] = all(initial[k].dtype == first[k].dtype
                                                  and initial[k].tobytes() == first[k].tobytes() for k in first)
            if condition != "zero":
                regression_pairs[case["name"]] = compare_physical_prefix(zero_data, data, case)
            print(json.dumps({"case": name, "completed": data["completed_control_intervals"],
                              "passed": score["passed"], "checks": score["checks"]}), flush=True)
            if (data["error"] is not None or controls > 6000 or native > 30000
                    or any(not row["passed"] for row in regression_pairs.values())):
                stop = True
                break
        if stop:
            break
    integrity = (not stop and len(scores) == 10 and len(initial_pairs) == 5
                 and all(initial_pairs.values()) and len(regression_pairs) == 5
                 and all(row["passed"] for row in regression_pairs.values())
                 and zero_hold_equivalence is not None
                 and all(zero_hold_equivalence.values()) and check_hashes(inputs) and check_frozen())
    summary = {"integrity_passed": integrity, "actual_controls": controls, "actual_native_calls": native,
               "same_case_initial_pairs": initial_pairs, "zero_hold_readiness_equivalence": zero_hold_equivalence,
               "physical_prefix_regressions": regression_pairs,
               "scores": scores, "receipts": receipts,
               "learned_task_passed": integrity and all(scores[f"{c['name']}__final_policy_131072"]["passed"]
                                                        for c in protocol["cases"]),
               "full_objective_complete": False}
    write_json(args.output / "summary.json", tagged(summary))
    write_manifest(args.output)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("scores", "receipts")}, indent=2))
    return 0 if integrity else 1


if __name__ == "__main__":
    raise SystemExit(main())
