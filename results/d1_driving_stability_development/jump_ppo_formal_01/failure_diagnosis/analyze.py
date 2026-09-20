"""File-only diagnosis; imports no simulator/controller/policy and performs no rollouts."""
from __future__ import annotations
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np


def read_json(path):
    return json.loads(path.read_text())


def rows(path):
    with gzip.open(path, "rt") as stream:
        for line in stream:
            yield json.loads(line)


def runs(mask):
    start = None
    result = []
    for i, value in enumerate([*mask, False]):
        if value and start is None:
            start = i
        if not value and start is not None:
            result.append((start, i))
            start = None
    return result


def action_summary(actions):
    a = np.asarray(actions, dtype=float)
    if not a.size:
        return {"count": 0}
    leg = a[:, :4]
    total = float(np.sum(leg * leg))
    return {
        "count": len(a), "mean": a.mean(axis=0).tolist(),
        "rms": np.sqrt(np.mean(a * a, axis=0)).tolist(),
        "near_boundary_fraction_abs_ge_0p95": np.mean(abs(a) >= .95, axis=0).tolist(),
        "actual_boundary_fraction_abs_ge_1": np.mean(abs(a) >= 1., axis=0).tolist(),
        "leg_common_squared_norm_fraction": float(np.sum(4 * leg.mean(axis=1)**2) / total) if total else None,
        "common_mode_definition": "Orthogonal projection onto span((1,1,1,1)); accounting, not causal attribution.",
    }


def request_native_summary(record, start, stop):
    window = record["intervals"][5*start:5*stop]
    gap = np.array([x["endpoint_min_gap_m"] for x in window])
    margin = np.array([x["contact_margin_m"] for x in window])
    counts = np.array([x["active_wheel_contacts"] for x in window])
    loads = np.array([x["wheel_normal_load_n"] for x in window])
    unloaded = np.all(counts == 0, axis=1) & np.all(loads == 0., axis=1) & (gap > margin)
    groups = runs(unloaded.tolist())
    vz = np.array([x["start_com_vz_mps"] for x in window])
    peak = int(np.argmax(gap))
    vp = int(np.argmax(vz))
    result = {
        "executed_window": [start, stop], "native_count": len(window),
        "maximum_simultaneous_gap_m": float(gap.max()),
        "maximum_net_gap_m": float(np.maximum(0., gap-margin).max()),
        "peak_gap_native_index": window[peak]["index"],
        "peak_gap_returned_time_s": window[peak]["end_time_s"],
        "unloaded_native_count": int(unloaded.sum()),
        "longest_unloaded_ms": 2 * max((b-a for a,b in groups), default=0),
        "max_start_com_vz_mps": float(vz.max()),
        "max_com_vz_native_index": window[vp]["index"],
        "sampled_unloaded_runs": [],
    }
    for a, b in groups:
        segment = window[a:b]
        result["sampled_unloaded_runs"].append({
            "first_native_index": segment[0]["index"], "end_native_index_exclusive": segment[-1]["index"]+1,
            "duration_s": sum(x["actual_dt_s"] for x in segment),
            "onset_com_vz_mps": segment[0]["start_com_vz_mps"],
            "last_returned_com_vz_mps": segment[-1]["returned_com_velocity_mps"][2],
            "net_gap_max_m": max(max(0.,x["endpoint_min_gap_m"]-x["contact_margin_m"]) for x in segment),
        })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    work = args.work
    inputs = []
    protocol_path = work / "rl_jump_training_plan_01/evaluation_protocol.json"
    protocol = read_json(protocol_path)
    inputs.append(protocol_path)
    evaluation = []
    leg_indices = [i for i in range(16) if i % 4 != 3]
    for case in protocol["cases"]:
        for condition in protocol["conditions"]:
            folder = work / "rl_jump_evaluation_01" / f"{case['name']}__{condition}"
            record_path, trace_path = folder / "records.json", folder / "trace.jsonl.gz"
            inputs.extend((record_path, trace_path, folder / "physical_score.json"))
            record, trace = read_json(record_path), list(rows(trace_path))
            start, stop = (200, 600) if case["request_tick"] is None else (case["request_tick"], case["request_tick"]+120)
            selected = trace[start:stop]
            targets = np.array([x["controller"]["leg_extension_target_m"] for x in selected])
            requests = np.array([x["controller"]["requested_extension_m"] for x in selected])
            rate = np.array([x["controller"]["joint_target_rate_limited"] for x in selected])[:, leg_indices]
            torque = np.array([x["controller"]["torque_limited"] for x in selected])
            endpoints = record["endpoints"]
            epw = endpoints[start+1:stop+1]
            terms = {k: sum(x["info"]["reward_terms"][k] for x in trace) for k in trace[0]["info"]["reward_terms"]}
            raw_heights = sorted(set(x["raw_command"]["clearance_m"] for x in selected))
            score = read_json(folder / "physical_score.json")
            evaluation.append({
                "case": case["name"], "condition": condition,
                "window_meaning": "hold postsettling" if case["request_tick"] is None else "executions [s,s+120), endpoints S[s+1:s+121], native[5s:5(s+120)]",
                "request_native": request_native_summary(record, start, stop),
                "control_endpoint_window_max_gap_m": max(x["min_gap_m"] for x in epw),
                "control_endpoint_window_max_net_gap_m": max(max(0.,x["min_gap_m"]-x["contact_margin_m"]) for x in epw),
                "raw_heights_m": raw_heights,
                "all_raw_v_yaw_zero": all(x["raw_command"]["forward_velocity_mps"] == 0. and x["raw_command"]["yaw_rate_rps"] == 0. for x in trace),
                "actions": action_summary([x["action"] for x in selected]),
                "leg_target_clipped_intervals": int(np.any(rate,axis=1).sum()),
                "leg_target_clipped_axis_intervals": int(rate.sum()),
                "ik_extension_clipped_intervals": int(np.any(abs(targets-requests)>1e-12,axis=1).sum()),
                "torque_protected_intervals": int(np.any(torque,axis=1).sum()),
                "peak_requested_leg_torque_nm": max(abs(x["controller"]["requested_torque_nm"][j]) for x in selected for j in leg_indices),
                "peak_requested_wheel_torque_nm": max(abs(x["controller"]["requested_torque_nm"][j]) for x in selected for j in (3,7,11,15)),
                "peak_planar_displacement_m": max(x["planar_displacement_m"] for x in endpoints),
                "peak_abs_roll_pitch_rad": max(max(abs(x['roll_rad']),abs(x['pitch_rad'])) for x in endpoints),
                "peak_heading_error_rad": max(abs(x["heading_error_rad"]) for x in endpoints),
                "total_reward": sum(terms.values()), "reward_terms": terms,
                "score_passed": score["passed"], "score_checks": score["checks"],
                "late_settled": score["late_settled"],
            })
    training = work / "rl_jump_formal_01"
    episodes_path, final_path = training / "episodes.jsonl.gz", training / "final_snapshot.json"
    episodes = {x["episode_index"]: dict(x, completed=True) for x in rows(episodes_path)}
    final = read_json(final_path)
    if not final["episode_finished"]:
        episodes[final["episode_index"]] = {"episode_index": final["episode_index"], "spec": final["spec"], "completed": False}
    sums = defaultdict(lambda: defaultdict(float))
    actions = defaultdict(list)
    stage = defaultdict(lambda: {"episodes": [], "requested_control_steps": 0, "fully_unloaded_control_steps": 0, "positive_task_bonus_steps": 0,
                                 "flight_event_steps": 0, "clearance_event_steps": 0, "landing_event_steps": 0,
                                 "maximum_raw_request_window_gap_m": -1e30, "maximum_window_net_gap_m": 0.,
                                 "protected_request_steps": 0, "rate_flagged_request_steps_all16": 0})
    events = []
    count = 0
    trace_path = training / "training_trace.jsonl.gz"
    for row in rows(trace_path):
        count += 1
        eid = row["episode_index"]
        spec = episodes[eid]["spec"]
        key = str(spec["net_clearance_m"])
        for name,value in row["reward_terms"].items(): sums[eid][name] += value
        request = spec["request_tick"]
        window = request is not None and request <= row["executed_tick"] < request+120
        group = key + ("_request" if window else "_hold" if request is None else "_outside_request")
        actions[group].append(row["action"])
        if window:
            st = stage[key]
            st["requested_control_steps"] += 1
            st["fully_unloaded_control_steps"] += int(row["endpoint"]["fully_unloaded"])
            st["positive_task_bonus_steps"] += int(any(row["reward_terms"][n]>0 for n in ("clearance_progress","flight_once","landing_once")))
            for term,field in (("flight_once","flight_event_steps"),("clearance_progress","clearance_event_steps"),("landing_once","landing_event_steps")):
                st[field] += int(row["reward_terms"][term]>0)
            st["maximum_raw_request_window_gap_m"] = max(st["maximum_raw_request_window_gap_m"],row["geometry"]["simultaneous_minimum_gap_m"])
            st["maximum_window_net_gap_m"] = max(st["maximum_window_net_gap_m"],row["endpoint"]["net_gap_m"])
            st["protected_request_steps"] += int(row["torque_protected_axes"]>0)
            st["rate_flagged_request_steps_all16"] += int(row["rate_limited_axes"]>0)
        if row["reward_terms"]["flight_once"] > 0:
            events.append({"episode_index":eid,"total_transition":row["total_transition"],"stage_goal_m":spec["net_clearance_m"],
                           "offset":row["executed_tick"]-request,"credited_gap_m":row["progress_event"]["credited_net_gap_m"],
                           "action":row["action"],"reward":row["reward"]})
    for eid, episode in episodes.items():
        episode["reward_terms_recomputed"] = dict(sums[eid])
        episode["return_recomputed"] = sum(sums[eid].values())
        stage[str(episode["spec"]["net_clearance_m"])]["episodes"].append(episode)
    stages = {}
    for key,st in stage.items():
        eps = st.pop("episodes")
        complete = [x for x in eps if x["completed"]]
        flight = [x for x in complete if x["reward_terms_recomputed"].get("flight_once",0)>0]
        noflight = [x for x in complete if x not in flight]
        allterms = {name:sum(x["reward_terms_recomputed"].get(name,0) for x in complete) for name in sums[0]}
        stages[key] = {**st,"completed_episodes":len(complete),"partial_episodes":len(eps)-len(complete),
                       "flight_episodes":len(flight),"landing_episodes":sum(x["reward_terms_recomputed"].get("landing_once",0)>0 for x in complete),
                       "mean_episode_return":float(np.mean([x["return_recomputed"] for x in complete])) if complete else None,
                       "mean_flight_episode_return":float(np.mean([x["return_recomputed"] for x in flight])) if flight else None,
                       "mean_nonflight_episode_return":float(np.mean([x["return_recomputed"] for x in noflight])) if noflight else None,
                       "complete_episode_reward_sums":allterms}
        if stages[key]["maximum_raw_request_window_gap_m"] == -1e30:
            stages[key]["maximum_raw_request_window_gap_m"] = None
    updates_path = training / "updates.jsonl"
    updates = [json.loads(x) for x in updates_path.read_text().splitlines()]
    inputs.extend((trace_path, episodes_path, final_path, updates_path, training/"receipt.json",work/"rl_formal_full_log_audit_01.json"))
    report={"schema":"d1-jump-ppo-failure-file-diagnosis-v1","new_physics_steps":0,"new_training_steps":0,
            "evaluation":evaluation,"training":{"rows":count,"stages":stages,"actions":{k:action_summary(v) for k,v in actions.items()},
                "flight_events":events,"training_progress_is_control_quantized_not_independent_native_evaluation":True,
                "rate_flag_warning":"Training compact rate_limited_axes includes wheel slots whose target copies current wheel position; do not interpret it as actual leg slew saturation. Evaluation excludes wheel slots.",
                "update_first":updates[0],"update_last":updates[-1]},
            "limits":["No rollout or controller/policy reconstruction performed.","Requested gap statistics exclude reset settling and use [s,s+120) executions; control endpoints and2ms native returned geometry are separately labeled.","Current reward/action associations and projection fractions are descriptive, not interventions or proof of achievable20mm hops.","Training event labels are saved control-quantized accounting, not a retrospective physical qualification of all training episodes."],
            "input_sha256":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}}
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"output":str(args.output),"sha256":hashlib.sha256(args.output.read_bytes()).hexdigest(),"training_rows":count,
                      "stage_counts":{k:{n:v for n,v in s.items() if n in ('completed_episodes','flight_episodes','positive_task_bonus_steps','requested_control_steps','landing_episodes','mean_episode_return')} for k,s in stages.items()},"new_physics_steps":0},indent=2))


if __name__ == "__main__":
    main()
