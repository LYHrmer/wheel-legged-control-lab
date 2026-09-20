"""File-only audit of the immutable 600-row legacy jump trace; no model imports."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path


def span_runs(rows, predicate):
    runs, start = [], None
    for index in range(len(rows)+1):
        on = index < len(rows) and predicate(rows[index])
        if on and start is None:
            start = index
        if not on and start is not None:
            runs.append({"first_row":start,"last_row":index-1,
                         "first_endpoint_tick":start+1,"last_endpoint_tick":index,
                         "first_time_s":rows[start]["time_s"],
                         "last_time_s":rows[index-1]["time_s"],
                         "sample_count":index-start,
                         "endpoint_span_s":rows[index-1]["time_s"]-rows[start]["time_s"],
                         "nominal_sample_bin_duration_s":.01*(index-start)})
            start = None
    return runs


def audit(folder):
    paths = [folder/name for name in ("trace.json.gz","summary.json","protocol.json","jump_probe.py")]
    raw = gzip.decompress(paths[0].read_bytes())
    rows = json.loads(raw)
    summary, protocol = [json.loads(p.read_text()) for p in paths[1:3]]
    if len(rows) != 600:
        raise ValueError("expected exact original 600-row trace")
    if hashlib.sha256(raw).hexdigest() != protocol["raw_trace"]["sha256"] or len(raw) != protocol["raw_trace"]["bytes"]:
        raise ValueError("decompressed original trace identity failed")
    minima = [min(row["wheel_bottoms_m"]) for row in rows]
    peak_i = max(range(200,400),key=lambda i:minima[i])
    base_i = max(range(200,400),key=lambda i:rows[i]["height_m"])
    base0 = rows[199]["height_m"]

    def point(index):
        row=rows[index]; q=row["qpos"]; w,x,y,z=q[3:7]
        tilt=[math.atan2(2*(w*x+y*z),1-2*(x*x+y*y)),math.asin(max(-1.,min(1.,2*(w*y-z*x))))]
        return {"row":index,"endpoint_tick":index+1,"time_s":row["time_s"],"phase":row["phase"],
                "base_origin_height_m":row["height_m"],"base_origin_rise_from_precommand_m":row["height_m"]-base0,
                "base_origin_vertical_speed_mps":row["vertical_speed_mps"],
                "simultaneous_min_wheel_bottom_m":minima[index],"wheel_bottoms_m":row["wheel_bottoms_m"],
                "requested_vertical_feedforward_n":row["requested_vertical_force_n"],
                "allocated_support_request_sum_n":sum(row["upward_support_n"]),
                "actual_quaternion_roll_pitch_rad":tilt,"legacy_logged_roll_pitch_rad":row["roll_pitch"],
                "peak_abs_requested_torque_nm":max(abs(v) for v in row["torque"])}

    phases=[]
    for name in dict.fromkeys(row["phase"] for row in rows):
        selected=[i for i,row in enumerate(rows) if row["phase"]==name]
        phases.append({"phase":name,"count":len(selected),"runs":span_runs(rows,lambda row:row["phase"]==name),
                       "max_min_wheel_bottom_m":max(minima[i] for i in selected),
                       "min_min_wheel_bottom_m":min(minima[i] for i in selected),
                       "peak_base_origin_height_m":max(rows[i]["height_m"] for i in selected),
                       "allocated_support_request_sum_range_n":[min(sum(rows[i]["upward_support_n"]) for i in selected),max(sum(rows[i]["upward_support_n"]) for i in selected)]})
    thresholds={str(t):span_runs(rows,lambda row,t=t:min(row["wheel_bottoms_m"])>t) for t in (0.,.001,.002,.005,.015,.020)}
    limits=[80.,80.,80.,12.]*4
    torque_counts=[sum(abs(row["torque"][j])>=limits[j] for row in rows) for j in range(16)]
    final=rows[500:600]
    checks={"original_clearance_peak_exact":minima[peak_i]==summary["max_all_wheel_clearance_m"],
            "original_base_origin_rise_exact":rows[base_i]["height_m"]-base0==summary["max_base_rise_m"],
            "original_peak_time_exact":rows[peak_i]["time_s"]==summary["clearance_peak_time_s"],
            "all_times_match_saved_endpoints":all(abs(row["time_s"]-(i+1)*.01)<1e-10 for i,row in enumerate(rows)),
            "no_saved_nonwheel_contact":all(row["nonwheel_contacts"]==0 for row in rows)}
    return {"schema":"legacy-jump-file-only-audit-v1","passed":all(checks.values()),"checks":checks,
            "new_physics_steps":0,"new_model_or_controller_calls":0,"new_contact_solves":0,
            "input_sha256":{str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            "decompressed_trace_sha256":hashlib.sha256(raw).hexdigest(),"saved_control_intervals":600,
            "peak_simultaneous_clearance":point(peak_i),"peak_base_origin_height":point(base_i),
            "precommand_state":point(199),"phase_summaries":phases,
            "geometry_threshold_runs_m":thresholds,
            "threshold_run_interpretation":"Runs describe 10 ms sampled endpoint geometry only. Neither sample-bin duration nor endpoint span proves uninterrupted no-load flight between samples. Contacts/native normal forces were not saved.",
            "rated_torque_counts_by_joint":torque_counts,
            "peak_abs_leg_requested_torque_nm":max(abs(row["torque"][j]) for row in rows for j in range(16) if j%4!=3),
            "peak_abs_wheel_requested_torque_nm":max(abs(row["torque"][j]) for row in rows for j in (3,7,11,15)),
            "final_second_base_origin_height_range_m":[min(x["height_m"] for x in final),max(x["height_m"] for x in final)],
            "final_second_peak_abs_base_vertical_speed_mps":max(abs(x["vertical_speed_mps"]) for x in final),
            "last_state":point(599),
            "evidence_boundaries":["base_origin_rise is qpos[2], not whole-robot COM ballistic rise", "upward_support_n is an allocated controller request, not measured contact load", "phase flight and completed schedule are not measured takeoff/landing", "saved geometry on a native plane establishes sampled clearance, not traversal of a box", "no torque split, native per-substep contact loads, or requested-vs-actual actuator receipt exists in this legacy trace"]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    report=audit(args.input)
    with args.output.open("x") as stream:
        json.dump(report,stream,indent=2,allow_nan=False)
        stream.write("\n")
    print(json.dumps({"passed":report["passed"],"new_physics_steps":0,
                      "clearance_peak":report["peak_simultaneous_clearance"],
                      "base_peak":report["peak_base_origin_height"],
                      "clearance_runs":report["geometry_threshold_runs_m"]},indent=2))


if __name__=="__main__":
    main()
