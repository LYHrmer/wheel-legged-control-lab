"""Independent physical gates for the five frozen stationary-hop evaluations."""
from __future__ import annotations

import json
import math

import numpy as np

from scripts.d1_jump_readiness import sampled_true_runs
from scripts.d1_jump_readiness_records import _execution_valid, _late_settled

FIXED_GATES = {
    "completed_controls": 600, "heading_peak_rad": math.radians(5),
    "roll_or_pitch_peak_rad": math.radians(10), "planar_displacement_peak_m": .1,
    "jump_native_consecutive_unloaded_intervals_min": 10, "jump_sampled_airtime_min_s": .02,
    "jump_simultaneous_net_clearance_min_m": .02, "landing_before_endpoint_tick": 400,
    "late_endpoint_start_inclusive": 400, "late_endpoint_stop_exclusive": 600,
    "late_held_height_rmse_max_m": .015, "late_abs_body_forward_velocity_max_mps": .03,
    "late_abs_whole_robot_com_vertical_velocity_max_mps": .03,
    "late_path_state_start_inclusive": 400, "late_path_state_stop_inclusive": 600,
    "late_planar_path_max_m": .05, "load_occupancy_final_native_intervals": 1000,
    "per_wheel_positive_normal_load_fraction_min": .95,
    "hold_forbid_certified_flight_from_executed_tick": 200,
}


def score_jump_episode(episode, case, protocol):
    """Score saved native records, never shaped reward or time-driven phase labels."""
    if protocol.get("schema") != "d1-jump-ppo-independent-evaluation-v1":
        raise ValueError("unsupported jump evaluation protocol")
    if any(isinstance(protocol["gates"].get(k), bool) or protocol["gates"].get(k) != v
           for k, v in FIXED_GATES.items()):
        raise ValueError("frozen physical evaluation gates changed")
    if json.dumps(case, sort_keys=True) not in [json.dumps(c, sort_keys=True) for c in protocol["cases"]]:
        raise ValueError("case is not a declared frozen evaluation member")
    execution, late = _execution_valid(episode), _late_settled(episode)
    rows = episode.get("intervals", [])
    result = {"schema": "d1-jump-ppo-physical-score-v1", "case": case["name"],
              "passed": False, "execution": execution, "late_settled": late,
              "flight_runs": [], "requested_qualified_runs": [], "max_qualified_net_gap_m": None,
              "loaded_return": None, "checks": {"execution": execution["passed"], "late": late["passed"]},
              "notes": ["Native solved-contact samples and returned-pose geometry are distinct recorded phases.",
                        "Sampled flight evidence does not prove unsampled continuous-time clearance."]}
    record_keys = ("native_time_continuous", "native_contact_fields_valid", "endpoint_tick_sequence")
    if not all(execution["checks"][key] for key in record_keys):
        result["checks"]["valid_native_records"] = False
        return result
    try:
        gaps = np.array([r["endpoint_min_gap_m"] for r in rows], dtype=float)
        margins = np.array([r["contact_margin_m"] for r in rows], dtype=float)
        vz = np.array([r["start_com_vz_mps"] for r in rows], dtype=float)
        loads = np.array([r["wheel_normal_load_n"] for r in rows], dtype=float)
        counts = np.array([r["active_wheel_contacts"] for r in rows], dtype=int)
        starts = np.array([r["start_time_s"] for r in rows], dtype=float)
        ends = np.array([r["end_time_s"] for r in rows], dtype=float)
    except (KeyError, TypeError, ValueError):
        result["checks"]["valid_native_geometry"] = False
        return result
    finite = all(np.isfinite(a).all() for a in (gaps, margins, vz, loads, starts, ends))
    if not finite or np.any(margins < 0.) or loads.shape != (3000, 4):
        result["checks"]["valid_native_geometry"] = False
        return result
    net = np.maximum(0., gaps-margins)
    unloaded = (np.all(counts == 0, axis=1) & np.all(loads == 0., axis=1) & (gaps > margins))
    native_indices = np.arange(3000)
    result["peak_per_wheel_normal_load_n"] = np.max(loads, axis=0).tolist()

    def runs(mask):
        qualified = []
        for run in sampled_true_runs(mask, starts, ends):
            first, stop = int(run["first_index"]), int(run["end_index_exclusive"])
            if (run["intervals"] >= 10 and run["summed_interval_duration_s"] + 1e-10 >= .02
                    and vz[first] > 0.):
                qualified.append({**dict(run), "start_com_vz_mps": float(vz[first]),
                                  "max_net_gap_m": float(np.max(net[first:stop]))})
        return qualified

    result["flight_runs"] = runs(unloaded)
    if case["request_tick"] is None:
        # Certify the real onset before checking overlap; tick200 is not a new takeoff.
        hold_flights = [run for run in result["flight_runs"] if run["end_index_exclusive"] > 1000]
        result["checks"]["no_flight_after_settling"] = not hold_flights
        result["extra_flights"] = len(hold_flights)
    else:
        start = int(case["request_tick"]) * 5
        stop = (int(case["request_tick"]) + 120) * 5
        qualified = runs(unloaded & (native_indices >= start) & (native_indices < stop))
        result["requested_qualified_runs"] = qualified
        useful = [run for run in qualified if run["max_net_gap_m"] >= .02]
        result["max_qualified_net_gap_m"] = max((r["max_net_gap_m"] for r in qualified), default=None)
        result["checks"]["requested_flight"] = bool(qualified)
        result["checks"]["useful_net_clearance"] = bool(useful)
        loaded = np.all(loads > 0., axis=1) & np.all(counts >= 1, axis=1)
        for run in useful:
            valid = np.flatnonzero(loaded & (native_indices >= run["end_index_exclusive"])
                                  & (ends < 4.-1e-10))
            if valid.size:
                i = int(valid[0])
                result["loaded_return"] = {"native_index": i, "end_time_s": float(ends[i]),
                                           "normal_load_n": loads[i].tolist()}
                break
        result["checks"]["loaded_return_before_endpoint400"] = result["loaded_return"] is not None
        result["extra_flights"] = max(0, len(result["flight_runs"])-1)
    result["passed"] = all(result["checks"].values())
    return result
