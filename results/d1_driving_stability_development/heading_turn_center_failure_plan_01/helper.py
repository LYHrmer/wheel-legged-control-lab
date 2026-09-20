"""Rebuild saved turn kinematics and compare existing force logs; no integration.

Only mj_kinematics, mj_comPos, mj_comVel, mj_jacBody, mj_jac and
mj_objectVelocity are called on scratch data populated from saved qpos/qvel.
No env/controller, force solve, collision refresh, rollout or parameter search.
"""
import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np

from wheel_legged_control.d1.model import D1_JOINT_NAMES, LEG_PREFIXES, build_d1_model


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    with gzip.open(path, "rt") as stream:
        return [json.loads(line) for line in stream]


def fit(y, vx, weights=None):
    y, vx = np.asarray(y), np.asarray(vx)
    weights = np.ones(len(y)) if weights is None else np.asarray(weights)
    if not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("yaw fit requires positive total archived load")
    weights = weights / weights.sum()
    centered = y - weights @ y
    spread = float(weights @ centered**2)
    if spread <= 1e-4:
        raise ValueError("degenerate contact/center lateral geometry")
    return float(-(weights * centered) @ vx / spread)


WINDOWS = ((200, 210), (210, 225), (225, 250), (200, 250),
           (250, 275), (275, 300), (250, 300), (300, 450), (450, 800))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    repo, work = args.repo.resolve(), args.work.resolve()
    hashes = {str(Path(__file__).resolve()): sha(__file__)}
    frozen = json.loads((repo / "results/d1_budget_study/protocol.json").read_text())["source_sha256"]
    assert len(frozen) == 77
    for relative, expected in frozen.items():
        assert sha(repo / relative) == expected
        hashes[str(repo / relative)] = expected
    for relative in ("scripts/d1_turn_center_compensation.py", "scripts/probe_d1_heading_turn_center.py",
                     "scripts/d1_turn_contact_diagnostics.py", "scripts/d1_native_contact_diagnostics.py"):
        hashes[str(repo / relative)] = sha(repo / relative)
    model = build_d1_model()
    data = mujoco.MjData(model)
    assert model.nhfield == 0
    base = model.body("base_link").id
    wheel_ids = [model.body(leg + "_foot").id for leg in LEG_PREFIXES]
    dofs = np.array([model.jnt_dofadr[model.joint(name).id] for name in D1_JOINT_NAMES]).reshape(4, 4)
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    velocity = np.zeros(6)
    calls = Counter()

    def api(name, *values):
        calls[name] += 1
        return getattr(mujoco, name)(*values)

    results = []
    global_checks = {"body_rate_closure_max_rps": 0., "point_velocity_closure_max_mps": 0.,
                     "point_decomposition_closure_max_mps": 0., "yaw_fit_identity_max_rps": 0.,
                     "center_log_closure_max_mps": 0., "native_moment_closure_max_nm": 0.}
    for label, dirname in (("baseline", "flat_plane_02"), ("center_candidate", "plane_turn_center_01")):
        source = work / dirname
        for filename in ("protocol.json", "summary.json", "manifest.json"):
            hashes[str(source / filename)] = sha(source / filename)
        root_manifest = json.loads((source / "manifest.json").read_text())
        for relative, identity in root_manifest.items():
            path = source / relative
            assert path.stat().st_size == identity["bytes"] and sha(path) == identity["sha256"]
            hashes[str(path)] = identity["sha256"]
        for side, sign in (("left", 1), ("right", -1)):
            folder = source / f"stationary_turn_{side}_hold"
            manifest = json.loads((folder / "complete_manifest.json").read_text())
            hashes[str(folder / "complete_manifest.json")] = sha(folder / "complete_manifest.json")
            for relative, identity in manifest.items():
                path = folder / relative
                assert path.stat().st_size == identity["bytes"] and sha(path) == identity["sha256"]
                hashes[str(path)] = identity["sha256"]
            trace = read_rows(folder / "trace.jsonl.gz")
            diag = read_rows(folder / "plane_diagnostics.jsonl.gz")
            native = read_rows(folder / "native_physics_entries.jsonl.gz")
            compensation = read_rows(folder / "turn_compensation.jsonl.gz") if label == "center_candidate" else None
            summary = json.loads((folder / "summary.json").read_text())
            with np.load(folder / "states.npz", allow_pickle=False) as z:
                qpos, qvel = z["qpos"].copy(), z["qvel"].copy()
            assert qpos.shape == (801, model.nq) and qvel.shape == (801, model.nv)
            assert len(trace) == len(diag) == 800 and len(native) == 4000
            assert all(entry["returned"] and abs(entry["actual_dt_s"]-.002) < 1e-12 for entry in native)
            records, decisions = [], []
            for k in range(200, 801):
                data.qpos[:] = qpos[k]
                data.qvel[:] = qvel[k]
                warm = data.qacc_warmstart.copy()
                api("mj_kinematics", model, data)
                api("mj_comPos", model, data)
                api("mj_comVel", model, data)
                rotation = data.xmat[base].reshape(3, 3)
                yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
                heading = np.array([np.cos(yaw), np.sin(yaw), 0.])
                lateral_axis = np.array([-np.sin(yaw), np.cos(yaw), 0.])
                lateral = (data.xpos[wheel_ids] - data.xpos[base]) @ lateral_axis
                center_u, center_u_body = [], []
                for leg, body in enumerate(wheel_ids):
                    api("mj_jacBody", model, data, jp, jr, body)
                    relative = jp[:, dofs[leg, :3]] @ qvel[k, dofs[leg, :3]]
                    center_u.append(float(heading @ relative))
                    center_u_body.append(float(rotation[:, 0] @ relative))
                center_u = np.asarray(center_u)
                api("mj_objectVelocity", model, data, mujoco.mjtObj.mjOBJ_BODY, base, velocity, 0)
                body_rate = float(rotation[:, 2] @ velocity[:3])
                body_closure = abs(body_rate-trace[k-1]["heading_task"]["truth_yaw_rate_after_rps"])
                global_checks["body_rate_closure_max_rps"] = max(global_checks["body_rate_closure_max_rps"], body_closure)
                if k < 800:
                    recorded_u = compensation[k]["record"]["leg_center_forward_mps"] if compensation else diag[k]["leg_forward_velocity_before_mps"]
                    calculated_u = center_u if compensation else center_u_body
                    global_checks["center_log_closure_max_mps"] = max(global_checks["center_log_closure_max_mps"], float(np.max(np.abs(np.asarray(recorded_u)-calculated_u))))
                endpoint = diag[k-1]["endpoint_contacts"]
                assert abs(endpoint["measurement_time_s"]-.01*k) < 1e-11
                contacts = endpoint["contacts"]
                loads = np.array([c["normal_force_n"] for c in contacts])
                assert np.all(loads >= 0) and loads.sum() > 0
                y = (np.array([c["pos_world_m"] for c in contacts])-data.xpos[base]) @ lateral_axis
                components = {name: fit(y, np.array([c[field] for c in contacts]) @ heading, loads)
                    for name, field in (("base", "free_base_velocity_world_mps"), ("leg", "leg_dof_velocity_world_mps"),
                                        ("wheel", "wheel_dof_velocity_world_mps"), ("slip", "relative_velocity_world_mps"))}
                closure = abs(components["base"]+components["leg"]+components["wheel"]-components["slip"])
                global_checks["yaw_fit_identity_max_rps"] = max(global_checks["yaw_fit_identity_max_rps"], closure)
                for c in contacts:
                    assert np.array_equal(c["normal_world"], [0., 0., 1.])
                    api("mj_jac", model, data, jp, jr, np.array(c["pos_world_m"]), c["robot_body_id"])
                    global_checks["point_velocity_closure_max_mps"] = max(global_checks["point_velocity_closure_max_mps"], float(np.max(np.abs(jp@qvel[k]-c["relative_velocity_world_mps"]))))
                    global_checks["point_decomposition_closure_max_mps"] = max(global_checks["point_decomposition_closure_max_mps"], float(c["decomposition_residual_norm_mps"]))
                matched_center = fit(y, np.array([center_u[c["wheel_index"]] for c in contacts]), loads)
                r_w = fit(lateral, .087*qvel[k, dofs[:, 3]])
                a_center = -fit(lateral, center_u)
                record = {"endpoint_tick": k, "actual_heading_rad": yaw, "body_yaw_rate_rps": body_rate,
                    "heading_error_rad": trace[k-1]["heading_error_rad"],
                    "point_component_yaw_fit_rps": components, "contact_spin_yaw_rps": -components["wheel"],
                    "full_leg_gap_rps": components["leg"], "center_matched_gap_rps": matched_center,
                    "carrier_rotation_gap_rps": components["leg"]-matched_center, "slip_gap_rps": -components["slip"],
                    "spin_minus_base_gap_rps": -components["wheel"]-components["base"],
                    "center_wheel_yaw_fit_rps": r_w, "center_a_rps": a_center,
                    "center_u_heading_mps": center_u.tolist(), "common_center_u_mps": float(center_u.mean()),
                    "center_relation_residual_rps": body_rate-r_w-a_center,
                    "point_tangent_speed_rms_mps": float(np.sqrt(np.average([c["tangential_relative_speed_mps"]**2 for c in contacts], weights=loads))),
                    "endpoint_normal_load_n": float(loads.sum()),
                    "loaded_wheels": len({c["wheel_index"] for c in contacts if c["normal_force_n"] > 0})}
                records.append(record)
                if k < 800:
                    d = diag[k]
                    target_fit = fit(lateral, .087*np.array(d["wheel_target_rad_s"]))
                    effective = float(d["effective_yaw_request_rps"])
                    decisions.append({"execution_tick": k,
                        "body_yaw_rate_rps": body_rate, "effective_yaw_request_rps": effective,
                        "body_effective_yaw_error_rps": effective-body_rate,
                        "wheel_target_yaw_fit_rps": target_fit, "wheel_yaw_tracking_error_rps": target_fit-r_w,
                        "body_minus_wheel_error_rps": effective-body_rate-(target_fit-r_w),
                        "center_a_rps": a_center, "center_relation_residual_rps": record["center_relation_residual_rps"],
                        "active": bool(compensation[k]["record"]["active"]) if compensation else False,
                        "max_abs_wheel_error_rad_s": float(np.max(np.abs(d["wheel_error_before_rad_s"]))),
                        "mean_abs_wheel_error_rad_s": float(np.mean(np.abs(d["wheel_error_before_rad_s"]))),
                        "max_abs_wheel_target_rad_s": float(np.max(np.abs(d["wheel_target_rad_s"]))),
                        "max_abs_wheel_torque_nm": float(np.max(np.abs(np.array(d["requested_torque_nm"])[[3,7,11,15]]))),
                        "inner_cap_occupied": bool(d["inner_yaw_limit_occupied"])})
                assert data.time == 0. and np.array_equal(data.qpos, qpos[k]) and np.array_equal(data.qvel, qvel[k]) and np.array_equal(data.qacc_warmstart, warm)
            windows = []
            for lo, hi in WINDOWS:
                ep = [r for r in records if lo < r["endpoint_tick"] <= hi]
                dec = [r for r in decisions if lo <= r["execution_tick"] < hi]
                nt = native[5*lo:5*hi]
                dt = np.array([e["actual_dt_s"] for e in nt])
                moments, normals, friction = [], [], []
                for entry in nt:
                    info = entry["contacts"]
                    assert info["max_horizontal_normal"] == 0. and info["max_vertical_normal_error"] == 0.
                    mx = my = mt = 0.
                    for c in info["contacts"]:
                        arm = np.array(c["pos_world_m"])-info["reference_world_m"]
                        force = np.array(c["force_world_n"])
                        mx += -arm[1]*force[0]
                        my += arm[0]*force[1]
                        mt += c["torque_world_nm"][2]
                    global_checks["native_moment_closure_max_nm"] = max(global_checks["native_moment_closure_max_nm"], abs(mx+my+mt-info["total_wrench_world_6"][5]))
                    moments.append([mx, my, mt, info["total_wrench_world_6"][5]])
                    normals.append(info["summed_normal_force_world_n"][2])
                    friction.append(sum(np.linalg.norm(c["tangent_force_world_n"]) for c in info["contacts"]))
                mean = lambda key: float(np.mean([r[key] for r in ep]))
                dmean = lambda key: float(np.mean([r[key] for r in dec]))
                window = {"execution_ticks_half_open": [lo, hi], "endpoint_ticks_inclusive": [lo+1, hi],
                    "endpoint_means": {key: mean(key) for key in ("body_yaw_rate_rps", "contact_spin_yaw_rps", "full_leg_gap_rps",
                        "center_matched_gap_rps", "carrier_rotation_gap_rps", "slip_gap_rps", "spin_minus_base_gap_rps",
                        "center_wheel_yaw_fit_rps", "center_a_rps", "center_relation_residual_rps")},
                    "center_u_rms_mps": float(np.sqrt(np.mean([np.square(r["center_u_heading_mps"]) for r in ep]))),
                    "center_common_u_rms_mps": float(np.sqrt(np.mean([r["common_center_u_mps"]**2 for r in ep]))),
                    "point_tangent_speed_rms_mps": float(np.sqrt(np.mean([r["point_tangent_speed_rms_mps"]**2 for r in ep]))),
                    "frames_without_all_four_loaded_wheels": sum(r["loaded_wheels"] < 4 for r in ep),
                    "decision_means": {key: dmean(key) for key in ("body_yaw_rate_rps", "effective_yaw_request_rps", "body_effective_yaw_error_rps",
                        "wheel_target_yaw_fit_rps", "wheel_yaw_tracking_error_rps", "body_minus_wheel_error_rps", "center_a_rps", "mean_abs_wheel_error_rad_s")},
                    "native_time_weighted_yaw_moment_world_x_world_y_contact_torque_net_nm": np.average(moments, axis=0, weights=dt).tolist(),
                    "native_time_weighted_normal_load_n": float(np.average(normals, weights=dt)),
                    "native_sum_tangent_magnitude_over_mu_normal_load": float(dt@np.array(friction)/(.9*(dt@np.array(normals)))),
                    "native_no_active_wheel_contact_substeps": sum(not any(e["contacts"]["active_contacts_by_wheel"]) for e in nt),
                    "peak_abs_wheel_target_rad_s": max(r["max_abs_wheel_target_rad_s"] for r in dec),
                    "peak_abs_wheel_torque_nm": max(r["max_abs_wheel_torque_nm"] for r in dec),
                    "inner_cap_intervals": sum(r["inner_cap_occupied"] for r in dec)}
                windows.append(window)
            peak = max(trace, key=lambda row: abs(row["heading_error_rad"]))
            results.append({"source": label, "case": folder.name, "turn_sign": sign, "gates": summary["gates"],
                "peak_heading_error_rad": float(peak["heading_error_rad"]), "peak_endpoint_tick": peak["endpoint_tick"],
                "actual_heading_at_peak_rad": peak["heading_task"]["truth_heading_after"],
                "reference_heading_at_peak_rad": peak["heading_task"]["reference_heading_after"],
                "protected_joint_intervals": sum(sum(d["torque_protected"]) for d in diag),
                "applied_all_five_equal_request": all(np.array_equal(d["applied_torque_nm"], np.tile(d["requested_torque_nm"], (5,1))) for d in diag),
                "windows": windows, "endpoints": records, "decisions": decisions,
                "boundary_endpoints": [r for r in records if r["endpoint_tick"] in (200,201,205,210,215,225,240,249,250,251,255,275,300,450,800)]})
    comparisons = []
    for side in ("left", "right"):
        b = next(r for r in results if r["source"] == "baseline" and side in r["case"])
        c = next(r for r in results if r["source"] == "center_candidate" and side in r["case"])
        delta_windows = []
        for bw, cw in zip(b["windows"], c["windows"]):
            changes = {key: cw["endpoint_means"][key]-bw["endpoint_means"][key] for key in bw["endpoint_means"]}
            spin_change = changes["contact_spin_yaw_rps"]
            delta_windows.append({"execution_ticks_half_open": cw["execution_ticks_half_open"], "changes_candidate_minus_baseline_rps": changes,
                "incremental_accounting_not_causal": {key: changes[key]/spin_change if abs(spin_change)>1e-12 else None
                    for key in ("body_yaw_rate_rps", "full_leg_gap_rps", "slip_gap_rps")}})
        comparisons.append({"side": side,
            "peak_error_reduction_fraction": 1-abs(c["peak_heading_error_rad"])/abs(b["peak_heading_error_rad"]),
            "heading_at_250_ratio": c["actual_heading_at_peak_rad"]/b["actual_heading_at_peak_rad"],
            "required_heading_at_250_magnitude_rad": .3-np.deg2rad(5.),
            "windows": delta_windows})
    assert global_checks["body_rate_closure_max_rps"] < 1e-12
    assert global_checks["point_velocity_closure_max_mps"] < 1e-12
    assert global_checks["center_log_closure_max_mps"] < 1e-12
    assert global_checks["yaw_fit_identity_max_rps"] < 1e-12
    assert global_checks["native_moment_closure_max_nm"] < 1e-10
    assert all(sha(path) == expected for path, expected in hashes.items())
    report = {"schema": "d1-turn-center-failure-saved-records-v1", "new_physics_steps": 0,
        "new_controller_compute_calls": 0, "new_contact_force_solves": 0, "mujoco_version": mujoco.__version__,
        "input_sha256": hashes, "inputs_unchanged": True, "frozen77_match": True,
        "actual_kinematic_api_calls": dict(calls), "all_scratch_times_zero_and_qpos_qvel_warmstart_unchanged": True,
        "closure_checks": global_checks,
        "phase_semantics": {"decision": "saved S[t] and its actual controller target/memory", "endpoint": "archived synchronized contact cache at S[t+1]",
                            "native": "archived native solved cache; force moments use its own stored world reference and contact positions"},
        "fit_identity": "contact spin yaw - free-base yaw fit = full leg yaw fit - slip yaw fit; loaded point fits share weights and lateral coordinates",
        "limitations": ["Kinematic component fractions are accounting identities, not causal/energy shares",
            "No native force is multiplied by a differently phased endpoint velocity", "Positive archived endpoint normal loads distinguish loaded contacts from geometry candidates",
            "No new controller, rollout, contact solve or gain/cap/PI search; a changing yaw-fit error does not predict counterfactual closed-loop stability"],
        "cases": results, "comparisons": comparisons}
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"report_sha256": sha(args.output), "checks": global_checks, "api_calls": dict(calls),
        "comparisons": [{k:v for k,v in r.items() if k != "windows"} for r in comparisons]}))


if __name__ == "__main__":
    def forbidden(*args, **kwargs):
        raise AssertionError("saved-record diagnosis must not integrate or solve contacts")
    from contextlib import ExitStack
    with ExitStack() as guards:
        for name in ("mj_step", "mj_step1", "mj_step2", "mj_forward", "mj_inverse", "mj_collision", "mj_fwdPosition", "mj_fwdVelocity", "mj_fwdActuation", "mj_fwdAcceleration", "mj_fwdConstraint"):
            guards.enter_context(patch.object(mujoco, name, forbidden))
        main()
