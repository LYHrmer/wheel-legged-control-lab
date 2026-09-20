"""Offline native-plane turn decomposition and one-step kinematic target audit.

No integration, new collision solve, controller compute or candidate rollout.
All contact forces/point velocities are read from archived synchronized/native
caches. Geometry/Jacobians are calculated at the exact saved endpoint only.
"""
import argparse
import gzip
import hashlib
import json
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


def rms(a):
    return float(np.sqrt(np.mean(np.square(a))))


def fit(y, values, weights=None):
    weights = np.ones(len(y))/len(y) if weights is None else weights/np.sum(weights)
    centered = y - weights @ y
    denominator = float(weights @ centered**2)
    if denominator <= 1e-4:
        raise ValueError("degenerate lateral geometry cannot be zero-filled")
    return float(-(weights*centered) @ values / denominator)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    repo, work = args.repo.resolve(), args.work.resolve()
    source = work / "flat_plane_02"
    hashes = {str(Path(__file__).resolve()): sha(__file__)}
    for path in (source / "protocol.json", source / "summary.json",
                 repo / "scripts/d1_heading_tracking_env.py", repo / "scripts/d1_heading_reference.py"):
        hashes[str(path)] = sha(path)
    frozen = json.loads((repo / "results/d1_budget_study/protocol.json").read_text())["source_sha256"]
    assert len(frozen) == 77
    for path, expected in frozen.items():
        assert sha(repo / path) == expected
        hashes[str(repo / path)] = expected
    model = build_d1_model()
    data = mujoco.MjData(model)
    assert model.nhfield == 0
    base = model.body("base_link").id
    wheels = [model.body(leg + "_foot").id for leg in LEG_PREFIXES]
    dofs = np.array([model.jnt_dofadr[model.joint(n).id] for n in D1_JOINT_NAMES]).reshape(4, 4)
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    velocity = np.zeros(6)
    result = []
    for side, sign in (("left", 1), ("right", -1)):
        folder = source / f"stationary_turn_{side}_hold"
        manifest = json.loads((folder / "complete_manifest.json").read_text())
        hashes[str(folder / "complete_manifest.json")] = sha(folder / "complete_manifest.json")
        for name, identity in manifest.items():
            assert sha(folder / name) == identity["sha256"]
            hashes[str(folder / name)] = identity["sha256"]
        diagnostics = read_rows(folder / "plane_diagnostics.jsonl.gz")
        trace = read_rows(folder / "trace.jsonl.gz")
        native = read_rows(folder / "native_physics_entries.jsonl.gz")
        summary = json.loads((folder / "summary.json").read_text())
        with np.load(folder / "states.npz", allow_pickle=False) as z:
            qpos, qvel = z["qpos"].copy(), z["qvel"].copy()
        assert len(qpos) == 801 and len(trace) == len(diagnostics) == 800 and len(native) == 4000
        records, shadow = [], []
        for k in range(200, 801):
            data.qpos[:] = qpos[k]
            data.qvel[:] = qvel[k]
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)
            mujoco.mj_comVel(model, data)
            assert data.time == 0. and np.array_equal(data.qpos, qpos[k]) and np.array_equal(data.qvel, qvel[k])
            rotation = data.xmat[base].reshape(3, 3)
            yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
            forward = np.array([np.cos(yaw), np.sin(yaw), 0.])
            lateral_axis = np.array([-np.sin(yaw), np.cos(yaw), 0.])
            lateral = (data.xpos[wheels] - data.xpos[base]) @ lateral_axis
            center_u, center_u_body = [], []
            for leg, body in enumerate(wheels):
                mujoco.mj_jacBody(model, data, jp, jr, body)
                relative = jp[:, dofs[leg, :3]] @ qvel[k, dofs[leg, :3]]
                center_u.append(float(forward @ relative))
                center_u_body.append(float(rotation[:, 0] @ relative))
            center_u = np.asarray(center_u)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, base, velocity, 0)
            body_yaw = float(rotation[:, 2] @ velocity[:3])
            assert abs(body_yaw - trace[k - 1]["heading_task"]["truth_yaw_rate_after_rps"]) <= 1e-12
            if k < 800:
                np.testing.assert_allclose(center_u_body, diagnostics[k]["leg_forward_velocity_before_mps"], rtol=0, atol=1e-12)
            endpoint = diagnostics[k - 1]["endpoint_contacts"]
            assert abs(endpoint["measurement_time_s"] - .01*k) < 1e-11
            contacts = endpoint["contacts"]
            loads = np.array([c["normal_force_n"] for c in contacts])
            assert np.all(loads >= 0) and loads.sum() > 0
            y = (np.array([c["pos_world_m"] for c in contacts]) - data.xpos[base]) @ lateral_axis
            components = {}
            for name, field in (("base", "free_base_velocity_world_mps"), ("leg", "leg_dof_velocity_world_mps"),
                                ("wheel", "wheel_dof_velocity_world_mps"), ("slip", "relative_velocity_world_mps")):
                world = np.array([c[field] for c in contacts])
                components[name] = fit(y, world @ forward, loads)
            assert abs(components["base"] + components["leg"] + components["wheel"] - components["slip"]) < 1e-12
            for contact in contacts:
                assert np.array_equal(contact["normal_world"], [0., 0., 1.])
                mujoco.mj_jac(model, data, jp, jr, np.asarray(contact["pos_world_m"]), contact["robot_body_id"])
                np.testing.assert_allclose(jp @ qvel[k], contact["relative_velocity_world_mps"], rtol=0, atol=1e-12)
            center_at_contact = np.array([center_u[c["wheel_index"]] for c in contacts])
            center_fit_matched = fit(y, center_at_contact, loads)
            wheel_fit = fit(lateral, .087*qvel[k, dofs[:, 3]])
            center_a = -fit(lateral, center_u)
            record = {"endpoint_tick": k, "actual_yaw_rad": yaw, "body_yaw_rate_rps": body_yaw,
                "heading_error_rad": trace[k - 1]["heading_error_rad"], "wheel_center_yaw_fit_rps": wheel_fit,
                "a_center_rps": center_a, "center_relative_vx_heading_mps": center_u.tolist(),
                "center_relative_vx_common_mps": float(center_u.mean()),
                "point_component_yaw_fit_rps": components,
                "point_weighted_center_leg_fit_rps": center_fit_matched,
                "wheel_carrier_leg_rotation_remainder_rps": components["leg"] - center_fit_matched,
                "spin_minus_base_gap_rps": -components["wheel"] - components["base"],
                "slip_yaw_gap_rps": -components["slip"],
                "point_tangent_speed_rms_mps": float(np.sqrt(np.average([c["tangential_relative_speed_mps"]**2 for c in contacts], weights=loads))),
                "endpoint_normal_load_n": float(loads.sum()),
                "loaded_wheels": len(set(c["wheel_index"] for c in contacts if c["normal_force_n"] > 0))}
            records.append(record)
            if 200 <= k < 250:
                d = diagnostics[k]
                assert d["raw_user_command"]["forward_velocity_mps"] == 0.
                assert d["raw_user_command"]["yaw_rate_rps"] == sign*.6
                original_target = np.asarray(d["wheel_target_rad_s"])
                effective = float(d["effective_yaw_request_rps"])
                np.testing.assert_allclose(original_target, -.6*sign*lateral/.087, rtol=0, atol=1e-12)
                candidate_unclipped = (-effective*lateral + center_u)/.087
                candidate_target = np.clip(candidate_unclipped, -30., 30.)
                err = candidate_target - qvel[k, dofs[:, 3]]
                old_integral = np.asarray(d["wheel_integral_before_nm"])
                trial_integral = np.clip(old_integral + 3.*.01*err, -4., 4.)
                trial_request = 2.2*err + trial_integral
                accepted = (np.abs(trial_request) <= 12.) | (trial_request*err < 0)
                after_integral = np.where(accepted, trial_integral, old_integral)
                final_request = 2.2*err + after_integral
                shadow.append({"execution_tick": k, "baseline_target_rad_s": original_target.tolist(),
                    "candidate_target_rad_s": candidate_target.tolist(), "center_u_mps": center_u.tolist(),
                    "candidate_minus_baseline_target_rad_s": (candidate_target-original_target).tolist(),
                    "candidate_equivalent_wheel_yaw_target_rps": fit(lateral, .087*candidate_target),
                    "actual_effective_body_yaw_request_rps": effective,
                    "candidate_one_step_wheel_request_nm": final_request.tolist(),
                    "candidate_one_step_integral_after_nm": after_integral.tolist(),
                    "any_target_speed_clipped": bool(np.any(candidate_target != candidate_unclipped)),
                    "any_torque_limit_exceeded": bool(np.any(np.abs(final_request) > 12.)),
                    "any_integral_clipped_or_rejected": bool(np.any(abs(trial_integral) >= 4.) or np.any(~accepted))})
        windows = []
        for lo, hi in ((200, 210), (210, 225), (225, 250), (200, 250), (250, 300), (300, 450), (450, 800)):
            ep = [r for r in records if lo < r["endpoint_tick"] <= hi]
            nt = native[5*lo:5*hi]
            duration = np.array([e["actual_dt_s"] for e in nt])
            force_x_moment, force_y_moment, normal, tangent_magnitude = [], [], [], []
            for entry in nt:
                info = entry["contacts"]
                assert info["max_horizontal_normal"] == 0.
                mx = my = mt = 0.
                for c in info["contacts"]:
                    r = np.asarray(c["pos_world_m"]) - info["reference_world_m"]
                    f = np.asarray(c["force_world_n"])
                    mx += -r[1]*f[0]
                    my += r[0]*f[1]
                    mt += c["torque_world_nm"][2]
                assert abs(mx + my + mt - info["total_wrench_world_6"][5]) < 1e-10
                force_x_moment.append(mx)
                force_y_moment.append(my)
                normal.append(info["summed_normal_force_world_n"][2])
                tangent_magnitude.append(sum(np.linalg.norm(c["tangent_force_world_n"]) for c in info["contacts"]["contacts"]) if isinstance(info["contacts"], dict) else sum(np.linalg.norm(c["tangent_force_world_n"]) for c in info["contacts"]))
            mean = lambda name: float(np.mean([r[name] for r in ep]))
            gap = mean("spin_minus_base_gap_rps")
            leg = float(np.mean([r["point_component_yaw_fit_rps"]["leg"] for r in ep]))
            center = mean("point_weighted_center_leg_fit_rps")
            windows.append({"execution_ticks": [lo, hi], "end_exclusive": True, "endpoint_ticks_inclusive": [lo+1, hi],
                "mean_body_yaw_rps": mean("body_yaw_rate_rps"), "mean_wheel_center_yaw_fit_rps": mean("wheel_center_yaw_fit_rps"),
                "mean_a_center_rps": mean("a_center_rps"), "mean_contact_spin_yaw_rps": -float(np.mean([r["point_component_yaw_fit_rps"]["wheel"] for r in ep])),
                "mean_gap_rps": gap, "mean_full_contact_leg_gap_rps": leg,
                "mean_point_matched_center_gap_rps": center,
                "mean_wheel_carrier_rotation_gap_rps": mean("wheel_carrier_leg_rotation_remainder_rps"),
                "mean_slip_gap_rps": mean("slip_yaw_gap_rps"),
                "leg_over_gap_not_causal": leg/gap if abs(gap)>1e-12 else None,
                "center_over_gap_not_causal": center/gap if abs(gap)>1e-12 else None,
                "endpoint_tangent_speed_rms_mps": rms([r["point_tangent_speed_rms_mps"] for r in ep]),
                "center_common_vx_rms_mps": rms([r["center_relative_vx_common_mps"] for r in ep]),
                "frames_without_all_four_loaded_wheels": sum(r["loaded_wheels"] < 4 for r in ep),
                "native_normal_fz_time_mean_n": float(np.average(normal, weights=duration)),
                "native_yaw_moment_world_x_force_time_mean_nm": float(np.average(force_x_moment, weights=duration)),
                "native_yaw_moment_world_y_force_time_mean_nm": float(np.average(force_y_moment, weights=duration)),
                "native_net_yaw_moment_time_mean_nm": float(np.average([e["contacts"]["total_wrench_world_6"][5] for e in nt], weights=duration)),
                "native_sum_tangent_magnitude_over_mu_normal_load": float(duration @ np.asarray(tangent_magnitude)/(.9*(duration @ np.asarray(normal)))),
                "native_no_active_wheel_contact_substeps": sum(not any(e["contacts"]["active_contacts_by_wheel"]) for e in nt),
                "raw_yaw_mean_rps": float(np.mean([d["raw_user_command"]["yaw_rate_rps"] for d in diagnostics[lo:hi]])),
                "servo_yaw_mean_rps": float(np.mean([d["servo_command"]["yaw_rate_rps"] for d in diagnostics[lo:hi]])),
                "inner_cap_occupied_intervals": sum(d["inner_yaw_limit_occupied"] for d in diagnostics[lo:hi]),
                "mean_abs_wheel_error_rad_s": float(np.mean(np.abs([d["wheel_error_before_rad_s"] for d in diagnostics[lo:hi]])))})
        peak = max(trace, key=lambda r: abs(r["heading_error_rad"]))
        result.append({"case": folder.name, "original_gate_failures": summary["gates"]["failed"],
            "peak_endpoint_tick": peak["endpoint_tick"], "peak_heading_error_rad": peak["heading_error_rad"],
            "actual_heading_at_peak_rad": peak["heading_task"]["truth_heading_after"],
            "raw_reference_at_peak_rad": peak["heading_task"]["reference_heading_after"],
            "requested_torque_equals_all_applied_substeps": all(np.array_equal(np.asarray(d["applied_torque_nm"]), np.tile(d["requested_torque_nm"], (5,1))) for d in diagnostics),
            "protected_joint_intervals": sum(sum(d["torque_protected"]) for d in diagnostics),
            "peak_abs_actual_wheel_torque_nm": max(abs(d["requested_torque_nm"][i]) for d in diagnostics for i in (3,7,11,15)),
            "windows": windows, "endpoints": records, "one_step_shadow": shadow,
            "one_step_shadow_summary": {"max_abs_target_rad_s": max(abs(v) for r in shadow for v in r["candidate_target_rad_s"]),
                "max_abs_target_increment_rad_s": max(abs(v) for r in shadow for v in r["candidate_minus_baseline_target_rad_s"]),
                "max_abs_wheel_request_nm": max(abs(v) for r in shadow for v in r["candidate_one_step_wheel_request_nm"]),
                "target_speed_clip_count": sum(r["any_target_speed_clipped"] for r in shadow),
                "torque_limit_exceeded_count": sum(r["any_torque_limit_exceeded"] for r in shadow),
                "integral_clip_or_reject_count": sum(r["any_integral_clipped_or_rejected"] for r in shadow)}})
    assert all(sha(path) == expected for path, expected in hashes.items())
    output = {"schema": "d1-plane-turn-offline-kinematic-diagnosis-v1", "new_physics_steps": 0,
        "controller_compute_calls": 0, "all_audit_times_zero": True, "frozen77_match": True,
        "mujoco_version": mujoco.__version__, "input_sha256": hashes, "inputs_unchanged": True,
        "fit": "yaw_fit(v)=-weighted_cov(contact_lateral, horizontal-heading vx)/weighted_var(contact_lateral); contact fits use saved endpoint normal loads, center fit uses equal wheel weight",
        "identity": "contact spin yaw - free-base yaw fit = contact leg yaw fit - slip yaw fit; fractions are kinematic accounting, not causal shares",
        "candidate": "only raw-forward exactly0 and raw-yaw nonzero: omega_target=clip((raw/servo_forward-effective_yaw*lateral+u_center_heading)/.087, -30,30), coefficient1",
        "limitations": ["Native forces and endpoint velocities remain separate; no mixed-phase energy computation",
            "One-step shadow uses each original baseline PI memory independently, never feeds a candidate state/memory forward",
            "Center correction omits wheel-carrier angular velocity, contact-point/axis geometry and full roll/pitch coupling",
            "No candidate trajectory, discrete closed-loop stability or task gate prediction is claimed"], "cases": result}
    with args.output.open("x") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"report_sha256": sha(args.output), "cases": [{k:v for k,v in r.items() if k not in ("endpoints","one_step_shadow")} for r in result]}))


if __name__ == "__main__":
    def forbidden(*args, **kwargs):
        raise AssertionError("offline audit must not integrate physics")
    with patch.object(mujoco, "mj_step", forbidden), patch.object(mujoco, "mj_step1", forbidden), patch.object(mujoco, "mj_step2", forbidden):
        main()
