"""Fixed 209-pose mechanical qualification, with zero integration or contact solve.

Only one candidate mechanism: pulse-only, fixed-b, body-x wheel-center leg
damping on the ORIGINAL plane-zero wheel controller. Center-candidate poses
are an archived-state stress set, not authorization to combine controllers.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
from contextlib import ExitStack
from unittest.mock import patch

import mujoco
import numpy as np

from wheel_legged_control.d1.model import (
    D1_JOINT_NAMES, LEG_PREFIXES, JOINT_TORQUE_LIMIT, JOINT_VELOCITY_LIMIT,
    JOINT_POSITION_LOW, JOINT_POSITION_HIGH, build_d1_model,
)

B = 126.4374005337902
TICKS = [199, *range(200, 250), 250]
RADIUS = .087


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    with gzip.open(path, "rt") as stream:
        return [json.loads(line) for line in stream]


def safe_torque(request, q, qd):
    safe = np.clip(request, -JOINT_TORQUE_LIMIT, JOINT_TORQUE_LIMIT)
    outward = ((q >= JOINT_POSITION_HIGH) & (safe > 0)) | ((q <= JOINT_POSITION_LOW) & (safe < 0))
    outward |= (np.abs(qd) >= JOINT_VELOCITY_LIMIT) & (safe*qd > 0)
    safe[outward] = 0.
    return safe


def generalized_decay(mass, damping):
    lower = np.linalg.cholesky(mass)
    transformed = np.linalg.solve(lower, np.linalg.solve(lower, damping).T).T
    eigenvalues = np.linalg.eigvalsh(.5*(transformed+transformed.T))
    assert eigenvalues.min() >= -1e-8
    return float(max(0., eigenvalues[-1])*.01)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("/home/lyh/wheel-legged-control-lab"))
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    work, repo = args.work.resolve(), args.repo.resolve()
    hashes = {}

    def remember(path, expected=None):
        path = Path(path).resolve()
        digest = sha(path)
        assert expected is None or digest == expected, path
        hashes[str(path)] = digest

    remember(__file__)
    remember(Path(__file__).with_name("fixed_method_contract.md"))
    for name in ("report.json", "qualify_body_rate.py", "body_rate_qualification.json"):
        remember(work/"turn_center_failure_plan_01"/name)
    diagnosis = json.loads((work/"turn_center_failure_plan_01/report.json").read_text())
    for path, expected in diagnosis["input_sha256"].items():
        remember(path, expected)
    frozen = json.loads((repo/"results/d1_budget_study/protocol.json").read_text())["source_sha256"]
    assert len(frozen) == 77
    for path, expected in frozen.items():
        remember(repo/path, expected)
    model, calls = build_d1_model(), {}
    assert model.nhfield == 0 and model.nq == 23 and model.nv == 22
    data = mujoco.MjData(model)
    base = model.body("base_link").id
    bodies = [model.body(p+"_foot").id for p in LEG_PREFIXES]
    joints = np.array([model.joint(n).id for n in D1_JOINT_NAMES])
    dofs = model.jnt_dofadr[joints].reshape(4, 4)
    qadr = model.jnt_qposadr[joints]
    leg_dofs, wheel_dofs = dofs[:, :3].ravel(), dofs[:, 3]
    leg_local = np.array([i for i in range(16) if i % 4 != 3])
    wheel_local = np.array([3, 7, 11, 15])
    jp, jr, mass = np.zeros((3, model.nv)), np.zeros((3, model.nv)), np.zeros((model.nv, model.nv))
    pose_count = 0

    def invoke(name, *a):
        calls[name] = calls.get(name, 0)+1
        return getattr(mujoco, name)(*a)

    def geometry(qpos, qvel):
        nonlocal pose_count
        pose_count += 1
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        warmstart = data.qacc_warmstart.copy()
        invoke("mj_kinematics", model, data)
        invoke("mj_comPos", model, data)
        invoke("mj_comVel", model, data)
        invoke("mj_crb", model, data)
        invoke("mj_fullM", model, data, mass)
        assert data.time == 0. and data.qpos.tobytes() == qpos.tobytes() and data.qvel.tobytes() == qvel.tobytes()
        assert data.qacc_warmstart.tobytes() == warmstart.tobytes()
        rotation = data.xmat[base].reshape(3, 3).copy()
        yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        horizontal = np.array([np.cos(yaw), np.sin(yaw), 0.])
        lateral_axis = np.array([-np.sin(yaw), np.cos(yaw), 0.])
        jacobians, jx = [], np.zeros((4, model.nv))
        for leg, body in enumerate(bodies):
            invoke("mj_jacBody", model, data, jp, jr, body)
            jacobians.append(jp.copy())
            jx[leg, dofs[leg, :3]] = rotation[:, 0] @ jp[:, dofs[leg, :3]]
        lateral = (data.xpos[bodies]-data.xpos[base]) @ lateral_axis
        return rotation, horizontal, lateral, jacobians, jx

    def fixed_center_yaw_mode(rotation, jacobians):
        mode = np.zeros(model.nv)
        mode[3:6] = rotation.T @ np.array([0., 0., 1.])
        for leg, jac in enumerate(jacobians):
            mode[dofs[leg, :3]] = np.linalg.solve(jac[:, dofs[leg, :3]], -jac @ mode)
        residual = max(np.linalg.norm(j @ mode) for j in jacobians)
        assert residual < 1e-10
        return mode

    initial_path = work/"flat_plane_02/stationary_turn_left_hold/states.npz"
    with np.load(initial_path, allow_pickle=False) as z:
        rotation, horizontal, lateral, jacs, jx = geometry(z["qpos"][0], z["qvel"][0])
    nominal_mode = fixed_center_yaw_mode(rotation, jacs)
    longitudinal = np.zeros(model.nv)
    longitudinal[0] = 1.
    for leg, jac in enumerate(jacs):
        longitudinal[dofs[leg, :3]] = np.linalg.solve(jac[:, dofs[leg, :3]], -jac @ longitudinal)
    mx = float(longitudinal @ mass @ longitudinal)
    kx = float(80.*(longitudinal[leg_dofs] @ longitudinal[leg_dofs]))
    bx = float(3.*(longitudinal[leg_dofs] @ longitudinal[leg_dofs])+longitudinal @ (model.dof_damping*longitudinal))
    assert abs((2.*np.sqrt(mx*kx)-bx)/4.-B) < 1e-10
    mtheta = float(nominal_mode @ mass @ nominal_mode)
    ktheta = float(80.*(nominal_mode[leg_dofs] @ nominal_mode[leg_dofs]))
    btheta = float(3.*(nominal_mode[leg_dofs] @ nominal_mode[leg_dofs])+nominal_mode @ (model.dof_damping*nominal_mode))
    extra_theta = float(B*np.sum((jx @ nominal_mode)**2))
    nominal = {"translation_gain_recomputed": float((2.*np.sqrt(mx*kx)-bx)/4.),
        "M_theta_kg_m2": mtheta, "K_theta_nm_per_rad": ktheta,
        "B_theta_original_nms_per_rad": btheta, "B_theta_added_nms_per_rad": extra_theta,
        "single_mode_zeta_before": btheta/(2.*np.sqrt(mtheta*ktheta)),
        "single_mode_zeta_after": (btheta+extra_theta)/(2.*np.sqrt(mtheta*ktheta)),
        "unit_world_yaw_centers_fixed_mode": nominal_mode.tolist(),
        "meaning": "constrained-center kinematic mode projected onto free mass and joint gains; not an identified yaw closed-loop mode"}
    episodes = []
    for source, directory in (("baseline", "flat_plane_02"), ("center_state_stress_only", "plane_turn_center_01")):
        for side, sign in (("left", 1), ("right", -1)):
            folder = work/directory/f"stationary_turn_{side}_hold"
            manifest = json.loads((folder/"complete_manifest.json").read_text())
            remember(folder/"complete_manifest.json")
            for name, item in manifest.items():
                assert (folder/name).stat().st_size == item["bytes"]
                remember(folder/name, item["sha256"])
            with np.load(folder/"states.npz", allow_pickle=False) as z:
                qpos, qvel = z["qpos"].copy(), z["qvel"].copy()
            diagnostics = read_rows(folder/"plane_diagnostics.jsonl.gz")
            records = []
            for tick in TICKS:
                d, endpoint = diagnostics[tick], diagnostics[tick-1]["endpoint_contacts"]
                rotation, horizontal, lateral, jacs, jx = geometry(qpos[tick], qvel[tick])
                u = jx @ qvel[tick]
                np.testing.assert_allclose(u, d["leg_forward_velocity_before_mps"], rtol=0, atol=2e-12)
                centered = lateral-lateral.mean()
                common = np.full(4, u.mean())
                differential = centered*float(centered @ u/(centered @ centered))
                remainder = u-common-differential
                assert abs(u @ u-common @ common-differential @ differential-remainder @ remainder) < 1e-12
                damping = B*jx.T @ jx
                potential_delta = -damping @ qvel[tick]
                raw = d["raw_user_command"]
                active = raw["forward_velocity_mps"] == 0. and raw["yaw_rate_rps"] != 0.
                assert active == (200 <= tick < 250)
                delta = potential_delta[dofs.ravel()] if active else np.zeros(16)
                assert np.array_equal(delta[wheel_local], np.zeros(4))
                assert np.array_equal(damping[:, wheel_dofs], np.zeros((model.nv, 4)))
                instant_power = float(delta @ qvel[tick, dofs.ravel()])
                assert abs(instant_power-(-B*float(u @ u) if active else 0.)) < 1e-10
                before_request = np.array(d["unlimited_torque_nm"])
                original_safe = safe_torque(before_request, qpos[tick, qadr], qvel[tick, dofs.ravel()])
                np.testing.assert_array_equal(original_safe, d["requested_torque_nm"])
                request = before_request+delta
                safe = safe_torque(request, qpos[tick, qadr], qvel[tick, dofs.ravel()])
                safe_power = float((safe-original_safe) @ qvel[tick, dofs.ravel()])
                old_diagonal = model.dof_damping.copy()
                old_diagonal[leg_dofs] += 3.
                old_diagonal[wheel_dofs] += 2.2+3.*.01
                yaw_mode = fixed_center_yaw_mode(rotation, jacs)
                qtheta = float(yaw_mode @ potential_delta)
                base_yaw_velocity = np.zeros(model.nv)
                base_yaw_velocity[3:6] = rotation.T @ np.array([0., 0., sign*.6])
                original_target = -sign*.6*lateral/RADIUS
                rigid_velocity = base_yaw_velocity.copy()
                rigid_velocity[wheel_dofs] = original_target
                contact_summaries = []
                for leg, body in enumerate(bodies):
                    contacts = [c for c in endpoint["contacts"] if c["wheel_index"] == leg and c["normal_force_n"] > 0]
                    assert contacts, "fixed selected sample lacks loaded wheel; do not invent zero-slip support"
                    weights = np.array([c["normal_force_n"] for c in contacts]); weights /= weights.sum()
                    point = weights @ np.array([c["pos_world_m"] for c in contacts])
                    invoke("mj_jac", model, data, jp, jr, point, body)
                    representative_jac = jp.copy()
                    # Unique 4x4 kinematic solve, NOT a constraint-force solve.
                    # Three contact-velocity equations plus center-leg-x=0.
                    local = dofs[leg]
                    a = np.vstack([jp[:, local], jx[leg, local]])
                    rhs = np.r_[-jp @ base_yaw_velocity, 0.]
                    singular = np.linalg.svd(a, compute_uv=False)
                    solution = np.linalg.solve(a, rhs)
                    local_velocity = base_yaw_velocity.copy()
                    local_velocity[local] = solution
                    representative_residual = np.linalg.norm(representative_jac @ local_velocity)
                    assert representative_residual < 1e-10 and abs(jx[leg] @ local_velocity) < 1e-10
                    axle = data.xaxis[joints[4*leg+3]].copy(); axle[2] = 0.
                    assert np.linalg.norm(axle) > 1e-6
                    axle /= np.linalg.norm(axle)
                    fixed_side, movable_residual, velocity_closure = [], [], []
                    for contact in contacts:
                        np.testing.assert_array_equal(contact["normal_world"], [0., 0., 1.])
                        invoke("mj_jac", model, data, jp, jr, np.array(contact["pos_world_m"]), body)
                        velocity_closure.append(float(np.max(np.abs(jp @ qvel[tick]-contact["relative_velocity_world_mps"]))))
                        fixed_side.append(float(axle @ (jp @ rigid_velocity)))
                        movable_residual.append(float(np.linalg.norm(jp @ local_velocity)))
                    assert max(velocity_closure) < 2e-12
                    contact_summaries.append({"wheel": leg, "loaded_contact_count": len(contacts),
                        "frozen_leg_axle_side_rms_mps": float(np.sqrt(weights @ np.square(fixed_side))),
                        "representative_singular_values_m_per_rad": singular.tolist(),
                        "representative_condition_number": float(singular[0]/singular[-1]),
                        "representative_required_leg_rad_s": solution[:3].tolist(),
                        "representative_required_wheel_rad_s": float(solution[3]),
                        "required_wheel_minus_original_target_rad_s": float(solution[3]-original_target[leg]),
                        "representative_contact_residual_mps": float(representative_residual),
                        "all_loaded_points_velocity_residual_rms_mps": float(np.sqrt(weights @ np.square(movable_residual))),
                        "max_representative_actuator_speed_ratio": float(np.max(np.abs(solution)/JOINT_VELOCITY_LIMIT[4*leg:4*leg+4])),
                        "saved_contact_velocity_closure_max_mps": max(velocity_closure)})
                records.append({"state_tick": tick, "active": active, "u_body_x_mps": u.tolist(),
                    "u_squared_sum": float(u @ u), "u_common_squared_sum": float(common @ common),
                    "u_yaw_differential_squared_sum": float(differential @ differential),
                    "u_remaining_squared_sum": float(remainder @ remainder),
                    "delta_joint_torque_nm": delta.tolist(), "unclipped_delta_power_w": instant_power,
                    "safe_actual_increment_power_w": safe_power,
                    "request_plus_delta_limit_ratio": float(np.max(np.abs(request)/JOINT_TORQUE_LIMIT)),
                    "candidate_protected_joint_count": int(np.count_nonzero(safe != request)),
                    "new_protected_joint_count": int(np.count_nonzero((safe != request) & (original_safe == before_request))),
                    "actual_joint_speed_ratio": float(np.max(np.abs(qvel[tick, dofs.ravel()])/JOINT_VELOCITY_LIMIT)),
                    "minimum_leg_position_limit_margin_rad": float(np.min(np.minimum(qpos[tick, qadr][leg_local]-JOINT_POSITION_LOW[leg_local], JOINT_POSITION_HIGH[leg_local]-qpos[tick, qadr][leg_local]))),
                    "free_inertia_new_damping_dt_lambda": generalized_decay(mass, damping),
                    "free_inertia_simple_old_plus_new_dt_lambda": generalized_decay(mass, damping+np.diag(old_diagonal)),
                    "yaw_mode_virtual_force_nm": qtheta,
                    "raw_sign_times_yaw_mode_virtual_force_nm": sign*qtheta,
                    "geometry": contact_summaries})
            active_records = [r for r in records if r["active"]]
            energy = sum(r["u_squared_sum"] for r in active_records)
            geometry_rows = [g for r in active_records for g in r["geometry"]]
            summary = {"source": source, "side": side, "poses": len(records), "active_pose_samples": len(active_records),
                "yaw_differential_u_squared_share": sum(r["u_yaw_differential_squared_sum"] for r in active_records)/energy,
                "common_u_squared_share": sum(r["u_common_squared_sum"] for r in active_records)/energy,
                "remaining_u_squared_share": sum(r["u_remaining_squared_sum"] for r in active_records)/energy,
                "sum_sample_dt_times_unclipped_delta_power_j_not_held_input_energy": .01*sum(r["unclipped_delta_power_w"] for r in active_records),
                "maximum_unclipped_delta_power_w": max(r["unclipped_delta_power_w"] for r in active_records),
                "maximum_safe_increment_power_w": max(r["safe_actual_increment_power_w"] for r in active_records),
                "peak_abs_delta_nm": max(abs(v) for r in active_records for v in r["delta_joint_torque_nm"]),
                "maximum_request_limit_ratio": max(r["request_plus_delta_limit_ratio"] for r in active_records),
                "new_protected_joint_samples": sum(r["new_protected_joint_count"] for r in active_records),
                "candidate_protected_joint_samples": sum(r["candidate_protected_joint_count"] for r in active_records),
                "maximum_actual_joint_speed_ratio": max(r["actual_joint_speed_ratio"] for r in active_records),
                "minimum_leg_position_limit_margin_rad": min(r["minimum_leg_position_limit_margin_rad"] for r in active_records),
                "maximum_free_inertia_new_dt_lambda": max(r["free_inertia_new_damping_dt_lambda"] for r in active_records),
                "maximum_free_inertia_simple_old_plus_new_dt_lambda": max(r["free_inertia_simple_old_plus_new_dt_lambda"] for r in active_records),
                "mean_raw_signed_virtual_yaw_force_nm": float(np.mean([r["raw_sign_times_yaw_mode_virtual_force_nm"] for r in active_records])),
                "opposing_virtual_yaw_force_samples": sum(r["raw_sign_times_yaw_mode_virtual_force_nm"] < 0 for r in active_records),
                "frozen_legs_side_speed_rms_over_wheels_samples_mps": float(np.sqrt(np.mean([g["frozen_leg_axle_side_rms_mps"]**2 for g in geometry_rows]))),
                "maximum_representative_condition_number": max(g["representative_condition_number"] for g in geometry_rows),
                "maximum_representative_required_speed_ratio": max(g["max_representative_actuator_speed_ratio"] for g in geometry_rows),
                "maximum_representative_wheel_target_difference_rad_s": max(abs(g["required_wheel_minus_original_target_rad_s"]) for g in geometry_rows),
                "all_loaded_points_residual_rms_over_wheels_samples_mps": float(np.sqrt(np.mean([g["all_loaded_points_velocity_residual_rms_mps"]**2 for g in geometry_rows]))),
                "maximum_all_loaded_points_residual_mps": max(g["all_loaded_points_velocity_residual_rms_mps"] for g in geometry_rows)}
            source_case = next(c for c in diagnosis["cases"] if c["source"] == ("baseline" if source == "baseline" else "center_candidate") and c["turn_sign"] == sign)
            summary["archived_native_force_windows"] = [{"execution_ticks_half_open": window["execution_ticks_half_open"],
                "world_x_world_y_contact_torque_net_yaw_moment_nm": window["native_time_weighted_yaw_moment_world_x_world_y_contact_torque_net_nm"],
                "normal_load_n": window["native_time_weighted_normal_load_n"],
                "tangent_over_mu_load": window["native_sum_tangent_magnitude_over_mu_normal_load"]}
                for window in source_case["windows"] if window["execution_ticks_half_open"] in ([200, 210], [210, 225], [225, 250], [200, 250])]
            episodes.append({"summary": summary, "samples": records})
    assert pose_count == 209 and all(sha(path) == expected for path, expected in hashes.items())
    report = {"schema": "d1-turn-pulse-longitudinal-leg-damper-mechanical-qualification-v1",
        "algebra_checks_passed": True, "new_physics_steps": 0, "new_controller_compute_calls": 0,
        "new_contact_force_solves": 0, "saved_pose_samples": 208, "nominal_pose_samples": 1,
        "fixed_sample_ticks": TICKS, "all_scratch_times_zero": True, "frozen77_unchanged": True,
        "input_sha256": hashes, "inputs_unchanged": True, "mujoco_version": mujoco.__version__,
        "fixed_b_nspm": B, "nominal_mode": nominal,
        "candidate": "original plane-zero wheel controller + raw-pure-turn-pulse-only -b Jbodyx.T Jbodyx qdot_leg; no center compensation and no stop latch",
        "wheel_spin_damping": {"damper_wheel_rows_and_columns_zero": True,
            "original_direct_wheel_PI_error_derivative": -1., "original_local_instantaneous_PI_torque_derivative_nm_per_rad_s": -(2.2+3.*.01)},
        "actual_nonintegrating_api_calls": calls,
        "limitations": ["No proposed-controller trajectory, closed-loop stability certificate or task pass prediction.",
            "Free-inertia spectral values do not include loaded-contact dynamics or the full controller. They report sampling risk rather than automatically rejecting a bounded probe.",
            "Sampled incremental power is before integration and is not actual held-input work. Safety-filtered increments are reported separately.",
            "Yaw-mode virtual force is a coordinate projection of leg torques, not a measured external yaw moment.",
            "Representative contact-point kinematics do not prove every finite contact point can remain no-slip. All loaded-point residuals and required wheel-target differences are retained.",
            "Archived native contact forces are read from the hash-checked prior diagnosis; no native force is multiplied by endpoint velocity.",
            "Center-candidate state shadows are an envelope stress test and do not authorize combining mechanisms."],
        "episodes": episodes}
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False); stream.write("\n")
    print(json.dumps({"report": str(args.output), "sha256": sha(args.output), "new_physics_steps": 0,
        "pose_count": pose_count, "nominal": nominal, "episodes": [e["summary"] for e in episodes]}, allow_nan=False))


if __name__ == "__main__":
    def forbidden(*_args, **_kwargs):
        raise AssertionError("zero-physics qualification forbids integration or fresh dynamics/contact solve")
    with ExitStack() as stack:
        for api in ("mj_step", "mj_step1", "mj_step2", "mj_forward", "mj_inverse", "mj_collision"):
            stack.enter_context(patch.object(mujoco, api, forbidden))
        main()
