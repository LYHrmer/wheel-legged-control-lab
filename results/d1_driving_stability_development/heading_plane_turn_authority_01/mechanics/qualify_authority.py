"""208 saved-state qualifications of the original uncapped inner yaw formula.

No rollout, controller compute, gain/cap search, state propagation or contact
force solve. PI headroom is diagnostic only; it is not another controller.
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

TICKS = [199, *range(200, 250), 250]
WHEELS = np.array([3, 7, 11, 15])
R, KP, KI, DT = .087, 2.2, 3., .01


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    with gzip.open(path, "rt") as stream:
        return [json.loads(line) for line in stream]


def wheel_pi(target, omega, memory):
    error = target-omega
    trial_unclipped = memory+KI*DT*error
    trial = np.clip(trial_unclipped, -4., 4.)
    trial_request = KP*error+trial
    accepted = (np.abs(trial_request) <= 12.) | (trial_request*error < 0.)
    after = np.where(accepted, trial, memory)
    request = KP*error+after
    damping = KP+KI*DT*(accepted & (np.abs(trial_unclipped) < 4.))
    return {"error": error, "after": after, "request": request,
        "integral_rejected": ~accepted, "integral_clipped": trial != trial_unclipped,
        "unprotected_direct_spin_damping": damping}


def safety(request, q, qd):
    safe = np.clip(request, -JOINT_TORQUE_LIMIT, JOINT_TORQUE_LIMIT)
    outward = ((q >= JOINT_POSITION_HIGH) & (safe > 0)) | ((q <= JOINT_POSITION_LOW) & (safe < 0))
    outward |= (np.abs(qd) >= JOINT_VELOCITY_LIMIT) & (safe*qd > 0)
    safe[outward] = 0.
    return safe, outward


def headroom(forward, lateral, omega, memory):
    """Exact request<=12 interval for the frozen antiwindup, on these samples.

    Since |I|<=4<12, error has the same sign as any over-limit torque. Any
    integral update that would cause |tau|>12 is rejected. Thus the final
    pre-safety request is within 12 iff |Kp*error+I_before|<=12.
    The saved sample targets below are strictly inside +/-30; therefore the
    wheel-target clip creates no additional remote feasible rays here.
    """
    assert np.max(np.abs(memory)) <= 4.
    low = omega+(-12.-memory)/KP
    high = omega+(12.-memory)/KP
    assert np.all(low > -30.) and np.all(high < 30.)
    assert np.min(np.abs(lateral)) > 1e-4
    pairs = np.stack([(forward-R*low)/lateral, (forward-R*high)/lateral], axis=1)
    interval = [float(np.max(np.min(pairs, axis=1))), float(np.min(np.max(pairs, axis=1)))]
    assert interval[0] <= interval[1]
    for value in interval:
        target = (forward-value*lateral)/R
        result = wheel_pi(target, omega, memory)
        assert abs(float(np.max(np.abs(result["request"])))-12.) < 2e-10
    return interval


def native_windows(native):
    result = []
    for low, high in ((200, 210), (210, 225), (225, 250), (200, 250)):
        selected = native[5*low:5*high]
        dt = np.array([e["actual_dt_s"] for e in selected])
        moments, normals, ratios = [], [], []
        for entry in selected:
            assert entry["returned"] and abs(entry["actual_dt_s"]-.002) < 1e-12
            c = entry["contacts"]
            assert c["sampling"] == "native_step_solved_cache_not_synchronized_endpoint"
            assert c["max_horizontal_normal"] <= 1e-12 and c["max_vertical_normal_error"] <= 1e-12
            xmoment = ymoment = contact_torque = tangent = 0.
            for contact in c["contacts"]:
                point = np.array(contact["pos_world_m"])-c["reference_world_m"]
                force = np.array(contact["force_world_n"])
                xmoment -= point[1]*force[0]
                ymoment += point[0]*force[1]
                contact_torque += contact["torque_world_nm"][2]
                tangent += np.linalg.norm(contact["tangent_force_world_n"])
            assert abs(xmoment+ymoment+contact_torque-c["total_wrench_world_6"][5]) < 1e-10
            normal = c["summed_normal_force_world_n"][2]
            assert normal > 0.
            normals.append(normal); ratios.append(tangent)
            moments.append([xmoment, ymoment, contact_torque, c["total_wrench_world_6"][5]])
        result.append({"executed_ticks_half_open": [low, high], "native_substeps": len(selected),
            "actual_world_x_world_y_contact_torque_net_yaw_moment_nm": np.average(moments, axis=0, weights=dt).tolist(),
            "actual_normal_fz_n": float(np.average(normals, weights=dt)),
            "sum_tangent_over_mu_normal": float(dt @ np.array(ratios)/(.9*(dt @ np.array(normals))))})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("/home/lyh/wheel-legged-control-lab"))
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    repo, work, hashes = args.repo.resolve(), args.work.resolve(), {}

    def remember(path, expected=None):
        path = Path(path).resolve(); digest = sha(path)
        assert expected is None or digest == expected, path
        hashes[str(path)] = digest

    remember(__file__)
    remember(Path(__file__).with_name("fixed_contract.md"))
    remember(Path(__file__).with_name("sampling_note.md"))
    frozen_path = repo/"results/d1_budget_study/protocol.json"
    remember(frozen_path)
    frozen = json.loads(frozen_path.read_text())["source_sha256"]
    assert len(frozen) == 77
    for path, expected in frozen.items():
        remember(repo/path, expected)
    g1_path = repo/"results/d1_driving_stability_development/heading_g1_01/evaluation_protocol.json"
    remember(g1_path, "cf5dbd51042bfafa163116f4a7fb2a990726d47c5a6000558f28aa6866a664fc")
    for name in ("d1_heading_tracking_env.py", "d1_heading_reference.py", "d1_flat_plane_env.py", "d1_turn_leg_damping.py"):
        remember(repo/"scripts"/name)
    model = build_d1_model(); data = mujoco.MjData(model)
    assert model.nhfield == 0 and (model.nq, model.nv, model.nu) == (23, 22, 16)
    base = model.body("base_link").id
    bodies = [model.body(p+"_foot").id for p in LEG_PREFIXES]
    joint_ids = [model.joint(n).id for n in D1_JOINT_NAMES]
    dofs, qadr = model.jnt_dofadr[joint_ids], model.jnt_qposadr[joint_ids]
    velocity, pose_count, cases = np.zeros(6), 0, []
    mass, jp = np.zeros((model.nv, model.nv)), np.zeros((3, model.nv))
    for source, directory in (("baseline", "flat_plane_02"), ("damper_state_stress_only", "plane_turn_damping_01")):
        for side, sign in (("left", 1), ("right", -1)):
            folder = work/directory/f"stationary_turn_{side}_hold"
            remember(folder/"complete_manifest.json")
            manifest = json.loads((folder/"complete_manifest.json").read_text())
            for name, item in manifest.items():
                assert (folder/name).stat().st_size == item["bytes"]
                remember(folder/name, item["sha256"])
            diagnostic, trace, native = [rows(folder/name) for name in
                ("plane_diagnostics.jsonl.gz", "trace.jsonl.gz", "native_physics_entries.jsonl.gz")]
            damping = rows(folder/"turn_damping.jsonl.gz") if source != "baseline" else None
            summary = json.loads((folder/"summary.json").read_text())
            with np.load(folder/"states.npz", allow_pickle=False) as z:
                qpos, qvel = z["qpos"].copy(), z["qvel"].copy()
            assert len(trace) == len(diagnostic) == 800 and len(qpos) == len(qvel) == 801 and len(native) == 4000
            samples = []
            for tick in TICKS:
                pose_count += 1
                data.qpos[:] = qpos[tick]; data.qvel[:] = qvel[tick]
                warmstart = data.qacc_warmstart.copy()
                mujoco.mj_kinematics(model, data); mujoco.mj_comPos(model, data); mujoco.mj_comVel(model, data)
                assert data.time == 0. and data.qpos.tobytes() == qpos[tick].tobytes() and data.qvel.tobytes() == qvel[tick].tobytes()
                assert data.qacc_warmstart.tobytes() == warmstart.tobytes()
                rotation = data.xmat[base].reshape(3, 3)
                yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
                lateral = (data.xpos[bodies]-data.xpos[base]) @ np.array([-np.sin(yaw), np.cos(yaw), 0.])
                mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, base, velocity, 0)
                body_rate = float(rotation[:, 2] @ velocity[:3])
                d = diagnostic[tick]; raw = d["raw_user_command"]; servo = d["servo_command"]["yaw_rate_rps"]
                assert abs(body_rate-d["body_yaw_rate_before_rps"]) < 2e-12
                assert raw["forward_velocity_mps"] == d["servo_command"]["forward_velocity_mps"] == 0.
                active = raw["yaw_rate_rps"] != 0.
                assert active == (200 <= tick < 250)
                if active:
                    assert raw["yaw_rate_rps"] == sign*.6
                original_unclipped = servo+4.*(servo-body_rate)
                original_effective = float(np.clip(original_unclipped, -.6, .6))
                assert abs(original_effective-d["effective_yaw_request_rps"]) < 1e-12
                old_target = np.clip(-original_effective*lateral/R, -30., 30.)
                np.testing.assert_allclose(old_target, d["wheel_target_rad_s"], rtol=0, atol=2e-12)
                restored = original_unclipped if active else original_effective
                unclipped_target = -restored*lateral/R
                target = np.clip(unclipped_target, -30., 30.)
                memory, omega = np.array(d["wheel_integral_before_nm"]), qvel[tick, dofs[WHEELS]]
                old_pi, candidate_pi = wheel_pi(old_target, omega, memory), wheel_pi(target, omega, memory)
                np.testing.assert_allclose(old_pi["after"], d["wheel_integral_after_nm"], rtol=0, atol=2e-12)
                np.testing.assert_allclose(old_pi["request"], np.array(d["unlimited_torque_nm"])[WHEELS], rtol=0, atol=2e-12)
                # Remove the archived damper from the stress sample. This
                # candidate changes only wheel authority on the original law.
                original_request = np.array(d["unlimited_torque_nm"] if damping is None else damping[tick]["record"]["base_requested_torque_nm"])
                new_request = original_request.copy(); new_request[WHEELS] = candidate_pi["request"]
                safe, outward = safety(new_request, qpos[tick, qadr], qvel[tick, dofs])
                assert np.isfinite(safe).all() and np.all(np.abs(safe) <= JOINT_TORQUE_LIMIT)
                np.testing.assert_array_equal(new_request[np.setdiff1d(np.arange(16), WHEELS)], original_request[np.setdiff1d(np.arange(16), WHEELS)])
                interval = headroom(0., lateral, omega, memory)
                assert interval[0]-1e-10 <= original_effective <= interval[1]+1e-10
                masked = abs(original_unclipped) > .6
                outer_saturated = trace[tick]["heading_task"]["servo_yaw_rate_saturated_before"]
                derivative = -4. if outer_saturated else -6.
                # Scale diagnostic only: this rigid-leg geometric-wheel mode
                # violates lateral contact constraints, quantified below.
                mujoco.mj_crb(model, data); mujoco.mj_fullM(model, data, mass)
                mode = np.zeros(model.nv)
                mode[3:6] = rotation.T @ np.array([0., 0., 1.])
                mode[dofs[WHEELS]] = -lateral/R
                modal_mass = float(mode @ mass @ mode)
                assert modal_mass > 0.
                body_rate_per_mode = float(rotation[2, 2])
                old_target_derivative = -lateral/R*(0. if masked else derivative)*body_rate_per_mode
                restored_derivative = derivative if active or not masked else 0.
                new_target_derivative = -lateral/R*restored_derivative*body_rate_per_mode
                new_target_derivative[target != unclipped_target] = 0.
                omega_mode = mode[dofs[WHEELS]]
                old_error_derivative = old_target_derivative-omega_mode
                new_error_derivative = new_target_derivative-omega_mode
                passive = float(mode @ (model.dof_damping*mode))
                old_modal_damping = float(-omega_mode @ (old_pi["unprotected_direct_spin_damping"]*old_error_derivative)+passive)
                new_modal_damping = float(-omega_mode @ (candidate_pi["unprotected_direct_spin_damping"]*new_error_derivative)+passive)
                safe_slopes = candidate_pi["unprotected_direct_spin_damping"]
                safe_slopes = np.where((np.abs(candidate_pi["request"]) < 12.) & ~outward[WHEELS], safe_slopes, 0.)
                safe_modal_damping = float(-omega_mode @ (safe_slopes*new_error_derivative)+passive)
                endpoint = diagnostic[tick-1]["endpoint_contacts"]
                residual_squared, lateral_squared, loads = [], [], []
                for contact in endpoint["contacts"]:
                    load = contact["normal_force_n"]
                    if load <= 0.:
                        continue
                    wheel = contact["wheel_index"]
                    mujoco.mj_jac(model, data, jp, None, np.array(contact["pos_world_m"]), bodies[wheel])
                    point_velocity = jp @ mode
                    axle = data.xaxis[joint_ids[4*wheel+3]].copy(); axle[2] = 0.
                    axle /= np.linalg.norm(axle)
                    residual_squared.append(float(point_velocity @ point_velocity))
                    lateral_squared.append(float(axle @ point_velocity)**2)
                    loads.append(load)
                assert sum(loads) > 0.
                sampling = {"assumption": "unit world-yaw, rigid legs, original geometric wheel-speed map; not a loaded contact constraint mode",
                    "modal_mass_kg_m2": modal_mass, "original_local_B_nms": old_modal_damping,
                    "restored_preprotection_local_B_nms": new_modal_damping,
                    "restored_after_rated_protection_local_B_nms": safe_modal_damping,
                    "original_dt_B_over_M": DT*old_modal_damping/modal_mass,
                    "restored_preprotection_dt_B_over_M": DT*new_modal_damping/modal_mass,
                    "restored_after_protection_dt_B_over_M": DT*safe_modal_damping/modal_mass,
                    "unit_mode_loaded_contact_velocity_rms_m_per_rad": float(np.sqrt(np.average(residual_squared, weights=loads))),
                    "unit_mode_loaded_contact_side_velocity_rms_m_per_rad": float(np.sqrt(np.average(lateral_squared, weights=loads)))}
                samples.append({"execution_tick": tick, "active": active,
                    "raw_yaw_rps": raw["yaw_rate_rps"], "servo_yaw_rps": servo, "body_yaw_rps": body_rate,
                    "original_unclipped_inner_rps": original_unclipped, "original_capped_inner_rps": original_effective,
                    "restored_inner_rps": restored, "original_inner_feedback_masked": bool(masked),
                    "original_capped_body_rate_derivative_including_outer": 0. if masked else derivative,
                    "restored_body_rate_derivative_including_outer": derivative if active or not masked else 0.,
                    "exact_all_wheel_PI_headroom_yaw_interval_rps_diagnostic_only": interval,
                    "restored_outside_unsaturated_authority_interval": bool(restored < interval[0] or restored > interval[1]),
                    "lateral_m": lateral.tolist(), "wheel_velocity_rad_s": omega.tolist(), "I_before_nm": memory.tolist(),
                    "old_target_rad_s": old_target.tolist(), "restored_unclipped_target_rad_s": unclipped_target.tolist(),
                    "restored_target_rad_s": target.tolist(), "target_clipped": (target != unclipped_target).tolist(),
                    "one_sample_I_after_nm": candidate_pi["after"].tolist(),
                    "one_sample_wheel_unlimited_request_nm": candidate_pi["request"].tolist(),
                    "one_sample_wheel_actual_safe_nm": safe[WHEELS].tolist(),
                    "integral_rejected": candidate_pi["integral_rejected"].tolist(),
                    "integral_clipped": candidate_pi["integral_clipped"].tolist(),
                    "rated_wheel_torque_saturated": (np.abs(candidate_pi["request"]) > 12.).tolist(),
                    "outward_protected_axes": outward.tolist(),
                    "unprotected_direct_wheel_spin_damping_nm_per_rad_s": candidate_pi["unprotected_direct_spin_damping"].tolist(),
                    "actual_saved_joint_speed_ratio": float(np.max(np.abs(qvel[tick, dofs])/JOINT_VELOCITY_LIMIT)),
                    "sampling_scale_diagnostic": sampling})
            pulse = [s for s in samples if s["active"]]
            cases.append({"source": source, "side": side, "poses": len(samples),
                "original_heading_peak_rad": summary["heading_peak_rad"], "original_failed_gates": summary["gates"]["failed"],
                "active_intervals": len(pulse), "original_feedback_masked_intervals": sum(s["original_inner_feedback_masked"] for s in pulse),
                "min_abs_original_unclipped_inner_rps": min(abs(s["original_unclipped_inner_rps"]) for s in pulse),
                "max_abs_original_unclipped_inner_rps": max(abs(s["original_unclipped_inner_rps"]) for s in pulse),
                "max_abs_restored_target_rad_s": max(abs(v) for s in pulse for v in s["restored_target_rad_s"]),
                "max_abs_restored_unlimited_wheel_request_nm": max(abs(v) for s in pulse for v in s["one_sample_wheel_unlimited_request_nm"]),
                "max_abs_restored_actual_wheel_torque_nm": max(abs(v) for s in pulse for v in s["one_sample_wheel_actual_safe_nm"]),
                "rated_saturated_wheel_samples": sum(sum(s["rated_wheel_torque_saturated"]) for s in pulse),
                "target_clipped_wheel_samples": sum(sum(s["target_clipped"]) for s in pulse),
                "integral_rejected_wheel_samples": sum(sum(s["integral_rejected"]) for s in pulse),
                "integral_clipped_wheel_samples": sum(sum(s["integral_clipped"]) for s in pulse),
                "outward_protected_joint_samples": sum(sum(s["outward_protected_axes"]) for s in pulse),
                "maximum_actual_saved_joint_speed_ratio": max(s["actual_saved_joint_speed_ratio"] for s in pulse),
                "max_original_proxy_dt_B_over_M": max(s["sampling_scale_diagnostic"]["original_dt_B_over_M"] for s in pulse),
                "max_restored_preprotection_proxy_dt_B_over_M": max(s["sampling_scale_diagnostic"]["restored_preprotection_dt_B_over_M"] for s in pulse),
                "max_restored_after_protection_proxy_dt_B_over_M": max(s["sampling_scale_diagnostic"]["restored_after_protection_dt_B_over_M"] for s in pulse),
                "proxy_preprotection_above_two_samples": sum(s["sampling_scale_diagnostic"]["restored_preprotection_dt_B_over_M"] > 2. for s in pulse),
                "first_pulse_headroom_interval_rps": pulse[0]["exact_all_wheel_PI_headroom_yaw_interval_rps_diagnostic_only"],
                "original_native_force_windows": native_windows(native), "samples": samples})
    old_cap = []
    old_root = work/"heading_turn_limit_02"
    remember(old_root/"protocol.json")
    old_protocol = json.loads((old_root/"protocol.json").read_text())
    assert old_protocol["conditions"] == {"limit0p6": .6, "limit1p0": 1.}
    for side in ("left", "right"):
        folder = old_root/f"stationary_turn_{side}_hold"/"limit1p0"
        for name in ("summary.json", "episode_metadata.json"):
            remember(folder/name)
        old_summary = json.loads((folder/"summary.json").read_text())
        old_meta = json.loads((folder/"episode_metadata.json").read_text())["episode_metadata"]
        old_cap.append({"side": side, "fixed_cap_rps": 1., "heading_peak_rad": old_summary["heading_peak_rad"],
            "failed_gates": old_summary["gates"]["failed"], "old_recorded_terrain": old_meta["terrain"],
            "comparability": "legacy hfield experiment, not the separately identified native-plane plant; fixed cap also retained the inner clipping operation"})
    assert pose_count == 208 and all(sha(path) == expected for path, expected in hashes.items())
    report = {"schema": "d1-pure-turn-inner-feedback-actuator-authority-qualification-v1",
        "algebra_checks_passed": True, "new_physics_steps": 0, "controller_compute_calls": 0,
        "new_contact_force_solves": 0, "saved_pose_count": pose_count, "fixed_sample_ticks": TICKS,
        "all_scratch_times_zero": True, "frozen77_unchanged": True, "inputs_unchanged": True,
        "mujoco_version": mujoco.__version__, "input_sha256": hashes,
        "single_candidate_formula": "raw v==0 and raw yaw!=0: r_eff=servo+4*(servo-body_yaw); otherwise original +/-0.6; final original wheel +/-30, PI/antiwindup and rated/outward protection",
        "dynamic_headroom_limiter_in_candidate": False, "damper_or_center_compensation_in_candidate": False,
        "headroom_identity": "Given |I_before|<=4<12, final original antiwindup request is within +/-12 iff |2.2*wheel_error+I_before|<=12. This is diagnostic only.",
        "wheel_damping_boundary": "target has no direct dependence on wheel velocity at fixed body/pose. Pre-protection PI retains positive 2.2 or 2.23 Nm/(rad/s) wheel-spin damping; rated saturation can temporarily flatten actual torque response.",
        "limitations": ["Masked feedback is an exact local controller fact; it does not alone prove the cause of body tracking failure.",
            "One-sample PI restarts from the archived memory at each state; no counterfactual state or PI memory is propagated.",
            "Rated saturation is expected evidence, not an automatic failure or a stability proof. All saved old states are inside their original speed bounds; future candidate speeds remain unpredicted.",
            "Native forces keep their original solved-cache phase and are not multiplied by endpoint velocities.",
            "Lateral leg/contact incompatibility and wheel slip remain possible limiting mechanisms. This qualification does not predict original gate pass.",
            "Only one feedback-restoration formula is proposed; no numeric cap grid or follow-up cap selection is authorized."],
        "legacy_fixed_one_cap_evidence": old_cap, "cases": cases}
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False); stream.write("\n")
    print(json.dumps({"report": str(args.output), "sha256": sha(args.output), "new_physics_steps": 0,
        "saved_pose_count": pose_count, "cases": [{k:v for k,v in c.items() if k not in ("samples", "original_native_force_windows")} for c in cases]}, allow_nan=False))


if __name__ == "__main__":
    def forbidden(*_args, **_kwargs):
        raise AssertionError("qualification forbids integration, forward/inverse dynamics and contact solve")
    with ExitStack() as stack:
        for api in ("mj_step", "mj_step1", "mj_step2", "mj_forward", "mj_inverse", "mj_collision"):
            stack.enter_context(patch.object(mujoco, api, forbidden))
        main()
