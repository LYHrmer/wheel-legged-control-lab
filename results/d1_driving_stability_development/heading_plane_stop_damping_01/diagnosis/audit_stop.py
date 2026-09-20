"""Bounded offline diagnosis of two completed native-plane stop trajectories.

No plant/environment/controller is instantiated. Three integration APIs are
guarded. Saved endpoint geometry, velocities and inertia are calculated only;
contact forces come from the saved native and synchronized-endpoint receipts.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np
from scipy.signal import find_peaks

from wheel_legged_control.d1.model import D1_JOINT_NAMES, JOINT_TORQUE_LIMIT, LEG_PREFIXES, build_d1_model

FIXED_B = 126.4374005337902
COMPONENTS = ("slip_x", "contact_normal_x", "base_angular", "leg", "wheel_geometry")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    with gzip.open(path, "rt") as stream:
        return [json.loads(line) for line in stream]


def rms(values):
    return float(np.sqrt(np.mean(np.square(values))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    repo, work = args.repo.resolve(), args.work.resolve()
    source = work / "flat_plane_01"
    hashes = {str(Path(__file__).resolve()): sha(__file__)}
    for path in (source / "protocol.json", work / "flat_plane_partial_audit.json",
                 work / "flat_plane_preflight_01/report.json", work / "stop_kinematic_diagnosis_01/report.json"):
        hashes[str(path)] = sha(path)
    frozen_path = repo / "results/d1_budget_study/protocol.json"
    frozen = json.loads(frozen_path.read_text())["source_sha256"]
    assert len(frozen) == 77
    for name, expected in frozen.items():
        assert sha(repo / name) == expected
        hashes[str(repo / name)] = expected
    model = build_d1_model()
    data = mujoco.MjData(model)
    assert model.nhfield == 0 and model.geom_type[model.geom("floor").id] == mujoco.mjtGeom.mjGEOM_PLANE
    base = model.body("base_link").id
    wheels = [model.body(leg + "_foot").id for leg in LEG_PREFIXES]
    dofs = np.array([model.jnt_dofadr[model.joint(name).id] for name in D1_JOINT_NAMES]).reshape(4, 4)
    leg_dofs = dofs[:, :3].ravel()
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    velocity = np.zeros(6)
    mass = np.zeros((model.nv, model.nv))

    def kinematics(qpos, qvel):
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        mujoco.mj_comVel(model, data)
        assert np.array_equal(data.qpos, qpos) and np.array_equal(data.qvel, qvel) and data.time == 0.

    with np.load(source / "flat_forward_stop/states.npz", allow_pickle=False) as z:
        kinematics(z["qpos"][0], z["qvel"][0])
    mujoco.mj_crb(model, data)
    mujoco.mj_fullM(model, data, mass)
    mode = np.zeros(model.nv)
    mode[0] = 1.
    for leg, body in enumerate(wheels):
        mujoco.mj_jacBody(model, data, jp, jr, body)
        mode[dofs[leg, :3]] = np.linalg.solve(jp[:, dofs[leg, :3]], [-1., 0., 0.])
    m_eff = float(mode @ mass @ mode)
    k_eff = float(80. * (mode[leg_dofs] @ mode[leg_dofs]))
    b_old = float(3. * (mode[leg_dofs] @ mode[leg_dofs]) + mode @ (model.dof_damping * mode))
    b_critical = float(2. * np.sqrt(m_eff * k_eff))
    derived_b = (b_critical - b_old) / 4.
    assert abs(derived_b - FIXED_B) < 1e-10
    episodes = []
    for case in ("flat_forward_stop", "flat_reverse_stop"):
        folder = source / case
        manifest = json.loads((folder / "complete_manifest.json").read_text())
        hashes[str(folder / "complete_manifest.json")] = sha(folder / "complete_manifest.json")
        for name, identity in manifest.items():
            path = folder / name
            assert sha(path) == identity["sha256"] and path.stat().st_size == identity["bytes"]
            hashes[str(path)] = identity["sha256"]
        with np.load(folder / "states.npz", allow_pickle=False) as z:
            qpos, qvel, positions = z["qpos"].copy(), z["qvel"].copy(), z["truth_positions_world_m"].copy()
        trace = rows(folder / "trace.jsonl.gz")
        diagnostic = rows(folder / "plane_diagnostics.jsonl.gz")
        native = rows(folder / "native_physics_entries.jsonl.gz")
        summary = json.loads((folder / "summary.json").read_text())
        assert len(qpos) == len(qvel) == 801 and len(trace) == len(diagnostic) == 800 and len(native) == 4000
        assert all(r["tick"] == t and r["endpoint_tick"] == t + 1 for t, r in enumerate(trace))
        assert all(r["tick"] == t and r["endpoint_tick"] == t + 1 for t, r in enumerate(diagnostic))
        assert all(e["returned"] for e in native)
        assert all(d["raw_user_command"]["forward_velocity_mps"] == d["servo_command"]["forward_velocity_mps"] for d in diagnostic)
        raw = np.array([d["raw_user_command"]["forward_velocity_mps"] for d in diagnostic])
        assert np.all(raw[400:] == 0.) and raw[399] != 0.
        errors, records = [], []
        for tick in range(400, 801):
            kinematics(qpos[tick], qvel[tick])
            forward = data.xmat[base].reshape(3, 3)[:, 0]
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, base, velocity, 0)
            body_vx = float(forward @ velocity[3:])
            com, omega = data.xipos[base].copy(), velocity[:3].copy()
            rolling = .087 * qvel[tick, dofs[:, 3]]
            error = abs(body_vx - trace[tick - 1]["body_forward_mps"])
            errors.append(error)
            assert error <= 1e-12
            center_u, delta, damping_matrix = [], np.zeros(16), np.zeros((model.nv, model.nv))
            for leg, body in enumerate(wheels):
                mujoco.mj_jacBody(model, data, jp, jr, body)
                jx = forward @ jp[:, dofs[leg, :3]]
                u = float(jx @ qvel[tick, dofs[leg, :3]])
                center_u.append(u)
                delta[4*leg:4*leg+3] = -FIXED_B * jx * u
                row = np.zeros(model.nv)
                row[dofs[leg, :3]] = jx
                damping_matrix += FIXED_B * np.outer(row, row)
            if tick < 800:
                np.testing.assert_allclose(center_u, diagnostic[tick]["leg_forward_velocity_before_mps"], rtol=0, atol=1e-12)
                assert abs(body_vx - diagnostic[tick]["body_forward_before_mps"]) <= 1e-12
                assert abs(np.mean(rolling) - diagnostic[tick]["wheel_rolling_before_mps"]) <= 1e-12
            endpoint = diagnostic[tick - 1]["endpoint_contacts"]
            assert endpoint["sampling"] == "synchronized_endpoint_not_control_average"
            assert abs(endpoint["measurement_time_s"] - .01*tick) < 1e-11
            values, loads, contact_records = [], [], []
            for contact in endpoint["contacts"]:
                leg = contact["wheel_index"]
                normal = np.asarray(contact["normal_world"])
                assert np.array_equal(normal, [0., 0., 1.])
                point = np.asarray(contact["pos_world_m"])
                mujoco.mj_jac(model, data, jp, jr, point, wheels[leg])
                actual_point = jp @ qvel[tick]
                np.testing.assert_allclose(actual_point, contact["relative_velocity_world_mps"], rtol=0, atol=1e-12)
                leg_velocity = jp[:, leg_dofs] @ qvel[tick, leg_dofs]
                wheel_velocity = jp[:, dofs[:, 3]] @ qvel[tick, dofs[:, 3]]
                np.testing.assert_allclose(leg_velocity, contact["leg_dof_velocity_world_mps"], rtol=0, atol=1e-12)
                np.testing.assert_allclose(wheel_velocity, contact["wheel_dof_velocity_world_mps"], rtol=0, atol=1e-12)
                tangent = actual_point - normal * (normal @ actual_point)
                components = np.array([
                    forward @ tangent,
                    forward @ (actual_point - tangent),
                    -forward @ np.cross(omega, point - com),
                    -forward @ leg_velocity,
                    -forward @ wheel_velocity - rolling[leg],
                ])
                mismatch = body_vx - rolling[leg]
                closure = float(mismatch - components.sum())
                assert abs(closure) <= 1e-12
                weight = max(0., float(contact["normal_force_n"]))
                values.append(np.r_[mismatch, components, rolling[leg]])
                loads.append(weight)
                contact_records.append({"wheel": leg, "normal_force_n": weight,
                    "tangent_speed_mps": float(np.linalg.norm(tangent)), "mismatch_mps": float(mismatch),
                    **{name: float(v) for name, v in zip(COMPONENTS, components)}, "closure_error_mps": closure})
            assert sum(loads) > 0., "No loaded endpoint contact: do not fabricate zero-slip support"
            weighted = np.average(np.asarray(values), axis=0, weights=loads)
            per_leg = [
                np.average([values[i] for i, c in enumerate(contact_records) if c["wheel"] == leg], axis=0,
                           weights=[loads[i] for i, c in enumerate(contact_records) if c["wheel"] == leg])
                if sum(loads[i] for i, c in enumerate(contact_records) if c["wheel"] == leg) > 0 else None
                for leg in range(4)
            ]
            equal_wheel = np.mean(per_leg, axis=0) if all(p is not None for p in per_leg) else None
            record = {"state_tick": tick, "recorded_time_s": endpoint["measurement_time_s"],
                "audit_time_s": float(data.time), "body_com_vx_mps": body_vx,
                "four_wheel_mean_rolling_mps": float(np.mean(rolling)),
                "four_wheel_mismatch_mps": float(body_vx - np.mean(rolling)),
                "leg_center_relative_vx_by_leg_mps": center_u,
                "endpoint_normal_load_n": float(sum(loads)),
                "loaded_wheels": sum(p is not None for p in per_leg),
                "normal_load_weighted": {"mismatch": float(weighted[0]),
                    **{name: float(v) for name, v in zip(COMPONENTS, weighted[1:6])},
                    "rolling": float(weighted[6]),
                    "tangent_speed_rms": float(np.sqrt(np.average([c["tangent_speed_mps"]**2 for c in contact_records], weights=loads)))},
                "equal_wheel_normal_weighted": None if equal_wheel is None else {
                    "mismatch": float(equal_wheel[0]), **{name: float(v) for name, v in zip(COMPONENTS, equal_wheel[1:6])}},
                "contacts": contact_records}
            if tick < 800:
                mujoco.mj_crb(model, data)
                mujoco.mj_fullM(model, data, mass)
                decay = float(np.linalg.eigvals(np.linalg.solve(mass, damping_matrix)).real.max())
                base_request = np.asarray(diagnostic[tick]["unlimited_torque_nm"])
                record["hypothetical_fixed_b"] = {
                    "delta_torque_nm": delta.tolist(),
                    "joint_power_w": float(delta @ qvel[tick, dofs.ravel()]),
                    "requested_plus_delta_limit_ratio": float(np.max(np.abs(base_request + delta)/JOINT_TORQUE_LIMIT)),
                    "free_inertia_dt_lambda_max": decay*.01,
                }
            records.append(record)
        windows = []
        for lo, hi in ((400, 450), (450, 500), (500, 600), (500, 800), (700, 800)):
            selected = [r for r in records if lo <= r["state_tick"] < hi]
            x = np.array([r["normal_load_weighted"]["mismatch"] for r in selected])
            w = {"state_ticks": [lo, hi], "end_exclusive": True,
                "body_com_vx_rms_mps": rms([r["body_com_vx_mps"] for r in selected]),
                "wheel_rolling_rms_mps": rms([r["four_wheel_mean_rolling_mps"] for r in selected]),
                "four_wheel_mismatch_rms_mps": rms([r["four_wheel_mismatch_mps"] for r in selected]),
                "normal_load_weighted_component_rms_mps": {name: rms([r["normal_load_weighted"][name] for r in selected]) for name in ("mismatch", *COMPONENTS, "tangent_speed_rms")},
                "projection_shares_not_causal": {name: float(x @ np.array([r["normal_load_weighted"][name] for r in selected])/(x @ x)) for name in COMPONENTS},
                "endpoint_normal_load_mean_n": float(np.mean([r["endpoint_normal_load_n"] for r in selected])),
                "frames_not_all_four_wheels_loaded": sum(r["loaded_wheels"] < 4 for r in selected)}
            entries = native[lo*5:hi*5]
            durations = np.array([e["actual_dt_s"] for e in entries])
            nf = np.array([e["contacts"]["summed_normal_force_world_n"] for e in entries])
            tf = np.array([e["contacts"]["summed_tangent_force_world_n"] for e in entries])
            ratio_numerator = np.array([sum(np.linalg.norm(c["tangent_force_world_n"]) for c in e["contacts"]["contacts"]) for e in entries])
            ratio_denominator = .9*nf[:, 2]
            w["native_executed_loads"] = {"interval_ticks": [lo, hi], "end_exclusive": True,
                "sampling": "native_step_solved_cache_not_synchronized_endpoint",
                "substeps": len(entries), "normal_force_z_time_mean_n": float(np.average(nf[:, 2], weights=durations)),
                "tangent_force_x_time_mean_n": float(np.average(tf[:, 0], weights=durations)),
                "tangent_force_x_rms_n": rms(tf[:, 0]),
                "sum_tangent_magnitude_over_mu_normal_load": float(durations @ ratio_numerator / (durations @ ratio_denominator)),
                "substeps_without_active_wheel_contacts": sum(not any(e["contacts"]["active_contacts_by_wheel"]) for e in entries),
                "max_horizontal_normal": max(e["contacts"]["max_horizontal_normal"] for e in entries)}
            windows.append(w)
        direction = 1. if case == "flat_forward_stop" else -1.
        directional = direction*np.array([r["body_com_vx_mps"] for r in records])
        extrema = {}
        for name, sign in (("positive_peaks", 1), ("negative_peaks", -1)):
            indices = find_peaks(sign*directional, distance=35, prominence=.01)[0]
            extrema[name] = [{"state_tick": records[i]["state_tick"], "directional_body_com_vx_mps": float(directional[i])} for i in indices]
        extrema["first_reverse_state_tick"] = next((r["state_tick"] for r in records if direction*r["body_com_vx_mps"] < 0.), None)
        torque_receipts = []
        for tick in (399, 400, 401, 405, 425, 450, 475, 500, 550, 600, 700, 799):
            d = diagnostic[tick]
            request = np.asarray(d["requested_torque_nm"])[[3, 7, 11, 15]]
            applied = np.asarray(d["applied_torque_nm"])
            native_ctrl = np.array([e["ctrl_nm"] for e in native[5*tick:5*(tick+1)]])
            assert np.array_equal(applied, native_ctrl)
            torque_receipts.append({"execution_tick": tick, "mean_wheel_requested_nm": float(np.mean(request)),
                "mean_wheel_applied_by_substep_nm": applied[:, [3, 7, 11, 15]].mean(axis=1).tolist(),
                "integral_before_mean_nm": float(np.mean(d["wheel_integral_before_nm"])),
                "integral_after_mean_nm": float(np.mean(d["wheel_integral_after_nm"])),
                "P_mean_nm": float(np.mean(request) - np.mean(d["wheel_integral_after_nm"])),
                "wheel_error_before_mean_rad_s": float(np.mean(d["wheel_error_before_rad_s"])),
                "native_ctrl_matches_applied": True})
        scored_velocity = np.array([r["body_forward_mps"] for r in trace])
        computed_scores = {"velocity_rmse_mps": rms(scored_velocity - raw),
            "late_stop_max_abs_body_vx_mps": max(abs(r["body_com_vx_mps"]) for r in records if 500 <= r["state_tick"] < 800),
            "late_stop_cumulative_planar_path_m": float(np.linalg.norm(np.diff(positions[500:701, :2], axis=0), axis=1).sum())}
        for name, value in computed_scores.items():
            assert abs(value - summary[name]) <= 1e-12, (name, value, summary[name])
        hypothetical = [r["hypothetical_fixed_b"] for r in records if "hypothetical_fixed_b" in r]
        episodes.append({"case": case, "scores_recomputed": computed_scores,
            "original_failed_gates": summary["gates"]["failed"], "original_stop_motion": summary["stop_motion"],
            "windows": windows, "extrema": extrema, "selected_wheel_torque_receipts": torque_receipts,
            "max_body_com_logged_velocity_error_mps": max(errors),
            "max_per_contact_identity_closure_mps": max(abs(c["closure_error_mps"]) for r in records for c in r["contacts"]),
            "torque_protected_joint_intervals": sum(sum(d["torque_protected"]) for d in diagnostic),
            "all_requested_torque_equals_each_applied_substep": all(np.array_equal(np.asarray(d["applied_torque_nm"]), np.tile(d["requested_torque_nm"], (5, 1))) for d in diagnostic),
            "hypothetical_fixed_b_envelope": {
                "max_abs_delta_nm": max(abs(v) for r in hypothetical for v in r["delta_torque_nm"]),
                "max_requested_plus_delta_limit_ratio": max(r["requested_plus_delta_limit_ratio"] for r in hypothetical),
                "max_joint_power_w": max(r["joint_power_w"] for r in hypothetical),
                "min_joint_power_w": min(r["joint_power_w"] for r in hypothetical),
                "max_free_inertia_dt_lambda": max(r["free_inertia_dt_lambda_max"] for r in hypothetical)},
            "per_endpoint": records})
    assert all(sha(path) == expected for path, expected in hashes.items())
    report = {"schema": "d1-plane-stop-offline-diagnosis-v1", "new_physics_steps": 0,
        "integration_attempts": 0, "controller_compute_calls": 0, "all_audit_times_zero": True,
        "mujoco_version": mujoco.__version__, "frozen77_match": True,
        "native_plane_model_mass_kg": float(model.body_mass.sum()),
        "fixed_gain_not_retuned": FIXED_B,
        "nominal_mode_recalculation": {"M_eff_kg": m_eff, "K_eff_npm": k_eff,
            "B_old_nspm": b_old, "B_critical_nspm": b_critical, "derived_per_leg_b_nspm": derived_b,
            "mode_frequency_hz": float(np.sqrt(k_eff/m_eff)/(2*np.pi)), "mode_vector": mode.tolist()},
        "kinematic_identity": "body COM vx - r*qdot_wheel = contact tangent x + contact normal x - [omega cross (point-COM)]_x - J_leg,x*qdot_leg - (J_wheel,x*qdot_wheel + r*qdot_wheel)",
        "aggregation": "within each endpoint all contacts are weighted by nonnegative recorded synchronized normal force; rolling uses exactly the same wheel weights; equal-wheel aggregation is recorded separately when all four wheels loaded",
        "phase": "states[k], trace[k-1] endpoint, diagnostic[k-1].endpoint_contacts are time k*0.01; diagnostic[k] before fields agree for k<800; native force windows are executed intervals [lo,hi), kept separate",
        "limitations": ["No new trajectory or repair claim", "Projection shares are an additive kinematic identity, not causal or energy fractions",
            "Endpoint forces are saved synchronized mj_forward cache results, not native substep forces",
            "Actual executed native contact loads are summarized independently; native force is not multiplied by an endpoint velocity from another phase",
            "Nominal fixed-center pure-x mode is not a closed-loop identification; sampled nonpositive delta power does not prove held-input passivity",
            "Hypothetical deltas, torque margin and free-inertia eigenvalues use the old saved trajectory only"],
        "episodes": episodes, "input_sha256": hashes, "inputs_unchanged": True}
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"report": str(args.output), "report_sha256": sha(args.output),
        "gain": report["nominal_mode_recalculation"],
        "episodes": [{k: v for k, v in e.items() if k not in ("per_endpoint",)} for e in episodes]}, allow_nan=False))


if __name__ == "__main__":
    def forbidden(*args, **kwargs):
        raise AssertionError("offline diagnosis must not integrate physics")

    with patch.object(mujoco, "mj_step", forbidden), patch.object(mujoco, "mj_step1", forbidden), patch.object(mujoco, "mj_step2", forbidden):
        main()
