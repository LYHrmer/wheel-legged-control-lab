"""Read-only saved-state kinematics; no time integration or control execution.

Run from the repository with PYTHONDONTWRITEBYTECODE=1 and its PYTHONPATH.
Only the explicitly requested new JSON output is written, using exclusive mode.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.signal import find_peaks

from wheel_legged_control.d1.locomotion_terrain import D1LocomotionTerrainConfig
from wheel_legged_control.d1.model import (
    D1_JOINT_NAMES,
    JOINT_TORQUE_LIMIT,
    LEG_PREFIXES,
    build_d1_model,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rms(a):
    return float(np.sqrt(np.mean(np.asarray(a) ** 2)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    repo = Path(__file__).resolve()
    # Repository location is explicit and recorded; the source result protocol
    # authenticates its frozen model inputs before any geometric calculation.
    repo = Path("/home/lyh/wheel-legged-control-lab")
    protocol = json.loads((args.input / "protocol.json").read_text())
    input_hashes = {str(args.input / "protocol.json"): sha(args.input / "protocol.json")}
    frozen_checks = {}
    for name, expected in protocol["input_sha256"].items():
        p = Path(name)
        if not p.is_absolute():
            p = repo / p
        if p.exists():
            actual = sha(p)
            frozen_checks[str(p)] = actual == expected
            input_hashes[str(p)] = actual
    if not all(frozen_checks.values()):
        raise RuntimeError("An input recorded by the source protocol changed")
    input_hashes[str(Path(__file__).resolve())] = sha(__file__)

    model = build_d1_model(locomotion_terrain=D1LocomotionTerrainConfig(layout="flat"))
    data = mujoco.MjData(model)
    base = model.body("base_link").id
    wheel_ids = [model.body(leg + "_foot").id for leg in LEG_PREFIXES]
    dofs = np.array([model.jnt_dofadr[model.joint(n).id] for n in D1_JOINT_NAMES]).reshape(4, 4)
    qaddrs = np.array([model.jnt_qposadr[model.joint(n).id] for n in D1_JOINT_NAMES]).reshape(4, 4)
    terrain_ids = {
        i for i in range(model.ngeom)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith(("floor", "terrain_"))
    }
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    velocity = np.zeros(6)
    mass = np.zeros((model.nv, model.nv))

    def kinematics(qpos, qvel):
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        # These update geometric/velocity caches, never qpos, qvel or time.
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        mujoco.mj_comVel(model, data)

    nominal_npz = args.input / "flat_forward_stop/bypass/states.npz"
    with np.load(nominal_npz) as z:
        nominal_q = z["qpos"][0].copy()
    kinematics(nominal_q, np.zeros(model.nv))
    mujoco.mj_crb(model, data)
    mujoco.mj_fullM(model, data, mass)
    mode = np.zeros(model.nv)
    mode[0] = 1.0  # nominal unit visible-origin world-x translation
    jrows = []
    leg_mode_details = []
    for leg, wid in enumerate(wheel_ids):
        mujoco.mj_jacBody(model, data, jp, jr, wid)
        jac = jp[:, dofs[leg, :3]]
        dqdx = np.linalg.solve(jac, -np.array([1.0, 0.0, 0.0]))
        mode[dofs[leg, :3]] = dqdx
        jrows.append(jac[0].copy())
        leg_mode_details.append({"leg": LEG_PREFIXES[leg], "jacobian": jac.tolist(), "dq_dx_rad_pm": dqdx.tolist()})
    leg_dofs = dofs[:, :3].ravel()
    m_eff = float(mode @ mass @ mode)
    k_eff = float(80.0 * (mode[leg_dofs] @ mode[leg_dofs]))
    b_existing = float(3.0 * (mode[leg_dofs] @ mode[leg_dofs]) + mode @ (model.dof_damping * mode))
    b_critical = float(2.0 * np.sqrt(m_eff * k_eff))
    per_leg = float(max(0.0, b_critical - b_existing) / 4.0)
    gain_provenance = {
        "status": "untested_nominal_mode_design_not_an_identified_closed_loop_model",
        "source_state": str(nominal_npz) + ":qpos[0]",
        "mode_assumptions": "visible base translates in world x; roll/pitch/y/z fixed; four wheel centers fixed; wheel spins fixed; leg IK differential compensates translation",
        "total_mass_kg": float(model.body_mass.sum()),
        "effective_mass_kg": m_eff, "joint_pd_stiffness_Npm": k_eff,
        "existing_joint_and_passive_damping_Nspm": b_existing,
        "critical_total_damping_Nspm": b_critical,
        "proposed_extra_per_leg_damping_Nspm": per_leg,
        "undamped_frequency_hz": float(np.sqrt(k_eff / m_eff) / (2 * np.pi)),
        "existing_nominal_damping_ratio": b_existing / b_critical,
        "mode_vector": mode.tolist(), "legs": leg_mode_details,
        "excluded_effects": ["gravity/support geometric stiffness", "body pitch coupling", "rolling/sliding contacts", "wheel PI feedback", "actuator saturation", "future trajectories"],
    }
    episodes = []
    for case in ("flat_forward_stop", "flat_reverse_stop"):
        direction = 1 if "forward" in case else -1
        for condition in ("bypass", "release_0p5"):
            folder = args.input / case / condition
            for filename in ("states.npz", "trace.jsonl.gz", "episode_metadata.json", "summary.json"):
                path = folder / filename
                input_hashes[str(path)] = sha(path)
            z = np.load(folder / "states.npz")
            with gzip.open(folder / "trace.jsonl.gz", "rt") as f:
                trace = [json.loads(line) for line in f]
            metadata = json.loads((folder / "episode_metadata.json").read_text())["episode_metadata"]
            if any(v != 1.0 for v in metadata["domain"].values()):
                raise RuntimeError("This helper assumes the recorded nominal domain")
            points, closure, velocity_error, invariance, increments, bounds, powers = [], [], [], [], [], [], []
            for tick in range(400, 800):
                q, v = z["qpos"][tick], z["qvel"][tick]
                kinematics(q, v)
                mujoco.mj_collision(model, data)
                rotation = data.xmat[base].reshape(3, 3)
                forward = rotation[:, 0]
                mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, base, velocity, 0)
                body = float(forward @ velocity[3:])
                angular, com = velocity[:3].copy(), data.xipos[base].copy()
                components, leg_records = [], []
                damping_matrix = np.zeros((model.nv, model.nv))
                delta = np.zeros(16)
                for leg, wid in enumerate(wheel_ids):
                    contacts = []
                    for contact in data.contact:
                        g1, g2 = int(contact.geom1), int(contact.geom2)
                        if ((g1 in terrain_ids and model.geom_bodyid[g2] == wid)
                                or (g2 in terrain_ids and model.geom_bodyid[g1] == wid)):
                            contacts.append(contact.pos.copy())
                    # No dynamics/contact-force solve: a collision candidate is
                    # not evidence of load-bearing support. Missing candidates
                    # use a named geometric bottom-point proxy, never 'slip'.
                    point = np.mean(contacts, axis=0) if contacts else data.xpos[wid] - np.array([0.0, 0.0, 0.087])
                    mujoco.mj_jac(model, data, jp, jr, point, wid)
                    j = forward @ jp
                    tangent = float(j @ v)
                    attitude = float(-forward @ np.cross(angular, point - com))
                    leg_part = float(-j[dofs[leg, :3]] @ v[dofs[leg, :3]])
                    geometry = float(-(j[dofs[leg, 3]] + 0.087) * v[dofs[leg, 3]])
                    components.append([tangent, attitude, leg_part, geometry])
                    mujoco.mj_jacBody(model, data, jp, jr, wid)
                    jcenter = forward @ jp[:, dofs[leg, :3]]
                    u = float(jcenter @ v[dofs[leg, :3]])
                    delta[4 * leg:4 * leg + 3] = -per_leg * jcenter * u
                    row = np.zeros(model.nv)
                    row[dofs[leg, :3]] = jcenter
                    damping_matrix += per_leg * np.outer(row, row)
                    leg_records.append({"leg": LEG_PREFIXES[leg], "collision_candidate_points": len(contacts), "uses_bottom_proxy": not bool(contacts), "center_relative_x_speed_mps": u})
                wheel = float(0.087 * np.mean(v[dofs[:, 3]]))
                mismatch = body - wheel
                average = np.mean(components, axis=0)
                error = float(mismatch - np.sum(average))
                truth_error = float(body - trace[tick]["body_forward_before_mps"])
                closure.append(abs(error)); velocity_error.append(abs(truth_error))
                mujoco.mj_crb(model, data)
                mujoco.mj_fullM(model, data, mass)
                decay = float(np.max(np.linalg.eigvals(np.linalg.solve(mass, damping_matrix)).real))
                power = float(delta @ v[dofs.ravel()])
                total = np.array(trace[tick]["controller_unlimited_torque_nm"]) + delta
                increments.append(float(np.max(np.abs(delta))))
                bounds.append(float(np.max(np.abs(total) / JOINT_TORQUE_LIMIT)))
                powers.append(power)
                invariance.append(bool(np.array_equal(data.qpos, q) and np.array_equal(data.qvel, v) and data.time == 0.0))
                points.append({
                    "tick": tick, "body_vx_mps": body, "wheel_roll_mps": wheel,
                    "mismatch_mps": mismatch, "point_tangent_mps": float(average[0]),
                    "base_angular_component_mps": float(average[1]),
                    "leg_component_mps": float(average[2]), "wheel_axis_geometry_component_mps": float(average[3]),
                    "closure_error_mps": error, "logged_body_velocity_error_mps": truth_error,
                    "legs": leg_records, "hypothetical_damper_delta_torque_nm": delta.tolist(),
                    "hypothetical_damper_joint_power_W": power,
                    "hypothetical_free_inertia_damping_dt_lambda_max": decay * 0.01,
                    "hypothetical_total_torque_limit_ratio": bounds[-1],
                    "wheel_integral_mean_before_Nm": float(np.mean(trace[tick]["wheel_integral_before_nm"])),
                    "wheel_integral_mean_after_Nm": float(np.mean(trace[tick]["wheel_integral_after_nm"])),
                })
            windows = []
            for start, end in ((400, 450), (450, 500), (500, 600), (500, 800)):
                window = points[start - 400:end - 400]
                values = {field: rms([p[field] for p in window]) for field in (
                    "body_vx_mps", "wheel_roll_mps", "mismatch_mps", "point_tangent_mps",
                    "base_angular_component_mps", "leg_component_mps", "wheel_axis_geometry_component_mps")}
                values.update({"start_tick": start, "end_tick_exclusive": end,
                    "leg_mismatch_correlation": float(np.corrcoef([p["leg_component_mps"] for p in window], [p["mismatch_mps"] for p in window])[0, 1]),
                    "wheel_frames_without_collision_candidate": sum(leg["uses_bottom_proxy"] for p in window for leg in p["legs"]),
                    "total_wheel_frames": len(window) * 4})
                windows.append(values)
            vb = direction * np.array([r["body_forward_before_mps"] for r in trace])
            pitch = direction * np.r_[0, [r["actual_roll_pitch_rad"][1] for r in trace[:-1]]] * 180 / np.pi
            extrema = {}
            # Fixed descriptive peak extraction only, not a controller parameter.
            for name, a, prominence in (("directional_body_vx", vb, 0.01), ("directional_pitch_deg", pitch, 0.1)):
                for label, sign in (("max", 1), ("min", -1)):
                    peaks = find_peaks(sign * a[400:], distance=35, prominence=prominence)[0] + 400
                    extrema[name + "_" + label] = [{"tick": int(t), "value": float(a[t])} for t in peaks]
            crossings = np.where((vb[:-1] > 0) & (vb[1:] <= 0))[0] + 1
            extrema["first_reverse_tick"] = next((int(t) for t in crossings if t >= 400), None)
            first = []
            for t in (399, 400, 401, 449, 500):
                row = trace[t]
                request = z["requested_torque_nm"][t, [3, 7, 11, 15]]
                applied = z["actuator_applied_nm"][t][:, [3, 7, 11, 15]]
                first.append({"tick": t, "mean_requested_wheel_Nm": float(request.mean()),
                    "mean_applied_wheel_Nm_by_substep": applied.mean(axis=1).tolist(),
                    "applied_equals_request_all_substeps": bool(np.array_equal(applied, np.tile(request, (5, 1)))),
                    "integral_before_mean_Nm": float(np.mean(row["wheel_integral_before_nm"])),
                    "integral_after_mean_Nm": float(np.mean(row["wheel_integral_after_nm"])),
                    "P_mean_Nm": float(request.mean() - np.mean(row["wheel_integral_after_nm"]))})
            episodes.append({
                "case": case, "condition": condition, "windows": windows, "extrema": extrema,
                "selected_wheel_torque_receipts": first, "max_closure_error_mps": max(closure),
                "max_logged_body_velocity_error_mps": max(velocity_error),
                "qpos_qvel_time_unchanged_all_frames": all(invariance),
                "hypothetical_damper_envelope": {
                    "max_abs_increment_Nm": max(increments), "max_total_torque_limit_ratio": max(bounds),
                    "max_joint_power_W": max(powers), "min_joint_power_W": min(powers),
                    "max_free_inertia_dt_lambda": max(p["hypothetical_free_inertia_damping_dt_lambda_max"] for p in points),
                    "meaning": "algebra on old saved trajectories only; neither a rollout nor proof of closed-loop stability or future saturation bounds"},
                "per_tick": points,
            })
    report = {
        "schema": "d1-stop-saved-state-kinematic-diagnosis-v1",
        "mujoco_version": mujoco.__version__, "numpy_version": np.__version__,
        "input_sha256": input_hashes, "recorded_input_checks": frozen_checks,
        "actual_mujoco_calculation_apis": ["mj_kinematics", "mj_comPos", "mj_comVel", "mj_collision", "mj_objectVelocity", "mj_jac", "mj_jacBody", "mj_crb", "mj_fullM"],
        "simulation_steps": 0, "time_integrations": 0, "controller_compute_calls": 0,
        "identity": "body COM vx - 0.087 mean wheel qdot = mean point tangent vx - mean [omega_base cross (point-COM)]_x - mean J_leg,x qdot_leg - mean (J_wheel,x+0.087) qdot_wheel; all projected into visible base x",
        "contact_limitations": ["mj_collision returns geometric collision candidates, not load-bearing constraints", "no contact force solve; no slip-force or work attribution", "when no candidate exists, point is wheel-center minus world-z radius and is explicitly a proxy", "instantaneous kinematic components do not prove separate causal contributions"],
        "hypothetical_damper_gain_provenance": gain_provenance,
        "episodes": episodes,
    }
    with args.output.open("x") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps({"output": str(args.output), "sha256": sha(args.output), "gain": gain_provenance,
        "episodes": [{k: v for k, v in e.items() if k not in ("per_tick",)} for e in episodes]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
