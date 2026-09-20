"""Read saved JSON/NPZ only; no MuJoCo import, model, forward, or physics steps."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    with gzip.open(path, "rt") as stream:
        return [json.loads(line) for line in stream]


def stats(values):
    a = np.asarray(values, dtype=float)
    return {"mean": float(a.mean()), "rms": float(np.sqrt(np.mean(a*a))),
            "min": float(a.min()), "max": float(a.max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    inputs = {str(Path(__file__).resolve()): sha(Path(__file__)),
              str(args.input / "protocol.json"): sha(args.input / "protocol.json"),
              str(args.input / "summary.json"): sha(args.input / "summary.json")}
    result = {"schema": "d1-turn-contact-offline-decomposition-v1",
              "new_physics_steps": 0, "plant_instantiated": False,
              "window": "executed ticks200..249; ALL kinematics/contact fits at endpoints201..250",
              "fit": "Each contact velocity component is first projected onto its measured tangent plane, then u_forward = intercept - omega_fit * contact_lateral_offset; equal total weight per active wheel, equal weight per contact within that wheel",
              "identity": "wheel_spin_yaw_minus_base_fit = leg_fit_yaw_minus_slip_fit_yaw (static terrain); signed window means are kinematic accounting, not causal percentages",
              "contact_sampling": "synchronized endpoint, not substep impulse or time-integrated slip",
              "cases": {}}
    for side, sign in (("left", 1), ("right", -1)):
        for condition in ("limit0p6", "limit1p0"):
            root = args.input / f"stationary_turn_{side}_hold" / condition
            for filename in ("turn_diagnostics.jsonl.gz", "trace.jsonl.gz", "states.npz", "summary.json"):
                p = root / filename
                inputs[str(p)] = sha(p)
            diagnostic = read_rows(root / "turn_diagnostics.jsonl.gz")
            trace = read_rows(root / "trace.jsonl.gz")
            with np.load(root / "states.npz") as z:
                qpos = z["qpos"].copy()
            assert len(diagnostic) == len(trace) == 800 and len(qpos) == 801
            endpoint_rows, decomposition_residuals, friction_ratios, contact_powers = [], [], [], []
            normal_horizontal = []
            excluded_fit_endpoints = []
            torque_request = np.asarray([r["requested_torque_nm"] for r in diagnostic])
            torque_applied = np.asarray([r["applied_torque_nm"] for r in diagnostic])
            integral = np.asarray([r["wheel_integral_after_nm"] for r in diagnostic])
            assert torque_request.shape == (800, 16) and torque_applied.shape == (800, 5, 16)
            for k in range(200, 250):
                row = diagnostic[k]
                assert row["tick"] == k and row["endpoint_tick"] == k + 1
                cinfo = row["contacts"]
                assert abs(cinfo["measurement_time_s"] - (k + 1)*.01) < 1e-9
                contacts = cinfo["contacts"]
                assert contacts, "no-contact endpoint cannot be treated as zero slip"
                quat = qpos[k + 1, 3:7]
                w, x, y, z = quat
                yaw = np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))
                rotation = np.array([[np.cos(yaw), np.sin(yaw), 0.],
                                     [-np.sin(yaw), np.cos(yaw), 0.], [0., 0., 1.]])
                reference = np.asarray(cinfo["reference_world_m"])
                points = np.asarray([c["pos_world_m"] for c in contacts])
                offsets = (points-reference) @ rotation.T
                wheel_ids = np.asarray([c["wheel_index"] for c in contacts])
                active = np.unique(wheel_ids)
                weights = np.asarray([1./(len(active)*np.count_nonzero(wheel_ids == wid)) for wid in wheel_ids])
                assert abs(weights.sum()-1) < 1e-12
                centered_y = offsets[:, 1] - weights @ offsets[:, 1]
                denominator = float(weights @ centered_y**2)
                if denominator <= 1e-4 or len(active) < 2:
                    excluded_fit_endpoints.append(k+1)
                    continue
                normals = np.asarray([c["normal_world"] for c in contacts])
                assert np.max(np.abs(np.linalg.norm(normals, axis=1)-1)) < 1e-10
                normal_horizontal.extend(np.linalg.norm(normals[:, :2], axis=1).tolist())
                components = {}
                for label, key in (("base", "free_base_velocity_world_mps"),
                                   ("leg", "leg_dof_velocity_world_mps"),
                                   ("wheel", "wheel_dof_velocity_world_mps"),
                                   ("terrain", "terrain_point_velocity_world_mps"),
                                   ("slip", "relative_velocity_world_mps")):
                    world_v = np.asarray([c[key] for c in contacts])
                    tangent_v = world_v - np.sum(world_v*normals, axis=1)[:, None]*normals
                    v = tangent_v @ rotation.T
                    components[label] = {"yaw_fit_rps": float(-(weights*centered_y) @ v[:, 0] / denominator),
                                         "forward_rms_mps": float(np.sqrt(weights @ v[:, 0]**2)),
                                         "lateral_rms_mps": float(np.sqrt(weights @ v[:, 1]**2)),
                                         "planar_rms_mps": float(np.sqrt(weights @ np.sum(v[:, :2]**2, axis=1)))}
                world_base = np.asarray([c["free_base_velocity_world_mps"] for c in contacts])
                world_leg = np.asarray([c["leg_dof_velocity_world_mps"] for c in contacts])
                world_wheel = np.asarray([c["wheel_dof_velocity_world_mps"] for c in contacts])
                world_robot = np.asarray([c["robot_point_velocity_world_mps"] for c in contacts])
                res = float(np.max(np.abs(world_base+world_leg+world_wheel-world_robot)))
                decomposition_residuals.append(res)
                assert res < 1e-12
                omega = {key: value["yaw_fit_rps"] for key, value in components.items()}
                assert abs(omega["base"]+omega["leg"]+omega["wheel"]-omega["terrain"]-omega["slip"]) < 1e-12
                assert abs(omega["terrain"]) < 1e-12
                force = np.asarray([c["force_world_n"] for c in contacts]) @ rotation.T
                direct_moment = np.asarray([c["torque_world_nm"] for c in contacts])[:, 2]
                longitudinal_moment = float(np.sum(-offsets[:, 1] * force[:, 0]))
                lateral_moment = float(np.sum(offsets[:, 0] * force[:, 1]))
                moment = longitudinal_moment + lateral_moment + float(direct_moment.sum())
                assert abs(moment-cinfo["total_wrench_world_6"][5]) < 1e-10
                for c in contacts:
                    fn = c["normal_force_n"]
                    if fn > 1e-6:
                        friction_ratios.append(c["tangential_force_magnitude_n"]/(.9*fn))
                    contact_powers.append(float(np.dot(c["force_world_n"], c["tangential_relative_velocity_world_mps"])))
                gap = -omega["wheel"]-omega["base"]
                endpoint_rows.append({"endpoint_tick": k+1, "contact_count": len(contacts),
                    "active_wheels": len(active), "components": components,
                    "body_yaw_rate_after_rps": row["body_yaw_rate_after_rps"],
                    "wheel_rolling_fit_same_endpoint_rps": diagnostic[k+1]["wheel_fit_yaw_before_rps"],
                    "contact_wheel_spin_yaw_rps": -omega["wheel"],
                    "wheel_spin_minus_base_fit_gap_rps": gap,
                    "gap_leg_contribution_rps": omega["leg"],
                    "gap_slip_contribution_rps": -omega["slip"],
                    "yaw_moment_longitudinal_force_nm": longitudinal_moment,
                    "yaw_moment_lateral_force_nm": lateral_moment,
                    "yaw_moment_total_nm": moment})
            endpoint = lambda field: [r[field] for r in endpoint_rows]
            gap = float(np.mean(endpoint("wheel_spin_minus_base_fit_gap_rps")))
            leg = float(np.mean(endpoint("gap_leg_contribution_rps")))
            slip = float(np.mean(endpoint("gap_slip_contribution_rps")))
            assert abs(gap-leg-slip) < 1e-12
            limits = np.tile([80., 80., 80., 12.], 4)
            request_delta = torque_applied-torque_request[:, None, :]
            peak_row = max(trace, key=lambda r: abs(r["heading_error_rad"]))
            summary = {"side": side, "condition": condition, "candidate_passed": False,
                "peak_heading_error_rad": abs(peak_row["heading_error_rad"]),
                "peak_endpoint_tick": peak_row["endpoint_tick"],
                "fit_valid_endpoint_count": len(endpoint_rows),
                "fit_excluded_endpoint_ticks": excluded_fit_endpoints,
                "fit_exclusion_rule": "fewer than2 active wheels or weighted lateral variance <=1e-4m^2; no missing sample is zero-filled",
                "fit_statistic_window": "only the listed valid endpoint rows; contact moment and power below share this window",
                "endpoint_mean_body_yaw_rps": float(np.mean(endpoint("body_yaw_rate_after_rps"))),
                "endpoint_mean_rolling_fit_yaw_rps": float(np.mean(endpoint("wheel_rolling_fit_same_endpoint_rps"))),
                "endpoint_mean_contact_wheel_spin_yaw_rps": float(np.mean(endpoint("contact_wheel_spin_yaw_rps"))),
                "signed_mean_gap_rps": sign*gap,
                "signed_mean_gap_leg_contribution_rps": sign*leg,
                "signed_mean_gap_slip_contribution_rps": sign*slip,
                "signed_mean_gap_leg_fraction": leg/gap,
                "signed_mean_gap_slip_fraction": slip/gap,
                "fit_component_mean_yaw_rps": {key: float(np.mean([r["components"][key]["yaw_fit_rps"] for r in endpoint_rows])) for key in ("base", "leg", "wheel", "slip")},
                "component_window_rms_mps": {key: {axis: float(np.sqrt(np.mean([r["components"][key][axis]**2 for r in endpoint_rows]))) for axis in ("forward_rms_mps", "lateral_rms_mps", "planar_rms_mps")} for key in ("base", "leg", "wheel", "slip")},
                "yaw_moment_signed_mean_nm": {label: sign*float(np.mean(endpoint(field))) for label,field in (("longitudinal", "yaw_moment_longitudinal_force_nm"), ("lateral", "yaw_moment_lateral_force_nm"), ("net", "yaw_moment_total_nm"))},
                "all_episode_requested_wheel_torque_max_abs_nm": float(np.max(np.abs(torque_request[:, [3,7,11,15]]))),
                "all_episode_applied_wheel_torque_max_abs_nm": float(np.max(np.abs(torque_applied[:, :, [3,7,11,15]]))),
                "all_episode_actuator_request_applied_max_abs_difference_nm": float(np.max(np.abs(request_delta))),
                "all_episode_applied_axis_torque_limit_occupancy": float(np.mean(np.abs(torque_applied) >= limits[None,None,:]-1e-9)),
                "all_episode_wheel_integral_abs_max_nm": float(np.max(np.abs(integral))),
                "contact_friction_ratio_ft_over_0p9fn": stats(friction_ratios),
                "contact_friction_ratio_fraction_ge0p98": float(np.mean(np.asarray(friction_ratios)>=.98)),
                "contact_normal_horizontal_component_abs_max": max(normal_horizontal),
                "endpoint_contact_tangent_power_sum_mean_w": float(np.sum(contact_powers)/len(endpoint_rows)),
                "max_full_jacobian_velocity_decomposition_residual_mps": max(decomposition_residuals),
                "endpoint_rows": endpoint_rows}
            result["cases"][f"{side}_{condition}"] = summary
            print(json.dumps({k:v for k,v in summary.items() if k != "endpoint_rows"}, allow_nan=False))
    assert all(sha(Path(path)) == value for path,value in inputs.items())
    result["source_sha256"] = inputs
    result["inputs_unchanged"] = True
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
