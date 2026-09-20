"""Independent saved-state algebra for a yaw-error measurement replacement.

No controller imports/calls, MuJoCo, state propagation, memory recurrence,
integration, gain choice or contact-force/velocity energy mixing.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    with gzip.open(path, "rt") as stream:
        return [json.loads(line) for line in stream]


def fit(y, linear):
    centered = y-y.mean()
    return float(-centered@linear/(centered@centered))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--diagnosis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    hashes = {str(Path(__file__).resolve()): sha(__file__), str(args.diagnosis.resolve()): sha(args.diagnosis)}
    diagnosis = json.loads(args.diagnosis.read_text())
    for path, expected in diagnosis["input_sha256"].items():
        assert sha(path) == expected
        hashes[path] = expected
    cases = []
    for source, directory in (("baseline", "flat_plane_02"), ("center_candidate", "plane_turn_center_01")):
        for side, sign in (("left", 1), ("right", -1)):
            folder = args.work / directory / f"stationary_turn_{side}_hold"
            diag = rows(folder / "plane_diagnostics.jsonl.gz")
            compensation = rows(folder / "turn_compensation.jsonl.gz") if source == "center_candidate" else None
            reconstructed = next(c for c in diagnosis["cases"] if c["source"] == source and c["turn_sign"] == sign)
            samples = []
            for tick in range(200, 250):
                d = diag[tick]
                assert d["raw_user_command"]["forward_velocity_mps"] == 0.
                assert d["raw_user_command"]["yaw_rate_rps"] == sign*.6
                effective = d["effective_yaw_request_rps"]
                assert effective == sign*.6
                original_target = np.array(d["wheel_target_rad_s"])
                original_error = np.array(d["wheel_error_before_rad_s"])
                wheel_velocity = original_target-original_error
                # Active old targets are far inside +/-30; reconstruct only the
                # original lateral geometry, never infer a new dynamic state.
                base_unclipped = np.array(compensation[tick]["record"]["base_unclipped_rad_s"]) if compensation else original_target
                assert np.max(np.abs(base_unclipped)) < 30.
                lateral = -.087*base_unclipped/effective
                centered = lateral-lateral.mean()
                spread = float(centered@centered)
                assert spread > 1e-4
                body_rate = d["body_yaw_rate_before_rps"]
                wheel_fit = fit(lateral, .087*wheel_velocity)
                reference = next(v for v in reconstructed["decisions"] if v["execution_tick"] == tick)
                assert abs(body_rate-reference["body_yaw_rate_rps"]) < 1e-12
                expected_wheel_fit = reference["wheel_target_yaw_fit_rps"]-reference["wheel_yaw_tracking_error_rps"]
                assert abs(wheel_fit-expected_wheel_fit) < 1e-12
                correction = centered*(body_rate-wheel_fit)/.087
                alternative_unclipped = base_unclipped+correction
                alternative_target = np.clip(alternative_unclipped, -30., 30.)
                error = alternative_target-wheel_velocity
                projection_error = fit(lateral, .087*error)
                identity_residual = projection_error-(effective-body_rate)
                # Each sample restarts from that recorded trajectory's own old
                # memory. No alternative integral is used in the next sample.
                original_memory = np.array(d["wheel_integral_before_nm"])
                trial_memory_unclipped = original_memory+3.*.01*error
                trial_memory = np.clip(trial_memory_unclipped, -4., 4.)
                trial_request = 2.2*error+trial_memory
                accepted = (np.abs(trial_request) <= 12.) | (trial_request*error < 0)
                final_memory = np.where(accepted, trial_memory, original_memory)
                request = 2.2*error+final_memory
                # A body-fixed differential wheel-speed variation is exactly
                # cancelled by the new velocity target's direct wheel term.
                wheel_mode = centered/.087
                d_wheel_fit = fit(lateral, .087*wheel_mode)
                target_mode_derivative = centered*(0.-d_wheel_fit)/.087
                wheel_error_mode_derivative = target_mode_derivative-wheel_mode
                samples.append({"execution_tick": tick, "lateral_m": lateral.tolist(),
                    "body_yaw_rate_rps": body_rate, "center_wheel_yaw_fit_rps": wheel_fit,
                    "effective_yaw_request_rps": effective, "raw_body_error_rps": effective-body_rate,
                    "replacement_correction_rad_s": correction.tolist(),
                    "original_nominal_unclipped_rad_s": base_unclipped.tolist(),
                    "replacement_target_rad_s": alternative_target.tolist(),
                    "replacement_target_yaw_fit_rps": fit(lateral, .087*alternative_target),
                    "wheel_error_projection_rps": projection_error,
                    "body_error_identity_residual_rps": identity_residual,
                    "common_target_increment_rad_s": float(correction.mean()),
                    "one_sample_requested_wheel_torque_nm": request.tolist(),
                    "one_sample_integral_nm": final_memory.tolist(),
                    "target_clip": bool(np.any(alternative_unclipped != alternative_target)),
                    "integral_clip": bool(np.any(trial_memory != trial_memory_unclipped)),
                    "integral_reject": bool(np.any(~accepted)),
                    "torque_over_12": bool(np.any(np.abs(request)>12.)),
                    "differential_free_wheel_error_derivative_norm": float(np.linalg.norm(wheel_error_mode_derivative))})
            cases.append({"source": source, "side": side, "independent_samples": len(samples),
                "max_abs_body_error_identity_residual_rps": max(abs(s["body_error_identity_residual_rps"]) for s in samples),
                "max_abs_common_target_increment_rad_s": max(abs(s["common_target_increment_rad_s"]) for s in samples),
                "max_abs_target_rad_s": max(abs(v) for s in samples for v in s["replacement_target_rad_s"]),
                "max_abs_wheel_request_nm": max(abs(v) for s in samples for v in s["one_sample_requested_wheel_torque_nm"]),
                "max_abs_integral_nm": max(abs(v) for s in samples for v in s["one_sample_integral_nm"]),
                "max_abs_target_equivalent_yaw_rps": max(abs(s["replacement_target_yaw_fit_rps"]) for s in samples),
                "target_clip_count": sum(s["target_clip"] for s in samples),
                "integral_clip_count": sum(s["integral_clip"] for s in samples),
                "integral_reject_count": sum(s["integral_reject"] for s in samples),
                "torque_over_12_count": sum(s["torque_over_12"] for s in samples),
                "max_differential_free_wheel_error_derivative_norm": max(s["differential_free_wheel_error_derivative_norm"] for s in samples),
                "samples": samples})
    assert all(sha(p)==h for p,h in hashes.items())
    report = {"schema": "d1-body-yaw-error-replacement-algebra-qualification-v1",
        "new_physics_steps": 0, "controller_compute_calls": 0, "new_contact_solves": 0,
        "independent_saved_state_samples": 200, "input_sha256": hashes, "inputs_unchanged": True,
        "formula": "y0=y-mean(y); rw=-R*y0.dot(omega)/(y0.dot(y0)); omega_target=(v-reff*y)/R+y0*(rbody-rw)/R; final original +/-30 clip",
        "identity_when_unclipped": "-R*y0.dot(omega_target-omega)/(y0.dot(y0)) == reff-rbody; correction common mode is zero",
        "gain_origin": "no new numeric feedback gain: unit coefficient follows exact yaw-error projection; original Kp=2.2 Ki=3 dt=.01 R=.087 preserved",
        "critical_limitation": "This replacement cancels the original wheel PI's proportional/instant integral response to the pure differential wheel-spin velocity mode at fixed body yaw rate. Its unloaded/weak-contact dynamics must be qualified before any physical candidate is authorized; this is not a passivity or closed-loop stability proof.",
        "cases": cases}
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"sha256": sha(args.output), "cases": [{k:v for k,v in c.items() if k != "samples"} for c in cases]}))


if __name__ == "__main__":
    main()
