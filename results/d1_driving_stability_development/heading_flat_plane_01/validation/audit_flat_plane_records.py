"""Independent file-only recomputation of plane scores, forces, timing, and budgets."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


def load_rows(path):
    with gzip.open(path, "rt") as stream:
        return [json.loads(line) for line in stream]


def same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def near(a, b, tol=1e-10):
    np.testing.assert_allclose(a, b, rtol=0, atol=tol)


def expected_raw(case, tick):
    c = case["command"]
    forward = c["forward_target_mps"]*np.clip((tick-c["settle_ticks"])/c["ramp_ticks"], 0., 1.)
    if c["stop_tick"] is not None and tick >= c["stop_tick"]:
        forward = 0.
    p = c["yaw_pulse"]
    yaw = p["user_yaw_rate_rps"] if p and p["start_tick"] <= tick < p["end_tick_exclusive"] else 0.
    return {"forward_velocity_mps": float(forward), "yaw_rate_rps": yaw, "clearance_m": c["height_m"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    protocol = json.loads((args.input/"protocol.json").read_text())
    completed = (args.input/"summary.json").exists()
    if completed:
        aggregate = json.loads((args.input/"summary.json").read_text())
        cases = protocol["cases"]
    else:
        failure = json.loads((args.input/"batch_failure.json").read_text())
        aggregate = {"actual_control_transitions": failure["completed_control_intervals_from_clock"],
                     "actual_physics_substeps": failure["actual_physics_substeps_from_clock"]}
        cases = [c for c in protocol["cases"] if (args.input/c["name"]/"plane_summary.json").exists()]
        assert cases == protocol["cases"][:2] and aggregate["actual_control_transitions"] == 1600
    reports, total = [], 0
    for case in cases:
        folder = args.input/case["name"]
        rows = load_rows(folder/"trace.jsonl.gz")
        diag = load_rows(folder/"plane_diagnostics.jsonl.gz")
        native = load_rows(folder/"native_physics_entries.jsonl.gz")
        summary = json.loads((folder/"plane_summary.json").read_text())
        n = len(rows)
        assert n == len(diag) and len(native) == n*5
        assert n <= case["max_transitions"]
        with np.load(folder/"states.npz") as z:
            states = {k: z[k].copy() for k in z.files}
        with np.load(folder/"execution_states.npz") as z:
            assert len(z["qpos"]) == n+2
            assert same(z["qpos"][:-1], states["qpos"])
            assert same(z["qvel"][:-1], states["qvel"])
            near(z["time_s"][:-1], np.arange(n+1)*.01)
            assert same(z["qpos"][-1], z["qpos"][-2])
        assert all(len(states[k]) == n+1 for k in ("qpos", "qvel", "observations", "truth_positions_world_m"))
        assert states["requested_actions"].shape == states["applied_actions"].shape == (n, 8)
        assert not np.any(states["requested_actions"]) and not np.any(states["applied_actions"])
        raw_errors, heading_errors, normals = [], [], []
        for k, (row, d) in enumerate(zip(rows, diag)):
            assert row["tick"] == d["tick"] == k and row["endpoint_tick"] == k+1
            raw = expected_raw(case, k)
            assert d["raw_user_command"] == row["heading_task"]["user_command_before"] == raw
            raw_errors.append(row["body_forward_mps"]-raw["forward_velocity_mps"])
            near(row["user_forward_error_mps"], raw_errors[-1])
            heading_errors.append(row["heading_error_rad"])
            for offset, e in enumerate(native[k*5:(k+1)*5]):
                assert e["returned"] and e["contacts"]["sampling"] == "native_step_solved_cache_not_synchronized_endpoint"
                near(e["start_time_s"], (k*5+offset)*.002)
                near(e["actual_dt_s"], .002)
                near(e["ctrl_nm"], d["applied_torque_nm"][offset], tol=0)
                pulse = case["external_wrench"]
                expected = np.zeros(6)
                if pulse and pulse["start_tick"] <= k < pulse["end_tick_exclusive"]:
                    expected = np.asarray(pulse["force_xyz_n"]+pulse["torque_xyz_nm"])
                near(e["wrench_world"], expected, tol=0)
                near(e["wrench_world"], row["physics_wrench_samples"][offset]["wrench_world"], tol=0)
                c = e["contacts"]
                normals.append(c["max_horizontal_normal"])
                assert c["max_horizontal_normal"] <= 1e-12 and c["max_vertical_normal_error"] <= 1e-12
                force, moment, normal_sum, tangent_sum = (np.zeros(3) for _ in range(4))
                counts = [0]*4
                for p in c["contacts"]:
                    counts[p["wheel"]] += 1
                    f = np.asarray(p["force_world_n"])
                    normal = np.asarray(p["normal_terrain_to_robot_world"])
                    assert np.linalg.norm(normal[:2]) <= 1e-12
                    near(p["normal_force_world_n"], np.dot(f, normal)*normal)
                    near(f, np.asarray(p["normal_force_world_n"])+p["tangent_force_world_n"])
                    force += f
                    moment += np.cross(np.asarray(p["pos_world_m"])-c["reference_world_m"], f)+p["torque_world_nm"]
                    normal_sum += p["normal_force_world_n"]
                    tangent_sum += p["tangent_force_world_n"]
                assert counts == c["active_contacts_by_wheel"]
                near(c["total_wrench_world_6"], np.r_[force, moment])
                near(c["summed_normal_force_world_n"], normal_sum)
                near(c["summed_tangent_force_world_n"], tangent_sum)
        rmse = float(np.sqrt(np.mean(np.square(raw_errors))))
        near(rmse, summary["velocity_rmse_mps"])
        near(max(abs(v) for v in heading_errors), summary["heading_peak_rad"])
        impulse = sum(e["wrench_world"][5]*e["actual_dt_s"] for e in native)
        near(impulse, summary["actual_signed_yaw_impulse_nms"])
        item = {"case": case["name"], "transitions": n, "physics_substeps": len(native),
                "velocity_rmse_mps": rmse, "signed_impulse_nms": impulse,
                "max_horizontal_contact_normal": max(normals)}
        if case["command"]["stop_tick"] is not None:
            speed = max(abs(r["body_forward_mps"]) for r in rows if 500 <= r["endpoint_tick"] < 800)
            pos = states["truth_positions_world_m"]
            path = float(np.linalg.norm(np.diff(pos[500:701, :2], axis=0), axis=1).sum())
            near(speed, summary["late_stop_max_abs_body_vx_mps"])
            near(path, summary["late_stop_cumulative_planar_path_m"])
            assert summary["gates"]["checks"]["late_stop_speed"] == (speed <= .03)
            assert summary["gates"]["checks"]["late_stop_planar_path"] == (path <= .05)
            item.update(late_peak_speed_mps=speed, late_path_m=path)
        reports.append(item)
        total += n
    assert total == aggregate["actual_control_transitions"] <= 5600
    assert total*5 == aggregate["actual_physics_substeps"] <= 28000
    root = Path('/home/lyh/wheel-legged-control-lab')
    frozen = json.loads((root/'results/d1_budget_study/protocol.json').read_text())["source_sha256"]
    assert len(frozen) == 77
    assert all(hashlib.sha256((root/p).read_bytes()).hexdigest() == h for p,h in frozen.items())
    result = {"passed": True, "full_batch_completed": completed, "file_only_new_physics_steps": 0, "episodes": reports,
              "actual_control_transitions": total, "actual_physics_substeps": total*5,
              "frozen77_unchanged": True, "task_gate_success_is_separate": True}
    with args.output.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
