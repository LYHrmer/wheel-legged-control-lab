"""Read back round-86 ramp gates and synthetic PPO updates without a physics or ML runtime."""

import argparse
import json
import math
import struct
import sys
from pathlib import Path

if __package__:
    from . import audit_terrain_preview85 as prev
else:  # Also usable with python -IS when ROS has installed a package named scripts.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import audit_terrain_preview85 as prev

DEV = (8517, 8529, 8543)
FRESH = (8607, 8619, 8631)
FIXTURE = (8603, 8611, 8623)
MODEL_SHA256 = "c2637a4988d1530fdcc5bf81883b3d86d120caf4071d531c48726e5a5b5be66b"


def surface(x, sign, toe):
    """Interpolate the two float32 ramp-map vertices that enclose the recorded pose."""
    coordinate = (x + 6.0) / .01
    index = min(math.floor(coordinate), 1199)
    prev.check(0 <= index < 1200, "body outside the finite ramp")
    vertices = []
    for i in (index, index + 1):
        d = -6.0 + i * .01 - toe
        u = min(1., max(0., d / .12))
        h = sign * math.tan(math.radians(4)) * (.12 * (u**3 - .5 * u**4) + max(d - .12, 0.))
        vertices.append(struct.unpack("<f", struct.pack("<f", (h + .7) / 1.4))[0])
    difference = vertices[1] - vertices[0]
    return -.7 + 1.4 * (vertices[0] + (coordinate - index) * difference), 1.4 * difference / .01


def verify_tick(row, step, config, direction):
    x, y, z, w, i, j, k = row["qpos"][:7]
    roll = math.atan2(2 * (w*i + j*k), 1 - 2 * (i*i + j*j))
    pitch = math.asin(max(-1., min(1., 2 * (w*j - k*i))))
    yaw = math.atan2(2 * (w*k + i*j), 1 - 2 * (j*j + k*k))
    height, slope = surface(x, direction, config["toe_x_m"])
    grade = -math.atan(slope * math.cos(yaw))
    fallen = z - height < .22 or max(abs(roll), abs(pitch)) > .85
    reason = "fallen" if fallen else ("out_of_bounds" if max(abs(x), abs(y)) > 5.5 else None)
    now = (step - 1) * .01
    est = max(0., now - .02)
    prev.fields(row, {"step": step, "time_s": step * .01, "control_s": now, "estimator_time_s": est,
                      "estimator_age_s": now - est, "truth_pitch_rad": pitch, "truth_roll_rad": roll,
                      "ground_pitch_rad": grade, "base_x_m": x, "termination_reason": reason,
                      "pitch_error_rad": pitch - config["operating_state"][1] - grade}, "tick")
    error = (row["lqr_distance_m"] - row["lqr_reference_distance_m"],
             row["estimated_pitch_rad"] - (config["operating_state"][1] + row["command_pitch_rad"]),
             row["estimated_speed_mps"] - row["command_speed_mps"], row["estimated_pitch_rate_rps"])
    raw_fx = -math.fsum(a * b for a, b in zip(error, config["gain"][0], strict=True))
    fx = max(-180., min(raw_fx, 180.))
    applied, limits = row["applied_torque_nm"], config["joint_torque_limits_nm"]
    prev.check(len(limits) == 16 and all(math.isfinite(v) and v > 0 for v in limits), "axis limits")
    violations = saturation = 0
    for requested, actual, limit in zip(row["requested_torque_nm"], applied, limits, strict=True):
        prev.check(math.isfinite(requested) and math.isfinite(actual), "nonfinite torque")
        prev.equal(actual, max(-limit, min(requested, limit)), "applied axis torque")
        violations += max(abs(requested), abs(actual)) > limit + 1e-9
        saturation += abs(actual) >= limit - 1e-9
    prev.equal(sum(applied[3::4]) / .087, fx, "wheel torques versus limited Fx")
    prev.fields(row, {"baseline_force_n": raw_fx, "applied_longitudinal_force_n": fx,
                      "torque_bound_violations": violations, "saturated_axes": saturation,
                      "speed_error_mps": row["truth_speed_mps"] - row["command_speed_mps"],
                      "torque_mean_square_nm2": math.fsum(v*v for v in applied) / 16}, "force")


def route(path, terrain, seed, damped):
    config = prev.read(path / "configuration.json")
    metrics = prev.read(path / "metrics.json")
    windows = prev.read(path / "sensor_windows.json")
    rows, frames = (prev.lines(path / name) for name in ("ticks.jsonl", "captures.jsonl"))
    prev.fields(config, {"toe_x_m": prev.TOE_M, "nearest_beam_deg": 82, "fx_limit_n": 180.,
                         "proprioceptive_delay_s": .02, "range_latency_s": .04, "task_step_limit": 600,
                         "model_sha256": MODEL_SHA256, "input_cost": 1. / 180**2,
                         "damping_enabled": damped}, path.name)
    prev.check(0 < len(rows) <= 600, "route length")
    direction = 1 if terrain == "up" else -1
    for step, row in enumerate(rows, 1):
        verify_tick(row, step, config, direction)
        prev.check(step == len(rows) or row["termination_reason"] is None, "steps after termination")
    terminal = rows[-1]["termination_reason"]
    prev.check(len(rows) == 600 or terminal is not None, "incomplete route with a complete receipt")
    full = len(rows) == 600 and terminal is None
    capture_steps = [math.ceil(i * 100 / 6) for i in range(36) if math.ceil(i * 100 / 6) < len(rows)]
    for seq, (c, idx) in enumerate(zip(frames, capture_steps, strict=True)):
        prev.equal(set(c["mounts"]), {"82"}, "single shadow sensor")
        prev.fields(c, {"sequence": seq, "captured_s": idx * .01, "available_s": idx * .01 + .04,
                        "estimator_s": max(0., idx * .01 - .02),
                        "estimated_roll_rad": rows[idx]["estimated_roll_rad"],
                        "estimated_pitch_rad": rows[idx]["estimated_pitch_rad"]}, "exposure")
    start = prev.first_entry(rows, prev.TOE_M)
    after = rows[start:] if start is not None else []
    contact = after[0]["time_s"] if after else None
    end = next((i for i, r in enumerate(after) if r["front_track_x_m"] >= prev.TOE_M + .4), None)
    grades = prev.grade_samples(rows, frames, "82")
    primary = prev.mount_result(rows, grades, "82", 0, direction, contact)
    chords = prev.mount_result(rows, grades, "82", 2, direction, contact)
    prev.fields(windows["measured"], primary, "primary sensor")
    prev.fields(windows["matched_ground_chords"], chords, "actual ground chord")
    first = windows["first_registered_window"]
    confirmed = False
    if primary["passed"]:
        chosen = [(r, g) for r, g in zip(rows, grades, strict=True)
                  if primary["first_streak_start_s"] <= r["control_s"] <= primary["first_confirmation_s"]]
        prev.equal(len(chosen), 3, "first packet duration")
        past = []
        for (row, (_, _, grade)), detail in zip(chosen, first["ticks"], strict=True):
            actual = grade is not None and direction * grade >= math.radians(.5)
            sample = row["mounts"]["82"]
            prev.equal(sample["sequence"], primary["first_qualifying_packet"], "first packet")
            prev.fields(detail, {"control_s": row["control_s"], "captured_s": sample["captured_s"],
                                 "query_x_m": sample["query_x_m"], "correct_on_both_floor_chords": actual,
                                 "measured_slope_rad": sample["slope_rad"],
                                 "online_pair_start_beam_by_track": sample["online_pair_start_beam_by_track"],
                                 "true_floor_chord_rad_by_track": sample["true_floor_chord_rad_by_track"]}, "first")
            past.append(actual)
        confirmed = all(past)
        prev.fields(first, {"sequence": primary["first_qualifying_packet"], "all_three_confirmed": confirmed}, "first")
    else:
        prev.equal(first, None, "missing first window")
    violations = sum(r["torque_bound_violations"] for r in rows)
    result = {"terrain": terrain, "seed": seed, "normalized_input_cost": True, "task_steps": len(rows),
              "full_horizon": full, "termination_reason": terminal or "time_limit", "first_front_contact_s": contact,
              "crossed_window_s": after[end]["time_s"] if end is not None else None,
              "zero_window_complete": end is not None, "zero_window_pitch_rmse_deg": None if end is None else
              prev.rmse_deg([r["pitch_error_rad"] for r in after[:end + 1]]),
              "max_progress_after_toe_m": max((r["front_track_x_m"] - prev.TOE_M for r in after), default=None),
              "sensor_passed": primary["passed"], "true_chord_passed": chords["passed"],
              "first_sensor_window_corroborated": confirmed, "first_confirmation_s": primary["first_confirmation_s"],
              "fx_limited_ticks": sum(abs(r["applied_longitudinal_force_n"]) >= 180. - 1e-9 for r in rows),
              "torque_bound_violations": violations,
              "speed_rmse_mps": math.sqrt(math.fsum(r["speed_error_mps"]**2 for r in rows) / len(rows)),
              "torque_rms_nm": math.sqrt(math.fsum(r["torque_mean_square_nm2"] for r in rows) / len(rows)),
              "passed": full and end is not None and primary["passed"] and violations == 0}
    prev.fields(metrics, result, path.name)
    return {**result, "precontact_wrong_packets": primary["precontact_wrong_packets"]}


def fixture(path):
    checked = []
    group = prev.read(path / "summary.json")
    for seed, recorded in zip(FIXTURE, group["cases"], strict=True):
        folder = path / f"seed_{seed}"
        rows = prev.lines(folder / "transaction.jsonl")
        summary = prev.read(folder / "summary.json")
        update = prev.read(folder / "ppo_update.json")
        probes = prev.read(folder / "reload_probes.json")
        params = prev.read(folder / "diagnostic_parameters.json")
        prev.equal(summary, recorded, "fixture case receipt")
        prev.check(len(rows) == 128 and [r["segment_tick"] for r in rows] == list(range(1, 65))*2
                   and [r["segment"] for r in rows] == [1]*64 + [2]*64, "PPO fixture segments")
        prev.check(all(not r["terminated"] and r["truncated"] == (r["segment_tick"] == 64) for r in rows),
                   "fixture truncation")
        prev.fields(summary, {"seed": seed, "status": "COMPLETE", "optimizer_steps": 1, "updates_attempted": 1,
                              "d1_optimizer_steps": 0, "synthetic_optimizer_steps": 1,
                              "recorded_transitions": len(rows), "fixed_probe_reload_exact": True}, "fixture")
        prev.equal(update["optimizer_steps"], 1, "update count")
        seen = set()
        for change in update["parameters"]:
            role, field = change["parameter"].split(".")
            idx = change["flat_index"]
            key = (role, field, idx)
            prev.check(key not in seen, "duplicate parameter")
            seen.add(key)
            saved = params[role][field] if field == "bias" else params[role][field][idx]
            delta = -.0003 * change["gradient"]
            prev.fields(change, {"before": 0., "after": saved, "delta": delta}, str(key))
            prev.equal(saved, delta, "plain one-step SGD")
        size = len(rows[0]["observation"])
        prev.equal(seen, {(r, f, i) for r in ("actor", "critic") for f, n in (("weight", size), ("bias", 1))
                          for i in range(n)}, "all saved parameters")
        prev.check(any(params["actor"]["weight"]), "actor mean-head update")
        prev.equal(probes["zero_based_rows"], [0, 63, 64, 127], "probe indices")
        for role, key in (("actor", "mean"), ("critic", "value")):
            prev.equal(probes[key], probes["reloaded_" + key], role + " exact reload")
            for idx, values, prediction in zip(probes["zero_based_rows"], probes["observations"], probes[key], strict=True):
                prev.equal(values, rows[idx]["observation"], "probe inputs")
                dot = math.fsum(x * w for x, w in zip(values, params[role]["weight"], strict=True)) + params[role]["bias"]
                prev.equal(prediction[0] if key == "mean" else prediction, dot, role + " readback")
        checked.append({"seed": seed, "optimizer_steps": 1, "transitions": len(rows), "reload_probes": 4})
    prev.fields(group, {"status": "COMPLETE", "passed": True, "synthetic_optimizer_steps": 3,
                        "synthetic_transitions": 384, "d1_optimizer_steps": 0, "d1_controls": 0}, "fixture group")
    return {"status": "VERIFIED", "cases": checked, "synthetic_optimizer_steps": 3, "d1_optimizer_steps": 0}


def audit(root, report):
    report["stages"] = {}
    errors = []
    qualified = None
    for name in ("development", "fallback", "fresh", "fixture"):
        path = root / name
        if not path.exists():
            report["stages"][name] = {"status": "NOT_RUN"}
            continue
        if (path / "partial.json").exists() or not (path / "summary.json").exists():
            unfinished = {}
            pattern = "*/transaction.jsonl" if name == "fixture" else "*/ticks.jsonl"
            for raw in path.glob(pattern):
                with raw.open(encoding="utf-8") as stream:
                    unfinished[raw.parent.name] = sum(1 for _ in stream)
            report["stages"][name] = {"status": "PARTIAL", "retained_tick_counts": unfinished}
            errors.append(f"{name} has an unfinished receipt")
            continue
        report["stages"][name] = {"status": "CHECKING", "cases": []}
        if name == "fixture":
            report["stages"][name] = fixture(path)
            continue
        receipt = prev.read(path / "summary.json")
        prev.fields(receipt, {"stage": name, "status": "COMPLETE", "d1_optimizer_steps": 0}, name)
        if name == "fallback":
            prev.check(report["stages"]["development"].get("passed") is False, "fallback admission")
        expected = [(t, s) for t in ("up", "down") for s in (FRESH if name == "fresh" else DEV)]
        found = {p.name for p in path.iterdir() if p.is_dir()}
        if not receipt["cases"]:
            prev.check(not found and name != "development" and not receipt["passed"], "invalid skipped stage")
            if name == "fallback":
                prev.equal(prev.read(path / "fit.json")["passed"], False, "fallback fit gate")
            else:
                prev.check(qualified is None, "fresh skipped despite completed development")
            prev.fields(receipt, {"controls": 0, "native_steps": 0}, name)
            report["stages"][name] = {"status": "SKIPPED", "reason": receipt["reason"]}
            continue
        prev.check((name != "fallback" or qualified is None) and (name != "fresh" or qualified is not None),
                   "stage ran without its registration condition")
        prev.equal(found, {f"{t}_{s}" for t, s in expected}, "expected six cases")
        cases = report["stages"][name]["cases"]
        for (terrain, seed), recorded in zip(expected, receipt["cases"], strict=True):
            case_dir = path / f"{terrain}_{seed}"
            prev.equal(recorded, prev.read(case_dir / "metrics.json"), case_dir.name)
            cases.append(route(case_dir, terrain, seed, name == "fallback" or qualified == "fallback"))
        passed = all(c["passed"] and c["first_sensor_window_corroborated"] for c in cases)
        ticks = sum(c["task_steps"] for c in cases)
        prev.fields(receipt, {"passed": passed, "controls": ticks, "native_steps": 5*ticks}, name)
        if name == "fresh":
            prev.equal(receipt["selected"], qualified, "qualification source")
        elif passed:
            qualified = name
        report["stages"][name] = {"status": "VERIFIED", "passed": passed, "controls": ticks, "cases": cases}
    prev.check(any(s["status"] != "NOT_RUN" for s in report["stages"].values()), "no recorded stages")
    prev.check(not errors, "; ".join(errors))
    report["contact_evidence"] = "Recorded maximum x of a positively loaded front-wheel/floor contact. Individual solver wrenches were not logged."
    report["auditor_native_steps"] = report["auditor_optimizer_steps"] = 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"status": "PARTIAL"}
    with args.output.open("x", encoding="utf-8") as out:
        try:
            audit(args.root, report)
            report["status"] = "COMPLETE"
        except Exception as exc:
            report["error"] = repr(exc)
            raise
        finally:
            json.dump(report, out, indent=2, allow_nan=False)
            out.write("\n")


if __name__ == "__main__":
    main()
