"""Recompute the round-85 gates from recorded samples with Python's standard library."""

import argparse
import ast
import hashlib
import json
import math
import struct
from bisect import bisect_right
from pathlib import Path
from zipfile import ZipFile

PLAN = Path(__file__).resolve().parents[1] / "docs/experiments85/terrain_action85.json"
OP_PITCH = 0.003862749186185274  # Archived round-84 legacy model, radians.
TOE_M = 0.569214057025901  # Fixed placement in the round-84 ramp.
ARMS = {"negative": -1, "zero": 0, "positive": 1}
DT = 0.01


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def lines(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def check(condition, message):
    if not condition:
        raise ValueError(message)


def equal(actual, expected, label):
    if isinstance(expected, float):
        matches = type(actual) in (int, float) and math.isclose(
            actual, expected, rel_tol=1e-12, abs_tol=1e-10,
        )
    else:
        matches = actual == expected
    check(matches, f"{label}: recorded {actual!r}, recomputed {expected!r}")


def fields(recorded, recomputed, label):
    for key, value in recomputed.items():
        equal(recorded[key], value, f"{label}/{key}")


def bracket(xs, zs, query, mask):
    indices = []
    for i in range(3):
        if not (mask[i] and mask[i + 1]):
            continue
        if any(v is None for v in (xs[i], xs[i + 1], zs[i], zs[i + 1])):
            continue
        if xs[i] <= query < xs[i + 1] and xs[i + 1] - xs[i] >= 0.035:
            indices.append(i)
    return indices[0] if len(indices) == 1 else None


def slopes(x, z, query, hit_ids, floor=None):
    grades = []
    for xs, zs, ids in zip(x, z, hit_ids, strict=True):
        mask = [floor is None or ident == floor for ident in ids]
        i = bracket(xs, zs, query, mask)
        grades.append(None if i is None else math.atan2(zs[i + 1] - zs[i], xs[i + 1] - xs[i]))
    covered = all(g is not None for g in grades)
    valid = covered and max(map(abs, grades)) <= math.radians(12) and (
        abs(grades[1] - grades[0]) <= math.radians(3)
    )
    return covered, (sum(grades) / 2 if valid else None)


def raw_pitch(r, label):
    value = r["truth_pitch_rad"] - OP_PITCH - r["ground_pitch_rad"]
    equal(r["pitch_error_rad"], value, label + "/pitch error")
    return value


def first_entry(rows, toe):
    return next((i for i, r in enumerate(rows)
                 if r["front_contact_x_m"] is not None and r["front_contact_x_m"] >= toe), None)


def rmse_deg(values):
    return math.degrees(math.sqrt(math.fsum(v * v for v in values) / len(values)))


def capture_geometry(capture, angle):
    """Rebuild optical endpoints using raw slant ranges, beam poses and both attitudes."""
    mount = capture["mounts"][angle]
    truth = capture["score_only_truth"]
    rot, base = truth["body_rotation"], truth["base_xyz_world_m"]
    yaw = math.atan2(rot[1][0], rot[0][0])
    heading = (math.cos(yaw), math.sin(yaw), 0.0)
    for value, expected in zip(truth["forward_heading_world"], heading, strict=True):
        equal(value, expected, angle + "/true forward")
    points = truth["mounts"][angle]
    r, p = capture["estimated_roll_rad"], capture["estimated_pitch_rad"]
    floor_tracks = []
    for lane, y in enumerate((0.19, -0.19)):
        floor_track = []
        for i, beam in enumerate((int(angle), 56, 43, 33)):
            d = mount["raw_range_m"][lane][i]
            geom = mount["first_hit_geom"][lane][i]
            a = math.radians(beam)
            body = (0.355 + d * math.cos(a), y, 0.055 - d * math.sin(a))
            rolled_z = math.sin(r) * y + math.cos(r) * body[2]
            ok = 0.14 <= d <= 1.45
            expected_x = math.cos(p) * body[0] + math.sin(p) * rolled_z if ok else None
            expected_z = -math.sin(p) * body[0] + math.cos(p) * rolled_z if ok else None
            equal(mount["local_x_m"][lane][i], expected_x, angle + "/estimated beam x")
            equal(mount["local_z_m"][lane][i], expected_z, angle + "/estimated beam z")
            relative = [math.fsum(a * b for a, b in zip(axis, body, strict=True)) for axis in rot]
            world = [b + q for b, q in zip(base, relative, strict=True)]
            observed = points["first_hit_world_xyz_m"][lane][i]
            if d < 0:
                equal(observed, None, angle + "/missing return")
            else:
                for value, expected in zip(observed, world, strict=True):
                    equal(value, expected, angle + "/world endpoint")
            floor_x = floor_z = None
            if ok and geom == mount["floor_geom_id"]:
                floor_x = math.fsum(v * h for v, h in zip(relative, heading, strict=True))
                floor_z = world[2]
            equal(points["in_range_floor_forward_from_base_m"][lane][i], floor_x, angle + "/floor x")
            equal(points["in_range_floor_world_z_m"][lane][i], floor_z, angle + "/floor z")
            floor_track.append((floor_x, floor_z))
        floor_tracks.append(floor_track)
    return floor_tracks


def grade_samples(rows, captures, angle):
    delivery = [c["available_s"] for c in captures]
    endpoints = [capture_geometry(c, angle) for c in captures]
    grades = []
    for row in rows:
        now = row["control_s"]
        sample = row["mounts"][angle]
        choice = bisect_right(delivery, now + 1e-12) - 1
        equal(sample["sequence"], choice if choice >= 0 else None, angle + "/delivered sequence")
        if choice < 0:
            fields(sample, {"slope_rad": None, "ground_slope_rad": None, "corroborated_slope_rad": None,
                            "reason": "waiting", "ground_reason": "waiting"}, angle)
            grades.append((None, None, None))
            continue
        capture = captures[choice]
        for key in ("captured_s", "available_s"):
            equal(sample[key], capture[key], angle + "/" + key)
        check(now - capture["captured_s"] <= 0.25 + 1e-12, angle + "/expired packet")
        mount = capture["mounts"][angle]
        x, z, ids = mount["local_x_m"], mount["local_z_m"], mount["first_hit_geom"]
        _, grade = slopes(x, z, sample["query_x_m"], ids)
        _, ground = slopes(x, z, sample["query_x_m"], ids, mount["floor_geom_id"])
        equal(sample["slope_rad"], grade, angle + "/slope")
        equal(sample["ground_slope_rad"], ground, angle + "/ground slope")
        equal(sample["reason"] == "valid", grade is not None, angle + "/valid")
        equal(sample["ground_reason"] == "valid", ground is not None, angle + "/ground valid")
        chords = []
        for lane in range(2):
            i = bracket(x[lane], z[lane], sample["query_x_m"], [True] * 4)
            equal(sample["online_pair_start_beam_by_track"][lane], i, angle + "/selected pair")
            chord = None
            if i is not None:
                (x1, z1), (x2, z2) = endpoints[choice][lane][i:i + 2]
                if x1 is not None and x2 is not None and x2 > x1:
                    chord = math.atan2(z2 - z1, x2 - x1)
            equal(sample["true_floor_chord_rad_by_track"][lane], chord, angle + "/actual floor chord")
            chords.append(chord)
        matched = None
        if grade is not None and None not in chords:
            lo, hi = min(grade, *chords), max(grade, *chords)
            matched = lo if lo > 0 else (hi if hi < 0 else 0.0)
        equal(sample["corroborated_slope_rad"], matched, angle + "/matched slope")
        grades.append((grade, ground, matched))
    return grades


def mount_result(rows, grades, angle, mode, direction, contact):
    valid, wrong, early_wrong, early_correct, delivered = set(), set(), set(), set(), set()
    first_receipt_valid = set()
    first_valid = first_correct = first_start = first_confirm = first_seq = None
    longest = streak = valid_ticks = wrong_ticks = early_wrong_ticks = 0
    last_seq = None
    for index, (row, values) in enumerate(zip(rows, grades, strict=True)):
        now, choice, grade = row["control_s"], row["mounts"][angle]["sequence"], values[mode]
        if choice is None:
            continue
        if grade is not None:
            valid.add(choice)
            valid_ticks += 1
            if first_valid is None:
                first_valid = now
            if choice not in delivered:
                first_receipt_valid.add(choice)
        delivered.add(choice)
        before = contact is not None and now <= contact - DT + 1e-9
        signed = None if grade is None else direction * grade
        if signed is not None and signed <= -math.radians(0.5):
            wrong.add(choice)
            wrong_ticks += 1
            if before:
                early_wrong.add(choice)
                early_wrong_ticks += 1
        correct = signed is not None and signed >= math.radians(0.5)
        if correct and first_correct is None:
            first_correct = now
        if correct and before:
            early_correct.add(choice)
            streak = streak + 1 if last_seq == choice else 1
            longest = max(streak, longest)
            if streak == 3 and first_confirm is None:
                first_start, first_confirm, first_seq = rows[index - 2]["control_s"], now, choice
        else:
            streak = 0
        last_seq = choice
    return {
        "delivered_packets": len(delivered), "both_tracks_valid_at_first_receipt": len(first_receipt_valid),
        "both_tracks_valid_packet_fraction": len(valid) / len(delivered) if delivered else None,
        "both_tracks_valid_packets_any_tick": len(valid), "valid_ticks": valid_ticks,
        "wrong_direction_packets": len(wrong), "wrong_direction_ticks": wrong_ticks,
        "precontact_wrong_packets": len(early_wrong), "precontact_wrong_ticks": early_wrong_ticks,
        "precontact_correct_packets": len(early_correct), "first_correct_s": first_correct,
        "first_valid_s": first_valid, "first_streak_start_s": first_start,
        "first_confirmation_s": first_confirm, "first_qualifying_packet": first_seq,
        "longest_same_packet_precontact_ticks": longest, "passed": first_confirm is not None,
    }


def sensing_case(path, terrain):
    metrics, rows, captures = read(path / "metrics.json"), lines(path / "ticks.jsonl"), lines(path / "captures.jsonl")
    check(0 < len(rows) <= 600, path.name + "/control budget")
    expected_captures = [k for k in range(600) if math.ceil(k * 100 / 6) < len(rows)]
    equal([c["sequence"] for c in captures], expected_captures, path.name + "/capture sequence")
    for c in captures:
        equal(c["captured_s"], math.ceil(c["sequence"] * 100 / 6) * DT, path.name + "/exposure")
        equal(c["available_s"], c["captured_s"] + 0.04, path.name + "/delivery")
        equal(c["estimator_s"], max(0.0, c["captured_s"] - 0.02), path.name + "/delayed estimator")
    for i, r in enumerate(rows):
        equal(r["step"], i + 1, path.name + "/step")
        equal(r["time_s"], (i + 1) * DT, path.name + "/post-step time")
        equal(r["control_s"], i * DT, path.name + "/pre-step time")
        raw_pitch(r, path.name)
        torques = zip(r["requested_torque_nm"], r["applied_torque_nm"], metrics["joint_torque_limits_nm"], strict=True)
        check(len(metrics["joint_torque_limits_nm"]) == 16, path.name + "/axes")
        violations = 0
        for requested, applied, limit in torques:
            check(limit > 0 and math.isfinite(requested), path.name + "/torque")
            equal(applied, max(-limit, min(requested, limit)), path.name + "/applied torque")
            violations += max(abs(requested), abs(applied)) > limit + 1e-9
        equal(r["torque_bound_violations"], violations, path.name + "/torque violations")
    toe = metrics["toe_x_m"]
    equal(toe, TOE_M, path.name + "/toe")
    start = first_entry(rows, toe)
    contact = None if start is None else rows[start]["time_s"]
    end = next((i for i in range(start, len(rows)) if rows[i]["front_track_x_m"] >= toe + 0.4),
               None) if start is not None else None
    result = {
        "task_steps": len(rows), "first_front_contact_s": contact,
        "crossed_window_s": rows[end]["time_s"] if end is not None else None,
        "zero_window_complete": end is not None,
        "zero_window_pitch_rmse_deg": rmse_deg([raw_pitch(r, path.name) for r in rows[start:end + 1]])
            if end is not None else None,
        "termination_reason": rows[-1]["termination_reason"] or "time_limit",
        "torque_bound_violations": sum(r["torque_bound_violations"] for r in rows),
    }
    check(len(rows) == 600 or rows[-1]["termination_reason"] is not None, path.name + "/partial case")
    fields(metrics, result, path.name)
    keys = ("mounts", "true_ground_only_diagnostic", "matched_ground_chord_diagnostic")
    for key in (*keys, "first_window_on_true_floor"):
        result[key] = {}
    direction = 1 if terrain == "up" else -1
    for angle in ("74", "82"):
        grades = grade_samples(rows, captures, angle)
        for mode, key in enumerate(keys):
            score = mount_result(rows, grades, angle, mode, direction, contact)
            fields(metrics[key][angle], score, path.name + "/" + angle + "/" + key)
            result[key][angle] = score
        first = result["mounts"][angle]
        evidence = metrics["first_registered_window_ground_check"][angle]
        confirmed = None
        if first["passed"]:
            seq = first["first_qualifying_packet"]
            equal(evidence["sequence"], seq, angle + "/first window packet")
            chosen = [i for i, r in enumerate(rows) if first["first_streak_start_s"] <= r["control_s"]
                      <= first["first_confirmation_s"]]
            equal(len(chosen), 3, angle + "/first window duration")
            on_floor = []
            for i, detail in zip(chosen, evidence["ticks"], strict=True):
                sample = rows[i]["mounts"][angle]
                equal(sample["sequence"], seq, angle + "/first window sequence")
                on_floor.append(grades[i][2] is not None and direction * grades[i][2] >= math.radians(0.5))
                fields(detail, {"control_s": rows[i]["control_s"], "captured_s": sample["captured_s"],
                                "query_x_m": sample["query_x_m"], "measured_slope_rad": grades[i][0],
                                "correct_on_both_floor_chords": on_floor[-1]}, angle + "/first window")
            confirmed = all(on_floor)
            equal(evidence["all_three_confirmed"], confirmed, angle + "/three matched chords")
        else:
            equal(evidence, None, angle + "/missing first window")
        result["first_window_on_true_floor"][angle] = confirmed
    floors = {m["floor_geom_id"] for c in captures for m in c["mounts"].values()}
    check(len(floors) == 1, path.name + "/ambiguous ground geometry")
    return result, floors.pop()


def checkpoint(path):
    record = read(path / "checkpoint_python.json")
    python = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    vectors = {}
    with ZipFile(path / "checkpoint_native.npz") as archive:
        for filename in archive.namelist():
            name = Path(filename).stem
            with archive.open(filename) as stream:
                check(stream.read(6) == b"\x93NUMPY", path.name + "/checkpoint format")
                major, _minor = stream.read(2)
                check(major in (1, 2, 3), path.name + "/npy version")
                size = 2 if major == 1 else 4
                header_len = int.from_bytes(stream.read(size), "little")
                header = ast.literal_eval(stream.read(header_len).decode("utf-8"))
                payload = stream.read()
            check(header["descr"] == "<f8" and len(header["shape"]) == 1 and
                  len(payload) == 8 * header["shape"][0], path.name + "/native-state shape")
            check(all(math.isfinite(x) for (x,) in struct.iter_unpack("<d", payload)), path.name + "/native-state precision")
            vectors[name] = hashlib.sha256(payload).hexdigest()
    equal(set(vectors), {"plant", "measurement", "estimator"}, path.name + "/checkpoint buffers")
    expected = {"python_sha256": hashlib.sha256(python).hexdigest(), "integration_sha256": vectors}
    for arm in ARMS:
        equal(read(path / f"{arm}_start.json"), expected, path.name + "/restored " + arm)
    low = record[0]["controller"]["fields"]["low_level"]["fields"]
    return low["total_mass_kg"] * low["gravity_mps2"], expected


def actual_loads(r, label):
    loads = r["wheel_normal_load_n"]
    check(len(loads) == 4 and all(math.isfinite(x) and x >= 0 for x in loads), label + "/wheel contact")
    front, rear = sum(loads[:2]), sum(loads[2:])
    equal(r["front_normal_load_n"], float(front), label + "/front load")
    equal(r["rear_normal_load_n"], float(rear), label + "/rear load")
    return front, rear


def action_row(r, label):
    limits, raw, applied = r["joint_torque_limits_nm"], r["preclip_torque_nm"], r["applied_torque_nm"]
    check(len(raw) == len(limits) == len(applied) == 16, label + "/actuator axes")
    for value, limit, used in zip(raw, limits, applied, strict=True):
        check(math.isfinite(value) and math.isfinite(limit) and limit > 0, label + "/invalid torque")
        equal(used, max(-limit, min(value, limit)), label + "/actuator clip")
    violations = sum(abs(value) > limit + 1e-9 for value, limit in zip(applied, limits, strict=True))
    equal(r["torque_bound_violations"], violations, label + "/limit count")
    shifts = [a - b for a, b in zip(r["allocated_support_n"], r["baseline_support_n"], strict=True)]
    check(len(shifts) == 4, label + "/legs")
    delta = r["allocated_transfer_n"]
    for i, shift in enumerate(shifts):
        equal(shift, delta / 2 if i < 2 else -delta / 2, label + "/leg transfer")
    equal(r["support_sum_error_n"], math.fsum(shifts), label + "/support sum")
    raw_pitch(r, label)
    return violations, abs(math.fsum(shifts)), actual_loads(r, label)


def action_case(path, prefix_limit):
    metrics, prefix = read(path / "metrics.json"), lines(path / "prefix.jsonl")
    toe = metrics["toe_m"]
    equal(toe, TOE_M, path.name + "/toe")
    check(0 < len(prefix) <= prefix_limit, path.name + "/prefix budget")
    outcome = "prefix_cap"
    prefix_violations = 0
    for i, r in enumerate(prefix, 1):
        equal(r["step"], i, path.name + "/prefix step")
        equal(r["requested_transfer_n"], 0.0, path.name + "/prefix request")
        prefix_violations += action_row(r, path.name)[0]
        pos = r["front_contact_x_m"]
        if r["termination_reason"] or (pos is not None and pos >= toe - 0.05):
            check(i == len(prefix), path.name + "/missed first trigger")
            outcome = r["termination_reason"] or ("checkpoint" if pos < toe else "missed_precontact_band")
    if outcome == "prefix_cap":
        equal(len(prefix), prefix_limit, path.name + "/partial prefix")
    result = {"prefix_outcome": outcome, "prefix_steps": len(prefix)}
    fields(metrics, result, path.name)
    result["prefix_torque_violations"] = prefix_violations
    if outcome != "checkpoint":
        fields(metrics, {"terminal_branches": 0, "branch_steps": 0, "passed": False}, path.name)
        check(not any((path / f"{a}.jsonl").exists() for a in ARMS), path.name + "/unregistered branch")
        return {**result, "branch_steps": 0, "passed": False}
    weight, start_hash = checkpoint(path)
    branches = {key: lines(path / f"{key}.jsonl") for key in ARMS}
    diagnostics = {}
    for arm, rows in branches.items():
        check(0 < len(rows) <= 64, path.name + "/branch duration")
        readings = []
        for step, r in enumerate(rows, 1):
            equal(r["branch_tick"], step, path.name + "/" + arm)
            equal(r["step"], len(prefix) + step, path.name + "/absolute tick")
            equal(r["time_s"], (len(prefix) + step) * DT, path.name + "/branch clock")
            request = 0.02 * weight * ARMS[arm] if step <= 16 else 0.0
            equal(r["requested_transfer_n"], request, path.name + "/pulse schedule")
            readings.append(action_row(r, path.name + "/" + arm))
            check(r["termination_reason"] is None or step == len(rows), path.name + "/early terminal step")
        check(len(rows) == 64 or rows[-1]["termination_reason"] is not None, path.name + "/partial branch")
        diagnostics[arm] = readings
    contacts = {a: first_entry(rows, toe) for a, rows in branches.items()}
    shortest = min(len(rows) for rows in branches.values())
    first = max(contacts.values()) if None not in contacts.values() else None
    window = range(first, shortest) if first is not None else range(0)
    pitch = {a: rmse_deg([raw_pitch(rows[i], path.name) for i in window]) if window else None
             for a, rows in branches.items()}
    difference = pitch["positive"] - pitch["negative"] if window else None
    positive, zero, negative = [branches[a][0] for a in ("positive", "zero", "negative")]
    wheel = (3, 7, 11, 15)
    leg_delta = max(abs(positive["applied_torque_nm"][i] - negative["applied_torque_nm"][i])
                    for i in range(16) if i not in wheel)
    wheel_delta = max(abs(r["applied_torque_nm"][i] - zero["applied_torque_nm"][i])
                      for r in (negative, positive) for i in wheel)
    drive_delta = max(abs(r["longitudinal_drive_n"] - zero["longitudinal_drive_n"]) for r in (negative, positive))
    load_delta = []
    for i in range(shortest):
        loads = {arm: records[i][2] for arm, records in diagnostics.items()}
        if all(min(pair) > 0 for pair in loads.values()):
            front, rear = loads["positive"]
            other_front, other_rear = loads["negative"]
            load_delta.append((front - rear) - (other_front - other_rear))
    max_load = max(map(abs, load_delta), default=None)
    max_support_error = max(v[1] for samples in diagnostics.values() for v in samples)
    violations = {arm: sum(s[0] for s in samples) for arm, samples in diagnostics.items()}
    ends = {arm: rows[-1]["termination_reason"] for arm, rows in branches.items()}
    checks = {
        "all_three_reach_64_ticks": all(len(rows) == 64 for rows in branches.values()),
        "first_applied_torques_separate": leg_delta > 1e-8,
        "first_wheel_and_longitudinal_commands_match": wheel_delta <= 1e-8 and drive_delta <= 1e-8,
        "support_transfer_conserves_total": max_support_error <= 1e-8,
        "loaded_front_rear_response_separates": max_load is not None and max_load > 1e-8,
        "common_entry_rmse_separates": difference is not None and abs(difference) >= 0.01,
        "no_added_fall": ends["zero"] == "fallen" or "fallen" not in ends.values(),
        "in_bounds": "out_of_bounds" not in ends.values(),
        "no_added_torque_limit_violations": max(violations.values()) <= violations["zero"],
    }
    result.update({
        "terminal_branches": len(branches), "branch_steps": sum(len(v) for v in branches.values()),
        "common_entry_ticks": len(window), "positive_minus_negative_pitch_rmse_deg": difference,
        "first_pos_neg_applied_torque_linf_nm": leg_delta, "first_wheel_torque_spread_vs_zero_nm": wheel_delta,
        "first_longitudinal_force_spread_vs_zero_n": drive_delta, "max_per_tick_support_total_error_n": max_support_error,
        "max_abs_common_loaded_pos_minus_neg_axle_difference_n": max_load,
        "both_axles_loaded_all_arms_ticks": len(load_delta),
        "common_entry_start_tick": first + 1 if window else None,
        "common_entry_end_tick": shortest if window else None,
        "checks": checks, "passed": all(checks.values()),
    })
    fields(metrics, {k: v for k, v in result.items() if k != "prefix_torque_violations"}, path.name)
    for arm, value in pitch.items():
        fields(metrics["branches"][arm], {"common_entry_pitch_rmse_deg": value,
               "first_contact_tick": None if contacts[arm] is None else contacts[arm] + 1,
               "termination": ends[arm] or "branch_horizon", "torque_bound_violations": violations[arm],
               "steps": len(branches[arm])}, path.name + "/" + arm)
    result.update({"common_entry_rmse_deg": pitch, "actuator_violations": violations, "start_digests": start_hash})
    return result


def offline_check(path, floor):
    rows = lines(path / "archive_frames.jsonl")
    check(len({(r["case"], r["sequence"]) for r in rows}) == len(rows) == 308, "historical frame identity")
    pools = {}
    for angle in ("74", "82"):
        counts = {"frames": len(rows), "both_tracks_covered": 0, "ground_covered": 0, "ground_valid": 0, "valid": 0}
        for r in rows:
            m = r["mounts"][angle]
            for ground in (False, True):
                covered, grade = slopes(m["x_m"], m["z_m"], r["query_x_m"], m["first_hit_geom"], floor if ground else None)
                counts["ground_covered" if ground else "both_tracks_covered"] += int(covered)
                counts["ground_valid" if ground else "valid"] += int(grade is not None)
        fields(read(path / "summary.json")["pool"][angle], counts, "archived " + angle)
        pools[angle] = counts
    check((pools["74"]["ground_covered"], pools["74"]["ground_valid"]) == (82, 80), "historical geometry calibration")
    return pools


def audit(sensing, action, run_summary, offline=None, plan_path=PLAN):
    plan = read(plan_path)
    heldout, development = plan["heldout_sensing"], plan["axle_response"]
    check(set(heldout["seeds"]).isdisjoint(development["seeds"]), "seed partitions overlap")
    sensor_scores = {}
    floors = set()
    for terrain in heldout["terrain"]:
        for seed in heldout["seeds"]:
            case = f"{terrain}_{seed}"
            sensor_scores[case], floor = sensing_case(sensing / case, terrain)
            floors.add(floor)
    check(len(sensor_scores) == 6 and len(floors) == 1, "held-out design or ground identity")
    heldout_ticks = sum(c["task_steps"] for c in sensor_scores.values())
    sensor_passed = all(c["mounts"]["82"]["passed"] for c in sensor_scores.values())
    window_count = sum(c["zero_window_complete"] for c in sensor_scores.values())
    angles = ("74", "82")
    correct_counts = {a: sum(c["mounts"][a]["passed"] for c in sensor_scores.values()) for a in angles}
    chord_counts = {a: sum(c["matched_ground_chord_diagnostic"][a]["passed"] for c in sensor_scores.values())
                    for a in angles}
    first_chords = {a: sum(c["first_window_on_true_floor"][a] is True for c in sensor_scores.values())
                    for a in angles}
    falls = sum(c["termination_reason"] == "fallen" for c in sensor_scores.values())
    fields(read(sensing / "summary.json"), {
        "status": "COMPLETE", "task_steps": heldout_ticks, "zero_window_complete_cases": window_count,
        "correct_precontact_cases_by_angle": correct_counts, "true_chord_correct_precontact_cases_by_angle": chord_counts,
        "first_registered_windows_corroborated_by_angle": first_chords, "falls": falls,
        "passed": sensor_passed, "training_updates": 0,
    }, "sensing summary")
    budget = plan["budget"]
    cold = budget["single_cold_identification_controls"]
    prefix_limit = min(development["prefix_per_point_cap"], (
        budget["all_controls_including_identification_max"] - cold - heldout_ticks - budget["branch_controls_max"]
    ) // 6)
    responses = {}
    for terrain in development["terrain"]:
        for seed in development["seeds"]:
            key = f"{terrain}_{seed}"
            responses[key] = action_case(action / key, prefix_limit)
    check(len(responses) == 6, "development design")
    prefix_ticks = sum(c["prefix_steps"] for c in responses.values())
    branch_ticks = sum(c["branch_steps"] for c in responses.values())
    response_passed = all(c["passed"] for c in responses.values())
    triggered = sum(c["prefix_outcome"] == "checkpoint" for c in responses.values())
    count_branches = 3 * triggered
    fields(read(action / "summary.json"), {
        "status": "COMPLETE", "max_prefix_steps_each": prefix_limit, "prefix_steps": prefix_ticks,
        "branch_steps": branch_ticks, "task_steps": prefix_ticks + branch_ticks,
        "triggered_points": triggered, "terminal_branches": count_branches,
        "passed": response_passed, "training_updates": 0,
    }, "action summary")
    total = heldout_ticks + prefix_ticks + branch_ticks + cold
    check(heldout_ticks <= budget["heldout_controls_max"] and prefix_ticks <= budget["prefix_controls_max"]
          and branch_ticks <= budget["branch_controls_max"] and total <= budget["all_controls_including_identification_max"]
          and 5 * total <= budget["all_native_steps_including_identification_max"], "physical budget exceeded")
    gates = {"sensing": sensor_passed, "isolated_axle_response": response_passed,
             "six_zero_residual_score_windows": window_count == 6}
    combined = read(run_summary)
    ledger = {"identification": cold * 5, "sensing": heldout_ticks * 5, "action": (prefix_ticks + branch_ticks) * 5}
    fields(combined, {
        "status": "COMPLETE", "protocol": plan["protocol"], "gates": gates,
        "eligible_for_learning": all(gates.values()), "training_updates": 0, "heldout_controls": heldout_ticks,
        "prefix_limit_per_point": prefix_limit, "prefix_controls": prefix_ticks, "branch_controls": branch_ticks,
        "triggered_points": triggered, "terminal_branches": count_branches, "zero_window_complete_cases": window_count,
        "sensing_correct_by_angle": correct_counts, "sensing_falls": falls,
        "native_steps_by_stage": ledger, "total_native_steps": total * 5, "total_controls_including_identification": total,
    }, "runtime ledger")
    fields(combined["identification_cache"], {"misses": 1, "currsize": 1}, "shared identification")
    snapshot = hashlib.sha256((run_summary.parent / "identified_model.npz").read_bytes()).hexdigest()
    fields(combined["model"], {"model_sha256": snapshot, "operating_pitch_rad": OP_PITCH,
                               "pitch_gain_n_per_rad": -93618.1946018094}, "legacy model")
    return {
        "schema": "terrain-axle85-independent-readback-v1", "audit": "PASS",
        "new_physics_steps": 0, "new_training_updates": 0,
        "historic_counts": offline_check(offline, floors.pop()) if offline is not None else None,
        "heldout": sensor_scores, "development": responses, "heldout_task_controls": heldout_ticks,
        "prefix_controls": prefix_ticks, "branch_controls": branch_ticks, "accounted_cold_controls": cold,
        "verified_native_steps_by_stage": ledger, "model_sha256": snapshot,
        "correct_cases_by_angle": correct_counts, "matched_chord_cases_by_angle": chord_counts,
        "first_true_floor_windows_by_angle": first_chords, "heldout_falls": falls,
        "total_controls": total, "total_native_steps": total * 5,
        "sensor_passed": sensor_passed, "response_passed": response_passed, "complete_040m_windows": window_count,
        "joint_gate_passed": sensor_passed and response_passed and window_count == 6,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("sensing", "action", "run-summary", "output"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--offline", type=Path)
    parser.add_argument("--plan", type=Path, default=PLAN)
    args = parser.parse_args()
    with args.output.open("x", encoding="utf-8") as stream:
        try:
            result = audit(args.sensing, args.action, args.run_summary, args.offline, args.plan)
        except Exception as exc:
            json.dump({"audit": "FAIL", "error": repr(exc), "new_physics_steps": 0}, stream, indent=2)
            stream.write("\n")
            raise
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: result[k] for k in ("audit", "joint_gate_passed", "total_controls", "total_native_steps")}))


if __name__ == "__main__":
    main()
