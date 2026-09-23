"""Pure, supplementary archive audit: faithful recording does not certify geometry.

This module never imports MuJoCo or constructs a plant. The original scorer and
all physics evidence remain immutable; no output can grant qualification or RL.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.abc
import json
import sys
from pathlib import Path

import numpy as np

from scripts.d1_single_step_contact_diagnostics import audit_contact
from scripts.d1_single_step_scoring import (
    TORQUE_LIMITS_NM,
    _f,
    _mat,
    _parse_endpoint,
    _parse_trace,
    _quat_to_rpy,
    _vec,
)

FROZEN_CONTRACT_SHA256 = "1ea04dc310486ed7025a73ed1e61a7977fdaafe0545f3de859905835d1b5c238"
NEXT_CONTRACT_SHA256 = "157f0a33b386cea0ef99556a0d693370126ae57479db0460e4b211dd7b27e7d9"
FROZEN_MANIFEST_SHA256 = "80b7a829b6be1332bf050d68b2774903457961a5400c5efa8b59da094e366f46"
FROZEN_FIXTURE_SHA256 = "91cb3c9d1137b11c39f071de01ddb9e2f9ba83ca90af8e04e3cc565800852a55"
FROZEN_STATIC_SHA256 = "0c46a59f8905e1310b7a73d4c6c749440264161ddb27a4d067ac0ce005d30bc2"


class DenyPhysicsImports(importlib.abc.MetaPathFinder):
    """CLI additionally rejects environments that already imported physics."""

    def __init__(self):
        self.attempts = []

    def find_spec(self, fullname, path=None, target=None):
        if (fullname == "mujoco" or fullname.startswith(("mujoco.", "wheel_legged_control"))
                or fullname in {"scripts.d1_single_step_env", "scripts.d1_single_step_plant",
                                "scripts.probe_d1_single_step", "scripts.qualify_d1_single_step"}):
            self.attempts.append(fullname)
            raise ImportError("zero-call forensic contract forbids import: "+fullname)


def require(condition, code):
    if not condition:
        raise ValueError(code)


def audit_native_state(row, index, previous, trace):
    """Validate a complete native interval and its held control, pure arithmetic."""
    require(type(row["index"]) is int and row["index"] == index, "native_index")
    require(row["returned"] is True, "native_not_returned")
    require(row["error"] is None, "native_error")
    require(row["contact_sample_tag"] == "native_step_solved_cache_not_synchronized_endpoint",
            "native_contact_sampling")
    for field, expected in (("start_time_s", index*.002),
                            ("end_time_s", (index+1)*.002), ("actual_dt_s", .002)):
        require(abs(_f(row[field], field)-expected) <= 1e-9, "native_time:"+field)
    for field, size in (("qpos_before", 23), ("qpos_returned", 23),
                        ("qvel_before", 22), ("qvel_returned", 22),
                        ("qacc_warmstart_before", 22), ("qacc_warmstart_after", 22)):
        _vec(row[field], size, field)
    for field in ("qpos_before", "qpos_returned"):
        _quat_to_rpy(np.asarray(row[field][3:7]), field)
    for field in ("act_before", "act_after"):
        _vec(row[field], 0, field)
    ctrl = _vec(row["ctrl_nm"], 16, "ctrl_nm")
    require(np.all(abs(ctrl) <= TORQUE_LIMITS_NM), "native_torque_limits")
    require(np.array_equal(ctrl, trace[index//5]["torque_nm"]), "held_control_mismatch")
    require(np.array_equal(_mat(row["xfrc_applied"], (18, 6), "xfrc"), np.zeros((18, 6))),
            "foreign_body_wrench")
    require(np.array_equal(_vec(row["qfrc_applied"], 22, "qfrc"), np.zeros(22)),
            "foreign_generalized_force")
    if previous is not None:
        for field in ("qpos", "qvel"):
            require(np.array_equal(previous[field+"_returned"], row[field+"_before"]),
                    field+"_chain_broken")


def audit_endpoint(ep, tick, native, manifest):
    """Bind endpoint state and every collision identity to the compiled manifest."""
    _parse_endpoint(ep, tick, "endpoint")
    row = native[0 if tick == 0 else 5*tick-1]
    suffix = "_before" if tick == 0 else "_returned"
    q, v = np.asarray(row["qpos"+suffix]), np.asarray(row["qvel"+suffix])
    # Reproduce the original endpoint quaternion arithmetic without renormalizing.
    w, x, y, z = q[3:7]
    rpy = [np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y)),
           np.arcsin(np.clip(2*(w*y-z*x), -1, 1)),
           np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))]
    forward = np.array([1-2*(y*y+z*z), 2*(x*y+w*z), 2*(x*z-w*y)])
    vx = float(forward@v[:3]+np.cross(v[3:6], manifest["base_body_ipos_local_m"])[0])
    require(np.array_equal(q[:3], ep["base_position_m"]), "endpoint_position_link")
    require(np.allclose(rpy, ep["rpy_rad"], rtol=0., atol=1e-12), "endpoint_rotation_link")
    require(abs(vx-ep["body_vx_mps"]) <= 1e-12, "endpoint_base_com_velocity_link")
    com_v = _vec(ep["com_velocity_mps"], 3, "com_velocity")
    _vec(ep["com_position_m"], 3, "com_position")
    require(ep["com_vz_mps"] == com_v[2], "endpoint_whole_com_velocity_link")
    robot = {g["identity"]: g for g in manifest["geoms"] if g["body_id"] > 0 and g["collision"]}
    bounds = ep["collision_bounds"]
    require(len(bounds) == len(robot) and {b["identity"] for b in bounds} == set(robot),
            "endpoint_collision_identity_set")
    for bound in bounds:
        for field in ("geom_id", "body_name", "geom_type", "wheel_index", "margin_m"):
            require(bound[field] == robot[bound["identity"]][field], "collision_binding:"+field)


def audit_payload(payload):
    """Visit all available records, including geometrically anomalous contacts."""
    raw, geometry = [], []
    counts = {"native": 0, "endpoints": 0, "controls": 0, "contact_rows": 0,
              "box_contact_rows": 0, "positive_box_contact_rows": 0}
    loaded_geometry_invalid = False

    def check(location, function, *args):
        try:
            function(*args)
        except Exception as exc:  # noqa: BLE001 -- every malformed record remains explicitly invalid
            raw.append({"location": location, "code": type(exc).__name__+":"+str(exc)[:240]})

    try:
        manifest = payload["geometry_manifest"]
        geoms = {g["geom_id"]: g for g in manifest["geoms"]}
        native, endpoints, trace = (payload[k] for k in ("native", "endpoints", "trace"))
        require(isinstance(native, list) and isinstance(endpoints, list) and isinstance(trace, list),
                "collections_not_lists")
        require(len(geoms) == len(manifest["geoms"]) and set(geoms) == set(range(len(geoms))),
                "compiled_geom_mapping")
        require(len({g["identity"] for g in geoms.values()}) == len(geoms), "duplicate_geom_identity")
        _vec(manifest["base_body_ipos_local_m"], 3, "base_body_ipos")
        for g in geoms.values():
            require(type(g["geom_id"]) is int and type(g["body_id"]) is int, "compiled_ids_not_int")
            _vec(g["position_local_m"], 3, "compiled_geom_position")
            _vec(g["size_m"], 3, "compiled_geom_size")
            require(_f(g["margin_m"], "compiled_margin") >= 0., "negative_compiled_margin")
    except Exception as exc:  # noqa: BLE001 -- fail closed on malformed schema
        return {"raw_record_links_valid": False, "raw_issues": [{"code": "schema:"+str(exc)}],
                "counts": counts, "all_candidate_geometry_valid": False,
                "positive_load_candidate_geometry_valid": False, "geometry_issues": [],
                "original_score_overridden": False, "qualification_granted": False, "rl_gate_open": False}
    check("counts", require, len(native) == 6000 and len(endpoints) == 1201 and len(trace) == 1200,
          "incomplete_frozen_trial")
    previous = None
    for index, row in enumerate(native):
        counts["native"] += 1
        check(f"native[{index}]", audit_native_state, row, index, previous, trace)
        previous = row
        loads, box_loads = np.zeros(4), np.zeros(4)
        nonwheel, geometric_box = 0, False
        try:
            cdata = row["contacts"]
            contacts = cdata["contacts"]
            require(isinstance(contacts, list), "contacts_not_list")
            require(cdata["sampling"] == row["contact_sample_tag"], "contact_sampling_link")
        except Exception as exc:  # noqa: BLE001 -- retain the rest of an invalid archive
            raw.append({"location": f"native[{index}]", "code": "contacts_schema:"+str(exc)})
            continue
        for j, contact in enumerate(contacts):
            counts["contact_rows"] += 1
            if not isinstance(contact, dict):
                raw.append({"location": f"native[{index}].contacts[{j}]", "code": "contact_not_dict"})
                continue
            result = audit_contact(contact, geoms)
            location = f"native[{index}].contacts[{j}]"
            for issue in result["raw_issues"]:
                raw.append({"location": location, "code": issue})
            check(location, require, contact.get("index") == j, "contact_index")
            for issue in result["geometry_issues"]:
                geometry.append({"native_index": index, "contact_index": j, "code": issue,
                    "positive_load": result["positive_load"], "diagnostic": result["box_feature"],
                    "raw_contact": contact})
            is_box, wheel = result["is_box"], result["wheel_index"]
            geometric_box |= is_box
            counts["box_contact_rows"] += int(is_box)
            if is_box and result["positive_load"]:
                counts["positive_box_contact_rows"] += 1
                loaded_geometry_invalid |= bool(result["geometry_issues"] or result["raw_issues"])
            if contact.get("terrain_geom_id") is not None:
                if wheel is None:
                    nonwheel += 1
                elif result["positive_load"] and isinstance(wheel, int) and 0 <= wheel < 4:
                    loads[wheel] += result["normal_load_n"]
                    if is_box:
                        box_loads[wheel] += result["normal_load_n"]
        for field, expected in (("wheel_positive_normal_load_n", loads),
                                ("wheel_box_positive_normal_load_n", box_loads),
                                ("nonwheel_terrain_contacts", nonwheel),
                                ("geometric_box_contact", geometric_box)):
            check(f"native[{index}]", require, np.array_equal(cdata.get(field), expected),
                  "contact_aggregate:"+field)
    for tick, ep in enumerate(endpoints):
        counts["endpoints"] += 1
        check(f"endpoints[{tick}]", audit_endpoint, ep, tick, native, manifest)
    for tick, tr in enumerate(trace):
        counts["controls"] += 1
        check(f"trace[{tick}]", _parse_trace, tr, tick, payload.get("obstacle_enabled"), "trace")
        check(f"trace[{tick}]", require, isinstance(tr, dict) and tr.get("endpoint_tick") == tick+1,
              "trace_endpoint_tick")
    events = payload.get("native_entry_events", [])
    check("events", require, len(events) == len(native), "native_entry_event_count")
    def event_check(ev, index):
        require(ev["attempt"] == index and abs(_f(ev["time_s"], "event_time")-index*.002) <= 1e-9,
                "native_entry_event_link")
    for index, ev in enumerate(events):
        check(f"entry_events[{index}]", event_check, ev, index)
    if "receipt" in payload:
        def receipt_check():
            rec = payload["receipt"]
            require(rec["completed_control_intervals"] == len(trace) and rec["clock_native_steps"] == len(native),
                    "receipt_count_link")
            require(rec["error"] is None and rec["terminated"] is False and rec["truncated"] is True,
                    "receipt_terminal_status")
            require(rec["callback_count"] == len(endpoints), "receipt_callback_count")
            nr = rec["native_receipt"]
            require(nr["attempted_native_calls"] == nr["returned_native_calls"] == nr["recorded_entries"] == len(native),
                    "native_receipt_count_link")
            require(nr["failed_native_calls"] == nr["foreign_pass_through_calls"] == 0, "native_receipt_failure")
        check("receipt", receipt_check)
    else:
        raw.append({"location": "receipt", "code": "missing_receipt"})
    return {"raw_record_links_valid": not raw, "raw_issues": raw, "counts": counts,
        "all_candidate_geometry_valid": counts["contact_rows"] > 0 and not geometry and not raw,
        "positive_load_candidate_geometry_valid": (len(native) == 6000
                                                   and counts["positive_box_contact_rows"] == 11032
                                                   and not loaded_geometry_invalid),
        "geometry_issues": geometry, "original_score_overridden": False,
        "qualification_granted": False, "rl_gate_open": False,
        "scope": "native/control/endpoint identities, timing, finite state, raw contact links, aggregates, original command and torque gates; no force re-solve or continuous collision proof"}


def cylinder_support_proof(fixture):
    row = next(r for r in fixture["static_forensic"]["reconstructions"]
               if r["native_index"] == 3486 and r["pose"] == "before")
    center = np.asarray(row["wheel_center_world_m"])
    axis = np.asarray(row["wheel_rotation_world"])[:, 2]
    radius, half_length = row["wheel_size_m"][:2]
    up = np.array([0., 0., 1.])
    require(abs(axis[2]) < 1., "vertical_axis_outside_fixed_fixture")
    point = center-half_length*np.sign(axis[2])*axis-radius*(up-axis[2]*axis)/np.sqrt(1-axis[2]**2)
    box = next(g for g in fixture["geometry_manifest"]["geoms"] if g["terrain_kind"] == "box")
    bc, half = np.asarray(box["position_local_m"]), np.asarray(box["size_m"])
    gap = float(point[2]-bc[2]-half[2])
    return {"minimum_support_point_m": point.tolist(), "box_top_m": float(bc[2]+half[2]),
            "vertical_gap_m": gap, "xy_inside_box": bool(np.all(abs(point[:2]-bc[:2]) <= half[:2])),
            "saved_geom_distance_m": row["geom_distance_m"],
            "analytic_minus_saved_distance_m": gap-row["geom_distance_m"],
            "new_mujoco_calls": 0, "qualification_granted": False}


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def verify_anchor_hashes(manifest_sha, fixture_sha):
    """Reject changed trial/fixture identities before any full-pass budget charge."""
    require(manifest_sha == FROZEN_MANIFEST_SHA256, "frozen_archive_manifest_changed")
    require(fixture_sha == FROZEN_FIXTURE_SHA256, "fixture_sha256_changed")


def verify_fixture(fixture_path, archive, static_report):
    """Bind the copied real samples to their four frozen source files."""
    fixture = read_json(fixture_path)
    expected = {
        "single_step_readiness_01/single_15mm_box/native.jsonl.gz": archive/"single_15mm_box/native.jsonl.gz",
        "single_step_readiness_01/single_15mm_box/geometry_manifest.json": archive/"single_15mm_box/geometry_manifest.json",
        "static_contact_forensic_01.json": static_report,
        "single_step_readiness_01/manifest.json": archive/"manifest.json",
    }
    sources = {}
    for key, value in fixture["sources"].items():
        if "/stability_20260922/" in key:
            relative = key.split("/stability_20260922/", 1)[1]
        elif key.startswith("stability_20260922/"):
            relative = key.split("/", 1)[1]
        else:
            relative = key
        require(relative not in sources, "duplicate_fixture_source_path")
        sources[relative] = value
    require(set(sources) == set(expected), "fixture_source_paths")
    for relative, path in expected.items():
        source = sources[relative]
        require(file_sha(path) == source["sha256"] and path.stat().st_size == source["bytes"],
                "fixture_source_changed:"+relative)
    require(fixture["geometry_manifest"] == read_json(
        expected["single_step_readiness_01/single_15mm_box/geometry_manifest.json"]),
            "fixture_geometry_manifest_link")
    require(fixture["static_forensic"] == read_json(expected["static_contact_forensic_01.json"]),
            "fixture_static_forensic_link")
    requested = {int(row["index"]): row for row in fixture["native_samples"]}
    require(set(requested) == {3485, 3486, 3487}, "fixture_native_sample_indices")
    bad = requested[fixture["expected_bad_native_index"]]["contacts"]["contacts"]
    require(bad[fixture["expected_bad_contact_index"]]["index"] == 5,
            "fixture_bad_contact_index")
    return fixture


def write_json(path, obj):
    with path.open("x") as stream:
        json.dump(obj, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--diagnostic-contract", type=Path, required=True)
    parser.add_argument("--static-report", type=Path, required=True)
    parser.add_argument("--budget", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not any(k == "mujoco" or k.startswith("mujoco.") for k in sys.modules), "physics_already_imported")
    guard = DenyPhysicsImports()
    sys.meta_path.insert(0, guard)
    require(not args.output.exists(), "output_exists")
    protocol, manifest = read_json(args.archive/"protocol.json"), read_json(args.archive/"manifest.json")
    require(file_sha(args.contract) == FROZEN_CONTRACT_SHA256, "frozen_contract_changed")
    require(file_sha(args.diagnostic_contract) == NEXT_CONTRACT_SHA256, "next_contract_changed")
    verify_anchor_hashes(file_sha(args.archive/"manifest.json"), file_sha(args.fixture))
    require(protocol["contract_sha256"] == FROZEN_CONTRACT_SHA256, "protocol_contract_link")
    require(len(manifest) == 24, "archive_manifest_entry_count")
    fixture = verify_fixture(args.fixture, args.archive, args.static_report)
    source_bad = [p for p, h in protocol["input_sha256"].items() if file_sha(args.root/p) != h]
    archive_bad = [p for p, v in manifest.items() if file_sha(args.archive/p) != v["sha256"]
                   or (args.archive/p).stat().st_size != v["bytes"]]
    require(not source_bad and not archive_bad, "source_or_archive_identity_changed")
    budget = read_json(args.budget) if args.budget.exists() else {"attempted_full_passes": 0, "max_full_passes": 2}
    require(type(budget["attempted_full_passes"]) is int
            and 0 <= budget["attempted_full_passes"] <= 2
            and type(budget["max_full_passes"]) is int
            and budget["max_full_passes"] == 2, "invalid_offline_budget_record")
    require(budget["attempted_full_passes"] < 2, "offline_budget_exhausted")
    budget["attempted_full_passes"] += 1
    args.budget.write_text(json.dumps(budget, indent=2)+"\n")
    args.output.mkdir()
    results = {}
    for name in ("plane_only", "single_15mm_box"):
        folder = args.archive/name
        p = {"obstacle_enabled": name == "single_15mm_box",
             "geometry_manifest": read_json(folder/"geometry_manifest.json"),
             "receipt": read_json(folder/"receipt.json")}
        for key in ("native", "endpoints", "trace", "native_entry_events"):
            with gzip.open(folder/(key+".jsonl.gz"), "rt") as stream:
                p[key] = [json.loads(line) for line in stream]
        result = audit_payload(p)
        result.update(archive_identity_valid=True, source_identity_valid=True,
                      archive_identity_scope="all 24 manifest entries: SHA256 and byte count",
                      source_identity_scope="all protocol input SHA256 entries, frozen contract, fixture and its four sources",
                      original_frozen_score=read_json(folder/"score.json"))
        write_json(args.output/(name+".json"), result)
        results[name] = {k: result[k] for k in ("raw_record_links_valid", "counts",
            "all_candidate_geometry_valid", "positive_load_candidate_geometry_valid")}
        print(json.dumps({"case": name, **results[name]}), flush=True)
    write_json(args.output/"support_proof.json", cylinder_support_proof(fixture))
    write_json(args.output/"receipt.json", {"full_pass_number": budget["attempted_full_passes"],
        "cases": results, "new_control": 0, "new_native": 0, "new_static_mujoco": 0,
        "forbidden_import_attempts": guard.attempts, "mujoco_imported": "mujoco" in sys.modules,
        "archive_manifest_sha256": file_sha(args.archive/"manifest.json"),
        "fixture_sha256": file_sha(args.fixture), "qualification_granted": False, "rl_gate_open": False})


if __name__ == "__main__":
    main()
