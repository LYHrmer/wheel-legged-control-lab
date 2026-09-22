"""Describe saved trajectories; never override the original readiness verdict."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from scripts.audit_d1_single_step_contact import contact_anomalies
from scripts.d1_single_step_records import PhysicsCallLedger
from scripts.probe_d1_single_step import load_case, quaternion_rpy, write_json


def observations(payload):
    ep, native = payload["endpoints"], payload["native"]
    poses = [np.asarray(e["rpy_rad"]) for e in ep]
    poses += [quaternion_rpy(n[k][3:7]) for n in native
              for k in ("qpos_before", "qpos_returned")]
    rpy = np.asarray(poses)
    yaw = (rpy[:, 2]-ep[0]["rpy_rad"][2]+np.pi) % (2*np.pi)-np.pi
    lateral = [e["base_position_m"][1] for e in ep]
    lateral += [n[k][1] for n in native for k in ("qpos_before", "qpos_returned")]
    features, wheels, first_loaded = Counter(), set(), None
    for row in native:
        for c in row["contacts"]["contacts"]:
            if c["box_feature"] and c["wheel_index"] is not None and c["active"] and c["normal_load_n"] > 0:
                features[c["box_feature"]["feature"]] += 1
                wheels.add(c["wheel_index"])
                if first_loaded is None:
                    first_loaded = row["index"]
    late = ep[1101:1201]
    loads = np.asarray([n["contacts"]["wheel_positive_normal_load_n"] for n in native[5500:6000]])
    far_edge = ep[0]["base_position_m"][0]+.88
    return {"source_identity_valid": payload["source_identity_valid"],
        "supplementary_observations_only": True, "qualification_granted": False,
        "original_score_overridden": False, "controls": len(payload["trace"]),
        "native": len(native), "max_abs_roll_deg": float(np.rad2deg(abs(rpy[:, 0]).max())),
        "max_abs_pitch_deg": float(np.rad2deg(abs(rpy[:, 1]).max())),
        "max_abs_heading_deg": float(np.rad2deg(abs(yaw).max())),
        "max_abs_lateral_m": float(np.max(abs(np.asarray(lateral)-ep[0]["base_position_m"][1]))),
        "late_max_abs_body_vx_mps": max(abs(e["body_vx_mps"]) for e in late),
        "late_max_abs_whole_com_vz_mps": max(abs(e["com_vz_mps"]) for e in late),
        "late_world_z_rmse_m": float(np.sqrt(np.mean([(e["base_position_m"][2]-.455)**2 for e in late]))),
        "late_wheel_positive_load_fractions": np.mean(loads > 0., axis=0).tolist(),
        "nonwheel_native_contact_rows": sum(n["contacts"]["nonwheel_terrain_contacts"] for n in native),
        "nonwheel_endpoint_contact_rows": sum(e["nonwheel_terrain_contacts"] for e in ep),
        "final_whole_robot_clearance_beyond_far_edge_m": min(
            b["minimum_world_m"][0]-far_edge-b["margin_m"] for b in ep[-1]["collision_bounds"]),
        "far_edge_m": far_edge, "positive_box_feature_counts": dict(features),
        "positive_box_wheels": sorted(wheels), "first_positive_box_native_index": first_loaded,
        "contact_inventory": contact_anomalies(native)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    with PhysicsCallLedger(native_limit=0) as ledger:
        p = load_case(args.case)
        if not p["source_identity_valid"]:
            raise ValueError("original input or archive identity mismatch")
        report = observations(p)
    report["ledger"] = ledger.receipt()
    write_json(args.output/"observations.json", report)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = p["endpoints"]
    t = [e["time_s"] for e in ep]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    fig.suptitle("15 mm box: saved observations; original qualification REJECTED")
    axes[0, 0].plot(t, [min(b["minimum_world_m"][0]-b["margin_m"]
        for b in e["collision_bounds"])-report["far_edge_m"] for e in ep])
    axes[0, 0].axhline(0., color="gray", ls="--")
    axes[0, 0].set_ylabel("Whole robot clearance past far edge (m)")
    axes[0, 1].plot(t, [np.rad2deg(e["rpy_rad"][1]) for e in ep])
    axes[0, 1].set_ylabel("Pitch (deg)")
    axes[1, 0].plot(t, [e["body_vx_mps"] for e in ep], label="Base body vx")
    axes[1, 0].plot(t, [e["com_vz_mps"] for e in ep], label="Whole COM vz")
    axes[1, 0].set_ylabel("Velocity (m/s)")
    axes[1, 0].legend()
    axes[1, 1].plot([n["start_time_s"] for n in p["native"]],
        [sum(n["contacts"]["wheel_box_positive_normal_load_n"]) for n in p["native"]])
    axes[1, 1].set_ylabel("Total positive wheel-box load (N)")
    for ax in axes.flat:
        ax.axvline(3486*.002, color="red", ls=":", label="Unloaded normal anomaly")
        ax.set_xlabel("Time (s)")
        ax.grid(alpha=.2)
    fig.savefig(args.output/"observations.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
