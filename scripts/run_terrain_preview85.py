"""Execute the two registered round-85 probes in a single supervised CPU process."""

import argparse
import hashlib
import json
import math
import tempfile
import zipfile
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np
import scipy
from scipy.linalg import solve_discrete_are

from scripts import terrain_preview84_trials as prior
from scripts.terrain_preview85_action import run_action_cases
from scripts.terrain_preview85_sensing import run_offline, run_sensor_cases
from wheel_legged_control.d1.hierarchical import D1_OUTER_Q, D1_OUTER_R
from wheel_legged_control.d1.linear_model import identify_sagittal_model

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "docs/experiments85/terrain_action85.json"
ARCHIVE = ROOT / "docs/assets/project84/recordings.zip"
SOURCES = (
    "scripts/run_terrain_preview85.py", "scripts/terrain_preview85_common.py",
    "scripts/terrain_preview85_sensing.py", "scripts/terrain_preview85_action.py",
    "scripts/terrain_preview84_mount_audit.py", "src/wheel_legged_control/d1/controllers.py",
    "scripts/run_pressure83.py", "scripts/resource_pressure83.py",
    "runtime/d1_b22/runtime_manifest_v27.json",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_archived(output):
    """A clean checkout needs only the committed 84 ZIP; no private run path is used."""
    with tempfile.TemporaryDirectory(prefix="terrain85-archive-") as temp, zipfile.ZipFile(ARCHIVE) as z:
        for entry in z.infolist():
            if entry.filename.startswith("trials/"):
                if ".." in Path(entry.filename).parts:
                    raise ValueError("invalid archived path")
                z.extract(entry, temp)
        source = Path(temp) / "trials"
        previous = json.loads((source / "protocol.json").read_text())
        for name, expected in previous["sources_sha256"].items():
            if sha(ROOT / name) != expected:
                raise ValueError(f"round 84 source identity changed: {name}")
        archived_case = json.loads((source / "cases/up_8417_off/metrics.json").read_text())
        return run_offline(source, output), archived_case


def calibrate_once(output, reference):
    if identify_sagittal_model.cache_info().currsize:
        raise RuntimeError("fresh model cache required for charging calibration")
    model = identify_sagittal_model(prior.DT, "legacy")
    with (output / "identified_model.npz").open("xb") as dst:
        np.savez_compressed(dst, **asdict(model))
    p = solve_discrete_are(model.a, model.b, D1_OUTER_Q, D1_OUTER_R)
    gain = np.linalg.solve(D1_OUTER_R + model.b.T @ p @ model.b, model.b.T @ p @ model.a)
    if not (math.isclose(gain[0, 1], reference["pitch_gain_n_per_rad"], rel_tol=1e-10)
            and math.isclose(model.operating_state[1], reference["operating_pitch_rad"], abs_tol=1e-12)):
        raise ValueError("identified baseline differs from the archived round 84 controller")
    return {"pitch_gain_n_per_rad": float(gain[0, 1]), "operating_pitch_rad": float(model.operating_state[1]),
            "model_sha256": sha(output / "identified_model.npz")}


def prefix_cap(plan, sensing_steps):
    b = plan["budget"]
    available = (b["all_controls_including_identification_max"] - b["single_cold_identification_controls"]
                 - sensing_steps - b["branch_controls_max"])
    return min(plan["axle_response"]["prefix_per_point_cap"], available // 6)


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    plan = json.loads(PLAN.read_text())
    budget = plan["budget"]
    identity = {name: sha(ROOT / name) for name in SOURCES}
    prior.write_json(output / "started.json", {
        "protocol": plan["protocol"], "plan_sha256": sha(PLAN), "sources_sha256": identity,
        "prior_archive_sha256": sha(ARCHIVE), "mujoco": mujoco.__version__, "numpy": np.__version__,
        "scipy": scipy.__version__, "status_until_summary": "PARTIAL",
    })
    counts = Counter()
    phase = "archive"
    native_step = mujoco.mj_step

    def metered_step(model, data, nstep=1):
        if phase == "archive" or sum(counts.values()) + nstep > budget["all_native_steps_including_identification_max"]:
            raise RuntimeError(f"unexpected physical integration in {phase} or exhausted native-step budget")
        native_step(model, data, nstep)
        counts[phase] += nstep

    try:
        if sha(ARCHIVE) != plan["previous_recordings_sha256"]:
            raise ValueError("archived recordings SHA-256 mismatch")
        with patch.object(mujoco, "mj_step", metered_step):
            offline, reference = run_archived(output / "archive")
            if not offline["passed"]:
                raise ValueError("archived 74-degree measurement reference did not reproduce")
            phase = "identification"
            calibration = calibrate_once(output, reference)
            if counts[phase] != budget["single_cold_identification_controls"] * 5:
                raise RuntimeError("unexpected number of native identification steps")
            phase = "sensing"
            sensing = run_sensor_cases(output / "sensing", tuple(plan["heldout_sensing"]["seeds"]))
            if (counts[phase] != sensing["task_steps"] * 5 or len(sensing["cases"]) != 6
                    or sensing["task_steps"] > budget["heldout_controls_max"]):
                raise ValueError("incomplete or miscounted sensing cases")
            phase = "action"
            max_prefix = prefix_cap(plan, sensing["task_steps"])
            action = run_action_cases(output / "action", tuple(plan["axle_response"]["seeds"]),
                                      max_prefix_steps=max_prefix)
        total = (budget["single_cold_identification_controls"] + sensing["task_steps"]
                 + action["prefix_steps"] + action["branch_steps"])
        if (len(action["cases"]) != 6 or action["branch_steps"] > budget["branch_controls_max"]
                or action["prefix_steps"] > max_prefix * 6 or counts["action"] != action["task_steps"] * 5
                or sum(counts.values()) != total * 5 or identify_sagittal_model.cache_info().misses != 1):
            raise ValueError("incomplete action cases or native-step ledger mismatch")
        if identity != {name: sha(ROOT / name) for name in SOURCES}:
            raise RuntimeError("implementation or frozen source identity changed during trial")
        gates = {"sensing": sensing["passed"], "isolated_axle_response": action["passed"],
                 "six_zero_residual_score_windows": sensing["zero_window_complete_cases"] == 6}
        result = {
            "protocol": plan["protocol"], "status": "COMPLETE", "gates": gates,
            "eligible_for_learning": all(gates.values()), "training_updates": 0,
            "archive_counts": offline["pool"], "model": calibration,
            "sensing_correct_by_angle": sensing["correct_precontact_cases_by_angle"],
            "zero_window_complete_cases": sensing["zero_window_complete_cases"],
            "sensing_falls": sensing["falls"], "triggered_points": action["triggered_points"],
            "terminal_branches": action["terminal_branches"],
            "heldout_controls": sensing["task_steps"], "prefix_limit_per_point": max_prefix,
            "prefix_controls": action["prefix_steps"], "branch_controls": action["branch_steps"],
            "total_controls_including_identification": total, "native_steps_by_stage": dict(counts),
            "total_native_steps": sum(counts.values()), "identification_cache": identify_sagittal_model.cache_info()._asdict(),
        }
        prior.write_json(output / "summary.json", result)
        return result
    except BaseException as exc:
        prior.write_json(output / "partial.json", {"status": "PARTIAL", "stage": phase,
                                                  "native_steps_by_stage": dict(counts), "error": repr(exc)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().output), indent=2, allow_nan=False))
