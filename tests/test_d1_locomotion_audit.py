"""Independent arithmetic and deliberately corrupted locomotion evaluation artifacts.

The fixture is a tiny hand-checkable directory: two synthetic roads, two oracle
noise-seed labels per road with byte-identical trajectories, three 0.01 s ticks
each. Nothing is simulated; every number is a literal chosen so the expected
metrics can be written in closed form inside the tests.
"""

import ast
import csv
import hashlib
import io
import json
import math
import tarfile
from pathlib import Path

import numpy as np
import pytest

from scripts.audit_d1_locomotion import audit_evaluation, audit_matrix, paired_deltas

AUDIT_PATH = Path(__file__).resolve().parents[1] / "scripts/audit_d1_locomotion.py"
DT = 0.01
DURATION = 0.03
SEEDS = (101, 202)
QUALITY_LIMITS = {
    "velocity_rmse_mps": 0.08,
    "yaw_rmse_rps": 0.08,
    "height_rmse_m": 0.025,
    "attitude_rmse_rad": 0.15,
}
TERRAINS = (
    {"name": "road0", "kind": "synthetic_rough", "amplitude_m": 0.03},
    {"name": "road1", "kind": "synthetic_steps", "amplitude_m": 0.05},
)
SCHEDULE = {"segments": [{"forward_velocity_mps": 0.5, "yaw_rate_rps": 0.0, "clearance_m": 0.42}]}
COMMANDS = {
    "command_forward_velocity_mps": 0.5,
    "command_yaw_rate_rps": 0.0,
    "command_clearance_m": 0.42,
}
ERRORS = {
    "velocity_error_mps": (0.04, -0.08, 0.04),
    "yaw_rate_error_rps": (0.02, 0.04, -0.06),
    "height_error_m": (0.01, -0.02, 0.01),
    "roll_error_rad": (0.03, -0.05, 0.04),
    "pitch_error_rad": (0.06, 0.02, -0.03),
}
STEPS_XY = ((0.01, 0.0), (0.0, 0.02), (0.01, 0.0))
FLAGS = (False, True, True)
ACTIONS = ((0.2, 0.4), (0.5, 0.5), (0.6, 0.0))
GROUND = (0.02, 0.05, 0.02)
POWER = (12.0, 15.0, 18.0)
SOURCE_FILES = ("scripts/run_d1_locomotion.py", "src/wheel_legged_control/d1/locomotion_env.py")
DEFAULT_GAINS = {
    "wheel_kp": 2.2,
    "wheel_ki": 3.0,
    "yaw_feedback_gain": 4.0,
    "leg_feedback_scale": 1.0,
    "attitude_feedback_scale": 1.0,
}
V2 = "d1-wheel-leg-joint-pd-wheel-pi-yaw-feedback-v2"
V3 = "d1-wheel-leg-joint-pd-wheel-pi-yaw-feedback-configured-v3"
V4 = "d1-wheel-leg-joint-pd-wheel-pi-yaw-feedback-configured-v4"


def build_rows(ticks, scale, zero=False):
    """One CSV row per control tick, all quantities exactly as recorded."""
    rows, x, y = [], -3.8, 0.0
    for tick in range(ticks):
        x, y = x + STEPS_XY[tick][0], y + STEPS_XY[tick][1]
        errors = {key: value[tick] for key, value in ERRORS.items()}
        errors["velocity_error_mps"] *= scale
        clearance = COMMANDS["command_clearance_m"] + errors["height_error_m"]
        rows.append(
            {
                "time_s": (tick + 1) * DT,
                "x_m": x,
                "y_m": y,
                "z_m": GROUND[tick] + clearance,
                "ground_height_m": GROUND[tick],
                "clearance_m": clearance,
                "velocity_mps": COMMANDS["command_forward_velocity_mps"]
                + errors["velocity_error_mps"],
                "yaw_rate_rps": COMMANDS["command_yaw_rate_rps"] + errors["yaw_rate_error_rps"],
                **COMMANDS,
                **errors,
                "mechanical_power_w": POWER[tick],
                "action_mean_square": 0.0
                if zero
                else math.fsum(a * a for a in ACTIONS[tick]) / len(ACTIONS[tick]),
                "nonflat_now": FLAGS[tick],
            }
        )
    return rows


def hand_metrics(rows):
    """Expected endpoints from plain-Python sums, never from the audit itself."""
    n = len(rows)
    rms = lambda key: math.sqrt(math.fsum(row[key] ** 2 for row in rows) / n)
    return {
        "velocity_rmse_mps": rms("velocity_error_mps"),
        "yaw_rmse_rps": rms("yaw_rate_error_rps"),
        "height_rmse_m": rms("height_error_m"),
        "attitude_rmse_rad": math.sqrt(
            math.fsum(r["roll_error_rad"] ** 2 + r["pitch_error_rad"] ** 2 for r in rows) / n
        ),
        "mean_mechanical_power_w": math.fsum(r["mechanical_power_w"] for r in rows) / n,
        "action_rms": math.sqrt(math.fsum(r["action_mean_square"] for r in rows) / n),
    }


def hand_exposure(rows):
    n = len(rows)
    increments = [math.hypot(*STEPS_XY[tick]) for tick in range(n)]
    nonflat = [tick for tick in range(n) if FLAGS[tick]]
    cells = {
        (math.floor(rows[t]["x_m"] / 0.25), math.floor(rows[t]["y_m"] / 0.25)) for t in nonflat
    }
    return {
        "nonflat_steps": len(nonflat),
        "nonflat_fraction": len(nonflat) / n,
        "nonflat_now": bool(FLAGS[n - 1]),
        "path_m": math.fsum(increments),
        "nonflat_path_m": math.fsum(increments[tick] for tick in nonflat),
        "unique_nonflat_0p25m_cells": len(cells),
    }


def csv_bytes(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows([{key: repr(value) for key, value in row.items()} for row in rows])
    return stream.getvalue().encode()


def npz_bytes(rows):
    qpos = np.zeros((len(rows) + 1, 23))
    qpos[:, 3] = 1.0
    qpos[0, :3] = (-3.8, 0.0, 0.455)
    for index, row in enumerate(rows):
        qpos[index + 1, :3] = (row["x_m"], row["y_m"], row["z_m"])
    stream = io.BytesIO()
    actions = [
        ACTIONS[index] if row["action_mean_square"] else (0.0, 0.0)
        for index, row in enumerate(rows)
    ]
    np.savez(stream, qpos=qpos, actions=np.asarray(actions, dtype=np.float32))
    return stream.getvalue()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def write_source(directory, tree):
    manifest = {}
    for name in SOURCE_FILES:
        path = tree / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# Deliberately synthetic stand-in for {name}, not real source.\n")
        manifest[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    archive_path = directory / "source.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for name in manifest:
            archive.add(tree / name, arcname=name)
    write_json(directory / "source_consistency.json", {"unchanged": True, "changed": []})
    write_json(
        directory / "source.json",
        {
            "sha256": manifest,
            "archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        },
    )


def write_directory(root, *, policy, scale=1.0, short_roads=()):
    root.mkdir(parents=True)
    write_source(root, root.parent / f"{root.name}_tree")
    arguments = {
        "mode": "evaluate",
        "duration": DURATION,
        "split": "development",
        "baseline": "lqr",
        "source": "oracle",
        "history": 1,
        "measurement_delay": 0.0,
        "policy": policy,
    }
    write_json(
        root / "protocol.json",
        {
            "schema": "d1-command-locomotion-experiment-v1",
            "arguments": arguments,
            "quality_thresholds": QUALITY_LIMITS,
            "evaluation_seeds": {"development": list(SEEDS), "holdout": [909]},
            "terrain_splits": {"development": list(TERRAINS), "holdout": [TERRAINS[0]]},
        },
    )
    records = []
    for road, terrain in enumerate(TERRAINS):
        ticks = 2 if road in short_roads else 3
        rows = build_rows(ticks, scale, zero=policy is None)
        metrics, exposure = hand_metrics(rows), hand_exposure(rows)
        completed = ticks == round(DURATION / DT)
        for seed in SEEDS:
            case = f"road{road}_seed{seed}"
            (root / f"{case}.csv").write_bytes(csv_bytes(rows))
            (root / f"{case}.npz").write_bytes(npz_bytes(rows))
            write_json(
                root / f"{case}_episode.json",
                {
                    "terrain": terrain,
                    "provider": {"kind": arguments["source"]},
                    "baseline": arguments["baseline"],
                    "duration_s": DURATION,
                    "schedule": SCHEDULE,
                },
            )
            records.append(
                {
                    "case": case,
                    "seed": seed,
                    "terrain": terrain,
                    "executed_steps": ticks,
                    "duration_s": ticks * DT,
                    "terminal_reason": "time_limit" if completed else "fall_or_body_contact",
                    "completed": completed,
                    "quality_pass": completed
                    and all(metrics[k] <= limit for k, limit in QUALITY_LIMITS.items()),
                    "terrain_exposure": exposure,
                    **metrics,
                }
            )
    write_json(root / "evaluation.json", records)
    return root


def rewrite_json(path, mutate):
    value = json.loads(path.read_text())
    mutate(value)
    write_json(path, value)


def rewrite_csv(path, mutate):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        names, rows = reader.fieldnames, list(reader)
    mutate(rows)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def rewrite_npz(path, mutate):
    with np.load(path) as archive:
        arrays = {name: archive[name] for name in archive.files}
    mutate(arrays)
    np.savez(path, **arrays)


@pytest.fixture
def policy_dir(tmp_path):
    return write_directory(tmp_path / "policy_eval", policy="runs/lqr_seed24000/checkpoint.zip")


@pytest.fixture
def zero_dir(tmp_path):
    return write_directory(tmp_path / "zero_eval", policy=None, scale=2.0)


def test_audit_implementation_imports_no_simulator_or_trainer():
    roots = set()
    for node in ast.walk(ast.parse(AUDIT_PATH.read_text())):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            roots.add((node.module or "").split(".")[0])
    assert roots == {
        "__future__",
        "argparse",
        "csv",
        "hashlib",
        "json",
        "math",
        "tarfile",
        "pathlib",
        "numpy",
    }
    assert "__import__" not in AUDIT_PATH.read_text()


def test_audit_recomputes_endpoints_and_leaves_artifacts_untouched(policy_dir):
    before = {path.name: path.read_bytes() for path in sorted(policy_dir.iterdir())}
    report = audit_evaluation(policy_dir)
    assert report["source"]["file_count"] == 2
    assert len(report["cases"]) == 4
    case = next(item for item in report["cases"] if item["case"] == "road0_seed101")
    assert case["velocity_rmse_mps"] == pytest.approx(math.sqrt((0.04**2 + 0.08**2 + 0.04**2) / 3))
    assert case["yaw_rmse_rps"] == pytest.approx(math.sqrt((0.02**2 + 0.04**2 + 0.06**2) / 3))
    assert case["height_rmse_m"] == pytest.approx(math.sqrt((0.01**2 + 0.02**2 + 0.01**2) / 3))
    assert case["attitude_rmse_rad"] == pytest.approx(math.sqrt(0.0099 / 3))
    assert case["mean_mechanical_power_w"] == pytest.approx((12.0 + 15.0 + 18.0) / 3)
    assert case["action_rms"] == pytest.approx(math.sqrt((0.1 + 0.25 + 0.18) / 3))
    assert case["completed"] is True and case["quality_pass"] is True
    assert case["duration_s"] == pytest.approx(0.03)
    assert case["terrain_exposure"] == {
        "nonflat_steps": 2,
        "nonflat_fraction": pytest.approx(2 / 3),
        "nonflat_now": True,
        "path_m": pytest.approx(0.04),
        "nonflat_path_m": pytest.approx(0.03),
        "unique_nonflat_0p25m_cells": 1,
    }
    assert case["schedule"] == SCHEDULE
    assert {path.name: path.read_bytes() for path in sorted(policy_dir.iterdir())} == before


def test_duplicate_oracle_labels_collapse_to_two_roads(policy_dir, zero_dir):
    comparison = paired_deltas(audit_evaluation(policy_dir), audit_evaluation(zero_dir))
    assert [pair["paired_case_labels"] for pair in comparison["pairs"]] == [
        ["road0_seed101", "road0_seed202"],
        ["road1_seed101", "road1_seed202"],
    ]
    assert [pair["terrain"] for pair in comparison["pairs"]] == list(TERRAINS)
    assert comparison["bootstrap_ci"] is None
    for pair in comparison["pairs"]:
        assert pair["comparable_full_horizon"] is True
        assert pair["policy_quality_pass"] is True and pair["zero_quality_pass"] is False
        assert pair["delta_policy_minus_zero"]["velocity_rmse_mps"] == pytest.approx(
            math.sqrt(0.0096 / 3) - math.sqrt(0.0384 / 3)
        )
        assert pair["delta_policy_minus_zero"]["action_rms"] == pytest.approx(math.sqrt(0.53 / 3))


def test_incomplete_valid_fall_yields_no_full_horizon_delta(tmp_path, policy_dir):
    zero = write_directory(tmp_path / "zero_fall", policy=None, scale=2.0, short_roads=(1,))
    first, second = paired_deltas(audit_evaluation(policy_dir), audit_evaluation(zero))["pairs"]
    assert first["comparable_full_horizon"] is True
    assert first["delta_policy_minus_zero"]["velocity_rmse_mps"] < 0
    assert second["paired_case_labels"] == ["road1_seed101", "road1_seed202"]
    assert second["comparable_full_horizon"] is False
    assert second["delta_policy_minus_zero"] is None
    assert second["zero_completed"] is False and second["zero_quality_pass"] is False
    assert second["zero_terminal_reason"] == "fall_or_body_contact"
    assert second["zero_duration_s"] == pytest.approx(0.02)
    assert second["policy_duration_s"] == pytest.approx(0.03)
    assert second["zero_exposure"] == {
        "nonflat_steps": 1,
        "nonflat_fraction": pytest.approx(0.5),
        "nonflat_now": True,
        "path_m": pytest.approx(0.03),
        "nonflat_path_m": pytest.approx(0.02),
        "unique_nonflat_0p25m_cells": 1,
    }


@pytest.mark.parametrize(
    "key,value",
    (
        ("source", "imu_encoder_fusion"),
        ("history", 2),
        ("duration", 0.05),
        ("policy", "runs/other/checkpoint.zip"),
    ),
)
def test_incompatible_configuration_prevents_pairing(policy_dir, zero_dir, key, value):
    policy, zero = audit_evaluation(policy_dir), audit_evaluation(zero_dir)
    zero["arguments"][key] = value
    with pytest.raises(ValueError):
        paired_deltas(policy, zero)


def test_oracle_duplicates_must_be_byte_identical(policy_dir, zero_dir):
    rewrite_csv(
        policy_dir / "road0_seed202.csv",
        lambda rows: rows[0].update({"mechanical_power_w": "12.000"}),
    )
    policy = audit_evaluation(policy_dir)
    with pytest.raises(ValueError, match="not identical"):
        paired_deltas(policy, audit_evaluation(zero_dir))


def corrupt(directory, fault):
    records, csv_path = directory / "evaluation.json", directory / "road0_seed101.csv"
    npz_path, episode = directory / "road0_seed101.npz", directory / "road0_seed101_episode.json"
    if fault == "declared_metric":
        rewrite_json(records, lambda r: r[0].update({"action_rms": 0.5}))
    elif fault == "declared_quality":
        rewrite_json(records, lambda r: r[0].update({"quality_pass": False}))
    elif fault == "declared_steps":
        rewrite_json(records, lambda r: r[0].update({"executed_steps": 2}))
    elif fault == "incomplete_matrix":
        rewrite_json(records, lambda r: r.pop())
    elif fault == "terminal_reason":
        rewrite_json(records, lambda r: r[0].update({"terminal_reason": "battery_empty"}))
    elif fault == "exposure_value":
        rewrite_json(records, lambda r: r[0]["terrain_exposure"].update({"nonflat_path_m": 0.04}))
    elif fault == "exposure_type":
        rewrite_json(records, lambda r: r[0]["terrain_exposure"].update({"nonflat_steps": True}))
    elif fault == "clock":
        rewrite_csv(csv_path, lambda rows: rows[1].update({"time_s": "0.05"}))
    elif fault == "exposure_flag":
        rewrite_csv(csv_path, lambda rows: rows[2].update({"nonflat_now": "maybe"}))
    elif fault == "recorded_error":
        rewrite_csv(csv_path, lambda rows: rows[0].update({"velocity_error_mps": "0.05"}))
    elif fault == "negative_power":
        rewrite_csv(csv_path, lambda rows: rows[0].update({"mechanical_power_w": "-12.0"}))
    elif fault == "npz_action":
        rewrite_npz(npz_path, lambda a: a["actions"].__setitem__((0, 0), 0.3))
    elif fault == "npz_bounds":
        rewrite_npz(npz_path, lambda a: a["actions"].__setitem__((0, 0), 1.5))
    elif fault == "npz_pose":
        rewrite_npz(npz_path, lambda a: a["qpos"].__setitem__((2, 1), 0.5))
    elif fault == "npz_initial_pose":
        rewrite_npz(npz_path, lambda a: a["qpos"].__setitem__((0, 2), 0.5))
    else:
        rewrite_json(episode, lambda e: e.update({"terrain": dict(TERRAINS[1])}))


@pytest.mark.parametrize(
    "fault",
    (
        "declared_metric",
        "declared_quality",
        "declared_steps",
        "incomplete_matrix",
        "terminal_reason",
        "exposure_value",
        "exposure_type",
        "clock",
        "exposure_flag",
        "recorded_error",
        "negative_power",
        "npz_action",
        "npz_bounds",
        "npz_pose",
        "npz_initial_pose",
        "episode_terrain",
    ),
)
def test_corrupted_evaluation_artifacts_are_rejected(policy_dir, fault):
    corrupt(policy_dir, fault)
    with pytest.raises(ValueError):
        audit_evaluation(policy_dir)


@pytest.mark.parametrize(
    "fault",
    (
        "changed_source",
        "empty_manifest",
        "unsafe_manifest_name",
        "archive_digest",
        "member_digest",
        "traversal_member",
        "symlink_member",
        "symlink_artifact",
    ),
)
def test_source_provenance_faults_are_rejected_without_extraction(policy_dir, fault):
    archive_path, document = policy_dir / "source.tar.gz", policy_dir / "source.json"
    if fault == "changed_source":
        write_json(
            policy_dir / "source_consistency.json",
            {"unchanged": False, "changed": [SOURCE_FILES[0]]},
        )
    elif fault == "empty_manifest":
        rewrite_json(document, lambda d: d.update({"sha256": {}}))
    elif fault == "unsafe_manifest_name":
        rewrite_json(document, lambda d: d["sha256"].update({"../outside.py": "0" * 64}))
    elif fault == "member_digest":
        rewrite_json(document, lambda d: d["sha256"].update({SOURCE_FILES[0]: "1" * 64}))
    elif fault == "archive_digest":
        archive_path.write_bytes(archive_path.read_bytes() + b"\n")
    elif fault == "symlink_artifact":
        moved = policy_dir / "records.json"
        (policy_dir / "evaluation.json").rename(moved)
        (policy_dir / "evaluation.json").symlink_to(moved)
    else:
        payload = b"escaped\n"
        with tarfile.open(archive_path, "w:gz") as archive:
            for name in SOURCE_FILES:
                member = tarfile.TarInfo(name)
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))
            entry = tarfile.TarInfo("../outside.py" if fault == "traversal_member" else "linked.py")
            if fault == "symlink_member":
                entry.type, entry.linkname = tarfile.SYMTYPE, SOURCE_FILES[0]
            else:
                entry.size = len(payload)
            archive.addfile(entry, io.BytesIO(payload) if entry.size else None)
        rewrite_json(
            document,
            lambda d: d.update(
                {"archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest()}
            ),
        )
    with pytest.raises(ValueError):
        audit_evaluation(policy_dir)
    assert not (policy_dir.parent / "outside.py").exists()
    assert not (policy_dir / "linked.py").exists()


@pytest.mark.parametrize("fault", ("nan_literal", "duplicate_key"))
def test_nonfinite_or_duplicated_json_is_rejected(policy_dir, fault):
    path = policy_dir / "protocol.json"
    text = path.read_text()
    if fault == "nan_literal":
        text = text.replace('"duration": 0.03', '"duration": NaN')
    else:
        text = text.replace('"mode": "evaluate"', '"mode": "evaluate",\n    "mode": "replay"')
    assert "NaN" in text or text.count('"mode"') == 2
    path.write_text(text)
    with pytest.raises(ValueError):
        audit_evaluation(policy_dir)


def test_nonzero_actions_cannot_be_relabelled_as_zero_residual(policy_dir):
    rewrite_json(policy_dir / "protocol.json", lambda p: p["arguments"].update(policy=None))
    with pytest.raises(ValueError, match="zero residual"):
        audit_evaluation(policy_dir)


def test_eight_axis_controller_parameters_must_match_for_pairing(policy_dir, zero_dir):
    for directory, ki in ((policy_dir, 0.75), (zero_dir, 1.5)):
        gains = {"wheel_kp": 0.55, "wheel_ki": ki, "yaw_feedback_gain": 4.0}
        as_wheel_directory(directory, gains, V3, arguments=gains)
    policy, zero = audit_evaluation(policy_dir), audit_evaluation(zero_dir)
    assert len(policy["cases"]) == 4
    with pytest.raises(ValueError, match="controller parameters"):
        paired_deltas(policy, zero)


def read_case_names(directory):
    return [case["case"] for case in json.loads((directory / "evaluation.json").read_text())]


def as_wheel_directory(directory, gains, schema, *, arguments=None):
    rewrite_json(
        directory / "protocol.json",
        lambda p: p["arguments"].update(baseline="wheel_leg", **(arguments or {})),
    )
    for case in read_case_names(directory):

        def update_episode(episode):
            episode.update(baseline="wheel_leg", controller_schema=schema)
            if gains is not None:
                episode["controller_parameters"] = gains

        rewrite_json(directory / f"{case}_episode.json", update_episode)
        rewrite_npz(
            directory / f"{case}.npz", lambda a: a.update(actions=np.tile(a["actions"], (1, 4)))
        )


@pytest.mark.parametrize(
    "gains,schema,arguments",
    (
        (None, V2, {}),
        ({k: v for k, v in DEFAULT_GAINS.items() if "scale" not in k}, V2, {}),
        (
            {"wheel_kp": 0.55, "wheel_ki": 1.5, "yaw_feedback_gain": 4.0},
            V3,
            {"wheel_kp": 0.55, "wheel_ki": 1.5},
        ),
        (DEFAULT_GAINS, V2, {}),
        ({**DEFAULT_GAINS, "wheel_ki": 0}, V3, {"wheel_ki": 0}),
        ({**DEFAULT_GAINS, "leg_feedback_scale": 0.25}, V4, {"leg_feedback_scale": 0.25}),
        (
            {**DEFAULT_GAINS, "attitude_feedback_scale": 0.25},
            V4,
            {"attitude_feedback_scale": 0.25},
        ),
    ),
)
def test_wheel_records_normalize_without_rewriting_artifacts(policy_dir, gains, schema, arguments):
    as_wheel_directory(policy_dir, gains, schema, arguments=arguments)
    before = {path.name: path.read_bytes() for path in policy_dir.iterdir()}
    cases = audit_evaluation(policy_dir)["cases"]
    assert all(case["controller_parameters"] == {**DEFAULT_GAINS, **arguments} for case in cases)
    assert {path.name: path.read_bytes() for path in policy_dir.iterdir()} == before


@pytest.mark.parametrize(
    "gains,schema,arguments",
    (
        (None, V3, {"wheel_kp": 0.55}),
        (None, V4, {"leg_feedback_scale": 0.25}),
        (None, V2, {"wheel_kp": 0.55}),
        (DEFAULT_GAINS, V3, {}),
        (DEFAULT_GAINS, V4, {}),
        (DEFAULT_GAINS, None, {}),
        (DEFAULT_GAINS, "unknown", {}),
        (
            {"wheel_kp": 2.2, "wheel_ki": 3.0, "yaw_feedback_gain": 4.0},
            V4,
            {"leg_feedback_scale": 0.25},
        ),
        ({**DEFAULT_GAINS, "leg_feedback_scale": 0.25}, V2, {}),
        ({**DEFAULT_GAINS, "attitude_feedback_scale": 0.25}, V4, {"leg_feedback_scale": 0.25}),
        ({**DEFAULT_GAINS, "invented_gain": 1.0}, V2, {}),
        ({k: v for k, v in DEFAULT_GAINS.items() if k != "leg_feedback_scale"}, V2, {}),
        ([], V2, {}),
    ),
)
def test_schema_and_gain_record_corruption_is_rejected(policy_dir, gains, schema, arguments):
    as_wheel_directory(policy_dir, gains, schema, arguments=arguments)
    with pytest.raises(ValueError, match="controller"):
        audit_evaluation(policy_dir)


def test_explicit_null_gains_are_not_legacy_missing_gains(policy_dir):
    as_wheel_directory(policy_dir, None, V2)
    rewrite_json(
        policy_dir / "road0_seed101_episode.json",
        lambda e: e.update(controller_parameters=None),
    )
    with pytest.raises(ValueError, match="controller parameters must be an object"):
        audit_evaluation(policy_dir)


@pytest.mark.parametrize("location", ("arguments", "episode"))
@pytest.mark.parametrize(
    "key,value",
    (
        ("wheel_kp", 0),
        ("wheel_ki", -1),
        ("yaw_feedback_gain", True),
        ("leg_feedback_scale", 0),
        ("attitude_feedback_scale", -0.25),
        ("leg_feedback_scale", True),
        ("attitude_feedback_scale", "1"),
        ("leg_feedback_scale", [1.0]),
        ("attitude_feedback_scale", float("inf")),
        ("leg_feedback_scale", float("nan")),
    ),
)
def test_invalid_gain_scalars_are_rejected(policy_dir, location, key, value):
    as_wheel_directory(policy_dir, DEFAULT_GAINS, V2)
    if location == "arguments":
        rewrite_json(policy_dir / "protocol.json", lambda p: p["arguments"].update({key: value}))
    else:
        rewrite_json(
            policy_dir / "road0_seed101_episode.json",
            lambda e: e["controller_parameters"].update({key: value}),
        )
    with pytest.raises(ValueError):
        audit_evaluation(policy_dir)


def test_wheel_scales_must_match_for_pairing(policy_dir, zero_dir):
    for directory, scale in ((policy_dir, 0.25), (zero_dir, 0.5)):
        as_wheel_directory(
            directory,
            {**DEFAULT_GAINS, "leg_feedback_scale": scale},
            V4,
            arguments={"leg_feedback_scale": scale},
        )
    with pytest.raises(ValueError, match="paired controller parameters differ"):
        paired_deltas(audit_evaluation(policy_dir), audit_evaluation(zero_dir))


def test_legacy_three_gain_and_new_five_gain_records_can_pair(policy_dir, zero_dir):
    three_gains = {"wheel_kp": 0.55, "wheel_ki": 1.5, "yaw_feedback_gain": 4.0}
    as_wheel_directory(policy_dir, three_gains, V3, arguments=three_gains)
    as_wheel_directory(zero_dir, {**DEFAULT_GAINS, **three_gains}, V3, arguments=three_gains)
    pairs = paired_deltas(audit_evaluation(policy_dir), audit_evaluation(zero_dir))["pairs"]
    assert len(pairs) == 2
    assert all(pair["comparable_full_horizon"] for pair in pairs)


@pytest.fixture
def matrix_root(tmp_path):
    """Tiny phase-4-shaped archive, not trained policies or simulated evidence."""
    for baseline in ("lqr", "wheel_leg"):
        for seed in (24000, 25000, 26000):
            directory = tmp_path / "d1_v3_locomotion_ppo" / f"{baseline}_seed{seed}"
            directory.mkdir(parents=True)
            write_source(directory, directory.parent / f"{directory.name}_tree")
            (directory / "checkpoint.zip").write_bytes(b"synthetic checkpoint, never deserialize")
            sha = hashlib.sha256((directory / "checkpoint.zip").read_bytes()).hexdigest()
            write_json(
                directory / "training.json", {"num_timesteps": 32768, "checkpoint_sha256": sha}
            )
            write_json(
                directory / "checkpoint.json",
                {
                    "extra": {"num_timesteps": 32768},
                    "model_sha256": sha,
                    "controller_schema": V2,
                    "recorded_episode": {"controller_schema": V2},
                },
            )
            write_json(
                directory / "protocol.json",
                {
                    "arguments": {
                        "seed": seed,
                        "baseline": baseline,
                        "steps": 32768,
                        "source": "oracle",
                        "history": 1,
                        "workers": 4,
                    }
                },
            )
        for split in ("development", "holdout"):
            zero_label = "wheel" if baseline == "wheel_leg" and split == "development" else baseline
            zero = write_directory(
                tmp_path / f"d1_v3_locomotion_zero_{zero_label}_{split}", policy=None
            )
            evaluations = [zero]
            for seed in (24000, 25000, 26000):
                evaluations.append(
                    write_directory(
                        tmp_path / "d1_v3_locomotion_evaluation" / f"{baseline}_seed{seed}_{split}",
                        policy=f"runs/{baseline}_seed{seed}/checkpoint.zip",
                    )
                )
            for directory in evaluations:
                if baseline == "wheel_leg":
                    as_wheel_directory(directory, None, V2)

                def update_protocol(protocol, split=split):
                    protocol["arguments"]["split"] = split
                    protocol["evaluation_seeds"]["holdout"] = list(SEEDS)
                    protocol["terrain_splits"]["holdout"] = list(TERRAINS)

                rewrite_json(directory / "protocol.json", update_protocol)
    return tmp_path


def test_phase4_accepts_historical_unscaled_matrix(matrix_root):
    report = audit_matrix(matrix_root)
    assert len(report["training"]) == 6
    assert report["splits"]["holdout"]["wheel_leg"]["full_horizon_pairs"] == 6


@pytest.mark.parametrize(
    "target", ("training_cli", "checkpoint", "recorded_episode", "zero", "policy")
)
def test_phase4_rejects_bandwidth_changes_even_in_valid_v4_records(matrix_root, target):
    directory = matrix_root / "d1_v3_locomotion_ppo" / "wheel_leg_seed24000"
    if target == "training_cli":
        rewrite_json(
            directory / "protocol.json",
            lambda p: p["arguments"].update(leg_feedback_scale=0.25),
        )
    elif target in ("checkpoint", "recorded_episode"):

        def mutate_checkpoint(metadata):
            record = metadata if target == "checkpoint" else metadata["recorded_episode"]
            record.update(
                controller_schema=V4,
                controller_parameters={**DEFAULT_GAINS, "leg_feedback_scale": 0.25},
            )

        rewrite_json(directory / "checkpoint.json", mutate_checkpoint)
    else:
        directory = (
            matrix_root / "d1_v3_locomotion_zero_wheel_development"
            if target == "zero"
            else matrix_root / "d1_v3_locomotion_evaluation" / "wheel_leg_seed24000_development"
        )
        rewrite_json(
            directory / "protocol.json",
            lambda p: p["arguments"].update(leg_feedback_scale=0.25),
        )
        for case in read_case_names(directory):
            rewrite_json(
                directory / f"{case}_episode.json",
                lambda e: e.update(
                    controller_schema=V4,
                    controller_parameters={**DEFAULT_GAINS, "leg_feedback_scale": 0.25},
                ),
            )
    with pytest.raises(ValueError, match="controller schema|phase4 wheel parameters changed"):
        audit_matrix(matrix_root)


@pytest.mark.parametrize("field", ("success_rate_pct", "quality_pass"))
def test_unverified_or_missing_case_fields_are_rejected(policy_dir, field):
    def mutate(records):
        if field == "success_rate_pct":
            records[0][field] = 100.0
        else:
            del records[0][field]

    rewrite_json(policy_dir / "evaluation.json", mutate)
    with pytest.raises(ValueError, match="case fields"):
        audit_evaluation(policy_dir)


def test_clearance_cannot_disagree_with_ground_and_height(policy_dir):
    rewrite_csv(
        policy_dir / "road0_seed101.csv", lambda rows: rows[0].update(ground_height_m="0.03")
    )
    with pytest.raises(ValueError, match="clearance mismatch"):
        audit_evaluation(policy_dir)
