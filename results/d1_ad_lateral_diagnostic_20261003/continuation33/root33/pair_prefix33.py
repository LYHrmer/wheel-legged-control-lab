"""Pure online C33/C30 reset and first-200-control entry bridge."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from worker30 import identity


ARRAYS = ("qpos", "qvel", "act", "ctrl", "qacc_warmstart", "observation", "time")


def _prefix_rows(path):
    rows = []
    with gzip.open(path, "rt") as stream:
        for _, line in zip(range(200), stream):
            rows.append(json.loads(line))
    if len(rows) != 200:
        raise RuntimeError("C33 entry bridge lacks 200 real control rows")
    return rows


def _row(row):
    if (row["actor"] != "B" or row["policy_predict_called"] is not True
            or row["skill30"] is not None or row["macro_events30"]):
        raise RuntimeError("C33 entry bridge has altered pre-side authority")
    chosen = {key: row[key] for key in ("input_observation99", "raw_action16",
        "action16", "policy_input_action", "raw_command", "controller_handoff_receipt",
        "side_start_receipt")}
    chosen["info"] = {key: row["info"][key] for key in
        ("controller_record", "raw_operator_command", "consumed_command", "servo_receipt")}
    return chosen


def _digest_rows(rows):
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(_row(row), sort_keys=True, separators=(",", ":"),
                                 allow_nan=False).encode()+b"\n")
    return digest.hexdigest()


def pair_prefix33(baseline_report_path, new_folder, sources):
    report_path = Path(baseline_report_path).resolve(strict=True)
    if sources.get(str(report_path)) != identity(report_path):
        raise RuntimeError("C33 baseline report left source closure")
    report = json.loads(report_path.read_text())
    old = Path(report["run"]).resolve(strict=True)/"episode_0"
    new = Path(new_folder)
    for name in ("initial_state.npz", "states.npz", "controls.jsonl.gz"):
        source = old/name
        if sources.get(str(source)) != identity(source):
            raise RuntimeError("C33 original entry archive left source closure: "+str(source))
    arrays = {}
    with np.load(old/"initial_state.npz") as a, np.load(new/"initial_state.npz") as b:
        for key in ARRAYS:
            arrays["initial_"+key] = (a[key].dtype == b[key].dtype and
                a[key].shape == b[key].shape and a[key].tobytes() == b[key].tobytes())
    with np.load(old/"states.npz") as a, np.load(new/"states.npz") as b:
        for key in ARRAYS:
            arrays["prefix_"+key] = (len(a[key]) > 200 and len(b[key]) > 200
                and a[key][:201].dtype == b[key][:201].dtype
                and a[key][:201].shape == b[key][:201].shape
                and a[key][:201].tobytes() == b[key][:201].tobytes())
    old_sha = _digest_rows(_prefix_rows(old/"controls.jsonl.gz"))
    new_sha = _digest_rows(_prefix_rows(new/"controls.jsonl.gz"))
    result = {"schema": "d1-c33-online-prefix-bridge-v1", "baseline": str(report_path),
              "baseline_identity": identity(report_path), "array_checks": arrays,
              "baseline_preparation_sha256": old_sha,
              "new_preparation_sha256": new_sha,
              "passed": all(arrays.values()) and old_sha == new_sha}
    if not result["passed"]:
        raise RuntimeError("C33 original/C33 actual reset and 200-control entry differ")
    return result
