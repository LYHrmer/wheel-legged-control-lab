import json
from pathlib import Path
from zipfile import ZipFile

from scripts.audit_terrain_preview85 import audit

ASSETS = Path(__file__).resolve().parents[1] / "docs/assets/project85"


def test_published_recordings_reconstruct_the_three_preregistered_decisions(tmp_path):
    with ZipFile(ASSETS / "recordings.zip") as archive:
        for member in archive.namelist():
            if member.startswith("trials/"):
                archive.extract(member, tmp_path)
    trials = tmp_path / "trials"
    report = audit(trials / "sensing", trials / "action", trials / "summary.json", trials / "archive")
    published = json.loads((ASSETS / "independent_audit.json").read_text())
    for field in ("audit", "sensor_passed", "response_passed", "complete_040m_windows",
                  "joint_gate_passed", "total_controls", "total_native_steps", "historic_counts",
                  "correct_cases_by_angle", "first_true_floor_windows_by_angle", "heldout_falls"):
        assert report[field] == published[field]
    assert report["sensor_passed"] and not report["response_passed"]
    assert report["complete_040m_windows"] == 0 and not report["joint_gate_passed"]
    for name, case in report["development"].items():
        assert case["checks"] == published["development"][name]["checks"]
