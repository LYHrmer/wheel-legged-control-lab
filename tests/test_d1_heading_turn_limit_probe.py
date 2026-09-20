"""Archive lifecycle regression without instantiating or stepping a plant."""
import gzip
import json

import pytest

from scripts import probe_d1_heading_turn_limit as probe


def test_completed_g1_manifest_is_preserved_and_diagnostics_get_complete_manifest(tmp_path, monkeypatch):
    original_constructor = probe.g1.D1HeadingTrackingEnv
    case = {"name": "stationary_turn_left_hold"}

    def fake_completed_run(output, _case, _model, _gates):
        output.mkdir()
        row = {"body_yaw_rate_before_rps": .1, "wheel_fit_yaw_before_rps": .4,
               "contacts": {"contacts": [], "total_wrench_world_6": [0.]*6}}
        with gzip.open(output / "turn_diagnostics.jsonl.gz", "xt") as stream:
            for _ in range(800):
                stream.write(json.dumps(row) + "\n")
        probe.write_json(output / "manifest.json", {"original": "retain"})
        probe.write_json(output / "turn_execution_receipt.json", {"actual_physics_substeps_from_clock": 4000})
        return {"gates": {"passed": False}, "heading_peak_rad": .25,
                "actual_observed_physics_substeps": 4000}, [{}]*800

    monkeypatch.setattr(probe.g1, "run_episode", fake_completed_run)
    output = tmp_path / "episode"
    summary, _ = probe.run_instrumented(output, case, "limit0p6", {})
    assert summary["actual_control_transitions"] == 800
    assert json.loads((output / "manifest.json").read_text()) == {"original": "retain"}
    complete = json.loads((output / "complete_manifest.json").read_text())
    assert "turn_execution_receipt.json" in complete
    assert "turn_diagnostics.jsonl.gz" in complete
    assert probe.g1.D1HeadingTrackingEnv is original_constructor


def test_constructor_binding_restored_on_failure(tmp_path, monkeypatch):
    original_constructor = probe.g1.D1HeadingTrackingEnv

    def failed(*args):
        raise RuntimeError("synthetic prephysics failure")

    monkeypatch.setattr(probe.g1, "run_episode", failed)
    with pytest.raises(RuntimeError, match="synthetic prephysics"):
        probe.run_instrumented(tmp_path, {}, "limit0p6", {})
    assert probe.g1.D1HeadingTrackingEnv is original_constructor
