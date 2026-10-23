"""Pure development-diagnostic and exact evaluation-derivation boundaries."""
import ast
import builtins
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sigma50_eval_tests", ROOT / "scripts/eval_b22_sigma50.py")
e = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = e
spec.loader.exec_module(e)

ARMS = {f"{label}_s{seed % 100}": {"seed": seed, "sigma_multiplier": multiplier}
        for seed in (201051, 201061, 201071) for label, multiplier in (("baseline", 1.), ("candidate", .5))}


def records(baseline=(.8, .9, 1.), candidate=(.82, .86, .9)):
    rows = {}
    for index, seed in enumerate((201051, 201061, 201071)):
        for name, values in (("baseline", baseline), ("candidate", candidate)):
            arm = f"{name}_s{seed % 100}"
            rows[arm] = {"status": "COMPLETE", "training_arm": arm, "training_seed": seed,
                         "diagnostic": {"primary_actor": "mirror50", "primary_defined": True,
                                        "R": {"mirror50": values[index]}}}
    return rows


def test_three_pair_direction_is_descriptive_and_does_not_grant_qualification():
    value = e.compare_diagnostics50(records(), ARMS)
    assert value["direction_pass"] is True and value["n_complete_pairs"] == 3
    assert value["range"]["baseline"] == pytest.approx(.2)
    assert value["range"]["candidate"] == pytest.approx(.08)
    assert value["qualification_granted"] is False
    assert value["statistical_significance_claimed"] is False


def test_narrower_range_with_worse_mean_is_not_a_pass():
    value = e.compare_diagnostics50(records(candidate=(.93, .94, .95)), ARMS)
    assert value["range"]["candidate"] < value["range"]["baseline"]
    assert value["direction_pass"] is False


@pytest.mark.parametrize("failure", ["INCOMPLETE", "FAILED", "NOT_RUN"])
def test_failed_partial_or_missing_pair_is_not_replaced_or_dropped(failure):
    data = records()
    if failure == "NOT_RUN":
        data.pop("candidate_s71")
    else:
        data["candidate_s71"]["status"] = failure
    value = e.compare_diagnostics50(data, ARMS)
    assert value["status"] == "INCOMPLETE" and value["direction_pass"] is None
    assert value["n_complete_pairs"] == 2 and len(value["rows"]) == 6
    assert value["mean"] is None and value["range"] is None
    assert next(row for row in value["rows"] if row["arm"] == "candidate_s71")["status"] == failure


def test_foreign_seed_cannot_fill_a_registered_slot():
    data = records()
    data["candidate_s71"]["training_seed"] = 201072
    assert e.compare_diagnostics50(data, ARMS)["direction_pass"] is None


def task_scores():
    cases = tuple("task" + str(i) for i in range(6))
    rows = [{"case_id": case, "experiment_actor": actor, "completed_controls": 100,
             "windows": {"control_cap": 100}, "rl_terms": {"sse_total": 10. if actor == "zero" else 8.}}
            for case in cases for actor in e.ACTORS]
    return cases, rows


def test_primary_r_uses_full_fixed_drive_pool():
    cases, rows = task_scores()
    value = e.descriptive50(rows, cases)
    assert value["pooled_drive_sse"]["zero"] == 60
    assert value["R"]["mirror50"] == pytest.approx(.8)
    rows[0]["completed_controls"] = 99
    value = e.descriptive50(rows, cases)
    assert value["primary_defined"] is False and value["R"]["mirror50"] is None


def test_raw_secondary_cannot_replace_or_screen_primary_mirror():
    cases, rows = task_scores()
    next(row for row in rows if row["experiment_actor"] == "raw50")["completed_controls"] = 99
    value = e.descriptive50(rows, cases)
    assert value["primary_defined"] is True and value["R"]["raw50"] is None
    assert value["all_original_cases_full_horizon"] is False


def test_zero_denominator_or_duplicate_task_cannot_manufacture_a_ratio():
    cases, rows = task_scores()
    for row in rows:
        if row["experiment_actor"] == "zero":
            row["rl_terms"]["sse_total"] = 0.
    assert e.descriptive50(rows, cases)["primary_defined"] is False
    rows[-1] = dict(rows[0])
    with pytest.raises(ValueError, match="matrix"):
        e.descriptive50(rows, cases)


@pytest.mark.parametrize("kind,reading", [("recipes", False), ("heldout", False), ("score", True), ("eval", True)])
def test_evaluation_derivation_is_bounded_to_declared_identity_nodes(kind, reading):
    tree, proof = e.derive50(kind, reading=reading)
    assert isinstance(tree, ast.Module) and proof["inverse_ast_exact"] is True
    compile(tree, "<sigma50-static-derivation>", "exec")
    if kind == "score":
        functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
        # The original contribution function still delegates to frozen c18 math.
        calls = [node for node in ast.walk(functions["score_pairs"]) if isinstance(node, ast.Call)]
        assert any(isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                   and node.func.value.id == "c18" and node.func.attr == "score_pairs" for node in calls)


def test_numpy_paths_are_configured_before_import_and_origin_is_checked(monkeypatch, tmp_path):
    events = []
    path = tmp_path / "numpy.py"
    module = SimpleNamespace(__file__=str(path))
    expected = {"bytes": 5, "sha256": "a" * 64}
    real_import = builtins.__import__

    def import_hook(name, *args, **kwargs):
        if name == "numpy":
            assert events == ["configure"]
            events.append("import")
            return module
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(e, "configure_paths50", lambda: events.append("configure"))
    monkeypatch.setattr(builtins, "__import__", import_hook)
    monkeypatch.setattr(e, "identity", lambda _path: events.append("identity") or expected)
    assert e.configured_numpy50({str(path): expected}) is module
    assert events == ["configure", "import", "identity"]


def test_foreign_already_loaded_numpy_is_rejected_after_configuration(monkeypatch, tmp_path):
    module = SimpleNamespace(__file__=str(tmp_path / "foreign_numpy.py"))
    real_import = builtins.__import__

    def import_hook(name, *args, **kwargs):
        return module if name == "numpy" else real_import(name, *args, **kwargs)

    monkeypatch.setattr(e, "configure_paths50", lambda: None)
    monkeypatch.setattr(builtins, "__import__", import_hook)
    with pytest.raises(ValueError, match="current closure"):
        e.configured_numpy50({str(tmp_path / "registered_numpy.py"): {"bytes": 1, "sha256": "a" * 64}})


def test_export_wrong_dependency_fails_before_any_model_load(monkeypatch, tmp_path):
    monkeypatch.delenv("LD_PRELOAD", raising=False)
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    plan = {"source_hashes": {}}
    monkeypatch.setattr(e, "verified_plan50", lambda _path: plan)
    monkeypatch.setattr(e, "identity", lambda _path: {"bytes": 1, "sha256": "a" * 64})
    monkeypatch.setattr(e, "configure_paths50", lambda: None)
    monkeypatch.setattr(e, "load", lambda _name, _path: SimpleNamespace(SIMULATORS=set(), block_imports=lambda _: None))

    def refused(_sources):
        raise ValueError("foreign dependency origin")

    monkeypatch.setattr(e, "configured_numpy50", refused)
    result = e.export50(tmp_path)
    assert result["status"] == "INCOMPLETE"
    assert result["error"] == "ValueError: foreign dependency origin"
    assert result["ppo_load_attempts"] == result["torch_load_attempts"] == result["sb3_predict_attempts"] == 0
    assert (tmp_path / "export/report.json").is_file()


def test_direction_does_not_hide_failed_coverage_or_contribution():
    data = records()
    data["candidate_s51"].update(training_coverage_valid=False,
                                  training_coverage={"coverage_valid": False, "missing": "slot"},
                                  contribution={"mirror50": {"passed": False}})
    result = e.compare_diagnostics50(data, ARMS)
    row = next(item for item in result["rows"] if item["arm"] == "candidate_s51")
    assert row["training_coverage_valid"] is False
    assert row["training_coverage"]["missing"] == "slot"
    assert row["contribution"]["mirror50"]["passed"] is False
    assert result["direction_pass"] is True and result["qualification_granted"] is False
