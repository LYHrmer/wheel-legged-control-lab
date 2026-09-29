"""Import-boundary regressions; never create a model or call physics."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

from floor_bridge16 import probe_import_isolated
from offline_floor16 import run as offline_run

HERE = Path(__file__).resolve().parent
W = HERE.parent


def test_loaded_torch_blocks_direct_reader_but_isolated_probe_succeeds(
    monkeypatch, tmp_path,
):
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    source = W / "verify_course_e_08_03.py"
    spec = importlib.util.spec_from_file_location("direct_reader_forbidden16", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with pytest.raises(RuntimeError, match="physics/policy already imported"):
        spec.loader.exec_module(module)

    artifact_dir = tmp_path / "probe"
    result = probe_import_isolated(artifact_dir=artifact_dir)
    receipt = json.loads((artifact_dir / "bridge_receipt.json").read_text())
    assert result["probe_import_passed"] is True
    assert result["forbidden_modules_loaded"] is False
    assert receipt["exit_code"] == 0
    assert receipt["same_process_group"] is True
    assert receipt["loader_injection_removed"] is True
    assert receipt["model_loaded_by_bridge"] is False


def test_offline_entry_refuses_loader_injection_and_loaded_model(monkeypatch):
    monkeypatch.setenv("LD_PRELOAD", "/synthetic/never-loaded.so")
    with pytest.raises(RuntimeError, match="loader injection"):
        offline_run(floor=None, probe_import=True)
    monkeypatch.delenv("LD_PRELOAD")
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    with pytest.raises(RuntimeError, match="model/physics module already loaded"):
        offline_run(floor=None, probe_import=True)
