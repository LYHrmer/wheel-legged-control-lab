"""Source-admission rejection cases without reading the large runtime closure."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/audit_b22_sources50.py"
SPEC = importlib.util.spec_from_file_location("sources50_test_module", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def test_exact_single_approved_change():
    assert audit.classify_change(audit.LIBXTST_PATH, audit.LIBXTST_OLD,
                                 audit.LIBXTST_NEW) == "approved_libxtst_replacement"
    assert audit.classify_change(audit.LIBXTST_PATH, audit.LIBXTST_OLD,
                                 audit.LIBXTST_OLD) == "unchanged"


@pytest.mark.parametrize("path,old,new", [
    ("/unexpected/lib.so", audit.LIBXTST_OLD, audit.LIBXTST_NEW),
    (audit.LIBXTST_PATH, {**audit.LIBXTST_OLD, "sha256": "0" * 64}, audit.LIBXTST_NEW),
    (audit.LIBXTST_PATH, audit.LIBXTST_OLD, {**audit.LIBXTST_NEW, "bytes": 1}),
    (audit.LIBXTST_PATH, audit.LIBXTST_OLD, {**audit.LIBXTST_NEW, "sha256": "f" * 64}),
])
def test_other_changes_are_rejected(path, old, new):
    assert audit.classify_change(path, old, new) == "unexpected_change"


@pytest.fixture
def small_closure(tmp_path, monkeypatch):
    source = tmp_path / "source.bin"
    source.write_bytes(b"pinned source\0\xff")
    manifest = tmp_path / "upstream.json"
    manifest.write_text(json.dumps({"source_hashes": {str(source): audit.file_identity(source)}}))
    monkeypatch.setattr(audit, "UPSTREAM_PATH", manifest)
    monkeypatch.setattr(audit, "UPSTREAM_SHA256", hashlib.sha256(manifest.read_bytes()).hexdigest())
    monkeypatch.setattr(audit, "UPSTREAM_COUNT", 1)
    return tmp_path, source, manifest


def test_complete_audit_is_independently_checked(small_closure):
    root, source, _ = small_closure
    output = root / "attempt01"
    report = audit.run_audit(output)
    assert report["status"] == "COMPLETE"
    assert audit.verify_audit50(output / "report.json", rehash=True) == report
    source.write_bytes(b"source changed after audit")
    with pytest.raises(ValueError, match="current source changed"):
        audit.verify_audit50(output / "report.json", rehash=True)


def test_changed_source_is_retained_and_refused(small_closure):
    root, source, _ = small_closure
    source.write_bytes(b"unexpected source")
    output = root / "rejected"
    report = audit.run_audit(output)
    assert report["status"] == "REJECTED"
    assert report["errors"][0]["path"] == str(source)
    with pytest.raises(ValueError, match="not complete"):
        audit.verify_audit50(output / "report.json")
    with pytest.raises(FileExistsError):
        audit.run_audit(output)


def test_forged_success_and_forgotten_sources_do_not_pass(small_closure):
    root, _, _ = small_closure
    output = root / "forged"
    report = audit.run_audit(output)
    report["current_source_hashes"] = {}
    (output / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="omitted or added"):
        audit.verify_audit50(output / "report.json")


def test_ledger_tampering_is_rejected(small_closure):
    root, _, _ = small_closure
    output = root / "tampered"
    audit.run_audit(output)
    (output / "sources.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="ledger changed"):
        audit.verify_audit50(output / "report.json")


def test_upstream_change_is_not_rebound(small_closure):
    root, _, manifest = small_closure
    manifest.write_text("{}")
    output = root / "bad_upstream"
    report = audit.run_audit(output)
    assert report["status"] == "INCOMPLETE"
    assert "upstream v27 identity changed" in report["errors"][0]["error"]
