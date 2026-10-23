"""Audit an independent C50 source closure without changing upstream v27.

Only the exact libXtst replacement declared in the approved proposal is allowed.
This standard-library tool neither imports learning libraries nor constructs models.
Run its CLI inside the serial bounded systemd job used for other C50 operations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import signal
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_PATH = ROOT / "runtime/d1_b22/runtime_manifest_v27.json"
UPSTREAM_SHA256 = "d3c38429362e82745179f2c526b2ee52d805526df390d52346c805e5a51fff09"
UPSTREAM_COUNT = 4072
LIBXTST_PATH = "/usr/lib/x86_64-linux-gnu/libXtst.so.6.1.0"
LIBXTST_OLD = {"bytes": 27128, "sha256":
               "0c81be070f7857bdcab6bd528161d1049c374f1237aabf76d92e097b2d8ba16a"}
LIBXTST_NEW = {"bytes": 27128, "sha256":
               "4198b32dc9fba7dfc0ee2bb1b6bc85067c89adfc68d0279059ad82c207cc3c35"}
SCHEMA = "b22-independent-source-audit50-v1"


def file_identity(path: str | Path) -> dict:
    """Stream a file, rejecting a changed identity while it is being read."""
    path = Path(path)
    resolved = path.resolve(strict=True)
    before = path.stat()
    digest, size = hashlib.sha256(), 0
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
            size += len(block)
    after = path.stat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if (resolved != path.resolve(strict=True) or size != before.st_size or
            any(getattr(before, key) != getattr(after, key) for key in fields)):
        raise ValueError(f"source changed while hashing: {path}")
    return {"bytes": size, "sha256": digest.hexdigest()}


def classify_change(path: str, upstream: dict, current: dict) -> str:
    if upstream == current:
        return "unchanged"
    if path == LIBXTST_PATH and upstream == LIBXTST_OLD and current == LIBXTST_NEW:
        return "approved_libxtst_replacement"
    return "unexpected_change"


def upstream_manifest() -> tuple[dict, dict]:
    identity = file_identity(UPSTREAM_PATH)
    if identity["sha256"] != UPSTREAM_SHA256:
        raise ValueError("upstream v27 identity changed")
    value = json.loads(UPSTREAM_PATH.read_text())
    if len(value["source_hashes"]) != UPSTREAM_COUNT:
        raise ValueError("upstream source count changed")
    return value, identity


def verify_audit50(path: str | Path, *, rehash: bool = False) -> dict:
    """Validate a saved audit, optionally rehashing every admitted current input.

    The fixed upstream identity and exact per-file difference policy are checked
    independently of the report's status. Callers must also bind this report's
    file identity in their own registration before using it as an input.
    """
    path = Path(path)
    report = json.loads(path.read_text())
    upstream, identity = upstream_manifest()
    if report.get("schema") != SCHEMA or report.get("status") != "COMPLETE":
        raise ValueError("source audit is not complete")
    if (report.get("upstream_manifest") != str(UPSTREAM_PATH) or
            report.get("upstream_identity") != identity or
            report.get("audit_script_identity") != file_identity(__file__)):
        raise ValueError("source audit producer or upstream identity changed")
    current = report["current_source_hashes"]
    original = upstream["source_hashes"]
    if set(current) != set(original):
        raise ValueError("source audit omitted or added upstream inputs")
    accepted = []
    for name in sorted(original):
        kind = classify_change(name, original[name], current[name])
        if kind == "unexpected_change":
            raise ValueError(f"unregistered source change: {name}")
        if kind != "unchanged":
            accepted.append({"path": name, "upstream": original[name],
                             "current": current[name], "classification": kind})
        if rehash and file_identity(name) != current[name]:
            raise ValueError(f"current source changed since audit: {name}")
    if (report.get("accepted_differences") != accepted or report.get("errors") != [] or
            report.get("checked_entries") != UPSTREAM_COUNT or
            report.get("total_bytes") != sum(item["bytes"] for item in current.values())):
        raise ValueError("source audit summary differs from its source identities")
    ledger = path.parent / "sources.jsonl"
    if report.get("ledger_identity") != file_identity(ledger):
        raise ValueError("source audit ledger changed")
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    if len(rows) != UPSTREAM_COUNT or [row["path"] for row in rows] != sorted(original):
        raise ValueError("source audit ledger coverage/order changed")
    for row in rows:
        name = row["path"]
        if (row.get("upstream") != original[name] or row.get("current") != current[name] or
                row.get("classification") != classify_change(name, original[name], current[name])):
            raise ValueError(f"source audit ledger differs from registry: {name}")
    return report


def run_audit(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"schema": SCHEMA, "status": "INCOMPLETE", "upstream_manifest": str(UPSTREAM_PATH),
              "audit_script_identity": file_identity(__file__), "upstream_identity": None,
              "current_source_hashes": {}, "accepted_differences": [], "errors": [],
              "checked_entries": 0, "total_bytes": 0, "training_controls": 0,
              "model_constructions": 0, "policy_calls": 0, "physics_steps": 0,
              "retry_permitted": False}
    ledger = output / "sources.jsonl"
    try:
        upstream, report["upstream_identity"] = upstream_manifest()
        with ledger.open("x") as stream:
            for name, previous in sorted(upstream["source_hashes"].items()):
                row = {"path": name, "upstream": previous}
                try:
                    current = file_identity(name)
                    kind = classify_change(name, previous, current)
                    row.update(current=current, classification=kind)
                    report["current_source_hashes"][name] = current
                    report["total_bytes"] += current["bytes"]
                    if kind == "unexpected_change":
                        report["errors"].append(row.copy())
                    elif kind != "unchanged":
                        report["accepted_differences"].append(row.copy())
                except (OSError, ValueError) as exc:
                    row.update(classification="unreadable", error=f"{type(exc).__name__}: {exc}")
                    report["errors"].append(row.copy())
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                stream.flush()
                report["checked_entries"] += 1
        if file_identity(UPSTREAM_PATH) != report["upstream_identity"]:
            raise ValueError("upstream manifest changed during audit")
        report["status"] = "REJECTED" if report["errors"] else "COMPLETE"
    except BaseException as exc:  # noqa: BLE001 -- preserve interrupted source checks
        report["errors"].append({"error": f"{type(exc).__name__}: {exc}"})
    finally:
        report["elapsed_s"] = time.monotonic() - started
        report["ledger_identity"] = file_identity(ledger) if ledger.exists() else None
        with (output / "report.json").open("x") as stream:
            json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    def interrupted(signum, _frame):
        raise RuntimeError(f"source audit received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    report = run_audit(args.output)
    print(json.dumps({key: report[key] for key in
                      ("status", "checked_entries", "total_bytes", "accepted_differences",
                       "errors", "elapsed_s")}, indent=2), flush=True)
    return int(report["status"] != "COMPLETE")


if __name__ == "__main__":
    raise SystemExit(main())
