"""Tighter read-only admission for new83 jobs; frozen42 remains unchanged.

PSI thresholds are conservative new engineering limits, not a predictor or
replication of systemd-oomd. No system setting, unrelated group or process is
changed. Available-byte admission/reserve are still enforced by the supervisor.
"""

import math
from pathlib import Path

from scripts.resource_guard42 import ResourceGuardError, _directory, _read, _scope, check_memory

STARTUP_FULL_AVG10_MAX = 1.0
RUNTIME_FULL_AVG10_MAX = 10.0


def parse_pressure(text):
    result = {}
    for line in text.splitlines():
        items = line.split()
        if len(items) != 5 or items[0] not in ("some", "full") or items[0] in result:
            raise ValueError("invalid/duplicate PSI record")
        fields = {}
        for item in items[1:]:
            if item.count("=") != 1:
                raise ValueError("invalid PSI field")
            key, value = item.split("=")
            if key in fields or key not in ("avg10", "avg60", "avg300", "total"):
                raise ValueError("unknown/duplicate PSI field")
            if key == "total":
                if not value.isascii() or not value.isdigit():
                    raise ValueError("PSI total must be a nonnegative integer")
                fields[key] = int(value)
            else:
                number = float(value)
                if not math.isfinite(number) or not 0 <= number <= 100:
                    raise ValueError("PSI average outside [0,100]")
                fields[key] = number
        if set(fields) != {"avg10", "avg60", "avg300", "total"}:
            raise ValueError("incomplete PSI fields")
        result[items[0]] = fields
    if set(result) != {"some", "full"}:
        raise ValueError("both some/full PSI records required")
    if any(
        result["full"][key] > result["some"][key] for key in ("avg10", "avg60", "avg300", "total")
    ):
        raise ValueError("full PSI cannot exceed some PSI")
    return result


def read_pressures(snapshot, cgroup_root=Path("/sys/fs/cgroup")):
    check_memory(snapshot)
    root = Path(cgroup_root)
    if not root.is_absolute() or root.resolve(strict=True) != root:
        raise ValueError("canonical cgroup root required")
    result = {}
    for row in snapshot["raw_ancestors"]:
        path = row["path"]
        directory = root if path == "/" else _directory(root, _scope(path))
        before = directory.stat()
        pressure = parse_pressure(_read(directory / "memory.pressure"))
        after = directory.stat()
        if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
            raise ValueError("pressure scope identity changed")
        result[path] = pressure
    return result


def check_pressure(snapshot, pressures, *, startup=False):
    check_memory(snapshot)
    errors = []
    required = {r["path"] for r in snapshot["raw_ancestors"]}
    if set(pressures) != required:
        errors.append("pressure coverage must include owned and every ancestor")
    ceiling = STARTUP_FULL_AVG10_MAX if startup else RUNTIME_FULL_AVG10_MAX
    for path, pressure in pressures.items():
        try:
            value = pressure["full"]["avg10"]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError("invalid full avg10")
            if not 0 <= value <= 100 or value >= ceiling:
                errors.append(f"pressure {path} full.avg10={value} >= {ceiling}")
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"pressure {path}: {exc}")
    for event in ("high", "max"):
        value = snapshot["owned"]["events"].get(event)
        if type(value) is not int or value != 0:
            errors.append(f"owned memory.{event} event is nonzero or unknown")
    if errors:
        diagnostic = dict(
            snapshot,
            errors=list(snapshot["errors"]) + errors,
            pressures=pressures,
            pressure_stage="startup" if startup else "runtime",
        )
        raise ResourceGuardError(diagnostic)
    return snapshot
