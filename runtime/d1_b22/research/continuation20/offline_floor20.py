"""Clean-process floor readback entrypoint; never loads a model or engine."""

from __future__ import annotations

import argparse
import hashlib
import importlib.abc
import json
import os
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
W = HERE.parent
FORBIDDEN = frozenset((
    "mujoco", "glfw", "torch", "stable_baselines3", "gym", "gymnasium",
    "engine_binding", "wheel_legged_control",
))


class _ForbidModelPhysicsImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: object = None,
                  target: object = None) -> None:
        if fullname.partition(".")[0] in FORBIDDEN:
            raise RuntimeError("offline reader attempted model/physics import: " + fullname)
        return None


def _assert_clean(label: str) -> None:
    if any(name in os.environ for name in ("LD_PRELOAD", "LD_LIBRARY_PATH")):
        raise RuntimeError(label + ": loader injection environment is present")
    loaded = sorted(name for name in sys.modules
                    if name.partition(".")[0] in FORBIDDEN)
    if loaded:
        raise RuntimeError(label + ": model/physics module already loaded: "
                           + ",".join(loaded[:5]))
    try:
        mappings = Path("/proc/self/maps").read_text(encoding="utf-8").lower()
    except OSError as error:
        raise RuntimeError(label + ": cannot inspect loaded native libraries") from error
    if "libmujoco" in mappings or "libepa01_engine" in mappings:
        raise RuntimeError(label + ": model/physics native library is mapped")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_exclusive(path: Path, value: dict[str, Any]) -> None:
    if not path.is_absolute() or not path.parent.is_dir():
        raise ValueError("offline result needs an absolute path in an existing folder")
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def run(*, floor: Path | None, probe_import: bool) -> dict[str, Any]:
    _assert_clean("before reader import")
    sys.meta_path.insert(0, _ForbidModelPhysicsImports())
    for directory in (HERE, W, W / "continuation18", W / "continuation15", W / "continuation13"):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    import read_eval20

    _assert_clean("after reader import")
    reader = Path(read_eval20.__file__).resolve(strict=True)
    if reader != HERE / "read_eval20.py":
        raise RuntimeError("offline reader resolved outside frozen continuation20")
    if probe_import:
        if floor is not None:
            raise ValueError("import probe cannot request a floor")
        return {"schema": "d1-stage20-offline-import-probe-v1",
                "probe_import_passed": True,
                "reader_source": str(reader),
                "reader_sha256": _sha256(reader),
                "forbidden_modules_loaded": False,
                "model_loaded": False, "physics_called": False}
    if floor is None or not floor.is_absolute():
        raise ValueError("offline floor requires an absolute folder")
    result = read_eval20.read_floor(floor)
    _assert_clean("after floor readback")
    if type(result.get("numeric_gate_passed")) is not bool:
        raise RuntimeError("offline floor reader returned no numeric gate")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--floor", type=Path)
    choice.add_argument("--probe-import", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.floor is not None and args.output is None:
        parser.error("--floor requires --output")
    try:
        result = run(floor=args.floor, probe_import=args.probe_import)
        if args.output is None:
            print(json.dumps(result, sort_keys=True, allow_nan=False))
        else:
            _write_exclusive(args.output, result)
    except BaseException as error:
        print(f"offline floor failed: {type(error).__name__}: {error}",
              file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
