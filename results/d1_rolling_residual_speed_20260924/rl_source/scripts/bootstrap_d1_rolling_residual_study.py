"""Bounded import-environment repair before the frozen rolling study entry.

OpenCV's fixed loader appends its lib64 path to LD_LIBRARY_PATH while SB3 is
imported. This entry records that exact mutation, removes only that key, then
delegates to the unchanged study, which repeats its freeze and GOT checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

CV2_DIR = Path("/home/lyh/.local/lib/python3.10/site-packages/cv2")
CV2_SHA256 = {
    "__init__.py": "69192904dead3ada37c8b305ee850ac7474924730881a9d5e0b2390944eb5aae",
    "config.py": "974e2d4096ee1a9a9a341df1bf33e16973683bf1ac733006de27ff2f23bc584d",
    "config-3.py": "9a7aadf724b822001f5e963b01fd4e375d45e2cbbe328d2ca7b42a440c083a1c",
    "load_config_py3.py": "9c22766454a9ac2f6869d37830b2efc699e7b8c3125484e9976c6e677c066029",
    "version.py": "e718587212508acdcb296839c287c7fa346edb621005ffb199c26397592544ed",
    "cv2.abi3.so": "9e29605abcc31c9942d0e7dffc53b4ccaa601676c794327b0ea71a570a0c26c2",
}
ENVIRONMENT_CONTRACT_SHA256 = "5d1bc8ae1e2fed63bfda7db13d6198444508fc661335b7de82133b4ee3e20d16"
EXPECTED_MUTATION = "/home/lyh/.local/lib/python3.10/site-packages/cv2/../../lib64:"
RECEIPT_NAME = "environment_mutation_receipt.json"
FAILURE_NAME = "environment_bootstrap_failure.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_exclusive(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    return parser.parse_args(argv)


def _required_hashes(freeze: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = freeze["files"]
    if not isinstance(rows, dict):
        raise TypeError("freeze file list is malformed")
    if (CV2_DIR / "config-3.10.py").exists():
        raise ValueError("unexpected OpenCV version-specific loader config")
    required = [CV2_DIR / name for name in CV2_SHA256]
    for path in required:
        entry = rows.get(str(path))
        if (not isinstance(entry, dict) or not path.is_file()
                or path.stat().st_size != entry.get("bytes")
                or _sha256(path) != entry.get("sha256")
                or entry["sha256"] != CV2_SHA256[path.name]):
            raise ValueError(f"OpenCV loader input is absent or differs: {path}")
    return {str(path): rows[str(path)] for path in required}


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    receipt_path = args.output / RECEIPT_NAME
    stage = "startup"
    observed_mutation: str | None = None
    cv2_hashes: dict[str, dict[str, Any]] = {}
    cache_before: dict[str, int] | None = None
    cache_after: dict[str, int] | None = None
    startup = {
        "ld_library_path_key_absent": "LD_LIBRARY_PATH" not in os.environ,
        "cv2_module_absent": "cv2" not in sys.modules,
        "ld_preload": os.environ.get("LD_PRELOAD"),
        "ld_bind_now": os.environ.get("LD_BIND_NOW"),
        "python_version": list(sys.version_info[:3]),
    }
    try:
        if sys.version_info[:2] != (3, 10):
            raise RuntimeError("this frozen bootstrap requires Python 3.10")
        if not startup["ld_library_path_key_absent"] or not startup["cv2_module_absent"]:
            raise RuntimeError("bootstrap requires absent LD_LIBRARY_PATH and unloaded cv2")
        if (startup["ld_preload"] != str(args.library.resolve())
                or startup["ld_bind_now"] != "1"):
            raise RuntimeError("isolated DSO preload and eager binding are required")
        if not args.output.is_dir() or receipt_path.exists():
            raise ValueError("root-created output directory or exclusive receipt slot is unavailable")

        # The original study's top level has only standard-library imports.
        stage = "original_freeze_before_dependencies"
        from scripts import run_d1_rolling_residual_study as study

        freeze = study._check_freeze(args)
        environment_contract = freeze.get("environment_contract_path")
        environment_sha = freeze.get("environment_contract_sha256")
        if (not isinstance(environment_contract, str)
                or not Path(environment_contract).is_absolute()
                or environment_sha != ENVIRONMENT_CONTRACT_SHA256
                or freeze["files"].get(environment_contract, {}).get("sha256") != environment_sha
                or _sha256(Path(environment_contract)) != environment_sha):
            raise ValueError("environment-repair contract is not frozen")
        cv2_hashes = _required_hashes(freeze)

        # Exactly the heavy import graph from original study.main(), without
        # constructing EngineGuard, model, data, PPO, or any physical entry.
        stage = "original_dependency_imports"
        # isort: off -- import order is part of the observed environment failure
        import mujoco
        import mujoco._functions
        import mujoco._rollout

        from wheel_legged_control.d1.wheel_leg_controller import _nominal_geometry

        cache_before = _nominal_geometry.cache_info()._asdict()
        if cache_before != {"hits": 0, "misses": 0, "maxsize": 1, "currsize": 0}:
            raise RuntimeError("nominal geometry cache was warm before study dependencies")
        from scripts import d1_rolling_engine_runtime as runtime_module
        from scripts import d1_rolling_native_monitor as monitor_module
        from scripts import evaluate_d1_rolling_residual as eval_module
        from scripts import train_d1_rolling_residual_ppo as train_module

        if "cv2" not in sys.modules:
            raise RuntimeError("the original study imports did not load OpenCV")
        import cv2
        # isort: on

        stage = "verify_import_mutation"
        if (Path(cv2.__file__) != CV2_DIR / "__init__.py"
                or Path(cv2._native.__file__) != CV2_DIR / "cv2.abi3.so"):
            raise RuntimeError("loaded OpenCV Python/native modules differ from pinned files")
        if any(name not in sys.modules for name in (
            "mujoco", "mujoco._functions", "mujoco._rollout", "cv2",
            "scripts.d1_rolling_engine_runtime", "scripts.d1_rolling_native_monitor",
            "scripts.evaluate_d1_rolling_residual", "scripts.train_d1_rolling_residual_ppo",
        )):
            raise RuntimeError("the original study dependency graph was not fully imported")
        if not all((runtime_module, monitor_module, eval_module, train_module, mujoco)):
            raise RuntimeError("a required dependency module is unavailable")
        cache_after = _nominal_geometry.cache_info()._asdict()
        if cache_after != cache_before:
            raise RuntimeError("imports unexpectedly constructed nominal geometry")
        if (CV2_DIR / "config-3.10.py").exists():
            raise RuntimeError("OpenCV loader config selection changed during import")
        observed_mutation = os.environ.get("LD_LIBRARY_PATH")
        if observed_mutation != EXPECTED_MUTATION:
            raise RuntimeError("OpenCV LD_LIBRARY_PATH mutation differs from fixed loader")
        for name, entry in cv2_hashes.items():
            path = Path(name)
            if path.stat().st_size != entry["bytes"] or _sha256(path) != entry["sha256"]:
                raise RuntimeError(f"OpenCV loader input changed during import: {name}")

        stage = "restore_and_record"
        os.environ.pop("LD_LIBRARY_PATH")
        if ("LD_LIBRARY_PATH" in os.environ
                or os.environ.get("LD_PRELOAD") != startup["ld_preload"]
                or os.environ.get("LD_BIND_NOW") != startup["ld_bind_now"]):
            raise RuntimeError("engine environment was not restored exactly")
        _write_exclusive(receipt_path, {
            "schema": "d1-rolling-cv2-import-environment-repair-v1",
            "status": "restored_before_original_study",
            "startup": startup,
            "environment_contract_path": environment_contract,
            "environment_contract_sha256": environment_sha,
            "cv2_python_file": cv2.__file__,
            "cv2_native_file": cv2._native.__file__,
            "cv2_version_observed": getattr(cv2, "__version__", None),
            "cv2_frozen_files": cv2_hashes,
            "observed_ld_library_path_after_import": observed_mutation,
            "expected_single_opencv_mutation": EXPECTED_MUTATION,
            "restored_ld_library_path_key_absent": True,
            "restored_ld_preload": os.environ["LD_PRELOAD"],
            "restored_ld_bind_now": os.environ["LD_BIND_NOW"],
            "nominal_geometry_cache_before_import": cache_before,
            "nominal_geometry_cache_after_import": cache_after,
            "bootstrap_explicit_engine_guard_instantiation": False,
            "bootstrap_explicit_model_or_data_construction": False,
            "bootstrap_explicit_integration_call": False,
            "native_counter_zero_requires_original_runtime_proof": True,
            "next_entry": "unchanged run_d1_rolling_residual_study.main",
        })
    except BaseException as exc:
        # Preserve the sole prepaid attempt's failure before any study entry.
        observed = os.environ.get("LD_LIBRARY_PATH")
        os.environ.pop("LD_LIBRARY_PATH", None)
        failure_path = args.output / FAILURE_NAME
        if args.output.is_dir() and not failure_path.exists():
            _write_exclusive(failure_path, {
                "schema": "d1-rolling-cv2-import-environment-repair-v1",
                "status": "failed_before_original_study",
                "stage": stage, "failure_type": type(exc).__name__,
                "failure_message": str(exc), "startup": startup,
                "observed_ld_library_path_after_import": observed,
                "expected_single_opencv_mutation": EXPECTED_MUTATION,
                "restored_ld_library_path_key_absent": "LD_LIBRARY_PATH" not in os.environ,
                "cv2_frozen_files_checked": cv2_hashes,
                "nominal_geometry_cache_before_import": cache_before,
                "nominal_geometry_cache_after_import": cache_after,
                "original_study_main_called": False,
            })
        raise
    return study.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
