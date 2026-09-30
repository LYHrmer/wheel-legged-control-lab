"""Root-only bounded 08-R pure test round; never imports the engine."""

from __future__ import annotations

import argparse
import hashlib
import importlib.abc
import importlib.machinery
import json
import os
import sys
import time
import types
from pathlib import Path


WORK = Path(__file__).resolve().parent
REPO = Path("/home/lyh/wheel-legged-control-lab")
TESTS = (
    WORK / "course_impl08/test_residual16_math_08.py",
    WORK / "course_impl08/test_course_seams_08.py",
)
SOURCES = (
    WORK / "run_course_pure_tests_08.py",
    WORK / "next_rl_pilot_contract_08.md",
    *TESTS,
    *(
        WORK / "course_impl08" / name
        for name in (
            "residual16_math_08.py",
            "course_ground_08.py",
            "full_drive_command_08.py",
            "full_drive_servo_08.py",
            "full_drive_observation_08.py",
            "full_drive_loop_08.py",
            "full_drive_controller_08.py",
            "full_drive_env_08.py",
            "full_drive_schedule_08.py",
        )
    ),
)

FORBIDDEN = {"mujoco", "glfw", "torch", "stable_baselines3", "gym", "gymnasium"}
BLOCKED: list[str] = []


class NoPhysicsImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.partition(".")[0] in FORBIDDEN:
            BLOCKED.append(fullname)
            raise RuntimeError("08-R pure-test physics import forbidden: " + fullname)
        return None


class TestReceipts:
    def __init__(self) -> None:
        self.results: list[dict] = []
        self.collected = 0
        self.collection_errors: list[str] = []

    def pytest_collection_finish(self, session) -> None:
        self.collected = len(session.items)
        if self.collected > 16:
            raise RuntimeError("08-R pure round collected more than sixteen cases")

    def pytest_collectreport(self, report) -> None:
        if report.failed:
            self.collection_errors.append(str(report.longrepr))

    def pytest_runtest_logreport(self, report) -> None:
        if report.when == "call" or report.failed:
            self.results.append(
                {
                    "nodeid": report.nodeid,
                    "when": report.when,
                    "outcome": report.outcome,
                }
            )


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", type=int, choices=(1, 2), required=True)
    args = parser.parse_args()
    activation = WORK / f"pure_course_round_{args.round}_activation.json"
    receipt = WORK / f"pure_course_round_{args.round}_receipt.json"
    if activation.exists() or receipt.exists():
        raise FileExistsError("08-R pure round has already been consumed")
    if args.round == 2 and not (WORK / "pure_course_round_1_receipt.json").is_file():
        raise RuntimeError("round two requires a saved first round")
    if any(not path.is_file() for path in SOURCES):
        raise FileNotFoundError("08-R pure source absent")
    before = {str(path): sha(path) for path in SOURCES}
    with activation.open("x", encoding="utf-8") as stream:
        json.dump(
            {"round": args.round, "test_limit": 16, "source_sha256": before},
            stream,
            indent=2,
            sort_keys=True,
        )
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(WORK / "course_impl08"))
    # A ROS dist-packages regular package named ``scripts`` can shadow the
    # repository's namespace directory even when REPO is first on sys.path.
    # Bind only the exact local package path for this pure process.
    if "scripts" in sys.modules:
        raise RuntimeError("scripts package was imported before local pure bootstrap")
    package = types.ModuleType("scripts")
    package.__path__ = [str(REPO / "scripts")]
    package.__spec__ = importlib.machinery.ModuleSpec(
        "scripts", loader=None, is_package=True
    )
    package.__spec__.submodule_search_locations = package.__path__
    sys.modules["scripts"] = package
    sys.meta_path.insert(0, NoPhysicsImports())
    if any(name.partition(".")[0] in FORBIDDEN for name in sys.modules):
        raise RuntimeError("physics dependency already loaded before pure tests")
    import pytest

    tracker = TestReceipts()
    started = time.monotonic()
    code = int(
        pytest.main(
            ["-q", "-p", "no:cacheprovider", *map(str, TESTS)], plugins=[tracker]
        )
    )
    changed = [name for name, digest in before.items() if sha(Path(name)) != digest]
    calls = [row for row in tracker.results if row["when"] == "call"]
    passed = (
        code == 0
        and tracker.collected == 16
        and len(calls) == tracker.collected
        and all(row["outcome"] == "passed" for row in calls)
        and not BLOCKED
        and not changed
        and not tracker.collection_errors
    )
    result = {
        "schema": "d1-course-rl16-pure-round-v1",
        "round": args.round,
        "exit_code": code,
        "collected": tracker.collected,
        "results": tracker.results,
        "blocked_engine_imports": BLOCKED,
        "collection_errors": tracker.collection_errors,
        "sources_changed_during_tests": changed,
        "all_passed": passed,
        "source_sha256": before,
        "file_sha256": {str(path): sha(path) for path in TESTS},
        "elapsed_s": time.monotonic() - started,
    }
    with receipt.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
