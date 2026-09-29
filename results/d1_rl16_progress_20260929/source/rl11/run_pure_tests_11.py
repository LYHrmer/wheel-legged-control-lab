"""Root-only bounded short-plan and heldout scorer integration tests, with no engine or policy imports."""

from __future__ import annotations

import argparse
import hashlib
import importlib.abc
import json
import os
import signal
import sys
import time
import traceback
from pathlib import Path

W = Path(__file__).resolve().parent
R = Path("/home/lyh/wheel-legged-control-lab")
CONTRACT = W / "next_short_training_contract_11.md"
CONTRACT_SHA = "e6e9d9d61d43d5eb3184c6c0cb31493f9f95e76cf4ab5632de02ce28135cd929"
TESTS = (W / "test_rl16_heldout_score_11_pure.py",
         W / "test_short_curriculum_11_pure.py")
SOURCES = (
    Path(__file__).resolve(), CONTRACT, W / "short_seed_semantics_11.md", *TESTS,
    *(W / name for name in (
        "rl16_heldout_score_11.py", "short_curriculum_11.py",
        "short_corridor_11.py", "short_episode_plan_11.py",
        "short_plan_recipe_11.py", "short_heldout_recipe_11.py", "short_heldout_11.py",
        "budget_spec_11.py",
    )),
    W.parent / "upright11/world_upright_course_11.py",
    R / "scripts/run_d1_latest_rl.py",
    *sorted((W.parent / "course_impl08").glob("*.py")),
    W.parent / "upright11/reference_readback_01.json",
    W.parent / "upright11/reference_run_01/ramp_zero_reference/geometry_manifest.json",
    W / "model_preflight_01/receipt.json",
)


FORBIDDEN = {
    "mujoco",
    "glfw",
    "torch",
    "stable_baselines3",
    "gym",
    "gymnasium",
    "wheel_legged_control",
    "engine_binding",
    "cv2",
}
BLOCKED = []


class NoEngine(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.partition(".")[0] in FORBIDDEN:
            BLOCKED.append(fullname)
            raise RuntimeError("upright pure test forbids engine/model: " + fullname)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


class Results:
    def __init__(self):
        self.collected = 0
        self.rows = []
        self.collection_errors = []

    def pytest_collection_finish(self, session):
        self.collected = len(session.items)
        if not 1 <= self.collected <= 8:
            raise RuntimeError("terminal round must collect 1..8 new pure cases")

    def pytest_collectreport(self, report):
        if report.failed:
            self.collection_errors.append(str(report.longrepr))

    def pytest_runtest_logreport(self, report):
        if report.when == "call" or report.failed:
            self.rows.append(
                {
                    "nodeid": report.nodeid,
                    "when": report.when,
                    "outcome": report.outcome,
                    "failure": str(report.longrepr) if report.failed else None,
                }
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", type=int, choices=(1, 2), required=True)
    parser.add_argument("--correction-note", type=Path)
    args = parser.parse_args()
    out = W / f"pure_round_{args.round:02d}"
    if out.exists():
        raise RuntimeError("upright pure round already consumed")
    if sha(CONTRACT) != CONTRACT_SHA:
        raise RuntimeError("pure test scope changed")
    correction = None
    if args.round == 2:
        first = json.loads((W / "pure_round_01/receipt.json").read_text())
        if first.get("all_passed") or args.correction_note is None:
            raise RuntimeError(
                "second round requires failed first round and concrete correction"
            )
        correction = {
            "path": str(args.correction_note.resolve()),
            "sha256": sha(args.correction_note),
            "text": args.correction_note.read_text(),
        }
    before = {str(path): sha(path) for path in SOURCES}
    out.mkdir()
    save(
        out / "reservation.json",
        {
            "round": args.round,
            "max_cases": 8,
            "wallclock_limit_s": 30,
            "source_sha256": before,
            "robot_controls": 0,
            "native_calls": 0,
            "correction": correction,
        },
    )
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    sys.path[:0] = [
        str(W), str(W.parent / "upright11"),
        str(W.parent / "course_impl08"), str(R / "src"), str(R),
        "/home/lyh/.local/lib/python3.10/site-packages",
        str(R / ".local-deps"),
    ]
    if any(name.partition(".")[0] in FORBIDDEN for name in sys.modules):
        raise RuntimeError("pure worker began with an engine/model already imported")
    sys.meta_path.insert(0, NoEngine())
    tracker = Results()
    started = time.monotonic()
    error = None
    code = None

    def expired(_sig, _frame):
        raise TimeoutError("30s upright pure round wall limit")

    signal.signal(signal.SIGALRM, expired)
    signal.alarm(30)
    try:
        import pytest

        code = int(
            pytest.main(["-q", "-p", "no:cacheprovider", *(str(path) for path in TESTS)], plugins=[tracker])
        )
    except BaseException as exc:  # noqa: BLE001 -- preserve consumed round even on interruption
        error = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        signal.alarm(0)
        elapsed = time.monotonic() - started
        changed = [
            name
            for name, expected in before.items()
            if not Path(name).is_file() or sha(Path(name)) != expected
        ]
        calls = [row for row in tracker.rows if row["when"] == "call"]
        passed = (
            code == 0
            and error is None
            and 1 <= tracker.collected <= 8
            and len(calls) == tracker.collected
            and all(row["outcome"] == "passed" for row in calls)
            and not BLOCKED
            and not changed
            and not tracker.collection_errors
            and elapsed <= 30
        )
        receipt = {
            "round": args.round,
            "all_passed": passed,
            "collected": tracker.collected,
            "exit_code": code,
            "error": error,
            "results": tracker.rows,
            "source_sha256": before,
            "sources_changed": changed,
            "blocked_engine_imports": BLOCKED,
            "collection_errors": tracker.collection_errors,
            "elapsed_s": elapsed,
            "robot_controls": 0,
            "native_calls": 0,
        }
        save(out / "receipt.json", receipt)
    print(
        json.dumps(
            {
                "all_passed": passed,
                "collected": tracker.collected,
                "exit_code": code,
                "error": error,
                "elapsed_s": elapsed,
            }
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
