"""Compare D1 compute latency on one frozen, reference-generated state trace.

Only the reference controller module comes from Git; both versions use this
checkout's model and state classes. Use trusted local revisions: loading the
reference executes its Python source. This host benchmark is not a real-time
guarantee or a CI performance gate. No thread settings are changed by this script.
"""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import io
import json
import os
import platform
import pstats
import subprocess
import sys
import types
from datetime import datetime, timezone
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from time import perf_counter_ns

import numpy as np
from scipy.spatial.transform import Rotation

from wheel_legged_control.d1.controllers import D1Command
from wheel_legged_control.d1.inverse_dynamics import D1InverseDynamicsController
from wheel_legged_control.d1.model import D1Plant
from wheel_legged_control.d1.state_estimation import D1MujocoTruthStateSource
from wheel_legged_control.provenance import capture_git_provenance

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "src/wheel_legged_control/d1/inverse_dynamics.py"
STEPS, REPETITIONS, WARMUP_CALLS = 600, 3, 10


def _source_snapshot() -> dict[str, dict[str, str]]:
    """Bind imported project modules to this checkout, then fingerprint them."""
    paths = [Path(__file__).resolve()]
    for name in (
        "wheel_legged_control.d1.inverse_dynamics", "wheel_legged_control.d1.model",
        "wheel_legged_control.d1.state_estimation", "wheel_legged_control.d1.controllers",
        "wheel_legged_control.provenance",
    ):
        actual = Path(import_module(name).__file__).resolve()
        expected = (ROOT / "src").joinpath(*name.split(".")).with_suffix(".py").resolve()
        if actual != expected:
            raise RuntimeError(f"module {name} imported from {actual}, expected {expected}")
        paths.append(actual)
    return {
        str(path.relative_to(ROOT)): {
            "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in paths
    }


def _reference(revision: str) -> tuple[type, str, str]:
    def git(*args: str) -> bytes:
        return subprocess.check_output(("git", *args), cwd=ROOT)

    commit = git("rev-parse", "--verify", "--end-of-options", revision + "^{commit}").decode().strip()
    source = git("show", f"{commit}:{SOURCE}")
    name = "wheel_legged_control.d1._benchmark_reference"
    module = types.ModuleType(name)
    module.__package__ = "wheel_legged_control.d1"
    module.__file__ = f"git:{commit}:{SOURCE}"
    sys.modules[name] = module
    # Loading a trusted reference module intentionally executes its source.
    exec(compile(source, module.__file__, "exec"), module.__dict__)  # noqa: S102
    return module.D1InverseDynamicsController, commit, hashlib.sha256(source).hexdigest()


def _trace(controller_type: type) -> tuple[list, list[float]]:
    plant = D1Plant(control_dt=.01, arena="flat")
    initial_rpy = np.r_[np.random.default_rng(22).uniform(-.01, .01, 2), 0.0]
    quaternion = Rotation.from_euler("xyz", initial_rpy).as_quat()[[3, 0, 1, 2]]
    plant.reset(base_quaternion=quaternion)
    source = D1MujocoTruthStateSource(plant)
    state = source.reset(seed=22)
    controller = controller_type(control_dt=.01)
    trace = []
    for step in range(STEPS):
        command = D1Command(forward_velocity_mps=.5 if 100 <= step < 300 else 0.0)
        trace.append((command, state))
        plant.step(controller.compute(command, state))
        state = source.read()
    return trace, initial_rpy.tolist()


def _run(controller_type: type, trace: list, *, warm_start: bool | None = None) -> tuple[dict, list]:
    start = perf_counter_ns()
    kwargs = {} if warm_start is None else {"warm_start": warm_start}
    controller = controller_type(control_dt=.01, **kwargs)
    constructor_ms = (perf_counter_ns() - start) / 1e6
    errors = []

    def compute(command: D1Command, state, phase: str):
        start = perf_counter_ns()
        try:
            torque = controller.compute(command, state)
        except (ValueError, RuntimeError) as error:
            elapsed = (perf_counter_ns() - start) / 1e6
            errors.append({"phase": phase, "error": str(error)})
            return elapsed, None
        elapsed = (perf_counter_ns() - start) / 1e6
        result = controller.last_result
        if result is None or result.status != "solved":
            errors.append({"phase": phase, "error": "compute returned without solved result"})
            return elapsed, None
        output = np.concatenate((torque, result.generalized_acceleration,
                                 result.contact_force_world_n.ravel()))
        if not np.isfinite(output).all():
            errors.append({"phase": phase, "error": "nonfinite output"})
            return elapsed, None
        return elapsed, output

    first_ms, _ = compute(*trace[0], "first_call")
    for index in range(WARMUP_CALLS):
        compute(*trace[0], f"warmup:{index}")
    timings, outputs = [], []
    for index, (command, state) in enumerate(trace):
        elapsed, output = compute(command, state, f"trace:{index}")
        timings.append(elapsed)
        outputs.append(output)
    rejected = sum(output is None for output in outputs)
    return {
        "constructor_ms": constructor_ms, "first_compute_ms": first_ms,
        "compute_ms": timings, "accepted": len(outputs) - rejected,
        "rejected": rejected, "all_phase_rejections": len(errors), "errors": errors,
    }, outputs


def _summary(passes: list[dict]) -> dict:
    values = np.concatenate([run["compute_ms"] for run in passes])
    return {
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "p99_9_ms": float(np.percentile(values, 99.9)),
        "max_ms": float(np.max(values)),
        "over_10ms_fraction": float(np.mean(values > 10.0)),
        "accepted": sum(run["accepted"] for run in passes),
        "rejected": sum(run["rejected"] for run in passes),
        "all_phase_rejections": sum(run["all_phase_rejections"] for run in passes),
        "passes": passes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", default="6faf813", help="trusted local Git revision")
    parser.add_argument("--output", type=Path, required=True, help="new JSON file; never overwritten")
    parser.add_argument("--profile", action="store_true", help="separate, untimed candidate pass")
    parser.add_argument("--warm-start", action="store_true",
                        help="experimental candidate; performance acceptance is not equivalence")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    sources_before = _source_snapshot()
    provenance = capture_git_provenance(ROOT)
    reference, commit, source_hash = _reference(args.reference)
    trace, initial_rpy = _trace(reference)
    runs, differences, orders = {"reference": [], "candidate": []}, [], []
    for repetition in range(REPETITIONS):
        order = ("reference", "candidate") if repetition % 2 == 0 else ("candidate", "reference")
        orders.append(order)
        outputs = {}
        for name in order:
            kind = reference if name == "reference" else D1InverseDynamicsController
            run, outputs[name] = _run(
                kind, trace, warm_start=args.warm_start if name == "candidate" else None
            )
            runs[name].append(run)
        for old, new in zip(outputs["reference"], outputs["candidate"], strict=True):
            if old is not None and new is not None:
                differences.append(np.abs(new - old))
    summaries = {name: _summary(passes) for name, passes in runs.items()}
    maximum = np.max(differences, axis=0) if differences else np.full(50, np.inf)
    difference = {
        name: float(values.max()) if differences else None
        for name, values in zip(("torque_nm", "qacc", "contact_force_n"),
                                (maximum[:16], maximum[16:38], maximum[38:]), strict=True)
    }
    reductions = {key: 1.0 - summaries["candidate"][key] / summaries["reference"][key]
                  for key in ("p50_ms", "p99_ms")}
    equivalence_gates = {
        "all_comparisons_accepted": len(differences) == STEPS * REPETITIONS,
        "torque_difference_le_1e_3_nm": bool(differences) and difference["torque_nm"] <= 1e-3,
        "qacc_difference_le_0_01": bool(differences) and difference["qacc"] <= .01,
        "force_difference_le_0_05_n": bool(differences) and difference["contact_force_n"] <= .05,
    }
    performance_gates = {
        "no_rejections": all(item["all_phase_rejections"] == 0 for item in summaries.values()),
        "median_reduction_ge_25_percent": reductions["p50_ms"] >= .25,
        "p99_reduction_ge_15_percent": reductions["p99_ms"] >= .15,
    }
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "reference_commit": commit, "reference_module_sha256": source_hash,
        "candidate_provenance": provenance,
        "candidate_module_sha256": sources_before[SOURCE]["sha256"],
        "project_sources_before": sources_before,
        "candidate_configuration": {"warm_start": args.warm_start},
        "requires_closed_loop_validation": args.warm_start,
        "acceptance_scope": (
            "experimental performance only; not numerical equivalence or closed-loop acceptance"
            if args.warm_start else "performance and numerical equivalence on the recorded trace"
        ),
        "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                    "dependencies": {name: version(name) for name in ("numpy", "scipy", "mujoco", "osqp")},
                    "thread_environment": {name: os.environ.get(name) for name in (
                        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                        "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")}},
        "trace": {"seed": 22, "scenario": "drive_brake", "steps": STEPS,
                  "control_dt_s": .01, "initial_rpy_rad": initial_rpy,
                  "contact_masks": sorted({"".join(str(int(x)) for x in state.wheel_contact)
                                           for _, state in trace})},
        "method": {"repetitions": REPETITIONS, "pass_order": orders,
                   "warmup": f"first compute separately, then {WARMUP_CALLS} repeats of trace[0]",
                   "measurement": "all 600 trace calls after warmup; fresh controller per pass",
                   "scope": "reference module only; model/state/dependencies from current checkout",
                   "limitation": "host benchmark, not a CI threshold or hard-real-time guarantee"},
        "results": summaries, "maximum_absolute_difference": difference,
        "paired_accepted_calls": len(differences), "latency_reduction_fraction": reductions,
        "numerical_equivalence_gates": equivalence_gates,
        "numerical_equivalence_passed": all(equivalence_gates.values()),
        "performance_gates": performance_gates,
        "profile_status": "not_requested",
    }
    if args.profile:
        profiler = cProfile.Profile()
        try:
            controller = D1InverseDynamicsController(control_dt=.01, warm_start=args.warm_start)
            for _ in range(WARMUP_CALLS + 1):
                controller.compute(*trace[0])
            with profiler:
                for command, state in trace:
                    controller.compute(command, state)
            stream = io.StringIO()
            pstats.Stats(profiler, stream=stream).sort_stats("cumulative").print_stats(25)
            report["separate_candidate_profile"] = stream.getvalue()
            report["profile_status"] = "completed"
        except Exception as error:  # noqa: BLE001 - retain report, then exit nonzero
            # Profiling is independent: preserve the completed timing report,
            # including when construction or warmup fails before profiling starts.
            report["profile_status"] = "failed"
            report["separate_candidate_profile_error"] = f"{type(error).__name__}: {error}"
    try:
        report["project_sources_after"] = _source_snapshot()
        performance_gates["project_sources_unchanged"] = (
            report["project_sources_after"] == sources_before
        )
    except (OSError, RuntimeError) as error:
        report["project_sources_after"] = None
        report["project_sources_error"] = str(error)
        performance_gates["project_sources_unchanged"] = False
    performance_gates["requested_profile_completed"] = report["profile_status"] != "failed"
    equivalence_gates.update({
        name: performance_gates[name] for name in ("no_rejections", "project_sources_unchanged")
    })
    report["numerical_equivalence_passed"] = all(equivalence_gates.values())
    gates = (performance_gates.copy() if args.warm_start
             else {**performance_gates, **equivalence_gates})
    report["gates"] = gates
    report["performance_passed"] = all(performance_gates.values())
    report["passed"] = all(gates.values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output": str(args.output), "passed": report["passed"],
                      "numerical_equivalence_passed": report["numerical_equivalence_passed"],
                      "requires_closed_loop_validation": args.warm_start,
                      "gates": gates, "latency_reduction_fraction": reductions}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
