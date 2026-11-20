"""Budgeted single-environment control and training gates for the D1 terrain smoke."""

import argparse
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import mujoco

from scripts import terrain_baseline86 as baseline
from scripts import terrain_preview84_trials as trial
from scripts import terrain_preview85_action as pulse
from scripts.terrain_axle86 import audit_first_ticks
from scripts.terrain_smoke86_env import PhysicalSmokeEnv, advance

ROOT = trial.ROOT
PLAN = ROOT / "docs/experiments86/terrain_training_smoke.json"
ARCHIVE = baseline.ARCHIVED_MODEL
STAGE_LIMITS = {"development": 3600, "fallback": 3600, "fresh": 3600, "pulses": 3552, "policy": 1884}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_inputs(plan):
    if sha(ARCHIVE) != plan["archive_sha256"] or sha(
        ROOT / "runtime/d1_b22/runtime_manifest_v27.json"
    ) != plan["frozen_manifest_sha256"]:
        raise ValueError("archived input or frozen runtime identity changed")
    with ZipFile(ARCHIVE) as raw:
        previous = json.loads(raw.read("trials/started.json"))
    for filename, digest in previous["sources_sha256"].items():
        if sha(ROOT / filename) != digest:
            raise ValueError(f"round-85 source identity changed: {filename}")


@contextmanager
def meter_native(count, limit):
    native = mujoco.mj_step

    def counted(model, data, nstep=1):
        if count[0] + nstep > limit:
            raise RuntimeError("registered physics budget exhausted")
        native(model, data, nstep)
        count[0] += nstep

    with patch.object(mujoco, "mj_step", counted):
        yield


def previous_stage(root, name):
    receipt = read(root / name / "summary.json")
    if receipt["status"] != "COMPLETE" or receipt["protocol_sha256"] != sha(PLAN):
        raise ValueError(f"{name} lacks a complete matching receipt")
    for case in receipt.get("cases", []):
        label = f"{case['terrain']}_{case['seed']}"
        if read(root / name / label / "metrics.json") != case:
            raise ValueError(f"{name}/{label} differs from its raw case receipt")
    return receipt


def permitted_zero_cases(cases, design):
    expected = [(t, s) for t in design["terrains"] for s in design["seeds"]]
    return [(c["terrain"], c["seed"]) for c in cases] == expected and all(
        c["passed"] and c["full_horizon"] and c["task_steps"] == design["controls_per_case"]
        and c["zero_window_complete"] and c["sensor_passed"] and c["first_sensor_window_corroborated"]
        and not c["torque_bound_violations"] for c in cases
    )


def selected_baseline(root, plan):
    for stage in ("development", "fallback"):
        path = root / stage / "summary.json"
        if path.is_file():
            result = previous_stage(root, stage)
            if result["passed"] and permitted_zero_cases(result["cases"], plan["development"]):
                model = read(root / stage / "model.json") if stage == "fallback" else None
                if stage == "fallback" and sha(root / stage / "model.json") != result["damping_model_sha256"]:
                    raise ValueError("qualified damping parameters differ from the recorded fit")
                return stage, model
    return None, None


def zero_routes(directory, design, model, damping):
    cases = []
    for terrain in design["terrains"]:
        for seed in design["seeds"]:
            cases.append(baseline.run_zero_case(
                directory / f"{terrain}_{seed}", terrain, seed, model,
                normalized_input_cost=True, damping_model=damping,
            ))
    return {"passed": permitted_zero_cases(cases, design), "cases": cases,
            "controls": sum(c["task_steps"] for c in cases)}


def pulse_route(directory, terrain, seed, model, damping, design):
    directory.mkdir()
    trial.write_json(directory / "started.json", {"terrain": terrain, "seed": seed, "status_until_metrics": "PARTIAL"})
    ep = baseline.new_episode(terrain, seed, model, normalized_input_cost=True,
                              side_allocator=True, damping_model=damping)
    prefix = 0
    trigger = pulse._trigger(trial.front_contact_x(ep.plant), ep.toe)
    with (directory / "prefix.jsonl").open("x") as stream:
        while trigger == "approach" and prefix < design["prefix_limit_each"]:
            row = advance(ep, 0.0, damping)
            prefix += 1
            pulse._log(stream, row)
            trigger = row["termination_reason"] or pulse._trigger(row["front_contact_x_m"], ep.toe)
    result = {"terrain": terrain, "seed": seed, "prefix_steps": prefix, "toe_m": ep.toe,
              "prefix_outcome": "prefix_cap" if trigger == "approach" else trigger,
              "branch_steps": 0, "terminal_branches": 0, "passed": False}
    if trigger == "checkpoint":
        checkpoint = pulse.BranchCheckpoint(ep, directory)
        weight = ep.controller.low_level.total_mass_kg * ep.controller.low_level.gravity_mps2
        branches = {}
        for arm, sign in pulse.ARMS.items():
            trial.write_json(directory / f"{arm}_start.json", checkpoint.restore(ep))
            rows = []
            with (directory / f"{arm}.jsonl").open("x") as stream:
                for tick in range(1, design["branch_controls"] + 1):
                    row = advance(ep, pulse.pulse_at(sign, tick, weight), damping)
                    row["branch_tick"] = tick
                    pulse._log(stream, row)
                    rows.append(row)
                    if row["termination_reason"]:
                        break
            branches[arm] = rows
        result.update(pulse.compare_branches(branches, ep.toe))
        result["terminal_branches"] = len(branches)
        result["branch_steps"] = sum(map(len, branches.values()))
        result["both_signs_nonzero_first"] = (
            branches["negative"][0]["allocated_transfer_n"] < -1e-8
            and branches["positive"][0]["allocated_transfer_n"] > 1e-8
        )
    trial.write_json(directory / "metrics.json", result)
    return result


def physics_stage(stage, directory, plan):
    root = directory.parent
    model = baseline.load_model()
    if stage == "development":
        reference = audit_first_ticks(ARCHIVE)
        trial.write_json(directory / "allocation_readback.json", reference)
        if reference["old_reconstruction_max_error_nm"] > 1e-9:
            raise ValueError("published first torques failed to reproduce")
        return zero_routes(directory, plan["development"], model, None)
    if stage == "fallback":
        if previous_stage(root, "development")["passed"]:
            raise ValueError("the registered fallback is only for a failed development gate")
        from scripts.terrain_damping86 import fit_predictor

        offline, damping = fit_predictor(ARCHIVE)
        trial.write_json(directory / "fit.json", offline)
        if not offline["passed"]:
            return {"passed": False, "reason": "held-out velocity prediction", "cases": [], "controls": 0}
        trial.write_json(directory / "model.json", damping)
        return {**zero_routes(directory, plan["development"], model, damping),
                "damping_model_sha256": sha(directory / "model.json")}
    selected, damping = selected_baseline(root, plan)
    if selected is None:
        return {"passed": False, "reason": "development gate closed", "cases": [], "controls": 0}
    if stage == "fresh":
        return {**zero_routes(directory, plan["fresh_qualification"], model, damping), "selected": selected}
    fresh = previous_stage(root, "fresh")
    if fresh.get("selected") != selected or not permitted_zero_cases(fresh["cases"], plan["fresh_qualification"]):
        return {"passed": False, "reason": "fresh zero-route gate closed", "cases": [], "controls": 0}
    design = plan["pulse_qualification"]
    if stage == "pulses":
        cases = [pulse_route(directory / f"{t}_{s}", t, s, model, damping, design)
                 for t in design["terrains"] for s in design["seeds"]]
        return {"passed": all(c["passed"] for c in cases), "selected": selected, "cases": cases,
                "controls": sum(c["prefix_steps"] + c["branch_steps"] for c in cases)}
    pulses = previous_stage(root, "pulses")
    combos = [(t, s) for t in design["terrains"] for s in design["seeds"]]
    pulse_cases = pulses["cases"]
    if (pulses.get("selected") != selected or [(c["terrain"], c["seed"]) for c in pulse_cases] != combos
            or not all(c["passed"] and c["terminal_branches"] == 3 and c["branch_steps"] == 192
                       for c in pulse_cases)):
        return {"passed": False, "reason": "pulse gate closed", "cases": [], "controls": 0}
    fixture = read(root / "fixture/summary.json")
    if not (fixture["passed"] and fixture["synthetic_optimizer_steps"] == 3 and fixture["d1_optimizer_steps"] == 0):
        raise ValueError("PPO software fixture failed admission")
    from scripts.terrain_ppo86 import run_case

    outcomes = []
    controls = 0
    for seed in plan["ppo_smoke"]["seeds"]:
        with PhysicalSmokeEnv(directory / f"physical_{seed}", model, damping) as env:
            outcomes.append(run_case(env, seed, directory / f"seed_{seed}", task="physical", physical_gate_passed=True))
            controls += env.controls
    return {"passed": all(x["passed"] for x in outcomes), "selected": selected, "policies": outcomes,
            "controls": controls, "d1_optimizer_steps": sum(x["d1_optimizer_steps"] for x in outcomes)}


def run(stage, root):
    directory = root / stage
    directory.mkdir(parents=True, exist_ok=False)
    plan = read(PLAN)
    count = [0]
    sources = ["scripts/run_terrain_smoke86.py", "scripts/terrain_baseline86.py",
               "scripts/terrain_axle86.py", "scripts/terrain_smoke86_env.py"]
    if stage != "development":
        sources.append("scripts/terrain_damping86.py")
    if stage == "policy":
        sources.append("scripts/terrain_ppo86.py")
    trial.write_json(directory / "started.json", {"stage": stage, "protocol_sha256": sha(PLAN),
                     "sources_sha256": {f: sha(ROOT / f) for f in sources}, "status_until_summary": "PARTIAL"})
    try:
        verify_inputs(plan)
        with meter_native(count, 5 * STAGE_LIMITS[stage]):
            result = physics_stage(stage, directory, plan)
        if count[0] != 5 * result["controls"]:
            raise ValueError("native integration counter disagrees with recorded control steps")
        result = {"status": "COMPLETE", "stage": stage, "protocol_sha256": sha(PLAN),
                  "native_steps": count[0], "d1_optimizer_steps": 0, **result}
        trial.write_json(directory / "summary.json", result)
        return result
    except BaseException as exc:
        trial.write_json(directory / "partial.json", {"status": "PARTIAL", "stage": stage,
                         "native_steps": count[0], "error": repr(exc)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGE_LIMITS)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    receipt = run(args.stage, args.root)
    print(json.dumps({k: receipt[k] for k in ("stage", "passed", "controls", "native_steps", "d1_optimizer_steps")}))
