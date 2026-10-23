"""Independent, serial C50 parent; default only describes the registered plan.

The historical parent CLI is disabled before its source is compiled into this
new namespace. Its bounded ownership/counting functions are reused under exact
source/AST evidence, with new paths, reservation, source admission and pairing.
No historical release/output/once-only entrypoint is executed or resumed.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b22_sigma50_contract as contract50


def derived_parent50():
    derive = contract50.ExactDerivation50("scripts/run_b22_seed47.py")
    for name, old, new in (
        ("CONTRACT", "'C47_B22_seed_replication_v1'", repr(contract50.CONTRACT)),
        ("OUTPUT", "ROOT/'results/b22_seed47_20261018_02'", "ROOT/'results/b22_sigma50_20261024_01'"),
        ("RELEASE", "ROOT/'results/b22_seed47_plan_20261018_02.json'", "ROOT/'results/b22_sigma50_plan_20261024_01.json'"),
        ("ADMISSIONS", "ROOT/'results/b22_seed47_admissions_20261018_02'", "ROOT/'results/b22_sigma50_admissions_20261024_01'"),
        ("DRIVER", "ROOT/'scripts/run_b22_seed47.py'", "ROOT/'scripts/run_b22_sigma50.py'"),
        ("WORKER", "ROOT/'scripts/worker_b22_seed47.py'", "ROOT/'scripts/worker_b22_sigma50.py'"),
        ("READER", "ROOT/'scripts/read_b22_seed47.py'", "ROOT/'scripts/read_b22_sigma50.py'"),
    ):
        derive.assignment(name, old, new)
    derive.import_module("b22_seed47", "b22_sigma50_contract", count=3)
    derive.remove_main_guard()
    return derive.finish()


_tree50, DERIVATION50 = derived_parent50()
exec(compile(_tree50, __file__, "exec"), globals())  # noqa: S102 -- pinned, inverse-checked source in independent namespace

ROOT = globals()["ROOT"]
OUTPUT = globals()["OUTPUT"]
MANIFEST = globals()["MANIFEST"]
MANIFEST_SHA = globals()["MANIFEST_SHA"]
PINS = globals()["PINS"]
LIMITS = globals()["LIMITS"]
RESOURCE_POLICY = globals()["RESOURCE_POLICY"]
require = globals()["require"]
identity = globals()["identity"]
read_json = globals()["read_json"]
write_json = globals()["write_json"]
supervision = globals()["supervision"]
worker_api = globals()["worker_api"]
reader_api = globals()["reader_api"]
stages42 = globals()["stages42"]
main = globals()["main"]
_describe47 = globals()["describe42"]
_execute47 = globals()["execute42"]
_source_identity47 = globals()["source_identity42"]

# The worker's one-arm numeric protocol stays 32768 / 32 / 1024 / 256 / 4.
require(LIMITS["controls"] == 65536 and LIMITS["normal_native"] == 327680
        and LIMITS["compiler_native"] == 4 and LIMITS["total_wall_s"] == 4500,
        "50 historical parent budget seam changed")
LIMITS = {**LIMITS, "controls": 196608, "normal_native": 983040, "compiler_native": 12,
          "worker_pure_servo_advances": 960000, "reader_pure_servo_advances": 480000,
          "combined_pure_servo_advances": 1440000, "total_wall_s": 10800,
          "start_free_disk_bytes": 22 << 30, "output_quota_bytes": 20 << 30}
RESOURCE_POLICY = {**RESOURCE_POLICY, "unit": "control50-wheel-train-20261024-01.service",
                   "owned_scope": "/user.slice/user-1000.slice/user@1000.service/app.slice/control50-wheel-train-20261024-01.service"}
AUTHORED = (
    "scripts/b22_sigma50_contract.py", "scripts/run_b22_sigma50.py", "scripts/worker_b22_sigma50.py",
    "scripts/read_b22_sigma50.py", "scripts/eval_b22_sigma50.py", "scripts/audit_b22_sources50.py",
    "scripts/admit_b22_sigma50.py", "docs/experiments50/training_admission50_preregistered.json",
    "tests/test_b22_sigma50_admission.py",
    "tests/test_b22_sigma50_worker.py", "tests/test_b22_sigma50_reader.py", "tests/test_b22_sigma50_eval.py",
    "tests/test_b22_sources50.py", "docs/b22_sigma50_preregistration.md",
    "src/wheel_legged_control/d1/yaw_feedforward44.py", "scripts/b22_yawtrain44.py",
    "scripts/b22_reward46.py", "scripts/b22_tracking_reward42.py", "scripts/resource_guard42.py",
)


def unused_budget_origin42():
    return {"schema": "b22-sigma50-independent-new-budget-v1", "previous_executions": 0,
            "archive48_reused": False, "parent_checkpoint": "complete_B22"}


def predecessor_failure42():
    return {"schema": "b22-sigma50-no-resumption-v1", "inputs": {}}


def describe42():
    value = _describe47()
    value.update(
        initial_intervention_schema=contract50.INITIAL_SCHEMA,
        independent_source_audit=str(contract50.SOURCE_AUDIT),
        accepted_environment_difference="exact approved libXtst replacement; v27 unchanged",
        seeds=list(contract50.SEEDS), arms=contract50.ARMS,
        training_comparison={"primary_actor": "mirror50", "secondary_actor": "raw50",
                             "question": "three-seed range decreases without increasing mean on fixed development tasks",
                             "first1024_trajectory_equality_required": False,
                             "automatic_deployment": False},
        selection="all six complete valid finals evaluated; any incomplete attempt prevents the primary comparison",
        parent_derivation=DERIVATION50,
        namespace_overrides50=["AUTHORED", "LIMITS", "RESOURCE_POLICY", "inputs42", "source_identity42",
                               "describe42", "verify_training_pair42", "execute42", "admission_proof50",
                               "unused_budget_origin42", "predecessor_failure42"],
    )
    return value


def source_identity42():
    value = _source_identity47()
    value["source_audit_identity"] = identity(contract50.SOURCE_AUDIT)
    value["parent_derivation50"] = DERIVATION50
    value["model_admission50"] = admission_proof50()
    return value


def admission_proof50():
    """A real, separately terminated model worker and cold reader are required."""
    import admit_b22_sigma50 as admission

    admission.validate_plan50()
    folder = admission.OUTPUT
    report = read_json(folder / "report.json")
    worker = read_json(folder / "worker_report.json")
    reader = read_json(folder / "reader_report.json")
    registration = admission.verify_registration50()
    require(all(row.get("status") == "COMPLETE" for row in (report, worker, reader))
            and report.get("worker_identity") == reader.get("worker_report_identity") == identity(folder / "worker_report.json")
            and report.get("reader_identity") == identity(folder / "reader_report.json")
            and worker.get("registration_identity") == identity(folder / "registration.json")
            and reader.get("no_torch_or_simulator_imported") is True
            and reader.get("pid") != worker.get("pid"), "50 complete independent model admission required")
    counts = worker.get("actual_model_api_counts")
    require(counts == report.get("actual_counts") and counts["limits"] == admission.LIMITS
            and all(counts["counts"].get(name, {}).get("returned", 0) == limit
                    for name, limit in admission.LIMITS.items())
            and all(row["attempted"] == row["returned"] for row in counts["counts"].values())
            and counts["counts"]["predict"]["rows_returned"] == 64,
            "50 admission model API/probe-row count differs")
    require(all(worker.get(name) == 0 for name in ("native_steps", "mujoco_constructions", "training_controls", "training_attempts_consumed"))
            and worker.get("stub_instances") == 2 and worker.get("simulator_modules_loaded") == []
            and worker.get("simulator_import_rejections") == []
            and worker.get("zero_api_counts") == {name: {"attempted": 0, "returned": 0, "rejected": 0}
                                                  for name in admission.ZERO_APIS}
            and worker.get("models_reused_for_training") is False,
            "50 admission zero-learning/zero-physics reservation differs")
    require(set(reader.get("arms", {})) == set(worker.get("arms", {})) == set(admission.ARMS)
            and all(reader["arms"][arm].get("actor_critic_adam_rng_unchanged") is True
                    and reader["arms"][arm].get("arm") == arm
                    and reader["arms"][arm].get("seed") == contract50.ARMS[arm]["seed"]
                    for arm in admission.ARMS), "50 admission two independent initial proofs required")
    # Reconstruct the complete admitted inventory, including current authored
    # bytes. The original parent validate_release42 subsequently rehashes every
    # member of this same inventory before any worker may start.
    require(registration["source_hashes"] == admission.registration50()["source_hashes"],
            "50 source inventory changed after model admission")
    files = {str(path): identity(path) for path in folder.rglob("*") if path.is_file()}
    return {"schema": "b22-sigma50-admission-binding-v1", "path": str(folder / "report.json"),
            "report_identity": identity(folder / "report.json"), "files": files,
            "training_attempts_consumed": 0, "predict_api_calls": 2, "probe_rows": 64}


def inputs42(source):
    supervision()
    require(identity(MANIFEST)["sha256"] == MANIFEST_SHA == contract50.MANIFEST_SHA,
            "50 immutable v27 manifest changed")
    audit = contract50.source_audit50()
    require(source["source_audit_identity"] == identity(contract50.SOURCE_AUDIT),
            "50 registration source audit changed")
    inputs = dict(audit["current_source_hashes"])
    admission = admission_proof50()
    require(source["model_admission50"] == admission, "50 registered model admission changed")
    mappings = (
        source["authored"], worker_api().required_inputs42(), reader_api().required_inputs42(),
        admission["files"],
        contract50.pinned_inputs50(), {str(ROOT / name): identity(ROOT / name) for name in PINS},
        {str(MANIFEST): identity(MANIFEST), str(contract50.SOURCE_AUDIT): identity(contract50.SOURCE_AUDIT),
         str(contract50.SOURCE_AUDIT.parent / "sources.jsonl"): audit["ledger_identity"]},
    )
    for mapping in mappings:
        require(type(mapping) is dict, "50 source identity map required")
        for name, entry in mapping.items():
            require(name not in inputs or inputs[name] == entry, "50 conflicting registered source identity: " + name)
            inputs[name] = entry
    return inputs


def verify_training_pair42():
    pairs = []
    compare = ("seed", "course_arm", "command_seed", "measurement_seed", "source_episode_offset",
               "initial_state", "parent_hashes", "rng_sha256", "environment_randomness_scope")
    for seed in contract50.SEEDS:
        names = [f"{label}_s{seed % 100}" for label in ("baseline", "candidate")]
        rows = {name: read_json(OUTPUT / ("train_" + name + "_readback.json")) for name in names}
        evidence = {}
        for name, row in rows.items():
            require(row.get("status") == "COMPLETE" and row.get("training_valid") is True,
                    "50 pairing requires every complete training readback")
            proof = row.get("pairing42", {})
            require(proof.get("schema") == "b22-sigma50-exogenous-pairing-v1"
                    and proof.get("seed") == seed and proof.get("course_arm") == "B"
                    and proof.get("command_seed") == contract50.COMMAND_SEED
                    and proof.get("measurement_seed") == contract50.MEASUREMENT_SEED
                    and proof.get("source_episode_offset") == 0
                    and proof.get("first1024_trajectory_equality_required") is False
                    and all(key in proof for key in compare), "50 complete paired exogenous proof required")
            evidence[name] = proof
        first, second = (evidence[name] for name in names)
        require(all(first[key] == second[key] for key in compare),
                "50 paired initial state, parent, RNG or external configuration differs")
        episodes = {}
        for name in names:
            rows = evidence[name].get("exogenous_episodes")
            require(type(rows) is list and rows and sum(row["completed_ticks"] for row in rows) == 32768,
                    "50 complete source-episode/tick ledger required")
            require([row["episode_index"] for row in rows] == list(range(len(rows)))
                    and [row["source_episode_index"] for row in rows] == list(range(len(rows))),
                    "50 source-episode ledger is not contiguous from zero")
            episodes[name] = {row["source_episode_index"]: row for row in rows}
        shared = sorted(set(episodes[names[0]]) & set(episodes[names[1]]))
        require(shared, "50 paired training has no shared source episode")
        shared_ticks = 0
        for source in shared:
            a, b = (episodes[name][source] for name in names)
            require(all(a[key] == b[key] for key in ("raw_command_sha256", "scheduled_ticks",
                    "episode_reset_seed", "provider_measurement_seed")), "50 external condition differs by source episode")
            shared_ticks += min(a["completed_ticks"], b["completed_ticks"])
        pairs.append({"seed": seed, "readers": {name: identity(OUTPUT / ("train_" + name + "_readback.json"))
                                                for name in names}, "proofs": evidence,
                      "shared_source_episode_count": len(shared), "shared_source_episode_tick_count": shared_ticks})
    return {"schema": "b22-sigma50-pair-verification-v1", "passed": True, "pairs": pairs,
            "post_intervention_action_or_state_equality_required": False,
            "independent_environment_noise_replications_claimed": False}


def execute42():
    result = _execute47()
    status = {}
    for stage in stages42():
        folder = OUTPUT / stage["name"]
        row = {"status": "NOT_RUN", "actual_counts": None}
        if folder.exists():
            row["status"] = "INCOMPLETE"
            receipt = folder / "worker_receipt.json"
            if receipt.exists():
                value = read_json(receipt)
                row.update(worker_status=value.get("status"), actual_counts=value.get("actual_counts"),
                           known_partial_counts=value.get("known_partial_counts"), receipt_identity=identity(receipt))
                readback = OUTPUT / (stage["name"] + "_readback.json")
                if stage["mode"] == "bootstrap" and value.get("status") == "COMPLETE":
                    row["status"] = "COMPLETE"
                elif readback.exists():
                    saved = read_json(readback)
                    if saved.get("status") == "COMPLETE" and saved.get("training_valid") is True:
                        row["status"] = "COMPLETE"
        status[stage["name"]] = row
    result["stage_status50"] = status
    result["unknown_stage_counts50"] = [name for name, row in status.items()
                                        if row["status"] == "INCOMPLETE" and row.get("known_partial_counts") is None]
    if result.get("execution_started"):
        terminal = identity(OUTPUT / "report.json")
        supplement = OUTPUT / "stage_status50.json"
        write_json(supplement, {
            "schema": "b22-sigma50-stage-status-v1", "status": result["status"],
            "terminal_report_path": str(OUTPUT / "report.json"), "terminal_report_identity": terminal,
            "stage_status50": status, "unknown_stage_counts50": result["unknown_stage_counts50"],
        })
        require(identity(OUTPUT / "report.json") == terminal, "50 historical-owner terminal receipt changed")
        result["stage_status50_path"] = str(supplement)
        result["stage_status50_identity"] = identity(supplement)
        result["terminal_report_identity"] = terminal
    return result


if __name__ == "__main__":
    raise SystemExit(main())
