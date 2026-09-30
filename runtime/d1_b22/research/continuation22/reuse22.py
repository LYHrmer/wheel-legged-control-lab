"""Pure evidence bridge for the one frozen C20 A1 reused by C22.

Reads hashes/JSON/Python AST only. No policy deserialization, imports of the
learning implementation, environment construction, prediction or simulation.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import os
import signal
from pathlib import Path

HERE = Path(__file__).resolve().parent
W = HERE.parent
OLD = W / "continuation20"
PREVIOUS = W / "continuation21"
A_RUN = OLD / "train_A_1"
CONTRACT_ID = "C22_fixed_B_eval_repair_v1"
A_MODEL_SHA256 = "1a630229fe9b44abe75abe4e47c98be50a55a043f0163f20f57b0b97640904e5"
A_READER_SHA256 = "189792963dc0ff54513fe5091b0feb34d9547dfc0f9ccf45f517f06a68d26925"
PARENT_SHA256 = "1fcfe833d8a7cfdbbc224c9007a35e44c035674ce948b51557434b77838476fb"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def file_identity(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "missing regular reuse source: " + str(path))
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1 << 20), b""):
            digest.update(part)
            size += len(part)
    return {"sha256": digest.hexdigest(), "bytes": size}


def validate_spec_equivalence(old, new):
    """All shared learning/controller values remain fixed; only B cycle1 changes."""
    require(new["parent"] == old["parent"] and new["controller"] == old["controller"],
            "C22 parent or numerical C18 controller contract changed")
    fields = ("B_bumps_vx_cycles", "B_cycle_period", "B_override_slots",
        "actor_clip_l2", "critic_clip_l2", "batch_kl_stage_stop_strict_above", "batch_size",
        "clip_range", "command_seed", "controls_per_arm_per_stage", "ent_coef", "episode_ticks",
        "gae_lambda", "gamma", "learning_rate", "measurement_seed", "min_optimizer_steps_per_train",
        "n_epochs_max", "n_steps", "net_arch", "raw_start", "reset_actor", "reset_adam",
        "rollouts_per_arm_per_stage", "sb3_early_stop_strict_above", "target_kl", "terrain_slots",
        "vf_coef")
    require(all(new["training"][key] == old["training"][key] for key in fields),
            "shared A/B learning, seed, action permission or A-course rules changed")
    require(new["training"]["ppo_seed_by_stage"][0] == old["training"]["ppo_seed_by_stage"][0] == 201001,
            "shared stage1 PPO seed changed")
    cycles = copy.deepcopy(old["training"]["B_yaw_cycles"])
    cycles[1] = {"vx": 1., "segments": [[425,525,-.25],[525,725,.25],[725,825,-.25]]}
    require(new["training"]["B_yaw_cycles"] == cycles,
            "C22 B-cycle1 is not the sole preregistered course repair")
    expected_evaluation = copy.deepcopy(old["evaluation"])
    for split,edges,amplitude in (("development",(430,530,730,830),.30),
                                  ("final_sealed",(440,540,740,840),.28)):
        yaw_cases = [row for row in expected_evaluation[split] if "yaw" in row["case_id"]]
        require(len(yaw_cases) == 2, "frozen yaw case count differs")
        for row in yaw_cases:
            sign = 1 if row["case_id"].endswith("left") else -1
            row["yaw_segments"] = [[edges[i],edges[i+1],sign*amplitude*(1 if i != 1 else -1)]
                                   for i in range(3)]
    for key in ("actors_order", "regression_actor_order", "regression", "development", "final_sealed",
                "new_case_absolute_gates", "new_case_weight", "regression_pool"):
        require(new["evaluation"][key] == expected_evaluation[key],
                "C22 fixed evaluation values changed: " + key)
    expected_floor = {key:value for key,value in old["evaluation"]["floor"].items() if key != "seed_stage2"}
    require(new["evaluation"]["floor"] == expected_floor
            and new["training"]["max_stages"] == 1 and new["training"]["arms"] == ["B"]
            and new["training"]["total_controls_max"] == 32768,
            "C22 floor protocol or single-new-B scope changed")
    return {"shared_learning_values_equal": True, "A_recipe_rules_equal": True,
            "only_B_cycle1_commands_changed": True, "evaluation_numeric_gates_equal": True,
            "only_four_declared_evaluation_yaw_windows_changed": True}


class _ModuleNames(ast.NodeTransformer):
    """Only protocol-preserving module-location renames are accepted."""
    def visit_Constant(self, node):
        if node.value in ("spec21.json", "spec22.json"):
            node.value = "spec20.json"
        return node

    def visit_ImportFrom(self, node):
        if node.module in ("learning21", "checkpoint21", "recipes21", "learning22", "checkpoint22", "recipes22"):
            node.module = node.module[:-2] + "20"
        return self.generic_visit(node)


def source_signature(source):
    tree = _ModuleNames().visit(ast.parse(source))
    # Comments and module docstrings do not affect the numerical implementation.
    if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant) and isinstance(tree.body[0].value.value, str):
        tree.body.pop(0)
    return ast.dump(tree, include_attributes=False)


def recipe_equivalent(old_source, new_source):
    """Permit exactly the added source0..79 rejection immediately after indexing."""
    tree = ast.parse(new_source)
    functions = [node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name == "select_episode"]
    require(len(functions) == 1, "new finite selector definition absent/duplicated")
    body = functions[0].body
    guard = ast.parse('if not 0 <= source <= 79:\n    raise ValueError("C21 source index is outside the sealed finite range 0..79")').body[0]
    assignment = ast.parse("source = index + source_offset").body[0]
    require(len(body) > 3 and ast.dump(body[2],include_attributes=False) == ast.dump(assignment,include_attributes=False)
            and ast.dump(body[3],include_attributes=False) == ast.dump(guard,include_attributes=False),
            "new selector finite guard differs from its sole allowed nonnumerical rejection")
    body.pop(3)
    require(source_signature(old_source) == source_signature(ast.unparse(tree)),
            "selector differs beyond the fixed C22 spec path and exact source0..79 guard")
    return True


def validate_a_documents(old_spec, new_spec, session, manifest, metadata, reader, parentage):
    spec_proof = validate_spec_equivalence(old_spec, new_spec)
    require(new_spec["A_reuse"]["model_sha256"] == A_MODEL_SHA256
            and new_spec["A_reuse"]["final_model"] == str(A_RUN/"final_checkpoint/final_model.zip")
            and new_spec["A_reuse"]["reader"] == str(OLD/"train_A_1_readback.json")
            and new_spec["A_reuse"]["run"] == str(A_RUN),
            "C22 scientific spec does not select the frozen A1 evidence")
    require(session["schema"] == "d1-coverage-session-20-v1" and session["mode"] == "train"
            and session["arm"] == "A" and session["stage"] == 1
            and session["control_limit"] == 32768 and session["normal_native_limit"] == 163840
            and session["compiler_native_limit"] == 2 and session["controller_variant"] == "combined"
            and session["source_episode_offset"] == 0
            and (session["ppo_seed"],session["command_seed"],session["measurement_seed"]) == (201001,201002,201003)
            and session["parentpaths"]["model_sha256"] == PARENT_SHA256,
            "reused A1 session/parent/seed/budget differs")
    require(reader["schema"] == "d1-stage20-full-training-independent-readback-v1"
            and reader["run"] == str(A_RUN) and reader["arm"] == "A" and reader["stage"] == 1
            and all(reader[key] is True for key in ("training_valid", "learning_valid", "coverage_valid",
                "archive_valid", "source_closure_verified", "five_native_per_control_verified"))
            and reader["complete_controls"] == 32768
            and reader["episodes_started"] == reader["next_source_episode_index"] == 33
            and reader["reader_model_calls"] == reader["reader_physics_calls"] == 0,
            "reused A1 lacks a complete independent training/coverage readback")
    require(manifest["folder"] == str(A_RUN/"final_checkpoint")
            and manifest["stage20_course_arm"] == "A" and manifest["stage20_stage"] == 1
            and manifest["files"]["final_model.zip"]["sha256"] == A_MODEL_SHA256
            and reader["actual_learning"]["checkpoint_model_identity"] == manifest["files"]["final_model.zip"]
            and manifest["actual_optimizer_steps"] == 351,
            "reused A1 is not its unique complete final checkpoint")
    receipt = metadata["training_receipt"]
    expected_hashes = {key:old_spec["parent"][key] for key in
                       ("policy_state","optimizer_state","actor_mean_state","log_std_state")}
    require(parentage == metadata["stage20_parentage"]
            and parentage["parent_model_identity"]["sha256"] == PARENT_SHA256
            and parentage["parent_hashes"] == receipt["initial_hashes"] == expected_hashes
            and parentage["parent_optimizer_steps"] == 1280
            and parentage["parent_num_timesteps"] == 16384 and parentage["parent_n_updates"] == 64
            and parentage["actor_reinitialized"] is False and parentage["optimizer_reinitialized"] is False
            and parentage["stage_seed"] == metadata["ppo_seed"] == 201001
            and parentage["arm"] == "A" and parentage["stage"] == 1
            and receipt["qualified_for_final_checkpoint"] is True
            and receipt["actual"] == reader["actual_learning"]["actual"]
            and receipt["actual"]["optimizer_steps"] == 351
            and receipt["construction"]["optimizer_lr"] == .0003
            and receipt["construction"]["target_kl"] == .03
            and receipt["normal_kl_threshold"] == .045 and receipt["hard_kl_threshold"] == .30,
            "reused A1 parent PPO/Adam continuity or actual optimization differs")
    dimensions = {"observation_size":99,"action_size":16,"control_dt_s":.01,
        "physics_dt_s":.002,"physics_steps_per_control":5,"wheel_action_scale_rad_s":4.,
        "observation_schema":"d1-full-drive-proprio-course99-action16-v1",
        "action_schema":"d1-course-residual16-leg-angle-wheel-speed-v1",
        "controller_schema":"d1-course18-yawlimit-filteredcommonintegral-adapter-v1",
        "control_loop_schema":"d1-full-drive-world-upright-loop-v1",
        "reference_schema":"d1-course-world-upright-roll-pitch-zero-v1",
        "reward_schema":"d1-course-world-upright-bodycom-yaw-terrain-action-torque-v1",
        "task_schema":"d1-course-world-upright-rl16-task-v1"}
    require(all(metadata[key] == value for key,value in dimensions.items())
            and metadata["net_arch"] == {"pi":[128,128],"vf":[128,128]}
            and metadata["command_seed"] == 201002 and metadata["measurement_seed_stream"] == 201003
            and manifest["reload_verification"]["loaded_policy_state_sha256"] == receipt["final_hashes"]["policy_state"]
            and manifest["reload_verification"]["probe_actions_byte_exact"] is True
            and manifest["reload_verification"]["matches_training_final_policy_state"] is True,
            "reused A1 99D/16D/reward/C18 or strict final-reload identity differs")
    return {**spec_proof,"A_training_valid":True,"A_coverage_valid":True,
        "A_checkpoint_sha256":A_MODEL_SHA256,"parent_checkpoint_sha256":PARENT_SHA256,
        "parent_adam_step":1280,"A_actual_optimizer_steps":351,"A_final_adam_step":1631,
        "A_actual_training_controls":32768,"A_new_training_controls_in_C22":0,
        "first1024_cross_run_pair_required_separately":True,
        "reset_equivalence_basis":"same frozen deterministic reset/provider/reward/C18 numerical sources; actual A20/B22 first1024 initial arrays and control reset are checked separately"}


def verify_a_reuse(session, frozen_inputs, *, evaluation_identity=None):
    """Bind A1 evidence and the new numerical sources to this source-GO closure."""
    used = {}
    def bound(path):
        path = Path(path)
        actual = file_identity(path)
        require(str(path) in frozen_inputs and frozen_inputs[str(path)] == actual,
                "A reuse evidence absent/changed in C22 GO: " + str(path))
        used[str(path)] = actual
        return path
    def doc(path):
        return json.loads(bound(path).read_text())
    require(session.get("execution_contract_id") == CONTRACT_ID and session.get("stage",1) == 1,
            "A reuse requested outside the fixed single-stage C22 contract")
    bound(HERE/"reuse22.py")
    new_spec, old_spec = doc(HERE/"spec22.json"), doc(OLD/"spec20.json")
    for label,expected in new_spec["prior_evidence"].items():
        name = expected.get("path",label)
        bound(name)
        require(used[name] == {k:expected[k] for k in ("sha256","bytes")},
                "pinned prior evidence changed: " + name)
    finite = new_spec["finite_geometry"]
    previous_bridge = doc(finite["prior_A_reuse_path"])
    require(used[finite["prior_A_reuse_path"]] == finite["prior_A_reuse_identity"]
            and previous_bridge["schema"] == "d1-c21-reused-A1-evidence-v1"
            and previous_bridge["execution_contract_id"] == "C21_fixed_B_repair_v1"
            and previous_bridge["passed"] is True, "prior C21 A bridge is not its frozen success")
    for name,expected in previous_bridge["inputs"].items():
        bound(name)
        require(used[name] == expected, "C21 A bridge historical source changed: " + name)
    previous_spec = doc(finite["prior_spec_path"])
    require(used[finite["prior_spec_path"]] == finite["prior_spec_identity"],
            "prior C21 spec changed")
    # Only these three descriptive contract identifiers were renamed by C22.
    previous_spec = copy.deepcopy(previous_spec)
    previous_spec["training"]["A_recipe"] = previous_spec["training"]["A_recipe"].replace("C21","C22")
    prior_gate = previous_spec["evaluation"]["final_execution_gate"]
    prior_gate["fresh_first1024_A20_B22_pair_valid"] = prior_gate.pop("fresh_first1024_A20_B21_pair_valid")
    previous_spec["evaluation"]["seal_rule"] = previous_spec["evaluation"]["seal_rule"].replace("C21","C22").replace("B21_1","B22_1")
    require(new_spec["training"] == previous_spec["training"]
            and new_spec["A_reuse"] == previous_spec["A_reuse"]
            and new_spec["evaluation"]["final_execution_gate"] == previous_spec["evaluation"]["final_execution_gate"]
            and new_spec["evaluation"]["final_claim_gates"] == previous_spec["evaluation"]["final_claim_gates"],
            "C21 to C22 learning, A selection, or scientific decision gates changed")
    expected_evaluation = copy.deepcopy(previous_spec["evaluation"])
    for split in ("development","final_sealed"):
        for row,actual in zip(expected_evaluation[split],new_spec["evaluation"][split]):
            if "yaw" in row["case_id"]:
                row["yaw_segments"] = actual["yaw_segments"]
    require(new_spec["evaluation"] == expected_evaluation,
            "C21 to C22 evaluation changed beyond the four declared yaw windows")
    prior_session = doc(A_RUN/"session.json")
    manifest = doc(A_RUN/"final_checkpoint_manifest.json")
    metadata = doc(A_RUN/"final_checkpoint/final_metadata.json")
    reader_path = OLD/"train_A_1_readback.json"
    reader, reader_host = doc(reader_path), doc(OLD/"train_A_1_readback_host_receipt.json")
    require(used[str(reader_path)]["sha256"] == A_READER_SHA256
            and reader_host["failure"] is None and reader_host["exit_code"] == 0
            and reader_host["output_identity"] == used[str(reader_path)]
            and reader_host["reader_identity"] == file_identity(bound(OLD/"read_training20.py"))
            and reader_host["controls"] == reader_host["model_calls"] == 0,
            "A1 independent-reader host/source/output closure differs")
    parentage = doc(A_RUN/"parentage.json")
    result = validate_a_documents(old_spec,new_spec,prior_session,manifest,metadata,reader,parentage)
    worker, host = doc(A_RUN/"worker_receipt.json"), doc(A_RUN/"host_receipt.json")
    require(worker["execution_complete"] is True and worker["failure"] is None
            and worker["cleanup_errors"] == [] and worker["warnings"] == []
            and worker["session_identity"] == used[str(A_RUN/"session.json")]
            and host["exit_code"] == 0 and host["failure"] is None
            and host["changed_sources"] == [] and host["postcheck_complete"] is True
            and host["no_live_owned_processes"] is True,
            "A1 physical worker/host/source closure differs")
    for name, expected in prior_session["source_hashes"].items():
        require(frozen_inputs.get(name) == expected,
                "A1 historical source not preserved in C22 closure: " + name)
    for name, expected in manifest["files"].items():
        require(file_identity(bound(A_RUN/"final_checkpoint"/name)) == expected,
                "A1 checkpoint/probe payload changed")
    require(used[str(A_RUN/"final_checkpoint/final_metadata.json")]["sha256"] == manifest["metadata_sha256"],
            "A1 final metadata differs from strict manifest")
    entry = bound(HERE/"train22.py")
    imports = {node.module for node in ast.walk(ast.parse(entry.read_text())) if isinstance(node,ast.ImportFrom)}
    algorithm_bridges = {}
    for family in ("learning","checkpoint"):
        selected = imports & {family+"20",family+"21",family+"22"}
        require(len(selected) == 1, "C22 train entry lacks one explicit numerical implementation: " + family)
        module = selected.pop()
        old_source = bound(OLD/(family+"20.py"))
        new_source = bound(({"20":OLD,"21":PREVIOUS,"22":HERE}[module[-2:]])/(module+".py"))
        require(source_signature(old_source.read_text()) == source_signature(new_source.read_text()),
                "C22 numerical algorithm differs beyond permitted module rename: " + module)
        algorithm_bridges[family] = {"old_source":str(old_source),"new_source":str(new_source),
            "old_identity":used[str(old_source)],"new_identity":used[str(new_source)],
            "AST_equal_after_module_rename":True}
    old_recipe,new_recipe = bound(OLD/"recipes20.py"),bound(HERE/"recipes22.py")
    recipe_equivalent(old_recipe.read_text(),new_recipe.read_text())
    common = [W/"continuation18"/n for n in ("controller18.py","residual18.py")]
    common += [W/"course_impl08"/n for n in ("full_drive_env_08.py","full_drive_controller_08.py",
        "full_drive_loop_08.py","full_drive_servo_08.py","full_drive_observation_08.py","course_plant_08.py")]
    common += [W/"upright11/world_upright_course_11.py"]
    repository = Path("/home/lyh/wheel-legged-control-lab")
    common += [repository/"src/wheel_legged_control/d1"/n for n in ("control_loop.py","control_context.py",
        "state_provider.py","state_estimation.py","wheel_leg_controller.py","model.py","actuator_channel.py")]
    for path in common:
        bound(path)
        require(used[str(path)] == prior_session["source_hashes"].get(str(path)),
                "A1/C22 numerical controller/reward/reset source changed")
    if evaluation_identity is not None:
        require(session["eval_manifests"]["A"] == manifest
                and evaluation_identity["checkpoint_sha256_by_actor"]["A"] == A_MODEL_SHA256
                and evaluation_identity["checkpoint_paths_by_actor"]["A"] == str(A_RUN/"final_checkpoint/final_model.zip"),
                "C22 evaluated A differs from the preselected reused A1")
    return {"schema":"d1-c22-reused-A1-evidence-v1","execution_contract_id":CONTRACT_ID,
        "passed":True,**result,"inputs":dict(sorted({**prior_session["source_hashes"],**used}.items())),
        "model_calls":0,"physics_calls":0,
        "source_freeze_verified":True,"A_reader_identity":used[str(reader_path)],
        "prior_C21_A_reuse_identity":used[finite["prior_A_reuse_path"]],
        "C21_to_C22_shared_training_and_claim_gates_equal":True,
        "algorithm_source_bridges":algorithm_bridges,
        "A_selector_source_equivalence":"AST equal except spec path and exact source0..79 guard; all actual A sources0..32 satisfy guard"}


def collect_reuse_inputs():
    """Hash the full historical source closure and the minimal bridge documents."""
    prior_session = json.loads((A_RUN/"session.json").read_text())
    new_spec = json.loads((HERE/"spec22.json").read_text())
    previous_bridge = json.loads(Path(new_spec["finite_geometry"]["prior_A_reuse_path"]).read_text())
    frozen = dict(previous_bridge["inputs"])
    for name,expected in prior_session["source_hashes"].items():
        require(frozen.get(name) == expected, "C21 bridge omitted original A source: " + name)
    for name,expected in frozen.items():
        require(file_identity(name) == expected, "historical A1 frozen source changed: " + name)
    new_spec = json.loads((HERE/"spec22.json").read_text())
    required = [HERE/"spec22.json",HERE/"reuse22.py",HERE/"train22.py",HERE/"recipes22.py",
        OLD/"spec20.json",OLD/"read_training20.py",OLD/"train_A_1_readback.json",
        OLD/"train_A_1_readback_host_receipt.json",A_RUN/"session.json",A_RUN/"worker_receipt.json",
        A_RUN/"host_receipt.json",A_RUN/"parentage.json",A_RUN/"final_checkpoint_manifest.json",
        A_RUN/"final_checkpoint/final_metadata.json"]
    required += [Path(expected.get("path",name)) for name,expected in new_spec["prior_evidence"].items()]
    required += [Path(new_spec["finite_geometry"][key]) for key in ("prior_A_reuse_path","prior_spec_path")]
    manifest = json.loads((A_RUN/"final_checkpoint_manifest.json").read_text())
    required += [A_RUN/"final_checkpoint"/name for name in manifest["files"]]
    entry = ast.parse((HERE/"train22.py").read_text())
    imports = {node.module for node in ast.walk(entry) if isinstance(node,ast.ImportFrom)}
    for family in ("learning","checkpoint"):
        selected = imports & {family+"20",family+"21",family+"22"}
        require(len(selected) == 1, "cannot identify selected C22 numerical module")
        name = selected.pop()
        required += [OLD/(family+"20.py"),({"20":OLD,"21":PREVIOUS,"22":HERE}[name[-2:]])/(name+".py")]
    for path in required:
        actual = file_identity(path)
        require(str(path) not in frozen or frozen[str(path)] == actual,
                "historical source would be overwritten in reuse closure: " + str(path))
        frozen[str(path)] = actual
    return dict(sorted(frozen.items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=HERE/"a_reuse_22.json")
    args = parser.parse_args()
    require(not args.output.exists(), "A reuse output already exists; do not overwrite evidence")
    def timeout(_signum,_frame):
        raise TimeoutError("C22 A reuse pure preflight exceeded 600 seconds")
    signal.signal(signal.SIGALRM,timeout)
    signal.alarm(600)
    inputs = {}
    try:
        inputs = collect_reuse_inputs()
        result = verify_a_reuse({"execution_contract_id":CONTRACT_ID,"stage":1},inputs)
    except Exception as error:
        result = {"schema":"d1-c22-reused-A1-evidence-v1","execution_contract_id":CONTRACT_ID,
            "passed":False,"failure":repr(error),"inputs":inputs,"model_calls":0,"physics_calls":0}
    finally:
        signal.alarm(0)
    with args.output.open("x",encoding="utf-8") as stream:
        json.dump(result,stream,sort_keys=True,indent=2,allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"output":str(args.output),"passed":result["passed"],
        "inputs":len(result["inputs"]),"identity":file_identity(args.output),
        "model_calls":0,"physics_calls":0},sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
