"""C22 pure reuse/score tests; no checkpoint loading or physics imports."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parent
W = HERE.parent
for directory in (W/"course_impl08", W/"continuation18", HERE):
    sys.path.insert(0,str(directory))
from reuse22 import (A_RUN, OLD, A_MODEL_SHA256, validate_a_documents,
                     validate_spec_equivalence, source_signature, recipe_equivalent)
import score22


def documents():
    def read(path):
        return json.loads(path.read_text())
    return [read(OLD/"spec20.json"),read(HERE/"spec22.json"),
        read(A_RUN/"session.json"),read(A_RUN/"final_checkpoint_manifest.json"),
        read(A_RUN/"final_checkpoint/final_metadata.json"),
        read(OLD/"train_A_1_readback.json"),read(A_RUN/"parentage.json")]


def test_existing_A1_metadata_proves_unique_valid_351_update_final():
    result = validate_a_documents(*documents())
    assert result["A_training_valid"] and result["A_coverage_valid"]
    assert result["A_checkpoint_sha256"] == A_MODEL_SHA256
    assert result["A_actual_optimizer_steps"] == 351 and result["A_final_adam_step"] == 1631
    assert result["A_new_training_controls_in_C22"] == 0
    assert result["first1024_cross_run_pair_required_separately"]


@pytest.mark.parametrize("tamper",("seed","checkpoint","parent_adam","optimizer_count","actor_reset","hyperparameter"))
def test_reuse_rejects_changed_scientific_or_checkpoint_identity(tamper):
    args = documents()
    if tamper == "seed":
        args[2]["ppo_seed"] += 1
    elif tamper == "checkpoint":
        args[3]["files"]["final_model.zip"]["sha256"] = "0"*64
    elif tamper == "parent_adam":
        args[6]["parent_optimizer_steps"] = 0
    elif tamper == "optimizer_count":
        args[3]["actual_optimizer_steps"] = 512
    elif tamper == "actor_reset":
        args[6]["actor_reinitialized"] = True
    else:
        args[1]["training"]["target_kl"] = .04
    with pytest.raises(ValueError):
        validate_a_documents(*args)


def test_numerical_source_normalization_does_not_hide_arithmetic_changes():
    old = "from learning20 import ClipAudit20\nVALUE = .03\n"
    new = "from learning21 import ClipAudit20\nVALUE = .03\n"
    assert source_signature(old) == source_signature(new)
    assert source_signature(old) != source_signature(new.replace(".03", ".04"))
    assert source_signature(old) != source_signature(new.replace("ClipAudit20", "OtherAudit"))


def test_exact_finite_guard_is_the_only_allowed_selector_difference():
    old = (OLD/"recipes20.py").read_text()
    new = (HERE/"recipes22.py").read_text()
    assert recipe_equivalent(old,new)
    with pytest.raises(ValueError):
        recipe_equivalent(old,new.replace("0 <= source <= 79","0 <= source <= 78"))
    with pytest.raises(ValueError):
        recipe_equivalent(old,new.replace("if source < 8:","if source < 7:"))


def test_C22_fixed_schedules_only_change_four_declared_yaw_windows():
    old,new = documents()[:2]
    assert validate_spec_equivalence(old,new)["only_four_declared_evaluation_yaw_windows_changed"]
    commands = score22.expected_raw_commands("final_yaw_left")
    assert commands[199]["forward_velocity_mps"] == 0.
    assert commands[200]["forward_velocity_mps"] == 1.1
    assert commands[440]["yaw_rate_rps"] == .28 and commands[540]["yaw_rate_rps"] == -.28
    assert score22.case_windows("final_yaw_left")["drive"] == [200,880]
    assert score22.case_windows("flat_1p2_yaw_mirror")["hold"] == [415,815]
    bad = deepcopy(new)
    bad["evaluation"]["development"][0]["speed_mps"] = .4
    with pytest.raises(ValueError):
        validate_spec_equivalence(old,bad)


@pytest.mark.parametrize("split,index,edge", (("development",0,530),("development",1,730),
                                               ("final_sealed",0,540),("final_sealed",1,740)))
def test_reuse_rejects_any_other_yaw_window(split,index,edge):
    old,new = documents()[:2]
    new["evaluation"][split][index]["yaw_segments"][0][1] = edge+1
    with pytest.raises(ValueError):
        validate_spec_equivalence(old,new)
