"""Archive identity and authorization checks without constructing a simulator."""
import json

import numpy as np
import pytest

from scripts.probe_d1_jump_readiness import bitwise_equal, tagged, validate_preflight


def test_numeric_identity_includes_dtype_and_bits():
    assert bitwise_equal(np.zeros(3), np.zeros(3))
    assert not bitwise_equal(np.zeros(3), np.zeros(3, dtype=np.float32))
    assert not bitwise_equal(np.array([0.]), np.array([-0.]))
    with pytest.raises(TypeError, match="pointer"):
        bitwise_equal([{"x": 1}], [{"x": 1}])


def test_invalid_preflight_fails_before_model_construction(tmp_path):
    file = tmp_path / "preflight.json"
    file.write_text(json.dumps({"passed": True, "root_execution_authorized": True,
                                "frozen77_unchanged": True, "new_budget_control": 1201,
                                "new_budget_native_substeps": 6000}))
    with pytest.raises(ValueError, match="1200/6000"):
        validate_preflight(file, tmp_path / "contract", tmp_path / "cases")


def test_nonfinite_records_remain_explicit_and_json_encodable():
    values = tagged({"state": np.array([float("nan"), float("inf"), 3.])})
    assert values == {"state": [{"nonfinite": "nan"}, {"nonfinite": "inf"}, 3.]}
    json.dumps(values, allow_nan=False)
