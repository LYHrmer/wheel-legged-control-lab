import json
from io import StringIO

import mujoco
import numpy as np

from scripts import run_terrain_preview85 as run
from scripts import terrain_preview84_trials as prior
from scripts import terrain_preview85_common as common
from scripts.terrain_preview85_action import BranchCheckpoint, SupportTransferLQR
from scripts.terrain_preview85_sensing import PairedRangeClock
from wheel_legged_control.d1.linear_model import D1SagittalLinearModel
from wheel_legged_control.d1.terrain_preview import PreviewChannel


def test_complete_physical_budget_charges_calibration_even_for_full_heldout_set():
    plan = json.loads(run.PLAN.read_text())
    for n in range(3601):
        cap = run.prefix_cap(plan, n)
        assert cap <= 400 and 511 + n + 1152 + 6 * cap <= 7152
    assert run.prefix_cap(plan, 3600) == 314


def test_native_checkpoint_and_sensor_logging_without_integrating(tmp_path, monkeypatch):
    def forbidden(*args):
        raise AssertionError("preflight must not integrate")

    monkeypatch.setattr(mujoco, "mj_step", forbidden)
    model = D1SagittalLinearModel(0.9 * np.eye(4), np.ones((4, 1)), np.zeros(4), np.zeros(4), prior.DT)
    monkeypatch.setattr(common, "identify_sagittal_model", lambda *_: model)
    ep = common.new_episode("up", 0, controller_class=SupportTransferLQR)
    cmd = common.baseline_command(ep)
    reference, _, _, _, _ = prior.collect_commands(
        ep.plant, ep.controller, ep.source, ep.state, PreviewChannel(), "off",
    )
    assert cmd == reference
    capture = StringIO()
    signal = PairedRangeClock().observe(ep, cmd, capture)
    assert signal["82"]["reason"] == "waiting" and json.loads(capture.getvalue())["sequence"] == 0
    torque = ep.controller.compute(cmd, ep.state)
    assert common.physical_sample(ep, torque)["time_s"] == 0
    station = BranchCheckpoint(ep, tmp_path)
    saved = station.restore(ep)
    ep.plant.data.qvel[0] = 1
    assert station.restore(ep) == saved and ep.plant.data.qvel[0] == 0
