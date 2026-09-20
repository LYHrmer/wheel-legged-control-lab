"""Retain native physics entries and partial states when a bounded G1 probe fails.

This is instrumentation only. It delegates the original G1 execution and never
steps, resets, refreshes, repairs, or changes the plant or its command.
"""
from __future__ import annotations

import gzip
import json
import math

import mujoco
import numpy as np

from scripts import probe_d1_heading_g1 as g1
from scripts.run_d1_wheel_common_mean_study import write_json


class ProbeStateArchive:
    """Raw integrator snapshots; failed intervals need not end on a control tick."""

    def __init__(self, plant, output):
        self.plant, self.output = plant, output
        self.states = []
        self.closed = False

    def capture(self, tag):
        self.states.append((float(self.plant.data.time), self.plant.data.qpos.copy(),
                            self.plant.data.qvel.copy(), tag))

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.output is None:
            return
        # Keep the actual last integrator state even if the caller failed after
        # env.step, or a reset failed before it could publish an observation.
        self.capture("close")
        np.savez_compressed(self.output / "execution_states.npz",
            time_s=np.asarray([s[0] for s in self.states]),
            qpos=np.asarray([s[1] for s in self.states]),
            qvel=np.asarray([s[2] for s in self.states]),
            tag=np.asarray([s[3] for s in self.states]))


def run_archived_g1_episode(output, case, label, gates, env_factory, *, contact_sampler=None):
    """Run once, restore constructor/observer bindings, preserve failed substeps.

    Only use sequentially in a dedicated probe process, as required by G1's
    existing module-level native-step observer. ``env_factory`` receives the
    original constructor kwargs and must not start an episode itself.
    """
    original_env = g1.D1HeadingTrackingEnv
    original_observer = g1.NativeWrenchObserver
    entries = []
    failure = None
    owns_output = False

    def construct_env(**kwargs):
        nonlocal owns_output
        # G1 calls its constructor only after exclusive output.mkdir succeeds.
        owns_output = True
        return env_factory(**kwargs)

    class RetainedObserver(original_observer):
        def __enter__(self):
            result = super().__enter__()
            original_observed_step = mujoco.mj_step

            def retained_step(model, data, *args, **kwargs):
                if model is not self.plant.model or data is not self.plant.data:
                    return original_observed_step(model, data, *args, **kwargs)
                row = {"start_time_s": float(data.time),
                       "wrench_world": data.xfrc_applied[self.plant.base_body_id].tolist(),
                       "ctrl_nm": data.ctrl.copy().tolist(), "returned": False}
                try:
                    value = original_observed_step(model, data, *args, **kwargs)
                    row["returned"] = True
                    if contact_sampler is not None:
                        before_sample = (data.qpos.copy(), data.qvel.copy(), data.qacc_warmstart.copy(), float(data.time))
                        row["contacts"] = contact_sampler(self.plant)
                        after_sample = (data.qpos, data.qvel, data.qacc_warmstart, float(data.time))
                        if any(not np.array_equal(a, b, equal_nan=True) for a, b in zip(before_sample, after_sample)):
                            raise RuntimeError("native contact observer changed integrator or warm start")
                    return value
                except BaseException as error:
                    row["error"] = {"type": type(error).__name__, "message": str(error)}
                    row["qpos_after_error"] = data.qpos.copy().tolist()
                    row["qvel_after_error"] = data.qvel.copy().tolist()
                    raise
                finally:
                    row["end_time_s"] = float(data.time)
                    row["actual_dt_s"] = float(data.time) - row["start_time_s"]
                    entries.append(row)

            mujoco.mj_step = retained_step
            return result

    g1.D1HeadingTrackingEnv = construct_env
    g1.NativeWrenchObserver = RetainedObserver
    try:
        return g1.run_episode(output, case, {"kind": "zero_action", "label": label}, gates)
    except BaseException as error:
        failure = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        g1.D1HeadingTrackingEnv = original_env
        g1.NativeWrenchObserver = original_observer
        # G1 creates this directory before construction; a directory-creation
        # failure must not cause us to overwrite somebody else's evidence.
        if owns_output:
            nonfinite, paths = [], []

            def json_safe(value, path="", *, record=True):
                if isinstance(value, float) and not math.isfinite(value):
                    if record:
                        nonfinite.append(value)
                        paths.append(path)
                    return {"nonfinite_float": "NaN" if math.isnan(value) else "+Infinity" if value > 0 else "-Infinity"}
                if isinstance(value, dict):
                    return {k: json_safe(v, path+"/"+k, record=record) for k, v in value.items()}
                if isinstance(value, list):
                    return [json_safe(v, path+"/"+str(i), record=record) for i, v in enumerate(value)]
                return value

            archival_error = None
            try:
                encoded = [json_safe(entry, str(i)) for i, entry in enumerate(entries)]
                if nonfinite:
                    np.savez_compressed(output / "native_nonfinite_values.npz",
                        paths=np.asarray(paths), values=np.asarray(nonfinite),
                        ctrl_nm=np.asarray([e["ctrl_nm"] for e in entries]),
                        wrench_world=np.asarray([e["wrench_world"] for e in entries]))
                with gzip.open(output / "native_physics_entries.jsonl.gz", "xt") as stream:
                    for entry in encoded:
                        stream.write(json.dumps(entry, allow_nan=False) + "\n")
            except Exception as error:  # noqa: BLE001 -- preserve the original physics failure and write an archive error receipt
                archival_error = {"type": type(error).__name__, "message": str(error)}
            finally:
                write_json(output / "native_entry_receipt.json", json_safe({
                    "observed_native_calls": len(entries),
                    "returned_native_calls": sum(e["returned"] for e in entries),
                    "summed_actual_dt_s": sum(e["actual_dt_s"] for e in entries),
                    "failure": failure, "archival_error": archival_error,
                    "nonfinite_value_count": len(nonfinite),
                    "nonfinite_encoding": "tagged JSON object; original float values and paths in NPZ",
                    "no_retry_or_padding": True,
                }, record=False))
            if archival_error is not None and failure is None:
                raise RuntimeError(f"native evidence archival failed: {archival_error}")
