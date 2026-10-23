"""One-shot C50 model admission; no training or simulator is permitted.

The controller launches a cold model worker, waits for its exit, then launches
a cold NumPy reader. Admission models are discarded and cannot seed a run.
"""

from __future__ import annotations

import argparse
import gc
import importlib.abc
import io
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b22_sigma50_contract as contract

ROOT = contract.ROOT
SCRIPT = Path(__file__).resolve()
PLAN = ROOT / "docs/experiments50/training_admission50_preregistered.json"
OUTPUT = ROOT / "results/b22_sigma50_model_admission_20261024_01"
ARMS = ("baseline_s51", "candidate_s51")
UNIT = "control50-wheel-model-admission-20261024-01.service"
SCOPE = "/user.slice/user-1000.slice/user@1000.service/app.slice/" + UNIT
LIMITS = {"load": 2, "torch_load": 6, "predict": 2, "save": 0, "learn": 0,
          "train": 0, "forward": 0, "evaluate_actions": 0, "predict_values": 0, "backward": 0}
ZERO_APIS = ("optimizer_step", "autograd_backward", "autograd_grad", "stub_reset", "stub_step", "audited_train")
SIMULATORS = frozenset({"mujoco", "mujoco_py", "engine_binding", "pybullet", "isaacgym", "isaaclab"})
QUOTA = 512 << 20
SCHEMA = "b22-sigma50-model-admission-v1"


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return contract.read_json(path)


def write(path, value):
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    write_bytes(path, data)


def write_bytes(path, data):
    path = Path(path)
    require(path.is_relative_to(OUTPUT) and not path.is_symlink(), "admission output escapes its reservation")
    used = sum(p.stat().st_size for p in OUTPUT.rglob("*") if p.is_file())
    require(used + len(data) <= QUOTA, "admission output quota exhausted")
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


class Writer50:
    def commit_json(self, path, value):
        write(path, value)

    def commit_npz(self, path, arrays):
        import numpy as np

        memory = io.BytesIO()
        np.savez(memory, **arrays)
        write_bytes(path, memory.getvalue())


class BlockImports50(importlib.abc.MetaPathFinder):
    def __init__(self, forbidden):
        self.forbidden = forbidden
        self.rejected = []

    def find_spec(self, fullname, path=None, target=None):
        if fullname.partition(".")[0] in self.forbidden:
            self.rejected.append(fullname)
            raise ImportError("admission forbidden dependency: " + fullname)


def block_imports(forbidden):
    require(not any(name.partition(".")[0] in forbidden for name in sys.modules),
            "admission requires a cold dependency boundary")
    blocker = BlockImports50(forbidden)
    sys.meta_path.insert(0, blocker)
    return blocker


def interrupted(signum, _frame):
    raise InterruptedError("admission interrupted by " + signal.Signals(signum).name)


def install_signals():
    for number in (signal.SIGINT, signal.SIGTERM, signal.SIGALRM):
        signal.signal(number, interrupted)


def resource50(minimum):
    from resource_guard42 import read_memory

    snapshot = read_memory(SCOPE)
    require(snapshot["effective_available_bytes"] >= minimum, "admission effective memory below reservation")
    require(os.getpriority(os.PRIO_PROCESS, 0) >= 10, "admission requires Nice >= 10")
    require(all(os.environ.get(key) == "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")),
            "admission requires single-thread numeric libraries")
    return snapshot


def validate_plan50():
    plan = read(PLAN)
    require(plan == {"schema": SCHEMA, "arms": list(ARMS), "output": str(OUTPUT), "unit": UNIT,
                     "owned_scope": SCOPE, "model_api_limits": LIMITS, "zero_api_limits": dict.fromkeys(ZERO_APIS, 0),
                     "predict_rows_each": 32, "predict_deterministic": True, "stub_instances": 2,
                     "observation_box": {"shape": [99], "dtype": "float32", "low": -5.0, "high": 5.0},
                     "action_box": {"shape": [16], "dtype": "float32", "low": -1.0, "high": 1.0},
                     "training_controls": 0, "native_steps": 0, "mujoco_constructions": 0,
                     "independent_reader_after_model_worker_exit": True, "training_attempts_consumed": 0,
                     "models_discarded": True, "resume_permitted": False, "retry_permitted": False,
                     "soft_wall_s": 900, "quota_bytes": QUOTA, "start_effective_bytes": 4 << 30,
                     "reserve_effective_bytes": 2 << 30, "memory_high_bytes": 2 << 30,
                     "memory_max_bytes": 3 << 30, "swap_max_bytes": 0, "nice": 10,
                     "minimum_complete_counts": {"load": 2, "torch_load": 6, "predict": 2, "predict_rows": 64}},
            "admission preregistration differs from the exact implementation")
    return plan


def registration50():
    import run_b22_sigma50 as parent

    inputs = dict(contract.dependency_hashes50())
    for mapping in (parent.worker_api().required_inputs42(), parent.reader_api().required_inputs42(),
                    {str(ROOT / name): contract.identity(ROOT / name) for name in parent.AUTHORED},
                    {str(PLAN): contract.identity(PLAN), str(SCRIPT): contract.identity(SCRIPT)}):
        for path, expected in mapping.items():
            require(path not in inputs or inputs[path] == expected, "conflicting admission source: " + path)
            inputs[path] = expected
    return {"schema": SCHEMA, "plan_identity": contract.identity(PLAN), "script_identity": contract.identity(SCRIPT),
            "source_audit_identity": contract.identity(contract.SOURCE_AUDIT), "source_hashes": inputs,
            "parent_pid": os.getpid(), "started_monotonic": time.monotonic(), "retry_permitted": False}


def verify_registration50(*, rehash=False, child=False):
    validate_plan50()
    registration = read(OUTPUT / "registration.json")
    require(registration["schema"] == SCHEMA and registration["plan_identity"] == contract.identity(PLAN)
            and registration["script_identity"] == contract.identity(SCRIPT)
            and registration["source_audit_identity"] == contract.identity(contract.SOURCE_AUDIT)
            and registration["retry_permitted"] is False, "admission registration identity changed")
    if child:
        require(registration["parent_pid"] == os.getppid(), "admission child lacks its original parent")
    if rehash:
        for path, expected in registration["source_hashes"].items():
            require(contract.identity(path) == expected, "admission frozen source changed: " + path)
    return registration


def configure_paths50():
    for path in reversed(read(contract.MANIFEST)["runtime_environment"]["PYTHONPATH"].split(os.pathsep)):
        if path not in sys.path:
            sys.path.insert(0, path)
    require("LD_PRELOAD" not in os.environ and "LD_LIBRARY_PATH" not in os.environ,
            "model-only admission must not preload a simulator")


class ZeroCounter50:
    def __init__(self):
        self.counts = {name: {"attempted": 0, "returned": 0, "rejected": 0} for name in ZERO_APIS}
        self.restore = []

    def reject(self, name):
        self.counts[name]["rejected"] += 1
        raise RuntimeError("admission zero API budget: " + name)

    def patch(self, owner, name, key):
        original = owner.__dict__[name]

        def forbidden(*_args, **_kwargs):
            self.reject(key)

        setattr(owner, name, forbidden)
        self.restore.append((owner, name, original))

    def close(self):
        for owner, name, original in reversed(self.restore):
            setattr(owner, name, original)
        self.restore.clear()


def worker50():
    registration = verify_registration50(child=True)
    install_signals()
    signal.setitimer(signal.ITIMER_REAL, max(.01, registration["started_monotonic"] + 850 - time.monotonic()))
    blocker = block_imports(SIMULATORS)
    report = {"schema": SCHEMA, "status": "INCOMPLETE", "pid": os.getpid(), "arms": {},
              "actual_model_api_counts": None, "known_model_api_counts": None, "error": None,
              "registration_identity": contract.identity(OUTPUT / "registration.json")}
    counter = None
    zero = ZeroCounter50()
    stubs = 0
    try:
        resource50(4 << 30)
        configure_paths50()
        import gymnasium as gym
        import numpy as np
        import torch
        import worker_b22_sigma50 as worker
        from budget_spec_11 import BudgetSpec
        from runtime_support22 import ModelCounter

        torch.set_num_threads(1)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        zero.patch(torch.optim.Adam, "step", "optimizer_step")
        zero.patch(torch.autograd, "backward", "autograd_backward")
        zero.patch(torch.autograd, "grad", "autograd_grad")

        class SpaceStub(gym.Env):
            def __init__(self):
                self.observation_space = gym.spaces.Box(-5.0, 5.0, (99,), dtype=np.float32)
                self.action_space = gym.spaces.Box(-1.0, 1.0, (16,), dtype=np.float32)

            def reset(self, **_kwargs):
                zero.reject("stub_reset")

            def step(self, _action):
                zero.reject("stub_step")

        writer = Writer50()
        counter = ModelCounter(False, dict(LIMITS))
        with counter:
            for arm in ARMS:
                resource50(2 << 30)
                folder = OUTPUT / arm
                folder.mkdir(exist_ok=False)
                session = worker.session_defaults42("train", arm)
                write(folder / "session.json", session)
                worker._initial_context50 = {"session": session, "output": folder, "writer": writer}
                try:
                    learning = worker.seed_learning_module42(session)
                    require(sys.modules["learning21"] is learning and learning.STAGE_SEEDS[1] == session["ppo_seed"],
                            "admission private learning alias/seed differs from formal worker")
                    import rl16_learning_11 as strict

                    model_class = strict._audited_ppo_class()
                    if not any(owner is model_class for owner, _name, _original in zero.restore):
                        zero.patch(model_class, "train", "audited_train")
                    require(stubs < 2, "admission stub construction cap exhausted")
                    stubs += 1
                    env = SpaceStub()
                    model, audit, clip, parentage = learning.load_warm_start20(
                        env, BudgetSpec(32768, 1024, 256, 4), session["ppo_seed"], "B",
                        session["parentpaths"], target_kl=.03, stage=2)
                    probe_folder = Path(session["parentpaths"]["model"]).parent
                    observations = np.fromfile(probe_folder / "final_probe_observations_f32.bin", dtype=np.float32).reshape(32, 99)
                    expected = np.fromfile(probe_folder / "final_probe_actions_f32.bin", dtype=np.float32).reshape(32, 16)
                    counter.phase = "probe"
                    actions, state = model.predict(observations, deterministic=True)
                    require(state is None and actions.dtype == np.dtype("float32") and actions.shape == (32, 16)
                            and actions.tobytes() == expected.tobytes(), "admission deterministic parent probe differs")
                    write_bytes(folder / "probe_actions.f32.bin", actions.tobytes())
                    write(folder / "parentage.json", parentage)
                    write(folder / "initial_hashes.json", audit.initial_hashes)
                    report["arms"][arm] = {"status": "COMPLETE", "probe_rows": 32, "probe_byte_exact": True,
                                           "proof_identity": contract.identity(folder / "initial_sigma50/proof.json"),
                                           "arrays_identity": contract.identity(folder / "initial_sigma50/tensors.npz"),
                                           "parentage_identity": contract.identity(folder / "parentage.json"),
                                           "initial_hashes_identity": contract.identity(folder / "initial_hashes.json"),
                                           "probe_identity": contract.identity(folder / "probe_actions.f32.bin"),
                                           "session_identity": contract.identity(folder / "session.json")}
                    model.get_env().close()
                    del model, audit, clip, learning, env
                    gc.collect()
                finally:
                    worker._initial_context50 = None
        counts = counter.summary()
        require(all(counts["counts"].get(name, {}).get("returned", 0) == limit for name, limit in LIMITS.items())
                and counts["counts"]["predict"]["rows_returned"] == 64
                and all(row["attempted"] == row["returned"] for row in counts["counts"].values()),
                "admission actual API counts differ")
        require(all(row == {"attempted": 0, "returned": 0, "rejected": 0} for row in zero.counts.values())
                and not blocker.rejected and stubs == 2, "admission forbidden call was attempted")
        report.update(status="COMPLETE", actual_model_api_counts=counts)
    except BaseException as error:  # noqa: BLE001 -- retain partial counts for interrupts; never retry
        report["error"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        zero.close()
        report.update(known_model_api_counts=None if counter is None else counter.summary(), zero_api_counts=zero.counts,
                      simulator_import_rejections=blocker.rejected, stub_instances=stubs,
                      simulator_modules_loaded=[name for name in sys.modules if name.partition(".")[0] in SIMULATORS],
                      native_steps=0, mujoco_constructions=0, training_controls=0, training_attempts_consumed=0,
                      models_reused_for_training=False)
        write(OUTPUT / "worker_report.json", report)
    return 0 if report["status"] == "COMPLETE" else 2


def reader50():
    verify_registration50(child=True)
    install_signals()
    block_imports(SIMULATORS | {"torch", "stable_baselines3", "gymnasium", "gym", "tensorflow", "jax"})
    report = {"schema": SCHEMA, "status": "INCOMPLETE", "pid": os.getpid(), "error": None, "arms": {}}
    try:
        configure_paths50()
        import read_b22_sigma50 as reader

        np = reader.checked_numpy50()
        worker = read(OUTPUT / "worker_report.json")
        require(worker["status"] == "COMPLETE" and worker["pid"] != os.getpid(), "admission model worker incomplete")
        require(not (Path("/proc") / str(worker["pid"])).exists(), "model worker must exit before independent reader")
        for arm in ARMS:
            folder = OUTPUT / arm
            row = worker["arms"][arm]
            for key, name in (("proof_identity", "initial_sigma50/proof.json"), ("arrays_identity", "initial_sigma50/tensors.npz"),
                              ("parentage_identity", "parentage.json"), ("initial_hashes_identity", "initial_hashes.json"),
                              ("probe_identity", "probe_actions.f32.bin"), ("session_identity", "session.json")):
                require(contract.identity(folder / name) == row[key], "admission worker artifact changed: " + name)
            session = read(folder / "session.json")
            parentage = read(folder / "parentage.json")

            def arrays(path):
                with np.load(path, allow_pickle=False) as archive:
                    return {name: archive[name] for name in archive.files}

            proof = reader.verify_initial50(folder, session, {"initial_hashes": read(folder / "initial_hashes.json")},
                                            parentage, documents=read, arrays=arrays)
            expected = Path(session["parentpaths"]["model"]).with_name("final_probe_actions_f32.bin")
            require((folder / "probe_actions.f32.bin").read_bytes() == expected.read_bytes(),
                    "independent parent deterministic probe differs")
            report["arms"][arm] = proof
        require(not any(name.partition(".")[0] in SIMULATORS | {"torch", "stable_baselines3"} for name in sys.modules),
                "independent initial reader imported forbidden frameworks")
        report.update(status="COMPLETE", worker_report_identity=contract.identity(OUTPUT / "worker_report.json"),
                      no_torch_or_simulator_imported=True)
    except BaseException as error:  # noqa: BLE001 -- retain the failed independent-read receipt on interruption
        report["error"] = {"type": type(error).__name__, "message": str(error)}
    write(OUTPUT / "reader_report.json", report)
    return 0 if report["status"] == "COMPLETE" else 2


def run50():
    validate_plan50()
    install_signals()
    resource50(4 << 30)
    OUTPUT.mkdir(exist_ok=False)
    report = {"schema": SCHEMA, "status": "INCOMPLETE", "error": None, "stages": {}, "actual_counts": None}
    child = None
    try:
        write(OUTPUT / "registration.json", registration50())
        verify_registration50(rehash=True)
        environment = os.environ.copy()
        for key in ("LD_PRELOAD", "LD_LIBRARY_PATH"):
            environment.pop(key, None)
        environment["PYTHONPATH"] = read(contract.MANIFEST)["runtime_environment"]["PYTHONPATH"]
        for mode in ("worker", "read"):
            resource50(4 << 30)
            with (OUTPUT / (mode + ".log")).open("xb") as log:
                child = subprocess.Popen([sys.executable, "-B", str(SCRIPT), "--mode", mode],
                                         cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
                deadline = read(OUTPUT / "registration.json")["started_monotonic"] + 900
                while True:
                    require(time.monotonic() < deadline, "admission total wall budget exhausted")
                    resource50(2 << 30)
                    require(sum(path.stat().st_size for path in OUTPUT.rglob("*") if path.is_file()) < QUOTA - (1 << 20),
                            "admission output reserve exhausted")
                    try:
                        code = child.wait(timeout=.5)
                        break
                    except subprocess.TimeoutExpired:
                        continue
                report["stages"][mode] = {"returncode": code, "pid": child.pid}
                child = None
            require(code == 0, "admission stage failed: " + mode)
        verify_registration50(rehash=True)
        report.update(status="COMPLETE", actual_counts=read(OUTPUT / "worker_report.json")["actual_model_api_counts"],
                      worker_identity=contract.identity(OUTPUT / "worker_report.json"),
                      reader_identity=contract.identity(OUTPUT / "reader_report.json"))
    except BaseException as error:  # noqa: BLE001 -- terminate owned child and preserve the one-shot terminal receipt
        report["error"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=15)
        report["known_partial_model_counts"] = (read(OUTPUT / "worker_report.json").get("known_model_api_counts")
                                                 if (OUTPUT / "worker_report.json").exists() else None)
        report["unknown_counts"] = report["actual_counts"] is None and report["known_partial_model_counts"] is None
        report.update(training_attempts_consumed=0, training_controls=0, native_steps=0,
                      mujoco_constructions=0, resume_permitted=False, retry_permitted=False)
        write(OUTPUT / "report.json", report)
    print(json.dumps({"status": report["status"], "output": str(OUTPUT), "error": report["error"]}))
    return 0 if report["status"] == "COMPLETE" else 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("describe", "run", "worker", "read"), default="describe")
    mode = parser.parse_args().mode
    if mode == "describe":
        print(json.dumps(validate_plan50(), indent=2, sort_keys=True))
        return 0
    return {"run": run50, "worker": worker50, "read": reader50}[mode]()


if __name__ == "__main__":
    raise SystemExit(main())
