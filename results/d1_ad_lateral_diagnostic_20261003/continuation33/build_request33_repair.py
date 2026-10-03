"""Prepare one exclusive C33 development_02 request from a new reviewed GO.

This only writes a request. It does not import the robot stack or start a host.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
CONTRACT = "C33_fixed_S_curve_feasibility_v1"
SITE = "/home/lyh/.local/lib/python3.10/site-packages"
OUTPUT = HERE / "development_02"
MODEL_LIMITS = {"load": 1, "torch_load": 3, "predict": 4801,
                "forward": 0, "evaluate_actions": 0, "predict_values": 0,
                "backward": 0, "learn": 0, "train": 0, "save": 0}
STATIC_CAPS = {"copy": 12128, "forward": 1351552, "fullM": 12000,
               "jac": 48000, "jacBody": 1158144, "objectVelocity": 24008}


def build(source_go: Path, output: Path):
    source_go = source_go.resolve(strict=True)
    if source_go == HERE / "source_go33.json":
        raise RuntimeError("C33 repair requires a new superseding source GO")
    if output.exists():
        raise FileExistsError(output)
    go = json.loads(source_go.read_text())
    if (go.get("decision") != "GO" or
            go.get("execution_contract_id") != CONTRACT or
            set(go.get("arms", {})) != {"development"}):
        raise RuntimeError("C33 repair requires reviewed single-arm C33 GO")
    spec = go["arms"]["development"]
    required = {"arm": "development", "mode": "fixed", "render": False,
                "retry_permitted": False, "control_limit": 17600,
                "cycles_limit": 8, "macros_limit": 32,
                "soft_s": 900, "close_s": 960, "hard_s": 1020,
                "outer_s": 1200, "normal_native_cap": 88000,
                "compiler_native_cap": 2, "side_arm": "teacher",
                "seed": 271001, "spawn_position_m": [-8.0, -4.7, 0.455],
                "output_directory": str(OUTPUT), "model_limits": MODEL_LIMITS,
                "static_API_global_caps": STATIC_CAPS}
    if any(spec.get(key) != value for key, value in required.items()):
        raise RuntimeError("C33 repair changed the fixed campaign or budgets")
    if (spec.get("execution_contract_id") != CONTRACT or
            go.get("worker") != str(HERE / "root33/worker33.py") or
            go.get("host") != str(HERE / "root33/host33.py")):
        raise RuntimeError("C33 repair changed the frozen worker/host authority")
    shared = go["shared_session"]
    if (shared.get("execution_contract_id") != CONTRACT or
            shared.get("development_cases") != spec.get("development_cases") or
            shared.get("baseline_paths33") != spec.get("baseline_paths33")):
        raise RuntimeError("C33 repair changed the shared cases or baselines")
    env = go["runtime_environment"]
    path = env.get("PYTHONPATH")
    parts = path.split(":") if isinstance(path, str) else []
    needed = {str(HERE / "root33"), str(HERE / "sol33"),
              str(HERE.parent / "continuation31/root31"),
              str(HERE.parent / "continuation31/sol31"),
              str(HERE.parent / "continuation30/root30")}
    if (not parts or parts[0] != SITE or
            not needed.issubset(parts) or env.get("LD_PRELOAD") != shared.get("library") or
            env.get("LD_BIND_NOW") != "1" or env.get("LD_LIBRARY_PATH", "missing") is not None or
            any(env.get(key) != "1" for key in
                ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"))):
        raise RuntimeError("C33 repaired import environment differs")
    if OUTPUT.exists() or OUTPUT.with_name(OUTPUT.name + "_reservation.json").exists():
        raise FileExistsError("development_02 was already reserved")
    request = {"schema": "d1-c33-single-request-v1", "arm": "development",
               "source_go_path": str(source_go), "spec": spec,
               "x11_events": None, "supersedes_attempt": "development_01"}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(request, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-go", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    request = build(args.source_go, args.output)
    print(json.dumps({"request": str(args.output), "arm": request["arm"],
                      "output_directory": request["spec"]["output_directory"]}))


if __name__ == "__main__":
    main()
