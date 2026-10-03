"""Create the single C33 host request from a reviewed source GO; no execution."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def build(source_go: Path, output: Path):
    if output.exists():
        raise FileExistsError(output)
    source_go = source_go.resolve(strict=True)
    go = json.loads(source_go.read_text())
    if (go.get("decision") != "GO" or go.get("execution_contract_id") !=
            "C33_fixed_S_curve_feasibility_v1" or set(go.get("arms", {})) != {"development"}):
        raise RuntimeError("C33 requires reviewed single-arm source GO")
    spec = go["arms"]["development"]
    if (spec.get("output_directory") != str(Path(__file__).resolve().parent/"development_01")
            or spec.get("control_limit") != 17600
            or spec.get("cycles_limit") != 8 or spec.get("macros_limit") != 32
            or spec.get("outer_s") != 1200 or spec.get("render") is not False):
        raise RuntimeError("C33 finite request differs from frozen campaign")
    request = {"schema": "d1-c33-single-request-v1", "arm": "development",
               "source_go_path": str(source_go), "spec": spec, "x11_events": None}
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
