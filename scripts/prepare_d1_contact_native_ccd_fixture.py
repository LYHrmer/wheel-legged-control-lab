"""Prepare the bounded native CCD adapter inputs from frozen JSON only."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from scripts.prepare_d1_contact_kernel_fixture import digest, prepare, require

CONTRACT_SHA256 = "0869d5836add5dc12a295d5ac4618bff984244c9703516202f9485a29fdf4768"
FIXTURE_SHA256 = "91cb3c9d1137b11c39f071de01ddb9e2f9ba83ca90af8e04e3cc565800852a55"
LIB_SHA256 = "bd3f702ace8a31e1046f746880387858a981d11b01772a55ebef48ffd55ea5b8"
SOURCE_HASHES = {
    "engine_collision_convex.h": "3df5241f900dd9c01565c857191eeb740104b09f75a11af56500b73fbaefddfe",
    "engine_collision_gjk.h": "756622585c9b5d1c5f824eb8e7d69c8c8393addd74733cfddbb01321ff3a219e",
    "user_objects.cc": "e7f1f426540802d54cdd6c1230a0ba0b4d2156e993ac546caea57bbdf49b0ff1",
}
VEC3_SHA256 = "89c4b4e52ec27f24108a6ff18a65879bd6ad31475ebbaf712488220ee5369934"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--lib", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--vec3-header", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(digest(args.fixture) == FIXTURE_SHA256, "fixture_sha256_changed")
    require(digest(args.contract) == CONTRACT_SHA256, "native_ccd_contract_sha256_changed")
    require(digest(args.lib) == LIB_SHA256, "installed_library_sha256_changed")
    require(digest(args.vec3_header) == VEC3_SHA256, "libccd_vec3_header_sha256_changed")
    for filename, expected in SOURCE_HASHES.items():
        require(digest(args.sources / filename) == expected, "official_source_changed:" + filename)

    fixture = json.loads(args.fixture.read_text())
    old_header, _unused_xml, record = prepare(fixture, args.lib)
    wheel_size = fixture["geometry_manifest"]["geoms"][59]["size_m"]
    box_size = fixture["geometry_manifest"]["geoms"][1]["size_m"]
    wheel_rbound = math.sqrt(wheel_size[0] * wheel_size[0] + wheel_size[1] * wheel_size[1])
    box_rbound = math.sqrt(
        box_size[0] * box_size[0] + box_size[1] * box_size[1] + box_size[2] * box_size[2]
    )
    require(old_header.endswith("#endif\n"), "upstream_header_shape")
    header = old_header.removesuffix("#endif\n") + "\n".join([
        f'#define KERNEL_CONTRACT_02_SHA256 "{CONTRACT_SHA256}"',
        f"static const double kWheelRbound = {wheel_rbound.hex()};",
        f"static const double kBoxRbound = {box_rbound.hex()};",
        "#endif",
        "",
    ])
    record.update({
        "schema": "d1-native-ccd-primitive-input-v1",
        "entry": "mjc_initCCDObj + mjc_ccdSize + mjc_ccd, copied local perturbation wrapper",
        "model_or_data_allocated": False,
        "primitive_carrier_arrays": ["geom_type", "geom_size", "geom_xpos", "geom_xmat"],
        "rbound": {"wheel_m": wheel_rbound, "box_m": box_rbound,
                   "source": "user_objects.cc primitive rbound; ordered multiply/add/sqrt"},
        "ccd_config": {"max_iterations": 35, "tolerance": 1e-6, "max_contacts": 1,
                       "dist_cutoff": 0, "npolygonmax": 0, "nmeshdegmax": 0},
        "source_sha256": SOURCE_HASHES,
        "libccd_vec3_header_sha256": VEC3_SHA256,
        "contract_sha256": CONTRACT_SHA256,
    })
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "engine").mkdir()
    for filename in ("engine_collision_convex.h", "engine_collision_gjk.h"):
        (args.output / "engine" / filename).write_bytes((args.sources / filename).read_bytes())
    (args.output / "d1_contact_native_ccd_input.h").write_text(header)
    (args.output / "d1_contact_native_ccd_input.json").write_text(
        json.dumps(record, indent=2, allow_nan=False) + "\n"
    )
    files = {filename: digest(args.output / filename) for filename in (
        "d1_contact_native_ccd_input.h", "d1_contact_native_ccd_input.json",
        "engine/engine_collision_convex.h", "engine/engine_collision_gjk.h",
    )}
    (args.output / "preparation_receipt.json").write_text(json.dumps({
        "schema": "d1-native-ccd-preparation-v1",
        "files_sha256": files,
        "fixture_sha256": FIXTURE_SHA256,
        "contract_sha256": CONTRACT_SHA256,
        "installed_library_sha256": LIB_SHA256,
        "official_header_sha256": SOURCE_HASHES,
        "libccd_vec3_header_sha256": VEC3_SHA256,
        "mujoco_calls": 0,
        "qualification_granted": False,
        "original_score_overridden": False,
        "rl_gate_open": False,
    }, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
