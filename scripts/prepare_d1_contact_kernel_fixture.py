"""Prepare exact archived cylinder/box kernel inputs without loading MuJoCo."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

FIXTURE_SHA256 = "91cb3c9d1137b11c39f071de01ddb9e2f9ba83ca90af8e04e3cc565800852a55"
CONTRACT_SHA256 = "0a442f05ba09693d9e97bdc171cee05fef9e1c835225a6bd414209c4c9e374b9"
LIB_SHA256 = "bd3f702ace8a31e1046f746880387858a981d11b01772a55ebef48ffd55ea5b8"
OLD_WHEEL_GEOM = 59
OLD_BOX_GEOM = 1
POSE_INDICES = (3485, 3486, 3487)


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def require(condition: bool, code: str) -> None:
    if not condition:
        raise ValueError(code)


def c_number(value: float) -> str:
    require(type(value) in (int, float) and math.isfinite(value), "nonfinite_or_nonnumeric_input")
    return float(value).hex()


def c_array(values) -> str:
    return "{" + ", ".join(c_number(value) for value in values) + "}"


def flatten(matrix: list[list[float]]) -> list[float]:
    require(len(matrix) == 3 and all(len(row) == 3 for row in matrix), "rotation_shape")
    return [value for row in matrix for value in row]


def prepare(fixture: dict, lib_path: Path) -> tuple[str, str, dict]:
    manifest = fixture["geometry_manifest"]
    geoms = {geom["geom_id"]: geom for geom in manifest["geoms"]}
    wheel, box = geoms[OLD_WHEEL_GEOM], geoms[OLD_BOX_GEOM]
    require(wheel["geom_type"] == 5 and box["geom_type"] == 6, "compiled_geom_types")
    require(wheel["size_m"][:2] == [0.087, 0.02], "wheel_size_changed")
    require(box["size_m"] == [0.18, 0.62, 0.0075], "box_size_changed")
    require(box["position_local_m"] == [-3.0999999999999996, 0.0, 0.0075], "box_center_changed")
    require(wheel["margin_m"] == 0.001 and box["margin_m"] == 0.0, "geom_margins_changed")
    require(fixture["expected_bad_native_index"] == 3486
            and fixture["expected_bad_contact_index"] == 5, "bad_candidate_identity_changed")
    native = {row["index"]: row for row in fixture["native_samples"]}
    reconstructed = {(row["native_index"], row["pose"]): row
                     for row in fixture["static_forensic"]["reconstructions"]}
    require(set(native) == set(POSE_INDICES), "native_sample_set")

    poses = []
    for native_index in POSE_INDICES:
        row = reconstructed[(native_index, "before")]
        source_pairs = [contact for contact in native[native_index]["contacts"]["contacts"]
                        if (contact["geom1"], contact["geom2"]) == (OLD_WHEEL_GEOM, OLD_BOX_GEOM)]
        static_pairs = [contact for contact in row["contacts"]
                        if (contact["geom1"], contact["geom2"]) == (OLD_WHEEL_GEOM, OLD_BOX_GEOM)]
        require(len(source_pairs) == len(static_pairs) == 3, "archived_pair_count")
        for saved, rebuilt in zip(source_pairs, static_pairs, strict=True):
            for field in ("position_world_m", "distance_m", "frame_geom1_to_geom2"):
                require(saved[field] == rebuilt[field], "static_native_pair_disagreement:"+field)
        require(row["state_unchanged"] is True, "static_reconstruction_mutated_state")
        require(row["wheel_size_m"] == wheel["size_m"], "static_wheel_size_disagreement")
        center = row["wheel_center_world_m"]
        rotation = row["wheel_rotation_world"]
        axis = [rotation[i][2] for i in range(3)]
        require(abs(axis[2]) < 1.0, "vertical_cylinder_axis")
        radial = [(float(i == 2) - axis[2] * axis[i]) / math.sqrt(1 - axis[2] ** 2)
                  for i in range(3)]
        lowest = [center[i] - wheel["size_m"][1] * math.copysign(1.0, axis[2]) * axis[i]
                  - wheel["size_m"][0] * radial[i] for i in range(3)]
        gap = lowest[2] - box["position_local_m"][2] - box["size_m"][2]
        poses.append({
            "native_index": native_index,
            "wheel_center_world_m": row["wheel_center_world_m"],
            "wheel_rotation_world": row["wheel_rotation_world"],
            "box_center_world_m": box["position_local_m"],
            "box_rotation_world": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "analytic_cylinder_support": {
                "lowest_point_world_m": lowest,
                "vertical_gap_to_box_top_m": gap,
                "xy_inside_box": all(abs(lowest[i] - box["position_local_m"][i])
                                     <= box["size_m"][i] for i in (0, 1)),
            },
            "reference_pairs": [{
                "original_contact_index": saved["index"],
                "position_world_m": saved["position_world_m"],
                "distance_m": saved["distance_m"],
                "frame_geom1_to_geom2": saved["frame_geom1_to_geom2"],
                "original_geoms": [OLD_WHEEL_GEOM, OLD_BOX_GEOM],
            } for saved in source_pairs],
        })

    lines = [
        "/* Generated only from fixed JSON; every floating literal is binary64 hex. */",
        "#ifndef D1_CONTACT_KERNEL_INPUT_H",
        "#define D1_CONTACT_KERNEL_INPUT_H",
        "#define KERNEL_POSE_COUNT 3",
        "#define KERNEL_MAX_REFERENCE 5",
        f'#define KERNEL_EXPECTED_LIB_PATH "{lib_path.resolve()}"',
        f'#define KERNEL_EXPECTED_LIB_SHA256 "{LIB_SHA256}"',
        f'#define KERNEL_FIXTURE_SHA256 "{FIXTURE_SHA256}"',
        "typedef struct { int original_index; double pos[3]; double dist; double frame[9]; } KernelReference;",
        "typedef struct { int native_index; double wheel_pos[3]; double wheel_mat[9];",
        "  double box_pos[3]; double box_mat[9]; int nref;",
        "  KernelReference refs[KERNEL_MAX_REFERENCE]; } KernelPose;",
        f"static const double kWheelSize[3] = {c_array(wheel['size_m'])};",
        f"static const double kBoxSize[3] = {c_array(box['size_m'])};",
        f"static const double kWheelMargin = {c_number(wheel['margin_m'])};",
        f"static const double kBoxMargin = {c_number(box['margin_m'])};",
        f"static const double kPairMargin = {c_number(0.001)};",
        "static const KernelPose kPoses[KERNEL_POSE_COUNT] = {",
    ]
    for pose in poses:
        lines.extend([
            "  {",
            f"    {pose['native_index']}, {c_array(pose['wheel_center_world_m'])},",
            f"    {c_array(flatten(pose['wheel_rotation_world']))},",
            f"    {c_array(pose['box_center_world_m'])},",
            f"    {c_array(flatten(pose['box_rotation_world']))},",
            f"    {len(pose['reference_pairs'])}, {{",
        ])
        for ref in pose["reference_pairs"]:
            lines.append("      {" + f"{ref['original_contact_index']}, "
                         + c_array(ref["position_world_m"]) + ", "
                         + c_number(ref["distance_m"]) + ", "
                         + c_array(flatten(ref["frame_geom1_to_geom2"])) + "},")
        lines.extend(["    }", "  },"])
    lines.extend(["};", "#endif", ""])

    # XML numbers are shortest-roundtrip decimal; the harness overwrites world caches with the hex header.
    def xml_vec(values) -> str:
        return " ".join(repr(float(value)) for value in values)

    xml = "\n".join([
        '<mujoco model="d1_cylinder_box_kernel">',
        '  <compiler angle="radian"/>',
        '  <option ccd_iterations="35" ccd_tolerance="1e-6"/>',
        '  <worldbody>',
        (f'    <geom name="archived_wheel_59" type="cylinder" size="{xml_vec(wheel["size_m"][:2])}"'
         f' pos="{xml_vec(poses[0]["wheel_center_world_m"])}" margin="0.001"/>'),
        (f'    <geom name="archived_box_1" type="box" size="{xml_vec(box["size_m"])}"'
         f' pos="{xml_vec(box["position_local_m"])}" margin="0"/>'),
        '  </worldbody>',
        '</mujoco>',
        '',
    ])
    record = {
        "schema": "d1-contact-kernel-input-v1",
        "pose_order": list(POSE_INDICES),
        "old_geom_identity": {"wheel": [OLD_WHEEL_GEOM, wheel["identity"]],
                              "box": [OLD_BOX_GEOM, box["identity"]]},
        "new_geom_identity": {"wheel": 0, "box": 1},
        "input_source": "fixture static_forensic before; original native pair values cross-checked",
        "hexfloat_encoding": "Python float.hex, no matrix/quaternion conversion",
        "expected_library_path": str(lib_path.resolve()),
        "expected_library_sha256": LIB_SHA256,
        "expected_ccd": {"nativeccd_enabled": True, "multiccd_enabled": True,
                         "ccd_iterations": 35, "ccd_tolerance": 1e-6},
        "pair_generator_margin_m": 0.001,
        "gap_m": 0.0,
        "poses": poses,
        "qualification_granted": False,
        "original_score_overridden": False,
        "rl_gate_open": False,
    }
    return "\n".join(lines), xml, record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--lib", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(digest(args.fixture) == FIXTURE_SHA256, "fixture_sha256_changed")
    require(digest(args.contract) == CONTRACT_SHA256, "kernel_contract_sha256_changed")
    require(digest(args.lib) == LIB_SHA256, "installed_mujoco_library_sha256_changed")
    header, xml, record = prepare(json.loads(args.fixture.read_text()), args.lib)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "d1_contact_kernel_input.h").write_text(header)
    (args.output / "d1_contact_kernel.xml").write_text(xml)
    (args.output / "d1_contact_kernel_input.json").write_text(
        json.dumps(record, indent=2, allow_nan=False) + "\n"
    )
    files = {name: digest(args.output / name) for name in (
        "d1_contact_kernel_input.h", "d1_contact_kernel.xml", "d1_contact_kernel_input.json")}
    receipt = {"schema": "d1-contact-kernel-preparation-v1", "files_sha256": files,
               "fixture_sha256": FIXTURE_SHA256, "contract_sha256": CONTRACT_SHA256,
               "installed_library_sha256": LIB_SHA256, "new_mujoco_calls": 0,
               "new_pair_attempts": 0, "qualification_granted": False, "rl_gate_open": False}
    (args.output / "preparation_receipt.json").write_text(
        json.dumps(receipt, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
