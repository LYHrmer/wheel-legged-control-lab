"""Supplemental read-only contact-normal/wrench audit; never imports MuJoCo."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    sources = {str(Path(__file__).resolve()): sha(Path(__file__))}
    report = {"schema": "d1-turn-contact-geometry-sensitivity-v1", "new_physics_steps": 0,
              "window": "endpoints201..250 for force statistics; geometrically valid subset explicitly listed for yaw fits",
              "caution": "tilted normals are observed hfield contacts, not proof of a model defect; endpoint contact forces are not continuous impulse or dissipated energy",
              "cases": {}}
    for side, sign in (("left", 1), ("right", -1)):
        for condition in ("limit0p6", "limit1p0"):
            folder = args.input / f"stationary_turn_{side}_hold" / condition
            p = folder / "turn_diagnostics.jsonl.gz"
            sources[str(p)] = sha(p)
            sources[str(folder / "states.npz")] = sha(folder / "states.npz")
            with gzip.open(p, "rt") as stream:
                rows = [json.loads(line) for line in stream]
            with np.load(folder / "states.npz") as z:
                qpos = z["qpos"].copy()
            force_stats, normal_stats, moment_rows, weighted_fits = [], [], [], []
            excluded, examples = [], []
            for k in range(200, 250):
                info = rows[k]["contacts"]
                contacts = info["contacts"]
                w, x, y, z = qpos[k+1, 3:7]
                yaw = np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))
                rotation = np.array([[np.cos(yaw), np.sin(yaw), 0.],
                                     [-np.sin(yaw), np.cos(yaw), 0.], [0., 0., 1.]])
                r = (np.asarray([c["pos_world_m"] for c in contacts])-info["reference_world_m"]) @ rotation.T
                n = np.asarray([c["normal_world"] for c in contacts])
                f = np.asarray([c["force_world_n"] for c in contacts])
                f_normal = np.sum(f*n, axis=1)[:, None]*n
                f_tangent = f-f_normal
                fn = np.asarray([c["normal_force_n"] for c in contacts])
                normal_stats.extend(zip(np.linalg.norm(n[:, :2], axis=1).tolist(), fn.tolist()))
                fm = {}
                for name, world in (("normal", f_normal), ("tangent", f_tangent), ("total", f)):
                    body = world @ rotation.T
                    fm[name+"_longitudinal"] = float(np.sum(-r[:, 1]*body[:, 0]))
                    fm[name+"_lateral"] = float(np.sum(r[:, 0]*body[:, 1]))
                    fm[name+"_net"] = fm[name+"_longitudinal"]+fm[name+"_lateral"]
                assert abs(fm["total_net"]-info["total_wrench_world_6"][5]) < 1e-10
                moment_rows.append(fm)
                force_stats.append(float(fn.sum()))
                for c in contacts:
                    if np.linalg.norm(c["normal_world"][:2])>.9 and c["normal_force_n"]>50 and len(examples)<4:
                        examples.append({"endpoint_tick":k+1, **{key:c[key] for key in
                            ("wheel_index", "robot_geom_id", "terrain_geom_id", "pos_world_m", "normal_world", "normal_force_n", "force_world_n")}})
                if fn.sum() <= 1e-6:
                    excluded.append(k+1)
                    continue
                weights = fn/fn.sum()
                cy = r[:, 1]-weights @ r[:, 1]
                den = float(weights @ cy**2)
                active = {c["wheel_index"] for c in contacts if c["normal_force_n"]>1e-6}
                if den <= 1e-4 or len(active)<2:
                    excluded.append(k+1)
                    continue
                omegas = {}
                for name,key in (("base", "free_base_velocity_world_mps"),
                                 ("leg", "leg_dof_velocity_world_mps"),
                                 ("wheel", "wheel_dof_velocity_world_mps"),
                                 ("slip", "relative_velocity_world_mps")):
                    v = np.asarray([c[key] for c in contacts])
                    v = (v-np.sum(v*n, axis=1)[:,None]*n) @ rotation.T
                    omegas[name] = float(-(weights*cy) @ v[:,0]/den)
                assert abs(omegas["base"]+omegas["leg"]+omegas["wheel"]-omegas["slip"])<1e-12
                weighted_fits.append(omegas)
            means = {key:float(np.mean([r[key] for r in weighted_fits])) for key in weighted_fits[0]}
            gap = -means["wheel"]-means["base"]
            ns = np.asarray(normal_stats)
            case = {"normal_force_weighted_fit_valid_endpoints":len(weighted_fits),
                "normal_force_weighted_fit_excluded_endpoints":excluded,
                "normal_force_weighted_signed_gap_rps":sign*gap,
                "normal_force_weighted_leg_gap_fraction":means["leg"]/gap,
                "normal_force_weighted_slip_gap_fraction":-means["slip"]/gap,
                "contact_count":len(ns),
                "normal_horizontal_gt0p1_contact_fraction":float(np.mean(ns[:,0]>.1)),
                "normal_horizontal_gt0p1_fraction_of_normal_load_sum":float(ns[ns[:,0]>.1,1].sum()/ns[:,1].sum()),
                "normal_horizontal_gt0p9_fraction_of_normal_load_sum":float(ns[ns[:,0]>.9,1].sum()/ns[:,1].sum()),
                "endpoint_normal_force_sum_mean_n":float(np.mean(force_stats)),
                "full_50_endpoint_signed_yaw_moment_means_nm":{key:sign*float(np.mean([r[key] for r in moment_rows])) for key in moment_rows[0]},
                "high_load_tilted_normal_examples":examples}
            report["cases"][f"{side}_{condition}"] = case
            print(json.dumps({k:v for k,v in case.items() if k != "high_load_tilted_normal_examples"}, allow_nan=False))
    assert all(sha(Path(path))==value for path,value in sources.items())
    report["source_sha256"] = sources
    report["inputs_unchanged"] = True
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
