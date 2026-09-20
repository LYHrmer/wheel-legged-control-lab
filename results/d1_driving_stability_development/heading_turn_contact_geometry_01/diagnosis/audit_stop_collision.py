"""Same-pose stopping collision geometry only; no dynamics or force claims."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from audit_flat_collision import collision, sha
from wheel_legged_control.d1.locomotion_terrain import D1LocomotionTerrainConfig
from wheel_legged_control.d1.model import build_d1_model


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    hfield=build_d1_model(locomotion_terrain=D1LocomotionTerrainConfig(layout="flat"))
    plane=build_d1_model()
    floor=mujoco.mj_name2id(hfield,mujoco.mjtObj.mjOBJ_GEOM,"floor")
    assert np.all(hfield.hfield_data==0)
    np.testing.assert_array_equal(hfield.geom_pos[floor],plane.geom_pos[floor])
    np.testing.assert_array_equal(hfield.geom_friction,plane.geom_friction)
    sources={str(Path(__file__).resolve()):sha(Path(__file__)),
             str(Path(__file__).with_name("audit_flat_collision.py")):sha(Path(__file__).with_name("audit_flat_collision.py"))}
    report={"schema":"d1-stop-saved-pose-collision-v1","new_physics_steps":0,
        "position_collision_calls":0,"plant_instantiated":False,"mujoco_version":mujoco.__version__,
        "method":"mj_fwdPosition only, saved poses; no mj_step, no time integration, no control modification",
        "limit":"Original stop traces have no contact-force samples. This report establishes geometry, not per-contact load or a plane stopping trajectory.",
        "cases":{}}
    for name in ("flat_forward_stop","flat_reverse_stop"):
        for condition in ("bypass","release_0p5"):
            p=args.input/name/condition/"states.npz"
            sources[str(p)]=sha(p)
            with np.load(p) as z:qpos=z["qpos"].copy();qvel=z["qvel"].copy()
            snapshots=[]
            for tick in (400,401,425,449,475,500,550):
                a=collision(hfield,qpos[tick],qvel[tick],tick*.01)
                b=collision(plane,qpos[tick],qvel[tick],tick*.01)
                snapshots.append({"state_tick":tick,"hfield":a,"plane":b})
                report["position_collision_calls"]+=2
            report["cases"][name+"_"+condition]=snapshots
            print(json.dumps({"case":name,"condition":condition,
                "hfield_horizontal_normal_max_by_tick":{r["state_tick"]:r["hfield"]["max_horizontal_normal"] for r in snapshots},
                "plane_horizontal_normal_max":max(r["plane"]["max_horizontal_normal"] for r in snapshots)}))
    assert all(sha(Path(p))==v for p,v in sources.items())
    report["input_sha256"]=sources
    report["inputs_unchanged"]=True
    args.output.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")


if __name__=="__main__":main()
