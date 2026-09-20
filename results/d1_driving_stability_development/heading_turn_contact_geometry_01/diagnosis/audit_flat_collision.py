"""Offline collision-only snapshots: frozen flat hfield versus z=0 plane.

Build two models and reuse saved qpos/qvel; call mj_fwdPosition only. Never
construct a plant, call mj_step, integrate time, tune a controller, or simulate
the counterfactual plane trajectory.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from wheel_legged_control.d1.locomotion_terrain import D1LocomotionTerrainConfig
from wheel_legged_control.d1.model import build_d1_model


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collision(model, qpos, qvel, time):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    data.time = time
    original = data.qpos.copy(), data.qvel.copy(), float(data.time)
    mujoco.mj_fwdPosition(model, data)
    assert np.array_equal(data.qpos, original[0])
    assert np.array_equal(data.qvel, original[1]) and data.time == original[2]
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    contacts = []
    for index,c in enumerate(data.contact):
        if floor not in (int(c.geom1), int(c.geom2)):
            continue
        robot_geom = int(c.geom2) if int(c.geom1)==floor else int(c.geom1)
        body = int(model.geom_bodyid[robot_geom])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body)
        if not name.endswith("_foot"):
            continue
        contacts.append({"index":index, "robot_body":name, "robot_geom":robot_geom,
            "pos":np.asarray(c.pos).tolist(), "normal":np.asarray(c.frame).reshape(3,3)[0].tolist(),
            "dist":float(c.dist), "efc_address":int(c.efc_address)})
    return {"contact_count":len(contacts),
            "active_contact_count":sum(c["efc_address"]>=0 for c in contacts),
            "max_horizontal_normal":max((float(np.linalg.norm(c["normal"][:2])) for c in contacts),default=0),
            "contacts":contacts}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    hfield=build_d1_model(locomotion_terrain=D1LocomotionTerrainConfig(layout="flat"))
    plane=build_d1_model()
    assert (hfield.nq,hfield.nv,hfield.nbody,hfield.ngeom)==(plane.nq,plane.nv,plane.nbody,plane.ngeom)
    np.testing.assert_array_equal(hfield.body_mass,plane.body_mass)
    np.testing.assert_array_equal(hfield.geom_friction,plane.geom_friction)
    floor=mujoco.mj_name2id(hfield,mujoco.mjtObj.mjOBJ_GEOM,"floor")
    hid=int(hfield.geom_dataid[floor])
    assert hfield.geom_type[floor]==mujoco.mjtGeom.mjGEOM_HFIELD
    assert plane.geom_type[floor]==mujoco.mjtGeom.mjGEOM_PLANE
    height=np.asarray(hfield.hfield_data)
    assert np.all(height==0)
    assert np.array_equal(hfield.geom_pos[floor],plane.geom_pos[floor])
    half_x,half_y=hfield.hfield_size[hid,:2]
    dx=2*half_x/(int(hfield.hfield_ncol[hid])-1)
    dy=2*half_y/(int(hfield.hfield_nrow[hid])-1)
    sources={str(Path(__file__).resolve()):sha(Path(__file__))}
    result={"schema":"d1-flat-hfield-plane-offline-collision-v1", "new_physics_steps":0,
        "plant_instantiated":False,"position_collision_calls":0,"mujoco_version":mujoco.__version__,
        "call":"mj_fwdPosition only on newly allocated snapshot MjData; no mj_step or time integration",
        "floor_position":hfield.geom_pos[floor].tolist(),"hfield_size":hfield.hfield_size[hid].tolist(),
        "hfield_rows":int(hfield.hfield_nrow[hid]),"hfield_cols":int(hfield.hfield_ncol[hid]),
        "hfield_data_all_zero":True,"grid_dx_m":float(dx),"grid_dy_m":float(dy),"snapshots":[],
        "limit":"Plane is tested only for contacts at already-observed hfield poses; no plane trajectory or control outcome is inferred."}
    for side,condition,endpoints in (("left","limit0p6",(200,211,230,245,250)),
                                     ("left","limit1p0",(211,250)),
                                     ("right","limit0p6",(211,)),
                                     ("right","limit1p0",(211,))):
        folder=args.input/f"stationary_turn_{side}_hold"/condition
        for name in ("states.npz","turn_diagnostics.jsonl.gz"):
            sources[str(folder/name)]=sha(folder/name)
        with np.load(folder/"states.npz") as z: qpos=z["qpos"].copy();qvel=z["qvel"].copy()
        with gzip.open(folder/"turn_diagnostics.jsonl.gz","rt") as f: rows=[json.loads(line) for line in f]
        for endpoint in endpoints:
            a=collision(hfield,qpos[endpoint],qvel[endpoint],endpoint*.01)
            b=collision(plane,qpos[endpoint],qvel[endpoint],endpoint*.01)
            saved=rows[endpoint-1]["contacts"]["contacts"]
            active=[c for c in a["contacts"] if c["efc_address"]>=0]
            assert len(saved)==len(active)
            for old,new in zip(saved,active):
                assert old["robot_geom_id"]==new["robot_geom"]
                np.testing.assert_allclose(old["pos_world_m"],new["pos"],rtol=0,atol=1e-12)
                np.testing.assert_allclose(old["normal_world"],new["normal"],rtol=0,atol=1e-12)
            for c in a["contacts"]:
                xp,yp=c["pos"][:2]
                c["nearest_hfield_x_gridline_distance_m"]=float(abs((xp+half_x)/dx-round((xp+half_x)/dx))*dx)
                c["nearest_hfield_y_gridline_distance_m"]=float(abs((yp+half_y)/dy-round((yp+half_y)/dy))*dy)
            item={"side":side,"condition":condition,"endpoint":endpoint,
                  "hfield_reproduces_saved_contact_geometry":True,"hfield":a,"plane":b}
            result["snapshots"].append(item)
            result["position_collision_calls"]+=2
            print(json.dumps({"case":side+"_"+condition,"endpoint":endpoint,
                "hfield_count":a["contact_count"],"hfield_max_horizontal_normal":a["max_horizontal_normal"],
                "plane_count":b["contact_count"],"plane_max_horizontal_normal":b["max_horizontal_normal"]}))
    assert all(sha(Path(p))==v for p,v in sources.items())
    result["input_sha256"]=sources
    result["inputs_unchanged"]=True
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")


if __name__=="__main__": main()
