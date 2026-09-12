import json,math
from pathlib import Path
import numpy as np
from wheel_legged_control.d1.interactive import D1InteractiveSimulation
out=Path('/home/lyh/wheel-legged-control-lab-work/recovery-20260912/velocity_hold_probe');out.mkdir(exist_ok=False)
reports=[]
for gain,igain in [(0,0),(1.2,.8),(2,1.5)]:
 for yaw in [0.,1.2]:
  sim=D1InteractiveSimulation(arena='flat');sim.reset('start');integral=0.;rows=[]
  for tick in range(1100):
   state=sim.teleop._state
   error=-float(state.base_linear_velocity_body[0])
   if tick>=100:integral=float(np.clip(integral+error*.01,-.25,.25))
   cmd=0 if tick<100 else float(np.clip(gain*error+igain*integral,-.5,.5))
   sim.teleop.forward_velocity_mps=cmd;sim.teleop.yaw_rate_rps=0 if tick<100 else yaw
   status=sim.step();state=sim.teleop._state
   rows.append([float(state.base_position[0]),float(state.base_position[1]),float(state.base_rpy[2]),float(state.base_linear_velocity_body[0]),float(state.base_rpy[0]),float(state.base_rpy[1]),status.safety_mode,cmd])
  yaws=np.unwrap([r[2] for r in rows]);report=dict(gain=gain,igain=igain,yaw=yaw,yaw_change_rad=float(yaws[-1]-yaws[99]),xy_change=(np.array(rows[-1][:2])-rows[99][:2]).tolist(),mean_last_v=float(np.mean([r[3] for r in rows[-200:]])),max_roll=max(abs(r[4]) for r in rows),max_pitch=max(abs(r[5]) for r in rows),fallen_steps=sum(r[6]=='recovery' for r in rows),last_command=rows[-1][7])
  reports.append(report);print(json.dumps(report),flush=True)
with (out/'summary.json').open('x') as f:json.dump(reports,f,indent=2)
