import csv,json,math,hashlib
from pathlib import Path
import numpy as np
from wheel_legged_control.d1.interactive import D1InteractiveSimulation
out=Path('/home/lyh/wheel-legged-control-lab-work/recovery-20260912/yaw_response_probe_high')
out.mkdir(exist_ok=False)
reports=[]
for arena,zone,target in [('flat','start',0.0),('flat','start',1.2),('flat','start',-1.2),('course','rough',1.2),('course','rough',-1.2)]:
 sim=D1InteractiveSimulation(arena=arena)
 sim.reset(zone)
 rows=[]
 for tick in range(900):
  command=0 if tick<100 else target
  sim.teleop.yaw_rate_rps=command
  status=sim.step()
  state=sim.teleop._state
  rows.append(dict(tick=tick,command=command,yaw=float(state.base_rpy[2]),roll=float(state.base_rpy[0]),pitch=float(state.base_rpy[1]),x=float(state.base_position[0]),y=float(state.base_position[1]),z=float(state.base_position[2]),safety=status.safety_mode,actual_yaw_rate=float(state.base_angular_velocity_body[2])))
 name=f'{arena}_{zone}_{target}'
 with (out/(name+'.csv')).open('x') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 yaw=np.unwrap([r['yaw'] for r in rows])
 report=dict(name=name,yaw_command_rad=target*8,yaw_change_rad=float(yaw[-1]-yaw[99]),max_roll=max(abs(r['roll']) for r in rows),max_pitch=max(abs(r['pitch']) for r in rows),fallen_steps=sum(r['safety']=='recovery' for r in rows),xy_change=[rows[-1]['x']-rows[99]['x'],rows[-1]['y']-rows[99]['y']])
 reports.append(report);print(json.dumps(report),flush=True)
with (out/'summary.json').open('x') as f:json.dump(reports,f,indent=2)
