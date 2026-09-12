import csv,json,math
from pathlib import Path
from scripts.d1_course_controls import CourseKeyboardCommands,wrap_angle
from wheel_legged_control.d1.interactive import D1InteractiveSimulation
out=Path('/home/lyh/wheel-legged-control-lab-work/recovery-20260912/heading_damping_probe');out.mkdir(exist_ok=False)
reports=[]
for kp,kd in [(1.5,.8),(2.5,1.),(1.,.6),(1.,.2)]:
 arena,zone,key='flat','start','Q'
 sim=D1InteractiveSimulation(arena=arena);sim.reset(zone);clock=[0.]
 c=CourseKeyboardCommands(clock=lambda:clock[0]);c.velocity_kp=.6;c.velocity_ki=.4;c.reset(0,sim.teleop._state.base_position[:2]);rows=[]
 for step in range(1600):
  st=sim.teleop._state;clock[0]=step*.01
  c.set_state(yaw_rad=float(st.base_rpy[2]),position_xy=st.base_position[:2],forward_velocity_mps=float(st.base_linear_velocity_body[0]),fallen=bool(st.has_fallen()),simulation_time_s=clock[0])
  c.update_pressed({ord(key)} if 100<=step<200 else set())
  cmd=c(clock[0]);sim.teleop.forward_velocity_mps=cmd.forward_velocity_mps;sim.teleop.yaw_rate_rps=max(-1.,min(1.,kp*wrap_angle(c.goal_yaw_rad-float(st.base_rpy[2]))-kd*float(st.base_angular_velocity_body[2])))
  status=sim.step();st=sim.teleop._state
  rows.append(dict(step=step,goal=c.goal_yaw_rad,yaw=float(st.base_rpy[2]),vx=float(st.base_linear_velocity_body[0]),x=float(st.base_position[0]),y=float(st.base_position[1]),request_vx=cmd.forward_velocity_mps,request_yaw=cmd.yaw_rate_rps,safety=status.safety_mode))
 name=f'{arena}_{zone}_{key}_{kp}_{kd}'
 with (out/(name+'.csv')).open('x') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 report=dict(name=name,target_rad=rows[-1]['goal'],actual_rad=rows[-1]['yaw'],error_rad=abs(wrap_angle(rows[-1]['goal']-rows[-1]['yaw'])),xy_change=[rows[-1]['x']-rows[99]['x'],rows[-1]['y']-rows[99]['y']],fallen_steps=sum(r['safety']=='recovery' for r in rows),final_speed=rows[-1]['vx'])
 reports.append(report);print(json.dumps(report),flush=True)
with (out/'summary.json').open('x') as f:json.dump(reports,f,indent=2)

