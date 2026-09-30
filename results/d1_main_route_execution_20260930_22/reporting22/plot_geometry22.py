"""Plot saved nominal traces only; no servo integration, model or physics."""
from pathlib import Path
import hashlib
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

C = Path(__file__).resolve().parent.parent
old = C.parent/'continuation21/finite_geometry_preflight_21/all_nominal_traces21.npz'
new = C/'finite_geometry_preflight_22/four_revised_nominal_traces22.npz'
report = C/'finite_geometry_22.json'
out = C/'figures22'
out.mkdir(exist_ok=False)
rows = {r['case_id']: r for r in json.loads(report.read_text())['new_rows']}
plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
fig, axes = plt.subplots(2, 2, figsize=(11, 6.4), sharex=True, sharey=True)
with np.load(old, allow_pickle=False) as a, np.load(new, allow_pickle=False) as b:
    for ax, case in zip(axes.flat, ('dev_yaw_left','dev_yaw_right','final_yaw_left','final_yaw_right')):
        x = a['eval_'+case+'_path_xy_m']
        y = b[case+'_path_xy_m']
        radius = rows[case]['reset_robot_radial_bound_m']
        ax.axhspan(-6.6, -6, color='#fbe9e7')
        ax.axhline(-6, color='#b71c1c', lw=1.3, label='Map safety edge')
        for edge in (-5.8, -3.6):
            ax.axhline(edge, color='#787878', lw=.9, ls=':', label='Flat lane center limits' if edge == -3.6 else None)
        ax.fill_between(x[:,0], x[:,1]-radius, x[:,1]+radius, color='#cc7130', alpha=.10)
        ax.fill_between(y[:,0], y[:,1]-radius, y[:,1]+radius, color='#1565a0', alpha=.12)
        ax.plot(x[:,0], x[:,1], color='#cc7130', lw=1.8, label='C21 rejected schedule')
        ax.plot(y[:,0], y[:,1], color='#1565a0', lw=1.8, label='C22 qualified schedule')
        ax.scatter([x[-1,0], y[-1,0]], [x[-1,1], y[-1,1]], c=['#cc7130','#1565a0'], s=18, zorder=5)
        ax.set_title(case.replace('_',' '), loc='left', fontweight='bold')
        ax.set_xlim(-8.2, .3)
        ax.set_ylim(-6.5, -3)
        ax.grid(alpha=.15)
        ax.text(.03,.05, f'C22 map Y margin: {rows[case]["y_safety_margin_m"]:.3f} m', transform=ax.transAxes, fontsize=9)
for ax in axes[-1]: ax.set_xlabel('World X (m)')
for ax in axes[:,0]: ax.set_ylabel('World Y (m)')
handles, labels = axes[0,0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(.5,.045))
fig.suptitle('Finite geometry qualification before robot execution', x=.09, ha='left', fontweight='bold')
fig.text(.09,.018,'Saved servo-integrated nominal paths; shading is the reset radius in Y, not an observed robot trajectory.',fontsize=9)
fig.tight_layout(rect=(0,.13,1,.94))
for suffix in ('png','svg'):
    fig.savefig(out/('nominal_geometry22.'+suffix), dpi=180)
plt.close(fig)
def identity(p):
    data=p.read_bytes()
    return {'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
receipt={'schema':'d1-c22-saved-nominal-plot-v1','inputs':{str(p):identity(p) for p in (old,new,report,Path(__file__))},
         'outputs':{p.name:identity(p) for p in out.iterdir()},'model_calls':0,'physics_calls':0,'servo_calls':0}
with (out/'plot_receipt22.json').open('x') as f: json.dump(receipt,f,indent=2,sort_keys=True)
print(json.dumps({'outputs':list(receipt['outputs']), 'model_calls':0,'physics_calls':0,'servo_calls':0}))
