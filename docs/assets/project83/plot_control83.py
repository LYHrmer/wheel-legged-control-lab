"""Saved-data visualization only; all three registered development seeds."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root = Path('/home/lyh/wheel-legged-control-lab')
new = root / 'results/offset_mpc83_20261104_01'
old = root / 'results/mpc49_preflight_20261022_01'
report = json.loads((new / 'report.json').read_text())
assert report['status'] == report['verification_after'] == 'COMPLETE'
assert len(report['episodes']) == 12 and not report['decision']['accepted']
definitions = [('Original49',old,'baseline','#777777'),
               ('Delay82',new,'baseline','#2274a5'),
               ('Load observer83',new,'candidate','#cb5a27')]
fig, axes = plt.subplots(2,2,figsize=(10.5,6.3),sharex=True,layout='constrained')
labels = ['Height tracking error (mm)','Forward tracking error (m/s)',
          'Pitch (degrees)','Applied leg force (N)']
for title, folder, arm, color in definitions:
    traces=[]
    for seed in (490101,490102,490103):
        with (folder / f'{seed}_mismatch_delay_benchmark_{arm}/trajectory.csv').open() as f:
            rows=[{k:float(v) for k,v in row.items()} for row in csv.DictReader(f)]
        assert len(rows)==400
        traces.append([[1000*(r['extension']-r['target_extension']),r['vx']-r['target_vx'],
                        np.degrees(r['pitch']),r['leg_force']] for r in rows])
    values=np.asarray(traces)
    times=np.asarray([r['time_s'] for r in rows])
    for i, ax in enumerate(axes.flat):
        ax.fill_between(times,values[:,:,i].min(0),values[:,:,i].max(0),color=color,alpha=.13,linewidth=0)
        ax.plot(times,values[:,:,i].mean(0),color=color,label=title,linewidth=1.6)
for ax, label in zip(axes.flat, labels, strict=True):
    ax.set_ylabel(label)
    ax.grid(alpha=.17)
    ax.spines[['top','right']].set_visible(False)
for ax in axes[-1]:
    ax.set_xlabel('Simulation time (s)')
axes[0,0].legend(loc='lower right',fontsize=8)
fig.suptitle('Height improves; original speed, pitch and effort gates still fail\n'
             'All 3 paired development seeds; line = mean, band = range (not confidence interval)',fontsize=11)
target=root/'docs/assets/project83/mpc83_tradeoff.png'
target.parent.mkdir(parents=True,exist_ok=True)
if target.exists():
    raise FileExistsError(target)
fig.savefig(target,dpi=160)
plt.close(fig)
print(json.dumps({'status':'PASS','output':str(target),'bytes':target.stat().st_size,
                  'new_physics_steps':0,'new_model_calls':0,'parameter_updates':0}))
