"""Plot already validated C24 archives; no model or physics execution."""
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
report = json.loads((HERE/'independent_read_01.json').read_text())
assert report['B22_true_15mm_qualified'] and report['zero_true_15mm_qualified']
fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
for actor, color in [('zero', '#52616b'), ('B', '#0068b4')]:
    folder = HERE/'step_pair_01/heldout'/('true_15mm_0p2_'+actor)
    rows = []
    for path in sorted(folder.glob('control_records_*.jsonl.gz')):
        with gzip.open(path, 'rt') as stream:
            rows.extend(json.loads(line) for line in stream)
    with gzip.open(folder/'endpoint_geometry24.jsonl.gz', 'rt') as stream:
        geometry = [json.loads(line) for line in stream]
    assert len(rows) == 1200 and len(geometry) == 1201
    t = np.arange(1, 1201)*.01
    metrics = [row['info']['metrics'] for row in rows]
    label = 'B22 + C18' if actor == 'B' else 'C18, zero residual'
    axes[0, 0].plot(t, [m['body_com_vx_mps'] for m in metrics], color=color, label=label)
    q = np.asarray([row['qpos'][3:7] for row in geometry])
    yaw = np.unwrap(np.arctan2(2*(q[:, 0]*q[:, 3]+q[:, 1]*q[:, 2]),
                              1-2*(q[:, 2]**2+q[:, 3]**2)))
    axes[0, 1].plot(np.arange(1201)*.01, np.rad2deg(yaw-yaw[0]), color=color)
    margin = [min(shape['minimum_world_m'][0]+2.92-shape['margin_m']
                  for shape in row['collision_bounds']) for row in geometry]
    axes[1, 0].plot(np.arange(1201)*.01, margin, color=color)
    axes[1, 1].plot(t[-100:], [abs(m['body_com_vx_mps']) for m in metrics[-100:]], color=color)
axes[0, 0].plot(np.arange(1200)*.01, [.2 if 200 <= t < 1000 else 0 for t in range(1200)],
                '--', color='#bd6c2c', label='Raw forward command', linewidth=1.2)
axes[0, 0].set(ylabel='Body forward speed (m/s)', title='Fixed 0.20 m/s roll-over')
axes[0, 0].legend(fontsize=9)
axes[0, 1].set(ylabel='Heading change (deg)', title='Heading remains within the 5 deg gate')
axes[0, 1].axhline(5, color='#bd6c2c', linestyle='--', linewidth=1)
axes[1, 0].set(ylabel='Whole-robot far-edge margin (m)', title='All collision shapes clear the real box')
axes[1, 0].axhline(0, color='#bd6c2c', linestyle='--', linewidth=1)
axes[1, 1].set(ylabel='Absolute body forward speed (m/s)', title='Final 100 endpoints: stop qualification')
axes[1, 1].axhline(.03, color='#bd6c2c', linestyle='--', linewidth=1)
for ax in axes.flat:
    ax.set_xlabel('Simulation time (s)')
    ax.grid(alpha=.22)
fig.suptitle('Real native 15 mm single box: both actors pass the fixed scenario\n'
             'Saved C24 evidence only; one matched pair, no RL-superiority claim', fontsize=13)
fig.savefig(HERE/'true15mm_evidence.png', dpi=180)
fig.savefig(HERE/'true15mm_evidence.svg')
plt.close(fig)
