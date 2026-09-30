"""Plot saved nominal paths only; never load a policy or physics engine."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

C = Path(__file__).resolve().parent.parent
report = json.loads((C / 'offline_corridor_20_02.json').read_text())
original = report['B18_failure']['actual_reset_flat_guard']
candidate = report['single_preselected_candidate']
with np.load(original['nominal_path_file'], allow_pickle=False) as saved:
    old = saved['xy_m']
new = np.asarray(candidate['nominal_path_xy_m'])
limit = -6 + candidate['radius_m']
folder = C / 'figures20'
folder.mkdir(exist_ok=False)
fig, ax = plt.subplots(figsize=(9, 3.8), constrained_layout=True)
ax.axhspan(-6.1, limit, color='#fde6e1', label='Center outside envelope-safe region')
ax.axhline(limit, color='#9d453b', linestyle='--', linewidth=1,
           label='Y boundary + saved reset envelope')
ax.plot(old[:, 0], old[:, 1], color='#b73d32', linewidth=2, label='Original B18: rejected')
ax.plot(new[:, 0], new[:, 1], color='#176b85', linewidth=2, label='Single three-phase candidate: static pass')
ax.scatter(*old[0], color='#242c35', s=28, zorder=5)
ax.set(xlim=(-8.3, -.5), ylim=(-5.86, -4.55), xlabel='World X (m)', ylabel='World Y (m)',
       title='C20 nominal servo-integrated paths — not robot trajectories')
ax.grid(alpha=.18)
ax.legend(loc='upper right', fontsize=8, framealpha=.96)
for suffix in ('png', 'svg'):
    fig.savefig(folder / ('nominal_corridor_20.' + suffix), dpi=180)
plt.close(fig)
print(str(folder))
