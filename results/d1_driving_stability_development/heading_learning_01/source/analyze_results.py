"""Recompute physical metrics and plot every checkpoint from complete saved traces."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    summary = json.loads((ROOT/'development/summary.json').read_text())['episodes']
    rows, traces, states = [], {}, {}
    for label, original in summary.items():
        directory = ROOT/'development'/label
        with gzip.open(directory/'trace.jsonl.gz', 'rt') as stream:
            trace = [json.loads(line) for line in stream]
        with np.load(directory/'states.npz') as archive:
            qpos = archive['qpos'].copy()
            assert len(qpos) == len(trace)+1 == len(archive['observations'])
            assert archive['observations'].shape[1] == 85
            assert len(archive['applied_actions']) == len(trace)
        assert len(trace) == original['actual_transitions']
        assert all(not r['terminated'] and not r['truncated'] for r in trace[:-1])
        assert trace[-1]['terminated'] or trace[-1]['truncated']
        assert np.allclose(np.linalg.norm(qpos[:, 3:7], axis=1), 1., atol=1e-10)
        w, x, y, z = qpos[1:, 3:7].T
        roll = np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y))
        pitch = np.arcsin(np.clip(2*(w*y-z*x), -1., 1.))
        yaw = np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))
        recorded_yaw = np.array([r['metrics']['yaw_rad'] for r in trace])
        assert np.allclose(np.arctan2(np.sin(yaw-recorded_yaw), np.cos(yaw-recorded_yaw)), 0., atol=1e-12)
        error = np.array([r['heading_task']['heading_error_after'] for r in trace])
        assert np.isclose(np.sqrt(np.mean(error**2)), original['heading_rmse_rad'])
        assert np.isclose(qpos[-1, 1]-qpos[0, 1], original['cross_track_final_m'])
        row = {
            'model': label,
            **{key: original[key] for key in (
                'actual_transitions', 'time_s', 'completed_duration', 'terminal_reason',
                'velocity_rmse_mps', 'height_rmse_m', 'heading_rmse_rad', 'heading_peak_rad',
                'cross_track_peak_m', 'cross_track_final_m', 'nonwheel_contact_steps',
                'task_return', 'servo_command_return', 'heading_goal_return',
            )},
            'max_abs_actual_roll_rad': float(np.max(np.abs(roll))),
            'max_abs_actual_pitch_rad': float(np.max(np.abs(pitch))),
        }
        rows.append(row)
        traces[label], states[label] = trace, qpos
    with (ROOT/'physical_metrics.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    early = [row for row in rows if row['model'].endswith('16384')]
    late = [row for row in rows if row['model'].endswith('65536')]
    checks = {
        'all_7_trace_lengths_and_terminal_boundaries_checked': True,
        'all_quaternion_yaw_and_heading_rms_and_final_cross_track_recomputed': True,
        'actual_roll_pitch_from_qpos_distinct_from_ground_relative_metrics': True,
        'actual_development_transitions': sum(row['actual_transitions'] for row in rows),
        'early_16k_completed': sum(row['completed_duration'] for row in early),
        'late_65k_completed': sum(row['completed_duration'] for row in late),
        'zero_completed': summary['zero']['completed_duration'],
        'early_check_extra_transitions': sum(s['actual_transitions'] for s in json.loads(
            (ROOT/'first_16k_check/summary.json').read_text())['episodes'].values()),
        'claim_limit': 'Opened development road only; no stopping, turning, perturbation or GUI acceptance.',
    }
    (ROOT/'analysis.json').write_text(json.dumps(checks, indent=2)+'\n')
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), constrained_layout=True)
    styles = [('zero', 'Zero residual', '#687682', '-', 1.3),
              ('step16384', '16k checkpoint', '#c7772c', '--', 1.6),
              ('step65536', '65k checkpoint', '#176c9d', '-', 1.8)]
    for col, seed in enumerate((49001, 49002, 49003)):
        for suffix, legend, color, line, width in styles:
            label = 'zero' if suffix == 'zero' else f'seed{seed}_{suffix}'
            qpos, trace = states[label], traces[label]
            axes[0, col].plot(qpos[:, 0], qpos[:, 1], label=legend, color=color, ls=line, lw=width)
            axes[0, col].plot(qpos[-1, 0], qpos[-1, 1], 'x' if not summary[label]['completed_duration'] else 'o', color=color, ms=5)
            axes[1, col].plot([r['metrics']['time_s'] for r in trace],
                             [r['metrics']['velocity_mps'] for r in trace], color=color, ls=line, lw=width)
        axes[0, col].axvline(5.3, color='#b94343', ls=':', lw=1)
        axes[0, col].axhline(0., color='#b0b5ba', lw=.6)
        axes[0, col].set(title=f'Seed {seed}', xlabel='World x (m)', xlim=(-4., 5.5), ylim=(-.15, .15))
        trace = traces['zero']
        axes[1, col].plot([r['metrics']['time_s'] for r in trace],
                         [r['heading_task']['user_command_before']['forward_velocity_mps'] for r in trace],
                         color='#252a30', ls=':', lw=1.2)
        axes[1, col].set(xlabel='Simulation time (s)', xlim=(0., 32.), ylim=(-.05, .55))
        for axis in axes[:, col]:
            axis.grid(alpha=.2)
    axes[0, 0].set_ylabel('World y (m)')
    axes[1, 0].set_ylabel('Forward velocity (m/s)')
    axes[0, 0].legend(loc='lower left', fontsize=8)
    fig.suptitle('Opened development road: all 16k and 65k checkpoints\nCross markers: early termination; red dotted line: x boundary', fontsize=12)
    for extension in ('png', 'svg'):
        fig.savefig(ROOT/f'checkpoint_comparison.{extension}', dpi=180)
    plt.close(fig)
    files = {str(p.relative_to(ROOT)): {'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'bytes': p.stat().st_size}
             for p in sorted(ROOT.rglob('*')) if p.is_file() and p != ROOT/'manifest.json'}
    (ROOT/'manifest.json').write_text(json.dumps(files, indent=2)+'\n')
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
