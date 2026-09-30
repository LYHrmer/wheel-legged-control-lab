"""Plot saved controller records only; no simulation or policy execution."""
from pathlib import Path
import builtins
import gzip
import hashlib
import json

original_import = builtins.__import__


def guarded(name, *args, **kwargs):
    if name.partition('.')[0] in {'torch', 'mujoco', 'stable_baselines3',
                                  'wheel_legged_control', 'gymnasium'}:
        raise RuntimeError('model or physics import prohibited')
    return original_import(name, *args, **kwargs)


builtins.__import__ = guarded
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

C = Path(__file__).resolve().parent.parent
sources = {}


def read(path):
    data = path.read_bytes()
    sources[str(path)] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
    return data


def series(run, case, actor):
    folder = C / run / 'heldout' / (case + '_' + actor)
    receipt = json.loads(read(folder / 'case_receipt.json'))
    rows = []
    for block in receipt['controller_record_blocks']:
        raw = read(folder / block['file'])
        if hashlib.sha256(raw).hexdigest() != block['sha256']:
            raise ValueError('record hash mismatch')
        part = [json.loads(line) for line in gzip.decompress(raw).splitlines()]
        if len(part) != block['rows']:
            raise ValueError('record row count mismatch')
        rows.extend(part)
    if len(rows) != receipt['completed_controls']:
        raise ValueError('case incomplete')
    return {
        't': np.arange(len(rows)) * .01,
        'vx': np.array([r['info']['metrics']['body_com_vx_mps'] for r in rows]),
        'yaw': np.array([r['info']['metrics']['body_yaw_rate_rps'] for r in rows]),
        'vx_ref': np.array([r['info']['consumed_command']['forward_velocity_mps'] for r in rows]),
        'yaw_ref': np.array([r['info']['consumed_command']['yaw_rate_rps'] for r in rows]),
    }


def main():
    for name in ('primary', 'mirror'):
        report = json.loads(read(C / f'qualification_{name}_readback_17.json'))
        if not report['source_closure_verified'] or report['physical_calls_performed_by_reader'] != 0:
            raise ValueError('readback not closed')
    output = C / 'figures17'
    output.mkdir(exist_ok=False)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), layout='constrained')
    colors = {'baseline': '#c65d3b', 'combined': '#146c94', 'zero': '#6a737b'}
    primary = 'qualification_primary_01'
    for ax, case, key, title, window in (
        (axes[0, 0], 'flat_1p2_yaw', 'yaw', 'Original steering script', (3.8, 8.5)),
        (axes[0, 1], 'ramp_0p45_complete', 'vx', 'Full ramp: tracking window', (5., 13.55)),
    ):
        old = series('development_baseline_01', case, 'grouped_continue')
        new = series(primary, case, 'grouped_continue')
        ax.plot(old['t'], old[key], color=colors['baseline'], lw=1.5, label='Original controller + RL')
        ax.plot(new['t'], new[key], color=colors['combined'], lw=1.5, label='Combined repair + same RL')
        ax.plot(new['t'], new[key + '_ref'], 'k--', lw=1., label='Consumed reference')
        ax.set(xlim=window, title=title, ylabel='Yaw rate (rad/s)' if key == 'yaw' else 'Body forward speed (m/s)')
    for ax, run, case, key, title, window in (
        (axes[1, 0], 'qualification_mirror_01', 'flat_1p2_yaw', 'yaw', 'Mirrored steering script', (3.8, 8.5)),
        (axes[1, 1], primary, 'flat_1p6', 'vx', '1.6 m/s: drive and release', (0., 16.)),
    ):
        zero = series(run, case, 'zero')
        policy = series(run, case, 'grouped_continue')
        ax.plot(zero['t'], zero[key], color=colors['zero'], lw=1.3, label='Combined repair + zero residual')
        ax.plot(policy['t'], policy[key], color=colors['combined'], lw=1.5, label='Combined repair + RL')
        ax.plot(policy['t'], policy[key + '_ref'], 'k--', lw=1., label='Consumed reference')
        ax.set(xlim=window, title=title, ylabel='Yaw rate (rad/s)' if key == 'yaw' else 'Body forward speed (m/s)')
    for ax in axes.flat:
        ax.set_xlabel('Simulation time (s)')
        ax.grid(alpha=.2)
        ax.legend(fontsize=7.5, loc='best')
    fig.suptitle('C17: fixed-script qualification with the unchanged grouped RL checkpoint', fontsize=13)
    fig.savefig(output / 'tracking_17.png', dpi=180)
    fig.savefig(output / 'tracking_17.svg')
    plt.close(fig)
    identity = {'schema': 'd1-saved-data-figure-17-v1', 'inputs': sources,
                'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'model_calls': 0, 'physical_calls': 0,
                'note': 'Speed is base-body inertial COM in body axes, not whole-robot COM.'}
    with (output / 'figure_receipt_17.json').open('x') as stream:
        json.dump(identity, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps({'output': str(output), 'input_files': len(sources)}))


if __name__ == '__main__':
    main()
