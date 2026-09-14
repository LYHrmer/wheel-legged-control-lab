"""Read-only spot audit of two taps and a cancellation; no physics imports."""
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path('/home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260914')
SOURCE = ROOT/'side_intent_integration_01/physics_01'
manifest = json.loads((SOURCE/'manifest.json').read_text())
inputs, results = {}, []
for name, direction in (('tap_A', 1), ('tap_D', -1), ('cancel_swing', 1)):
    directory = SOURCE/name
    for filename in ('states.npz', 'telemetry.csv', 'physical_side_trace.jsonl', 'physical_gate.json'):
        path = directory/filename
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == manifest[f'{name}/{filename}']['sha256']
        inputs[f'{name}/{filename}'] = digest
    with np.load(directory/'states.npz', allow_pickle=False) as data:
        position = data['qpos']
        times = data['segment_time_s']
        segment = data['segment_id']
    with (directory/'telemetry.csv').open() as stream:
        telemetry = list(csv.DictReader(stream))
    trace = [json.loads(row) for row in (directory/'physical_side_trace.jsonl').read_text().splitlines()]
    published = json.loads((directory/'physical_gate.json').read_text())
    side_indices = [i for i, row in enumerate(trace) if row['torque_source'] != 'legacy']
    start, end = side_indices[0], side_indices[-1]+1
    assert side_indices == list(range(start, end)), 'multiple side intervals or early handoff'
    first = int(telemetry[start]['state_before_index'])
    last = int(telemetry[end-1]['state_after_index'])
    quaternion = position[first, 3:7]
    w, x, y, z = quaternion
    yaw = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    left = np.asarray([-math.sin(yaw), math.cos(yaw)])
    displacement = float(direction*np.dot(position[last, :2]-position[first, :2], left))
    duration = float(times[last]-times[first])
    cycle = published['cycles'][0]
    assert len(published['cycles']) == 1
    assert abs(displacement-cycle['signed_body_lateral_m']) < 1e-12
    assert abs(duration-cycle['cycle_seconds']) < 1e-9
    assert all(trace[end-1]['wheel_contacts_after']) and trace[end-1]['pending_leg'] is None
    assert trace[end]['torque_source'] == 'legacy'
    assert all(row['torque_source'] == 'legacy' and not row['active_after'] for row in trace[end:])
    result = {'case': name, 'inferred_start_tick': start, 'inferred_end_tick': end,
              'actual_cycle_seconds': duration, 'signed_body_lateral_m': displacement,
              'four_contacts_and_no_pending_leg_before_handoff': True,
              'one_cycle_and_no_later_restart': True}
    if name.startswith('tap_'):
        retained_index = int(telemetry[end+399]['state_after_index'])
        signed = direction*((position[last:retained_index+1, :2]-position[first, :2])@left)
        retained = float(signed[-1])
        rollback = float(max(0., displacement-np.min(signed)))
        ratio = retained/displacement
        assert abs(times[retained_index]-times[last]-4.) < 1e-9
        assert segment[first] == segment[last] == segment[retained_index]
        assert abs(retained-cycle['retained_lateral_after_4s_m']) < 1e-12
        assert abs(ratio-cycle['retained_fraction_after_4s']) < 1e-12
        assert abs(rollback-cycle['max_handoff_rollback_m']) < 1e-12
        assert ratio >= .9 and rollback <= .005 and .025 <= displacement <= .05
        assert abs(duration-10.56) < 1e-9
        result.update(retained_lateral_after_4s_m=retained, retained_fraction_after_4s=ratio,
                      max_handoff_rollback_m=rollback, all_tap_gate_thresholds_passed=True)
    else:
        cancel = round(published['cancel_time_s']/.01)
        packet = telemetry[cancel]
        assert trace[cancel]['phase_before'] == 'swing'
        assert trace[cancel]['phase_after'] == 'abort_land'
        assert trace[cancel]['failure'] == 'cancelled'
        assert not all(trace[cancel]['wheel_contacts_after'])
        assert packet['side_raw_held_direction'] == '1' and packet['side_resolved_direction'] == '0'
        assert all(row['torque_source'] != 'legacy' for row in trace[cancel:end])
        result.update(cancel_time_s=trace[cancel]['before_time_s'], phase_at_cancel='swing',
                      phase_after_cancel='abort_land', airborne_at_cancel=True,
                      actual_handoff_time_s=trace[end]['before_time_s'],
                      held_input_preserved_but_resolved_zero=True,
                      side_owned_every_tick_until_safe_completion=True)
    results.append(result)
report = {'schema': 'd1-side-intent-independent-spot-audit-v1', 'all_passed': True,
          'new_physics_steps': 0, 'audited_cases': results, 'input_sha256': inputs,
          'helper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'scope': 'Independent arithmetic of tap_A/tap_D and swing cancellation only; other cases not claimed reaudited.'}
with (ROOT/'side_intent_physics_spot_audit.json').open('x') as stream:
    json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
    stream.write('\n')
print(json.dumps(report, allow_nan=False))
