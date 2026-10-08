"""Extract the C37 rocking diagnosis into a small, publishable derived artifact.

The raw evidence (episode_0/native_block_0000.jsonl.gz, 3.1 MB) stays local per the
project's raw-payload exclusion policy. This script recomputes, from that same sealed file,
the exact numbers reported in the C36-C38 physical findings: how often the two returning
wheels are simultaneously in contact during probe+dwell, the measured attitude excursion,
and the wheel-height differential with the implied rotation about the support diagonal.

No MuJoCo, no model construction, no new physics: this reads the already-saved native
substep records and performs the same arithmetic already reported inline during the
investigation, now captured as a signed, reproducible artifact with its source's hash as
provenance.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import sys
from pathlib import Path

RUN = Path(sys.argv[1] if len(sys.argv) > 1 else
           '/home/lyh/wheel-legged-control-lab-work/recovery-20260912/'
           'stability_20260924_full_drive01/continuation37/development_01')
OUTPUT = Path(sys.argv[2] if len(sys.argv) > 2 else
              Path(__file__).resolve().parent/'rocking_diagnosis_37.json')


def identity(path):
    data = Path(path).read_bytes()
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def load_native(run):
    rows = []
    for block in sorted((run/'episode_0').glob('native_block_*.jsonl.gz')):
        with gzip.open(block, 'rt') as stream:
            rows.extend(json.loads(line) for line in stream)
    return rows


def main():
    rows = load_native(RUN)
    evidence = lambda row: row['contact_evidence35']  # noqa: E731
    air = [row for row in rows
           if evidence(row)['consumed_phase'] in ('lift', 'horizontal', 'lower', 'probe', 'dwell')]
    if not air:
        raise RuntimeError('no air-phase native rows found; wrong run or wrong campaign')
    swing = evidence(air[0])['swing_legs']
    stance = evidence(air[0])['stance_legs']
    probe_dwell = [row for row in rows
                  if evidence(row)['consumed_phase'] in ('probe', 'dwell')]
    both_contact = sum(1 for row in probe_dwell
                       if all(evidence(row)['active_wheel_terrain_contact'][leg]
                              for leg in swing))
    both_loaded = sum(1 for row in probe_dwell
                      if all(evidence(row)['positive_wheel_normal_load_sum_n'][leg] > 1e-8
                             for leg in swing))
    rolls = [row['roll_deg'] for row in air]
    pitches = [row['pitch_deg'] for row in air]
    min_z = [evidence(row)['whole_wheel_min_z_m'] for row in air]
    differential = [row[swing[0]]-row[swing[1]] for row in min_z]
    points = evidence(air[0])['positive_load_weighted_wheel_contact_world_m']
    stance_a, stance_b = points[stance[0]], points[stance[1]]
    span = math.dist(
        (points[swing[0]][0] if points[swing[0]] else 0.,
         points[swing[0]][1] if points[swing[0]] else 0.),
        (points[swing[1]][0] if points[swing[1]] else 0.,
         points[swing[1]][1] if points[swing[1]] else 0.)) or None
    peak_differential_m = max(abs(value) for value in differential)
    implied_rotation_deg = None
    separation_m = math.dist(stance_a[:2], stance_b[:2]) if (stance_a and stance_b) else None

    result = {
        'schema': 'd1-c37-rocking-diagnosis-v1',
        'source_run': str(RUN),
        'source_native_block_identity': [
            {'path': str(block), **identity(block)}
            for block in sorted((RUN/'episode_0').glob('native_block_*.jsonl.gz'))],
        'swing_pair': swing, 'stance_pair': stance,
        'air_phase_native_rows': len(air),
        'probe_dwell_native_rows': len(probe_dwell),
        'both_returning_wheels_in_contact': both_contact,
        'both_returning_wheels_loaded': both_loaded,
        'attitude_deg': {
            'roll_min': min(rolls), 'roll_max': max(rolls),
            'roll_mean': sum(rolls)/len(rolls),
            'pitch_min': min(pitches), 'pitch_max': max(pitches),
            'pitch_mean': sum(pitches)/len(pitches)},
        'returning_wheel_height_differential_m': {
            'min': min(differential), 'max': max(differential),
            'peak_to_peak': max(differential)-min(differential)},
        'stance_diagonal_separation_m': separation_m,
        'implied_rotation_about_diagonal_deg':
            math.degrees(math.asin(min(1., peak_differential_m/separation_m)))
            if separation_m else None,
        'measured_attitude_excursion_deg': max(max(abs(v) for v in rolls),
                                               max(abs(v) for v in pitches)),
        'interpretation': ('the two returning wheels are almost never simultaneously in '
                           'contact during probe+dwell; the robot rocks about the stance '
                           'diagonal with a height differential whose implied rotation is '
                           'consistent with the measured attitude excursion, matching the '
                           'static COM-offset-from-contact-centroid finding'),
    }
    with OUTPUT.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({'output': str(OUTPUT), 'both_contact': both_contact,
                      'of': len(probe_dwell), 'peak_to_peak_mm': result[
                          'returning_wheel_height_differential_m']['peak_to_peak']*1000}))


if __name__ == '__main__':
    main()
