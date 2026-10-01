"""Bounded read-only diagnostic of the saved D1 lateral-step records.

Reads only restored saved evidence (trace / telemetry / states / protocol) and
reports arithmetic over it. No controller, model, simulator, MuJoCo or torch
import; no physics step, no training, no threshold search. Phase names here are
*saved labels* from the recorded trace rows, not a claim about what the
controller computed inside a tick.

Usage:
    python analyze_saved_side27.py --inputs ../saved_input_27 --output NEW.json

The output path must not already exist; it is opened with mode 'x'.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

SCHEMA = 'd1-c27-saved-side-speed-diagnostic-v1'
CASES = ('tap_A', 'tap_D')
EVIDENCE_INDEX = Path('/home/lyh/wheel-legged-control-lab/results/d1_driving_stability_development'
                      '/side_input_commit/evidence_index.json')
REFERENCE_HELPER = Path('/home/lyh/wheel-legged-control-lab/results/d1_driving_stability_development'
                        '/side_input_commit/helpers/audit_side_intent_spot.py')
PROTOCOL_KEYS = ('control_dt_s', 'physics_dt_s', 'sampling_mode', 'policy', 'controller_class',
                 'manual_gui_tested', 'side_step_profile', 'side_step_distance_m', 'side_input_mode',
                 'compiled_model_sha256', 'validation_harness_sha256', 'legacy_measurement_timing',
                 'segment_rule', 'state_timing')
PROTOCOL_SOURCES = ('scripts/d1_fast_side_step.py', 'scripts/d1_side_step.py',
                    'scripts/d1_course_side_step.py', 'scripts/d1_course_side_intent.py',
                    'scripts/run_d1_course_drive.py', 'src/wheel_legged_control/d1/interactive.py')

HANDOFF_OBSERVATION_S = 4.
NOMINAL_CONTROL_DT_S = .01
DT_TOLERANCE_S = 1e-9
TIME_TOLERANCE_S = 1e-6

#: Old baseline of the same two taps, fixed before this module was written.
BASELINE = {
    'side_interval_ticks': [200, 1256],
    'cycle_seconds': 10.56,
    'signed_body_lateral_m': {'tap_A': .03703154847442854, 'tap_D': .03704205955635278},
    'sampling_mode': 'legacy_mixed',
    'source': ('physical_gate.json / matrix summary of stability_20260914'
               ' side_intent_integration_01/physics_01 tap_A and tap_D, restated in'
               ' continuation27/saved_side_timing_27.json'),
    'independent_arithmetic_reference': str(REFERENCE_HELPER),
}


class DataError(Exception):
    """Saved data did not satisfy a documented precondition of this diagnostic."""


# --------------------------------------------------------------------- primitives

def sha256_and_bytes(path):
    """Digest and size of a file, read in bounded chunks."""
    digest = hashlib.sha256()
    size = 0
    with Path(path).open('rb') as stream:
        while True:
            chunk = stream.read(1 << 20)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    return {'sha256': digest.hexdigest(), 'bytes': size}


def json_safe_number(name, value):
    """Return ``value`` as a float, refusing NaN/inf before it reaches the report."""
    number = float(value)
    if not math.isfinite(number):
        raise DataError(f'{name} is not finite: {value!r}')
    return number


def parse_floats(name, values):
    """Parse saved text/numeric cells into a finite float array."""
    parsed = []
    for index, value in enumerate(values):
        text = value.strip() if isinstance(value, str) else value
        if text is None or text == '':
            raise DataError(f'{name}[{index}] is empty in the saved record')
        try:
            parsed.append(float(text))
        except (TypeError, ValueError):
            raise DataError(f'{name}[{index}] is not a number: {value!r}') from None
    return ensure_finite(name, parsed)


def ensure_finite(name, values):
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        raise DataError(f'{name} contains non-finite samples')
    return array


def wrap_angle(angle):
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def yaw_from_quaternion(quaternion):
    """Yaw of a (w, x, y, z) quaternion, matching the reference helper exactly."""
    w, x, y, z = (float(v) for v in quaternion)
    return math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))


def yaw_series(quaternions):
    q = ensure_finite('quaternions', quaternions)
    if q.ndim != 2 or q.shape[1] != 4:
        raise DataError('quaternions must be an (n, 4) array')
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))


def body_axes(yaw):
    """Forward and left unit axes of the body frame at ``yaw``."""
    return (np.asarray([math.cos(yaw), math.sin(yaw)]),
            np.asarray([-math.sin(yaw), math.cos(yaw)]))


def body_frame_track(positions_xy, origin_xy, yaw, direction):
    """Project world xy onto the start body frame.

    ``lateral`` is signed by the commanded ``direction`` (positive = commanded
    side), ``longitudinal`` keeps the body forward sign. Both are invariant to a
    world rotation applied to the positions and the yaw together.
    """
    if direction not in (-1, 1):
        raise DataError(f'direction must be -1 or +1, got {direction!r}')
    forward, left = body_axes(yaw)
    positions = ensure_finite('positions_xy', positions_xy)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise DataError('positions_xy must be an (n, 2) array')
    delta = positions - ensure_finite('origin_xy', origin_xy)
    return direction*(delta@left), delta@forward


def uniform_dt(times, nominal=NOMINAL_CONTROL_DT_S, tolerance=DT_TOLERANCE_S):
    """Exact saved sample step, rejecting gaps, repeats and reversals."""
    t = ensure_finite('times', times)
    if t.ndim != 1 or t.size < 2:
        raise DataError('need at least two time samples')
    steps = np.diff(t)
    if float(steps.min()) <= 0.:
        raise DataError('saved times are not strictly increasing')
    if float(steps.max() - steps.min()) > tolerance:
        raise DataError(f'non-uniform saved time step: {float(steps.min())!r}'
                        f'..{float(steps.max())!r}')
    step = float(steps.mean())
    if nominal is not None and abs(step - nominal) > tolerance:
        raise DataError(f'saved time step {step!r} differs from nominal {nominal!r}')
    return step


def contiguous_tick_times(before_times, after_times, tolerance=1e-12):
    """Validate that each tick ends where the next begins; return the exact dt."""
    before = ensure_finite('before_time_s', before_times)
    after = ensure_finite('after_time_s', after_times)
    if before.size != after.size or before.size < 2:
        raise DataError('tick time columns must align and hold 2+ ticks')
    gaps = np.abs(before[1:] - after[:-1])
    if float(gaps.max()) > tolerance:
        raise DataError('saved tick times are not contiguous across the interval')
    return uniform_dt(np.append(before, after[-1]))


def sole_contiguous_interval(flags):
    """Infer the single contiguous run of truthy flags; end is exclusive."""
    marked = [i for i, flag in enumerate(flags) if flag]
    if not marked:
        raise DataError('no side-owned ticks in the saved trace')
    start, end = marked[0], marked[-1] + 1
    if marked != list(range(start, end)):
        raise DataError('saved trace has more than one side-owned interval; refusing to pick one')
    return start, end


def require_single_segment(segment_id, first, last):
    """Refuse to measure displacement across a simulator reset."""
    window = np.asarray(segment_id)[first:last + 1]
    if window.size == 0:
        raise DataError('empty segment window')
    unique = sorted({int(value) for value in window})
    if len(unique) != 1:
        raise DataError(f'samples {first}..{last} cross segment ids {unique}')
    return unique[0]


def validate_state_chain(before_index, after_index, first, last):
    """Each saved tick must advance the state index by exactly one, with no jump."""
    before = np.asarray(before_index, dtype=np.int64)
    after = np.asarray(after_index, dtype=np.int64)
    if before.size != after.size or before.size == 0:
        raise DataError('state index columns must align and be non-empty')
    if not np.array_equal(after, before + 1):
        raise DataError('a saved tick does not advance the state index by exactly one')
    if not np.array_equal(before[1:], after[:-1]):
        raise DataError('saved tick state indices are discontinuous')
    if int(before[0]) != first or int(after[-1]) != last:
        raise DataError('state index chain does not match the inferred interval endpoints')
    return True


# ----------------------------------------------------------------------- measures

def net_and_path_speeds(times, lateral, longitudinal):
    """Separate the net displacement rate from the peak and path-length rates."""
    t = ensure_finite('times', times)
    y = ensure_finite('lateral', lateral)
    x = ensure_finite('longitudinal', longitudinal)
    if not (t.size == y.size == x.size) or t.size < 2:
        raise DataError('times, lateral and longitudinal must align and hold 2+ samples')
    duration = float(t[-1] - t[0])
    if duration <= 0.:
        raise DataError('non-positive duration')
    steps = np.diff(t)
    if float(steps.min()) <= 0.:
        raise DataError('saved times are not strictly increasing')
    d_lat, d_lon = np.diff(y), np.diff(x)
    planar = np.hypot(d_lat, d_lon)
    net_lateral = float(y[-1] - y[0])
    net_longitudinal = float(x[-1] - x[0])
    path_length = float(planar.sum())
    return {
        'duration_s': duration,
        'net_lateral_m': net_lateral,
        'net_longitudinal_drift_m': net_longitudinal,
        'net_signed_lateral_speed_mps': net_lateral/duration,
        'net_longitudinal_drift_speed_mps': net_longitudinal/duration,
        'path_length_m': path_length,
        'mean_path_speed_mps': path_length/duration,
        'peak_sample_planar_speed_mps': float((planar/steps).max()),
        'peak_abs_lateral_rate_mps': float((np.abs(d_lat)/steps).max()),
        'path_to_net_ratio': path_length/abs(net_lateral) if net_lateral else None,
        'max_lateral_m': float(y.max()),
        'min_lateral_m': float(y.min()),
        'time_of_max_lateral_s': float(t[int(np.argmax(y))] - t[0]),
        'time_of_min_lateral_s': float(t[int(np.argmin(y))] - t[0]),
        'max_abs_longitudinal_m': float(np.abs(x).max()),
    }


def reversal_metrics(times, lateral):
    """Initial negative-direction travel before the start position is recrossed."""
    t = ensure_finite('times', times)
    y = ensure_finite('lateral', lateral)
    if t.size != y.size or t.size < 2:
        raise DataError('times and lateral must align and hold 2+ samples')
    duration = float(t[-1] - t[0])
    if duration <= 0.:
        raise DataError('non-positive duration')
    steps = np.diff(t)
    negative = y[:-1] < 0.
    negative_time = float(steps[negative].sum())
    result = {
        'sample_count': int(y.size),
        'negative_sample_count': int((y < 0.).sum()),
        'negative_time_s': negative_time,
        'negative_fraction_of_cycle': negative_time/duration,
        'first_negative_sample_offset': None,
        'first_negative_time_s': None,
        'initial_negative_excursion_m': None,
        'recrossed_start_position': False,
        'time_to_recross_start_s': None,
        'fraction_of_cycle_before_recrossing': None,
    }
    if not negative.any():
        return result
    first = int(np.argmax(negative))
    result['first_negative_sample_offset'] = first
    result['first_negative_time_s'] = float(t[first] - t[0])
    later = np.flatnonzero(y[first:] >= 0.)
    stop = first + int(later[0]) if later.size else int(y.size - 1)
    result['initial_negative_excursion_m'] = float(-y[first:stop + 1].min())
    if later.size:
        result['recrossed_start_position'] = True
        result['time_to_recross_start_s'] = float(t[stop] - t[0])
        result['fraction_of_cycle_before_recrossing'] = float(t[stop] - t[0])/duration
    return result


def handoff_retention(times, lateral, cycle_end_sample, observation_s=HANDOFF_OBSERVATION_S,
                      tolerance=DT_TOLERANCE_S):
    """Retention and worst rollback of the net lateral over the saved observation.

    Only samples at or after the cycle end are searched, so a later segment that
    restarts ``segment_time_s`` cannot supply the observation endpoint.
    """
    t = ensure_finite('times', times)
    y = ensure_finite('lateral', lateral)
    if t.size != y.size:
        raise DataError('times and lateral must align')
    if not 0 <= cycle_end_sample < t.size - 1:
        raise DataError('cycle end sample is outside the saved states')
    target = float(t[cycle_end_sample]) + float(observation_s)
    tail = t[cycle_end_sample:]
    matches = np.flatnonzero(np.abs(tail - target) <= tolerance)
    if matches.size != 1:
        raise DataError(f'no unique saved sample {observation_s}s after the cycle end')
    stop = cycle_end_sample + int(matches[0])
    window = y[cycle_end_sample:stop + 1]
    net = float(y[cycle_end_sample])
    retained = float(window[-1])
    return {
        'observation_s': float(observation_s),
        'observation_is_saved_evidence_not_a_forced_wait': True,
        'observation_end_sample': stop,
        'retained_lateral_m': retained,
        'retained_fraction': retained/net if net else None,
        'max_rollback_m': float(max(0., net - float(window.min()))),
        'min_lateral_in_window_m': float(window.min()),
        'max_lateral_in_window_m': float(window.max()),
    }


def label_durations(labels, step_durations, expected_total, tolerance=TIME_TOLERANCE_S):
    """Sum saved per-tick durations by label and validate the total."""
    steps = ensure_finite('step_durations', step_durations)
    if len(labels) != steps.size:
        raise DataError('labels and step durations must align')
    totals, counts = {}, {}
    for label, step in zip(labels, steps):
        key = 'null' if label is None else str(label)
        totals[key] = totals.get(key, 0.) + float(step)
        counts[key] = counts.get(key, 0) + 1
    total = float(sum(totals.values()))
    if abs(total - float(expected_total)) > tolerance:
        raise DataError(f'label durations sum to {total!r}, not the interval duration'
                        f' {float(expected_total)!r}')
    return {'durations_s': totals, 'tick_counts': counts, 'sum_s': total,
            'sums_to_interval_duration': True}


def contiguous_label_segments(labels, before_times, after_times, tick_offset=0):
    """Contiguous runs of one saved label, with their exact saved start/end times."""
    if not (len(labels) == len(before_times) == len(after_times)):
        raise DataError('labels and times must align')
    if not labels:
        raise DataError('no labels')
    segments = []
    start = 0
    for index in range(1, len(labels) + 1):
        if index < len(labels) and labels[index] == labels[start]:
            continue
        label = labels[start]
        segments.append({
            'phase': 'null' if label is None else str(label),
            'start_tick': tick_offset + start,
            'end_tick': tick_offset + index,
            'tick_count': index - start,
            'start_time_s': json_safe_number('start_time_s', before_times[start]),
            'end_time_s': json_safe_number('end_time_s', after_times[index - 1]),
            'duration_s': json_safe_number('duration_s',
                                           after_times[index - 1] - before_times[start]),
            'next_phase': None if index >= len(labels) else (
                'null' if labels[index] is None else str(labels[index])),
        })
        start = index
    return segments


# ------------------------------------------------------------------------ loading

TRACE_FIELDS = ('tick', 'before_time_s', 'after_time_s', 'phase_before', 'phase_after',
                'torque_source', 'failure', 'active_after', 'wheel_contacts_after',
                'base_origin_speed_after_mps', 'support_margin_m', 'contact_dwell_s',
                'pending_leg', 'torque_within_limits')
TELEMETRY_FIELDS = ('tick', 'segment_id', 'state_before_index', 'state_after_index',
                    'side_direction', 'side_phase', 'side_active_after',
                    'measured_wheel_contacts_after', 'torque_saturation_fraction',
                    'applied_controller')


def load_trace(path):
    """Saved per-tick side trace, keeping only the fields this diagnostic uses."""
    rows = []
    with Path(path).open() as stream:
        for number, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            missing = [name for name in TRACE_FIELDS if name not in row]
            if missing:
                raise DataError(f'{path}:{number} missing trace fields {missing}')
            rows.append({name: row[name] for name in TRACE_FIELDS})
    if not rows:
        raise DataError(f'{path} is empty')
    if [row['tick'] for row in rows] != list(range(1, len(rows) + 1)):
        raise DataError(f'{path} ticks are not 1..n')
    return rows


def load_telemetry(path):
    """Saved per-tick telemetry, restricted to the explicit state indexing columns."""
    rows = []
    with Path(path).open(newline='') as stream:
        reader = csv.DictReader(stream)
        missing = [name for name in TELEMETRY_FIELDS if name not in (reader.fieldnames or ())]
        if missing:
            raise DataError(f'{path} missing telemetry columns {missing}')
        for row in reader:
            rows.append({name: row[name] for name in TELEMETRY_FIELDS})
    if not rows:
        raise DataError(f'{path} has no rows')
    if [int(row['tick']) for row in rows] != list(range(1, len(rows) + 1)):
        raise DataError(f'{path} ticks are not 1..n')
    return rows


def load_states(path):
    """qpos / segment_time_s / segment_id as documented in the reference helper."""
    with np.load(path, allow_pickle=False) as data:
        for key in ('qpos', 'segment_time_s', 'segment_id'):
            if key not in data:
                raise DataError(f'{path} has no {key!r} array')
        position = np.asarray(data['qpos'], dtype=float)
        times = np.asarray(data['segment_time_s'], dtype=float)
        segment = np.asarray(data['segment_id'])
    if position.ndim != 2 or position.shape[1] < 7:
        raise DataError(f'{path} qpos must be (n, >=7)')
    if not (position.shape[0] == times.size == segment.size):
        raise DataError(f'{path} arrays disagree in length')
    ensure_finite('qpos[:, :7]', position[:, :7])
    ensure_finite('segment_time_s', times)
    return position, times, segment


def saved_direction(telemetry, start, end):
    """Commanded side direction taken from saved telemetry, not inferred from motion."""
    values = set()
    for row in telemetry[start:end]:
        text = (row['side_direction'] or '').strip()
        if not text:
            continue
        value = int(float(text))
        if value:
            values.add(value)
    if len(values) != 1:
        raise DataError(f'saved telemetry does not give one side direction: {sorted(values)}')
    direction = values.pop()
    if direction not in (-1, 1):
        raise DataError(f'saved side direction {direction} is not -1 or +1')
    return direction


# ------------------------------------------------------------------- verification

def verify_inputs(inputs_dir, evidence_index_path):
    """Check restored bytes against the restore receipt and the public evidence index."""
    receipt_path = inputs_dir/'restore_receipt.json'
    receipt = json.loads(receipt_path.read_text())
    index = None
    index_meta = {'path': str(evidence_index_path), 'available': False, 'has_files_table': False}
    if evidence_index_path.is_file():
        index_meta.update(sha256_and_bytes(evidence_index_path), available=True)
        index = json.loads(evidence_index_path.read_text()).get('files')
        index_meta['has_files_table'] = isinstance(index, dict)
        if not index_meta['has_files_table']:
            index = None
    records = {}
    for key, record in sorted(receipt.get('records', {}).items()):
        original = record.get('original', {})
        restored = Path(record['restored'])
        expected = inputs_dir/Path(key).parent.name/Path(key).name
        entry = {
            'restored_path': str(restored),
            'restored_path_inside_inputs': restored == expected,
            'receipt_claims_byte_exact': bool(record.get('byte_exact')),
            'exists': restored.is_file(),
            'measured_sha256': None,
            'measured_bytes': None,
            'receipt_sha256_match': None,
            'receipt_bytes_match': None,
            'evidence_index_key': None,
            'evidence_index_sha256_match': None,
            'evidence_index_bytes_match': None,
        }
        if entry['exists']:
            measured = sha256_and_bytes(restored)
            entry.update(measured_sha256=measured['sha256'], measured_bytes=measured['bytes'],
                         receipt_sha256_match=measured['sha256'] == original.get('sha256'),
                         receipt_bytes_match=measured['bytes'] == original.get('bytes'))
            if index is not None and key in index:
                entry.update(evidence_index_key=key,
                             evidence_index_sha256_match=(measured['sha256']
                                                          == index[key].get('sha256')),
                             evidence_index_bytes_match=(measured['bytes']
                                                         == index[key].get('bytes')))
        records[key] = entry
    present = [entry for entry in records.values() if entry['exists']]
    indexed = [entry for entry in present if entry['evidence_index_key'] is not None]
    return {
        'restore_receipt': dict(sha256_and_bytes(receipt_path), path=str(receipt_path)),
        'receipt_model_calls': receipt.get('model_calls'),
        'receipt_physics_calls': receipt.get('physics_calls'),
        'evidence_index': index_meta,
        'records': records,
        'record_count': len(records),
        'all_restored_files_present': bool(records) and len(present) == len(records),
        'all_match_receipt_sha256': bool(present) and all(e['receipt_sha256_match']
                                                          for e in present),
        'all_match_receipt_bytes': bool(present) and all(e['receipt_bytes_match'] for e in present),
        'evidence_index_checked_count': len(indexed),
        'all_checked_match_evidence_index': (None if not indexed else
                                             all(e['evidence_index_sha256_match']
                                                 and e['evidence_index_bytes_match']
                                                 for e in indexed)),
        'records_without_evidence_index_entry': sorted(key for key, entry in records.items()
                                                       if entry['exists']
                                                       and entry['evidence_index_key'] is None),
    }


# ----------------------------------------------------------------------- analysis

def analyze_case(case_dir, name):
    """All saved-data measures for one side-step record."""
    trace = load_trace(case_dir/'physical_side_trace.jsonl')
    telemetry = load_telemetry(case_dir/'telemetry.csv')
    position, times, segment = load_states(case_dir/'states.npz')
    protocol = json.loads((case_dir/'protocol.json').read_text())
    if len(trace) != len(telemetry):
        raise DataError(f'{name}: trace and telemetry tick counts differ')

    start, end = sole_contiguous_interval([row['torque_source'] != 'legacy' for row in trace])
    owned_trace = trace[start:end]
    owned_telemetry = telemetry[start:end]
    direction = saved_direction(telemetry, start, end)

    before_times = ensure_finite('before_time_s', [row['before_time_s'] for row in owned_trace])
    after_times = ensure_finite('after_time_s', [row['after_time_s'] for row in owned_trace])
    step_durations = after_times - before_times
    trace_dt = contiguous_tick_times(before_times, after_times)
    trace_duration = float(after_times[-1] - before_times[0])

    first = int(owned_telemetry[0]['state_before_index'])
    last = int(owned_telemetry[-1]['state_after_index'])
    validate_state_chain([int(row['state_before_index']) for row in owned_telemetry],
                         [int(row['state_after_index']) for row in owned_telemetry], first, last)
    if not 0 <= first < last < position.shape[0]:
        raise DataError(f'{name}: state indices {first}..{last} outside the saved states')
    cycle_segment = require_single_segment(segment, first, last)
    state_dt = uniform_dt(times[first:last + 1])

    yaw_start = yaw_from_quaternion(position[first, 3:7])
    lateral_all, longitudinal_all = body_frame_track(position[:, :2], position[first, :2],
                                                     yaw_start, direction)
    cycle = slice(first, last + 1)
    speeds = net_and_path_speeds(times[cycle], lateral_all[cycle], longitudinal_all[cycle])
    cycle_seconds = float(times[last] - times[first])
    if abs(cycle_seconds - trace_duration) > TIME_TOLERANCE_S:
        raise DataError(f'{name}: state cycle time {cycle_seconds!r} disagrees with the'
                        f' trace duration {trace_duration!r}')

    yaws = yaw_series(position[cycle, 3:7])
    yaw_deviation = np.asarray([wrap_angle(float(value) - yaw_start) for value in yaws])

    retention, retention_note = None, None
    try:
        retention = handoff_retention(times, lateral_all, last)
        require_single_segment(segment, first, retention['observation_end_sample'])
        uniform_dt(times[last:retention['observation_end_sample'] + 1])
        retention_note = ('saved samples only, same segment id as the cycle;'
                          ' observation window, not an enforced dwell')
    except DataError as error:
        retention_note = f'unavailable: {error}'

    margins = ensure_finite('support_margin_m', [row['support_margin_m'] for row in owned_trace])
    saturation = parse_floats('torque_saturation_fraction',
                              [row['torque_saturation_fraction'] for row in telemetry])
    four_contacts = np.asarray([bool(row['wheel_contacts_after'])
                                and all(row['wheel_contacts_after']) for row in owned_trace])
    airborne_runs, run = [], 0
    for flag in four_contacts:
        if flag:
            if run:
                airborne_runs.append(run)
            run = 0
        else:
            run += 1
    if run:
        airborne_runs.append(run)

    handoff_tick = trace[end] if end < len(trace) else None
    return {
        'case': name,
        'saved_direction': direction,
        'interval': {
            'inferred_start_tick': start,
            'inferred_end_tick': end,
            'sole_contiguous_side_owned_interval': True,
            'ownership_rule': "trace torque_source != 'legacy'",
            'tick_count': end - start,
            'state_before_index': first,
            'state_after_index': last,
            'state_index_source': 'telemetry state_before_index / state_after_index columns',
            'segment_id': cycle_segment,
            'trace_dt_s': trace_dt,
            'state_dt_s': state_dt,
            'trace_duration_s': trace_duration,
            'state_cycle_seconds': cycle_seconds,
            'trace_and_state_duration_agree': True,
            'total_saved_ticks': len(trace),
        },
        'displacement': {
            'frame': 'start-tick body frame from the saved quaternion; lateral signed by the'
                     ' saved commanded direction',
            'start_yaw_rad': yaw_start,
            'net_signed_body_lateral_m': speeds['net_lateral_m'],
            'net_body_longitudinal_drift_m': speeds['net_longitudinal_drift_m'],
            'max_abs_body_longitudinal_m': speeds['max_abs_longitudinal_m'],
            'net_lateral_is_correct_signed': speeds['net_lateral_m'] > 0.,
        },
        'speeds': {
            'definition': 'net signed speed is net body-lateral displacement divided by the saved'
                          ' simulated cycle time; path-length and peak sample rates are different'
                          ' quantities and are reported separately',
            **{key: speeds[key] for key in (
                'duration_s', 'net_signed_lateral_speed_mps', 'net_longitudinal_drift_speed_mps',
                'path_length_m', 'mean_path_speed_mps', 'peak_sample_planar_speed_mps',
                'peak_abs_lateral_rate_mps', 'path_to_net_ratio')},
            'max_saved_base_origin_speed_mps': float(ensure_finite(
                'base_origin_speed_after_mps',
                [row['base_origin_speed_after_mps'] for row in owned_trace]).max()),
        },
        'full_cycle_extrema': {key: speeds[key] for key in (
            'max_lateral_m', 'min_lateral_m', 'time_of_max_lateral_s', 'time_of_min_lateral_s')},
        'reversal': dict(reversal_metrics(times[cycle], lateral_all[cycle]),
                         note='negative lateral is travel opposite the commanded side before the'
                              ' start position is recrossed; sample times are saved simulated time'),
        'yaw': {
            'source': 'saved base quaternion (w, x, y, z) in qpos[3:7]',
            'net_yaw_change_rad': wrap_angle(float(yaws[-1]) - yaw_start),
            'max_abs_yaw_deviation_rad': float(np.abs(yaw_deviation).max()),
        },
        'handoff_retention': retention,
        'handoff_retention_note': retention_note,
        'phase_labels': {
            'note': 'saved per-tick labels at control-tick resolution; phase_before and phase_after'
                    ' are reported separately and neither is a claim about the exact'
                    ' inside-compute phase of a tick',
            'dt_s': trace_dt,
            'interval_duration_s': trace_duration,
            'phase_before': label_durations([row['phase_before'] for row in owned_trace],
                                            step_durations, trace_duration),
            'phase_after': label_durations([row['phase_after'] for row in owned_trace],
                                           step_durations, trace_duration),
            'telemetry_side_phase': label_durations([row['side_phase'] for row in owned_telemetry],
                                                    step_durations, trace_duration),
            'contiguous_phase_before_segments': contiguous_label_segments(
                [row['phase_before'] for row in owned_trace], before_times, after_times,
                tick_offset=start),
        },
        'saved_safety_checks': {
            'all_ticks_torque_within_limits': all(bool(row['torque_within_limits'])
                                                  for row in owned_trace),
            'ticks_outside_torque_limits': sum(0 if row['torque_within_limits'] else 1
                                               for row in owned_trace),
            'max_saved_torque_saturation_fraction_cycle': float(saturation[start:end].max()),
            'max_saved_torque_saturation_fraction_run': float(saturation.max()),
            'min_saved_support_margin_m': float(margins.min()),
            'min_saved_support_margin_four_contacts_m': (float(margins[four_contacts].min())
                                                         if four_contacts.any() else None),
            'ticks_without_four_contacts': int((~four_contacts).sum()),
            'longest_run_without_four_contacts_ticks': max(airborne_runs) if airborne_runs else 0,
            'four_contacts_at_last_owned_tick': bool(four_contacts[-1]),
            'pending_leg_none_at_last_owned_tick': owned_trace[-1]['pending_leg'] is None,
            'failure_labels_in_interval': sorted({str(row['failure']) for row in owned_trace}),
            'handoff_tick_is_legacy': (None if handoff_tick is None
                                       else handoff_tick['torque_source'] == 'legacy'),
            'no_side_ownership_after_handoff': all(row['torque_source'] == 'legacy'
                                                   and not row['active_after']
                                                   for row in trace[end:]),
        },
        'provenance': {
            'protocol': {key: protocol.get(key) for key in PROTOCOL_KEYS},
            'protocol_source_sha256': {key: protocol.get('source_sha256', {}).get(key)
                                       for key in PROTOCOL_SOURCES},
            'sampling_mode_is_legacy_mixed': protocol.get('sampling_mode') == 'legacy_mixed',
            'manual_gui_tested': protocol.get('manual_gui_tested'),
            'applied_controller_labels_in_interval': sorted(
                {row['applied_controller'] for row in owned_telemetry}),
        },
    }


def compare_to_baseline(name, case):
    """Difference of this run's arithmetic against the fixed old baseline."""
    expected = BASELINE['signed_body_lateral_m'].get(name)
    interval = case['interval']
    net = case['displacement']['net_signed_body_lateral_m']
    ticks = [interval['inferred_start_tick'], interval['inferred_end_tick']]
    return {
        'baseline_signed_body_lateral_m': expected,
        'measured_minus_baseline_m': None if expected is None else net - expected,
        'baseline_cycle_seconds': BASELINE['cycle_seconds'],
        'measured_minus_baseline_cycle_s': (interval['state_cycle_seconds']
                                            - BASELINE['cycle_seconds']),
        'baseline_interval_ticks': BASELINE['side_interval_ticks'],
        'measured_interval_ticks': ticks,
        'interval_matches_baseline': ticks == BASELINE['side_interval_ticks'],
    }


def build_report(inputs_dir, evidence_index_path):
    inputs_dir = Path(inputs_dir)
    cases, comparison = {}, {}
    for name in CASES:
        case_dir = inputs_dir/name
        if not case_dir.is_dir():
            raise DataError(f'missing saved case directory {case_dir}')
        cases[name] = analyze_case(case_dir, name)
        comparison[name] = compare_to_baseline(name, cases[name])
    reference = (dict(sha256_and_bytes(REFERENCE_HELPER), path=str(REFERENCE_HELPER))
                 if REFERENCE_HELPER.is_file() else {'path': str(REFERENCE_HELPER),
                                                     'available': False})
    return {
        'schema': SCHEMA,
        'scope': 'Arithmetic over the restored saved tap_A/tap_D records only. No controller,'
                 ' model, simulator or policy was loaded; no physics was stepped; no window or'
                 ' threshold was chosen after seeing a result.',
        'inputs_dir': str(inputs_dir),
        'module_sha256': sha256_and_bytes(__file__)['sha256'],
        'reference_schema_helper': reference,
        'input_verification': verify_inputs(inputs_dir, evidence_index_path),
        'baseline_provenance': BASELINE,
        'cases': cases,
        'baseline_comparison': comparison,
        'activity': {
            'model_calls': 0,
            'physics_calls': 0,
            'simulator_runs': 0,
            'training_steps': 0,
            'gui_sessions': 0,
            'files_modified_outside_this_directory': 0,
            'old_sampling_mode': 'legacy_mixed',
            'new_b22_qualification': False,
            'new_gui_qualification': False,
        },
        'caveats': [
            'Phase durations are saved per-tick labels at the saved control dt; phase_before and'
            ' phase_after differ by one tick of labelling and neither resolves sub-tick timing'
            ' inside the controller.',
            'legacy_mixed sampling keeps the previous final physics substep in the saved'
            ' measurements; they are not synchronized measurements.',
            'Nothing here establishes a physical cause for the observed timing or for the initial'
            ' reverse excursion; only saved values are reported.',
            'The 4 s handoff window is saved observation, not an enforced wait, and retention is'
            ' only claimed for that saved window.',
            'No new B22, GUI or policy qualification is implied; this module runs no model.',
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description='Bounded diagnostic of saved D1 side-step records.')
    parser.add_argument('--inputs', default='../saved_input_27',
                        help='directory holding restore_receipt.json and the saved case folders')
    parser.add_argument('--output', required=True, help='new JSON report path (must not exist)')
    parser.add_argument('--evidence-index', default=str(EVIDENCE_INDEX),
                        help='public evidence index used for original SHA/bytes verification')
    args = parser.parse_args(argv)

    output = Path(args.output)
    if output.exists():
        raise SystemExit(f'refusing to overwrite existing output {output}')
    report = build_report(args.inputs, Path(args.evidence_index))
    with output.open('x') as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'schema': SCHEMA, 'output': str(output),
                      'net_signed_body_lateral_m': {
                          name: case['displacement']['net_signed_body_lateral_m']
                          for name, case in report['cases'].items()},
                      'net_signed_lateral_speed_mps': {
                          name: case['speeds']['net_signed_lateral_speed_mps']
                          for name, case in report['cases'].items()}},
                     sort_keys=True, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
