"""Independent saved-only GUI23 qualification reader.

Only stdlib/NumPy and separately frozen pure readers are admissible. Runtime
worker completion booleans are never a replacement for saved numerical joins.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.abc
import json
import math
import os
from pathlib import Path
import sys
import struct
import zlib
from collections import Counter

FORBIDDEN = {"mujoco", "glfw", "torch", "stable_baselines3", "gym", "gymnasium",
             "engine_binding", "wheel_legged_control"}


class NoExecution(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.partition(".")[0] in FORBIDDEN:
            raise RuntimeError("GUI saved reader forbids execution import: " + fullname)


if any(n.partition(".")[0] in FORBIDDEN for n in sys.modules):
    raise RuntimeError("GUI reader was not launched cold")
sys.meta_path.insert(0, NoExecution())
sys.dont_write_bytecode = True
import numpy as np

W = Path(__file__).resolve().parents[2]
for directory in (W, W/"course_impl08", W/"rl11", W/"continuation18"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
from verify_control18 import check_controller
from verify_course_e_08_03 import (check_binding, check_contact, _body_com_velocity,
                                 _roll_pitch_deg, FAMILIES, family)
from verify_rl16_training_08 import _ground_hit

MODEL_SHA = "7bd58b5d6114745b12b4284be3f861f973a5d31d9d596365a5b54cb5e0d76691"
ARCHIVE = "d1-archive-transaction-13-v1"
STATE_KEYS = ("qpos", "qvel", "ctrl", "qacc_warmstart", "observation", "time")
PROFILE = {"flat_0p6":(.6,0.), "flat_1p6":(1.6,0.), "yaw_1p2":(1.2,.3),
           "bumps_0p4":(.4,0.), "rough_0p35":(.35,0.), "ramp_0p45_complete":(.45,0.)}
KEYS = {"w", "q", "e", "r", "space", "x", "escape"}


def require(value, message):
    if not value:
        raise ValueError(message)


def identity(path):
    h = hashlib.sha256()
    n = 0
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1 << 20), b""):
            h.update(block)
            n += len(block)
    return {"bytes":n, "sha256":h.hexdigest()}


def reject_constant(value):
    raise ValueError("nonfinite JSON constant: " + value)


def document(path):
    value = json.loads(Path(path).read_text(), parse_constant=reject_constant)
    require(isinstance(value, dict), "JSON document must be an object: " + str(path))
    return value


def committed(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "missing/symlink payload: " + str(path))
    manifest = document(path.with_name(path.name + ".manifest.json"))
    items = manifest.get("payloads")
    require(manifest.get("schema") == ARCHIVE and isinstance(items, list) and items,
            "missing atomic transaction members")
    names = [r.get("file") for r in items]
    require(len(names) == len(set(names)) and path.name in names, "duplicate/missing transaction payload")
    for row in items:
        name = row["file"]
        require(isinstance(name, str) and Path(name).name == name, "unsafe transaction payload name")
        file = path.parent/name
        require(file.is_file() and not file.is_symlink()
                and identity(file) == {"bytes":row["bytes"], "sha256":row["sha256"]},
                "atomic payload differs: " + str(file))
    return manifest


def saved_document(path):
    committed(path)
    return document(path)


def saved_value(path):
    committed(path)
    return json.loads(Path(path).read_text(), parse_constant=reject_constant)


def saved_rows(path):
    committed(path)
    with gzip.open(path, "rt") as stream:
        for line in stream:
            value = json.loads(line, parse_constant=reject_constant)
            require(isinstance(value, dict), "nonobject archive row")
            yield value


def saved_arrays(path):
    committed(path)
    with np.load(path, allow_pickle=False) as archive:
        require(len(archive.files) == len(set(archive.files)), "duplicate NPZ member")
        result = {k:archive[k].copy() for k in archive.files}
    require(all(a.dtype.kind in "fiub" and np.isfinite(a).all() for a in result.values()),
            "nonfinite/non-numeric NPZ member")
    return result


def byte_equal(first, second):
    a, b = np.asarray(first), np.asarray(second)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def close(first, second, tolerance=1e-10, *, atol=None):
    if atol is not None:
        tolerance = atol
    a, b = np.asarray(first), np.asarray(second)
    return a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all() and np.allclose(a,b,rtol=0,atol=tolerance)


def equal(first, second):
    return np.array_equal(np.asarray(first), np.asarray(second))


def percentile95(values):
    require(bool(values), "empty performance window")
    return sorted(values)[max(0, math.ceil(.95*len(values))-1)]


def input_replay(profile, snapshots):
    """Independent arithmetic of input authority; receives actual consumed snapshots.

    Provenance/merge joins to the main-thread publication log are checked by the
    caller. No production InputController or its output booleans are imported.
    """
    require(profile in PROFILE, "unknown fixed input profile")
    prior = None
    owner_time = -1
    disarmed = reset_used = exited = False
    results = []
    for record in snapshots:
        s, now = record["snapshot"], record["now_ns"]
        require(type(now) is int and isinstance(s,dict), "bad owner input clock/snapshot")
        seq, stamp = s["sequence"], s["timestamp_ns"]
        held, events = set(s["held"]), s["events"]
        require(type(seq) is int and seq >= 0 and type(stamp) is int and stamp >= 0,
                "bad snapshot identity/clock")
        require(type(s["focused"]) is bool and type(s["closed"]) is bool
                and type(s["events_truncated"]) is bool, "bad snapshot boolean")
        require(len(held) == len(s["held"]) and all(isinstance(k,str) for k in held),
                "bad held-key list")
        require(isinstance(events,list) and all(isinstance(e,dict) for e in events),
                "bad saved event list")
        for event in events:
            require(event["kind"] in ("press","release","focus_lost","focus_gained","close"),
                    "unknown saved input event")
        repeat = prior is not None and seq == prior["sequence"]
        result = dict(forward_mps=0.,yaw_rps=0.,reason=None,consumed_sequence=None,
                      reset_requested=False,exit_requested=exited,reused_snapshot=False,
                      clearance_m=.455,lateral_mps=0.,jump_requested=False)
        closing = s["closed"] or "escape" in held or any(
            e["kind"] == "close" or e["kind"] == "press" and e.get("key") == "escape"
            for e in events)
        if closing:
            exited = disarmed = True
            result.update(reason="exit_requested",exit_requested=True)
        elif exited:
            result.update(reason="exit_latched",exit_requested=True)
        elif now < 0 or now < owner_time:
            disarmed = True
            result["reason"] = "backward_owner_clock"
        else:
            owner_time = now
            invalid = None
            if stamp > now:
                invalid = "future_snapshot"
            elif now-stamp > 250_000_000:
                invalid = "stale_snapshot"
            elif prior is not None and seq < prior["sequence"]:
                invalid = "reordered_snapshot"
            elif repeat and s != prior:
                invalid = "same_sequence_changed"
            elif not repeat and prior is not None and stamp < prior["timestamp_ns"]:
                invalid = "backward_snapshot_clock"
            if invalid:
                disarmed = True
                result["reason"] = invalid
            else:
                previous = prior
                if not repeat:
                    prior = s
                result.update(consumed_sequence=seq,reused_snapshot=repeat)
                new_events = [] if repeat else events
                if s["events_truncated"] or not s["focused"] or any(e["kind"] == "focus_lost" for e in new_events):
                    disarmed = True
                    result["reason"] = "event_overflow" if s["events_truncated"] else "focus_lost"
                elif held-KEYS or any(e.get("key") not in KEYS for e in new_events if e["kind"] in ("press","release")):
                    disarmed = True
                    result["reason"] = "unsupported_key"
                elif any(e["kind"] == "press" and e.get("key") in ("space","x") for e in new_events) or held & {"space","x"}:
                    disarmed = True
                    result["reason"] = "operator_stop"
                elif not repeat and (any(e["kind"] == "press" and e.get("key") == "r" for e in new_events)
                        or "r" in held and (previous is None or "r" not in previous["held"])):
                    disarmed = True
                    result.update(reason="reset_limit" if reset_used else "simulation_reset_request",
                                  reset_requested=not reset_used)
                    reset_used = True
                elif "w" not in held:
                    if not repeat:
                        disarmed = False
                    result["reason"] = "released"
                elif disarmed:
                    result["reason"] = "release_w_to_rearm"
                else:
                    vx, yaw = PROFILE[profile]
                    result.update(forward_mps=vx,yaw_rps=yaw*(int("q" in held)-int("e" in held)),
                                  reason="held_command")
        results.append(result)
    return results


def check_performance_data(run, controls, start_ns, end_ns, polls, frames, snapshot_sources):
    """Recompute declared windows and cross-link every displayed frame to a producer."""
    require(type(start_ns) is int and type(end_ns) is int and start_ns < end_ns,
            "missing active render time boundary")
    require(0 < controls <= 1200, "GUI control count outside qualification contract")
    require(isinstance(polls,list) and polls == sorted(polls)
            and all(type(t) is int for t in polls), "invalid actual poll clock")
    require(isinstance(frames,list) and 0 < len(frames) <= 1000, "invalid render frame count")
    duration = (end_ns-start_ns)/1e9
    rtf = .01*controls/duration
    active_polls = [t for t in polls if start_ns <= t <= end_ns]
    gaps = [b-a for a,b in zip(active_polls,active_polls[1:])]
    sources = {(r["control_index"],r["source_wall_ns"]):r for r in snapshot_sources}
    require(len(sources) == len(snapshot_sources), "duplicate published frame identity")
    draws = []
    previous_time = -1
    captures = {}
    for frame in frames:
        require(type(frame.get("wall_ns")) is int and frame["wall_ns"] >= previous_time,
                "render clock order differs")
        previous_time = frame["wall_ns"]
        if frame["status"] not in ("rendered","reused"):
            continue
        require(type(frame.get("snapshot_age_ns")) is int and frame["snapshot_age_ns"] >= 0,
                "invalid displayed snapshot age")
        source_time = frame["source_wall_ns"]
        key = (frame["control_index"], source_time)
        require(key in sources and 0 <= key[0] <= controls,
                "rendered frame not bound to actual owner publication")
        source = sources[key]
        publication = source['publication']
        require(publication['status'] == 'published'
                and publication['frame_seq'] == frame['frame_seq']
                and publication['wall_ns'] == frame['published_wall_ns']
                and source_time <= publication['wall_ns'] <= frame['render_return_ns']
                and frame['wall_ns'] <= frame['render_return_ns']
                and max(0,frame['wall_ns']-publication['wall_ns']) <= frame['snapshot_age_ns']
                <= frame['render_return_ns']-publication['wall_ns'],
                'snapshot age not bounded by actual publication/render clocks')
        if start_ns <= frame["wall_ns"] <= end_ns:
            draws.append(frame)
        capture = frame.get("capture")
        if capture is not None:
            require(isinstance(capture,dict), "invalid capture identity")
            name = Path(capture["path"]).name
            require(name not in captures and name in ("frame_initial.png","frame_drive.png","frame_final.png"),
                    "unexpected/repeated screenshot")
            image_path = Path(run)/name
            require(str(image_path) == capture["path"] and identity(image_path) ==
                    {"bytes":capture["bytes"],"sha256":capture["png_sha256"]},
                    "screenshot payload identity differs")
            pixels = png_rgb(image_path)
            require(pixels.shape == (500,800,3) and float(pixels.std()) > 2
                    and hashlib.sha256(pixels[::-1].tobytes()).hexdigest()
                    == capture["pixels_sha256"], "screenshot raster is empty or differs")
            captures[name] = frame
    require(set(captures) == {"frame_initial.png","frame_drive.png","frame_final.png"},
            "three GUI screenshots required")
    require(captures["frame_initial.png"]["control_index"] == 0
            and captures["frame_final.png"]["control_index"] == controls,
            "initial/final screenshot indices differ")
    require(0 < captures["frame_drive.png"]["control_index"] < controls,
            "moving screenshot has no executed controls")
    for boundary in (0,controls):
        require(sum(f.get("status") in ("rendered","reused") and f.get("control_index") == boundary
                    for f in frames) >= 2, "missing stable no-control display pair")
    require(len(gaps) >= 8 and bool(draws), "insufficient actual performance samples")
    ages = [r["snapshot_age_ns"] for r in draws]
    fps = len(draws)/duration
    p95_poll, p95_age = percentile95(gaps),percentile95(ages)
    return {"passed":rtf >= .8 and fps >= 8 and p95_poll <= 250_000_000 and p95_age <= 250_000_000,
            "active_rtf":rtf,"draw_fps":fps,"drawn_active_frames":len(draws),
            "poll_interval_p95_ns":p95_poll,"snapshot_age_p95_ns":p95_age,
            "captures_verified":sorted(captures)}


def check_input_publication_links(raw_rows, consumed_rows):
    """Recover lossless pending edges between actual main-thread publications."""
    published = {}
    last_sequence = -1
    for row in raw_rows:
        s = row["snapshot"]
        seq = s["sequence"]
        require(type(seq) is int and seq > last_sequence, "poll sequence not strictly increasing")
        last_sequence = seq
        require(type(row["published"]) is bool, "publication flag missing")
        if row["published"]:
            require(type(row["publish_wall_ns"]) is int
                    and s['timestamp_ns'] <= row['publish_begin_ns'] <= row['publish_wall_ns'],
                    "publication precedes actual poll")
            published[seq] = row
    previous = -1
    previous_snapshot = None
    for row in consumed_rows:
        s = row["snapshot"]
        seq = s["sequence"]
        require(seq in published, "owner consumed snapshot absent from actual publication")
        raw = published[seq]
        require(raw["publish_begin_ns"] <= row["now_ns"], "owner consumed future publication")
        for key in ("sequence","timestamp_ns","held","focused","closed"):
            require(s[key] == raw["snapshot"][key], "delivered snapshot changed main-thread state: " + key)
        if seq == previous:
            require(s == previous_snapshot, "repeated delivered snapshot mutated")
            continue
        require(seq > previous, "owner snapshot order went backwards")
        pending = [e for k in sorted(published) if previous < k <= seq
                   for e in published[k]["snapshot"]["events"]]
        truncated = any(published[k]["snapshot"]["events_truncated"]
                        for k in published if previous < k <= seq)
        require(not truncated and s["events_truncated"] is False,
                "input event overflow cannot qualify usable GUI")
        require(s["events"] == pending, "owner lost or invented pending keyboard/focus events")
        previous,previous_snapshot = seq,s
    return {"published_snapshot_count":len(published),"owner_consumptions":len(consumed_rows),
            "pending_edges_preserved":True}


def png_rgb(path: Path) -> np.ndarray:
    """Decode the fixed RGB8 Pillow capture without adding an image dependency."""
    data = path.read_bytes()
    require(data.startswith(b"\x89PNG\r\n\x1a\n"), f"PNG signature missing: {path}")
    offset, width, height, payload = 8, None, None, bytearray()
    while offset + 12 <= len(data):
        length = struct.unpack_from(">I", data, offset)[0]
        tag = data[offset + 4:offset + 8]
        end = offset + 12 + length
        require(end <= len(data), f"PNG chunk truncated: {path}")
        body = data[offset + 8:end - 4]
        require(zlib.crc32(tag + body) == struct.unpack_from(">I", data, end - 4)[0],
             f"PNG chunk CRC differs: {path}")
        if tag == b"IHDR":
            width, height, bits, colour, compression, filtering, interlace = struct.unpack(
                ">IIBBBBB", body)
            require(0 < width <= 4096 and 0 < height <= 4096 and bits == 8
                 and colour == 2 and compression == filtering == interlace == 0,
                 f"PNG RGB8 dimensions/format differ: {path}")
        elif tag == b"IDAT":
            payload.extend(body)
        elif tag == b"IEND":
            require(end == len(data), f"PNG trailing bytes: {path}")
            break
        offset = end
    require(width is not None and height is not None and bool(payload), f"PNG data missing: {path}")
    raw = zlib.decompress(payload)
    stride = width * 3
    require(len(raw) == height * (stride + 1), f"PNG raster length differs: {path}")
    image = np.empty((height, stride), dtype=np.uint8)
    for row in range(height):
        start = row * (stride + 1)
        method = raw[start]
        require(method in (0, 1, 2, 3, 4), f"PNG filter invalid: {path}")
        source = raw[start + 1:start + 1 + stride]
        for col, value in enumerate(source):
            left = int(image[row, col - 3]) if col >= 3 else 0
            above = int(image[row - 1, col]) if row else 0
            upper_left = int(image[row - 1, col - 3]) if row and col >= 3 else 0
            if method == 0:
                predictor = 0
            elif method == 1:
                predictor = left
            elif method == 2:
                predictor = above
            elif method == 3:
                predictor = (left + above) // 2
            else:
                p = left + above - upper_left
                distances = (abs(p - left), abs(p - above), abs(p - upper_left))
                predictor = (left, above, upper_left)[distances.index(min(distances))]
            image[row, col] = (value + predictor) & 255
    return image.reshape(height, width, 3)


def check_native(row, five, calc, binding, geometry, native_start):
    info = row["info"]
    traces = row["native_actuator_traces"]
    interval = info["native_interval_summary"]
    tick = row["episode_tick"]
    require(len(five) == len(traces) == interval["native_returns"] == 5
            and close(interval["start_time_s"], tick*.01)
            and close(interval["end_time_s"], (tick+1)*.01),
            "training control lacks five native returns")
    nonwheel = 0
    families = Counter()
    positive = set()
    pre, post = row["pre_state"], row["post_state"]
    for substep, (native, trace) in enumerate(zip(five, traces)):
        require(native["native_index"] == native_start+substep
                and close(native["start_time_s"], tick*.01+substep*.002)
                and close(native["end_time_s"], tick*.01+(substep+1)*.002)
                and native["contact_count"] == len(native["contacts"]),
                "training native index/clock differs")
        for key in ("qpos", "qvel", "qacc_warmstart"):
            expected_before = pre[key] if substep == 0 else five[substep-1]["after"][key]
            require(equal(native["before"][key], expected_before),
                    "training native integrator before-state differs")
            if substep == 4:
                require(equal(native["after"][key], post[key]),
                        "training native integrator endpoint differs")
        require(equal(native["before"]["ctrl"], trace["applied_nm"])
                and equal(native["after"]["ctrl"], trace["applied_nm"])
                and close(native["after"]["actuator_force"], trace["applied_nm"])
                and all(close(trace[name], calc["computed"]["safe_torque_nm"])
                        for name in ("requested_nm", "limited_nm", "delayed_nm", "applied_nm")),
                "training native actuator torque chain differs")
        roll, pitch = _roll_pitch_deg(native["after"]["qpos"])
        require(close((roll, pitch), (native["roll_deg"], native["pitch_deg"])),
                "training native posture differs")
        per_native = Counter()
        candidates = 0
        for contact in native["contacts"]:
            loaded, family = check_contact(contact, geometry, binding)
            if family is not None:
                candidate = contact["robot_wheel_index"] is None
                require(contact["nonwheel_ground_candidate"] is candidate
                        and contact["nonwheel_ground_contact"] is candidate
                        and contact["nonwheel_active_solver_contact"] is (
                            candidate and contact["efc_address"] >= 0),
                        "training contact solver classification differs")
            candidates += bool(contact["nonwheel_ground_contact"])
            if loaded:
                per_native[family] += 1
                positive.add(family)
        require(candidates == native["nonwheel_contact_count"]
                and dict(per_native) == native["terrain_family_positive_wheel_load"],
                "training native contact/force summary differs")
        nonwheel += candidates
        families.update(per_native)
    require(nonwheel == interval["nonwheel_contact_count"]
            and dict(families) == interval["terrain_family_positive_wheel_load"],
            "training five-native interval summary differs")
    return positive


def geometry_map(manifest, binding):
    rows = manifest['world_collision_geoms']
    require(len(rows) == 92 and manifest['control_dt_s'] == .01
            and manifest['native_dt_s'] == .002
            and Counter(family(r['name']) for r in rows) == FAMILIES,
            'actual compiled course layout/timing differs')
    mapping = {}
    kinds = {'plane':set(),'box':set()}
    for row in rows:
        gid = row['geom_id']
        require(type(gid) is int and gid not in mapping
                and 0 <= gid < len(binding['geom_bodyid'])
                and row['body_id'] == binding['geom_bodyid'][gid] == 0
                and row['collision'] is True
                and (binding['geom_contype'][gid] != 0 or binding['geom_conaffinity'][gid] != 0)
                and close(binding['geom_size'][gid],row['size_m'])
                and row['type'] == ('plane' if row['name'] == 'floor' else 'box'),
                'compiled course terrain identity differs')
        q = np.asarray(row['quaternion_wxyz'],dtype=float)
        require(q.shape == (4,) and close(q@q,1.,atol=1e-12)
                and np.isfinite(row['position_m']).all(), 'invalid terrain transform')
        kinds[row['type']].add(int(binding['geom_type'][gid]))
        mapping[gid] = row
    require(len(kinds['plane']) == len(kinds['box']) == 1
            and kinds['plane'] != kinds['box'], 'compiled plane/box types differ')
    return mapping


def check_reset(control, initial):
    require(set(initial) == set(STATE_KEYS)
            and initial['qpos'].shape == (23,) and initial['qvel'].shape == (22,)
            and initial['ctrl'].shape == (16,) and initial['observation'].shape == (99,)
            and initial['observation'].dtype == np.dtype('float32')
            and equal(initial['observation'][59:79],np.zeros(20,dtype=np.float32))
            and close(initial['time'],0.)
            and control['variant'] == 'combined'
            and equal(control['wheel_integral_nm'],[0.]*4)
            and control['wheel_common_reference_z_rad_s'] == 0.
            and control['previous_servo_forward_mps'] is None
            and control['stop_latched'] is False
            and control['servo_last_tick'] == 0
            and control['provider_sequence'] == 0
            and control['provider_control_time_s'] == 0.
            and equal(control['previous_action'],np.zeros(16)),
            'actual initial five-state/control hidden state differs')
    applied = control['servo_applied']
    require(applied['forward_velocity_mps'] == applied['lateral_velocity_mps']
            == applied['yaw_rate_rps'] == 0. and applied['clearance_m'] == .455
            and applied['jump_requested'] is False, 'reset prepared command not zero')


def check_episode(run, index, offset, session, binding, geometry):
    folder = run/f'episode_{index}'
    reset = saved_document(folder/'reset.json')
    initial = saved_arrays(folder/'initial_state.npz')
    states = saved_arrays(folder/'states.npz')
    receipt = saved_document(folder/'segment_receipt.json')
    rows = list(saved_rows(folder/'controls.jsonl.gz'))
    n = len(rows)
    require(1 <= n <= session['control_limit']-offset
            and receipt['episode_index'] == index and receipt['completed_controls'] == n
            and reset['seed'] == session['seed']
            and reset['model_address'] > 0 and reset['data_address'] > 0,
            'episode identity/control count differs')
    check_reset(reset['control_reset_state'],initial)
    require(set(states) == set(STATE_KEYS) and all(
        states[k].shape == (n+1,*initial[k].shape)
        and byte_equal(states[k][0],initial[k]) for k in STATE_KEYS),
        'endpoint array/initial bytes differ')
    require(close(states['time'],np.arange(n+1)*.01), 'episode control clock differs')
    for section,fields in (('C_state',('control_attempts','control_returns',
                                      'construction_attempts','construction_returns')),
                           ('python',('control_attempted','control_completed',
                                      'native_attempted','native_returned'))):
        require(all(reset['before'][section][k] == reset['after'][section][k] for k in fields),
                'reset altered/refunded physical ledger')
    require(reset['before']['python']['control_completed'] == offset
            and reset['before']['python']['native_returned'] == 5*offset,
            'reset global ledger differs')
    guard = receipt['native_segment']
    require(guard['mode'] == 'heldout' and guard['record_valid'] is True
            and guard['native_attempted'] == guard['native_returned'] == 5*n
            and guard['failure'] is None and guard['archive_failure'] is None
            and guard['partial_native_interval'] is False
            and guard['force_sampling_performed'] is True
            and guard['full_contact_qualification_recorded'] is True,
            'episode complete native sampling proof differs')
    names = guard['native_files']
    require(len(names) == len(set(names)) and guard['archive_block_manifests']
            == [name+'.manifest.json' for name in names], 'native archive list differs')
    native = []
    for name in names:
        require(Path(name).name == name, 'unsafe native archive name')
        native.extend(saved_rows(folder/name))
    require(len(native) == 5*n, 'native archive count differs')
    servo_vx = servo_yaw = z = 0.
    pi = np.zeros(4)
    prior_servo = None
    stop_latched = False
    action_nonzero = False
    safety = True
    for tick,row in enumerate(rows):
        info = row['info']
        require(row['control_index'] == offset+tick and row['episode_index'] == index
                and row['episode_tick'] == tick and row['actor'] == session['actor']
                and row['checkpoint_sha256'] == (MODEL_SHA if session['actor'] == 'B' else None)
                and row['policy_predict_called'] is (session['actor'] == 'B')
                and row['control_stages18'] == info['controller_record']
                and info['completed_control_intervals'] == tick+1,
                'control identity/prepared-consumed index differs')
        for key in STATE_KEYS:
            require(equal(row['pre_state'][key],states[key][tick])
                    and equal(row['post_state'][key],states[key][tick+1]),
                    'complete state/control join differs: '+key)
        require(equal(row['input_observation99'],states['observation'][tick])
                and equal(row['raw_action16'],row['action16'])
                and equal(row['action16'],row['policy_input_action']),
                '99D/16D policy archive join differs')
        action = np.asarray(row['action16'])
        require(action.shape == (16,) and np.isfinite(action).all(), 'invalid policy action')
        if session['actor'] == 'zero':
            require(equal(action,np.zeros(16)), 'zero actor output not zero')
        action_nonzero |= bool(np.any(np.asarray(info['applied_action']) != 0))
        raw = row['raw_command']
        require(raw == info['raw_operator_command'] and raw['lateral_velocity_mps'] == 0.
                and raw['clearance_m'] == .455 and raw['jump_requested'] is False,
                'prepared raw command changed before consumption')
        if session['mode'] == 'script':
            require(raw['forward_velocity_mps'] == (.6 if 175 <= offset+tick < 425 else 0.)
                    and raw['yaw_rate_rps'] == 0., 'qualification script changed')
        servo_vx += float(np.clip(raw['forward_velocity_mps']-servo_vx,-.005,.005))
        servo_yaw += float(np.clip(raw['yaw_rate_rps']-servo_yaw,-.006,.006))
        command = info['consumed_command']
        require(close(command['forward_velocity_mps'],servo_vx)
                and close(command['yaw_rate_rps'],servo_yaw)
                and command['lateral_velocity_mps'] == 0.
                and command['clearance_m'] == .455 and command['jump_requested'] is False,
                'independent command servo arithmetic differs')
        adapter = info['controller_record']
        after_stop = False if servo_vx != 0 else (
            True if prior_servo is not None and prior_servo != 0 else stop_latched)
        require(adapter['mode'] == 'eval' and adapter['controller_variant'] == 'combined'
                and adapter['test_actor_probe'] is False
                and adapter['previous_servo_forward_mps'] == prior_servo
                and adapter['stop_latched_before'] is stop_latched
                and adapter['stop_latched_after'] is after_stop,
                'C18 stop latch/previous servo chain differs')
        calc = check_controller(row,offset+tick,pre_qpos=states['qpos'][tick],
            pre_qvel=states['qvel'][tick],
            pre_body_forward_mps=float(_body_com_velocity(states['qpos'][tick],states['qvel'][tick],binding)[0]),
            servo_yaw_rps=servo_yaw)
        actual = calc['calculation']
        require(close(actual['wheel_integral_before_nm'],pi)
                and close(actual['wheel_common_reference_z_before_rad_s'],z)
                and equal(actual['consumed_joint_position_rad'],states['qpos'][tick][binding['qpos_addresses']])
                and equal(actual['consumed_joint_velocity_rad_s'],states['qvel'][tick][binding['dof_addresses']]),
                'C18 hidden state/current joint state differs')
        pi = np.asarray(calc['computed']['wheel_integral_after_nm'])
        z = calc['computed']['wheel_common_reference_z_after_rad_s']
        prior_servo, stop_latched = servo_vx, after_stop
        five = native[5*tick:5*tick+5]
        check_native(row,five,calc,binding,geometry,5*(offset+tick))
        for step in five:
            q = step['after']['qpos']
            ground,_,_ = _ground_hit(geometry,q[0],q[1])
            safety &= (abs(step['roll_deg']) <= 10 and abs(step['pitch_deg']) <= 10
                       and q[2]-ground >= .28 and step['nonwheel_contact_count'] == 0
                       and abs(q[0]) <= 11 and abs(q[1]) <= 6)
        q,v = states['qpos'][tick+1],states['qvel'][tick+1]
        metrics = info['metrics']
        ground,_,gid = _ground_hit(geometry,q[0],q[1])
        require(close(metrics['body_com_vx_mps'],_body_com_velocity(q,v,binding)[0],atol=1e-6)
                and close(metrics['body_yaw_rate_rps'],v[5],atol=1e-6)
                and close(metrics['clearance_m'],q[2]-ground,atol=1e-6)
                and metrics['ground_geom_id'] == gid
                and close(row['reward'],sum(info['reward_terms'].values())),
                'saved endpoint metrics/reward-sum linkage differs')
        require(row['terminated'] is False, 'GUI encountered a physical task failure')
    require(receipt['boundary']['python']['control_completed'] == offset+n
            and receipt['boundary']['python']['native_returned'] == 5*(offset+n),
            'episode final ledger differs')
    return {'index':index,'controls':n,'initial':initial,'states':states,'reset':reset,
            'rows':rows,'native':native,'guard':guard,'safety_passed':bool(safety),
            'nonzero_effective_residual':action_nonzero}


def check_ledgers(result, session, controls):
    c,p,g = result['C_final'],result['python'],result['native_guard']
    require(c['construction_attempts'] == c['construction_returns'] == 2
            and c['control_attempts'] == c['control_returns'] == 5*controls
            and c['phase'] == c['target_model'] == c['target_data'] == 0
            and c['violations'] == 0 and c['native_construction_caller_verified'] is True
            and c['ccd_attempts'] == c['ccd_returns']
            and (c['ccd_attempts'] == 0 or c['native_ccd_caller_verified'] is True)
            and p['control_attempted'] == p['control_completed'] == controls
            and p['native_attempted'] == p['native_returned'] == p['clock_advanced_substeps'] == 5*controls
            and p['forbidden_entries'] == 0 and p['fatal_latched'] is False
            and g['native_attempted'] == g['native_returned'] == g['native_checked'] == 5*controls
            and g['failure'] is None and result['thread_violations'] == []
            and result['warnings'] == [], 'actual C/Python/native/thread ledger differs')
    calls = result['model_calls']
    counts = calls['counts']
    actor_b = session['actor'] == 'B'
    expected = dict(load=int(actor_b),torch_load=3*int(actor_b),predict=(controls+1)*int(actor_b),
                    save=0,learn=0,train=0,forward=0,evaluate_actions=0,predict_values=0,backward=0)
    require(calls['limits'] == session['model_limits']
            and all(counts.get(k,{}).get('attempted',0) == counts.get(k,{}).get('returned',0) == v
                    for k,v in expected.items())
            and result['policy_predictions'] == controls*int(actor_b)
            and counts.get('predict',{}).get('rows_attempted',0)
            == counts.get('predict',{}).get('rows_returned',0) == (controls+32)*int(actor_b)
            and calls['actor_rows'] == (controls+32)*int(actor_b) and calls['critic_rows'] == 0,
            'fixed model API/probe/control rows differ')
    edges,mailbox = result['copy_edges'],result['mailbox']
    require(edges['rejected'] == 0 and edges['live_to_measurement'] >= controls
            and edges['measurement_to_writing'] == mailbox['published']
            and edges['reading_to_display'] == mailbox['acquired'] == mailbox['released']
            and mailbox['copy_failed'] == 0, 'exclusive render copy/lease ledger differs')
    require(type(result['owner_thread_ident']) is int and type(result['main_thread_ident']) is int
            and result['owner_thread_ident'] != result['main_thread_ident'], 'owner/main thread identity differs')
    addresses = result['data_addresses']
    address_values = [addresses['live'],addresses['measurement'],addresses['display'],*addresses['slots']]
    require(len(address_values) == len(set(address_values)) == 6
            and all(type(v) is int and v > 0 for v in address_values), 'six MjData identities not distinct')


def check_sources(run, session, worker):
    require(session['execution_contract_id'] == 'C23_B22_GUI_v1'
            and session['schema'] == 'd1-c23-gui-worker-v1'
            and session['qualification'] is True and session['retry_permitted'] is False
            and session['actor'] == 'B' and session['controller_variant'] == 'combined'
            and worker['schema'] == 'd1-c23-gui-worker-receipt-v1'
            and worker['execution_contract_id'] == session['execution_contract_id']
            and worker['arm'] == session['arm'] and worker['session_identity'] == identity(run/'session.json')
            and worker['failure'] is None and worker['warnings'] == []
            and worker['archive_failed'] is False and worker['writer_partial'] == [],
            'worker/session identity or archive failure differs')
    arm_spec = {'script_headless':('script',False,'flat_0p6',600,231001),
                'script_gui':('script',True,'flat_0p6',600,231001),
                'keyboard_gui':('keyboard',True,'yaw_1p2',1200,231002)}
    require(session['arm'] in arm_spec and tuple(session[k] for k in
            ('mode','render','profile','control_limit','seed')) == arm_spec[session['arm']],
            'fixed qualification arm differs')
    host,supervisor = document(run/'host_receipt.json'),document(run/'supervisor_receipt.json')
    require(host['arm'] == session['arm'] and host['exit_code'] == 0
            and host['failure'] is None and host['source_mismatches'] == []
            and host['owned_no_orphans'] is True and host['reservation_closed'] is True
            and host['reserved_controls'] == session['control_limit'] and host['retry_permitted'] is False
            and supervisor['exit_code'] == 0 and supervisor['failure'] is None
            and supervisor['cleanup']['remaining'] == {} and supervisor['outer_limit_s'] == 600
            and supervisor['elapsed_s'] <= 600, 'host/supervisor has not closed successfully')
    go_path = Path(session['source_go_path'])
    go = document(go_path)
    require(identity(go_path) == session['source_go_identity'] and go['decision'] == 'GO'
            and all(session.get(k) == v for k,v in go['arms'][session['arm']].items()),
            'actual session differs from source GO')
    sources = session['source_hashes']
    require(all(sources.get(k) == v for k,v in go['inputs'].items()), 'GO closure missing from session')
    for name,expected in sources.items():
        require(identity(Path(name)) == expected, 'frozen source/payload changed: '+name)
    for path in (Path(__file__).resolve(),Path(__file__).resolve().parents[1]/'gui23/run_gui23.py',
                 Path(__file__).resolve().parents[1]/'gui23/input23.py'):
        require(str(path) in sources, 'critical new GUI source absent from frozen closure')
    model = Path(session['checkpoint_folder'])/'final_model.zip'
    manifest,relocation = session['checkpoint_manifest'],session['checkpoint_relocation']
    original_path = Path(relocation['original_manifest_path'])
    original = document(original_path)
    require(identity(original_path) == relocation['original_manifest_identity'] == sources.get(str(original_path))
            and relocation['changed_fields'] == ['folder']
            and original['folder'] == relocation['original_folder']
            and relocation['current_folder'] == session['checkpoint_folder']
            and manifest == {**original,'folder':session['checkpoint_folder']}
            and identity(model) == manifest['files']['final_model.zip']
            and identity(model)['sha256'] == session['checkpoint_sha256'] == MODEL_SHA,
            'B22 ZIP/manifest folder-only relocation differs')
    load = saved_document(run/'strict_load.json')
    require(load['probe_rows'] == 32 and load['probe_actions_byte_exact'] is True
            and load['matches_training_final_policy_state'] is True
            and load['verified_without_engine_or_reset'] is True
            and load['observation_space']['shape'] == [99]
            and load['action_space']['shape'] == [16], 'strict B22 loaded policy/probe identity differs')
    return go


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def pair_runs(first, second):
    require(first['session']['arm'] == 'script_headless' and second['session']['arm'] == 'script_gui'
            and len(first['episodes']) == len(second['episodes']) == 1,
            'same-trajectory pair needs exactly the two fixed script episodes')
    a,b = first['episodes'][0],second['episodes'][0]
    checks = {'initial_'+k:byte_equal(a['initial'][k],b['initial'][k]) for k in STATE_KEYS}
    checks.update({'states_'+k:byte_equal(a['states'][k],b['states'][k]) for k in STATE_KEYS})
    checks['complete_control_reset'] = a['reset']['control_reset_state'] == b['reset']['control_reset_state']
    checks['native_all_fields'] = canonical_digest(a['native']) == canonical_digest(b['native'])
    for key in ('input_observation99','raw_action16','action16','policy_input_action','raw_command',
                'reward','terminated','truncated','info','native_actuator_traces'):
        checks['controls_'+key] = canonical_digest([r[key] for r in a['rows']]) == canonical_digest([r[key] for r in b['rows']])
    require(all(checks.values()), 'headless/GUI complete numerical trajectory differs')
    return {'passed':True,'controls_per_arm':600,'native_per_arm':3000,'checks':checks,
            'interpretation':'same B22 policy, reset state, 99D inputs, 16D actions, complete controller and 5T native records'}


GLFW_KEYS = {87:'w',81:'q',69:'e',82:'r',32:'space',88:'x',256:'escape',65:'a',68:'d'}


def check_raw_poll_mapping(raw_rows):
    pending = []
    overflow = False
    for row in raw_rows:
        poll,sample = row['poll'],row['snapshot']
        events = []
        for event in poll['events']:
            if event['type'] == 'focus':
                events.append({'kind':'focus_gained' if event['focused'] else 'focus_lost','key':None})
            elif event['type'] == 'key' and event['action'] in (0,1):
                events.append({'kind':'press' if event['action'] == 1 else 'release',
                               'key':GLFW_KEYS.get(event['key'],'unsupported')})
        for event in row['raw_button_events']:
            require(event['type'] == 'button'
                    and (event['key'] == 'space' and 650 <= event['x'] <= 790 and 20 <= event['y'] < 60
                         or event['key'] == 'r' and 650 <= event['x'] <= 790 and 60 <= event['y'] < 100),
                    'button callback not in actual visible hit rectangle')
            events.append({'kind':'press','key':event['key']})
        pending.extend(events)
        overflow |= poll['events_truncated']
        require(not overflow and len(pending) <= 4096, 'raw GLFW input overflow cannot qualify')
        require(sample['timestamp_ns'] == poll['wall_ns']
                and sample['held'] == sorted(GLFW_KEYS[k] for k in poll['held_keys'])
                and sample['focused'] is poll['focused'] and sample['closed'] is poll['window_close']
                and sample['events'] == pending and sample['events_truncated'] is False,
                'immutable snapshot does not represent actual GLFW/button events')
        if row['published']:
            pending = []
            overflow = False


def check_prepared_inputs(session, raw_rows, decisions, episodes):
    check_raw_poll_mapping(raw_rows)
    controls = [r for e in episodes for r in e['rows']]
    by_control = {}
    actual_consumptions = [r for r in decisions if r['snapshot'] is not None]
    links = check_input_publication_links(raw_rows,actual_consumptions)
    if session['mode'] == 'keyboard':
        expected = input_replay(session['profile'],actual_consumptions)
        for row,computed in zip(actual_consumptions,expected):
            require(row['decision'] == computed and row['reason'] == computed['reason'],
                    'independent input state-machine decision differs')
    for row in decisions:
        require(type(row['executed']) is bool
                and close(row['prepare_time_s'],row['episode_tick']*.01),
                'prepared tick/time/execution marker differs')
        if row['snapshot'] is None:
            require(row['decision'] is None and row['reason'] in ('bootstrap_zero','no_snapshot','frozen_script'),
                    'invented keyboard authority without a real publication')
            if row['reason'] == 'bootstrap_zero':
                require(row['episode_tick'] == 0, 'bootstrap only at actual reset tick zero')
        if row['executed']:
            index = row['consumed_by_control_index']
            require(index == row['global_control_index'] and index not in by_control
                    and 0 <= index < len(controls), 'prepared command consumed twice/out of order')
            by_control[index] = row
            control = controls[index]
            require(row['episode_index'] == control['episode_index']
                    and row['episode_tick'] == control['episode_tick']
                    and row['raw_command'] == control['raw_command']
                    and control['input_snapshot_sequence']
                    == (None if row['snapshot'] is None else row['snapshot']['sequence']),
                    'prepared input not consumed by its actual observation/control')
            if session['mode'] == 'keyboard':
                decision = row['decision']
                require(row['raw_command']['forward_velocity_mps'] == (0. if decision is None else decision['forward_mps'])
                        and row['raw_command']['yaw_rate_rps'] == (0. if decision is None else decision['yaw_rps'])
                        and (decision is None or not decision['reset_requested'] and not decision['exit_requested']),
                        'keyboard decision/physical command or R/exit boundary differs')
        else:
            require(row['consumed_by_control_index'] is None, 'unexecuted prepared record owns a control')
    require(set(by_control) == set(range(len(controls))), 'some controls lack exactly one preparation')
    resets = [r for r in decisions if r.get('reset_before_control_index') is not None]
    require(len(resets) == len(episodes)-1 and len(resets) <= 1, 'R/reset episode count differs')
    for row,episode in zip(resets,episodes[1:]):
        require(row['executed'] is False and row['decision']['reset_requested'] is True
                and row['reset_before_control_index'] == episode['rows'][0]['control_index']
                and all(byte_equal(episodes[0]['initial'][k],episode['initial'][k]) for k in STATE_KEYS)
                and episodes[0]['reset']['control_reset_state'] == episode['reset']['control_reset_state'],
                'R does not reproduce actual complete physical/control reset')
    return {**links,'controls_with_exact_preparation':len(by_control),'actual_reset_count':len(resets)}


def check_real_events(run, session, go, raw_rows, decisions):
    plan = go['x11_events_by_arm'][session['arm']]
    driver = document(run/'x11_events_receipt.json')
    events = driver['events']
    sent = [r for r in events if r['operation'] != 'initial_focus']
    require(driver['schema'] == 'd1-c23-real-xtest-events-v1' and driver['all_sent'] is True
            and driver['sent_count'] == driver['expected_count'] == len(plan) == len(sent)
            and driver['physical_human_keyboard_tested'] is False,
            'real XTest event plan incomplete')
    for expected,actual in zip(plan,sent):
        require(all(actual[k] == v for k,v in expected.items())
                and actual['observed_completed_controls'] >= expected['tick'],
                'actual OS event differs from preregistered control-index plan')
    # Use actual callback timestamps, not the planned injection tick, as evidence.
    callbacks = []
    for row in raw_rows:
        for event in row['poll']['events']:
            if event['type'] == 'key' and event['action'] in (0,1):
                callbacks.append({'operation':'key_down' if event['action'] == 1 else 'key_up',
                    'key':GLFW_KEYS.get(event['key'],'unsupported'),'wall_ns':event['wall_ns']})
            elif event['type'] == 'focus':
                callbacks.append({'operation':'focus_return' if event['focused'] else 'focus_lost',
                                  'wall_ns':event['wall_ns']})
        callbacks.extend({'operation':'button_click','button':'stop' if e['key'] == 'space' else 'reset',
                          'wall_ns':e['wall_ns']} for e in row['raw_button_events'])
    used = set()
    for event in sent:
        key = event.get('key','').lower()
        key = 'escape' if key == 'escape' else key
        matches = [i for i,c in enumerate(callbacks) if i not in used
                   and c['operation'] == event['operation']
                   and (event['operation'] not in ('key_down','key_up') or c.get('key') == key)
                   and (event['operation'] != 'button_click' or c.get('button') == event['button'])
                   and event['wall_ns']-50_000_000 <= c['wall_ns'] <= event['wall_ns']+1_000_000_000]
        require(bool(matches), 'OS event lacks bounded actual GLFW callback: '+str(event))
        used.add(matches[0])
    computed = [r['decision'] for r in decisions if r['decision'] is not None]
    reasons = {r['reason'] for r in computed}
    required = {'held_command','released','operator_stop','focus_lost','stale_snapshot',
                'release_w_to_rearm','simulation_reset_request','reset_limit','exit_requested'}
    require(required <= reasons and sum(r['reset_requested'] for r in computed) == 1
            and any(r['yaw_rps'] > 0 and r['forward_mps'] == 1.2 for r in computed)
            and any(r['yaw_rps'] < 0 and r['forward_mps'] == 1.2 for r in computed),
            'required real keyboard safety/control path not covered')
    require(any(e['operation'] == 'button_click' and e['button'] == 'stop' for e in sent)
            and any(e['operation'] == 'key_down' and e.get('key','').lower() in ('space','x') for e in sent),
            'both actual button and keyboard Stop required')
    return {'passed':True,'preregistered_events':len(sent),'actual_callbacks_matched':len(used),
            'decision_reasons':sorted(reasons),'physical_human_keyboard_tested':False,
            'event_source':'actual XTest OS input through GLFW callbacks and owner preparation'}


def stop_window(rows):
    tail = rows[-100:]
    raw_servo_zero = len(tail) == 100 and all(
        r['raw_command']['forward_velocity_mps'] == r['raw_command']['yaw_rate_rps'] == 0.
        and r['info']['consumed_command']['forward_velocity_mps'] == 0.
        and r['info']['consumed_command']['yaw_rate_rps'] == 0. for r in tail)
    vx = float(np.mean([abs(r['info']['metrics']['body_com_vx_mps']) for r in tail]))
    yaw = float(np.mean([abs(r['info']['metrics']['body_yaw_rate_rps']) for r in tail]))
    std = float(np.std([r['info']['metrics']['clearance_m'] for r in tail]))
    return {'passed':bool(raw_servo_zero and vx <= .04 and yaw <= .05 and std <= .02),
            'complete_raw_servo_zero_last100':raw_servo_zero,'mean_abs_body_vx_mps':vx,
            'mean_abs_yaw_rate_rps':yaw,'clearance_std_m':std,
            'thresholds':{'mean_abs_body_vx_mps':.04,'mean_abs_yaw_rate_rps':.05,'clearance_std_m':.02}}


def read_run_details(run):
    run = Path(run).resolve(strict=True)
    session = document(run/'session.json')
    worker = saved_document(run/'worker_receipt.json')
    go = check_sources(run,session,worker)
    result = worker['result']
    controls = result['completed_controls']
    require(type(controls) is int and 0 < controls <= session['control_limit'], 'invalid actual controls')
    if session['mode'] == 'script':
        require(controls == 600, 'fixed script did not complete 600 controls')
    construction = saved_document(run/'construction_receipt.json')
    proof = construction['proof']
    require(proof['passed'] is True and proof['dso_path'] == session['library']
            and proof['dso_sha256'] == session['source_hashes'][session['library']]['sha256']
            and len(proof['jump_slots']) == 4 and all(r['passed'] is True for r in proof['jump_slots'])
            and construction['nominal_cache']['misses'] == 1
            and construction['C_state']['construction_attempts'] == construction['C_state']['construction_returns'] == 2,
            'cold two-compiler/engine binding proof differs')
    binding = check_binding(construction['geometry_binding'])
    geometry = geometry_map(construction['compiled_geometry'],binding)
    reference_path = Path(session['reference_construction_path'])
    require(str(reference_path) in session['source_hashes'], 'frozen C22 reference binding absent')
    reference = document(reference_path)
    require(construction['geometry_binding'] == reference['geometry_binding']
            and construction['compiled_geometry'] == reference['compiled_geometry'],
            'GUI changed compiled C22 robot/course geometry')
    episodes = []
    offset = 0
    for index in range(len(result['segment_receipts'])):
        episode = check_episode(run,index,offset,session,binding,geometry)
        require(episode['guard'] == result['segment_receipts'][index], 'episode/worker segment differs')
        offset += episode['controls']
        episodes.append(episode)
    require(offset == controls and 1 <= len(episodes) <= (2 if session['mode'] == 'keyboard' else 1)
            and {p.name for p in run.glob('episode_*') if p.is_dir()}
            == {f'episode_{i}' for i in range(len(episodes))}, 'episode set/global total differs')
    check_ledgers(result,session,controls)
    require(result['control_step_caller_verified'] is True, 'actual control native caller unverified')
    origins = saved_document(run/'loaded_origins_final.json')
    require(all(name in origins['module_origins'] for name in ('torch','numpy','stable_baselines3','gymnasium'))
            and all(path in session['source_hashes'] for path in origins['module_origins'].values())
            and all(row['identity_verified'] is True
                    and row['defining_source'] in session['source_hashes']
                    for row in origins['synthetic_module_aliases'].values()),
            'actual loaded numerical module origins unbound')
    runtime_origins = saved_document(run/'runtime_module_origins.json')
    expected_origins = {
        'run_gui23':W/'continuation23/gui23/run_gui23.py',
        'input23':W/'continuation23/gui23/input23.py',
        'controller18':W/'continuation18/controller18.py',
        'residual18':W/'continuation18/residual18.py',
        'world_upright_course_11':W/'upright11/world_upright_course_11.py',
        'full_drive_env_08':W/'course_impl08/full_drive_env_08.py',
        'gui13_bridge':W/'continuation13/gui13_revision02/gui13_bridge.py',
        'async_course_renderer_12':W/'gui12/async_course_renderer_12.py',
        'latest_frame_mailbox_12':W/'gui12/latest_frame_mailbox_12.py',
        'engine_binding':Path(session['binding_module'])}
    require(runtime_origins == {k:str(v.resolve()) for k,v in expected_origins.items()}
            and all(path in session['source_hashes'] for path in runtime_origins.values()),
            'actual loaded GUI/control modules left the reviewed bundle tree')
    for filename in ('mapped_libraries_before_load.json','mapped_libraries_final.json'):
        mappings = saved_document(run/filename)['paths']
        require(mappings and all(path in session['source_hashes'] for path in mappings),
                'actual compute DSO mapping absent from frozen closure')
    pauses = result['pause_render_proofs']
    require([r['stage'] for r in pauses] == ['initial','final']
            and all(r['unchanged'] is True and r['before_boundary'] == r['after_boundary'] for r in pauses),
            'stable render pauses changed physical state/counters')
    inputs = list(saved_rows(run/'input_snapshots.jsonl.gz'))
    decisions = list(saved_rows(run/'owner_decisions.jsonl.gz'))
    require(worker['input_polls'] == len(inputs) and worker['owner_decisions'] == len(decisions),
            'input archive count differs')
    input_report = check_prepared_inputs(session,inputs,decisions,episodes)
    performance = saved_document(run/'performance.json')
    frames = saved_value(run/'render_frames.json')
    require(performance['render_frames'] == frames and worker['render_frames'] == len(frames)
            and performance['active_start_ns'] == result['active_start_ns']
            and performance['active_end_ns'] == result['active_end_ns'], 'render/worker windows differ')
    published = [{**r['metadata'],'publication':r['publication']}
                 for r in performance['published_frame_snapshots']]
    require(len(published) == result['mailbox']['published']
            and [r['publication']['frame_seq'] for r in published] == list(range(1,len(published)+1)),
            'owner publication/frame sequence count differs')
    for meta in published:
        episode = episodes[meta['episode_id']]
        tick = meta['episode_tick']
        start = episode['rows'][0]['control_index']
        require(meta['control_index'] == start+tick and 0 <= tick <= episode['controls']
                and close(meta['sim_time_s'],episode['states']['time'][tick]),
                'render source detached from actual episode/control endpoint')
    render = {'rendered':False,'render_api_calls':0,'passed':True}
    if session['render']:
        require(0 < performance['render_api_calls'] <= session['frame_limit'] == 1000
                and performance['poll_timestamps_ns'] == [r['poll']['wall_ns'] for r in inputs],
                'actual render API/poll counts differ')
        render = check_performance_data(run,controls,result['active_start_ns'],result['active_end_ns'],
            performance['poll_timestamps_ns'],frames,published)
        render.update(rendered=True,render_api_calls=performance['render_api_calls'])
    else:
        require(frames == [] and inputs == [] and performance['render_api_calls'] == 0
                and result['copy_edges']['reading_to_display'] == 0, 'headless run rendered/polled GLFW')
    events = None
    if session['mode'] == 'keyboard':
        events = check_real_events(run,session,go,inputs,decisions)
        require(result['input_reset_used'] is True and input_report['actual_reset_count'] == 1
                and result['stop_reason'] == 'operator_exit', 'keyboard R/exit not actually closed')
        pause = session['test_input_pause']
        duration = (performance['input_pause_end_ns']-performance['input_pause_start_ns'])/1e9
        require(0 < pause['duration_s'] <= .4 and duration >= pause['duration_s'],
                'actual bounded input-publication pause absent')
        events['ttl_injection'] = {'kind':'publication pause with callback edges retained',
                                  'declared_duration_s':pause['duration_s'],'actual_duration_s':duration}
    all_rows = [r for e in episodes for r in e['rows']]
    stopped = stop_window(episodes[-1]['rows'])
    safe = all(e['safety_passed'] for e in episodes)
    actual_residual = any(e['nonzero_effective_residual'] for e in episodes)
    passed = bool(safe and actual_residual and render['passed'] and stopped['passed'])
    report = {'schema':'d1-c23-independent-saved-readback-v1','run':str(run),
        'execution_contract_id':session['execution_contract_id'],'arm':session['arm'],
        'session_identity':identity(run/'session.json'),'worker_receipt_identity':identity(run/'worker_receipt.json'),
        'record_valid':True,'arm_qualification_passed':passed,'source_and_actual_origin_verified':True,
        'checkpoint_sha256':MODEL_SHA,'observation_action_shape':[99,16],
        'completed_controls':controls,'normal_native_returns':5*controls,'compiler_native_returns':2,
        'model_calls':result['model_calls'],'input':input_report,'real_events':events,
        'render':render,'last100_stop':stopped,'physical_safety_passed':safe,
        'actual_nonzero_B22_residual':actual_residual,'actual_episode_count':len(episodes),
        'renderer_changes_physics_pending_pair':session['mode'] == 'script',
        'scope':'one finite B22/C18 GUI arm; no all-profile GUI retest, arbitrary driving, or RL superiority claim',
        'reader_model_calls':0,'reader_physics_calls':0}
    return {'report':report,'session':session,'episodes':episodes,'rows':all_rows}


def read_run(run):
    return read_run_details(run)['report']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--pair-headless',type=Path)
    args = parser.parse_args()
    actual = read_run_details(args.run)
    report = actual['report']
    if args.pair_headless is not None:
        reference = read_run_details(args.pair_headless)
        require(reference['report']['arm_qualification_passed'] and report['arm_qualification_passed'],
                'pair requires two independently qualified script arms')
        report['same_trajectory_pair'] = pair_runs(reference,actual)
        report['paired_headless_readback'] = reference['report']
        report['renderer_changes_physics_pending_pair'] = False
    with args.output.open('x') as stream:
        json.dump(report,stream,sort_keys=True,indent=2,allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    return 0 if report['arm_qualification_passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
