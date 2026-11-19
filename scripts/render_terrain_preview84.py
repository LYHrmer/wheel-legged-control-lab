"""Draw the two preregistered terrain-preview pairs from recorded MuJoCo poses."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SEED, FPS, FINAL_TICK = 8417, 20, 600
PANE_W, PANE_H, GAP = 448, 244, 4
WIDTH, HEIGHT, SCENE_Y = 2 * PANE_W + GAP, 428, 76
BG, INK, MUTED = '#172530', '#f4f6f7', '#bac7cc'
COLORS = {'off': '#90b9e6', 'on': '#68d4b1'}
ARM_LABELS = {'off': 'Proprioception only', 'on': 'Proprioception + terrain preview'}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as src:
        for block in iter(lambda: src.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def tick(time_s: float) -> int:
    if not math.isfinite(time_s):
        raise ValueError('nonfinite recording clock')
    result = round(time_s * 100)
    if abs(time_s - result / 100) > 1e-6 or not 0 <= result <= FINAL_TICK:
        raise ValueError(f'off-grid recording time {time_s}')
    return result


@dataclass
class Recording:
    arm: str
    metrics: dict
    states: dict
    telemetry: dict
    terrain: dict
    hashes: dict

    @classmethod
    def read(cls, directory: Path, slope: str, arm: str) -> Recording:
        hashes = {name: sha256(directory / name) for name in
                  ('metrics.json', 'states.npz', 'telemetry.csv', 'terrain.npz')
                  if (directory / name).is_file()}
        metrics, states, telemetry, terrain = {}, {}, {}, {}
        if 'metrics.json' in hashes:
            metrics = json.loads((directory / 'metrics.json').read_text())
            if (metrics['terrain'], metrics['seed'], metrics['arm']) != (slope, SEED, arm):
                raise ValueError(f'case identity mismatch in {directory}')
        if 'terrain.npz' in hashes:
            with np.load(directory / 'terrain.npz', allow_pickle=False) as z:
                terrain = {k: z[k].copy() for k in
                           ('hfield', 'ramp_x_m', 'ramp_width_m', 'slope_deg')}
        if 'states.npz' in hashes:
            with np.load(directory / 'states.npz', allow_pickle=False) as z:
                times, positions, velocities = (z[k] for k in ('time_s', 'qpos', 'qvel'))
            if positions.shape != (len(times), 23) or velocities.shape != (len(times), 22):
                raise ValueError(f'invalid recorded D1 dimensions in {directory}')
            previous = -1
            for t, pos, vel in zip(times, positions, velocities, strict=True):
                index = tick(float(t))
                if index <= previous:
                    raise ValueError(f'unordered pose clock in {directory}')
                previous = index
                if np.isfinite(pos).all() and np.isfinite(vel).all():
                    states[index] = (pos, vel)
        if 'telemetry.csv' in hashes:
            with (directory / 'telemetry.csv').open(newline='') as src:
                for row in csv.DictReader(src):
                    index = tick(float(row['time_s']))
                    if index in telemetry:
                        raise ValueError(f'duplicate measurement clock in {directory}')
                    telemetry[index] = row
        return cls(arm, metrics, states, telemetry, terrain, hashes)


def number(value, fmt: str, scale: float = 1.0) -> str:
    if value is None or value == '':
        return '--'
    v = float(value) * scale
    return format(v, fmt) if math.isfinite(v) else '--'


def footer(draw, x, record, index, fonts, status):
    _, label, small = fonts
    y = SCENE_Y + PANE_H + 7
    row = record.telemetry.get(index, {})
    error = number(row.get('pitch_error_rad'), '+.2f', 180 / math.pi)
    delta = number(row.get('applied_delta_pitch_rad'), '+.1f', 180_000 / math.pi)
    draw.text((x + 12, y), f'Pitch error  {error}°      Preview correction  {delta} mdeg',
              font=label, fill=INK)
    events = (('Seen', 'first_seen_s'), ('Corrected', 'first_correct_correction_s'),
              ('Front contact', 'first_front_contact_s'))
    for col, (name, key) in enumerate(events):
        t = record.metrics.get(key)
        shown = t is not None and math.isfinite(float(t)) and float(t) <= index / 100 + 1e-9
        event_time = f'{float(t):.2f}s' if shown else '--'
        draw.text((x + 12 + col * 140, y + 23), f'{name}  {event_time}', font=small,
                  fill=COLORS[record.arm] if shown else MUTED)
    draw.text((x + 12, y + 43), status, font=small, fill=MUTED)


def add_marks(mujoco, scene, x_axis, road_z, ramp_x, center_x):
    # World-fixed visual marks follow the archived collision surface at true scale.
    start = math.ceil((center_x - 1.6) * 4)
    marks = [(n / 4, False) for n in range(start, start + 14) if abs(n / 4 - ramp_x) > .025]
    for x, entry in [*marks, (ramp_x, True)]:
        if not x_axis[0] <= x <= x_axis[-1] or scene.ngeom >= scene.maxgeom:
            continue
        z = float(np.interp(x, x_axis, road_z)) + .002
        geom = scene.geoms[scene.ngeom]
        color = (0.94, 0.52, 0.11, 1.0) if entry else (0.50, 0.57, 0.60, 1.0)
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                           np.eye(3).ravel(), np.array(color, dtype=np.float32))
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, .006 if entry else .0018,
                             np.array((x, -.60, z)), np.array((x, .60, z)))
        geom.category = int(mujoco.mjtCatBit.mjCAT_DECOR)
        scene.ngeom += 1


def read_geometry(records):
    geometries = [r.terrain for r in records if r.terrain]
    if not geometries:
        raise ValueError('no recorded collision geometry for either side')
    terrain = geometries[0]
    for other in geometries[1:]:
        if any(not np.array_equal(v, other[k]) for k, v in terrain.items()):
            raise ValueError('paired cases have different archived terrain')
    return terrain


def render_pair(run, out, slope, verdict, fonts):
    import mujoco

    from wheel_legged_control.d1.model import build_d1_model
    from wheel_legged_control.d1.training_terrain import TrainingTerrainConfig

    dirs = [run / 'cases' / f'{slope}_{SEED}_{arm}' for arm in ARM_LABELS]
    records = [Recording.read(d, slope, arm) for d, arm in zip(dirs, ARM_LABELS, strict=True)]
    terrain = read_geometry(records)
    model = build_d1_model(training_terrain=TrainingTerrainConfig())
    floor = model.geom('floor').id
    hid = int(model.geom_dataid[floor])
    hfield = terrain['hfield']
    expected = int(model.hfield_nrow[hid]), int(model.hfield_ncol[hid])
    if (hfield.shape != expected or not np.isfinite(hfield).all()
            or np.any((hfield < 0) | (hfield > 1)) or not np.all(hfield == hfield[0])):
        raise ValueError('invalid recorded longitudinal heightfield')
    adr = int(model.hfield_adr[hid])
    model.hfield_data[adr:adr + hfield.size] = hfield.ravel()
    size = model.hfield_size[hid]
    x_axis = model.geom_pos[floor, 0] + np.linspace(-size[0], size[0], expected[1])
    road_z = model.geom_pos[floor, 2] + hfield[0].astype(float) * size[2]
    ramp_x = float(terrain['ramp_x_m'])
    model.vis.global_.offwidth, model.vis.global_.offheight = PANE_W, PANE_H
    model.geom_matid[floor] = -1
    model.geom_rgba[floor] = (.72, .77, .79, 1.)
    model.vis.headlight.ambient[:] = .48
    camera, options, data = mujoco.MjvCamera(), mujoco.MjvOption(), mujoco.MjData(model)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance, camera.azimuth, camera.elevation = 1.66, 96., -14.
    options.geomgroup[3] = 0
    frames = out / f'{slope}_png'
    frames.mkdir()
    heading, label, small = fonts
    row_count, gap_count, terminal_count = 0, [0, 0], [0, 0]
    name = 'Uphill' if slope == 'up' else 'Downhill'
    with mujoco.Renderer(model, height=PANE_H, width=PANE_W) as renderer:
        for frame, index in enumerate(range(0, FINAL_TICK + 1, 100 // FPS)):
            canvas = Image.new('RGB', (WIDTH, HEIGHT), color=BG)
            draw = ImageDraw.Draw(canvas)
            draw.text((12, 9), f'D1   {name} {float(terrain["slope_deg"]):+.0f}°',
                      font=heading, fill=INK)
            draw.text((WIDTH - 14, 17), f'0.35 m/s     seed {SEED}     t = {index / 100:.2f} s',
                      font=label, fill=MUTED, anchor='ra')
            for side, r in enumerate(records):
                x = side * (PANE_W + GAP)
                draw.rectangle((x, 49, x + 5, SCENE_Y - 3), fill=COLORS[r.arm])
                draw.text((x + 12, 50), ARM_LABELS[r.arm], font=label, fill=INK)
                reason = r.metrics.get('termination_reason')
                stop = int(r.metrics.get('steps', FINAL_TICK))
                ended = bool(reason and reason != 'time_limit' and index >= stop)
                source = stop if ended else index
                state = r.states.get(source)
                if state is not None and r.terrain:
                    data.qpos[:], data.qvel[:], data.time = *state, source / 100
                    mujoco.mj_forward(model, data)
                    row_count += 1
                    bx, by = data.qpos[:2]
                    camera.lookat[:] = (bx + .14, by, np.interp(bx, x_axis, road_z) + .24)
                    renderer.update_scene(data, camera=camera, scene_option=options)
                    add_marks(mujoco, renderer.scene, x_axis, road_z, ramp_x, bx)
                    canvas.paste(Image.fromarray(renderer.render()), (x, SCENE_Y))
                    status = 'Recorded pose'
                    if ended:
                        terminal_count[side] += 1
                        status = f'Last recorded pose at {source / 100:.2f} s  |  {reason}'
                    elif index == FINAL_TICK and r.metrics.get('full_horizon'):
                        status = 'Reached 6.00 s'
                else:
                    gap_count[side] += 1
                    draw.rectangle((x, SCENE_Y, x + PANE_W - 1, SCENE_Y + PANE_H - 1),
                                   fill='#33424c')
                    draw.text((x + PANE_W // 2, SCENE_Y + PANE_H // 2), 'No recorded frame',
                              font=label, fill=INK, anchor='mm')
                    status = f'Ended at {stop / 100:.2f} s  |  {reason}' if ended else 'Missing data'
                footer(draw, x, r, index, fonts, status)
            draw.line((0, HEIGHT - 30, WIDTH, HEIGHT - 30), fill='#42515a')
            draw.text((12, HEIGHT - 23), 'Orange stripe: start of ramp   |   True terrain scale',
                      font=small, fill=MUTED)
            draw.text((WIDTH - 12, HEIGHT - 23), f'27-case screen: {verdict}', font=small,
                      fill=INK, anchor='ra')
            canvas.save(frames / f'{frame:04d}.png')
    for r, d in zip(records, dirs, strict=True):
        if any(sha256(d / name) != digest for name, digest in r.hashes.items()):
            raise ValueError(f'source recording changed during rendering: {d}')
    return {'png_directory': frames.name, 'png_frames': frame + 1, 'forward_calls': row_count,
            'unavailable_frames': gap_count, 'terminal_pose_frames': terminal_count,
            'arms': {r.arm: {'source_sha256': r.hashes, 'recorded_poses': len(r.states),
                             'termination_reason': r.metrics.get('termination_reason'),
                             'pitch_rmse_deg': r.metrics.get('pitch_rmse_deg')}
                     for r in records}}


def ffmpeg(*args):
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-n',
                    '-threads', '1', '-filter_threads', '1', '-filter_complex_threads', '1',
                    *map(str, args)], check=True, timeout=240)


def encode(out, pairs):
    files = {}
    inputs = []
    for slope, pair in pairs.items():
        pattern = out / pair['png_directory'] / '%04d.png'
        args = ['-threads', '1', '-framerate', str(FPS), '-i', str(pattern)]
        inputs.extend(args)
        gif = out / f'terrain_preview84_{slope}.gif'
        graph = ('[0:v]fps=10,scale=810:-2:flags=lanczos,split[a][b];'
                 '[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer:bayer_scale=4')
        ffmpeg(*args, '-filter_complex', graph, '-loop', '0', '-threads', '1', gif)
        if gif.stat().st_size > 5_000_000:
            raise ValueError(f'{gif.name} exceeds 5 MB')
        with Image.open(gif) as im:
            duration_ms = 0
            for i in range(im.n_frames):
                im.seek(i)
                im.load()
                duration_ms += im.info['duration']
            if not 5_950 <= duration_ms <= 6_150:
                raise ValueError(f'invalid GIF duration {duration_ms} ms')
            files[gif.name] = {'bytes': gif.stat().st_size, 'sha256': sha256(gif),
                               'decoded_frames': im.n_frames, 'duration_ms': duration_ms}
    video = out / 'terrain_preview84.mp4'
    ffmpeg(*inputs, '-filter_complex', '[0:v][1:v]concat=n=2:v=1:a=0[v]', '-map', '[v]',
           '-an', '-r', FPS, '-c:v', 'libx264', '-preset', 'veryfast', '-crf', 20, '-pix_fmt', 'yuv420p',
           '-movflags', '+faststart', '-threads', 1, video)
    probe = subprocess.run(['ffprobe', '-v', 'error', '-threads', '1', '-count_frames',
                            '-select_streams', 'v:0', '-show_entries',
                            'stream=width,height,nb_read_frames,duration', '-of', 'json', str(video)],
                           check=True, capture_output=True, text=True, timeout=120)
    stream, = json.loads(probe.stdout)['streams']
    if (stream['width'], stream['height'], int(stream['nb_read_frames'])) != (
        WIDTH, HEIGHT, sum(p['png_frames'] for p in pairs.values())
    ):
        raise ValueError('MP4 frame count or dimensions differ from the PNG sequences')
    files[video.name] = {'bytes': video.stat().st_size, 'sha256': sha256(video), **stream}
    return files


def render(run: Path, output: Path) -> dict:
    if output.resolve().is_relative_to((run / 'cases').resolve()):
        raise ValueError('choose a new output outside the source case directories')
    output.mkdir(parents=True, exist_ok=False)
    result = {'status': 'INCOMPLETE', 'source': str(run.resolve()), 'seed': SEED,
              'explicit_physics_steps': 0, 'policy_calls': 0, 'fps': FPS, 'pairs': {}}
    try:
        os.environ.setdefault('MUJOCO_GL', 'egl')
        os.environ.setdefault('LIBGL_ALWAYS_SOFTWARE', '1')
        sys.path.insert(0, str(ROOT / 'src'))
        path = run / 'summary.json'
        summary = json.loads(path.read_text()) if path.is_file() else {}
        result['source_summary_sha256'] = sha256(path) if path.is_file() else None
        passed = (summary.get('gate') or {}).get('passed')
        verdict = ('pass' if passed else 'failed') if (
            summary.get('status') == 'COMPLETE' and isinstance(passed, bool)
        ) else 'incomplete'
        result['source_verdict'] = verdict
        fonts = tuple(ImageFont.truetype(f, size) for f, size in
                      [('DejaVuSans-Bold.ttf', 24), ('DejaVuSans.ttf', 16),
                       ('DejaVuSans.ttf', 13)])
        for slope in ('up', 'down'):
            result['pairs'][slope] = render_pair(run, output, slope, verdict, fonts)
        result['media'] = encode(output, result['pairs'])
        if path.is_file() and sha256(path) != result['source_summary_sha256']:
            raise ValueError('source summary changed during rendering')
        result['status'] = 'COMPLETE'
    except BaseException as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        with (output / 'render_report.json').open('x') as dst:
            json.dump(result, dst, indent=2, allow_nan=False)
            dst.write('\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path, help='recorded 84 suite directory')
    parser.add_argument('--output', required=True, type=Path, help='new media directory')
    args = parser.parse_args()
    print(json.dumps(render(args.run, args.output), indent=2))
