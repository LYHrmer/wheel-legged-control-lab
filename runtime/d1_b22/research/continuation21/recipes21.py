"""Pure C21 finite recipes; C20 numeric protocol retained. Commands are fixed before any model or physical call."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
from budget_spec_11 import BudgetSpec
from full_drive_command_08 import FullDriveCommand
from full_drive_schedule_08 import COURSE_SPAWNS_M
from short_episode_plan_11 import FrozenShortPlan, ShortEpisodeSchedule
from short_heldout_11 import WorldUprightHeldoutSchedule
from recipes18 import floor_schedule, make_schedule as original_schedule

SPEC = json.loads((Path(__file__).parent / 'spec21.json').read_text())
T = SPEC['training']
TRAIN_BUDGET = BudgetSpec(T['controls_per_arm_per_stage'], T['n_steps'],
                          T['batch_size'], T['n_epochs_max'])
PLAN_SCHEMA = 'd1-c18-command-coverage-short1000-c20-v1'
PPO_SEED = T['ppo_seed_by_stage'][0]
MEASUREMENT_SEED = T['measurement_seed']
SELECTION_SEED = T['command_seed']
_WARMUP = ((.4, 0., 1), (.6, 0., 1), (.8, 0., 1), (1., 0., 1),
           (1.2, 0., 1), (1.6, 0., 1), (.6, .15, 1), (.9, .2, -1))
_SLOTS = (('flat', (.4, .9), None), ('flat', (1.1, 1.6), None),
          ('flat', (.6, 1.2), (.15, .3)), ('bumps', (.25, .4), None),
          ('rough', (.25, .4), None), ('ramp', (.35, .5), None),
          ('flat', 1.6, None), ('ramp', (.35, .5), None))


def select_episode(index, arm, *, source_offset=0):
    if type(index) is not int or index < 0 or arm not in ('A', 'B'):
        raise ValueError('invalid episode/arm')
    if type(source_offset) is not int or source_offset < 0:
        raise ValueError('invalid cumulative episode offset')
    source = index + source_offset
    if not 0 <= source <= 79:
        raise ValueError("C21 source index is outside the sealed finite range 0..79")
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([SELECTION_SEED, source])))
    slot = cycle = None
    if source < 8:
        terrain = 'flat'
        speed, amplitude, sign = _WARMUP[source]
    else:
        slot, cycle = (source-8) % 8, (source-8) // 8
        terrain, speed_rule, yaw_rule = _SLOTS[slot]
        speed = float(rng.uniform(*speed_rule)) if isinstance(speed_rule, tuple) else speed_rule
        amplitude = float(rng.uniform(*yaw_rule)) if yaw_rule else 0.
        sign = (1 if int(rng.integers(0, 2)) == 0 else -1) if yaw_rule else 1
    segments = [[450, 550, amplitude*sign], [550, 750, -amplitude*sign],
                [750, 850, amplitude*sign]] if amplitude else []
    if arm == 'B' and source >= 8 and slot == 2:
        override = T['B_yaw_cycles'][cycle % T['B_cycle_period']]
        speed, segments = override['vx'], override['segments']
    if arm == 'B' and source >= 8 and slot == 3:
        speed = T['B_bumps_vx_cycles'][cycle % T['B_cycle_period']]
    raw = tuple(FullDriveCommand(
        forward_velocity_mps=speed if tick >= 175 else 0.,
        yaw_rate_rps=next((yaw for start, end, yaw in segments if start <= tick < end), 0.),
    ) for tick in range(1000))
    choice = {'schema': PLAN_SCHEMA, 'arm': arm, 'episode_index': index,
              'source_episode_index': source, 'source_offset': source_offset,
              'selection_seed': SELECTION_SEED, 'slot': slot, 'cycle': cycle,
              'terrain': terrain, 'speed_mps': speed, 'yaw_segments': segments,
              'warmup': source < 8}
    return ShortEpisodeSchedule(index, f'c20_{arm}_{source:06d}_{terrain}', terrain,
        COURSE_SPAWNS_M[terrain], SELECTION_SEED,
        json.dumps(choice, sort_keys=True, separators=(',', ':'), allow_nan=False), raw)


def build_plan(*, source_sha256, arm, source_offset=0):
    return FrozenShortPlan(PLAN_SCHEMA, source_sha256,
        lambda index: select_episode(index, arm, source_offset=source_offset))


@dataclass(frozen=True)
class Task:
    case_id: str
    terrain: str
    speed_mps: float
    yaw_amplitude_rps: float
    seed: int
    horizon: int
    hold_start: int
    hold_end: int
    release_tick: int
    final_start: int
    drive_start: int


CASE_SPECS = {s['case_id']: s for group in ('regression', 'development', 'final_sealed')
              for s in SPEC['evaluation'][group]}
TASK_BY_ID = {name: Task(name, s['terrain'], s['speed_mps'],
    s.get('yaw_amplitude_rps', max((abs(a[2]) for a in s.get('yaw_segments', [])), default=0.)),
    s['seed'], s['horizon'], *s['hold'], s['release_tick'], s['final'][0], s['drive_start'])
    for name, s in CASE_SPECS.items()}
ORDER = tuple(s['case_id'] for s in SPEC['evaluation']['regression'][:6])


def make_schedule(case_id, seed=None, mirror=False):
    s = CASE_SPECS[case_id]
    if seed is not None and seed != s['seed']:
        raise ValueError('case seed differs from preregistered identity')
    if 'source_case_id' in s:
        old = original_schedule(s['source_case_id'], s['seed'], mirror=s['mirror'])
        return WorldUprightHeldoutSchedule(case_id, old.terrain, old.seed,
                                           old.spawn_position_m, old.raw_commands)
    if mirror:
        raise ValueError('new cases carry explicit signed commands')
    commands = tuple(FullDriveCommand(
        forward_velocity_mps=s['speed_mps'] if s['drive_start'] <= t < s['release_tick'] else 0.,
        yaw_rate_rps=next((y for a, b, y in s['yaw_segments'] if a <= t < b), 0.),
    ) for t in range(s['horizon']))
    return WorldUprightHeldoutSchedule(case_id, s['terrain'], s['seed'],
        COURSE_SPAWNS_M[s['terrain']], commands)


def serialized_tables():
    """Seal every eval command and a bounded prefix of source training recipes."""
    return {'evaluation': {name: {'sha256': make_schedule(name).command_sha256,
                'raw_commands': [asdict(c) for c in make_schedule(name).raw_commands]}
            for name in CASE_SPECS},
            'training_first_80_sources': {arm: [
                {'source_index': i, 'command_sha256': select_episode(i, arm).command_sha256,
                 'choice': json.loads(select_episode(i, arm).choice_json)}
                for i in range(80)] for arm in ('A', 'B')}}
