"""Canonical C31 learning macros from actual C30 state/native 5T intervals."""
from __future__ import annotations

import copy

from macro30 import build_macros30, components30


SCHEMA31 = 'd1-c31-actual-event-macros-v1'
ALIASES = {
    'progress': 'progress_potential_difference',
    'backtrack': 'backtrack_normalized',
    'longitudinal_square_normalized_s': 'longitudinal_normalized_square_integral_s',
    'lateral_goal_square_normalized_s': 'lateral_goal_error_normalized_square_integral_s',
    'yaw_square_normalized_s': 'yaw_error_normalized_square_integral_s',
    'actual_torque_square_normalized_s': 'torque_normalized_square_integral_s',
    'longitudinal_square_m2_s': 'longitudinal_square_integral_m2_s',
    'lateral_goal_square_m2_s': 'lateral_goal_error_square_integral_m2_s',
    'yaw_square_rad2_s': 'yaw_error_square_integral_rad2_s',
}


def canonical_components31(value: dict, control_range: tuple[int,int]) -> dict:
    """Only rename old exact C30 physical values; never invent a transition."""
    if not set(ALIASES).issubset(value) or 'elapsed_s' not in value:
        raise ValueError('C31 old actual macro components are incomplete')
    begin,end=map(int,control_range)
    if begin < 0 or end <= begin or abs(value['elapsed_s']-(end-begin)*.01)>1e-12:
        raise ValueError('C31 macro component duration differs from actual controls')
    return {'control_range':[begin,end], 'elapsed_controls':end-begin,
            'elapsed_s':value['elapsed_s'],
            **{name:value[alias] for alias,name in ALIASES.items()},
            'reward_weights_defined':False,
            'weighted_scalar_reward':None,
            'torque_cost_is_energy':False,
            'source_component_schema':'d1-c30-macros-v1'}


def build_macros31(states, rows, events, *, torque_limits, origin_xy, yaw0,
                   direction, end_reason, global_control_offset=0) -> dict:
    """Keep original C30 transitions and add only actual-index/canonical fields."""
    if (type(global_control_offset) is not int or global_control_offset < 0
            or not 1 <= len(events) <= 4):
        raise ValueError('C31 macro requires one to four true saved leg latches')
    original=build_macros30(states,rows,events,torque_limits=torque_limits,
                            origin_xy=origin_xy,yaw0=yaw0,direction=direction,
                            end_reason=end_reason)
    if len(original)!=len(events):
        raise RuntimeError('C31 actual C30 macro count differs from latches')
    native=[[trace['applied_nm'] for trace in row['native_actuator_traces']]
            for row in rows]
    whole_old=components30(states['qpos'],native,torque_limits,
                           origin_xy=origin_xy,yaw0=yaw0,direction=direction)
    output=[]
    for index,(transition,latch) in enumerate(zip(original,events)):
        start,end=int(transition['control_start']),int(transition['control_end'])
        skill=latch.get('skill31_latch')
        if (transition['macro_index']!=index or latch['control_index']!=start
                or not isinstance(skill,dict)
                or skill.get('local_control_index')!=start
                or skill.get('observation31',{}).get('macro_index')!=index
                or transition['consumed_action']!=list(latch['consumed_action'])):
            raise ValueError('C31 actual latch differs from C30 state/native macro')
        mapped=copy.deepcopy(transition)
        mapped['schema']='d1-c31-actual-event-transition-v1'
        mapped['source_transition30_schema']=transition['schema']
        mapped['original_reward_components30']=copy.deepcopy(transition['reward_components'])
        mapped['control_range']=[start,end]
        mapped['global_control_range']=[global_control_offset+start,
                                        global_control_offset+end]
        mapped['reward_components']=canonical_components31(
            transition['reward_components'],(start,end))
        mapped['skill31_latch']=copy.deepcopy(skill)
        output.append(mapped)
    full=canonical_components31(whole_old,(0,len(rows)))
    return {'schema':SCHEMA31,'events':copy.deepcopy(events),
            'transitions':output,
            'full_case_components':full,
            'original_full_case_components30':whole_old,
            'elapsed_controls':len(rows),
            'event_transition_count':len(output),
            'global_control_offset':global_control_offset,
            'direction':direction,'origin_xy':list(origin_xy),'yaw0':float(yaw0),
            'end_reason':end_reason,'training':True}
