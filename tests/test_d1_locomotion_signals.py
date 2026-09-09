"""Check v3 observation semantics and hand-computed reward contributions."""

from dataclasses import fields, replace

import gymnasium as gym
import numpy as np
import pytest

from wheel_legged_control.d1.actuator_channel import (
    ActuatorChannel,
    ActuatorChannelConfig,
    ActuatorTrace,
)
from wheel_legged_control.d1.control_context import (
    D1AppliedAction,
    D1ControllerMemory,
    D1ForceBaseline,
    D1JointTargetBaseline,
)
from wheel_legged_control.d1.control_loop import (
    D1ControlLoop,
    D1ForceControllerAdapter,
    D1MotionCommand,
    D1WheelLegControllerAdapter,
)
from wheel_legged_control.d1.controllers import D1Command
from wheel_legged_control.d1.hierarchical import D1LQRVMCController
from wheel_legged_control.d1.locomotion_observation import (
    LOCOMOTION_OBSERVATION_SCHEMA,
    LOCOMOTION_OBSERVATION_SIZE,
    OBSERVATION_SLICES,
    encode_d1_locomotion_observation,
)
from wheel_legged_control.d1.locomotion_rewards import (
    D1LocomotionRewardConfig,
    d1_locomotion_reward_terms,
)
from wheel_legged_control.d1.model import (
    JOINT_TORQUE_LIMIT,
    JOINT_VELOCITY_LIMIT,
    NOMINAL_JOINT_POSITION,
    D1Plant,
)
from wheel_legged_control.d1.observation_history import D1ObservationHistory
from wheel_legged_control.d1.state_provider import D1StateProviderConfig, build_d1_state_provider
from wheel_legged_control.d1.training_terrain import TrainingGroundReference
from wheel_legged_control.d1.wheel_leg_controller import D1WheelLegController


def make_loop(action_size=2):
    actuator = ActuatorChannel(ActuatorChannelConfig(torque_limit_nm=tuple(JOINT_TORQUE_LIMIT)))
    plant = D1Plant(sampling_mode="synchronized", actuator_channel=actuator)
    provider = build_d1_state_provider(
        plant,
        D1StateProviderConfig("oracle"),
        oracle_ground_query=lambda x, y: TrainingGroundReference(0, 0, 0),
    )
    adapter = (
        D1ForceControllerAdapter(D1LQRVMCController(plant))
        if action_size == 2
        else D1WheelLegControllerAdapter(D1WheelLegController())
    )
    return D1ControlLoop(plant, provider, adapter)


@pytest.fixture(scope="module")
def transition():
    """One real 10 ms control interval; replacements below are explicit test data."""
    loop = make_loop()
    loop.reset(seed=5)
    loop.prepare(D1MotionCommand())
    return loop.step(np.zeros(2))


@pytest.fixture
def ground():
    return TrainingGroundReference(0.0, 0.0, 0.0)


def state_with_motion(state, *, velocity=(0, 0, 0), angular=(0, 0, 0), rotation=None, **kwargs):
    rotation = np.eye(3) if rotation is None else np.asarray(rotation)
    return replace(
        state,
        base_rotation=rotation,
        base_linear_velocity_body=np.asarray(velocity),
        base_linear_velocity_world=rotation @ velocity,
        base_angular_velocity_body=np.asarray(angular),
        base_angular_velocity_world=rotation @ angular,
        **kwargs,
    )


def zero_trace():
    return ActuatorTrace(*(np.zeros(16) for _ in range(5)))


@pytest.fixture
def distinctive_decision(transition):
    """Values chosen for hand-checked, different entries in every public slice."""
    leg_indices = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]
    normalized_legs = np.asarray((-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8))
    position = NOMINAL_JOINT_POSITION.copy()
    position[leg_indices] += normalized_legs * np.tile((0.8, 2, 1), 4)
    position[[3, 7, 11, 15]] = (100, 200, 300, 400)  # wheel angles must not enter leg slots
    state = state_with_motion(
        transition.state,
        sequence=1,
        control_time_s=0.05,
        measurement_time_s=0.025,
        base_position=np.asarray((8, 9, 0.64)),
        rotation=((0, -1, 0), (1, 0, 0), (0, 0, 1)),
        velocity=(0.12, -0.24, 0.36),
        angular=(0.4, -0.8, 1.2),
        joint_position=position,
        joint_velocity=JOINT_VELOCITY_LIMIT * np.linspace(-0.8, 0.7, 16),
    )
    proposal = replace(
        transition.decision.context.proposal,
        tick=1,
        control_time_s=0.05,
        baseline=D1ForceBaseline(90, 300, -125),
        memory=D1ControllerMemory(distance_m=1, distance_reference_m=1.275, yaw_integral_nm=1),
    )
    context = replace(
        transition.decision.context,
        state=state,
        proposal=proposal,
        previous_applied_action=D1AppliedAction(0, 0.04, 0.05, np.asarray((0.25, -0.5))),
    )
    return replace(
        transition.decision,
        context=context,
        motion_command=D1MotionCommand(0.3, 0.25, 0.495),
        world_command=D1Command(0.3, 0.25, 0.6, 0.06, -0.09),
    )


def test_default_reward_config_constructs_with_slots():
    config = D1LocomotionRewardConfig()
    assert config.velocity_weight == 1.0
    assert config.termination_cost == 2.0


def test_observation_slice_map_covers_exactly_82_values_without_gaps():
    assert LOCOMOTION_OBSERVATION_SIZE == 82
    assert OBSERVATION_SLICES == {
        "com_velocity_body_mps": (0, 3),
        "angular_velocity_body_over4": (3, 6),
        "projected_gravity": (6, 9),
        "height_error_over_0p08m": (9, 10),
        "leg_position_error_normalized": (10, 22),
        "joint_velocity_over_limit": (22, 38),
        "command_vx_yaw_clearance_roll_pitch": (38, 43),
        "measurement_age_over_0p05s": (43, 44),
        "baseline_force_or_leg_and_wheel_targets": (44, 60),
        "baseline_kind_force_joint": (60, 62),
        "distance_error_yaw_integral_wheel_integrals": (62, 68),
        "controller_memory_present": (68, 74),
        "previous_applied_action_padded8": (74, 82),
    }
    assert [
        index for start, end in OBSERVATION_SLICES.values() for index in range(start, end)
    ] == list(range(82))


def test_all_force_observation_values_match_the_documented_physical_mapping(distinctive_decision):
    observation = encode_d1_locomotion_observation(distinctive_decision)
    expected = np.asarray(
        [
            0.12,
            -0.24,
            0.36,
            0.1,
            -0.2,
            0.3,
            0,
            0,
            -1,
            0.5,
            -0.3,
            -0.2,
            -0.1,
            0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            -0.8,
            -0.7,
            -0.6,
            -0.5,
            -0.4,
            -0.3,
            -0.2,
            -0.1,
            0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.5,
            0.5,
            0.5,
            0.2,
            -0.3,
            0.5,
            0.5,
            0.5,
            -0.25,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0.5,
            0.25,
            0,
            0,
            0,
            0,
            1,
            1,
            0,
            0,
            0,
            0,
            0.25,
            -0.5,
            0,
            0,
            0,
            0,
            0,
            0,
        ],
        dtype=np.float32,
    )
    assert observation.shape == (82,) and observation.dtype == np.float32
    np.testing.assert_allclose(observation, expected, atol=1e-7, rtol=0)


def test_joint_baseline_is_target_geometry_not_force_and_all_eight_previous_actions_are_visible(
    distinctive_decision,
):
    target = NOMINAL_JOINT_POSITION.copy()
    target[[0, 4, 8, 12]] += (0.08, -0.16, 0.24, -0.32)
    target[[3, 7, 11, 15]] = (40, 50, 60, 70)  # wheel angle is not wheel-speed target
    proposal = replace(
        distinctive_decision.context.proposal,
        baseline=D1JointTargetBaseline(target, np.asarray((3, -6, 9, -12))),
        memory=D1ControllerMemory(wheel_integral_nm=np.asarray((0, 1, -2, 3))),
    )
    context = replace(
        distinctive_decision.context,
        proposal=proposal,
        action_size=8,
        action_schema="fixture-wheel-leg-eight-v1",
        previous_applied_action=D1AppliedAction(
            0, 0.04, 0.05, np.asarray((-0.8, -0.6, -0.4, -0.2, 0.2, 0.4, 0.6, 0.8))
        ),
    )
    observation = encode_d1_locomotion_observation(replace(distinctive_decision, context=context))
    np.testing.assert_allclose(
        observation[44:60],
        (0.1, 0, 0, -0.2, 0, 0, 0.3, 0, 0, -0.4, 0, 0, 0.1, -0.2, 0.3, -0.4),
        atol=1e-7,
    )
    np.testing.assert_array_equal(observation[60:62], (0, 1))
    np.testing.assert_array_equal(observation[62:68], (0, 0, 0, 0.25, -0.5, 0.75))
    np.testing.assert_array_equal(observation[68:74], (0, 0, 1, 1, 1, 1))
    np.testing.assert_allclose(observation[74:82], (-0.8, -0.6, -0.4, -0.2, 0.2, 0.4, 0.6, 0.8))


def test_zero_controller_memory_is_distinct_from_absent_memory(distinctive_decision):
    encoded = []
    for memory in (D1ControllerMemory(), D1ControllerMemory(0, 0, 0, np.zeros(4))):
        proposal = replace(distinctive_decision.context.proposal, memory=memory)
        context = replace(distinctive_decision.context, proposal=proposal)
        encoded.append(
            encode_d1_locomotion_observation(replace(distinctive_decision, context=context))
        )
    np.testing.assert_array_equal(encoded[0][62:68], encoded[1][62:68])
    np.testing.assert_array_equal(encoded[0][68:74], 0)
    np.testing.assert_array_equal(encoded[1][68:74], 1)


@pytest.mark.parametrize("action_size", (2, 8))
def test_reading_observation_does_not_advance_real_control_or_expose_last_action_as_current_baseline(
    action_size,
):
    loop = make_loop(action_size)
    loop.reset(seed=3)
    command = D1MotionCommand(0.15, 0.08)
    initial = loop.prepare(command)
    expected = encode_d1_locomotion_observation(initial)
    for _ in range(3):
        observation = encode_d1_locomotion_observation(loop.prepare(command))
        np.testing.assert_array_equal(observation, expected)
        observation[:] = -99
    assert loop.provider.read().sequence == 0 and loop.plant.data.time == 0
    np.testing.assert_array_equal(expected[74:], 0)
    loop.step(np.ones(action_size) * 1.4)
    current = loop.prepare(command)
    encoded = encode_d1_locomotion_observation(current)
    assert current.context.proposal.tick == 1
    np.testing.assert_array_equal(encoded[74 : 74 + action_size], 1)
    np.testing.assert_array_equal(encoded[74 + action_size :], 0)
    expected_kind = (1, 0) if action_size == 2 else (0, 1)
    np.testing.assert_array_equal(encoded[60:62], expected_kind)
    assert loop.provider.read().sequence == 1 and loop.plant.data.time == pytest.approx(0.01)


class EncodedDecisionEnv(gym.Env):
    """No simulator reference: history must use only the explicitly emitted vectors."""

    observation_schema = LOCOMOTION_OBSERVATION_SCHEMA
    source_schema = "test-only-fixed-decision-v1"

    def __init__(self, decisions):
        self.decisions = decisions
        self.observation_space = gym.spaces.Box(-5, 5, (82,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1, 1, (2,), dtype=np.float32)

    @property
    def plant(self):
        raise AssertionError("history must not access simulator state")

    @property
    def data(self):
        raise AssertionError("history must not access simulator state")

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.index = 0
        return encode_d1_locomotion_observation(self.decisions[0]), {}

    def step(self, action):
        self.index += 1
        return encode_d1_locomotion_observation(self.decisions[self.index]), 0.0, True, False, {}


def test_history_contains_only_successive_emitted_82_value_observations(distinctive_decision):
    next_decision = replace(distinctive_decision, motion_command=D1MotionCommand(0.3, 0.25, 0.415))
    base = EncodedDecisionEnv((distinctive_decision, next_decision))
    env = D1ObservationHistory(base, 2)
    first, _ = env.reset()
    np.testing.assert_array_equal(first[:82], first[82:])
    second, *_ = env.step(np.zeros(2, np.float32))
    np.testing.assert_array_equal(second[:82], first[:82])
    np.testing.assert_array_equal(second[82:], encode_d1_locomotion_observation(next_decision))
    assert first[40] == 0.5 and second[82 + 40] == -0.5
    assert env.source_schema == "test-only-fixed-decision-v1"


def test_encoder_clips_finite_extremes_without_mutating_state_and_rejects_nonfinite_commands(
    distinctive_decision,
):
    context = replace(
        distinctive_decision.context,
        state=state_with_motion(distinctive_decision.context.state, velocity=(10, -10, 0)),
    )
    extreme = replace(distinctive_decision, context=context)
    observed = encode_d1_locomotion_observation(extreme)
    np.testing.assert_array_equal(observed[:3], (5, -5, 0))
    np.testing.assert_array_equal(context.state.base_linear_velocity_body, (10, -10, 0))
    bad = replace(
        distinctive_decision,
        world_command=replace(distinctive_decision.world_command, roll_rad=np.nan),
    )
    with pytest.raises(ValueError, match="observation"):
        encode_d1_locomotion_observation(bad)


def test_encoder_rejects_an_unversioned_action_dimension(transition):
    context = replace(transition.decision.context, action_size=4)
    with pytest.raises(ValueError, match="2 or 8"):
        encode_d1_locomotion_observation(replace(transition.decision, context=context))


def trace_with_pairs(*, requested, applied, velocity):
    """Synthetic two-active-axis samples for hand-calculated mechanical power."""

    def vector(pair):
        value = np.zeros(16, dtype=np.float64)
        value[:2] = pair
        return value

    applied_vector = vector(applied)
    return ActuatorTrace(
        requested_nm=vector(requested),
        limited_nm=applied_vector,
        delayed_nm=applied_vector,
        applied_nm=applied_vector,
        joint_velocity_rps=vector(velocity),
    )


def varied_interval(transition, *, dt, action):
    receipt = replace(
        transition.receipt,
        end_time_s=transition.receipt.start_time_s + dt,
        normalized_action=np.asarray(action, dtype=np.float64),
    )
    return replace(transition, receipt=receipt)


def test_mechanical_power_uses_applied_torque_without_axis_cancellation(transition, ground):
    # Signed powers [+6, -8] W and [+10, -2] W give sums of magnitudes 14 and 12 W.
    # The equally weighted sample mean is 13 W; requested torques are irrelevant.
    traces = (
        trace_with_pairs(requested=(1e6, -1e6), applied=(3, -4), velocity=(2, 2)),
        trace_with_pairs(requested=(-1e6, 1e6), applied=(5, -1), velocity=(2, 2)),
    )
    stationary = varied_interval(transition, dt=0.01, action=(0, 0))
    terms = d1_locomotion_reward_terms(stationary, ground, traces, terminated=False)
    assert terms["mechanical_power"] == pytest.approx(-0.000013, rel=1e-12)
    ignored_request = tuple(replace(trace, requested_nm=np.full(16, -2e6)) for trace in traces)
    assert d1_locomotion_reward_terms(
        stationary, ground, ignored_request, terminated=False
    ) == pytest.approx(terms)


def test_action_change_integrates_over_dt_but_termination_is_an_event(transition, ground):
    idle = (zero_trace(),)
    # Same normalized action derivative: twice the change over twice the interval.
    fast = varied_interval(transition, dt=0.01, action=(0.1, -0.2))
    slow = varied_interval(transition, dt=0.02, action=(0.2, -0.4))
    fast_terms = d1_locomotion_reward_terms(fast, ground, idle, terminated=True)
    slow_terms = d1_locomotion_reward_terms(slow, ground, idle, terminated=True)
    assert fast_terms["action_change"] == pytest.approx(-0.00025, rel=1e-12)
    assert slow_terms["action_change"] == pytest.approx(-0.0005, rel=1e-12)
    assert fast_terms["termination"] == slow_terms["termination"] == -2
    for tick, terminated_terms in ((fast, fast_terms), (slow, slow_terms)):
        alive_terms = d1_locomotion_reward_terms(tick, ground, idle, terminated=False)
        assert alive_terms["termination"] == 0
        assert sum(terminated_terms.values()) - sum(alive_terms.values()) == pytest.approx(-2)


def test_all_reward_contributions_and_total_match_a_worked_physical_example(transition):
    # 0.2 m/s forward error; 0.4 rad/s yaw error; 25 mm clearance error;
    # 0.15 rad roll error; 13 W absolute mechanical power; action delta (.1, -.2).
    c, s = np.cos(0.15), np.sin(0.15)
    truth = state_with_motion(
        transition.truth,
        velocity=(0.2, 0, 0),
        angular=(0, 0, 0.4),
        base_position=np.asarray((0, 0, 0.58)),
        rotation=((1, 0, 0), (0, c, -s), (0, s, c)),
    )
    example = replace(varied_interval(transition, dt=0.01, action=(0.1, -0.2)), truth=truth)
    traces = (
        trace_with_pairs(requested=(1e6, -1e6), applied=(3, -4), velocity=(2, 2)),
        trace_with_pairs(requested=(1e6, -1e6), applied=(5, -1), velocity=(2, 2)),
    )
    terms = d1_locomotion_reward_terms(
        example, TrainingGroundReference(0.1, 0, 0), traces, terminated=False
    )
    expected = {
        "tracking_velocity": 0.003678794411714423,
        "tracking_yaw": 0.0001831563888873418,
        "height": -0.0025,
        "attitude": -0.0005,
        "mechanical_power": -0.000013,
        "action_change": -0.00025,
        "termination": 0,
    }
    assert terms == pytest.approx(expected, abs=1e-12, rel=1e-12)
    assert sum(terms.values()) == pytest.approx(0.0005989508006017648, abs=1e-12)


def test_reward_uses_poststep_truth_clearance_not_predecision_world_target(transition):
    truth = state_with_motion(transition.truth, base_position=np.asarray((4, 0, 0.655)))
    example = replace(transition, truth=truth)
    terms = d1_locomotion_reward_terms(
        example, TrainingGroundReference(0.2, 0, 0), (zero_trace(),), terminated=False
    )
    assert terms["height"] == pytest.approx(0, abs=1e-25)
    stale_ground = d1_locomotion_reward_terms(
        example, TrainingGroundReference(0, 0, 0), (zero_trace(),), terminated=False
    )
    assert stale_ground["height"] == pytest.approx(-0.16)
    # Actor feedback and the old world target cannot substitute for reward truth.
    changed_decision = replace(
        example.decision, world_command=replace(example.decision.world_command, base_height_m=9)
    )
    contaminated = replace(
        example,
        state=replace(example.state, base_position=np.asarray((0, 0, 9))),
        decision=changed_decision,
    )
    assert d1_locomotion_reward_terms(
        contaminated, TrainingGroundReference(0.2, 0, 0), (zero_trace(),), terminated=False
    ) == pytest.approx(terms)


def test_poststep_slope_target_rotates_with_heading_instead_of_reusing_old_world_command(
    transition,
):
    # dh/dx=tan(0.2), heading +pi/2: normal corresponds to roll=-0.2, pitch=0.
    slope = 0.2
    c, s = np.cos(-slope), np.sin(-slope)
    yaw = np.asarray(((0, -1, 0), (1, 0, 0), (0, 0, 1)))
    roll = np.asarray(((1, 0, 0), (0, c, -s), (0, s, c)))
    truth = state_with_motion(transition.truth, rotation=yaw @ roll)
    terms = d1_locomotion_reward_terms(
        replace(transition, truth=truth),
        TrainingGroundReference(0, -slope, 0),
        (zero_trace(),),
        terminated=False,
    )
    assert terms["attitude"] == pytest.approx(0, abs=1e-25)


def test_normal_wheel_rolling_is_not_penalized_as_stance_foot_slip(transition, ground):
    resting = state_with_motion(
        transition.truth, base_position=np.asarray((0, 0, 0.455)), joint_velocity=np.zeros(16)
    )
    spin = np.zeros(16)
    spin[[3, 7, 11, 15]] = 2
    # Radius .087 m: 2 rad/s corresponds to .174 m/s forward motion. A wheel
    # center translating at .174 m/s is not a stationary stance-foot violation.
    rolling = state_with_motion(
        resting,
        velocity=(0.174, 0, 0),
        joint_velocity=spin,
        wheel_contact=np.ones(4, dtype=bool),
    )
    idle_terms = d1_locomotion_reward_terms(
        replace(transition, truth=resting), ground, (zero_trace(),), terminated=False
    )
    rolling_trace = replace(zero_trace(), joint_velocity_rps=spin)
    decision = replace(
        transition.decision,
        motion_command=D1MotionCommand(0.174),
        world_command=replace(transition.decision.world_command, forward_velocity_mps=0.174),
    )
    rolling_terms = d1_locomotion_reward_terms(
        replace(transition, truth=rolling, decision=decision),
        ground,
        (rolling_trace,),
        terminated=False,
    )
    assert rolling_terms == pytest.approx(idle_terms)
    assert sum(rolling_terms.values()) == pytest.approx(0.02)


@pytest.mark.parametrize("field", [field.name for field in fields(D1LocomotionRewardConfig)])
@pytest.mark.parametrize("bad", (np.nan, np.inf, -1, True))
def test_every_reward_coefficient_rejects_nonfinite_negative_and_boolean_values(field, bad):
    with pytest.raises((ValueError, TypeError)):
        D1LocomotionRewardConfig(**{field: bad})


@pytest.mark.parametrize("field", ("velocity_weight", "velocity_sigma_mps"))
def test_reward_coefficients_are_scalars_not_single_element_arrays(field):
    # A length-one array passes NumPy truth tests but makes reward terms arrays.
    with pytest.raises((ValueError, TypeError)):
        D1LocomotionRewardConfig(**{field: np.asarray([0.2])})


@pytest.mark.parametrize(
    "field",
    (
        "velocity_sigma_mps",
        "yaw_sigma_rps",
        "height_scale_m",
        "attitude_scale_rad",
        "mechanical_power_scale_w",
        "action_change_scale",
    ),
)
def test_reward_denominators_cannot_be_zero(field):
    with pytest.raises(ValueError, match="positive"):
        D1LocomotionRewardConfig(**{field: 0})


def test_zero_weights_disable_terms_but_empty_physics_trace_is_not_faked(transition, ground):
    weights = {
        field.name: 0
        for field in fields(D1LocomotionRewardConfig)
        if "weight" in field.name or field.name == "termination_cost"
    }
    config = D1LocomotionRewardConfig(**weights)
    terms = d1_locomotion_reward_terms(
        transition, ground, (zero_trace(),), terminated=True, config=config
    )
    assert sum(terms.values()) == 0
    with pytest.raises(ValueError, match="traces"):
        d1_locomotion_reward_terms(transition, ground, (), terminated=False)
