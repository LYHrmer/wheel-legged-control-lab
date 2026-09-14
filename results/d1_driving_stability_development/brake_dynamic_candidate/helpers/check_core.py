"""Pure mock checks for the work-only core, without MuJoCo or repository edits."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


class Controller:
    def __init__(self, distance=1.0, reference=1.5):
        self.distance = distance
        self._distance_reference_m = reference
        self._prepared_control = None

    @property
    def control_memory(self):
        return SimpleNamespace(distance_m=self.distance,
                               distance_reference_m=self._distance_reference_m)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--core', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('checked_core', args.core)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    core = module.DynamicBrakeHold
    checks = []

    def check(name, function):
        try:
            function()
            checks.append({'name': name, 'passed': True})
        except (AssertionError, ValueError, TypeError, RuntimeError) as error:
            checks.append({'name': name, 'passed': False, 'error': str(error),
                           'type': type(error).__name__})

    def arithmetic_atomic():
        helper, controller = core(), Controller()
        helper.observe_applied(.5, eligible=True)
        before = copy.deepcopy(helper.__dict__)
        try:
            helper.prepare(controller, 0., 1e308, eligible=True)
        except ValueError:
            pass
        else:
            raise AssertionError('overflow did not fail')
        assert helper.__dict__ == before, 'overflow consumed helper edge/counters'
        assert controller._distance_reference_m == 1.5

    def hold_actual_error():
        helper, controller = core(), Controller()
        helper.observe_applied(.5, eligible=True)
        for _ in range(30):
            row = helper.prepare(controller, 0., 0., eligible=True)
            helper.observe_applied(0., eligible=True)
        assert row['hold_entered'] and helper.hold_count == 1
        controller.distance += .2
        row = helper.prepare(controller, 0., 0., eligible=True)
        assert not row['reference_written']
        assert row['new_error_m'] == controller._distance_reference_m-controller.distance, 'hold reports hypothetical clamped error instead of actual retained error'

    def edge_and_dwell():
        helper, controller = core(), Controller()
        assert not helper.prepare(controller, 0., 0., eligible=True)['release_started']
        helper.observe_applied(.5, eligible=True)
        row = helper.prepare(controller, 0., -.2, eligible=True)
        assert row['release_started'] and abs(row['new_error_m']-.04) < 1e-12
        helper.observe_applied(0., eligible=True)
        for _ in range(29):
            row = helper.prepare(controller, 0., .01, eligible=True)
            helper.observe_applied(0., eligible=True)
            assert not row['hold_entered']
        helper.prepare(controller, 0., .031, eligible=True)
        helper.observe_applied(0., eligible=True)
        for i in range(30):
            row = helper.prepare(controller, 0., .01, eligible=True)
            helper.observe_applied(0., eligible=True)
            assert row['hold_entered'] == (i == 29)
        assert helper.release_count == 1 and helper.hold_count == 1
        helper.prepare(controller, .5, .01, eligible=True)
        helper.observe_applied(.5, eligible=True)
        assert helper.prepare(controller, 0., .2, eligible=True)['release_started']
        assert helper.release_count == 2

    def ownership_and_actual_command():
        helper, controller = core(), Controller()
        helper.prepare(controller, .5, 0., eligible=True)
        helper.observe_applied(0., eligible=True)
        assert not helper.prepare(controller, 0., 0., eligible=True)['release_started']
        helper.observe_applied(.5, eligible=True)
        helper.prepare(controller, 0., 0., eligible=False)
        helper.observe_applied(0., eligible=False)
        assert not helper.prepare(controller, 0., 0., eligible=True)['release_started']
        helper.reset()
        assert helper.release_count == 0 and helper.phase == 'idle'

    def signed_bound():
        for old_error in (-.5, .5):
            for velocity in (-.2, .2):
                helper, controller = core(), Controller(reference=1.+old_error)
                helper.observe_applied(velocity, eligible=True)
                row = helper.prepare(controller, 0., velocity, eligible=True)
                assert abs(row['new_error_m']-(-.04 if old_error < 0 else .04)) < 1e-12

    def pending_atomic():
        helper, controller = core(), Controller()
        helper.observe_applied(.5, eligible=True)
        controller._prepared_control = object()
        before = copy.deepcopy(helper.__dict__)
        try:
            helper.prepare(controller, 0., .2, eligible=True)
        except RuntimeError:
            pass
        else:
            raise AssertionError('pending proposal not refused')
        assert helper.__dict__ == before and controller._distance_reference_m == 1.5

    for function in (arithmetic_atomic, hold_actual_error, edge_and_dwell,
                     ownership_and_actual_command, signed_bound, pending_atomic):
        check(function.__name__, function)
    result = {'checks': checks, 'all_passed': all(row['passed'] for row in checks),
              'physics_steps': 0}
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result))
    return int(not result['all_passed'])


if __name__ == '__main__':
    raise SystemExit(main())
