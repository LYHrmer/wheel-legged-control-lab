"""Regression for real C35 receipt layout, without reading any trajectory."""
import copy
import unittest

from read35 import validate_cycle35_shape


def fixture35():
    return dict(schema='d1-c35-cycle-receipt-v1', case_id='pair_stand_zero',
        case_index=0, direction=0, kind='in_place',
        terminal_kind='fatal_incomplete_or_unsafe', control_range=[0,200],
        native_range=[0,1000], pair_events=[],
        command=dict(definition={'id':'pair_stand_zero'}, side_begin_control=200,
            side_end_control=None, stop_control_index=None, cancel_control_index=None),
        result=dict(completed_controls=200,actual_normal_native=1000,
            policy_predictions=200,pair_exchanges=0,pending_control=None,
            terminal_kind='fatal_incomplete_or_unsafe',task=None,failure={'type':'fixture'}),
        side_access={'case_index':0},archive_completed=False)


class ReceiptSchema35PureTest(unittest.TestCase):
    def test_actual_top_level_pair_event_receipt(self):
        validate_cycle35_shape(fixture35())

    def test_old_nested_learning_cycle_does_not_satisfy_new_receipt(self):
        value=copy.deepcopy(fixture35())
        value['learning_cycle']={'macros':value.pop('pair_events')}
        with self.assertRaises((KeyError,ValueError)):
            validate_cycle35_shape(value)


if __name__ == '__main__':
    unittest.main()
