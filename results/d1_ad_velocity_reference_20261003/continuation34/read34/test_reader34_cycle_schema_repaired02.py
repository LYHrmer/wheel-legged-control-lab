"""One saved-receipt schema fixture for the C34 reader-only entry repair."""
import json
from pathlib import Path
import unittest


RECEIPT = (Path(__file__).resolve().parents[1] /
           'development_01/episode_0/cycle_receipt.json')


class SavedCycleSchema(unittest.TestCase):
    def test_actual_cycle_receipt_has_nested_learning_macros(self):
        cycle = json.loads(RECEIPT.read_text())
        self.assertEqual(cycle['schema'], 'd1-c31-cycle-receipt-v1')
        self.assertIn(cycle['terminal_kind'],
                      ('success', 'controlled_failure', 'safe_cancel'))
        self.assertNotIn('macros', cycle)
        self.assertIsInstance(cycle['learning_cycle']['macros'], list)
        self.assertGreater(len(cycle['learning_cycle']['macros']), 0)
        for event in cycle['learning_cycle']['macros']:
            self.assertIsNone(event['skill31_latch']['policy_receipt'])


if __name__ == '__main__':
    unittest.main()
