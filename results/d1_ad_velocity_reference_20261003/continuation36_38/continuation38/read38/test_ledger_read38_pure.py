"""Pure test for the C38 ledger-ceiling generalisation. No MuJoCo, no physics, no model."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'continuation35'/'read35'))
import read35  # noqa: F401,E402

from ledger_read38 import predict_ceiling38  # noqa: E402


class CeilingGeneralisation(unittest.TestCase):
    def test_reduces_to_the_frozen_c35_literals_at_five_cases(self):
        session = {'prepare_controls': 200, 'retention_controls': 400,
                   'cases': [{}]*5}
        self.assertEqual(predict_ceiling38(session), (3001, 3032))

    def test_c38_two_case_budget(self):
        session = {'prepare_controls': 200, 'retention_controls': 400,
                   'cases': [{}]*2}
        self.assertEqual(predict_ceiling38(session), (1201, 1232))

    def test_scales_linearly_with_case_count(self):
        for cases in (1, 2, 3, 5, 8):
            session = {'prepare_controls': 200, 'retention_controls': 400,
                       'cases': [{}]*cases}
            predict, rows = predict_ceiling38(session)
            self.assertEqual(predict, cases*600+1)
            self.assertEqual(rows, cases*600+32)


if __name__ == '__main__':
    unittest.main(verbosity=2)
