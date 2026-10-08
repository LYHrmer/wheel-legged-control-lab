"""Pure tests for the C37 fullM destination predicate. No MuJoCo, no physics, no model."""
from __future__ import annotations

import unittest

import numpy as np
from runtime37 import registered_destination37


class LegacyScratch:
    """Stands in for a MuJoCo build that exposes the packed inertia as qM."""

    def __init__(self, packed):
        self.qM = packed


class ModernScratch:
    """Stands in for MuJoCo 3.12, which exposes M and has no qM at all."""

    def __init__(self, dense):
        self.M = dense


class ShortCircuit(unittest.TestCase):
    def test_modern_binding_never_touches_the_legacy_attribute(self):
        scratch = ModernScratch(np.zeros((3, 3)))
        self.assertFalse(hasattr(scratch, 'qM'))
        # the frozen predicate raised AttributeError here; this must simply accept
        self.assertTrue(registered_destination37(True, np.zeros((3, 3)), scratch))
        self.assertTrue(registered_destination37(True, None, scratch))

    def test_frozen_expression_would_have_raised_on_the_same_input(self):
        scratch = ModernScratch(np.zeros((3, 3)))
        third = np.zeros((3, 3))
        with self.assertRaises(AttributeError):
            bool(isinstance(third, np.ndarray) and np.shares_memory(third, scratch.qM))


class LegacyAcceptance(unittest.TestCase):
    def test_legacy_binding_still_requires_shared_memory(self):
        packed = np.zeros(9)
        scratch = LegacyScratch(packed)
        self.assertTrue(registered_destination37(False, packed, scratch))
        self.assertTrue(registered_destination37(False, packed[2:], scratch))

    def test_legacy_binding_rejects_an_unregistered_array(self):
        scratch = LegacyScratch(np.zeros(9))
        self.assertFalse(registered_destination37(False, np.zeros(9), scratch))

    def test_non_array_third_argument_is_rejected(self):
        scratch = LegacyScratch(np.zeros(9))
        self.assertFalse(registered_destination37(False, [0.]*9, scratch))
        self.assertFalse(registered_destination37(False, None, scratch))

    def test_modern_scratch_with_legacy_call_is_rejected_not_crashed(self):
        scratch = ModernScratch(np.zeros((3, 3)))
        self.assertFalse(registered_destination37(False, np.zeros(9), scratch))


class EquivalenceWhereTheFrozenCheckWasDefined(unittest.TestCase):
    """Where the frozen predicate could run at all, C37 must agree with it exactly."""

    def frozen(self, modern, third, scratch):
        packed = isinstance(third, np.ndarray) and np.shares_memory(third, scratch.qM)
        return bool(modern or packed)

    def test_agrees_on_every_legacy_combination(self):
        packed = np.zeros(9)
        scratch = LegacyScratch(packed)
        others = (packed, packed[1:], np.zeros(9), [0.]*9, None)
        for modern in (True, False):
            for third in others:
                self.assertEqual(registered_destination37(modern, third, scratch),
                                 self.frozen(modern, third, scratch),
                                 f'modern={modern} third={type(third).__name__}')


if __name__ == '__main__':
    unittest.main(verbosity=2)
