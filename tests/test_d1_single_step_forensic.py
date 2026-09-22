"""Bounded unit tests for contact anomaly inventory bookkeeping.

Synthetic record dictionaries only: no MuJoCo, no case loading, no filesystem,
no subprocess, and ``main`` is never invoked.
"""
from __future__ import annotations

import unittest

from scripts.audit_d1_single_step_contact import contact_anomalies


def contact(index=0, active=True, load=1.0, valid=True, box=True):
    feature = {"geometric_support_valid": valid} if box else None
    return {"index": index, "active": active, "normal_load_n": load,
            "box_feature": feature, "robot_geom_id": 3, "terrain_geom_id": 7}


def row(index, *contacts):
    return {"index": index, "contacts": {"contacts": list(contacts)}}


class ContactAnomalyInventoryTest(unittest.TestCase):
    def test_unloaded_invalid_stays_anomaly_despite_loaded_valid_row(self):
        native = [row(0, contact(index=0, active=False, load=0.0, valid=False)),
                  row(1, contact(index=1, active=True, load=12.5, valid=True))]
        inv = contact_anomalies(native)
        self.assertEqual(inv["box_contact_rows"], 2)
        self.assertEqual(inv["positive_load_rows"], 1)
        self.assertEqual(len(inv["anomalies"]), 1)
        anomaly = inv["anomalies"][0]
        self.assertEqual(anomaly["native_index"], 0)
        self.assertEqual(anomaly["contact_index"], 0)
        self.assertIs(anomaly["loaded"], False)
        self.assertFalse(inv["all_box_features_valid"])
        self.assertTrue(inv["all_positive_load_features_valid"])
        self.assertFalse(inv["original_qualification_overridden"])

    def test_active_flag_alone_is_not_positive_load(self):
        native = [row(0, contact(active=True, load=0.0, valid=False))]
        inv = contact_anomalies(native)
        self.assertEqual(inv["box_contact_rows"], 1)
        self.assertEqual(inv["positive_load_rows"], 0)
        self.assertIs(inv["anomalies"][0]["loaded"], False)
        self.assertFalse(inv["all_positive_load_features_valid"])
        self.assertFalse(inv["original_qualification_overridden"])

    def test_negative_and_zero_loads_are_not_qualified(self):
        native = [row(0, contact(index=0, active=True, load=-4.0, valid=True),
                      contact(index=1, active=True, load=0.0, valid=True)),
                  row(1, contact(index=0, active=False, load=9.0, valid=True))]
        inv = contact_anomalies(native)
        self.assertEqual(inv["box_contact_rows"], 3)
        self.assertEqual(inv["positive_load_rows"], 0)
        self.assertEqual(inv["anomalies"], [])
        self.assertTrue(inv["all_box_features_valid"])
        self.assertFalse(inv["all_positive_load_features_valid"])
        self.assertFalse(inv["original_qualification_overridden"])

    def test_positive_invalid_feature_blocks_positive_load_validity(self):
        native = [row(4, contact(index=2, active=True, load=30.0, valid=False))]
        inv = contact_anomalies(native)
        self.assertEqual(inv["positive_load_rows"], 1)
        self.assertIs(inv["anomalies"][0]["loaded"], True)
        self.assertFalse(inv["all_box_features_valid"])
        self.assertFalse(inv["all_positive_load_features_valid"])
        self.assertFalse(inv["original_qualification_overridden"])

    def test_no_box_contacts_cannot_qualify(self):
        for native in ([], [row(0)], [row(0, contact(box=False, load=50.0))]):
            inv = contact_anomalies(native)
            self.assertEqual(inv["box_contact_rows"], 0)
            self.assertEqual(inv["positive_load_rows"], 0)
            self.assertEqual(inv["anomalies"], [])
            self.assertFalse(inv["all_box_features_valid"])
            self.assertFalse(inv["all_positive_load_features_valid"])
            self.assertFalse(inv["original_qualification_overridden"])

    def test_no_positive_loads_cannot_qualify_even_when_all_valid(self):
        native = [row(0, contact(active=False, load=0.0, valid=True))]
        inv = contact_anomalies(native)
        self.assertTrue(inv["all_box_features_valid"])
        self.assertFalse(inv["all_positive_load_features_valid"])
        self.assertFalse(inv["original_qualification_overridden"])

    def test_override_flag_is_always_false(self):
        cases = [[], [row(0, contact())],
                 [row(0, contact(valid=False))],
                 [row(0, contact(active=False, load=0.0, valid=False))],
                 [row(0, contact(box=False)), row(1, contact(load=-1.0))]]
        for native in cases:
            self.assertFalse(contact_anomalies(native)["original_qualification_overridden"])


if __name__ == "__main__":
    unittest.main()
