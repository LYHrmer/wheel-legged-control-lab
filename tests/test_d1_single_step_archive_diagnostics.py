"""Pure checks of archived native contacts; no physics is reconstructed."""

from __future__ import annotations

import copy
import hashlib
import json
import unittest
from pathlib import Path

import numpy as np

from scripts.d1_single_step_contact_diagnostics import audit_contact
from scripts.diagnose_d1_single_step_archive import (
    FROZEN_FIXTURE_SHA256,
    FROZEN_MANIFEST_SHA256,
    audit_native_state,
    audit_payload,
    verify_anchor_hashes,
)


class SingleStepArchiveDiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).parent / "fixtures" / "d1_single_step_native_normal_anomaly.json"
        cls.fixture = json.loads(path.read_text())
        cls.geoms = {g["geom_id"]: g for g in cls.fixture["geometry_manifest"]["geoms"]}

    def bad_contact(self):
        row = self.fixture["native_samples"][1]
        return copy.deepcopy(row["contacts"]["contacts"][5])

    def partial_payload(self):
        return {
            "geometry_manifest": self.fixture["geometry_manifest"],
            "native": copy.deepcopy(self.fixture["native_samples"]),
            "endpoints": [],
            "trace": [],
            "obstacle_enabled": True,
            "native_entry_events": [],
        }

    def test_01_real_anomaly_retains_raw_link_and_fails_geometry(self):
        contact = self.bad_contact()
        before = copy.deepcopy(contact)
        result = audit_contact(contact, self.geoms)
        self.assertEqual(result["raw_issues"], [])
        self.assertIn("box_normal_outside_support_cone", result["geometry_issues"])
        self.assertFalse(result["positive_load"])
        self.assertEqual(contact, before)

    def test_02_positive_load_does_not_repair_bad_geometry(self):
        contact = self.bad_contact()
        contact["local_force_torque"][0] = 5.0
        contact["normal_load_n"] = 5.0
        contact["force_on_robot_world_n"] = (
            -np.asarray(contact["frame_geom1_to_geom2"]).T
            @ np.asarray(contact["local_force_torque"][:3])
        ).tolist()
        result = audit_contact(contact, self.geoms)
        self.assertEqual(result["raw_issues"], [])
        self.assertTrue(result["positive_load"])
        self.assertIn("box_normal_outside_support_cone", result["geometry_issues"])

    def test_03_reported_normal_sign_tamper_breaks_raw_link(self):
        contact = self.bad_contact()
        contact["normal_terrain_to_robot_world"] = [
            -value for value in contact["normal_terrain_to_robot_world"]
        ]
        self.assertTrue(audit_contact(contact, self.geoms)["raw_issues"])

    def test_04_changed_manifest_anchor_is_rejected(self):
        sources = self.fixture["sources"]
        manifest = next(value for key, value in sources.items()
                        if key.endswith("/single_step_readiness_01/manifest.json"))
        self.assertEqual(manifest["sha256"], FROZEN_MANIFEST_SHA256)
        changed = hashlib.sha256(b"changed manifest").hexdigest()
        with self.assertRaisesRegex(ValueError, "frozen_archive_manifest_changed"):
            verify_anchor_hashes(changed, FROZEN_FIXTURE_SHA256)

    def test_05_load_summary_tamper_breaks_raw_link(self):
        contact = self.bad_contact()
        contact["normal_load_n"] = 1.0
        self.assertTrue(audit_contact(contact, self.geoms)["raw_issues"])

    def test_06_nonfinite_position_fails_both_audits(self):
        contact = self.bad_contact()
        contact["position_world_m"][0] = float("nan")
        result = audit_contact(contact, self.geoms)
        self.assertTrue(result["raw_issues"])
        self.assertTrue(result["geometry_issues"])

    def test_07_real_top_tangent_remains_valid(self):
        contact = self.fixture["reference_contacts"]["top_tangent"]["contact"]
        result = audit_contact(contact, self.geoms)
        self.assertEqual(result["raw_issues"], [])
        self.assertEqual(result["geometry_issues"], [])

    def test_08_real_front_top_edge_remains_valid(self):
        contact = self.fixture["reference_contacts"]["front_top_edge"]["contact"]
        result = audit_contact(contact, self.geoms)
        self.assertEqual(result["raw_issues"], [])
        self.assertEqual(result["geometry_issues"], [])

    def test_09_changed_fixture_anchor_is_rejected(self):
        path = Path(__file__).parent / "fixtures" / "d1_single_step_native_normal_anomaly.json"
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), FROZEN_FIXTURE_SHA256)
        changed = hashlib.sha256(b"changed fixture").hexdigest()
        with self.assertRaisesRegex(ValueError, "fixture_sha256_changed"):
            verify_anchor_hashes(FROZEN_MANIFEST_SHA256, changed)

    def test_10_aggregate_tamper_is_reported_in_partial_payload(self):
        payload = self.partial_payload()
        payload["native"][0]["contacts"]["wheel_positive_normal_load_n"][0] += 1.0
        result = audit_payload(payload)
        self.assertFalse(result["raw_record_links_valid"])
        self.assertTrue(any("contact_aggregate:wheel_positive_normal_load_n" in issue["code"]
                            for issue in result["raw_issues"]))

    def test_11_partial_payload_cannot_gain_qualification(self):
        result = audit_payload(self.partial_payload())
        self.assertTrue(any("incomplete_frozen_trial" in issue["code"]
                            for issue in result["raw_issues"]))
        self.assertFalse(result["qualification_granted"])
        self.assertFalse(result["rl_gate_open"])
        self.assertFalse(result["original_score_overridden"])

    def test_12_native_qvel_chain_tamper_is_reported(self):
        row = copy.deepcopy(self.fixture["native_samples"][1])
        row["qvel_before"][0] += 0.01
        trace = [{"torque_nm": row["ctrl_nm"]} for _ in range(698)]
        with self.assertRaisesRegex(ValueError, "qvel_chain_broken"):
            audit_native_state(row, 3486, self.fixture["native_samples"][0], trace)


if __name__ == "__main__":
    unittest.main()
