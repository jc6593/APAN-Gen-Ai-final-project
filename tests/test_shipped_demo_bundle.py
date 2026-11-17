"""Smoke tests for the committed twelve-product serving artifact."""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from app.main import _default_bundle_dir
from app.repository import DemoBundleRepository


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_DIR = PROJECT_ROOT / "demo_bundle_v2"


class ShippedDemoBundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repository = DemoBundleRepository(BUNDLE_DIR)

    def test_v2_is_the_default_bundle(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(BUNDLE_DIR, _default_bundle_dir())

    def test_catalog_contract(self) -> None:
        health = self.repository.health()
        products = self.repository.products()["products"]
        reports = [
            self.repository.report(product["product_id"])
            for product in products
        ]

        self.assertEqual("featured_laptops_v2", health["bundle_version"])
        self.assertEqual(12, health["product_count"])
        self.assertEqual(12, len(products))
        self.assertEqual(300, sum(
            report["sample"]["reviews_analyzed"] for report in reports
        ))
        self.assertEqual(
            [20] * 6 + [30] * 6,
            sorted(report["sample"]["reviews_analyzed"] for report in reports),
        )
        self.assertEqual(549, sum(
            report["sample"]["available_review_count"] for report in reports
        ))

    def test_equal_cohort_comparison_has_traceable_evidence(self) -> None:
        comparison = self.repository.compare(
            "acer_aspire5", "dell_inspiron_b077xx31vl"
        )
        self.assertEqual(20, comparison["cohort_size_per_product"])
        self.assertTrue(comparison["aligned_issues"])

        selected = None
        for row in comparison["aligned_issues"]:
            for side, product_id in (
                ("left", "acer_aspire5"),
                ("right", "dell_inspiron_b077xx31vl"),
            ):
                for sentiment in ("positive", "negative"):
                    signal = row[side][sentiment]
                    if signal is not None:
                        selected = (product_id, signal)
                        break
                if selected:
                    break
            if selected:
                break

        self.assertIsNotNone(selected)
        product_id, signal = selected
        evidence = self.repository.evidence(
            product_id,
            signal["finding_id"],
            cohort_size=20,
        )
        self.assertEqual(
            signal["supporting_review_count"],
            evidence["finding"]["supporting_review_count"],
        )
        self.assertTrue(evidence["evidence"])


if __name__ == "__main__":
    unittest.main()
