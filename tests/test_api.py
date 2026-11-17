"""Portable API tests using a synthetic bundle (no model or project cache)."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Callable

from fastapi.testclient import TestClient

from app.main import create_app
from app.repository import BundleValidationError, DemoBundleRepository


ProductMutator = Callable[[dict[str, Any]], None]
ManifestMutator = Callable[[dict[str, Any]], None]


def _product_document(
    product_id: str,
    display_name: str,
    role: str,
) -> dict[str, Any]:
    return {
        "schema_version": "reviewlens.demo_product.v1",
        "product": {
            "product_id": product_id,
            "display_name": display_name,
            "brand": "Example",
            "operating_system": "TestOS",
            "role": role,
            "status": (
                "demo_cohort_not_separately_evaluated"
                if role == "benchmark"
                else "product_specific_accuracy_not_established"
            ),
        },
        "sample": {
            "reviews_analyzed": 2,
            "available_review_count": 12,
            "rating_distribution": {"1": 0, "2": 0, "3": 1, "4": 0, "5": 1},
        },
        "coverage": {
            "mapped_review_coverage": 0.5,
            "reviews_with_findings": 1,
            "reviews_with_other_verified_signals": 1,
        },
        "findings": [
            {
                "finding_id": f"{product_id}_finding_001",
                "scope": "current_product",
                "category": "performance",
                "issue_group": "slow_startup",
                "sentiment": "negative",
                "supporting_review_count": 1,
                "supporting_review_rate": 0.5,
                "evidence": [
                    {
                        "review_id": f"{product_id}_review_001",
                        "quote": "Startup takes longer than expected.",
                        "rating": 3,
                        "exact_source_verified": True,
                    }
                ],
            }
        ],
        "signals": [
            {
                "signal_id": f"{product_id}_signal_001",
                "review_id": f"{product_id}_review_002",
                "quote": "The screen is easy to read outdoors.",
                "rating": 5,
                "exact_source_verified": True,
                "reason": "No safe taxonomy mapping",
            }
        ],
        "report": {
            "executive_summary": (
                "A directional two-review example with traceable source evidence."
            ),
            "recommended_actions": ["Inspect startup performance before prioritizing."],
        },
        "limitations": [
            "This synthetic fixture is only for API contract verification."
        ],
    }


def _json_bytes(document: dict[str, Any]) -> bytes:
    return json.dumps(
        document,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8") + b"\n"


def _write_bundle(
    root: Path,
    *,
    product_mutator: ProductMutator | None = None,
    manifest_mutator: ManifestMutator | None = None,
) -> Path:
    bundle = root / "demo_bundle"
    products_dir = bundle / "products"
    products_dir.mkdir(parents=True)
    definitions = [
        ("alpha_laptop", "Alpha Laptop", "benchmark"),
        ("beta_laptop", "Beta Laptop", "transfer_demo"),
    ]
    entries: list[dict[str, str]] = []
    for product_id, display_name, role in definitions:
        document = _product_document(product_id, display_name, role)
        if product_mutator is not None and product_id == "alpha_laptop":
            product_mutator(document)
        payload = _json_bytes(document)
        relative_path = f"products/{product_id}.json"
        (bundle / relative_path).write_bytes(payload)
        entries.append(
            {
                "product_id": product_id,
                "file": relative_path,
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    manifest: dict[str, Any] = {
        "schema_version": "reviewlens.demo_bundle.v1",
        "bundle_version": "test-bundle-1",
        "featured_product_ids": [item[0] for item in definitions],
        "products": entries,
        "methodology": {
            "runtime": "frozen_precomputed_results",
            "pipeline": [
                "layer_1_review_filtering",
                "atomic_observation_extraction",
                "constrained_claim_normalization",
                "exact_evidence_verification",
            ],
        },
        "limitations": [
            "Directional convenience samples are not population estimates."
        ],
    }
    if manifest_mutator is not None:
        manifest_mutator(manifest)
    (bundle / "manifest.json").write_bytes(_json_bytes(manifest))
    return bundle


class ReviewLensApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.bundle = _write_bundle(self.root)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_home_health_and_product_catalog(self) -> None:
        with TestClient(create_app(self.bundle)) as client:
            homepage = client.get("/")
            health = client.get("/health")
            products = client.get("/api/products")

        self.assertEqual(homepage.status_code, 200)
        self.assertIn("ReviewLens", homepage.text)
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["status"], "ok")
        self.assertEqual(health.json()["product_count"], 2)
        self.assertRegex(health.json()["bundle_manifest_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(products.status_code, 200)
        self.assertEqual(
            [item["product_id"] for item in products.json()["products"]],
            ["alpha_laptop", "beta_laptop"],
        )

    def test_report_findings_evidence_signals_and_methodology(self) -> None:
        with TestClient(create_app(self.bundle)) as client:
            report = client.get("/api/products/alpha_laptop/report")
            findings = client.get("/api/products/alpha_laptop/findings")
            finding_id = findings.json()["findings"][0]["finding_id"]
            evidence = client.get(
                f"/api/products/alpha_laptop/findings/{finding_id}/evidence"
            )
            signals = client.get("/api/products/alpha_laptop/signals")
            methodology = client.get("/api/methodology")

        self.assertEqual(report.status_code, 200)
        self.assertEqual(report.json()["sample"]["reviews_analyzed"], 2)
        self.assertNotIn("evidence", findings.json()["findings"][0])
        self.assertEqual(findings.json()["findings"][0]["evidence_count"], 1)
        self.assertTrue(evidence.json()["evidence"][0]["exact_source_verified"])
        self.assertEqual(len(signals.json()["signals"]), 1)
        self.assertEqual(
            methodology.json()["schema_version"], "reviewlens.demo_bundle.v1"
        )

    def test_compare_is_equal_sample_and_directional(self) -> None:
        with TestClient(create_app(self.bundle)) as client:
            response = client.post(
                "/api/compare",
                json={
                    "left_product_id": "alpha_laptop",
                    "right_product_id": "beta_laptop",
                },
            )
            same_product = client.post(
                "/api/compare",
                json={
                    "left_product_id": "alpha_laptop",
                    "right_product_id": "alpha_laptop",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["comparison_type"], "directional_equal_sample")
        self.assertEqual(response.json()["cohort_size_per_product"], 2)
        self.assertEqual(same_product.status_code, 422)

    def test_unknown_product_and_finding_return_not_found(self) -> None:
        with TestClient(create_app(self.bundle)) as client:
            missing_product = client.get("/api/products/not_here/report")
            missing_finding = client.get(
                "/api/products/alpha_laptop/findings/not_here/evidence"
            )

        self.assertEqual(missing_product.status_code, 404)
        self.assertEqual(missing_finding.status_code, 404)


class DemoBundleIntegrityTests(unittest.TestCase):
    def test_product_hash_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle = _write_bundle(Path(directory))
            product_path = bundle / "products" / "alpha_laptop.json"
            product = json.loads(product_path.read_text(encoding="utf-8"))
            product["report"]["executive_summary"] = "Tampered after signing"
            product_path.write_bytes(_json_bytes(product))

            with self.assertRaisesRegex(BundleValidationError, "hash mismatch"):
                DemoBundleRepository(bundle)

    def test_wrong_manifest_schema_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle = _write_bundle(
                Path(directory),
                manifest_mutator=lambda manifest: manifest.update(
                    {"schema_version": "reviewlens.demo_bundle.v0"}
                ),
            )
            with self.assertRaisesRegex(BundleValidationError, "schema_version"):
                DemoBundleRepository(bundle)

    def test_path_traversal_is_rejected(self) -> None:
        def escape_bundle(manifest: dict[str, Any]) -> None:
            manifest["products"][0]["file"] = "../outside.json"

        with tempfile.TemporaryDirectory() as directory:
            bundle = _write_bundle(Path(directory), manifest_mutator=escape_bundle)
            with self.assertRaisesRegex(BundleValidationError, "safe relative path"):
                DemoBundleRepository(bundle)

    def test_full_review_text_fields_are_rejected_even_with_valid_hash(self) -> None:
        def add_raw_review(product: dict[str, Any]) -> None:
            product["signals"][0]["raw_text"] = "A full review must never ship."

        with tempfile.TemporaryDirectory() as directory:
            bundle = _write_bundle(Path(directory), product_mutator=add_raw_review)
            with self.assertRaisesRegex(BundleValidationError, "forbidden full-review"):
                DemoBundleRepository(bundle)

    def test_raw_model_trace_fields_are_rejected_even_with_valid_hash(self) -> None:
        def add_model_trace(product: dict[str, Any]) -> None:
            product["signals"][0]["model_result_raw"] = {"hidden": "trace"}

        with tempfile.TemporaryDirectory() as directory:
            bundle = _write_bundle(Path(directory), product_mutator=add_model_trace)
            with self.assertRaisesRegex(BundleValidationError, "forbidden full-review"):
                DemoBundleRepository(bundle)

    def test_service_scope_cannot_enter_product_findings(self) -> None:
        def change_scope(product: dict[str, Any]) -> None:
            product["findings"][0]["scope"] = "purchase_service"

        with tempfile.TemporaryDirectory() as directory:
            bundle = _write_bundle(Path(directory), product_mutator=change_scope)
            with self.assertRaisesRegex(BundleValidationError, "outside current-product"):
                DemoBundleRepository(bundle)

    def test_repository_returns_defensive_copies(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = DemoBundleRepository(_write_bundle(Path(directory)))
            first = repository.report("alpha_laptop")
            first["report"]["executive_summary"] = "Changed by caller"
            second = repository.report("alpha_laptop")

        self.assertNotEqual(
            first["report"]["executive_summary"],
            second["report"]["executive_summary"],
        )


if __name__ == "__main__":
    unittest.main()
