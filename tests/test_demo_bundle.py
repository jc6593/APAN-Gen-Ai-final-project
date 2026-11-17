import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from app.repository import DemoBundleRepository
from reviewlens_ef.demo_bundle import (
    DemoBundleError,
    DemoBundleImmutableError,
    build_demo_bundle,
    verify_demo_bundle,
)


REVIEW_TEXT = "The screen is dim. Service was slow. Overall limited."


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def evidence(quote):
    start = REVIEW_TEXT.index(quote)
    return {
        "sentence_id": f"S_{start}",
        "source_field": "clean_text",
        "quote": quote,
        "quote_start_char_sentence": 0,
        "quote_end_char_sentence": len(quote),
        "quote_start_char_source": start,
        "quote_end_char_source": start + len(quote),
        "quote_start_char_clean": start,
        "quote_end_char_clean": start + len(quote),
        "sentence_text": quote,
    }


def claim(observation_id, mapping_status, attribution, quote, *, category=None,
          issue_group=None, sentiment="negative", text=None):
    return {
        "observation_id": observation_id,
        "mapping_status": mapping_status,
        "category": category,
        "issue_group": issue_group,
        "claim": text or quote,
        "sentiment": sentiment,
        "attribution": attribution,
        "statement_type": "actual_experience",
        "evidence": [evidence(quote)],
    }


def make_fixture(root, sample_sizes=None):
    sample_sizes = tuple(sample_sizes or (30,) * 6)
    featured = []
    results = []
    for product_number, sample_size in enumerate(sample_sizes):
        product_id = f"laptop_{product_number}"
        config_path = root / "configs" / "products" / f"{product_id}.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            json.dumps({"product_id": product_id, "display_name": f"Laptop {product_number}"}),
            encoding="utf-8",
        )
        featured.append(SimpleNamespace(
            config_path=str(config_path.relative_to(root)),
            brand=f"Brand {product_number}",
            operating_system="ChromeOS" if product_number else "Windows 11",
            evaluation_status=(
                "benchmark_source_product" if product_number == 0 else "transfer_demo"
            ),
            cohort_status="frozen_30",
            presentation_name=f"Laptop {product_number}",
            reviews_to_analyze=sample_size,
        ))
        output_dir = root / "outputs" / product_id
        run_dir = output_dir / "evidence_first" / "runs" / "fingerprint"
        review_ids = [
            f"{product_id}_r{number:02d}" for number in range(sample_size)
        ]
        reviews = [
            {
                "review_id": review_id,
                "product_id": product_id,
                "product_name": f"Laptop {product_number}",
                "rating": float((index % 5) + 1),
                "timestamp": f"2026-01-{(index % 28) + 1:02d}",
                "title": "Short title",
                "raw_text": REVIEW_TEXT,
                "clean_text": REVIEW_TEXT,
            }
            for index, review_id in enumerate(review_ids)
        ]
        observations = [
            {
                "product_id": product_id,
                "review_id": review_id,
                "run_fingerprint": "fingerprint",
                "accepted_observations": ([{"observation_id": "O001"}] if index < 5 else []),
                "selected_raw_response": "must never ship",
            }
            for index, review_id in enumerate(review_ids)
        ]
        claims = []
        for index, review_id in enumerate(review_ids):
            accepted = []
            if index in {0, 1}:
                accepted.append(claim(
                    "O001", "mapped", "current_product", "The screen is dim.",
                    category="display", issue_group="screen_quality",
                ))
                if index == 0:
                    accepted.append(claim(
                        "O002", "mapped", "current_product", "The screen is dim.",
                        category="display", issue_group="screen_quality",
                    ))
            elif index == 2:
                accepted.append(claim(
                    "O001", "mapped", "seller_platform", "Service was slow.",
                    category="purchase_service", issue_group="customer_service",
                ))
            elif index == 3:
                accepted.append(claim(
                    "O001", "unmapped", "current_product", "Overall limited.",
                    text="The product has another specific limitation.",
                ))
            elif index == 4:
                accepted.append(claim(
                    "O001", "mapped", "other_product", "The screen is dim.",
                    category="display", issue_group="screen_quality",
                ))
            claims.append({
                "product_id": product_id,
                "review_id": review_id,
                "run_fingerprint": "fingerprint",
                "accepted_claims": accepted,
                "model_result_raw": {"must": "never ship"},
                "attempts": ["must never ship"],
            })
        reviews_path = output_dir / "prepared_reviews.jsonl"
        observations_path = run_dir / "observations.jsonl"
        claims_path = run_dir / "claims.jsonl"
        write_jsonl(reviews_path, reviews)
        write_jsonl(observations_path, observations)
        write_jsonl(claims_path, claims)
        results.append({
            "product_id": product_id,
            "status": "transfer_demo_unvalidated",
            "selected_review_ids": review_ids,
            "summary": {"product_name": f"Laptop {product_number}"},
            "output_dir": str(output_dir),
            "run_dir": str(run_dir),
            "run_fingerprint": "fingerprint",
            "paths": {
                "observations": str(observations_path),
                "claims": str(claims_path),
            },
        })
    demo_path = root / "configs" / "featured_demo.json"
    demo_path.write_text(json.dumps({"demo_id": "final-demo-v1"}), encoding="utf-8")
    demo_hash = hashlib.sha256(demo_path.read_bytes()).hexdigest()
    demo_config = SimpleNamespace(
        demo_id="final-demo-v1",
        reviews_per_product=30,
        l1_model_id="Qwen/Qwen3-1.7B-MLX-4bit",
        l1_revision="l1-frozen",
        l2_model_id="mlx-community/Qwen3-4B-Instruct-2507-4bit",
        l2_revision="l2-frozen",
        products=tuple(featured),
        max_retries=1,
        bundle_output_directory="outputs/demo_bundle",
        config_path=str(demo_path.relative_to(root)),
        config_sha256=demo_hash,
        comparison_reviews_per_product=min(sample_sizes),
    )
    return demo_config, results


class DemoBundleTests(unittest.TestCase):
    def test_builds_strict_model_free_contract_and_partitions_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, results = make_fixture(root)
            built = build_demo_bundle(root, config, results)
            self.assertTrue(built.was_created)
            self.assertEqual("final-demo-v1", built.bundle_version)
            manifest = json.loads(built.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual("reviewlens.demo_bundle.v1", manifest["schema_version"])
            self.assertEqual(6, len(manifest["featured_product_ids"]))
            self.assertEqual(
                manifest["featured_product_ids"],
                [item["product_id"] for item in manifest["products"]],
            )
            product = json.loads(built.product_paths[0].read_text(encoding="utf-8"))
            self.assertEqual("reviewlens.demo_product.v1", product["schema_version"])
            self.assertEqual("benchmark", product["product"]["role"])
            self.assertEqual("frozen_30", product["product"]["status"])
            self.assertEqual(30, product["sample"]["reviews_analyzed"])
            self.assertEqual(30, product["sample"]["available_review_count"])
            self.assertEqual(30, sum(product["sample"]["rating_distribution"].values()))
            self.assertEqual(0.066667, product["coverage"]["mapped_review_coverage"])

            # Three atomic claims collapse to two unique review supporters.
            self.assertEqual(1, len(product["findings"]))
            finding = product["findings"][0]
            self.assertEqual("current_product", finding["scope"])
            self.assertEqual(2, finding["supporting_review_count"])
            self.assertEqual(round(2 / 30, 6), finding["supporting_review_rate"])
            self.assertEqual(2, len(finding["supporting_review_ids"]))
            self.assertEqual(1, len(product["report"]["service_context"]))
            self.assertEqual("purchase_service", product["report"]["service_context"][0]["scope"])
            self.assertEqual(1, len(product["signals"]))
            self.assertTrue(product["signals"][0]["exact_source_verified"])
            self.assertEqual(1.0, finding["evidence"][0]["rating"])
            self.assertTrue(finding["evidence"][0]["exact_source_verified"])

            repository = DemoBundleRepository(built.output_directory)
            self.assertEqual(6, repository.health()["product_count"])
            self.assertEqual(
                0.066667,
                repository.findings("laptop_0")["findings"][0][
                    "supporting_review_rate"
                ],
            )

            serialized = built.product_paths[0].read_text(encoding="utf-8")
            for forbidden in (
                '"raw_text"', '"clean_text":', '"raw_response"',
                '"model_result_raw"', '"attempts"', REVIEW_TEXT,
            ):
                self.assertNotIn(forbidden, serialized)

    def test_verify_and_second_build_are_deterministic_and_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, results = make_fixture(root)
            first = build_demo_bundle(root, config, results)
            verified = verify_demo_bundle(
                root, output_directory=config.bundle_output_directory
            )
            second = build_demo_bundle(root, config, results)
            self.assertEqual(first.bundle_id, verified.bundle_id)
            self.assertEqual(first.bundle_id, second.bundle_id)
            self.assertFalse(second.was_created)
            self.assertEqual(
                first.product_paths[0].read_bytes(), second.product_paths[0].read_bytes()
            )

    def test_variable_report_samples_keep_a_smaller_equal_comparison_cohort(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, results = make_fixture(root, sample_sizes=(30, 20))
            built = build_demo_bundle(root, config, results)
            manifest = json.loads(built.manifest_path.read_text(encoding="utf-8"))

            self.assertEqual(2, manifest["methodology"]["featured_product_count"])
            self.assertIsNone(manifest["methodology"]["reviews_per_product"])
            self.assertEqual(
                {"laptop_0": 30, "laptop_1": 20},
                manifest["methodology"]["reviews_per_product_by_id"],
            )
            self.assertEqual(
                20, manifest["methodology"]["comparison_reviews_per_product"]
            )
            self.assertEqual(
                20,
                json.loads(built.product_paths[1].read_text(encoding="utf-8"))[
                    "sample"
                ]["reviews_analyzed"],
            )
            verify_demo_bundle(root, output_directory=config.bundle_output_directory)

    def test_finding_evidence_prefers_distinct_supporting_reviews(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, results = make_fixture(root, sample_sizes=(3, 3))
            claims_path = Path(results[0]["paths"]["claims"])
            rows = [json.loads(line) for line in claims_path.read_text().splitlines()]
            rows[0]["accepted_claims"].append(claim(
                "O003", "mapped", "current_product", "Service was slow.",
                category="display", issue_group="screen_quality",
            ))
            write_jsonl(claims_path, rows)

            built = build_demo_bundle(root, config, results)
            product = json.loads(built.product_paths[0].read_text(encoding="utf-8"))
            evidence_rows = product["findings"][0]["evidence"]

            self.assertEqual(3, len(evidence_rows))
            self.assertEqual(
                ["laptop_0_r00", "laptop_0_r01"],
                [item["review_id"] for item in evidence_rows[:2]],
            )

    def test_tampering_is_detected_and_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, results = make_fixture(root)
            built = build_demo_bundle(root, config, results)
            product = built.product_paths[0]
            product.write_bytes(product.read_bytes() + b" ")
            with self.assertRaisesRegex(DemoBundleError, "hash mismatch"):
                verify_demo_bundle(root, output_directory=config.bundle_output_directory)
            with self.assertRaises(DemoBundleImmutableError):
                build_demo_bundle(root, config, results)

    def test_inexact_source_offset_fails_before_bundle_write(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, results = make_fixture(root)
            claims_path = Path(results[0]["paths"]["claims"])
            rows = [json.loads(line) for line in claims_path.read_text().splitlines()]
            rows[0]["accepted_claims"][0]["evidence"][0]["quote_start_char_source"] = 1
            write_jsonl(claims_path, rows)
            with self.assertRaisesRegex(DemoBundleError, "offsets"):
                build_demo_bundle(root, config, results)
            self.assertFalse((root / config.bundle_output_directory).exists())

    def test_requires_at_least_two_products_and_matching_product_cohorts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, results = make_fixture(root)
            short_config = SimpleNamespace(**{
                **vars(config), "products": config.products[:1],
            })
            with self.assertRaisesRegex(DemoBundleError, "2-16"):
                build_demo_bundle(root, short_config, results[:1])
            mismatched_product = SimpleNamespace(**{
                **vars(config.products[0]), "reviews_to_analyze": 29,
            })
            wrong_cohort = SimpleNamespace(**{
                **vars(config),
                "products": (mismatched_product, *config.products[1:]),
            })
            with self.assertRaisesRegex(DemoBundleError, "expected 29"):
                build_demo_bundle(root, wrong_cohort, results)


if __name__ == "__main__":
    unittest.main()
