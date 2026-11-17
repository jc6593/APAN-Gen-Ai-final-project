import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

from reviewlens_ef.cli import run_mode


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TAXONOMY = {"display": {"screen_quality": "display quality"}}
PRODUCT_CONFIG = "configs/products/hp_chromebook_14.json"
CATALOG_CONFIG = "configs/catalogs/laptop_catalog_v1.json"
CATALOG_V2_CONFIG = "configs/catalogs/laptop_catalog_v2.json"
DEMO_CONFIG = "configs/demo/featured_laptops_v1.json"


class LaptopCliDispatchTests(unittest.TestCase):
    def test_catalog_is_read_only_and_returns_available_products(self):
        products = [{
            "product_id": "hp_chromebook_14",
            "display_name": "HP Chromebook 14",
            "catalog_scope": "product_family",
            "available_review_count": 12,
            "config_path": PRODUCT_CONFIG,
        }]
        with mock.patch(
            "reviewlens_ef.laptop_products.list_laptop_products",
            return_value=products,
        ) as catalog:
            result = run_mode(
                "laptop-catalog", PROJECT_ROOT, TAXONOMY, "unused-model"
            )
        catalog.assert_called_once_with(PROJECT_ROOT.resolve())
        self.assertEqual(result, products)

    def test_pilot_defaults_to_ten_and_passes_only_laptop_arguments(self):
        expected = {"product_id": "hp_chromebook_14"}
        with mock.patch(
            "reviewlens_ef.laptop_pipeline.run_laptop_pilot",
            return_value=expected,
        ) as runner:
            result = run_mode(
                "laptop-pilot",
                PROJECT_ROOT,
                TAXONOMY,
                "fake-qwen-4b",
                product_config=PRODUCT_CONFIG,
                show=7,
            )
        self.assertEqual(result, expected)
        runner.assert_called_once_with(
            project_root=PROJECT_ROOT.resolve(),
            product_config=PRODUCT_CONFIG,
            limit=10,
            show=7,
            l2_model_id="fake-qwen-4b",
            max_retries=1,
        )

    def test_summary_is_cache_only_and_uses_requested_prefix(self):
        expected = {"status": "transfer_demo_unvalidated"}
        with mock.patch(
            "reviewlens_ef.laptop_pipeline.summarize_laptop_run",
            return_value=expected,
        ) as summarizer:
            result = run_mode(
                "laptop-summary",
                PROJECT_ROOT,
                TAXONOMY,
                "unused-model",
                product_config=PRODUCT_CONFIG,
                limit=5,
                show=3,
            )
        self.assertEqual(result, expected)
        summarizer.assert_called_once_with(
            project_root=PROJECT_ROOT.resolve(),
            product_config=PRODUCT_CONFIG,
            limit=5,
            show=3,
        )

    def test_catalog_expand_dispatches_data_only_builder(self):
        expected = {"status": "frozen_transfer_demo_catalog"}
        with mock.patch(
            "reviewlens_ef.catalog_collection.run_laptop_catalog_expansion",
            return_value=expected,
        ) as builder:
            result = run_mode(
                "laptop-catalog-expand",
                PROJECT_ROOT,
                TAXONOMY,
                "unused-model",
                catalog_config=CATALOG_CONFIG,
            )
        self.assertEqual(expected, result)
        builder.assert_called_once_with(
            project_root=PROJECT_ROOT.resolve(),
            catalog_config=CATALOG_CONFIG,
        )

    def test_catalog_promote_reuses_local_checkpoint(self):
        expected = {"status": "frozen_transfer_demo_catalog"}
        with mock.patch(
            "reviewlens_ef.catalog_collection.promote_laptop_catalog",
            return_value=expected,
        ) as promoter:
            result = run_mode(
                "laptop-catalog-promote",
                PROJECT_ROOT,
                TAXONOMY,
                "unused-model",
                catalog_config=CATALOG_V2_CONFIG,
                source_catalog_config=CATALOG_CONFIG,
            )
        self.assertEqual(expected, result)
        promoter.assert_called_once_with(
            project_root=PROJECT_ROOT.resolve(),
            source_catalog_config=CATALOG_CONFIG,
            target_catalog_config=CATALOG_V2_CONFIG,
        )

    def test_featured_precompute_uses_frozen_demo_contract(self):
        expected = {"status": "featured_precompute_complete"}
        with mock.patch(
            "reviewlens_ef.laptop_batch.run_featured_precompute",
            return_value=expected,
        ) as runner:
            result = run_mode(
                "laptop-demo-precompute",
                PROJECT_ROOT,
                TAXONOMY,
                "ignored-runtime-model",
                demo_config=DEMO_CONFIG,
                show=2,
            )
        self.assertEqual(expected, result)
        runner.assert_called_once_with(
            project_root=PROJECT_ROOT.resolve(),
            demo_config=DEMO_CONFIG,
            show=2,
        )

    def test_laptop_modes_enforce_config_and_read_only_flags(self):
        with self.assertRaisesRegex(ValueError, "product-config is required"):
            run_mode("laptop-pilot", PROJECT_ROOT, TAXONOMY, "unused-model")
        with self.assertRaisesRegex(ValueError, "not available"):
            run_mode(
                "laptop-summary",
                PROJECT_ROOT,
                TAXONOMY,
                "unused-model",
                product_config=PRODUCT_CONFIG,
                force=True,
            )
        with self.assertRaisesRegex(ValueError, "only for laptop modes"):
            run_mode(
                "ef-summary",
                PROJECT_ROOT,
                TAXONOMY,
                "unused-model",
                product_config=PRODUCT_CONFIG,
            )
        with self.assertRaisesRegex(ValueError, "not available"):
            run_mode(
                "laptop-catalog",
                PROJECT_ROOT,
                TAXONOMY,
                "unused-model",
                limit=1,
            )
        with self.assertRaisesRegex(ValueError, "catalog-config is required"):
            run_mode(
                "laptop-catalog-expand", PROJECT_ROOT, TAXONOMY, "unused-model"
            )
        with self.assertRaisesRegex(ValueError, "only for catalog expansion"):
            run_mode(
                "laptop-catalog",
                PROJECT_ROOT,
                TAXONOMY,
                "unused-model",
                catalog_config=CATALOG_CONFIG,
            )
        with self.assertRaisesRegex(ValueError, "source-catalog-config is required"):
            run_mode(
                "laptop-catalog-promote",
                PROJECT_ROOT,
                TAXONOMY,
                "unused-model",
                catalog_config=CATALOG_V2_CONFIG,
            )
        with self.assertRaisesRegex(ValueError, "only for laptop-catalog-promote"):
            run_mode(
                "laptop-catalog-expand",
                PROJECT_ROOT,
                TAXONOMY,
                "unused-model",
                catalog_config=CATALOG_CONFIG,
                source_catalog_config=CATALOG_CONFIG,
            )
        with self.assertRaisesRegex(ValueError, "demo-config is required"):
            run_mode(
                "laptop-demo-precompute", PROJECT_ROOT, TAXONOMY, "unused-model"
            )
        with self.assertRaisesRegex(ValueError, "fixed by"):
            run_mode(
                "laptop-demo-precompute", PROJECT_ROOT, TAXONOMY, "unused-model",
                demo_config=DEMO_CONFIG, limit=1,
            )
        with self.assertRaisesRegex(ValueError, "only for laptop-demo-precompute"):
            run_mode(
                "laptop-catalog", PROJECT_ROOT, TAXONOMY, "unused-model",
                demo_config=DEMO_CONFIG,
            )

    def test_main_help_exposes_catalog_pilot_summary_and_config(self):
        completed = subprocess.run(
            [sys.executable, "noteboks/15_review_analysis.py", "--help"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        for token in (
            "laptop-catalog", "laptop-catalog-expand", "laptop-catalog-promote",
            "laptop-pilot",
            "laptop-summary", "laptop-demo-precompute", "--product-config",
            "--catalog-config", "--source-catalog-config", "--demo-config"
        ):
            self.assertIn(token, completed.stdout)


if __name__ == "__main__":
    unittest.main()
