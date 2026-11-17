import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from reviewlens_ef.catalog_collection import (
    CatalogBuildError,
    CatalogImmutableOutputError,
    _iter_remote_jsonl,
    build_candidate_plan,
    classify_laptop_title,
    collect_from_records,
    freeze_catalog,
    load_catalog_build_config,
    promote_laptop_catalog,
    run_laptop_catalog_expansion,
)
from reviewlens_ef.laptop_products import list_laptop_products


ROOT = Path(__file__).resolve().parents[1]


class LaptopTitleClassifierTests(unittest.TestCase):
    def test_accepts_complete_devices_and_rejects_accessories(self):
        devices = [
            "Apple 2022 MacBook Air Laptop with M2 chip, 13.6-inch display, 8GB RAM, 512GB SSD",
            "Acer Chromebook 314 Laptop, Intel Celeron, 14-inch display, 64GB eMMC, Chrome OS",
            "Lenovo IdeaPad 3 Laptop, Ryzen 5, 15.6-inch FHD, 8GB RAM, 256GB SSD, Windows 11",
        ]
        accessories = [
            "SOOPII USB C Cable for MacBook Pro Laptop 100W",
            "Incipio Feather for 13-Inch MacBook Pro Retina - Black",
            "A-Tech 16GB SODIMM Laptop Memory Kit for Dell Notebook",
            "Protective Case Compatible with Microsoft Surface Laptop 3 13.5-inch",
        ]
        self.assertTrue(all(classify_laptop_title(title) for title in devices))
        self.assertTrue(all(classify_laptop_title(title) is None for title in accessories))
        renewed = (
            "Dell Latitude Laptop, Intel Core i5, 14-inch display, 8GB RAM, "
            "256GB SSD, Windows 10 (Renewed)"
        )
        self.assertIsNone(classify_laptop_title(renewed))
        self.assertEqual(
            "renewed", classify_laptop_title(renewed, include_renewed=True)["condition"]
        )


class CatalogCollectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "configs/catalogs").mkdir(parents=True)
        (self.root / "configs/products").mkdir(parents=True)
        (self.root / "configs/taxonomies").mkdir(parents=True)
        (self.root / "data").mkdir()
        (self.root / "configs/taxonomies/laptop_v1.json").write_bytes(
            (ROOT / "configs/taxonomies/laptop_v1.json").read_bytes()
        )
        self.candidates = [
            self._candidate(
                "B000000001", 900, "Apple MacBook Air Laptop M2, 13-inch Retina display, 8GB RAM, 256GB SSD"
            ),
            self._candidate(
                "B000000002", 800, "Acer Chromebook 314 Laptop, Intel Celeron, 14-inch display, 64GB eMMC, Chrome OS"
            ),
            self._candidate(
                "B000000003", 700, "Lenovo IdeaPad 3 Laptop, Ryzen 5, 15-inch display, 8GB RAM, 256GB SSD, Windows 11"
            ),
            self._candidate(
                "B000000004", 600, "Dell Inspiron Laptop, Intel Core i5, 15-inch display, 8GB RAM, 512GB SSD, Windows 11"
            ),
            self._candidate(
                "B000000005", 500, "ASUS VivoBook Laptop, Intel Core i3, 14-inch display, 8GB RAM, 256GB SSD, Windows 11"
            ),
            self._candidate(
                "B000000006", 400, "HP Pavilion Laptop, Intel Core i5, 15-inch display, 8GB RAM, 256GB SSD, Windows 10"
            ),
            self._candidate(
                "B000000007", 300, "USB C Cable Compatible with MacBook Laptop, 100W charging"
            ),
        ]
        self._write_jsonl(self.root / "data/candidates.jsonl", self.candidates)
        self.config_document = {
            "schema_version": "reviewlens.laptop_catalog_build.v1",
            "catalog_id": "fixture_catalog_v1",
            "category": "laptop",
            "source": {
                "candidate_metadata_jsonl": "data/candidates.jsonl",
                "review_repository": "fixture/reviews",
                "review_jsonl_path": "reviews.jsonl",
            },
            "selection": {
                "target_new_products": 4,
                "candidate_pool_size": 6,
                "minimum_reviews_per_product": 2,
                "maximum_reviews_per_product": 3,
                "minimum_text_characters": 40,
                "maximum_scan_records": 100,
                "checkpoint_interval_records": 5,
                "maximum_products_per_brand": 1,
                "minimum_distinct_brands": 4,
                "required_operating_systems": ["macos", "chromeos"],
                "seed": 7,
                "include_renewed": False,
                "excluded_parent_asins": [],
            },
            "output": {
                "directory": "data/fixture_catalog_v1",
                "generated_config_directory": "configs/products/fixture_catalog_v1",
                "generated_default_limit": 2,
            },
        }
        self.config_path = self.root / "configs/catalogs/fixture.json"
        self.config_path.write_text(json.dumps(self.config_document, indent=2))
        self.config = load_catalog_build_config(
            self.root, "configs/catalogs/fixture.json"
        )
        self.plan = build_candidate_plan(self.root, self.config)

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def _candidate(parent_asin, rating_number, title):
        return {
            "parent_asin": parent_asin,
            "title": title,
            "average_rating": 4.2,
            "rating_number": rating_number,
        }

    @staticmethod
    def _write_jsonl(path, rows):
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )

    def _review_stream(self):
        rows = []
        line_number = 0
        for round_number in range(1, 4):
            for candidate in self.plan["candidate_pool"]:
                line_number += 1
                rows.append(
                    (
                        line_number,
                        {
                            "parent_asin": candidate["parent_asin"],
                            "asin": f"C{line_number:09d}",
                            "rating": float((round_number - 1) % 5 + 1),
                            "title": f"Review {line_number}",
                            "text": "This is a sufficiently detailed laptop review for collection testing.",
                            "timestamp": line_number,
                            "verified_purchase": True,
                            "helpful_vote": 0,
                        },
                    )
                )
        return rows

    def test_plan_filters_accessories_and_records_bounded_scope(self):
        asins = {row["parent_asin"] for row in self.plan["candidate_pool"]}
        self.assertNotIn("B000000007", asins)
        self.assertEqual(6, len(asins))
        self.assertIn("bounded metadata", self.plan["scope_note"])
        self.assertEqual(1, self.plan["rejected_counts"]["not_high_confidence_laptop"])

    def test_collect_freeze_is_diverse_capped_and_idempotent(self):
        state = collect_from_records(
            self.config,
            self.plan,
            self._review_stream(),
            source_revision="fixture-sha",
        )
        self.assertEqual("ready_to_freeze", state["status"])
        self.assertEqual(4, len(state["selected_parent_asins"]))
        manifest = freeze_catalog(self.root, self.config, self.plan, state)
        self.assertEqual(4, len(manifest["products"]))
        self.assertEqual(4, len({row["brand"] for row in manifest["products"]}))
        for product in manifest["products"]:
            self.assertEqual(2, product["collected_review_count"])
            review_path = self.root / product["reviews_path"]
            config_path = self.root / product["config_path"]
            self.assertTrue(review_path.is_file())
            self.assertTrue(config_path.is_file())
            generated = json.loads(config_path.read_text())
            self.assertEqual(2, generated["selection"]["default_limit"])
            self.assertEqual(
                sorted(generated["product_family"]["allowed_child_asins"]),
                generated["product_family"]["allowed_child_asins"],
            )
        self.assertTrue(
            (self.root / "configs/products/fixture_catalog_v1/.published").is_file()
        )
        self.assertEqual(
            {row["product_id"] for row in manifest["products"]},
            {row["product_id"] for row in list_laptop_products(self.root)},
        )
        second = freeze_catalog(self.root, self.config, self.plan, state)
        self.assertEqual(manifest, second)
        verified = run_laptop_catalog_expansion(
            self.root, "configs/catalogs/fixture.json"
        )
        self.assertEqual(manifest, verified)

    def test_promotes_verified_checkpoint_without_remote_scan(self):
        state = collect_from_records(
            self.config,
            self.plan,
            self._review_stream(),
            source_revision="fixture-sha",
        )
        source_output = self.root / "data/fixture_catalog_v1"
        source_output.mkdir(parents=True, exist_ok=True)
        (source_output / "candidate_plan.json").write_text(
            json.dumps(
                self.plan, ensure_ascii=False, indent=2, sort_keys=True
            ) + "\n",
            encoding="utf-8",
        )
        source_manifest = freeze_catalog(
            self.root, self.config, self.plan, state
        )
        source_manifest_bytes = (
            source_output / "catalog_manifest.json"
        ).read_bytes()

        target = json.loads(json.dumps(self.config_document))
        target["catalog_id"] = "fixture_catalog_v2"
        target["selection"].update({
            "target_new_products": 5,
            "minimum_reviews_per_product": 1,
            "maximum_products_per_brand": 1,
            "minimum_distinct_brands": 5,
        })
        target["output"] = {
            "directory": "data/fixture_catalog_v2",
            "generated_config_directory": "configs/products/fixture_catalog_v2",
            "generated_default_limit": 1,
        }
        target_path = self.root / "configs/catalogs/fixture_v2.json"
        target_path.write_text(json.dumps(target, indent=2), encoding="utf-8")

        promoted = promote_laptop_catalog(
            self.root,
            "configs/catalogs/fixture.json",
            "configs/catalogs/fixture_v2.json",
        )
        self.assertEqual("fixture_catalog_v2", promoted["catalog_id"])
        self.assertEqual(5, len(promoted["products"]))
        self.assertEqual(
            5, len({row["brand"] for row in promoted["products"]})
        )
        self.assertEqual(
            source_manifest_bytes,
            (source_output / "catalog_manifest.json").read_bytes(),
        )
        self.assertEqual(
            [row["product_id"] for row in source_manifest["products"]],
            [
                row["product_id"]
                for row in json.loads(source_manifest_bytes)["products"]
            ],
        )
        second = promote_laptop_catalog(
            self.root,
            "configs/catalogs/fixture.json",
            "configs/catalogs/fixture_v2.json",
        )
        self.assertEqual(promoted, second)

    def test_promotion_rejects_tampered_source_checkpoint(self):
        state = collect_from_records(
            self.config,
            self.plan,
            self._review_stream(),
            source_revision="fixture-sha",
        )
        source_output = self.root / "data/fixture_catalog_v1"
        source_output.mkdir(parents=True, exist_ok=True)
        (source_output / "candidate_plan.json").write_text(
            json.dumps(
                self.plan, ensure_ascii=False, indent=2, sort_keys=True
            ) + "\n",
            encoding="utf-8",
        )
        freeze_catalog(self.root, self.config, self.plan, state)
        checkpoint_path = source_output / "collection_checkpoint.json"
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        first_asin = next(iter(checkpoint["collected"]))
        checkpoint["collected"][first_asin][0]["rating"] = 99
        checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")

        target = json.loads(json.dumps(self.config_document))
        target["catalog_id"] = "fixture_catalog_v2"
        target["selection"].update({
            "target_new_products": 5,
            "minimum_reviews_per_product": 1,
            "maximum_products_per_brand": 1,
            "minimum_distinct_brands": 5,
        })
        target["output"] = {
            "directory": "data/fixture_catalog_v2",
            "generated_config_directory": "configs/products/fixture_catalog_v2",
            "generated_default_limit": 1,
        }
        (self.root / "configs/catalogs/fixture_v2.json").write_text(
            json.dumps(target, indent=2), encoding="utf-8"
        )
        with self.assertRaises(CatalogBuildError):
            promote_laptop_catalog(
                self.root,
                "configs/catalogs/fixture.json",
                "configs/catalogs/fixture_v2.json",
            )

    def test_resume_skips_prior_lines_and_tamper_is_rejected(self):
        records = self._review_stream()
        partial = collect_from_records(
            self.config,
            self.plan,
            records[:6],
            source_revision="fixture-sha",
        )
        resumed = collect_from_records(
            self.config,
            self.plan,
            records,
            source_revision="fixture-sha",
            state=partial,
        )
        manifest = freeze_catalog(self.root, self.config, self.plan, resumed)
        first = self.root / manifest["products"][0]["reviews_path"]
        first.write_text("tampered\n")
        with self.assertRaises(CatalogImmutableOutputError):
            freeze_catalog(self.root, self.config, self.plan, resumed)

    def test_ready_checkpoint_freezes_same_selection_without_consuming_stream(self):
        ready = collect_from_records(
            self.config,
            self.plan,
            self._review_stream(),
            source_revision="fixture-sha",
        )

        def must_not_read():
            raise AssertionError("ready checkpoint should not consume the source")
            yield

        resumed = collect_from_records(
            self.config,
            self.plan,
            must_not_read(),
            source_revision="fixture-sha",
            state=ready,
        )
        self.assertEqual(ready["selected_parent_asins"], resumed["selected_parent_asins"])

        config_dir = self.root / "configs/products/fixture_catalog_v1"
        config_dir.mkdir(parents=True, exist_ok=True)
        (config_dir / "unexpected.json").write_text("{}")
        with self.assertRaises(CatalogImmutableOutputError):
            freeze_catalog(self.root, self.config, self.plan, resumed)

    def test_tampered_checkpoint_and_manifest_fail_closed(self):
        records = self._review_stream()
        partial = collect_from_records(
            self.config,
            self.plan,
            records[:6],
            source_revision="fixture-sha",
        )
        first_asin = next(iter(partial["collected"]))
        partial["collected"][first_asin][0]["rating"] = 99
        with self.assertRaises(CatalogBuildError):
            collect_from_records(
                self.config,
                self.plan,
                records,
                source_revision="fixture-sha",
                state=partial,
            )

        complete = collect_from_records(
            self.config,
            self.plan,
            records,
            source_revision="fixture-sha",
        )
        freeze_catalog(self.root, self.config, self.plan, complete)
        manifest_path = self.root / "data/fixture_catalog_v1/catalog_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["products"] = []
        manifest_path.write_text(json.dumps(manifest, indent=2))
        with self.assertRaises(CatalogImmutableOutputError):
            run_laptop_catalog_expansion(
                self.root, "configs/catalogs/fixture.json"
            )

    def test_invalid_config_and_insufficient_stream_fail_closed(self):
        bad = json.loads(json.dumps(self.config_document))
        bad["selection"]["minimum_reviews_per_product"] = 4
        bad["selection"]["maximum_reviews_per_product"] = 3
        self.config_path.write_text(json.dumps(bad))
        with self.assertRaises(CatalogBuildError):
            load_catalog_build_config(self.root, "configs/catalogs/fixture.json")

        sparse = collect_from_records(
            self.config,
            self.plan,
            self._review_stream()[:3],
            source_revision="fixture-sha",
        )
        with self.assertRaises(CatalogBuildError):
            freeze_catalog(self.root, self.config, self.plan, sparse)

    def test_remote_binary_offset_resumes_at_next_record(self):
        payload = b"".join(
            json.dumps({"parent_asin": f"B{index:09d}"}).encode() + b"\n"
            for index in range(1, 4)
        )

        class FakeFileSystem:
            def open(self, path, mode):
                self.path = path
                self.mode = mode
                return io.BytesIO(payload)

        with mock.patch("huggingface_hub.HfFileSystem", return_value=FakeFileSystem()):
            first_pass = list(_iter_remote_jsonl(self.config, "fixture-sha"))
            resumed = list(
                _iter_remote_jsonl(
                    self.config,
                    "fixture-sha",
                    start_after=1,
                    source_byte_offset=first_pass[0][2],
                )
            )
        self.assertEqual([1, 2, 3], [row[0] for row in first_pass])
        self.assertEqual([2, 3], [row[0] for row in resumed])
        self.assertEqual("B000000002", resumed[0][1]["parent_asin"])


if __name__ == "__main__":
    unittest.main()
