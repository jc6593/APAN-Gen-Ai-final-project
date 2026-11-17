import ast
import json
import tempfile
import unittest
from pathlib import Path

from reviewlens_ef.laptop_products import (
    ImmutablePreparedOutputError,
    LaptopConfigError,
    list_laptop_products,
    load_laptop_product_config,
    load_laptop_taxonomy,
    prepare_laptop_product,
    select_product_family_reviews,
)


ROOT = Path(__file__).resolve().parents[1]


class LaptopProductRealConfigTests(unittest.TestCase):
    def test_laptop_taxonomy_exactly_matches_frozen_acer_taxonomy(self):
        tree = ast.parse((ROOT / "noteboks/15_review_analysis.py").read_text())
        frozen = None
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "ISSUE_GROUPS"
                for target in node.targets
            ):
                frozen = ast.literal_eval(node.value)
                break
        self.assertIsNotNone(frozen)
        self.assertEqual(
            frozen,
            load_laptop_taxonomy(ROOT, "configs/taxonomies/laptop_v1.json"),
        )

    def test_catalog_is_read_only_and_has_expected_local_cohorts(self):
        products = {row["product_id"]: row for row in list_laptop_products(ROOT)}
        self.assertTrue(
            {"acer_aspire5", "hp_chromebook_14", "macbook_pro_2019"}
            <= set(products)
        )
        self.assertEqual(189, products["acer_aspire5"]["available_review_count"])
        self.assertEqual(12, products["hp_chromebook_14"]["available_review_count"])
        self.assertEqual(20, products["macbook_pro_2019"]["available_review_count"])
        self.assertEqual("product_family", products["hp_chromebook_14"]["catalog_scope"])
        self.assertEqual("product_family", products["macbook_pro_2019"]["catalog_scope"])

    def test_hp_frozen_pilot_prefix_covers_ratings_one_through_five(self):
        config = load_laptop_product_config(
            ROOT, "configs/products/hp_chromebook_14.json"
        )
        rows = select_product_family_reviews(ROOT, config, limit=5)
        self.assertEqual(
            [
                "electronics_99975",
                "electronics_274208",
                "electronics_1202809",
                "electronics_1831057",
                "electronics_8883",
            ],
            [row["review_id"] for row in rows],
        )
        self.assertEqual([1, 2, 3, 4, 5], [int(row["rating"]) for row in rows])
        self.assertEqual(
            {"B099M5J8MP", "B09SVSBVP1"},
            set(config.allowed_child_asins),
        )
        self.assertEqual(64, len(config.config_sha256))
        self.assertEqual(64, len(config.taxonomy_sha256))
        taxonomy = load_laptop_taxonomy(ROOT, config)
        self.assertIn("battery", taxonomy)
        self.assertIn("battery_runtime", taxonomy["battery"])


class LaptopProductPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "configs/products").mkdir(parents=True)
        (self.root / "configs/taxonomies").mkdir(parents=True)
        (self.root / "data").mkdir()
        taxonomy_source = ROOT / "configs/taxonomies/laptop_v1.json"
        (self.root / "configs/taxonomies/laptop_v1.json").write_bytes(
            taxonomy_source.read_bytes()
        )
        self.rows = []
        for rating in range(1, 6):
            self.rows.append(
                {
                    "review_id": f"r{rating}",
                    "parent_asin": "B000000001",
                    "product_name": "Fixture Laptop",
                    "asin": "B000000101" if rating % 2 else "B000000102",
                    "rating": float(rating),
                    "title": f"<b>Title {rating}</b>",
                    "text": f"<p>Battery {rating} is good.</p><div>Screen works!</div>",
                    "timestamp": rating,
                    "verified_purchase": True,
                    "helpful_vote": 0,
                }
            )
        self.rows.extend(
            [
                {
                    "review_id": "r6",
                    "parent_asin": "B000000001",
                    "product_name": "Fixture Laptop",
                    "asin": "B000000101",
                    "rating": 5.0,
                    "title": "Extra",
                    "text": "Extra review.",
                },
                {
                    "review_id": "ignored",
                    "parent_asin": "B000000099",
                    "product_name": "Other",
                    "asin": "B000000999",
                    "rating": 1.0,
                    "title": "Other",
                    "text": "Other product.",
                },
            ]
        )
        self._write_jsonl(self.root / "data/reviews.jsonl", self.rows)
        self.config = {
            "schema_version": "reviewlens.product_config.v1",
            "product_id": "fixture_laptop",
            "category": "laptop",
            "catalog_scope": "product_family",
            "product_family": {
                "display_name": "Fixture Laptop",
                "model_label": "Fixture family",
                "parent_asin": "B000000001",
                "allowed_child_asins": ["B000000101", "B000000102"],
            },
            "source": {"reviews_jsonl": "data/reviews.jsonl"},
            "taxonomy": {
                "config_path": "configs/taxonomies/laptop_v1.json",
                "schema_version": "reviewlens.taxonomy.laptop.v1",
            },
            "selection": {
                "strategy": "rating_round_robin_v1",
                "seed": 7,
                "default_limit": 5,
                "pilot_review_ids": ["r1", "r2", "r3", "r4", "r5"],
            },
            "output": {"directory": "outputs/laptops/fixture_laptop"},
        }
        self._write_config(self.config)
        frozen = self.root / "outputs/amazon/evidence_first/frozen.txt"
        frozen.parent.mkdir(parents=True)
        frozen.write_text("untouched", encoding="utf-8")
        self.frozen = frozen

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def _write_jsonl(path, rows):
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )

    def _write_config(self, document):
        (self.root / "configs/products/fixture_laptop.json").write_text(
            json.dumps(document, indent=2), encoding="utf-8"
        )

    def test_prepare_is_exact_idempotent_and_isolated(self):
        result = prepare_laptop_product(
            self.root, "configs/products/fixture_laptop.json"
        )
        self.assertTrue(result.was_created)
        self.assertEqual(("r1", "r2", "r3", "r4", "r5"), result.selected_review_ids)
        self.assertEqual(
            (self.root / "outputs/laptops/fixture_laptop").resolve(), result.output_dir
        )
        reviews = [json.loads(line) for line in result.reviews_path.read_text().splitlines()]
        sentences = [
            json.loads(line) for line in result.sentences_path.read_text().splitlines()
        ]
        titles = [json.loads(line) for line in result.titles_path.read_text().splitlines()]
        self.assertEqual(self.rows[0]["text"], reviews[0]["raw_text"])
        self.assertEqual(self.rows[0]["title"], reviews[0]["raw_title"])
        self.assertEqual("Title 1", reviews[0]["title"])
        self.assertEqual("Battery 1 is good.\nScreen works!", reviews[0]["clean_text"])
        review_by_id = {row["review_id"]: row for row in reviews}
        for sentence in sentences:
            source = review_by_id[sentence["review_id"]]["clean_text"]
            self.assertEqual(
                sentence["sentence_text"],
                source[sentence["start_char"] : sentence["end_char"]],
            )
            self.assertEqual("clean_text", sentence["source_field"])
        self.assertTrue(all(row["sentence_id"].endswith("_TITLE") for row in titles))
        self.assertTrue(all(row["source_field"] == "title" for row in titles))
        manifest = json.loads(result.manifest_path.read_text())
        self.assertEqual(list(result.selected_review_ids), manifest["selection"]["selected_review_ids"])
        self.assertEqual(6, manifest["selection"]["available_review_count"])
        self.assertIn("no statistical balance", manifest["selection"]["sampling_note"])
        self.assertEqual("untouched", self.frozen.read_text())

        second = prepare_laptop_product(
            self.root, "configs/products/fixture_laptop.json"
        )
        self.assertFalse(second.was_created)
        self.assertEqual(result.selected_review_ids, second.selected_review_ids)

    def test_tampered_output_is_never_overwritten(self):
        result = prepare_laptop_product(
            self.root, "configs/products/fixture_laptop.json"
        )
        result.reviews_path.write_text("tampered\n", encoding="utf-8")
        with self.assertRaises(ImmutablePreparedOutputError):
            prepare_laptop_product(
                self.root, "configs/products/fixture_laptop.json"
            )
        self.assertEqual("tampered\n", result.reviews_path.read_text())

    def test_rejects_unknown_keys_absolute_paths_and_traversal(self):
        bad = json.loads(json.dumps(self.config))
        bad["unknown"] = True
        self._write_config(bad)
        with self.assertRaises(LaptopConfigError):
            load_laptop_product_config(
                self.root, "configs/products/fixture_laptop.json"
            )

        self._write_config(self.config)
        with self.assertRaises(LaptopConfigError):
            load_laptop_product_config(
                self.root, str(self.root / "configs/products/fixture_laptop.json")
            )

        bad = json.loads(json.dumps(self.config))
        bad["source"]["reviews_jsonl"] = "../outside.jsonl"
        self._write_config(bad)
        with self.assertRaises(LaptopConfigError):
            load_laptop_product_config(
                self.root, "configs/products/fixture_laptop.json"
            )

    def test_rejects_child_asin_not_allowed_by_config(self):
        self.rows[0]["asin"] = "B000000777"
        self._write_jsonl(self.root / "data/reviews.jsonl", self.rows)
        config = load_laptop_product_config(
            self.root, "configs/products/fixture_laptop.json"
        )
        with self.assertRaisesRegex(LaptopConfigError, "not allowed"):
            select_product_family_reviews(self.root, config)


if __name__ == "__main__":
    unittest.main()
