import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from reviewlens_ef import laptop_batch, laptop_products
from reviewlens_ef.laptop_batch import FeaturedDemoConfig, FeaturedProduct
from reviewlens_ef.laptop_pipeline import run_laptop_pilot


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _prepared_fixture(root):
    product_id = "test_laptop"
    review_id = "review_1"
    text = "The screen is clear and bright."
    title = "Clear screen"
    output = root / "outputs/laptops" / product_id
    reviews_path = output / "prepared_reviews.jsonl"
    sentences_path = output / "sentences.jsonl"
    titles_path = output / "titles.jsonl"
    manifest_path = output / "prepare_manifest.json"
    _write_jsonl(
        reviews_path,
        [{
            "review_id": review_id,
            "product_id": product_id,
            "product_name": "Test Laptop",
            "product_model": "Test 14",
            "asin": "CHILD00001",
            "parent_asin": "PARENT0001",
            "rating": 5.0,
            "title": title,
            "raw_text": text,
            "clean_text": text,
        }],
    )
    _write_jsonl(
        sentences_path,
        [{
            "sentence_id": f"{review_id}_S001",
            "review_id": review_id,
            "source_field": "clean_text",
            "sentence_text": text,
            "start_char": 0,
            "end_char": len(text),
            "rating": 5.0,
        }],
    )
    _write_jsonl(
        titles_path,
        [{
            "sentence_id": f"{review_id}_TITLE",
            "review_id": review_id,
            "source_field": "title",
            "sentence_text": title,
            "start_char": 0,
            "end_char": len(title),
        }],
    )
    manifest_path.write_text(
        json.dumps({"selected_review_ids": [review_id]}), encoding="utf-8"
    )
    return SimpleNamespace(
        product_id=product_id,
        output_dir=output,
        reviews_path=reviews_path,
        sentences_path=sentences_path,
        titles_path=titles_path,
        manifest_path=manifest_path,
        selected_review_ids=(review_id,),
        config_sha256="config-hash",
        taxonomy_sha256="taxonomy-hash",
    )


class _FakeLayer1:
    def __init__(self):
        self.calls = 0
        self.closed = False

    def __call__(self, prompt, sentence):
        self.calls += 1
        return json.dumps([{"term": "screen", "sentiment": "positive"}])

    def close(self):
        self.closed = True


class _TrackedModel:
    def __init__(self, name, events):
        self.name = name
        self.events = events
        self.calls = 0
        self.closed = 0

    def __call__(self, *args):
        self.calls += 1
        self.events.append(f"{self.name}:infer")
        return "unused"

    def close(self):
        self.closed += 1
        self.events.append(f"{self.name}:close")


class LaptopStageTests(unittest.TestCase):
    def _product_patches(self, prepared):
        config = SimpleNamespace(config_sha256="config-hash")
        taxonomy = {"display": {"screen_quality": "screen clarity"}}
        return (
            mock.patch.object(
                laptop_products, "load_laptop_product_config", return_value=config
            ),
            mock.patch.object(
                laptop_products, "prepare_laptop_product", return_value=prepared
            ),
            mock.patch.object(
                laptop_products, "load_laptop_taxonomy", return_value=taxonomy
            ),
        )

    def test_layer1_stage_returns_before_layer2_and_leaves_shared_model_open(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = _prepared_fixture(root)
            layer1 = _FakeLayer1()
            forbidden_layer2 = mock.Mock(
                side_effect=AssertionError("Layer 2 must not run in the Layer 1 stage")
            )
            product_config, product_prepare, taxonomy = self._product_patches(prepared)
            with product_config, product_prepare, taxonomy:
                result = run_laptop_pilot(
                    root,
                    "configs/products/test.json",
                    limit=1,
                    show=0,
                    max_retries=0,
                    layer1_generator=layer1,
                    generator=forbidden_layer2,
                    layer1_system_prompt="SemEval aspect extraction",
                    stage="layer1",
                    manage_model_lifecycle=False,
                )

            self.assertEqual(result["stage"], "layer1_complete")
            self.assertEqual(layer1.calls, 1)
            self.assertFalse(layer1.closed)
            forbidden_layer2.assert_not_called()
            self.assertTrue(Path(result["paths"]["layer1"]).is_file())
            self.assertFalse(Path(result["paths"]["observations"]).exists())

    def test_layer2_stage_fails_closed_when_layer1_cache_is_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = _prepared_fixture(root)
            forbidden_layer2 = mock.Mock(
                side_effect=AssertionError("Layer 2 must not run without Layer 1 cache")
            )
            product_config, product_prepare, taxonomy = self._product_patches(prepared)
            with product_config, product_prepare, taxonomy:
                with self.assertRaisesRegex(
                    ValueError, "requires a complete Layer 1 cache"
                ):
                    run_laptop_pilot(
                        root,
                        "configs/products/test.json",
                        limit=1,
                        show=0,
                        max_retries=0,
                        generator=forbidden_layer2,
                        layer1_system_prompt="SemEval aspect extraction",
                        stage="layer2",
                        manage_model_lifecycle=False,
                    )
            forbidden_layer2.assert_not_called()


class FeaturedBatchTests(unittest.TestCase):
    def test_v2_config_supports_variable_report_samples_and_equal_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            product_dir = root / "configs/products"
            product_dir.mkdir(parents=True)
            products = []
            for index, sample_size in enumerate((30, 20), 1):
                path = product_dir / f"product_{index}.json"
                path.write_text("{}", encoding="utf-8")
                products.append({
                    "config_path": path.relative_to(root).as_posix(),
                    "brand": f"Brand {index}",
                    "operating_system": "Windows",
                    "evaluation_status": (
                        "benchmark_source_product" if index == 1 else "transfer_demo"
                    ),
                    "cohort_status": "directional_demo",
                    "presentation_name": f"Laptop {index}",
                    "reviews_to_analyze": sample_size,
                })
            config_path = root / "configs/demo/v2.json"
            config_path.parent.mkdir(parents=True)
            config_path.write_text(json.dumps({
                "schema_version": "reviewlens.featured_demo_config.v2",
                "demo_id": "fixture_v2",
                "maximum_reviews_per_product": 30,
                "comparison_reviews_per_product": 20,
                "models": {
                    "layer1": {"model_id": "l1", "revision": "1" * 40},
                    "layer2": {"model_id": "l2", "revision": "2" * 40},
                },
                "products": products,
                "max_retries": 1,
                "bundle_output_directory": "demo_bundle_v2",
            }), encoding="utf-8")

            loaded = laptop_batch.load_featured_demo_config(
                root, config_path.relative_to(root)
            )

        self.assertEqual(30, loaded.reviews_per_product)
        self.assertEqual(20, loaded.comparison_reviews_per_product)
        self.assertEqual([30, 20], [p.reviews_to_analyze for p in loaded.products])
        self.assertEqual("Laptop 2", loaded.products[1].presentation_name)

    def test_two_products_share_one_model_per_layer_and_finish_layer1_first(self):
        events = []
        products = tuple(
            FeaturedProduct(
                config_path=f"configs/products/product_{number}.json",
                brand="Brand",
                operating_system="OS",
                evaluation_status=(
                    "benchmark_source_product" if number == 1 else "transfer_demo"
                ),
                cohort_status="demo",
            )
            for number in (1, 2)
        )
        config = FeaturedDemoConfig(
            demo_id="test_demo",
            reviews_per_product=30,
            l1_model_id="layer1-model",
            l1_revision="1" * 40,
            l2_model_id="layer2-model",
            l2_revision="2" * 40,
            products=products,
            max_retries=0,
            bundle_output_directory="demo_bundle",
            config_path="configs/demo/test.json",
            config_sha256="config-hash",
        )
        layer1_model = _TrackedModel("l1", events)
        layer2_model = _TrackedModel("l2", events)

        def fake_run(root, product_config, **kwargs):
            stage = kwargs["stage"]
            events.append(f"{stage}:{product_config}")
            if stage == "layer1":
                kwargs["layer1_generator"]("prompt", "sentence")
                return {"product_id": product_config, "stage": "layer1_complete"}
            kwargs["generator"]("prompt", {}, False)
            return {"product_id": product_config, "stage": "complete"}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                mock.patch.object(
                    laptop_batch, "load_featured_demo_config", return_value=config
                ),
                mock.patch.object(laptop_batch, "_verify_local_revisions"),
                mock.patch.object(
                    laptop_batch,
                    "_LaptopLayer1Generator",
                    return_value=layer1_model,
                ) as layer1_constructor,
                mock.patch.object(
                    laptop_batch, "_MlxGenerator", return_value=layer2_model
                ) as layer2_constructor,
                mock.patch.object(
                    laptop_batch, "run_laptop_pilot", side_effect=fake_run
                ),
            ):
                result = laptop_batch.run_featured_precompute(
                    root, "configs/demo/test.json", build_bundle=False
                )

        self.assertEqual(result["layer1_model_loads"], 1)
        self.assertEqual(result["layer2_model_loads"], 1)
        layer1_constructor.assert_called_once()
        layer2_constructor.assert_called_once()
        self.assertEqual(layer1_model.calls, 2)
        self.assertEqual(layer2_model.calls, 2)
        self.assertEqual(layer1_model.closed, 1)
        self.assertEqual(layer2_model.closed, 1)
        first_layer2 = next(
            index for index, event in enumerate(events) if event.startswith("layer2:")
        )
        last_layer1 = max(
            index for index, event in enumerate(events) if event.startswith("layer1:")
        )
        self.assertLess(last_layer1, first_layer2)
        self.assertLess(events.index("l1:close"), first_layer2)


if __name__ == "__main__":
    unittest.main()
