import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from reviewlens_ef import laptop_products
from reviewlens_ef.laptop_pipeline import TRANSFER_STATUS, run_laptop_pilot, summarize_laptop_run
from reviewlens_ef.laptop_prompts import build_laptop_observation_prompt


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def prepared_fixture(root, product_id="test_laptop"):
    output = root / "outputs/laptops" / product_id
    review_ids = ("review_1", "review_2")
    reviews, sentences, titles = [], [], []
    for number, review_id in enumerate(review_ids, 1):
        text, title = "The screen is clear and bright.", "Clear screen"
        reviews.append({
            "review_id": review_id, "product_id": product_id,
            "product_name": "Test Laptop Family", "product_model": "Test 14",
            "asin": f"CHILD0000{number}", "parent_asin": "PARENT0001",
            "rating": float(number + 3), "title": title,
            "raw_text": text, "clean_text": text,
        })
        sentences.append({
            "sentence_id": f"{review_id}_S001", "review_id": review_id,
            "source_field": "clean_text", "sentence_text": text,
            "start_char": 0, "end_char": len(text), "rating": float(number + 3),
        })
        titles.append({
            "sentence_id": f"{review_id}_TITLE", "review_id": review_id,
            "source_field": "title", "sentence_text": title,
            "start_char": 0, "end_char": len(title),
        })
    reviews_path = output / "prepared_reviews.jsonl"
    sentences_path = output / "sentences.jsonl"
    titles_path = output / "titles.jsonl"
    manifest_path = output / "prepare_manifest.json"
    write_jsonl(reviews_path, reviews)
    write_jsonl(sentences_path, sentences)
    write_jsonl(titles_path, titles)
    manifest_path.write_text(json.dumps({"selected_review_ids": list(review_ids)}), encoding="utf-8")
    return SimpleNamespace(
        product_id=product_id, output_dir=output, reviews_path=reviews_path,
        sentences_path=sentences_path, titles_path=titles_path,
        manifest_path=manifest_path, selected_review_ids=review_ids,
        config_sha256="config-hash", taxonomy_sha256="taxonomy-hash",
    )


class FakeLayer1:
    def __init__(self):
        self.calls, self.closed = 0, False

    def __call__(self, prompt, sentence):
        self.calls += 1
        return json.dumps([{"term": "screen", "sentiment": "positive"}])

    def close(self):
        self.closed = True


class FakeLayer2:
    def __init__(self, layer1):
        self.layer1, self.calls = layer1, 0

    def __call__(self, prompt, payload, repair):
        if not self.layer1.closed:
            raise AssertionError("Layer 1 must be released before Layer 2")
        self.calls += 1
        if "validated_observations" in payload:
            return json.dumps({"claims": [{
                "observation_id": item["observation_id"],
                "mapping_status": "mapped", "category": "display",
                "issue_group": "screen_quality", "claim": item["observation"],
            } for item in payload["validated_observations"]]})
        source = next(
            item for item in payload["sentences_and_layer1_clues"]
            if item["source_field"] == "clean_text"
        )
        expected = [{"term": "screen", "sentiment": "positive"}]
        if source["layer1_aspects"] != expected:
            raise AssertionError("Layer 1 clue did not reach Layer 2")
        return json.dumps({"observations": [{
            "sentence_id": source["sentence_id"], "evidence_text": source["text"],
            "observation": "The screen is clear and bright.",
            "target": "current_product", "polarity": "positive",
            "statement_type": "actual_experience",
        }]})


class LaptopPipelineTests(unittest.TestCase):
    def test_prompt_is_product_neutral(self):
        prompt = build_laptop_observation_prompt()
        self.assertNotIn("Acer", prompt)
        self.assertNotIn("A515", prompt)
        self.assertIn("product_name", prompt)

    def test_layer_order_cache_resume_summary_and_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = prepared_fixture(root)
            config = SimpleNamespace(config_sha256="config-hash")
            taxonomy = {"display": {"screen_quality": "screen clarity"}}
            with (
                mock.patch.object(laptop_products, "load_laptop_product_config", return_value=config),
                mock.patch.object(laptop_products, "prepare_laptop_product", return_value=prepared),
                mock.patch.object(laptop_products, "load_laptop_taxonomy", return_value=taxonomy),
            ):
                layer1 = FakeLayer1()
                layer2 = FakeLayer2(layer1)
                result = run_laptop_pilot(
                    root, "configs/products/test.json", limit=2, show=0,
                    max_retries=0, layer1_generator=layer1, generator=layer2,
                    layer1_system_prompt="SemEval aspect extraction",
                )
                self.assertEqual(layer1.calls, 2)
                self.assertTrue(layer1.closed)
                self.assertEqual(layer2.calls, 4)
                self.assertEqual(result["status"], TRANSFER_STATUS)
                self.assertEqual(result["selected_review_ids"], ["review_1", "review_2"])
                self.assertEqual(result["summary"]["issues"][0]["supporting_review_count"], 2)
                self.assertEqual(result["summary"]["issues"][0]["supporting_review_rate"], 1.0)
                self.assertTrue(
                    Path(result["run_dir"]).is_relative_to(
                        (root / "outputs/laptops/test_laptop").resolve()
                    )
                )
                manifest = json.loads(Path(result["paths"]["manifest"]).read_text())
                self.assertIn("model_revision", manifest["layer1"])
                self.assertIn("model_revision", manifest["layer2"])

                def forbidden(*args, **kwargs):
                    raise AssertionError("complete cache called a model")

                cached = run_laptop_pilot(
                    root, "configs/products/test.json", limit=2, show=0,
                    max_retries=0, layer1_generator=forbidden, generator=forbidden,
                    layer1_system_prompt="SemEval aspect extraction",
                )
                self.assertEqual(cached["run_fingerprint"], result["run_fingerprint"])
                read_only = summarize_laptop_run(root, "configs/products/test.json", limit=2, show=0)
                self.assertEqual(read_only["summary"], result["summary"])


if __name__ == "__main__":
    unittest.main()
