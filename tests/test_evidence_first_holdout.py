import json
import tempfile
import unittest
from pathlib import Path

from reviewlens_ef.core import PILOT_REVIEW_IDS
from reviewlens_ef.holdout import (
    _selection_document,
    prepare_holdout,
    run_holdout,
)


TAXONOMY = {"battery": {"battery_runtime": "battery life per charge"}}
MODEL_ID = "fake-qwen"


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def make_project(root, reverse=False):
    reviews = []
    sentences = []
    layer1 = []
    for rating in range(1, 6):
        for index in range(5):
            review_id = f"electronics_holdout_{rating}_{index}"
            text = (
                "Battery lasts all day."
                + " Very good." * index
                + f" Rating {rating}."
            )
            sentence_id = f"{review_id}_S001"
            reviews.append({
                "review_id": review_id,
                "rating": float(rating),
                "title": f"Battery review {rating}-{index}",
                "clean_text": text,
                "product_name": "test laptop",
                "product_model": "test",
            })
            sentences.append({
                "review_id": review_id,
                "sentence_id": sentence_id,
                "sentence_text": text,
                "start_char": 0,
                "end_char": len(text),
            })
            layer1.append({
                "review_id": review_id,
                "sentence_id": sentence_id,
                "sentence_text": text,
                "start_char": 0,
                "parse_success": True,
                "aspects": [
                    {"term": f"battery-{number}", "sentiment": "positive"}
                    for number in range(index + 1)
                ],
            })
    for index, review_id in enumerate(PILOT_REVIEW_IDS):
        text = "Battery lasts all day in this pilot review."
        sentence_id = f"{review_id}_S001"
        reviews.append({
            "review_id": review_id,
            "rating": float(index % 5 + 1),
            "title": "Excluded pilot review",
            "clean_text": text,
            "product_name": "test laptop",
            "product_model": "test",
        })
        sentences.append({
            "review_id": review_id,
            "sentence_id": sentence_id,
            "sentence_text": text,
            "start_char": 0,
            "end_char": len(text),
        })
        layer1.append({
            "review_id": review_id,
            "sentence_id": sentence_id,
            "sentence_text": text,
            "start_char": 0,
            "parse_success": True,
            "aspects": [{"term": "battery", "sentiment": "positive"}],
        })
    if reverse:
        reviews.reverse()
        sentences.reverse()
        layer1.reverse()
    write_jsonl(
        root / "data/processed/amazon/acer_aspire5_reviews_clean.jsonl", reviews
    )
    write_jsonl(
        root / "data/processed/amazon/acer_aspire5_sentences.jsonl", sentences
    )
    write_jsonl(
        root / "outputs/amazon/acer_aspire5_full_predictions.jsonl", layer1
    )


class HoldoutTests(unittest.TestCase):
    def test_selection_is_fixed_balanced_and_order_invariant(self):
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            first = Path(first_dir)
            second = Path(second_dir)
            make_project(first)
            make_project(second, reverse=True)
            first_doc = _selection_document(first, TAXONOMY, MODEL_ID, 5560)
            second_doc = _selection_document(second, TAXONOMY, MODEL_ID, 5560)
            self.assertEqual(first_doc, second_doc)
            selected = first_doc["selected_reviews"]
            self.assertEqual(25, len(selected))
            ids = [row["review_id"] for row in selected]
            self.assertEqual(25, len(set(ids)))
            self.assertFalse(set(ids) & set(PILOT_REVIEW_IDS))
            self.assertEqual([1, 2, 3, 4, 5], [
                row["rating"] for row in selected[:5]
            ])
            for rating in range(1, 6):
                rows = [row for row in selected if row["rating"] == rating]
                self.assertEqual(5, len(rows))
                self.assertEqual(set(range(5)), {row["length_bin"] for row in rows})
                self.assertEqual(set(range(5)), {row["density_bin"] for row in rows})

    def test_prepare_run_resume_and_force_are_isolated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_project(root)
            document = prepare_holdout(root, TAXONOMY, MODEL_ID)
            holdout_dir = root / "outputs/amazon/evidence_first/holdout"
            selection_path = holdout_dir / "selection_manifest.json"
            self.assertTrue(selection_path.exists())
            self.assertFalse((holdout_dir / "run_manifest.json").exists())
            self.assertEqual("frozen_before_inference", document["status"])
            selection_before = selection_path.read_bytes()
            pilot_sentinel = root / "outputs/amazon/evidence_first/claims_pilot.jsonl"
            pilot_sentinel.write_text('{"pilot": true}\n', encoding="utf-8")
            calls = []

            def fake_generator(system_prompt, payload, repair):
                self.assertTrue(selection_path.exists())
                self.assertTrue((holdout_dir / "run_manifest.json").exists())
                if "sentences_and_layer1_clues" in payload:
                    calls.append((payload["review_id"], "observation"))
                    source = next(
                        item for item in payload["sentences_and_layer1_clues"]
                        if item["source_field"] == "clean_text"
                    )
                    return json.dumps({"observations": [{
                        "sentence_id": source["sentence_id"],
                        "evidence_text": source["text"],
                        "observation": "The battery lasts all day.",
                        "target": "current_product",
                        "polarity": "positive",
                        "statement_type": "actual_experience",
                    }]})
                calls.append((payload["review_id"], "claim"))
                observation = payload["validated_observations"][0]
                return json.dumps({"claims": [{
                    "observation_id": observation["observation_id"],
                    "mapping_status": "mapped",
                    "category": "battery",
                    "issue_group": "battery_runtime",
                    "claim": observation["observation"],
                }]})

            first = run_holdout(
                root, TAXONOMY, MODEL_ID, limit=5, max_retries=0,
                show=0, generator=fake_generator,
            )
            self.assertEqual(5, len(first["claim_rows"]))
            self.assertEqual(10, len(calls))
            second = run_holdout(
                root, TAXONOMY, MODEL_ID, limit=25, max_retries=0,
                show=0, generator=fake_generator,
            )
            self.assertEqual(25, len(second["claim_rows"]))
            self.assertEqual(50, len(calls))

            def should_not_run(*args):
                raise AssertionError("fully cached holdout should resume")

            run_holdout(
                root, TAXONOMY, MODEL_ID, limit=25, max_retries=0,
                show=0, generator=should_not_run,
            )
            self.assertEqual(selection_before, selection_path.read_bytes())
            self.assertEqual(
                '{"pilot": true}\n', pilot_sentinel.read_text(encoding="utf-8")
            )

            run_holdout(
                root, TAXONOMY, MODEL_ID, limit=5, force=True,
                max_retries=0, show=0, generator=fake_generator,
            )
            self.assertEqual(selection_before, selection_path.read_bytes())
            self.assertEqual(
                '{"pilot": true}\n', pilot_sentinel.read_text(encoding="utf-8")
            )
            self.assertTrue(list(holdout_dir.glob("claims_holdout.backup_*.jsonl")))
            self.assertFalse(list(
                holdout_dir.glob("selection_manifest.backup_*.json")
            ))

    def test_stale_source_and_config_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_project(root)
            prepare_holdout(root, TAXONOMY, MODEL_ID)
            with self.assertRaisesRegex(ValueError, "immutable"):
                prepare_holdout(root, TAXONOMY, "different-model")

            reviews_path = (
                root / "data/processed/amazon/acer_aspire5_reviews_clean.jsonl"
            )
            rows = [json.loads(line) for line in reviews_path.read_text().splitlines()]
            rows[0]["clean_text"] += " changed"
            write_jsonl(reviews_path, rows)
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                run_holdout(
                    root, TAXONOMY, MODEL_ID, limit=5, max_retries=0,
                    show=0, generator=lambda *_: self.fail("must not call model"),
                )

    def test_run_requires_explicit_prepare(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_project(root)
            with self.assertRaisesRegex(ValueError, "ef-holdout-prepare"):
                run_holdout(
                    root, TAXONOMY, MODEL_ID, limit=5, max_retries=0,
                    show=0, generator=lambda *_: self.fail("must not call model"),
                )


if __name__ == "__main__":
    unittest.main()
