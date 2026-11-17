import json
import tempfile
import unittest
from pathlib import Path

from reviewlens_ef.core import PILOT_REVIEW_IDS
from reviewlens_ef.pipeline import run_pilot


TAXONOMY = {"battery": {"battery_runtime": "battery life per charge"}}


class PipelineV11Tests(unittest.TestCase):
    def _make_project(self, root):
        reviews_path = root / "data/processed/amazon/acer_aspire5_reviews_clean.jsonl"
        sentences_path = root / "data/processed/amazon/acer_aspire5_sentences.jsonl"
        layer1_path = root / "outputs/amazon/acer_aspire5_full_predictions.jsonl"
        reviews_path.parent.mkdir(parents=True)
        layer1_path.parent.mkdir(parents=True)
        reviews, sentences, layer1 = [], [], []
        for review_id in PILOT_REVIEW_IDS:
            text = "Battery lasts all day."
            title = "Excellent battery life"
            sentence_id = f"{review_id}_S001"
            reviews.append({
                "review_id": review_id,
                "clean_text": text,
                "title": title,
                "rating": 5.0,
                "product_name": "test",
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
                "aspects": [{"term": "Battery", "sentiment": "positive"}],
            })
        for path, rows in (
            (reviews_path, reviews),
            (sentences_path, sentences),
            (layer1_path, layer1),
        ):
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

    def test_title_source_merged_business_finding_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._make_project(root)
            calls = []

            def fake_generator(system_prompt, payload, repair):
                calls.append(payload["review_id"])
                if "sentences_and_layer1_clues" in payload:
                    sources = payload["sentences_and_layer1_clues"]
                    title, body = sources
                    self.assertEqual("title", title["source_field"])
                    self.assertIsNone(title["layer1_aspects"])
                    self.assertEqual("clean_text", body["source_field"])
                    return json.dumps({"observations": [
                        {
                            "sentence_id": title["sentence_id"],
                            "evidence_text": title["text"],
                            "observation": "The title praises battery life.",
                            "target": "current_product",
                            "polarity": "positive",
                            "statement_type": "actual_experience",
                        },
                        {
                            "sentence_id": body["sentence_id"],
                            "evidence_text": "Battery lasts all day",
                            "observation": "The battery lasts all day.",
                            "target": "current_product",
                            "polarity": "positive",
                            "statement_type": "actual_experience",
                        },
                    ]})
                observations = payload["validated_observations"]
                self.assertEqual("title", observations[0]["source_field"])
                self.assertNotIn("context_before", observations[0])
                self.assertNotIn("context_after", observations[0])
                return json.dumps({"claims": [
                    {
                        "observation_id": observation["observation_id"],
                        "mapping_status": "mapped",
                        "category": "battery",
                        "issue_group": "battery_runtime",
                        "claim": observation["observation"],
                    }
                    for observation in observations
                ]})

            first = run_pilot(
                root, TAXONOMY, "fake-model", limit=10,
                max_retries=0, show=0, generator=fake_generator,
            )
            self.assertEqual(20, len(calls))
            first_observation = first["observation_rows"][0]["accepted_observations"][0]
            self.assertEqual("title", first_observation["source_field"])
            self.assertIsNone(first_observation["evidence_start_char_clean"])
            row = first["claim_rows"][0]
            self.assertEqual(2, len(row["accepted_claims"]))
            self.assertEqual(1, len(row["merged_claims"]))
            self.assertEqual(1, row["duplicate_claims_merged"])
            self.assertEqual(2, len(row["merged_claims"][0]["evidence"]))
            self.assertTrue(row["model_normalization_complete"])
            self.assertEqual(0, row["fallback_unmapped_count"])
            self.assertTrue(row["normalization_complete"])

            def should_not_run(*args):
                raise AssertionError("valid v1.4 cache must resume")

            second = run_pilot(
                root, TAXONOMY, "fake-model", limit=10,
                max_retries=0, show=0, generator=should_not_run,
            )
            self.assertEqual(10, len(second["claim_rows"]))

            ef_dir = root / "outputs/amazon/evidence_first"
            observations_path = ef_dir / "observations_pilot.jsonl"
            observations_before = observations_path.read_bytes()
            nli_path = ef_dir / "nli_pilot.jsonl"
            nli_path.write_text('{"old": true}\n', encoding="utf-8")
            calls_before = len(calls)
            run_pilot(
                root, TAXONOMY, "fake-model", limit=10, force_claims=True,
                max_retries=0, show=0, generator=fake_generator,
            )
            self.assertEqual(10, len(calls) - calls_before)
            self.assertEqual(observations_before, observations_path.read_bytes())
            self.assertEqual(
                [], list(ef_dir.glob("observations_pilot.backup_*.jsonl"))
            )
            self.assertEqual(
                1, len(list(ef_dir.glob("claims_pilot.backup_*.jsonl")))
            )
            self.assertEqual(
                1, len(list(ef_dir.glob("pilot_manifest.backup_*.json")))
            )
            self.assertFalse(nli_path.exists())
            backups = list(ef_dir.glob("nli_pilot.backup_*.jsonl"))
            self.assertEqual(1, len(backups))
            self.assertEqual(
                '{"old": true}\n', backups[0].read_text(encoding="utf-8")
            )

            with self.assertRaisesRegex(ValueError, "mutually exclusive"):
                run_pilot(
                    root, TAXONOMY, "fake-model", limit=10,
                    force=True, force_claims=True, max_retries=0,
                    show=0, generator=should_not_run,
                )

            nli_path.write_text('{"new": true}\n', encoding="utf-8")
            run_pilot(
                root, TAXONOMY, "fake-model", limit=10, force=True,
                max_retries=0, show=0, generator=fake_generator,
            )
            self.assertEqual(
                1, len(list(ef_dir.glob("observations_pilot.backup_*.jsonl")))
            )
            self.assertEqual(
                2, len(list(ef_dir.glob("nli_pilot.backup_*.jsonl")))
            )

    def test_force_claims_validates_observations_before_backing_up(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._make_project(root)
            ef_dir = root / "outputs/amazon/evidence_first"
            ef_dir.mkdir(parents=True)
            claims_path = ef_dir / "claims_pilot.jsonl"
            manifest_path = ef_dir / "pilot_manifest.json"
            claims_path.write_text('{"sentinel": "claims"}\n', encoding="utf-8")
            manifest_path.write_text('{"sentinel": "manifest"}\n', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "complete compatible"):
                run_pilot(
                    root,
                    TAXONOMY,
                    "fake-model",
                    limit=10,
                    force_claims=True,
                    max_retries=0,
                    show=0,
                    generator=lambda *_: self.fail("model must not run"),
                )

            self.assertEqual(
                '{"sentinel": "claims"}\n',
                claims_path.read_text(encoding="utf-8"),
            )
            self.assertEqual(
                '{"sentinel": "manifest"}\n',
                manifest_path.read_text(encoding="utf-8"),
            )
            self.assertEqual([], list(ef_dir.glob("*.backup_*")))


if __name__ == "__main__":
    unittest.main()
