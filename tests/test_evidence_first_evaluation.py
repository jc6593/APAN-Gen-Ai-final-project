import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from reviewlens_ef.evaluation import summarize
from reviewlens_ef.nli import _advisory_status, run_nli


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class EvidenceFirstEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.review_id = "electronics_test"
        review_text = "The speakers are terrible."
        write_jsonl(
            self.root / "data/processed/amazon/acer_aspire5_reviews_clean.jsonl",
            [{
                "review_id": self.review_id,
                "rating": 1.0,
                "title": "Terrible speakers",
                "clean_text": review_text,
            }],
        )
        write_jsonl(
            self.root / "data/processed/amazon/acer_aspire5_sentences.jsonl",
            [{
                "sentence_id": f"{self.review_id}_S001",
                "review_id": self.review_id,
                "sentence_text": review_text,
                "start_char": 0,
                "end_char": len(review_text),
            }],
        )
        write_jsonl(
            self.root / "outputs/amazon/acer_aspire5_business_full_v3.jsonl",
            [{
                "review_id": self.review_id,
                "accepted_feedback": [{
                    "category": "audio",
                    "issue_group": "speaker_sound",
                    "sentiment": "negative",
                    "attribution": "current_product",
                    "summary": "The speakers sound bad.",
                    "evidence": [{
                        "sentence_id": f"{self.review_id}_S001",
                        "text": "speakers are terrible",
                    }],
                }],
            }],
        )
        write_jsonl(
            self.root / "outputs/amazon/evidence_first/observations_pilot.jsonl",
            [{
                "review_id": self.review_id,
                "accepted_observations": [
                    {
                        "observation_id": "O001",
                        "sentence_id": f"{self.review_id}_TITLE",
                        "source_field": "title",
                        "evidence_text": "Terrible speakers",
                        "evidence_start_char_source": 0,
                        "evidence_end_char_source": 17,
                        "observation": "The title criticizes the speakers.",
                        "target": "current_product",
                        "polarity": "negative",
                    },
                    {
                        "observation_id": "O002",
                        "sentence_id": f"{self.review_id}_S001",
                        "source_field": "clean_text",
                        "evidence_text": "speakers are terrible",
                        "observation": "The reviewer criticizes the speakers.",
                        "target": "current_product",
                        "polarity": "negative",
                    },
                ],
                "rejected_observations": [],
            }],
        )
        write_jsonl(
            self.root / "outputs/amazon/evidence_first/claims_pilot.jsonl",
            [{
                "review_id": self.review_id,
                "accepted_claims": [
                    {
                        "claim_id": "C001",
                        "observation_id": "O001",
                        "mapping_status": "mapped",
                        "category": "audio",
                        "issue_group": "speaker_sound",
                        "claim": "The review title reports terrible speakers.",
                        "sentiment": "negative",
                        "attribution": "current_product",
                        "evidence": [{
                            "sentence_id": f"{self.review_id}_TITLE",
                            "source_field": "title",
                            "evidence_text": "Terrible speakers",
                            "quote_start_char_source": 0,
                            "quote_end_char_source": 17,
                        }],
                    },
                    {
                        "claim_id": "C002",
                        "observation_id": "O002",
                        "mapping_status": "mapped",
                        "category": "audio",
                        "issue_group": "speaker_sound",
                        "claim": "The reviewer reports terrible speakers.",
                        "sentiment": "negative",
                        "attribution": "current_product",
                        "evidence": [{
                            "sentence_id": f"{self.review_id}_S001",
                            "source_field": "clean_text",
                            "evidence_text": "speakers are terrible",
                            "quote_start_char_clean": 4,
                            "quote_end_char_clean": 25,
                        }],
                    },
                ],
                "merged_claims": [{
                    "merged_claim_id": "M001",
                    "mapping_status": "mapped",
                    "category": "audio",
                    "issue_group": "speaker_sound",
                    "sentiment": "negative",
                    "attribution": "current_product",
                    "component_claims": [
                        "The review title reports terrible speakers.",
                        "The reviewer reports terrible speakers.",
                    ],
                    "source_claim_ids": ["C001", "C002"],
                    "observation_ids": ["O001", "O002"],
                    "evidence": [],
                }],
                "duplicate_claims_merged": 1,
                "rejected_claims": [],
            }],
        )
        write_jsonl(
            self.root / "outputs/amazon/evidence_first/nli_pilot.jsonl",
            [{
                "review_id": self.review_id,
                "claim_id": "C001",
                "advisory_status": "supported",
                "components": {
                    name: {
                        "label": "entailment",
                        "probabilities": {
                            "entailment": 0.9,
                            "neutral": 0.08,
                            "contradiction": 0.02,
                        },
                    }
                    for name in ("claim", "sentiment", "target")
                },
            }],
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_summary_aligns_signatures_and_writes_review_sheets(self):
        result = summarize(self.root, show=0)
        self.assertEqual(result["pilot_reviews"], 1)
        self.assertEqual(result["totals"]["shared_issue_tuples"], 1)
        self.assertEqual(result["totals"]["baseline_claim_count"], 1)
        self.assertEqual(result["totals"]["baseline_unique_issue_tuple_count"], 1)
        self.assertEqual(result["totals"]["ef_claim_count"], 2)
        self.assertEqual(result["totals"]["ef_atomic_claim_count"], 2)
        self.assertEqual(result["totals"]["ef_business_finding_count"], 1)
        self.assertEqual(result["totals"]["duplicate_claims_merged"], 1)
        self.assertEqual(result["totals"]["ef_fallback_unmapped_count"], 0)
        self.assertEqual(
            result["totals"]["ef_deterministic_recovery_count"], 0
        )
        self.assertEqual(
            result["totals"]["ef_category_canonicalization_count"], 0
        )
        self.assertEqual(result["totals"]["ef_taxonomy_repair_count"], 0)

        output = self.root / "outputs/amazon/evidence_first"
        with (output / "claim_precision_review.csv").open(
            encoding="utf-8-sig", newline=""
        ) as handle:
            precision = list(csv.DictReader(handle))
        self.assertEqual(len(precision), 3)
        self.assertNotIn("system", precision[0])
        self.assertNotIn("claim_id", precision[0])
        self.assertIn("statement_type", precision[0])
        self.assertIn("normalization_source", precision[0])
        self.assertIn("normalization_adjustments", precision[0])
        self.assertTrue(all(row["blind_id"].startswith("BLIND_") for row in precision))
        self.assertEqual(
            sorted(row["nli_status"] for row in precision),
            ["not_run", "not_run", "supported"],
        )
        self.assertEqual(len(result["precision_blind_key"]), 3)
        self.assertTrue(all(row["locator_valid"] == "True" for row in precision))

        with (output / "extraction_coverage_review.csv").open(
            encoding="utf-8-sig", newline=""
        ) as handle:
            coverage = list(csv.DictReader(handle))
        self.assertEqual(len(coverage), 1)
        self.assertEqual(coverage[0]["review_title"], "Terrible speakers")
        self.assertEqual(coverage[0]["ef_observation_count"], "2")
        self.assertEqual(coverage[0]["ef_business_finding_count"], "1")
        self.assertEqual(coverage[0]["ef_fallback_unmapped_count"], "0")
        self.assertEqual(coverage[0]["ef_deterministic_recovery_count"], "0")

    def test_summary_preserves_human_columns(self):
        result = summarize(self.root, show=0)
        path = self.root / "outputs/amazon/evidence_first/claim_precision_review.csv"
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
            fieldnames = list(rows[0])
        rows[0]["human_supported"] = "yes"
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        result = summarize(self.root, show=0)
        with path.open(encoding="utf-8-sig", newline="") as handle:
            after = {row["blind_id"]: row for row in csv.DictReader(handle)}
        self.assertEqual(after[rows[0]["blind_id"]]["human_supported"], "yes")
        system = result["precision_blind_key"][rows[0]["blind_id"]]["system"]
        metrics = result["human_evaluation"]["claim_precision_by_system"][system]
        self.assertEqual(metrics["components"]["supported"]["labeled"], 1)
        self.assertEqual(metrics["components"]["supported"]["rate"], 1.0)


class FakeScorer:
    def __init__(self, model_id, device, batch_size):
        self.model_id = model_id
        self.device = "cpu"
        self.batch_size = batch_size
        self.model_revision = "fake-revision"

    def score_pairs(self, pairs):
        return [
            {
                "label": "entailment",
                "logits": [3.0, 0.0, -1.0],
                "probabilities": {
                    "entailment": 0.9,
                    "neutral": 0.08,
                    "contradiction": 0.02,
                },
            }
            for _ in pairs
        ]


class EvidenceFirstNliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        review_id = "electronics_nli"
        sentence = "The speakers are terrible."
        write_jsonl(
            self.root / "data/processed/amazon/acer_aspire5_reviews_clean.jsonl",
            [{
                "review_id": review_id,
                "title": "Terrible speakers",
                "clean_text": sentence,
            }],
        )
        write_jsonl(
            self.root / "data/processed/amazon/acer_aspire5_sentences.jsonl",
            [{
                "sentence_id": f"{review_id}_S001",
                "review_id": review_id,
                "sentence_text": sentence,
                "start_char": 0,
                "end_char": len(sentence),
            }],
        )
        write_jsonl(
            self.root / "outputs/amazon/evidence_first/claims_pilot.jsonl",
            [{
                "review_id": review_id,
                "accepted_claims": [{
                    "claim_id": "C001",
                    "mapping_status": "mapped",
                    "category": "audio",
                    "issue_group": "speaker_sound",
                    "claim": "The reviewer dislikes the speakers.",
                    "sentiment": "negative",
                    "attribution": "current_product",
                    "evidence": [{
                        "sentence_id": f"{review_id}_TITLE",
                        "source_field": "title",
                        "evidence_text": "Terrible speakers",
                    }],
                }],
            }],
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_run_nli_uses_mock_scorer_and_resumes(self):
        with patch("reviewlens_ef.nli.TransformersNliScorer", FakeScorer):
            first = run_nli(self.root, model_id="fake/model", show=0)
            second = run_nli(self.root, model_id="fake/model", show=0)
        self.assertEqual(first["pending_run"], 1)
        self.assertEqual(second["pending_run"], 0)
        self.assertEqual(first["status_counts"], {"supported": 1})
        rows = _read(self.root / "outputs/amazon/evidence_first/nli_pilot.jsonl")
        self.assertEqual(rows[0]["model_revision"], "fake-revision")
        self.assertEqual(set(rows[0]["components"]), {"claim", "sentiment", "target"})
        self.assertIn("Terrible speakers", rows[0]["premise"])

    def test_stale_model_cache_is_rejected(self):
        with patch("reviewlens_ef.nli.TransformersNliScorer", FakeScorer):
            run_nli(self.root, model_id="fake/model", show=0)
            with self.assertRaisesRegex(ValueError, "Stale NLI cache"):
                run_nli(self.root, model_id="different/model", show=0)

    def test_advisory_status_is_conservative(self):
        supported = {
            name: {"probabilities": {"entailment": 0.8, "contradiction": 0.1}}
            for name in ("claim", "sentiment", "target")
        }
        contradicted = dict(supported)
        contradicted["claim"] = {
            "probabilities": {"entailment": 0.05, "contradiction": 0.9}
        }
        unclear = dict(supported)
        unclear["target"] = {
            "probabilities": {"entailment": 0.5, "contradiction": 0.1}
        }
        self.assertEqual(_advisory_status(supported), "supported")
        self.assertEqual(_advisory_status(contradicted), "contradicted")
        self.assertEqual(_advisory_status(unclear), "unclear")


def _read(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


if __name__ == "__main__":
    unittest.main()
