import csv
import json
import tempfile
import unittest
from pathlib import Path

from reviewlens_ef.evaluation_v2 import summarize_evaluation


TAXONOMY = {"audio": {"speaker_sound": "speaker quality"}}


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def read_jsonl(path):
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class EvaluationV2Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.review_id = "electronics_eval"
        text = "The speakers are terrible."
        sentence_id = f"{self.review_id}_S001"
        write_jsonl(
            self.root / "data/processed/amazon/acer_aspire5_reviews_clean.jsonl",
            [{
                "review_id": self.review_id,
                "rating": 1.0,
                "title": "Terrible speakers",
                "clean_text": text,
            }],
        )
        write_jsonl(
            self.root / "data/processed/amazon/acer_aspire5_sentences.jsonl",
            [{
                "review_id": self.review_id,
                "sentence_id": sentence_id,
                "sentence_text": text,
                "start_char": 0,
                "end_char": len(text),
            }],
        )
        evidence = [{"sentence_id": sentence_id, "text": text}]
        write_jsonl(
            self.root / "outputs/amazon/acer_aspire5_business_full_v3.jsonl",
            [{
                "review_id": self.review_id,
                "accepted_feedback": [{
                    "category": "audio",
                    "issue_group": "speaker_sound",
                    "sentiment": "negative",
                    "attribution": "current_product",
                    "summary": "The speakers sound terrible.",
                    "evidence": evidence,
                }],
            }],
        )
        observations = [
            {
                "observation_id": "O001",
                "statement_type": "actual_experience",
                "observation": "The speakers are terrible.",
            },
            {
                "observation_id": "O002",
                "statement_type": "overall_opinion",
                "observation": "The reviewer is dissatisfied.",
            },
        ]
        write_jsonl(
            self.root / "outputs/amazon/evidence_first/observations_pilot.jsonl",
            [{
                "review_id": self.review_id,
                "parse_success": True,
                "reference_checks_passed": True,
                "accepted_observations": observations,
            }],
        )
        ef_evidence = [{
            "sentence_id": sentence_id,
            "source_field": "clean_text",
            "quote": text,
        }]
        mapped = {
            "claim_id": "C001",
            "observation_id": "O001",
            "mapping_status": "mapped",
            "category": "audio",
            "issue_group": "speaker_sound",
            "sentiment": "negative",
            "attribution": "current_product",
            "statement_type": "actual_experience",
            "claim": "The speakers are terrible.",
            "evidence": ef_evidence,
        }
        unmapped = {
            "claim_id": "C002",
            "observation_id": "O002",
            "mapping_status": "unmapped",
            "category": None,
            "issue_group": None,
            "sentiment": "negative",
            "attribution": "current_product",
            "statement_type": "overall_opinion",
            "claim": "The reviewer is dissatisfied.",
            "evidence": ef_evidence,
        }
        merged = {
            "business_finding_id": "B001",
            "mapping_status": "mapped",
            "category": "audio",
            "issue_group": "speaker_sound",
            "sentiment": "negative",
            "attribution": "current_product",
            "component_claims": ["The speakers are terrible."],
            "evidence": ef_evidence,
        }
        write_jsonl(
            self.root / "outputs/amazon/evidence_first/claims_pilot.jsonl",
            [{
                "review_id": self.review_id,
                "parse_success": True,
                "normalization_complete": True,
                "model_normalization_complete": True,
                "accepted_claims": [mapped, unmapped],
                "merged_claims": [merged],
            }],
        )

    def tearDown(self):
        self.temp.cleanup()

    def read_csv(self, name):
        path = self.root / "outputs/amazon/evidence_first" / name
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    def test_fair_units_blinding_and_diagnostics(self):
        result = summarize_evaluation(
            self.root, TAXONOMY, cohort="pilot", show=0
        )
        primary = self.read_csv("business_finding_precision_blind.csv")
        self.assertEqual(2, len(primary))
        self.assertEqual(1, result["counts"]["v3_findings"])
        self.assertEqual(1, result["counts"]["ef_findings"])
        forbidden = {
            "system", "item_id", "claim_id", "mapping_status",
            "statement_type", "normalization_source", "locator_valid", "nli_status",
        }
        self.assertTrue(forbidden.isdisjoint(primary[0]))
        key = json.loads((
            self.root
            / "outputs/amazon/evidence_first/business_finding_precision_key.json"
        ).read_text())
        self.assertEqual(
            {"v3", "evidence_first"},
            {value["system"] for value in key["rows"].values()},
        )
        atomic = self.read_csv("ef_atomic_evidence_audit.csv")
        abstention = self.read_csv("ef_abstention_audit.csv")
        gold = self.read_csv("gold_coverage_review.csv")
        self.assertEqual(["The speakers are terrible."], [row["claim"] for row in atomic])
        self.assertEqual(
            ["The reviewer is dissatisfied."],
            [row["observation"] for row in abstention],
        )
        self.assertEqual(1, len(gold))
        self.assertNotIn("baseline_claims", gold[0])
        self.assertNotIn("ef_claims", gold[0])
        summarize_evaluation(self.root, TAXONOMY, cohort="pilot", show=0)
        self.assertEqual(
            [],
            list((self.root / "outputs/amazon/evidence_first").glob(
                "*.backup_*.csv"
            )),
        )

    def test_stable_ids_preserve_labels(self):
        summarize_evaluation(self.root, TAXONOMY, show=0)
        path = (
            self.root
            / "outputs/amazon/evidence_first/business_finding_precision_blind.csv"
        )
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
            fields = list(rows[0])
        first_id = rows[0]["blind_id"]
        rows[0]["human_supported"] = "yes"
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        result = summarize_evaluation(self.root, TAXONOMY, show=0)
        after = {row["blind_id"]: row for row in self.read_csv(
            "business_finding_precision_blind.csv"
        )}
        self.assertEqual("yes", after[first_id]["human_supported"])
        key = json.loads((
            self.root
            / "outputs/amazon/evidence_first/business_finding_precision_key.json"
        ).read_text())
        system = key["rows"][first_id]["system"]
        self.assertEqual(
            1,
            result["metrics"]["primary_finding_precision_by_system"][system]
            ["components"]["supported"]["labeled"],
        )

    def test_complete_gold_validation_and_metrics(self):
        summarize_evaluation(self.root, TAXONOMY, show=0)
        path = self.root / "outputs/amazon/evidence_first/gold_coverage_review.csv"
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
            fields = list(rows[0])
        rows[0]["human_gold_complete"] = "yes"
        rows[0]["human_gold_findings_json"] = "not-json"
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaisesRegex(ValueError, self.review_id):
            summarize_evaluation(self.root, TAXONOMY, show=0)

        rows[0]["human_gold_findings_json"] = json.dumps([{
            "category": "audio",
            "issue_group": "speaker_sound",
            "sentiment": "negative",
            "attribution": "current_product",
            "evidence_quote": "The speakers are terrible.",
        }])
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        result = summarize_evaluation(self.root, TAXONOMY, show=0)
        coverage = result["metrics"]["coverage_by_system"]
        for system in ("v3", "evidence_first"):
            self.assertEqual(1.0, coverage[system]["issue_pair"]["micro"]["recall"])
            self.assertEqual(1.0, coverage[system]["full_tuple"]["review_macro"]["recall"])
        self.assertEqual(1, result["metrics"]["gold_label_status"]["complete"])

    def test_stage_failure_has_end_to_end_and_conditional_coverage(self):
        failed_id = "electronics_failed"
        failed_text = "The speakers are awful."
        failed_sentence_id = f"{failed_id}_S001"
        reviews_path = (
            self.root / "data/processed/amazon/acer_aspire5_reviews_clean.jsonl"
        )
        reviews = read_jsonl(reviews_path)
        reviews.append({
            "review_id": failed_id,
            "rating": 1.0,
            "title": "Awful speakers",
            "clean_text": failed_text,
        })
        write_jsonl(reviews_path, reviews)

        sentences_path = (
            self.root / "data/processed/amazon/acer_aspire5_sentences.jsonl"
        )
        sentences = read_jsonl(sentences_path)
        sentences.append({
            "review_id": failed_id,
            "sentence_id": failed_sentence_id,
            "sentence_text": failed_text,
            "start_char": 0,
            "end_char": len(failed_text),
        })
        write_jsonl(sentences_path, sentences)

        baseline_path = (
            self.root / "outputs/amazon/acer_aspire5_business_full_v3.jsonl"
        )
        baseline = read_jsonl(baseline_path)
        baseline.append({
            "review_id": failed_id,
            "accepted_feedback": [{
                "category": "audio",
                "issue_group": "speaker_sound",
                "sentiment": "negative",
                "attribution": "current_product",
                "summary": "The speakers are awful.",
                "evidence": [{
                    "sentence_id": failed_sentence_id,
                    "text": failed_text,
                }],
            }],
        })
        write_jsonl(baseline_path, baseline)

        observation_path = (
            self.root / "outputs/amazon/evidence_first/observations_pilot.jsonl"
        )
        observation_rows = read_jsonl(observation_path)
        observation_rows.append({
            "review_id": failed_id,
            "parse_success": False,
            "reference_checks_passed": False,
            "accepted_observations": [],
            "rejected_observations": [],
            "validation_errors": ["parse_error: expected one JSON object"],
        })
        write_jsonl(observation_path, observation_rows)

        claim_path = (
            self.root / "outputs/amazon/evidence_first/claims_pilot.jsonl"
        )
        claim_rows = read_jsonl(claim_path)
        claim_rows.append({
            "review_id": failed_id,
            "parse_success": True,
            "normalization_complete": True,
            "model_normalization_complete": True,
            "accepted_claims": [],
            "merged_claims": [],
        })
        write_jsonl(claim_path, claim_rows)

        initial = summarize_evaluation(self.root, TAXONOMY, show=0)
        status = initial["metrics"]["stage_status"]
        self.assertEqual(2, status["cache_complete_reviews"])
        self.assertEqual(1, status["valid_generation_reviews"])
        self.assertEqual([failed_id], status["stage_failure_review_ids"])
        self.assertEqual(
            "observation_extraction",
            status["failure_details"][failed_id][0]["stage"],
        )

        gold_path = (
            self.root / "outputs/amazon/evidence_first/gold_coverage_review.csv"
        )
        with gold_path.open(encoding="utf-8-sig", newline="") as handle:
            gold_rows = list(csv.DictReader(handle))
            fields = list(gold_rows[0])
        self.assertEqual(2, len(gold_rows))
        self.assertNotIn("generation_status", gold_rows[0])
        for row in gold_rows:
            quote = (
                "The speakers are terrible."
                if row["review_id"] == self.review_id else failed_text
            )
            row["human_gold_complete"] = "yes"
            row["human_gold_findings_json"] = json.dumps([{
                "category": "audio",
                "issue_group": "speaker_sound",
                "sentiment": "negative",
                "attribution": "current_product",
                "evidence_quote": quote,
            }])
        with gold_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(gold_rows)

        result = summarize_evaluation(self.root, TAXONOMY, show=0)
        end_to_end = result["metrics"]["coverage_by_system_end_to_end"]
        conditional = result["metrics"][
            "coverage_by_system_conditional_on_valid_generation"
        ]
        self.assertEqual(
            0.5, end_to_end["evidence_first"]["full_tuple"]["micro"]["recall"]
        )
        self.assertEqual(
            1.0, conditional["evidence_first"]["full_tuple"]["micro"]["recall"]
        )
        self.assertEqual(
            1.0, end_to_end["v3"]["full_tuple"]["micro"]["recall"]
        )
        self.assertEqual(
            [failed_id],
            result["metrics"]["coverage_scopes"]
            ["conditional_on_valid_generation"]["excluded_review_ids"],
        )


if __name__ == "__main__":
    unittest.main()
