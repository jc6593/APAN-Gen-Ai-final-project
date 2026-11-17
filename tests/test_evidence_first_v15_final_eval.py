import ast
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from reviewlens_ef.cli import run_mode
from reviewlens_ef.v15_final_eval import (
    _blind_primary_items,
    _diagnostic_items,
    _validate_locked_inputs,
    evaluate_v15_final,
    prepare_v15_final_evaluation,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_frozen_taxonomy():
    tree = ast.parse((PROJECT_ROOT / "noteboks/15_review_analysis.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "ISSUE_GROUPS"
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError("ISSUE_GROUPS not found")


TAXONOMY = load_frozen_taxonomy()


def copy_locked_fixture(destination):
    destination = Path(destination)
    for relative in (
        "data/processed/amazon/acer_aspire5_reviews_clean.jsonl",
        "outputs/amazon/acer_aspire5_business_full_v3.jsonl",
    ):
        source = PROJECT_ROOT / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    final_source = PROJECT_ROOT / "outputs/amazon/evidence_first/v15_final"
    final_target = destination / "outputs/amazon/evidence_first/v15_final"
    final_target.mkdir(parents=True)
    for name in (
        "selection_manifest.json",
        "gold_reference.json",
        "run_manifest.json",
        "observations_v15_final.jsonl",
        "claims_v15_final.jsonl",
        "completion_manifest.json",
    ):
        shutil.copy2(final_source / name, final_target / name)


def write_complete_labels(project_root, one_v3_error=False):
    root = Path(project_root)
    locked = _validate_locked_inputs(root, TAXONOMY)
    _, primary_key = _blind_primary_items(locked)
    _, diagnostic_key = _diagnostic_items(locked)
    precision_rows = []
    error_written = False
    for blind_id, item in primary_key.items():
        supported = "yes"
        if one_v3_error and item["system"] == "v3" and not error_written:
            supported = "no"
            error_written = True
        precision_rows.append({
            "blind_id": blind_id,
            "supported": supported,
            "category_correct": "yes",
            "sentiment_correct": "yes",
            "target_correct": "yes",
            "notes": "fixture",
        })
    diagnostic_rows = []
    for diagnostic_id, item in diagnostic_key.items():
        if item["mapping_status"] == "mapped":
            diagnostic_rows.append({
                "diagnostic_id": diagnostic_id,
                "supported": "yes",
                "atomic": "yes",
                "category_correct": "yes",
                "sentiment_correct": "yes",
                "target_correct": "yes",
                "notes": "fixture",
            })
        else:
            diagnostic_rows.append({
                "diagnostic_id": diagnostic_id,
                "should_map": "no",
                "expected_category": None,
                "expected_issue_group": None,
                "statement_type_correct": "yes",
                "notes": "fixture",
            })
    output = root / "outputs/amazon/evidence_first/v15_final"
    (output / "precision_labels.json").write_text(
        json.dumps({"labels": precision_rows}, indent=2, sort_keys=True) + "\n"
    )
    (output / "diagnostic_labels.json").write_text(
        json.dumps({"labels": diagnostic_rows}, indent=2, sort_keys=True) + "\n"
    )


class V15FinalEvaluationTests(unittest.TestCase):
    def test_complete_labels_produce_reproducible_locked_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            copy_locked_fixture(temporary)
            blind = prepare_v15_final_evaluation(temporary, TAXONOMY)
            self.assertEqual(len(blind["items"]), 82)
            self.assertEqual(len(blind["diagnostic_items"]), 81)
            self.assertEqual(
                sum(
                    item["mapping_status"] == "mapped"
                    for item in blind["diagnostic_items"]
                ),
                33,
            )
            write_complete_labels(temporary, one_v3_error=True)
            result = evaluate_v15_final(temporary, TAXONOMY, show=0)
            rerun = evaluate_v15_final(temporary, TAXONOMY, show=0)
            self.assertEqual(result, rerun)
            self.assertEqual(result["counts"]["gold_full_tuples"], 60)
            self.assertEqual(
                result["primary_strict_precision"]["v3"]
                ["strict_fully_correct"]["yes"],
                51,
            )
            self.assertEqual(
                result["primary_strict_precision"]["evidence_first"]
                ["strict_fully_correct"]["yes"],
                30,
            )
            self.assertEqual(
                result["gold_recall"]["v3"]["full_tuple"]["micro"],
                {"hits": 32, "gold": 60, "rate": 0.533333},
            )
            self.assertEqual(
                result["gold_recall"]["evidence_first"]
                ["full_tuple"]["micro"],
                {"hits": 25, "gold": 60, "rate": 0.416667},
            )
            self.assertEqual(
                result["gold_recall"]["v3"]["issue_pair"]["micro"],
                {"hits": 36, "gold": 56, "rate": 0.642857},
            )
            self.assertEqual(
                result["gold_recall"]["evidence_first"]
                ["issue_pair"]["micro"],
                {"hits": 26, "gold": 56, "rate": 0.464286},
            )
            self.assertEqual(
                result["improvement_gate"]["decision"],
                "precision_recall_tradeoff",
            )
            self.assertFalse(result["improvement_gate"]["passed"])
            self.assertEqual(
                result["atomic_and_abstention_diagnostics"]
                ["mapped_atomic"]["strict_fully_correct"]["yes"],
                33,
            )
            self.assertEqual(
                result["atomic_and_abstention_diagnostics"]
                ["unmapped_abstentions"]["should_map"]["yes"],
                0,
            )
            self.assertEqual(
                result["completion_and_failures"]["v3"]
                ["failure_review_ids"],
                ["electronics_1807797"],
            )
            output = (
                Path(temporary)
                / "outputs/amazon/evidence_first/v15_final/final_evaluation.json"
            )
            self.assertTrue(output.exists())

    def test_incomplete_labels_and_locked_cache_tamper_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            copy_locked_fixture(temporary)
            prepare_v15_final_evaluation(temporary, TAXONOMY)
            write_complete_labels(temporary)
            output = Path(temporary) / "outputs/amazon/evidence_first/v15_final"
            labels = json.loads((output / "precision_labels.json").read_text())
            labels["labels"].pop()
            (output / "precision_labels.json").write_text(json.dumps(labels) + "\n")
            with self.assertRaisesRegex(ValueError, "Incomplete final precision labels"):
                evaluate_v15_final(temporary, TAXONOMY)

        with tempfile.TemporaryDirectory() as temporary:
            copy_locked_fixture(temporary)
            claims = (
                Path(temporary)
                / "outputs/amazon/evidence_first/v15_final/claims_v15_final.jsonl"
            )
            claims.write_text(claims.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "completion manifest mismatch"):
                prepare_v15_final_evaluation(temporary, TAXONOMY)

    def test_cli_forbids_final_eval_mutation_flags(self):
        for kwargs in ({"force": True}, {"limit": 5}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                run_mode(
                    "ef-v15-final-eval",
                    PROJECT_ROOT,
                    TAXONOMY,
                    "unused-model",
                    **kwargs,
                )


if __name__ == "__main__":
    unittest.main()
