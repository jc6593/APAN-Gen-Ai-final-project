import ast
import hashlib
import json
import shutil
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

from reviewlens_ef.cli import run_mode
from reviewlens_ef.core import taxonomy_hash
from reviewlens_ef.v15_final import (
    EVALUATION_PROTOCOL,
    FINAL_RATING_TARGETS,
    FINAL_SIZE,
    prepare_v15_final,
    run_v15_final,
    selection_document_v15_final,
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
MODEL_ID = "mlx-community/Qwen3-4B-Instruct-2507-4bit"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def copy_project_inputs(destination):
    destination = Path(destination)
    source_data = PROJECT_ROOT / "data/processed/amazon"
    target_data = destination / "data/processed/amazon"
    target_data.mkdir(parents=True)
    for name in ("acer_aspire5_reviews_clean.jsonl", "acer_aspire5_sentences.jsonl"):
        shutil.copy2(source_data / name, target_data / name)

    source_outputs = PROJECT_ROOT / "outputs/amazon"
    target_outputs = destination / "outputs/amazon"
    target_outputs.mkdir(parents=True)
    for name in (
        "acer_aspire5_full_predictions.jsonl",
        "acer_aspire5_business_full_v3.jsonl",
        "acer_aspire5_business_pilot_v3.jsonl",
    ):
        shutil.copy2(source_outputs / name, target_outputs / name)
    pilot_target = target_outputs / "evidence_first"
    pilot_target.mkdir(parents=True)
    shutil.copy2(
        source_outputs / "evidence_first/pilot_manifest.json",
        pilot_target / "pilot_manifest.json",
    )
    for cohort, names in {
        "holdout": ("selection_manifest.json",),
        "v15_dev": (
            "selection_manifest.json",
            "run_manifest.json",
            "observations_v15_dev.jsonl",
            "claims_v15_dev.jsonl",
        ),
    }.items():
        target = target_outputs / "evidence_first" / cohort
        target.mkdir(parents=True)
        for name in names:
            shutil.copy2(source_outputs / "evidence_first" / cohort / name, target / name)


class EmptyObservationGenerator:
    def __init__(self):
        self.calls = 0

    def __call__(self, system_prompt, payload, repair=None):
        self.calls += 1
        return json.dumps({"observations": []})


class FailingGenerator:
    def __call__(self, system_prompt, payload, repair=None):
        raise AssertionError("completed final cache must not call the generator")


class InterruptingGenerator:
    def __init__(self, successful_calls):
        self.successful_calls = successful_calls
        self.calls = 0

    def __call__(self, system_prompt, payload, repair=None):
        self.calls += 1
        if self.calls > self.successful_calls:
            raise RuntimeError("simulated interruption")
        return json.dumps({"observations": []})


def write_empty_gold(project_root):
    output = Path(project_root) / "outputs/amazon/evidence_first/v15_final"
    selection_path = output / "selection_manifest.json"
    selection = json.loads(selection_path.read_text())
    selected_ids = [row["review_id"] for row in selection["selected_reviews"]]
    reviews = {
        row["review_id"]: row
        for row in (
            json.loads(line)
            for line in (
                Path(project_root)
                / "data/processed/amazon/acer_aspire5_reviews_clean.jsonl"
            ).read_text().splitlines()
        )
    }
    review_rows = []
    for index, review_id in enumerate(selected_ids):
        source = reviews[review_id]
        findings = []
        no_finding_reason = "No specific taxonomy finding in this test fixture."
        if index == 0:
            quote = source["title"]
            findings = [{
                "category": "battery",
                "issue_group": "charging",
                "sentiment": "negative",
                "attribution": "current_product",
                "evidence": [{
                    "source_field": "title",
                    "quote": quote,
                    "start_char": 0,
                    "end_char": len(quote),
                }],
            }]
            no_finding_reason = None
        review_rows.append({
            "review_id": review_id,
            "gold_complete": True,
            "source_text_hashes": {
                "title": hashlib.sha256(source["title"].encode()).hexdigest(),
                "clean_text": hashlib.sha256(source["clean_text"].encode()).hexdigest(),
            },
            "gold_findings": findings,
            "no_finding_reason": no_finding_reason,
        })
    document = {
        "schema_version": "reviewlens_v15_final_gold_v1",
        "status": "frozen_complete_before_final_inference",
        "selection_manifest_sha256": sha256(selection_path),
        "evaluation_protocol_hash": selection["evaluation_protocol_hash"],
        "taxonomy_hash": taxonomy_hash(TAXONOMY),
        "selected_review_ids": selected_ids,
        "reviews": review_rows,
    }
    (output / "gold_reference.json").write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )


class V15FinalSelectionTests(unittest.TestCase):
    def test_real_selection_is_balanced_disjoint_and_length_only(self):
        document = selection_document_v15_final(PROJECT_ROOT)
        rows = document["selected_reviews"]
        self.assertEqual(len(rows), FINAL_SIZE)
        self.assertEqual(Counter(row["rating"] for row in rows), Counter(FINAL_RATING_TARGETS))
        self.assertEqual(document["exclusion_union_count"], 54)
        self.assertEqual(document["eligible_review_count"], 135)
        self.assertEqual(
            document["eligible_rating_distribution"],
            {"1": 7, "2": 3, "3": 3, "4": 18, "5": 104},
        )
        self.assertEqual(
            [row["review_id"] for row in rows],
            [
                "electronics_395247", "electronics_1046023", "electronics_800032",
                "electronics_52771", "electronics_1213818", "electronics_1549899",
                "electronics_943842", "electronics_219260", "electronics_1170992",
                "electronics_830445", "electronics_1807797", "electronics_1444882",
                "electronics_502169", "electronics_956109", "electronics_977364",
            ],
        )
        excluded = set().union(*(
            set(values) for values in document["excluded_review_ids"].values()
        ))
        self.assertFalse(excluded & {row["review_id"] for row in rows})
        self.assertEqual(document["selection_features"], [
            "review_id", "rating", "len(clean_text)"
        ])
        self.assertEqual(document["evaluation_protocol"], EVALUATION_PROTOCOL)
        for row in rows:
            self.assertEqual(
                row["sampling_weight_for_eligible_pool"], row["length_bin_size"]
            )

    def test_frozen_v15_model_files_are_unchanged(self):
        expected = {
            "reviewlens_ef/v15_core.py": "bea22366bb75ddfe76b8d625ef5ab9dfa3d3efa43e0eaee1b2b2d1950c66ff0e",
            "reviewlens_ef/v15_dev.py": "55501a21bc801aeb9dfcef6e6d1860eaf62c5c053ae86eed022eb360ed0d11f1",
            "reviewlens_ef/v15_prompts.py": "cd75ec935d6d5568fc5263b7e749633ebfefe45542bb3798baaab0f6b8614d05",
        }
        self.assertEqual(
            {name: sha256(PROJECT_ROOT / name) for name in expected}, expected
        )

    def test_prepare_is_immutable_and_does_not_create_inference_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            copy_project_inputs(temporary)
            first = prepare_v15_final(temporary)
            second = prepare_v15_final(temporary)
            self.assertEqual(first, second)
            output = Path(temporary) / "outputs/amazon/evidence_first/v15_final"
            self.assertEqual(
                sorted(path.name for path in output.iterdir()),
                ["selection_manifest.json"],
            )


class V15FinalRunTests(unittest.TestCase):
    def test_locked_run_completes_and_resume_never_calls_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            copy_project_inputs(temporary)
            fake_snapshot = Path(temporary) / "fake-model-snapshot"
            fake_snapshot.mkdir()
            generator = EmptyObservationGenerator()
            with mock.patch(
                "reviewlens_ef.v15_final._local_model_revision",
                return_value="fake-revision",
            ), mock.patch(
                "reviewlens_ef.v15_final._local_model_snapshot",
                return_value=fake_snapshot.resolve(),
            ):
                prepare_v15_final(temporary)
                write_empty_gold(temporary)
                result = run_v15_final(
                    temporary, TAXONOMY, MODEL_ID, generator=generator
                )
                self.assertEqual(generator.calls, FINAL_SIZE)
                self.assertEqual(len(result["observation_rows"]), FINAL_SIZE)
                self.assertEqual(len(result["claim_rows"]), FINAL_SIZE)
                resumed = run_v15_final(
                    temporary, TAXONOMY, MODEL_ID, generator=FailingGenerator()
                )
            self.assertEqual(result["selected_review_ids"], resumed["selected_review_ids"])
            output = Path(temporary) / "outputs/amazon/evidence_first/v15_final"
            self.assertTrue((output / "completion_manifest.json").exists())
            self.assertEqual(
                len((output / "observations_v15_final.jsonl").read_text().splitlines()),
                FINAL_SIZE,
            )
            self.assertEqual(
                len((output / "claims_v15_final.jsonl").read_text().splitlines()),
                FINAL_SIZE,
            )

    def test_interruption_resumes_but_completed_tamper_never_regenerates(self):
        with tempfile.TemporaryDirectory() as temporary:
            copy_project_inputs(temporary)
            fake_snapshot = Path(temporary) / "fake-model-snapshot"
            fake_snapshot.mkdir()
            with mock.patch(
                "reviewlens_ef.v15_final._local_model_revision",
                return_value="fake-revision",
            ), mock.patch(
                "reviewlens_ef.v15_final._local_model_snapshot",
                return_value=fake_snapshot.resolve(),
            ):
                prepare_v15_final(temporary)
                write_empty_gold(temporary)
                interrupted = InterruptingGenerator(4)
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    run_v15_final(
                        temporary, TAXONOMY, MODEL_ID, generator=interrupted
                    )
                output = Path(temporary) / "outputs/amazon/evidence_first/v15_final"
                self.assertEqual(
                    len((output / "observations_v15_final.jsonl").read_text().splitlines()),
                    4,
                )
                resumed = EmptyObservationGenerator()
                run_v15_final(temporary, TAXONOMY, MODEL_ID, generator=resumed)
                self.assertEqual(resumed.calls, FINAL_SIZE - 4)

                claims_path = output / "claims_v15_final.jsonl"
                lines = claims_path.read_text().splitlines()
                claims_path.write_text("\n".join(lines[:-1]) + "\n")
                with self.assertRaisesRegex(ValueError, "completion manifest"):
                    run_v15_final(
                        temporary,
                        TAXONOMY,
                        MODEL_ID,
                        generator=FailingGenerator(),
                    )

    def test_model_and_taxonomy_are_bound_to_completed_dev(self):
        with tempfile.TemporaryDirectory() as temporary:
            copy_project_inputs(temporary)
            fake_snapshot = Path(temporary) / "fake-model-snapshot"
            fake_snapshot.mkdir()
            with mock.patch(
                "reviewlens_ef.v15_final._local_model_revision",
                return_value="fake-revision",
            ), mock.patch(
                "reviewlens_ef.v15_final._local_model_snapshot",
                return_value=fake_snapshot.resolve(),
            ):
                prepare_v15_final(temporary)
                write_empty_gold(temporary)
                with self.assertRaisesRegex(ValueError, "Final model is frozen"):
                    run_v15_final(
                        temporary, TAXONOMY, "different-model", generator=FailingGenerator()
                    )
                changed_taxonomy = json.loads(json.dumps(TAXONOMY))
                changed_taxonomy["performance"]["slow_performance"] += " changed"
                with self.assertRaisesRegex(ValueError, "model/taxonomy mismatch"):
                    run_v15_final(
                        temporary,
                        changed_taxonomy,
                        MODEL_ID,
                        generator=FailingGenerator(),
                    )

    def test_gold_requires_exact_source_evidence_and_nonempty_completion(self):
        with tempfile.TemporaryDirectory() as temporary:
            copy_project_inputs(temporary)
            fake_snapshot = Path(temporary) / "fake-model-snapshot"
            fake_snapshot.mkdir()
            with mock.patch(
                "reviewlens_ef.v15_final._local_model_revision",
                return_value="fake-revision",
            ), mock.patch(
                "reviewlens_ef.v15_final._local_model_snapshot",
                return_value=fake_snapshot.resolve(),
            ):
                prepare_v15_final(temporary)
                write_empty_gold(temporary)
                gold_path = (
                    Path(temporary)
                    / "outputs/amazon/evidence_first/v15_final/gold_reference.json"
                )
                gold = json.loads(gold_path.read_text())
                evidence = gold["reviews"][0]["gold_findings"][0]["evidence"][0]
                evidence.update({
                    "quote": "THIS QUOTE DOES NOT EXIST",
                    "start_char": 0,
                    "end_char": len("THIS QUOTE DOES NOT EXIST"),
                })
                gold_path.write_text(json.dumps(gold) + "\n")
                with self.assertRaisesRegex(ValueError, "exact source span"):
                    run_v15_final(
                        temporary, TAXONOMY, MODEL_ID, generator=FailingGenerator()
                    )

                write_empty_gold(temporary)
                gold = json.loads(gold_path.read_text())
                for row in gold["reviews"]:
                    row["gold_findings"] = []
                    row["no_finding_reason"] = "Adjudicated no finding."
                gold_path.write_text(json.dumps(gold) + "\n")
                with self.assertRaisesRegex(ValueError, "contains no findings"):
                    run_v15_final(
                        temporary, TAXONOMY, MODEL_ID, generator=FailingGenerator()
                    )

    def test_cli_forbids_peeking_and_regeneration(self):
        for mode in ("ef-v15-final-prepare", "ef-v15-final"):
            with self.subTest(mode=mode):
                with self.assertRaises(ValueError):
                    run_mode(mode, PROJECT_ROOT, TAXONOMY, "fake-model", force=True)
                with self.assertRaises(ValueError):
                    run_mode(mode, PROJECT_ROOT, TAXONOMY, "fake-model", limit=5)
        with self.assertRaisesRegex(ValueError, "requires --max-retries 1"):
            run_mode(
                "ef-v15-final",
                PROJECT_ROOT,
                TAXONOMY,
                "fake-model",
                max_retries=0,
            )


if __name__ == "__main__":
    unittest.main()
