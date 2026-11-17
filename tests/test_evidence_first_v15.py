import hashlib
import json
import shutil
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

from reviewlens_ef import core as v14
from reviewlens_ef.cli import run_mode
from reviewlens_ef.v15_core import (
    complete_claims_v15,
    normalization_payload_v15,
    parse_observation_attempt_v15,
    parse_observation_response_v15,
    recovery_candidates_v15,
)
from reviewlens_ef.v15_dev import (
    DEV_RATING_TARGETS,
    LEGACY_PILOT_REVIEW_IDS,
    prepare_v15_dev,
    run_v15_dev,
    selection_document_v15,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TAXONOMY = {
    "performance": {
        "slow_performance": "slow",
        "processing_capability": "task performance",
    },
    "system_stability": {"boot_failure": "boot failure"},
    "battery": {
        "battery_runtime": "runtime",
        "charging": "charging hardware or behavior",
    },
    "display": {
        "screen_quality": "screen quality",
        "touchscreen": "touchscreen",
    },
    "audio": {
        "low_volume": "low volume",
        "speaker_sound": "speaker sound",
    },
    "input_devices": {
        "keyboard": "keyboard",
        "touchpad_mouse": "touchpad",
    },
    "connectivity_ports": {
        "wifi_bluetooth": "wireless",
        "other_connectivity": "other connectivity",
        "usb_other_ports": "physical ports",
    },
    "software_os": {
        "s_mode_restriction": "S Mode",
        "app_compatibility": "app compatibility",
        "os_setup": "OS setup",
    },
    "memory_storage": {
        "ram_capacity": "RAM amount",
        "storage_capacity": "storage amount",
        "upgradeability": "RAM/storage upgrades",
        "storage_performance": "drive performance",
    },
    "build_quality": {
        "hinge_assembly": "hinge",
        "housing_damage": "housing damage",
        "durability": "durability",
    },
    "design_portability": {
        "weight": "travel portability",
        "appearance": "appearance",
        "dimensions": "size and fit",
    },
    "price_value": {"value_for_money": "value"},
    "purchase_service": {"delivery_damage": "delivery damage"},
    "other": {"other_specific": "other supported issue"},
}


def observation(
    evidence,
    *,
    polarity="negative",
    statement_type="actual_experience",
    target="current_product",
    before=None,
    after=None,
):
    row = {
        "observation_id": "O001",
        "sentence_id": "R1_S001",
        "source_field": "clean_text",
        "evidence_text": evidence,
        "evidence_start_char_sentence": 0,
        "evidence_end_char_sentence": len(evidence),
        "evidence_start_char_source": 0,
        "evidence_end_char_source": len(evidence),
        "evidence_start_char_clean": 0,
        "evidence_end_char_clean": len(evidence),
        "sentence_text": evidence,
        "context_before_sentence_id": "R1_S000" if before else None,
        "context_before": before,
        "context_after_sentence_id": "R1_S002" if after else None,
        "context_after": after,
        "observation": evidence,
        "target": target,
        "polarity": polarity,
        "statement_type": statement_type,
        "locator_adjustments": [],
    }
    for direction, value in (("before", before), ("after", after)):
        if value:
            row[f"context_{direction}_provenance_validated"] = True
            row[f"context_{direction}_source_field"] = "clean_text"
            row[f"context_{direction}_start_char_source"] = 0
            row[f"context_{direction}_end_char_source"] = len(value)
            row[f"context_{direction}_start_char_clean"] = 0
            row[f"context_{direction}_end_char_clean"] = len(value)
        else:
            row[f"context_{direction}_provenance_validated"] = False
    return row


def normalized_gate(obs, category=None, issue_group=None):
    mapped = category is not None
    return v14.validate_claims(
        {
            "claims": [{
                "observation_id": "O001",
                "mapping_status": "mapped" if mapped else "unmapped",
                "category": category,
                "issue_group": issue_group,
                "claim": obs["observation"],
            }]
        },
        [obs],
        TAXONOMY,
    )


class V15ParserTests(unittest.TestCase):
    def test_bare_observation_list_is_wrapped_then_fully_validated(self):
        item = {
            "sentence_id": "R1_S001",
            "evidence_text": "This laptop is fast.",
            "observation": "This laptop is fast.",
            "target": "current_product",
            "polarity": "positive",
            "statement_type": "actual_experience",
        }
        raw = "<think></think>\n```json\n" + json.dumps([item]) + "\n```"
        parsed, adjustments = parse_observation_response_v15(raw)
        self.assertEqual(parsed, {"observations": [item]})
        self.assertEqual(adjustments[0]["type"], "wrapped_top_level_observation_list")

        review = {"review_id": "R1", "title": "", "clean_text": item["evidence_text"]}
        sentences = [{
            "review_id": "R1",
            "sentence_id": "R1_S001",
            "sentence_text": item["evidence_text"],
            "source_field": "clean_text",
            "start_char": 0,
            "end_char": len(item["evidence_text"]),
        }]
        _, gate, success = parse_observation_attempt_v15(raw, review, sentences)
        self.assertTrue(success)
        self.assertEqual(len(gate["accepted_observations"]), 1)
        self.assertFalse(gate["strict_object_shape"])
        self.assertEqual(
            gate["accepted_observations"][0]["evidence_text"],
            item["evidence_text"],
        )

    def test_v14_and_claim_parsers_remain_object_only(self):
        with self.assertRaisesRegex(ValueError, "expected one JSON object"):
            v14.parse_json_object("[]")
        with self.assertRaisesRegex(ValueError, "expected one JSON object"):
            v14.parse_json_object('[{"observation_id":"O001"}]')

    def test_unique_adjacent_antecedent_is_recovered_with_source_offsets(self):
        first = "The keyboard backlight is useful."
        second = "It is very dim."
        clean = first + " " + second
        sentences = [
            {
                "review_id": "R1",
                "sentence_id": "R1_S001",
                "sentence_text": first,
                "source_field": "clean_text",
                "start_char": 0,
                "end_char": len(first),
            },
            {
                "review_id": "R1",
                "sentence_id": "R1_S002",
                "sentence_text": second,
                "source_field": "clean_text",
                "start_char": len(first) + 1,
                "end_char": len(clean),
            },
        ]
        item = {
            "sentence_id": "R1_S002",
            "evidence_text": second,
            "observation": "The keyboard backlight is very dim.",
            "target": "current_product",
            "polarity": "negative",
            "statement_type": "actual_experience",
        }
        review = {"review_id": "R1", "title": "", "clean_text": clean}
        base = v14.validate_observations({"observations": [item]}, review, sentences)
        self.assertEqual(base["accepted_observations"], [])

        _, gate, success = parse_observation_attempt_v15(
            json.dumps({"observations": [item]}),
            review,
            sentences,
        )
        self.assertTrue(success)
        self.assertEqual(gate["context_recovery_count"], 1)
        accepted = gate["accepted_observations"][0]
        self.assertEqual(accepted["evidence_text"], second)
        context = accepted["context_recovery_adjustments"][0]
        self.assertEqual(context["sentence_id"], "R1_S001")
        self.assertEqual(context["quote_start_char_source"], 0)
        self.assertEqual(context["quote_end_char_source"], len(first))
        self.assertEqual(clean[0:len(first)], context["quote"])
        completed = complete_claims_v15(
            normalized_gate(accepted),
            [accepted],
            TAXONOMY,
        )
        claim = completed["accepted_claims"][0]
        self.assertEqual(
            (claim["category"], claim["issue_group"]),
            ("input_devices", "keyboard"),
        )
        self.assertEqual(claim["mapping_status"], "mapped")

    def test_ambiguous_or_generic_context_fragments_stay_rejected(self):
        texts = [
            "The keyboard backlight is useful.",
            "It is average.",
            "The keyboard backlight is uneven.",
        ]
        clean = " ".join(texts)
        sentences = []
        cursor = 0
        for number, text in enumerate(texts, start=1):
            sentences.append({
                "review_id": "R1",
                "sentence_id": f"R1_S{number:03d}",
                "sentence_text": text,
                "source_field": "clean_text",
                "start_char": cursor,
                "end_char": cursor + len(text),
            })
            cursor += len(text) + 1
        item = {
            "sentence_id": "R1_S002",
            "evidence_text": texts[1],
            "observation": "The keyboard backlight is average.",
            "target": "current_product",
            "polarity": "neutral",
            "statement_type": "actual_experience",
        }
        _, gate, _ = parse_observation_attempt_v15(
            json.dumps({"observations": [item]}),
            {"review_id": "R1", "title": "", "clean_text": clean},
            sentences,
        )
        self.assertEqual(gate["accepted_observations"], [])
        self.assertEqual(gate.get("context_recovery_count", 0), 0)

    def test_context_entity_must_match_without_competing_entities(self):
        for provider in (
            "The touchpad is useful.",
            "The keyboard and display are both poor.",
        ):
            current = "It is very dim."
            clean = provider + " " + current
            sentences = [
                {
                    "review_id": "R1",
                    "sentence_id": "R1_S001",
                    "sentence_text": provider,
                    "source_field": "clean_text",
                    "start_char": 0,
                    "end_char": len(provider),
                },
                {
                    "review_id": "R1",
                    "sentence_id": "R1_S002",
                    "sentence_text": current,
                    "source_field": "clean_text",
                    "start_char": len(provider) + 1,
                    "end_char": len(clean),
                },
            ]
            item = {
                "sentence_id": "R1_S002",
                "evidence_text": current,
                "observation": "The keyboard backlight is very dim.",
                "target": "current_product",
                "polarity": "negative",
                "statement_type": "actual_experience",
            }
            with self.subTest(provider=provider):
                _, gate, _ = parse_observation_attempt_v15(
                    json.dumps({"observations": [item]}),
                    {"review_id": "R1", "title": "", "clean_text": clean},
                    sentences,
                )
                self.assertEqual(gate["accepted_observations"], [])
                self.assertEqual(gate.get("context_recovery_count", 0), 0)


class V15RecoveryPolicyTests(unittest.TestCase):
    def assert_pair(self, obs, pair):
        candidates = recovery_candidates_v15(obs, TAXONOMY)
        self.assertEqual(set(candidates), {pair})
        return candidates[pair]

    def test_targeted_positive_and_negative_boundaries(self):
        cases = [
            (
                observation("Missing power cord"),
                ("battery", "charging"),
                observation("The power cord is short"),
            ),
            (
                observation(
                    "I use this laptop at home and travel better than my old one.",
                    polarity="positive",
                ),
                ("design_portability", "weight"),
                observation(
                    "I plan to travel with it",
                    polarity="positive",
                    statement_type="intended_use",
                ),
            ),
            (
                observation("It is a fairly good looking computer.", polarity="positive"),
                ("design_portability", "appearance"),
                observation("Take a good look at the computer", polarity="positive"),
            ),
            (
                observation(
                    "I can barely hear anything!",
                    before="The laptop sound is awful.",
                ),
                ("audio", "low_volume"),
                observation(
                    "I can barely hear the fan!",
                    before="The fan is quiet.",
                ),
            ),
            (
                observation(
                    "This laptop is so so upgrade-able",
                    polarity="positive",
                    after="As far as memory and SSD drives.",
                    statement_type="capability_or_limitation",
                ),
                ("memory_storage", "upgradeability"),
                observation(
                    "The battery is upgradeable",
                    polarity="positive",
                    after="The battery can be replaced.",
                    statement_type="capability_or_limitation",
                ),
            ),
            (
                observation(
                    "It does the job well",
                    polarity="positive",
                    before="I used it for lesson planning and school projects.",
                ),
                ("performance", "processing_capability"),
                observation("It does the job well", polarity="positive"),
            ),
        ]
        for positive, expected, negative in cases:
            with self.subTest(evidence=positive["evidence_text"]):
                self.assert_pair(positive, expected)
                self.assertEqual(recovery_candidates_v15(negative, TAXONOMY), {})

    def test_context_recovery_is_audited_and_keeps_exact_evidence(self):
        obs = observation(
            "I can barely hear anything!",
            before="The laptop sound is awful.",
        )
        result = complete_claims_v15(normalized_gate(obs), [obs], TAXONOMY)
        claim = result["accepted_claims"][0]
        self.assertEqual((claim["category"], claim["issue_group"]), ("audio", "low_volume"))
        self.assertEqual(claim["evidence"][0]["quote"], obs["evidence_text"])
        adjustment = claim["normalization_adjustments"][-1]
        self.assertEqual(adjustment["type"], "v15_recall_recovery")
        context = adjustment["rule_matches"][0]["context_evidence"]
        self.assertEqual(context[0]["sentence_id"], "R1_S000")
        self.assertEqual(context[0]["source_field"], "clean_text")
        self.assertEqual(context[0]["quote_start_char_source"], 0)
        self.assertEqual(
            context[0]["quote_end_char_source"],
            len("The laptop sound is awful."),
        )
        self.assertEqual(result["v15_recall_recovery_count"], 1)

    def test_known_holdout_phrasings_and_ambiguity_guards(self):
        expected = [
            (
                observation("It took me TWO HOURS to set them up"),
                ("software_os", "os_setup"),
            ),
            (
                observation("There is so much packaged into windows that bloats it!"),
                ("software_os", "os_setup"),
            ),
            (
                observation(
                    "in order to remove S mode, you have to buy Windows Home or Pro "
                    "and use the product key to remove S mode",
                    statement_type="capability_or_limitation",
                ),
                ("software_os", "s_mode_restriction"),
            ),
            (
                observation("Poor storage performace"),
                ("memory_storage", "storage_performance"),
            ),
            (
                observation(
                    "It's also really big??",
                    after="I can't find a nice case, so traveling is inconvenient.",
                ),
                ("design_portability", "dimensions"),
            ),
            (
                observation(
                    "Sure it’s a lil plastic-y on the bottom and around the screen.",
                    statement_type="feature_description",
                ),
                ("build_quality", "durability"),
            ),
            (
                observation(
                    "The shipping box was dented",
                    target="seller_platform",
                    statement_type="service_experience",
                ),
                ("purchase_service", "delivery_damage"),
            ),
        ]
        for obs, pair in expected:
            with self.subTest(evidence=obs["evidence_text"]):
                self.assert_pair(obs, pair)

        for unsafe in [
            observation("Fast transfers.", polarity="positive"),
            observation("It charges fast.", polarity="positive"),
            observation("Shipping was pretty fast.", polarity="positive"),
            observation("I used it for 5 hours"),
            observation("No issues with USB", polarity="positive"),
            observation("The power cord is short"),
            observation("Damage box and laptop"),
            observation(
                "This thing has next to no RAM/storage and needs to be upgraded"
            ),
        ]:
            with self.subTest(unsafe=unsafe["evidence_text"]):
                self.assertEqual(recovery_candidates_v15(unsafe, TAXONOMY), {})

        storage_speed = recovery_candidates_v15(
            observation("The storage performance is low"),
            TAXONOMY,
        )
        self.assertEqual(
            set(storage_speed),
            {("memory_storage", "storage_performance")},
        )

    def test_multiple_supported_pairs_remain_unmapped(self):
        obs = observation(
            "good battery life, easy to transport, works well for email and basic web browsing",
            polarity="positive",
            statement_type="capability_or_limitation",
        )
        candidates = recovery_candidates_v15(obs, TAXONOMY)
        self.assertGreaterEqual(len(candidates), 3)
        result = complete_claims_v15(normalized_gate(obs), [obs], TAXONOMY)
        self.assertEqual(result["accepted_claims"][0]["mapping_status"], "unmapped")
        self.assertEqual(result["v15_recall_recovery_count"], 0)

    def test_high_confidence_sibling_category_repairs(self):
        cases = [
            (
                observation("the top right side of the laptop has a dent in it"),
                ("build_quality", "hinge_assembly"),
                ("build_quality", "housing_damage"),
            ),
            (
                observation("the Acer box the laptop came in was damaged"),
                ("build_quality", "housing_damage"),
                ("purchase_service", "delivery_damage"),
            ),
            (
                observation('They removed the "create local account" option'),
                ("software_os", "s_mode_restriction"),
                ("software_os", "os_setup"),
            ),
            (
                observation("often would disconnect online till it would not connect at all"),
                ("connectivity_ports", "wifi_bluetooth"),
                ("connectivity_ports", "other_connectivity"),
            ),
            (
                observation(
                    "$300 black screen",
                    after="Defective...wouldn’t boot up, just gives me a black screen.",
                ),
                ("display", "screen_quality"),
                ("system_stability", "boot_failure"),
            ),
        ]
        for obs, model_pair, expected_pair in cases:
            with self.subTest(evidence=obs["evidence_text"]):
                gate = normalized_gate(obs, *model_pair)
                result = complete_claims_v15(gate, [obs], TAXONOMY)
                claim = result["accepted_claims"][0]
                self.assertEqual((claim["category"], claim["issue_group"]), expected_pair)
                self.assertEqual(result["v15_category_repair_count"], 1)
                self.assertEqual(
                    claim["normalization_adjustments"][-1]["type"],
                    "canonicalized_v15_taxonomy_pair",
                )

    def test_supported_model_pair_is_not_overwritten_by_unrelated_candidate(self):
        obs = observation(
            "This is a good-looking computer for the price",
            polarity="positive",
            statement_type="overall_opinion",
        )
        gate = normalized_gate(obs, "price_value", "value_for_money")
        result = complete_claims_v15(gate, [obs], TAXONOMY)
        claim = result["accepted_claims"][0]
        self.assertEqual(
            (claim["category"], claim["issue_group"]),
            ("price_value", "value_for_money"),
        )
        self.assertEqual(result["v15_category_repair_count"], 0)

    def test_unaudited_model_context_mapping_is_forced_to_abstain(self):
        obs = observation(
            "It is awkward to use.",
            before="The keyboard backlight is useful.",
        )
        obs["context_recovery_adjustments"] = [{
            "type": "adjacent_aspect_antecedent",
            "sentence_id": "R1_S000",
        }]
        gate = normalized_gate(obs, "software_os", "os_setup")
        result = complete_claims_v15(gate, [obs], TAXONOMY)
        claim = result["accepted_claims"][0]
        self.assertEqual(claim["mapping_status"], "unmapped")
        self.assertEqual(
            claim["normalization_adjustments"][-1]["type"],
            "v15_rejected_unaudited_context_mapping",
        )

    def test_context_is_not_exposed_to_normalization_model(self):
        obs = observation(
            "I can barely hear anything!",
            before="The laptop sound is awful.",
        )
        payload = normalization_payload_v15({
            "review_id": "R1",
            "accepted_observations": [obs],
        })
        item = payload["validated_observations"][0]
        self.assertNotIn("context_before", item)
        self.assertNotIn("context_after", item)

    def test_correct_model_context_mapping_gets_audit_adjustment(self):
        obs = observation(
            "It lasts 8 hours",
            polarity="positive",
            before="The battery performance is good.",
        )
        gate = normalized_gate(obs, "battery", "battery_runtime")
        result = complete_claims_v15(gate, [obs], TAXONOMY)
        claim = result["accepted_claims"][0]
        self.assertEqual(
            claim["normalization_adjustments"][-1]["type"],
            "v15_context_grounding_validated",
        )
        self.assertEqual(
            claim["normalization_source"],
            "model_with_v15_context_validation",
        )

    def test_contextual_wrong_pair_counts_as_category_repair(self):
        obs = observation(
            "I can barely hear anything!",
            before="The speaker volume is low.",
        )
        obs["context_recovery_adjustments"] = [{
            "type": "adjacent_aspect_antecedent",
            "sentence_id": "R1_S000",
        }]
        gate = normalized_gate(obs, "display", "touchscreen")
        result = complete_claims_v15(gate, [obs], TAXONOMY)
        claim = result["accepted_claims"][0]
        self.assertEqual(
            (claim["category"], claim["issue_group"]),
            ("audio", "low_volume"),
        )
        self.assertEqual(result["v15_category_repair_count"], 1)
        self.assertEqual(result["v15_recall_recovery_count"], 0)

    def test_direction_specific_context_uses_only_the_antecedent_side(self):
        obs = observation(
            "I can barely hear anything!",
            before="The speaker volume is low.",
            after="The sound remains quiet.",
        )
        candidates = recovery_candidates_v15(obs, TAXONOMY)
        matches = candidates[("audio", "low_volume")]
        context = matches[0]["context_evidence"]
        self.assertEqual(len(context), 1)
        self.assertEqual(context[0]["direction"], "before")

    def test_intentions_and_expectations_stay_unmapped(self):
        for statement_type, evidence in [
            ("intended_use", "I plan to travel with it"),
            ("expectation_or_assumption", "I wish it had a 12 hour battery"),
        ]:
            obs = observation(evidence, statement_type=statement_type)
            self.assertEqual(recovery_candidates_v15(obs, TAXONOMY), {})


class V15SelectionAndIntegrationTests(unittest.TestCase):
    def test_cli_defaults_to_ten_and_prepare_rejects_force(self):
        with mock.patch("reviewlens_ef.v15_dev.run_v15_dev") as runner:
            run_mode(
                "ef-v15-dev",
                PROJECT_ROOT,
                TAXONOMY,
                "fake-model",
                limit=0,
            )
        self.assertEqual(runner.call_args.kwargs["limit"], 10)
        with self.assertRaisesRegex(ValueError, "cannot overwrite"):
            run_mode(
                "ef-v15-dev-prepare",
                PROJECT_ROOT,
                TAXONOMY,
                "fake-model",
                force=True,
            )

    def test_real_selection_is_deterministic_disjoint_and_prefix_balanced(self):
        first = selection_document_v15(PROJECT_ROOT)
        second = selection_document_v15(PROJECT_ROOT)
        self.assertEqual(first, second)
        rows = first["selected_reviews"]
        self.assertEqual(len(rows), 20)
        self.assertEqual(
            Counter(row["rating"] for row in rows),
            Counter(DEV_RATING_TARGETS),
        )
        self.assertEqual(
            Counter(row["rating"] for row in rows[:10]),
            Counter({1: 2, 2: 2, 3: 2, 4: 2, 5: 2}),
        )
        selected = {row["review_id"] for row in rows}
        self.assertFalse(selected & set(v14.PILOT_REVIEW_IDS))
        self.assertFalse(selected & set(LEGACY_PILOT_REVIEW_IDS))
        reused = set(first["reused_diagnostic_3star_review_ids"])
        self.assertEqual(len(reused), 4)
        self.assertEqual(
            selected & set(first["excluded_diagnostic_holdout_review_ids"]),
            reused,
        )
        self.assertEqual(len(first["reserved_unseen_3star_review_ids"]), 3)
        self.assertFalse(selected & set(first["reserved_unseen_3star_review_ids"]))

    def test_frozen_selection_rejects_source_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_dir = root / "data/processed/amazon"
            source_dir.mkdir(parents=True)
            for name in (
                "acer_aspire5_reviews_clean.jsonl",
                "acer_aspire5_sentences.jsonl",
            ):
                shutil.copy2(PROJECT_ROOT / "data/processed/amazon" / name, source_dir / name)
            output_dir = root / "outputs/amazon"
            output_dir.mkdir(parents=True)
            shutil.copy2(
                PROJECT_ROOT / "outputs/amazon/acer_aspire5_full_predictions.jsonl",
                output_dir / "acer_aspire5_full_predictions.jsonl",
            )
            holdout_dir = output_dir / "evidence_first/holdout"
            holdout_dir.mkdir(parents=True)
            shutil.copy2(
                PROJECT_ROOT / "outputs/amazon/evidence_first/holdout/selection_manifest.json",
                holdout_dir / "selection_manifest.json",
            )
            frozen = prepare_v15_dev(root)
            reviews_path = source_dir / "acer_aspire5_reviews_clean.jsonl"
            reviews = [json.loads(line) for line in reviews_path.read_text().splitlines()]
            changed_id = frozen["selected_reviews"][0]["review_id"]
            changed = next(row for row in reviews if row["review_id"] == changed_id)
            changed["title"] = str(changed.get("title", "")) + " changed"
            reviews_path.write_text(
                "".join(json.dumps(row) + "\n" for row in reviews),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "immutable"):
                prepare_v15_dev(root)
            selection_path = output_dir / "evidence_first/v15_dev/selection_manifest.json"
            self.assertEqual(json.loads(selection_path.read_text()), frozen)

    def test_prepare_run_resume_and_old_artifact_isolation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_dir = root / "data/processed/amazon"
            source_dir.mkdir(parents=True)
            for name in (
                "acer_aspire5_reviews_clean.jsonl",
                "acer_aspire5_sentences.jsonl",
            ):
                shutil.copy2(PROJECT_ROOT / "data/processed/amazon" / name, source_dir / name)
            output_dir = root / "outputs/amazon"
            output_dir.mkdir(parents=True)
            shutil.copy2(
                PROJECT_ROOT / "outputs/amazon/acer_aspire5_full_predictions.jsonl",
                output_dir / "acer_aspire5_full_predictions.jsonl",
            )
            old_pilot = output_dir / "evidence_first/observations_pilot.jsonl"
            old_pilot.parent.mkdir(parents=True)
            old_pilot.write_text("old-pilot-sentinel\n", encoding="utf-8")
            old_holdout_dir = output_dir / "evidence_first/holdout"
            old_holdout_dir.mkdir(parents=True)
            shutil.copy2(
                PROJECT_ROOT / "outputs/amazon/evidence_first/holdout/selection_manifest.json",
                old_holdout_dir / "selection_manifest.json",
            )
            old_claims = old_holdout_dir / "claims_holdout.jsonl"
            old_claims.write_text("old-holdout-sentinel\n", encoding="utf-8")

            prepare_v15_dev(root)
            self.assertEqual(old_pilot.read_text(), "old-pilot-sentinel\n")
            self.assertEqual(old_claims.read_text(), "old-holdout-sentinel\n")

            calls = []

            def fake_generator(system_prompt, payload, repair):
                self.assertIsNone(repair)
                calls.append((system_prompt, payload["review_id"]))
                if "extract evidence-first observations" in system_prompt:
                    source = payload["sentences_and_layer1_clues"][0]
                    return json.dumps({
                        "observations": [{
                            "sentence_id": source["sentence_id"],
                            "evidence_text": source["text"],
                            "observation": source["text"],
                            "target": "current_product",
                            "polarity": "neutral",
                            "statement_type": "overall_opinion",
                        }]
                    })
                return json.dumps({
                    "claims": [{
                        "observation_id": item["observation_id"],
                        "mapping_status": "unmapped",
                        "category": None,
                        "issue_group": None,
                        "claim": item["observation"],
                    } for item in payload["validated_observations"]]
                })

            first = run_v15_dev(
                root,
                {"other": {"other_specific": "other"}},
                "fake-model",
                limit=10,
                max_retries=0,
                show=0,
                generator=fake_generator,
            )
            self.assertEqual(len(first["selected_review_ids"]), 10)
            first_call_count = len(calls)
            self.assertGreaterEqual(first_call_count, 10)

            calls.clear()
            second = run_v15_dev(
                root,
                {"other": {"other_specific": "other"}},
                "fake-model",
                limit=20,
                max_retries=0,
                show=0,
                generator=fake_generator,
            )
            self.assertEqual(len(second["selected_review_ids"]), 20)
            self.assertEqual(
                Counter(row["cohort_origin"] for row in second["claim_rows"]),
                Counter({"new_dev": 16, "diagnostic_3star_reuse": 4}),
            )
            observation_calls = [
                call for call in calls
                if "extract evidence-first observations" in call[0]
            ]
            self.assertEqual(len(observation_calls), 10)

            calls.clear()
            run_v15_dev(
                root,
                {"other": {"other_specific": "other"}},
                "fake-model",
                limit=20,
                max_retries=0,
                show=0,
                generator=fake_generator,
            )
            self.assertEqual(calls, [])
            self.assertEqual(old_pilot.read_text(), "old-pilot-sentinel\n")
            self.assertEqual(old_claims.read_text(), "old-holdout-sentinel\n")

            v15_dir = output_dir / "evidence_first/v15_dev"
            (v15_dir / "run_manifest.json").unlink()
            with self.assertRaisesRegex(ValueError, "without their run manifest"):
                run_v15_dev(
                    root,
                    {"other": {"other_specific": "other"}},
                    "fake-model",
                    limit=20,
                    max_retries=0,
                    show=0,
                    generator=fake_generator,
                )

            calls.clear()
            run_v15_dev(
                root,
                {"other": {"other_specific": "other"}},
                "fake-model",
                limit=20,
                force=True,
                max_retries=0,
                show=0,
                generator=fake_generator,
            )
            self.assertGreater(len(calls), 0)
            self.assertTrue(list(v15_dir.glob("observations_v15_dev.backup_*.jsonl")))
            self.assertTrue(list(v15_dir.glob("claims_v15_dev.backup_*.jsonl")))

            observation_cache = v15_dir / "observations_v15_dev.jsonl"
            rows = [json.loads(line) for line in observation_cache.read_text().splitlines()]
            rows[0]["accepted_observations"][0]["observation"] += " CORRUPTED"
            observation_cache.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "record_hash does not match"):
                run_v15_dev(
                    root,
                    {"other": {"other_specific": "other"}},
                    "fake-model",
                    limit=20,
                    max_retries=0,
                    show=0,
                    generator=fake_generator,
                )
            self.assertEqual(old_pilot.read_text(), "old-pilot-sentinel\n")
            self.assertEqual(old_claims.read_text(), "old-holdout-sentinel\n")

    def test_frozen_v14_model_files_remain_byte_identical(self):
        expected = {
            "reviewlens_ef/core.py": "09370ae6320f7660015bc2b376b2c09c6b1821d103b1bbdc82bb5b76da58cb2c",
            "reviewlens_ef/pipeline.py": "6fdbb60a3bb83a423dce774923e183ab334eca13bc4f5513f0ca8109e5e87f95",
            "reviewlens_ef/prompts.py": "99db5d01d150a939677436c37073e17ee216a6e232057b9f4dcea842dc4d5c78",
            "reviewlens_ef/holdout.py": "45d9cc33f731956afecc261c163f476fd3f04f482e3575805f63a3158ea5277c",
        }
        for relative, digest in expected.items():
            with self.subTest(file=relative):
                actual = hashlib.sha256((PROJECT_ROOT / relative).read_bytes()).hexdigest()
                self.assertEqual(actual, digest)


if __name__ == "__main__":
    unittest.main()
