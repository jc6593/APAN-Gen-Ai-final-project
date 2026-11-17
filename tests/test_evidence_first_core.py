import json
import tempfile
import unittest
from pathlib import Path

from reviewlens_ef.core import (
    CLAIM_SCHEMA_VERSION,
    OBSERVATION_SCHEMA_VERSION,
    PILOT_REVIEW_IDS,
    complete_claims_with_unmapped_fallbacks,
    hash_json,
    load_stage_cache,
    merge_claims_by_tuple,
    parse_json_object,
    select_pilot_ids,
    taxonomy_hash,
    validate_claims,
    validate_observations,
)
from reviewlens_ef.pipeline import run_pilot


TAXONOMY = {
    "battery": {"battery_runtime": "battery life per charge"},
    "display": {"screen_quality": "screen quality"},
}

BOUNDARY_TAXONOMY = {
    "performance": {
        "slow_performance": "lag, slow performance or poor responsiveness",
        "processing_capability": "processor/graphics ability to handle tasks or games",
    },
    "system_stability": {
        "freezing_crashing": "freezing, crashes or restarts",
    },
    "design_portability": {
        "weight": "weight and portability",
        "appearance": "appearance and design",
    },
    "memory_storage": {
        "storage_capacity": "storage space",
        "storage_performance": "storage speed",
        "upgradeability": "adding or replacing RAM/storage",
    },
    "battery": {
        "battery_runtime": "battery life per charge",
        "charging": "charging behavior",
    },
    "audio": {"speaker_sound": "speaker sound quality"},
    "input_devices": {
        "keyboard": "keyboard experience",
        "touchpad_mouse": "touchpad experience",
    },
    "connectivity_ports": {"usb_other_ports": "physical ports"},
    "price_value": {
        "price": "price itself",
        "value_for_money": "value relative to price",
    },
    "purchase_service": {"support_service": "support interaction"},
    "display": {"screen_quality": "screen quality"},
    "software_os": {"app_compatibility": "app compatibility"},
}


class ObservationValidationTests(unittest.TestCase):
    def setUp(self):
        self.review = {
            "review_id": "r1",
            "clean_text": "Battery good. Screen bad.",
        }
        self.sentences = [
            {
                "review_id": "r1",
                "sentence_id": "r1_S001",
                "sentence_text": "Battery good.",
                "start_char": 0,
                "end_char": 13,
            },
            {
                "review_id": "r1",
                "sentence_id": "r1_S002",
                "sentence_text": "Screen bad.",
                "start_char": 14,
                "end_char": 25,
            },
        ]

    def test_exact_quotes_get_python_offsets_and_stable_ids(self):
        # Reverse model order to prove IDs follow source positions.
        result = validate_observations(
            {
                "observations": [
                    {
                        "sentence_id": "r1_S002",
                        "evidence_text": "Screen bad",
                        "observation": "The screen is criticized.",
                        "target": "current_product",
                        "polarity": "negative",
                        "statement_type": "actual_experience",
                    },
                    {
                        "sentence_id": "r1_S001",
                        "evidence_text": "Battery good",
                        "observation": "The battery is praised.",
                        "target": "current_product",
                        "polarity": "positive",
                        "statement_type": "actual_experience",
                    },
                ]
            },
            self.review,
            self.sentences,
        )
        self.assertTrue(result["reference_checks_passed"])
        observations = result["accepted_observations"]
        self.assertEqual(["O001", "O002"], [row["observation_id"] for row in observations])
        self.assertEqual("r1_S001", observations[0]["sentence_id"])
        self.assertEqual(0, observations[0]["evidence_start_char_clean"])
        self.assertEqual("Battery good", self.review["clean_text"][0:12])
        self.assertEqual(14, observations[1]["evidence_start_char_clean"])

    def test_unique_case_mismatch_is_canonicalized_from_source(self):
        result = validate_observations(
            {
                "observations": [
                    {
                        "sentence_id": "r1_S001",
                        "evidence_text": "battery",
                        "observation": "The battery is praised.",
                        "target": "current_product",
                        "polarity": "positive",
                        "statement_type": "actual_experience",
                    }
                ]
            },
            self.review,
            self.sentences,
        )
        self.assertTrue(result["reference_checks_passed"])
        observation = result["accepted_observations"][0]
        self.assertEqual("Battery", observation["evidence_text"])
        self.assertEqual(0, observation["evidence_start_char_sentence"])
        self.assertEqual(
            "case_canonicalized_from_source",
            observation["locator_adjustments"][0]["type"],
        )

    def test_ambiguous_repeated_quote_is_rejected(self):
        review = {"review_id": "r2", "clean_text": "screen screen."}
        sentences = [
            {
                "review_id": "r2",
                "sentence_id": "r2_S001",
                "sentence_text": "screen screen.",
                "start_char": 0,
                "end_char": 8,
            }
        ]
        result = validate_observations(
            {
                "observations": [
                    {
                        "sentence_id": "r2_S001",
                        "evidence_text": "screen",
                        "observation": "The product is criticized.",
                        "target": "current_product",
                        "polarity": "negative",
                        "statement_type": "actual_experience",
                    }
                ]
            },
            review,
            sentences,
        )
        self.assertEqual([], result["accepted_observations"])
        self.assertIn("multiple times", result["validation_errors"][0])

    def test_cross_review_sentence_is_rejected(self):
        sentences = [dict(self.sentences[0], review_id="another")]
        result = validate_observations(
            {
                "observations": [
                    {
                        "sentence_id": "r1_S001",
                        "evidence_text": "good",
                        "observation": "The battery is praised.",
                        "target": "current_product",
                        "polarity": "positive",
                        "statement_type": "actual_experience",
                    }
                ]
            },
            self.review,
            sentences,
        )
        self.assertEqual([], result["accepted_observations"])
        self.assertTrue(
            any("another review" in error for error in result["validation_errors"])
        )

    def test_title_is_localized_separately_and_context_is_derived(self):
        review = {
            "review_id": "r3",
            "title": "Power port failed",
            "clean_text": "Stopped charging.",
        }
        sources = [
            {
                "review_id": "r3",
                "sentence_id": "r3_TITLE",
                "sentence_text": review["title"],
                "source_field": "title",
                "start_char": 0,
            },
            {
                "review_id": "r3",
                "sentence_id": "r3_S001",
                "sentence_text": review["clean_text"],
                "source_field": "clean_text",
                "start_char": 0,
            },
        ]
        result = validate_observations(
            {
                "observations": [{
                    "sentence_id": "r3_TITLE",
                    "evidence_text": "Power port failed",
                    "observation": "The power port failed.",
                    "target": "current_product",
                    "polarity": "negative",
                    "statement_type": "actual_experience",
                }]
            },
            review,
            sources,
        )
        observation = result["accepted_observations"][0]
        self.assertEqual("title", observation["source_field"])
        self.assertEqual(0, observation["evidence_start_char_source"])
        self.assertIsNone(observation["evidence_start_char_clean"])
        self.assertEqual("r3_S001", observation["context_after_sentence_id"])
        self.assertEqual("Stopped charging.", observation["context_after"])

    def test_multi_aspect_bundle_is_rejected(self):
        review = {
            "review_id": "r4",
            "clean_text": "Screen, keyboard, USB ports, and speakers are good.",
        }
        result = validate_observations(
            {"observations": [{
                "sentence_id": "r4_S001",
                "evidence_text": review["clean_text"],
                "observation": "The screen, keyboard, USB ports, and speakers are good.",
                "target": "current_product",
                "polarity": "positive",
                "statement_type": "actual_experience",
            }]},
            review,
            [{
                "review_id": "r4",
                "sentence_id": "r4_S001",
                "sentence_text": review["clean_text"],
                "start_char": 0,
            }],
        )
        self.assertEqual([], result["accepted_observations"])
        self.assertIn("multiple product aspects", " ".join(result["validation_errors"]))

    def test_short_generic_quote_is_not_self_sufficient(self):
        text = "Chrome will not install, so forget it."
        result = validate_observations(
            {"observations": [{
                "sentence_id": "r7_S001",
                "evidence_text": "forget it",
                "observation": "Chrome cannot be installed.",
                "target": "current_product",
                "polarity": "negative",
                "statement_type": "actual_experience",
            }]},
            {"review_id": "r7", "clean_text": text},
            [{
                "review_id": "r7",
                "sentence_id": "r7_S001",
                "sentence_text": text,
                "start_char": 0,
            }],
        )
        self.assertEqual([], result["accepted_observations"])
        self.assertIn("not self-sufficient", " ".join(result["validation_errors"]))

        generic_emotion = validate_observations(
            {"observations": [{
                "sentence_id": "r7_S001",
                "evidence_text": "two disappointed kids",
                "observation": "The children dislike the browser restriction.",
                "target": "current_product",
                "polarity": "negative",
                "statement_type": "overall_opinion",
            }]},
            {"review_id": "r7", "clean_text": "two disappointed kids"},
            [{
                "review_id": "r7",
                "sentence_id": "r7_S001",
                "sentence_text": "two disappointed kids",
                "start_char": 0,
            }],
        )
        self.assertEqual([], generic_emotion["accepted_observations"])
        self.assertIn(
            "not self-sufficient",
            " ".join(generic_emotion["validation_errors"]),
        )

        long_generic = "It's still average compared to most other laptops."
        generic_comparison = validate_observations(
            {"observations": [{
                "sentence_id": "r7_S001",
                "evidence_text": long_generic,
                "observation": "Battery performance is average.",
                "target": "current_product",
                "polarity": "negative",
                "statement_type": "capability_or_limitation",
            }]},
            {"review_id": "r7", "clean_text": long_generic},
            [{
                "review_id": "r7",
                "sentence_id": "r7_S001",
                "sentence_text": long_generic,
                "start_char": 0,
            }],
        )
        self.assertEqual([], generic_comparison["accepted_observations"])
        self.assertIn(
            "not self-sufficient",
            " ".join(generic_comparison["validation_errors"]),
        )

        cross_aspect = validate_observations(
            {"observations": [{
                "sentence_id": "r7_S001",
                "evidence_text": "Battery life is great.",
                "observation": "The speakers sound great.",
                "target": "current_product",
                "polarity": "positive",
                "statement_type": "actual_experience",
            }]},
            {"review_id": "r7", "clean_text": "Battery life is great."},
            [{
                "review_id": "r7",
                "sentence_id": "r7_S001",
                "sentence_text": "Battery life is great.",
                "start_char": 0,
            }],
        )
        self.assertEqual([], cross_aspect["accepted_observations"])
        self.assertIn(
            "not self-sufficient",
            " ".join(cross_aspect["validation_errors"]),
        )

    def test_intention_cannot_claim_actual_experience(self):
        text = "My intention was to use it for web browsing."
        result = validate_observations(
            {"observations": [{
                "sentence_id": "r5_S001",
                "evidence_text": text,
                "observation": "The reviewer intended to browse the web.",
                "target": "current_product",
                "polarity": "neutral",
                "statement_type": "actual_experience",
            }]},
            {"review_id": "r5", "clean_text": text},
            [{
                "review_id": "r5",
                "sentence_id": "r5_S001",
                "sentence_text": text,
                "start_char": 0,
            }],
        )
        self.assertEqual([], result["accepted_observations"])
        self.assertIn("cannot be actual_experience", " ".join(result["validation_errors"]))

    def test_statement_types_require_grounded_reverse_cues(self):
        cases = [
            (
                "This computer is lightweight and easy to carry.",
                "intended_use",
                "requires explicit reviewer intention",
            ),
            (
                "Looked around and this was the best price available!",
                "expectation_or_assumption",
                "requires explicit expectation evidence",
            ),
            (
                "The battery needs recharging sooner than expected.",
                "service_experience",
                "requires explicit service, warranty, or seller evidence",
            ),
            (
                "I'm assuming the battery can be upgraded later.",
                "capability_or_limitation",
                "requires expectation_or_assumption",
            ),
        ]
        for number, (text, statement_type, expected_error) in enumerate(cases):
            with self.subTest(statement_type=statement_type):
                review_id = f"statement_{number}"
                result = validate_observations(
                    {"observations": [{
                        "sentence_id": f"{review_id}_S001",
                        "evidence_text": text,
                        "observation": text,
                        "target": "current_product",
                        "polarity": "positive",
                        "statement_type": statement_type,
                    }]},
                    {"review_id": review_id, "clean_text": text},
                    [{
                        "review_id": review_id,
                        "sentence_id": f"{review_id}_S001",
                        "sentence_text": text,
                        "start_char": 0,
                    }],
                )
                self.assertEqual([], result["accepted_observations"])
                self.assertIn(expected_error, " ".join(result["validation_errors"]))

    def test_two_group_product_list_is_rejected_as_non_atomic(self):
        text = (
            "Excellent base laptop with good features, a decent screen, "
            "not a lot of crapware and reliable operation."
        )
        result = validate_observations(
            {"observations": [{
                "sentence_id": "bundle_S001",
                "evidence_text": text,
                "observation": (
                    "The laptop has a decent screen, little crapware, and is reliable."
                ),
                "target": "current_product",
                "polarity": "positive",
                "statement_type": "actual_experience",
            }]},
            {"review_id": "bundle", "clean_text": text},
            [{
                "review_id": "bundle",
                "sentence_id": "bundle_S001",
                "sentence_text": text,
                "start_char": 0,
            }],
        )
        self.assertEqual([], result["accepted_observations"])
        self.assertIn("bundle multiple product aspects", result["validation_errors"][0])

    def test_manufacturer_service_is_not_current_product_target(self):
        text = "Acer warranty repair turnaround takes forever."
        result = validate_observations(
            {"observations": [{
                "sentence_id": "r6_S001",
                "evidence_text": text,
                "observation": "Acer warranty service is slow.",
                "target": "current_product",
                "polarity": "negative",
                "statement_type": "service_experience",
            }]},
            {"review_id": "r6", "clean_text": text},
            [{
                "review_id": "r6",
                "sentence_id": "r6_S001",
                "sentence_text": text,
                "start_char": 0,
            }],
        )
        self.assertEqual([], result["accepted_observations"])
        self.assertIn("seller_platform or unclear", " ".join(result["validation_errors"]))

    def test_ambiguous_pronoun_after_previous_product_is_not_current(self):
        review = {
            "review_id": "r8",
            "clean_text": "My previous HP was awful. It is so slow.",
        }
        sources = [
            {
                "review_id": "r8",
                "sentence_id": "r8_S001",
                "sentence_text": "My previous HP was awful.",
                "start_char": 0,
            },
            {
                "review_id": "r8",
                "sentence_id": "r8_S002",
                "sentence_text": "It is so slow.",
                "start_char": 26,
            },
        ]
        result = validate_observations(
            {"observations": [{
                "sentence_id": "r8_S002",
                "evidence_text": "It is so slow.",
                "observation": "The current laptop is slow.",
                "target": "current_product",
                "polarity": "negative",
                "statement_type": "actual_experience",
            }]},
            review,
            sources,
        )
        self.assertEqual([], result["accepted_observations"])
        self.assertIn("ambiguous pronoun", " ".join(result["validation_errors"]))


class ClaimValidationTests(unittest.TestCase):
    def setUp(self):
        gate = validate_observations(
            {
                "observations": [
                    {
                        "sentence_id": "r1_S001",
                        "evidence_text": "Battery good",
                        "observation": "The reviewer praises battery performance.",
                        "target": "current_product",
                        "polarity": "positive",
                        "statement_type": "actual_experience",
                    },
                    {
                        "sentence_id": "r1_S002",
                        "evidence_text": "Screen bad",
                        "observation": "The reviewer criticizes the screen.",
                        "target": "previous_product",
                        "polarity": "negative",
                        "statement_type": "actual_experience",
                    },
                ]
            },
            {"review_id": "r1", "clean_text": "Battery good. Screen bad."},
            [
                {
                    "review_id": "r1",
                    "sentence_id": "r1_S001",
                    "sentence_text": "Battery good.",
                    "start_char": 0,
                },
                {
                    "review_id": "r1",
                    "sentence_id": "r1_S002",
                    "sentence_text": "Screen bad.",
                    "start_char": 14,
                },
            ],
        )
        self.observations = gate["accepted_observations"]

    def _observation(
        self,
        text,
        observation,
        statement_type="actual_experience",
        polarity="negative",
    ):
        gate = validate_observations(
            {"observations": [{
                "sentence_id": "case_S001",
                "evidence_text": text,
                "observation": observation,
                "target": "current_product",
                "polarity": polarity,
                "statement_type": statement_type,
            }]},
            {"review_id": "case", "clean_text": text},
            [{
                "review_id": "case",
                "sentence_id": "case_S001",
                "sentence_text": text,
                "start_char": 0,
            }],
        )
        self.assertEqual([], gate["validation_errors"])
        return gate["accepted_observations"]

    def _mapped(self, observations, category, issue_group):
        return validate_claims(
            {"claims": [{
                "observation_id": "O001",
                "mapping_status": "mapped",
                "category": category,
                "issue_group": issue_group,
                "claim": observations[0]["observation"],
            }]},
            observations,
            BOUNDARY_TAXONOMY,
        )

    def _complete_unmapped(self, observations, taxonomy=BOUNDARY_TAXONOMY):
        gate = validate_claims(
            {"claims": [{
                "observation_id": "O001",
                "mapping_status": "unmapped",
                "category": None,
                "issue_group": None,
                "claim": observations[0]["observation"],
            }]},
            observations,
            taxonomy,
        )
        return complete_claims_with_unmapped_fallbacks(
            gate,
            observations,
            taxonomy,
        )

    def test_mapped_and_unmapped_inherit_protected_fields(self):
        result = validate_claims(
            {
                "claims": [
                    {
                        "observation_id": "O001",
                        "mapping_status": "mapped",
                        "category": "battery",
                        "issue_group": "battery_runtime",
                        "claim": "The reviewer reports a positive battery experience.",
                    },
                    {
                        "observation_id": "O002",
                        "mapping_status": "unmapped",
                        "category": None,
                        "issue_group": None,
                        "claim": "The reviewer criticizes a previous product's screen.",
                    },
                ]
            },
            self.observations,
            TAXONOMY,
        )
        self.assertTrue(result["normalization_complete"])
        first, second = result["accepted_claims"]
        self.assertEqual("positive", first["sentiment"])
        self.assertEqual("current_product", first["attribution"])
        self.assertEqual("actual_experience", first["statement_type"])
        self.assertEqual("valid_unmapped", second["deterministic_status"])
        self.assertEqual("previous_product", second["attribution"])
        self.assertIsNone(second["category"])
        self.assertEqual("Battery good", first["evidence"][0]["quote"])

    def test_invalid_taxonomy_pair_and_missing_id_are_rejected(self):
        result = validate_claims(
            {
                "claims": [
                    {
                        "observation_id": "O001",
                        "mapping_status": "mapped",
                        "category": "battery",
                        "issue_group": "screen_quality",
                        "claim": "Invalid forced mapping.",
                    },
                    {
                        "observation_id": "O999",
                        "mapping_status": "unmapped",
                        "category": None,
                        "issue_group": None,
                        "claim": "Unknown.",
                    },
                ]
            },
            self.observations,
            TAXONOMY,
        )
        self.assertFalse(result["normalization_complete"])
        self.assertEqual([], result["accepted_claims"])
        self.assertEqual(["O001", "O002"], result["missing_observation_ids"])
        joined = " ".join(result["validation_errors"])
        self.assertIn("screen_quality requires explicit visual-display evidence", joined)
        self.assertIn("unknown observation_id", joined)

    def test_unmapped_cannot_smuggle_taxonomy_values(self):
        result = validate_claims(
            {
                "claims": [
                    {
                        "observation_id": "O001",
                        "mapping_status": "unmapped",
                        "category": "battery",
                        "issue_group": "battery_runtime",
                        "claim": "Claim.",
                    }
                ]
            },
            self.observations[:1],
            TAXONOMY,
        )
        self.assertEqual([], result["accepted_claims"])
        self.assertIn("require null", " ".join(result["validation_errors"]))

    def test_high_risk_performance_boundaries(self):
        gaming = self._observation(
            "Not a gaming machine or storage powerhouse.",
            "The laptop is not intended for demanding games.",
            "capability_or_limitation",
        )
        slow = self._mapped(gaming, "performance", "slow_performance")
        self.assertEqual([], slow["accepted_claims"])
        self.assertIn("explicit slow", " ".join(slow["validation_errors"]))
        capability = self._mapped(gaming, "performance", "processing_capability")
        self.assertTrue(capability["normalization_complete"])

        crashes = self._observation(
            "It crashes every time I play.",
            "The laptop crashes during games.",
        )
        wrong = self._mapped(crashes, "performance", "slow_performance")
        self.assertIn("cannot map to slow_performance", " ".join(wrong["validation_errors"]))
        correct = self._mapped(crashes, "system_stability", "freezing_crashing")
        self.assertTrue(correct["normalization_complete"])

    def test_unique_issue_group_parent_is_safely_canonicalized(self):
        power_port = self._observation(
            "The power port failed.",
            "The laptop power port failed.",
        )
        charging = self._mapped(
            power_port, "connectivity_ports", "charging"
        )
        self.assertTrue(charging["normalization_complete"])
        charging_claim = charging["accepted_claims"][0]
        self.assertEqual("battery", charging_claim["category"])
        self.assertEqual("valid_mapped_canonicalized", charging_claim["deterministic_status"])
        self.assertEqual(
            "connectivity_ports",
            charging_claim["normalization_adjustments"][0]["from_category"],
        )

        routed = self._mapped(
            power_port, "connectivity_ports", "usb_other_ports"
        )
        self.assertTrue(routed["normalization_complete"])
        routed_claim = routed["accepted_claims"][0]
        self.assertEqual("battery", routed_claim["category"])
        self.assertEqual("charging", routed_claim["issue_group"])
        self.assertEqual(
            "canonicalized_power_port_to_charging",
            routed_claim["normalization_adjustments"][0]["type"],
        )

        neutral_port = self._observation(
            "This laptop has a power port.",
            "The laptop has a power port.",
            "feature_description",
            "neutral",
        )
        neutral_route = self._mapped(
            neutral_port, "connectivity_ports", "usb_other_ports"
        )
        self.assertEqual([], neutral_route["accepted_claims"])
        self.assertIn(
            "belongs to battery/charging",
            " ".join(neutral_route["validation_errors"]),
        )

        port_location = self._observation(
            "The power port is in a bad location.",
            "The power port is poorly located.",
            "feature_description",
            "negative",
        )
        location_route = self._mapped(
            port_location, "connectivity_ports", "usb_other_ports"
        )
        self.assertEqual([], location_route["accepted_claims"])
        self.assertIn(
            "belongs to battery/charging",
            " ".join(location_route["validation_errors"]),
        )
        direct_charging = self._mapped(
            port_location, "battery", "charging"
        )
        self.assertEqual([], direct_charging["accepted_claims"])
        self.assertIn(
            "behavior or failure",
            " ".join(direct_charging["validation_errors"]),
        )

        for text in (
            "The laptop has a charging port.",
            "The charging port is on the left side.",
            "The charger is fine but the battery is dead.",
            "The power port works, but the battery is dead.",
        ):
            with self.subTest(charging_rejection=text):
                neutral_or_negative = self._observation(
                    text,
                    text,
                    "feature_description",
                    "negative" if "dead" in text else "neutral",
                )
                result = self._mapped(
                    neutral_or_negative, "battery", "charging"
                )
                self.assertEqual([], result["accepted_claims"])
                self.assertIn(
                    "behavior or failure",
                    " ".join(result["validation_errors"]),
                )

        charging_successes = (
            ("The charging port works.", "positive"),
            ("The charging port failed.", "negative"),
            ("The laptop charges slowly.", "negative"),
        )
        for text, polarity in charging_successes:
            with self.subTest(charging_support=text):
                behavior = self._observation(
                    text,
                    text,
                    "actual_experience",
                    polarity,
                )
                result = self._mapped(behavior, "battery", "charging")
                self.assertTrue(result["normalization_complete"])

        crashes = self._observation(
            "StarCraft II constantly crashes.",
            "StarCraft II constantly crashes on the laptop.",
        )
        stability = self._mapped(crashes, "performance", "freezing_crashing")
        self.assertTrue(stability["normalization_complete"])
        self.assertEqual(
            "system_stability", stability["accepted_claims"][0]["category"]
        )

        battery_upgrade = self._observation(
            "The battery can be upgraded later.",
            "The battery can be upgraded later.",
            "capability_or_limitation",
        )
        unsafe = self._mapped(battery_upgrade, "battery", "upgradeability")
        self.assertEqual([], unsafe["accepted_claims"])
        self.assertIn(
            "requires RAM/storage expansion evidence",
            " ".join(unsafe["validation_errors"]),
        )

        unsupported_repair = self._observation(
            "Battery life is great.",
            "Battery life is great.",
            "actual_experience",
            "positive",
        )
        appearance = self._mapped(
            unsupported_repair, "battery", "appearance"
        )
        self.assertEqual([], appearance["accepted_claims"])
        self.assertIn(
            "appearance requires explicit design or appearance evidence",
            " ".join(appearance["validation_errors"]),
        )

    def test_canonical_claim_and_aspect_gates_stop_shifted_normalization(self):
        battery = self._observation(
            "Battery life lasts eight hours.",
            "The battery lasts eight hours.",
            "actual_experience",
            "positive",
        )
        canonical = validate_claims(
            {"claims": [{
                "observation_id": "O001",
                "mapping_status": "mapped",
                "category": "battery",
                "issue_group": "battery_runtime",
                "claim": "The speakers sound average.",
            }]},
            battery,
            BOUNDARY_TAXONOMY,
        )
        self.assertTrue(canonical["normalization_complete"])
        canonical_claim = canonical["accepted_claims"][0]
        self.assertEqual("The battery lasts eight hours.", canonical_claim["claim"])
        self.assertEqual("The speakers sound average.", canonical_claim["model_claim"])
        self.assertIn(
            "canonicalized_claim_from_validated_observation",
            [
                item["type"]
                for item in canonical_claim["normalization_adjustments"]
            ],
        )

        sound = self._observation(
            "Sound is decent, but not stellar.",
            "The sound is decent but not excellent.",
            "feature_description",
            "mixed",
        )
        wrong_touchpad = self._mapped(
            sound, "input_devices", "touchpad_mouse"
        )
        self.assertIn(
            "pointing-device evidence",
            " ".join(wrong_touchpad["validation_errors"]),
        )

        battery_average = self._observation(
            "Battery performance is average.",
            "Battery performance is average.",
            "capability_or_limitation",
            "mixed",
        )
        wrong_speaker = self._mapped(
            battery_average, "audio", "speaker_sound"
        )
        self.assertIn(
            "speaker or sound evidence",
            " ".join(wrong_speaker["validation_errors"]),
        )

        keyboard = self._observation(
            "I really like the backlit keyboard.",
            "The reviewer likes the backlit keyboard.",
            "feature_description",
            "positive",
        )
        correct_keyboard = self._mapped(
            keyboard, "input_devices", "keyboard"
        )
        self.assertTrue(correct_keyboard["normalization_complete"])

        generic_look = self._observation(
            "Take a look at this key feature.",
            "Take a look at this key feature.",
            "feature_description",
            "neutral",
        )
        appearance = self._mapped(
            generic_look, "design_portability", "appearance"
        )
        self.assertIn(
            "explicit design or appearance evidence",
            " ".join(appearance["validation_errors"]),
        )
        generic_key = self._mapped(
            generic_look, "input_devices", "keyboard"
        )
        self.assertIn(
            "keyboard or key evidence",
            " ".join(generic_key["validation_errors"]),
        )

        secondhand = self._observation(
            "A friend said warranty turnaround takes forever.",
            "A friend reports slow warranty turnaround.",
            "overall_opinion",
            "negative",
        )
        service = self._mapped(
            secondhand, "purchase_service", "support_service"
        )
        self.assertIn(
            "secondhand service anecdotes",
            " ".join(service["validation_errors"]),
        )

    def test_processing_and_storage_boundaries_cover_pilot_phrasing(self):
        task_speed = self._observation(
            "It navigated quickly between web pages and Word opens instantly.",
            "Web pages load quickly and Word opens instantly.",
            "capability_or_limitation",
            "positive",
        )
        processing = self._mapped(
            task_speed, "performance", "processing_capability"
        )
        self.assertTrue(processing["normalization_complete"])

        multitasking = self._observation(
            "The system handles a few tasks without slowing down.",
            "The system handles multiple tasks without slowing down.",
            "capability_or_limitation",
            "positive",
        )
        processing = self._mapped(
            multitasking, "performance", "processing_capability"
        )
        self.assertTrue(processing["normalization_complete"])

        expansion = self._observation(
            "An SSD can be added to the open storage slot.",
            "An SSD can be added to the open storage slot.",
            "capability_or_limitation",
            "positive",
        )
        wrong_capacity = self._mapped(
            expansion, "memory_storage", "storage_capacity"
        )
        self.assertIn(
            "belongs to upgradeability",
            " ".join(wrong_capacity["validation_errors"]),
        )
        upgrade = self._mapped(expansion, "memory_storage", "upgradeability")
        self.assertTrue(upgrade["normalization_complete"])

    def test_invalid_model_decision_gets_stable_unmapped_fallback(self):
        observations = self._observation(
            "The hinge failed.",
            "The laptop hinge failed.",
        )
        invalid = validate_claims(
            {"claims": [{
                "observation_id": "O001",
                "mapping_status": "mapped",
                "category": "battery",
                "issue_group": "not_a_group",
                "claim": "Invalid mapping.",
            }]},
            observations,
            BOUNDARY_TAXONOMY,
        )
        completed = complete_claims_with_unmapped_fallbacks(
            invalid, observations, BOUNDARY_TAXONOMY
        )
        self.assertTrue(completed["normalization_complete"])
        self.assertFalse(completed["model_normalization_complete"])
        self.assertEqual(1, completed["fallback_unmapped_count"])
        fallback = completed["accepted_claims"][0]
        self.assertEqual("C001", fallback["claim_id"])
        self.assertEqual("unmapped", fallback["mapping_status"])
        self.assertEqual(
            "deterministic_unmapped_fallback",
            fallback["normalization_source"],
        )
        self.assertTrue(completed["rejected_claims"])
        self.assertTrue(completed["model_validation_errors"])

    def test_high_precision_recovery_maps_only_explicit_evidence(self):
        cases = [
            (
                "Best deal on a basic laptop available!",
                "This is the best deal on a basic laptop available.",
                "overall_opinion",
                "positive",
                ("price_value", "value_for_money"),
                "explicit_positive_value_judgment",
            ),
            (
                "Looked around and this was the best price available!",
                "This was the best price available.",
                "overall_opinion",
                "positive",
                ("price_value", "price"),
                "explicit_positive_price_comparison",
            ),
            (
                "No good if power port goes bad and it did!",
                "The power port failed.",
                "capability_or_limitation",
                "negative",
                ("battery", "charging"),
                "explicit_power_hardware_failure",
            ),
            (
                "To me it's definitely a waste of money.",
                "The purchase is a waste of money.",
                "overall_opinion",
                "negative",
                ("price_value", "value_for_money"),
                "explicit_negative_value_judgment",
            ),
            (
                "This laptop is not worth the money.",
                "The laptop is not worth the money.",
                "overall_opinion",
                "negative",
                ("price_value", "value_for_money"),
                "explicit_negative_value_judgment",
            ),
            (
                "The software blocks Chrome from being downloaded.",
                "The software prevents downloading Chrome.",
                "capability_or_limitation",
                "negative",
                ("software_os", "app_compatibility"),
                "explicit_app_block_with_action",
            ),
        ]
        for text, observation, statement_type, polarity, pair, rule_id in cases:
            with self.subTest(text=text):
                observations = self._observation(
                    text,
                    observation,
                    statement_type,
                    polarity,
                )
                completed = self._complete_unmapped(observations)
                claim = completed["accepted_claims"][0]
                self.assertEqual("mapped", claim["mapping_status"])
                self.assertEqual(pair, (claim["category"], claim["issue_group"]))
                self.assertEqual(
                    "deterministic_high_precision_recovery",
                    claim["normalization_source"],
                )
                self.assertEqual(
                    "valid_mapped_by_high_precision_recovery",
                    claim["deterministic_status"],
                )
                self.assertIn(
                    rule_id,
                    claim["normalization_adjustments"][-1]["rule_ids"],
                )
                self.assertEqual(1, completed["deterministic_recovery_count"])
                self.assertEqual(0, completed["fallback_unmapped_count"])

    def test_high_precision_recovery_keeps_unsafe_cases_unmapped(self):
        cases = [
            (
                "Forget about Google Chrome",
                "The reviewer advises against Chrome.",
                "overall_opinion",
                "negative",
            ),
            (
                "If you want to use Google Chrome, forget it.",
                "The reviewer is told to forget using Chrome.",
                "capability_or_limitation",
                "negative",
            ),
            (
                "I wanted an inexpensive laptop for school.",
                "The reviewer wanted an inexpensive laptop.",
                "intended_use",
                "positive",
            ),
            (
                "The power port is in a bad location.",
                "The power port is poorly located.",
                "feature_description",
                "negative",
            ),
            (
                "This is the best deal at the best price.",
                "The reviewer praises both the deal and price.",
                "overall_opinion",
                "positive",
            ),
            (
                "Best deal available.",
                "The reviewer calls it the best deal.",
                "overall_opinion",
                "negative",
            ),
            (
                "Gaming heavily is not this model's strength.",
                "The laptop is not strong at heavy gaming.",
                "capability_or_limitation",
                "negative",
            ),
            (
                "You need to recharge sooner than expected.",
                "The battery needs recharging sooner than expected.",
                "capability_or_limitation",
                "negative",
            ),
            (
                "Chrome opens fine, but the keyboard does not work.",
                "The keyboard does not work.",
                "actual_experience",
                "negative",
            ),
            (
                "Chrome opens fine and the touchpad does not work.",
                "The touchpad does not work.",
                "actual_experience",
                "negative",
            ),
            (
                "The power port is convenient, but the screen failed.",
                "The screen failed.",
                "actual_experience",
                "negative",
            ),
            (
                "The power port sits next to a dead screen pixel.",
                "The screen has a dead pixel.",
                "feature_description",
                "negative",
            ),
            (
                "The power adapter came from a broker.",
                "The power adapter came from a broker.",
                "feature_description",
                "negative",
            ),
            (
                "The charger broke the screen.",
                "The charger damaged the screen.",
                "actual_experience",
                "negative",
            ),
            (
                "The browser blocks popups from opening.",
                "The browser blocks popups.",
                "feature_description",
                "negative",
            ),
            (
                "I cannot work without Chrome.",
                "The reviewer cannot work without Chrome.",
                "actual_experience",
                "negative",
            ),
            (
                "The team cannot work unless Chrome is available.",
                "The team needs Chrome for its work.",
                "actual_experience",
                "negative",
            ),
            (
                "Chrome does not make work easier.",
                "Chrome does not make the work easier.",
                "overall_opinion",
                "negative",
            ),
            (
                "An ad blocker prevents the browser from opening popups.",
                "The ad blocker prevents popups.",
                "feature_description",
                "negative",
            ),
            (
                "Chrome works, but the antivirus blocks a PDF from being downloaded.",
                "The antivirus blocks a PDF download.",
                "actual_experience",
                "negative",
            ),
            (
                "The charger is fine but the battery is dead.",
                "The battery is dead while the charger is fine.",
                "actual_experience",
                "negative",
            ),
            (
                "The power port works, but the battery is dead.",
                "The battery is dead while the power port works.",
                "actual_experience",
                "negative",
            ),
            (
                "This is not a broken power port.",
                "The power port is not broken.",
                "feature_description",
                "negative",
            ),
            (
                "A charger failed my safety test.",
                "The charger failed a safety test.",
                "actual_experience",
                "negative",
            ),
            (
                "This is not the best deal.",
                "The reviewer does not call it the best deal.",
                "overall_opinion",
                "positive",
            ),
            (
                "This is not a good value.",
                "The reviewer does not call it a good value.",
                "overall_opinion",
                "positive",
            ),
            (
                "This is not the best price.",
                "The reviewer does not call it the best price.",
                "overall_opinion",
                "positive",
            ),
            (
                "This is not a bad deal.",
                "The reviewer does not call it a bad deal.",
                "overall_opinion",
                "negative",
            ),
            (
                "This is not worth the money.",
                "The reviewer says it is not worth the money.",
                "overall_opinion",
                "positive",
            ),
        ]
        for text, observation, statement_type, polarity in cases:
            with self.subTest(text=text):
                observations = self._observation(
                    text,
                    observation,
                    statement_type,
                    polarity,
                )
                completed = self._complete_unmapped(observations)
                claim = completed["accepted_claims"][0]
                self.assertEqual("unmapped", claim["mapping_status"])
                self.assertEqual("model", claim["normalization_source"])
                self.assertEqual(0, completed["deterministic_recovery_count"])

    def test_recovered_fallback_preserves_rejected_model_audit(self):
        observations = self._observation(
            "No good if power port goes bad and it did!",
            "The laptop power port failed.",
            "capability_or_limitation",
            "negative",
        )
        invalid = validate_claims(
            {"claims": [{
                "observation_id": "O001",
                "mapping_status": "mapped",
                "category": "connectivity_ports",
                "issue_group": "power_port",
                "claim": "The laptop power port failed.",
            }]},
            observations,
            BOUNDARY_TAXONOMY,
        )
        completed = complete_claims_with_unmapped_fallbacks(
            invalid,
            observations,
            BOUNDARY_TAXONOMY,
        )
        claim = completed["accepted_claims"][0]
        self.assertEqual("mapped", claim["mapping_status"])
        self.assertEqual(("battery", "charging"), (
            claim["category"], claim["issue_group"]
        ))
        self.assertEqual(1, completed["deterministic_recovery_count"])
        self.assertEqual(0, completed["fallback_unmapped_count"])
        self.assertFalse(completed["model_normalization_complete"])
        self.assertTrue(completed["rejected_claims"])
        self.assertTrue(completed["model_validation_errors"])
        adjustment = claim["normalization_adjustments"][-1]
        self.assertEqual(
            "deterministic_unmapped_fallback",
            adjustment["from_normalization_source"],
        )

    def test_intentions_and_unsupported_taxonomy_pairs_must_abstain(self):
        intended = self._observation(
            "Bought for using Microsoft Office.",
            "The reviewer intends to use Microsoft Office.",
            "intended_use",
            "neutral",
        )
        app = self._mapped(intended, "software_os", "app_compatibility")
        self.assertIn("must be unmapped", " ".join(app["validation_errors"]))

        generic_chrome = self._observation(
            "Forget about Google Chrome.",
            "The reviewer advises against Chrome.",
            "overall_opinion",
            "negative",
        )
        generic_app = self._mapped(
            generic_chrome, "software_os", "app_compatibility"
        )
        self.assertIn(
            "blocked, unavailable, or incompatible",
            " ".join(generic_app["validation_errors"]),
        )

        popup_blocker = self._observation(
            "The browser blocks popups from opening.",
            "The browser blocks popups.",
            "feature_description",
            "negative",
        )
        popup_app = self._mapped(
            popup_blocker, "software_os", "app_compatibility"
        )
        self.assertIn(
            "blocked, unavailable, or incompatible",
            " ".join(popup_app["validation_errors"]),
        )

        blocked_chrome = self._observation(
            "The software blocks Chrome from being downloaded.",
            "The software blocks downloading Chrome.",
            "capability_or_limitation",
            "negative",
        )
        supported_app = self._mapped(
            blocked_chrome, "software_os", "app_compatibility"
        )
        self.assertTrue(supported_app["normalization_complete"])

        assumption = self._observation(
            "I'm assuming the battery can be upgraded later.",
            "The reviewer assumes the battery can be replaced.",
            "expectation_or_assumption",
            "positive",
        )
        charging = self._mapped(assumption, "battery", "charging")
        self.assertIn("must be unmapped", " ".join(charging["validation_errors"]))

        suitability = self._observation(
            "Good purchase if you do not expect a gaming machine.",
            "The laptop suits users without gaming needs.",
            "capability_or_limitation",
            "positive",
        )
        weight = self._mapped(suitability, "design_portability", "weight")
        self.assertIn("weight requires", " ".join(weight["validation_errors"]))

        slot = self._observation(
            "An SSD can be added in the open storage slot.",
            "The SSD storage can be expanded through an open slot.",
            "feature_description",
            "positive",
        )
        storage_speed = self._mapped(
            slot, "memory_storage", "storage_performance"
        )
        self.assertIn("storage plus speed", " ".join(storage_speed["validation_errors"]))
        upgrade = self._mapped(slot, "memory_storage", "upgradeability")
        self.assertTrue(upgrade["normalization_complete"])

        recommendation = self._observation(
            "Highly recommended!",
            "The reviewer highly recommends the laptop.",
            "overall_opinion",
            "positive",
        )
        value = self._mapped(recommendation, "price_value", "value_for_money")
        self.assertIn("price or value", " ".join(value["validation_errors"]))

        explicit_value = self._observation(
            "This laptop is inexpensive for what it offers.",
            "The laptop is inexpensive for its features.",
            "overall_opinion",
            "positive",
        )
        value = self._mapped(
            explicit_value, "price_value", "value_for_money"
        )
        self.assertTrue(value["normalization_complete"])

    def test_power_port_title_can_support_charging_claim(self):
        title = "No good if power port goes bad and it did!!!"
        observations = validate_observations(
            {"observations": [{
                "sentence_id": "case_TITLE",
                "evidence_text": "power port goes bad and it did",
                "observation": "The laptop's power port failed.",
                "target": "current_product",
                "polarity": "negative",
                "statement_type": "actual_experience",
            }]},
            {"review_id": "case", "title": title, "clean_text": "In six months."},
            [{
                "review_id": "case",
                "sentence_id": "case_TITLE",
                "sentence_text": title,
                "source_field": "title",
                "start_char": 0,
            }],
        )["accepted_observations"]
        result = self._mapped(observations, "battery", "charging")
        self.assertTrue(result["normalization_complete"])
        evidence = result["accepted_claims"][0]["evidence"][0]
        self.assertEqual("title", evidence["source_field"])
        self.assertIsNone(evidence["quote_start_char_clean"])

    def test_merge_claims_keeps_atomic_trace_and_unique_evidence(self):
        claims = [
            {
                "claim_id": "C001",
                "observation_id": "O001",
                "mapping_status": "mapped",
                "category": "battery",
                "issue_group": "battery_runtime",
                "claim": "Battery life is short.",
                "sentiment": "negative",
                "attribution": "current_product",
                "statement_type": "actual_experience",
                "evidence": [{"sentence_id": "S1", "quote": "four hours"}],
            },
            {
                "claim_id": "C002",
                "observation_id": "O002",
                "mapping_status": "mapped",
                "category": "battery",
                "issue_group": "battery_runtime",
                "claim": "Battery is below the advertised runtime.",
                "sentiment": "negative",
                "attribution": "current_product",
                "statement_type": "actual_experience",
                "evidence": [{"sentence_id": "S2", "quote": "less than advertised"}],
            },
            {
                "claim_id": "C003",
                "observation_id": "O003",
                "mapping_status": "unmapped",
                "category": None,
                "issue_group": None,
                "claim": "Overall praise.",
                "sentiment": "positive",
                "attribution": "current_product",
                "statement_type": "overall_opinion",
                "evidence": [{"sentence_id": "S3", "quote": "great"}],
            },
        ]
        findings = merge_claims_by_tuple(claims)
        self.assertEqual(1, len(findings))
        self.assertEqual(["C001", "C002"], findings[0]["source_claim_ids"])
        self.assertEqual(["O001", "O002"], findings[0]["observation_ids"])
        self.assertEqual(2, len(findings[0]["component_claims"]))
        self.assertEqual(2, len(findings[0]["evidence"]))


class CacheAndSelectionTests(unittest.TestCase):
    def test_fixed_pilot_order_and_safe_limit(self):
        available = set(PILOT_REVIEW_IDS)
        self.assertEqual(list(PILOT_REVIEW_IDS[:3]), select_pilot_ids(available, 3))
        self.assertEqual(list(PILOT_REVIEW_IDS), select_pilot_ids(available, 0))
        with self.assertRaises(ValueError):
            select_pilot_ids(available, 11)
        with self.assertRaises(ValueError):
            select_pilot_ids(set(PILOT_REVIEW_IDS[:-1]), 10)

    def test_hashes_are_order_independent_for_objects(self):
        self.assertEqual(hash_json({"a": 1, "b": 2}), hash_json({"b": 2, "a": 1}))
        self.assertEqual(taxonomy_hash(TAXONOMY), taxonomy_hash(dict(TAXONOMY)))

    def test_cache_fingerprint_accepts_exact_row_and_rejects_stale(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.jsonl"
            row = {
                "review_id": "r1",
                "schema_version": OBSERVATION_SCHEMA_VERSION,
                "model_id": "model",
                "prompt_hash": "prompt",
                "taxonomy_hash": "taxonomy",
                "input_hash": "input",
            }
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            loaded = load_stage_cache(
                path,
                schema_version=OBSERVATION_SCHEMA_VERSION,
                model_id="model",
                prompt_hash="prompt",
                taxonomy_digest="taxonomy",
                input_hashes={"r1": "input"},
            )
            self.assertEqual(row, loaded["r1"])
            with self.assertRaisesRegex(ValueError, "stale Evidence-first cache"):
                load_stage_cache(
                    path,
                    schema_version=OBSERVATION_SCHEMA_VERSION,
                    model_id="different",
                    prompt_hash="prompt",
                    taxonomy_digest="taxonomy",
                    input_hashes={"r1": "input"},
                )

    def test_qwen_wrappers_are_removed_before_json_parse(self):
        parsed = parse_json_object(
            '<think>hidden</think>\n\n```json\n{"observations": []}\n```'
        )
        self.assertEqual({"observations": []}, parsed)


class PipelineIntegrationTests(unittest.TestCase):
    def _make_project(self, root):
        reviews_path = root / "data/processed/amazon/acer_aspire5_reviews_clean.jsonl"
        sentences_path = root / "data/processed/amazon/acer_aspire5_sentences.jsonl"
        layer1_path = root / "outputs/amazon/acer_aspire5_full_predictions.jsonl"
        reviews_path.parent.mkdir(parents=True)
        layer1_path.parent.mkdir(parents=True)
        reviews, sentences, layer1 = [], [], []
        for review_id in PILOT_REVIEW_IDS:
            text = f"{review_id} battery is good."
            sentence_id = f"{review_id}_S001"
            reviews.append(
                {
                    "review_id": review_id,
                    "clean_text": text,
                    "rating": 5.0,
                    "product_name": "test",
                    "product_model": "test",
                }
            )
            sentences.append(
                {
                    "review_id": review_id,
                    "sentence_id": sentence_id,
                    "sentence_text": text,
                    "start_char": 0,
                    "end_char": len(text),
                }
            )
            layer1.append(
                {
                    "review_id": review_id,
                    "sentence_id": sentence_id,
                    "sentence_text": text,
                    "start_char": 0,
                    "parse_success": True,
                    "aspects": [{"term": "battery", "sentiment": "positive"}],
                }
            )
        for path, rows in (
            (reviews_path, reviews),
            (sentences_path, sentences),
            (layer1_path, layer1),
        ):
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

    def test_fake_model_run_resumes_and_leaves_old_artifacts_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._make_project(root)
            old_path = root / "outputs/amazon/evidence/evidence_pack.json"
            old_path.parent.mkdir(parents=True)
            old_path.write_text('{"old": true}', encoding="utf-8")
            calls = []

            def fake_generator(system_prompt, payload, repair):
                calls.append(payload["review_id"])
                if "sentences_and_layer1_clues" in payload:
                    sentence = payload["sentences_and_layer1_clues"][0]
                    return json.dumps(
                        {
                            "observations": [
                                {
                                    "sentence_id": sentence["sentence_id"],
                                    "evidence_text": "battery is good",
                                    "observation": "The battery is praised.",
                                    "target": "current_product",
                                    "polarity": "positive",
                                    "statement_type": "actual_experience",
                                }
                            ]
                        }
                    )
                return json.dumps(
                    {
                        "claims": [
                            {
                                "observation_id": "O001",
                                "mapping_status": "mapped",
                                "category": "battery",
                                "issue_group": "battery_runtime",
                                "claim": "The reviewer reports a positive battery experience.",
                            }
                        ]
                    }
                )

            first = run_pilot(
                root,
                TAXONOMY,
                "fake-model",
                limit=10,
                max_retries=0,
                show=0,
                generator=fake_generator,
            )
            self.assertEqual(20, len(calls))
            self.assertEqual(10, len(first["claim_rows"]))
            self.assertEqual('{"old": true}', old_path.read_text(encoding="utf-8"))
            output_dir = root / "outputs/amazon/evidence_first"
            self.assertEqual(
                10,
                len((output_dir / "observations_pilot.jsonl").read_text().splitlines()),
            )
            self.assertEqual(
                10,
                len((output_dir / "claims_pilot.jsonl").read_text().splitlines()),
            )

            def should_not_run(*args):
                raise AssertionError("resumed pilot must not call the model")

            second = run_pilot(
                root,
                TAXONOMY,
                "fake-model",
                limit=10,
                max_retries=0,
                show=0,
                generator=should_not_run,
            )
            self.assertEqual(10, len(second["observation_rows"]))


if __name__ == "__main__":
    unittest.main()
