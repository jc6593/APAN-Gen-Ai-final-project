"""Isolated deterministic policy for the Evidence-first v1.5 recall repair.

The frozen v1.4 core is intentionally imported, never edited.  V1.5 keeps its
exact-evidence contract, then adds narrowly scoped parser recovery, adjacent
context grounding, and auditable taxonomy-pair repairs.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

from . import core as v14


OBSERVATION_SCHEMA_VERSION_V15 = "evidence_first_observation_v1_5"
CLAIM_SCHEMA_VERSION_V15 = "evidence_first_claim_v1_5"
POLICY_VERSION_V15 = "evidence_first_recall_repair_v1_5"


def _strip_model_wrappers(raw: str) -> str:
    if not isinstance(raw, str):
        raise ValueError("model response must be text")
    text = re.sub(r"^\s*<think>.*?</think>\s*", "", raw, flags=re.DOTALL).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```\s*$", "", text)
    return text


def parse_observation_response_v15(
    raw: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Parse an observation object, safely wrapping a bare top-level list.

    The wrapped items are still untrusted and must pass the complete v1.4
    evidence locator and semantic gate.  Claim parsing remains object-only.
    """
    parsed = json.loads(_strip_model_wrappers(raw))
    if isinstance(parsed, dict):
        return parsed, []
    if isinstance(parsed, list):
        return {"observations": parsed}, [{
            "type": "wrapped_top_level_observation_list",
            "item_count": len(parsed),
        }]
    raise ValueError("expected one JSON object or an observation list")


def parse_observation_attempt_v15(
    raw: str,
    review: Mapping[str, Any],
    sentences: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any], bool]:
    """Parse and fully validate one v1.5 observation response."""
    try:
        parsed, adjustments = parse_observation_response_v15(raw)
        gate = validate_observations_v15(parsed, review, sentences)
        gate["parse_adjustments"] = adjustments
        gate["strict_object_shape"] = bool(
            not adjustments
            and set(parsed) == {"observations"}
            and isinstance(parsed.get("observations"), list)
        )
        return parsed, gate, True
    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        return None, {
            "accepted_observations": [],
            "rejected_observations": [],
            "validation_errors": [f"parse_error: {exc}"],
            "reference_checks_passed": False,
            "parse_adjustments": [],
            "strict_object_shape": False,
        }, False


def _aspect_groups(text: str) -> set[str]:
    return {
        name
        for name, pattern in v14.ASPECT_CUE_GROUPS.items()
        if pattern.search(text)
    }


_CONTEXT_ENTITY_CUES = {
    "keyboard": re.compile(r"\b(?:keyboard|keys?|backlit|backlight)\b", re.I),
    "touchpad": re.compile(
        r"\b(?:touchpad|trackpad|mouse|cursor|pointer)\w*\b",
        re.I,
    ),
    "display": re.compile(r"\b(?:screen|display|monitor)\b", re.I),
    "speaker": re.compile(r"\b(?:speaker|sound|audio|volume)\w*\b", re.I),
}


def _context_entities(text: str) -> set[str]:
    return {
        name
        for name, pattern in _CONTEXT_ENTITY_CUES.items()
        if pattern.search(text)
    }


def _validated_context_metadata(
    sentence: Mapping[str, Any],
    review: Mapping[str, Any],
) -> dict[str, Any] | None:
    quote = _text(sentence.get("sentence_text"))
    source_field = sentence.get("source_field", "clean_text")
    source_start = sentence.get("start_char")
    source_text = review.get(source_field)
    if (
        not quote
        or sentence.get("review_id") != review.get("review_id")
        or source_field not in v14.ALLOWED_SOURCE_FIELDS
        or not isinstance(source_start, int)
        or source_start < 0
        or not isinstance(source_text, str)
    ):
        return None
    source_end = source_start + len(quote)
    if source_end > len(source_text) or source_text[source_start:source_end] != quote:
        return None
    return {
        "source_field": source_field,
        "quote": quote,
        "quote_start_char_sentence": 0,
        "quote_end_char_sentence": len(quote),
        "quote_start_char_source": source_start,
        "quote_end_char_source": source_end,
        "quote_start_char_clean": source_start if source_field == "clean_text" else None,
        "quote_end_char_clean": source_end if source_field == "clean_text" else None,
    }


def _context_evidence_row(
    sentence: Mapping[str, Any],
    review: Mapping[str, Any],
    matched_aspect: str,
    direction: str,
) -> dict[str, Any] | None:
    metadata = _validated_context_metadata(sentence, review)
    if metadata is None:
        return None
    return {
        "type": "adjacent_aspect_antecedent",
        "direction": direction,
        "sentence_id": sentence.get("sentence_id"),
        **metadata,
        "matched_aspect_groups": [matched_aspect],
    }


def _attach_context_provenance(
    row: dict[str, Any],
    sentence_lookup: Mapping[str, Mapping[str, Any]],
    review: Mapping[str, Any],
) -> None:
    for direction in ("before", "after"):
        sentence_id = row.get(f"context_{direction}_sentence_id")
        quote = row.get(f"context_{direction}")
        sentence = sentence_lookup.get(sentence_id)
        metadata = (
            _validated_context_metadata(sentence, review)
            if isinstance(sentence, Mapping)
            else None
        )
        valid = bool(
            metadata is not None
            and isinstance(quote, str)
            and metadata["quote"] == quote
        )
        row[f"context_{direction}_provenance_validated"] = valid
        if valid and metadata is not None:
            row[f"context_{direction}_source_field"] = metadata["source_field"]
            row[f"context_{direction}_start_char_source"] = metadata[
                "quote_start_char_source"
            ]
            row[f"context_{direction}_end_char_source"] = metadata[
                "quote_end_char_source"
            ]
            row[f"context_{direction}_start_char_clean"] = metadata[
                "quote_start_char_clean"
            ]
            row[f"context_{direction}_end_char_clean"] = metadata[
                "quote_end_char_clean"
            ]


def validate_observations_v15(
    data: Mapping[str, Any],
    review: Mapping[str, Any],
    sentences: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Run the frozen gate, then recover one narrow adjacent antecedent case.

    Recovery is allowed only when the base gate's sole item-level objection is
    aspect self-sufficiency, the exact quote contains a concrete predicate plus
    a pronoun/continuation, and exactly one adjacent source record supplies the
    single missing aspect. Generic fragments remain rejected.
    """
    gate = v14.validate_observations(data, review, sentences)
    sentence_lookup = {
        row.get("sentence_id"): row
        for row in sentences
        if isinstance(row.get("sentence_id"), str)
    }
    sentence_order = {
        row.get("sentence_id"): index
        for index, row in enumerate(sentences)
        if isinstance(row.get("sentence_id"), str)
    }
    recovered: list[dict[str, Any]] = []
    recovered_indices: set[int] = set()
    expected_error = "evidence quote is not self-sufficient about the observation aspect"
    generic_fragment = re.compile(
        r"^\s*(?:(?:it|this|that)\s+(?:is|was|'s|’s)\s+)?"
        r"(?:very\s+)?(?:average|good|great|bad|fine|okay|ok|terrible|"
        r"awful|perfect|nice|forget it)\W*$",
        re.I,
    )
    bridge_predicate = re.compile(
        r"\b(?:dim|bright|touchy|responsive|work\w*|broken|missing|"
        r"low|loud|quiet|heavy|light|fast|slow|crack\w*|damag\w*|"
        r"disconnect\w*|fail\w*|barely|hard\s+to|no\s+option|"
        r"can(?:not|'t)|won't|does(?: not|n't)|isn't|wasn't)\b",
        re.I,
    )
    bridge_reference = re.compile(
        r"^\s*(?:also|and|but|however)?\s*"
        r"(?:it|this|that|they|them|these|those|its|their)\b|"
        r"\b(?:it|they|them|its|their)\b",
        re.I,
    )

    for rejected in gate.get("rejected_observations", []):
        index = rejected.get("index")
        item = rejected.get("item")
        reasons = rejected.get("errors")
        if (
            not isinstance(index, int)
            or not isinstance(item, dict)
            or reasons != [expected_error]
        ):
            continue
        evidence = _text(item.get("evidence_text"))
        observation_text = _text(item.get("observation"))
        if (
            not evidence
            or not observation_text
            or generic_fragment.fullmatch(evidence)
            or not bridge_reference.search(evidence)
            or not bridge_predicate.search(evidence)
        ):
            continue
        missing = _aspect_groups(observation_text) - _aspect_groups(evidence)
        if len(missing) != 1:
            continue
        missing_aspect = next(iter(missing))
        observation_entities = _context_entities(observation_text)
        if len(observation_entities) != 1:
            continue
        missing_entity = next(iter(observation_entities))
        sentence_id = item.get("sentence_id")
        position = sentence_order.get(sentence_id)
        if position is None:
            continue
        providers: list[tuple[str, Mapping[str, Any]]] = []
        # Pronouns and continuation fragments resolve backward. V1.5's
        # pair-specific normalization rules may use a following qualifier, but
        # the general observation gate never guesses a cataphoric referent.
        adjacent_index = position - 1
        if 0 <= adjacent_index < len(sentences):
            adjacent = sentences[adjacent_index]
            adjacent_text = _text(adjacent.get("sentence_text"))
            if (
                missing_aspect in _aspect_groups(adjacent_text)
                and _context_entities(adjacent_text) == {missing_entity}
            ):
                providers.append(("before", adjacent))
        if len(providers) != 1:
            continue

        # Re-run every frozen locator/type/target check with only the already
        # audited self-sufficiency objection neutralized. The original readable
        # observation is restored after this surrogate passes.
        surrogate_item = dict(item)
        surrogate_item["observation"] = evidence
        surrogate_gate = v14.validate_observations(
            {"observations": [surrogate_item]},
            review,
            sentences,
        )
        if (
            surrogate_gate.get("validation_errors")
            or len(surrogate_gate.get("accepted_observations", [])) != 1
        ):
            continue
        row = dict(surrogate_gate["accepted_observations"][0])
        row["observation"] = observation_text
        direction, adjacent = providers[0]
        context_adjustment = _context_evidence_row(
            adjacent,
            review,
            f"{missing_aspect}:{missing_entity}",
            direction,
        )
        if context_adjustment is None:
            continue
        row["context_recovery_adjustments"] = [context_adjustment]
        recovered.append(row)
        recovered_indices.add(index)

    if not recovered:
        for row in gate.get("accepted_observations", []):
            row.setdefault("context_recovery_adjustments", [])
            _attach_context_provenance(row, sentence_lookup, review)
        gate["context_recovery_count"] = 0
        return gate

    accepted = [dict(row) for row in gate.get("accepted_observations", [])]
    for row in accepted:
        row.setdefault("context_recovery_adjustments", [])
    accepted.extend(recovered)
    for row in accepted:
        _attach_context_provenance(row, sentence_lookup, review)
    accepted.sort(key=lambda row: (
        sentence_order.get(row.get("sentence_id"), len(sentence_order)),
        row.get("evidence_start_char_sentence", 0),
        row.get("evidence_end_char_sentence", 0),
        row.get("observation", ""),
        row.get("target", ""),
        row.get("polarity", ""),
    ))
    for number, row in enumerate(accepted, start=1):
        row["observation_id"] = f"O{number:03d}"

    rejected = [
        entry for entry in gate.get("rejected_observations", [])
        if entry.get("index") not in recovered_indices
    ]
    errors = [
        error for error in gate.get("validation_errors", [])
        if not any(
            error.startswith(f"observation[{index}]: ")
            for index in recovered_indices
        )
    ]
    return {
        "accepted_observations": accepted,
        "rejected_observations": rejected,
        "validation_errors": errors,
        "reference_checks_passed": not errors,
        "context_recovery_count": len(recovered),
    }


def normalization_payload_v15(
    observation_row: Mapping[str, Any],
) -> dict[str, Any]:
    """Keep context away from the model; Python alone performs audited recovery."""
    return {
        "review_id": observation_row["review_id"],
        "validated_observations": [
            {
                "observation_id": item["observation_id"],
                "sentence_id": item["sentence_id"],
                "evidence_text": item["evidence_text"],
                "observation": item["observation"],
                "target": item["target"],
                "polarity": item["polarity"],
                "statement_type": item["statement_type"],
                "source_field": item["source_field"],
            }
            for item in observation_row.get("accepted_observations", [])
        ],
    }


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _matched_contexts(
    observation: Mapping[str, Any],
    pattern: str | re.Pattern[str],
) -> list[dict[str, Any]]:
    regex = re.compile(pattern, re.I) if isinstance(pattern, str) else pattern
    matches: list[dict[str, str]] = []
    for direction in ("before", "after"):
        quote = _text(observation.get(f"context_{direction}"))
        sentence_id = observation.get(f"context_{direction}_sentence_id")
        source_field = observation.get(f"context_{direction}_source_field")
        source_start = observation.get(f"context_{direction}_start_char_source")
        source_end = observation.get(f"context_{direction}_end_char_source")
        if (
            quote
            and isinstance(sentence_id, str)
            and observation.get(f"context_{direction}_provenance_validated") is True
            and source_field in v14.ALLOWED_SOURCE_FIELDS
            and isinstance(source_start, int)
            and isinstance(source_end, int)
            and source_end - source_start == len(quote)
            and regex.search(quote)
        ):
            matches.append({
                "direction": direction,
                "sentence_id": sentence_id,
                "source_field": source_field,
                "quote": quote,
                "quote_start_char_source": source_start,
                "quote_end_char_source": source_end,
                "quote_start_char_clean": observation.get(
                    f"context_{direction}_start_char_clean"
                ),
                "quote_end_char_clean": observation.get(
                    f"context_{direction}_end_char_clean"
                ),
            })
    return matches


def _directional_context(
    observation: Mapping[str, Any],
    pattern: str | re.Pattern[str],
    directions: Sequence[str],
) -> list[dict[str, Any]]:
    """Choose one rule-specific direction; never combine adjacent records."""
    matches = _matched_contexts(observation, pattern)
    for direction in directions:
        directional = [row for row in matches if row.get("direction") == direction]
        if len(directional) == 1:
            return directional
    return []


def _rule(
    rule_id: str,
    *,
    context: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    context_rows = [dict(row) for row in context]
    return {
        "rule_id": rule_id,
        "grounding_scope": "adjacent_context" if context_rows else "exact",
        "context_evidence": context_rows,
    }


def recovery_candidates_v15(
    observation: Mapping[str, Any],
    taxonomy: Mapping[str, Mapping[str, str]],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Return uniquely auditable v1.5 taxonomy candidates.

    Model-authored observation prose is deliberately excluded from every rule.
    Only the exact source quote and at most one immediately adjacent, already
    located source record may supply a cue.
    """
    target = observation.get("target")
    statement_type = observation.get("statement_type")
    if target not in {"current_product", "seller_platform"}:
        return {}
    if statement_type in {
        "intended_use",
        "expectation_or_assumption",
        "unclear",
    }:
        return {}

    exact = _text(observation.get("evidence_text"))
    if not exact:
        return {}
    polarity = observation.get("polarity")
    candidates: dict[tuple[str, str], list[dict[str, Any]]] = {}

    def add(
        category: str,
        issue_group: str,
        rule_id: str,
        *,
        polarities: set[str] | None = None,
        context: Sequence[Mapping[str, Any]] = (),
        allow_service_pair: bool = False,
    ) -> None:
        if allow_service_pair:
            if target not in {"current_product", "seller_platform"}:
                return
        elif target != "current_product" or statement_type == "service_experience":
            return
        if polarities is not None and polarity not in polarities:
            return
        if category not in taxonomy or issue_group not in taxonomy[category]:
            return
        candidates.setdefault((category, issue_group), []).append(
            _rule(rule_id, context=context)
        )

    # Do not force either half of an explicitly composite damage statement.
    composite_box_and_product_damage = bool(
        re.search(
            r"\b(?:box|packag\w*)\b(?:\s+\w+){0,3}\s+"
            r"(?:and|&)\s+(?:\w+\s+){0,2}(?:laptop|computer|device)\b|"
            r"\b(?:laptop|computer|device)\b(?:\s+\w+){0,3}\s+"
            r"(?:and|&)\s+(?:\w+\s+){0,2}(?:box|packag\w*)\b",
            exact,
            re.I,
        )
        and re.search(r"\b(?:damag\w*|dent\w*|crush\w*)\b", exact, re.I)
    )

    if re.search(
        r"\b(?:(?:missing|no|without|not included)\s+(?:a\s+)?"
        r"(?:power|charging)\s+(?:cord|adapter|charger)|"
        r"(?:power|charging)\s+(?:cord|adapter|charger)\s+(?:was\s+)?missing|"
        r"(?:did|does)\s+not\s+come\s+with\s+(?:a\s+)?"
        r"(?:power|charging)\s+(?:cord|adapter|charger))\b",
        exact,
        re.I,
    ):
        add("battery", "charging", "missing_power_cord", polarities={"negative"})

    if re.search(
        r"\b(?:travel(?:s|ed|ing)?\s+(?:better|well|easily)|"
        r"easy\s+to\s+travel\s+with|travel\s+portability)\b",
        exact,
        re.I,
    ) and not re.search(r"\b(?:plan|intend|hope|will)\b", exact, re.I):
        add(
            "design_portability",
            "weight",
            "travel_portability_comparison",
            polarities={"positive", "negative", "mixed"},
        )

    if re.search(
        r"\b(?:light\s*weight|lightweight|heavy|portable|easy to (?:carry|transport))\b",
        exact,
        re.I,
    ):
        add(
            "design_portability",
            "weight",
            "explicit_weight_or_portability",
            polarities={"positive", "negative", "mixed", "neutral"},
        )

    if re.search(
        r"\b(?:good|great|nice|attractive|modern)[ -]?looking\b|"
        r"\blooks?\s+(?:good|great|nice|attractive|modern)\b",
        exact,
        re.I,
    ):
        add(
            "design_portability",
            "appearance",
            "adjective_looking_appearance",
            polarities={"positive", "negative", "mixed", "neutral"},
        )

    case_context = _directional_context(
        observation,
        r"\b(?:case|fit|size|dimension|travel|bringing it places)\b",
        ("after",),
    )
    explicit_dimensions = re.search(
        r"\b(?:laptop|computer|device|it(?:'s|’s)?)\b(?:\s+\w+){0,3}\s+"
        r"(?:really\s+|very\s+|too\s+)?(?:big|large|oversized)\b|"
        r"\b(?:case|sleeve)\b(?:\s+\w+){0,5}\s+"
        r"(?:fit|fits|fitting|size|large|small)\b",
        exact,
        re.I,
    )
    if explicit_dimensions and not re.search(r"\b(?:screen|display)\b", exact, re.I):
        needs_context = bool(
            re.search(r"^\s*(?:it|this|that)\b", exact, re.I)
            and not re.search(r"\b(?:laptop|computer|device|case|sleeve)\b", exact, re.I)
        )
        if not needs_context or case_context:
            add(
                "design_portability",
                "dimensions",
                "device_size_or_case_fit",
                polarities={"positive", "negative", "mixed", "neutral"},
                context=case_context if needs_context else (),
            )

    keyboard_context = [
        row for row in _directional_context(
            observation,
            r"\b(?:keyboard|keys?|backlit|backlight)\b",
            ("before",),
        )
        if _context_entities(_text(row.get("quote"))) == {"keyboard"}
    ]
    if keyboard_context and re.search(
        r"\b(?:dim|bright|lit|responsive|work\w*|broken|missing|"
        r"no\s+option|can(?:not|'t)|won't|does(?: not|n't))\b",
        exact,
        re.I,
    ):
        add(
            "input_devices",
            "keyboard",
            "keyboard_state_with_antecedent",
            polarities={"positive", "negative", "mixed", "neutral"},
            context=keyboard_context,
        )

    touchpad_context = [
        row for row in _directional_context(
            observation,
            r"\b(?:touchpad|trackpad|mouse|cursor|pointer)\w*\b",
            ("before",),
        )
        if _context_entities(_text(row.get("quote"))) == {"touchpad"}
    ]
    if touchpad_context and re.search(
        r"\b(?:touchy|wacky|responsive|unresponsive|work\w*|broken|"
        r"can(?:not|'t)|won't|does(?: not|n't))\b",
        exact,
        re.I,
    ):
        add(
            "input_devices",
            "touchpad_mouse",
            "touchpad_state_with_antecedent",
            polarities={"positive", "negative", "mixed", "neutral"},
            context=touchpad_context,
        )

    display_context = [
        row for row in _directional_context(
            observation,
            r"\b(?:screen|display|monitor)\b",
            ("before",),
        )
        if _context_entities(_text(row.get("quote"))) == {"display"}
    ]
    if display_context and re.search(
        r"\b(?:dim|bright|dark|clear|flicker\w*|work\w*|broken|"
        r"can(?:not|'t)|won't|does(?: not|n't))\b",
        exact,
        re.I,
    ):
        add(
            "display",
            "screen_quality",
            "display_state_with_antecedent",
            polarities={"positive", "negative", "mixed", "neutral"},
            context=display_context,
        )

    speaker_context = [
        row for row in _directional_context(
            observation,
            r"\b(?:speaker|sound|audio|volume)\w*\b",
            ("before",),
        )
        if _context_entities(_text(row.get("quote"))) == {"speaker"}
    ]
    if speaker_context and re.search(
        r"\b(?:quiet|low|loud|barely|hardly|hard\s+to)\b",
        exact,
        re.I,
    ):
        add(
            "audio",
            "low_volume",
            "speaker_volume_with_antecedent",
            polarities={"positive", "negative", "mixed", "neutral"},
            context=speaker_context,
        )
    if speaker_context and re.search(
        r"\b(?:work\w*|broken|fail\w*|can(?:not|'t)|won't|"
        r"does(?: not|n't))\b",
        exact,
        re.I,
    ):
        add(
            "audio",
            "speaker_sound",
            "speaker_failure_with_antecedent",
            polarities={"negative", "mixed", "neutral"},
            context=speaker_context,
        )

    storage_cue = re.search(r"\b(?:storage|ssd|hard drive|disk|drive)\b", exact, re.I)
    storage_speed = re.search(
        r"\b(?:read|write|speed|fast|slow|performance|performace|behavio(?:u)?r)\w*\b",
        exact,
        re.I,
    )
    if storage_cue and storage_speed:
        add(
            "memory_storage",
            "storage_performance",
            "storage_speed_or_behavior",
            polarities={"positive", "negative", "mixed", "neutral"},
        )

    storage_capacity = storage_cue and re.search(
        r"\b(?:capacity|space|room|full|\d+\s*(?:gb|tb|gig))\b|"
        r"\b(?:low|little|next to no)\s+(?:storage|space|capacity)\b",
        exact,
        re.I,
    )
    if storage_capacity and not re.search(
        r"\b(?:add|slot|replace|expand|upgrad)\w*\b", exact, re.I
    ):
        add(
            "memory_storage",
            "storage_capacity",
            "explicit_storage_capacity",
            polarities={"positive", "negative", "mixed", "neutral"},
        )

    if re.search(r"\b(?:ram|memory)\b", exact, re.I) and re.search(
        r"\b(?:\d+\s*(?:gb|gig)|adequate|enough|subpar|low|little|"
        r"next to no|amount|capacity)\b",
        exact,
        re.I,
    ) and not re.search(r"\b(?:add|slot|replace|expand|upgrad)\w*\b", exact, re.I):
        add(
            "memory_storage",
            "ram_capacity",
            "explicit_ram_amount_or_adequacy",
            polarities={"positive", "negative", "mixed", "neutral"},
        )

    upgrade_context = _directional_context(
        observation,
        r"\b(?:ram|memory|storage|ssd|hard drive|disk|drive)\b",
        ("after", "before"),
    )
    composite_memory_upgrade = bool(
        re.search(r"\bram\b", exact, re.I)
        and re.search(r"\bstorage\b", exact, re.I)
        and re.search(r"\b(?:next to no|low|little|inadequate)\b", exact, re.I)
        and re.search(r"\bupgrad\w*\b", exact, re.I)
    )
    if (
        re.search(r"\bupgrad(?:e|ed|ing|able|-able|ability)\b", exact, re.I)
        and not composite_memory_upgrade
    ):
        exact_memory = re.search(
            r"\b(?:ram|memory|storage|ssd|hard drive|disk|drive)\b",
            exact,
            re.I,
        )
        if (exact_memory or upgrade_context) and not re.search(
            r"\bbattery\b", exact, re.I
        ):
            add(
                "memory_storage",
                "upgradeability",
                "upgradeable_with_memory_context",
                polarities={"positive", "negative", "mixed", "neutral"},
                context=() if exact_memory else upgrade_context,
            )

    audio_context = _directional_context(
        observation,
        r"\b(?:sound|speaker|audio|volume|loud|quiet)\w*\b",
        ("before",),
    )
    exact_audio = re.search(
        r"\b(?:sound|speaker|audio|volume|loud|quiet)\w*\b", exact, re.I
    )
    audibility = re.search(
        r"\b(?:barely|hardly|can(?:not|'t)|could(?: not|n't))\b"
        r"(?:\s+\w+){0,4}\s+hear\b|\bhard\s+to\s+hear\b",
        exact,
        re.I,
    )
    fan_noise = re.search(r"\b(?:fan|coil|noise cancellation)\b", exact, re.I)
    if audibility and not fan_noise and (exact_audio or audio_context):
        add(
            "audio",
            "low_volume",
            "audibility_with_audio_context",
            polarities={"negative", "mixed"},
            context=() if exact_audio else audio_context,
        )
    if re.search(r"\bvolume\b(?:\s+\w+){0,3}\s+\b(?:low|quiet)\b", exact, re.I):
        add("audio", "low_volume", "explicit_low_volume", polarities={"negative", "mixed"})
    prior_low_volume = _directional_context(
        observation,
        r"\bvolume\b(?:\s+\w+){0,4}\s+\b(?:low|quiet)\b|"
        r"\b(?:barely|hardly|can't|cannot)\b(?:\s+\w+){0,4}\s+hear\b",
        ("before",),
    )
    if re.search(r"\b(?:headphones?|earphones?|buds?)\b", exact, re.I) and re.search(
        r"\b(?:solve|need|sell|moving on|switch)\w*\b", exact, re.I
    ) and prior_low_volume:
        add(
            "audio",
            "low_volume",
            "low_volume_workaround_continuation",
            polarities={"negative", "mixed"},
            context=prior_low_volume,
        )

    task_context = _directional_context(
        observation,
        r"\b(?:lesson planning|school projects?|email|web brows\w*|office|"
        r"word|games?|gaming|editing|streaming|workload|tasks?)\b",
        ("before",),
    )
    positive_speed = re.search(
        r"\bgood\s+speed\s+for\s+(?:most\s+)?(?:activities|tasks?|uses?)\b|"
        r"\blightning\s+fast\b|\beasy\s+to\s+navigate\b|"
        r"\b(?:it|this\s+(?:laptop|computer)|the\s+(?:laptop|computer)|"
        r"processor|cpu|system)\s+(?:is|was|feels?|runs?)\s+"
        r"(?:very\s+|really\s+)?(?:fast|quick|responsive)\b",
        exact,
        re.I,
    )
    if (
        positive_speed
        and polarity == "positive"
        and not storage_cue
        and not re.search(
            r"\b(?:boot|startup|start-up|transfer|download|upload|network|internet|"
            r"charg\w*|battery|shipping|delivery|support|service|setup|set-up)\w*\b",
            exact,
            re.I,
        )
    ):
        add(
            "performance",
            "processing_capability",
            "positive_general_speed",
            polarities={"positive"},
        )
    if re.search(r"\bdoes?\s+the\s+job\s+well\b", exact, re.I) and task_context:
        add(
            "performance",
            "processing_capability",
            "task_success_with_context",
            polarities={"positive"},
            context=task_context,
        )
    if re.search(
        r"\bworks?\s+well\s+for\s+(?:email|web|brows\w*|office|"
        r"school|tasks?|projects?)\b",
        exact,
        re.I,
    ):
        add(
            "performance",
            "processing_capability",
            "explicit_task_success",
            polarities={"positive"},
        )

    literal_s_mode = bool(re.search(r"\bs\s*mode\b", exact, re.I))
    missing_product_key = bool(
        re.search(
            r"\b(?:missing|no|without|does(?: not|n't)|did(?: not|n't))\b"
            r"(?:\s+\w+){0,6}\s+\bproduct key\b|"
            r"\bproduct key\b(?:\s+\w+){0,4}\s+\b(?:missing|not included)\b",
            exact,
            re.I,
        )
    )
    account_setup_issue = bool(
        re.search(r"\b(?:microsoft|local)\s+account\b", exact, re.I)
        and re.search(
            r"\b(?:must|force\w*|sign\s+in|log\s+in|remove\w*|"
            r"no\s+option|create\w*|set\s*up|setup)\b",
            exact,
            re.I,
        )
    )
    os_setup = bool(
        re.search(r"\b(?:bloatware|crapware)\b", exact, re.I)
        or re.search(r"\bwindows\b", exact, re.I)
        and re.search(r"\b(?:bloat\w*|packaged)\b", exact, re.I)
        or missing_product_key
        or account_setup_issue
        or re.search(r"\b(?:set\s*up|setup)\b", exact, re.I)
        and re.search(r"\b(?:hours?|restart|network|account|terrible|worst)\b", exact, re.I)
        or re.search(
            r"\b(?:hours?|minutes?)\b(?:\s+\w+){0,5}\s+"
            r"\bset\s+(?:(?:them|it|the\s+computers?)\s+)?up\b",
            exact,
            re.I,
        )
        or re.search(r"\bnetwork entry (?:section|settings?)\b", exact, re.I)
    )
    if os_setup:
        add(
            "software_os",
            "os_setup",
            "explicit_os_setup_or_bloatware",
            polarities={"positive", "negative", "mixed", "neutral"},
        )
    if literal_s_mode and not missing_product_key and re.search(
        r"\b(?:remove|exit|get out|bypass|restrict|prevent|force|hours?)\w*\b",
        exact,
        re.I,
    ):
        add(
            "software_os",
            "s_mode_restriction",
            "literal_s_mode_restriction",
            polarities={"negative", "mixed", "neutral"},
        )

    if re.search(
        r"\b(?:can(?:not|'t)|will not|won't|does(?: not|n't))\b"
        r"(?:\s+\w+){0,4}\s+(?:run|install|download|open)\b",
        exact,
        re.I,
    ) and re.search(r"\b(?:app|application|chrome|browser|google maps)\b", exact, re.I):
        add(
            "software_os",
            "app_compatibility",
            "explicit_app_action_block",
            polarities={"negative"},
        )

    wireless = re.search(r"\b(?:wi-?fi|wireless|bluetooth)\b", exact, re.I)
    connection_failure = re.search(
        r"\b(?:off\s*line|offline|disconnect\w*|not connect|"
        r"would not connect|won't connect|cannot connect)\b",
        exact,
        re.I,
    )
    if connection_failure:
        add(
            "connectivity_ports",
            "wifi_bluetooth" if wireless else "other_connectivity",
            "explicit_wireless_failure" if wireless else "unspecified_connection_failure",
            polarities={"negative", "mixed"},
        )

    physical_port = r"(?:(?:sd|micro\s*sd)[ -]?card(?:\s+slot)?|(?:usb|hdmi)(?:\s+port)?)"
    missing_physical_port = re.search(
        rf"\b(?:no|without|missing)\s+(?:an?\s+)?{physical_port}\b|"
        rf"\b(?:has|have)\s+no\s+(?:an?\s+)?{physical_port}\b|"
        rf"\b(?:does(?: not|n't)|do\s+not)\s+have\s+(?:an?\s+)?{physical_port}\b|"
        rf"\b{physical_port}\b(?:\s+\w+){{0,3}}\s+\b(?:missing|absent)\b",
        exact,
        re.I,
    )
    if missing_physical_port:
        add(
            "connectivity_ports",
            "usb_other_ports",
            "explicit_missing_physical_port",
            polarities={"negative", "neutral", "mixed"},
        )

    if re.search(r"\b(?:touch\s*screen|touchscreen)\b", exact, re.I):
        add(
            "display",
            "touchscreen",
            "explicit_touchscreen_capability",
            polarities={"positive", "negative", "neutral", "mixed"},
        )

    if not composite_box_and_product_damage and not re.search(
        r"\b(?:box|packag\w*)\b", exact, re.I
    ) and re.search(
        r"\b(?:dent\w*|dented)\b", exact, re.I
    ) and not re.search(r"\bhinge\b", exact, re.I):
        add(
            "build_quality",
            "housing_damage",
            "explicit_housing_dent",
            polarities={"negative", "mixed"},
        )

    if not composite_box_and_product_damage and re.search(
        r"\b(?:box|packag\w*)\b", exact, re.I
    ) and re.search(r"\b(?:damag\w*|dent\w*|crush\w*|torn|tear)\b", exact, re.I):
        add(
            "purchase_service",
            "delivery_damage",
            "damaged_product_packaging",
            polarities={"negative", "mixed"},
            allow_service_pair=True,
        )

    boot_context = _directional_context(
        observation,
        r"\b(?:would(?: not|n['’]t)|won't|cannot|can't|does(?: not|n't))\s+"
        r"(?:boot|start)|\bboot failure\b",
        ("before", "after"),
    )
    exact_boot = re.search(
        r"\b(?:would(?: not|n['’]t)|won't|cannot|can't|does(?: not|n't))\s+"
        r"(?:boot|start)\b",
        exact,
        re.I,
    )
    if re.search(r"\bblack screen\b", exact, re.I) and (exact_boot or boot_context):
        add(
            "system_stability",
            "boot_failure",
            "black_screen_with_boot_failure",
            polarities={"negative", "mixed"},
            context=() if exact_boot else boot_context,
        )

    if re.search(
        r"\b(?:broke|broken|breakage)\b(?:\s+\w+){0,5}\s+"
        r"(?:within|under|less than|after)\s+\d+\s+years?\b|"
        r"\bdestroyed\s+by\s+poor\s+mechanical\s+design\b",
        exact,
        re.I,
    ):
        add(
            "build_quality",
            "durability",
            "explicit_early_breakage_or_mechanical_failure",
            polarities={"negative", "mixed"},
        )

    if re.search(r"\b(?:debris|sharp|rough)\b", exact, re.I) and re.search(
        r"\b(?:plastic|body|casing|chassis|material)\b", exact, re.I
    ) and re.search(r"\b(?:cut|scratch|injur|sharp)\w*\b", exact, re.I):
        add(
            "build_quality",
            "durability",
            "explicit_material_hazard",
            polarities={"negative", "mixed", "neutral"},
        )

    if re.search(
        r"\b(?:plastic[ -]?(?:y|ky)|plastic\s+(?:feel|feeling))\b",
        exact,
        re.I,
    ) and re.search(r"\b(?:bottom|body|screen|casing|chassis|lid)\b", exact, re.I):
        add(
            "build_quality",
            "durability",
            "explicit_plastic_material_quality",
            polarities={"positive", "negative", "mixed", "neutral"},
        )

    if re.search(r"\b(?:no|without|missing|does(?: not|n't) have)\b", exact, re.I) and re.search(
        r"\b(?:dvd|optical)\s+drive\b", exact, re.I
    ):
        add(
            "other",
            "other_specific",
            "explicit_missing_optical_drive",
            polarities={"negative", "neutral", "mixed"},
        )

    positive_value = re.search(
        r"\b(?:good|great|best|excellent)\s+(?:laptop|computer|device)\s+"
        r"for\s+(?:the\s+)?price\b|\bbest\s+(?:value|deal)\b|"
        r"\bworth\s+(?:the\s+)?money\b",
        exact,
        re.I,
    )
    negative_value = re.search(
        r"\bexpensive\s+mistake\b|\bwaste\s+of\s+money\b|"
        r"\bnot\s+worth\s+(?:the\s+)?(?:money|cost)\b",
        exact,
        re.I,
    )
    budget_value = re.search(r"\bon\s+a\s+budget\b", exact, re.I) and re.search(
        r"\b(?:love|recommend|best|good|great)\b", exact, re.I
    )
    if positive_value or budget_value:
        add(
            "price_value",
            "value_for_money",
            "explicit_positive_value_judgment",
            polarities={"positive", "mixed"},
        )
    if negative_value:
        add(
            "price_value",
            "value_for_money",
            "explicit_negative_value_judgment",
            polarities={"negative", "mixed"},
        )

    battery_context = _directional_context(
        observation,
        r"\b(?:battery|runtime|charge)\b",
        ("before", "after"),
    )
    battery_duration = re.search(
        r"\b(?:\d+\s*(?:to\s*\d+\s*)?hours?|runtime|per\s+charge|"
        r"between\s+charges)\b",
        exact,
        re.I,
    )
    exact_battery = re.search(r"\b(?:battery|runtime|per\s+charge)\b", exact, re.I)
    if re.search(r"\bbattery\s+life\b", exact, re.I) or (
        battery_duration and (exact_battery or battery_context)
    ):
        add(
            "battery",
            "battery_runtime",
            "explicit_battery_runtime",
            polarities={"positive", "negative", "mixed", "neutral"},
            context=() if exact_battery else battery_context,
        )

    return candidates


def _repair_adjustment(
    *,
    adjustment_type: str,
    matches: Sequence[Mapping[str, Any]],
    from_status: Any,
    from_category: Any,
    from_issue_group: Any,
    to_category: str,
    to_issue_group: str,
) -> dict[str, Any]:
    return {
        "type": adjustment_type,
        "policy_version": POLICY_VERSION_V15,
        "rule_matches": [dict(item) for item in matches],
        "from_mapping_status": from_status,
        "from_category": from_category,
        "from_issue_group": from_issue_group,
        "to_category": to_category,
        "to_issue_group": to_issue_group,
    }


_CATEGORY_REPAIR_ALLOWLIST = {
    (
        "missing_power_cord",
        ("connectivity_ports", "usb_other_ports"),
        ("battery", "charging"),
    ),
    (
        "missing_power_cord",
        ("purchase_service", "delivery_damage"),
        ("battery", "charging"),
    ),
    (
        "explicit_os_setup_or_bloatware",
        ("software_os", "s_mode_restriction"),
        ("software_os", "os_setup"),
    ),
    (
        "literal_s_mode_restriction",
        ("software_os", "os_setup"),
        ("software_os", "s_mode_restriction"),
    ),
    (
        "explicit_housing_dent",
        ("build_quality", "hinge_assembly"),
        ("build_quality", "housing_damage"),
    ),
    (
        "damaged_product_packaging",
        ("build_quality", "housing_damage"),
        ("purchase_service", "delivery_damage"),
    ),
    (
        "unspecified_connection_failure",
        ("connectivity_ports", "wifi_bluetooth"),
        ("connectivity_ports", "other_connectivity"),
    ),
    (
        "black_screen_with_boot_failure",
        ("display", "screen_quality"),
        ("system_stability", "boot_failure"),
    ),
    (
        "positive_general_speed",
        ("performance", "slow_performance"),
        ("performance", "processing_capability"),
    ),
    (
        "storage_speed_or_behavior",
        ("performance", "slow_performance"),
        ("memory_storage", "storage_performance"),
    ),
    (
        "storage_speed_or_behavior",
        ("performance", "processing_capability"),
        ("memory_storage", "storage_performance"),
    ),
    (
        "storage_speed_or_behavior",
        ("memory_storage", "storage_capacity"),
        ("memory_storage", "storage_performance"),
    ),
    (
        "explicit_ram_amount_or_adequacy",
        ("memory_storage", "storage_capacity"),
        ("memory_storage", "ram_capacity"),
    ),
    (
        "device_size_or_case_fit",
        ("design_portability", "appearance"),
        ("design_portability", "dimensions"),
    ),
    (
        "device_size_or_case_fit",
        ("design_portability", "weight"),
        ("design_portability", "dimensions"),
    ),
}


def _category_repair_allowed(
    matches: Sequence[Mapping[str, Any]],
    from_pair: tuple[Any, Any],
    to_pair: tuple[str, str],
) -> bool:
    return any(
        (match.get("rule_id"), from_pair, to_pair) in _CATEGORY_REPAIR_ALLOWLIST
        for match in matches
    )


def complete_claims_v15(
    gate: Mapping[str, Any],
    observations: Sequence[Mapping[str, Any]],
    taxonomy: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    """Complete v1.4-valid claims, then apply unique v1.5 repairs only."""
    result = v14.complete_claims_with_unmapped_fallbacks(
        gate,
        observations,
        taxonomy,
    )
    observation_by_id = {
        item.get("observation_id"): item
        for item in observations
        if isinstance(item.get("observation_id"), str)
    }
    completed: list[dict[str, Any]] = []
    recall_recoveries = 0
    category_repairs = 0

    for claim_value in result.get("accepted_claims", []):
        claim = dict(claim_value)
        observation = observation_by_id.get(claim.get("observation_id"))
        candidates = (
            recovery_candidates_v15(observation, taxonomy)
            if isinstance(observation, Mapping)
            else {}
        )
        context_recovered = bool(
            isinstance(observation, Mapping)
            and observation.get("context_recovery_adjustments")
        )
        originally_mapped = claim.get("mapping_status") == "mapped"
        original_pair = (claim.get("category"), claim.get("issue_group"))
        context_pair_repair = False
        if context_recovered and claim.get("mapping_status") == "mapped":
            unique_pair = next(iter(candidates)) if len(candidates) == 1 else None
            current_pair = (claim.get("category"), claim.get("issue_group"))
            if unique_pair != current_pair:
                adjustments = list(claim.get("normalization_adjustments", []) or [])
                adjustments.append({
                    "type": "v15_rejected_unaudited_context_mapping",
                    "policy_version": POLICY_VERSION_V15,
                    "from_category": claim.get("category"),
                    "from_issue_group": claim.get("issue_group"),
                    "observation_context_adjustments": list(
                        observation.get("context_recovery_adjustments", []) or []
                    ),
                })
                claim.update({
                    "mapping_status": "unmapped",
                    "category": None,
                    "issue_group": None,
                    "deterministic_status": "valid_unmapped_after_v15_context_guard",
                    "normalization_source": "deterministic_v15_context_guard",
                    "normalization_adjustments": adjustments,
                })
                context_pair_repair = unique_pair is not None
        if len(candidates) == 1:
            (category, issue_group), matches = next(iter(candidates.items()))
            current_pair = (claim.get("category"), claim.get("issue_group"))
            if claim.get("mapping_status") == "unmapped":
                adjustments = list(claim.get("normalization_adjustments", []) or [])
                adjustments.append(_repair_adjustment(
                    adjustment_type=(
                        "canonicalized_v15_context_taxonomy_pair"
                        if context_pair_repair
                        else "v15_recall_recovery"
                    ),
                    matches=matches,
                    from_status="mapped" if context_pair_repair else claim.get("mapping_status"),
                    from_category=original_pair[0] if context_pair_repair else claim.get("category"),
                    from_issue_group=original_pair[1] if context_pair_repair else claim.get("issue_group"),
                    to_category=category,
                    to_issue_group=issue_group,
                ))
                claim.update({
                    "mapping_status": "mapped",
                    "category": category,
                    "issue_group": issue_group,
                    "deterministic_status": (
                        "valid_mapped_canonicalized_v15_context"
                        if context_pair_repair
                        else "valid_mapped_by_v15_recall_recovery"
                    ),
                    "normalization_source": (
                        "model_with_v15_context_category_repair"
                        if context_pair_repair
                        else "deterministic_v15_recall_recovery"
                    ),
                    "normalization_adjustments": adjustments,
                })
                if context_pair_repair:
                    category_repairs += 1
                else:
                    recall_recoveries += 1
            elif current_pair == (category, issue_group) and (
                context_recovered
                or any(
                    match.get("grounding_scope") == "adjacent_context"
                    for match in matches
                )
            ):
                adjustments = list(claim.get("normalization_adjustments", []) or [])
                adjustments.append({
                    "type": "v15_context_grounding_validated",
                    "policy_version": POLICY_VERSION_V15,
                    "rule_matches": [dict(item) for item in matches],
                    "observation_context_adjustments": (
                        list(observation.get("context_recovery_adjustments", []) or [])
                        if isinstance(observation, Mapping)
                        else []
                    ),
                })
                claim["normalization_adjustments"] = adjustments
                claim["normalization_source"] = (
                    "model_with_v15_context_validation"
                    if originally_mapped
                    else claim.get("normalization_source")
                )
            elif (
                current_pair != (category, issue_group)
                and _category_repair_allowed(
                    matches,
                    current_pair,
                    (category, issue_group),
                )
            ):
                adjustments = list(claim.get("normalization_adjustments", []) or [])
                adjustments.append(_repair_adjustment(
                    adjustment_type="canonicalized_v15_taxonomy_pair",
                    matches=matches,
                    from_status=claim.get("mapping_status"),
                    from_category=claim.get("category"),
                    from_issue_group=claim.get("issue_group"),
                    to_category=category,
                    to_issue_group=issue_group,
                ))
                claim.update({
                    "category": category,
                    "issue_group": issue_group,
                    "deterministic_status": "valid_mapped_canonicalized_v15",
                    "normalization_source": "model_with_v15_deterministic_adjustments",
                    "normalization_adjustments": adjustments,
                })
                category_repairs += 1
        completed.append(claim)

    result = dict(result)
    result["accepted_claims"] = completed
    result["fallback_unmapped_count"] = sum(
        claim.get("mapping_status") == "unmapped"
        and claim.get("normalization_source") == "deterministic_unmapped_fallback"
        for claim in completed
    )
    result["v15_recall_recovery_count"] = recall_recoveries
    result["v15_category_repair_count"] = category_repairs
    result["deterministic_recovery_count"] = (
        int(result.get("deterministic_recovery_count", 0)) + recall_recoveries
    )
    result["policy_version"] = POLICY_VERSION_V15
    return result


__all__ = [
    "CLAIM_SCHEMA_VERSION_V15",
    "OBSERVATION_SCHEMA_VERSION_V15",
    "POLICY_VERSION_V15",
    "complete_claims_v15",
    "normalization_payload_v15",
    "parse_observation_attempt_v15",
    "parse_observation_response_v15",
    "recovery_candidates_v15",
]
