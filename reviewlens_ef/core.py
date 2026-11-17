"""Deterministic contracts and cache helpers for the Evidence-first pipeline.

This module deliberately uses only the Python standard library.  Model output is
untrusted input: evidence locations, observation IDs, inherited labels, and
taxonomy pairs are all decided or checked here.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PILOT_REVIEW_IDS = (
    "electronics_1155658",
    "electronics_245706",
    "electronics_1801744",
    "electronics_980004",
    "electronics_70840",
    "electronics_1005118",
    "electronics_1583730",
    "electronics_229659",
    "electronics_218793",
    "electronics_539422",
)

OBSERVATION_SCHEMA_VERSION = "evidence_first_observation_v1_3"
CLAIM_SCHEMA_VERSION = "evidence_first_claim_v1_4"
ALLOWED_TARGETS = {
    "current_product",
    "previous_product",
    "other_product",
    "seller_platform",
    "unclear",
}
ALLOWED_POLARITIES = {"positive", "negative", "neutral", "mixed"}
ALLOWED_MAPPING_STATUSES = {"mapped", "unmapped"}
ALLOWED_STATEMENT_TYPES = {
    "actual_experience",
    "capability_or_limitation",
    "intended_use",
    "expectation_or_assumption",
    "feature_description",
    "service_experience",
    "overall_opinion",
    "unclear",
}
ALLOWED_SOURCE_FIELDS = {"clean_text", "title"}
SAFE_PARENT_CANONICALIZATION_GROUPS = {
    "slow_performance",
    "freezing_crashing",
    "processing_capability",
    "weight",
    "storage_performance",
    "storage_capacity",
    "upgradeability",
    "charging",
    "value_for_money",
    "screen_quality",
    "app_compatibility",
    "speaker_sound",
    "keyboard",
    "touchpad_mouse",
    "appearance",
    "price",
    "usb_other_ports",
}

OBSERVATION_MODEL_KEYS = {
    "sentence_id",
    "evidence_text",
    "observation",
    "target",
    "polarity",
    "statement_type",
}
CLAIM_MODEL_KEYS = {
    "observation_id",
    "mapping_status",
    "category",
    "issue_group",
    "claim",
}


# Deliberately conservative lexical gates. They catch high-risk category
# mistakes; they are not intended to solve semantic classification by regex.
ASPECT_CUE_GROUPS = {
    "display": re.compile(r"\b(?:screen|display|monitor|resolution|brightness)\b", re.I),
    "input": re.compile(
        r"\b(?:keyboard|keys|backlit|backlight|touchpad|trackpad|mouse|cursor)\b",
        re.I,
    ),
    "ports": re.compile(r"\b(?:usb|hdmi|port|sd[- ]?card)\b", re.I),
    "audio": re.compile(r"\b(?:speaker|sound|audio|headphone|microphone)\b", re.I),
    "memory_storage": re.compile(r"\b(?:ram|memory|storage|ssd|hard drive|disk)\b", re.I),
    "performance": re.compile(
        r"\b(?:processor|cpu|gpu|graphics|performance|speed|fast|slow|lag|"
        r"quick\w*|instant\w*|load\w*|navigat\w*|tasks?|handl\w*|responsive)\w*\b",
        re.I,
    ),
    "battery": re.compile(
        r"\b(?:battery|charg\w*|recharg\w*|power adapter)\b",
        re.I,
    ),
    "design": re.compile(
        r"\b(?:weight|lightweight|light weight|portable|carry|thin|design|"
        r"appearance|looks?\s+(?:great|good|nice|beautiful|sleek))\b",
        re.I,
    ),
    "software": re.compile(
        r"\b(?:windows|software|app(?:lication)?s?|browser|chrome|s mode|"
        r"crapware|bloatware)\b",
        re.I,
    ),
    "stability": re.compile(r"\b(?:crash\w*|freez\w*|restart\w*|blue screen|shut\w* down)\b", re.I),
    "service": re.compile(r"\b(?:warranty|support|seller|delivery|shipping|refund|return|repair)\b", re.I),
    "price_value": re.compile(
        r"\b(?:price|value|cost|worth|money|budget|deal|inexpensive|affordable)\b",
        re.I,
    ),
}

INTENDED_USE_CUE = re.compile(
    r"\b(?:I\s+(?:just\s+)?(?:really\s+)?(?:want|wanted|need|needed)\b|"
    r"I(?:'m| am)\s+(?:planning|looking)\s+to\b|"
    r"(?:my|his|her|their|whose)\s+intention\b|"
    r"intention (?:was |is )?to\b|intended to\b|"
    r"plan(?:ning)? (?:to|on)\b|bought (?:it |this )?for\b|"
    r"purpose (?:was |is )?to\b|going to use\b)\b",
    re.I,
)
EXPECTATION_CUE = re.compile(
    r"\b(?:assum\w*|expect\w*|hop\w*|should|supposed|thought|"
    r"anticipat\w*|believ\w*)\b",
    re.I,
)
EXPECTATION_ONLY_CUE = re.compile(
    r"\b(?:should|supposed|assuming|expecting|hoping|thought\s+it\s+would)\b",
    re.I,
)
SERVICE_CUE = re.compile(
    r"\b(?:warranty|support|service|seller|amazon|manufacturer|acer|"
    r"refund|return|repair|turn\s*around|delivery|shipping)\b",
    re.I,
)
SERVICE_EXPERIENCE_CUE = re.compile(
    r"\b(?:warranty|support|service|seller|amazon|refund|return|repair|"
    r"turn\s*around|delivery|shipping)\b",
    re.I,
)
POWER_HARDWARE_PATTERN = (
    r"(?:(?:power|charging)\s+port|charger|power adapter|ac adapter)"
)
POWER_HARDWARE_CUE = re.compile(
    rf"\b{POWER_HARDWARE_PATTERN}\b",
    re.I,
)
POWER_FAILURE_RELATION = re.compile(
    rf"\b{POWER_HARDWARE_PATTERN}\b(?:"
    r"\s+(?:has\s+|had\s+)?fail(?:s|ed|ing)?\b"
    r"(?=$|\s+(?:again|completely|suddenly|intermittently|after|within|"
    r"in|during|yesterday|today)\b)"
    r"|\s+(?:(?:is|was|has\s+been|had\s+been)\s+)?(?:broken|dead)\b"
    r"|\s+(?:goes|went|has\s+gone|had\s+gone)\s+bad\b"
    r"|\s+(?:stops|stopped)\s+working\b"
    r"|\s+(?:won't|doesn't|does\s+not|cannot|can't|isn't|is\s+not)"
    r"\s+work(?:ing)?\b"
    r"|\s+broke\s+down\b"
    r"|\s+failure\b)"
    rf"|\b(?:broken|dead|failed)\s+{POWER_HARDWARE_PATTERN}\b",
    re.I,
)
POWER_WORKING_RELATION = re.compile(
    rf"\b{POWER_HARDWARE_PATTERN}\b\s+"
    r"(?:(?:still|now)\s+)?(?:works?|worked|is\s+working|was\s+working)\b",
    re.I,
)
NEGATIVE_CHARGING_PREDICATE = re.compile(
    r"\b(?:doesn't|does\s+not|won't|cannot|can't|fails?\s+to|"
    r"stops?\s+|is\s+not|not\s+)(?:re)?charg\w*\b"
    r"|\b(?:re)?charg(?:e|es|ed|ing)\b"
    r"(?:\s+(?:is|was|very|too|quite|really)){0,3}\s+"
    r"(?:slow\w*|fails?|failed|stops?|stopped)\b",
    re.I,
)
POSITIVE_CHARGING_PREDICATE = re.compile(
    r"\b(?:re)?charg(?:e|es|ed|ing)\b"
    r"(?:\s+(?:is|was|very|quite|really)){0,3}\s+"
    r"(?:fast\w*|quick\w*|well|properly|works?|working)\b",
    re.I,
)
EVIDENCE_CLAUSE_SPLIT = re.compile(
    r"(?:[.!?;,:]+|\b(?:and|but|however|whereas|while|although)\b)",
    re.I,
)
APP_ENTITY_PATTERN = r"(?:google\s+chrome|chrome|browser|app(?:lication)?s?)"
APP_TRANSFER_ACTION_PATTERN = (
    r"(?:download\w*|install\w*|run\w*|launch\w*)"
)
APP_DIRECT_ACTION_PATTERN = (
    r"(?:download\w*|install\w*|run\w*|open\w*|launch\w*)"
)
APP_BLOCK_OBJECT_RELATION = re.compile(
    r"\b(?:software|computer|system|product|laptop|windows|it)\b"
    r"(?:\s+\w+){0,4}\s+"
    r"\b(?:block(?:s|ed|ing)?|prevent(?:s|ed|ing)?)\b"
    rf"(?:\s+\w+){{0,4}}\s+\b{APP_ENTITY_PATTERN}\b"
    rf"(?:\s+(?:from|being|to)){{0,3}}\s+"
    rf"\b{APP_TRANSFER_ACTION_PATTERN}\b",
    re.I,
)
APP_PASSIVE_BLOCK_RELATION = re.compile(
    rf"\b{APP_ENTITY_PATTERN}\b(?:\s+\w+){{0,3}}\s+"
    r"\b(?:is|was|gets?|got)\s+(?:blocked|prevented)\b"
    rf"(?:\s+(?:from|being|to)){{0,3}}\s+"
    rf"\b{APP_TRANSFER_ACTION_PATTERN}\b",
    re.I,
)
APP_INABILITY_RELATION = re.compile(
    rf"\b{APP_ENTITY_PATTERN}\b(?:\s+\w+){{0,3}}\s+"
    r"\b(?:cannot|can't|won't|doesn't|does\s+not|fails?\s+to|"
    r"is\s+unable\s+to)\b"
    rf"(?:\s+\w+){{0,2}}\s+\b{APP_DIRECT_ACTION_PATTERN}\b"
    r"|\b(?:cannot|can't|unable\s+to|fails?\s+to)\b"
    rf"\s+\b{APP_DIRECT_ACTION_PATTERN}\b"
    rf"(?:\s+(?:the|a|an)){{0,1}}\s+\b{APP_ENTITY_PATTERN}\b",
    re.I,
)
APP_INCOMPATIBILITY_RELATION = re.compile(
    rf"\b{APP_ENTITY_PATTERN}\b(?:\s+\w+){{0,3}}\s+"
    r"\b(?:is\s+|was\s+)?(?:incompatib\w*|unsupported|not\s+supported)\b"
    r"|\b(?:incompatib\w*|unsupported)\b"
    rf"(?:\s+\w+){{0,3}}\s+\b{APP_ENTITY_PATTERN}\b",
    re.I,
)


def _evidence_clauses(text: str) -> list[str]:
    return [
        clause.strip()
        for clause in EVIDENCE_CLAUSE_SPLIT.split(text)
        if clause.strip()
    ]


def _aspect_cue_names(text: str) -> set[str]:
    return {
        name for name, pattern in ASPECT_CUE_GROUPS.items()
        if pattern.search(text)
    }


def _power_failure_supported(text: str) -> bool:
    return any(
        _aspect_cue_names(clause) <= {"battery", "ports"}
        and _has_unnegated_match(POWER_FAILURE_RELATION, clause)
        for clause in _evidence_clauses(text)
    )


def _charging_behavior_supported(text: str, polarity: Any = None) -> bool:
    negative_supported = _power_failure_supported(text)
    positive_supported = False
    for clause in _evidence_clauses(text):
        if not _aspect_cue_names(clause) <= {"battery", "ports", "performance"}:
            continue
        if POWER_WORKING_RELATION.search(clause):
            positive_supported = True
        # Remove hardware nouns so "charging port" cannot itself count as a
        # charging predicate. A remaining predicate must express behavior.
        without_hardware = POWER_HARDWARE_CUE.sub(" ", clause)
        if NEGATIVE_CHARGING_PREDICATE.search(without_hardware):
            negative_supported = True
        if POSITIVE_CHARGING_PREDICATE.search(without_hardware):
            positive_supported = True
    if polarity == "negative":
        return negative_supported
    if polarity == "positive":
        return positive_supported
    if polarity == "mixed":
        return negative_supported and positive_supported
    if polarity == "neutral":
        return False
    return negative_supported or positive_supported


def _app_limitation_supported(text: str) -> bool:
    for clause in _evidence_clauses(text):
        if not _aspect_cue_names(clause) <= {"software"}:
            continue
        if any(pattern.search(clause) for pattern in (
            APP_BLOCK_OBJECT_RELATION,
            APP_PASSIVE_BLOCK_RELATION,
            APP_INABILITY_RELATION,
            APP_INCOMPATIBILITY_RELATION,
        )):
            return True
    return False


def _has_unnegated_match(
    pattern: str | re.Pattern[str],
    text: str,
) -> bool:
    matches = (
        pattern.finditer(text)
        if isinstance(pattern, re.Pattern)
        else re.finditer(pattern, text, re.I)
    )
    for match in matches:
        prefix = text[max(0, match.start() - 40):match.start()]
        if re.search(
            r"\b(?:not|never|hardly|barely|isn't|wasn't|aren't|weren't)"
            r"(?:\s+\w+){0,4}\s*$",
            prefix,
            re.I,
        ):
            continue
        return True
    return False


def canonical_json(value: Any) -> str:
    """Serialize data deterministically for cache fingerprints."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def hash_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def taxonomy_hash(taxonomy: Mapping[str, Mapping[str, str]]) -> str:
    return hash_json(taxonomy)


def parse_json_object(raw: str) -> dict[str, Any]:
    """Parse one model JSON object, tolerating Qwen think/fence wrappers only."""
    if not isinstance(raw, str):
        raise ValueError("model response must be text")
    text = re.sub(r"^\s*<think>.*?</think>\s*", "", raw, flags=re.DOTALL).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```\s*$", "", text)
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("expected one JSON object")
    return parsed


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(value)
    return rows


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def backup_existing(path: Path) -> Path | None:
    """Move only the requested Evidence-first file to a timestamped backup."""
    if not path.exists():
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    destination = path.with_name(f"{path.stem}.backup_{stamp}{path.suffix}")
    shutil.move(path, destination)
    return destination


def select_pilot_ids(
    available_review_ids: Iterable[str],
    limit: int = 10,
) -> list[str]:
    """Return the fixed, reviewable pilot in its declared order.

    A zero limit is treated as the safe 10-review default.  This function never
    expands beyond the fixed pilot, so it cannot accidentally trigger all 189
    reviews.
    """
    if limit == 0:
        limit = len(PILOT_REVIEW_IDS)
    if limit < 1 or limit > len(PILOT_REVIEW_IDS):
        raise ValueError(f"Evidence-first pilot --limit must be 1..{len(PILOT_REVIEW_IDS)}")
    available = set(available_review_ids)
    selected = list(PILOT_REVIEW_IDS[:limit])
    missing = [review_id for review_id in selected if review_id not in available]
    if missing:
        raise ValueError(f"Pilot reviews missing from source/cache: {missing}")
    return selected


def _top_level_list(
    data: Mapping[str, Any],
    key: str,
) -> tuple[list[Any], list[str]]:
    errors: list[str] = []
    if set(data) != {key}:
        errors.append(f"top level must contain exactly {key}")
    values = data.get(key)
    if not isinstance(values, list):
        errors.append(f"{key} must be a list")
        return [], errors
    return values, errors


def validate_observations(
    data: Mapping[str, Any],
    review: Mapping[str, Any],
    sentences: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate and localize raw observation extraction output.

    Exact evidence offsets are calculated here.  A quote must appear exactly
    once inside its cited sentence; ambiguous duplicate spans are quarantined.
    """
    candidates, errors = _top_level_list(data, "observations")
    review_id = review.get("review_id")
    clean_text = review.get("clean_text")
    title = review.get("title", "")
    if not isinstance(review_id, str) or not isinstance(clean_text, str):
        raise ValueError("review requires string review_id and clean_text")
    if not isinstance(title, str):
        raise ValueError("review title must be text when present")

    sentence_lookup: dict[str, Mapping[str, Any]] = {}
    sentence_order: dict[str, int] = {}
    ordered_sentences: list[Mapping[str, Any]] = []
    for sentence in sentences:
        sentence_id = sentence.get("sentence_id")
        if isinstance(sentence_id, str):
            if sentence_id in sentence_lookup:
                raise ValueError(f"duplicate sentence ID: {sentence_id}")
            sentence_lookup[sentence_id] = sentence
            sentence_order[sentence_id] = len(ordered_sentences)
            ordered_sentences.append(sentence)

    def context_at(source_index: int, direction: int) -> tuple[str | None, str | None]:
        context_index = source_index + direction
        if not 0 <= context_index < len(ordered_sentences):
            return None, None
        context = ordered_sentences[context_index]
        context_id = context.get("sentence_id")
        context_text = context.get("sentence_text")
        return (
            context_id if isinstance(context_id, str) else None,
            context_text if isinstance(context_text, str) else None,
        )

    located: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_exact: set[str] = set()

    for index, item in enumerate(candidates):
        reasons: list[str] = []
        if not isinstance(item, dict):
            rejected.append({"index": index, "item": item, "errors": ["not an object"]})
            continue
        if set(item) != OBSERVATION_MODEL_KEYS:
            reasons.append(
                f"wrong fields: {sorted(set(item) ^ OBSERVATION_MODEL_KEYS)}"
            )

        sentence_id = item.get("sentence_id")
        evidence_text = item.get("evidence_text")
        observation = item.get("observation")
        target = item.get("target")
        polarity = item.get("polarity")
        statement_type = item.get("statement_type")

        sentence = sentence_lookup.get(sentence_id) if isinstance(sentence_id, str) else None
        if sentence is None:
            reasons.append("unknown sentence_id")
        elif sentence.get("review_id") != review_id:
            reasons.append("sentence belongs to another review")

        if not isinstance(evidence_text, str) or not evidence_text.strip():
            reasons.append("evidence_text must be nonempty text")
        if not isinstance(observation, str) or not observation.strip():
            reasons.append("observation must be nonempty text")
        if target not in ALLOWED_TARGETS:
            reasons.append("invalid target")
        if polarity not in ALLOWED_POLARITIES:
            reasons.append("invalid polarity")
        if statement_type not in ALLOWED_STATEMENT_TYPES:
            reasons.append("invalid statement_type")

        semantic_text = " ".join(
            value for value in (evidence_text, observation)
            if isinstance(value, str)
        )
        evidence_groups = [
            name for name, pattern in ASPECT_CUE_GROUPS.items()
            if isinstance(evidence_text, str) and pattern.search(evidence_text)
        ]
        aspect_groups = sorted(
            name for name, pattern in ASPECT_CUE_GROUPS.items()
            if pattern.search(semantic_text)
        )
        enumeration_text = (
            re.sub(
                r"^\s*(?:plus|however|but|also|overall|that said),\s*",
                "",
                evidence_text,
                flags=re.I,
            )
            if isinstance(evidence_text, str) else ""
        )
        list_bundles_multiple_aspects = (
            len(evidence_groups) >= 2
            and (
                enumeration_text.count(",") >= 2
                or re.search(r"\bor\b", enumeration_text, re.I)
            )
        )
        if len(aspect_groups) >= 3 or list_bundles_multiple_aspects:
            reasons.append(
                "observation appears to bundle multiple product aspects: "
                + ", ".join(aspect_groups)
            )
        observation_groups = [
            name for name, pattern in ASPECT_CUE_GROUPS.items()
            if isinstance(observation, str) and pattern.search(observation)
        ]
        if (
            isinstance(evidence_text, str)
            and observation_groups
            and not set(observation_groups).intersection(evidence_groups)
        ):
            reasons.append(
                "evidence quote is not self-sufficient about the observation aspect"
            )
        grounded_evidence = evidence_text if isinstance(evidence_text, str) else ""
        if (
            statement_type == "actual_experience"
            and (
                INTENDED_USE_CUE.search(grounded_evidence)
                or EXPECTATION_ONLY_CUE.search(grounded_evidence)
            )
        ):
            reasons.append(
                "expectation/intention language cannot be actual_experience"
            )
        if (
            statement_type in {"capability_or_limitation", "feature_description"}
            and EXPECTATION_ONLY_CUE.search(grounded_evidence)
        ):
            reasons.append(
                "assumption/expectation evidence requires expectation_or_assumption"
            )
        if (
            statement_type == "intended_use"
            and not INTENDED_USE_CUE.search(grounded_evidence)
        ):
            reasons.append(
                "intended_use requires explicit reviewer intention or purpose evidence"
            )
        if (
            statement_type == "expectation_or_assumption"
            and not EXPECTATION_CUE.search(grounded_evidence)
        ):
            reasons.append(
                "expectation_or_assumption requires explicit expectation evidence"
            )
        if (
            statement_type == "service_experience"
            and not SERVICE_EXPERIENCE_CUE.search(grounded_evidence)
        ):
            reasons.append(
                "service_experience requires explicit service, warranty, or seller evidence"
            )
        if (
            statement_type == "service_experience"
            and target == "current_product"
            and SERVICE_CUE.search(semantic_text)
        ):
            reasons.append(
                "manufacturer/seller/warranty service target must be "
                "seller_platform or unclear"
            )

        sentence_text = sentence.get("sentence_text") if sentence is not None else None
        source_field = (
            sentence.get("source_field", "clean_text")
            if sentence is not None else None
        )
        local_start: int | None = None
        source_start: int | None = None
        source_end: int | None = None
        clean_start: int | None = None
        clean_end: int | None = None
        locator_adjustments: list[dict[str, str]] = []
        source_position = (
            sentence_order.get(str(sentence_id))
            if sentence_id is not None else None
        )
        before_id: str | None = None
        before_text: str | None = None
        after_id: str | None = None
        after_text: str | None = None
        if source_position is not None:
            before_id, before_text = context_at(source_position, -1)
            after_id, after_text = context_at(source_position, 1)
        if source_field not in ALLOWED_SOURCE_FIELDS:
            reasons.append("invalid source_field")
        if isinstance(sentence_text, str) and isinstance(evidence_text, str) and evidence_text:
            occurrences = sentence_text.count(evidence_text)
            if occurrences == 0:
                case_matches = list(re.finditer(
                    re.escape(evidence_text), sentence_text, flags=re.I
                ))
                if len(case_matches) == 1:
                    local_start, local_end = case_matches[0].span()
                    model_evidence = evidence_text
                    evidence_text = sentence_text[local_start:local_end]
                    locator_adjustments.append({
                        "type": "case_canonicalized_from_source",
                        "model_evidence_text": model_evidence,
                    })
                    occurrences = 1
                else:
                    reasons.append(
                        "evidence_text is not an exact case-sensitive sentence substring"
                    )
            if occurrences > 1:
                reasons.append("evidence_text occurs multiple times in cited sentence")
            elif occurrences == 1:
                if local_start is None:
                    local_start = sentence_text.find(evidence_text)
                sentence_start = sentence.get("start_char")
                if not isinstance(sentence_start, int) or sentence_start < 0:
                    reasons.append("sentence start_char is invalid")
                else:
                    source_start = sentence_start + local_start
                    source_end = source_start + len(evidence_text)
                    source_text = clean_text if source_field == "clean_text" else title
                    if (
                        source_start < 0
                        or source_end > len(source_text)
                        or source_text[source_start:source_end] != evidence_text
                    ):
                        reasons.append("review-level source offset mismatch")
                    elif source_field == "clean_text":
                        clean_start, clean_end = source_start, source_end

        if (
            target == "current_product"
            and isinstance(evidence_text, str)
            and re.match(r"^\s*(?:it|this|that)\b", evidence_text, re.I)
            and isinstance(before_text, str)
            and re.search(
                r"\b(?:previous|previously|older|old|former|other)\b",
                before_text,
                re.I,
            )
        ):
            reasons.append(
                "ambiguous pronoun after previous/other product context "
                "cannot be current_product"
            )

        exact_key = hash_json(item)
        if exact_key in seen_exact:
            reasons.append("duplicate observation")
        seen_exact.add(exact_key)

        if reasons:
            rejected.append({"index": index, "item": item, "errors": reasons})
            continue

        assert sentence is not None
        assert isinstance(sentence_text, str)
        assert isinstance(evidence_text, str)
        assert isinstance(observation, str)
        assert isinstance(local_start, int)
        assert isinstance(source_start, int)
        assert isinstance(source_end, int)
        assert isinstance(source_field, str)
        assert isinstance(statement_type, str)
        assert source_position is not None
        located.append(
            {
                "_source_index": index,
                "_sentence_order": source_position,
                "sentence_id": sentence_id,
                "source_field": source_field,
                "evidence_text": evidence_text,
                "evidence_start_char_sentence": local_start,
                "evidence_end_char_sentence": local_start + len(evidence_text),
                "evidence_start_char_source": source_start,
                "evidence_end_char_source": source_end,
                "evidence_start_char_clean": clean_start,
                "evidence_end_char_clean": clean_end,
                "sentence_text": sentence_text,
                "context_before_sentence_id": before_id,
                "context_before": before_text,
                "context_after_sentence_id": after_id,
                "context_after": after_text,
                "observation": observation.strip(),
                "target": target,
                "polarity": polarity,
                "statement_type": statement_type,
                "locator_adjustments": locator_adjustments,
            }
        )

    # Source order makes IDs stable even if the model returns the same valid
    # observations in a different array order.
    located.sort(
        key=lambda row: (
            row["_sentence_order"],
            row["evidence_start_char_sentence"],
            row["evidence_end_char_sentence"],
            row["observation"],
            row["target"],
            row["polarity"],
            row["_source_index"],
        )
    )
    accepted: list[dict[str, Any]] = []
    for number, row in enumerate(located, start=1):
        row = dict(row)
        row.pop("_source_index", None)
        row.pop("_sentence_order", None)
        row["observation_id"] = f"O{number:03d}"
        accepted.append(row)

    errors.extend(
        f"observation[{entry['index']}]: {reason}"
        for entry in rejected
        for reason in entry["errors"]
    )
    return {
        "accepted_observations": accepted,
        "rejected_observations": rejected,
        "validation_errors": errors,
        "reference_checks_passed": not errors,
    }


def _evidence_from_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "sentence_id": observation["sentence_id"],
        "source_field": observation["source_field"],
        "quote": observation["evidence_text"],
        "quote_start_char_sentence": observation["evidence_start_char_sentence"],
        "quote_end_char_sentence": observation["evidence_end_char_sentence"],
        "quote_start_char_source": observation["evidence_start_char_source"],
        "quote_end_char_source": observation["evidence_end_char_source"],
        "quote_start_char_clean": observation.get("evidence_start_char_clean"),
        "quote_end_char_clean": observation.get("evidence_end_char_clean"),
        "sentence_text": observation["sentence_text"],
    }


def _grounded_observation_text(observation: Mapping[str, Any]) -> str:
    # Taxonomy gates trust only the exact quote. The model-authored observation
    # is useful as the canonical readable claim after it passes aspect
    # alignment, but it may never supply a missing category cue.
    evidence = observation.get("evidence_text")
    return evidence.strip() if isinstance(evidence, str) else ""


def _normalization_boundary_errors(
    issue_group: str,
    observation: Mapping[str, Any],
) -> list[str]:
    """Return narrow, explainable reasons a mapped issue is unsafe."""
    statement_type = observation.get("statement_type")
    if statement_type in {"intended_use", "expectation_or_assumption"}:
        return [f"{statement_type} observations must be unmapped"]

    text = _grounded_observation_text(observation)
    errors: list[str] = []
    stability = re.search(
        r"\b(?:crash\w*|freez\w*|restart\w*|blue screen|shut\w* down)\b",
        text,
        re.I,
    )
    slow = re.search(
        r"\b(?:slow\w*|lag\w*|sluggish|unresponsive|wait\w*|delay\w*)\b|"
        r"\bs\s+l\s*o\s*w\b|takes? forever",
        text,
        re.I,
    )

    if issue_group == "slow_performance":
        if stability:
            errors.append("crash/freeze/restart evidence cannot map to slow_performance")
        if not slow:
            errors.append("slow_performance requires explicit slow/lag/wait evidence")
    elif issue_group == "freezing_crashing" and not stability:
        errors.append("freezing_crashing requires explicit crash/freeze/restart evidence")
    elif issue_group == "processing_capability" and not re.search(
        r"\b(?:gaming|game|tasks?|workload|processor|cpu|gpu|graphics|"
        r"perform\w*|adequate|capab\w*|powerhouse|editing|multitask\w*|"
        r"stream\w*|brows\w*|web|office|word|program\w*|video|photo|"
        r"open\w*|load\w*|navigat\w*|quick\w*|instant\w*|handl\w*|"
        r"fast\w*|responsive)\b",
        text,
        re.I,
    ):
        errors.append("processing_capability requires explicit task or compute evidence")
    elif issue_group == "weight" and not re.search(
        r"\b(?:weight|lightweight|light weight|heavy|portable|portability|carry)\w*\b",
        text,
        re.I,
    ):
        errors.append("weight requires explicit weight or portability evidence")
    elif issue_group == "appearance" and not re.search(
        r"\b(?:appearance|design|sleek|thin|color)\w*\b|"
        r"\blooks?\s+(?:great|good|nice|beautiful|modern|attractive)\b",
        text,
        re.I,
    ):
        errors.append("appearance requires explicit design or appearance evidence")
    elif issue_group == "storage_performance":
        has_storage = re.search(r"\b(?:storage|ssd|hard drive|disk|drive)\b", text, re.I)
        has_speed = re.search(
            r"\b(?:speed|fast|slow|performance|read|write)\w*\b", text, re.I
        )
        if not has_storage or not has_speed:
            errors.append("storage_performance requires storage plus speed evidence")
    elif issue_group == "storage_capacity":
        expansion = re.search(
            r"\b(?:upgrad\w*|add\w*|slot|replac\w*|expand\w*)\b",
            text,
            re.I,
        )
        capacity = re.search(
            r"\b(?:capacity|space|room|full|storage (?:amount|size)|"
            r"\d+\s*(?:gb|tb)|gigabytes?|terabytes?)\b",
            text,
            re.I,
        )
        if expansion:
            errors.append(
                "RAM/storage slot, addition, or replacement belongs to upgradeability"
            )
        elif not capacity:
            errors.append(
                "storage_capacity requires explicit capacity, space, GB, or TB evidence"
            )
    elif issue_group == "upgradeability":
        has_upgrade = re.search(
            r"\b(?:upgrad\w*|add\w*|slot|replac\w*|expand\w*)\b", text, re.I
        )
        has_memory_storage = re.search(
            r"\b(?:ram|memory|storage|ssd|hard drive|disk)\b", text, re.I
        )
        if not has_upgrade or not has_memory_storage:
            errors.append("upgradeability requires RAM/storage expansion evidence")
    elif issue_group == "charging" and not _charging_behavior_supported(
        text,
        observation.get("polarity"),
    ):
        errors.append(
            "charging requires explicit charging hardware plus behavior or failure"
        )
    elif issue_group == "battery_runtime" and not re.search(
        r"\b(?:battery|runtime|hours?|recharg\w*|per charge)\b",
        text,
        re.I,
    ):
        errors.append("battery_runtime requires explicit battery-duration evidence")
    elif issue_group == "speaker_sound" and not re.search(
        r"\b(?:speaker|sound|audio)\w*\b",
        text,
        re.I,
    ):
        errors.append("speaker_sound requires explicit speaker or sound evidence")
    elif issue_group == "low_volume" and not re.search(
        r"\b(?:volume|loud|quiet)\w*\b",
        text,
        re.I,
    ):
        errors.append("low_volume requires explicit volume evidence")
    elif issue_group == "audio_connectivity" and not re.search(
        r"\b(?:headphone|microphone|audio (?:jack|port)|headset)\w*\b",
        text,
        re.I,
    ):
        errors.append("audio_connectivity requires explicit audio-port evidence")
    elif issue_group == "keyboard" and not re.search(
        r"\b(?:keyboard|keys|backlit|backlight)\b",
        text,
        re.I,
    ):
        errors.append("keyboard requires explicit keyboard or key evidence")
    elif issue_group == "touchpad_mouse" and not re.search(
        r"\b(?:touchpad|trackpad|mouse|cursor|pointer)\w*\b",
        text,
        re.I,
    ):
        errors.append("touchpad_mouse requires explicit pointing-device evidence")
    elif issue_group == "usb_other_ports":
        if re.search(
            r"\b(?:power port|charging port|charger|power adapter|ac adapter)\b",
            text,
            re.I,
        ):
            errors.append("power/charging hardware belongs to battery/charging")
        elif not re.search(
            r"\b(?:usb|hdmi|sd[- ]?card|port)\w*\b",
            text,
            re.I,
        ):
            errors.append("usb_other_ports requires explicit physical-port evidence")
    elif issue_group == "price" and not re.search(
        r"(?:\$|\b(?:price|cost|dollars?)\b)",
        text,
        re.I,
    ):
        errors.append("price requires explicit price or cost evidence")
    elif issue_group == "value_for_money" and not re.search(
        r"\b(?:price|value|cost|worth|money|budget|deal|hard to beat|"
        r"inexpensive|affordable)\b",
        text,
        re.I,
    ):
        errors.append("value_for_money requires explicit price or value evidence")
    elif issue_group == "screen_quality" and not re.search(
        r"\b(?:screen|display|monitor|picture|resolution|brightness|color|clear)\w*\b",
        text,
        re.I,
    ):
        errors.append("screen_quality requires explicit visual-display evidence")
    elif issue_group == "app_compatibility" and not _app_limitation_supported(text):
        errors.append(
            "app_compatibility requires an explicit blocked, unavailable, or "
            "incompatible app action"
        )
    elif issue_group in {"support_service", "return_warranty"}:
        if not re.search(
            r"\b(?:support|service|warranty|repair|return|seller|amazon|"
            r"refund|turn\s*around)\b",
            text,
            re.I,
        ):
            errors.append(
                f"{issue_group} requires explicit support, warranty, or return evidence"
            )
        if re.search(r"\b(?:friend|heard|someone else|they told me)\b", text, re.I):
            errors.append("secondhand service anecdotes must remain unmapped")
    return errors


def _high_precision_recovery_candidates(
    observation: Mapping[str, Any],
    taxonomy: Mapping[str, Mapping[str, str]],
) -> dict[tuple[str, str], list[str]]:
    """Find uniquely supported taxonomy pairs for an abstained observation.

    The rules cover only explicit mechanisms that the normalizer can safely
    under-map. They use the exact quote, never the model-authored observation
    or adjacent context, and still pass through the existing boundary gate.
    """
    if observation.get("target") != "current_product":
        return {}
    if observation.get("statement_type") in {
        "intended_use",
        "expectation_or_assumption",
        "service_experience",
        "unclear",
    }:
        return {}

    text = _grounded_observation_text(observation)
    polarity = observation.get("polarity")
    candidates: dict[tuple[str, str], list[str]] = {}

    def add(
        rule_id: str,
        category: str,
        issue_group: str,
        *,
        allowed_polarities: set[str],
    ) -> None:
        if polarity not in allowed_polarities:
            return
        if category not in taxonomy or issue_group not in taxonomy[category]:
            return
        if _normalization_boundary_errors(issue_group, observation):
            return
        candidates.setdefault((category, issue_group), []).append(rule_id)

    if _app_limitation_supported(text):
        add(
            "explicit_app_block_with_action",
            "software_os",
            "app_compatibility",
            allowed_polarities={"negative"},
        )

    if _power_failure_supported(text):
        add(
            "explicit_power_hardware_failure",
            "battery",
            "charging",
            allowed_polarities={"negative"},
        )

    positive_value = _has_unnegated_match(
        r"\b(?:best\s+deal|(?:good|great|excellent|amazing)\s+"
        r"(?:deal|value)|worth\s+(?:the\s+)?(?:money|cost))\b",
        text,
    )
    negative_value = (
        _has_unnegated_match(
            r"\b(?:waste\s+of\s+money|(?:poor|bad|terrible|worst)\s+"
            r"(?:deal|value))\b",
            text,
        )
        or re.search(
            r"\bnot\s+worth\s+(?:the\s+)?(?:money|cost)\b",
            text,
            re.I,
        )
    )
    if positive_value:
        add(
            "explicit_positive_value_judgment",
            "price_value",
            "value_for_money",
            allowed_polarities={"positive"},
        )
    if negative_value:
        add(
            "explicit_negative_value_judgment",
            "price_value",
            "value_for_money",
            allowed_polarities={"negative"},
        )

    if _has_unnegated_match(
        r"\b(?:best|lowest|cheapest)\s+price\b",
        text,
    ):
        add(
            "explicit_positive_price_comparison",
            "price_value",
            "price",
            allowed_polarities={"positive"},
        )

    return candidates


def validate_claims(
    data: Mapping[str, Any],
    observations: Sequence[Mapping[str, Any]],
    taxonomy: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    """Constrain normalizations to validated upstream observations."""
    candidates, errors = _top_level_list(data, "claims")
    observation_lookup = {
        observation.get("observation_id"): observation
        for observation in observations
        if isinstance(observation.get("observation_id"), str)
    }
    if len(observation_lookup) != len(observations):
        raise ValueError("accepted observations have missing or duplicate IDs")
    issue_group_parents: dict[str, list[str]] = {}
    for parent_category, groups in taxonomy.items():
        for child_group in groups:
            issue_group_parents.setdefault(child_group, []).append(parent_category)

    accepted_by_id: dict[str, dict[str, Any]] = {}
    rejected: list[dict[str, Any]] = []
    referenced_ids: set[str] = set()

    for index, item in enumerate(candidates):
        reasons: list[str] = []
        if not isinstance(item, dict):
            rejected.append({"index": index, "item": item, "errors": ["not an object"]})
            continue
        if set(item) != CLAIM_MODEL_KEYS:
            reasons.append(f"wrong fields: {sorted(set(item) ^ CLAIM_MODEL_KEYS)}")

        observation_id = item.get("observation_id")
        mapping_status = item.get("mapping_status")
        category = item.get("category")
        issue_group = item.get("issue_group")
        claim = item.get("claim")
        normalization_adjustments: list[dict[str, Any]] = []

        observation = (
            observation_lookup.get(observation_id)
            if isinstance(observation_id, str)
            else None
        )
        if observation is None:
            reasons.append("unknown observation_id")
        else:
            if observation_id in referenced_ids:
                reasons.append("duplicate normalization for observation_id")
            referenced_ids.add(observation_id)

        if mapping_status not in ALLOWED_MAPPING_STATUSES:
            reasons.append("invalid mapping_status")
        elif mapping_status == "mapped":
            grounded_text = (
                _grounded_observation_text(observation)
                if observation is not None else ""
            )
            if (
                category == "connectivity_ports"
                and issue_group == "usb_other_ports"
                and "battery" in taxonomy
                and "charging" in taxonomy["battery"]
                and _power_failure_supported(grounded_text)
            ):
                normalization_adjustments.append({
                    "type": "canonicalized_power_port_to_charging",
                    "from_category": category,
                    "from_issue_group": issue_group,
                    "to_category": "battery",
                    "to_issue_group": "charging",
                })
                category = "battery"
                issue_group = "charging"
            parents = (
                issue_group_parents.get(issue_group, [])
                if isinstance(issue_group, str) else []
            )
            if (
                issue_group in SAFE_PARENT_CANONICALIZATION_GROUPS
                and len(parents) == 1
                and category != parents[0]
            ):
                original_category = category
                category = parents[0]
                normalization_adjustments.append({
                    "type": "canonicalized_unique_issue_group_parent",
                    "from_category": original_category,
                    "to_category": category,
                    "issue_group": issue_group,
                })
            if not isinstance(category, str) or category not in taxonomy:
                reasons.append("invalid category")
            elif (
                not isinstance(issue_group, str)
                or issue_group not in taxonomy[category]
            ):
                reasons.append("invalid category/issue_group pair")
            elif observation is not None:
                reasons.extend(
                    _normalization_boundary_errors(issue_group, observation)
                )
        elif category is not None or issue_group is not None:
            reasons.append("unmapped claims require null category and issue_group")

        if not isinstance(claim, str) or not claim.strip():
            reasons.append("claim must be nonempty text")

        if reasons:
            rejected.append({"index": index, "item": item, "errors": reasons})
            continue

        assert observation is not None
        assert isinstance(observation_id, str)
        assert isinstance(claim, str)
        canonical_claim = observation.get("observation")
        if not isinstance(canonical_claim, str) or not canonical_claim.strip():
            raise ValueError("validated observation is missing observation text")
        model_claim = claim.strip()
        canonical_claim = canonical_claim.strip()
        if model_claim != canonical_claim:
            normalization_adjustments.append({
                "type": "canonicalized_claim_from_validated_observation",
                "model_claim": model_claim,
            })
        taxonomy_adjusted = any(
            adjustment.get("type") in {
                "canonicalized_unique_issue_group_parent",
                "canonicalized_power_port_to_charging",
            }
            for adjustment in normalization_adjustments
        )
        accepted_by_id[observation_id] = {
            "observation_id": observation_id,
            "mapping_status": mapping_status,
            "category": category,
            "issue_group": issue_group,
            "claim": canonical_claim,
            "model_claim": model_claim,
            # These fields are inherited, never regenerated by the normalizer.
            "sentiment": observation["polarity"],
            "attribution": observation["target"],
            "statement_type": observation["statement_type"],
            "evidence": [_evidence_from_observation(observation)],
            "deterministic_status": (
                "valid_mapped_canonicalized"
                if mapping_status == "mapped" and taxonomy_adjusted
                else "valid_mapped"
                if mapping_status == "mapped"
                else "valid_unmapped"
            ),
            "normalization_source": (
                "model_with_deterministic_adjustments"
                if normalization_adjustments else "model"
            ),
            "normalization_adjustments": normalization_adjustments,
        }

    ordered_ids = [str(observation["observation_id"]) for observation in observations]
    missing_ids = [observation_id for observation_id in ordered_ids if observation_id not in accepted_by_id]
    for observation_id in missing_ids:
        errors.append(f"missing valid normalization for {observation_id}")
    errors.extend(
        f"claim[{entry['index']}]: {reason}"
        for entry in rejected
        for reason in entry["errors"]
    )

    accepted: list[dict[str, Any]] = []
    for number, observation_id in enumerate(ordered_ids, start=1):
        claim_row = accepted_by_id.get(observation_id)
        if claim_row is None:
            continue
        claim_row = dict(claim_row)
        claim_row["claim_id"] = f"C{number:03d}"
        accepted.append(claim_row)

    return {
        "accepted_claims": accepted,
        "rejected_claims": rejected,
        "missing_observation_ids": missing_ids,
        "validation_errors": errors,
        "normalization_complete": not errors and len(accepted) == len(observations),
    }


def complete_claims_with_unmapped_fallbacks(
    gate: Mapping[str, Any],
    observations: Sequence[Mapping[str, Any]],
    taxonomy: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    """Preserve every observation and recover only a unique explicit mapping.

    Rejected model candidates remain available for audit. Only observations
    without a valid accepted decision receive a deterministic unmapped claim.
    Final abstentions may be mapped only when the exact evidence triggers one
    distinct high-precision taxonomy pair and the normal boundary gate passes.
    """
    accepted_by_id = {
        claim.get("observation_id"): dict(claim)
        for claim in gate.get("accepted_claims", []) or []
        if isinstance(claim, Mapping)
        and isinstance(claim.get("observation_id"), str)
    }
    completed: list[dict[str, Any]] = []
    recovery_count = 0
    for number, observation in enumerate(observations, start=1):
        observation_id = observation.get("observation_id")
        if not isinstance(observation_id, str):
            raise ValueError("validated observation is missing observation_id")
        claim = accepted_by_id.get(observation_id)
        if claim is None:
            observation_text = observation.get("observation")
            if not isinstance(observation_text, str) or not observation_text.strip():
                raise ValueError("validated observation is missing observation text")
            claim = {
                "observation_id": observation_id,
                "mapping_status": "unmapped",
                "category": None,
                "issue_group": None,
                "claim": observation_text.strip(),
                "sentiment": observation["polarity"],
                "attribution": observation["target"],
                "statement_type": observation["statement_type"],
                "evidence": [_evidence_from_observation(observation)],
                "deterministic_status": (
                    "fallback_unmapped_after_invalid_normalization"
                ),
                "normalization_source": "deterministic_unmapped_fallback",
                "normalization_adjustments": [],
            }
        if claim.get("mapping_status") == "unmapped":
            candidates = _high_precision_recovery_candidates(
                observation,
                taxonomy,
            )
            if len(candidates) == 1:
                (category, issue_group), rule_ids = next(iter(candidates.items()))
                prior_source = claim.get("normalization_source")
                adjustments = list(claim.get("normalization_adjustments", []) or [])
                adjustments.append({
                    "type": "recovered_unique_supported_taxonomy_pair",
                    "rule_ids": sorted(rule_ids),
                    "from_mapping_status": "unmapped",
                    "from_normalization_source": prior_source,
                    "to_category": category,
                    "to_issue_group": issue_group,
                })
                claim.update({
                    "mapping_status": "mapped",
                    "category": category,
                    "issue_group": issue_group,
                    "deterministic_status": (
                        "valid_mapped_by_high_precision_recovery"
                    ),
                    "normalization_source": (
                        "deterministic_high_precision_recovery"
                    ),
                    "normalization_adjustments": adjustments,
                })
                recovery_count += 1
        claim["claim_id"] = f"C{number:03d}"
        completed.append(claim)

    model_errors = list(gate.get("validation_errors", []) or [])
    model_missing = list(gate.get("missing_observation_ids", []) or [])
    result = dict(gate)
    result.update({
        "accepted_claims": completed,
        "model_normalization_complete": bool(
            gate.get("normalization_complete", False)
        ),
        "model_validation_errors": model_errors,
        "model_missing_observation_ids": model_missing,
        "fallback_unmapped_count": sum(
            claim.get("mapping_status") == "unmapped"
            and claim.get("normalization_source")
            == "deterministic_unmapped_fallback"
            for claim in completed
        ),
        "deterministic_recovery_count": recovery_count,
        "missing_observation_ids": [],
        "validation_errors": [],
        "normalization_complete": len(completed) == len(observations),
    })
    return result


def merge_claims_by_tuple(
    claims: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Merge mapped atomic claims into review-level business findings.

    Atomic claims stay available for precision evaluation. This derived view
    prevents evidence splitting from inflating review-level issue counts.
    """
    grouped: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    order: list[tuple[str, str, str, str]] = []
    for claim in claims:
        if claim.get("mapping_status") != "mapped":
            continue
        values = (
            claim.get("category"),
            claim.get("issue_group"),
            claim.get("sentiment"),
            claim.get("attribution"),
        )
        if not all(isinstance(value, str) and value for value in values):
            raise ValueError("mapped claim is missing its business tuple")
        key = values  # type: ignore[assignment]
        if key not in grouped:
            grouped[key] = {
                "mapping_status": "mapped",
                "category": key[0],
                "issue_group": key[1],
                "sentiment": key[2],
                "attribution": key[3],
                "source_claim_ids": [],
                "observation_ids": [],
                "statement_types": [],
                "component_claims": [],
                "evidence": [],
            }
            order.append(key)
        finding = grouped[key]
        claim_id = claim.get("claim_id")
        observation_id = claim.get("observation_id")
        statement_type = claim.get("statement_type")
        claim_text = claim.get("claim")
        if isinstance(claim_id, str) and claim_id not in finding["source_claim_ids"]:
            finding["source_claim_ids"].append(claim_id)
        if (
            isinstance(observation_id, str)
            and observation_id not in finding["observation_ids"]
        ):
            finding["observation_ids"].append(observation_id)
        if (
            isinstance(statement_type, str)
            and statement_type not in finding["statement_types"]
        ):
            finding["statement_types"].append(statement_type)
        if isinstance(claim_text, str) and claim_text not in finding["component_claims"]:
            finding["component_claims"].append(claim_text)
        existing_evidence = {hash_json(value) for value in finding["evidence"]}
        for evidence in claim.get("evidence", []) or []:
            if not isinstance(evidence, Mapping):
                continue
            evidence_copy = dict(evidence)
            digest = hash_json(evidence_copy)
            if digest not in existing_evidence:
                finding["evidence"].append(evidence_copy)
                existing_evidence.add(digest)

    findings: list[dict[str, Any]] = []
    for number, key in enumerate(order, start=1):
        finding = dict(grouped[key])
        finding["business_finding_id"] = f"B{number:03d}"
        finding["deterministic_status"] = "merged_mapped_business_finding"
        findings.append(finding)
    return findings


def stage_record_hash(row: Mapping[str, Any], accepted_key: str) -> str:
    """Fingerprint only the deterministic accepted result used downstream."""
    return hash_json(
        {
            "review_id": row.get("review_id"),
            "schema_version": row.get("schema_version"),
            accepted_key: row.get(accepted_key, []),
        }
    )


def load_stage_cache(
    path: Path,
    *,
    schema_version: str,
    model_id: str,
    prompt_hash: str,
    taxonomy_digest: str,
    input_hashes: Mapping[str, str],
    upstream_hashes: Mapping[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Load a resumable stage cache and reject any stale or mixed row."""
    if not path.exists():
        return {}
    saved: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        review_id = row.get("review_id")
        if not isinstance(review_id, str) or review_id in saved:
            raise ValueError(f"duplicate or missing cached review_id in {path}: {review_id}")
        expected_input = input_hashes.get(review_id)
        if expected_input is None:
            raise ValueError(f"unknown cached review in {path}: {review_id}")
        expected = {
            "schema_version": schema_version,
            "model_id": model_id,
            "prompt_hash": prompt_hash,
            "taxonomy_hash": taxonomy_digest,
            "input_hash": expected_input,
        }
        mismatches = [key for key, value in expected.items() if row.get(key) != value]
        if upstream_hashes is not None:
            expected_upstream = upstream_hashes.get(review_id)
            if expected_upstream is None or row.get("upstream_hash") != expected_upstream:
                mismatches.append("upstream_hash")
        if mismatches:
            raise ValueError(
                f"stale Evidence-first cache row {review_id} in {path}; "
                f"mismatched {sorted(set(mismatches))}. Use --force to back it up."
            )
        saved[review_id] = row
    return saved
