"""Fair, human-first evaluation for frozen Evidence-first cohorts.

This module deliberately writes new artifacts.  The legacy ``ef-summary``
files remain untouched for backward compatibility.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .core import ALLOWED_POLARITIES, ALLOWED_TARGETS


PRIMARY_HUMAN_COLUMNS = (
    "human_supported",
    "human_category_correct",
    "human_sentiment_correct",
    "human_target_correct",
    "notes",
)
ATOMIC_HUMAN_COLUMNS = (
    "human_supported",
    "human_atomic",
    "human_category_correct",
    "human_sentiment_correct",
    "human_target_correct",
    "notes",
)
ABSTENTION_HUMAN_COLUMNS = (
    "human_should_map",
    "human_expected_category",
    "human_expected_issue_group",
    "human_statement_type_correct",
    "notes",
)
GOLD_HUMAN_COLUMNS = (
    "human_gold_findings_json",
    "human_gold_complete",
    "notes",
)

PRIMARY_FIELDS = [
    "blind_id", "review_id", "review_title", "review_text", "category",
    "issue_group", "sentiment", "attribution", "finding", "evidence_quotes",
    *PRIMARY_HUMAN_COLUMNS,
]
ATOMIC_FIELDS = [
    "audit_id", "review_id", "review_title", "review_text", "category",
    "issue_group", "sentiment", "attribution", "claim", "evidence_quotes",
    *ATOMIC_HUMAN_COLUMNS,
]
ABSTENTION_FIELDS = [
    "abstention_id", "review_id", "review_title", "review_text",
    "statement_type", "observation", "evidence_quotes",
    *ABSTENTION_HUMAN_COLUMNS,
]
GOLD_FIELDS = [
    "review_id", "rating", "review_title", "review_text",
    *GOLD_HUMAN_COLUMNS,
]


def _read_jsonl(path: Path, required: bool = True) -> list[dict[str, Any]]:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def _unique_review_index(
    rows: Sequence[Mapping[str, Any]],
    label: str,
) -> dict[str, Mapping[str, Any]]:
    index: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        review_id = row.get("review_id")
        if not isinstance(review_id, str) or not review_id or review_id in index:
            raise ValueError(f"Duplicate or missing review_id in {label}: {review_id}")
        index[review_id] = row
    return index


def _stable_id(prefix: str, *parts: Any) -> str:
    payload = json.dumps(
        parts, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return prefix + hashlib.sha256(payload).hexdigest()[:16].upper()


def _quote(evidence: Mapping[str, Any]) -> str:
    for key in ("evidence_text", "quote", "text"):
        value = evidence.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _evidence_list(item: Mapping[str, Any]) -> list[dict[str, Any]]:
    evidence = item.get("evidence", []) or []
    if isinstance(evidence, dict):
        evidence = [evidence]
    return [value for value in evidence if isinstance(value, dict)]


def _evidence_quotes(item: Mapping[str, Any]) -> str:
    quotes: list[str] = []
    for evidence in _evidence_list(item):
        quote = _quote(evidence)
        if quote and quote not in quotes:
            quotes.append(quote)
    return " || ".join(quotes)


def _signature(item: Mapping[str, Any]) -> tuple[str, str, str, str] | None:
    if item.get("mapping_status", "mapped") != "mapped":
        return None
    values = (
        item.get("category"),
        item.get("issue_group"),
        item.get("sentiment") or item.get("polarity"),
        item.get("attribution") or item.get("target"),
    )
    if not all(isinstance(value, str) and value for value in values):
        return None
    return values  # type: ignore[return-value]


def _finding_text(item: Mapping[str, Any]) -> str:
    for key in ("summary", "claim"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    components = item.get("component_claims", []) or []
    if isinstance(components, list):
        clean = [str(value).strip() for value in components if str(value).strip()]
        if clean:
            return " || ".join(clean)
    return ""


def _human_boolean(value: Any) -> bool | None:
    normalized = str(value or "").strip().casefold()
    if normalized in {"yes", "y", "true", "1"}:
        return True
    if normalized in {"no", "n", "false", "0"}:
        return False
    return None


def _read_preserved(
    path: Path,
    key_column: str,
    human_columns: Iterable[str],
) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if key_column not in (reader.fieldnames or []):
            raise ValueError(f"Existing evaluation sheet lacks {key_column}: {path}")
        return {
            str(row[key_column]): {
                column: str(row.get(column, "")) for column in human_columns
            }
            for row in reader
            if row.get(key_column)
        }


def _render_csv(rows: Sequence[Mapping[str, Any]], fieldnames: list[str]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return "\ufeff" + buffer.getvalue()


def _write_csv_preserving(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
    key_column: str,
    human_columns: Iterable[str],
) -> None:
    preserved = _read_preserved(path, key_column, human_columns)
    for row in rows:
        for column, value in preserved.get(str(row[key_column]), {}).items():
            row[column] = value
        for column in human_columns:
            row.setdefault(column, "")
    content = _render_csv(rows, fieldnames)
    encoded = content.encode("utf-8")
    # Compare raw bytes so universal-newline decoding does not turn CRLF into
    # LF and create a pointless backup on every unchanged rerun.
    if path.exists() and path.read_bytes() == encoded:
        return
    if path.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup = path.with_name(f"{path.stem}.backup_{stamp}{path.suffix}")
        shutil.copy2(path, backup)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)


def _write_json_if_changed(path: Path, value: Mapping[str, Any]) -> None:
    content = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _source_index(
    sentences: Sequence[Mapping[str, Any]],
    reviews: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    sources = {
        str(row["sentence_id"]): dict(row, source_field="clean_text")
        for row in sentences
    }
    for review in reviews:
        review_id = review.get("review_id")
        title = review.get("title")
        if not isinstance(review_id, str) or not isinstance(title, str) or not title.strip():
            continue
        sentence_id = f"{review_id}_TITLE"
        if sentence_id in sources:
            raise ValueError(f"Synthetic title ID collision: {sentence_id}")
        sources[sentence_id] = {
            "sentence_id": sentence_id,
            "review_id": review_id,
            "sentence_text": title,
            "source_field": "title",
        }
    return sources


def _locator_valid(
    review_id: str,
    item: Mapping[str, Any],
    source_by_id: Mapping[str, Mapping[str, Any]],
) -> bool:
    evidence = _evidence_list(item)
    if not evidence:
        return False
    for value in evidence:
        source = source_by_id.get(str(value.get("sentence_id")))
        quote = _quote(value)
        if (
            source is None
            or source.get("review_id") != review_id
            or not quote
            or quote not in str(source.get("sentence_text", ""))
        ):
            return False
    return True


def _cohort_paths(root: Path, cohort: str) -> dict[str, Path]:
    ef_dir = root / "outputs/amazon/evidence_first"
    if cohort == "pilot":
        return {
            "observations": ef_dir / "observations_pilot.jsonl",
            "claims": ef_dir / "claims_pilot.jsonl",
            "selection": ef_dir / "pilot_manifest.json",
            "output_dir": ef_dir,
        }
    if cohort == "holdout":
        holdout = ef_dir / "holdout"
        return {
            "observations": holdout / "observations_holdout.jsonl",
            "claims": holdout / "claims_holdout.jsonl",
            "selection": holdout / "selection_manifest.json",
            "output_dir": holdout / "evaluation",
        }
    raise ValueError(f"Unknown evaluation cohort: {cohort}")


def _build_primary_rows(
    selected_ids: Sequence[str],
    baseline: Mapping[str, Mapping[str, Any]],
    claims: Mapping[str, Mapping[str, Any]],
    reviews: Mapping[str, Mapping[str, Any]],
    source_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    key: dict[str, dict[str, Any]] = {}
    for review_id in selected_ids:
        system_items = (
            ("v3", baseline[review_id].get("accepted_feedback", []) or []),
            ("evidence_first", claims[review_id].get("merged_claims", []) or []),
        )
        for system, items in system_items:
            for item in items:
                if not isinstance(item, dict):
                    continue
                signature = _signature(item)
                if signature is None:
                    continue
                finding = _finding_text(item)
                quotes = _evidence_quotes(item)
                item_id = (
                    item.get("business_finding_id")
                    or item.get("feedback_id")
                    or _stable_id("ITEM_", review_id, signature, finding, quotes)
                )
                blind_id = _stable_id(
                    "BLIND_", system, review_id, signature, finding, quotes
                )
                if blind_id in key:
                    raise ValueError(
                        f"Duplicate primary evaluation item in {review_id}: {blind_id}"
                    )
                review = reviews[review_id]
                rows.append({
                    "blind_id": blind_id,
                    "review_id": review_id,
                    "review_title": review.get("title", ""),
                    "review_text": review.get("clean_text", ""),
                    "category": signature[0],
                    "issue_group": signature[1],
                    "sentiment": signature[2],
                    "attribution": signature[3],
                    "finding": finding,
                    "evidence_quotes": quotes,
                })
                key[blind_id] = {
                    "system": system,
                    "review_id": review_id,
                    "item_id": str(item_id),
                    "locator_valid": _locator_valid(
                        review_id, item, source_by_id
                    ),
                }
    rows.sort(key=lambda row: str(row["blind_id"]))
    return rows, key


def _build_diagnostic_rows(
    selected_ids: Sequence[str],
    claims: Mapping[str, Mapping[str, Any]],
    observations: Mapping[str, Mapping[str, Any]],
    reviews: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    atomic_rows: list[dict[str, Any]] = []
    abstention_rows: list[dict[str, Any]] = []
    for review_id in selected_ids:
        review = reviews[review_id]
        observation_by_id = {
            str(item.get("observation_id")): item
            for item in observations[review_id].get("accepted_observations", []) or []
            if isinstance(item, dict)
        }
        for item in claims[review_id].get("accepted_claims", []) or []:
            if not isinstance(item, dict):
                continue
            claim_id = item.get("claim_id") or _stable_id(
                "CLAIM_", review_id, item
            )
            common = {
                "review_id": review_id,
                "review_title": review.get("title", ""),
                "review_text": review.get("clean_text", ""),
                "evidence_quotes": _evidence_quotes(item),
            }
            if item.get("mapping_status") == "mapped":
                atomic_rows.append({
                    "audit_id": _stable_id("ATOMIC_", review_id, claim_id),
                    **common,
                    "category": item.get("category", ""),
                    "issue_group": item.get("issue_group", ""),
                    "sentiment": item.get("sentiment", ""),
                    "attribution": item.get("attribution", ""),
                    "claim": item.get("claim", ""),
                })
            elif item.get("mapping_status") == "unmapped":
                observation = observation_by_id.get(str(item.get("observation_id")), {})
                abstention_rows.append({
                    "abstention_id": _stable_id(
                        "ABSTAIN_", review_id, claim_id
                    ),
                    **common,
                    "statement_type": (
                        observation.get("statement_type")
                        or item.get("statement_type", "")
                    ),
                    "observation": (
                        observation.get("observation") or item.get("claim", "")
                    ),
                })
    atomic_rows.sort(key=lambda row: str(row["audit_id"]))
    abstention_rows.sort(key=lambda row: str(row["abstention_id"]))
    return atomic_rows, abstention_rows


def _component_metrics(
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
) -> dict[str, Any]:
    components: dict[str, Any] = {}
    for column in columns:
        labels = [
            label
            for row in rows
            if (label := _human_boolean(row.get(column))) is not None
        ]
        yes = sum(labels)
        components[column.removeprefix("human_")] = {
            "rows": len(rows),
            "labeled": len(labels),
            "blank_or_invalid": len(rows) - len(labels),
            "yes": yes,
            "no": len(labels) - yes,
            "rate": round(yes / len(labels), 6) if labels else None,
        }
    fully_correct: list[bool] = []
    for row in rows:
        labels = [_human_boolean(row.get(column)) for column in columns]
        if all(label is not None for label in labels):
            fully_correct.append(all(labels))
    return {
        "rows": len(rows),
        "components": components,
        "fully_correct": {
            "labeled": len(fully_correct),
            "blank_or_partial": len(rows) - len(fully_correct),
            "yes": sum(fully_correct),
            "rate": (
                round(sum(fully_correct) / len(fully_correct), 6)
                if fully_correct else None
            ),
        },
    }


def _primary_metrics(
    rows: Sequence[Mapping[str, Any]],
    blind_key: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for system in ("v3", "evidence_first"):
        system_rows = [
            row for row in rows
            if blind_key.get(str(row.get("blind_id")), {}).get("system") == system
        ]
        result[system] = _component_metrics(
            system_rows,
            PRIMARY_HUMAN_COLUMNS[:-1],
        )
    return result


def _abstention_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    should_map = [
        label
        for row in rows
        if (label := _human_boolean(row.get("human_should_map"))) is not None
    ]
    statement_type = [
        label
        for row in rows
        if (
            label := _human_boolean(row.get("human_statement_type_correct"))
        ) is not None
    ]
    correct_abstentions = sum(not label for label in should_map)
    expected_pair_labeled = sum(
        bool(str(row.get("human_expected_category", "")).strip())
        and bool(str(row.get("human_expected_issue_group", "")).strip())
        for row in rows
    )
    return {
        "rows": len(rows),
        "should_map": {
            "labeled": len(should_map),
            "blank_or_invalid": len(rows) - len(should_map),
            "yes": sum(should_map),
            "rate": (
                round(sum(should_map) / len(should_map), 6)
                if should_map else None
            ),
        },
        "correct_abstention": {
            "labeled": len(should_map),
            "yes": correct_abstentions,
            "rate": (
                round(correct_abstentions / len(should_map), 6)
                if should_map else None
            ),
        },
        "expected_taxonomy_pair_labeled": expected_pair_labeled,
        "statement_type_correct": {
            "labeled": len(statement_type),
            "blank_or_invalid": len(rows) - len(statement_type),
            "yes": sum(statement_type),
            "rate": (
                round(sum(statement_type) / len(statement_type), 6)
                if statement_type else None
            ),
        },
    }


def _parse_gold(
    rows: Sequence[Mapping[str, Any]],
    taxonomy: Mapping[str, Mapping[str, str]],
) -> tuple[dict[str, list[dict[str, str]]], dict[str, Any]]:
    complete: dict[str, list[dict[str, str]]] = {}
    incomplete = 0
    explicit_no = 0
    for row in rows:
        review_id = str(row.get("review_id", ""))
        complete_label = _human_boolean(row.get("human_gold_complete"))
        if complete_label is not True:
            incomplete += 1
            explicit_no += int(complete_label is False)
            continue
        raw = str(row.get("human_gold_findings_json", "")).strip()
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError(f"Invalid complete gold JSON for {review_id}: {exc}") from exc
        if not isinstance(parsed, list):
            raise ValueError(f"Invalid complete gold schema for {review_id}: expected list")
        clean: list[dict[str, str]] = []
        required = {
            "category", "issue_group", "sentiment", "attribution", "evidence_quote"
        }
        review_text = str(row.get("review_text", ""))
        review_title = str(row.get("review_title", ""))
        for index, item in enumerate(parsed, 1):
            if not isinstance(item, dict) or set(item) != required:
                raise ValueError(
                    f"Invalid complete gold item for {review_id} item {index}: "
                    f"expected exactly {sorted(required)}"
                )
            if not all(isinstance(item[key], str) and item[key].strip() for key in required):
                raise ValueError(
                    f"Invalid complete gold item for {review_id} item {index}: "
                    "all fields must be non-empty strings"
                )
            category = item["category"]
            issue_group = item["issue_group"]
            if category not in taxonomy or issue_group not in taxonomy[category]:
                raise ValueError(
                    f"Invalid complete gold taxonomy for {review_id} item {index}: "
                    f"{category}/{issue_group}"
                )
            if item["sentiment"] not in ALLOWED_POLARITIES:
                raise ValueError(
                    f"Invalid complete gold sentiment for {review_id} item {index}"
                )
            if item["attribution"] not in ALLOWED_TARGETS:
                raise ValueError(
                    f"Invalid complete gold attribution for {review_id} item {index}"
                )
            if (
                item["evidence_quote"] not in review_text
                and item["evidence_quote"] not in review_title
            ):
                raise ValueError(
                    f"Invalid complete gold evidence for {review_id} item {index}: "
                    "quote is not verbatim in title or review text"
                )
            clean.append({key: item[key].strip() for key in required})
        complete[review_id] = clean
    return complete, {
        "rows": len(rows),
        "complete": len(complete),
        "incomplete_or_blank": incomplete,
        "explicit_incomplete": explicit_no,
    }


def _tuple_dict(value: tuple[str, ...]) -> dict[str, str]:
    keys = (
        ("category", "issue_group")
        if len(value) == 2
        else ("category", "issue_group", "sentiment", "attribution")
    )
    return dict(zip(keys, value))


def _coverage_level(
    predictions: Mapping[str, set[tuple[str, ...]]],
    gold: Mapping[str, set[tuple[str, ...]]],
    ratings: Mapping[str, int],
    weights: Mapping[str, float],
) -> dict[str, Any]:
    matched_total = 0
    gold_total = 0
    prediction_total = 0
    macro_values: list[float] = []
    zero_gold = 0
    weighted_matched = 0.0
    weighted_gold = 0.0
    by_rating: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    missed_by_review: dict[str, list[dict[str, str]]] = {}
    extra_by_review: dict[str, list[dict[str, str]]] = {}
    for review_id, gold_set in gold.items():
        predicted_set = predictions.get(review_id, set())
        matched = gold_set & predicted_set
        missed = gold_set - predicted_set
        extra = predicted_set - gold_set
        matched_total += len(matched)
        gold_total += len(gold_set)
        prediction_total += len(predicted_set)
        weight = float(weights.get(review_id, 1.0))
        weighted_matched += weight * len(matched)
        weighted_gold += weight * len(gold_set)
        rating = ratings.get(review_id)
        if rating is not None:
            by_rating[rating][0] += len(matched)
            by_rating[rating][1] += len(gold_set)
        if gold_set:
            macro_values.append(len(matched) / len(gold_set))
        else:
            zero_gold += 1
        if missed:
            missed_by_review[review_id] = [
                _tuple_dict(value) for value in sorted(missed)
            ]
        if extra:
            extra_by_review[review_id] = [
                _tuple_dict(value) for value in sorted(extra)
            ]
    rating_recalls = {
        str(rating): (
            round(values[0] / values[1], 6) if values[1] else None
        )
        for rating, values in sorted(by_rating.items())
    }
    valid_rating_recalls = [
        value for value in rating_recalls.values() if value is not None
    ]
    return {
        "reviews": len(gold),
        "micro": {
            "matched": matched_total,
            "gold": gold_total,
            "predicted": prediction_total,
            "missed": gold_total - matched_total,
            "extra": sum(len(values) for values in extra_by_review.values()),
            "recall": (
                round(matched_total / gold_total, 6) if gold_total else None
            ),
        },
        "review_macro": {
            "eligible_reviews": len(macro_values),
            "zero_gold_reviews": zero_gold,
            "recall": (
                round(sum(macro_values) / len(macro_values), 6)
                if macro_values else None
            ),
        },
        "sampling_weighted_micro_recall": (
            round(weighted_matched / weighted_gold, 6)
            if weighted_gold else None
        ),
        "rating_recall": rating_recalls,
        "rating_macro_recall": (
            round(sum(valid_rating_recalls) / len(valid_rating_recalls), 6)
            if valid_rating_recalls else None
        ),
        "missed_by_review": missed_by_review,
        "extra_by_review": extra_by_review,
    }


def _coverage_metrics(
    selected_ids: Sequence[str],
    baseline: Mapping[str, Mapping[str, Any]],
    claims: Mapping[str, Mapping[str, Any]],
    gold_items: Mapping[str, Sequence[Mapping[str, str]]],
    reviews: Mapping[str, Mapping[str, Any]],
    weights: Mapping[str, float],
) -> dict[str, Any]:
    ratings = {
        review_id: int(float(reviews[review_id].get("rating", 0)))
        for review_id in gold_items
    }
    result: dict[str, Any] = {}
    for system in ("v3", "evidence_first"):
        pair_predictions: dict[str, set[tuple[str, ...]]] = {}
        full_predictions: dict[str, set[tuple[str, ...]]] = {}
        pair_gold: dict[str, set[tuple[str, ...]]] = {}
        full_gold: dict[str, set[tuple[str, ...]]] = {}
        for review_id in selected_ids:
            if review_id not in gold_items:
                continue
            items = (
                baseline[review_id].get("accepted_feedback", []) or []
                if system == "v3"
                else claims[review_id].get("merged_claims", []) or []
            )
            signatures = {
                signature
                for item in items
                if isinstance(item, dict) and (signature := _signature(item))
            }
            pair_predictions[review_id] = {
                (signature[0], signature[1]) for signature in signatures
            }
            full_predictions[review_id] = set(signatures)
            gold_signatures = {
                (
                    item["category"], item["issue_group"], item["sentiment"],
                    item["attribution"],
                )
                for item in gold_items[review_id]
            }
            pair_gold[review_id] = {
                (signature[0], signature[1]) for signature in gold_signatures
            }
            full_gold[review_id] = gold_signatures
        result[system] = {
            "issue_pair": _coverage_level(
                pair_predictions, pair_gold, ratings, weights
            ),
            "full_tuple": _coverage_level(
                full_predictions, full_gold, ratings, weights
            ),
        }
    return result


def _stage_status(
    ordered_universe: Sequence[str],
    observations: Mapping[str, Mapping[str, Any]],
    claims: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Separate cache presence from a usable end-to-end pipeline result."""
    cache_complete_ids: list[str] = []
    valid_generation_ids: list[str] = []
    pending_ids: list[str] = []
    failure_details: dict[str, list[dict[str, Any]]] = {}
    partial_validation_ids: list[str] = []
    model_normalization_incomplete_ids: list[str] = []
    claim_model_parse_failure_ids: list[str] = []
    status_unknown_ids: list[str] = []

    for review_id in ordered_universe:
        missing = []
        if review_id not in observations:
            missing.append("observation_cache")
        if review_id not in claims:
            missing.append("claim_cache")
        if missing:
            pending_ids.append(review_id)
            continue

        cache_complete_ids.append(review_id)
        observation_row = observations[review_id]
        claim_row = claims[review_id]
        failures: list[dict[str, Any]] = []

        if "parse_success" not in observation_row:
            status_unknown_ids.append(review_id)
        elif observation_row.get("parse_success") is False:
            failures.append({
                "stage": "observation_extraction",
                "failure": "parse_failure",
                "errors": list(observation_row.get("validation_errors") or []),
            })

        if "normalization_complete" not in claim_row:
            if review_id not in status_unknown_ids:
                status_unknown_ids.append(review_id)
        elif claim_row.get("normalization_complete") is False:
            failures.append({
                "stage": "claim_normalization",
                "failure": "incomplete_normalization",
                "errors": list(claim_row.get("validation_errors") or []),
            })

        # These conditions are reported as diagnostics, but they do not make
        # the final deterministic output unusable.
        if observation_row.get("reference_checks_passed") is False:
            partial_validation_ids.append(review_id)
        if claim_row.get("model_normalization_complete") is False:
            model_normalization_incomplete_ids.append(review_id)
        if claim_row.get("parse_success") is False:
            claim_model_parse_failure_ids.append(review_id)

        if failures:
            failure_details[review_id] = failures
        else:
            # Missing status keys are accepted for legacy fixtures/artifacts;
            # current v1.4 rows carry both and are listed above when missing.
            valid_generation_ids.append(review_id)

    return {
        "cohort_reviews": len(ordered_universe),
        "cache_complete_reviews": len(cache_complete_ids),
        "valid_generation_reviews": len(valid_generation_ids),
        "stage_failure_reviews": len(failure_details),
        "pending_reviews": len(pending_ids),
        "cache_complete_review_ids": cache_complete_ids,
        "valid_generation_review_ids": valid_generation_ids,
        "stage_failure_review_ids": list(failure_details),
        "pending_review_ids": pending_ids,
        "failure_details": failure_details,
        "partial_validation_review_ids": partial_validation_ids,
        "model_normalization_incomplete_review_ids": (
            model_normalization_incomplete_ids
        ),
        "claim_model_parse_failure_review_ids": claim_model_parse_failure_ids,
        "status_unknown_review_ids": status_unknown_ids,
    }


def summarize_evaluation(
    project_root: str | Path,
    taxonomy: Mapping[str, Mapping[str, str]],
    cohort: str = "pilot",
    show: int = 10,
) -> dict[str, Any]:
    """Create comparable human-evaluation artifacts for pilot or holdout."""
    root = Path(project_root).resolve()
    paths = _cohort_paths(root, cohort)
    observation_rows = _read_jsonl(paths["observations"])
    claim_rows = _read_jsonl(paths["claims"])
    baseline_rows = _read_jsonl(
        root / "outputs/amazon/acer_aspire5_business_full_v3.jsonl"
    )
    review_rows = _read_jsonl(
        root / "data/processed/amazon/acer_aspire5_reviews_clean.jsonl"
    )
    sentence_rows = _read_jsonl(
        root / "data/processed/amazon/acer_aspire5_sentences.jsonl"
    )
    observations = _unique_review_index(observation_rows, str(paths["observations"]))
    claims = _unique_review_index(claim_rows, str(paths["claims"]))
    baseline = _unique_review_index(
        baseline_rows, "outputs/amazon/acer_aspire5_business_full_v3.jsonl"
    )
    reviews = _unique_review_index(
        review_rows, "data/processed/amazon/acer_aspire5_reviews_clean.jsonl"
    )

    if cohort == "holdout":
        if not paths["selection"].exists():
            raise FileNotFoundError(paths["selection"])
        selection = json.loads(paths["selection"].read_text(encoding="utf-8"))
        selected_records = selection.get("selected_reviews", [])
        ordered_universe = [str(row["review_id"]) for row in selected_records]
        weights = {
            str(row["review_id"]): float(row.get("sampling_weight", 1.0))
            for row in selected_records
        }
    else:
        ordered_universe = list(dict.fromkeys(
            [str(row["review_id"]) for row in claim_rows]
            + [str(row["review_id"]) for row in observation_rows]
        ))
        weights = {review_id: 1.0 for review_id in ordered_universe}

    stage_status = _stage_status(ordered_universe, observations, claims)
    cache_complete_ids = stage_status["cache_complete_review_ids"]
    valid_generation_ids = stage_status["valid_generation_review_ids"]
    if not cache_complete_ids:
        raise ValueError(f"No completed {cohort} reviews are available for evaluation")
    missing_baseline = [value for value in ordered_universe if value not in baseline]
    missing_reviews = [value for value in ordered_universe if value not in reviews]
    if missing_baseline or missing_reviews:
        raise ValueError(
            f"Evaluation alignment failed: missing baseline={missing_baseline}, "
            f"missing reviews={missing_reviews}"
        )
    source_by_id = _source_index(sentence_rows, review_rows)
    primary_rows, blind_key = _build_primary_rows(
        cache_complete_ids, baseline, claims, reviews, source_by_id
    )
    atomic_rows, abstention_rows = _build_diagnostic_rows(
        cache_complete_ids, claims, observations, reviews
    )
    gold_rows = [
        {
            "review_id": review_id,
            "rating": reviews[review_id].get("rating", ""),
            "review_title": reviews[review_id].get("title", ""),
            "review_text": reviews[review_id].get("clean_text", ""),
        }
        for review_id in ordered_universe
    ]

    output_dir = paths["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    primary_path = output_dir / "business_finding_precision_blind.csv"
    key_path = output_dir / "business_finding_precision_key.json"
    atomic_path = output_dir / "ef_atomic_evidence_audit.csv"
    abstention_path = output_dir / "ef_abstention_audit.csv"
    gold_path = output_dir / "gold_coverage_review.csv"
    metrics_path = output_dir / "human_metrics.json"

    # Validate any existing completed gold labels before refreshing another
    # evaluation artifact, so malformed gold cannot leave a partial update.
    preserved_gold = _read_preserved(
        gold_path, "review_id", GOLD_HUMAN_COLUMNS
    )
    for row in gold_rows:
        for column, value in preserved_gold.get(str(row["review_id"]), {}).items():
            row[column] = value
        for column in GOLD_HUMAN_COLUMNS:
            row.setdefault(column, "")
    gold_items, gold_status = _parse_gold(gold_rows, taxonomy)

    _write_csv_preserving(
        primary_path, primary_rows, PRIMARY_FIELDS, "blind_id", PRIMARY_HUMAN_COLUMNS
    )
    _write_csv_preserving(
        atomic_path, atomic_rows, ATOMIC_FIELDS, "audit_id", ATOMIC_HUMAN_COLUMNS
    )
    _write_csv_preserving(
        abstention_path, abstention_rows, ABSTENTION_FIELDS,
        "abstention_id", ABSTENTION_HUMAN_COLUMNS,
    )
    _write_csv_preserving(
        gold_path, gold_rows, GOLD_FIELDS, "review_id", GOLD_HUMAN_COLUMNS
    )
    _write_json_if_changed(key_path, {
        "schema_version": "reviewlens_blind_finding_key_v1",
        "cohort": cohort,
        "warning": "Keep this key separate from human raters until labels are frozen.",
        "rows": blind_key,
    })
    coverage_end_to_end = _coverage_metrics(
        cache_complete_ids, baseline, claims, gold_items, reviews, weights
    )
    coverage_conditional = _coverage_metrics(
        valid_generation_ids, baseline, claims, gold_items, reviews, weights
    )
    valid_generation_set = set(valid_generation_ids)
    metrics = {
        "schema_version": "reviewlens_human_evaluation_v2_1",
        "cohort": cohort,
        "cohort_reviews": len(ordered_universe),
        # Backward-compatible alias: completed means both cache rows exist;
        # it does not assert that model generation succeeded.
        "completed_reviews": len(cache_complete_ids),
        "cache_complete_reviews": len(cache_complete_ids),
        "valid_generation_reviews": len(valid_generation_ids),
        "missing_stage_review_ids": stage_status["pending_review_ids"],
        "stage_status": stage_status,
        "primary_finding_precision_by_system": _primary_metrics(
            primary_rows, blind_key
        ),
        "atomic_evidence_audit": _component_metrics(
            atomic_rows, ATOMIC_HUMAN_COLUMNS[:-1]
        ),
        "abstention_audit": _abstention_metrics(abstention_rows),
        "gold_label_status": gold_status,
        # Existing consumers retain the end-to-end interpretation.
        "coverage_by_system": coverage_end_to_end,
        "coverage_by_system_end_to_end": coverage_end_to_end,
        "coverage_by_system_conditional_on_valid_generation": (
            coverage_conditional
        ),
        "coverage_scopes": {
            "end_to_end": {
                "review_count": len(cache_complete_ids),
                "review_ids": cache_complete_ids,
                "description": (
                    "All cache-complete reviews; failed EF generation remains "
                    "an empty prediction and counts against coverage."
                ),
            },
            "conditional_on_valid_generation": {
                "review_count": len(valid_generation_ids),
                "review_ids": valid_generation_ids,
                "excluded_review_ids": [
                    review_id for review_id in cache_complete_ids
                    if review_id not in valid_generation_set
                ],
                "description": (
                    "Both systems evaluated on the same subset with a valid "
                    "EF pipeline output."
                ),
            },
        },
        "notes": [
            "The primary comparison uses review-level findings for both systems.",
            "Atomic and abstention audits are diagnostics, not primary precision.",
            "Blank or unrecognized human labels are excluded and reported.",
            "Coverage is calculated only for reviews marked human_gold_complete=yes.",
            "End-to-end coverage counts a failed EF generation as an empty prediction.",
            "Conditional coverage uses the same valid-generation subset for both systems.",
            "No improvement claim is valid before human labels are complete.",
        ],
    }
    _write_json_if_changed(metrics_path, metrics)

    counts = {
        "v3_findings": sum(
            blind_key[str(row["blind_id"])]["system"] == "v3"
            for row in primary_rows
        ),
        "ef_findings": sum(
            blind_key[str(row["blind_id"])]["system"] == "evidence_first"
            for row in primary_rows
        ),
        "ef_mapped_atomic": len(atomic_rows),
        "ef_unmapped_abstentions": len(abstention_rows),
        "cache_complete_reviews": len(cache_complete_ids),
        "valid_generation_reviews": len(valid_generation_ids),
        "stage_failure_reviews": stage_status["stage_failure_reviews"],
    }
    print(f"\n=== REVIEWLENS FAIR EVALUATION: {cohort.upper()} ===")
    print(
        f"Cache complete: {len(cache_complete_ids)}/{len(ordered_universe)}; "
        f"pending stages: {stage_status['pending_reviews']}"
    )
    print(
        f"Valid generation outputs: {len(valid_generation_ids)}/"
        f"{len(ordered_universe)}; stage failures: "
        f"{stage_status['stage_failure_reviews']}"
    )
    for review_id, failures in stage_status["failure_details"].items():
        labels = ", ".join(
            f"{item['stage']}:{item['failure']}" for item in failures
        )
        print(f"  STAGE FAILURE {review_id}: {labels}")
    print("Evaluation rows:", counts)
    for review_id in cache_complete_ids[: max(show, 0)]:
        v3_count = len(baseline[review_id].get("accepted_feedback", []) or [])
        ef_count = len(claims[review_id].get("merged_claims", []) or [])
        print(f"  {review_id}: v3_findings={v3_count} ef_findings={ef_count}")
    print("Blind finding precision:", primary_path)
    print("Keep hidden from raters:", key_path)
    print("EF atomic evidence audit:", atomic_path)
    print("EF abstention audit:", abstention_path)
    print("Independent gold coverage:", gold_path)
    print("Human metrics:", metrics_path)
    print("No model was loaded or called.")
    return {"counts": counts, "metrics": metrics, "paths": {
        "primary": str(primary_path),
        "key": str(key_path),
        "atomic": str(atomic_path),
        "abstention": str(abstention_path),
        "gold": str(gold_path),
        "metrics": str(metrics_path),
    }}


__all__ = ["summarize_evaluation"]
