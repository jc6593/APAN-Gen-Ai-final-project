"""Reproducible evaluation for the locked Evidence-first v1.5 final cohort.

This module is deliberately separate from the frozen inference implementation.
It validates every locked artifact before reading labels, reconstructs blinded
primary items deterministically, and writes evaluation-only artifacts.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from .core import hash_json, read_jsonl, stage_record_hash, taxonomy_hash
from .pipeline import _claim_record_hash


PRIMARY_LABELS = (
    "supported",
    "category_correct",
    "sentiment_correct",
    "target_correct",
)
MAPPED_DIAGNOSTIC_LABELS = (
    "supported",
    "atomic",
    "category_correct",
    "sentiment_correct",
    "target_correct",
)
YES_NO = {"yes", "no"}


def _paths(root: Path) -> dict[str, Path]:
    output = root / "outputs/amazon/evidence_first/v15_final"
    return {
        "output": output,
        "selection": output / "selection_manifest.json",
        "gold": output / "gold_reference.json",
        "run": output / "run_manifest.json",
        "observations": output / "observations_v15_final.jsonl",
        "claims": output / "claims_v15_final.jsonl",
        "completion": output / "completion_manifest.json",
        "precision_labels": output / "precision_labels.json",
        "diagnostic_labels": output / "diagnostic_labels.json",
        "blind_items": output / "blind_precision_items.json",
        "evaluation": output / "final_evaluation.json",
        "reviews": root / "data/processed/amazon/acer_aspire5_reviews_clean.jsonl",
        "baseline": root / "outputs/amazon/acer_aspire5_business_full_v3.jsonl",
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path, label: str) -> dict[str, Any]:
    if not path.exists():
        raise ValueError(f"Missing {label}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Invalid {label}: expected one JSON object")
    return value


def _unique_index(
    rows: Sequence[Mapping[str, Any]], label: str
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        review_id = row.get("review_id")
        if not isinstance(review_id, str) or not review_id or review_id in result:
            raise ValueError(f"Duplicate or missing review_id in {label}: {review_id}")
        result[review_id] = row
    return result


def _stable_id(prefix: str, *parts: Any) -> str:
    payload = json.dumps(
        parts, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return prefix + hashlib.sha256(payload).hexdigest()[:16].upper()


def _quote(evidence: Mapping[str, Any]) -> str:
    for name in ("text", "evidence_text", "quote"):
        value = evidence.get(name)
        if isinstance(value, str) and value:
            return value
    return ""


def _evidence_quotes(item: Mapping[str, Any]) -> list[str]:
    evidence = item.get("evidence", []) or []
    if isinstance(evidence, Mapping):
        evidence = [evidence]
    quotes: list[str] = []
    for value in evidence:
        if not isinstance(value, Mapping):
            continue
        quote = _quote(value)
        if quote and quote not in quotes:
            quotes.append(quote)
    return quotes


def _finding_text(item: Mapping[str, Any]) -> str:
    for name in ("summary", "claim"):
        value = item.get(name)
        if isinstance(value, str) and value:
            return value
    components = item.get("component_claims", []) or []
    if isinstance(components, list):
        return " || ".join(str(value) for value in components if str(value))
    return ""


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


def _validate_locked_inputs(
    root: Path,
    taxonomy: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    paths = _paths(root)
    selection = _load_object(paths["selection"], "final selection")
    gold = _load_object(paths["gold"], "final gold reference")
    run = _load_object(paths["run"], "final run manifest")
    completion = _load_object(paths["completion"], "final completion manifest")
    observations = read_jsonl(paths["observations"])
    claims = read_jsonl(paths["claims"])

    selected_rows = selection.get("selected_reviews")
    if not isinstance(selected_rows, list) or len(selected_rows) != 15:
        raise ValueError("Final selection must contain exactly 15 reviews")
    selected_ids = [row.get("review_id") for row in selected_rows]
    if not all(isinstance(value, str) and value for value in selected_ids):
        raise ValueError("Invalid final selected review IDs")
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError("Duplicate final selected review IDs")

    expected_completion = {
        "schema_version": "reviewlens_v15_final_completion_v1",
        "status": "complete_locked_no_regeneration",
        "selected_review_ids": selected_ids,
        "selection_manifest_sha256": _sha256(paths["selection"]),
        "run_manifest_sha256": _sha256(paths["run"]),
        "observations_sha256": _sha256(paths["observations"]),
        "claims_sha256": _sha256(paths["claims"]),
        "gold_reference_sha256": _sha256(paths["gold"]),
    }
    mismatches = [
        key for key, value in expected_completion.items()
        if completion.get(key) != value
    ]
    if mismatches:
        raise ValueError(f"Final completion manifest mismatch: {mismatches}")

    if run.get("selected_review_ids") != selected_ids:
        raise ValueError("Final run review order mismatch")
    if run.get("selection_manifest_hash") != hash_json(selection):
        raise ValueError("Final run does not match selection content")
    if run.get("taxonomy_hash") != taxonomy_hash(taxonomy):
        raise ValueError("Evaluation taxonomy differs from the frozen run")
    if gold.get("selected_review_ids") != selected_ids:
        raise ValueError("Final gold review order mismatch")
    if gold.get("selection_manifest_sha256") != _sha256(paths["selection"]):
        raise ValueError("Final gold does not match the frozen selection file")
    if gold.get("evaluation_protocol_hash") != selection.get(
        "evaluation_protocol_hash"
    ):
        raise ValueError("Final gold evaluation protocol mismatch")
    if gold.get("taxonomy_hash") != run.get("taxonomy_hash"):
        raise ValueError("Final gold taxonomy mismatch")
    gold_reviews = gold.get("reviews")
    if not isinstance(gold_reviews, list) or [
        row.get("review_id") for row in gold_reviews if isinstance(row, Mapping)
    ] != selected_ids:
        raise ValueError("Final gold rows are incomplete or out of order")

    if [row.get("review_id") for row in observations] != selected_ids:
        raise ValueError("Final observation rows are incomplete or out of order")
    if [row.get("review_id") for row in claims] != selected_ids:
        raise ValueError("Final claim rows are incomplete or out of order")
    for row in observations:
        if row.get("record_hash") != stage_record_hash(
            row, "accepted_observations"
        ):
            raise ValueError(f"Corrupt final observation row: {row.get('review_id')}")
    for observation_row, claim_row in zip(observations, claims):
        if claim_row.get("record_hash") != _claim_record_hash(claim_row):
            raise ValueError(f"Corrupt final claim row: {claim_row.get('review_id')}")
        if claim_row.get("upstream_hash") != observation_row.get("record_hash"):
            raise ValueError(
                f"Final claim upstream mismatch: {claim_row.get('review_id')}"
            )

    source_hashes = selection.get("source_artifact_hashes", {})
    if _sha256(paths["baseline"]) != source_hashes.get("baseline_v3"):
        raise ValueError("Frozen baseline v3 artifact changed")
    if _sha256(paths["reviews"]) != source_hashes.get("reviews"):
        raise ValueError("Frozen review source artifact changed")

    reviews = _unique_index(read_jsonl(paths["reviews"]), "review source")
    baseline = _unique_index(read_jsonl(paths["baseline"]), "baseline v3")
    if any(review_id not in reviews for review_id in selected_ids):
        raise ValueError("A final review is missing from the source artifact")
    if any(review_id not in baseline for review_id in selected_ids):
        raise ValueError("A final review is missing from baseline v3")

    return {
        "paths": paths,
        "selection": selection,
        "gold": gold,
        "run": run,
        "completion": completion,
        "selected_ids": selected_ids,
        "reviews": reviews,
        "baseline": baseline,
        "observations": observations,
        "claims": claims,
        "gold_reviews": _unique_index(gold_reviews, "gold reference"),
    }


def _blind_primary_items(
    locked: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    items: list[dict[str, Any]] = []
    key: dict[str, dict[str, Any]] = {}
    claims_by_id = _unique_index(locked["claims"], "final claims")
    systems = (
        ("A", "v3", locked["baseline"], "accepted_feedback"),
        ("B", "evidence_first", claims_by_id, "merged_claims"),
    )
    for code, system, source, field in systems:
        for review_id in locked["selected_ids"]:
            raw_items = source[review_id].get(field, []) or []
            if not isinstance(raw_items, list):
                raise ValueError(f"Invalid {system} findings for {review_id}")
            for index, item in enumerate(raw_items):
                if not isinstance(item, Mapping):
                    raise ValueError(f"Invalid {system} finding for {review_id}")
                signature = _signature(item)
                if signature is None:
                    raise ValueError(f"Invalid mapped {system} finding for {review_id}")
                quotes = _evidence_quotes(item)
                finding = _finding_text(item)
                compact_signature = json.dumps(
                    list(signature),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                payload = (
                    f"{code}|{review_id}|{index}|{compact_signature}|"
                    f"{finding}|{'||'.join(quotes)}"
                )
                blind_id = "BLIND_" + hashlib.sha256(
                    payload.encode("utf-8")
                ).hexdigest()[:16].upper()
                if blind_id in key:
                    raise ValueError(f"Duplicate final blind ID: {blind_id}")
                review = locked["reviews"][review_id]
                items.append({
                    "blind_id": blind_id,
                    "review_id": review_id,
                    "review_title": review.get("title", ""),
                    "review_text": review.get("clean_text", ""),
                    "proposed_category": signature[0],
                    "proposed_issue_group": signature[1],
                    "proposed_sentiment": signature[2],
                    "proposed_attribution": signature[3],
                    "proposed_finding": finding,
                    "evidence_quotes": quotes,
                })
                key[blind_id] = {
                    "system": system,
                    "review_id": review_id,
                    "signature": signature,
                }
    items.sort(
        key=lambda row: hashlib.sha256(
            f"shuffle|{row['blind_id']}".encode("utf-8")
        ).hexdigest()
    )
    return items, key


def _diagnostic_items(
    locked: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    items: list[dict[str, Any]] = []
    key: dict[str, dict[str, Any]] = {}
    observations_by_review = _unique_index(
        locked["observations"], "final observations"
    )
    for claim_row in locked["claims"]:
        review_id = str(claim_row["review_id"])
        observation_by_id = {
            str(item.get("observation_id")): item
            for item in observations_by_review[review_id].get(
                "accepted_observations", []
            )
            if isinstance(item, Mapping)
        }
        for claim in claim_row.get("accepted_claims", []) or []:
            if not isinstance(claim, Mapping):
                raise ValueError(f"Invalid atomic claim for {review_id}")
            status = claim.get("mapping_status")
            claim_id = claim.get("claim_id")
            if status not in {"mapped", "unmapped"} or not isinstance(
                claim_id, str
            ):
                raise ValueError(f"Invalid atomic claim identity for {review_id}")
            prefix = "ATOMIC_" if status == "mapped" else "ABSTAIN_"
            diagnostic_id = _stable_id(prefix, review_id, claim_id)
            if diagnostic_id in key:
                raise ValueError(f"Duplicate diagnostic ID: {diagnostic_id}")
            observation = observation_by_id.get(str(claim.get("observation_id")), {})
            review = locked["reviews"][review_id]
            item = {
                "diagnostic_id": diagnostic_id,
                "review_id": review_id,
                "review_title": review.get("title", ""),
                "review_text": review.get("clean_text", ""),
                "observation_id": claim.get("observation_id"),
                "mapping_status": status,
                "proposed_category": claim.get("category"),
                "proposed_issue_group": claim.get("issue_group"),
                "proposed_sentiment": claim.get("sentiment"),
                "proposed_attribution": claim.get("attribution"),
                "statement_type": (
                    observation.get("statement_type")
                    or claim.get("statement_type")
                ),
                "claim": claim.get("claim", ""),
                "evidence_quotes": _evidence_quotes(claim),
            }
            items.append(item)
            key[diagnostic_id] = {"mapping_status": status, "claim": claim}
    items.sort(key=lambda row: str(row["diagnostic_id"]))
    return items, key


def _blind_document(
    locked: Mapping[str, Any],
    primary_items: Sequence[Mapping[str, Any]],
    diagnostic_items: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    paths = locked["paths"]
    return {
        "schema_version": "reviewlens_v15_final_blind_items_v1",
        "status": "locked_items_awaiting_complete_labels",
        "selection_manifest_sha256": _sha256(paths["selection"]),
        "completion_manifest_sha256": _sha256(paths["completion"]),
        "gold_reference_sha256": _sha256(paths["gold"]),
        "instructions": {
            "primary_labels": list(PRIMARY_LABELS),
            "primary_fully_correct": "all four primary labels are yes",
            "mapped_diagnostic_labels": list(MAPPED_DIAGNOSTIC_LABELS),
            "unmapped_diagnostic_labels": [
                "should_map",
                "expected_category",
                "expected_issue_group",
                "statement_type_correct",
            ],
            "yes_no_values": ["yes", "no"],
            "blind_rule": "Judge source support without inferring system identity.",
        },
        "items": list(primary_items),
        "diagnostic_items": list(diagnostic_items),
    }


def _write_immutable(path: Path, document: Mapping[str, Any]) -> None:
    content = json.dumps(
        document, ensure_ascii=False, indent=2, sort_keys=True
    ) + "\n"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != document:
            raise ValueError(f"Existing immutable evaluation input differs: {path}")
        return
    path.write_text(content, encoding="utf-8")


def prepare_v15_final_evaluation(
    project_root: str | Path,
    taxonomy: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    """Validate locked outputs and materialize deterministic annotation items."""
    root = Path(project_root).resolve()
    locked = _validate_locked_inputs(root, taxonomy)
    primary_items, _ = _blind_primary_items(locked)
    diagnostic_items, _ = _diagnostic_items(locked)
    document = _blind_document(locked, primary_items, diagnostic_items)
    _write_immutable(locked["paths"]["blind_items"], document)
    return document


def _normalized_yes_no(value: Any, label: str) -> str:
    normalized = str(value or "").strip().casefold()
    if normalized not in YES_NO:
        raise ValueError(f"{label} must be yes or no")
    return normalized


def _label_index(
    path: Path,
    id_field: str,
    expected_ids: set[str],
    label: str,
) -> dict[str, dict[str, Any]]:
    document = _load_object(path, label)
    rows = document.get("labels")
    if not isinstance(rows, list):
        raise ValueError(f"Invalid {label}: labels must be a list")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"Invalid {label} row")
        row_id = row.get(id_field)
        if not isinstance(row_id, str) or not row_id or row_id in result:
            raise ValueError(f"Duplicate or missing {id_field} in {label}: {row_id}")
        result[row_id] = row
    missing = sorted(expected_ids - set(result))
    extra = sorted(set(result) - expected_ids)
    if missing or extra:
        raise ValueError(
            f"Incomplete {label}: missing={missing[:5]} extra={extra[:5]}"
        )
    return result


def _validate_precision_labels(
    path: Path, primary_key: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    labels = _label_index(
        path, "blind_id", set(primary_key), "final precision labels"
    )
    for blind_id, row in labels.items():
        for field in PRIMARY_LABELS:
            row[field] = _normalized_yes_no(
                row.get(field), f"{blind_id}.{field}"
            )
    return labels


def _validate_diagnostic_labels(
    path: Path,
    diagnostic_key: Mapping[str, Mapping[str, Any]],
    taxonomy: Mapping[str, Mapping[str, str]],
) -> dict[str, dict[str, Any]]:
    labels = _label_index(
        path, "diagnostic_id", set(diagnostic_key), "final diagnostic labels"
    )
    for diagnostic_id, row in labels.items():
        status = diagnostic_key[diagnostic_id]["mapping_status"]
        if status == "mapped":
            for field in MAPPED_DIAGNOSTIC_LABELS:
                row[field] = _normalized_yes_no(
                    row.get(field), f"{diagnostic_id}.{field}"
                )
            continue
        row["should_map"] = _normalized_yes_no(
            row.get("should_map"), f"{diagnostic_id}.should_map"
        )
        row["statement_type_correct"] = _normalized_yes_no(
            row.get("statement_type_correct"),
            f"{diagnostic_id}.statement_type_correct",
        )
        category = row.get("expected_category")
        issue_group = row.get("expected_issue_group")
        if category in {"", None} and issue_group in {"", None}:
            category = issue_group = None
        elif (
            not isinstance(category, str)
            or category not in taxonomy
            or not isinstance(issue_group, str)
            or issue_group not in taxonomy[category]
        ):
            raise ValueError(
                f"Invalid expected taxonomy pair for {diagnostic_id}"
            )
        if row["should_map"] == "yes" and category is None:
            raise ValueError(
                f"A should_map=yes label needs an expected pair: {diagnostic_id}"
            )
        if row["should_map"] == "no" and category is not None:
            raise ValueError(
                f"A should_map=no label must leave the expected pair null: "
                f"{diagnostic_id}"
            )
        row["expected_category"] = category
        row["expected_issue_group"] = issue_group
    return labels


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def _component_metrics(
    rows: Sequence[Mapping[str, Any]], fields: Sequence[str]
) -> dict[str, Any]:
    components = {}
    for field in fields:
        yes = sum(row[field] == "yes" for row in rows)
        components[field] = {
            "yes": yes,
            "no": len(rows) - yes,
            "total": len(rows),
            "rate": _rate(yes, len(rows)),
        }
    strict = sum(all(row[field] == "yes" for field in fields) for row in rows)
    return {
        "rows": len(rows),
        "components": components,
        "strict_fully_correct": {
            "yes": strict,
            "no": len(rows) - strict,
            "total": len(rows),
            "rate": _rate(strict, len(rows)),
        },
    }


def _primary_metrics(
    labels: Mapping[str, Mapping[str, Any]],
    primary_key: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    result = {}
    for system in ("v3", "evidence_first"):
        rows = [
            labels[blind_id]
            for blind_id, item in primary_key.items()
            if item["system"] == system
        ]
        result[system] = _component_metrics(rows, PRIMARY_LABELS)
    return result


def _gold_signatures(
    gold_review: Mapping[str, Any]
) -> set[tuple[str, str, str, str]]:
    signatures = set()
    for finding in gold_review.get("gold_findings", []) or []:
        signature = _signature(finding)
        if signature is None:
            raise ValueError(f"Invalid gold finding: {finding}")
        signatures.add(signature)
    return signatures


def _recall_metrics(
    locked: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    claims_by_id = _unique_index(locked["claims"], "final claims")
    rating_by_id = {
        str(row["review_id"]): str(int(row["rating"]))
        for row in locked["selection"]["selected_reviews"]
    }
    systems = {
        "v3": (
            locked["baseline"],
            "accepted_feedback",
        ),
        "evidence_first": (
            claims_by_id,
            "merged_claims",
        ),
    }
    result: dict[str, Any] = {}
    exact_proxy: dict[str, Any] = {}
    for system, (source, field) in systems.items():
        full_hits = full_gold = pair_hits = pair_gold = 0
        full_ratios: list[float] = []
        pair_ratios: list[float] = []
        per_review: list[dict[str, Any]] = []
        rating_rows = {
            str(value): {
                "full_tuple_hits": 0,
                "full_tuple_gold": 0,
                "issue_pair_hits": 0,
                "issue_pair_gold": 0,
            }
            for value in range(1, 6)
        }
        predicted_rows = 0
        exact_prediction_hits = 0
        for review_id in locked["selected_ids"]:
            gold_signatures = _gold_signatures(
                locked["gold_reviews"][review_id]
            )
            gold_pairs = {value[:2] for value in gold_signatures}
            raw_items = source[review_id].get(field, []) or []
            predicted_signatures: list[tuple[str, str, str, str]] = []
            for item in raw_items:
                signature = _signature(item)
                if signature is None:
                    raise ValueError(
                        f"Invalid {system} evaluation finding: {review_id}"
                    )
                predicted_signatures.append(signature)
            predicted_set = set(predicted_signatures)
            predicted_pairs = {value[:2] for value in predicted_set}
            review_full_hits = len(gold_signatures & predicted_set)
            review_pair_hits = len(gold_pairs & predicted_pairs)
            full_hits += review_full_hits
            full_gold += len(gold_signatures)
            pair_hits += review_pair_hits
            pair_gold += len(gold_pairs)
            predicted_rows += len(predicted_signatures)
            exact_prediction_hits += sum(
                value in gold_signatures for value in predicted_signatures
            )
            if gold_signatures:
                full_ratios.append(review_full_hits / len(gold_signatures))
                pair_ratios.append(review_pair_hits / len(gold_pairs))
            rating = rating_by_id[review_id]
            rating_rows[rating]["full_tuple_hits"] += review_full_hits
            rating_rows[rating]["full_tuple_gold"] += len(gold_signatures)
            rating_rows[rating]["issue_pair_hits"] += review_pair_hits
            rating_rows[rating]["issue_pair_gold"] += len(gold_pairs)
            per_review.append({
                "review_id": review_id,
                "rating": int(rating),
                "predicted_findings": len(predicted_signatures),
                "full_tuple_hits": review_full_hits,
                "full_tuple_gold": len(gold_signatures),
                "issue_pair_hits": review_pair_hits,
                "issue_pair_gold": len(gold_pairs),
            })
        for row in rating_rows.values():
            row["full_tuple_recall"] = _rate(
                row["full_tuple_hits"], row["full_tuple_gold"]
            )
            row["issue_pair_recall"] = _rate(
                row["issue_pair_hits"], row["issue_pair_gold"]
            )
        result[system] = {
            "full_tuple": {
                "micro": {
                    "hits": full_hits,
                    "gold": full_gold,
                    "rate": _rate(full_hits, full_gold),
                },
                "macro_nonempty_gold_reviews": {
                    "reviews": len(full_ratios),
                    "excluded_zero_gold_reviews": len(locked["selected_ids"])
                    - len(full_ratios),
                    "rate": round(sum(full_ratios) / len(full_ratios), 6),
                },
            },
            "issue_pair": {
                "micro": {
                    "hits": pair_hits,
                    "gold": pair_gold,
                    "rate": _rate(pair_hits, pair_gold),
                },
                "macro_nonempty_gold_reviews": {
                    "reviews": len(pair_ratios),
                    "excluded_zero_gold_reviews": len(locked["selected_ids"])
                    - len(pair_ratios),
                    "rate": round(sum(pair_ratios) / len(pair_ratios), 6),
                },
            },
            "rating_stratified": rating_rows,
            "per_review": per_review,
        }
        exact_proxy[system] = {
            "exact_signature_hits": exact_prediction_hits,
            "predicted_findings": predicted_rows,
            "rate": _rate(exact_prediction_hits, predicted_rows),
            "note": (
                "Conservative exact-tuple proxy, not the blinded primary "
                "precision metric."
            ),
        }
    return result, exact_proxy


def _diagnostic_metrics(
    labels: Mapping[str, Mapping[str, Any]],
    diagnostic_key: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    mapped = [
        labels[item_id]
        for item_id, item in diagnostic_key.items()
        if item["mapping_status"] == "mapped"
    ]
    unmapped = [
        labels[item_id]
        for item_id, item in diagnostic_key.items()
        if item["mapping_status"] == "unmapped"
    ]
    should_map = sum(row["should_map"] == "yes" for row in unmapped)
    statement_type = sum(
        row["statement_type_correct"] == "yes" for row in unmapped
    )
    return {
        "mapped_atomic": _component_metrics(
            mapped, MAPPED_DIAGNOSTIC_LABELS
        ),
        "unmapped_abstentions": {
            "rows": len(unmapped),
            "should_map": {
                "yes": should_map,
                "no": len(unmapped) - should_map,
                "rate": _rate(should_map, len(unmapped)),
            },
            "correct_abstention": {
                "yes": len(unmapped) - should_map,
                "no": should_map,
                "rate": _rate(len(unmapped) - should_map, len(unmapped)),
            },
            "statement_type_correct": {
                "yes": statement_type,
                "no": len(unmapped) - statement_type,
                "rate": _rate(statement_type, len(unmapped)),
            },
            "expected_pair_labeled": sum(
                row.get("expected_category") is not None for row in unmapped
            ),
        },
    }


def _completion_metrics(locked: Mapping[str, Any]) -> dict[str, Any]:
    baseline_failures = [
        review_id for review_id in locked["selected_ids"]
        if not locked["baseline"][review_id].get("parse_success")
    ]
    observation_failures = [
        row["review_id"] for row in locked["observations"]
        if not row.get("parse_success")
    ]
    claim_failures = [
        row["review_id"] for row in locked["claims"]
        if not row.get("parse_success")
    ]
    accepted_claims = [
        claim
        for row in locked["claims"]
        for claim in row.get("accepted_claims", []) or []
    ]
    return {
        "v3": {
            "reviews": len(locked["selected_ids"]),
            "parse_success": len(locked["selected_ids"]) - len(baseline_failures),
            "parse_failures": len(baseline_failures),
            "failure_review_ids": baseline_failures,
            "business_findings": sum(
                len(locked["baseline"][review_id].get("accepted_feedback", []))
                for review_id in locked["selected_ids"]
            ),
        },
        "evidence_first": {
            "reviews": len(locked["selected_ids"]),
            "observation_parse_success": len(locked["observations"])
            - len(observation_failures),
            "observation_parse_failures": len(observation_failures),
            "observation_failure_review_ids": observation_failures,
            "claim_parse_success": len(locked["claims"]) - len(claim_failures),
            "claim_parse_failures": len(claim_failures),
            "claim_failure_review_ids": claim_failures,
            "accepted_observations": sum(
                len(row.get("accepted_observations", []))
                for row in locked["observations"]
            ),
            "rejected_observations": sum(
                len(row.get("rejected_observations", []))
                for row in locked["observations"]
            ),
            "mapped_atomic_claims": sum(
                claim.get("mapping_status") == "mapped"
                for claim in accepted_claims
            ),
            "unmapped_atomic_claims": sum(
                claim.get("mapping_status") == "unmapped"
                for claim in accepted_claims
            ),
            "merged_business_findings": sum(
                len(row.get("merged_claims", [])) for row in locked["claims"]
            ),
        },
    }


def _improvement_gate(
    primary: Mapping[str, Any], recall: Mapping[str, Any]
) -> dict[str, Any]:
    v3_precision = primary["v3"]["strict_fully_correct"]["rate"]
    ef_precision = primary["evidence_first"]["strict_fully_correct"]["rate"]
    v3_recall = recall["v3"]["full_tuple"]["micro"]["rate"]
    ef_recall = recall["evidence_first"]["full_tuple"]["micro"]["rate"]
    precision_improved = ef_precision > v3_precision
    recall_not_lower = ef_recall >= v3_recall
    passed = precision_improved and recall_not_lower
    if passed:
        decision = "demonstrated_improvement"
    elif (ef_precision > v3_precision and ef_recall < v3_recall) or (
        ef_precision < v3_precision and ef_recall > v3_recall
    ):
        decision = "precision_recall_tradeoff"
    else:
        decision = "no_demonstrated_improvement"
    return {
        "rule": (
            "strict v1.5 finding precision must exceed v3 and end-to-end "
            "full-tuple micro recall must not be lower"
        ),
        "strict_precision_improved": precision_improved,
        "full_tuple_recall_not_lower": recall_not_lower,
        "passed": passed,
        "decision": decision,
        "strict_precision_delta": round(ef_precision - v3_precision, 6),
        "full_tuple_recall_delta": round(ef_recall - v3_recall, 6),
    }


def evaluate_v15_final(
    project_root: str | Path,
    taxonomy: Mapping[str, Mapping[str, str]],
    show: int = 5,
) -> dict[str, Any]:
    """Evaluate the complete locked cohort from complete human/AI labels."""
    root = Path(project_root).resolve()
    locked = _validate_locked_inputs(root, taxonomy)
    primary_items, primary_key = _blind_primary_items(locked)
    diagnostic_items, diagnostic_key = _diagnostic_items(locked)
    blind_document = _blind_document(locked, primary_items, diagnostic_items)
    _write_immutable(locked["paths"]["blind_items"], blind_document)

    precision_labels = _validate_precision_labels(
        locked["paths"]["precision_labels"], primary_key
    )
    diagnostic_labels = _validate_diagnostic_labels(
        locked["paths"]["diagnostic_labels"], diagnostic_key, taxonomy
    )
    primary = _primary_metrics(precision_labels, primary_key)
    recall, exact_proxy = _recall_metrics(locked)
    diagnostics = _diagnostic_metrics(diagnostic_labels, diagnostic_key)
    completion = _completion_metrics(locked)
    gate = _improvement_gate(primary, recall)
    paths = locked["paths"]
    document = {
        "schema_version": "reviewlens_v15_final_evaluation_v1",
        "status": "complete_locked_evaluation",
        "cohort_label": locked["selection"].get("cohort_label"),
        "selected_review_ids": list(locked["selected_ids"]),
        "input_hashes": {
            "selection_manifest_sha256": _sha256(paths["selection"]),
            "gold_reference_sha256": _sha256(paths["gold"]),
            "run_manifest_sha256": _sha256(paths["run"]),
            "completion_manifest_sha256": _sha256(paths["completion"]),
            "observations_sha256": _sha256(paths["observations"]),
            "claims_sha256": _sha256(paths["claims"]),
            "baseline_v3_sha256": _sha256(paths["baseline"]),
            "blind_items_sha256": _sha256(paths["blind_items"]),
            "precision_labels_sha256": _sha256(paths["precision_labels"]),
            "diagnostic_labels_sha256": _sha256(paths["diagnostic_labels"]),
        },
        "counts": {
            "reviews": len(locked["selected_ids"]),
            "gold_full_tuples": sum(
                len(_gold_signatures(locked["gold_reviews"][review_id]))
                for review_id in locked["selected_ids"]
            ),
            "blind_primary_items": len(primary_items),
            "diagnostic_atomic_claims": len(diagnostic_items),
        },
        "primary_strict_precision": primary,
        "gold_recall": recall,
        "exact_signature_precision_proxy": exact_proxy,
        "atomic_and_abstention_diagnostics": diagnostics,
        "completion_and_failures": completion,
        "improvement_gate": gate,
        "reporting_limits": locked["selection"]["evaluation_protocol"].get(
            "reporting_limits", []
        ),
    }
    paths["evaluation"].write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("\n=== EVIDENCE-FIRST v1.5 FINAL EVALUATION ===")
    print(
        "Strict finding precision: "
        f"v3={primary['v3']['strict_fully_correct']['yes']}/"
        f"{primary['v3']['strict_fully_correct']['total']} "
        f"({primary['v3']['strict_fully_correct']['rate']:.3f}); "
        f"v1.5={primary['evidence_first']['strict_fully_correct']['yes']}/"
        f"{primary['evidence_first']['strict_fully_correct']['total']} "
        f"({primary['evidence_first']['strict_fully_correct']['rate']:.3f})"
    )
    print(
        "Full-tuple recall: "
        f"v3={recall['v3']['full_tuple']['micro']['hits']}/"
        f"{recall['v3']['full_tuple']['micro']['gold']}; "
        f"v1.5={recall['evidence_first']['full_tuple']['micro']['hits']}/"
        f"{recall['evidence_first']['full_tuple']['micro']['gold']}"
    )
    print("Frozen improvement gate:", gate["decision"])
    if show > 0:
        print("Rating-stratified full-tuple recall:")
        for rating in list(recall["v3"]["rating_stratified"])[:show]:
            v3 = recall["v3"]["rating_stratified"][rating]
            ef = recall["evidence_first"]["rating_stratified"][rating]
            print(
                f"  {rating} star: v3={v3['full_tuple_hits']}/"
                f"{v3['full_tuple_gold']} v1.5={ef['full_tuple_hits']}/"
                f"{ef['full_tuple_gold']}"
            )
    print("Saved:", paths["evaluation"])
    print("No models were loaded or called.")
    return document


__all__ = [
    "evaluate_v15_final",
    "prepare_v15_final_evaluation",
]
