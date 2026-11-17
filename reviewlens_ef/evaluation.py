"""Descriptive v3/EF comparison and human-review sheet generation."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PRECISION_HUMAN_COLUMNS = (
    "human_supported",
    "human_category_correct",
    "human_sentiment_correct",
    "human_target_correct",
    "notes",
)
COVERAGE_HUMAN_COLUMNS = (
    "human_missing_feedback_count",
    "missing_feedback_notes",
    "human_coverage_complete",
    "notes",
)


def _read_jsonl(path: Path, required: bool = True) -> list[dict[str, Any]]:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def _stable_id(*parts: Any) -> str:
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return "BLIND_" + hashlib.sha256(payload).hexdigest()[:14].upper()


def _signature(item: dict[str, Any]) -> tuple[str, str, str, str] | None:
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


def _signature_dict(signature: tuple[str, str, str, str]) -> dict[str, str]:
    return dict(zip(("category", "issue_group", "sentiment", "attribution"), signature))


def _quote(evidence: dict[str, Any]) -> str:
    for key in ("evidence_text", "quote", "text"):
        value = evidence.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _evidence_list(item: dict[str, Any]) -> list[dict[str, Any]]:
    evidence = item.get("evidence", []) or []
    if isinstance(evidence, dict):
        evidence = [evidence]
    return [value for value in evidence if isinstance(value, dict)]


def _with_title_sources(
    sentences: list[dict[str, Any]],
    reviews: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Add review titles to the same source index used for evidence checks."""
    sources = [dict(row, source_field=row.get("source_field", "clean_text"))
               for row in sentences]
    existing_ids = {row.get("sentence_id") for row in sources}
    for review in reviews:
        review_id = review.get("review_id")
        title = review.get("title")
        if not isinstance(review_id, str) or not isinstance(title, str) or not title.strip():
            continue
        sentence_id = f"{review_id}_TITLE"
        if sentence_id in existing_ids:
            raise ValueError(f"Synthetic title ID collides with source sentence: {sentence_id}")
        existing_ids.add(sentence_id)
        sources.append({
            "sentence_id": sentence_id,
            "review_id": review_id,
            "sentence_text": title,
            "source_field": "title",
            "start_char": 0,
            "end_char": len(title),
        })
    return sources


def _locator_valid(
    review_id: str,
    item: dict[str, Any],
    sentence_by_id: dict[str, dict[str, Any]],
    review_by_id: dict[str, dict[str, Any]],
) -> bool:
    evidence = _evidence_list(item)
    if not evidence:
        return False
    review = review_by_id.get(review_id, {})
    clean_text = str(review.get("clean_text", ""))
    title = str(review.get("title", ""))
    for value in evidence:
        sentence_id = value.get("sentence_id")
        quote = _quote(value)
        sentence = sentence_by_id.get(sentence_id)
        if (
            not quote
            or sentence is None
            or sentence.get("review_id") != review_id
            or quote not in str(sentence.get("sentence_text", ""))
        ):
            return False
        source_field = value.get("source_field") or sentence.get("source_field") or "clean_text"
        if source_field not in {"clean_text", "title"}:
            return False
        if source_field == "title":
            source_text = title
            start = value.get(
                "quote_start_char_source",
                value.get("start_char_source", value.get("quote_start_char_title")),
            )
            end = value.get(
                "quote_end_char_source",
                value.get("end_char_source", value.get("quote_end_char_title")),
            )
        else:
            source_text = clean_text
            start = value.get(
                "quote_start_char_clean",
                value.get("start_char_clean", value.get("quote_start_char_source")),
            )
            end = value.get(
                "quote_end_char_clean",
                value.get("end_char_clean", value.get("quote_end_char_source")),
            )
        if start is not None or end is not None:
            try:
                if source_text[int(start):int(end)] != quote:
                    return False
            except (TypeError, ValueError):
                return False
    return True


def _json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _human_boolean(value: Any) -> bool | None:
    normalized = str(value or "").strip().casefold()
    if normalized in {"yes", "y", "true", "1"}:
        return True
    if normalized in {"no", "n", "false", "0"}:
        return False
    return None


def _human_evaluation_summary(
    precision_rows: list[dict[str, Any]],
    blind_key: dict[str, dict[str, str]],
    coverage_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Calculate metrics only from explicit human yes/no labels."""
    by_system: dict[str, Any] = {}
    for system in ("v3", "evidence_first"):
        system_rows = [
            row for row in precision_rows
            if blind_key.get(str(row.get("blind_id")), {}).get("system") == system
        ]
        components: dict[str, Any] = {}
        for column in PRECISION_HUMAN_COLUMNS[:-1]:
            labels = [
                label for row in system_rows
                if (label := _human_boolean(row.get(column))) is not None
            ]
            positives = sum(labels)
            components[column.removeprefix("human_")] = {
                "labeled": len(labels),
                "yes": positives,
                "rate": round(positives / len(labels), 4) if labels else None,
            }
        fully_labeled = []
        for row in system_rows:
            labels = [
                _human_boolean(row.get(column))
                for column in PRECISION_HUMAN_COLUMNS[:-1]
            ]
            if all(label is not None for label in labels):
                fully_labeled.append(all(labels))
        by_system[system] = {
            "rows": len(system_rows),
            "components": components,
            "fully_correct": {
                "labeled": len(fully_labeled),
                "yes": sum(fully_labeled),
                "rate": (
                    round(sum(fully_labeled) / len(fully_labeled), 4)
                    if fully_labeled else None
                ),
            },
        }

    coverage_labels = [
        label for row in coverage_rows
        if (label := _human_boolean(row.get("human_coverage_complete"))) is not None
    ]
    reported_missing = []
    for row in coverage_rows:
        raw = str(row.get("human_missing_feedback_count") or "").strip()
        if not raw:
            continue
        try:
            value = int(raw)
        except ValueError:
            continue
        if value >= 0:
            reported_missing.append(value)
    return {
        "claim_precision_by_system": by_system,
        "coverage": {
            "reviews": len(coverage_rows),
            "coverage_complete_labeled": len(coverage_labels),
            "coverage_complete_yes": sum(coverage_labels),
            "coverage_complete_rate": (
                round(sum(coverage_labels) / len(coverage_labels), 4)
                if coverage_labels else None
            ),
            "missing_feedback_count_labeled": len(reported_missing),
            "reported_missing_feedback_total": sum(reported_missing),
        },
        "label_note": "Unrecognized or blank human values are excluded from rates.",
    }


def _read_preserved(
    path: Path, key_column: str, human_columns: Iterable[str]
) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return {
            row[key_column]: {column: row.get(column, "") for column in human_columns}
            for row in reader
            if row.get(key_column)
        }


def _render_csv(rows: list[dict[str, Any]], fieldnames: list[str]) -> str:
    import io

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
    existing = path.read_text(encoding="utf-8-sig") if path.exists() else None
    # utf-8-sig strips the BOM, so normalise before comparing.
    normalised = content.removeprefix("\ufeff")
    if existing == normalised:
        return
    if path.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup = path.with_name(f"{path.stem}.backup_{stamp}{path.suffix}")
        shutil.copy2(path, backup)
    path.write_text(content, encoding="utf-8")


def _nli_index(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    index = {}
    for row in rows:
        review_id, claim_id = row.get("review_id"), row.get("claim_id")
        if isinstance(review_id, str) and isinstance(claim_id, str):
            index[(review_id, claim_id)] = row
    return index


def _precision_rows(
    selected_ids: list[str],
    baseline: dict[str, dict[str, Any]],
    claims: dict[str, dict[str, Any]],
    nli: dict[tuple[str, str], dict[str, Any]],
    sentence_by_id: dict[str, dict[str, Any]],
    review_by_id: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]]]:
    rows = []
    blind_key = {}
    for review_id in selected_ids:
        systems = (
            ("v3", baseline.get(review_id, {}).get("accepted_feedback", []) or []),
            ("evidence_first", claims.get(review_id, {}).get("accepted_claims", []) or []),
        )
        for system, items in systems:
            for index, item in enumerate(items, 1):
                if not isinstance(item, dict):
                    continue
                item_id = item.get("claim_id") or f"{review_id}_{system}_{index:03d}"
                blind_id = _stable_id(system, review_id, item_id, _signature(item), item.get("claim") or item.get("summary"))
                blind_key[blind_id] = {
                    "system": system,
                    "review_id": review_id,
                    "item_id": str(item_id),
                }
                nli_row = nli.get((review_id, str(item_id)), {}) if system == "evidence_first" else {}
                components = nli_row.get("components", {}) or {}
                rows.append({
                    "blind_id": blind_id,
                    "review_id": review_id,
                    "mapping_status": item.get("mapping_status", "mapped"),
                    "category": item.get("category"),
                    "issue_group": item.get("issue_group"),
                    "sentiment": item.get("sentiment") or item.get("polarity"),
                    "attribution": item.get("attribution") or item.get("target"),
                    "statement_type": item.get(
                        "statement_type",
                        "legacy_unspecified" if system == "v3" else "",
                    ),
                    "claim": item.get("claim") or item.get("summary"),
                    "evidence_quotes": " || ".join(
                        _quote(value) for value in _evidence_list(item) if _quote(value)
                    ),
                    "locator_valid": _locator_valid(
                        review_id, item, sentence_by_id, review_by_id
                    ),
                    "deterministic_status": item.get(
                        "deterministic_status",
                        "legacy_candidate_unverified" if system == "v3" else "accepted",
                    ),
                    "deterministic_reasons": _json_cell(item.get("deterministic_reasons", [])),
                    "normalization_source": item.get(
                        "normalization_source",
                        "legacy_model" if system == "v3" else "model",
                    ),
                    "normalization_adjustments": _json_cell(
                        item.get("normalization_adjustments", [])
                    ),
                    "nli_status": nli_row.get("advisory_status", "not_run"),
                    "nli_claim_label": (components.get("claim") or {}).get("label", ""),
                    "nli_sentiment_label": (components.get("sentiment") or {}).get("label", ""),
                    "nli_target_label": (components.get("target") or {}).get("label", ""),
                })
    rows.sort(key=lambda row: row["blind_id"])
    return rows, blind_key


def summarize(project_root: str | Path, show: int = 10) -> dict[str, Any]:
    """Compare the same pilot reviews and create two human evaluation sheets.

    Counts and tuple differences are descriptive only.  Accuracy and coverage
    are deliberately left for human labels in the generated CSV files.
    """
    root = Path(project_root).resolve()
    ef_dir = root / "outputs/amazon/evidence_first"
    observation_rows = _read_jsonl(ef_dir / "observations_pilot.jsonl")
    claim_rows = _read_jsonl(ef_dir / "claims_pilot.jsonl")
    nli_rows = _read_jsonl(ef_dir / "nli_pilot.jsonl", required=False)
    baseline_rows = _read_jsonl(root / "outputs/amazon/acer_aspire5_business_full_v3.jsonl")
    reviews = _read_jsonl(root / "data/processed/amazon/acer_aspire5_reviews_clean.jsonl")
    sentences = _read_jsonl(root / "data/processed/amazon/acer_aspire5_sentences.jsonl")

    observations = {row["review_id"]: row for row in observation_rows}
    claims = {row["review_id"]: row for row in claim_rows}
    baseline = {row["review_id"]: row for row in baseline_rows}
    review_by_id = {row["review_id"]: row for row in reviews}
    sources = _with_title_sources(sentences, reviews)
    sentence_by_id = {row["sentence_id"]: row for row in sources}
    nli = _nli_index(nli_rows)
    selected_ids = list(dict.fromkeys(
        [row["review_id"] for row in claim_rows]
        + [row["review_id"] for row in observation_rows]
    ))
    if not selected_ids:
        raise ValueError("Evidence-first pilot caches contain no reviews")
    missing_baseline = [review_id for review_id in selected_ids if review_id not in baseline]
    missing_reviews = [review_id for review_id in selected_ids if review_id not in review_by_id]
    if missing_baseline or missing_reviews:
        raise ValueError(
            f"Pilot alignment failed: missing baseline={missing_baseline}, "
            f"missing source reviews={missing_reviews}"
        )

    comparison_rows = []
    totals = Counter()
    for review_id in selected_ids:
        old_items = baseline[review_id].get("accepted_feedback", []) or []
        claim_row = claims.get(review_id, {})
        new_items = claim_row.get("accepted_claims", []) or []
        merged_present = "merged_claims" in claim_row
        merged_items = claim_row.get("merged_claims", []) or []
        business_items = merged_items if merged_present else new_items
        accepted_observations = observations.get(review_id, {}).get("accepted_observations", []) or []
        old_signatures = {signature for item in old_items if (signature := _signature(item))}
        new_signatures = {
            signature for item in business_items if (signature := _signature(item))
        }
        mapped = sum(item.get("mapping_status", "mapped") == "mapped" for item in new_items)
        unmapped = sum(item.get("mapping_status") == "unmapped" for item in new_items)
        duplicate_claims_merged = int(
            claim_row.get(
                "duplicate_claims_merged",
                max(mapped - len(new_signatures), 0),
            )
        )
        fallback_unmapped = int(claim_row.get(
            "fallback_unmapped_count",
            sum(
                item.get("normalization_source") == "deterministic_unmapped_fallback"
                for item in new_items
            ),
        ))
        deterministic_recoveries = int(claim_row.get(
            "deterministic_recovery_count",
            sum(
                item.get("normalization_source")
                == "deterministic_high_precision_recovery"
                for item in new_items
            ),
        ))
        category_canonicalizations = sum(
            any(
                adjustment.get("type")
                == "canonicalized_unique_issue_group_parent"
                for adjustment in item.get("normalization_adjustments", []) or []
                if isinstance(adjustment, dict)
            )
            for item in new_items
        )
        taxonomy_repairs = sum(
            any(
                adjustment.get("type") in {
                    "canonicalized_unique_issue_group_parent",
                    "canonicalized_power_port_to_charging",
                }
                for adjustment in item.get("normalization_adjustments", []) or []
                if isinstance(adjustment, dict)
            )
            for item in new_items
        )
        rejected_observations = len(observations.get(review_id, {}).get("rejected_observations", []) or [])
        rejected_claims = len(claims.get(review_id, {}).get("rejected_claims", []) or [])
        statuses = Counter(
            row.get("advisory_status", "not_run")
            for (rid, _), row in nli.items() if rid == review_id
        )
        row = {
            "review_id": review_id,
            "rating": review_by_id[review_id].get("rating"),
            "baseline_claim_count": len(old_items),
            "baseline_unique_issue_tuple_count": len(old_signatures),
            "ef_observation_count": len(accepted_observations),
            "ef_atomic_claim_count": len(new_items),
            "ef_mapped_atomic_claim_count": mapped,
            "ef_unmapped_atomic_claim_count": unmapped,
            "ef_business_finding_count": len(new_signatures),
            "ef_unique_mapped_issue_tuple_count": len(new_signatures),
            "duplicate_claims_merged": duplicate_claims_merged,
            "ef_fallback_unmapped_count": fallback_unmapped,
            "ef_deterministic_recovery_count": deterministic_recoveries,
            "ef_category_canonicalization_count": category_canonicalizations,
            "ef_taxonomy_repair_count": taxonomy_repairs,
            # Backward-compatible aliases for existing consumers.
            "ef_claim_count": len(new_items),
            "ef_mapped_count": mapped,
            "ef_unmapped_count": unmapped,
            "deterministic_rejected_observations": rejected_observations,
            "deterministic_rejected_claims": rejected_claims,
            "shared_issue_tuples": [_signature_dict(value) for value in sorted(old_signatures & new_signatures)],
            "baseline_only_issue_tuples": [_signature_dict(value) for value in sorted(old_signatures - new_signatures)],
            "ef_only_issue_tuples": [_signature_dict(value) for value in sorted(new_signatures - old_signatures)],
            "nli_advisory_counts": dict(statuses),
        }
        comparison_rows.append(row)
        for key in (
            "baseline_claim_count", "baseline_unique_issue_tuple_count",
            "ef_observation_count",
            "ef_atomic_claim_count", "ef_mapped_atomic_claim_count",
            "ef_unmapped_atomic_claim_count", "ef_business_finding_count",
            "ef_unique_mapped_issue_tuple_count", "duplicate_claims_merged",
            "ef_fallback_unmapped_count", "ef_deterministic_recovery_count",
            "ef_category_canonicalization_count",
            "ef_taxonomy_repair_count",
            "ef_claim_count", "ef_mapped_count", "ef_unmapped_count",
            "deterministic_rejected_observations", "deterministic_rejected_claims",
        ):
            totals[key] += row[key]
        totals["shared_issue_tuples"] += len(row["shared_issue_tuples"])
        totals["baseline_only_issue_tuples"] += len(row["baseline_only_issue_tuples"])
        totals["ef_only_issue_tuples"] += len(row["ef_only_issue_tuples"])

    precision_rows, blind_key = _precision_rows(
        selected_ids, baseline, claims, nli, sentence_by_id, review_by_id
    )
    coverage_rows = []
    for review_id in selected_ids:
        old_items = baseline[review_id].get("accepted_feedback", []) or []
        observation_items = observations.get(review_id, {}).get("accepted_observations", []) or []
        new_items = claims.get(review_id, {}).get("accepted_claims", []) or []
        merged_items = claims.get(review_id, {}).get("merged_claims", []) or []
        coverage_rows.append({
            "review_id": review_id,
            "rating": review_by_id[review_id].get("rating"),
            "review_title": review_by_id[review_id].get("title"),
            "review_text": review_by_id[review_id].get("clean_text"),
            "baseline_claim_count": len(old_items),
            "baseline_claims": _json_cell([
                {
                    "category": item.get("category"),
                    "issue_group": item.get("issue_group"),
                    "sentiment": item.get("sentiment"),
                    "attribution": item.get("attribution"),
                    "summary": item.get("summary"),
                }
                for item in old_items if isinstance(item, dict)
            ]),
            "ef_observation_count": len(observation_items),
            "ef_observations": _json_cell(observation_items),
            "ef_mapped_claim_count": sum(item.get("mapping_status", "mapped") == "mapped" for item in new_items),
            "ef_unmapped_claim_count": sum(item.get("mapping_status") == "unmapped" for item in new_items),
            "ef_fallback_unmapped_count": int(
                claims.get(review_id, {}).get("fallback_unmapped_count", 0)
            ),
            "ef_deterministic_recovery_count": int(
                claims.get(review_id, {}).get(
                    "deterministic_recovery_count",
                    sum(
                        item.get("normalization_source")
                        == "deterministic_high_precision_recovery"
                        for item in new_items
                    ),
                )
            ),
            "ef_claims": _json_cell(new_items),
            "ef_business_finding_count": len([
                item for item in merged_items if _signature(item)
            ]) if "merged_claims" in claims.get(review_id, {}) else len({
                signature for item in new_items if (signature := _signature(item))
            }),
            "ef_merged_claims": _json_cell(merged_items),
        })

    ef_dir.mkdir(parents=True, exist_ok=True)
    precision_path = ef_dir / "claim_precision_review.csv"
    coverage_path = ef_dir / "extraction_coverage_review.csv"
    precision_fields = [
        "blind_id", "review_id", "mapping_status", "category",
        "issue_group", "sentiment", "attribution", "statement_type", "claim",
        "evidence_quotes",
        "locator_valid", "deterministic_status", "deterministic_reasons",
        "normalization_source", "normalization_adjustments",
        "nli_status", "nli_claim_label", "nli_sentiment_label", "nli_target_label",
        *PRECISION_HUMAN_COLUMNS,
    ]
    coverage_fields = [
        "review_id", "rating", "review_title", "review_text", "baseline_claim_count",
        "baseline_claims", "ef_observation_count", "ef_observations",
        "ef_mapped_claim_count", "ef_unmapped_claim_count", "ef_claims",
        "ef_fallback_unmapped_count", "ef_deterministic_recovery_count",
        "ef_business_finding_count", "ef_merged_claims",
        *COVERAGE_HUMAN_COLUMNS,
    ]
    _write_csv_preserving(
        precision_path, precision_rows, precision_fields, "blind_id",
        PRECISION_HUMAN_COLUMNS,
    )
    _write_csv_preserving(
        coverage_path, coverage_rows, coverage_fields, "review_id",
        COVERAGE_HUMAN_COLUMNS,
    )
    human_evaluation = _human_evaluation_summary(
        precision_rows, blind_key, coverage_rows
    )

    comparison = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "descriptive_comparison_requires_human_evaluation",
        "pilot_reviews": len(selected_ids),
        "totals": dict(totals),
        "reviews": comparison_rows,
        "precision_blind_key": blind_key,
        "human_evaluation": human_evaluation,
        "notes": [
            "Tuple differences are descriptive and are not accuracy metrics.",
            "EF atomic claims are evaluated individually; business findings are merged by review-level issue tuple.",
            "duplicate_claims_merged prevents evidence splitting from looking like additional business issues.",
            "Fallback unmapped claims preserve evidence after invalid model normalizations; they are not mapped findings.",
            "Deterministic recoveries require one explicit evidence-only taxonomy pair and are reported separately for human review.",
            "Category canonicalizations are recorded deterministic repairs of a gated, uniquely owned issue group.",
            "Complete claim_precision_review.csv before calculating precision.",
            "Complete extraction_coverage_review.csv before judging coverage.",
            "NLI labels are advisory and not human ground truth.",
        ],
    }
    comparison_path = ef_dir / "comparison_v3_vs_ef.json"
    comparison_path.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n=== REVIEWLENS: LAYER 2 v3 vs EVIDENCE-FIRST PILOT ===")
    print("Aligned pilot reviews:", len(selected_ids))
    print(
        "Baseline candidates / unique issue tuples:",
        totals["baseline_claim_count"], "/",
        totals["baseline_unique_issue_tuple_count"],
    )
    print(
        "EF observations / atomic claims:",
        totals["ef_observation_count"], "/", totals["ef_atomic_claim_count"],
    )
    print(
        "EF atomic mapped / unmapped:",
        totals["ef_mapped_atomic_claim_count"], "/",
        totals["ef_unmapped_atomic_claim_count"],
    )
    print(
        "EF review-level business findings: "
        f"{totals['ef_business_finding_count']} "
        f"(duplicate atomic claims merged: {totals['duplicate_claims_merged']})"
    )
    print(
        "EF deterministic fallbacks / taxonomy repairs / recoveries:",
        totals["ef_fallback_unmapped_count"], "/",
        totals["ef_taxonomy_repair_count"], "/",
        totals["ef_deterministic_recovery_count"],
    )
    print("Descriptive tuple differences:", {
        key: totals[key] for key in (
            "shared_issue_tuples", "baseline_only_issue_tuples", "ef_only_issue_tuples"
        )
    })
    for row in comparison_rows[:max(0, show)]:
        print(
            f"  {row['review_id']}: v3={row['baseline_claim_count']} "
            f"observations={row['ef_observation_count']} "
            f"atomic={row['ef_atomic_claim_count']} "
            f"findings={row['ef_business_finding_count']} "
            f"merged={row['duplicate_claims_merged']} "
            f"fallbacks={row['ef_fallback_unmapped_count']} "
            f"recoveries={row['ef_deterministic_recovery_count']} "
            f"unmapped={row['ef_unmapped_atomic_claim_count']} "
            f"shared={len(row['shared_issue_tuples'])}"
        )
    print("Comparison:", comparison_path)
    print("Claim precision review:", precision_path)
    print("Extraction coverage review:", coverage_path)
    if any(
        metrics["components"]["supported"]["labeled"]
        for metrics in human_evaluation["claim_precision_by_system"].values()
    ):
        print("Human-label metrics:", human_evaluation)
    print("No improvement claim is made until the human columns are completed.")
    return comparison


__all__ = ["summarize"]
