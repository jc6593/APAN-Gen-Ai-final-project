"""Locked residual validation cohort for the frozen Evidence-first v1.5 model.

This module is intentionally separate from ``v15_dev``.  It selects an
untouched, rating-balanced cohort using only rating and review length, freezes
the model/evaluation fingerprints before inference, and permits resume but not
regeneration or partial peeking.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import core as core_module
from . import pipeline as pipeline_module
from . import prompts as prompts_module
from . import v15_core as v15_core_module
from . import v15_dev as v15_dev_module
from . import v15_prompts as v15_prompts_module
from .core import (
    PILOT_REVIEW_IDS,
    append_jsonl,
    hash_json,
    hash_text,
    merge_claims_by_tuple,
    read_jsonl,
    stage_record_hash,
    taxonomy_hash,
)
from .pipeline import (
    Generator,
    _MlxGenerator,
    _claim_record_hash,
    _observation_payload,
    _parse_claim_attempt,
    _prepare_inputs,
    _run_with_optional_repair,
    _source_rows,
)
from .v15_core import (
    CLAIM_SCHEMA_VERSION_V15,
    OBSERVATION_SCHEMA_VERSION_V15,
    POLICY_VERSION_V15,
    complete_claims_v15,
    normalization_payload_v15,
    parse_observation_attempt_v15,
)
from .v15_dev import LEGACY_PILOT_REVIEW_IDS, _empty_claim_row_v15
from .v15_prompts import OBSERVATION_PROMPT_V15, build_normalization_prompt_v15


FINAL_SIZE = 15
FINAL_SEED = 5562
FINAL_RATING_TARGETS = {1: 3, 2: 3, 3: 3, 4: 3, 5: 3}
FINAL_SELECTION_ALGORITHM = "rating_length_only_locked_v15_final_v1"
FINAL_MAX_RETRIES = 1
FINAL_MAX_TOKENS = 4096
FINAL_TEMPERATURE = 0.0

EVALUATION_PROTOCOL = {
    "protocol_version": "reviewlens_v15_final_eval_v1",
    "cohort_label": "internal_locked_validation_not_external_test",
    "primary_metric": {
        "name": "strict_review_level_business_finding_precision",
        "correct_when": [
            "supported=yes",
            "category_correct=yes",
            "sentiment_correct=yes",
            "target_correct=yes",
        ],
        "denominator": "every_generated_review_level_finding",
        "require_complete_labels": True,
    },
    "gold_protocol": {
        "created_from_source_before_system_predictions": True,
        "tuple": ["category", "issue_group", "sentiment", "attribution"],
        "shortest_verbatim_evidence": True,
        "deduplicate_same_tuple_and_sentiment": True,
        "opposite_sentiments_remain_separate": True,
        "exclude_generic_or_inferred_or_wrong_target_content": True,
    },
    "secondary_metrics": [
        "end_to_end_full_tuple_recall_micro_and_review_macro",
        "issue_pair_recall_micro_and_review_macro",
        "supported_category_sentiment_target_component_precision",
        "atomic_claim_fully_correct_rate",
        "abstention_correctness_and_should_map_rate",
        "generation_completion_and_failure_count",
        "rating_stratified_descriptive_counts",
    ],
    "improvement_rule": (
        "Only claim improvement when strict finding precision exceeds v3 and "
        "end-to-end full-tuple recall is not lower."
    ),
    "reporting_limits": [
        "report raw numerators and denominators",
        "no significance or population-prevalence claim from n=15",
        "blind and shuffle system identity for v3 versus v1.5 annotation",
        "generate annotation sheets only after all 15 final claims complete",
        "no tuning after final inference",
    ],
}


def _paths(project_root: Path) -> dict[str, Path]:
    output_dir = project_root / "outputs/amazon/evidence_first/v15_final"
    dev_dir = project_root / "outputs/amazon/evidence_first/v15_dev"
    holdout_dir = project_root / "outputs/amazon/evidence_first/holdout"
    return {
        "output_dir": output_dir,
        "selection": output_dir / "selection_manifest.json",
        "gold_reference": output_dir / "gold_reference.json",
        "run_manifest": output_dir / "run_manifest.json",
        "observations": output_dir / "observations_v15_final.jsonl",
        "claims": output_dir / "claims_v15_final.jsonl",
        "completion": output_dir / "completion_manifest.json",
        "reviews": project_root / "data/processed/amazon/acer_aspire5_reviews_clean.jsonl",
        "sentences": project_root / "data/processed/amazon/acer_aspire5_sentences.jsonl",
        "layer1": project_root / "outputs/amazon/acer_aspire5_full_predictions.jsonl",
        "baseline_v3": project_root / "outputs/amazon/acer_aspire5_business_full_v3.jsonl",
        "ef_pilot_manifest": (
            project_root / "outputs/amazon/evidence_first/pilot_manifest.json"
        ),
        "legacy_pilot_v3": (
            project_root / "outputs/amazon/acer_aspire5_business_pilot_v3.jsonl"
        ),
        "holdout_selection": holdout_dir / "selection_manifest.json",
        "dev_selection": dev_dir / "selection_manifest.json",
        "dev_run_manifest": dev_dir / "run_manifest.json",
        "dev_observations": dev_dir / "observations_v15_dev.jsonl",
        "dev_claims": dev_dir / "claims_v15_dev.jsonl",
    }


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path, label: str) -> dict[str, Any]:
    if not path.exists():
        raise ValueError(f"Missing {label}: {path}")
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Invalid {label}: {path}")
    return value


def _rating(value: Any) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid rating: {value}") from exc
    rating = int(numeric)
    if numeric != rating or rating not in FINAL_RATING_TARGETS:
        raise ValueError(f"Invalid rating: {value}")
    return rating


def _tie(seed: int, context: str, review_id: str) -> str:
    return hashlib.sha256(
        f"{seed}|{context}|{review_id}".encode("utf-8")
    ).hexdigest()


def _model_code_hashes() -> dict[str, str]:
    files = {
        "reviewlens_ef/core.py": Path(str(core_module.__file__)),
        "reviewlens_ef/pipeline.py": Path(str(pipeline_module.__file__)),
        "reviewlens_ef/prompts.py": Path(str(prompts_module.__file__)),
        "reviewlens_ef/v15_core.py": Path(str(v15_core_module.__file__)),
        "reviewlens_ef/v15_dev.py": Path(str(v15_dev_module.__file__)),
        "reviewlens_ef/v15_prompts.py": Path(str(v15_prompts_module.__file__)),
        "reviewlens_ef/v15_final.py": Path(__file__),
    }
    return {name: _file_hash(path.resolve()) for name, path in files.items()}


def _validate_completed_dev(paths: Mapping[str, Path]) -> dict[str, Any]:
    selection = _load_object(paths["dev_selection"], "v1.5 dev selection")
    run_manifest = _load_object(paths["dev_run_manifest"], "v1.5 dev run manifest")
    selected_rows = selection.get("selected_reviews")
    if not isinstance(selected_rows, list) or len(selected_rows) != 20:
        raise ValueError("V1.5 dev selection must contain 20 reviews")
    selected_ids = [row.get("review_id") for row in selected_rows]
    if not all(isinstance(value, str) for value in selected_ids):
        raise ValueError("Invalid v1.5 dev review IDs")
    if run_manifest.get("selection_manifest_hash") != hash_json(selection):
        raise ValueError("V1.5 dev run manifest does not match its selection")
    if run_manifest.get("selected_review_ids") != selected_ids:
        raise ValueError("V1.5 dev run manifest review order mismatch")
    current_dev_hashes = v15_dev_module._model_code_hashes()
    if run_manifest.get("model_code_hashes") != current_dev_hashes:
        raise ValueError(
            "The completed v1.5 dev run was produced by different model code"
        )

    observation_rows = read_jsonl(paths["dev_observations"])
    claim_rows = read_jsonl(paths["dev_claims"])
    if [row.get("review_id") for row in observation_rows] != selected_ids:
        raise ValueError("V1.5 dev observation cache is not complete and ordered")
    if [row.get("review_id") for row in claim_rows] != selected_ids:
        raise ValueError("V1.5 dev claim cache is not complete and ordered")
    observations_by_id = {row["review_id"]: row for row in observation_rows}
    for row in observation_rows:
        if row.get("record_hash") != stage_record_hash(row, "accepted_observations"):
            raise ValueError(f"Corrupt v1.5 dev observation: {row.get('review_id')}")
    for row in claim_rows:
        review_id = row.get("review_id")
        if row.get("record_hash") != _claim_record_hash(row):
            raise ValueError(f"Corrupt v1.5 dev claim: {review_id}")
        if row.get("upstream_hash") != observations_by_id[review_id].get("record_hash"):
            raise ValueError(f"V1.5 dev upstream mismatch: {review_id}")

    return {
        "selection": selection,
        "run_manifest": run_manifest,
        "selected_review_ids": selected_ids,
        "artifact_hashes": {
            "selection_manifest": _file_hash(paths["dev_selection"]),
            "run_manifest": _file_hash(paths["dev_run_manifest"]),
            "observations": _file_hash(paths["dev_observations"]),
            "claims": _file_hash(paths["dev_claims"]),
        },
    }


def _manifest_review_ids(document: Mapping[str, Any], label: str) -> list[str]:
    rows = document.get("selected_reviews")
    if not isinstance(rows, list):
        raise ValueError(f"Invalid {label} selected_reviews")
    ids = [row.get("review_id") for row in rows if isinstance(row, Mapping)]
    if len(ids) != len(rows) or not all(isinstance(value, str) for value in ids):
        raise ValueError(f"Invalid {label} review IDs")
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate {label} review IDs")
    return [str(value) for value in ids]


def _choose_rating_records(
    rows: Sequence[Mapping[str, Any]],
    rating: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    target = FINAL_RATING_TARGETS[rating]
    ordered = sorted(
        (dict(row) for row in rows),
        key=lambda row: (
            row["review_char_length"],
            _tie(seed, f"rating-{rating}-length-order", row["review_id"]),
        ),
    )
    bins: list[list[dict[str, Any]]] = [[] for _ in range(target)]
    for index, row in enumerate(ordered):
        slot = min(index * target // len(ordered), target - 1)
        bins[slot].append(row)
    selected: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for slot, options in enumerate(bins):
        if not options:
            raise ValueError(f"Empty final length bin: rating={rating}, slot={slot}")
        chosen = min(
            options,
            key=lambda row: _tie(
                seed,
                f"rating-{rating}-slot-{slot}-choice",
                row["review_id"],
            ),
        )
        item = dict(chosen)
        item.update({
            "length_bin": slot,
            "length_bin_size": len(options),
            "sampling_weight_for_eligible_pool": len(options),
        })
        selected.append(item)
        summaries.append({
            "rating": rating,
            "length_bin": slot,
            "bin_size": len(options),
            "min_review_char_length": min(row["review_char_length"] for row in options),
            "max_review_char_length": max(row["review_char_length"] for row in options),
        })
    return selected, summaries


def selection_document_v15_final(
    project_root: str | Path,
    seed: int = FINAL_SEED,
) -> dict[str, Any]:
    """Build the deterministic final selection without writing or inference."""
    root = Path(project_root).resolve()
    paths = _paths(root)
    dev = _validate_completed_dev(paths)
    holdout = _load_object(paths["holdout_selection"], "diagnostic holdout selection")
    holdout_ids = _manifest_review_ids(holdout, "diagnostic holdout")
    if len(holdout_ids) != 25:
        raise ValueError("Diagnostic holdout must contain 25 reviews")

    reviews = read_jsonl(paths["reviews"])
    review_ids = [row.get("review_id") for row in reviews]
    if len(reviews) != 189 or len(set(review_ids)) != 189:
        raise ValueError("Expected 189 unique source reviews")
    review_by_id = {str(row["review_id"]): row for row in reviews}
    dev_ids = dev["selected_review_ids"]
    exclusion_sets = {
        "evidence_first_pilot": set(PILOT_REVIEW_IDS),
        "legacy_pilot": set(LEGACY_PILOT_REVIEW_IDS),
        "diagnostic_holdout": set(holdout_ids),
        "v15_dev": set(dev_ids),
    }
    exclusion_union = set().union(*exclusion_sets.values())
    pair_overlaps = {
        f"{left}__{right}": sorted(exclusion_sets[left] & exclusion_sets[right])
        for index, left in enumerate(exclusion_sets)
        for right in list(exclusion_sets)[index + 1 :]
    }
    eligible = [
        {
            "review_id": review_id,
            "rating": _rating(review["rating"]),
            "review_char_length": len(str(review.get("clean_text", ""))),
        }
        for review_id, review in sorted(review_by_id.items())
        if review_id not in exclusion_union
    ]
    eligible_counts = Counter(row["rating"] for row in eligible)
    if any(eligible_counts[rating] < target for rating, target in FINAL_RATING_TARGETS.items()):
        raise ValueError(f"Insufficient final candidates: {dict(eligible_counts)}")

    chosen_by_rating: dict[int, list[dict[str, Any]]] = {}
    bin_summaries: list[dict[str, Any]] = []
    for rating in range(1, 6):
        chosen, summaries = _choose_rating_records(
            [row for row in eligible if row["rating"] == rating],
            rating,
            seed,
        )
        chosen_by_rating[rating] = chosen
        bin_summaries.extend(summaries)
    selected_reviews = [
        chosen_by_rating[rating][slot]
        for slot in range(3)
        for rating in range(1, 6)
    ]
    selected_ids = [row["review_id"] for row in selected_reviews]
    if len(selected_ids) != FINAL_SIZE or len(set(selected_ids)) != FINAL_SIZE:
        raise AssertionError("Final selection must contain 15 unique reviews")
    if set(selected_ids) & exclusion_union:
        raise AssertionError("Final selection overlaps a prior cohort")
    if Counter(row["rating"] for row in selected_reviews) != Counter(FINAL_RATING_TARGETS):
        raise AssertionError("Final selection is not rating-balanced")

    exact_text_hashes = [
        hash_text(str(review_by_id[review_id].get("clean_text", "")))
        for review_id in review_by_id
    ]
    if len(exact_text_hashes) != len(set(exact_text_hashes)):
        raise ValueError("Exact duplicate clean_text detected in source reviews")

    source_hashes = {
        "reviews": _file_hash(paths["reviews"]),
        "sentences": _file_hash(paths["sentences"]),
        "layer1": _file_hash(paths["layer1"]),
        "baseline_v3": _file_hash(paths["baseline_v3"]),
        "evidence_first_pilot_manifest": _file_hash(paths["ef_pilot_manifest"]),
        "legacy_pilot_v3": _file_hash(paths["legacy_pilot_v3"]),
        "diagnostic_holdout_selection": _file_hash(paths["holdout_selection"]),
        **{f"v15_dev_{key}": value for key, value in dev["artifact_hashes"].items()},
    }
    model_hashes = _model_code_hashes()
    dev_run_manifest = dev["run_manifest"]
    frozen_model_id = dev_run_manifest.get("model_id")
    if not isinstance(frozen_model_id, str) or not frozen_model_id:
        raise ValueError("Completed v1.5 dev run has no valid model_id")
    frozen_model_revision = _local_model_revision(frozen_model_id)
    frozen_snapshot = _local_model_snapshot(
        frozen_model_id,
        frozen_model_revision,
    )
    artifact_record_counts = {
        "reviews": len(reviews),
        "sentences": len(read_jsonl(paths["sentences"])),
        "layer1": len(read_jsonl(paths["layer1"])),
        "baseline_v3": len(read_jsonl(paths["baseline_v3"])),
        "legacy_pilot_v3": len(read_jsonl(paths["legacy_pilot_v3"])),
        "v15_dev_observations": 20,
        "v15_dev_claims": 20,
    }
    return {
        "schema_version": "reviewlens_v15_final_selection_v1",
        "status": "frozen_after_dev_before_final_inference",
        "cohort_label": "internal_locked_validation_not_external_test",
        "algorithm": FINAL_SELECTION_ALGORITHM,
        "selection_features": ["review_id", "rating", "len(clean_text)"],
        "seed": seed,
        "tie_breaker": "sha256(seed|context|review_id)",
        "length_bin_formula": "min(index * 3 // eligible_rating_count, 2)",
        "source_review_count": len(reviews),
        "source_rating_distribution": {
            str(rating): count
            for rating, count in sorted(Counter(
                _rating(row["rating"]) for row in reviews
            ).items())
        },
        "excluded_review_ids": {
            name: sorted(values) for name, values in exclusion_sets.items()
        },
        "exclusion_pair_overlaps": pair_overlaps,
        "exclusion_union_count": len(exclusion_union),
        "eligible_review_count": len(eligible),
        "eligible_rating_distribution": {
            str(rating): eligible_counts[rating] for rating in range(1, 6)
        },
        "eligible_pool_hash": hash_json(eligible),
        "scarcity_rationale": (
            "Three per rating is the largest balanced cohort with no prior v1.5 "
            "inference: only three eligible 2-star and three eligible 3-star "
            "reviews remain."
        ),
        "rating_targets": {str(key): value for key, value in FINAL_RATING_TARGETS.items()},
        "length_bins": bin_summaries,
        "selected_reviews": selected_reviews,
        "selected_review_ids_hash": hash_json(selected_ids),
        "source_artifact_hashes": source_hashes,
        "source_artifact_record_counts": artifact_record_counts,
        "frozen_model_code_hashes": model_hashes,
        "frozen_model_id": frozen_model_id,
        "frozen_model_revision": frozen_model_revision,
        "frozen_model_snapshot_path": str(frozen_snapshot),
        "frozen_taxonomy_hash": dev_run_manifest.get("taxonomy_hash"),
        "frozen_observation_prompt_hash": dev_run_manifest.get(
            "observation_prompt_hash"
        ),
        "frozen_claim_prompt_hash": dev_run_manifest.get("claim_prompt_hash"),
        "frozen_observation_schema_version": dev_run_manifest.get(
            "observation_schema_version"
        ),
        "frozen_claim_schema_version": dev_run_manifest.get(
            "claim_schema_version"
        ),
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "evaluation_protocol_hash": hash_json(EVALUATION_PROTOCOL),
        "reporting_note": (
            "Balanced internal validation over the 135-review residual pool; "
            "not representative of the original 189-review prevalence."
        ),
    }


def _without_created_at(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "created_at_utc"}


def prepare_v15_final(
    project_root: str | Path,
    seed: int = FINAL_SEED,
) -> dict[str, Any]:
    """Freeze or verify the final selection.  This never loads a model."""
    root = Path(project_root).resolve()
    paths = _paths(root)
    expected = selection_document_v15_final(root, seed)
    if paths["selection"].exists():
        existing = _load_object(paths["selection"], "v1.5 final selection")
        if _without_created_at(existing) != expected:
            raise ValueError(
                "Existing final selection is immutable and no longer matches "
                "its source, exclusion, code, or protocol fingerprints."
            )
        document = existing
        status = "already frozen and verified"
    else:
        unexpected = [
            paths[key]
            for key in (
                "gold_reference", "run_manifest", "observations", "claims", "completion"
            )
            if paths[key].exists()
        ]
        if unexpected:
            raise ValueError(
                "Final inference artifacts exist without a selection manifest: "
                f"{unexpected}"
            )
        paths["output_dir"].mkdir(parents=True, exist_ok=True)
        document = dict(expected)
        document["created_at_utc"] = datetime.now(timezone.utc).isoformat()
        temporary = paths["selection"].with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(paths["selection"])
        status = "frozen before final inference"
    print("\n=== EVIDENCE-FIRST v1.5 FINAL PREPARE ===")
    print(f"Selection {status}: {paths['selection']}")
    print("Reviews: 15; exactly 3 per star; no prior Evidence-first v1.5 inference")
    print("Evaluation protocol and frozen model hashes are stored in the manifest.")
    print("No model was loaded or called.")
    return document


def _validated_selection(root: Path) -> dict[str, Any]:
    paths = _paths(root)
    if not paths["selection"].exists():
        raise ValueError(
            "Final selection is not frozen. Run --mode ef-v15-final-prepare first."
        )
    existing = _load_object(paths["selection"], "v1.5 final selection")
    seed = existing.get("seed")
    if not isinstance(seed, int):
        raise ValueError("Invalid final selection seed")
    expected = selection_document_v15_final(root, seed)
    if _without_created_at(existing) != expected:
        raise ValueError("Frozen final selection failed immutability validation")
    return existing


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _local_model_revision(model_id: str) -> str:
    repository = Path.home() / ".cache/huggingface/hub" / (
        "models--" + model_id.replace("/", "--")
    )
    refs_main = repository / "refs/main"
    if refs_main.exists():
        value = refs_main.read_text(encoding="utf-8").strip()
        if value:
            return value
    snapshots = sorted(
        path.name for path in (repository / "snapshots").glob("*") if path.is_dir()
    )
    if len(snapshots) == 1:
        return snapshots[0]
    raise ValueError(
        f"Cannot freeze one local model revision for {model_id}; snapshots={snapshots}"
    )


def _local_model_snapshot(model_id: str, revision: str) -> Path:
    repository = Path.home() / ".cache/huggingface/hub" / (
        "models--" + model_id.replace("/", "--")
    )
    snapshot = (repository / "snapshots" / revision).resolve()
    if not snapshot.is_dir():
        raise ValueError(f"Frozen local model snapshot is missing: {snapshot}")
    return snapshot


def _write_or_validate_manifest(path: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    if path.exists():
        existing = _load_object(path, "v1.5 final run manifest")
        mismatches = [key for key, value in expected.items() if existing.get(key) != value]
        if mismatches:
            raise ValueError(
                f"Final run manifest is stale for {mismatches}; regeneration is forbidden."
            )
        return existing
    document = dict(expected)
    document["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return document


def _strict_stage_cache(
    path: Path,
    *,
    selected_ids: Sequence[str],
    stage: str,
    schema_version: str,
    model_id: str,
    prompt_hash: str,
    taxonomy_digest: str,
    input_hashes: Mapping[str, str],
    upstream_hashes: Mapping[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = read_jsonl(path)
    ids = [row.get("review_id") for row in rows]
    if ids != list(selected_ids[: len(ids)]):
        raise ValueError(f"Final {stage} cache must be an ordered prefix of the cohort")
    cache: dict[str, dict[str, Any]] = {}
    for row in rows:
        review_id = row["review_id"]
        expected_fields = {
            "schema_version": schema_version,
            "model_id": model_id,
            "prompt_hash": prompt_hash,
            "taxonomy_hash": taxonomy_digest,
            "input_hash": input_hashes[review_id],
        }
        if upstream_hashes is not None:
            expected_fields["upstream_hash"] = upstream_hashes[review_id]
        mismatches = [
            key for key, value in expected_fields.items() if row.get(key) != value
        ]
        if mismatches:
            raise ValueError(
                f"Stale final {stage} record for {review_id}: {mismatches}"
            )
        expected_hash = (
            stage_record_hash(row, "accepted_observations")
            if stage == "observations"
            else _claim_record_hash(row)
        )
        if row.get("record_hash") != expected_hash:
            raise ValueError(f"Corrupt final {stage} record for {review_id}")
        cache[review_id] = row
    return cache


def _validate_gold_reference(
    path: Path,
    *,
    selection_path: Path,
    selected_ids: Sequence[str],
    taxonomy: Mapping[str, Mapping[str, str]],
    source_reviews: Mapping[str, Mapping[str, Any]],
    taxonomy_digest: str,
    protocol_hash: str,
) -> tuple[dict[str, Any], str]:
    gold = _load_object(path, "v1.5 final source-only gold reference")
    expected_header = {
        "schema_version": "reviewlens_v15_final_gold_v1",
        "status": "frozen_complete_before_final_inference",
        "selection_manifest_sha256": _file_hash(selection_path),
        "evaluation_protocol_hash": protocol_hash,
        "taxonomy_hash": taxonomy_digest,
        "selected_review_ids": list(selected_ids),
    }
    mismatches = [
        key for key, value in expected_header.items() if gold.get(key) != value
    ]
    if mismatches:
        raise ValueError(f"Final gold reference header mismatch: {mismatches}")
    reviews = gold.get("reviews")
    if not isinstance(reviews, list) or [
        row.get("review_id") for row in reviews if isinstance(row, Mapping)
    ] != list(selected_ids):
        raise ValueError("Final gold reference must contain all 15 reviews in order")
    valid_sentiments = {"positive", "negative", "mixed", "neutral"}
    valid_targets = {
        "current_product", "previous_product", "other_product", "seller_platform"
    }
    total_findings = 0
    for review in reviews:
        review_id = review.get("review_id")
        source_review = source_reviews.get(review_id)
        if not isinstance(source_review, Mapping):
            raise ValueError(f"Gold review is absent from source data: {review_id}")
        if review.get("gold_complete") is not True:
            raise ValueError(f"Gold review is not marked complete: {review_id}")
        source_hashes = review.get("source_text_hashes")
        expected_source_hashes = {
            "title": hash_text(str(source_review.get("title", ""))),
            "clean_text": hash_text(str(source_review.get("clean_text", ""))),
        }
        if source_hashes != expected_source_hashes:
            raise ValueError(f"Gold source text hashes mismatch: {review_id}")
        findings = review.get("gold_findings")
        if not isinstance(findings, list):
            raise ValueError(f"Invalid gold_findings for {review_id}")
        if not findings and not isinstance(review.get("no_finding_reason"), str):
            raise ValueError(f"Empty gold review lacks adjudicated reason: {review_id}")
        total_findings += len(findings)
        seen: set[tuple[str, str, str, str]] = set()
        for finding in findings:
            if not isinstance(finding, Mapping):
                raise ValueError("Every gold finding must be an object")
            category = finding.get("category")
            issue_group = finding.get("issue_group")
            sentiment = finding.get("sentiment")
            attribution = finding.get("attribution")
            if (
                category not in taxonomy
                or issue_group not in taxonomy[category]
                or sentiment not in valid_sentiments
                or attribution not in valid_targets
            ):
                raise ValueError(f"Invalid final gold tuple: {finding}")
            key = (category, issue_group, sentiment, attribution)
            if key in seen:
                raise ValueError(f"Duplicate final gold tuple: {key}")
            seen.add(key)
            evidence = finding.get("evidence")
            if not isinstance(evidence, list) or not evidence:
                raise ValueError(f"Final gold finding lacks evidence: {finding}")
            for item in evidence:
                if (
                    not isinstance(item, Mapping)
                    or item.get("source_field") not in {"title", "clean_text"}
                    or not isinstance(item.get("quote"), str)
                    or not item["quote"]
                    or not isinstance(item.get("start_char"), int)
                    or not isinstance(item.get("end_char"), int)
                ):
                    raise ValueError(f"Invalid final gold evidence: {item}")
                source_text = str(source_review.get(item["source_field"], ""))
                start = item["start_char"]
                end = item["end_char"]
                if (
                    start < 0
                    or end != start + len(item["quote"])
                    or source_text[start:end] != item["quote"]
                ):
                    raise ValueError(
                        f"Final gold evidence is not an exact source span: {item}"
                    )
    if total_findings < 1:
        raise ValueError("Final gold reference contains no findings")
    return gold, _file_hash(path)


def _write_completion_manifest(
    path: Path,
    *,
    selection_path: Path,
    run_manifest_path: Path,
    observations_path: Path,
    claims_path: Path,
    gold_reference_path: Path,
    selected_ids: Sequence[str],
) -> dict[str, Any]:
    expected = {
        "schema_version": "reviewlens_v15_final_completion_v1",
        "status": "complete_locked_no_regeneration",
        "selected_review_ids": list(selected_ids),
        "selection_manifest_sha256": _file_hash(selection_path),
        "run_manifest_sha256": _file_hash(run_manifest_path),
        "observations_sha256": _file_hash(observations_path),
        "claims_sha256": _file_hash(claims_path),
        "gold_reference_sha256": _file_hash(gold_reference_path),
    }
    if path.exists():
        existing = _load_object(path, "v1.5 final completion manifest")
        if _without_created_at(existing) != expected:
            raise ValueError("Final completion manifest failed hash validation")
        return existing
    document = dict(expected)
    document["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return document


def run_v15_final(
    project_root: str | Path,
    taxonomy: Mapping[str, Mapping[str, str]],
    model_id: str,
    generator: Generator | None = None,
) -> dict[str, Any]:
    """Run or mechanically resume the complete locked 15-review cohort."""
    root = Path(project_root).resolve()
    paths = _paths(root)
    selection = _validated_selection(root)
    selected_rows = selection["selected_reviews"]
    selected_ids = [str(row["review_id"]) for row in selected_rows]
    if len(selected_ids) != FINAL_SIZE:
        raise ValueError("Final selection must contain exactly 15 reviews")
    if model_id != selection.get("frozen_model_id"):
        raise ValueError(
            f"Final model is frozen as {selection.get('frozen_model_id')}, not {model_id}"
        )
    origin_by_id = {
        review_id: "locked_residual_no_prior_v15" for review_id in selected_ids
    }

    reviews, sentences_by_review, predictions_by_review = _prepare_inputs(root)
    sources_by_review = {
        review_id: _source_rows(reviews[review_id], sentences_by_review[review_id])
        for review_id in selected_ids
    }
    observation_prompt_digest = hash_text(OBSERVATION_PROMPT_V15)
    normalization_prompt = build_normalization_prompt_v15(taxonomy)
    claim_prompt_digest = hash_text(normalization_prompt)
    taxonomy_digest = taxonomy_hash(taxonomy)
    frozen_bindings = {
        "frozen_taxonomy_hash": taxonomy_digest,
        "frozen_observation_prompt_hash": observation_prompt_digest,
        "frozen_claim_prompt_hash": claim_prompt_digest,
        "frozen_observation_schema_version": OBSERVATION_SCHEMA_VERSION_V15,
        "frozen_claim_schema_version": CLAIM_SCHEMA_VERSION_V15,
    }
    binding_mismatches = [
        key for key, value in frozen_bindings.items() if selection.get(key) != value
    ]
    if binding_mismatches:
        raise ValueError(f"Final frozen model/taxonomy mismatch: {binding_mismatches}")
    _, gold_reference_hash = _validate_gold_reference(
        paths["gold_reference"],
        selection_path=paths["selection"],
        selected_ids=selected_ids,
        taxonomy=taxonomy,
        source_reviews=reviews,
        taxonomy_digest=taxonomy_digest,
        protocol_hash=selection["evaluation_protocol_hash"],
    )
    observation_payloads = {
        review_id: _observation_payload(
            reviews[review_id],
            sources_by_review[review_id],
            predictions_by_review[review_id],
        )
        for review_id in selected_ids
    }
    observation_input_hashes = {
        review_id: hash_json(payload)
        for review_id, payload in observation_payloads.items()
    }
    current_code_hashes = _model_code_hashes()
    if selection.get("frozen_model_code_hashes") != current_code_hashes:
        raise ValueError("Frozen final model code hashes changed before inference")
    frozen_revision = str(selection["frozen_model_revision"])
    frozen_snapshot = _local_model_snapshot(model_id, frozen_revision)
    if str(frozen_snapshot) != selection.get("frozen_model_snapshot_path"):
        raise ValueError("Frozen final model snapshot path changed")

    if not paths["run_manifest"].exists():
        orphaned = [
            paths[key] for key in ("observations", "claims", "completion")
            if paths[key].exists()
        ]
        if orphaned:
            raise ValueError(f"Final artifacts exist without run manifest: {orphaned}")
    run_expected = {
        "schema_version": "reviewlens_v15_final_run_v1",
        "pipeline": "ReviewLens Evidence-first",
        "pipeline_version": "v1.5-locked-internal-validation",
        "policy_version": POLICY_VERSION_V15,
        "selection_manifest_hash": hash_json(selection),
        "selected_review_ids": selected_ids,
        "model_id": model_id,
        "model_revision": frozen_revision,
        "model_snapshot_path": str(frozen_snapshot),
        "model_code_hashes": current_code_hashes,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION_V15,
        "claim_schema_version": CLAIM_SCHEMA_VERSION_V15,
        "observation_prompt_hash": observation_prompt_digest,
        "claim_prompt_hash": claim_prompt_digest,
        "taxonomy_hash": taxonomy_digest,
        "max_retries": FINAL_MAX_RETRIES,
        "generation_temperature": FINAL_TEMPERATURE,
        "generation_max_tokens": FINAL_MAX_TOKENS,
        "mlx_version": _package_version("mlx"),
        "mlx_lm_version": _package_version("mlx-lm"),
        "source_artifact_hashes": selection["source_artifact_hashes"],
        "evaluation_protocol_hash": selection["evaluation_protocol_hash"],
        "gold_reference_sha256": gold_reference_hash,
        "output_isolation": "outputs/amazon/evidence_first/v15_final",
        "regeneration_policy": "forbidden_resume_missing_ordered_prefix_only",
    }
    _write_or_validate_manifest(paths["run_manifest"], run_expected)
    completed_locked = paths["completion"].exists()
    if completed_locked:
        missing = [
            paths[key] for key in ("observations", "claims") if not paths[key].exists()
        ]
        if missing:
            raise ValueError(
                f"Completed final run is missing locked cache files: {missing}"
            )
        # Validate every frozen artifact hash before any possible generator call.
        _write_completion_manifest(
            paths["completion"],
            selection_path=paths["selection"],
            run_manifest_path=paths["run_manifest"],
            observations_path=paths["observations"],
            claims_path=paths["claims"],
            gold_reference_path=paths["gold_reference"],
            selected_ids=selected_ids,
        )

    observation_cache = _strict_stage_cache(
        paths["observations"],
        selected_ids=selected_ids,
        stage="observations",
        schema_version=OBSERVATION_SCHEMA_VERSION_V15,
        model_id=model_id,
        prompt_hash=observation_prompt_digest,
        taxonomy_digest=taxonomy_digest,
        input_hashes=observation_input_hashes,
    )
    if completed_locked and len(observation_cache) != FINAL_SIZE:
        raise ValueError("Completed final observation cache is not complete")
    model_generator: Generator | None = generator

    def ensure_generator() -> Generator:
        nonlocal model_generator
        if model_generator is None:
            print("Loading frozen Evidence-first Layer 2:", model_id, "@", frozen_revision)
            model_generator = _MlxGenerator(str(frozen_snapshot))
        return model_generator

    for review_id in selected_ids[len(observation_cache) :]:
        raw, parsed, gate, parse_success, attempts = _run_with_optional_repair(
            generator=ensure_generator(),
            system_prompt=OBSERVATION_PROMPT_V15,
            payload=observation_payloads[review_id],
            parse_attempt=lambda response, current=review_id: parse_observation_attempt_v15(
                response,
                reviews[current],
                sources_by_review[current],
            ),
            accepted_key="accepted_observations",
            max_retries=FINAL_MAX_RETRIES,
        )
        row = {
            "review_id": review_id,
            "cohort_origin": origin_by_id[review_id],
            "rating": reviews[review_id].get("rating"),
            "schema_version": OBSERVATION_SCHEMA_VERSION_V15,
            "policy_version": POLICY_VERSION_V15,
            "model_id": model_id,
            "prompt_hash": observation_prompt_digest,
            "taxonomy_hash": taxonomy_digest,
            "input_hash": observation_input_hashes[review_id],
            "parse_success": parse_success,
            "strict_object_shape": gate.get("strict_object_shape", False),
            "parse_adjustments": gate.get("parse_adjustments", []),
            "context_recovery_count": gate.get("context_recovery_count", 0),
            "reference_checks_passed": gate["reference_checks_passed"],
            "accepted_observations": gate["accepted_observations"],
            "rejected_observations": gate["rejected_observations"],
            "validation_errors": gate["validation_errors"],
            "model_result_raw": parsed,
            "selected_raw_response": raw,
            "attempts": attempts,
        }
        row["record_hash"] = stage_record_hash(row, "accepted_observations")
        append_jsonl(paths["observations"], row)
        observation_cache[review_id] = row
        print(f"Final observation progress: {len(observation_cache)}/{FINAL_SIZE}")

    normalization_payloads = {
        review_id: normalization_payload_v15(observation_cache[review_id])
        for review_id in selected_ids
    }
    claim_input_hashes = {
        review_id: hash_json(payload)
        for review_id, payload in normalization_payloads.items()
    }
    upstream_hashes = {
        review_id: observation_cache[review_id]["record_hash"]
        for review_id in selected_ids
    }
    claim_cache = _strict_stage_cache(
        paths["claims"],
        selected_ids=selected_ids,
        stage="claims",
        schema_version=CLAIM_SCHEMA_VERSION_V15,
        model_id=model_id,
        prompt_hash=claim_prompt_digest,
        taxonomy_digest=taxonomy_digest,
        input_hashes=claim_input_hashes,
        upstream_hashes=upstream_hashes,
    )
    if completed_locked and len(claim_cache) != FINAL_SIZE:
        raise ValueError("Completed final claim cache is not complete")
    for review_id in selected_ids[len(claim_cache) :]:
        observations = observation_cache[review_id]["accepted_observations"]
        if not observations:
            row = _empty_claim_row_v15(
                review_id=review_id,
                model_id=model_id,
                prompt_digest=claim_prompt_digest,
                taxonomy_digest=taxonomy_digest,
                input_digest=claim_input_hashes[review_id],
                upstream_digest=upstream_hashes[review_id],
                cohort_origin=origin_by_id[review_id],
            )
        else:
            raw, parsed, gate, parse_success, attempts = _run_with_optional_repair(
                generator=ensure_generator(),
                system_prompt=normalization_prompt,
                payload=normalization_payloads[review_id],
                parse_attempt=lambda response, obs=observations: _parse_claim_attempt(
                    response, obs, taxonomy
                ),
                accepted_key="accepted_claims",
                max_retries=FINAL_MAX_RETRIES,
            )
            gate = complete_claims_v15(gate, observations, taxonomy)
            row = {
                "review_id": review_id,
                "cohort_origin": origin_by_id[review_id],
                "schema_version": CLAIM_SCHEMA_VERSION_V15,
                "policy_version": POLICY_VERSION_V15,
                "model_id": model_id,
                "prompt_hash": claim_prompt_digest,
                "taxonomy_hash": taxonomy_digest,
                "input_hash": claim_input_hashes[review_id],
                "upstream_hash": upstream_hashes[review_id],
                "parse_success": parse_success,
                "normalization_complete": gate["normalization_complete"],
                "model_normalization_complete": gate["model_normalization_complete"],
                "fallback_unmapped_count": gate["fallback_unmapped_count"],
                "deterministic_recovery_count": gate["deterministic_recovery_count"],
                "v15_recall_recovery_count": gate["v15_recall_recovery_count"],
                "v15_category_repair_count": gate["v15_category_repair_count"],
                "accepted_claims": gate["accepted_claims"],
                "rejected_claims": gate["rejected_claims"],
                "missing_observation_ids": gate["missing_observation_ids"],
                "validation_errors": gate["validation_errors"],
                "model_validation_errors": gate["model_validation_errors"],
                "model_missing_observation_ids": gate["model_missing_observation_ids"],
                "model_result_raw": parsed,
                "selected_raw_response": raw,
                "attempts": attempts,
            }
            row["merged_claims"] = merge_claims_by_tuple(row["accepted_claims"])
            row["duplicate_claims_merged"] = (
                sum(
                    claim.get("mapping_status") == "mapped"
                    for claim in row["accepted_claims"]
                ) - len(row["merged_claims"])
            )
            row["record_hash"] = _claim_record_hash(row)
        append_jsonl(paths["claims"], row)
        claim_cache[review_id] = row
        print(f"Final normalization progress: {len(claim_cache)}/{FINAL_SIZE}")

    if isinstance(model_generator, _MlxGenerator):
        del model_generator
        gc.collect()
        try:
            import mlx.core as mx

            mx.clear_cache()
        except (ImportError, AttributeError):
            pass

    _write_completion_manifest(
        paths["completion"],
        selection_path=paths["selection"],
        run_manifest_path=paths["run_manifest"],
        observations_path=paths["observations"],
        claims_path=paths["claims"],
        gold_reference_path=paths["gold_reference"],
        selected_ids=selected_ids,
    )
    print("\n=== EVIDENCE-FIRST v1.5 FINAL COMPLETE ===")
    print(f"Locked internal validation reviews: {FINAL_SIZE}/{FINAL_SIZE}")
    print("No result metrics are shown before blind annotation.")
    print("Saved only under:", paths["output_dir"])
    return {
        "selected_review_ids": selected_ids,
        "observation_rows": [observation_cache[value] for value in selected_ids],
        "claim_rows": [claim_cache[value] for value in selected_ids],
        "output_dir": str(paths["output_dir"]),
    }


__all__ = [
    "EVALUATION_PROTOCOL",
    "FINAL_RATING_TARGETS",
    "FINAL_SEED",
    "FINAL_SIZE",
    "prepare_v15_final",
    "run_v15_final",
    "selection_document_v15_final",
]
