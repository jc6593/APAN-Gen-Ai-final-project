"""Isolated 10-to-20 review development cohort for Evidence-first v1.5."""

from __future__ import annotations

import gc
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from . import core as core_module
from . import pipeline as pipeline_module
from . import prompts as prompts_module
from . import v15_core as v15_core_module
from . import v15_prompts as v15_prompts_module
from .core import (
    PILOT_REVIEW_IDS,
    append_jsonl,
    backup_existing,
    hash_json,
    hash_text,
    load_stage_cache,
    merge_claims_by_tuple,
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
from .v15_prompts import (
    OBSERVATION_PROMPT_V15,
    build_normalization_prompt_v15,
)


DEV_SIZE = 20
DEV_SEED = 5561
# The original notebook's five-review pilot predates the Evidence-first pilot.
# Excluding it prevents development leakage from either historical workflow.
LEGACY_PILOT_REVIEW_IDS = (
    "electronics_162754",
    "electronics_1095825",
    "electronics_1688574",
    "electronics_58628",
    "electronics_1230503",
)
# Only three unseen 3-star reviews remain after the diagnostic 25. Reuse four
# diagnostic 3-star reviews for development so all three unseen examples remain
# available to represent the scarce rating in the later final cohort.
DEV_RATING_TARGETS = {1: 4, 2: 4, 3: 4, 4: 4, 5: 4}
SELECTION_ALGORITHM = "rating_balanced_length_stratified_v15_dev_v1"


def _paths(project_root: Path) -> dict[str, Path]:
    output_dir = project_root / "outputs/amazon/evidence_first/v15_dev"
    return {
        "output_dir": output_dir,
        "selection": output_dir / "selection_manifest.json",
        "run_manifest": output_dir / "run_manifest.json",
        "observations": output_dir / "observations_v15_dev.jsonl",
        "claims": output_dir / "claims_v15_dev.jsonl",
        "diagnostic_holdout_selection": (
            project_root
            / "outputs/amazon/evidence_first/holdout/selection_manifest.json"
        ),
    }


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_code_hashes() -> dict[str, str]:
    files = {
        "reviewlens_ef/core.py": Path(str(core_module.__file__)),
        "reviewlens_ef/pipeline.py": Path(str(pipeline_module.__file__)),
        "reviewlens_ef/prompts.py": Path(str(prompts_module.__file__)),
        "reviewlens_ef/v15_core.py": Path(str(v15_core_module.__file__)),
        "reviewlens_ef/v15_prompts.py": Path(str(v15_prompts_module.__file__)),
        "reviewlens_ef/v15_dev.py": Path(__file__),
    }
    return {name: _file_hash(path.resolve()) for name, path in files.items()}


def _rating(value: Any) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid review rating: {value}") from exc
    integer = int(numeric)
    if numeric != integer or integer not in DEV_RATING_TARGETS:
        raise ValueError(f"Invalid review rating: {value}")
    return integer


def _tie(seed: int, context: str, review_id: str) -> str:
    return hashlib.sha256(
        f"{seed}|{context}|{review_id}".encode("utf-8")
    ).hexdigest()


def _load_diagnostic_holdout(path: Path) -> tuple[dict[str, Any], list[str]]:
    if not path.exists():
        raise ValueError(
            "The diagnostic 25-review selection is missing. Preserve/freeze it "
            "before creating the v1.5 dev cohort."
        )
    with path.open(encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict):
        raise ValueError(f"Invalid diagnostic holdout manifest: {path}")
    rows = document.get("selected_reviews")
    if not isinstance(rows, list):
        raise ValueError(f"Invalid diagnostic holdout selection: {path}")
    ids = [
        row.get("review_id")
        for row in rows
        if isinstance(row, Mapping) and isinstance(row.get("review_id"), str)
    ]
    if len(ids) != 25 or len(set(ids)) != 25:
        raise ValueError("Diagnostic holdout must contain 25 unique review IDs")
    return document, [str(value) for value in ids]


def _choose_rating_records(
    records: list[dict[str, Any]],
    rating: int,
    target: int,
    seed: int,
) -> list[dict[str, Any]]:
    ordered = sorted(
        records,
        key=lambda row: (
            row["review_char_length"],
            _tie(seed, f"rating-{rating}-length-order", str(row["review_id"])),
        ),
    )
    bins: dict[int, list[dict[str, Any]]] = {slot: [] for slot in range(target)}
    for index, row in enumerate(ordered):
        slot = min(index * target // len(ordered), target - 1)
        bins[slot].append(row)
    chosen: list[dict[str, Any]] = []
    for slot in range(target):
        options = bins[slot]
        if not options:
            raise AssertionError(f"Empty v1.5 dev length bin: rating={rating}, slot={slot}")
        row = min(
            options,
            key=lambda item: _tie(
                seed,
                f"rating-{rating}-slot-{slot}-choice",
                str(item["review_id"]),
            ),
        )
        selected = dict(row)
        selected["length_bin"] = slot
        selected["sampling_weight"] = round(len(records) / target, 12)
        chosen.append(selected)
    return chosen


def selection_document_v15(
    project_root: str | Path,
    seed: int = DEV_SEED,
) -> dict[str, Any]:
    """Build the deterministic dev selection without writing or calling a model."""
    root = Path(project_root).resolve()
    paths = _paths(root)
    diagnostic_manifest, diagnostic_ids = _load_diagnostic_holdout(
        paths["diagnostic_holdout_selection"]
    )
    reviews, sentences_by_review, predictions_by_review = _prepare_inputs(root)
    excluded = (
        set(PILOT_REVIEW_IDS)
        | set(LEGACY_PILOT_REVIEW_IDS)
        | set(diagnostic_ids)
    )
    def candidate_record(review_id: str, cohort_origin: str) -> dict[str, Any]:
        sentence_count = len(sentences_by_review[review_id])
        layer1_aspect_count = sum(
            len(row.get("aspects") or [])
            for row in predictions_by_review[review_id]
            if row.get("parse_success") and isinstance(row.get("aspects"), list)
        )
        return {
            "review_id": review_id,
            "rating": _rating(reviews[review_id].get("rating")),
            "cohort_origin": cohort_origin,
            "review_char_length": len(str(reviews[review_id].get("clean_text", ""))),
            "sentence_count": sentence_count,
            "layer1_aspect_count": layer1_aspect_count,
            "layer1_density": (
                round(layer1_aspect_count / sentence_count, 12)
                if sentence_count else 0.0
            ),
        }

    candidates = [
        candidate_record(review_id, "new_dev")
        for review_id in sorted(reviews)
        if review_id not in excluded
    ]
    distribution = Counter(int(row["rating"]) for row in candidates)
    insufficient = {
        rating: distribution[rating]
        for rating, target in DEV_RATING_TARGETS.items()
        if rating != 3 and distribution[rating] < target
    }
    if insufficient:
        raise ValueError(f"Insufficient v1.5 dev candidates by rating: {insufficient}")

    diagnostic_three_star = [
        candidate_record(review_id, "diagnostic_3star_reuse")
        for review_id in diagnostic_ids
        if _rating(reviews[review_id].get("rating")) == 3
        and review_id not in set(PILOT_REVIEW_IDS)
        and review_id not in set(LEGACY_PILOT_REVIEW_IDS)
    ]
    if len(diagnostic_three_star) < DEV_RATING_TARGETS[3]:
        raise ValueError(
            "The diagnostic cohort must provide at least four 3-star reviews "
            "so unseen 3-star reviews can be reserved for final evaluation"
        )

    chosen_by_rating = {
        rating: _choose_rating_records(
            (
                diagnostic_three_star
                if rating == 3
                else [row for row in candidates if row["rating"] == rating]
            ),
            rating,
            target,
            seed,
        )
        for rating, target in DEV_RATING_TARGETS.items()
    }
    selected_reviews = [
        chosen_by_rating[rating][slot]
        for slot in range(max(DEV_RATING_TARGETS.values()))
        for rating in range(1, 6)
        if slot < DEV_RATING_TARGETS[rating]
    ]
    selected_ids = [str(row["review_id"]) for row in selected_reviews]
    if len(selected_ids) != DEV_SIZE or len(set(selected_ids)) != DEV_SIZE:
        raise AssertionError("V1.5 dev selection did not produce 20 unique reviews")
    if set(selected_ids) & (set(PILOT_REVIEW_IDS) | set(LEGACY_PILOT_REVIEW_IDS)):
        raise AssertionError("V1.5 dev selection overlaps a historical pilot")
    reused_diagnostic_ids = {
        str(row["review_id"])
        for row in selected_reviews
        if row.get("cohort_origin") == "diagnostic_3star_reuse"
    }
    if (
        len(reused_diagnostic_ids) != DEV_RATING_TARGETS[3]
        or set(selected_ids) & set(diagnostic_ids) != reused_diagnostic_ids
    ):
        raise AssertionError("Only four documented diagnostic 3-star reviews may be reused")
    if Counter(int(row["rating"]) for row in selected_reviews[:10]) != Counter({
        1: 2,
        2: 2,
        3: 2,
        4: 2,
        5: 2,
    }):
        raise AssertionError("First ten v1.5 dev reviews are not rating-balanced")

    source_ids_for_hash = {
        str(row["review_id"]) for row in candidates + diagnostic_three_star
    }
    source_hash = hash_json({
        review_id: {
            "review": reviews[review_id],
            "sentences": sentences_by_review[review_id],
            "layer1": predictions_by_review[review_id],
        }
        for review_id in sorted(source_ids_for_hash)
    })
    reserved_unseen_three_star_ids = sorted(
        str(row["review_id"])
        for row in candidates
        if row["rating"] == 3
    )
    return {
        "schema_version": "reviewlens_v15_dev_selection_v1",
        "status": "frozen_before_v15_inference",
        "algorithm": SELECTION_ALGORITHM,
        "seed": seed,
        "dev_size": DEV_SIZE,
        "rating_targets": {
            str(key): value for key, value in DEV_RATING_TARGETS.items()
        },
        "excluded_pilot_review_ids": list(PILOT_REVIEW_IDS),
        "excluded_legacy_pilot_review_ids": list(LEGACY_PILOT_REVIEW_IDS),
        "excluded_diagnostic_holdout_review_ids": diagnostic_ids,
        "reused_diagnostic_3star_review_ids": sorted(reused_diagnostic_ids),
        "reserved_unseen_3star_review_ids": reserved_unseen_three_star_ids,
        "diagnostic_holdout_manifest_hash": hash_json(diagnostic_manifest),
        "new_candidate_count": len(candidates),
        "new_candidate_rating_distribution": {
            str(rating): distribution[rating] for rating in range(1, 6)
        },
        "candidate_pool_hash": source_hash,
        "selected_reviews": selected_reviews,
    }


def _without_created_at(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "created_at_utc"}


def _load_document(path: Path, label: str) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Invalid {label}: {path}")
    return value


def prepare_v15_dev(
    project_root: str | Path,
    seed: int = DEV_SEED,
) -> dict[str, Any]:
    """Freeze or verify the isolated 20-review dev selection; no model call."""
    root = Path(project_root).resolve()
    paths = _paths(root)
    expected = selection_document_v15(root, seed)
    if paths["selection"].exists():
        existing = _load_document(paths["selection"], "v1.5 dev selection")
        if _without_created_at(existing) != expected:
            raise ValueError(
                "Existing v1.5 dev selection is immutable and no longer matches "
                "its source/exclusion fingerprints. Do not overwrite it."
            )
        document = existing
        status = "already frozen and verified"
    else:
        unexpected = [
            paths[key] for key in ("run_manifest", "observations", "claims")
            if paths[key].exists()
        ]
        if unexpected:
            raise ValueError(
                "V1.5 dev inference artifacts exist without a selection manifest: "
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
        status = "frozen before inference"
    print("\n=== EVIDENCE-FIRST v1.5 DEV PREPARE ===")
    print(f"Selection {status}: {paths['selection']}")
    print("Reviews: 20; first 10 and full 20 are rating-balanced")
    print("Composition: 16 new + 4 diagnostic 3-star reviews reused for development")
    print("Reserved for final: all 3 remaining unseen 3-star reviews")
    print("No model was loaded or called.")
    return document


def _validated_selection(root: Path) -> dict[str, Any]:
    paths = _paths(root)
    if not paths["selection"].exists():
        return prepare_v15_dev(root)
    existing = _load_document(paths["selection"], "v1.5 dev selection")
    seed = existing.get("seed")
    if not isinstance(seed, int):
        raise ValueError("Invalid v1.5 dev selection seed")
    expected = selection_document_v15(root, seed)
    if _without_created_at(existing) != expected:
        raise ValueError(
            "Frozen v1.5 dev selection failed source/exclusion validation. "
            "Refusing inference."
        )
    return existing


def _write_or_validate_run_manifest(
    path: Path,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    if path.exists():
        existing = _load_document(path, "v1.5 dev run manifest")
        mismatches = [
            key for key, value in expected.items() if existing.get(key) != value
        ]
        if mismatches:
            raise ValueError(
                f"Stale v1.5 dev run manifest; mismatched {mismatches}. "
                "Use --force only to rebuild the isolated v1.5 dev caches."
            )
        return existing
    document = dict(expected)
    document["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return document


def _empty_claim_row_v15(
    *,
    review_id: str,
    model_id: str,
    prompt_digest: str,
    taxonomy_digest: str,
    input_digest: str,
    upstream_digest: str,
    cohort_origin: str,
) -> dict[str, Any]:
    row = {
        "review_id": review_id,
        "cohort_origin": cohort_origin,
        "schema_version": CLAIM_SCHEMA_VERSION_V15,
        "policy_version": POLICY_VERSION_V15,
        "model_id": model_id,
        "prompt_hash": prompt_digest,
        "taxonomy_hash": taxonomy_digest,
        "input_hash": input_digest,
        "upstream_hash": upstream_digest,
        "parse_success": True,
        "normalization_complete": True,
        "model_normalization_complete": True,
        "fallback_unmapped_count": 0,
        "deterministic_recovery_count": 0,
        "v15_recall_recovery_count": 0,
        "v15_category_repair_count": 0,
        "accepted_claims": [],
        "merged_claims": [],
        "duplicate_claims_merged": 0,
        "rejected_claims": [],
        "missing_observation_ids": [],
        "validation_errors": [],
        "model_validation_errors": [],
        "model_missing_observation_ids": [],
        "attempts": [],
        "note": "No validated observations; normalization model was not called.",
    }
    row["record_hash"] = _claim_record_hash(row)
    return row


def _validate_cached_record_hashes(
    cache: Mapping[str, Mapping[str, Any]],
    *,
    stage: str,
) -> None:
    for review_id, row in cache.items():
        if stage == "observations":
            expected = stage_record_hash(row, "accepted_observations")
        elif stage == "claims":
            expected = _claim_record_hash(row)
        else:
            raise ValueError(f"Unknown v1.5 cache stage: {stage}")
        if row.get("record_hash") != expected:
            raise ValueError(
                f"Corrupt v1.5 {stage} cache record for {review_id}: "
                "record_hash does not match content. Use --force to back up "
                "and rebuild only the v1.5 dev caches."
            )


def run_v15_dev(
    project_root: str | Path,
    taxonomy: Mapping[str, Mapping[str, str]],
    model_id: str,
    limit: int = 10,
    force: bool = False,
    max_retries: int = 1,
    show: int = 5,
    generator: Generator | None = None,
) -> dict[str, Any]:
    """Run/resume only the isolated v1.5 dev prefix (10 first, then 20)."""
    if max_retries not in {0, 1}:
        raise ValueError("max_retries must be 0 or 1")
    if not 1 <= limit <= DEV_SIZE:
        raise ValueError("Evidence-first v1.5 dev --limit must be between 1 and 20")
    root = Path(project_root).resolve()
    paths = _paths(root)
    selection = _validated_selection(root)
    selected_universe = [
        str(row["review_id"]) for row in selection["selected_reviews"]
    ]
    selection_by_id = {
        str(row["review_id"]): row for row in selection["selected_reviews"]
    }
    selected_ids = selected_universe[:limit]

    if force:
        for key in ("observations", "claims", "run_manifest"):
            backup = backup_existing(paths[key])
            if backup is not None:
                print("V1.5 dev backup:", backup)
    elif not paths["run_manifest"].exists():
        orphaned = [
            paths[key] for key in ("observations", "claims")
            if paths[key].exists()
        ]
        if orphaned:
            raise ValueError(
                "V1.5 dev caches exist without their run manifest: "
                f"{orphaned}. Refusing to adopt unverifiable caches; use "
                "--force to back them up and rebuild."
            )

    reviews, sentences_by_review, predictions_by_review = _prepare_inputs(root)
    sources_by_review = {
        review_id: _source_rows(reviews[review_id], sentences_by_review[review_id])
        for review_id in selected_universe
    }
    observation_prompt_digest = hash_text(OBSERVATION_PROMPT_V15)
    normalization_prompt = build_normalization_prompt_v15(taxonomy)
    claim_prompt_digest = hash_text(normalization_prompt)
    taxonomy_digest = taxonomy_hash(taxonomy)
    observation_payloads = {
        review_id: _observation_payload(
            reviews[review_id],
            sources_by_review[review_id],
            predictions_by_review[review_id],
        )
        for review_id in selected_universe
    }
    observation_input_hashes = {
        review_id: hash_json(payload)
        for review_id, payload in observation_payloads.items()
    }
    run_manifest_expected = {
        "schema_version": "reviewlens_v15_dev_run_v1",
        "pipeline": "ReviewLens Evidence-first",
        "pipeline_version": "v1.5-recall-repair-dev",
        "policy_version": POLICY_VERSION_V15,
        "selection_manifest_hash": hash_json(selection),
        "selected_review_ids": selected_universe,
        "model_id": model_id,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION_V15,
        "claim_schema_version": CLAIM_SCHEMA_VERSION_V15,
        "observation_prompt_hash": observation_prompt_digest,
        "claim_prompt_hash": claim_prompt_digest,
        "taxonomy_hash": taxonomy_digest,
        "model_code_hashes": _model_code_hashes(),
        "old_artifacts_policy": "read_only",
        "output_isolation": "outputs/amazon/evidence_first/v15_dev",
    }
    _write_or_validate_run_manifest(paths["run_manifest"], run_manifest_expected)

    observation_cache = load_stage_cache(
        paths["observations"],
        schema_version=OBSERVATION_SCHEMA_VERSION_V15,
        model_id=model_id,
        prompt_hash=observation_prompt_digest,
        taxonomy_digest=taxonomy_digest,
        input_hashes=observation_input_hashes,
    )
    _validate_cached_record_hashes(observation_cache, stage="observations")
    pending_observations = [
        review_id for review_id in selected_ids if review_id not in observation_cache
    ]
    print(
        "Evidence-first v1.5 dev observations: "
        f"selected={len(selected_ids)} "
        f"cached={len(selected_ids) - len(pending_observations)} "
        f"pending={len(pending_observations)}"
    )

    model_generator: Generator | None = generator

    def ensure_generator() -> Generator:
        nonlocal model_generator
        if model_generator is None:
            print("Loading Evidence-first Layer 2:", model_id)
            model_generator = _MlxGenerator(model_id)
        return model_generator

    for review_id in pending_observations:
        used_generator = ensure_generator()
        raw, parsed, gate, parse_success, attempts = _run_with_optional_repair(
            generator=used_generator,
            system_prompt=OBSERVATION_PROMPT_V15,
            payload=observation_payloads[review_id],
            parse_attempt=lambda response, current=review_id: (
                parse_observation_attempt_v15(
                    response,
                    reviews[current],
                    sources_by_review[current],
                )
            ),
            accepted_key="accepted_observations",
            max_retries=max_retries,
        )
        row = {
            "review_id": review_id,
            "cohort_origin": selection_by_id[review_id]["cohort_origin"],
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
        print(
            f"  {review_id}: observations={len(row['accepted_observations'])} "
            f"rejected={len(row['rejected_observations'])} "
            f"parse_recoveries={len(row['parse_adjustments'])} "
            f"context_recoveries={row['context_recovery_count']}"
        )

    normalization_payloads = {
        review_id: normalization_payload_v15(observation_cache[review_id])
        for review_id in selected_universe
        if review_id in observation_cache
    }
    claim_input_hashes = {
        review_id: hash_json(payload)
        for review_id, payload in normalization_payloads.items()
    }
    upstream_hashes = {
        review_id: observation_cache[review_id]["record_hash"]
        for review_id in normalization_payloads
    }
    claim_cache = load_stage_cache(
        paths["claims"],
        schema_version=CLAIM_SCHEMA_VERSION_V15,
        model_id=model_id,
        prompt_hash=claim_prompt_digest,
        taxonomy_digest=taxonomy_digest,
        input_hashes=claim_input_hashes,
        upstream_hashes=upstream_hashes,
    )
    _validate_cached_record_hashes(claim_cache, stage="claims")
    pending_claims = [
        review_id for review_id in selected_ids if review_id not in claim_cache
    ]
    print(
        "Evidence-first v1.5 dev normalization: "
        f"selected={len(selected_ids)} "
        f"cached={len(selected_ids) - len(pending_claims)} "
        f"pending={len(pending_claims)}"
    )

    for review_id in pending_claims:
        observation_row = observation_cache[review_id]
        observations = observation_row["accepted_observations"]
        if not observations:
            row = _empty_claim_row_v15(
                review_id=review_id,
                model_id=model_id,
                prompt_digest=claim_prompt_digest,
                taxonomy_digest=taxonomy_digest,
                input_digest=claim_input_hashes[review_id],
                upstream_digest=upstream_hashes[review_id],
                cohort_origin=str(selection_by_id[review_id]["cohort_origin"]),
            )
        else:
            used_generator = ensure_generator()
            raw, parsed, gate, parse_success, attempts = _run_with_optional_repair(
                generator=used_generator,
                system_prompt=normalization_prompt,
                payload=normalization_payloads[review_id],
                parse_attempt=lambda response, obs=observations: _parse_claim_attempt(
                    response, obs, taxonomy
                ),
                accepted_key="accepted_claims",
                max_retries=max_retries,
            )
            gate = complete_claims_v15(gate, observations, taxonomy)
            row = {
                "review_id": review_id,
                "cohort_origin": selection_by_id[review_id]["cohort_origin"],
                "schema_version": CLAIM_SCHEMA_VERSION_V15,
                "policy_version": POLICY_VERSION_V15,
                "model_id": model_id,
                "prompt_hash": claim_prompt_digest,
                "taxonomy_hash": taxonomy_digest,
                "input_hash": claim_input_hashes[review_id],
                "upstream_hash": upstream_hashes[review_id],
                "parse_success": parse_success,
                "normalization_complete": gate["normalization_complete"],
                "model_normalization_complete": gate[
                    "model_normalization_complete"
                ],
                "fallback_unmapped_count": gate["fallback_unmapped_count"],
                "deterministic_recovery_count": gate[
                    "deterministic_recovery_count"
                ],
                "v15_recall_recovery_count": gate[
                    "v15_recall_recovery_count"
                ],
                "v15_category_repair_count": gate[
                    "v15_category_repair_count"
                ],
                "accepted_claims": gate["accepted_claims"],
                "rejected_claims": gate["rejected_claims"],
                "missing_observation_ids": gate["missing_observation_ids"],
                "validation_errors": gate["validation_errors"],
                "model_validation_errors": gate["model_validation_errors"],
                "model_missing_observation_ids": gate[
                    "model_missing_observation_ids"
                ],
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
        mapped = sum(
            claim.get("mapping_status") == "mapped"
            for claim in row["accepted_claims"]
        )
        print(
            f"  {review_id}: atomic={len(row['accepted_claims'])} "
            f"mapped={mapped} unmapped={len(row['accepted_claims']) - mapped} "
            f"findings={len(row['merged_claims'])} "
            f"recall_repairs={row.get('v15_recall_recovery_count', 0)} "
            f"category_repairs={row.get('v15_category_repair_count', 0)}"
        )

    if isinstance(model_generator, _MlxGenerator):
        del model_generator
        gc.collect()
        try:
            import mlx.core as mx

            mx.clear_cache()
        except (ImportError, AttributeError):
            pass

    print("\n=== EVIDENCE-FIRST v1.5 DEV ===")
    print(
        f"Frozen dev cohort: {DEV_SIZE}; requested prefix: {len(selected_ids)}; "
        f"completed claims: {sum(value in claim_cache for value in selected_universe)}"
    )
    for review_id in selected_ids[: max(show, 0)]:
        observations = observation_cache[review_id]["accepted_observations"]
        claims = claim_cache[review_id]["accepted_claims"]
        mapped = sum(item.get("mapping_status") == "mapped" for item in claims)
        print(
            f"  {review_id}: observations={len(observations)} "
            f"origin={selection_by_id[review_id]['cohort_origin']} "
            f"atomic_mapped={mapped} unmapped={len(claims) - mapped} "
            f"findings={len(claim_cache[review_id]['merged_claims'])} "
            f"recall_repairs={claim_cache[review_id].get('v15_recall_recovery_count', 0)} "
            f"category_repairs={claim_cache[review_id].get('v15_category_repair_count', 0)}"
        )
    print("Saved only under:", paths["output_dir"])
    return {
        "selected_review_ids": selected_ids,
        "frozen_review_ids": selected_universe,
        "observation_rows": [observation_cache[value] for value in selected_ids],
        "claim_rows": [claim_cache[value] for value in selected_ids],
        "output_dir": str(paths["output_dir"]),
    }


__all__ = [
    "DEV_RATING_TARGETS",
    "DEV_SEED",
    "DEV_SIZE",
    "LEGACY_PILOT_REVIEW_IDS",
    "prepare_v15_dev",
    "run_v15_dev",
    "selection_document_v15",
]
