"""Frozen, resumable 25-review holdout for the Evidence-first v1.4 pipeline."""

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
from .core import (
    CLAIM_SCHEMA_VERSION,
    OBSERVATION_SCHEMA_VERSION,
    PILOT_REVIEW_IDS,
    append_jsonl,
    backup_existing,
    complete_claims_with_unmapped_fallbacks,
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
    _empty_claim_row,
    _normalization_payload,
    _observation_payload,
    _parse_claim_attempt,
    _parse_observation_attempt,
    _prepare_inputs,
    _run_with_optional_repair,
    _source_rows,
)
from .prompts import OBSERVATION_PROMPT, build_normalization_prompt


HOLDOUT_SIZE = 25
HOLDOUT_PER_RATING = 5
HOLDOUT_SEED = 5560
SELECTION_ALGORITHM = "rating_balanced_latin_hypercube_v1"


def _paths(project_root: Path) -> dict[str, Path]:
    output_dir = project_root / "outputs/amazon/evidence_first/holdout"
    return {
        "output_dir": output_dir,
        "selection": output_dir / "selection_manifest.json",
        "run_manifest": output_dir / "run_manifest.json",
        "observations": output_dir / "observations_holdout.jsonl",
        "claims": output_dir / "claims_holdout.jsonl",
        "nli": output_dir / "nli_holdout.jsonl",
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
        "reviewlens_ef/holdout.py": Path(__file__),
    }
    return {name: _file_hash(path.resolve()) for name, path in files.items()}


def _rating(value: Any) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid review rating: {value}") from exc
    integer = int(numeric)
    if numeric != integer or integer not in {1, 2, 3, 4, 5}:
        raise ValueError(f"Invalid review rating: {value}")
    return integer


def _tie(seed: int, context: str, review_id: str) -> str:
    return hashlib.sha256(
        f"{seed}|{context}|{review_id}".encode("utf-8")
    ).hexdigest()


def _rank_bins(
    records: list[dict[str, Any]],
    value_key: str,
    seed: int,
    context: str,
) -> dict[str, int]:
    ordered = sorted(
        records,
        key=lambda row: (
            row[value_key],
            _tie(seed, context, str(row["review_id"])),
        ),
    )
    size = len(ordered)
    return {
        str(row["review_id"]): min(index * HOLDOUT_PER_RATING // size, 4)
        for index, row in enumerate(ordered)
    }


def _selection_document(
    project_root: Path,
    taxonomy: Mapping[str, Mapping[str, str]],
    model_id: str,
    seed: int,
) -> dict[str, Any]:
    reviews, sentences_by_review, predictions_by_review = _prepare_inputs(project_root)
    candidates: list[dict[str, Any]] = []
    excluded = set(PILOT_REVIEW_IDS)
    for review_id in sorted(reviews):
        if review_id in excluded:
            continue
        sentence_count = len(sentences_by_review[review_id])
        layer1_aspect_count = sum(
            len(row.get("aspects") or [])
            for row in predictions_by_review[review_id]
            if row.get("parse_success") and isinstance(row.get("aspects"), list)
        )
        candidates.append({
            "review_id": review_id,
            "rating": _rating(reviews[review_id].get("rating")),
            "review_char_length": len(str(reviews[review_id].get("clean_text", ""))),
            "sentence_count": sentence_count,
            "layer1_aspect_count": layer1_aspect_count,
            "layer1_density": (
                round(layer1_aspect_count / sentence_count, 12)
                if sentence_count else 0.0
            ),
        })
    distribution = Counter(int(row["rating"]) for row in candidates)
    insufficient = {
        rating: distribution[rating]
        for rating in range(1, 6)
        if distribution[rating] < HOLDOUT_PER_RATING
    }
    if insufficient:
        raise ValueError(f"Insufficient holdout candidates by rating: {insufficient}")

    chosen_by_rating: dict[int, dict[int, dict[str, Any]]] = {}
    for rating in range(1, 6):
        rating_records = [row for row in candidates if row["rating"] == rating]
        length_bins = _rank_bins(
            rating_records, "review_char_length", seed, f"rating-{rating}-length"
        )
        density_bins = _rank_bins(
            rating_records, "layer1_density", seed, f"rating-{rating}-density"
        )
        used_density_bins: set[int] = set()
        chosen_by_rating[rating] = {}
        for length_bin in range(HOLDOUT_PER_RATING):
            options = [
                row for row in rating_records
                if length_bins[str(row["review_id"])] == length_bin
            ]
            chosen = min(
                options,
                key=lambda row: (
                    density_bins[str(row["review_id"])] in used_density_bins,
                    _tie(
                        seed,
                        f"rating-{rating}-length-{length_bin}-choice",
                        str(row["review_id"]),
                    ),
                ),
            )
            selected = dict(chosen)
            selected["length_bin"] = length_bin
            selected["density_bin"] = density_bins[str(chosen["review_id"])]
            selected["sampling_weight"] = round(
                distribution[rating] / HOLDOUT_PER_RATING, 12
            )
            chosen_by_rating[rating][length_bin] = selected
            used_density_bins.add(int(selected["density_bin"]))

    selected_reviews = [
        chosen_by_rating[rating][slot]
        for slot in range(HOLDOUT_PER_RATING)
        for rating in range(1, 6)
    ]
    if len(selected_reviews) != HOLDOUT_SIZE:
        raise AssertionError("Holdout selection did not produce exactly 25 reviews")
    selected_ids = [str(row["review_id"]) for row in selected_reviews]
    if len(set(selected_ids)) != HOLDOUT_SIZE or set(selected_ids) & excluded:
        raise AssertionError("Holdout selection contains duplicates or pilot reviews")

    source_hash = hash_json({
        review_id: {
            "review": reviews[review_id],
            "sentences": sentences_by_review[review_id],
            "layer1": predictions_by_review[review_id],
        }
        for review_id in sorted(reviews)
        if review_id not in excluded
    })
    normalization_prompt = build_normalization_prompt(taxonomy)
    return {
        "schema_version": "reviewlens_holdout_selection_v1",
        "status": "frozen_before_inference",
        "algorithm": SELECTION_ALGORITHM,
        "seed": seed,
        "holdout_size": HOLDOUT_SIZE,
        "per_rating": HOLDOUT_PER_RATING,
        "excluded_pilot_review_ids": list(PILOT_REVIEW_IDS),
        "candidate_count": len(candidates),
        "candidate_rating_distribution": {
            str(rating): distribution[rating] for rating in range(1, 6)
        },
        "candidate_pool_hash": source_hash,
        "selected_reviews": selected_reviews,
        "model_id": model_id,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "claim_schema_version": CLAIM_SCHEMA_VERSION,
        "observation_prompt_hash": hash_text(OBSERVATION_PROMPT),
        "claim_prompt_hash": hash_text(normalization_prompt),
        "taxonomy_hash": taxonomy_hash(taxonomy),
        "model_code_hashes": _model_code_hashes(),
    }


def _without_created_at(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "created_at_utc"}


def _load_selection(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Invalid holdout selection manifest: {path}")
    return value


def prepare_holdout(
    project_root: str | Path,
    taxonomy: Mapping[str, Mapping[str, str]],
    model_id: str,
    seed: int = HOLDOUT_SEED,
) -> dict[str, Any]:
    """Freeze the deterministic 25-review selection without calling a model."""
    root = Path(project_root).resolve()
    paths = _paths(root)
    expected = _selection_document(root, taxonomy, model_id, seed)
    if paths["selection"].exists():
        existing = _load_selection(paths["selection"])
        if _without_created_at(existing) != expected:
            raise ValueError(
                "Existing holdout selection is immutable and no longer matches "
                "the source/config/code fingerprints. Do not overwrite it."
            )
        document = existing
        status = "already frozen and verified"
    else:
        unexpected = [
            paths[key] for key in ("run_manifest", "observations", "claims", "nli")
            if paths[key].exists()
        ]
        if unexpected:
            raise ValueError(
                "Holdout inference artifacts exist without a frozen selection: "
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
    print("\n=== EVIDENCE-FIRST HOLDOUT PREPARE ===")
    print(f"Selection {status}: {paths['selection']}")
    print("Reviews: 25 total, 5 per star; pilot overlap: 0")
    print(
        "First five ratings:",
        [row["rating"] for row in document["selected_reviews"][:5]],
    )
    print("No model was loaded or called.")
    return document


def _validated_selection(
    root: Path,
    taxonomy: Mapping[str, Mapping[str, str]],
    model_id: str,
) -> dict[str, Any]:
    paths = _paths(root)
    if not paths["selection"].exists():
        raise ValueError(
            "Holdout selection is not frozen. Run --mode ef-holdout-prepare first."
        )
    existing = _load_selection(paths["selection"])
    seed = existing.get("seed")
    if not isinstance(seed, int):
        raise ValueError("Invalid holdout selection seed")
    expected = _selection_document(root, taxonomy, model_id, seed)
    if _without_created_at(existing) != expected:
        raise ValueError(
            "Frozen holdout selection failed source/config/code fingerprint "
            "validation. Refusing inference."
        )
    return existing


def _write_or_validate_run_manifest(
    path: Path,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            existing = json.load(handle)
        if not isinstance(existing, dict):
            raise ValueError(f"Invalid holdout run manifest: {path}")
        mismatches = [
            key for key, value in expected.items() if existing.get(key) != value
        ]
        if mismatches:
            raise ValueError(
                f"Stale holdout run manifest; mismatched {mismatches}. "
                "Use --force only if you intend to rebuild holdout inference caches."
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


def run_holdout(
    project_root: str | Path,
    taxonomy: Mapping[str, Mapping[str, str]],
    model_id: str,
    limit: int = HOLDOUT_SIZE,
    force: bool = False,
    max_retries: int = 1,
    show: int = 5,
    generator: Generator | None = None,
) -> dict[str, Any]:
    """Run/resume only the frozen holdout; never expand beyond its 25 IDs."""
    if max_retries not in {0, 1}:
        raise ValueError("max_retries must be 0 or 1")
    if not 1 <= limit <= HOLDOUT_SIZE:
        raise ValueError("Evidence-first holdout --limit must be between 1 and 25")
    root = Path(project_root).resolve()
    paths = _paths(root)
    selection = _validated_selection(root, taxonomy, model_id)
    selected_universe = [
        str(row["review_id"]) for row in selection["selected_reviews"]
    ]
    selected_ids = selected_universe[:limit]

    if force:
        for key in ("observations", "claims", "nli", "run_manifest"):
            backup = backup_existing(paths[key])
            if backup is not None:
                print("Holdout backup:", backup)

    reviews, sentences_by_review, predictions_by_review = _prepare_inputs(root)
    sources_by_review = {
        review_id: _source_rows(reviews[review_id], sentences_by_review[review_id])
        for review_id in selected_universe
    }
    observation_prompt_digest = hash_text(OBSERVATION_PROMPT)
    normalization_prompt = build_normalization_prompt(taxonomy)
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
        "schema_version": "reviewlens_holdout_run_v1",
        "pipeline": "ReviewLens Evidence-first",
        "pipeline_version": "v1.4-frozen-holdout",
        "selection_manifest_hash": hash_json(selection),
        "selected_review_ids": selected_universe,
        "model_id": model_id,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "claim_schema_version": CLAIM_SCHEMA_VERSION,
        "observation_prompt_hash": observation_prompt_digest,
        "claim_prompt_hash": claim_prompt_digest,
        "taxonomy_hash": taxonomy_digest,
        "model_code_hashes": _model_code_hashes(),
        "old_artifacts_policy": "read_only",
    }
    # This is intentionally persisted before ensure_generator can load MLX.
    _write_or_validate_run_manifest(paths["run_manifest"], run_manifest_expected)

    observation_cache = load_stage_cache(
        paths["observations"],
        schema_version=OBSERVATION_SCHEMA_VERSION,
        model_id=model_id,
        prompt_hash=observation_prompt_digest,
        taxonomy_digest=taxonomy_digest,
        input_hashes=observation_input_hashes,
    )
    pending_observations = [
        review_id for review_id in selected_ids if review_id not in observation_cache
    ]
    print(
        "Evidence-first holdout observations: "
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
            system_prompt=OBSERVATION_PROMPT,
            payload=observation_payloads[review_id],
            parse_attempt=lambda response, current=review_id: _parse_observation_attempt(
                response, reviews[current], sources_by_review[current]
            ),
            accepted_key="accepted_observations",
            max_retries=max_retries,
        )
        row = {
            "review_id": review_id,
            "rating": reviews[review_id].get("rating"),
            "schema_version": OBSERVATION_SCHEMA_VERSION,
            "model_id": model_id,
            "prompt_hash": observation_prompt_digest,
            "taxonomy_hash": taxonomy_digest,
            "input_hash": observation_input_hashes[review_id],
            "parse_success": parse_success,
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
            f"rejected={len(row['rejected_observations'])}"
        )

    normalization_payloads = {
        review_id: _normalization_payload(observation_cache[review_id])
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
        schema_version=CLAIM_SCHEMA_VERSION,
        model_id=model_id,
        prompt_hash=claim_prompt_digest,
        taxonomy_digest=taxonomy_digest,
        input_hashes=claim_input_hashes,
        upstream_hashes=upstream_hashes,
    )
    pending_claims = [
        review_id for review_id in selected_ids if review_id not in claim_cache
    ]
    print(
        "Evidence-first holdout normalization: "
        f"selected={len(selected_ids)} "
        f"cached={len(selected_ids) - len(pending_claims)} "
        f"pending={len(pending_claims)}"
    )

    for review_id in pending_claims:
        observation_row = observation_cache[review_id]
        observations = observation_row["accepted_observations"]
        if not observations:
            row = _empty_claim_row(
                review_id=review_id,
                model_id=model_id,
                prompt_digest=claim_prompt_digest,
                taxonomy_digest=taxonomy_digest,
                input_digest=claim_input_hashes[review_id],
                upstream_digest=upstream_hashes[review_id],
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
            gate = complete_claims_with_unmapped_fallbacks(
                gate, observations, taxonomy
            )
            row = {
                "review_id": review_id,
                "schema_version": CLAIM_SCHEMA_VERSION,
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
            f"merged={row['duplicate_claims_merged']} "
            f"fallbacks={row.get('fallback_unmapped_count', 0)} "
            f"recoveries={row.get('deterministic_recovery_count', 0)}"
        )

    if isinstance(model_generator, _MlxGenerator):
        del model_generator
        gc.collect()
        try:
            import mlx.core as mx

            mx.clear_cache()
        except (ImportError, AttributeError):
            pass

    print("\n=== EVIDENCE-FIRST HOLDOUT ===")
    print(
        f"Frozen cohort: {HOLDOUT_SIZE}; requested prefix: {len(selected_ids)}; "
        f"completed claims: {sum(review_id in claim_cache for review_id in selected_universe)}"
    )
    for review_id in selected_ids[: max(show, 0)]:
        observations = observation_cache[review_id]["accepted_observations"]
        claims = claim_cache[review_id]["accepted_claims"]
        mapped = sum(item.get("mapping_status") == "mapped" for item in claims)
        print(
            f"  {review_id}: observations={len(observations)} "
            f"atomic_mapped={mapped} unmapped={len(claims) - mapped} "
            f"findings={len(claim_cache[review_id]['merged_claims'])}"
        )
    print("Saved only under:", paths["output_dir"])
    return {
        "selected_review_ids": selected_ids,
        "frozen_review_ids": selected_universe,
        "observation_rows": [observation_cache[value] for value in selected_ids],
        "claim_rows": [claim_cache[value] for value in selected_ids],
        "output_dir": str(paths["output_dir"]),
    }


__all__ = ["prepare_holdout", "run_holdout"]
