"""Resumable two-stage Evidence-first pilot using the existing local MLX model."""

from __future__ import annotations

import gc
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

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
    parse_json_object,
    read_jsonl,
    select_pilot_ids,
    stage_record_hash,
    taxonomy_hash,
    validate_claims,
    validate_observations,
)
from .prompts import OBSERVATION_PROMPT, build_normalization_prompt


Generator = Callable[[str, Mapping[str, Any], Mapping[str, Any] | None], str]

OBSERVATION_CACHE_NAME = "observations_pilot.jsonl"
CLAIM_CACHE_NAME = "claims_pilot.jsonl"
MANIFEST_NAME = "pilot_manifest.json"


def _paths(project_root: Path) -> dict[str, Path]:
    output_dir = project_root / "outputs/amazon/evidence_first"
    return {
        "reviews": project_root / "data/processed/amazon/acer_aspire5_reviews_clean.jsonl",
        "sentences": project_root / "data/processed/amazon/acer_aspire5_sentences.jsonl",
        "layer1": project_root / "outputs/amazon/acer_aspire5_full_predictions.jsonl",
        "output_dir": output_dir,
        "observations": output_dir / OBSERVATION_CACHE_NAME,
        "claims": output_dir / CLAIM_CACHE_NAME,
        "nli": output_dir / "nli_pilot.jsonl",
        "manifest": output_dir / MANIFEST_NAME,
    }


def _prepare_inputs(project_root: Path) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
]:
    paths = _paths(project_root)
    reviews = read_jsonl(paths["reviews"])
    sentences = read_jsonl(paths["sentences"])
    predictions = read_jsonl(paths["layer1"])

    review_by_id: dict[str, dict[str, Any]] = {}
    for row in reviews:
        review_id = row.get("review_id")
        if not isinstance(review_id, str) or review_id in review_by_id:
            raise ValueError(f"duplicate or missing review ID: {review_id}")
        review_by_id[review_id] = row

    sentences_by_review: dict[str, list[dict[str, Any]]] = {}
    sentence_ids: set[str] = set()
    for row in sentences:
        sentence_id = row.get("sentence_id")
        review_id = row.get("review_id")
        if (
            not isinstance(sentence_id, str)
            or not isinstance(review_id, str)
            or sentence_id in sentence_ids
            or review_id not in review_by_id
        ):
            raise ValueError(f"invalid or duplicate source sentence: {sentence_id}")
        sentence_ids.add(sentence_id)
        sentences_by_review.setdefault(review_id, []).append(row)

    predictions_by_review: dict[str, list[dict[str, Any]]] = {}
    prediction_ids: set[str] = set()
    for row in predictions:
        sentence_id = row.get("sentence_id")
        review_id = row.get("review_id")
        if (
            not isinstance(sentence_id, str)
            or not isinstance(review_id, str)
            or sentence_id in prediction_ids
            or sentence_id not in sentence_ids
        ):
            raise ValueError(f"invalid or duplicate Layer 1 prediction: {sentence_id}")
        prediction_ids.add(sentence_id)
        predictions_by_review.setdefault(review_id, []).append(row)

    if prediction_ids != sentence_ids:
        raise ValueError(
            "Layer 1 full cache does not exactly match the prepared sentence set"
        )

    for review_id in review_by_id:
        sentences_by_review.setdefault(review_id, [])
        predictions_by_review.setdefault(review_id, [])
        sentences_by_review[review_id].sort(
            key=lambda row: (row.get("start_char", -1), row["sentence_id"])
        )
        predictions_by_review[review_id].sort(
            key=lambda row: (row.get("start_char", -1), row["sentence_id"])
        )
        if [row["sentence_id"] for row in sentences_by_review[review_id]] != [
            row["sentence_id"] for row in predictions_by_review[review_id]
        ]:
            raise ValueError(f"Layer 1/source sentence order mismatch: {review_id}")

    return review_by_id, sentences_by_review, predictions_by_review


def _observation_payload(
    review: Mapping[str, Any],
    sources: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    predictions_by_id = {row["sentence_id"]: row for row in predictions}
    return {
        "review_id": review["review_id"],
        "product_name": review.get("product_name"),
        "product_model": review.get("product_model"),
        "review_title": review.get("title"),
        "full_review": review["clean_text"],
        "sentences_and_layer1_clues": [
            {
                "sentence_id": row["sentence_id"],
                "text": row["sentence_text"],
                "source_field": row["source_field"],
                "layer1_aspects": (
                    predictions_by_id[row["sentence_id"]].get("aspects")
                    if row["sentence_id"] in predictions_by_id
                    and predictions_by_id[row["sentence_id"]].get("parse_success")
                    else None
                ),
            }
            for row in sources
        ],
    }


def _source_rows(
    review: Mapping[str, Any],
    body_sentences: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Expose the review title as a first-class, exactly locatable source."""
    review_id = str(review["review_id"])
    sources: list[dict[str, Any]] = []
    title = review.get("title")
    if isinstance(title, str) and title.strip():
        title_id = f"{review_id}_TITLE"
        if any(row.get("sentence_id") == title_id for row in body_sentences):
            raise ValueError(f"synthetic title ID collides with a body sentence: {title_id}")
        sources.append({
            "review_id": review_id,
            "sentence_id": title_id,
            "sentence_text": title,
            "source_field": "title",
            "start_char": 0,
            "end_char": len(title),
        })
    for sentence in body_sentences:
        source = dict(sentence)
        source["source_field"] = "clean_text"
        sources.append(source)
    return sources


def _normalization_payload(observation_row: Mapping[str, Any]) -> dict[str, Any]:
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


class _MlxGenerator:
    """Load Qwen lazily only when a cache stage actually has pending work."""

    def __init__(self, model_id: str):
        from mlx_lm import generate, load
        from mlx_lm.sample_utils import make_sampler

        self._generate = generate
        self.model, self.tokenizer = load(model_id)
        if hasattr(self.model, "eval"):
            self.model.eval()
        self.sampler = make_sampler(temp=0.0)

    def __call__(
        self,
        system_prompt: str,
        payload: Mapping[str, Any],
        repair: Mapping[str, Any] | None = None,
    ) -> str:
        user_payload: Mapping[str, Any]
        if repair is None:
            user_payload = payload
        else:
            user_payload = {
                "original_input": payload,
                "previous_response": repair.get("raw"),
                "validation_errors": repair.get("errors", []),
                "instruction": (
                    "Regenerate the whole JSON object and follow every validation "
                    "error literally. Split listed product aspects into separate "
                    "observations, classify statement type from the exact quote, "
                    "preserve protected attribution, and use the frozen parent "
                    "category for each issue group. Use unmapped when a correction "
                    "would otherwise invent or broaden an observation or claim."
                ),
            }
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps(user_payload, ensure_ascii=False),
            },
        ]
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return self._generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            sampler=self.sampler,
            max_tokens=4096,
            verbose=False,
        )


def _parse_observation_attempt(
    raw: str,
    review: Mapping[str, Any],
    sentences: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any], bool]:
    try:
        parsed = parse_json_object(raw)
        gate = validate_observations(parsed, review, sentences)
        return parsed, gate, True
    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        return None, {
            "accepted_observations": [],
            "rejected_observations": [],
            "validation_errors": [f"parse_error: {exc}"],
            "reference_checks_passed": False,
        }, False


def _parse_claim_attempt(
    raw: str,
    observations: Sequence[Mapping[str, Any]],
    taxonomy: Mapping[str, Mapping[str, str]],
) -> tuple[dict[str, Any] | None, dict[str, Any], bool]:
    try:
        parsed = parse_json_object(raw)
        gate = validate_claims(parsed, observations, taxonomy)
        return parsed, gate, True
    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        return None, {
            "accepted_claims": [],
            "rejected_claims": [],
            "missing_observation_ids": [
                observation.get("observation_id") for observation in observations
            ],
            "validation_errors": [f"parse_error: {exc}"],
            "normalization_complete": False,
        }, False


def _gate_score(gate: Mapping[str, Any], accepted_key: str) -> tuple[int, int, int]:
    errors = gate.get("validation_errors") or []
    accepted = gate.get(accepted_key) or []
    return (int(not errors), -len(errors), len(accepted))


def _run_with_optional_repair(
    *,
    generator: Generator,
    system_prompt: str,
    payload: Mapping[str, Any],
    parse_attempt: Callable[[str], tuple[dict[str, Any] | None, dict[str, Any], bool]],
    accepted_key: str,
    max_retries: int,
) -> tuple[str, dict[str, Any] | None, dict[str, Any], bool, list[dict[str, Any]]]:
    raw = generator(system_prompt, payload, None)
    parsed, gate, parse_success = parse_attempt(raw)
    attempts = [
        {
            "attempt": 1,
            "parse_success": parse_success,
            "validation_errors": gate["validation_errors"],
            "raw_response": raw,
        }
    ]
    best = (raw, parsed, gate, parse_success)
    if gate["validation_errors"] and max_retries:
        repaired_raw = generator(
            system_prompt,
            payload,
            {"raw": raw, "errors": gate["validation_errors"]},
        )
        repaired_parsed, repaired_gate, repaired_success = parse_attempt(repaired_raw)
        attempts.append(
            {
                "attempt": 2,
                "parse_success": repaired_success,
                "validation_errors": repaired_gate["validation_errors"],
                "raw_response": repaired_raw,
            }
        )
        if _gate_score(repaired_gate, accepted_key) > _gate_score(gate, accepted_key):
            best = (repaired_raw, repaired_parsed, repaired_gate, repaired_success)
    return (*best, attempts)


def _write_or_validate_manifest(
    path: Path,
    expected: Mapping[str, Any],
) -> None:
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            existing = json.load(handle)
        if not isinstance(existing, dict):
            raise ValueError(f"invalid Evidence-first manifest: {path}")
        mismatches = [
            key for key, value in expected.items() if existing.get(key) != value
        ]
        if mismatches:
            raise ValueError(
                f"stale Evidence-first manifest; mismatched {mismatches}. "
                "Use --force or --force-claims to back it up."
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    document = dict(expected)
    document["created_at_local"] = datetime.now().isoformat(timespec="seconds")
    with path.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, ensure_ascii=False, indent=2)


def _empty_claim_row(
    *,
    review_id: str,
    model_id: str,
    prompt_digest: str,
    taxonomy_digest: str,
    input_digest: str,
    upstream_digest: str,
) -> dict[str, Any]:
    row = {
        "review_id": review_id,
        "schema_version": CLAIM_SCHEMA_VERSION,
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


def _claim_record_hash(row: Mapping[str, Any]) -> str:
    return hash_json({
        "review_id": row.get("review_id"),
        "schema_version": row.get("schema_version"),
        "accepted_claims": row.get("accepted_claims", []),
        "merged_claims": row.get("merged_claims", []),
    })


def run_pilot(
    project_root: str | Path,
    taxonomy: Mapping[str, Mapping[str, str]],
    model_id: str,
    limit: int = 10,
    force: bool = False,
    max_retries: int = 1,
    show: int = 5,
    generator: Generator | None = None,
    force_claims: bool = False,
) -> dict[str, Any]:
    """Run/resume at most the fixed 10-review Evidence-first pilot.

    Existing Layer 1 and Layer 2 v3 files are read-only. All new state lives in
    the outputs/amazon/evidence_first directory.
    """
    if max_retries not in {0, 1}:
        raise ValueError("max_retries must be 0 or 1")
    if force and force_claims:
        raise ValueError("force and force_claims are mutually exclusive")
    root = Path(project_root).resolve()
    paths = _paths(root)
    paths["output_dir"].mkdir(parents=True, exist_ok=True)

    if force:
        # NLI rows are keyed by claim IDs; retaining them across a regenerated
        # pilot could silently attach old scores to different new claims.
        for key in ("observations", "claims", "nli", "manifest"):
            backup = backup_existing(paths[key])
            if backup is not None:
                print("Evidence-first backup:", backup)

    reviews, sentences_by_review, predictions_by_review = _prepare_inputs(root)
    sources_by_review = {
        review_id: _source_rows(reviews[review_id], sentences_by_review[review_id])
        for review_id in reviews
    }
    pilot_universe = select_pilot_ids(reviews, len(PILOT_REVIEW_IDS))
    selected_ids = select_pilot_ids(reviews, limit)

    observation_prompt_digest = hash_text(OBSERVATION_PROMPT)
    normalization_prompt = build_normalization_prompt(taxonomy)
    claim_prompt_digest = hash_text(normalization_prompt)
    taxonomy_digest = taxonomy_hash(taxonomy)

    observation_payloads = {
        review_id: _observation_payload(
            reviews[review_id], sources_by_review[review_id],
            predictions_by_review[review_id]
        )
        for review_id in pilot_universe
    }
    observation_input_hashes = {
        review_id: hash_json(payload)
        for review_id, payload in observation_payloads.items()
    }
    source_digest = hash_json(
        {
            review_id: {
                "review": reviews[review_id],
                "sources": sources_by_review[review_id],
                "layer1": predictions_by_review[review_id],
            }
            for review_id in pilot_universe
        }
    )
    manifest = {
        "pipeline": "ReviewLens Evidence-first",
        "pipeline_version": "v1.4",
        "pilot_review_ids": pilot_universe,
        "source_hash": source_digest,
        "model_id": model_id,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "claim_schema_version": CLAIM_SCHEMA_VERSION,
        "observation_prompt_hash": observation_prompt_digest,
        "claim_prompt_hash": claim_prompt_digest,
        "taxonomy_hash": taxonomy_digest,
        "old_artifacts_policy": "read_only",
    }
    if force_claims:
        # Validate upstream before moving downstream state. This mode promises
        # to preserve observations, not silently regenerate them.
        observation_cache = load_stage_cache(
            paths["observations"],
            schema_version=OBSERVATION_SCHEMA_VERSION,
            model_id=model_id,
            prompt_hash=observation_prompt_digest,
            taxonomy_digest=taxonomy_digest,
            input_hashes=observation_input_hashes,
        )
        missing_upstream = [
            review_id for review_id in selected_ids
            if review_id not in observation_cache
        ]
        if missing_upstream:
            raise ValueError(
                "--force-claims requires a complete compatible observation "
                f"cache; missing {missing_upstream}"
            )
        for key in ("claims", "nli", "manifest"):
            backup = backup_existing(paths[key])
            if backup is not None:
                print("Evidence-first backup:", backup)
        _write_or_validate_manifest(paths["manifest"], manifest)
    else:
        _write_or_validate_manifest(paths["manifest"], manifest)
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
        "Evidence-first observations: "
        f"selected={len(selected_ids)} cached={len(selected_ids)-len(pending_observations)} "
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
        review = reviews[review_id]
        payload = observation_payloads[review_id]
        used_generator = ensure_generator()
        raw, parsed, gate, parse_success, attempts = _run_with_optional_repair(
            generator=used_generator,
            system_prompt=OBSERVATION_PROMPT,
            payload=payload,
            parse_attempt=lambda response, r=review_id: _parse_observation_attempt(
                response,
                reviews[r],
                sources_by_review[r],
            ),
            accepted_key="accepted_observations",
            max_retries=max_retries,
        )
        row = {
            "review_id": review_id,
            "rating": review.get("rating"),
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
        for review_id in pilot_universe
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
        "Evidence-first normalization: "
        f"selected={len(selected_ids)} cached={len(selected_ids)-len(pending_claims)} "
        f"pending={len(pending_claims)}"
    )

    for review_id in pending_claims:
        observation_row = observation_cache[review_id]
        observations = observation_row["accepted_observations"]
        payload = normalization_payloads[review_id]
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
                payload=payload,
                parse_attempt=lambda response, obs=observations: _parse_claim_attempt(
                    response, obs, taxonomy
                ),
                accepted_key="accepted_claims",
                max_retries=max_retries,
            )
            gate = complete_claims_with_unmapped_fallbacks(
                gate,
                observations,
                taxonomy,
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
            f"  {review_id}: atomic_claims={len(row['accepted_claims'])} "
            f"mapped={mapped} unmapped={len(row['accepted_claims'])-mapped} "
            f"business_findings={len(row['merged_claims'])} "
            f"merged_duplicates={row['duplicate_claims_merged']} "
            f"fallbacks={row.get('fallback_unmapped_count', 0)} "
            f"recoveries={row.get('deterministic_recovery_count', 0)} "
            f"rejected={len(row['rejected_claims'])}"
        )

    if isinstance(model_generator, _MlxGenerator):
        del model_generator
        gc.collect()
        try:
            import mlx.core as mx

            mx.clear_cache()
        except (ImportError, AttributeError):
            pass

    print("\n=== EVIDENCE-FIRST PILOT ===")
    for review_id in selected_ids[: max(show, 0)]:
        observations = observation_cache[review_id]["accepted_observations"]
        claims = claim_cache[review_id]["accepted_claims"]
        business_findings = claim_cache[review_id]["merged_claims"]
        mapped = [claim for claim in claims if claim["mapping_status"] == "mapped"]
        print(
            f"{review_id}: observations={len(observations)} "
            f"atomic_mapped={len(mapped)} unmapped={len(claims)-len(mapped)} "
            f"business_findings={len(business_findings)} "
            f"merged_duplicates={claim_cache[review_id]['duplicate_claims_merged']} "
            f"fallbacks={claim_cache[review_id].get('fallback_unmapped_count', 0)} "
            f"recoveries={claim_cache[review_id].get('deterministic_recovery_count', 0)}"
        )
        for claim in claims[:3]:
            label = (
                f"{claim['category']}/{claim['issue_group']}"
                if claim["mapping_status"] == "mapped"
                else "unmapped"
            )
            print(f"  - {claim['claim_id']} {label}: {claim['claim']}")
    print("Saved only under:", paths["output_dir"])

    return {
        "selected_review_ids": selected_ids,
        "observation_rows": [observation_cache[review_id] for review_id in selected_ids],
        "claim_rows": [claim_cache[review_id] for review_id in selected_ids],
        "output_dir": str(paths["output_dir"]),
    }
