"""Cache-isolated Evidence-first transfer pipeline for configured laptops.

This path is separate from the frozen Aspire benchmark. Every output is labelled
as an unvalidated transfer demo until that product gets blind annotation.
"""

from __future__ import annotations

import gc
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .core import (
    append_jsonl, hash_json, hash_text, merge_claims_by_tuple, read_jsonl,
    stage_record_hash, taxonomy_hash,
)
from .laptop_prompts import (
    PROMPT_VERSION, build_laptop_normalization_prompt,
    build_laptop_observation_prompt,
)
from .pipeline import (
    Generator, _MlxGenerator, _claim_record_hash, _observation_payload,
    _parse_claim_attempt, _run_with_optional_repair,
)
from .v15_core import (
    CLAIM_SCHEMA_VERSION_V15, OBSERVATION_SCHEMA_VERSION_V15,
    POLICY_VERSION_V15, complete_claims_v15, normalization_payload_v15,
    parse_observation_attempt_v15,
)

TRANSFER_STATUS = "transfer_demo_unvalidated"
PIPELINE_VERSION = "reviewlens_laptop_evidence_first_v1"
LAYER1_SCHEMA_VERSION = "reviewlens_laptop_layer1_v1"
SUMMARY_SCHEMA_VERSION = "reviewlens_laptop_summary_v1"
DEFAULT_L1_MODEL = "Qwen/Qwen3-1.7B-MLX-4bit"
DEFAULT_L2_MODEL = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
DEFAULT_L1_ADAPTER = "outputs/adapters/reviewlens_layers16"
DEFAULT_L1_CHECKPOINT = "outputs/adapters/reviewlens_layers16/0001200_adapters.safetensors"
Layer1Generator = Callable[[str, str], str]


def _file_hash(path: Path) -> str:
    if not path.exists():
        return "missing"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_code_hashes(root: Path) -> dict[str, str]:
    """Bind a transfer run to both new orchestration and frozen validators."""
    names = (
        "reviewlens_ef/laptop_products.py",
        "reviewlens_ef/laptop_prompts.py",
        "reviewlens_ef/laptop_pipeline.py",
        "reviewlens_ef/core.py",
        "reviewlens_ef/pipeline.py",
        "reviewlens_ef/v15_core.py",
        "reviewlens_ef/v15_prompts.py",
    )
    return {name: _file_hash((root / name).resolve()) for name in names}


def _local_huggingface_revision(model_id: str) -> str | None:
    """Record the local Hub revision without loading or downloading a model."""
    repository = Path.home() / ".cache/huggingface/hub" / (
        "models--" + model_id.replace("/", "--")
    )
    reference = repository / "refs/main"
    if reference.is_file():
        revision = reference.read_text(encoding="utf-8").strip()
        if revision:
            return revision
    snapshots = sorted(
        path.name for path in (repository / "snapshots").glob("*") if path.is_dir()
    )
    return snapshots[0] if len(snapshots) == 1 else None


def _read_l1_system_prompt(root: Path) -> str:
    rows = read_jsonl(root / "data/processed/semeval/train.jsonl")
    try:
        prompt = rows[0]["messages"][0]["content"]
    except (IndexError, KeyError, TypeError) as exc:
        raise ValueError("SemEval Layer 1 system prompt is missing") from exc
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("SemEval Layer 1 system prompt is empty")
    return prompt


def _parse_l1(raw: str) -> list[dict[str, str]]:
    cleaned = re.sub(r"^\s*<think>.*?</think>\s*", "", raw, flags=re.DOTALL).strip()
    value = json.loads(cleaned)
    if not isinstance(value, list):
        raise ValueError("Layer 1 response must be a list")
    allowed = {"positive", "negative", "neutral", "conflict"}
    result: list[dict[str, str]] = []
    for item in value:
        if (
            not isinstance(item, dict)
            or set(item) != {"term", "sentiment"}
            or not isinstance(item.get("term"), str)
            or not item["term"].strip()
            or item.get("sentiment") not in allowed
        ):
            raise ValueError("invalid Layer 1 aspect record")
        result.append({"term": item["term"].strip(), "sentiment": str(item["sentiment"])})
    return result


class _LaptopLayer1Generator:
    """Load the existing fine-tuned Layer 1 only for an incomplete cache."""

    def __init__(self, model_id: str, adapter: Path, checkpoint: Path) -> None:
        if not adapter.is_dir():
            raise FileNotFoundError(f"Layer 1 adapter directory not found: {adapter}")
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Layer 1 checkpoint not found: {checkpoint}")
        import mlx.core as mx
        from mlx_lm import generate, load
        from mlx_lm.sample_utils import make_sampler

        self._generate = generate
        self.model, self.tokenizer = load(model_id, adapter_path=str(adapter))
        self.model.load_weights(str(checkpoint), strict=False)
        mx.eval(self.model.parameters())
        if hasattr(self.model, "eval"):
            self.model.eval()
        self.sampler = make_sampler(temp=0.0)

    def __call__(self, system_prompt: str, sentence: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": sentence},
        ]
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        if not prompt.rstrip().endswith("</think>") and prompt.endswith("<|im_start|>assistant\n"):
            prompt += "<think>\n\n</think>\n\n"
        return self._generate(
            self.model, self.tokenizer, prompt=prompt, sampler=self.sampler,
            max_tokens=256, verbose=False,
        )


def _release_model(value: Any) -> None:
    close = getattr(value, "close", None)
    if callable(close):
        close()
    del value
    gc.collect()
    # Do not import MLX just to release an injected fake generator in tests.
    # A real local model has already loaded this module by this point.
    mx = sys.modules.get("mlx.core")
    if mx is not None and hasattr(mx, "clear_cache"):
        mx.clear_cache()


def _write_or_validate_json(path: Path, document: Mapping[str, Any]) -> None:
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != document:
            raise ValueError(f"immutable laptop run document changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _prepared_path(prepared: Any, name: str) -> Path:
    value = getattr(prepared, name, None)
    if value is None:
        raise ValueError(f"prepared laptop result is missing {name}")
    return Path(value).resolve()


def _load_prefix_cache(
    path: Path, *, id_key: str, expected_ids: Sequence[str],
    common: Mapping[str, Any], input_hashes: Mapping[str, str],
    upstream_hashes: Mapping[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = read_jsonl(path)
    cached_ids = [row.get(id_key) for row in rows]
    if cached_ids != list(expected_ids[:len(cached_ids)]):
        raise ValueError(f"cache is not a complete ordered prefix: {path}")
    saved: dict[str, dict[str, Any]] = {}
    for row in rows:
        item_id = row.get(id_key)
        if not isinstance(item_id, str) or item_id in saved:
            raise ValueError(f"duplicate or missing cache ID in {path}: {item_id}")
        expected = {**common, "input_hash": input_hashes[item_id]}
        if upstream_hashes is not None:
            expected["upstream_hash"] = upstream_hashes[item_id]
        mismatches = [key for key, value in expected.items() if row.get(key) != value]
        if mismatches:
            raise ValueError(f"stale cache row {item_id} in {path}; mismatched {mismatches}")
        saved[item_id] = row
    return saved


def _load_inputs(prepared: Any) -> tuple[
    list[str], dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]],
    list[dict[str, Any]],
]:
    selected = [str(value) for value in prepared.selected_review_ids]
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("prepared selection must contain unique review IDs")
    review_rows = read_jsonl(_prepared_path(prepared, "reviews_path"))
    sentence_rows = read_jsonl(_prepared_path(prepared, "sentences_path"))
    title_path = _prepared_path(prepared, "titles_path")
    title_rows = read_jsonl(title_path) if title_path.exists() else []
    reviews = {str(row["review_id"]): row for row in review_rows}
    if list(reviews) != selected:
        raise ValueError("prepared reviews do not match configured selection order")

    bodies: dict[str, list[dict[str, Any]]] = {key: [] for key in selected}
    titles: dict[str, list[dict[str, Any]]] = {key: [] for key in selected}
    seen: set[str] = set()
    for source in [*title_rows, *sentence_rows]:
        row = dict(source)
        sid, rid = row.get("sentence_id"), row.get("review_id")
        if not isinstance(sid, str) or sid in seen or rid not in reviews:
            raise ValueError(f"invalid prepared source record: {sid}")
        seen.add(sid)
        field = row.get("source_field", "clean_text")
        if field not in {"title", "clean_text"}:
            raise ValueError(f"invalid source field: {field}")
        row["source_field"] = field
        source_text = reviews[str(rid)].get(field)
        start, end = row.get("start_char"), row.get("end_char")
        text = row.get("sentence_text")
        if (
            not isinstance(source_text, str) or not isinstance(start, int)
            or not isinstance(end, int) or not isinstance(text, str) or start < 0
            or end != start + len(text) or source_text[start:end] != text
        ):
            raise ValueError(f"source offsets do not locate exact text: {sid}")
        (titles if field == "title" else bodies)[str(rid)].append(row)

    sources: dict[str, list[dict[str, Any]]] = {}
    ordered_bodies: list[dict[str, Any]] = []
    for rid in selected:
        if len(titles[rid]) > 1:
            raise ValueError(f"multiple title records for {rid}")
        titles[rid].sort(key=lambda row: str(row["sentence_id"]))
        bodies[rid].sort(key=lambda row: (row["start_char"], row["sentence_id"]))
        sources[rid] = titles[rid] + bodies[rid]
        ordered_bodies.extend(bodies[rid])
    return selected, reviews, sources, ordered_bodies


def _empty_claim(
    product_id: str, review_id: str, model_id: str, prompt_hash: str,
    taxonomy_digest: str, input_hash: str, upstream_hash: str,
    fingerprint: str,
) -> dict[str, Any]:
    row = {
        "product_id": product_id, "review_id": review_id,
        "transfer_status": TRANSFER_STATUS, "run_fingerprint": fingerprint,
        "schema_version": CLAIM_SCHEMA_VERSION_V15, "model_id": model_id,
        "policy_version": POLICY_VERSION_V15,
        "prompt_hash": prompt_hash, "taxonomy_hash": taxonomy_digest,
        "input_hash": input_hash, "upstream_hash": upstream_hash,
        "parse_success": True, "normalization_complete": True,
        "model_normalization_complete": True, "fallback_unmapped_count": 0,
        "deterministic_recovery_count": 0, "v15_recall_recovery_count": 0,
        "v15_category_repair_count": 0, "accepted_claims": [],
        "merged_claims": [], "duplicate_claims_merged": 0,
        "rejected_claims": [], "missing_observation_ids": [],
        "validation_errors": [], "model_validation_errors": [],
        "model_missing_observation_ids": [], "attempts": [],
        "note": "No validated observations; normalization model was not called.",
    }
    row["record_hash"] = _claim_record_hash(row)
    return row


def build_product_summary(
    *, product_id: str, selected_review_ids: Sequence[str],
    reviews: Mapping[str, Mapping[str, Any]],
    observation_rows: Sequence[Mapping[str, Any]],
    claim_rows: Sequence[Mapping[str, Any]], run_fingerprint: str,
) -> dict[str, Any]:
    """Aggregate review support rather than inflated atomic-claim frequency."""
    denominator = len(selected_review_ids)
    issue_map: dict[tuple[str, str, str], dict[str, Any]] = {}
    reviews_with_findings: set[str] = set()
    mapped = unmapped = finding_count = 0
    for row in claim_rows:
        rid = str(row["review_id"])
        claims = list(row.get("accepted_claims", []) or [])
        mapped += sum(c.get("mapping_status") == "mapped" for c in claims)
        unmapped += sum(c.get("mapping_status") != "mapped" for c in claims)
        findings = list(row.get("merged_claims", []) or [])
        finding_count += len(findings)
        if findings:
            reviews_with_findings.add(rid)
        for finding in findings:
            key = (
                str(finding["category"]), str(finding["issue_group"]),
                str(finding["sentiment"]),
            )
            item = issue_map.setdefault(key, {
                "category": key[0], "issue_group": key[1], "sentiment": key[2],
                "supporting_review_ids": [], "evidence": [],
            })
            if rid not in item["supporting_review_ids"]:
                item["supporting_review_ids"].append(rid)
            for evidence in finding.get("evidence", []) or []:
                value = {"review_id": rid, **dict(evidence)}
                if value not in item["evidence"]:
                    item["evidence"].append(value)
    issues: list[dict[str, Any]] = []
    for key in sorted(issue_map):
        item = dict(issue_map[key])
        count = len(item["supporting_review_ids"])
        item["supporting_review_count"] = count
        item["supporting_review_rate"] = round(count / denominator, 6)
        issues.append(item)
    first = reviews[selected_review_ids[0]]
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION, "status": TRANSFER_STATUS,
        "validation_note": (
            "Product-specific accuracy has not been established by blind human "
            "annotation; this demonstrates configurable laptop transfer only."
        ),
        "product_id": product_id, "product_name": first.get("product_name"),
        "product_model": first.get("product_model"),
        "run_fingerprint": run_fingerprint,
        "selected_review_ids": list(selected_review_ids),
        "reviews_analyzed": denominator,
        "reviews_with_mapped_findings": len(reviews_with_findings),
        "mapped_finding_review_coverage": round(len(reviews_with_findings) / denominator, 6),
        "accepted_observation_count": sum(
            len(row.get("accepted_observations", []) or []) for row in observation_rows
        ),
        "atomic_mapped_count": mapped, "atomic_unmapped_count": unmapped,
        "review_level_business_finding_count": finding_count, "issues": issues,
    }


def run_laptop_pilot(
    project_root: str | Path,
    product_config: str | Path,
    *,
    limit: int = 10,
    show: int = 5,
    l2_model_id: str = DEFAULT_L2_MODEL,
    l2_model_revision: str | None = None,
    l1_model_id: str = DEFAULT_L1_MODEL,
    l1_adapter_path: str | Path | None = None,
    l1_checkpoint_path: str | Path | None = None,
    max_retries: int = 1,
    skip_layer1: bool = False,
    layer1_generator: Layer1Generator | None = None,
    generator: Generator | None = None,
    layer1_system_prompt: str | None = None,
    stage: str = "all",
    manage_model_lifecycle: bool = True,
) -> dict[str, Any]:
    """Run/resume a fixed, config-ordered laptop transfer pilot."""
    if limit < 1:
        raise ValueError("limit must be positive")
    if max_retries not in {0, 1}:
        raise ValueError("max_retries must be 0 or 1")
    if stage not in {"all", "layer1", "layer2"}:
        raise ValueError("stage must be one of: all, layer1, layer2")
    if stage == "all" and not manage_model_lifecycle:
        raise ValueError("stage=all must manage model lifecycle between layers")
    if stage == "layer2" and layer1_generator is not None:
        raise ValueError("stage=layer2 does not accept a Layer 1 generator")
    root = Path(project_root).resolve()
    from . import laptop_products

    config = laptop_products.load_laptop_product_config(root, product_config)
    prepared = laptop_products.prepare_laptop_product(root, product_config, limit=limit)
    taxonomy = laptop_products.load_laptop_taxonomy(root, config)
    product_id = str(prepared.product_id)
    output_dir = _prepared_path(prepared, "output_dir")
    expected_output = (root / "outputs/laptops" / product_id).resolve()
    if output_dir != expected_output:
        raise ValueError(
            f"laptop output isolation violation: expected {expected_output}, got {output_dir}"
        )
    selected, reviews, sources, bodies = _load_inputs(prepared)
    observation_prompt = build_laptop_observation_prompt()
    normalization_prompt = build_laptop_normalization_prompt(taxonomy)
    observation_prompt_hash = hash_text(observation_prompt)
    claim_prompt_hash = hash_text(normalization_prompt)
    taxonomy_digest = taxonomy_hash(taxonomy)
    l1_prompt = layer1_system_prompt or _read_l1_system_prompt(root)
    l1_prompt_hash = hash_text(l1_prompt)
    adapter = (
        Path(l1_adapter_path).resolve() if l1_adapter_path is not None
        else (root / DEFAULT_L1_ADAPTER).resolve()
    )
    checkpoint = (
        Path(l1_checkpoint_path).resolve() if l1_checkpoint_path is not None
        else (root / DEFAULT_L1_CHECKPOINT).resolve()
    )
    artifacts = {
        name: _prepared_path(prepared, name)
        for name in ("reviews_path", "sentences_path", "titles_path", "manifest_path")
    }
    config_hash = str(
        getattr(prepared, "config_sha256", None)
        or getattr(config, "config_sha256", None)
        or _file_hash((root / Path(product_config)).resolve())
    )
    l1_model_revision = _local_huggingface_revision(l1_model_id)
    resolved_l2_revision = (
        l2_model_revision or _local_huggingface_revision(l2_model_id)
    )
    fingerprint_data = {
        "pipeline_version": PIPELINE_VERSION, "prompt_version": PROMPT_VERSION,
        "product_id": product_id, "config_hash": config_hash,
        "model_code_hashes": _model_code_hashes(root),
        "prepared_artifact_hashes": {key: _file_hash(path) for key, path in artifacts.items()},
        "selected_review_ids": selected, "taxonomy_hash": taxonomy_digest,
        "observation_prompt_hash": observation_prompt_hash,
        "claim_prompt_hash": claim_prompt_hash,
        "layer1": {
            "enabled": not skip_layer1, "model_id": l1_model_id,
            "model_revision": l1_model_revision,
            "adapter_path": str(adapter),
            "adapter_config_hash": _file_hash(adapter / "adapter_config.json"),
            "checkpoint_hash": _file_hash(checkpoint),
            "system_prompt_hash": l1_prompt_hash, "temperature": 0.0,
            "max_tokens": 256,
        },
        "layer2": {
            "model_id": l2_model_id, "model_revision": resolved_l2_revision,
            "temperature": 0.0, "max_tokens": 4096,
            "max_retries": max_retries,
        },
        "policy_version": POLICY_VERSION_V15,
    }
    fingerprint = hash_json(fingerprint_data)
    run_dir = output_dir / "evidence_first/runs" / fingerprint
    paths = {
        "manifest": run_dir / "run_manifest.json",
        "layer1": run_dir / "layer1_predictions.jsonl",
        "observations": run_dir / "observations.jsonl",
        "claims": run_dir / "claims.jsonl",
        "summary": run_dir / "product_summary.json",
    }
    _write_or_validate_json(paths["manifest"], {
        "schema_version": "reviewlens_laptop_run_manifest_v1",
        "status": TRANSFER_STATUS, "run_fingerprint": fingerprint,
        **fingerprint_data,
    })

    # Titles remain valid Layer 2 evidence; the trained Layer 1 sees body sentences.
    sentence_ids = [str(row["sentence_id"]) for row in bodies]
    sentence_by_id = {str(row["sentence_id"]): row for row in bodies}
    l1_inputs = {
        sid: hash_json({
            "product_id": product_id, "sentence": sentence_by_id[sid],
            "prompt_hash": l1_prompt_hash,
        }) for sid in sentence_ids
    }
    l1_common = {
        "product_id": product_id, "run_fingerprint": fingerprint,
        "schema_version": LAYER1_SCHEMA_VERSION, "model_id": l1_model_id,
        "prompt_hash": l1_prompt_hash,
    }
    l1_cache = {} if skip_layer1 else _load_prefix_cache(
        paths["layer1"], id_key="sentence_id", expected_ids=sentence_ids,
        common=l1_common, input_hashes=l1_inputs,
    )
    pending_l1 = [sid for sid in sentence_ids if sid not in l1_cache]
    if stage == "layer2" and pending_l1:
        raise ValueError(
            f"stage=layer2 requires a complete Layer 1 cache; "
            f"{len(pending_l1)} sentence(s) are pending"
        )
    if pending_l1 and not manage_model_lifecycle and layer1_generator is None:
        raise ValueError(
            "manage_model_lifecycle=False requires an injected Layer 1 generator"
        )
    print(
        "Laptop Layer 1: explicitly skipped; using empty clues"
        if skip_layer1 else
        f"Laptop Layer 1: selected={len(sentence_ids)} cached={len(l1_cache)} pending={len(pending_l1)}"
    )
    active_l1 = layer1_generator
    if pending_l1:
        if active_l1 is None:
            print("Loading laptop Layer 1:", l1_model_id)
            active_l1 = _LaptopLayer1Generator(l1_model_id, adapter, checkpoint)
        for index, sid in enumerate(pending_l1, 1):
            source = sentence_by_id[sid]
            raw = active_l1(l1_prompt, str(source["sentence_text"]))
            try:
                aspects, success, error = _parse_l1(raw), True, None
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                aspects, success, error = None, False, str(exc)
            row = {
                **l1_common, "sentence_id": sid, "review_id": source["review_id"],
                "rating": source.get("rating"), "sentence_text": source["sentence_text"],
                "start_char": source["start_char"], "end_char": source["end_char"],
                "input_hash": l1_inputs[sid], "aspects": aspects,
                "parse_success": success, "parse_error": error, "raw_response": raw,
            }
            row["record_hash"] = hash_json({
                "sentence_id": sid, "schema_version": LAYER1_SCHEMA_VERSION,
                "aspects": aspects, "parse_success": success,
            })
            append_jsonl(paths["layer1"], row)
            l1_cache[sid] = row
            print(f"Laptop Layer 1 progress: {index}/{len(pending_l1)}")
    if active_l1 is not None and manage_model_lifecycle:
        _release_model(active_l1)
        active_l1 = None
    if stage == "layer1":
        return {
            "product_id": product_id,
            "status": TRANSFER_STATUS,
            "stage": "layer1_complete",
            "selected_review_ids": selected,
            "output_dir": str(output_dir),
            "run_dir": str(run_dir),
            "run_fingerprint": fingerprint,
            "paths": {key: str(value) for key, value in paths.items()},
        }

    predictions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if not skip_layer1:
        for sid in sentence_ids:
            predictions[str(l1_cache[sid]["review_id"])].append(l1_cache[sid])
    observation_payloads = {
        rid: _observation_payload(reviews[rid], sources[rid], predictions.get(rid, []))
        for rid in selected
    }
    observation_inputs = {
        rid: hash_json(payload) for rid, payload in observation_payloads.items()
    }
    observation_common = {
        "product_id": product_id, "transfer_status": TRANSFER_STATUS,
        "run_fingerprint": fingerprint,
        "schema_version": OBSERVATION_SCHEMA_VERSION_V15,
        "policy_version": POLICY_VERSION_V15,
        "model_id": l2_model_id, "prompt_hash": observation_prompt_hash,
        "taxonomy_hash": taxonomy_digest,
    }
    observation_cache = _load_prefix_cache(
        paths["observations"], id_key="review_id", expected_ids=selected,
        common=observation_common, input_hashes=observation_inputs,
    )
    pending_observations = [rid for rid in selected if rid not in observation_cache]
    print(
        f"Laptop observations: selected={len(selected)} cached={len(observation_cache)} "
        f"pending={len(pending_observations)}"
    )
    active_l2 = generator

    def ensure_l2() -> Generator:
        nonlocal active_l2
        if active_l2 is None:
            if not manage_model_lifecycle:
                raise ValueError(
                    "manage_model_lifecycle=False requires an injected Layer 2 generator"
                )
            print("Loading laptop Evidence-first Layer 2:", l2_model_id)
            active_l2 = _MlxGenerator(l2_model_id)
        return active_l2

    for index, rid in enumerate(pending_observations, 1):
        raw, parsed, gate, success, attempts = _run_with_optional_repair(
            generator=ensure_l2(), system_prompt=observation_prompt,
            payload=observation_payloads[rid],
            parse_attempt=lambda response, review_id=rid: parse_observation_attempt_v15(
                response, reviews[review_id], sources[review_id]
            ),
            accepted_key="accepted_observations", max_retries=max_retries,
        )
        row = {
            **observation_common, "review_id": rid,
            "rating": reviews[rid].get("rating"),
            "input_hash": observation_inputs[rid], "parse_success": success,
            "reference_checks_passed": gate["reference_checks_passed"],
            "context_recovery_count": gate.get("context_recovery_count", 0),
            "parse_adjustments": gate.get("parse_adjustments", []),
            "accepted_observations": gate["accepted_observations"],
            "rejected_observations": gate["rejected_observations"],
            "validation_errors": gate["validation_errors"],
            "model_result_raw": parsed, "selected_raw_response": raw,
            "attempts": attempts,
        }
        row["record_hash"] = stage_record_hash(row, "accepted_observations")
        append_jsonl(paths["observations"], row)
        observation_cache[rid] = row
        print(f"Laptop observation progress: {index}/{len(pending_observations)}")

    normalization_payloads = {
        rid: normalization_payload_v15(observation_cache[rid]) for rid in selected
    }
    claim_inputs = {
        rid: hash_json(payload) for rid, payload in normalization_payloads.items()
    }
    upstream = {rid: str(observation_cache[rid]["record_hash"]) for rid in selected}
    claim_common = {
        "product_id": product_id, "transfer_status": TRANSFER_STATUS,
        "run_fingerprint": fingerprint, "schema_version": CLAIM_SCHEMA_VERSION_V15,
        "policy_version": POLICY_VERSION_V15,
        "model_id": l2_model_id, "prompt_hash": claim_prompt_hash,
        "taxonomy_hash": taxonomy_digest,
    }
    claim_cache = _load_prefix_cache(
        paths["claims"], id_key="review_id", expected_ids=selected,
        common=claim_common, input_hashes=claim_inputs, upstream_hashes=upstream,
    )
    pending_claims = [rid for rid in selected if rid not in claim_cache]
    print(
        f"Laptop normalization: selected={len(selected)} cached={len(claim_cache)} "
        f"pending={len(pending_claims)}"
    )
    for index, rid in enumerate(pending_claims, 1):
        observations = observation_cache[rid]["accepted_observations"]
        if not observations:
            row = _empty_claim(
                product_id, rid, l2_model_id, claim_prompt_hash,
                taxonomy_digest, claim_inputs[rid], upstream[rid], fingerprint,
            )
        else:
            raw, parsed, gate, success, attempts = _run_with_optional_repair(
                generator=ensure_l2(), system_prompt=normalization_prompt,
                payload=normalization_payloads[rid],
                parse_attempt=lambda response, obs=observations: _parse_claim_attempt(
                    response, obs, taxonomy
                ),
                accepted_key="accepted_claims", max_retries=max_retries,
            )
            gate = complete_claims_v15(gate, observations, taxonomy)
            row = {
                **claim_common, "review_id": rid,
                "input_hash": claim_inputs[rid], "upstream_hash": upstream[rid],
                "parse_success": success,
                "normalization_complete": gate["normalization_complete"],
                "model_normalization_complete": gate["model_normalization_complete"],
                "fallback_unmapped_count": gate["fallback_unmapped_count"],
                "deterministic_recovery_count": gate["deterministic_recovery_count"],
                "v15_recall_recovery_count": gate.get("v15_recall_recovery_count", 0),
                "v15_category_repair_count": gate.get("v15_category_repair_count", 0),
                "accepted_claims": gate["accepted_claims"],
                "rejected_claims": gate["rejected_claims"],
                "missing_observation_ids": gate["missing_observation_ids"],
                "validation_errors": gate["validation_errors"],
                "model_validation_errors": gate["model_validation_errors"],
                "model_missing_observation_ids": gate["model_missing_observation_ids"],
                "model_result_raw": parsed, "selected_raw_response": raw,
                "attempts": attempts,
            }
            row["merged_claims"] = merge_claims_by_tuple(row["accepted_claims"])
            row["duplicate_claims_merged"] = (
                sum(c.get("mapping_status") == "mapped" for c in row["accepted_claims"])
                - len(row["merged_claims"])
            )
            row["record_hash"] = _claim_record_hash(row)
        append_jsonl(paths["claims"], row)
        claim_cache[rid] = row
        print(f"Laptop normalization progress: {index}/{len(pending_claims)}")

    if active_l2 is not None and manage_model_lifecycle:
        _release_model(active_l2)
    observation_rows = [observation_cache[rid] for rid in selected]
    claim_rows = [claim_cache[rid] for rid in selected]
    summary = build_product_summary(
        product_id=product_id, selected_review_ids=selected, reviews=reviews,
        observation_rows=observation_rows, claim_rows=claim_rows,
        run_fingerprint=fingerprint,
    )
    _write_or_validate_json(paths["summary"], summary)
    print("\n=== CONFIGURABLE LAPTOP TRANSFER PILOT ===")
    print("Product:", summary.get("product_name") or product_id)
    print("Status:", TRANSFER_STATUS)
    for rid in selected[:max(show, 0)]:
        observations = observation_cache[rid].get("accepted_observations", [])
        claims = claim_cache[rid].get("accepted_claims", [])
        mapped = sum(c.get("mapping_status") == "mapped" for c in claims)
        print(
            f"  {rid}: observations={len(observations)} mapped={mapped} "
            f"unmapped={len(claims)-mapped} "
            f"findings={len(claim_cache[rid].get('merged_claims', []))}"
        )
    print("Saved only under:", run_dir)
    print("Transfer performance is not claimed without product-specific annotation.")
    return {
        "product_id": product_id, "status": TRANSFER_STATUS,
        "selected_review_ids": selected, "observation_rows": observation_rows,
        "claim_rows": claim_rows, "summary": summary,
        "output_dir": str(output_dir), "run_dir": str(run_dir),
        "run_fingerprint": fingerprint,
        "paths": {key: str(value) for key, value in paths.items()},
    }


def summarize_laptop_run(
    project_root: str | Path,
    product_config: str | Path,
    *,
    limit: int = 10,
    show: int = 5,
) -> dict[str, Any]:
    """Read a completed matching run without loading either model."""
    root = Path(project_root).resolve()
    from . import laptop_products

    prepared = laptop_products.prepare_laptop_product(root, product_config, limit=limit)
    product_id = str(prepared.product_id)
    runs_dir = _prepared_path(prepared, "output_dir") / "evidence_first/runs"
    candidates: list[tuple[float, Path, dict[str, Any]]] = []
    paths = runs_dir.glob("*/product_summary.json") if runs_dir.exists() else []
    for path in paths:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        if (
            isinstance(value, dict) and value.get("product_id") == product_id
            and value.get("selected_review_ids") == list(prepared.selected_review_ids)
        ):
            candidates.append((path.stat().st_mtime, path, value))
    if not candidates:
        raise FileNotFoundError(
            f"No completed laptop pilot for {product_id}; run laptop-pilot first"
        )
    _, path, summary = max(candidates, key=lambda value: value[0])
    print("\n=== CONFIGURABLE LAPTOP TRANSFER SUMMARY ===")
    print("Product:", summary.get("product_name") or product_id)
    print("Status:", summary.get("status"))
    print("Reviews:", summary.get("reviews_analyzed"))
    for issue in summary.get("issues", [])[:max(show, 0)]:
        print(
            f"  {issue['category']}/{issue['issue_group']} {issue['sentiment']}: "
            f"{issue['supporting_review_count']}/{summary['reviews_analyzed']} reviews"
        )
    print("Summary:", path)
    return {
        "product_id": product_id, "status": TRANSFER_STATUS,
        "summary": summary, "summary_path": str(path),
        "run_dir": str(path.parent),
    }


__all__ = [
    "DEFAULT_L1_MODEL", "DEFAULT_L2_MODEL", "PIPELINE_VERSION",
    "TRANSFER_STATUS", "build_product_summary", "run_laptop_pilot",
    "summarize_laptop_run",
]
