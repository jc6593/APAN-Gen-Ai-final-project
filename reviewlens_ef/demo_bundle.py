"""Build immutable, model-free JSON artifacts for the ReviewLens demo.

The bundle is derived from completed Evidence-first runs. It contains short,
verified evidence excerpts, never full reviews or raw model responses.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

MANIFEST_SCHEMA_VERSION = "reviewlens.demo_bundle.v1"
PRODUCT_SCHEMA_VERSION = "reviewlens.demo_product.v1"
FEATURED_PRODUCT_COUNT = 6
REVIEWS_PER_PRODUCT = 30
MAX_QUOTE_CHARS = 240
MAX_EXCERPT_CHARS = 360
MAX_EVIDENCE = 3

_SAFE_ID = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
_FORBIDDEN_KEYS = {
    "raw_text", "clean_text", "raw_response", "selected_raw_response",
    "model_result_raw", "attempts", "full_review_text", "reviewer_id", "user_id",
}


class DemoBundleError(ValueError):
    """A completed run cannot be converted to the public demo contract."""


class DemoBundleImmutableError(DemoBundleError):
    """An existing content-addressed bundle would need to be overwritten."""


@dataclass(frozen=True)
class DemoBundleResult:
    bundle_id: str
    bundle_version: str
    output_directory: Path
    manifest_path: Path
    product_paths: tuple[Path, ...]
    was_created: bool


def _get(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _canonical(value: Any) -> bytes:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return (raw + "\n").encode("utf-8")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_id(prefix: str, *parts: Any) -> str:
    raw = json.dumps(parts, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"{prefix}_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:16]}"


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DemoBundleError(f"{label} must be non-empty text")
    return value.strip()


def _inside(root: Path, value: str | Path, label: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise DemoBundleError(f"{label} must be a filesystem path")
    candidate = Path(value)
    path = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise DemoBundleError(f"{label} must be inside project_root: {value}") from exc
    return path


def _required_file(root: Path, value: str | Path, label: str) -> Path:
    path = _inside(root, value, label)
    if not path.is_file():
        raise DemoBundleError(f"{label} does not exist: {path}")
    return path


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DemoBundleError(f"invalid JSON: {path}") from exc


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise DemoBundleError(f"{path}:{line_no} is not an object")
            rows.append(value)
    except DemoBundleError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DemoBundleError(f"invalid JSONL: {path}") from exc
    return rows


def _rating(value: Any, review_id: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DemoBundleError(f"review {review_id} has no numeric rating")
    rating = float(value)
    if rating not in {1.0, 2.0, 3.0, 4.0, 5.0}:
        raise DemoBundleError(f"review {review_id} rating must be an integer from 1-5")
    return rating


def _index_rows(rows: Sequence[Mapping[str, Any]], expected: Sequence[str],
                label: str) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    order: list[str] = []
    for row in rows:
        review_id = row.get("review_id")
        if not isinstance(review_id, str) or not review_id or review_id in result:
            raise DemoBundleError(f"{label} has a missing or duplicate review_id")
        result[review_id] = row
        order.append(review_id)
    if order != list(expected):
        raise DemoBundleError(f"{label} must use the same ordered 30-review cohort")
    return result


def _excerpt(sentence: str, quote_start: int, quote: str) -> tuple[str, str, int, int, bool]:
    public_quote = quote[:MAX_QUOTE_CHARS]
    quote_end = quote_start + len(public_quote)
    if len(sentence) <= MAX_EXCERPT_CHARS:
        start = 0
    else:
        room = MAX_EXCERPT_CHARS - len(public_quote)
        start = max(0, quote_start - max(0, room // 2))
        start = min(start, max(0, len(sentence) - MAX_EXCERPT_CHARS))
        if quote_end > start + MAX_EXCERPT_CHARS:
            start = quote_end - MAX_EXCERPT_CHARS
    excerpt = sentence[start:start + MAX_EXCERPT_CHARS]
    local_start = quote_start - start
    local_end = local_start + len(public_quote)
    if excerpt[local_start:local_end] != public_quote:
        raise DemoBundleError("could not create an exact evidence excerpt")
    return excerpt, public_quote, local_start, local_end, (
        public_quote != quote or excerpt != sentence
    )


def _public_evidence(product_id: str, review_id: str, review: Mapping[str, Any],
                     evidence: Mapping[str, Any]) -> dict[str, Any]:
    source_field = evidence.get("source_field")
    if source_field not in {"title", "clean_text"}:
        raise DemoBundleError(f"{review_id}: invalid evidence source_field")
    source = review.get(str(source_field))
    quote = evidence.get("quote")
    source_start = evidence.get("quote_start_char_source")
    source_end = evidence.get("quote_end_char_source")
    sentence = evidence.get("sentence_text")
    sentence_start = evidence.get("quote_start_char_sentence")
    sentence_end = evidence.get("quote_end_char_sentence")
    valid = (
        isinstance(source, str) and isinstance(quote, str) and bool(quote)
        and isinstance(source_start, int) and isinstance(source_end, int)
        and isinstance(sentence, str) and isinstance(sentence_start, int)
        and isinstance(sentence_end, int) and source_start >= 0 and sentence_start >= 0
        and source_end == source_start + len(quote)
        and sentence_end == sentence_start + len(quote)
        and source[source_start:source_end] == quote
        and sentence[sentence_start:sentence_end] == quote
    )
    if not valid:
        raise DemoBundleError(f"{review_id}: evidence offsets do not locate exact text")
    excerpt, public_quote, local_start, local_end, truncated = _excerpt(
        sentence, sentence_start, quote
    )
    public_source_end = source_start + len(public_quote)
    sentence_id = _text(evidence.get("sentence_id"), "evidence sentence_id")
    evidence_id = _stable_id(
        "ev", product_id, review_id, sentence_id, source_field,
        source_start, public_source_end, public_quote,
    )
    return {
        "evidence_id": evidence_id,
        "review_id": review_id,
        "rating": _rating(review.get("rating"), review_id),
        "sentence_id": sentence_id,
        "source_field": source_field,
        "quote": public_quote,
        "sentence_excerpt": excerpt,
        "quote_start_char_excerpt": local_start,
        "quote_end_char_excerpt": local_end,
        "quote_start_char_source": source_start,
        "quote_end_char_source": public_source_end,
        "exact_source_verified": True,
        "truncated": truncated,
    }


def _finding(product_id: str, scope: str, key: tuple[str, str, str],
             by_review: Mapping[str, list[dict[str, Any]]]) -> dict[str, Any]:
    category, issue_group, sentiment = key
    review_ids = sorted(by_review)
    evidence: list[dict[str, Any]] = []
    seen: set[str] = set()
    for review_id in review_ids:
        for item in by_review[review_id]:
            if item["evidence_id"] not in seen:
                evidence.append(item)
                seen.add(item["evidence_id"])
            if len(evidence) == MAX_EVIDENCE:
                break
        if len(evidence) == MAX_EVIDENCE:
            break
    if not evidence:
        raise DemoBundleError(f"finding {key} has no verified evidence")
    count = len(review_ids)
    return {
        "finding_id": _stable_id(
            "finding", product_id, scope, category, issue_group, sentiment
        ),
        "scope": scope,
        "category": category,
        "issue_group": issue_group,
        "sentiment": sentiment,
        "label": issue_group.replace("_", " ").title(),
        "supporting_review_count": count,
        "supporting_review_rate": round(count / REVIEWS_PER_PRODUCT, 6),
        "supporting_review_ids": review_ids,
        "evidence": evidence,
    }


def _signal(product_id: str, review_id: str, claim: Mapping[str, Any],
            evidence: list[dict[str, Any]]) -> dict[str, Any]:
    if not evidence:
        raise DemoBundleError(f"{review_id}: unmapped signal has no verified evidence")
    observation_id = _text(claim.get("observation_id"), "claim observation_id")
    claim_text = _text(claim.get("claim"), "unmapped claim")
    first = evidence[0]
    public_claim = claim_text[:MAX_EXCERPT_CHARS]
    return {
        "signal_id": _stable_id(
            "signal", product_id, review_id, observation_id, public_claim
        ),
        "review_id": review_id,
        "observation_id": observation_id,
        "sentiment": _text(claim.get("sentiment"), "claim sentiment"),
        "statement_type": _text(claim.get("statement_type"), "statement_type"),
        "claim": public_claim,
        "quote": first["quote"],
        "rating": first["rating"],
        "exact_source_verified": True,
        "evidence": evidence[:MAX_EVIDENCE],
    }


def _product_role(evaluation_status: Any) -> str:
    value = _text(evaluation_status, "evaluation_status")
    roles = {
        "benchmark_source_product": "benchmark",
        "benchmark": "benchmark",
        "transfer_demo": "transfer_demo",
    }
    if value not in roles:
        raise DemoBundleError(f"unsupported evaluation_status: {value}")
    return roles[value]


def _product_document(root: Path, featured: Any,
                      run: Mapping[str, Any]) -> dict[str, Any]:
    product_id = _text(run.get("product_id"), "run product_id")
    if not _SAFE_ID.fullmatch(product_id):
        raise DemoBundleError(f"unsafe product_id: {product_id}")
    paths = run.get("paths")
    if not isinstance(paths, Mapping):
        raise DemoBundleError(f"{product_id}: run paths are missing")
    output_dir = _inside(root, _text(run.get("output_dir"), "run output_dir"), "output_dir")
    reviews_path = _required_file(
        root, paths.get("prepared_reviews", output_dir / "prepared_reviews.jsonl"),
        f"{product_id} prepared reviews",
    )
    observations_path = _required_file(
        root, paths.get("observations"), f"{product_id} observations"
    )
    claims_path = _required_file(root, paths.get("claims"), f"{product_id} claims")
    config_path = _required_file(
        root, _get(featured, "config_path"), f"{product_id} config"
    )
    product_config = _read_json(config_path)
    if not isinstance(product_config, Mapping) or product_config.get("product_id") != product_id:
        raise DemoBundleError(
            f"{product_id}: featured product order does not match its config"
        )
    selected = run.get("selected_review_ids")
    if not isinstance(selected, list) or len(selected) != REVIEWS_PER_PRODUCT:
        raise DemoBundleError(f"{product_id}: completed run must select exactly 30 reviews")
    if len(set(selected)) != REVIEWS_PER_PRODUCT or not all(
        isinstance(value, str) and value for value in selected
    ):
        raise DemoBundleError(f"{product_id}: invalid selected review IDs")
    reviews_rows = _read_jsonl(reviews_path)
    reviews = _index_rows(reviews_rows, selected, "prepared reviews")
    observations = _index_rows(_read_jsonl(observations_path), selected, "observations")
    claims = _index_rows(_read_jsonl(claims_path), selected, "claims")
    fingerprint = _text(run.get("run_fingerprint"), "run_fingerprint")
    for label, indexed in (("observations", observations), ("claims", claims)):
        for review_id, row in indexed.items():
            if row.get("run_fingerprint") != fingerprint:
                raise DemoBundleError(f"{label} row {review_id} has stale fingerprint")
            if row.get("product_id") not in (None, product_id):
                raise DemoBundleError(f"{label} row {review_id} has wrong product_id")

    product_groups: dict[tuple[str, str, str], dict[str, list[dict[str, Any]]]] = {}
    service_groups: dict[tuple[str, str, str], dict[str, list[dict[str, Any]]]] = {}
    signals: list[dict[str, Any]] = []
    atomic_mapped = atomic_unmapped = excluded = 0
    product_atomic = service_atomic = 0
    mapped_reviews: set[str] = set()
    unmapped_reviews: set[str] = set()
    for review_id in selected:
        review = reviews[review_id]
        if review.get("product_id") not in (None, product_id):
            raise DemoBundleError(f"review {review_id} has wrong product_id")
        _rating(review.get("rating"), review_id)
        accepted = claims[review_id].get("accepted_claims", [])
        if not isinstance(accepted, list):
            raise DemoBundleError(f"{review_id}: accepted_claims must be a list")
        for claim in accepted:
            if not isinstance(claim, Mapping):
                raise DemoBundleError(f"{review_id}: accepted claim is not an object")
            raw_evidence = claim.get("evidence", [])
            if not isinstance(raw_evidence, list) or not all(
                isinstance(item, Mapping) for item in raw_evidence
            ):
                raise DemoBundleError(f"{review_id}: claim evidence must be objects")
            evidence = [
                _public_evidence(product_id, review_id, review, item)
                for item in raw_evidence
            ]
            mapping_status = claim.get("mapping_status")
            attribution = claim.get("attribution")
            category = claim.get("category")
            if mapping_status == "mapped":
                atomic_mapped += 1
                issue_group = claim.get("issue_group")
                sentiment = claim.get("sentiment")
                if not all(isinstance(value, str) and value for value in (
                    category, issue_group, sentiment
                )):
                    raise DemoBundleError(f"{review_id}: mapped claim has no tuple")
                key = (str(category), str(issue_group), str(sentiment))
                # Purchase/service is deliberately separate from product comparison.
                if category == "purchase_service" and attribution in {
                    "current_product", "seller_platform"
                }:
                    target = service_groups
                    service_atomic += 1
                elif attribution == "current_product" and category != "purchase_service":
                    target = product_groups
                    product_atomic += 1
                    mapped_reviews.add(review_id)
                else:
                    excluded += 1
                    continue
                if not evidence:
                    raise DemoBundleError(f"{review_id}: included claim has no evidence")
                target.setdefault(key, {}).setdefault(review_id, []).extend(evidence)
            elif mapping_status == "unmapped":
                atomic_unmapped += 1
                if attribution == "current_product":
                    signals.append(_signal(product_id, review_id, claim, evidence))
                    unmapped_reviews.add(review_id)
                else:
                    excluded += 1
            else:
                raise DemoBundleError(f"{review_id}: invalid mapping_status")

    findings = [
        _finding(product_id, "current_product", key, product_groups[key])
        for key in sorted(product_groups)
    ]
    service_context = [
        _finding(product_id, "purchase_service", key, service_groups[key])
        for key in sorted(service_groups)
    ]
    findings.sort(key=lambda item: (-item["supporting_review_count"], item["finding_id"]))
    service_context.sort(
        key=lambda item: (-item["supporting_review_count"], item["finding_id"])
    )
    signals.sort(key=lambda item: item["signal_id"])

    ratings = [_rating(reviews[rid].get("rating"), rid) for rid in selected]
    distribution = {
        str(star): sum(round(value) == star for value in ratings)
        for star in range(1, 6)
    }
    timestamps = sorted(
        str(reviews[rid].get("timestamp")) for rid in selected
        if reviews[rid].get("timestamp") not in (None, "")
    )
    accepted_observations = 0
    for review_id in selected:
        rows = observations[review_id].get("accepted_observations", [])
        if not isinstance(rows, list):
            raise DemoBundleError(f"{review_id}: accepted_observations must be a list")
        accepted_observations += len(rows)

    summary = run.get("summary") if isinstance(run.get("summary"), Mapping) else {}
    family = (
        product_config.get("product_family")
        if isinstance(product_config.get("product_family"), Mapping) else {}
    )
    display_name = (
        family.get("display_name") or product_config.get("display_name")
        or summary.get("product_name") or summary.get("product_model") or product_id
    )
    role = _product_role(_get(featured, "evaluation_status"))
    # Product-specific cohort status is part of the public interpretation;
    # the run's generic transfer status must not erase it.
    status = _get(featured, "cohort_status")
    document = {
        "schema_version": PRODUCT_SCHEMA_VERSION,
        "product": {
            "product_id": product_id,
            "display_name": _text(display_name, "display_name"),
            "brand": _text(_get(featured, "brand"), "brand"),
            "operating_system": _text(
                _get(featured, "operating_system"), "operating_system"
            ),
            "role": role,
            "status": _text(status, "status"),
        },
        "sample": {
            "reviews_analyzed": REVIEWS_PER_PRODUCT,
            "review_ids": list(selected),
            "average_rating": round(sum(ratings) / len(ratings), 3),
            "rating_distribution": distribution,
            "date_range": {
                "start": timestamps[0] if timestamps else None,
                "end": timestamps[-1] if timestamps else None,
            },
        },
        "coverage": {
            "accepted_observation_count": accepted_observations,
            "atomic_mapped_claim_count": atomic_mapped,
            "atomic_unmapped_claim_count": atomic_unmapped,
            "current_product_atomic_claim_count": product_atomic,
            "purchase_service_atomic_claim_count": service_atomic,
            "excluded_attribution_claim_count": excluded,
            "reviews_with_product_findings": len(mapped_reviews),
            "reviews_with_unmapped_signals": len(unmapped_reviews),
            "reviews_with_findings": len(mapped_reviews),
            "reviews_with_other_verified_signals": len(unmapped_reviews),
            "mapped_review_coverage": round(len(mapped_reviews) / REVIEWS_PER_PRODUCT, 6),
            "run_fingerprint": fingerprint,
            "source_sha256": {
                "config": _file_sha(config_path),
                "prepared_reviews": _file_sha(reviews_path),
                "observations": _file_sha(observations_path),
                "claims": _file_sha(claims_path),
            },
        },
        "findings": findings,
        "signals": signals,
        "report": {
            "title": f"ReviewLens evidence report: {display_name}",
            "executive_summary": (
                f"{len(findings)} product finding tuple(s) and {len(signals)} "
                "additional verified product signal(s) across 30 reviews."
            ),
            "top_finding_ids": [item["finding_id"] for item in findings[:3]],
            "signal_ids": [item["signal_id"] for item in signals],
            "service_context": service_context,
            "aggregation_note": (
                "Each review counts at most once per category/issue/sentiment tuple."
            ),
        },
        "limitations": [
            "This fixed 30-review sample is descriptive, not a population estimate.",
            "Transfer quality is not claimed without product-specific blind annotation.",
            "Purchase/service context is excluded from product findings and comparison.",
            "Verified unmapped signals are not forced into the product taxonomy.",
        ],
    }
    return document


def _manifest(demo_config: Any, documents: Sequence[Mapping[str, Any]],
              payloads: Sequence[bytes]) -> dict[str, Any]:
    product_ids = [str(item["product"]["product_id"]) for item in documents]
    version = _text(_get(demo_config, "demo_id"), "demo_id")
    core = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "bundle_version": version,
        "featured_product_ids": product_ids,
        "products": [
            {
                "product_id": product_id,
                "file": f"products/{product_id}.json",
                "sha256": _sha(payload),
            }
            for product_id, payload in zip(product_ids, payloads)
        ],
        "methodology": {
            "featured_product_count": FEATURED_PRODUCT_COUNT,
            "reviews_per_product": REVIEWS_PER_PRODUCT,
            "aggregation_unit": "unique review per category/issue_group/sentiment tuple",
            "scope_policy": {
                "current_product": "included in product findings",
                "purchase_service": "separate report context; excluded from comparison",
                "unmapped_current_product": "separate verified signals",
                "other_attribution": "excluded from product findings and signals",
            },
            "evidence_policy": "short verified excerpts with exact source offsets",
            "runtime_policy": "precomputed JSON; no model, MLX, or network required",
            "models": {
                "layer1": {
                    "model_id": _get(demo_config, "l1_model_id"),
                    "revision": _get(demo_config, "l1_revision"),
                },
                "layer2": {
                    "model_id": _get(demo_config, "l2_model_id"),
                    "revision": _get(demo_config, "l2_revision"),
                },
            },
            "demo_config_sha256": _get(demo_config, "config_sha256"),
        },
        "limitations": [
            "The bundle is a presentation artifact derived from completed runs.",
            "It contains no full reviews, user identifiers, or raw model responses.",
            "Cross-product comparisons are descriptive because samples are independent.",
        ],
    }
    return {**core, "bundle_id": _sha(_canonical(core))}


def _result(output_dir: Path, manifest: Mapping[str, Any],
            was_created: bool) -> DemoBundleResult:
    return DemoBundleResult(
        bundle_id=str(manifest["bundle_id"]),
        bundle_version=str(manifest["bundle_version"]),
        output_directory=output_dir,
        manifest_path=output_dir / "manifest.json",
        product_paths=tuple(
            output_dir / str(item["file"]) for item in manifest["products"]
        ),
        was_created=was_created,
    )


def build_demo_bundle(
    project_root: str | Path,
    demo_config: Any,
    product_results: Sequence[Mapping[str, Any]],
) -> DemoBundleResult:
    """Create or idempotently reuse the configured six-product demo bundle.

    This function only reads completed run artifacts. It never imports a model
    runtime and never accesses the network.
    """
    root = Path(project_root).resolve()
    if not root.is_dir():
        raise DemoBundleError(f"project_root does not exist: {root}")
    if _get(demo_config, "reviews_per_product") != REVIEWS_PER_PRODUCT:
        raise DemoBundleError("demo_config must freeze reviews_per_product at 30")
    featured = _get(demo_config, "products")
    if not isinstance(featured, Sequence) or isinstance(featured, (str, bytes)):
        raise DemoBundleError("demo_config.products must be a sequence")
    if len(featured) != FEATURED_PRODUCT_COUNT or len(product_results) != FEATURED_PRODUCT_COUNT:
        raise DemoBundleError("exactly six featured products and results are required")
    config_path = _required_file(
        root, _get(demo_config, "config_path"), "featured demo config"
    )
    configured_hash = _text(_get(demo_config, "config_sha256"), "config_sha256")
    if _file_sha(config_path) != configured_hash:
        raise DemoBundleError("featured demo config hash does not match its file")
    documents = [
        _product_document(root, product, result)
        for product, result in zip(featured, product_results)
    ]
    ids = [str(item["product"]["product_id"]) for item in documents]
    if len(set(ids)) != FEATURED_PRODUCT_COUNT:
        raise DemoBundleError("featured product IDs must be unique")
    payloads = [_canonical(document) for document in documents]
    manifest = _manifest(demo_config, documents, payloads)
    manifest_payload = _canonical(manifest)
    output_value = _get(demo_config, "bundle_output_directory")
    output_dir = _inside(root, output_value, "bundle_output_directory")
    expected = {
        "manifest.json": manifest_payload,
        **{
            f"products/{product_id}.json": payload
            for product_id, payload in zip(ids, payloads)
        },
    }
    if output_dir.exists():
        try:
            verified = verify_demo_bundle(root, output_directory=output_value)
        except DemoBundleError as exc:
            raise DemoBundleImmutableError(
                "existing demo bundle failed integrity verification and was not overwritten"
            ) from exc
        actual = {
            path.relative_to(output_dir).as_posix(): path.read_bytes()
            for path in output_dir.rglob("*") if path.is_file()
        }
        if actual != expected:
            raise DemoBundleImmutableError(
                "a different valid bundle already exists; choose a new output directory"
            )
        return DemoBundleResult(
            verified.bundle_id, verified.bundle_version, verified.output_directory,
            verified.manifest_path, verified.product_paths, False,
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".reviewlens-demo-",
                                     dir=output_dir.parent) as temporary:
        staging = Path(temporary) / output_dir.name
        (staging / "products").mkdir(parents=True)
        for product_id, payload in zip(ids, payloads):
            (staging / "products" / f"{product_id}.json").write_bytes(payload)
        (staging / "manifest.json").write_bytes(manifest_payload)
        os.replace(staging, output_dir)
    return verify_demo_bundle(
        root, output_directory=output_value, was_created=True
    )


def _walk_forbidden(value: Any, path: str = "$") -> None:
    if isinstance(value, Mapping):
        forbidden = set(value).intersection(_FORBIDDEN_KEYS)
        if forbidden:
            raise DemoBundleError(f"forbidden keys at {path}: {sorted(forbidden)}")
        for key, child in value.items():
            _walk_forbidden(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _walk_forbidden(child, f"{path}[{index}]")


def _verify_evidence(item: Any, product_id: str) -> None:
    if not isinstance(item, Mapping) or item.get("exact_source_verified") is not True:
        raise DemoBundleError(f"{product_id}: evidence is not source-verified")
    quote = item.get("quote")
    excerpt = item.get("sentence_excerpt")
    start = item.get("quote_start_char_excerpt")
    end = item.get("quote_end_char_excerpt")
    source_start = item.get("quote_start_char_source")
    source_end = item.get("quote_end_char_source")
    valid = (
        isinstance(quote, str) and bool(quote) and len(quote) <= MAX_QUOTE_CHARS
        and isinstance(excerpt, str) and len(excerpt) <= MAX_EXCERPT_CHARS
        and isinstance(start, int) and isinstance(end, int)
        and isinstance(source_start, int) and isinstance(source_end, int)
        and start >= 0 and source_start >= 0 and end == start + len(quote)
        and source_end == source_start + len(quote)
        and excerpt[start:end] == quote
    )
    if not valid:
        raise DemoBundleError(f"{product_id}: invalid evidence excerpt")
    expected = _stable_id(
        "ev", product_id, item.get("review_id"), item.get("sentence_id"),
        item.get("source_field"), source_start, source_end, quote,
    )
    if item.get("evidence_id") != expected:
        raise DemoBundleError(f"{product_id}: unstable evidence_id")


def _verify_finding(item: Any, product_id: str, review_ids: set[str],
                    expected_scope: str) -> None:
    if not isinstance(item, Mapping) or item.get("scope") != expected_scope:
        raise DemoBundleError(f"{product_id}: invalid finding scope")
    expected_id = _stable_id(
        "finding", product_id, expected_scope, item.get("category"),
        item.get("issue_group"), item.get("sentiment"),
    )
    if item.get("finding_id") != expected_id:
        raise DemoBundleError(f"{product_id}: unstable finding_id")
    support = item.get("supporting_review_ids")
    count = item.get("supporting_review_count")
    if (
        not isinstance(support, list) or len(support) != len(set(support))
        or not set(support).issubset(review_ids) or count != len(support)
        or item.get("supporting_review_rate") != round(len(support) / 30, 6)
    ):
        raise DemoBundleError(f"{product_id}: invalid finding support")
    evidence = item.get("evidence")
    if not isinstance(evidence, list) or not evidence or len(evidence) > MAX_EVIDENCE:
        raise DemoBundleError(f"{product_id}: invalid finding evidence")
    for row in evidence:
        if not isinstance(row, Mapping) or row.get("review_id") not in support:
            raise DemoBundleError(f"{product_id}: finding evidence is out of scope")
        _verify_evidence(row, product_id)


def _verify_product(document: Any, product_id: str) -> None:
    required = {
        "schema_version", "product", "sample", "coverage", "findings",
        "signals", "report", "limitations",
    }
    if (
        not isinstance(document, Mapping)
        or document.get("schema_version") != PRODUCT_SCHEMA_VERSION
        or set(document) != required
    ):
        raise DemoBundleError(f"{product_id}: invalid product API contract")
    product = document.get("product")
    if (
        not isinstance(product, Mapping) or product.get("product_id") != product_id
        or set(product) != {
            "product_id", "display_name", "brand", "operating_system", "role", "status"
        }
        or product.get("role") not in {"benchmark", "transfer_demo"}
    ):
        raise DemoBundleError(f"{product_id}: invalid product identity")
    sample = document.get("sample")
    review_list = sample.get("review_ids") if isinstance(sample, Mapping) else None
    distribution = sample.get("rating_distribution") if isinstance(sample, Mapping) else None
    if (
        not isinstance(sample, Mapping) or sample.get("reviews_analyzed") != 30
        or not isinstance(review_list, list) or len(review_list) != 30
        or len(set(review_list)) != 30 or not isinstance(distribution, Mapping)
        or set(distribution) != {"1", "2", "3", "4", "5"}
        or any(not isinstance(value, int) or value < 0 for value in distribution.values())
        or sum(distribution.values()) != 30
    ):
        raise DemoBundleError(f"{product_id}: invalid N=30 sample")
    review_ids = set(review_list)
    findings = document.get("findings")
    signals = document.get("signals")
    report = document.get("report")
    if not isinstance(findings, list) or not isinstance(signals, list):
        raise DemoBundleError(f"{product_id}: findings and signals must be lists")
    for finding in findings:
        _verify_finding(finding, product_id, review_ids, "current_product")
    if not isinstance(report, Mapping) or not isinstance(report.get("service_context"), list):
        raise DemoBundleError(f"{product_id}: invalid report")
    for finding in report["service_context"]:
        _verify_finding(finding, product_id, review_ids, "purchase_service")
    for signal in signals:
        if (
            not isinstance(signal, Mapping) or signal.get("review_id") not in review_ids
            or signal.get("exact_source_verified") is not True
            or not isinstance(signal.get("quote"), str) or not signal.get("quote")
        ):
            raise DemoBundleError(f"{product_id}: invalid signal")
        expected = _stable_id(
            "signal", product_id, signal.get("review_id"),
            signal.get("observation_id"), signal.get("claim"),
        )
        if signal.get("signal_id") != expected:
            raise DemoBundleError(f"{product_id}: unstable signal_id")
        evidence = signal.get("evidence")
        if not isinstance(evidence, list) or not evidence or len(evidence) > MAX_EVIDENCE:
            raise DemoBundleError(f"{product_id}: invalid signal evidence")
        for row in evidence:
            if not isinstance(row, Mapping) or row.get("review_id") != signal.get("review_id"):
                raise DemoBundleError(f"{product_id}: signal evidence is out of scope")
            _verify_evidence(row, product_id)
        if signal.get("quote") != evidence[0].get("quote"):
            raise DemoBundleError(f"{product_id}: signal quote differs from evidence")
    if not isinstance(document.get("limitations"), list):
        raise DemoBundleError(f"{product_id}: limitations must be a list")
    _walk_forbidden(document)


def verify_demo_bundle(
    project_root: str | Path,
    *,
    output_directory: str | Path,
    was_created: bool = False,
) -> DemoBundleResult:
    """Verify canonical JSON, hashes, strict schemas, cohort, and safe evidence."""
    root = Path(project_root).resolve()
    output_dir = _inside(root, output_directory, "output_directory")
    manifest_path = output_dir / "manifest.json"
    if not output_dir.is_dir() or not manifest_path.is_file():
        raise DemoBundleError(f"demo bundle does not exist: {output_dir}")
    manifest = _read_json(manifest_path)
    required = {
        "schema_version", "bundle_version", "featured_product_ids", "products",
        "methodology", "limitations", "bundle_id",
    }
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION
        or set(manifest) != required or manifest_path.read_bytes() != _canonical(manifest)
    ):
        raise DemoBundleError("invalid or non-canonical demo manifest")
    core = {key: value for key, value in manifest.items() if key != "bundle_id"}
    if manifest.get("bundle_id") != _sha(_canonical(core)):
        raise DemoBundleError("manifest bundle_id hash mismatch")
    ids = manifest.get("featured_product_ids")
    entries = manifest.get("products")
    methodology = manifest.get("methodology")
    if (
        not isinstance(ids, list) or len(ids) != 6 or len(set(ids)) != 6
        or not isinstance(entries, list) or len(entries) != 6
        or not isinstance(methodology, Mapping) or not methodology
        or methodology.get("reviews_per_product") != 30
        or not isinstance(manifest.get("limitations"), list)
    ):
        raise DemoBundleError("manifest does not freeze six products at N=30")
    entry_ids = [item.get("product_id") if isinstance(item, Mapping) else None for item in entries]
    if entry_ids != ids:
        raise DemoBundleError("featured_product_ids and products differ")
    expected_files = {"manifest.json"}
    for product_id, entry in zip(ids, entries):
        if not isinstance(product_id, str) or not _SAFE_ID.fullmatch(product_id):
            raise DemoBundleError("manifest contains an unsafe product_id")
        if not isinstance(entry, Mapping) or set(entry) != {"product_id", "file", "sha256"}:
            raise DemoBundleError(f"{product_id}: invalid manifest product entry")
        relative = f"products/{product_id}.json"
        if entry.get("file") != relative:
            raise DemoBundleError(f"{product_id}: unexpected product path")
        expected_files.add(relative)
        path = output_dir / relative
        if not path.is_file() or _file_sha(path) != entry.get("sha256"):
            raise DemoBundleError(f"{product_id}: product hash mismatch")
        document = _read_json(path)
        if path.read_bytes() != _canonical(document):
            raise DemoBundleError(f"{product_id}: product JSON is not canonical")
        _verify_product(document, product_id)
    actual_files = {
        path.relative_to(output_dir).as_posix()
        for path in output_dir.rglob("*") if path.is_file()
    }
    if actual_files != expected_files:
        raise DemoBundleError("bundle contains files not listed in the manifest")
    _walk_forbidden(manifest)
    return _result(output_dir, manifest, was_created)


__all__ = [
    "DemoBundleError", "DemoBundleImmutableError", "DemoBundleResult",
    "FEATURED_PRODUCT_COUNT", "MANIFEST_SCHEMA_VERSION", "PRODUCT_SCHEMA_VERSION",
    "REVIEWS_PER_PRODUCT", "build_demo_bundle", "verify_demo_bundle",
]
