"""Strict, reproducible preparation for the ReviewLens laptop catalog.

The catalog pipeline is independent of the frozen Acer evaluation pipeline. It
reads versioned configs, selects a deterministic cohort, and writes only to
``outputs/laptops/<product_id>/``.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
from collections import Counter, deque
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PRODUCT_CONFIG_SCHEMA_VERSION = "reviewlens.product_config.v1"
TAXONOMY_SCHEMA_VERSION = "reviewlens.taxonomy.laptop.v1"
PREPARED_REVIEW_SCHEMA_VERSION = "reviewlens.prepared_review.v1"
SOURCE_RECORD_SCHEMA_VERSION = "reviewlens.source_record.v1"
PREPARE_MANIFEST_SCHEMA_VERSION = "reviewlens.prepare_manifest.v1"
SELECTION_STRATEGY = "rating_round_robin_v1"


class LaptopConfigError(ValueError):
    """A product config, taxonomy, or configured source violates its contract."""


class ImmutablePreparedOutputError(RuntimeError):
    """An existing prepared artifact differs from deterministic output."""


@dataclass(frozen=True)
class LaptopProductConfig:
    schema_version: str
    product_id: str
    category: str
    catalog_scope: str
    display_name: str
    model_label: str
    parent_asin: str
    allowed_child_asins: tuple[str, ...]
    source_reviews_jsonl: str
    taxonomy_config_path: str
    taxonomy_schema_version: str
    selection_strategy: str
    selection_seed: int
    default_limit: int
    pilot_review_ids: tuple[str, ...]
    output_directory: str
    config_path: str
    config_sha256: str
    taxonomy_sha256: str

    @property
    def config_hash(self) -> str:
        return self.config_sha256

    @property
    def taxonomy_hash(self) -> str:
        return self.taxonomy_sha256

    @property
    def scope(self) -> str:
        return self.catalog_scope

    @property
    def product_model(self) -> str:
        return self.model_label


@dataclass(frozen=True)
class PrepareResult:
    product_id: str
    output_dir: Path
    reviews_path: Path
    sentences_path: Path
    titles_path: Path
    manifest_path: Path
    selected_review_ids: tuple[str, ...]
    config_sha256: str
    taxonomy_sha256: str
    source_sha256: str
    was_created: bool

    @property
    def config_hash(self) -> str:
        return self.config_sha256

    @property
    def taxonomy_hash(self) -> str:
        return self.taxonomy_sha256


class _ReviewHTMLCleaner(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in {"br", "p", "li", "div"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"p", "li", "div"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def clean_review_html(text: Any) -> str:
    """Normalize review HTML while retaining paragraph boundaries."""

    parser = _ReviewHTMLCleaner()
    parser.feed(str(text))
    parser.close()
    cleaned = html.unescape("".join(parser.parts))
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n\s*\n+", "\n", cleaned)
    return cleaned.strip()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any, *, indent: int | None = None) -> bytes:
    if indent is None:
        text = json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
    else:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=indent)
    return (text + "\n").encode("utf-8")


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical_json_bytes(dict(row)) for row in rows)


def _require_object(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise LaptopConfigError(f"{context} must be a JSON object")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], context: str
) -> None:
    actual = set(value)
    if actual != expected:
        raise LaptopConfigError(
            f"{context} keys do not match schema; "
            f"missing={sorted(expected - actual)}, unknown={sorted(actual - expected)}"
        )


def _require_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LaptopConfigError(f"{context} must be a non-empty string")
    return value


def _require_int(value: Any, context: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise LaptopConfigError(f"{context} must be an integer >= {minimum}")
    return value


def _project_root(project_root: str | Path) -> Path:
    root = Path(project_root).resolve()
    if not root.is_dir():
        raise LaptopConfigError(f"Project root does not exist: {root}")
    return root


def _resolve_project_relative(
    project_root: Path, raw_path: str | Path, context: str
) -> Path:
    raw_text = str(raw_path)
    relative = Path(raw_text)
    if not raw_text or "\x00" in raw_text:
        raise LaptopConfigError(f"{context} must be a project-relative path")
    if relative.is_absolute():
        raise LaptopConfigError(f"{context} must not be absolute: {raw_text}")
    if ".." in relative.parts:
        raise LaptopConfigError(f"{context} must not contain traversal: {raw_text}")
    resolved = (project_root / relative).resolve()
    try:
        resolved.relative_to(project_root)
    except ValueError as exc:
        raise LaptopConfigError(
            f"{context} resolves outside the project: {raw_text}"
        ) from exc
    return resolved


def _load_json(path: Path, context: str) -> tuple[dict[str, Any], bytes]:
    try:
        payload = path.read_bytes()
        document = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LaptopConfigError(f"Cannot read valid JSON from {context}: {path}") from exc
    return _require_object(document, context), payload


def _validate_taxonomy(document: dict[str, Any]) -> dict[str, dict[str, str]]:
    _require_exact_keys(
        document,
        {"schema_version", "taxonomy_id", "category", "issue_groups"},
        "taxonomy",
    )
    if document["schema_version"] != TAXONOMY_SCHEMA_VERSION:
        raise LaptopConfigError("Unsupported taxonomy schema_version")
    if document["taxonomy_id"] != "laptop_v1" or document["category"] != "laptop":
        raise LaptopConfigError("Taxonomy must be laptop_v1 for category laptop")
    groups = _require_object(document["issue_groups"], "taxonomy.issue_groups")
    if not groups:
        raise LaptopConfigError("taxonomy.issue_groups must not be empty")
    name_pattern = re.compile(r"^[a-z][a-z0-9_]*$")
    validated: dict[str, dict[str, str]] = {}
    for category, raw_issues in groups.items():
        if not isinstance(category, str) or not name_pattern.fullmatch(category):
            raise LaptopConfigError(f"Invalid taxonomy category: {category!r}")
        issues = _require_object(raw_issues, f"taxonomy.issue_groups.{category}")
        if not issues:
            raise LaptopConfigError(f"Taxonomy category {category!r} is empty")
        validated[category] = {}
        for issue, description in issues.items():
            if not isinstance(issue, str) or not name_pattern.fullmatch(issue):
                raise LaptopConfigError(f"Invalid taxonomy issue: {issue!r}")
            validated[category][issue] = _require_string(
                description, f"taxonomy description {category}.{issue}"
            )
    return validated


def load_laptop_product_config(
    project_root: str | Path, config_path: str | Path
) -> LaptopProductConfig:
    """Load and fully validate one project-relative product config."""

    root = _project_root(project_root)
    resolved = _resolve_project_relative(root, config_path, "config_path")
    document, config_bytes = _load_json(resolved, "product config")
    _require_exact_keys(
        document,
        {
            "schema_version", "product_id", "category", "catalog_scope",
            "product_family", "source", "taxonomy", "selection", "output",
        },
        "product config",
    )
    if document["schema_version"] != PRODUCT_CONFIG_SCHEMA_VERSION:
        raise LaptopConfigError("Unsupported product config schema_version")
    product_id = _require_string(document["product_id"], "product_id")
    if not re.fullmatch(r"[a-z0-9]+(?:_[a-z0-9]+)*", product_id):
        raise LaptopConfigError("product_id must use lower_snake_case")
    if document["category"] != "laptop":
        raise LaptopConfigError("category must be 'laptop'")
    catalog_scope = document["catalog_scope"]
    if catalog_scope not in {"canonical_product", "product_family"}:
        raise LaptopConfigError("Invalid catalog_scope")

    family = _require_object(document["product_family"], "product_family")
    _require_exact_keys(
        family,
        {"display_name", "model_label", "parent_asin", "allowed_child_asins"},
        "product_family",
    )
    display_name = _require_string(family["display_name"], "display_name")
    model_label = _require_string(family["model_label"], "model_label")
    parent_asin = _require_string(family["parent_asin"], "parent_asin")
    if not re.fullmatch(r"[A-Z0-9]{10}", parent_asin):
        raise LaptopConfigError("parent_asin must be a 10-character ASIN")
    children = family["allowed_child_asins"]
    if (
        not isinstance(children, list)
        or not children
        or any(
            not isinstance(child, str) or not re.fullmatch(r"[A-Z0-9]{10}", child)
            for child in children
        )
        or len(set(children)) != len(children)
    ):
        raise LaptopConfigError("allowed_child_asins must be unique valid ASINs")

    source = _require_object(document["source"], "source")
    _require_exact_keys(source, {"reviews_jsonl"}, "source")
    source_jsonl = _require_string(source["reviews_jsonl"], "source.reviews_jsonl")
    if not _resolve_project_relative(root, source_jsonl, "source.reviews_jsonl").is_file():
        raise LaptopConfigError(f"Review source does not exist: {source_jsonl}")

    taxonomy = _require_object(document["taxonomy"], "taxonomy")
    _require_exact_keys(taxonomy, {"config_path", "schema_version"}, "taxonomy")
    taxonomy_path_text = _require_string(
        taxonomy["config_path"], "taxonomy.config_path"
    )
    taxonomy_schema = _require_string(
        taxonomy["schema_version"], "taxonomy.schema_version"
    )
    if taxonomy_schema != TAXONOMY_SCHEMA_VERSION:
        raise LaptopConfigError("taxonomy.schema_version is not supported")
    taxonomy_document, taxonomy_bytes = _load_json(
        _resolve_project_relative(root, taxonomy_path_text, "taxonomy.config_path"),
        "taxonomy config",
    )
    _validate_taxonomy(taxonomy_document)

    selection = _require_object(document["selection"], "selection")
    _require_exact_keys(
        selection,
        {"strategy", "seed", "default_limit", "pilot_review_ids"},
        "selection",
    )
    strategy = _require_string(selection["strategy"], "selection.strategy")
    if strategy != SELECTION_STRATEGY:
        raise LaptopConfigError(f"selection.strategy must be {SELECTION_STRATEGY!r}")
    seed = _require_int(selection["seed"], "selection.seed")
    default_limit = _require_int(selection["default_limit"], "selection.default_limit", 1)
    pilot_ids = selection["pilot_review_ids"]
    if (
        not isinstance(pilot_ids, list)
        or any(not isinstance(item, str) or not item for item in pilot_ids)
        or len(set(pilot_ids)) != len(pilot_ids)
    ):
        raise LaptopConfigError("selection.pilot_review_ids must contain unique IDs")

    output = _require_object(document["output"], "output")
    _require_exact_keys(output, {"directory"}, "output")
    output_directory = _require_string(output["directory"], "output.directory")
    expected_output = f"outputs/laptops/{product_id}"
    if Path(output_directory).as_posix() != expected_output:
        raise LaptopConfigError(f"output.directory must be exactly {expected_output!r}")
    _resolve_project_relative(root, output_directory, "output.directory")

    return LaptopProductConfig(
        schema_version=PRODUCT_CONFIG_SCHEMA_VERSION,
        product_id=product_id,
        category="laptop",
        catalog_scope=catalog_scope,
        display_name=display_name,
        model_label=model_label,
        parent_asin=parent_asin,
        allowed_child_asins=tuple(children),
        source_reviews_jsonl=source_jsonl,
        taxonomy_config_path=taxonomy_path_text,
        taxonomy_schema_version=taxonomy_schema,
        selection_strategy=strategy,
        selection_seed=seed,
        default_limit=default_limit,
        pilot_review_ids=tuple(pilot_ids),
        output_directory=output_directory,
        config_path=Path(config_path).as_posix(),
        config_sha256=_sha256_bytes(config_bytes),
        taxonomy_sha256=_sha256_bytes(taxonomy_bytes),
    )


def load_laptop_taxonomy(
    project_root: str | Path,
    config_or_path: LaptopProductConfig | str | Path,
) -> dict[str, dict[str, str]]:
    """Return the raw category-to-issue mapping used by model prompts."""

    root = _project_root(project_root)
    path_text = (
        config_or_path.taxonomy_config_path
        if isinstance(config_or_path, LaptopProductConfig)
        else str(config_or_path)
    )
    document, _ = _load_json(
        _resolve_project_relative(root, path_text, "taxonomy config path"),
        "taxonomy config",
    )
    return _validate_taxonomy(document)


def _source_text(row: Mapping[str, Any], review_id: str) -> tuple[str, str]:
    if "text" in row:
        raw_text = str(row["text"])
    elif "raw_text" in row:
        raw_text = str(row["raw_text"])
    else:
        raise LaptopConfigError(f"Review {review_id!r} has no text/raw_text field")
    normalized = clean_review_html(raw_text)
    if not normalized:
        raise LaptopConfigError(f"Review {review_id!r} has empty clean text")
    if row.get("clean_text") is not None and str(row["clean_text"]) != normalized:
        raise LaptopConfigError(
            f"Review {review_id!r} clean_text does not match deterministic cleaning"
        )
    return raw_text, normalized


def _rating(row: Mapping[str, Any], review_id: str) -> int:
    value = row.get("rating")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LaptopConfigError(f"Review {review_id!r} has invalid rating")
    number = float(value)
    if not number.is_integer() or not 1 <= number <= 5:
        raise LaptopConfigError(f"Review {review_id!r} rating must be 1..5")
    return int(number)


def _load_family(
    root: Path, config: LaptopProductConfig
) -> tuple[list[dict[str, Any]], str]:
    source_path = _resolve_project_relative(
        root, config.source_reviews_jsonl, "source.reviews_jsonl"
    )
    source_hash = _sha256_file(source_path)
    matched: list[dict[str, Any]] = []
    seen: set[str] = set()
    with source_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = _require_object(json.loads(line), f"source row {line_number}")
            except json.JSONDecodeError as exc:
                raise LaptopConfigError(
                    f"Invalid JSONL at {config.source_reviews_jsonl}:{line_number}"
                ) from exc
            if row.get("parent_asin") != config.parent_asin:
                continue
            review_id = _require_string(row.get("review_id"), "source review_id")
            if review_id in seen:
                raise LaptopConfigError(f"Duplicate review_id {review_id!r}")
            seen.add(review_id)
            asin = _require_string(row.get("asin"), f"review {review_id}.asin")
            if asin not in config.allowed_child_asins:
                raise LaptopConfigError(
                    f"Review {review_id!r} child ASIN {asin!r} is not allowed by config"
                )
            raw_text, clean_text = _source_text(row, review_id)
            rating = _rating(row, review_id)
            title = row.get("title")
            if title is not None and not isinstance(title, str):
                raise LaptopConfigError(f"Review {review_id!r} title must be string/null")
            clean_title = clean_review_html(title) if title else ""
            matched.append(
                {
                    "schema_version": PREPARED_REVIEW_SCHEMA_VERSION,
                    "review_id": review_id,
                    "product_id": config.product_id,
                    "product_name": config.display_name,
                    "product_model": config.model_label,
                    "asin": asin,
                    "parent_asin": config.parent_asin,
                    "rating": float(rating),
                    "title": clean_title,
                    "raw_title": title,
                    "raw_text": raw_text,
                    "clean_text": clean_text,
                    "timestamp": row.get("timestamp"),
                    "verified_purchase": row.get("verified_purchase"),
                    "helpful_vote": row.get("helpful_vote"),
                    "provenance": {
                        "schema_version": SOURCE_RECORD_SCHEMA_VERSION,
                        "source_file": config.source_reviews_jsonl,
                        "source_sha256": source_hash,
                        "source_line_number": line_number,
                        "source_review_id": review_id,
                        "source_product_name": row.get("product_name"),
                    },
                }
            )
    if not matched:
        raise LaptopConfigError(f"No rows matched parent_asin {config.parent_asin!r}")
    available_ids = {row["review_id"] for row in matched}
    missing = [item for item in config.pilot_review_ids if item not in available_ids]
    if missing:
        raise LaptopConfigError(f"Configured pilot_review_ids not found: {missing}")
    return matched, source_hash


def _round_robin(rows: Sequence[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    buckets: dict[int, deque[dict[str, Any]]] = {}
    for rating in range(1, 6):
        bucket = [row for row in rows if int(row["rating"]) == rating]
        bucket.sort(
            key=lambda row: (
                hashlib.sha256(
                    f"{seed}\x1f{rating}\x1f{row['review_id']}".encode()
                ).hexdigest(),
                row["review_id"],
            )
        )
        if bucket:
            buckets[rating] = deque(bucket)
    ordered: list[dict[str, Any]] = []
    while buckets:
        for rating in sorted(tuple(buckets)):
            ordered.append(buckets[rating].popleft())
            if not buckets[rating]:
                del buckets[rating]
    return ordered


def _select(
    rows: Sequence[dict[str, Any]], config: LaptopProductConfig, limit: int | None
) -> list[dict[str, Any]]:
    count = config.default_limit if limit is None else _require_int(limit, "limit", 1)
    by_id = {row["review_id"]: row for row in rows}
    ordered = [by_id[item] for item in config.pilot_review_ids]
    pilot_set = set(config.pilot_review_ids)
    ordered.extend(
        _round_robin(
            [row for row in rows if row["review_id"] not in pilot_set],
            config.selection_seed,
        )
    )
    return ordered[: min(count, len(ordered))]


def select_product_family_reviews(
    project_root: str | Path,
    config: LaptopProductConfig,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Return prepared reviews in the config-bound deterministic order."""

    root = _project_root(project_root)
    rows, _ = _load_family(root, config)
    return _select(rows, config, limit)


_ABBREVIATIONS = {"dr", "e.g", "etc", "i.e", "jr", "mr", "mrs", "ms", "prof", "sr", "st", "vs"}
_CLOSERS = "\"')]}\u2019\u201d"


def _period_boundary(text: str, index: int) -> bool:
    if index > 0 and index + 1 < len(text):
        if text[index - 1].isdigit() and text[index + 1].isdigit():
            return False
    match = re.search(r"([A-Za-z](?:[A-Za-z.]*)?)$", text[:index])
    token = match.group(1).lower() if match else ""
    return token not in _ABBREVIATIONS and not (len(token) == 1 and token.isalpha())


def segment_with_offsets(text: str) -> list[dict[str, Any]]:
    """Split clean text while retaining exact inclusive/exclusive offsets."""

    if not isinstance(text, str) or not text.strip():
        raise ValueError("Cannot segment empty text")
    boundaries: list[int] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char == "\n":
            boundaries.append(index)
            index += 1
            continue
        if char in ".!?" and (char != "." or _period_boundary(text, index)):
            end = index + 1
            while end < len(text) and text[end] in ".!?":
                end += 1
            while end < len(text) and text[end] in _CLOSERS:
                end += 1
            if end == len(text) or text[end].isspace():
                boundaries.append(end)
                index = end
                continue
        index += 1
    if not boundaries or boundaries[-1] != len(text):
        boundaries.append(len(text))

    spans: list[dict[str, Any]] = []
    cursor = 0
    for boundary in boundaries:
        start, end = cursor, boundary
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start < end:
            spans.append(
                {"sentence_text": text[start:end], "start_char": start, "end_char": end}
            )
        cursor = boundary
    if not spans:
        raise ValueError("No sentences produced")
    for previous, current in zip(spans, spans[1:]):
        if text[previous["end_char"] : current["start_char"]].strip():
            raise ValueError("Sentence segmentation left uncovered text")
    for span in spans:
        if text[span["start_char"] : span["end_char"]] != span["sentence_text"]:
            raise ValueError("Sentence offset alignment failed")
    return spans


def _sentences(reviews: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for review in reviews:
        for index, span in enumerate(segment_with_offsets(review["clean_text"]), 1):
            records.append(
                {
                    "schema_version": SOURCE_RECORD_SCHEMA_VERSION,
                    "sentence_id": f"{review['review_id']}_S{index:03d}",
                    "review_id": review["review_id"],
                    "sentence_index": index,
                    "source_field": "clean_text",
                    "product_id": review["product_id"],
                    "asin": review["asin"],
                    "parent_asin": review["parent_asin"],
                    "product_model": review["product_model"],
                    "rating": review["rating"],
                    **span,
                }
            )
    return records


def _titles(reviews: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for review in reviews:
        title = clean_review_html(review["title"]) if review.get("title") else ""
        if title:
            records.append(
                {
                    "schema_version": SOURCE_RECORD_SCHEMA_VERSION,
                    "sentence_id": f"{review['review_id']}_TITLE",
                    "review_id": review["review_id"],
                    "source_field": "title",
                    "product_id": review["product_id"],
                    "asin": review["asin"],
                    "parent_asin": review["parent_asin"],
                    "product_model": review["product_model"],
                    "rating": review["rating"],
                    "sentence_text": title,
                    "start_char": 0,
                    "end_char": len(title),
                }
            )
    return records


def _ratings(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(int(row["rating"]) for row in rows)
    return {str(value): counts.get(value, 0) for value in range(1, 6)}


def _prepare_result(
    config: LaptopProductConfig,
    output_dir: Path,
    selected_ids: Sequence[str],
    source_hash: str,
    created: bool,
) -> PrepareResult:
    return PrepareResult(
        product_id=config.product_id,
        output_dir=output_dir,
        reviews_path=output_dir / "prepared_reviews.jsonl",
        sentences_path=output_dir / "sentences.jsonl",
        titles_path=output_dir / "titles.jsonl",
        manifest_path=output_dir / "prepare_manifest.json",
        selected_review_ids=tuple(selected_ids),
        config_sha256=config.config_sha256,
        taxonomy_sha256=config.taxonomy_sha256,
        source_sha256=source_hash,
        was_created=created,
    )


def prepare_laptop_product(
    project_root: str | Path,
    config_path: str | Path,
    limit: int | None = None,
) -> PrepareResult:
    """Prepare immutable files under the configured isolated output directory."""

    root = _project_root(project_root)
    config = load_laptop_product_config(root, config_path)
    available, source_hash = _load_family(root, config)
    selected = _select(available, config, limit)
    selected_ids = [row["review_id"] for row in selected]
    sentence_rows = _sentences(selected)
    title_rows = _titles(selected)
    reviews_bytes = _jsonl_bytes(selected)
    sentences_bytes = _jsonl_bytes(sentence_rows)
    titles_bytes = _jsonl_bytes(title_rows)
    requested_limit = config.default_limit if limit is None else limit
    manifest = {
        "schema_version": PREPARE_MANIFEST_SCHEMA_VERSION,
        "product": {
            "product_id": config.product_id,
            "display_name": config.display_name,
            "model_label": config.model_label,
            "catalog_scope": config.catalog_scope,
            "parent_asin": config.parent_asin,
            "allowed_child_asins": list(config.allowed_child_asins),
        },
        "inputs": {
            "config_path": config.config_path,
            "config_sha256": config.config_sha256,
            "source_path": config.source_reviews_jsonl,
            "source_sha256": source_hash,
            "taxonomy_path": config.taxonomy_config_path,
            "taxonomy_sha256": config.taxonomy_sha256,
        },
        "selection": {
            "strategy": config.selection_strategy,
            "seed": config.selection_seed,
            "requested_limit": requested_limit,
            "available_review_count": len(available),
            "selected_review_count": len(selected),
            "pilot_review_ids": list(config.pilot_review_ids),
            "selected_review_ids": selected_ids,
            "selected_review_ids_sha256": _sha256_bytes(
                _canonical_json_bytes(selected_ids)
            ),
            "available_rating_distribution": _ratings(available),
            "selected_rating_distribution": _ratings(selected),
            "sampling_note": (
                "The order is config-bound and deterministic. Rating buckets are "
                "covered when available; no statistical balance is claimed."
            ),
        },
        "title_policy": (
            "Titles are retained in reviews and emitted separately; they are never "
            "injected into clean_text sentences."
        ),
        "artifacts": {
            "prepared_reviews.jsonl": {
                "record_count": len(selected), "sha256": _sha256_bytes(reviews_bytes)
            },
            "sentences.jsonl": {
                "record_count": len(sentence_rows), "sha256": _sha256_bytes(sentences_bytes)
            },
            "titles.jsonl": {
                "record_count": len(title_rows), "sha256": _sha256_bytes(titles_bytes)
            },
        },
    }
    expected = {
        "prepared_reviews.jsonl": reviews_bytes,
        "sentences.jsonl": sentences_bytes,
        "titles.jsonl": titles_bytes,
        "prepare_manifest.json": _canonical_json_bytes(manifest, indent=2),
    }
    output_dir = _resolve_project_relative(root, config.output_directory, "output.directory")
    present = {name: (output_dir / name).exists() for name in expected}
    if all(present.values()):
        mismatched = [
            name for name, payload in expected.items()
            if (output_dir / name).read_bytes() != payload
        ]
        if mismatched:
            raise ImmutablePreparedOutputError(
                "Refusing to overwrite immutable artifacts: " + ", ".join(mismatched)
            )
        return _prepare_result(config, output_dir, selected_ids, source_hash, False)
    if any(present.values()):
        raise ImmutablePreparedOutputError("Prepared output is incomplete; refusing overwrite")

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in expected.items():
        try:
            with (output_dir / name).open("xb") as handle:
                handle.write(payload)
        except FileExistsError as exc:
            raise ImmutablePreparedOutputError(
                f"Refusing to overwrite immutable artifact: {name}"
            ) from exc
    return _prepare_result(config, output_dir, selected_ids, source_hash, True)


def list_laptop_products(project_root: str | Path) -> list[dict[str, Any]]:
    """List catalog products and source counts without writing or loading models."""

    root = _project_root(project_root)
    config_dir = _resolve_project_relative(root, "configs/products", "product config dir")
    if not config_dir.is_dir():
        raise LaptopConfigError("configs/products does not exist")
    products: list[dict[str, Any]] = []
    paths = list(config_dir.glob("*.json"))
    for directory in sorted(path for path in config_dir.iterdir() if path.is_dir()):
        # Catalog expansion writes configs into an isolated subdirectory.  A
        # final marker is written only after every source, config, and manifest
        # has been frozen, so interrupted builds never leak partial products.
        if (directory / ".published").is_file():
            paths.extend(directory.glob("*.json"))
    for path in sorted(paths):
        config = load_laptop_product_config(root, path.relative_to(root).as_posix())
        rows, _ = _load_family(root, config)
        products.append(
            {
                "product_id": config.product_id,
                "display_name": config.display_name,
                "catalog_scope": config.catalog_scope,
                "scope": config.catalog_scope,
                "available_review_count": len(rows),
                "default_limit": config.default_limit,
                "config_path": config.config_path,
                "parent_asin": config.parent_asin,
                "allowed_child_asins": list(config.allowed_child_asins),
            }
        )
    return sorted(products, key=lambda item: item["product_id"])


__all__ = [
    "ImmutablePreparedOutputError", "LaptopConfigError", "LaptopProductConfig",
    "PrepareResult", "clean_review_html", "list_laptop_products",
    "load_laptop_product_config", "load_laptop_taxonomy", "prepare_laptop_product",
    "segment_with_offsets", "select_product_family_reviews",
]
