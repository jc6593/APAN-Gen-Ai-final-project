"""Versioned, resumable collection for the ReviewLens laptop demo catalog.

This module expands the catalog data only.  It never imports or calls MLX,
Qwen, or the frozen Evidence-first evaluation code.  Existing product configs
and caches are treated as immutable; newly qualified products receive their own
review file and product config.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


CATALOG_BUILD_SCHEMA_VERSION = "reviewlens.laptop_catalog_build.v1"
CATALOG_PLAN_SCHEMA_VERSION = "reviewlens.laptop_catalog_plan.v1"
CATALOG_CHECKPOINT_SCHEMA_VERSION = "reviewlens.laptop_catalog_checkpoint.v1"
CATALOG_MANIFEST_SCHEMA_VERSION = "reviewlens.laptop_catalog_manifest.v1"
PRODUCT_CONFIG_SCHEMA_VERSION = "reviewlens.product_config.v1"
TAXONOMY_SCHEMA_VERSION = "reviewlens.taxonomy.laptop.v1"


class CatalogBuildError(RuntimeError):
    """The catalog build contract or source data is invalid."""


class CatalogImmutableOutputError(RuntimeError):
    """An existing frozen catalog artifact differs from expected output."""


@dataclass(frozen=True)
class CatalogBuildConfig:
    catalog_id: str
    config_path: str
    config_sha256: str
    candidate_metadata_jsonl: str
    review_repository: str
    review_jsonl_path: str
    target_new_products: int
    candidate_pool_size: int
    minimum_reviews_per_product: int
    maximum_reviews_per_product: int
    minimum_text_characters: int
    maximum_scan_records: int
    checkpoint_interval_records: int
    maximum_products_per_brand: int
    minimum_distinct_brands: int
    required_operating_systems: tuple[str, ...]
    selection_seed: int
    include_renewed: bool
    excluded_parent_asins: tuple[str, ...]
    output_directory: str
    generated_config_directory: str
    generated_default_limit: int


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


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _project_root(project_root: str | Path) -> Path:
    root = Path(project_root).resolve()
    if not root.is_dir():
        raise CatalogBuildError(f"Project root does not exist: {root}")
    return root


def _resolve_project_relative(root: Path, raw: str, context: str) -> Path:
    path = Path(raw)
    if not raw or "\x00" in raw or path.is_absolute() or ".." in path.parts:
        raise CatalogBuildError(f"{context} must be a safe project-relative path")
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise CatalogBuildError(f"{context} resolves outside the project") from exc
    return resolved


def _require_object(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CatalogBuildError(f"{context} must be a JSON object")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], context: str
) -> None:
    actual = set(value)
    if actual != expected:
        raise CatalogBuildError(
            f"{context} keys do not match schema; "
            f"missing={sorted(expected - actual)}, unknown={sorted(actual - expected)}"
        )


def _require_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CatalogBuildError(f"{context} must be a non-empty string")
    return value.strip()


def _require_int(value: Any, context: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CatalogBuildError(f"{context} must be an integer >= {minimum}")
    return value


def load_catalog_build_config(
    project_root: str | Path, config_path: str | Path
) -> CatalogBuildConfig:
    """Load the exact versioned contract for one catalog expansion."""

    root = _project_root(project_root)
    config_text = str(config_path)
    path = _resolve_project_relative(root, config_text, "catalog config path")
    try:
        payload = path.read_bytes()
        document = _require_object(json.loads(payload), "catalog config")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CatalogBuildError(f"Cannot read catalog config: {path}") from exc
    _require_exact_keys(
        document,
        {"schema_version", "catalog_id", "category", "source", "selection", "output"},
        "catalog config",
    )
    if document["schema_version"] != CATALOG_BUILD_SCHEMA_VERSION:
        raise CatalogBuildError("Unsupported catalog config schema_version")
    if document["category"] != "laptop":
        raise CatalogBuildError("Catalog category must be 'laptop'")
    catalog_id = _require_string(document["catalog_id"], "catalog_id")
    if not re.fullmatch(r"[a-z0-9]+(?:_[a-z0-9]+)*", catalog_id):
        raise CatalogBuildError("catalog_id must use lower_snake_case")

    source = _require_object(document["source"], "source")
    _require_exact_keys(
        source,
        {"candidate_metadata_jsonl", "review_repository", "review_jsonl_path"},
        "source",
    )
    metadata_path = _require_string(
        source["candidate_metadata_jsonl"], "source.candidate_metadata_jsonl"
    )
    if not _resolve_project_relative(
        root, metadata_path, "source.candidate_metadata_jsonl"
    ).is_file():
        raise CatalogBuildError(f"Candidate metadata does not exist: {metadata_path}")
    repository = _require_string(
        source["review_repository"], "source.review_repository"
    )
    if not re.fullmatch(r"[^/\s]+/[^/\s]+", repository):
        raise CatalogBuildError("review_repository must be namespace/name")
    review_path = _require_string(source["review_jsonl_path"], "review_jsonl_path")
    if review_path.startswith("/") or ".." in Path(review_path).parts:
        raise CatalogBuildError("review_jsonl_path must be repository-relative")

    selection = _require_object(document["selection"], "selection")
    _require_exact_keys(
        selection,
        {
            "target_new_products", "candidate_pool_size",
            "minimum_reviews_per_product", "maximum_reviews_per_product",
            "minimum_text_characters", "maximum_scan_records",
            "checkpoint_interval_records", "maximum_products_per_brand",
            "minimum_distinct_brands", "required_operating_systems",
            "seed", "include_renewed",
            "excluded_parent_asins",
        },
        "selection",
    )
    target = _require_int(selection["target_new_products"], "target_new_products", 1)
    pool_size = _require_int(selection["candidate_pool_size"], "candidate_pool_size", 1)
    minimum_reviews = _require_int(
        selection["minimum_reviews_per_product"], "minimum_reviews_per_product", 1
    )
    maximum_reviews = _require_int(
        selection["maximum_reviews_per_product"], "maximum_reviews_per_product", 1
    )
    maximum_scan = _require_int(
        selection["maximum_scan_records"], "maximum_scan_records", 1
    )
    checkpoint_interval = _require_int(
        selection["checkpoint_interval_records"], "checkpoint_interval_records", 1
    )
    brand_cap = _require_int(
        selection["maximum_products_per_brand"], "maximum_products_per_brand", 1
    )
    min_brands = _require_int(
        selection["minimum_distinct_brands"], "minimum_distinct_brands", 1
    )
    if pool_size < target:
        raise CatalogBuildError("candidate_pool_size must be >= target_new_products")
    if maximum_reviews < minimum_reviews:
        raise CatalogBuildError(
            "maximum_reviews_per_product must be >= minimum_reviews_per_product"
        )
    if min_brands > target:
        raise CatalogBuildError("minimum_distinct_brands cannot exceed target_new_products")
    required_os = selection["required_operating_systems"]
    if (
        not isinstance(required_os, list)
        or any(item not in {"windows", "macos", "chromeos"} for item in required_os)
        or len(set(required_os)) != len(required_os)
        or len(required_os) > target
    ):
        raise CatalogBuildError(
            "required_operating_systems must be unique supported OS labels"
        )
    include_renewed = selection["include_renewed"]
    if not isinstance(include_renewed, bool):
        raise CatalogBuildError("include_renewed must be boolean")
    excluded = selection["excluded_parent_asins"]
    if (
        not isinstance(excluded, list)
        or any(not isinstance(item, str) or not re.fullmatch(r"[A-Z0-9]{10}", item)
               for item in excluded)
        or len(set(excluded)) != len(excluded)
    ):
        raise CatalogBuildError("excluded_parent_asins must contain unique ASINs")

    output = _require_object(document["output"], "output")
    _require_exact_keys(
        output,
        {"directory", "generated_config_directory", "generated_default_limit"},
        "output",
    )
    output_dir = _require_string(output["directory"], "output.directory")
    generated_dir = _require_string(
        output["generated_config_directory"], "output.generated_config_directory"
    )
    _resolve_project_relative(root, output_dir, "output.directory")
    _resolve_project_relative(root, generated_dir, "output.generated_config_directory")
    default_limit = _require_int(
        output["generated_default_limit"], "output.generated_default_limit", 1
    )
    if default_limit > maximum_reviews:
        raise CatalogBuildError("generated_default_limit exceeds collected review cap")

    return CatalogBuildConfig(
        catalog_id=catalog_id,
        config_path=Path(config_text).as_posix(),
        config_sha256=_sha256_bytes(payload),
        candidate_metadata_jsonl=metadata_path,
        review_repository=repository,
        review_jsonl_path=review_path,
        target_new_products=target,
        candidate_pool_size=pool_size,
        minimum_reviews_per_product=minimum_reviews,
        maximum_reviews_per_product=maximum_reviews,
        minimum_text_characters=_require_int(
            selection["minimum_text_characters"], "minimum_text_characters", 1
        ),
        maximum_scan_records=maximum_scan,
        checkpoint_interval_records=checkpoint_interval,
        maximum_products_per_brand=brand_cap,
        minimum_distinct_brands=min_brands,
        required_operating_systems=tuple(required_os),
        selection_seed=_require_int(selection["seed"], "seed", 0),
        include_renewed=include_renewed,
        excluded_parent_asins=tuple(excluded),
        output_directory=output_dir,
        generated_config_directory=generated_dir,
        generated_default_limit=default_limit,
    )


_BRAND_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (brand, re.compile(pattern, re.IGNORECASE))
    for brand, pattern in (
        ("apple", r"\bapple\b|\bmacbook\b"),
        ("acer", r"\bacer\b"),
        ("asus", r"\basus\b"),
        ("dell", r"\bdell\b"),
        ("hp", r"\bhp\b|\bhewlett[ -]packard\b"),
        ("lenovo", r"\blenovo\b"),
        ("microsoft", r"\bmicrosoft\b|\bsurface laptop\b"),
        ("samsung", r"\bsamsung\b|\bgalaxy book\b"),
        ("msi", r"\bmsi\b"),
        ("lg", r"\blg\b|\blg gram\b"),
        ("razer", r"\brazer\b"),
        ("gateway", r"\bgateway\b"),
    )
)

_DEVICE_WORDS = re.compile(
    r"\b(?:laptop|notebook|ultrabook|chromebook|macbook|thinkpad|ideapad|"
    r"vivobook|zenbook|surface\s+laptop|galaxy\s+book|lg\s+gram)\b",
    re.IGNORECASE,
)
_STRONG_ACCESSORY = re.compile(
    r"\b(?:backpack|briefcase|bag|case|cover|cable|mouse|sleeve|"
    r"protective\s+case|hard\s+shell|hardshell|case\s+cover|skin|decal|stickers?|"
    r"screen\s+protector|privacy\s+filter|charger|power\s+adapter|usb\s+cable|"
    r"hdmi\s+cable|displayport\s+cable|docking\s+station|hub|laptop\s+stand|"
    r"wireless\s+mouse|computer\s+mouse|keyboard\s+and\s+mouse|wireless\s+keyboard|"
    r"earbuds?|headphones?|computer\s+speakers?|graphics?\s+(?:drawing\s+)?tablet|"
    r"webcam|cooling\s+pad|cooling\s+fan|heatsink|motherboard|bottom\s+case|"
    r"replacement\s+screen|replacement\s+battery|battery\s+for|"
    r"laptop\s+battery\s+(?:for|compatible)|"
    r"sodimm|so-dimm|memory\s+modules?|memory\s+kit|ram\s+upgrade|"
    r"external\s+(?:hard\s+)?drive|flash\s+drive|usb\s+drive|drive\s+enclosure|"
    r"external\s+(?:cd|dvd)|power\s+strip|surge\s+protector|"
    r"cleaning\s+kit|screen\s+cleaner|cleaner\s+solution)\b",
    re.IGNORECASE,
)
_ACCESSORY_RELATION = re.compile(
    r"\b(?:compatible\s+with|designed\s+for|case\s+for|cover\s+for|skin\s+for|fits)\b|"
    r"\bfor\s+(?:\d+(?:\.\d+)?[- ]?inch\s+)?(?:apple\s+)?"
    r"(?:macbook|chromebook|surface\s+laptop|dell|hp|lenovo|acer|asus)\b",
    re.IGNORECASE,
)
_CPU = re.compile(
    r"\b(?:intel|amd|ryzen|celeron|pentium|snapdragon|apple\s+m[1234]|m[1234]\s+(?:pro|max)|"
    r"core\s+i[3579]|i[3579]-\d{4,5}|mt\d{4}|processor|cpu)\b",
    re.IGNORECASE,
)
_MEMORY = re.compile(
    r"\b\d{1,3}\s*gb\s+(?:ram|memory|ddr\d?|lpddr\d?x?)\b",
    re.IGNORECASE,
)
_STORAGE = re.compile(
    r"\b(?:\d{1,4}\s*(?:gb|tb)\s+(?:ssd|hdd|emmc|storage)|ssd|emmc)\b",
    re.IGNORECASE,
)
_DISPLAY = re.compile(
    r"\b(?:\d{2}(?:\.\d)?[ -]?(?:inch|in\b)|display|screen|retina|fhd|"
    r"full\s+hd|ips|oled|touchscreen)\b",
    re.IGNORECASE,
)
_OS = re.compile(
    r"\b(?:windows\s*(?:10|11)?|win\s*(?:10|11)|chrome\s*os|macbook)\b",
    re.IGNORECASE,
)


def classify_laptop_title(title: str, *, include_renewed: bool = False) -> dict[str, str] | None:
    """Return high-precision device metadata, or ``None`` for a likely accessory."""

    if not isinstance(title, str) or not title.strip():
        return None
    normalized = re.sub(r"\s+", " ", title).strip()
    lower = normalized.casefold()
    if _STRONG_ACCESSORY.search(normalized) or _ACCESSORY_RELATION.search(normalized):
        return None
    if not _DEVICE_WORDS.search(normalized):
        return None
    brand = next(
        (name for name, pattern in _BRAND_PATTERNS if pattern.search(normalized)), None
    )
    if brand is None:
        return None
    condition = "renewed" if re.search(r"\b(?:renewed|refurbished)\b", lower) else "standard"
    if condition == "renewed" and not include_renewed:
        return None
    hardware_signals = sum(
        bool(pattern.search(normalized))
        for pattern in (_CPU, _MEMORY, _STORAGE, _DISPLAY, _OS)
    )
    if hardware_signals < 2:
        return None
    if brand == "apple" or "macbook" in lower:
        operating_system = "macos"
    elif "chromebook" in lower or re.search(r"chrome\s*os", lower):
        operating_system = "chromeos"
    elif "windows" in lower or re.search(r"\bwin\s*(?:10|11)\b", lower):
        operating_system = "windows"
    else:
        operating_system = "unknown"
    if re.search(r"\b(?:gaming|rog|nitro|predator|tuf|legion|omen|victus|razer)\b", lower):
        segment = "gaming"
    elif re.search(r"\b(?:thinkpad|latitude|elitebook|probook|surface\s+laptop)\b", lower):
        segment = "business"
    elif re.search(r"\b(?:macbook\s+pro|xps|zenbook|spectre|creator)\b", lower):
        segment = "premium_creator"
    else:
        segment = "consumer"
    family_tokens = (
        "macbook pro", "macbook air", "aspire", "swift", "nitro", "predator",
        "inspiron", "latitude", "xps", "vostro", "thinkpad", "ideapad", "yoga",
        "legion", "pavilion", "envy", "spectre", "elitebook", "probook", "victus",
        "omen", "stream", "vivobook", "zenbook", "rog", "tuf", "surface laptop",
        "galaxy book", "lg gram", "chromebook",
    )
    family = next((item.replace(" ", "_") for item in family_tokens if item in lower), "laptop")
    return {
        "brand": brand,
        "operating_system": operating_system,
        "segment": segment,
        "condition": condition,
        "family": family,
        "hardware_signal_count": str(hardware_signals),
    }


def _display_name(title: str) -> str:
    cleaned = re.sub(r"\s+", " ", title).strip()
    if "|" in cleaned:
        cleaned = cleaned.split("|", 1)[0].strip()
    match = re.match(r"(Apple\s+(?:20\d{2}\s+)?MacBook\s+(?:Air|Pro))\b", cleaned, re.I)
    if match:
        return match.group(1)
    if len(cleaned) <= 96:
        return cleaned
    shortened = cleaned[:96].rsplit(" ", 1)[0].rstrip(" ,;:-")
    return shortened or cleaned[:96]


def _product_id(candidate: Mapping[str, Any]) -> str:
    brand = re.sub(r"[^a-z0-9]+", "_", str(candidate["brand"]).casefold()).strip("_")
    family = re.sub(r"[^a-z0-9]+", "_", str(candidate["family"]).casefold()).strip("_")
    asin = str(candidate["parent_asin"]).casefold()
    return f"{brand}_{family}_{asin}"


def build_candidate_plan(
    project_root: str | Path, config: CatalogBuildConfig
) -> dict[str, Any]:
    """Filter local metadata into a deterministic, diverse scan pool."""

    root = _project_root(project_root)
    metadata_path = _resolve_project_relative(
        root, config.candidate_metadata_jsonl, "candidate metadata"
    )
    candidates_by_asin: dict[str, dict[str, Any]] = {}
    rejected = Counter()
    total_rows = 0
    with metadata_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            total_rows += 1
            try:
                row = _require_object(json.loads(line), f"metadata row {line_number}")
            except json.JSONDecodeError as exc:
                raise CatalogBuildError(
                    f"Invalid JSONL at {config.candidate_metadata_jsonl}:{line_number}"
                ) from exc
            asin = row.get("parent_asin")
            title = row.get("title")
            if not isinstance(asin, str) or not re.fullmatch(r"[A-Z0-9]{10}", asin):
                rejected["invalid_parent_asin"] += 1
                continue
            if asin in config.excluded_parent_asins:
                rejected["existing_product_excluded"] += 1
                continue
            classification = classify_laptop_title(
                title, include_renewed=config.include_renewed
            ) if isinstance(title, str) else None
            if classification is None:
                rejected["not_high_confidence_laptop"] += 1
                continue
            rating_number = row.get("rating_number")
            if isinstance(rating_number, bool) or not isinstance(rating_number, (int, float)):
                rating_number = 0
            average_rating = row.get("average_rating")
            candidate = {
                "parent_asin": asin,
                "title": title,
                "display_name": _display_name(title),
                "rating_number": int(max(0, rating_number)),
                "average_rating": average_rating,
                **classification,
            }
            previous = candidates_by_asin.get(asin)
            if previous is None or candidate["rating_number"] > previous["rating_number"]:
                candidates_by_asin[asin] = candidate

    ranked = sorted(
        candidates_by_asin.values(),
        key=lambda row: (-row["rating_number"], row["parent_asin"]),
    )
    brand_cap = max(4, math.ceil(config.candidate_pool_size / 5))
    selected: list[dict[str, Any]] = []
    selected_asins: set[str] = set()
    brand_counts: Counter[str] = Counter()

    def add(candidate: dict[str, Any]) -> bool:
        asin = candidate["parent_asin"]
        if asin in selected_asins or brand_counts[candidate["brand"]] >= brand_cap:
            return False
        selected.append(candidate)
        selected_asins.add(asin)
        brand_counts[candidate["brand"]] += 1
        return True

    # Reserve several scan slots for both non-Windows systems.  This improves
    # the chance of meeting catalog diversity without lowering the review bar.
    reserve_per_os = min(4, max(1, config.candidate_pool_size // 8))
    for operating_system in config.required_operating_systems:
        reserved = 0
        for candidate in ranked:
            if candidate["operating_system"] != operating_system:
                continue
            if add(candidate):
                reserved += 1
            if reserved >= reserve_per_os:
                break
    for candidate in ranked:
        if brand_counts[candidate["brand"]] >= brand_cap:
            continue
        add(candidate)
        if len(selected) >= config.candidate_pool_size:
            break
    selected.sort(key=lambda row: (-row["rating_number"], row["parent_asin"]))
    if len(selected) < config.target_new_products:
        raise CatalogBuildError(
            f"Only {len(selected)} high-confidence candidates remain; "
            f"need at least {config.target_new_products}"
        )
    return {
        "schema_version": CATALOG_PLAN_SCHEMA_VERSION,
        "catalog_id": config.catalog_id,
        "config_path": config.config_path,
        "config_sha256": config.config_sha256,
        "candidate_metadata_path": config.candidate_metadata_jsonl,
        "candidate_metadata_sha256": _sha256_file(metadata_path),
        "metadata_rows_read": total_rows,
        "high_confidence_candidates": len(ranked),
        "rejected_counts": dict(sorted(rejected.items())),
        "candidate_pool_brand_cap": brand_cap,
        "candidate_pool": selected,
        "scope_note": (
            "Candidates come from the configured bounded metadata file, not the full "
            "Amazon laptop market. Product-page rating_number is used only to rank "
            "the scan pool; it is not a collected review count."
        ),
    }


def _selected_products(
    config: CatalogBuildConfig,
    candidates: Sequence[Mapping[str, Any]],
    collected: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    qualified = [
        dict(candidate)
        for candidate in candidates
        if len(collected.get(str(candidate["parent_asin"]), ()))
        >= config.minimum_reviews_per_product
    ]
    qualified.sort(
        key=lambda row: (
            -len(collected[str(row["parent_asin"])]),
            -int(row["rating_number"]),
            str(row["parent_asin"]),
        )
    )
    chosen: list[dict[str, Any]] = []
    chosen_asins: set[str] = set()
    brand_counts: Counter[str] = Counter()

    def add(row: Mapping[str, Any]) -> bool:
        asin = str(row["parent_asin"])
        brand = str(row["brand"])
        if asin in chosen_asins or brand_counts[brand] >= config.maximum_products_per_brand:
            return False
        chosen.append(dict(row))
        chosen_asins.add(asin)
        brand_counts[brand] += 1
        return True

    for operating_system in config.required_operating_systems:
        row = next(
            (item for item in qualified if item["operating_system"] == operating_system),
            None,
        )
        if row is None or not add(row):
            return []
    for row in qualified:
        if len(chosen) >= config.target_new_products:
            break
        if row["brand"] not in brand_counts:
            add(row)
    for row in qualified:
        if len(chosen) >= config.target_new_products:
            break
        add(row)
    if len(chosen) < config.target_new_products:
        return []
    if len({row["brand"] for row in chosen}) < config.minimum_distinct_brands:
        return []
    return chosen[: config.target_new_products]


def _clean_review_record(
    line_number: int,
    record: Mapping[str, Any],
    candidate: Mapping[str, Any],
    config: CatalogBuildConfig,
) -> dict[str, Any] | None:
    text = str(record.get("text") or "").strip()
    if len(text) < config.minimum_text_characters:
        return None
    asin = record.get("asin")
    if not isinstance(asin, str) or not re.fullmatch(r"[A-Z0-9]{10}", asin):
        return None
    rating = record.get("rating")
    if isinstance(rating, bool) or not isinstance(rating, (int, float)):
        return None
    numeric_rating = float(rating)
    if not numeric_rating.is_integer() or not 1 <= numeric_rating <= 5:
        return None
    title = record.get("title")
    if title is not None and not isinstance(title, str):
        title = str(title)
    return {
        "review_id": f"electronics_{line_number}",
        "parent_asin": candidate["parent_asin"],
        "product_name": candidate["display_name"],
        "asin": asin,
        "rating": numeric_rating,
        "title": title,
        "text": text,
        "timestamp": record.get("timestamp"),
        "verified_purchase": record.get("verified_purchase"),
        "helpful_vote": record.get("helpful_vote"),
    }


def _empty_state(config: CatalogBuildConfig, plan_sha256: str, revision: str) -> dict[str, Any]:
    return {
        "schema_version": CATALOG_CHECKPOINT_SCHEMA_VERSION,
        "catalog_id": config.catalog_id,
        "config_sha256": config.config_sha256,
        "plan_sha256": plan_sha256,
        "source_revision": revision,
        "status": "in_progress",
        "scanned_records": 0,
        "source_byte_offset": 0,
        "target_matches": 0,
        "usable_records": 0,
        "rejected_target_records": 0,
        "collected": {},
    }


def _validate_checkpoint_state(
    state: Mapping[str, Any],
    config: CatalogBuildConfig,
    plan_hash: str,
    source_revision: str,
    candidates: Mapping[str, Mapping[str, Any]],
) -> None:
    if (
        state.get("schema_version") != CATALOG_CHECKPOINT_SCHEMA_VERSION
        or state.get("catalog_id") != config.catalog_id
        or state.get("config_sha256") != config.config_sha256
        or state.get("plan_sha256") != plan_hash
        or state.get("source_revision") != source_revision
        or state.get("status")
        not in {"in_progress", "ready_to_freeze", "scan_limit_reached", "complete"}
    ):
        raise CatalogBuildError("Checkpoint contract does not match this catalog run")
    for field in (
        "scanned_records", "source_byte_offset", "target_matches",
        "usable_records", "rejected_target_records",
    ):
        value = state.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise CatalogBuildError(f"Checkpoint {field} is invalid")
    if int(state["scanned_records"]) > config.maximum_scan_records:
        raise CatalogBuildError("Checkpoint exceeds the configured scan cap")
    collected = state.get("collected")
    if not isinstance(collected, dict):
        raise CatalogBuildError("Checkpoint collected must be an object")
    seen_ids: set[str] = set()
    retained = 0
    expected_keys = {
        "review_id", "parent_asin", "product_name", "asin", "rating", "title",
        "text", "timestamp", "verified_purchase", "helpful_vote",
    }
    for parent_asin, rows in collected.items():
        candidate = candidates.get(parent_asin)
        if candidate is None or not isinstance(rows, list):
            raise CatalogBuildError("Checkpoint contains an unplanned product")
        if len(rows) > config.maximum_reviews_per_product:
            raise CatalogBuildError("Checkpoint exceeds the per-product review cap")
        retained += len(rows)
        for row in rows:
            if not isinstance(row, dict) or set(row) != expected_keys:
                raise CatalogBuildError("Checkpoint contains a malformed review row")
            review_id = row.get("review_id")
            match = re.fullmatch(r"electronics_(\d+)", str(review_id))
            rating = row.get("rating")
            if (
                match is None
                or not isinstance(review_id, str)
                or int(match.group(1)) > int(state["scanned_records"])
                or review_id in seen_ids
                or row.get("parent_asin") != parent_asin
                or row.get("product_name") != candidate["display_name"]
                or not isinstance(row.get("asin"), str)
                or re.fullmatch(r"[A-Z0-9]{10}", row["asin"]) is None
                or isinstance(rating, bool)
                or not isinstance(rating, (int, float))
                or not float(rating).is_integer()
                or not 1 <= float(rating) <= 5
                or not isinstance(row.get("text"), str)
                or len(row["text"].strip()) < config.minimum_text_characters
            ):
                raise CatalogBuildError("Checkpoint review content is invalid")
            seen_ids.add(str(review_id))
    if int(state["usable_records"]) < retained:
        raise CatalogBuildError("Checkpoint usable count is smaller than retained rows")
    selected = state.get("selected_parent_asins", [])
    if (
        not isinstance(selected, list)
        or any(not isinstance(item, str) for item in selected)
        or len(set(selected)) != len(selected)
        or any(item not in candidates for item in selected)
    ):
        raise CatalogBuildError("Checkpoint selected_parent_asins is invalid")


def collect_from_records(
    config: CatalogBuildConfig,
    plan: Mapping[str, Any],
    records: Iterable[
        tuple[int, Mapping[str, Any]] | tuple[int, Mapping[str, Any], int]
    ],
    *,
    source_revision: str,
    state: Mapping[str, Any] | None = None,
    progress_callback: Any | None = None,
) -> dict[str, Any]:
    """Collect capped review rows from an injected stream (also used by tests)."""

    plan_hash = _sha256_bytes(_canonical_json_bytes(plan, indent=2))
    mutable = json.loads(json.dumps(state)) if state is not None else _empty_state(
        config, plan_hash, source_revision
    )
    candidates = {
        row["parent_asin"]: row for row in plan.get("candidate_pool", [])
    }
    _validate_checkpoint_state(
        mutable, config, plan_hash, source_revision, candidates
    )
    collected: dict[str, list[dict[str, Any]]] = {
        str(asin): list(rows) for asin, rows in mutable.get("collected", {}).items()
    }
    existing_ids = {
        row["review_id"] for rows in collected.values() for row in rows
    }
    mutable["collected"] = collected
    if mutable.get("status") in {"ready_to_freeze", "complete"}:
        chosen = _selected_products(config, plan["candidate_pool"], collected)
        chosen_asins = [row["parent_asin"] for row in chosen]
        if not chosen or chosen_asins != mutable.get("selected_parent_asins"):
            raise CatalogBuildError(
                "Checkpoint frozen selection does not match its collected reviews"
            )
        mutable["status"] = "ready_to_freeze"
        return mutable
    if progress_callback:
        # Keep a live reference in the caller so an interruption between regular
        # checkpoint intervals can still persist the most recent state.
        progress_callback(mutable)
    start_after = int(mutable.get("scanned_records", 0))
    for source_item in records:
        if len(source_item) == 2:
            line_number, record = source_item
            byte_offset = None
        elif len(source_item) == 3:
            line_number, record, byte_offset = source_item
        else:
            raise CatalogBuildError("Review stream yielded an invalid source tuple")
        if line_number <= start_after:
            continue
        if line_number > config.maximum_scan_records:
            break
        mutable["scanned_records"] = line_number
        if isinstance(byte_offset, int) and byte_offset >= 0:
            mutable["source_byte_offset"] = byte_offset
        parent_asin = record.get("parent_asin")
        candidate = candidates.get(parent_asin)
        catalog_may_have_changed = False
        if candidate is not None:
            mutable["target_matches"] = int(mutable.get("target_matches", 0)) + 1
            rows = collected.setdefault(str(parent_asin), [])
            clean = _clean_review_record(line_number, record, candidate, config)
            if clean is None:
                mutable["rejected_target_records"] = int(
                    mutable.get("rejected_target_records", 0)
                ) + 1
            elif clean["review_id"] not in existing_ids:
                mutable["usable_records"] = int(mutable.get("usable_records", 0)) + 1
                existing_ids.add(clean["review_id"])
                if len(rows) < config.maximum_reviews_per_product:
                    rows.append(clean)
                    catalog_may_have_changed = True
        mutable["collected"] = collected
        if progress_callback and line_number % config.checkpoint_interval_records == 0:
            progress_callback(mutable)
        if catalog_may_have_changed:
            chosen = _selected_products(config, plan["candidate_pool"], collected)
            if chosen:
                mutable["status"] = "ready_to_freeze"
                mutable["selected_parent_asins"] = [
                    row["parent_asin"] for row in chosen
                ]
                break
    mutable["collected"] = collected
    if mutable.get("status") != "ready_to_freeze":
        chosen = _selected_products(config, plan["candidate_pool"], collected)
        if chosen:
            mutable["status"] = "ready_to_freeze"
            mutable["selected_parent_asins"] = [row["parent_asin"] for row in chosen]
        elif int(mutable.get("scanned_records", 0)) >= config.maximum_scan_records:
            mutable["status"] = "scan_limit_reached"
    return mutable


def _write_immutable(path: Path, payload: bytes) -> bool:
    if path.exists():
        if path.read_bytes() != payload:
            raise CatalogImmutableOutputError(
                f"Refusing to overwrite a different frozen artifact: {path}"
            )
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
    except FileExistsError as exc:
        raise CatalogImmutableOutputError(f"Artifact appeared during write: {path}") from exc
    return True


def _write_checkpoint(path: Path, state: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_canonical_json_bytes(state, indent=2))
    temporary.replace(path)


def _rating_distribution(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter(int(float(row["rating"])) for row in rows)
    return {str(rating): counts.get(rating, 0) for rating in range(1, 6)}


def _publish_catalog_configs(
    root: Path,
    config: CatalogBuildConfig,
    manifest_path: Path,
    manifest_payload: bytes,
) -> None:
    """Publish nested generated configs only after the catalog is complete."""

    config_dir = _resolve_project_relative(
        root, config.generated_config_directory, "generated config directory"
    )
    manifest = _require_object(json.loads(manifest_payload), "catalog manifest")
    expected_configs = {
        _resolve_project_relative(root, product["config_path"], "published config")
        for product in manifest.get("products", [])
    }
    actual_configs = set(config_dir.glob("*.json")) if config_dir.is_dir() else set()
    if actual_configs != expected_configs:
        raise CatalogImmutableOutputError(
            "Generated config directory contains missing or extra product configs"
        )
    marker = {
        "catalog_id": config.catalog_id,
        "manifest_path": manifest_path.relative_to(root).as_posix(),
        "manifest_sha256": _sha256_bytes(manifest_payload),
    }
    _write_immutable(config_dir / ".published", _canonical_json_bytes(marker, indent=2))


def _generated_product_config(
    config: CatalogBuildConfig,
    candidate: Mapping[str, Any],
    source_path: str,
    child_asins: Sequence[str],
) -> dict[str, Any]:
    product_id = _product_id(candidate)
    return {
        "schema_version": PRODUCT_CONFIG_SCHEMA_VERSION,
        "product_id": product_id,
        "category": "laptop",
        "catalog_scope": "product_family",
        "product_family": {
            "display_name": candidate["display_name"],
            "model_label": f"{candidate['display_name']} family",
            "parent_asin": candidate["parent_asin"],
            "allowed_child_asins": list(child_asins),
        },
        "source": {"reviews_jsonl": source_path},
        "taxonomy": {
            "config_path": "configs/taxonomies/laptop_v1.json",
            "schema_version": TAXONOMY_SCHEMA_VERSION,
        },
        "selection": {
            "strategy": "rating_round_robin_v1",
            "seed": config.selection_seed,
            "default_limit": config.generated_default_limit,
            "pilot_review_ids": [],
        },
        "output": {"directory": f"outputs/laptops/{product_id}"},
    }


def freeze_catalog(
    project_root: str | Path,
    config: CatalogBuildConfig,
    plan: Mapping[str, Any],
    state: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze qualified products, review sources, configs, and an audit manifest."""

    root = _project_root(project_root)
    output_dir = _resolve_project_relative(root, config.output_directory, "output directory")
    candidates = {row["parent_asin"]: row for row in plan["candidate_pool"]}
    collected = state.get("collected", {})
    chosen = _selected_products(config, plan["candidate_pool"], collected)
    if not chosen:
        qualified = {
            asin: len(rows)
            for asin, rows in collected.items()
            if len(rows) >= config.minimum_reviews_per_product
        }
        raise CatalogBuildError(
            "The scan did not produce a sufficiently diverse catalog. "
            f"Qualified products: {qualified or 'none'}"
        )

    products: list[dict[str, Any]] = []
    for candidate in chosen:
        asin = candidate["parent_asin"]
        rows = list(collected[asin])[: config.maximum_reviews_per_product]
        product_id = _product_id(candidate)
        review_relative = (
            Path(config.output_directory) / "products" / product_id / "reviews.jsonl"
        ).as_posix()
        review_path = _resolve_project_relative(root, review_relative, "product reviews")
        review_payload = _jsonl_bytes(rows)
        _write_immutable(review_path, review_payload)
        child_asins = sorted({str(row["asin"]) for row in rows})
        generated = _generated_product_config(
            config, candidate, review_relative, child_asins
        )
        config_relative = (
            Path(config.generated_config_directory) / f"{product_id}.json"
        ).as_posix()
        generated_path = _resolve_project_relative(
            root, config_relative, "generated product config"
        )
        generated_payload = _canonical_json_bytes(generated, indent=2)
        _write_immutable(generated_path, generated_payload)
        timestamps = [
            row.get("timestamp") for row in rows
            if isinstance(row.get("timestamp"), (int, float))
        ]
        products.append(
            {
                "product_id": product_id,
                "parent_asin": asin,
                "display_name": candidate["display_name"],
                "brand": candidate["brand"],
                "operating_system": candidate["operating_system"],
                "segment": candidate["segment"],
                "condition": candidate["condition"],
                "metadata_title": candidate["title"],
                "metadata_rating_number": candidate["rating_number"],
                "collected_review_count": len(rows),
                "rating_distribution": _rating_distribution(rows),
                "timestamp_min": min(timestamps) if timestamps else None,
                "timestamp_max": max(timestamps) if timestamps else None,
                "child_asins": child_asins,
                "reviews_path": review_relative,
                "reviews_sha256": _sha256_bytes(review_payload),
                "config_path": config_relative,
                "config_sha256": _sha256_bytes(generated_payload),
            }
        )

    plan_payload = _canonical_json_bytes(plan, indent=2)
    manifest = {
        "schema_version": CATALOG_MANIFEST_SCHEMA_VERSION,
        "catalog_id": config.catalog_id,
        "status": "frozen_transfer_demo_catalog",
        "inputs": {
            "build_config_path": config.config_path,
            "build_config_sha256": config.config_sha256,
            "candidate_plan_path": (
                Path(config.output_directory) / "candidate_plan.json"
            ).as_posix(),
            "candidate_plan_sha256": _sha256_bytes(plan_payload),
            "review_repository": config.review_repository,
            "review_jsonl_path": config.review_jsonl_path,
            "source_revision": state.get("source_revision"),
        },
        "scan": {
            "scanned_records": state.get("scanned_records"),
            "target_matches": state.get("target_matches"),
            "usable_records_seen": state.get("usable_records"),
            "rejected_target_records": state.get("rejected_target_records"),
            "maximum_scan_records": config.maximum_scan_records,
            "minimum_text_characters": config.minimum_text_characters,
        },
        "selection": {
            "target_new_products": config.target_new_products,
            "minimum_reviews_per_product": config.minimum_reviews_per_product,
            "maximum_reviews_per_product": config.maximum_reviews_per_product,
            "maximum_products_per_brand": config.maximum_products_per_brand,
            "minimum_distinct_brands": config.minimum_distinct_brands,
            "required_operating_systems": list(config.required_operating_systems),
            "seed": config.selection_seed,
            "selected_product_ids": [row["product_id"] for row in products],
        },
        "products": products,
        "limitations": [
            "The candidate universe is the configured bounded metadata file, not all laptops.",
            "Reviews are a bounded stream-prefix convenience sample, not a random market sample.",
            (
                f"{config.minimum_reviews_per_product} to "
                f"{config.maximum_reviews_per_product} reviews support a course "
                "demo and issue discovery, not population claims."
            ),
            "Only Acer retains the project's formal benchmark status; new products are transfer demos.",
            "Source licensing was not established by this pipeline; do not claim commercial redistribution rights.",
        ],
    }
    manifest_path = output_dir / "catalog_manifest.json"
    manifest_payload = _canonical_json_bytes(manifest, indent=2)
    _write_immutable(manifest_path, manifest_payload)
    _publish_catalog_configs(root, config, manifest_path, manifest_payload)
    completed_state = dict(state)
    completed_state["status"] = "complete"
    completed_state["selected_parent_asins"] = [row["parent_asin"] for row in products]
    _write_checkpoint(output_dir / "collection_checkpoint.json", completed_state)
    return manifest


def _iter_remote_jsonl(
    config: CatalogBuildConfig,
    revision: str,
    *,
    start_after: int = 0,
    source_byte_offset: int = 0,
) -> Iterator[tuple[int, Mapping[str, Any], int]]:
    try:
        from huggingface_hub import HfFileSystem
    except ImportError as exc:
        raise CatalogBuildError(
            "huggingface_hub is required for laptop-catalog-expand"
        ) from exc
    fs = HfFileSystem()
    remote = (
        f"datasets/{config.review_repository}@{revision}/{config.review_jsonl_path}"
    )
    with fs.open(remote, "rb") as handle:
        line_number = 0
        if start_after and source_byte_offset:
            try:
                handle.seek(source_byte_offset)
                line_number = start_after
            except (OSError, ValueError):
                handle.seek(0)
        for raw_line in handle:
            line_number += 1
            if line_number > config.maximum_scan_records:
                break
            if line_number <= start_after:
                continue
            try:
                value = json.loads(raw_line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise CatalogBuildError(
                    f"Invalid remote JSON at record {line_number}"
                ) from exc
            if isinstance(value, dict):
                try:
                    byte_offset = int(handle.tell())
                except (OSError, ValueError):
                    byte_offset = 0
                yield line_number, value, byte_offset


def _dataset_revision(repository: str) -> str:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise CatalogBuildError(
            "huggingface_hub is required for laptop-catalog-expand"
        ) from exc
    revision = HfApi().dataset_info(repository).sha
    if not revision:
        raise CatalogBuildError("Could not resolve the dataset revision")
    return revision


def _verified_existing_manifest(
    root: Path, config: CatalogBuildConfig, plan: Mapping[str, Any]
) -> dict[str, Any] | None:
    output_dir = _resolve_project_relative(root, config.output_directory, "output directory")
    path = output_dir / "catalog_manifest.json"
    if not path.exists():
        return None
    try:
        payload = path.read_bytes()
        manifest = _require_object(json.loads(payload), "catalog manifest")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CatalogImmutableOutputError(f"Invalid frozen catalog manifest: {path}") from exc
    if payload != _canonical_json_bytes(manifest, indent=2):
        raise CatalogImmutableOutputError("Frozen catalog manifest is not canonical")
    if (
        manifest.get("schema_version") != CATALOG_MANIFEST_SCHEMA_VERSION
        or manifest.get("catalog_id") != config.catalog_id
        or manifest.get("status") != "frozen_transfer_demo_catalog"
    ):
        raise CatalogImmutableOutputError("Frozen catalog manifest contract is invalid")
    if manifest.get("inputs", {}).get("build_config_sha256") != config.config_sha256:
        raise CatalogImmutableOutputError("Frozen catalog uses a different build config")
    expected_plan_hash = _sha256_bytes(_canonical_json_bytes(plan, indent=2))
    if manifest.get("inputs", {}).get("candidate_plan_sha256") != expected_plan_hash:
        raise CatalogImmutableOutputError("Frozen catalog candidate plan changed")
    products = manifest.get("products")
    if not isinstance(products, list) or len(products) != config.target_new_products:
        raise CatalogImmutableOutputError("Frozen catalog has the wrong product count")
    product_ids = [product.get("product_id") for product in products if isinstance(product, dict)]
    parent_asins = [product.get("parent_asin") for product in products if isinstance(product, dict)]
    if (
        len(product_ids) != len(products)
        or len(set(product_ids)) != len(product_ids)
        or len(set(parent_asins)) != len(parent_asins)
        or manifest.get("selection", {}).get("selected_product_ids") != product_ids
    ):
        raise CatalogImmutableOutputError("Frozen catalog product identities are invalid")
    if len({product.get("brand") for product in products}) < config.minimum_distinct_brands:
        raise CatalogImmutableOutputError("Frozen catalog brand diversity is invalid")
    available_os = {product.get("operating_system") for product in products}
    if not set(config.required_operating_systems) <= available_os:
        raise CatalogImmutableOutputError("Frozen catalog OS coverage is invalid")

    expected_config_paths: set[Path] = set()
    for product in products:
        count = product.get("collected_review_count")
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or not config.minimum_reviews_per_product <= count <= config.maximum_reviews_per_product
        ):
            raise CatalogImmutableOutputError("Frozen catalog review count is invalid")
        review_path = _resolve_project_relative(root, product["reviews_path"], "reviews")
        product_config = _resolve_project_relative(root, product["config_path"], "config")
        expected_config_paths.add(product_config)
        if not review_path.is_file() or not product_config.is_file():
            raise CatalogImmutableOutputError("A frozen catalog artifact is missing")
        if _sha256_file(review_path) != product["reviews_sha256"]:
            raise CatalogImmutableOutputError(f"Frozen reviews changed: {review_path}")
        if _sha256_file(product_config) != product["config_sha256"]:
            raise CatalogImmutableOutputError(f"Generated config changed: {product_config}")
        review_rows = [line for line in review_path.read_text().splitlines() if line]
        if len(review_rows) != count:
            raise CatalogImmutableOutputError(f"Frozen review row count changed: {review_path}")
        try:
            from .laptop_products import load_laptop_product_config

            loaded = load_laptop_product_config(
                root, product_config.relative_to(root).as_posix()
            )
        except Exception as exc:
            raise CatalogImmutableOutputError(
                f"Generated product config is invalid: {product_config}"
            ) from exc
        if (
            loaded.product_id != product["product_id"]
            or loaded.parent_asin != product["parent_asin"]
            or loaded.source_reviews_jsonl != product["reviews_path"]
        ):
            raise CatalogImmutableOutputError("Generated config does not match manifest")
    config_dir = _resolve_project_relative(
        root, config.generated_config_directory, "generated config directory"
    )
    actual_config_paths = set(config_dir.glob("*.json")) if config_dir.is_dir() else set()
    if actual_config_paths != expected_config_paths:
        raise CatalogImmutableOutputError("Generated config directory is incomplete or has extras")
    _publish_catalog_configs(root, config, path, payload)
    return manifest


def run_laptop_catalog_expansion(
    project_root: str | Path, catalog_config: str | Path
) -> dict[str, Any]:
    """Plan, collect, and freeze a new laptop catalog without loading models."""

    root = _project_root(project_root)
    config = load_catalog_build_config(root, catalog_config)
    plan = build_candidate_plan(root, config)
    output_dir = _resolve_project_relative(root, config.output_directory, "output directory")
    plan_path = output_dir / "candidate_plan.json"
    plan_payload = _canonical_json_bytes(plan, indent=2)
    _write_immutable(plan_path, plan_payload)
    print("\n=== LAPTOP CATALOG EXPANSION (DATA ONLY) ===")
    print(
        f"Candidate metadata: {plan['metadata_rows_read']} rows; "
        f"high-confidence devices: {plan['high_confidence_candidates']}; "
        f"scan pool: {len(plan['candidate_pool'])}"
    )
    for row in plan["candidate_pool"]:
        print(
            f"  {row['parent_asin']} | {row['brand']} | {row['operating_system']} | "
            f"ratings_metadata={row['rating_number']} | {row['display_name']}"
        )
    existing = _verified_existing_manifest(root, config, plan)
    if existing is not None:
        print("Catalog is already frozen and all hashes were verified.")
        print("No models were loaded or called.")
        return existing

    checkpoint_path = output_dir / "collection_checkpoint.json"
    state: dict[str, Any] | None = None
    if checkpoint_path.exists():
        try:
            state = _require_object(
                json.loads(checkpoint_path.read_text()), "collection checkpoint"
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise CatalogBuildError("Collection checkpoint is invalid") from exc
        revision = _require_string(state.get("source_revision"), "source_revision")
        print(f"Resuming after {int(state.get('scanned_records', 0)):,} records.")
    else:
        revision = _dataset_revision(config.review_repository)
        print(f"Pinned dataset revision: {revision}")
    print(
        f"Collecting until {config.target_new_products} products reach at least "
        f"{config.minimum_reviews_per_product} usable reviews "
        f"(hard scan cap {config.maximum_scan_records:,})."
    )

    def save_progress(current: Mapping[str, Any]) -> None:
        _write_checkpoint(checkpoint_path, current)
        counts = sorted(
            (len(rows), asin) for asin, rows in current.get("collected", {}).items()
        )
        qualified = sum(count >= config.minimum_reviews_per_product for count, _ in counts)
        print(
            f"Catalog scan: {int(current['scanned_records']):,} records; "
            f"qualified={qualified}; best_counts={[count for count, _ in counts[-8:]]}"
        )

    latest_state: dict[str, Any] = {"value": state}

    original_save_progress = save_progress

    def save_progress(current: Mapping[str, Any]) -> None:
        latest_state["value"] = current
        original_save_progress(current)

    try:
        resume_after = int(state.get("scanned_records", 0)) if state else 0
        resume_offset = int(state.get("source_byte_offset", 0)) if state else 0
        state = collect_from_records(
            config,
            plan,
            _iter_remote_jsonl(
                config,
                revision,
                start_after=resume_after,
                source_byte_offset=resume_offset,
            ),
            source_revision=revision,
            state=state,
            progress_callback=save_progress,
        )
    except BaseException:
        current = latest_state.get("value")
        if current is not None:
            _write_checkpoint(checkpoint_path, current)
        raise
    _write_checkpoint(checkpoint_path, state)
    if state.get("status") != "ready_to_freeze":
        counts = sorted(
            (
                (asin, len(rows))
                for asin, rows in state.get("collected", {}).items()
            ),
            key=lambda item: (-item[1], item[0]),
        )
        raise CatalogBuildError(
            "Reached the configured scan cap before a diverse catalog qualified. "
            f"Best actual counts: {counts[:12]}"
        )
    manifest = freeze_catalog(root, config, plan, state)
    print("\n=== LAPTOP CATALOG FROZEN ===")
    for product in manifest["products"]:
        print(
            f"  {product['product_id']}: {product['collected_review_count']} reviews | "
            f"{product['brand']} | {product['operating_system']}"
        )
    print(f"Manifest: {output_dir / 'catalog_manifest.json'}")
    print("No models were loaded or called.")
    return manifest


def promote_laptop_catalog(
    project_root: str | Path,
    source_catalog_config: str | Path,
    target_catalog_config: str | Path,
) -> dict[str, Any]:
    """Freeze a new catalog from a verified local catalog checkpoint.

    Promotion is deliberately data-only. It reuses review rows already
    retained in a completed source checkpoint, applies a new versioned
    selection contract, and writes into the target catalog's separate output
    directories. It never opens the remote review stream and never mutates
    the source catalog.
    """

    root = _project_root(project_root)
    source = load_catalog_build_config(root, source_catalog_config)
    target = load_catalog_build_config(root, target_catalog_config)
    if source.catalog_id == target.catalog_id:
        raise CatalogBuildError("source and target catalog_id must differ")
    if (
        source.output_directory == target.output_directory
        or source.generated_config_directory == target.generated_config_directory
    ):
        raise CatalogBuildError("promotion target directories must differ from source")
    if (
        source.candidate_metadata_jsonl != target.candidate_metadata_jsonl
        or source.review_repository != target.review_repository
        or source.review_jsonl_path != target.review_jsonl_path
    ):
        raise CatalogBuildError("promotion must retain the same pinned review source")
    if target.maximum_reviews_per_product > source.maximum_reviews_per_product:
        raise CatalogBuildError(
            "promotion cannot recover reviews beyond the source checkpoint cap"
        )
    if target.minimum_text_characters < source.minimum_text_characters:
        raise CatalogBuildError(
            "promotion cannot relax the source checkpoint text-quality threshold"
        )

    source_plan = build_candidate_plan(root, source)
    source_manifest = _verified_existing_manifest(root, source, source_plan)
    if source_manifest is None:
        raise CatalogBuildError("source catalog must be frozen before promotion")
    source_output = _resolve_project_relative(
        root, source.output_directory, "source output directory"
    )
    source_checkpoint_path = source_output / "collection_checkpoint.json"
    source_plan_path = source_output / "candidate_plan.json"
    if not source_checkpoint_path.is_file() or not source_plan_path.is_file():
        raise CatalogBuildError("source catalog plan or checkpoint is missing")
    try:
        stored_source_plan = _require_object(
            json.loads(source_plan_path.read_text(encoding="utf-8")),
            "source candidate plan",
        )
        source_state = _require_object(
            json.loads(source_checkpoint_path.read_text(encoding="utf-8")),
            "source collection checkpoint",
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CatalogBuildError("source catalog plan or checkpoint is invalid") from exc
    if stored_source_plan != source_plan:
        raise CatalogImmutableOutputError("source candidate plan differs from config")
    source_plan_hash = _sha256_bytes(_canonical_json_bytes(source_plan, indent=2))
    source_revision = _require_string(
        source_state.get("source_revision"), "source checkpoint revision"
    )
    source_candidates = {
        row["parent_asin"]: row for row in source_plan["candidate_pool"]
    }
    _validate_checkpoint_state(
        source_state, source, source_plan_hash, source_revision, source_candidates
    )
    if source_state.get("status") != "complete":
        raise CatalogBuildError("source checkpoint must be complete")

    target_plan = build_candidate_plan(root, target)
    target_output = _resolve_project_relative(
        root, target.output_directory, "target output directory"
    )
    target_plan_payload = _canonical_json_bytes(target_plan, indent=2)
    _write_immutable(target_output / "candidate_plan.json", target_plan_payload)
    existing = _verified_existing_manifest(root, target, target_plan)
    if existing is not None:
        print("Promoted catalog is already frozen and all hashes were verified.")
        print("No remote data or models were loaded.")
        return existing

    target_asins = {
        str(row["parent_asin"]) for row in target_plan["candidate_pool"]
    }
    collected = {
        str(asin): list(rows)[: target.maximum_reviews_per_product]
        for asin, rows in source_state.get("collected", {}).items()
        if asin in target_asins
    }
    chosen = _selected_products(target, target_plan["candidate_pool"], collected)
    if not chosen:
        counts = sorted(
            ((asin, len(rows)) for asin, rows in collected.items()),
            key=lambda item: (-item[1], item[0]),
        )
        raise CatalogBuildError(
            "source checkpoint cannot satisfy the target selection contract; "
            f"best counts: {counts[:16]}"
        )
    target_plan_hash = _sha256_bytes(target_plan_payload)
    target_state = {
        "schema_version": CATALOG_CHECKPOINT_SCHEMA_VERSION,
        "catalog_id": target.catalog_id,
        "config_sha256": target.config_sha256,
        "plan_sha256": target_plan_hash,
        "source_revision": source_revision,
        "status": "ready_to_freeze",
        "scanned_records": int(source_state.get("scanned_records", 0)),
        "source_byte_offset": int(source_state.get("source_byte_offset", 0)),
        "target_matches": int(source_state.get("target_matches", 0)),
        "usable_records": sum(len(rows) for rows in collected.values()),
        "rejected_target_records": int(
            source_state.get("rejected_target_records", 0)
        ),
        "collected": collected,
        "selected_parent_asins": [row["parent_asin"] for row in chosen],
    }
    target_candidates = {
        row["parent_asin"]: row for row in target_plan["candidate_pool"]
    }
    _validate_checkpoint_state(
        target_state, target, target_plan_hash, source_revision, target_candidates
    )
    manifest = freeze_catalog(root, target, target_plan, target_state)
    print("\n=== LAPTOP CATALOG PROMOTED (LOCAL DATA ONLY) ===")
    print(
        f"Source: {source.catalog_id}; target: {target.catalog_id}; "
        f"remote records read: 0"
    )
    for product in manifest["products"]:
        print(
            f"  {product['product_id']}: {product['collected_review_count']} reviews | "
            f"{product['brand']} | {product['operating_system']}"
        )
    print(f"Manifest: {target_output / 'catalog_manifest.json'}")
    print("No remote data or models were loaded.")
    return manifest


__all__ = [
    "CatalogBuildConfig", "CatalogBuildError", "CatalogImmutableOutputError",
    "build_candidate_plan", "classify_laptop_title", "collect_from_records",
    "freeze_catalog", "load_catalog_build_config", "promote_laptop_catalog",
    "run_laptop_catalog_expansion",
]
