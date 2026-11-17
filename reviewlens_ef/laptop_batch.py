"""Two-stage precomputation for a versioned ReviewLens laptop catalog.

The batch keeps only one MLX model resident at a time: all Layer 1 caches are
completed first, then Layer 1 is released before one shared Layer 2 generator
finishes observations and normalization.  The existing single-product CLI
keeps its original lifecycle and behavior.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .laptop_pipeline import (
    DEFAULT_L1_ADAPTER,
    DEFAULT_L1_CHECKPOINT,
    _LaptopLayer1Generator,
    _local_huggingface_revision,
    _release_model,
    run_laptop_pilot,
)
from .pipeline import _MlxGenerator


DEMO_CONFIG_SCHEMA_VERSION = "reviewlens.featured_demo_config.v1"
DEMO_CONFIG_SCHEMA_VERSION_V2 = "reviewlens.featured_demo_config.v2"
MIN_FEATURED_PRODUCTS = 2
MAX_FEATURED_PRODUCTS = 16


class FeaturedDemoConfigError(ValueError):
    """The featured demo contract is missing, unsafe, or internally inconsistent."""


@dataclass(frozen=True)
class FeaturedProduct:
    config_path: str
    brand: str
    operating_system: str
    evaluation_status: str
    cohort_status: str
    presentation_name: str | None = None
    reviews_to_analyze: int = 30


@dataclass(frozen=True)
class FeaturedDemoConfig:
    demo_id: str
    reviews_per_product: int
    l1_model_id: str
    l1_revision: str
    l2_model_id: str
    l2_revision: str
    products: tuple[FeaturedProduct, ...]
    max_retries: int
    bundle_output_directory: str
    config_path: str
    config_sha256: str
    comparison_reviews_per_product: int = 30


def _safe_project_path(root: Path, raw: str, context: str) -> Path:
    path = Path(raw)
    if not raw or path.is_absolute() or ".." in path.parts or "\x00" in raw:
        raise FeaturedDemoConfigError(f"{context} must be project-relative")
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise FeaturedDemoConfigError(f"{context} resolves outside project") from exc
    return resolved


def _exact_keys(value: Mapping[str, Any], expected: set[str], context: str) -> None:
    if set(value) != expected:
        raise FeaturedDemoConfigError(
            f"{context} keys mismatch; missing={sorted(expected-set(value))}, "
            f"unknown={sorted(set(value)-expected)}"
        )


def _text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FeaturedDemoConfigError(f"{context} must be a non-empty string")
    return value.strip()


def load_featured_demo_config(
    project_root: str | Path, config_path: str | Path
) -> FeaturedDemoConfig:
    root = Path(project_root).resolve()
    if not root.is_dir():
        raise FeaturedDemoConfigError(f"Project root does not exist: {root}")
    raw_path = str(config_path)
    path = _safe_project_path(root, raw_path, "demo config")
    try:
        payload = path.read_bytes()
        document = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FeaturedDemoConfigError(f"Cannot read demo config: {path}") from exc
    if not isinstance(document, dict):
        raise FeaturedDemoConfigError("Demo config must be an object")
    schema_version = document.get("schema_version")
    if schema_version == DEMO_CONFIG_SCHEMA_VERSION:
        _exact_keys(
            document,
            {
                "schema_version", "demo_id", "reviews_per_product", "models",
                "products", "max_retries", "bundle_output_directory",
            },
            "demo config",
        )
        limit_key = "reviews_per_product"
        comparison_limit = document.get("reviews_per_product")
    elif schema_version == DEMO_CONFIG_SCHEMA_VERSION_V2:
        _exact_keys(
            document,
            {
                "schema_version", "demo_id", "maximum_reviews_per_product",
                "comparison_reviews_per_product", "models", "products",
                "max_retries", "bundle_output_directory",
            },
            "demo config",
        )
        limit_key = "maximum_reviews_per_product"
        comparison_limit = document.get("comparison_reviews_per_product")
    else:
        raise FeaturedDemoConfigError("Unsupported demo config schema_version")
    demo_id = _text(document["demo_id"], "demo_id")
    if not re.fullmatch(r"[a-z0-9]+(?:_[a-z0-9]+)*", demo_id):
        raise FeaturedDemoConfigError("demo_id must use lower_snake_case")
    limit = document[limit_key]
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise FeaturedDemoConfigError(f"{limit_key} must be positive")
    if schema_version == DEMO_CONFIG_SCHEMA_VERSION and limit != 30:
        raise FeaturedDemoConfigError("The v1 featured comparison cohort must be 30")
    if (
        isinstance(comparison_limit, bool)
        or not isinstance(comparison_limit, int)
        or comparison_limit < 1
        or comparison_limit > limit
    ):
        raise FeaturedDemoConfigError(
            "comparison_reviews_per_product must be between 1 and the maximum"
        )
    retries = document["max_retries"]
    if retries not in {0, 1} or isinstance(retries, bool):
        raise FeaturedDemoConfigError("max_retries must be 0 or 1")

    models = document["models"]
    if not isinstance(models, dict):
        raise FeaturedDemoConfigError("models must be an object")
    _exact_keys(models, {"layer1", "layer2"}, "models")
    parsed_models: dict[str, tuple[str, str]] = {}
    for layer in ("layer1", "layer2"):
        value = models[layer]
        if not isinstance(value, dict):
            raise FeaturedDemoConfigError(f"models.{layer} must be an object")
        _exact_keys(value, {"model_id", "revision"}, f"models.{layer}")
        model_id = _text(value["model_id"], f"models.{layer}.model_id")
        revision = _text(value["revision"], f"models.{layer}.revision")
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise FeaturedDemoConfigError(f"models.{layer}.revision must be a commit SHA")
        parsed_models[layer] = (model_id, revision)

    product_values = document["products"]
    if not isinstance(product_values, list):
        raise FeaturedDemoConfigError("products must be a list")
    if schema_version == DEMO_CONFIG_SCHEMA_VERSION:
        if len(product_values) != 6:
            raise FeaturedDemoConfigError("Exactly six featured products are required")
    elif not MIN_FEATURED_PRODUCTS <= len(product_values) <= MAX_FEATURED_PRODUCTS:
        raise FeaturedDemoConfigError(
            f"v2 requires {MIN_FEATURED_PRODUCTS}-{MAX_FEATURED_PRODUCTS} products"
        )
    products: list[FeaturedProduct] = []
    seen_paths: set[str] = set()
    for index, value in enumerate(product_values):
        if not isinstance(value, dict):
            raise FeaturedDemoConfigError(f"products[{index}] must be an object")
        expected_product_keys = {
            "config_path", "brand", "operating_system",
            "evaluation_status", "cohort_status",
        }
        if schema_version == DEMO_CONFIG_SCHEMA_VERSION_V2:
            expected_product_keys |= {"presentation_name", "reviews_to_analyze"}
        _exact_keys(value, expected_product_keys, f"products[{index}]")
        product_path = _text(value["config_path"], f"products[{index}].config_path")
        if product_path in seen_paths:
            raise FeaturedDemoConfigError("Featured product config paths must be unique")
        seen_paths.add(product_path)
        if not _safe_project_path(root, product_path, "product config").is_file():
            raise FeaturedDemoConfigError(f"Product config does not exist: {product_path}")
        evaluation = _text(value["evaluation_status"], "evaluation_status")
        if evaluation not in {"benchmark_source_product", "transfer_demo"}:
            raise FeaturedDemoConfigError(f"Invalid evaluation_status: {evaluation}")
        reviews_to_analyze = value.get("reviews_to_analyze", limit)
        if (
            isinstance(reviews_to_analyze, bool)
            or not isinstance(reviews_to_analyze, int)
            or not comparison_limit <= reviews_to_analyze <= limit
        ):
            raise FeaturedDemoConfigError(
                f"products[{index}].reviews_to_analyze must be between "
                "the comparison cohort and the configured maximum"
            )
        presentation_name = value.get("presentation_name")
        if presentation_name is not None:
            presentation_name = _text(
                presentation_name, f"products[{index}].presentation_name"
            )
        products.append(
            FeaturedProduct(
                config_path=product_path,
                brand=_text(value["brand"], "brand"),
                operating_system=_text(value["operating_system"], "operating_system"),
                evaluation_status=evaluation,
                cohort_status=_text(value["cohort_status"], "cohort_status"),
                presentation_name=presentation_name,
                reviews_to_analyze=reviews_to_analyze,
            )
        )
    if sum(item.evaluation_status == "benchmark_source_product" for item in products) != 1:
        raise FeaturedDemoConfigError("Exactly one benchmark source product is required")
    output = _text(document["bundle_output_directory"], "bundle_output_directory")
    _safe_project_path(root, output, "bundle output directory")
    l1_id, l1_revision = parsed_models["layer1"]
    l2_id, l2_revision = parsed_models["layer2"]
    return FeaturedDemoConfig(
        demo_id=demo_id,
        reviews_per_product=limit,
        l1_model_id=l1_id,
        l1_revision=l1_revision,
        l2_model_id=l2_id,
        l2_revision=l2_revision,
        products=tuple(products),
        max_retries=retries,
        bundle_output_directory=output,
        config_path=Path(raw_path).as_posix(),
        config_sha256=hashlib.sha256(payload).hexdigest(),
        comparison_reviews_per_product=comparison_limit,
    )


class _LazyLayer1:
    def __init__(self, model_id: str, adapter: Path, checkpoint: Path) -> None:
        self.model_id = model_id
        self.adapter = adapter
        self.checkpoint = checkpoint
        self.instance: Any | None = None
        self.load_count = 0

    def __call__(self, system_prompt: str, sentence: str) -> str:
        if self.instance is None:
            print("Loading shared featured-catalog Layer 1:", self.model_id)
            self.instance = _LaptopLayer1Generator(
                self.model_id, self.adapter, self.checkpoint
            )
            self.load_count += 1
        return self.instance(system_prompt, sentence)

    def close(self) -> None:
        if self.instance is not None:
            value, self.instance = self.instance, None
            _release_model(value)


class _LazyLayer2:
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.instance: Any | None = None
        self.load_count = 0

    def __call__(self, system_prompt: str, payload: Mapping[str, Any], repair: bool) -> str:
        if self.instance is None:
            print("Loading shared featured-catalog Layer 2:", self.model_id)
            self.instance = _MlxGenerator(self.model_id)
            self.load_count += 1
        return self.instance(system_prompt, payload, repair)

    def close(self) -> None:
        if self.instance is not None:
            value, self.instance = self.instance, None
            _release_model(value)


def _verify_local_revisions(config: FeaturedDemoConfig) -> None:
    actual_l1 = _local_huggingface_revision(config.l1_model_id)
    actual_l2 = _local_huggingface_revision(config.l2_model_id)
    if actual_l1 != config.l1_revision:
        raise FeaturedDemoConfigError(
            f"Layer 1 local revision is {actual_l1!r}; expected {config.l1_revision}"
        )
    if actual_l2 != config.l2_revision:
        raise FeaturedDemoConfigError(
            f"Layer 2 local revision is {actual_l2!r}; expected {config.l2_revision}"
        )


def run_featured_precompute(
    project_root: str | Path,
    demo_config: str | Path,
    *,
    show: int = 0,
    build_bundle: bool = True,
) -> dict[str, Any]:
    """Precompute a catalog with one model load per layer, then freeze a bundle."""

    root = Path(project_root).resolve()
    config = load_featured_demo_config(root, demo_config)
    _verify_local_revisions(config)
    adapter = (root / DEFAULT_L1_ADAPTER).resolve()
    checkpoint = (root / DEFAULT_L1_CHECKPOINT).resolve()
    layer1 = _LazyLayer1(config.l1_model_id, adapter, checkpoint)
    layer1_results: list[dict[str, Any]] = []
    print("\n=== FEATURED LAPTOPS: LAYER 1 ===")
    try:
        for index, product in enumerate(config.products, 1):
            print(f"Featured Layer 1 product {index}/{len(config.products)}: {product.config_path}")
            layer1_results.append(
                run_laptop_pilot(
                    root,
                    product.config_path,
                    limit=product.reviews_to_analyze,
                    show=0,
                    l1_model_id=config.l1_model_id,
                    l2_model_id=config.l2_model_id,
                    l2_model_revision=config.l2_revision,
                    max_retries=config.max_retries,
                    layer1_generator=layer1,
                    stage="layer1",
                    manage_model_lifecycle=False,
                )
            )
    finally:
        layer1.close()

    layer2 = _LazyLayer2(config.l2_model_id)
    completed: list[dict[str, Any]] = []
    print("\n=== FEATURED LAPTOPS: EVIDENCE-FIRST LAYER 2 ===")
    try:
        for index, product in enumerate(config.products, 1):
            print(f"Featured Layer 2 product {index}/{len(config.products)}: {product.config_path}")
            completed.append(
                run_laptop_pilot(
                    root,
                    product.config_path,
                    limit=product.reviews_to_analyze,
                    show=show,
                    l1_model_id=config.l1_model_id,
                    l2_model_id=config.l2_model_id,
                    l2_model_revision=config.l2_revision,
                    max_retries=config.max_retries,
                    generator=layer2,
                    stage="layer2",
                    manage_model_lifecycle=False,
                )
            )
    finally:
        layer2.close()

    result: dict[str, Any] = {
        "demo_id": config.demo_id,
        "status": "featured_precompute_complete",
        "maximum_reviews_per_product": config.reviews_per_product,
        "comparison_reviews_per_product": config.comparison_reviews_per_product,
        "product_results": completed,
        "layer1_model_loads": layer1.load_count,
        "layer2_model_loads": layer2.load_count,
    }
    if build_bundle:
        from .demo_bundle import build_demo_bundle

        bundle = build_demo_bundle(
            project_root=root,
            demo_config=config,
            product_results=completed,
        )
        result["bundle"] = bundle
    print("\n=== FEATURED LAPTOP PRECOMPUTE COMPLETE ===")
    print("Products:", len(completed))
    print("Layer 1 model loads:", layer1.load_count)
    print("Layer 2 model loads:", layer2.load_count)
    if build_bundle:
        print("Verified demo bundle:", bundle.output_directory)
    return result


__all__ = [
    "DEMO_CONFIG_SCHEMA_VERSION", "DEMO_CONFIG_SCHEMA_VERSION_V2",
    "FeaturedDemoConfig", "FeaturedDemoConfigError", "FeaturedProduct",
    "load_featured_demo_config", "run_featured_precompute",
]
