"""Strict read-only access to the frozen ReviewLens demo bundle."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping


BUNDLE_SCHEMA_VERSION = "reviewlens.demo_bundle.v1"
PRODUCT_SCHEMA_VERSION = "reviewlens.demo_product.v1"
PRODUCT_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")


class BundleValidationError(RuntimeError):
    """The demo bundle is missing, stale, or internally inconsistent."""


class ProductNotFoundError(KeyError):
    """A product ID is not present in the featured demo bundle."""


class FindingNotFoundError(KeyError):
    """A finding ID is not present for the selected product."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_object(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BundleValidationError(f"{context} must be a JSON object")
    return value


def _require_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BundleValidationError(f"{context} must be a non-empty string")
    return value.strip()


def _require_int(value: Any, context: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise BundleValidationError(f"{context} must be an integer >= {minimum}")
    return value


def _load_json(path: Path, context: str) -> tuple[dict[str, Any], bytes]:
    try:
        payload = path.read_bytes()
        document = _require_object(json.loads(payload), context)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BundleValidationError(f"Cannot read valid {context}: {path}") from exc
    return document, payload


def _safe_relative_path(root: Path, value: Any, context: str) -> Path:
    raw = _require_string(value, context)
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts or "\x00" in raw:
        raise BundleValidationError(f"{context} must be a safe relative path")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise BundleValidationError(f"{context} resolves outside the bundle") from exc
    return resolved


def _assert_no_full_review_fields(value: Any, context: str = "product") -> None:
    forbidden = {
        "raw_text", "clean_text", "raw_response", "selected_raw_response",
        "model_result_raw", "attempts", "full_review_text", "reviewer_id", "user_id",
    }
    if isinstance(value, dict):
        bad = forbidden.intersection(value)
        if bad:
            raise BundleValidationError(
                f"{context} exposes forbidden full-review/model fields: {sorted(bad)}"
            )
        for key, child in value.items():
            _assert_no_full_review_fields(child, f"{context}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_no_full_review_fields(child, f"{context}[{index}]")


class DemoBundleRepository:
    """Load a hash-bound bundle once and serve defensive copies."""

    def __init__(self, bundle_dir: str | Path):
        self.bundle_dir = Path(bundle_dir).expanduser().resolve()
        self.manifest_path = self.bundle_dir / "manifest.json"
        self.manifest: dict[str, Any] = {}
        self.manifest_sha256 = ""
        self._products: dict[str, dict[str, Any]] = {}
        self._finding_index: dict[str, dict[str, dict[str, Any]]] = {}
        self._load()

    def _load(self) -> None:
        if not self.bundle_dir.is_dir():
            raise BundleValidationError(
                f"Demo bundle directory does not exist: {self.bundle_dir}"
            )
        manifest, manifest_bytes = _load_json(self.manifest_path, "bundle manifest")
        if manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION:
            raise BundleValidationError("Unsupported demo bundle schema_version")
        bundle_version = _require_string(
            manifest.get("bundle_version"), "manifest.bundle_version"
        )
        methodology = _require_object(
            manifest.get("methodology"), "manifest.methodology"
        )
        if not methodology:
            raise BundleValidationError("manifest.methodology must not be empty")
        limitations = manifest.get("limitations")
        if (
            not isinstance(limitations, list)
            or not limitations
            or any(not isinstance(item, str) or not item.strip() for item in limitations)
        ):
            raise BundleValidationError("manifest.limitations must contain text")
        featured = manifest.get("featured_product_ids")
        if (
            not isinstance(featured, list)
            or not featured
            or any(
                not isinstance(item, str) or not PRODUCT_ID_PATTERN.fullmatch(item)
                for item in featured
            )
            or len(set(featured)) != len(featured)
        ):
            raise BundleValidationError("featured_product_ids are invalid")
        product_entries = manifest.get("products")
        if not isinstance(product_entries, list) or not product_entries:
            raise BundleValidationError("manifest.products must not be empty")

        products: dict[str, dict[str, Any]] = {}
        entry_ids: list[str] = []
        for index, raw_entry in enumerate(product_entries):
            entry = _require_object(raw_entry, f"manifest.products[{index}]")
            product_id = _require_string(
                entry.get("product_id"), f"manifest.products[{index}].product_id"
            )
            if not PRODUCT_ID_PATTERN.fullmatch(product_id) or product_id in products:
                raise BundleValidationError(f"Invalid or duplicate product_id: {product_id}")
            expected_hash = _require_string(
                entry.get("sha256"), f"manifest.products[{index}].sha256"
            )
            if not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
                raise BundleValidationError(f"Invalid product hash for {product_id}")
            path = _safe_relative_path(
                self.bundle_dir,
                entry.get("file"),
                f"manifest.products[{index}].file",
            )
            product, product_bytes = _load_json(path, f"product {product_id}")
            if _sha256(product_bytes) != expected_hash:
                raise BundleValidationError(f"Product hash mismatch: {product_id}")
            self._validate_product(product_id, product)
            products[product_id] = product
            entry_ids.append(product_id)
        if entry_ids != featured:
            raise BundleValidationError(
                "manifest.products order must exactly match featured_product_ids"
            )

        self.manifest = manifest
        self.manifest_sha256 = _sha256(manifest_bytes)
        self._products = products
        self._finding_index = {
            product_id: {
                finding["finding_id"]: finding for finding in product["findings"]
            }
            for product_id, product in products.items()
        }
        self.bundle_version = bundle_version

    @staticmethod
    def _validate_product(product_id: str, product: dict[str, Any]) -> None:
        if product.get("schema_version") != PRODUCT_SCHEMA_VERSION:
            raise BundleValidationError(f"Unsupported product schema: {product_id}")
        identity = _require_object(product.get("product"), f"{product_id}.product")
        if identity.get("product_id") != product_id:
            raise BundleValidationError(f"Product identity mismatch: {product_id}")
        for field in ("display_name", "brand", "operating_system", "role", "status"):
            _require_string(identity.get(field), f"{product_id}.product.{field}")
        if identity["role"] not in {"benchmark", "transfer_demo"}:
            raise BundleValidationError(f"Invalid product role: {product_id}")

        sample = _require_object(product.get("sample"), f"{product_id}.sample")
        reviews_analyzed = _require_int(
            sample.get("reviews_analyzed"), f"{product_id}.sample.reviews_analyzed", 1
        )
        rating_distribution = _require_object(
            sample.get("rating_distribution"),
            f"{product_id}.sample.rating_distribution",
        )
        if set(rating_distribution) != {"1", "2", "3", "4", "5"}:
            raise BundleValidationError(f"Incomplete rating distribution: {product_id}")
        rating_total = sum(
            _require_int(value, f"{product_id}.rating_distribution.{rating}")
            for rating, value in rating_distribution.items()
        )
        if rating_total != reviews_analyzed:
            raise BundleValidationError(f"Rating distribution total mismatch: {product_id}")

        _require_object(product.get("coverage"), f"{product_id}.coverage")
        findings = product.get("findings")
        if not isinstance(findings, list):
            raise BundleValidationError(f"{product_id}.findings must be a list")
        finding_ids: set[str] = set()
        for index, raw_finding in enumerate(findings):
            finding = _require_object(raw_finding, f"{product_id}.findings[{index}]")
            finding_id = _require_string(
                finding.get("finding_id"), f"{product_id}.findings[{index}].finding_id"
            )
            if finding_id in finding_ids:
                raise BundleValidationError(f"Duplicate finding_id: {finding_id}")
            finding_ids.add(finding_id)
            if finding.get("scope") != "current_product":
                raise BundleValidationError(
                    f"Product finding is outside current-product scope: {finding_id}"
                )
            for field in ("category", "issue_group", "sentiment"):
                _require_string(finding.get(field), f"{product_id}.{finding_id}.{field}")
            support = _require_int(
                finding.get("supporting_review_count"),
                f"{product_id}.{finding_id}.supporting_review_count",
                1,
            )
            if support > reviews_analyzed:
                raise BundleValidationError(f"Finding support exceeds sample: {finding_id}")
            rate = finding.get("supporting_review_rate")
            if isinstance(rate, bool) or not isinstance(rate, (int, float)):
                raise BundleValidationError(f"Invalid supporting rate: {finding_id}")
            expected_rate = round(support / reviews_analyzed, 6)
            if not math.isclose(float(rate), expected_rate, abs_tol=1e-9):
                raise BundleValidationError(f"Supporting rate mismatch: {finding_id}")
            evidence = finding.get("evidence")
            if not isinstance(evidence, list) or not evidence:
                raise BundleValidationError(f"Finding has no evidence: {finding_id}")
            evidence_reviews: set[str] = set()
            for evidence_index, raw_evidence in enumerate(evidence):
                item = _require_object(
                    raw_evidence,
                    f"{product_id}.{finding_id}.evidence[{evidence_index}]",
                )
                review_id = _require_string(
                    item.get("review_id"), f"{product_id}.{finding_id}.review_id"
                )
                quote = _require_string(
                    item.get("quote"), f"{product_id}.{finding_id}.quote"
                )
                if len(quote) > 1000:
                    raise BundleValidationError(f"Evidence quote is too long: {finding_id}")
                rating = item.get("rating")
                if (
                    isinstance(rating, bool)
                    or not isinstance(rating, (int, float))
                    or not 1 <= float(rating) <= 5
                ):
                    raise BundleValidationError(f"Invalid evidence rating: {finding_id}")
                if item.get("exact_source_verified") is not True:
                    raise BundleValidationError(f"Evidence is not source-verified: {finding_id}")
                evidence_reviews.add(review_id)
            if len(evidence_reviews) > support:
                raise BundleValidationError(f"Evidence exceeds supporting reviews: {finding_id}")

        signals = product.get("signals")
        if not isinstance(signals, list):
            raise BundleValidationError(f"{product_id}.signals must be a list")
        signal_ids: set[str] = set()
        for index, raw_signal in enumerate(signals):
            signal = _require_object(raw_signal, f"{product_id}.signals[{index}]")
            signal_id = _require_string(
                signal.get("signal_id"), f"{product_id}.signals[{index}].signal_id"
            )
            if signal_id in signal_ids:
                raise BundleValidationError(f"Duplicate signal_id: {signal_id}")
            signal_ids.add(signal_id)
            _require_string(signal.get("quote"), f"{product_id}.{signal_id}.quote")
            if signal.get("exact_source_verified") is not True:
                raise BundleValidationError(f"Signal is not source-verified: {signal_id}")

        _require_object(product.get("report"), f"{product_id}.report")
        limitations = product.get("limitations")
        if not isinstance(limitations, list) or any(
            not isinstance(item, str) or not item.strip() for item in limitations
        ):
            raise BundleValidationError(f"{product_id}.limitations must be text")
        _assert_no_full_review_fields(product, product_id)

    def _product(self, product_id: str) -> dict[str, Any]:
        try:
            return self._products[product_id]
        except KeyError as exc:
            raise ProductNotFoundError(product_id) from exc

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "service": "ReviewLens API",
            "api_version": "1.0.0",
            "bundle_version": self.bundle_version,
            "product_count": len(self._products),
            "bundle_manifest_sha256": self.manifest_sha256,
        }

    def products(self) -> dict[str, Any]:
        cards = []
        for product_id in self.manifest["featured_product_ids"]:
            document = self._products[product_id]
            identity = document["product"]
            sample = document["sample"]
            cards.append(
                {
                    **identity,
                    "reviews_analyzed": sample["reviews_analyzed"],
                    "available_review_count": sample.get("available_review_count"),
                }
            )
        return {"bundle_version": self.bundle_version, "products": cards}

    def report(self, product_id: str) -> dict[str, Any]:
        product = self._product(product_id)
        return copy.deepcopy(
            {
                "schema_version": product["schema_version"],
                "product": product["product"],
                "sample": product["sample"],
                "coverage": product["coverage"],
                "report": product["report"],
                "limitations": product["limitations"],
            }
        )

    def findings(self, product_id: str) -> dict[str, Any]:
        product = self._product(product_id)
        summaries = []
        for finding in product["findings"]:
            row = {key: value for key, value in finding.items() if key != "evidence"}
            row["evidence_count"] = len(finding["evidence"])
            summaries.append(row)
        return {"product_id": product_id, "findings": copy.deepcopy(summaries)}

    def evidence(self, product_id: str, finding_id: str) -> dict[str, Any]:
        product = self._product(product_id)
        try:
            finding = self._finding_index[product_id][finding_id]
        except KeyError as exc:
            raise FindingNotFoundError(finding_id) from exc
        return copy.deepcopy(
            {
                "product": product["product"],
                "finding": {
                    key: value for key, value in finding.items() if key != "evidence"
                },
                "evidence": finding["evidence"],
            }
        )

    def signals(self, product_id: str) -> dict[str, Any]:
        product = self._product(product_id)
        return {
            "product_id": product_id,
            "signals": copy.deepcopy(product["signals"]),
        }

    def compare(self, left_product_id: str, right_product_id: str) -> dict[str, Any]:
        if left_product_id == right_product_id:
            raise ValueError("Comparison requires two different products")
        left = self._product(left_product_id)
        right = self._product(right_product_id)
        left_size = left["sample"]["reviews_analyzed"]
        right_size = right["sample"]["reviews_analyzed"]
        if left_size != right_size:
            raise ValueError("Comparison requires equal analyzed sample sizes")

        def comparison_product(document: Mapping[str, Any]) -> dict[str, Any]:
            return {
                "product": document["product"],
                "coverage": document["coverage"],
                "findings": [
                    {
                        key: value
                        for key, value in finding.items()
                        if key != "evidence"
                    }
                    for finding in document["findings"]
                ],
            }

        return copy.deepcopy(
            {
                "comparison_type": "directional_equal_sample",
                "cohort_size_per_product": left_size,
                "products": [comparison_product(left), comparison_product(right)],
                "disclaimer": (
                    "Directional comparison of equal-sized frozen convenience samples; "
                    "not a market-wide ranking or population estimate."
                ),
            }
        )

    def methodology(self) -> dict[str, Any]:
        return copy.deepcopy(
            {
                "schema_version": BUNDLE_SCHEMA_VERSION,
                "bundle_version": self.bundle_version,
                "methodology": self.manifest["methodology"],
                "limitations": self.manifest["limitations"],
            }
        )
