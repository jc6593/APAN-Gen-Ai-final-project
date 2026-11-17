"""Public request and response schemas for the ReviewLens demo API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    status: str
    service: str
    api_version: str
    bundle_version: str
    product_count: int
    bundle_manifest_sha256: str


class ProductCard(BaseModel):
    product_id: str
    display_name: str
    brand: str
    operating_system: str
    role: str
    status: str
    reviews_analyzed: int
    available_review_count: int | None = None


class ProductsResponse(BaseModel):
    bundle_version: str
    products: list[ProductCard]


class CompareRequest(BaseModel):
    left_product_id: str = Field(min_length=1, max_length=120)
    right_product_id: str = Field(min_length=1, max_length=120)

    @field_validator("left_product_id", "right_product_id")
    @classmethod
    def normalize_product_id(cls, value: str) -> str:
        return value.strip()


class MethodologyResponse(BaseModel):
    schema_version: str
    bundle_version: str
    methodology: dict[str, Any]
    limitations: list[str]
