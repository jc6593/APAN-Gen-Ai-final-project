"""FastAPI entry point for the portable ReviewLens product demo."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .repository import (
    DemoBundleRepository,
    FindingNotFoundError,
    ProductNotFoundError,
)
from .schemas import CompareRequest, HealthResponse, MethodologyResponse, ProductsResponse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "static"


def get_repository(request: Request) -> DemoBundleRepository:
    """Return the immutable repository loaded during application startup."""
    return request.app.state.repository


RepositoryDependency = Annotated[DemoBundleRepository, Depends(get_repository)]


def _default_bundle_dir() -> Path:
    configured = os.environ.get("REVIEWLENS_BUNDLE_DIR")
    return (
        Path(configured).expanduser().resolve()
        if configured
        else PROJECT_ROOT / "demo_bundle_v2"
    )


def create_app(bundle_dir: str | Path | None = None) -> FastAPI:
    resolved_bundle = (
        Path(bundle_dir).expanduser().resolve() if bundle_dir else _default_bundle_dir()
    )

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.repository = DemoBundleRepository(resolved_bundle)
        yield

    application = FastAPI(
        title="ReviewLens API",
        version="1.0.0",
        description=(
            "Evidence-traceable Voice-of-Customer findings for featured laptop "
            "case studies. The API serves frozen, precomputed AI results and does "
            "not load a model at runtime."
        ),
        lifespan=lifespan,
    )
    application.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @application.get("/", include_in_schema=False)
    def home() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @application.get("/health", response_model=HealthResponse, tags=["system"])
    def health(repo: RepositoryDependency):
        return repo.health()

    @application.get("/api/products", response_model=ProductsResponse, tags=["products"])
    def products(repo: RepositoryDependency):
        return repo.products()

    def _not_found(exc: KeyError) -> HTTPException:
        return HTTPException(status_code=404, detail=str(exc.args[0]))

    @application.get("/api/products/{product_id}/report", tags=["products"])
    def product_report(product_id: str, repo: RepositoryDependency):
        try:
            return repo.report(product_id)
        except ProductNotFoundError as exc:
            raise _not_found(exc) from exc

    @application.get("/api/products/{product_id}/findings", tags=["findings"])
    def product_findings(product_id: str, repo: RepositoryDependency):
        try:
            return repo.findings(product_id)
        except ProductNotFoundError as exc:
            raise _not_found(exc) from exc

    @application.get(
        "/api/products/{product_id}/findings/{finding_id}/evidence",
        tags=["evidence"],
    )
    def finding_evidence(
        product_id: str,
        finding_id: str,
        repo: RepositoryDependency,
        cohort_size: int | None = Query(default=None, ge=1),
    ):
        try:
            return repo.evidence(product_id, finding_id, cohort_size=cohort_size)
        except (ProductNotFoundError, FindingNotFoundError) as exc:
            raise _not_found(exc) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @application.get("/api/products/{product_id}/signals", tags=["findings"])
    def product_signals(product_id: str, repo: RepositoryDependency):
        try:
            return repo.signals(product_id)
        except ProductNotFoundError as exc:
            raise _not_found(exc) from exc

    @application.post("/api/compare", tags=["comparison"])
    def compare(payload: CompareRequest, repo: RepositoryDependency):
        try:
            return repo.compare(payload.left_product_id, payload.right_product_id)
        except ProductNotFoundError as exc:
            raise _not_found(exc) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @application.get(
        "/api/methodology", response_model=MethodologyResponse, tags=["methodology"]
    )
    def methodology(repo: RepositoryDependency):
        return repo.methodology()

    return application


app = create_app()
