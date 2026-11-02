"""Map pipeline failures onto HTTP responses."""

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.services.errors import (
    EmptyDocumentError,
    InvalidDocumentError,
    ProviderConfigError,
    ProviderError,
    UnsupportedDocumentError,
    VectorStoreError,
)


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ProviderConfigError)
    async def provider_config_error(_request, exc: ProviderConfigError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(ProviderError)
    async def provider_error(_request, exc: ProviderError) -> JSONResponse:
        return JSONResponse(status_code=502, content={"detail": str(exc)})

    @app.exception_handler(UnsupportedDocumentError)
    async def unsupported_document(_request, exc: UnsupportedDocumentError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(InvalidDocumentError)
    async def invalid_document(_request, exc: InvalidDocumentError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(EmptyDocumentError)
    async def empty_document(_request, exc: EmptyDocumentError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(VectorStoreError)
    async def vector_store_error(_request, exc: VectorStoreError) -> JSONResponse:
        return JSONResponse(status_code=500, content={"detail": str(exc)})
