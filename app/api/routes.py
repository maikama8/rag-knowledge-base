"""HTTP routes for ingest and cited question answering."""

import asyncio
import re
from dataclasses import asdict
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app import __version__
from app.api.deps import get_knowledge_base, get_settings
from app.config import Settings
from app.schemas.knowledge import (
    DeleteResponse,
    DocumentListResponse,
    HealthResponse,
    IndexedDocumentModel,
    IngestedDocumentModel,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    ServiceInfo,
    TextIngestRequest,
)
from app.services.pipeline import KnowledgeBase

router = APIRouter()
_STATIC_INDEX = Path(__file__).resolve().parents[1] / "static" / "index.html"
_DOCUMENT_ID = re.compile(r"^[a-f0-9]{16}$")


def _reject_oversize(data: bytes, settings: Settings) -> None:
    if not data:
        raise HTTPException(status_code=400, detail="Upload is empty.")
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Upload exceeds the {settings.max_upload_bytes} byte limit.",
        )


@router.get("/", response_model=ServiceInfo, tags=["Health"])
def root(settings: Settings = Depends(get_settings)) -> ServiceInfo:
    """Service name, active providers, and links to the docs and the browser UI."""
    return ServiceInfo(
        name=settings.app_name,
        version=__version__,
        status="ok",
        embedding_provider=settings.embedding_provider,
        llm_provider=settings.llm_provider,
        vector_store=settings.vector_store,
        docs="/docs",
        ui="/ui",
    )


@router.get("/health", response_model=HealthResponse, tags=["Health"])
def health(
    settings: Settings = Depends(get_settings),
    knowledge_base: KnowledgeBase = Depends(get_knowledge_base),
) -> HealthResponse:
    """Liveness probe. Reports providers and how much is indexed."""
    documents = knowledge_base.documents()
    return HealthResponse(
        status="ok",
        embedding_provider=settings.embedding_provider,
        llm_provider=settings.llm_provider,
        vector_store=settings.vector_store,
        documents=len(documents),
        chunks=sum(document.chunks for document in documents),
    )


@router.get("/ui", include_in_schema=False)
def browser_ui() -> FileResponse:
    """Single-page UI for upload and questions."""
    return FileResponse(_STATIC_INDEX)


@router.post("/ingest", response_model=IngestResponse, tags=["Documents"])
async def ingest(
    files: list[UploadFile] = File(..., description="PDF, Markdown, TXT, or DOCX files."),
    settings: Settings = Depends(get_settings),
    knowledge_base: KnowledgeBase = Depends(get_knowledge_base),
) -> IngestResponse:
    """Index one or more files. Uploading the same file name replaces that document."""
    payloads: list[tuple[str | None, bytes]] = []
    for upload in files:
        data = await upload.read()
        _reject_oversize(data, settings)
        payloads.append((upload.filename, data))
    ingested = await asyncio.to_thread(knowledge_base.ingest_files, payloads)
    return IngestResponse(
        documents=[IngestedDocumentModel.model_validate(asdict(item)) for item in ingested],
        embedding_provider=settings.embedding_provider,
        vector_store=settings.vector_store,
    )


@router.post("/documents", response_model=IngestResponse, tags=["Documents"])
async def ingest_text(
    body: TextIngestRequest,
    settings: Settings = Depends(get_settings),
    knowledge_base: KnowledgeBase = Depends(get_knowledge_base),
) -> IngestResponse:
    """Index a raw Markdown or text document without a multipart upload."""
    ingested = await asyncio.to_thread(knowledge_base.ingest_text, body.filename, body.text)
    return IngestResponse(
        documents=[IngestedDocumentModel.model_validate(asdict(ingested))],
        embedding_provider=settings.embedding_provider,
        vector_store=settings.vector_store,
    )


@router.get("/documents", response_model=DocumentListResponse, tags=["Documents"])
def list_documents(
    knowledge_base: KnowledgeBase = Depends(get_knowledge_base),
) -> DocumentListResponse:
    """List files currently in the index."""
    documents = knowledge_base.documents()
    return DocumentListResponse(
        documents=[IndexedDocumentModel.model_validate(asdict(item)) for item in documents]
    )


@router.delete("/documents/{document_id}", response_model=DeleteResponse, tags=["Documents"])
def delete_document(
    document_id: str,
    knowledge_base: KnowledgeBase = Depends(get_knowledge_base),
) -> DeleteResponse:
    """Remove one indexed file and its chunks."""
    if _DOCUMENT_ID.fullmatch(document_id) is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    removed = knowledge_base.delete(document_id)
    if removed == 0:
        raise HTTPException(status_code=404, detail="Document not found.")
    return DeleteResponse(document_id=document_id, chunks_removed=removed)


@router.post("/query", response_model=QueryResponse, tags=["Query"])
async def query(
    body: QueryRequest,
    knowledge_base: KnowledgeBase = Depends(get_knowledge_base),
) -> QueryResponse:
    """Answer a question. Every claim cites the source file and the chunk or page it came from."""
    result = await asyncio.to_thread(knowledge_base.query, body.question, body.top_k)
    return QueryResponse.model_validate(asdict(result))
