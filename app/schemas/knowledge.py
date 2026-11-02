"""Request and response models for the knowledge-base API."""

from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4_000)
    top_k: int | None = Field(default=None, ge=1, le=20)

    @field_validator("question")
    @classmethod
    def _strip_question(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Question is empty.")
        return cleaned


class TextIngestRequest(BaseModel):
    filename: str = Field(min_length=1, max_length=255)
    text: str = Field(min_length=1, max_length=500_000)

    @field_validator("filename")
    @classmethod
    def _strip_filename(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Filename is empty.")
        return cleaned

    @field_validator("text")
    @classmethod
    def _strip_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Text is empty.")
        return cleaned


class CitationModel(BaseModel):
    source: str
    chunk_id: str
    chunk_index: int
    page: int | None = None
    score: float
    excerpt: str
    label: str


class ClaimModel(BaseModel):
    text: str
    citations: list[CitationModel]


class QueryResponse(BaseModel):
    question: str
    answer: str
    claims: list[ClaimModel]
    citations: list[CitationModel]
    provider: str
    refused: bool


class IngestedDocumentModel(BaseModel):
    document_id: str
    source: str
    chunks: int
    pages: int | None = None


class IngestResponse(BaseModel):
    documents: list[IngestedDocumentModel]
    embedding_provider: str
    vector_store: str


class IndexedDocumentModel(BaseModel):
    document_id: str
    source: str
    chunks: int


class DocumentListResponse(BaseModel):
    documents: list[IndexedDocumentModel]


class DeleteResponse(BaseModel):
    document_id: str
    chunks_removed: int


class ServiceInfo(BaseModel):
    name: str
    version: str
    status: str
    embedding_provider: str
    llm_provider: str
    vector_store: str
    docs: str
    ui: str


class HealthResponse(BaseModel):
    status: str
    embedding_provider: str
    llm_provider: str
    vector_store: str
    documents: int
    chunks: int
