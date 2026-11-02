"""Vector store interface."""

from typing import Protocol

from app.services.models import DocumentChunk, IndexedDocument, ScoredChunk


class VectorStore(Protocol):
    """Persist chunk vectors and return the nearest ones for a query."""

    name: str

    def upsert(self, chunks: list[DocumentChunk], vectors: list[list[float]]) -> None:
        """Insert chunks, replacing any already stored for the same document id."""

    def query(self, vector: list[float], top_k: int) -> list[ScoredChunk]:
        """Return up to `top_k` chunks, highest cosine similarity first."""

    def delete_document(self, document_id: str) -> int:
        """Remove one document. Return how many chunks were deleted."""

    def list_documents(self) -> list[IndexedDocument]:
        """List indexed files in the order they were first added."""

    def count(self) -> int:
        """Return the number of stored chunks."""
