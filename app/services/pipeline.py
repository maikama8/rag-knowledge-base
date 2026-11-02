"""Index documents and answer questions with citations."""

import threading

from app.services.chunking import chunk_pages
from app.services.documents.loaders import document_id_for, load_bytes, load_text
from app.services.embeddings.base import EmbeddingProvider
from app.services.errors import EmptyDocumentError
from app.services.llm.base import AnswerProvider
from app.services.models import (
    Answer,
    DocumentChunk,
    IndexedDocument,
    IngestedDocument,
    LoadedDocument,
)
from app.services.vectorstore.base import VectorStore


class KnowledgeBase:
    """The facade used by the API and the CLI."""

    def __init__(
        self,
        *,
        embeddings: EmbeddingProvider,
        store: VectorStore,
        answerer: AnswerProvider,
        chunk_size: int,
        chunk_overlap: int,
        top_k: int,
        min_score: float,
    ) -> None:
        self.embeddings = embeddings
        self.store = store
        self.answerer = answerer
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._top_k = top_k
        self._min_score = min_score
        self._lock = threading.Lock()

    def ingest_files(self, files: list[tuple[str | None, bytes]]) -> list[IngestedDocument]:
        """Load every file first, then embed and store. A bad file stores nothing new."""
        prepared = [self._prepare(load_bytes(name, data)) for name, data in files]
        return self._commit(prepared)

    def ingest_text(self, filename: str, text: str) -> IngestedDocument:
        """Index a raw string. The file name's suffix selects Markdown or plain text."""
        prepared = [self._prepare(load_text(filename, text))]
        return self._commit(prepared)[0]

    def query(self, question: str, top_k: int | None = None) -> Answer:
        """Retrieve chunks above the score floor and ask the answer provider to cite them."""
        cleaned = question.strip()
        limit = self._top_k if top_k is None else top_k
        vector = self.embeddings.embed([cleaned])[0]
        hits = self.store.query(vector, limit)
        grounded = [hit for hit in hits if hit.score >= self._min_score]
        return self.answerer.answer(cleaned, grounded)

    def documents(self) -> list[IndexedDocument]:
        return self.store.list_documents()

    def delete(self, document_id: str) -> int:
        return self.store.delete_document(document_id)

    def _prepare(self, loaded: LoadedDocument) -> tuple[LoadedDocument, str, list[DocumentChunk]]:
        document_id = document_id_for(loaded.source)
        chunks = chunk_pages(
            loaded.pages,
            source=loaded.source,
            document_id=document_id,
            chunk_size=self._chunk_size,
            overlap=self._chunk_overlap,
        )
        if not chunks:
            raise EmptyDocumentError(f"{loaded.source} has no extractable text.")
        return loaded, document_id, chunks

    def _commit(
        self, prepared: list[tuple[LoadedDocument, str, list[DocumentChunk]]]
    ) -> list[IngestedDocument]:
        results: list[IngestedDocument] = []
        with self._lock:
            for loaded, document_id, chunks in prepared:
                vectors = self.embeddings.embed([chunk.text for chunk in chunks])
                self.store.upsert(chunks, vectors)
                page_numbers = {page.number for page in loaded.pages if page.number is not None}
                results.append(
                    IngestedDocument(
                        document_id=document_id,
                        source=loaded.source,
                        chunks=len(chunks),
                        pages=len(page_numbers) if page_numbers else None,
                    )
                )
        return results
