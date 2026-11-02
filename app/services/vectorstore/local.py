"""A local cosine store backed by NumPy.

Vectors and chunk text are written under `DATA_DIR` as `vectors.npy` and
`index.json`. One process should own the directory. The on-disk index records
which embedding provider built it and refuses to load a mismatch.
"""

import json
import threading
from pathlib import Path

import numpy as np

from app.services.errors import VectorStoreError
from app.services.models import DocumentChunk, IndexedDocument, ScoredChunk

_INDEX_VERSION = 1


class LocalVectorStore:
    """Brute-force cosine similarity over L2-normalized vectors."""

    name = "local"

    def __init__(self, directory: Path, *, embedding_provider: str, dimension: int) -> None:
        if dimension < 1:
            raise ValueError("dimension must be positive.")
        self._directory = Path(directory)
        self._provider = embedding_provider
        self._dimension = dimension
        self._lock = threading.Lock()
        self._chunks: list[DocumentChunk] = []
        self._vectors = np.zeros((0, dimension), dtype=np.float32)
        self._load()

    def upsert(self, chunks: list[DocumentChunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise VectorStoreError("Chunk and vector counts do not match.")
        if not chunks:
            return
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != self._dimension:
            raise VectorStoreError(
                f"Expected vectors of dimension {self._dimension}, got shape {matrix.shape}."
            )
        document_ids = {chunk.document_id for chunk in chunks}
        with self._lock:
            self._drop_documents(document_ids)
            self._chunks.extend(chunks)
            self._vectors = np.vstack([self._vectors, _normalize_rows(matrix)])
            self._save()

    def query(self, vector: list[float], top_k: int) -> list[ScoredChunk]:
        if top_k < 1:
            return []
        with self._lock:
            if self._vectors.shape[0] == 0:
                return []
            query_vec = np.asarray(vector, dtype=np.float32)
            if query_vec.shape != (self._dimension,):
                raise VectorStoreError(f"Expected a query vector of dimension {self._dimension}.")
            norm = float(np.linalg.norm(query_vec))
            if norm == 0.0:
                return []
            scores = self._vectors @ (query_vec / norm)
            count = min(top_k, int(scores.shape[0]))
            order = np.argsort(-scores, kind="mergesort")[:count]
            return [
                ScoredChunk(chunk=self._chunks[int(index)], score=float(scores[int(index)]))
                for index in order
            ]

    def delete_document(self, document_id: str) -> int:
        with self._lock:
            before = len(self._chunks)
            self._drop_documents({document_id})
            removed = before - len(self._chunks)
            if removed:
                self._save()
            return removed

    def list_documents(self) -> list[IndexedDocument]:
        with self._lock:
            counts: dict[str, IndexedDocument] = {}
            for chunk in self._chunks:
                current = counts.get(chunk.document_id)
                if current is None:
                    counts[chunk.document_id] = IndexedDocument(
                        document_id=chunk.document_id,
                        source=chunk.source,
                        chunks=1,
                    )
                else:
                    counts[chunk.document_id] = IndexedDocument(
                        document_id=current.document_id,
                        source=current.source,
                        chunks=current.chunks + 1,
                    )
            return list(counts.values())

    def count(self) -> int:
        with self._lock:
            return len(self._chunks)

    def _drop_documents(self, document_ids: set[str]) -> None:
        keep = [
            index
            for index, chunk in enumerate(self._chunks)
            if chunk.document_id not in document_ids
        ]
        self._chunks = [self._chunks[index] for index in keep]
        if keep:
            self._vectors = self._vectors[keep]
        else:
            self._vectors = np.zeros((0, self._dimension), dtype=np.float32)

    def _load(self) -> None:
        self._directory.mkdir(parents=True, exist_ok=True)
        meta_path = self._directory / "index.json"
        vector_path = self._directory / "vectors.npy"
        if not meta_path.exists() and not vector_path.exists():
            return
        if not meta_path.exists() or not vector_path.exists():
            raise VectorStoreError(
                "DATA_DIR has a partial index. Delete the data directory and ingest again."
            )
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
            matrix = np.load(vector_path, allow_pickle=False)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise VectorStoreError("DATA_DIR has an index that could not be read.") from exc
        if not isinstance(payload, dict) or payload.get("version") != _INDEX_VERSION:
            raise VectorStoreError("DATA_DIR has an index written by a different version.")
        provider_matches = payload.get("embedding_provider") == self._provider
        dimension_matches = payload.get("dimension") == self._dimension
        if not provider_matches or not dimension_matches:
            raise VectorStoreError(
                "The index in DATA_DIR was built with a different embedding provider or "
                "dimension. Delete the data directory, or point DATA_DIR at a new folder."
            )
        raw_chunks = payload.get("chunks")
        if not isinstance(raw_chunks, list):
            raise VectorStoreError("DATA_DIR has an index that could not be read.")
        if matrix.shape != (len(raw_chunks), self._dimension):
            raise VectorStoreError("DATA_DIR vectors do not match the chunk metadata.")
        self._chunks = [_chunk_from_json(item) for item in raw_chunks]
        self._vectors = np.asarray(matrix, dtype=np.float32)

    def _save(self) -> None:
        self._directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": _INDEX_VERSION,
            "embedding_provider": self._provider,
            "dimension": self._dimension,
            "chunks": [_chunk_to_json(chunk) for chunk in self._chunks],
        }
        meta_path = self._directory / "index.json"
        vector_path = self._directory / "vectors.npy"
        temporary_meta = self._directory / "index.json.tmp"
        temporary_vectors = self._directory / "vectors.tmp.npy"
        temporary_meta.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        np.save(temporary_vectors, self._vectors)
        temporary_vectors.replace(vector_path)
        temporary_meta.replace(meta_path)


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe = np.where(norms == 0.0, 1.0, norms)
    return (matrix / safe).astype(np.float32)


def _chunk_to_json(chunk: DocumentChunk) -> dict[str, object]:
    return {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "source": chunk.source,
        "text": chunk.text,
        "chunk_index": chunk.chunk_index,
        "page": chunk.page,
    }


def _chunk_from_json(item: object) -> DocumentChunk:
    if not isinstance(item, dict):
        raise VectorStoreError("DATA_DIR has an index that could not be read.")
    try:
        page = item["page"]
        return DocumentChunk(
            chunk_id=str(item["chunk_id"]),
            document_id=str(item["document_id"]),
            source=str(item["source"]),
            text=str(item["text"]),
            chunk_index=int(item["chunk_index"]),
            page=None if page is None else int(page),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise VectorStoreError("DATA_DIR has an index that could not be read.") from exc
