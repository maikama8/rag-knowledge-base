"""Shared records for documents, hits, and cited answers."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PageContent:
    """Text from one PDF page, or one Markdown section when there is no page."""

    number: int | None
    text: str


@dataclass(frozen=True)
class LoadedDocument:
    """A file reduced to ordered pages or sections."""

    source: str
    pages: tuple[PageContent, ...]


@dataclass(frozen=True)
class DocumentChunk:
    """A slice of a source file that can be embedded and cited."""

    chunk_id: str
    document_id: str
    source: str
    text: str
    chunk_index: int
    page: int | None


@dataclass(frozen=True)
class ScoredChunk:
    """A retrieved chunk and its cosine similarity to the question."""

    chunk: DocumentChunk
    score: float


@dataclass(frozen=True)
class Citation:
    """A locator for one chunk: file name, page when the source has one, and chunk number."""

    source: str
    chunk_id: str
    chunk_index: int
    page: int | None
    score: float
    excerpt: str
    label: str


@dataclass(frozen=True)
class Claim:
    """One statement in an answer. Every claim carries at least one citation."""

    text: str
    citations: tuple[Citation, ...]


@dataclass(frozen=True)
class Answer:
    """A grounded answer. `refused` is true when nothing in the index supported a claim."""

    question: str
    answer: str
    claims: tuple[Claim, ...]
    citations: tuple[Citation, ...]
    provider: str
    refused: bool


@dataclass(frozen=True)
class IndexedDocument:
    """Summary of one file currently stored in the index."""

    document_id: str
    source: str
    chunks: int


@dataclass(frozen=True)
class IngestedDocument:
    """Result of indexing one upload."""

    document_id: str
    source: str
    chunks: int
    pages: int | None
