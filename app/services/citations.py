"""Turn retrieved chunks into an answer where every claim has a citation."""

from app.services.errors import UncitedClaimError
from app.services.models import Answer, Citation, Claim, DocumentChunk

NOT_FOUND_ANSWER = "I could not find this in the knowledge base."
_EXCERPT_LIMIT = 240


def format_citation(source: str, *, page: int | None, chunk_index: int) -> str:
    """Human-readable locator. Chunk numbers shown to people start at 1."""
    chunk_label = f"chunk {chunk_index + 1}"
    if page is None:
        return f"{source}, {chunk_label}"
    return f"{source}, page {page}, {chunk_label}"


def citation_from_chunk(chunk: DocumentChunk, score: float) -> Citation:
    """Build a citation record from a stored chunk."""
    return Citation(
        source=chunk.source,
        chunk_id=chunk.chunk_id,
        chunk_index=chunk.chunk_index,
        page=chunk.page,
        score=round(float(score), 4),
        excerpt=_excerpt(chunk.text),
        label=format_citation(chunk.source, page=chunk.page, chunk_index=chunk.chunk_index),
    )


def build_answer(
    *,
    question: str,
    claims: list[Claim] | tuple[Claim, ...],
    provider: str,
) -> Answer:
    """Assemble `answer` from `claims` so the prose cannot outrun the citations.

    An empty claim list is a refusal: the text says the index did not contain an answer,
    and both `claims` and `citations` are empty. A claim with no citations is an error.
    """
    if not claims:
        return Answer(
            question=question,
            answer=NOT_FOUND_ANSWER,
            claims=(),
            citations=(),
            provider=provider,
            refused=True,
        )

    ordered: list[Citation] = []
    positions: dict[str, int] = {}
    lines: list[str] = []
    normalized: list[Claim] = []
    for claim in claims:
        text = " ".join(claim.text.split()).strip()
        if not text:
            raise UncitedClaimError("A claim was empty.")
        if not claim.citations:
            raise UncitedClaimError("Every claim needs a source citation.")
        markers: list[str] = []
        for citation in claim.citations:
            if citation.chunk_id not in positions:
                positions[citation.chunk_id] = len(ordered) + 1
                ordered.append(citation)
            markers.append(str(positions[citation.chunk_id]))
        sentence = text if text[-1] in ".!?" else f"{text}."
        lines.append(f"{sentence} {''.join(f'[{marker}]' for marker in markers)}")
        normalized.append(Claim(text=text, citations=tuple(claim.citations)))

    return Answer(
        question=question,
        answer=" ".join(lines),
        claims=tuple(normalized),
        citations=tuple(ordered),
        provider=provider,
        refused=False,
    )


def _excerpt(text: str, limit: int = _EXCERPT_LIMIT) -> str:
    kept: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        kept.append(line)
    compact = " ".join(kept) if kept else " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."
