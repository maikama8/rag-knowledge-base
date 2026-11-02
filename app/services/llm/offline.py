"""Extractive answerer used when no API key is configured.

Sentences are copied from retrieved chunks. A sentence is kept when at least
half of the question's content words appear in it, so the claim stays tied to
wording the user can find in the source. This is not a paraphrasing model.
"""

from app.services.citations import build_answer, citation_from_chunk
from app.services.models import Answer, Claim, ScoredChunk
from app.services.text import overlap_score, split_sentences


class OfflineAnswerProvider:
    """Build a cited answer by quoting overlapping sentences."""

    name = "offline"

    def __init__(self, *, max_claims: int = 3, minimum_overlap: float = 0.5) -> None:
        if max_claims < 1:
            raise ValueError("max_claims must be positive.")
        if not 0.0 <= minimum_overlap <= 1.0:
            raise ValueError("minimum_overlap must be between 0 and 1.")
        self._max_claims = max_claims
        self._minimum_overlap = minimum_overlap

    def answer(self, question: str, chunks: list[ScoredChunk]) -> Answer:
        candidates: list[tuple[float, float, str, ScoredChunk]] = []
        seen: set[str] = set()
        for hit in chunks:
            for sentence in split_sentences(hit.chunk.text):
                key = " ".join(sentence.lower().split())
                if key in seen:
                    continue
                score = overlap_score(question, sentence)
                if score < self._minimum_overlap:
                    continue
                seen.add(key)
                candidates.append((score, hit.score, sentence, hit))

        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        claims: list[Claim] = []
        for _overlap, _retrieval, sentence, hit in candidates[: self._max_claims]:
            claims.append(
                Claim(
                    text=sentence,
                    citations=(citation_from_chunk(hit.chunk, hit.score),),
                )
            )
        return build_answer(question=question, claims=claims, provider=self.name)
