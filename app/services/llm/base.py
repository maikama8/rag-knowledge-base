"""Answer provider interface."""

from typing import Protocol

from app.services.models import Answer, ScoredChunk


class AnswerProvider(Protocol):
    """Write an answer whose claims are grounded in retrieved chunks."""

    name: str

    def answer(self, question: str, chunks: list[ScoredChunk]) -> Answer:
        """Answer `question` from `chunks` only. Refuse when they do not support a claim."""
