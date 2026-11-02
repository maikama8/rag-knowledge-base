"""OpenAI-compatible grounded answers.

Selected with LLM_PROVIDER=openai. The model must return JSON claims that name
chunk ids from the passages it was given. Claims that cite an unknown id, or
no id, are dropped. The displayed answer is then rebuilt so every remaining
claim has a citation marker.
"""

import json
import re

import httpx

from app.services.citations import build_answer, citation_from_chunk
from app.services.errors import ProviderError
from app.services.models import Answer, Claim, ScoredChunk

_SYSTEM_PROMPT = (
    "You answer questions using only the numbered passages. Reply with one JSON object "
    "with keys unknown (boolean) and claims (array of objects with text and chunk_ids). "
    "Every claim must be supported by one or more passages and list their chunk_ids. "
    "If the passages do not contain the answer, set unknown to true and claims to an empty array. "
    "Do not add a claim without a chunk_id from the passages. Do not invent facts."
)
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


class OpenAIAnswerProvider:
    """Ask a chat model for claims, then attach citations from the retrieved chunks."""

    name = "openai"

    def __init__(self, *, api_key: str, base_url: str, model: str, timeout: float) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout

    def answer(self, question: str, chunks: list[ScoredChunk]) -> Answer:
        if not chunks:
            return build_answer(question=question, claims=[], provider=self.name)
        claims = _claims_from_response(self._complete(question, chunks), chunks)
        return build_answer(question=question, claims=claims, provider=self.name)

    def _complete(self, question: str, chunks: list[ScoredChunk]) -> str:
        passages = []
        for hit in chunks:
            page = "none" if hit.chunk.page is None else str(hit.chunk.page)
            passages.append(
                f"chunk_id={hit.chunk.chunk_id} source={hit.chunk.source} page={page}\n"
                f"{hit.chunk.text}"
            )
        user = f"Question: {question}\n\nPassages:\n\n" + "\n\n".join(passages)
        body = {
            "model": self._model,
            "temperature": 0.0,
            "max_tokens": 1200,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
        }
        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.post(
                    f"{self._base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json=body,
                )
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:300]
            raise ProviderError(
                f"OpenAI answer request failed ({exc.response.status_code}): {detail}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError("OpenAI answer request failed.") from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise ProviderError("OpenAI answer request returned a non-JSON response.") from exc
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("OpenAI answer request returned an unexpected response.") from exc
        return _message_text(content)


def _message_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("text"):
                parts.append(str(item["text"]))
        return "".join(parts)
    raise ProviderError("OpenAI response did not include text content.")


def _claims_from_response(content: str, chunks: list[ScoredChunk]) -> list[Claim]:
    stripped = _FENCE_RE.sub("", content.strip())
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ProviderError("OpenAI answer request did not return valid JSON.") from exc
    if not isinstance(payload, dict):
        raise ProviderError("OpenAI answer request returned an unexpected JSON shape.")
    if payload.get("unknown") is True:
        return []

    raw_claims = payload.get("claims")
    if not isinstance(raw_claims, list):
        return []

    by_id = {hit.chunk.chunk_id: hit for hit in chunks}
    claims: list[Claim] = []
    for raw in raw_claims:
        if not isinstance(raw, dict):
            continue
        text = raw.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        chunk_ids = raw.get("chunk_ids")
        if not isinstance(chunk_ids, list):
            continue
        citations = []
        seen: set[str] = set()
        for chunk_id in chunk_ids:
            if not isinstance(chunk_id, str) or chunk_id in seen:
                continue
            hit = by_id.get(chunk_id)
            if hit is None:
                continue
            seen.add(chunk_id)
            citations.append(citation_from_chunk(hit.chunk, hit.score))
        if not citations:
            continue
        claims.append(Claim(text=text.strip(), citations=tuple(citations)))
    return claims
