"""Token and sentence helpers shared by the offline embedder and answerer."""

import re

# Question words are included so "how long is the refund window?" embeds as "refund window".
STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "did",
        "do",
        "does",
        "for",
        "from",
        "how",
        "in",
        "is",
        "it",
        "long",
        "many",
        "much",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
    }
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def content_tokens(text: str) -> list[str]:
    """Return lowercase word tokens with stopwords removed."""
    return [
        token
        for token in _TOKEN_RE.findall(text.lower())
        if token not in STOPWORDS and len(token) > 1
    ]


def overlap_score(question: str, sentence: str) -> float:
    """Share of the question's content words that also appear in `sentence`."""
    question_tokens = set(content_tokens(question))
    if not question_tokens:
        return 0.0
    sentence_tokens = set(content_tokens(sentence))
    return len(question_tokens & sentence_tokens) / len(question_tokens)


def split_sentences(text: str) -> list[str]:
    """Split prose into sentences. Markdown headings are not glued onto the next line."""
    sentences: list[str] = []
    for raw_line in re.split(r"\n+", text):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        flattened = re.sub(r"[ \t]+", " ", line)
        sentences.extend(part.strip() for part in _SENTENCE_RE.split(flattened) if part.strip())
    return sentences
