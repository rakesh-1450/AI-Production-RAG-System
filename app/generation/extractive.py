"""Retrieval-only answer: picks the most relevant sentences from the retrieved chunks.

Used when no LLM key is configured (or every LLM call fails) so the app still works.
"""
import re

from app.generation.prompt_builder import NOT_FOUND
from app.utils.text import tokenize

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def extractive_answer(query: str, docs, max_sentences: int = 2) -> str:
    query_terms = set(tokenize(query, drop_stopwords=True))
    if not docs or not query_terms:
        return NOT_FOUND

    candidates = []
    for rank, doc in enumerate(docs):
        for sentence in _SENTENCE_SPLIT.split(doc["content"]):
            sentence = sentence.strip()
            if len(sentence) < 15:
                continue
            overlap = len(query_terms & set(tokenize(sentence)))
            if overlap:
                # earlier (better reranked) chunks win ties
                candidates.append((overlap / len(query_terms) - rank * 0.01, sentence))

    if not candidates:
        return NOT_FOUND

    candidates.sort(key=lambda item: item[0], reverse=True)
    best = candidates[0][0]
    picked, seen = [], set()
    for score, sentence in candidates:
        if score < best * 0.6:  # drop weakly related sentences
            break
        if sentence not in seen:
            seen.add(sentence)
            picked.append(sentence)
        if len(picked) == max_sentences:
            break
    return " ".join(picked)
