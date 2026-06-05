from __future__ import annotations

from collections import Counter


def token_length(text: str) -> int:
    return len(text.split())


def repetition_rate(text: str, ngram_size: int = 3) -> float:
    tokens = text.lower().split()
    if len(tokens) < ngram_size:
        return 0.0
    ngrams = [
        tuple(tokens[i : i + ngram_size])
        for i in range(0, len(tokens) - ngram_size + 1)
    ]
    counts = Counter(ngrams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / max(1, len(ngrams))

