"""Knowledge-graph helpers: concept extraction and wiki-style links.

Concepts are the most frequent meaningful words and two-word phrases in a
note (title words count double). Notes that share a concept are connected
through that concept; ``[[Note Title]]`` in a note body creates an explicit
edge, the same convention Obsidian and Roam use.
"""

from __future__ import annotations

import re
from collections import Counter

from .embeddings import tokenize

WIKILINK = re.compile(r"\[\[([^\[\]\n]{1,200})\]\]")


def find_wikilinks(body: str) -> list[str]:
    return list(dict.fromkeys(m.strip() for m in WIKILINK.findall(body) if m.strip()))


def extract_concepts(title: str, body: str, limit: int = 6) -> list[str]:
    title_tokens = tokenize(title)
    body_tokens = tokenize(body)
    counts: Counter[str] = Counter()
    for tokens, weight in ((title_tokens, 2), (body_tokens, 1)):
        for word in tokens:
            if len(word) >= 4 and not word.isdigit():
                counts[word] += weight
        for a, b in zip(tokens, tokens[1:]):
            if len(a) >= 3 and len(b) >= 3:
                counts[f"{a} {b}"] += weight * 1.5  # phrases are more specific than words
    # A phrase must appear at least twice (or once in the title) to count.
    ranked = [c for c, n in counts.most_common() if n >= 2]
    concepts: list[str] = []
    for concept in ranked:
        # Skip a single word if a chosen phrase already contains it.
        if " " not in concept and any(concept in chosen.split() for chosen in concepts):
            continue
        concepts.append(concept)
        if len(concepts) == limit:
            break
    return concepts
