"""Text embeddings for semantic search.

Two interchangeable backends:

* ``GeminiEmbedder`` - calls Google's Gemini embedding REST API. Needs
  ``GEMINI_API_KEY`` in ``.env``. The key is sent in the ``x-goog-api-key``
  header (not the URL query string) so it never shows up in URLs, proxy
  logs or ``httpx`` exception messages.
  Request/response shapes follow https://ai.google.dev/gemini-api/docs/embeddings
  and the ``batchEmbedContents`` REST reference.
* ``LocalEmbedder`` - a dependency-free fallback (hashed bag of words +
  bigrams, the "hashing trick"). Lexical rather than truly semantic, but it
  lets the server work offline and in tests with no API key.
  The hashing trick is from Weinberger et al., "Feature Hashing for Large Scale
  Multitask Learning" (ICML 2009); see https://en.wikipedia.org/wiki/Feature_hashing
  (algorithm only, implementation written for this project).
- ``httpx.Client`` / ``raise_for_status`` usage: https://www.python-httpx.org/quickstart/
"""

from __future__ import annotations

import hashlib
import logging
import re
from abc import ABC, abstractmethod

import httpx
import numpy as np

from .security import RateLimitExceeded, SlidingWindowLimiter, UpstreamServiceError

logger = logging.getLogger("study_graph")

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_BATCH_SIZE = 50

_WORD = re.compile(r"[a-z][a-z0-9\-]{1,}")
STOPWORDS = frozenset(
    """a an and are as at be been but by can do does for from has have how i if in into is it its
    of on or our so than that the their them then there these they this to was we were what when
    where which while who why will with you your not no also may more most such use used using
    each other any all one two about over only both between through after before
    time unit example make makes made often usually much many like just want need""".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase words with stopwords removed and a light plural strip."""
    words = []
    for w in _WORD.findall(text.lower()):
        if w in STOPWORDS:
            continue
        if len(w) > 4 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        words.append(w)
    return words


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


class Embedder(ABC):
    model_name: str

    @abstractmethod
    def embed_documents(self, docs: list[tuple[str, str]]) -> np.ndarray:
        """Embed (title, text) pairs. Returns an L2-normalised (n, dim) matrix."""

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a search query. Returns an L2-normalised (dim,) vector."""


class LocalEmbedder(Embedder):
    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim
        self.model_name = f"local-hash-{dim}"

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = tokenize(text)
        features = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
        for feature in features:
            # blake2b instead of hash(): Python's hash() is randomised per process.
            digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[bucket] += sign
        return np.sign(vec) * np.log1p(np.abs(vec))  # sub-linear term frequency

    def embed_documents(self, docs: list[tuple[str, str]]) -> np.ndarray:
        if not docs:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _normalize(np.stack([self._vector(f"{title} {title} {text}") for title, text in docs]))

    def embed_query(self, query: str) -> np.ndarray:
        return _normalize(self._vector(query)[None, :])[0]


class GeminiEmbedder(Embedder):
    def __init__(self, api_key: str, model: str, dim: int, limiter: SlidingWindowLimiter,
                 client: httpx.Client | None = None) -> None:
        self.model_name = f"{model}-{dim}"
        self._model = model
        self._dim = dim
        self._limiter = limiter
        self._client = client or httpx.Client(
            base_url=GEMINI_BASE_URL,
            headers={"x-goog-api-key": api_key},
            timeout=httpx.Timeout(30.0),
        )

    def _request(self, texts: list[str]) -> np.ndarray:
        wait = self._limiter.try_acquire()
        if wait > 0:
            raise RateLimitExceeded("embedding API", wait)
        body = {
            "requests": [
                {
                    "model": f"models/{self._model}",
                    "content": {"parts": [{"text": t}]},
                    "outputDimensionality": self._dim,
                }
                for t in texts
            ]
        }
        try:
            response = self._client.post(f"/models/{self._model}:batchEmbedContents", json=body)
            response.raise_for_status()
            vectors = [item["values"] for item in response.json()["embeddings"]]
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            # Full details go to the server log only; the client gets a safe summary.
            logger.error("Gemini API error %s: %s", status, exc.response.text[:500])
            if status in (400, 401, 403):
                raise UpstreamServiceError(
                    "The embedding service rejected the server's credentials. "
                    "The server owner should check GEMINI_API_KEY in the .env file.",
                    detail=f"{exc} -> {exc.response.text[:300]}",
                ) from exc
            if status == 429:
                raise UpstreamServiceError("The embedding service quota is exhausted. Try again later.",
                                           detail=str(exc)) from exc
            raise UpstreamServiceError("The embedding service returned an error. Try again later.",
                                       detail=str(exc)) from exc
        except httpx.TransportError as exc:
            logger.error("Gemini API unreachable: %r", exc)
            raise UpstreamServiceError("The embedding service is unreachable. Check the internet connection.",
                                       detail=repr(exc)) from exc
        return _normalize(np.asarray(vectors, dtype=np.float32))

    def embed_documents(self, docs: list[tuple[str, str]]) -> np.ndarray:
        # Task-instruction formats quoted from the Gemini docs for gemini-embedding-2 retrieval:
        # https://ai.google.dev/gemini-api/docs/embeddings
        texts = [f"title: {title or 'none'} | text: {text}" for title, text in docs]
        parts = [self._request(texts[i:i + GEMINI_BATCH_SIZE]) for i in range(0, len(texts), GEMINI_BATCH_SIZE)]
        return np.vstack(parts) if parts else np.zeros((0, self._dim), dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return self._request([f"task: search result | query: {query}"])[0]
