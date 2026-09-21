import httpx
import numpy as np
import pytest

from study_graph.embeddings import GeminiEmbedder
from study_graph.security import RateLimitExceeded, SlidingWindowLimiter, UpstreamServiceError

KEY = "AIzaSyUNITTESTUNITTESTUNITTEST12345"


def make(handler, limit=100):
    client = httpx.Client(transport=httpx.MockTransport(handler),
                          base_url="https://example.test", headers={"x-goog-api-key": KEY})
    return GeminiEmbedder(KEY, "gemini-embedding-2", 4, SlidingWindowLimiter(limit), client=client)


def test_success_parses_and_normalises_and_keeps_key_out_of_url():
    seen = {}

    def handler(request: httpx.Request):
        seen["url"] = str(request.url)
        seen["header"] = request.headers["x-goog-api-key"]
        n = len(__import__("json").loads(request.content)["requests"])
        return httpx.Response(200, json={"embeddings": [{"values": [3, 4, 0, 0]}] * n})

    vecs = make(handler).embed_documents([("t", "a"), ("t", "b")])
    assert vecs.shape == (2, 4)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0)
    assert KEY not in seen["url"] and seen["header"] == KEY


def test_bad_key_gives_safe_message():
    embedder = make(lambda r: httpx.Response(400, json={"error": {"message": "API key not valid"}}))
    with pytest.raises(UpstreamServiceError) as exc:
        embedder.embed_query("hi")
    assert KEY not in exc.value.public_message
    assert "googleapis" not in exc.value.public_message


def test_outbound_api_is_rate_limited():
    embedder = make(lambda r: httpx.Response(200, json={"embeddings": [{"values": [1, 0, 0, 0]}]}), limit=2)
    embedder.embed_query("a")
    embedder.embed_query("b")
    with pytest.raises(RateLimitExceeded):
        embedder.embed_query("c")
