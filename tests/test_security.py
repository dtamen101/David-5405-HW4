import logging

import pytest
from mcp.server.fastmcp.exceptions import ToolError

from study_graph.security import (
    InvalidInputError, RateLimiter, RateLimitExceeded, RedactingFilter,
    SlidingWindowLimiter, TokenBucket, safe_endpoint,
)


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_token_bucket_allows_burst_then_blocks_then_refills():
    clock = FakeClock()
    bucket = TokenBucket(per_minute=6, clock=clock)  # 1 token / 10 s
    assert all(bucket.try_acquire() == 0 for _ in range(6))
    wait = bucket.try_acquire()
    assert wait == pytest.approx(10.0)
    clock.now = 10.0
    assert bucket.try_acquire() == 0


def test_sliding_window_hard_cap():
    clock = FakeClock()
    limiter = SlidingWindowLimiter(limit=2, window=60, clock=clock)
    assert limiter.try_acquire() == 0 and limiter.try_acquire() == 0
    assert limiter.try_acquire() == pytest.approx(60)
    clock.now = 61
    assert limiter.try_acquire() == 0


def test_rate_limiter_raises_user_safe_error():
    limiter = RateLimiter({"search": 1})
    limiter.check("search")
    with pytest.raises(RateLimitExceeded) as exc:
        limiter.check("search")
    assert "Try again in" in exc.value.public_message


def test_redacting_filter_removes_keys_from_messages_and_tracebacks():
    secret = "AIzaSyTESTTESTTESTTESTTESTTESTTEST123"
    f = RedactingFilter([secret])
    try:
        raise ValueError(f"bad url https://x.com/?key={secret}")
    except ValueError:
        import sys
        record = logging.LogRecord("t", logging.ERROR, __file__, 1, "key was %s", (secret,), sys.exc_info())
    f.filter(record)
    assert secret not in record.getMessage()
    assert secret not in record.exc_text


@pytest.mark.anyio
async def test_safe_endpoint_hides_unexpected_errors():
    @safe_endpoint("boom")
    def boom():
        raise RuntimeError("/home/david/secret/path.db is locked")

    with pytest.raises(ToolError) as exc:
        await boom()
    assert "/home/david" not in str(exc.value)
    assert "incident" in str(exc.value)


@pytest.mark.anyio
async def test_safe_endpoint_shows_public_message_only():
    @safe_endpoint("bad")
    def bad():
        raise InvalidInputError("Friendly message", detail="internal /abs/path detail")

    with pytest.raises(ToolError) as exc:
        await bad()
    assert str(exc.value) == "Friendly message"
