"""Best practices: rate limiting and safe error disclosure.

Two protections live here:

1. **Rate limiting** - a token bucket per tool category, plus a separate
   sliding-window limiter on outbound calls to the paid embedding API.
   An LLM stuck in a loop (or a prompt-injected document telling it to
   "search 500 times") cannot exhaust the API quota or hammer the disk.

2. **Error disclosure** - two layers. :class:`SafeFastMCP` catches
   argument-validation errors, which FastMCP raises *before* our code
   runs. Every tool and resource body is then wrapped by :func:`safe_endpoint`. Expected problems raise a :class:`StudyGraphError`
   whose ``public_message`` was written for the user. Anything else is
   logged in full (with secrets redacted) to ``logs/server.log`` and the
   client only receives a generic message plus an incident id. Without
   this, FastMCP returns ``str(exception)`` to the client, which can leak
   absolute file paths, SQL, library internals and even API keys.

The token-bucket algorithm follows the classic description at
https://en.wikipedia.org/wiki/Token_bucket (no code copied).
"""

from __future__ import annotations

import functools
import inspect
import logging
import re
import sys
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, TypeVar

import anyio
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ResourceError, ToolError
from pydantic import ValidationError

logger = logging.getLogger("study_graph")

F = TypeVar("F", bound=Callable[..., Any])


# ---------------------------------------------------------------------------
# User-safe exceptions
# ---------------------------------------------------------------------------
class StudyGraphError(Exception):
    """An expected error with two messages.

    ``public_message`` is written for the end user and is the only text the
    client ever sees. ``str(exc)`` (the ``detail``) may contain internals such
    as absolute paths or upstream API responses and only goes to the log.
    """

    def __init__(self, public_message: str, *, detail: str | None = None) -> None:
        super().__init__(detail or public_message)
        self.public_message = public_message


class InvalidInputError(StudyGraphError):
    pass


class NotFoundError(StudyGraphError):
    pass


class UpstreamServiceError(StudyGraphError):
    pass


class DocumentParseError(StudyGraphError):
    pass


class RateLimitExceeded(StudyGraphError):
    def __init__(self, bucket: str, retry_after: float) -> None:
        self.retry_after = retry_after
        super().__init__(
            f"Rate limit exceeded for '{bucket}' operations. "
            f"Try again in {retry_after:.0f} seconds."
        )


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------
class TokenBucket:
    """Allows short bursts up to ``capacity`` and refills at ``per_minute``/60 per second."""

    def __init__(self, per_minute: int, capacity: int | None = None, clock=time.monotonic) -> None:
        if per_minute <= 0:
            raise ValueError("per_minute must be positive")
        self.rate = per_minute / 60.0
        self.capacity = float(capacity or per_minute)
        self.tokens = self.capacity
        self._clock = clock
        self._updated = clock()
        self._lock = threading.Lock()

    def try_acquire(self, cost: float = 1.0) -> float:
        """Take ``cost`` tokens. Returns 0 on success, otherwise seconds to wait."""
        with self._lock:
            now = self._clock()
            self.tokens = min(self.capacity, self.tokens + (now - self._updated) * self.rate)
            self._updated = now
            if self.tokens >= cost:
                self.tokens -= cost
                return 0.0
            return (cost - self.tokens) / self.rate


class SlidingWindowLimiter:
    """Hard cap of ``limit`` events in any rolling ``window`` seconds (used for the paid API)."""

    def __init__(self, limit: int, window: float = 60.0, clock=time.monotonic) -> None:
        self.limit = limit
        self.window = window
        self._clock = clock
        self._events: deque[float] = deque()
        self._lock = threading.Lock()

    def try_acquire(self) -> float:
        with self._lock:
            now = self._clock()
            while self._events and now - self._events[0] >= self.window:
                self._events.popleft()
            if len(self._events) < self.limit:
                self._events.append(now)
                return 0.0
            return self.window - (now - self._events[0])


class RateLimiter:
    """Named token buckets, one per tool category (search / write / ingest)."""

    def __init__(self, limits: dict[str, int]) -> None:
        self._buckets = {name: TokenBucket(per_minute) for name, per_minute in limits.items()}

    def check(self, bucket: str) -> None:
        wait = self._buckets[bucket].try_acquire()
        if wait > 0:
            logger.warning("rate limit hit: bucket=%s retry_after=%.1fs", bucket, wait)
            raise RateLimitExceeded(bucket, wait)


# ---------------------------------------------------------------------------
# Secret redaction + logging
# ---------------------------------------------------------------------------
class RedactingFilter(logging.Filter):
    """Scrubs API keys from every log record, including tracebacks."""

    GOOGLE_KEY = re.compile(r"AIza[0-9A-Za-z_\-]{20,}")
    KEY_PARAM = re.compile(r"(?i)(key|token|api[_-]?key)=([^&\s'\"]+)")

    def __init__(self, secrets: list[str]) -> None:
        super().__init__()
        self._secrets = [s for s in secrets if s]

    def redact(self, text: str) -> str:
        for secret in self._secrets:
            text = text.replace(secret, "[REDACTED]")
        text = self.GOOGLE_KEY.sub("[REDACTED]", text)
        return self.KEY_PARAM.sub(r"\1=[REDACTED]", text)

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = self.redact(record.getMessage())
        record.args = None
        if record.exc_info:
            # Format the traceback now so we can redact it, then drop exc_info.
            record.exc_text = self.redact(logging.Formatter().formatException(record.exc_info))
            record.exc_info = None
        return True


def configure_logging(log_file: Path, secrets: list[str]) -> None:
    """Log to a rotating file and stderr. Never stdout: stdout carries the MCP protocol."""
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    redactor = RedactingFilter(secrets)
    file_handler = RotatingFileHandler(log_file, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    stderr_handler = logging.StreamHandler(sys.stderr)
    for handler in (file_handler, stderr_handler):
        handler.setFormatter(fmt)
        handler.addFilter(redactor)
        logger.addHandler(handler)
    logger.propagate = False


# ---------------------------------------------------------------------------
# The decorator every MCP endpoint goes through
# ---------------------------------------------------------------------------
def safe_endpoint(name: str, limiter: RateLimiter | None = None, bucket: str | None = None) -> Callable[[F], F]:
    """Wrap a tool/resource so it is rate limited and never leaks internals.

    The wrapper is ``async`` and runs the (blocking) body in a worker thread
    so SQLite and HTTP calls do not stall the MCP event loop.
    ``functools.wraps`` keeps the original signature, which FastMCP reads to
    build the tool's JSON schema.
    """

    def decorator(func: F) -> F:
        is_async = inspect.iscoroutinefunction(func)

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                if limiter and bucket:
                    limiter.check(bucket)
                if is_async:
                    return await func(*args, **kwargs)
                return await anyio.to_thread.run_sync(functools.partial(func, *args, **kwargs))
            except StudyGraphError as exc:
                logger.info("%s rejected: %s (%s)", name, exc.public_message, exc)
                raise ToolError(exc.public_message) from None
            except Exception:
                incident = uuid.uuid4().hex[:8]
                logger.exception("incident %s in %s", incident, name)
                raise ToolError(
                    f"An internal error occurred in '{name}'. "
                    f"The details were logged on the server (incident {incident})."
                ) from None

        return wrapper  # type: ignore[return-value]

    return decorator


# ---------------------------------------------------------------------------
# Argument-validation errors (raised by FastMCP before safe_endpoint runs)
# ---------------------------------------------------------------------------
def _exception_chain(exc: BaseException):
    """Yield exc and its causes, respecting ``raise ... from None``."""
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        yield exc
        exc = exc.__cause__ or (None if exc.__suppress_context__ else exc.__context__)


def _find(exc: BaseException, kind: type[BaseException]) -> BaseException | None:
    return next((e for e in _exception_chain(exc) if isinstance(e, kind)), None)


def describe_validation_error(error: ValidationError) -> str:
    """Plain-English summary that names the field but never echoes the input,
    the library, the internal model name or a documentation URL."""
    problems = []
    for item in error.errors()[:5]:
        field = ".".join(str(part) for part in item.get("loc", ())) or "arguments"
        kind = item.get("type", "")
        ctx = item.get("ctx") or {}
        if kind == "missing":
            problem = "is required"
        elif kind == "int_parsing_size":
            problem = "is too large"
        elif kind.startswith("int_") or kind.startswith("float_"):
            problem = "must be a whole number"
        elif kind.startswith("string_"):
            problem = "must be text"
        elif kind.startswith("list_"):
            problem = "must be a list"
        elif kind in ("less_than_equal", "less_than"):
            problem = f"must be at most {ctx.get('le', ctx.get('lt'))}"
        elif kind in ("greater_than_equal", "greater_than"):
            problem = f"must be at least {ctx.get('ge', ctx.get('gt'))}"
        else:
            problem = "has an invalid value"
        problems.append(f"'{field}' {problem}")
    return "; ".join(problems) + "."


class SafeFastMCP(FastMCP):
    """FastMCP with the error-disclosure policy applied at the protocol boundary.

    FastMCP validates tool arguments with pydantic *before* calling our
    function, and on failure sends the raw pydantic message to the client
    (library name, internal model name, the rejected input and a docs URL).
    These overrides replace such messages with a short, safe summary.
    Errors that already passed through :func:`safe_endpoint` are left alone.
    """

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        try:
            return await super().call_tool(name, arguments)
        except ToolError as exc:
            validation = _find(exc, ValidationError)
            if validation is None:
                raise
            logger.info("%s rejected arguments: %s", name, validation)
            raise ToolError(
                f"Invalid arguments for '{name}': {describe_validation_error(validation)}"
            ) from None

    async def read_resource(self, uri: Any) -> Any:
        try:
            return await super().read_resource(uri)
        except Exception as exc:
            validation = _find(exc, ValidationError)
            if validation is not None:
                logger.info("resource %s rejected: %s", uri, validation)
                raise ResourceError(f"Invalid resource address: {describe_validation_error(validation)}") from None
            public = _find(exc, ToolError)  # already sanitised by safe_endpoint
            if public is not None:
                raise ResourceError(str(public)) from None
            if str(exc).startswith("Unknown resource"):
                raise ResourceError(
                    "Unknown resource. Available: notes://sources, notes://graph, notes://note/{note_id}."
                ) from None
            incident = uuid.uuid4().hex[:8]
            logger.exception("incident %s reading resource %s", incident, uri)
            raise ResourceError(
                f"An internal error occurred while reading the resource (incident {incident})."
            ) from None
