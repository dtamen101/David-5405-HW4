"""Configuration and secret loading.

Secrets live in a ``.env`` file at the project root. We read it with
``dotenv_values`` (which returns a plain dict) instead of ``load_dotenv``
so the API key is never copied into the process-wide environment
variables, where child processes and crash reporters could see it.
All filesystem locations are built with ``pathlib.Path``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from dotenv import dotenv_values

# src/study_graph/config.py -> project root is two parents above the package.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"


def _as_int(value: str | None, default: int) -> int:
    try:
        return int(value) if value else default
    except ValueError:
        return default


def _resolve_dir(value: str | None, default: Path) -> Path:
    """Relative paths in .env are resolved against the project root."""
    path = Path(value).expanduser() if value else default
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


@dataclass(frozen=True)
class Settings:
    """Immutable runtime settings. Build one with :func:`load_settings`."""

    sources_dir: Path
    data_dir: Path
    log_dir: Path
    gemini_api_key: str | None = field(repr=False)  # repr=False: never print the key
    gemini_model: str = "gemini-embedding-2"
    embed_dim: int = 768
    # Rate limits (see security.py)
    search_per_minute: int = 20
    write_per_minute: int = 10
    ingest_per_minute: int = 3
    embed_api_calls_per_minute: int = 30
    # Resource limits
    max_file_mb: int = 25
    max_chunks_per_document: int = 300

    @property
    def db_path(self) -> Path:
        return self.data_dir / "study_graph.db"

    @property
    def log_file(self) -> Path:
        return self.log_dir / "server.log"

    @property
    def embedding_backend(self) -> str:
        return "gemini" if self.gemini_api_key else "local"


def load_settings(env_file: Path = ENV_FILE) -> Settings:
    env = dotenv_values(env_file) if env_file.exists() else {}
    settings = Settings(
        sources_dir=_resolve_dir(env.get("SOURCES_DIR"), PROJECT_ROOT / "sample_sources"),
        data_dir=_resolve_dir(env.get("DATA_DIR"), PROJECT_ROOT / "data"),
        log_dir=_resolve_dir(env.get("LOG_DIR"), PROJECT_ROOT / "logs"),
        gemini_api_key=(env.get("GEMINI_API_KEY") or "").strip() or None,
        gemini_model=env.get("GEMINI_EMBED_MODEL") or "gemini-embedding-2",
        search_per_minute=_as_int(env.get("RATE_SEARCH_PER_MIN"), 20),
        write_per_minute=_as_int(env.get("RATE_WRITE_PER_MIN"), 10),
        ingest_per_minute=_as_int(env.get("RATE_INGEST_PER_MIN"), 3),
        embed_api_calls_per_minute=_as_int(env.get("RATE_EMBED_API_PER_MIN"), 30),
        max_file_mb=_as_int(env.get("MAX_FILE_MB"), 25),
    )
    for folder in (settings.sources_dir, settings.data_dir, settings.log_dir):
        folder.mkdir(parents=True, exist_ok=True)
    return settings
