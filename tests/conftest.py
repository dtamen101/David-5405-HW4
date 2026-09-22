"""Shared pytest fixtures.

The ``anyio_backend`` fixture is the documented way to run async tests with AnyIO's pytest
plugin: https://anyio.readthedocs.io/en/stable/testing.html
``tmp_path`` is pytest's built-in fixture: https://docs.pytest.org/en/stable/how-to/tmp_path.html
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from study_graph.config import Settings
from study_graph.knowledge import KnowledgeBase

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    sources = tmp_path / "sources"
    shutil.copytree(ROOT / "sample_sources", sources)
    (tmp_path / "data").mkdir()
    (tmp_path / "logs").mkdir()
    return Settings(sources_dir=sources, data_dir=tmp_path / "data", log_dir=tmp_path / "logs", gemini_api_key=None)


@pytest.fixture
def kb(settings: Settings) -> KnowledgeBase:
    return KnowledgeBase(settings)


@pytest.fixture
def loaded_kb(kb: KnowledgeBase) -> KnowledgeBase:
    for name in ("week2_lean_operations.pptx", "reading_theory_of_constraints.pdf", "reading_agile_vs_waterfall.md"):
        kb.ingest_document(name, "ENMGT 5900")
    return kb
