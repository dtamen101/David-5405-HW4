"""SQLite persistence for notes, embeddings, concepts and graph links.

Every query uses ``?`` placeholders (parameter binding), so note text,
titles and search strings can never be interpreted as SQL.
Pattern from the Python docs: https://docs.python.org/3/library/sqlite3.html#sqlite3-placeholders
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS notes (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    title      TEXT NOT NULL,
    body       TEXT NOT NULL,
    course     TEXT NOT NULL DEFAULT '',
    source     TEXT NOT NULL DEFAULT '',      -- file name, or '' for hand-written notes
    location   TEXT NOT NULL DEFAULT '',      -- 'slide 3', 'page 2', ...
    tags       TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS embeddings (
    note_id INTEGER PRIMARY KEY REFERENCES notes(id) ON DELETE CASCADE,
    model   TEXT NOT NULL,
    vector  BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS concepts (
    note_id INTEGER NOT NULL REFERENCES notes(id) ON DELETE CASCADE,
    concept TEXT NOT NULL,
    PRIMARY KEY (note_id, concept)
);
CREATE TABLE IF NOT EXISTS links (
    src    INTEGER NOT NULL REFERENCES notes(id) ON DELETE CASCADE,
    dst    INTEGER NOT NULL REFERENCES notes(id) ON DELETE CASCADE,
    kind   TEXT NOT NULL,                     -- 'wikilink' | 'similar' | 'next'
    weight REAL NOT NULL DEFAULT 1.0,
    PRIMARY KEY (src, dst, kind)
);
CREATE INDEX IF NOT EXISTS idx_concepts_concept ON concepts(concept);
CREATE INDEX IF NOT EXISTS idx_notes_source ON notes(source);
"""


@dataclass(frozen=True)
class Note:
    id: int
    title: str
    body: str
    course: str
    source: str
    location: str
    tags: list[str]
    created_at: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Note":
        tags = [t for t in row["tags"].split(",") if t]
        return cls(row["id"], row["title"], row["body"], row["course"], row["source"],
                   row["location"], tags, row["created_at"])

    def citation(self) -> str:
        where = f"{self.source}, {self.location}" if self.source else "hand-written note"
        return f"[note {self.id}: {where}]"


class NoteStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        # One short-lived connection per operation: safe across worker threads.
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            with conn:  # commits on success, rolls back on exception
                yield conn
        finally:
            conn.close()

    # --- notes ---------------------------------------------------------------
    def insert_note(self, title: str, body: str, course: str = "", source: str = "",
                    location: str = "", tags: list[str] | None = None) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO notes (title, body, course, source, location, tags) VALUES (?, ?, ?, ?, ?, ?)",
                (title, body, course, source, location, ",".join(tags or [])),
            )
            return int(cur.lastrowid)

    def delete_source(self, source: str) -> int:
        with self._connect() as conn:
            return conn.execute("DELETE FROM notes WHERE source = ?", (source,)).rowcount

    def get_note(self, note_id: int) -> Note | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM notes WHERE id = ?", (note_id,)).fetchone()
        return Note.from_row(row) if row else None

    def get_notes(self, ids: list[int]) -> dict[int, Note]:
        if not ids:
            return {}
        marks = ",".join("?" * len(ids))  # only '?' characters are interpolated
        with self._connect() as conn:
            rows = conn.execute(f"SELECT * FROM notes WHERE id IN ({marks})", ids).fetchall()
        return {row["id"]: Note.from_row(row) for row in rows}

    def find_by_title(self, title: str) -> Note | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM notes WHERE lower(title) = lower(?) ORDER BY id LIMIT 1", (title,)
            ).fetchone()
        return Note.from_row(row) if row else None

    def note_ids(self, course: str | None = None) -> list[int]:
        with self._connect() as conn:
            if course:
                rows = conn.execute("SELECT id FROM notes WHERE course = ? COLLATE NOCASE", (course,))
            else:
                rows = conn.execute("SELECT id FROM notes")
            return [r["id"] for r in rows]

    def stats(self) -> dict:
        with self._connect() as conn:
            one = lambda sql: conn.execute(sql).fetchone()[0]  # noqa: E731
            return {
                "notes": one("SELECT COUNT(*) FROM notes"),
                "links": one("SELECT COUNT(*) FROM links"),
                "concepts": one("SELECT COUNT(DISTINCT concept) FROM concepts"),
                "courses": [r[0] for r in conn.execute(
                    "SELECT DISTINCT course FROM notes WHERE course != '' ORDER BY course")],
                "sources": [dict(r) for r in conn.execute(
                    "SELECT source, COUNT(*) AS notes FROM notes WHERE source != '' "
                    "GROUP BY source ORDER BY source")],
            }

    # --- embeddings ----------------------------------------------------------
    def save_embeddings(self, model: str, ids: list[int], matrix: np.ndarray) -> None:
        rows = [(i, model, matrix[k].astype(np.float32).tobytes()) for k, i in enumerate(ids)]
        with self._connect() as conn:
            conn.executemany("INSERT OR REPLACE INTO embeddings (note_id, model, vector) VALUES (?, ?, ?)", rows)

    def load_embeddings(self, model: str, ids: list[int] | None = None) -> tuple[list[int], np.ndarray]:
        with self._connect() as conn:
            rows = conn.execute("SELECT note_id, vector FROM embeddings WHERE model = ?", (model,)).fetchall()
        wanted = set(ids) if ids is not None else None
        rows = [r for r in rows if wanted is None or r["note_id"] in wanted]
        if not rows:
            return [], np.zeros((0, 0), dtype=np.float32)
        return [r["note_id"] for r in rows], np.stack([np.frombuffer(r["vector"], dtype=np.float32) for r in rows])

    def ids_missing_embedding(self, model: str) -> list[int]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT n.id FROM notes n LEFT JOIN embeddings e ON e.note_id = n.id AND e.model = ? "
                "WHERE e.note_id IS NULL", (model,)).fetchall()
        return [r["id"] for r in rows]

    # --- concepts & links ----------------------------------------------------
    def save_concepts(self, note_id: int, concepts: list[str]) -> None:
        with self._connect() as conn:
            conn.executemany("INSERT OR IGNORE INTO concepts (note_id, concept) VALUES (?, ?)",
                             [(note_id, c) for c in concepts])

    def concepts_for(self, note_id: int) -> list[str]:
        with self._connect() as conn:
            return [r[0] for r in conn.execute(
                "SELECT concept FROM concepts WHERE note_id = ? ORDER BY concept", (note_id,))]

    def notes_with_concept(self, concept: str, limit: int) -> list[int]:
        with self._connect() as conn:
            return [r[0] for r in conn.execute(
                "SELECT note_id FROM concepts WHERE concept = ? ORDER BY note_id LIMIT ?", (concept, limit))]

    def related_concepts(self, concept: str, limit: int) -> list[tuple[str, int]]:
        """Concepts that co-occur with ``concept`` in the same notes, most frequent first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT c2.concept, COUNT(*) AS n FROM concepts c1 "
                "JOIN concepts c2 ON c1.note_id = c2.note_id AND c2.concept != c1.concept "
                "WHERE c1.concept = ? GROUP BY c2.concept ORDER BY n DESC, c2.concept LIMIT ?",
                (concept, limit)).fetchall()
        return [(r[0], r[1]) for r in rows]

    def top_concepts(self, limit: int) -> list[tuple[str, int]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT concept, COUNT(*) AS n FROM concepts GROUP BY concept "
                "ORDER BY n DESC, concept LIMIT ?", (limit,)).fetchall()
        return [(r[0], r[1]) for r in rows]

    def add_link(self, src: int, dst: int, kind: str, weight: float = 1.0) -> None:
        if src == dst:
            return
        with self._connect() as conn:
            conn.execute("INSERT OR REPLACE INTO links (src, dst, kind, weight) VALUES (?, ?, ?, ?)",
                         (src, dst, kind, weight))

    def neighbors(self, note_id: int) -> list[tuple[int, str, float]]:
        """Links in both directions: (other_note_id, kind, weight)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT dst AS other, kind, weight FROM links WHERE src = ? "
                "UNION SELECT src AS other, kind, weight FROM links WHERE dst = ? "
                "ORDER BY weight DESC", (note_id, note_id)).fetchall()
        return [(r["other"], r["kind"], r["weight"]) for r in rows]
