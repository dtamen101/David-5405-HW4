"""Service layer: everything the MCP tools do, independent of MCP itself.

Keeping this separate from ``server.py`` means the logic can be unit tested
directly and the MCP layer stays a thin, declarative list of endpoints.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np

from .config import Settings
from .embeddings import Embedder, GeminiEmbedder, LocalEmbedder
from .extract import SUPPORTED_SUFFIXES, extract_chunks, resolve_source
from .graph import extract_concepts, find_wikilinks
from .security import InvalidInputError, NotFoundError, SlidingWindowLimiter
from .store import Note, NoteStore

logger = logging.getLogger("study_graph")

MAX_TITLE = 200
MAX_BODY = 20_000
MAX_QUERY = 500
MAX_NOTE_ID = 2**63 - 1  # largest integer SQLite can store
TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _\-]{0,39}$")
SIMILAR_LINKS_PER_NOTE = 3


def build_embedder(settings: Settings) -> Embedder:
    if settings.gemini_api_key:
        limiter = SlidingWindowLimiter(settings.embed_api_calls_per_minute)
        return GeminiEmbedder(settings.gemini_api_key, settings.gemini_model, settings.embed_dim, limiter)
    logger.warning("GEMINI_API_KEY not set - using the offline local embedder")
    return LocalEmbedder()


def _clean_text(value: str, field: str, max_len: int, required: bool = True) -> str:
    value = (value or "").strip()
    if required and not value:
        raise InvalidInputError(f"'{field}' must not be empty.")
    if len(value) > max_len:
        raise InvalidInputError(f"'{field}' is too long (max {max_len} characters).")
    return value


class KnowledgeBase:
    def __init__(self, settings: Settings, embedder: Embedder | None = None) -> None:
        self.settings = settings
        self.store = NoteStore(settings.db_path)
        self.embedder = embedder or build_embedder(settings)
        # Similarity scores differ by model: Gemini vectors are denser than hashed bags of words.
        self.min_similarity = 0.15 if isinstance(self.embedder, LocalEmbedder) else 0.6

    # ------------------------------------------------------------------ writes
    def ingest_document(self, filename: str, course: str = "") -> dict:
        course = _clean_text(course, "course", 60, required=False)
        path = resolve_source(self.settings.sources_dir, filename, self.settings.max_file_mb)
        chunks = extract_chunks(path)
        if not chunks:
            raise InvalidInputError("No readable text was found in that document (is it a scanned image?).")
        if len(chunks) > self.settings.max_chunks_per_document:
            raise InvalidInputError(
                f"Document has {len(chunks)} sections; the limit is {self.settings.max_chunks_per_document}.")

        relative = path.relative_to(self.settings.sources_dir.resolve()).as_posix()
        # Embed first: if the API fails, nothing has been written or deleted yet.
        vectors = self.embedder.embed_documents([(c.title, c.text) for c in chunks])
        replaced = self.store.delete_source(relative)

        ids = []
        for chunk in chunks:
            note_id = self.store.insert_note(chunk.title, chunk.text, course, relative, chunk.location)
            self.store.save_concepts(note_id, extract_concepts(chunk.title, chunk.text))
            ids.append(note_id)
        self.store.save_embeddings(self.embedder.model_name, ids, vectors)
        for prev, nxt in zip(ids, ids[1:]):
            self.store.add_link(prev, nxt, "next")
        links = sum(self._link_similar(i, vectors[k]) for k, i in enumerate(ids))

        return {
            "document": relative,
            "course": course,
            "notes_created": len(ids),
            "replaced_previous_notes": replaced,
            "similarity_links": links,
            "note_ids": ids,
            "sample_titles": [c.title for c in chunks[:5]],
        }

    def add_note(self, title: str, body: str, course: str = "", tags: list[str] | None = None) -> dict:
        title = _clean_text(title, "title", MAX_TITLE)
        body = _clean_text(body, "body", MAX_BODY)
        course = _clean_text(course, "course", 60, required=False)
        tags = [t.strip() for t in (tags or []) if t.strip()]
        if len(tags) > 10 or not all(TAG_RE.match(t) for t in tags):
            raise InvalidInputError("Tags must be at most 10 short words (letters, digits, spaces, - or _).")

        vector = self.embedder.embed_documents([(title, body)])[0]
        note_id = self.store.insert_note(title, body, course, tags=tags)
        self.store.save_embeddings(self.embedder.model_name, [note_id], vector[None, :])
        concepts = extract_concepts(title, body) + [t.lower() for t in tags]
        self.store.save_concepts(note_id, concepts)

        linked, unresolved = [], []
        for target_title in find_wikilinks(body):
            target = self.store.find_by_title(target_title)
            if target and target.id != note_id:
                self.store.add_link(note_id, target.id, "wikilink", 1.0)
                linked.append({"note_id": target.id, "title": target.title})
            else:
                unresolved.append(target_title)
        similar = self._link_similar(note_id, vector)
        return {
            "note_id": note_id,
            "concepts": concepts,
            "wikilinks": linked,
            "unresolved_wikilinks": unresolved,
            "similarity_links": similar,
        }

    def _link_similar(self, note_id: int, vector: np.ndarray) -> int:
        ids, matrix = self.store.load_embeddings(self.embedder.model_name)
        if not ids:
            return 0
        scores = matrix @ vector
        count = 0
        for idx in np.argsort(-scores)[: SIMILAR_LINKS_PER_NOTE + 1]:
            other = ids[idx]
            if other != note_id and scores[idx] >= self.min_similarity:
                self.store.add_link(note_id, other, "similar", round(float(scores[idx]), 3))
                count += 1
        return count

    # ------------------------------------------------------------------- reads
    def search(self, query: str, top_k: int = 5, course: str = "") -> dict:
        query = _clean_text(query, "query", MAX_QUERY)
        if not 1 <= top_k <= 20:
            raise InvalidInputError("top_k must be between 1 and 20.")
        self._backfill_embeddings()
        allowed = self.store.note_ids(course or None)
        ids, matrix = self.store.load_embeddings(self.embedder.model_name, allowed)
        if not ids:
            return {"query": query, "results": [], "hint": "No notes yet. Ingest a document or add a note first."}
        scores = matrix @ self.embedder.embed_query(query)
        order = np.argsort(-scores)[:top_k]
        notes = self.store.get_notes([ids[i] for i in order])
        results = []
        for i in order:
            note = notes[ids[i]]
            results.append({
                "note_id": note.id,
                "title": note.title,
                "citation": note.citation(),
                "score": round(float(scores[i]), 3),
                "snippet": note.body[:300] + ("..." if len(note.body) > 300 else ""),
                "concepts": self.store.concepts_for(note.id),
            })
        return {"query": query, "backend": self.embedder.model_name, "results": results}

    def _backfill_embeddings(self) -> None:
        """Embed notes created under a different backend (e.g. after adding an API key)."""
        missing = self.store.ids_missing_embedding(self.embedder.model_name)
        if not missing:
            return
        notes = self.store.get_notes(missing)
        vectors = self.embedder.embed_documents([(notes[i].title, notes[i].body) for i in missing])
        self.store.save_embeddings(self.embedder.model_name, missing, vectors)

    def explore(self, note_id: int | None = None, concept: str | None = None, limit: int = 10) -> dict:
        if (note_id is None) == (not concept):
            raise InvalidInputError("Provide exactly one of 'note_id' or 'concept'.")
        if not 1 <= limit <= 30:
            raise InvalidInputError("limit must be between 1 and 30.")
        if note_id is not None:
            return self._explore_note(note_id, limit)
        return self._explore_concept(_clean_text(concept or "", "concept", 80).lower(), limit)

    def _explore_note(self, note_id: int, limit: int) -> dict:
        note = self.get_note(note_id)
        edges = self.store.neighbors(note_id)[:limit]
        others = self.store.get_notes([e[0] for e in edges])
        concepts = self.store.concepts_for(note_id)
        shared = {}
        for concept in concepts:
            peers = [i for i in self.store.notes_with_concept(concept, limit + 1) if i != note_id]
            if peers:
                shared[concept] = peers[:limit]
        return {
            "note": {"note_id": note.id, "title": note.title, "citation": note.citation()},
            "concepts": concepts,
            "links": [
                {"note_id": other, "title": others[other].title, "relation": kind, "weight": weight}
                for other, kind, weight in edges if other in others
            ],
            "notes_sharing_concepts": shared,
        }

    def _explore_concept(self, concept: str, limit: int) -> dict:
        ids = self.store.notes_with_concept(concept, limit)
        if not ids:
            suggestions = [c for c, _ in self.store.top_concepts(200) if concept in c or c in concept][:5]
            raise NotFoundError(
                f"No notes are tagged with the concept '{concept}'."
                + (f" Similar concepts: {', '.join(suggestions)}." if suggestions else
                   " Try search_notes instead, or read notes://graph for the top concepts.")
            )
        notes = self.store.get_notes(ids)
        return {
            "concept": concept,
            "notes": [{"note_id": i, "title": notes[i].title, "citation": notes[i].citation()} for i in ids],
            "related_concepts": [{"concept": c, "shared_notes": n} for c, n in self.store.related_concepts(concept, limit)],
        }

    def get_note(self, note_id: int) -> Note:
        note = self.store.get_note(note_id)
        if note is None:
            raise NotFoundError(f"Note {note_id} does not exist.")
        return note

    @staticmethod
    def parse_note_id(raw: str) -> int:
        """Validate a note id taken from a resource URI (always a string)."""
        raw = (raw or "").strip()
        if not raw.isdigit() or not 1 <= int(raw) <= MAX_NOTE_ID:
            raise InvalidInputError("The note id must be a positive whole number, e.g. notes://note/12.")
        return int(raw)

    def note_markdown(self, note_id: int) -> str:
        note = self.get_note(note_id)
        meta = [f"Source: {note.citation()}"]
        if note.course:
            meta.append(f"Course: {note.course}")
        if note.tags:
            meta.append(f"Tags: {', '.join(note.tags)}")
        concepts = self.store.concepts_for(note_id)
        if concepts:
            meta.append(f"Concepts: {', '.join(concepts)}")
        return f"# {note.title}\n\n" + "\n".join(meta) + f"\n\n{note.body}\n"

    def list_sources(self) -> dict:
        root = self.settings.sources_dir.resolve()
        ingested = {s["source"]: s["notes"] for s in self.store.stats()["sources"]}
        files = []
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
                name = path.relative_to(root).as_posix()
                files.append({
                    "filename": name,
                    "size_kb": round(path.stat().st_size / 1024, 1),
                    "ingested_notes": ingested.get(name, 0),
                })
        # Deliberately no absolute path here: the client only needs relative names.
        return {"sources_folder": root.name, "files": files}

    def graph_overview(self) -> dict:
        stats = self.store.stats()
        stats["top_concepts"] = [{"concept": c, "notes": n} for c, n in self.store.top_concepts(25)]
        stats["embedding_backend"] = self.embedder.model_name
        return stats
