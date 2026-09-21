"""MCP endpoint definitions (the only module that knows about MCP).

Structure based on the FastMCP quickstart in the official MCP Python SDK
README: https://github.com/modelcontextprotocol/python-sdk#quickstart
(``FastMCP(...)``, ``@mcp.tool()``, ``@mcp.resource()``, ``@mcp.prompt()``, ``mcp.run()``).
"""

from __future__ import annotations

import json
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import Field

from .config import Settings, load_settings
from .knowledge import KnowledgeBase
from .security import RateLimiter, configure_logging, logger, safe_endpoint

INSTRUCTIONS = """\
Study Graph turns a student's lecture slides and readings into a searchable
knowledge graph. Typical flow: read notes://sources -> ingest_document ->
search_notes / explore_graph -> cite notes as [note N: file, slide M].
Always cite note ids when answering from the notes."""


def create_server(settings: Settings, kb: KnowledgeBase | None = None) -> FastMCP:
    kb = kb or KnowledgeBase(settings)
    limiter = RateLimiter({
        "search": settings.search_per_minute,
        "write": settings.write_per_minute,
        "ingest": settings.ingest_per_minute,
    })
    mcp = FastMCP("study-graph", instructions=INSTRUCTIONS)

    # ------------------------------------------------------------------ tools
    @mcp.tool(annotations=ToolAnnotations(title="Ingest slides or reading", destructiveHint=False,
                                          idempotentHint=True, openWorldHint=False))
    @safe_endpoint("ingest_document", limiter, "ingest")
    def ingest_document(
        filename: Annotated[str, Field(description="File name from notes://sources, e.g. 'week3_lecture.pptx'")],
        course: Annotated[str, Field(description="Optional course code to file the notes under, e.g. 'ENMGT 5900'")] = "",
    ) -> dict:
        """Import a slide deck (.pptx), reading (.pdf) or text/markdown file from the sources folder.

        Each slide, page or heading section becomes one note; notes are embedded for
        semantic search and linked into the knowledge graph. Re-ingesting the same file
        replaces its old notes.
        """
        return kb.ingest_document(filename, course)

    @mcp.tool(annotations=ToolAnnotations(title="Add a note", destructiveHint=False, openWorldHint=False))
    @safe_endpoint("add_note", limiter, "write")
    def add_note(
        title: Annotated[str, Field(description="Short, unique title for the note")],
        body: Annotated[str, Field(description="Note text in Markdown. Use [[Other Note Title]] to link notes.")],
        course: Annotated[str, Field(description="Optional course code")] = "",
        tags: Annotated[list[str] | None, Field(description="Optional tags, e.g. ['exam-2', 'definitions']")] = None,
    ) -> dict:
        """Save a hand-written note (e.g. your own summary or a class discussion).

        [[wikilinks]] to existing note titles become explicit graph edges.
        """
        return kb.add_note(title, body, course, tags)

    @mcp.tool(annotations=ToolAnnotations(title="Semantic search", readOnlyHint=True, openWorldHint=False))
    @safe_endpoint("search_notes", limiter, "search")
    def search_notes(
        query: Annotated[str, Field(description="Natural-language question or topic")],
        top_k: Annotated[int, Field(description="Number of results (1-20)", ge=1, le=20)] = 5,
        course: Annotated[str, Field(description="Optional course code to restrict the search")] = "",
    ) -> dict:
        """Find the notes most similar in meaning to a query, with citations back to the slide or page."""
        return kb.search(query, top_k, course)

    @mcp.tool(annotations=ToolAnnotations(title="Explore knowledge graph", readOnlyHint=True, openWorldHint=False))
    @safe_endpoint("explore_graph", limiter, "search")
    def explore_graph(
        note_id: Annotated[int | None, Field(description="Start from this note id")] = None,
        concept: Annotated[str | None, Field(description="Or start from a concept, e.g. 'supply chain'")] = None,
        limit: Annotated[int, Field(description="Max neighbours to return (1-30)", ge=1, le=30)] = 10,
    ) -> dict:
        """Walk the knowledge graph from a note or a concept.

        From a note: its concepts, linked notes (wikilink / similar / next slide) and notes
        sharing each concept. From a concept: the notes about it and co-occurring concepts.
        Give exactly one of note_id or concept.
        """
        return kb.explore(note_id, concept, limit)

    @mcp.tool(annotations=ToolAnnotations(title="Read full note", readOnlyHint=True, openWorldHint=False))
    @safe_endpoint("get_note", limiter, "search")
    def get_note(
        note_id: Annotated[int, Field(description="Id of the note to read in full")],
    ) -> str:
        """Return the complete text of one note with its citation and concepts.

        search_notes only returns a short snippet; use this when a snippet is cut off.
        """
        return kb.note_markdown(int(note_id))

    # -------------------------------------------------------------- resources
    @mcp.resource("notes://sources", name="sources", mime_type="application/json",
                  description="Documents available to ingest (relative names only).")
    @safe_endpoint("notes://sources")
    def sources() -> str:
        return json.dumps(kb.list_sources(), indent=2)

    @mcp.resource("notes://graph", name="graph_overview", mime_type="application/json",
                  description="Knowledge-graph statistics, courses, sources and top concepts.")
    @safe_endpoint("notes://graph")
    def graph() -> str:
        return json.dumps(kb.graph_overview(), indent=2)

    @mcp.resource("notes://note/{note_id}", name="note", mime_type="text/markdown",
                  description="Full text of one note with its source citation and concepts.")
    @safe_endpoint("notes://note")
    def note(note_id: int) -> str:
        return kb.note_markdown(int(note_id))

    # ---------------------------------------------------------------- prompts
    @mcp.prompt(title="Build a study guide")
    def study_guide(
        topic: Annotated[str, Field(description="Topic or exam unit to study")],
        course: Annotated[str, Field(description="Optional course code")] = "",
    ) -> str:
        """Create a cited study guide for a topic from your own slides and readings."""
        scope = f" for {course}" if course else ""
        return (
            f"Build a study guide on **{topic}**{scope} using only my notes in the Study Graph server.\n\n"
            "Steps:\n"
            f"1. Call `search_notes` with query=\"{topic}\"" + (f", course=\"{course}\"" if course else "")
            + " and top_k=8.\n"
            "2. Snippets are cut off, so call `get_note` for every result you plan to cite (and for any "
            "result whose snippet ends in '...') and work from the full text.\n"
            "3. For the two most relevant results, call `explore_graph` with their note_id to find "
            "connected ideas, and `explore_graph` with the most important concept. Also `get_note` any "
            "linked note (e.g. the next slide, or a note tagged with a key concept) that looks relevant "
            "but was not in the search results.\n"
            "4. Write the guide with these sections:\n"
            "   - Key ideas (5-8 bullets, each ending with a citation like [note 12: week3.pptx, slide 4])\n"
            "   - Definitions (term: one-sentence definition, cited)\n"
            "   - How the ideas connect (short paragraph based on the graph links)\n"
            "   - 3 practice questions, with answers hidden under a 'Answers' heading\n"
            "   - Gaps: anything the notes do not cover well\n"
            "5. Do not invent facts that are not in the notes; say so if coverage is thin.\n"
            "6. Finally, offer to save the guide with `add_note`, linking the cited notes with [[Title]]."
        )

    @mcp.prompt(title="Quiz me")
    def quiz_me(
        topic: Annotated[str, Field(description="Topic to be quizzed on")],
        questions: Annotated[str, Field(description="Number of questions, 1-10")] = "5",
    ) -> str:
        """Interactive quiz on a topic, graded against your notes."""
        n = max(1, min(10, int(questions))) if questions.isdigit() else 5
        return (
            f"Quiz me on **{topic}** using my Study Graph notes. First call `search_notes` for the topic. "
            f"Then ask me {n} questions one at a time, mixing recall and application. Wait for my answer "
            "before the next question, grade it against the notes and cite the note that has the answer. "
            "End with a score and the notes I should review."
        )

    logger.info("study-graph server ready (embedding backend: %s)", kb.embedder.model_name)
    return mcp


def main() -> None:
    settings = load_settings()
    configure_logging(settings.log_file, [settings.gemini_api_key or ""])
    create_server(settings).run()  # stdio transport, which Claude Desktop uses


if __name__ == "__main__":
    main()
