"""End-to-end tests through a real in-memory MCP client session."""

import json

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from study_graph.server import create_server


@pytest.mark.anyio
async def test_lists_tools_resources_prompts(settings):
    async with create_connected_server_and_client_session(create_server(settings)) as client:
        tools = {t.name for t in (await client.list_tools()).tools}
        assert tools == {"ingest_document", "add_note", "search_notes", "explore_graph", "get_note"}
        resources = {str(r.uri) for r in (await client.list_resources()).resources}
        assert {"notes://sources", "notes://graph"} <= resources
        templates = {t.uriTemplate for t in (await client.list_resource_templates()).resourceTemplates}
        assert "notes://note/{note_id}" in templates
        prompts = {p.name for p in (await client.list_prompts()).prompts}
        assert prompts == {"study_guide", "quiz_me"}


@pytest.mark.anyio
async def test_full_flow(settings):
    async with create_connected_server_and_client_session(create_server(settings)) as client:
        sources = json.loads((await client.read_resource("notes://sources")).contents[0].text)
        assert "week2_lean_operations.pptx" in [f["filename"] for f in sources["files"]]
        assert str(settings.sources_dir) not in json.dumps(sources)

        ingest = await client.call_tool("ingest_document", {"filename": "week2_lean_operations.pptx"})
        assert not ingest.isError
        search = await client.call_tool("search_notes", {"query": "takt time bottleneck", "top_k": 2})
        top = json.loads(search.content[0].text)["results"][0]
        note = await client.read_resource(f"notes://note/{top['note_id']}")
        assert "Source:" in note.contents[0].text

        prompt = await client.get_prompt("study_guide", {"topic": "bottlenecks"})
        assert "search_notes" in prompt.messages[0].content.text


@pytest.mark.anyio
async def test_errors_are_sanitised_over_mcp(settings):
    async with create_connected_server_and_client_session(create_server(settings)) as client:
        result = await client.call_tool("explore_graph", {"note_id": 2**70})
        assert result.isError
        text = result.content[0].text
        assert "incident" in text and "SQLite" not in text
