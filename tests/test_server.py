"""End-to-end tests through a real in-memory MCP client session.

``create_connected_server_and_client_session`` is the helper the MCP Python SDK uses in its own
test suite (``mcp/shared/memory.py``, v1.27.0):
https://github.com/modelcontextprotocol/python-sdk/blob/v1.27.0/src/mcp/shared/memory.py
"""

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
        # A huge id is now rejected by argument validation before it can reach SQLite.
        result = await client.call_tool("explore_graph", {"note_id": 2**70})
        assert result.isError
        text = result.content[0].text
        assert "must be at most" in text and "SQLite" not in text


@pytest.mark.anyio
@pytest.mark.parametrize("tool, args, expected", [
    ("explore_graph", {"note_id": "abc"}, "'note_id' must be a whole number"),
    ("explore_graph", {"note_id": 1.1805916207174113e21}, "'note_id' is too large"),
    ("explore_graph", {"note_id": 2**70}, "'note_id' must be at most"),
    ("get_note", {}, "'note_id' is required"),
    ("search_notes", {"query": "x", "top_k": 500}, "'top_k' must be at most 20"),
    ("add_note", {"title": "t", "body": "b", "tags": "notalist"}, "'tags' must be a list"),
])
async def test_bad_arguments_do_not_leak_validation_internals(settings, tool, args, expected):
    async with create_connected_server_and_client_session(create_server(settings)) as client:
        result = await client.call_tool(tool, args)
        text = result.content[0].text
        assert result.isError and expected in text
        for leak in ("pydantic", "input_value", "validation error for", "Arguments"):
            assert leak not in text


@pytest.mark.anyio
@pytest.mark.parametrize("uri", ["notes://note/abc", "notes://note/99999999999999999999999", "notes://note/0"])
async def test_bad_note_resource_ids_are_sanitised(settings, uri):
    from mcp.shared.exceptions import McpError

    async with create_connected_server_and_client_session(create_server(settings)) as client:
        with pytest.raises(McpError) as exc:
            await client.read_resource(uri)
        message = str(exc.value)
        assert "positive whole number" in message
        assert "pydantic" not in message and "Traceback" not in message


@pytest.mark.anyio
async def test_corrupted_database_returns_incident_id(settings):
    async with create_connected_server_and_client_session(create_server(settings)) as client:
        settings.db_path.write_bytes(b"not a database" * 100)
        result = await client.call_tool("search_notes", {"query": "bottleneck"})
        assert result.isError
        assert "incident" in result.content[0].text
        assert "database" not in result.content[0].text
