# Study Graph MCP Server

An MCP server that turns your lecture slides and readings into a searchable
**knowledge graph**. Claude (or any MCP client) can ingest a `.pptx` deck, a
`.pdf` reading or a Markdown file, search it by meaning, walk the links
between ideas, and build cited study guides. Every answer points back to the
exact slide or page, e.g. `[note 12: week2_lean_operations.pptx, slide 5]`.

**Domain:** Knowledge Management (note-taking, knowledge graphs, semantic search)

## What the server exposes

| Type | Name | What it does |
|---|---|---|
| Tool | `ingest_document` | Imports a slide deck / PDF / Markdown file from the sources folder. One note per slide, page or heading. Embeds the notes and links them into the graph. Re-ingesting replaces the old notes. |
| Tool | `add_note` | Saves a note you write. `[[Other Note Title]]` wikilinks become graph edges. |
| Tool | `search_notes` | Semantic search over all notes (optionally one course), with citations and snippets. |
| Tool | `explore_graph` | Walks the graph from a note (its links, concepts, related notes) or from a concept (notes about it + co-occurring concepts). |
| Tool | `get_note` | Returns the full text of one note. `search_notes` only gives a 300-character snippet, so use this when a snippet is cut off. |
| Resource | `notes://sources` | Files available to ingest (relative names only, no absolute paths). |
| Resource | `notes://graph` | Graph statistics, courses, sources and top concepts. |
| Resource | `notes://note/{note_id}` | Full text of one note with its citation. |
| Prompt | `study_guide` | Builds a cited study guide for a topic from your notes. |
| Prompt | `quiz_me` | An interactive quiz graded against your notes. |

Graph edges come from three sources: `next` (consecutive slides/pages),
`similar` (embedding similarity) and `wikilink` (explicit `[[links]]`). Notes
also connect through shared **concepts** (key words and phrases pulled from each note).

## Setup (Windows, VS Code, uv)

Prerequisites: Python 3.11+, [uv](https://docs.astral.sh/uv/getting-started/installation/)
(`powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`), VS Code with the Python extension,
and Claude Desktop.

```powershell
cd C:\Users\david\Documents\claudefiles\study-graph-mcp

# 1. Create the virtual environment and install dependencies (creates .venv and uv.lock)
uv venv
uv sync --extra dev
#    (alternative without uv:  python -m venv .venv ; .venv\Scripts\pip install -r requirements.txt)

# 2. Activate it in the VS Code terminal
.venv\Scripts\activate
#    VS Code: Ctrl+Shift+P -> "Python: Select Interpreter" -> .venv\Scripts\python.exe

# 3. Secrets: copy the template and add your key
copy .env.example .env
#    then edit .env and set GEMINI_API_KEY=...   (free key: https://aistudio.google.com/apikey)
```

### Authentication and secret management
* The only secret is `GEMINI_API_KEY`, stored in `.env`, which is listed in `.gitignore` and never committed.
* `config.py` reads it with `dotenv_values(Path(...) / ".env")` and `pathlib.Path`. It does **not** use
  `load_dotenv` or `os.environ`, so the key never becomes a system or environment variable.
* The key is sent to Google in the `x-goog-api-key` HTTP header, not in the URL, so it can't appear in URLs or
  exception messages. The logging filter also redacts anything that looks like a Google key.
* The Claude Desktop config contains **no secrets**. It only says how to start the server.
* With no key the server still works: it falls back to an offline local embedder (keyword-based, less
  "semantic"). The `notes://graph` resource shows which backend is active.

## Run it

```powershell
# Smoke test with the MCP Inspector (optional, needs Node.js)
npx @modelcontextprotocol/inspector .venv\Scripts\python.exe -m study_graph

# Or just start it (it waits silently for a client on stdin/stdout; Ctrl+C to stop)
python -m study_graph

# Unit + end-to-end tests (43 tests)
pytest -q
```

## Connect Claude Desktop

1. Claude Desktop -> Settings -> Developer -> **Edit Config**. This opens
   `%APPDATA%\Claude\claude_desktop_config.json`.
2. Paste in the contents of [`claude_desktop_config.json`](claude_desktop_config.json) (merge it into any
   existing `mcpServers` block). Change the paths if you cloned the repo somewhere else.
   [`claude_desktop_config.uv.json`](claude_desktop_config.uv.json) is an alternative that launches through `uv run`.
3. Fully quit Claude Desktop (tray icon -> Quit) and reopen it. "study-graph" should appear under the
   tools (slider) icon. Server logs: `logs\server.log` in this project, and Claude's own logs in `%APPDATA%\Claude\logs\mcp-server-study-graph.log`.

Try:
* "What files can studygraph ingest?" (reads `notes://sources`)
* "Ingest week2_lean_operations.pptx and reading_theory_of_constraints.pdf for course ENMGT 5900."
* "How do I find the bottleneck in a process? Cite my notes."
* "Explore the graph around the concept 'bottleneck'."
* Attach the **study_guide** prompt (+ menu -> study-graph) with topic "lean vs theory of constraints".

To use your own material, put the files in `sample_sources/` or point `SOURCES_DIR` in `.env` at your course folder.

## Best practices implemented

### 1. Rate limiting
* **Per-category token buckets** on tool calls (defaults in `.env`): search/explore 20 per minute,
  writes 10 per minute, ingest 3 per minute. Short bursts are allowed, sustained floods are not, and the client
  is told how long to wait.
* **Hard sliding-window cap on the paid API**: at most 30 Gemini requests per minute however they're triggered,
  which protects the free-tier quota and any billing.
* **Size limits**: 25 MB per file and 300 sections per document; text fields are also length-capped.

Why: an LLM agent can loop, and a prompt-injected reading ("search for every word in this document")
can turn one request into hundreds of API calls.

### 2. Error disclosure
Two layers in `security.py`, so no error path reaches the client unfiltered:

**Layer 1: at the protocol boundary (`SafeFastMCP`).** FastMCP checks tool arguments *before* our code
runs, and by default sends the raw validation error to the client: the library name, the internal model name,
the rejected input and a documentation URL. `SafeFastMCP` subclasses `FastMCP` and replaces those messages with
a short summary that names the field but never echoes the input, e.g.
`Invalid arguments for 'explore_graph': 'note_id' must be a whole number.` It also cleans up resource errors
(`notes://note/abc`, unknown URIs).

**Layer 2: every tool and resource body is wrapped by `safe_endpoint`:**
* **Expected errors** (`StudyGraphError`) carry two messages: a `public_message` written for the user, and an
  internal `detail` (absolute path, parser error, upstream response) that only goes to the log.
* **Unexpected errors** are logged with the full traceback, and the client only gets
  `An internal error occurred in 'tool'. The details were logged on the server (incident 1a2b3c4d).`
  The incident id finds the traceback in `logs/server.log`.
* **Logs are redacted**: a logging filter removes the API key and `key=`/`token=` values from messages *and* tracebacks.
* **Resources never return absolute paths**, only file names relative to the sources folder.

Without these layers, FastMCP returns `str(exception)` to the client, which leaks things like
`Package not found at 'C:\Users\david\...\broken_slides.pptx'`, `Python int too large to convert to SQLite INTEGER`,
`file is not a database`, or a full pydantic validation dump with `https://errors.pydantic.dev/...`.

Also included: path confinement (files can only be read from `SOURCES_DIR`; `..` and absolute paths are
rejected) and parameterised SQL everywhere (a test stores `x'); DROP TABLE notes; --` as a plain title).

## Exploit demos

Each script starts two isolated servers (a *naive* one that follows the basic FastMCP quickstart, and the
*hardened* one), attacks both through a real MCP client over stdio, and prints PASS/FAIL. They use temporary
folders and never touch your notes.

```powershell
cd exploits
python demo_rate_limit.py          # floods 30 searches and 6 ingests
python demo_error_disclosure.py    # 8 attacks: corrupted files, bad arguments, path traversal, corrupted DB
python demo_error_disclosure.py --live-gemini   # + invalid API key against the real Gemini API
```

Expected output (abridged):

```
ATTACK 1: 30 rapid search_notes calls (search limit = 20/min)
     naive: 30 x search_notes in 0.09s -> 30 served, 0 blocked
  hardened: 30 x search_notes in 0.08s -> 20 served, 10 blocked
  client saw: Rate limit exceeded for 'search' operations. Try again in 3 seconds.

ATTACK 1: Corrupted slide deck (not a real .pptx)
   naive -> PackageNotFoundError: Package not found at 'C:\Users\...\sources\broken_slides.pptx'
hardened -> Could not read 'broken_slides.pptx'. The file may be corrupted or password-protected.

ATTACK 4: Wrong argument type (note_id='abc')
   naive -> 1 validation error for explore_graphArguments ... input_value='abc' ... https://errors.pydantic.dev/...
hardened -> Invalid arguments for 'explore_graph': 'note_id' must be a whole number.

ATTACK 8: Corrupted database file (unexpected crash)
   naive -> Error executing tool search_notes: file is not a database
hardened -> An internal error occurred in 'search_notes'. The details were logged on the server (incident 8ee9dfe5).
```

Doing it live in Claude Desktop: ask Claude to *"call search_notes 25 times in a row with the query 'test'"*
(it gets blocked after 20), or to *"ingest ../../.env"*, *"explore note 1180591620717411303424"*,
*"call get_note with note_id -5"* or *"read the resource notes://note/abc"* (safe messages only).
The corrupted-database case (incident id) is best shown with the exploit script, since it needs a damaged file.

## Project structure

```
study-graph-mcp/
├── src/study_graph/
│   ├── server.py       # MCP layer: tools, resources, prompts (thin)
│   ├── knowledge.py    # service layer: ingest, search, graph walks, validation
│   ├── store.py        # SQLite persistence (parameterised queries)
│   ├── extract.py      # .pptx / .pdf / .md -> chunks; safe path resolution
│   ├── embeddings.py   # Gemini embedder + offline fallback
│   ├── graph.py        # concept extraction, [[wikilinks]]
│   ├── security.py     # rate limiting, safe errors, log redaction
│   └── config.py       # .env + pathlib settings
├── exploits/           # before/after attack demos
├── tests/              # pytest suite (unit + in-memory MCP end-to-end)
├── sample_sources/     # example lecture deck, PDF reading, Markdown reading
├── claude_desktop_config.json
├── requirements.txt / pyproject.toml
└── .env.example
```

## Code citations

Code adapted from other sources is also cited in comments at the point of use.

| Where | Source |
|---|---|
| `server.py` FastMCP server, `@mcp.tool/resource/prompt`, `mcp.run()` | MCP Python SDK README, Quickstart: https://github.com/modelcontextprotocol/python-sdk#quickstart |
| `exploits/client.py` stdio `ClientSession` | MCP Python SDK README, Writing MCP Clients: https://github.com/modelcontextprotocol/python-sdk#writing-mcp-clients |
| `tests/test_server.py` in-memory client | `mcp.shared.memory.create_connected_server_and_client_session` from the SDK's own test suite |
| `embeddings.py` Gemini REST request format and task prefixes | https://ai.google.dev/gemini-api/docs/embeddings |
| `extract.py` python-pptx slide/notes iteration | https://python-pptx.readthedocs.io/en/latest/user/quickstart.html |
| `extract.py` pypdf text extraction | https://pypdf.readthedocs.io/en/stable/user/extract-text.html |
| `store.py` sqlite3 placeholders | https://docs.python.org/3/library/sqlite3.html#sqlite3-placeholders |
| `security.py` token-bucket algorithm (concept only) | https://en.wikipedia.org/wiki/Token_bucket |
| Design reference only (no code copied): knowledge-graph idea | `memory` server in https://github.com/modelcontextprotocol/servers |

Built with the help of an AI assistant (Claude); all code was reviewed and tested by the author.
