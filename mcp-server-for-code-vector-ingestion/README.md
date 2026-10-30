# mcp-server-for-code-vector-ingestion

An MCP server that turns code repositories into searchable, well-documented
knowledge: it annotates missing docstrings (LLM, on a dedicated git branch),
chunks + embeds the code into ChromaDB, and serves semantic search with the
full source context behind every hit.

```
annotate_scan ──► annotate_apply ──► ingest_repository ──► query_repository ──► read_file_context
  (cheap,         (LLM + git         (chunk + embed,       (semantic +          (the whole
  read-only)       branch, commit)    destructive)          location metadata)   function/file)
```

Repos are registered the first time you touch them and get a stable UUID —
from then on you can address them by name (e.g. `object_detection`).

## Requirements

- [uv](https://docs.astral.sh/uv/) + Python 3.14+
- [Ollama](https://ollama.com) with two models:
  - `qwen3.8` — chat model, writes the docstrings
  - `qwen3-embedding:0.6b` — embeddings (used at **both** ingestion and query
    time — keep them the same or results will be garbage)
- `git` on PATH (the annotation step works on real branches)

## Setup

```bash
git clone <this repo> && cd mcp-server-for-code-vector-ingestion
uv sync
cp .env.example .env          # then point the *_BASE_URL / *_MODEL values
                              # at your Ollama (or any OpenAI-compatible) stack
```

All configuration is environment-driven (real env vars take priority over
`.env`); nothing model- or path-related is hardcoded. If a required value is
missing the server refuses to start and tells you which variable.

Useful knobs (full list in `.env.example`):

| Variable | Default | Purpose |
|---|---|---|
| `EMBEDDING_MODEL` / `EMBEDDING_BASE_URL` / `EMBEDDING_API_KEY` | required | embedding endpoint |
| `ANNOTATE_MODEL` / `ANNOTATE_BASE_URL` / `ANNOTATE_API_KEY` | required | docstring LLM endpoint |
| `CHROMA_DB_DIR` | `./db` (project root) | Chroma + registry storage |
| `ANNOTATION_BRANCH` | `annotated` | branch annotation commits land on |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `150` | chunker tuning |
| `MAX_FILE_BYTES` | `524288` | files bigger than this are skipped + reported |
| `MAX_CONTEXT_CHARS` | `16000` | `read_file_context` output cap |
| `MCP_TRANSPORT` | `stdio` | `stdio` or `streamable-http` |

## Run

```bash
uv run mcp-server-for-code-vector-ingestion
```

## Wire it into an MCP client

stdio (the default):

```json
{
  "mcpServers": {
    "code-vector-ingestion": {
      "command": "uv",
      "args": ["run", "--project", "/absolute/path/to/mcp-server-for-code-vector-ingestion",
               "mcp-server-for-code-vector-ingestion"],
      "env": {
        "EMBEDDING_MODEL": "qwen3-embedding:0.6b",
        "EMBEDDING_BASE_URL": "http://localhost:11434/v1",
        "EMBEDDING_API_KEY": "ollama",
        "ANNOTATE_MODEL": "qwen3.8",
        "ANNOTATE_BASE_URL": "http://localhost:11434/v1",
        "ANNOTATE_API_KEY": "ollama",
        "CHROMA_DB_DIR": "./db",
        "ANNOTATION_BRANCH": "annotated"
      }
    }
  }
}
```

Or set `MCP_TRANSPORT=streamable-http` and point the client at
`http://localhost:<port>/mcp` (fastmcp's default streamable-http path).

## The six tools

| Tool | What it does |
|---|---|
| `annotate_scan(repo)` | Read-only report of which files/functions lack docstrings. No LLM, no git writes. Run this first to size the job. |
| `annotate_apply(repo)` | LLM-generates + inserts docstrings on the `annotated` git branch, single commit. Never pushes. |
| `ingest_repository(repo)` | Chunk + embed + (re)build the repo's Chroma collection. **Destructive refresh** — the previous collection is deleted and replaced; the old chunk count is returned. |
| `query_repository(repo, query, k=5)` | Ranked chunks with location metadata (`source`, `absolute_path`, `language`, `chunk_index`, `start_index`). |
| `read_file_context(repo, file_path, start_index, mode)` | The full implementation behind a hit. Python defaults to the enclosing function/class (AST-precise); everything else gets a capped full-file read; `mode` can force either. |
| `list_repos()` | Registered repos with ingestion/annotation status. |

Full parameter/return/error contracts: **[AGENT.md](AGENT.md)**.

## Annotation style

`annotate_apply` produces docstrings in a deliberately fixed shape so that
follow-up questions can *follow the flow*, not just describe one function:

- **Module docstring** — birds-eye view: where the file sits in the system,
  its role, the modules it talks to.
- **Function docstring** — 1-3 plain sentences *plus* a `Flow:` block:

  ```text
  Run one alarm job inside a Ray actor: passes straight through to run_obj_detection.
  Flow:
  In: the job's detection settings and notification settings, enqueued by start_watchdog.py.
  Out: handed to run_obj_detection in start_detection.py, which performs the actual detection.
  Triggered by: Ray EventDetector actors — one call per alarm job.
  ```

The `Flow:` lines are grounded **only** in the file source plus a static
call-graph digest computed once per run (cross-file callers/callees; Python
via AST, other languages via the scanner's signature heuristics). Anything
the model can't ground is omitted — no invented callers, inputs, or
destinations. If a `Flow:` line is absent, read that as "no evidence in the
repo", not "not implemented".

## Safety model (what the server will and won't do)

- **Annotation never pushes.** It creates/updates one local branch
  (`annotated`), commits there, and leaves your working branch alone.
- **Merge conflicts are a hard stop**: the merge is aborted, the conflicted
  files are named, and the tool raises with a message you can hand to a human.
- **Dirty trees are rejected up front**, before anything is touched.
- **Per-file annotation failures are isolated** — one bad LLM reply fails
  that file only; the rest still get committed.
- **`read_file_context` can only read files inside a registered repository.**
  This is a retrieval server, not an "read any file on disk" oracle.
- **Ingestion is destructive by design** — but scoped to that repo's own
  collection. Skipped files (oversized/binary) are always reported.

## Tests

Fully offline (fake embeddings + ephemeral Chroma):

```bash
uv run pytest
```

## Typical session

```
1. annotate_scan("my_repo")            → "12 files, 43 gaps" — worth it?
2. annotate_apply("my_repo")           → 41 annotations on branch `annotated`
3. ingest_repository("my_repo")        → 312 chunks (replaced 289)
4. query_repository("my_repo", "how are redis queues configured?")
                                       → hits with source + start_index
5. read_file_context("my_repo", "pipeline/queue.py", start_index=214)
                                       → the whole function, not 400 chars of it
```

Merge the `annotated` branch in your repo when you're happy with what it
generated — that's the review step, and it's yours to make.
