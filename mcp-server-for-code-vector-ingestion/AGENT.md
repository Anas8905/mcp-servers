# AGENT.md — tool contracts for `code-vector-ingestion`

You are an agent calling this MCP server. This file defines exactly what
each tool takes, returns, and can throw. All tools return JSON dicts;
failures arrive as tool errors with a `message` you should forward to the
user verbatim when it mentions conflicts, dirty trees, or paths.

## Workflow

For a repository you have never seen:

```
annotate_scan → (report to user, ask go/no-go) → annotate_apply
    → ingest_repository → query_repository → read_file_context
```

`annotate_scan` and the scan inside `annotate_apply` are the *same* scanner,
so the pre-scan report predicts the apply report well. Only run `annotate_apply`
if the user approved the LLM cost; it is idempotent, but every run of a file
that still has unfixable gaps costs an LLM call.

`ingest_repository` must run **before** `query_repository`, and must re-run
after any `annotate_apply` you want reflected (annotation changes the source
text; the collection is the old text until refreshed).

Repos are addressed by `repo` in every tool: a filesystem path
(auto-registered, given a UUID) or a previously registered name/UUID.
`list_repos()` shows all registered repos with their status.

## Tool contracts

### `annotate_scan(repo: str) -> dict`

Read-only scan for missing docstrings. **No LLM calls, no git changes.**

| field | type | meaning |
|---|---|---|
| `repo` | dict | `repo_id`, `repo_name`, `repo_path`, `annotation_branch` |
| `files_scanned` | int | files eligible for annotation |
| `files_with_gaps` | int | files with at least one missing annotation |
| `total_gaps` | int | sum of missing module headers + functions/classes |
| `unknown_languages` | list[str] | files whose language has no annotation support |
| `skipped_too_large` / `skipped_binary` | list[str] | files skipped (repo-relative) |
| `files` | list | per-file: `file`, `language`, `module_missing`, `functions_missing: [{name, line}]` |
| `next_step` | str | actionable hint |

Python detection is AST-precise (zero false positives). Other languages are
best-effort heuristics and may miss functions or flag documented ones — an
"annotated file not flagged" is possible; a double-annotated file is not
(the apply step never inserts where documentation already exists).

### `annotate_apply(repo: str) -> dict`

LLM-generates + inserts docstrings, on the annotation branch, one commit.

Guarantees and preconditions:
- Working tree must be clean or a `RuntimeError` is raised.
- Creates the annotation branch if missing; if it exists, merges the default
  branch into it. **Merge conflict ⇒ hard stop**: the merge is aborted and a
  `RuntimeError` names every conflicted file. Forward that message to the
  user — do not retry blindly, the conflict is real.
- Never pushes. Your default branch is never moved or modified.
- Per-file failures are isolated: a bad LLM reply for one file does not
  abort the run; the report says which files failed and why.
- The commit happens only if at least one annotation landed.

| field | type | meaning |
|---|---|---|
| `branch` | dict | `annotation_branch`, `default_branch`, `already_existed`, `merged_default_into_annotation` |
| `files_scanned` / `files_annotated` / `functions_annotated` | int | counters |
| `commit` | str \| null | sha of the new commit, or `null` when nothing was annotated |
| `messages` | list[str] | merge notes and anomalies |
| `files` | list | per-file: `file`, `language`, `status` (`annotated` \| `skipped` \| `failed`), `module_applied`, `functions_applied`, `skipped` (with `reason`), `errors` |
| `next_step` | str | actionable hint |

Annotation style (what the LLM is asked to produce, and what a follow-up
"trace the flow" query relies on):
- Module docstring: birds-eye view — where the file sits, role, neighbors.
- Function docstring: 1-3 sentence description **plus a `Flow:` block**
  (`In:` / `Out:` / `Triggered by:`) placing the function on the source
  graph. Every Flow line is grounded strictly in file source or the
  static call-graph digest; **ungrounded lines are omitted, never
  invented**. A `Flow:` line missing means "no evidence in the repo",
  not "not implemented".
- The call graph is computed once per run (`call_context.build_call_index`):
  Python via AST (calls inside methods/lambdas are excluded when they can't
  be attributed), other languages via the same signature heuristics as the
  scanner + a `name(` regex. Cross-file edges only; builtins and external
  library calls are not in the graph and must not appear in a Flow block.
- If call-graph construction fails for any reason, the run continues and
  the affected files simply lose Flow grounding (noted in `messages`).

Known `skipped` reasons: `single-line definition; cannot insert a docstring
safely` (Python `def f(): return 1` cannot host a docstring — re-running
will keep reporting it; that's expected, not a bug), `signature line not
found` (heuristic couldn't relocate the function).

### `ingest_repository(repo: str) -> dict`

Chunks + embeds + **replaces** the repo's collection (destructive refresh).

| field | type | meaning |
|---|---|---|
| `repo` | dict | `repo_id`, `repo_name`, `collection_name` |
| `files_ingested` | int | |
| `chunks` | int | new chunk count |
| `previous_chunks_replaced` | int | the count in the collection being destroyed |
| `skipped_too_large` / `skipped_binary` | list[str] | |
| `next_step` | str | |

Chunking is language-aware (python/cpp/go/rust/js/ts/php/java/... get a
recursive splitter tuned to them; markdown gets a markdown-aware one;
everything else falls back to plain recursive).

### `query_repository(repo: str, query: str, k: int = 5, min_score: float | None = None) -> dict`

Ranked hits over the repo's chunks.

| field | type | meaning |
|---|---|---|
| `k` | int | requested number of hits |
| `note` | str \| null | present when hits are weak (all scores < 0.3) — say so to the user |
| `results` | list | per-hit: `rank`, `score` (0..1), `source` (repo-relative path), `absolute_path`, `language`, `chunk_index` (0-based; `null` only if the chunk pre-dates the metadata), `start_index` (0-based char offset), `text` |

Errors: unknown repo ⇒ tool error naming the lookup failure. Call
`list_repos()` to see what is registered.

Every hit has exactly the metadata you need to follow up:

```
read_file_context(repo, file_path=hit.source, start_index=hit.start_index)
```

### `read_file_context(repo: str, file_path: str, start_index: int | None = None, mode: str = "auto", max_chars: int | None = None) -> dict`

Full implementation behind a hit. Mode semantics:

- `auto` (default): Python files with a `start_index` get the **smallest
  enclosing function/class** (AST-precise — the whole function, not a
  window). Everything else degrades to a capped full-file read.
- `enclosing`: force the Python path; on non-Python files this degrades to
  a window around the offset (no AST for those).
- `full`: capped full-file read (the only safe mode for files where a
  window would be meaningless).

| field | type | meaning |
|---|---|---|
| `source` / `absolute_path` | str | where the content came from |
| `mode` | str | `enclosing` \| `full` \| `window` (what actually happened) |
| `total_chars` / `returned_chars` | int | if different, the file was longer than returned |
| `truncated` | bool | true means the server cut it off; consider raising `max_chars` |
| `content` | str | header line + source, ready to paste into context |

Security: `file_path` is resolved against the registered repo. Any path that
escapes the repo (e.g. `../../etc/passwd`) is rejected — this is intended.

### `list_repos() -> dict`

`repos: [{repo_id, repo_name, repo_path, ingested_at, ingestion_commit,
chunk_count, file_count, last_annotation_commit}]`. Use this whenever an
identifier doesn't resolve.

## Error messages you will see, and what they mean

| Error fragment | Meaning | What to do |
|---|---|---|
| `Working tree is not clean` | uncommitted changes in the repo | ask the user to commit or stash; do not force |
| `Merging 'main' into 'annotated' failed with conflicts in: a.py, b.go` | annotation branch and default branch edited the same lines (merge was aborted) | show the file list to the user; they must resolve; do not re-run until they do |
| `Annotation LLM did not return usable JSON` | the model produced prose instead of JSON on one file | that file alone failed; the rest of the run is fine |
| `Unknown repository` / no match | the identifier is not registered | call `list_repos()`, use one of those |
| `Path ... escapes the registered repository` | `read_file_context` was given a path outside the repo | you made a mistake; use the `source` value from the hit |
| `Configuration error: Missing required environment variable ...` | server not configured | tell the user which variable; nothing to fix in code |

## Gotchas worth remembering

1. **Embedding model consistency.** If you ever re-ingest with a different
   `EMBEDDING_MODEL` than the one that built the collection, scores become
   meaningless. In practice: don't change it without re-ingesting.
2. **Annotation is on a branch.** After `annotate_apply`, the *working
   tree of the default branch still has no docstrings* until the user merges
   `annotated`. Run `ingest_repository` on the **default branch state** —
   if they want the annotated version indexed, they merge first, then
   re-ingest.
3. **`chunk_index` is 0-based; `start_index` is a character offset, not a
   line number.**
4. **`min_score`** is a similarity floor (0..1); weak hits below it are
   dropped silently.
5. **Destructive ingestion is the design.** If the user wants the old
   collection back, they need to know before you call `ingest_repository`.
   Report `previous_chunks_replaced` in your summary.
