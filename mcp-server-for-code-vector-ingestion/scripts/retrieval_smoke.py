#!/usr/bin/env python3
"""Live retrieval smoke test (PoC parity).

Runs the same 8 natural-language queries as the original PoC
(``rag-experiments/rag_for_code/test_object_detection_ingestion.py``)
against the MCP server's *service layer* (registration -> ingestion ->
query -> context), so it validates the whole stack end to end with real
Ollama models.

Usage (from the project root):

    uv run python scripts/retrieval_smoke.py /path/to/repo
    uv run python scripts/retrieval_smoke.py /path/to/repo --context \
        --query "how is X configured?"

Notes:
- ingestion runs even if the repo is already registered (the PoC refresh
  the design requires — the previous collection is replaced and its size
  reported);
- ``--context`` fetches the enclosing definition behind each top hit that
  is a Python file (the PoC's two-stage flow);
- exit code 0 when every query produced at least one hit.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mcp_server_for_code_vector_ingestion.annotate import scan_repository
from mcp_server_for_code_vector_ingestion.config import get_settings
from mcp_server_for_code_vector_ingestion.context import read_file_context
from mcp_server_for_code_vector_ingestion.ingestion import ingest_repository
from mcp_server_for_code_vector_ingestion.query import query_repository
from mcp_server_for_code_vector_ingestion.registry import Registry

# the 8 queries from the PoC — the acceptance set for the object-detection repo
DEFAULT_QUERIES = [
    "How are incoming jobs picked up and queued into the pipeline?",
    "How is motion in the alarm video detected before object detection runs?",
    "How does the YOLO object detection model server request/response over Redis work?",
    "How are face recognition models trained and where are they stored?",
    "How does the plate OCR model server reply with the plate text?",
    "How is the VLM scene description generated for an alarm?",
    "How are the detection results committed to the database after object detection?",
    "What happens when a model server does not answer in time?",
]

K = 5


def _print_hit(rank: int, hit) -> None:
    snippet = " ".join(hit.text.split())[:140]
    print(f"  {rank}. {hit.source}  [chunk {hit.chunk_index}]  score={hit.score:.3f}  :: {snippet}")


def _maybe_context(
    repo_name: str,
    hit,
    settings,
    registry: Registry,
) -> None:
    if hit.language != "python" or hit.start_index is None:
        return
    try:
        result = read_file_context(
            repo_name,
            hit.source,
            start_index=hit.start_index,
            mode="enclosing",
            registry=registry,
            settings=settings,
        )
    except Exception as exc:  # context is a nicety; don't fail the smoke test on it
        print(f"     (context: {exc.__class__.__name__}: {exc})")
        return
    print(f"     enclosing [{result.mode}]:")
    for line in result.content.splitlines()[:40]:
        print(f"       {line}")
    if result.truncated:
        print("       ... [truncated]")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("repo", help="path to the repository (auto-registered)")
    parser.add_argument("-q", "--query", action="append", dest="queries",
                        help="override the default PoC queries (repeatable)")
    parser.add_argument("--context", action="store_true",
                        help="fetch the enclosing Python definition behind each top hit")
    parser.add_argument("--skip-ingest", action="store_true",
                        help="do not (re)build the collection; query the existing one")
    args = parser.parse_args(argv)

    settings = get_settings()
    registry = Registry(settings.registry_file)

    print(f"== repo: {args.repo}")
    scan = scan_repository(args.repo, settings, registry)
    print(f"   scan: {scan.files_scanned} files, {scan.total_gaps} annotation gap(s) "
          f"across {scan.files_with_gaps} file(s)")

    if not args.skip_ingest:
        print("== ingestion (destructive refresh)")
        ing = ingest_repository(args.repo, settings, registry=registry)
        record = registry.resolve(ing.repo_name)
        print(f"   {ing.files_ingested} files -> {ing.chunks} chunks "
              f"(replaced {ing.previous_chunks_replaced})")
        print(f"   collection: {record.collection_name}")
    else:
        record = registry.resolve(scan.repo["repo_name"])
        print(f"   (skipping ingestion; existing collection {record.collection_name})")

    queries = args.queries or DEFAULT_QUERIES
    repo_name = scan.repo["repo_name"]
    failed = 0
    for query in queries:
        print("\n" + "=" * 78)
        print(f"Q: {query}")
        print("=" * 78)
        result = query_repository(
            repo_name, query, k=K, settings=settings, registry=registry
        )
        if not result.results:
            failed += 1
            print("  (no hits — collection missing or empty? run without --skip-ingest)")
            continue
        if result.note:
            print(f"  note: {result.note}")
        for rank, hit in enumerate(result.results, 1):
            _print_hit(rank, hit)
            if args.context and rank == 1:
                _maybe_context(repo_name, hit, settings, registry)

    print("\n" + "=" * 78)
    if failed:
        print(f"{failed}/{len(queries)} queries produced no hits — FAIL")
        return 1
    print(f"all {len(queries)} queries produced hits — OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
