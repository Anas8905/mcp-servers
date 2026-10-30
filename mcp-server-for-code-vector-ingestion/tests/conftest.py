"""Shared fixtures: offline embeddings, isolated settings, a sample repo."""

from __future__ import annotations

import zlib
from pathlib import Path

import chromadb
import pytest
from langchain_core.embeddings import Embeddings

from mcp_server_for_code_vector_ingestion.config import Settings
from mcp_server_for_code_vector_ingestion.registry import Registry
from mcp_server_for_code_vector_ingestion.vectorstore import VectorStore


class FakeEmbeddings(Embeddings):
    """Deterministic, offline embeddings for tests.

    Keyword-based one-hot vectors: two texts sharing words are similar,
    so similarity_search has *meaningful* ordering without a network.
    """

    DIM = 16

    @staticmethod
    def _vec(text: str) -> list[float]:
        vec = [0.0] * FakeEmbeddings.DIM
        for word in __import__("re").findall(r"[a-z0-9]+", text.lower()):
            vec[zlib.crc32(word.encode()) % FakeEmbeddings.DIM] += 1.0
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        db_dir=tmp_path / "db",
        registry_file=tmp_path / "registry.json",
        embedding_model="fake-embed",
        embedding_base_url="http://fake",
        embedding_api_key="fake",
        annotate_model="fake-llm",
        annotate_base_url="http://fake",
        annotate_api_key="fake",
        annotation_branch="annotated",
        chunk_size=400,
        chunk_overlap=60,
        max_file_bytes=2000,
        max_context_chars=8000,
        transport="stdio",
    )


@pytest.fixture
def registry(settings: Settings) -> Registry:
    return Registry(settings.registry_file)


@pytest.fixture
def store(settings: Settings) -> VectorStore:
    return VectorStore(
        settings,
        embeddings=FakeEmbeddings(),
        client=chromadb.EphemeralClient(),
    )


MAIN_PY = '''"""Sample main module for tests."""
import os


def setup_redis_queue(queue_name: str) -> str:
    """Set up the redis queue for the pipeline."""
    pool = os.environ.get("REDIS_HOST", "localhost")
    return f"redis://{pool}/{queue_name}"


def process_job(job: dict) -> bool:
    """Process one queued job end to end."""
    if not job:
        return False
    job["done"] = True
    return True


def flush_metrics(metrics: list) -> None:
    """Flush accumulated metric counters."""
    del metrics[:]
'''

GO_FILE = '''package pipeline

// Setup wires the workers together.
func Setup(conf string) string {
\treturn conf + "#worker"
}

func Process(job map[string]bool) bool {
\treturn job["ok"]
}
'''

CPP_FILE = '''/* Sample C++ worker. */
#include <string>

std::string make_job(std::string name) {
    return "job:" + name;
}

int run_job(const char* name) {
    return static_cast<int>(name[0]);
}
'''

JS_FILE = '''export function startWorker(name) {
  return { name };
}

const stopWorker = (name) => ({ name, stopped: true });
'''

README_MD = """# Sample repo

Handles a demo job pipeline: redis queue, workers, metrics.
"""


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    root = tmp_path / "sample_repo"
    (root / "src").mkdir(parents=True)
    (root / "node_modules").mkdir()
    (root / ".venv").mkdir()
    (root / "docs").mkdir()

    (root / "main.py").write_text(MAIN_PY, encoding="utf-8")
    (root / "src" / "worker.go").write_text(GO_FILE, encoding="utf-8")
    (root / "src" / "worker.cpp").write_text(CPP_FILE, encoding="utf-8")
    (root / "src" / "worker.js").write_text(JS_FILE, encoding="utf-8")
    (root / "docs" / "README.md").write_text(README_MD, encoding="utf-8")
    (root / "node_modules" / "junk.js").write_text("ignored", encoding="utf-8")
    (root / ".venv" / "junk.py").write_text("ignored", encoding="utf-8")
    (root / "big.dat").write_bytes(b"\x00" * 5000)  # > max_file_bytes (2000)
    (root / "data.txt").write_bytes(b"\xff\xfe\x00\x01binary")  # non-UTF-8
    return root
