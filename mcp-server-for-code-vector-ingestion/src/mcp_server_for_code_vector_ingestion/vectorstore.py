"""Shared Chroma client and per-repo collection management.

One persistent Chroma client (configured in ``config.Settings``) holds
every repo's collection. Collection names are repo ids (UUIDs), so
there can be no name collisions or sanitization issues.

Design notes:
- Ingestion builds **all** documents first, then resets the collection
  (delete + create + add). If anything fails mid-way — say the embedding
  endpoint dies on file 40 of 60 — the previous collection is untouched.
- ``1 - distance`` is used as a similarity score in
  ``query_repository``.
"""

from __future__ import annotations

from chromadb import EphemeralClient, PersistentClient
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr

from .config import Settings, get_settings


def make_embeddings(settings: Settings | None = None) -> OpenAIEmbeddings:
    """OpenAI-compatible embeddings client (works with Ollama).

    ``check_embedding_ctx_length=False`` mirrors the PoC: small local
    models report (or don't report) context limits we want Ollama to
    override, and we size chunks well below the limit anyway.
    """
    s = settings or get_settings()
    return OpenAIEmbeddings(
        model=s.embedding_model,
        base_url=s.embedding_base_url,
        api_key=SecretStr(s.embedding_api_key),
        check_embedding_ctx_length=False,
        chunk_size=32,  # keep Ollama batches small enough for long context
    )


class CollectionNotFoundError(LookupError):
    """The repo has no chunk collection yet — ingestion has not run."""


class VectorStore:
    """Thin, testable facade over Chroma + langchain's Chroma wrapper."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        embeddings: OpenAIEmbeddings | None = None,
        client: PersistentClient | EphemeralClient | None = None,
    ) -> None:
        s = settings or get_settings()
        self._settings = s
        self._embeddings = embeddings or make_embeddings(s)
        self._client = client or PersistentClient(path=str(s.db_dir))
        self._collections: dict[str, Chroma] = {}

    # ------------------------------------------------------------------
    # collection lifecycle
    # ------------------------------------------------------------------

    def _collection(self, name: str) -> Chroma:
        if name not in self._collections:
            self._collections[name] = Chroma(
                collection_name=name,
                embedding_function=self._embeddings,
                client=self._client,
            )
        return self._collections[name]

    def exists(self, name: str) -> bool:
        try:
            self._client.get_collection(name)
            return True
        except Exception:  # chromadb raises several not-found flavors
            return False

    def delete(self, name: str) -> bool:
        try:
            self._client.delete_collection(name)
            self._collections.pop(name, None)
            return True
        except Exception:
            return False

    def clear(self, name: str) -> None:
        """Delete the collection if it exists (no-op otherwise)."""
        if self.exists(name):
            self.delete(name)

    def count(self, name: str) -> int:
        if not self.exists(name):
            return 0
        return int(self._client.get_collection(name).count())

    def add_documents(self, name: str, documents: list[Document]) -> int:
        """Create/replace the collection contents. Returns chunk count."""
        if not documents:
            raise ValueError("Refusing to create an empty collection")
        self._collections.pop(name, None)  # force fresh Chroma wrapper
        collection = self._collection(name)
        collection.add_documents(documents)
        return self.count(name)

    # ------------------------------------------------------------------
    # retrieval
    # ------------------------------------------------------------------

    def search(
        self, name: str, query: str, k: int
    ) -> list[tuple[Document, float]]:
        """Similarity search. Returns (document, distance) pairs.

        Raises:
            CollectionNotFoundError: the collection does not exist yet.
        """
        if not self.exists(name):
            raise CollectionNotFoundError(
                f"No collection {name!r}. Run ingest_repository first."
            )
        return self._collection(name).similarity_search_with_score(query, k=k)
