"""Knowledge base: the hybrid (BM25 + embedding) search from thai-rag-eval over its saved index."""
import os
from pathlib import Path
from typing import Protocol, Sequence

from thairag.chunking import Chunk
from thairag.embeddings import Embedder
from thairag.index import load_index
from thairag.retrievers import BM25Retriever, HybridRetriever, VectorRetriever

DEFAULT_INDEX_DIR = Path(__file__).resolve().parents[2] / "thai-rag-eval" / "index"


class KnowledgeBase(Protocol):
    def search(self, query: str, k: int = 4) -> Sequence[Chunk]: ...


def index_dir() -> Path:
    """Where the saved index lives. Set THAIRAG_INDEX to override."""
    return Path(os.getenv("THAIRAG_INDEX") or DEFAULT_INDEX_DIR)


class HybridKnowledgeBase:
    def __init__(self, chunks: Sequence[Chunk], retriever):
        self.by_id = {c.id: c for c in chunks}
        self.retriever = retriever

    @classmethod
    def from_index(cls, directory: Path, embedder: Embedder) -> "HybridKnowledgeBase":
        chunks, matrix, _meta = load_index(directory)
        bm25 = BM25Retriever(chunks)
        vector = VectorRetriever([c.id for c in chunks], matrix, embedder)
        return cls(chunks, HybridRetriever([bm25, vector]))

    def search(self, query: str, k: int = 4) -> list[Chunk]:
        return [self.by_id[chunk_id] for chunk_id, _ in self.retriever.search(query, k)]


class LazyKnowledgeBase:
    """Loads the index and creates the Gemini embedder on first use.

    The ADK web UI imports the agent module just to list it, so importing must not need
    an API key or a built index.
    """

    def __init__(self):
        self._kb: HybridKnowledgeBase | None = None

    def search(self, query: str, k: int = 4) -> list[Chunk]:
        if self._kb is None:
            from thairag.gemini import GeminiEmbedder, make_client

            directory = index_dir()
            if not (directory / "chunks.jsonl").exists():
                raise FileNotFoundError(
                    f"No index at {directory}. Build it with `python -m thairag ingest` in thai-rag-eval, "
                    "or set THAIRAG_INDEX."
                )
            self._kb = HybridKnowledgeBase.from_index(directory, GeminiEmbedder(make_client()))
        return self._kb.search(query, k)
