"""Function tools the agents call. They share one source list through the ADK session state."""
import logging
from typing import Callable

from google.adk.tools import ToolContext

from thai_agents.retrieval import KnowledgeBase

logger = logging.getLogger(__name__)

SOURCES_KEY = "sources"
MAX_PASSAGE_CHARS = 700
RESULTS_PER_SEARCH = 4


def _view(ref: str, source: dict) -> dict:
    return {"ref": ref, "title": source["title"], "section": source["section"], "text": source["text"]}


def make_tools(kb: KnowledgeBase) -> tuple[Callable, Callable]:
    """Build the two tools around a knowledge base.

    ADK reads each tool's name, signature and docstring to describe it to the model,
    so the docstrings below are part of the prompt.
    """

    def search_thai_wikipedia(query: str, tool_context: ToolContext) -> dict:
        """Search Thai Wikipedia articles about Thai provinces, temples, parks and landmarks.

        Args:
            query: What to look for, in Thai. Name the place and the fact, for example
                "ความสูงของดอยอินทนนท์".

        Returns:
            A dict. On success: status "ok" and a list of passages, each with a reference
            id (S1, S2, ...), the article title, the section and the text. The same
            passage always keeps the same reference id. On failure: status "error" and a
            message.
        """
        query = (query or "").strip()
        if not query:
            return {"status": "error", "message": "query is empty"}
        try:
            chunks = kb.search(query, RESULTS_PER_SEARCH)
        except Exception as exc:  # a failed search must not crash the agent run
            logger.exception("search failed")
            return {"status": "error", "message": f"search failed: {exc}"}

        sources: dict = dict(tool_context.state.get(SOURCES_KEY) or {})
        by_chunk = {s["chunk_id"]: ref for ref, s in sources.items()}
        passages = []
        for chunk in chunks:
            ref = by_chunk.get(chunk.id)
            if ref is None:
                ref = f"S{len(sources) + 1}"
                sources[ref] = {
                    "chunk_id": chunk.id,
                    "doc_id": chunk.doc_id,
                    "title": chunk.title,
                    "section": chunk.section,
                    "text": chunk.text[:MAX_PASSAGE_CHARS],
                }
                by_chunk[chunk.id] = ref
            passages.append(_view(ref, sources[ref]))
        tool_context.state[SOURCES_KEY] = sources
        return {"status": "ok", "passages": passages}

    def read_sources(tool_context: ToolContext) -> dict:
        """Return every passage the researcher has collected so far, with reference ids.

        Returns:
            A dict with status "ok" and the list of sources, or status "empty" when
            nothing has been collected yet.
        """
        sources = tool_context.state.get(SOURCES_KEY) or {}
        if not sources:
            return {"status": "empty", "message": "no sources collected yet"}
        return {"status": "ok", "sources": [_view(ref, s) for ref, s in sources.items()]}

    return search_thai_wikipedia, read_sources
