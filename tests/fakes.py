"""Stand-ins for Gemini and the knowledge base, so the agent wiring can be tested without an API key."""
import asyncio
import hashlib
from dataclasses import dataclass
from typing import AsyncGenerator

import numpy as np
from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.genai import types
from pydantic import Field

from thairag.chunking import Chunk
from thairag.embeddings import normalize
from thairag.retrievers import BM25Retriever
from thairag.tokenize import tokenize


@dataclass
class Call:
    """Ask the model to call a tool."""

    name: str
    args: dict


@dataclass
class Delay:
    """Sleep, then give `then` (used to test timeouts)."""

    seconds: float
    then: str = "late"


class ScriptedLlm(BaseLlm):
    """Plays back one scripted step per model call: text, a tool Call, an Exception to raise, or a Delay.

    Every request it receives is kept in `requests`, so tests can check what an agent was shown.
    """

    model: str = "scripted"
    steps: list = Field(default_factory=list)
    requests: list = Field(default_factory=list)

    @classmethod
    def supported_models(cls) -> list[str]:
        return ["scripted"]

    async def generate_content_async(self, llm_request: LlmRequest, stream: bool = False) -> AsyncGenerator[LlmResponse, None]:
        if not self.steps:
            raise AssertionError("ScriptedLlm ran out of steps: the agent called the model more often than scripted")
        self.requests.append(llm_request)
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        if isinstance(step, Delay):
            await asyncio.sleep(step.seconds)
            step = step.then
        if isinstance(step, Call):
            part = types.Part(function_call=types.FunctionCall(name=step.name, args=step.args))
        else:
            part = types.Part(text=step)
        yield LlmResponse(content=types.Content(role="model", parts=[part]))

    def seen_text(self, request_index: int) -> str:
        """All text and tool results shown to the model in one request, flattened for assertions."""
        out = []
        for content in self.requests[request_index].contents:
            for part in content.parts or []:
                if part.text:
                    out.append(part.text)
                if part.function_response:
                    out.append(str(part.function_response.response))
        return "\n".join(out)


class FakeKnowledgeBase:
    """Real BM25 search over a handful of chunks."""

    def __init__(self, chunks: list[Chunk]):
        self.by_id = {c.id: c for c in chunks}
        self.bm25 = BM25Retriever(chunks)
        self.queries: list[str] = []

    def search(self, query: str, k: int = 4) -> list[Chunk]:
        self.queries.append(query)
        return [self.by_id[i] for i, _ in self.bm25.search(query, k)]


class BrokenKnowledgeBase:
    def search(self, query: str, k: int = 4):
        raise RuntimeError("index is down")


class FakeEmbedder:
    """Hashed bag-of-words vectors, enough to build and query an index offline."""

    def __init__(self, dim: int = 64):
        self.dim = dim

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for token in tokenize(text):
            vec[int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim] += 1.0
        return normalize(vec)

    def embed_documents(self, texts):
        return np.stack([self._vector(t) for t in texts])

    def embed_query(self, text):
        return self._vector(text)


CHUNKS = [
    Chunk("nan#0", "nan", "จังหวัดน่าน", "ภูมิประเทศ", "จังหวัดน่านมีพื้นที่ประมาณ 12,130 ตารางกิโลเมตร ตั้งอยู่ในภาคเหนือของประเทศไทย"),
    Chunk("nan#1", "nan", "จังหวัดน่าน", "สัญลักษณ์", "ดอกไม้ประจำจังหวัดน่านคือดอกเสี้ยวดอกขาว"),
    Chunk("phuket#0", "phuket", "จังหวัดภูเก็ต", "ภูมิศาสตร์", "จังหวัดภูเก็ตเป็นเกาะที่ใหญ่ที่สุดของประเทศไทย ตั้งอยู่ในทะเลอันดามัน"),
    Chunk("phuket#1", "phuket", "จังหวัดภูเก็ต", "สัญลักษณ์", "ดอกไม้ประจำจังหวัดภูเก็ตคือดอกเฟื่องฟ้า"),
    Chunk("inthanon#0", "inthanon", "อุทยานแห่งชาติดอยอินทนนท์", "ภูมิประเทศ", "ดอยอินทนนท์สูง 2,565 เมตร เป็นยอดเขาที่สูงที่สุดในประเทศไทย"),
]
