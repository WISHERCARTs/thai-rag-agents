"""Call Gemini through thai-llm-gateway instead of directly.

The gateway (https://github.com/WISHERCARTs/thai-llm-gateway) speaks the OpenAI chat API and adds
rate limits, a daily budget, PII redaction, prompt-injection checks, a cost log and traces. ADK talks
to Gemini in its own request format, so `GatewayLlm` translates an ADK `LlmRequest` into an OpenAI
chat request and the reply back. `GatewayEmbedder` does the same for query embeddings.

Set LLM_GATEWAY_URL and LLM_GATEWAY_KEY to use it; without them the agents call Gemini directly.
"""
import json
import os
from contextvars import ContextVar
from typing import Any, AsyncGenerator, Optional

import httpx
import numpy as np
from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.genai import types
from pydantic import Field, PrivateAttr

from thairag.embeddings import normalize

# Set per question by the server, so the gateway groups that question's calls into one Langfuse session.
RUN_ID: ContextVar[Optional[str]] = ContextVar("run_id", default=None)

FINISH = {
    "stop": types.FinishReason.STOP,
    "tool_calls": types.FinishReason.STOP,
    "length": types.FinishReason.MAX_TOKENS,
    "content_filter": types.FinishReason.SAFETY,
}


class GatewayCallError(RuntimeError):
    """The gateway refused or failed a call. `code` is the gateway's error code, e.g. rate_limited."""

    def __init__(self, status: int, code: str, message: str, retry_after: Optional[str] = None):
        super().__init__(f"{status} {code}: {message}")
        self.status, self.code, self.message, self.retry_after = status, code, message, retry_after


def gateway_settings() -> Optional[tuple[str, str]]:
    url, key = os.getenv("LLM_GATEWAY_URL", "").strip(), os.getenv("LLM_GATEWAY_KEY", "").strip()
    return (url.rstrip("/"), key) if url and key else None


def _headers(key: str) -> dict[str, str]:
    headers = {"Authorization": f"Bearer {key}"}
    run_id = RUN_ID.get()
    if run_id:
        headers["x-session-id"] = run_id
    return headers


def _raise_for(resp: httpx.Response) -> None:
    if resp.status_code < 400:
        return
    try:
        err = resp.json().get("error") or {}
    except ValueError:
        err = {}
    raise GatewayCallError(
        resp.status_code,
        str(err.get("code") or "gateway_error"),
        str(err.get("message") or resp.text[:200]),
        resp.headers.get("retry-after"),
    )


# ---- ADK request -> OpenAI chat request ----


def _clean_schema(schema: Any) -> Any:
    """JSON schema without the pydantic titles, which add nothing for the model."""
    if isinstance(schema, dict):
        return {k: _clean_schema(v) for k, v in schema.items() if k != "title"}
    if isinstance(schema, list):
        return [_clean_schema(v) for v in schema]
    return schema


def _schema(decl: types.FunctionDeclaration) -> dict:
    if decl.parameters_json_schema:
        return _clean_schema(decl.parameters_json_schema)
    if decl.parameters:
        return _clean_schema(json.loads(decl.parameters.model_dump_json(exclude_none=True).lower()))
    return {"type": "object", "properties": {}}


def _tools(config: Optional[types.GenerateContentConfig]) -> list[dict]:
    out = []
    for tool in (config.tools if config else None) or []:
        for decl in getattr(tool, "function_declarations", None) or []:
            fn = {"name": decl.name, "parameters": _schema(decl)}
            if decl.description:
                fn["description"] = decl.description
            out.append({"type": "function", "function": fn})
    return out


def _system_text(instruction: Any) -> str:
    if instruction is None:
        return ""
    if isinstance(instruction, str):
        return instruction
    parts = getattr(instruction, "parts", None) or []
    return "\n".join(p.text for p in parts if getattr(p, "text", None))


def to_openai(request: LlmRequest, model: str) -> dict:
    """Translate an ADK request. Function calls without ids get stable ones so results can be matched."""
    messages: list[dict] = []
    system = _system_text(request.config.system_instruction if request.config else None)
    if system:
        messages.append({"role": "system", "content": system})

    pending: list[tuple[str, str]] = []  # (name, id) of calls waiting for a result, for calls without ids
    counter = 0
    for content in request.contents or []:
        texts, calls = [], []
        for part in content.parts or []:
            if getattr(part, "thought", False):
                continue
            if part.text:
                texts.append(part.text)
            if part.function_call:
                fc = part.function_call
                counter += 1
                call_id = fc.id or f"call_{counter}"
                pending.append((fc.name, call_id))
                calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": fc.name, "arguments": json.dumps(fc.args or {}, ensure_ascii=False)},
                    }
                )
            if part.function_response:
                fr = part.function_response
                call_id = fr.id
                if not call_id:
                    match = next((i for i, (name, _) in enumerate(pending) if name == fr.name), None)
                    call_id = pending.pop(match)[1] if match is not None else f"call_{fr.name}"
                else:
                    pending = [p for p in pending if p[1] != call_id]
                messages.append(
                    {"role": "tool", "tool_call_id": call_id, "content": json.dumps(fr.response, ensure_ascii=False, default=str)}
                )
        if content.role == "model":
            if texts or calls:
                msg: dict = {"role": "assistant", "content": "\n".join(texts) or None}
                if calls:
                    msg["tool_calls"] = calls
                messages.append(msg)
        elif texts:
            messages.append({"role": "user", "content": "\n".join(texts)})

    body: dict = {"model": model, "messages": messages}
    tools = _tools(request.config)
    if tools:
        body["tools"] = tools
    config = request.config
    if config is not None:
        if config.temperature is not None:
            body["temperature"] = config.temperature
        if config.top_p is not None:
            body["top_p"] = config.top_p
        if config.max_output_tokens is not None:
            body["max_tokens"] = config.max_output_tokens
    return body


# ---- OpenAI chat response -> ADK response ----


def from_openai(data: dict) -> LlmResponse:
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    parts: list[types.Part] = []
    if message.get("content"):
        parts.append(types.Part(text=message["content"]))
    for call in message.get("tool_calls") or []:
        fn = call.get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {}
        parts.append(types.Part(function_call=types.FunctionCall(id=call.get("id") or None, name=fn.get("name"), args=args)))
    usage = data.get("usage") or {}
    metadata = None
    if usage:
        metadata = types.GenerateContentResponseUsageMetadata(
            prompt_token_count=usage.get("prompt_tokens"),
            candidates_token_count=usage.get("completion_tokens"),
            total_token_count=usage.get("total_tokens"),
        )
    return LlmResponse(
        content=types.Content(role="model", parts=parts) if parts else None,
        usage_metadata=metadata,
        finish_reason=FINISH.get(choice.get("finish_reason") or "stop", types.FinishReason.OTHER),
    )


class GatewayLlm(BaseLlm):
    """An ADK model that sends every call through the gateway. `model` is the Gemini model name."""

    model: str = "gemini-2.5-flash"
    base_url: str
    api_key: str = Field(repr=False)
    timeout: float = 90.0
    _client: Optional[httpx.AsyncClient] = PrivateAttr(default=None)

    @classmethod
    def supported_models(cls) -> list[str]:
        return []  # created directly, never looked up by name

    def use_client(self, client: httpx.AsyncClient) -> "GatewayLlm":
        self._client = client
        return self

    async def generate_content_async(self, llm_request: LlmRequest, stream: bool = False) -> AsyncGenerator[LlmResponse, None]:
        body = to_openai(llm_request, self.model)
        url = f"{self.base_url}/v1/chat/completions"
        if self._client is not None:
            resp = await self._client.post(url, json=body, headers=_headers(self.api_key))
        else:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(url, json=body, headers=_headers(self.api_key))
        _raise_for(resp)
        yield from_openai(resp.json())


class GatewayEmbedder:
    """thairag's Embedder interface, through the gateway's /v1/embeddings (with Gemini task types)."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str = "gemini-embedding-001",
        dim: int = 768,
        batch_size: int = 50,
        timeout: float = 60.0,
        client: Optional[httpx.Client] = None,
    ):
        self.base_url, self.model, self.dim, self.batch_size = base_url.rstrip("/"), model, dim, batch_size
        self._key = api_key
        self._client = client or httpx.Client(timeout=timeout)

    def _embed(self, texts: list[str], task_type: str) -> np.ndarray:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            body = {"model": self.model, "input": batch, "task_type": task_type, "dimensions": self.dim}
            resp = self._client.post(f"{self.base_url}/v1/embeddings", json=body, headers=_headers(self._key))
            _raise_for(resp)
            data = sorted(resp.json()["data"], key=lambda d: d["index"])
            vectors.extend(d["embedding"] for d in data)
        # Same as thairag's GeminiEmbedder: truncated Gemini embeddings are not unit length.
        return normalize(np.asarray(vectors, dtype=np.float32))

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._embed(list(texts), "RETRIEVAL_DOCUMENT")

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._embed(list(texts), "RETRIEVAL_QUERY")

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_queries([text])[0]
