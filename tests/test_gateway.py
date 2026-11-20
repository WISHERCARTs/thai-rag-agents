"""The agents talking to thai-llm-gateway: request translation, the full loop over HTTP, embeddings, errors."""
import json

import httpx
import numpy as np
import pytest
from google.adk.models import LlmRequest
from google.genai import types

from thai_agents.agents import build_agents, default_model
from thai_agents.gateway import RUN_ID, GatewayCallError, GatewayEmbedder, GatewayLlm, from_openai, to_openai
from thai_agents.runner import ask_sync


def tool_call(call_id, name, args):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}


def reply(content=None, calls=None, tokens=(100, 10)):
    message = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = calls
    return {
        "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if calls else "stop"}],
        "usage": {"prompt_tokens": tokens[0], "completion_tokens": tokens[1], "total_tokens": sum(tokens)},
    }


class FakeGateway:
    """Answers /v1/chat/completions by agent (read from the system prompt) from scripted replies."""

    def __init__(self, scripts):
        self.scripts = scripts
        self.requests: list[dict] = []
        self.headers: list[httpx.Headers] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.read())
        self.requests.append(body)
        self.headers.append(request.headers)
        system = body["messages"][0]["content"]
        agent = "coordinator" if "ผู้ประสานงาน" in system else "researcher" if "นักค้นข้อมูล" in system else "fact_checker"
        return httpx.Response(200, json=self.scripts[agent].pop(0))


def gateway_model(fake: FakeGateway) -> GatewayLlm:
    client = httpx.AsyncClient(transport=httpx.MockTransport(fake.handler))
    return GatewayLlm(model="gemini-2.5-flash", base_url="https://gw.test", api_key="k").use_client(client)


def test_whole_run_through_the_gateway(kb):
    fake = FakeGateway(
        {
            "coordinator": [
                reply(calls=[tool_call("c1", "researcher", {"request": "ดอยอินทนนท์สูงกี่เมตร"})]),
                reply(calls=[tool_call("c2", "fact_checker", {"request": "สูง 2,565 เมตร [S1]"})]),
                reply("ดอยอินทนนท์สูง 2,565 เมตร [S1]"),
            ],
            "researcher": [
                reply(calls=[tool_call("r1", "search_thai_wikipedia", {"query": "ความสูงของดอยอินทนนท์"})]),
                reply("สูง 2,565 เมตร [S1]"),
            ],
            "fact_checker": [reply(calls=[tool_call("f1", "read_sources", {})]), reply("OK")],
        }
    )
    agent = build_agents(kb, model=gateway_model(fake))
    token = RUN_ID.set("run-1")
    try:
        result = ask_sync(agent, "ดอยอินทนนท์สูงกี่เมตร")
    finally:
        RUN_ID.reset(token)

    assert result.error is None, result.error
    assert result.answer == "ดอยอินทนนท์สูง 2,565 เมตร [S1]"
    assert [s["title"] for s in result.cited] == ["อุทยานแห่งชาติดอยอินทนนท์"]
    assert kb.queries == ["ความสูงของดอยอินทนนท์"]
    assert len(fake.requests) == 7
    assert all(h["authorization"] == "Bearer k" and h["x-session-id"] == "run-1" for h in fake.headers)
    # the tool result went back as an OpenAI tool message matched to the call id
    second_researcher = next(r for r in fake.requests if "นักค้นข้อมูล" in r["messages"][0]["content"] and len(r["messages"]) > 2)
    tool_msg = second_researcher["messages"][-1]
    assert tool_msg["role"] == "tool" and tool_msg["tool_call_id"] == "r1" and "2,565" in tool_msg["content"]
    # token usage flows into the trace
    assert result.trace.prompt_tokens == 700 and result.trace.output_tokens == 70


def test_request_translation_tools_and_config():
    request = LlmRequest(
        model="gemini-2.5-flash",
        contents=[
            types.Content(role="user", parts=[types.Part(text="ถาม")]),
            types.Content(role="model", parts=[types.Part(function_call=types.FunctionCall(name="read_sources", args={}))]),
            types.Content(
                role="user",
                parts=[types.Part(function_response=types.FunctionResponse(name="read_sources", response={"status": "empty"}))],
            ),
            types.Content(role="model", parts=[types.Part(text="คิดอยู่", thought=True), types.Part(text="ตอบ")]),
        ],
        config=types.GenerateContentConfig(
            system_instruction="ระบบ",
            temperature=0.2,
            max_output_tokens=256,
            tools=[
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(
                            name="search",
                            description="find",
                            parameters_json_schema={"title": "P", "type": "object", "properties": {"q": {"title": "Q", "type": "string"}}},
                        ),
                        types.FunctionDeclaration(name="read_sources"),
                    ]
                )
            ],
        ),
    )
    body = to_openai(request, "gemini-2.5-flash")
    roles = [m["role"] for m in body["messages"]]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]
    call_id = body["messages"][2]["tool_calls"][0]["id"]
    assert body["messages"][3]["tool_call_id"] == call_id  # ids are made up and matched when ADK has none
    assert body["messages"][4]["content"] == "ตอบ"  # thoughts are not sent back
    assert body["tools"][0]["function"] == {
        "name": "search",
        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
        "description": "find",
    }
    assert body["tools"][1]["function"]["parameters"] == {"type": "object", "properties": {}}
    assert body["temperature"] == 0.2 and body["max_tokens"] == 256


def test_response_translation():
    response = from_openai(reply("ข้อความ", calls=[tool_call("x", "search", {"q": "น่าน"})], tokens=(5, 2)))
    text, call = response.content.parts
    assert text.text == "ข้อความ"
    assert call.function_call.id == "x" and call.function_call.args == {"q": "น่าน"}
    assert response.usage_metadata.prompt_token_count == 5 and response.usage_metadata.candidates_token_count == 2


def test_gateway_refusal_becomes_a_typed_failure(kb):
    def handler(request):
        return httpx.Response(
            429, json={"error": {"code": "daily_budget_exceeded", "message": "budget"}}, headers={"retry-after": "60"}
        )

    model = GatewayLlm(base_url="https://gw.test", api_key="k").use_client(httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    result = ask_sync(build_agents(kb, model=model), "อะไรก็ได้")
    assert isinstance(result.failure, GatewayCallError)
    assert result.failure.code == "daily_budget_exceeded" and result.failure.retry_after == "60"


def test_embedder_batches_and_normalises():
    seen = []

    def handler(request):
        body = json.loads(request.read())
        seen.append(body)
        data = [{"index": i, "embedding": [3.0, 4.0]} for i in range(len(body["input"]))]
        return httpx.Response(200, json={"data": list(reversed(data))})

    embedder = GatewayEmbedder("https://gw.test/", "k", batch_size=2, client=httpx.Client(transport=httpx.MockTransport(handler)))
    matrix = embedder.embed_documents(["a", "b", "c"])
    assert matrix.shape == (3, 2) and np.allclose(matrix[0], [0.6, 0.8])
    assert [len(b["input"]) for b in seen] == [2, 1]
    assert seen[0]["task_type"] == "RETRIEVAL_DOCUMENT" and seen[0]["dimensions"] == 768
    embedder.embed_query("น่าน")
    assert seen[-1]["task_type"] == "RETRIEVAL_QUERY"


def test_embedder_raises_gateway_errors():
    handler = lambda request: httpx.Response(401, json={"error": {"code": "invalid_api_key", "message": "no"}})  # noqa: E731
    embedder = GatewayEmbedder("https://gw.test", "bad", client=httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(GatewayCallError) as err:
        embedder.embed_query("x")
    assert err.value.status == 401 and err.value.code == "invalid_api_key"


def test_default_model_uses_the_gateway_only_when_configured(monkeypatch):
    monkeypatch.delenv("LLM_GATEWAY_URL", raising=False)
    assert default_model() == "gemini-2.5-flash"
    monkeypatch.setenv("LLM_GATEWAY_URL", "https://gw.test/")
    monkeypatch.setenv("LLM_GATEWAY_KEY", "k")
    model = default_model("gemini-2.5-flash-lite")
    assert isinstance(model, GatewayLlm) and model.base_url == "https://gw.test" and model.model == "gemini-2.5-flash-lite"
    assert "k" not in repr(model)
