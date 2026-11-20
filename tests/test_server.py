"""The live demo endpoint: RecordedRun shape, limits, failures, CORS."""
import asyncio

import pytest
from fastapi.testclient import TestClient

from tests.fakes import Call, Delay, ScriptedLlm
from thai_agents.agents import build_agents
from thai_agents.gateway import GatewayCallError
from thai_agents.limits import DemoLimits, MemoryCounter
from thai_agents.server import create_app

QUESTION = "ดอยอินทนนท์สูงกี่เมตร"


def scripted_agents(kb):
    researcher = ScriptedLlm(steps=[Call("search_thai_wikipedia", {"query": "ความสูงของดอยอินทนนท์"}), "สูง 2,565 เมตร [S1]"])
    checker = ScriptedLlm(steps=[Call("read_sources", {}), "OK"])
    coordinator = ScriptedLlm(
        steps=[
            Call("researcher", {"request": QUESTION}),
            Call("fact_checker", {"request": "สูง 2,565 เมตร [S1]"}),
            "ดอยอินทนนท์สูง 2,565 เมตร [S1]",
        ]
    )
    return build_agents(kb, model=coordinator, researcher_model=researcher, checker_model=checker)


def client(kb, agent=None, **kwargs):
    app = create_app(root_agent=agent or scripted_agents(kb), knowledge_base=kb, counter=MemoryCounter(), preload=False, **kwargs)
    return TestClient(app)


def test_answer_comes_back_as_a_recorded_run(kb):
    r = client(kb).post("/ask", json={"question": f"  {QUESTION}  "})
    assert r.status_code == 200, r.text
    run = r.json()
    assert run["id"] == r.headers["x-run-id"] and run["type"] == "live"
    assert run["question"] == QUESTION and run["answer"] == "ดอยอินทนนท์สูง 2,565 เมตร [S1]"
    assert run["reference"] == "" and run["correct"] is None and run["abstained"] is False
    assert run["researcherCalls"] == 1 and run["usedFactChecker"] is True and run["modelCalls"] == 7
    assert run["searchQueries"] == ["ความสูงของดอยอินทนนท์"]
    cited = [s for s in run["sources"] if s["cited"]]
    assert [s["title"] for s in cited] == ["อุทยานแห่งชาติดอยอินทนนท์"] and "2,565" in cited[0]["text"]
    assert set(run) == {
        "id", "type", "question", "reference", "answer", "correct", "abstained", "researcherCalls",
        "usedFactChecker", "modelCalls", "tokens", "seconds", "searchQueries", "sources",
    }


@pytest.mark.parametrize("body", [{"question": "ab"}, {"question": "ก" * 301}, {"question": 5}, {}, []])
def test_bad_questions_are_refused_without_running(kb, body):
    agent = build_agents(kb, model=ScriptedLlm())  # would fail if it were called
    r = client(kb, agent).post("/ask", json=body)
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_question"


def test_visitor_limit_uses_the_real_client_address(kb, monkeypatch):
    monkeypatch.setenv("ASK_PER_VISITOR_HOUR", "1")
    c = client(kb, build_agents(kb, model=ScriptedLlm(steps=["ไม่พบข้อมูลในเอกสาร"] * 3)))
    first = c.post("/ask", json={"question": QUESTION}, headers={"x-forwarded-for": "6.6.6.6, 1.2.3.4"})
    assert first.status_code == 200 and first.json()["abstained"] is True
    # a made-up first entry does not give a new allowance: the last entry is the address Cloud Run saw
    again = c.post("/ask", json={"question": QUESTION}, headers={"x-forwarded-for": "7.7.7.7, 1.2.3.4"})
    assert again.status_code == 429 and int(again.headers["retry-after"]) > 0
    other = c.post("/ask", json={"question": QUESTION}, headers={"x-forwarded-for": "5.6.7.8"})
    assert other.status_code == 200


@pytest.mark.parametrize(
    "failure, status, code",
    [
        (GatewayCallError(400, "prompt_injection", "x"), 400, "blocked"),
        (GatewayCallError(429, "daily_budget_exceeded", "x", "120"), 429, "daily_budget_exceeded"),
        (GatewayCallError(502, "upstream_unavailable", "x"), 502, "agent_failed"),
    ],
)
def test_gateway_failures_map_to_clear_errors(kb, failure, status, code):
    r = client(kb, build_agents(kb, model=ScriptedLlm(steps=[failure]))).post("/ask", json={"question": QUESTION})
    assert r.status_code == status and r.json()["error"]["code"] == code
    if status == 429:
        assert r.headers["retry-after"] == "120"


def test_slow_runs_time_out(kb):
    agent = build_agents(kb, model=ScriptedLlm(steps=[Delay(1.0)]))
    r = client(kb, agent, timeout=0.1).post("/ask", json={"question": QUESTION})
    assert r.status_code == 504 and r.json()["error"]["code"] == "timeout"


def test_cors_only_for_allowed_origins(kb, monkeypatch):
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://portfolio.test")
    c = client(kb)
    ok = c.options("/ask", headers={"origin": "https://portfolio.test", "access-control-request-method": "POST"})
    assert ok.headers.get("access-control-allow-origin") == "https://portfolio.test"
    bad = c.options("/ask", headers={"origin": "https://evil.test", "access-control-request-method": "POST"})
    assert "access-control-allow-origin" not in bad.headers
    assert c.get("/health").json() == {"status": "ok"}


def test_site_wide_daily_cap():
    limits = DemoLimits(MemoryCounter(), per_hour=10, per_day=10, site_per_day=2, clock=lambda: 1_700_000_000)
    assert asyncio.run(limits.take("a")) is None
    assert asyncio.run(limits.take("b")) is None
    refusal = asyncio.run(limits.take("c"))
    assert refusal and "today" in refusal.message
