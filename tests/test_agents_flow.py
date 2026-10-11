"""The whole multi-agent loop, driven by scripted models: delegation, shared sources, tracing, failures."""
import pytest

from tests.fakes import BrokenKnowledgeBase, Call, Delay, ScriptedLlm
from thai_agents.agents import NOT_FOUND, build_agents
from thai_agents.runner import ask_sync
from thai_agents.tools import make_tools


def build(kb, coordinator, researcher, checker):
    agent = build_agents(kb, model=coordinator, researcher_model=researcher, checker_model=checker)
    return agent


def test_wiring_gives_the_coordinator_two_agents_as_tools(kb):
    agent = build(kb, ScriptedLlm(), ScriptedLlm(), ScriptedLlm())

    assert agent.name == "coordinator"
    assert [t.name for t in agent.tools] == ["researcher", "fact_checker"]
    researcher, checker = (t.agent for t in agent.tools)
    assert [t.__name__ for t in researcher.tools] == ["search_thai_wikipedia"]
    assert [t.__name__ for t in checker.tools] == ["read_sources"]
    assert NOT_FOUND in agent.instruction and NOT_FOUND in researcher.instruction


def test_single_question_flows_through_researcher_then_fact_checker(kb):
    researcher = ScriptedLlm(steps=[Call("search_thai_wikipedia", {"query": "ความสูงของดอยอินทนนท์"}), "ดอยอินทนนท์สูง 2,565 เมตร [S1]"])
    checker = ScriptedLlm(steps=[Call("read_sources", {}), "OK"])
    coordinator = ScriptedLlm(
        steps=[
            Call("researcher", {"request": "ดอยอินทนนท์สูงกี่เมตร"}),
            Call("fact_checker", {"request": "ดอยอินทนนท์สูง 2,565 เมตร [S1]"}),
            "ดอยอินทนนท์สูง 2,565 เมตร [S1]",
        ]
    )

    result = ask_sync(build(kb, coordinator, researcher, checker), "ดอยอินทนนท์สูงกี่เมตร")

    assert result.error is None
    assert result.answer == "ดอยอินทนนท์สูง 2,565 เมตร [S1]"
    assert [s["title"] for s in result.cited] == ["อุทยานแห่งชาติดอยอินทนนท์"]
    assert result.unknown_citations == []
    # the researcher was given the sub-question the coordinator wrote
    assert "ดอยอินทนนท์สูงกี่เมตร" in researcher.seen_text(0)
    # the fact checker could read the passage the researcher collected: sources are shared state
    assert "2,565" in checker.seen_text(1)
    assert kb.queries == ["ความสูงของดอยอินทนนท์"]


def test_trace_shows_who_called_what_with_nesting(kb):
    researcher = ScriptedLlm(steps=[Call("search_thai_wikipedia", {"query": "ดอยอินทนนท์"}), "สูง 2,565 เมตร [S1]"])
    checker = ScriptedLlm(steps=[Call("read_sources", {}), "OK"])
    coordinator = ScriptedLlm(
        steps=[Call("researcher", {"request": "q"}), Call("fact_checker", {"request": "draft"}), "ตอบ [S1]"]
    )

    result = ask_sync(build(kb, coordinator, researcher, checker), "ดอยอินทนนท์สูงเท่าไหร่")

    spans = [(s.tool, s.depth) for s in result.trace.spans]
    assert spans == [("researcher", 0), ("search_thai_wikipedia", 1), ("fact_checker", 0), ("read_sources", 1)]
    summary = result.trace.summary()
    assert summary["tool_calls"] == {"researcher": 1, "search_thai_wikipedia": 1, "fact_checker": 1, "read_sources": 1}
    assert summary["model_calls"] == {"coordinator": 3, "researcher": 2, "fact_checker": 2}
    assert all(s.duration is not None for s in result.trace.spans)
    rendered = result.trace.render()
    assert "- researcher('q')" in rendered
    assert "  - search_thai_wikipedia('ดอยอินทนนท์')" in rendered
    assert "1 passages" in rendered or "passages" in rendered


def test_comparison_question_uses_the_researcher_once_per_place(kb):
    researcher = ScriptedLlm(
        steps=[
            Call("search_thai_wikipedia", {"query": "ดอกไม้ประจำจังหวัดน่าน"}),
            "ดอกเสี้ยวดอกขาว [S1]",
            Call("search_thai_wikipedia", {"query": "ดอกไม้ประจำจังหวัดภูเก็ต"}),
            "ดอกเฟื่องฟ้า [S2]",
        ]
    )
    checker = ScriptedLlm(steps=[Call("read_sources", {}), "OK"])
    coordinator = ScriptedLlm(
        steps=[
            Call("researcher", {"request": "ดอกไม้ประจำจังหวัดน่านคืออะไร"}),
            Call("researcher", {"request": "ดอกไม้ประจำจังหวัดภูเก็ตคืออะไร"}),
            Call("fact_checker", {"request": "draft"}),
            "น่านคือดอกเสี้ยวดอกขาว [S1] ส่วนภูเก็ตคือดอกเฟื่องฟ้า [S2]",
        ]
    )

    result = ask_sync(build(kb, coordinator, researcher, checker), "ดอกไม้ประจำจังหวัดน่านกับภูเก็ตต่างกันอย่างไร")

    assert result.trace.tool_calls()["researcher"] == 2
    assert result.trace.tool_calls()["search_thai_wikipedia"] == 2
    assert [s["title"] for s in result.cited] == ["จังหวัดน่าน", "จังหวัดภูเก็ต"]
    # the checker sees passages from both researcher calls
    seen = checker.seen_text(1)
    assert "เสี้ยวดอกขาว" in seen and "ดอกเฟื่องฟ้า" in seen


def test_question_the_documents_cannot_answer_ends_with_not_found(kb):
    researcher = ScriptedLlm(steps=[Call("search_thai_wikipedia", {"query": "หอไอเฟลสูงกี่เมตร"}), NOT_FOUND])
    checker = ScriptedLlm(steps=[])
    coordinator = ScriptedLlm(steps=[Call("researcher", {"request": "หอไอเฟลสูงกี่เมตร"}), NOT_FOUND])

    result = ask_sync(build(kb, coordinator, researcher, checker), "หอไอเฟลสูงกี่เมตร")

    assert result.answer == NOT_FOUND
    assert result.cited == []
    assert "fact_checker" not in result.trace.tool_calls()


def test_search_failure_is_reported_to_the_researcher_and_traced():
    researcher = ScriptedLlm(steps=[Call("search_thai_wikipedia", {"query": "น่าน"}), NOT_FOUND])
    coordinator = ScriptedLlm(steps=[Call("researcher", {"request": "น่าน"}), NOT_FOUND])

    result = ask_sync(build(BrokenKnowledgeBase(), coordinator, researcher, ScriptedLlm()), "น่านมีอะไรบ้าง")

    assert result.error is None
    assert "index is down" in researcher.seen_text(1)  # the model was told the search failed
    assert "search failed" in result.trace.spans[1].result
    assert result.sources == {}


def test_a_model_error_becomes_an_error_result_not_an_exception(kb):
    coordinator = ScriptedLlm(steps=[RuntimeError("quota exceeded")])

    result = ask_sync(build(kb, coordinator, ScriptedLlm(), ScriptedLlm()), "อะไรก็ได้")

    assert result.answer == ""
    assert "quota exceeded" in result.error


def test_a_slow_model_times_out(kb):
    coordinator = ScriptedLlm(steps=[Delay(5)])

    result = ask_sync(build(kb, coordinator, ScriptedLlm(), ScriptedLlm()), "อะไรก็ได้", timeout=0.2)

    assert result.error.startswith("timed out")
    assert result.answer == ""


def test_citations_that_no_search_returned_are_flagged(kb):
    researcher = ScriptedLlm(steps=[Call("search_thai_wikipedia", {"query": "ดอยอินทนนท์"}), "สูง 2,565 เมตร [S1]"])
    checker = ScriptedLlm(steps=[Call("read_sources", {}), "OK"])
    coordinator = ScriptedLlm(
        steps=[Call("researcher", {"request": "q"}), Call("fact_checker", {"request": "d"}), "สูง 2,565 เมตร [S1] และยังมีอีก [S9]"]
    )

    result = ask_sync(build(kb, coordinator, researcher, checker), "ดอยอินทนนท์")

    assert result.unknown_citations == ["S9"]
    assert [s["ref"] for s in result.cited] == ["S1"]


def test_fact_checker_issues_reach_the_coordinator(kb):
    researcher = ScriptedLlm(steps=[Call("search_thai_wikipedia", {"query": "ดอยอินทนนท์"}), "สูง 2,565 เมตร [S1]"])
    checker = ScriptedLlm(steps=[Call("read_sources", {}), "ISSUES: ประโยคที่ 2 ไม่มีหลักฐาน"])
    coordinator = ScriptedLlm(
        steps=[Call("researcher", {"request": "q"}), Call("fact_checker", {"request": "ร่าง"}), "สูง 2,565 เมตร [S1]"]
    )

    ask_sync(build(kb, coordinator, researcher, checker), "ดอยอินทนนท์")

    assert "ISSUES" in coordinator.seen_text(2)  # the coordinator's last turn saw the checker's verdict
