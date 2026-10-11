from types import SimpleNamespace

from tests.fakes import BrokenKnowledgeBase
from thai_agents.tools import make_tools


def context() -> SimpleNamespace:
    return SimpleNamespace(state={})


def test_search_returns_passages_with_reference_ids(kb):
    search, _ = make_tools(kb)
    ctx = context()

    result = search("ดอยอินทนนท์สูงกี่เมตร", ctx)

    assert result["status"] == "ok"
    top = result["passages"][0]
    assert top["ref"] == "S1"
    assert top["title"] == "อุทยานแห่งชาติดอยอินทนนท์"
    assert "2,565" in top["text"]
    assert ctx.state["sources"]["S1"]["chunk_id"] == "inthanon#0"


def test_same_passage_keeps_the_same_reference_across_searches(kb):
    search, _ = make_tools(kb)
    ctx = context()

    first = search("ดอยอินทนนท์สูงกี่เมตร", ctx)["passages"]
    second = search("ยอดเขาที่สูงที่สุดในประเทศไทย", ctx)["passages"]

    ref_by_chunk = {s["chunk_id"]: ref for ref, s in ctx.state["sources"].items()}
    assert ref_by_chunk["inthanon#0"] == "S1"
    assert [p["ref"] for p in first if p["title"].endswith("อินทนนท์")] == ["S1"]
    assert "S1" in [p["ref"] for p in second]
    # no passage is stored twice
    assert len(ctx.state["sources"]) == len(ref_by_chunk)


def test_new_passages_get_new_reference_ids(kb):
    search, _ = make_tools(kb)
    ctx = context()

    search("ดอกไม้ประจำจังหวัดน่าน", ctx)
    search("ดอกไม้ประจำจังหวัดภูเก็ต", ctx)

    refs = list(ctx.state["sources"])
    assert refs == [f"S{i}" for i in range(1, len(refs) + 1)]
    titles = {s["title"] for s in ctx.state["sources"].values()}
    assert {"จังหวัดน่าน", "จังหวัดภูเก็ต"} <= titles


def test_empty_query_is_an_error_not_a_crash(kb):
    search, _ = make_tools(kb)
    assert search("   ", context())["status"] == "error"
    assert kb.queries == []


def test_a_failing_knowledge_base_becomes_an_error_result():
    search, _ = make_tools(BrokenKnowledgeBase())
    ctx = context()

    result = search("น่าน", ctx)

    assert result["status"] == "error"
    assert "index is down" in result["message"]
    assert "sources" not in ctx.state


def test_read_sources_is_empty_before_any_search(kb):
    _, read_sources = make_tools(kb)
    assert read_sources(context())["status"] == "empty"


def test_read_sources_returns_what_was_collected(kb):
    search, read_sources = make_tools(kb)
    ctx = context()
    search("ดอกไม้ประจำจังหวัดน่าน", ctx)

    result = read_sources(ctx)

    assert result["status"] == "ok"
    assert {s["ref"] for s in result["sources"]} == set(ctx.state["sources"])
    assert any("เสี้ยวดอกขาว" in s["text"] for s in result["sources"])


def test_long_passages_are_cut():
    from thairag.chunking import Chunk
    from tests.fakes import FakeKnowledgeBase
    from thai_agents.tools import MAX_PASSAGE_CHARS

    long_kb = FakeKnowledgeBase([Chunk("x#0", "x", "ชื่อ", "", "น่าน " * 400)])
    search, _ = make_tools(long_kb)
    ctx = context()

    search("น่าน", ctx)

    assert len(ctx.state["sources"]["S1"]["text"]) == MAX_PASSAGE_CHARS
