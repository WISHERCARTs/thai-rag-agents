from types import SimpleNamespace

import pytest

from tests.fakes import CHUNKS, FakeEmbedder
from thairag.index import build_index
from thai_agents import cli
from thai_agents.retrieval import HybridKnowledgeBase, LazyKnowledgeBase
from thai_agents.runner import RunResult
from thai_agents.trace import TracePlugin, preview_result


# ---- trace helpers --------------------------------------------------------

@pytest.mark.parametrize(
    "result, expected",
    [
        ({"status": "ok", "passages": [1, 2, 3]}, "3 passages"),
        ({"status": "ok", "sources": [1]}, "1 sources"),
        ({"status": "error", "message": "search failed: boom"}, "search failed: boom"),
        ({"status": "empty"}, "empty"),
        ("สั้น", "สั้น"),
    ],
)
def test_preview_result(result, expected):
    assert preview_result(result) == expected


def test_preview_result_truncates_long_text():
    out = preview_result("ก" * 300, limit=20)
    assert len(out) == 20 and out.endswith("…")


def test_trace_plugin_tracks_nesting_and_errors():
    import asyncio

    plugin = TracePlugin()
    outer, inner = SimpleNamespace(name="researcher"), SimpleNamespace(name="search")

    async def run():
        await plugin.before_tool_callback(tool=outer, tool_args={"request": "q"}, tool_context=None)
        await plugin.before_tool_callback(tool=inner, tool_args={"query": "x"}, tool_context=None)
        await plugin.on_tool_error_callback(tool=inner, tool_args={}, tool_context=None, error=ValueError("bad"))
        await plugin.after_tool_callback(tool=outer, tool_args={}, tool_context=None, result="ok")

    asyncio.run(run())

    assert [(s.tool, s.depth) for s in plugin.spans] == [("researcher", 0), ("search", 1)]
    assert plugin.spans[1].error == "ValueError: bad"
    assert plugin.spans[0].result == "ok"
    assert "ERROR ValueError: bad" in plugin.render()
    assert TracePlugin().render() == "(no tool calls)"


# ---- knowledge base -------------------------------------------------------

def test_hybrid_knowledge_base_loads_a_saved_index(tmp_path):
    embedder = FakeEmbedder()
    build_index(CHUNKS, embedder, tmp_path)

    kb = HybridKnowledgeBase.from_index(tmp_path, embedder)
    hits = kb.search("ดอกไม้ประจำจังหวัดภูเก็ต", 2)

    assert hits[0].id == "phuket#1"


def test_lazy_knowledge_base_explains_a_missing_index(tmp_path, monkeypatch):
    monkeypatch.setenv("THAIRAG_INDEX", str(tmp_path / "nothing-here"))

    with pytest.raises(FileNotFoundError) as info:
        LazyKnowledgeBase().search("น่าน")

    assert "python -m thairag ingest" in str(info.value)


def test_importing_the_adk_entry_point_needs_no_key_or_index(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    import importlib

    module = importlib.import_module("thai_agents.agent")
    assert module.root_agent.name == "coordinator"


# ---- command line ---------------------------------------------------------

def fake_result(**overrides) -> RunResult:
    base = dict(
        question="ดอยอินทนนท์สูงกี่เมตร",
        answer="สูง 2,565 เมตร [S1]",
        sources={"S1": {"title": "อุทยานแห่งชาติดอยอินทนนท์", "section": "ภูมิประเทศ", "text": "..." }},
        trace=TracePlugin(),
    )
    base.update(overrides)
    return RunResult(**base)


def test_cli_prints_answer_and_sources(monkeypatch, capsys):
    monkeypatch.setattr(cli, "build_agents", lambda *a, **k: object())
    monkeypatch.setattr(cli, "ask_sync", lambda agent, q, **k: fake_result())

    code = cli.main(["ask", "ดอยอินทนนท์สูงกี่เมตร"])

    out = capsys.readouterr().out
    assert code == 0
    assert "สูง 2,565 เมตร [S1]" in out
    assert "[S1] อุทยานแห่งชาติดอยอินทนนท์ — ภูมิประเทศ" in out
    assert "Trace" not in out


def test_cli_prints_the_trace_on_request(monkeypatch, capsys):
    monkeypatch.setattr(cli, "build_agents", lambda *a, **k: object())
    monkeypatch.setattr(cli, "ask_sync", lambda agent, q, **k: fake_result())

    cli.main(["ask", "x", "--trace"])

    out = capsys.readouterr().out
    assert "Trace:" in out and "Summary:" in out


def test_cli_reports_errors_with_a_failing_exit_code(monkeypatch, capsys):
    monkeypatch.setattr(cli, "build_agents", lambda *a, **k: object())
    monkeypatch.setattr(cli, "ask_sync", lambda agent, q, **k: fake_result(answer="", sources={}, error="quota exceeded"))

    code = cli.main(["ask", "x"])

    captured = capsys.readouterr()
    assert code == 1
    assert "quota exceeded" in captured.err
    assert "(no answer)" in captured.out


def test_cli_loads_the_api_key_from_a_dotenv_file(monkeypatch, tmp_path, capsys):
    (tmp_path / ".env").write_text("GEMINI_API_KEY=from-dotenv-file\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    seen = {}

    def fake_ask(agent, question, **kwargs):
        import os

        seen["key"] = os.getenv("GEMINI_API_KEY")
        return fake_result()

    monkeypatch.setattr(cli, "build_agents", lambda *a, **k: object())
    monkeypatch.setattr(cli, "ask_sync", fake_ask)

    cli.main(["ask", "x"])

    assert seen["key"] == "from-dotenv-file"


def test_a_key_already_in_the_environment_beats_the_dotenv_file(monkeypatch, tmp_path):
    (tmp_path / ".env").write_text("GEMINI_API_KEY=from-file\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "from-shell")

    from thai_agents.env import load_env

    load_env()

    import os

    assert os.environ["GEMINI_API_KEY"] == "from-shell"
