"""Turn one agent run into the `RecordedRun` shape the portfolio's AgentReplay component plays.

The recorded replays come from the evaluation, which also has a reference answer and a verdict.
A live question has neither, so `reference` is empty and `correct` is null.
"""
from thai_agents.agents import NOT_FOUND
from thai_agents.runner import RunResult

PREVIEW = 160  # passages the answer does not cite are cut to a preview, as in the recorded replays


def recorded_run(result: RunResult, run_id: str) -> dict:
    trace = result.trace
    calls = trace.tool_calls() if trace else {}
    cited = {s["ref"] for s in result.cited}
    sources = []
    for ref, s in result.sources.items():
        text = s["text"]
        if ref not in cited and len(text) > PREVIEW:
            text = text[:PREVIEW].rstrip() + "…"
        sources.append({"ref": ref, "title": s["title"], "section": s["section"], "text": text, "cited": ref in cited})
    return {
        "id": run_id,
        "type": "live",
        "question": result.question,
        "reference": "",
        "answer": result.answer,
        "correct": None,
        "abstained": NOT_FOUND in result.answer,
        "researcherCalls": calls.get("researcher", 0),
        "usedFactChecker": calls.get("fact_checker", 0) > 0,
        "modelCalls": sum(trace.model_calls.values()) if trace else 0,
        "tokens": (trace.prompt_tokens + trace.output_tokens) if trace else 0,
        "seconds": trace.summary()["seconds"] if trace else 0.0,
        "searchQueries": [
            str(sp.args.get("query", "")) for sp in (trace.spans if trace else []) if sp.tool == "search_thai_wikipedia"
        ],
        "sources": sources,
    }
