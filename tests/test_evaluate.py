import json

import pytest

from tests.fakes import Call, ScriptedLlm
from thai_agents import cli
from thai_agents.agents import NOT_FOUND, build_agents
from thai_agents.evaluate import (
    Case,
    estimate_calls,
    load_cases,
    save_results,
    score_case,
    summarize,
    summary_table,
)
from thai_agents.runner import RunResult, ask_sync
from thai_agents.trace import TracePlugin


def judge(correct=True, faithful=True):
    def reply(prompt: str) -> str:
        if "faithful" in prompt:
            return json.dumps({"faithful": faithful})
        return json.dumps({"correct": correct})

    return reply


def run(kb, coordinator, researcher, checker, question="q"):
    return ask_sync(build_agents(kb, model=coordinator, researcher_model=researcher, checker_model=checker), question)


def comparison_run(kb):
    researcher = ScriptedLlm(
        steps=[
            Call("search_thai_wikipedia", {"query": "ดอกไม้น่าน"}),
            "เสี้ยวดอกขาว [S1]",
            Call("search_thai_wikipedia", {"query": "ดอกไม้ภูเก็ต"}),
            "ดอกเฟื่องฟ้า [S2]",
        ]
    )
    checker = ScriptedLlm(steps=[Call("read_sources", {}), "OK"])
    coordinator = ScriptedLlm(
        steps=[
            Call("researcher", {"request": "น่าน"}),
            Call("researcher", {"request": "ภูเก็ต"}),
            Call("fact_checker", {"request": "ร่าง"}),
            "น่านคือเสี้ยวดอกขาว [S1] ภูเก็ตคือดอกเฟื่องฟ้า [S2]",
        ]
    )
    return run(kb, coordinator, researcher, checker)


COMPARISON = Case("c1", "comparison", "ดอกไม้ประจำจังหวัดน่านกับภูเก็ต", ["จังหวัดน่าน", "จังหวัดภูเก็ต"], "น่านเสี้ยวดอกขาว ภูเก็ตเฟื่องฟ้า", 2)


def test_comparison_case_scores_documents_decomposition_and_judges(kb):
    score = score_case(COMPARISON, comparison_run(kb), judge())

    assert score.doc_recall == 1.0
    assert score.correct is True and score.faithful is True
    assert score.researcher_calls == 2 and score.decomposed
    assert score.used_fact_checker
    assert score.cited == 2 and score.unknown_citations == 0
    assert score.model_calls == 4 + 4 + 2  # coordinator 4, researcher 4 (two searches), fact checker 2
    assert score.error is None


def test_missing_a_document_lowers_recall(kb):
    case = Case("c2", "comparison", "q", ["จังหวัดน่าน", "จังหวัดภูเก็ต", "จังหวัดกระบี่"], "x", 2)
    assert score_case(case, comparison_run(kb), judge()).doc_recall == pytest.approx(2 / 3)


def test_not_enough_researcher_calls_is_flagged(kb):
    case = Case("c3", "comparison", "q", ["จังหวัดน่าน"], "x", min_researcher_calls=3)
    assert score_case(case, comparison_run(kb), judge()).decomposed is False


def test_judge_verdict_is_used_for_answerable_questions(kb):
    score = score_case(COMPARISON, comparison_run(kb), judge(correct=False, faithful=False))
    assert score.correct is False and score.faithful is False


def test_unusable_judge_reply_leaves_correctness_unknown(kb):
    score = score_case(COMPARISON, comparison_run(kb), lambda prompt: "not json")
    assert score.correct is None and score.faithful is None


def abstaining_run(kb):
    researcher = ScriptedLlm(steps=[Call("search_thai_wikipedia", {"query": "หอไอเฟล"}), NOT_FOUND])
    coordinator = ScriptedLlm(steps=[Call("researcher", {"request": "หอไอเฟล"}), NOT_FOUND])
    return run(kb, coordinator, researcher, ScriptedLlm())


def test_unanswerable_question_is_correct_only_when_the_system_refuses(kb):
    case = Case("u1", "unanswerable", "หอไอเฟลสูงกี่เมตร", [], "", 1)
    refused = score_case(case, abstaining_run(kb), judge())

    assert refused.abstained and refused.correct is True
    assert refused.doc_recall is None and refused.faithful is None

    made_up = RunResult(case.question, "สูง 330 เมตร", {}, TracePlugin())
    assert score_case(case, made_up, judge()).correct is False

    # refusing but then stating a fact with a citation is not a clean refusal
    sources = {"S1": {"title": "จังหวัดน่าน", "section": "", "text": "..."}}
    hedged = RunResult(case.question, f"{NOT_FOUND} แต่น่านมีพื้นที่มาก [S1]", sources, TracePlugin())
    assert score_case(case, hedged, judge()).correct is False


def test_refusing_an_answerable_question_counts_as_wrong(kb):
    case = Case("s1", "single", "q", ["จังหวัดน่าน"], "เสี้ยวดอกขาว", 1)
    assert score_case(case, abstaining_run(kb), judge()).correct is False


def test_an_errored_run_counts_as_wrong_and_keeps_the_error(kb):
    broken = RunResult("q", "", {}, TracePlugin(), error="quota exceeded")
    score = score_case(COMPARISON, broken, judge())

    assert score.correct is False and score.error == "quota exceeded"


def test_summary_groups_by_type(kb):
    ok = score_case(COMPARISON, comparison_run(kb), judge())
    wrong = score_case(COMPARISON, comparison_run(kb), judge(correct=False))
    unanswerable = score_case(Case("u1", "unanswerable", "x", [], "", 1), abstaining_run(kb), judge())

    summary = summarize([ok, wrong, unanswerable])

    assert summary["overall"]["questions"] == 3
    assert summary["comparison"]["correct"] == 0.5
    assert summary["comparison"]["doc_recall"] == 1.0
    assert summary["unanswerable"]["correct"] == 1.0
    table = summary_table(summary)
    assert "| comparison |" in table and "| unanswerable |" in table


def test_results_are_saved_as_jsonl_json_and_markdown(kb, tmp_path):
    score = score_case(COMPARISON, comparison_run(kb), judge())
    save_results([score], tmp_path)

    row = json.loads((tmp_path / "results.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["id"] == "c1" and row["correct"] is True
    assert json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))["overall"]["questions"] == 1
    assert (tmp_path / "summary.md").read_text(encoding="utf-8").startswith("| group |")


def test_shipped_cases_are_well_formed():
    from pathlib import Path

    cases = load_cases(Path(__file__).resolve().parents[1] / "evals" / "cases.jsonl")

    assert len(cases) == 20
    assert len({c.id for c in cases}) == 20
    assert {c.type for c in cases} == {"single", "comparison", "unanswerable", "partial"}
    for case in cases:
        if case.type == "comparison":
            assert len(case.expected_docs) == 2 and case.min_researcher_calls == 2
        if case.type == "unanswerable":
            assert case.expected_docs == [] and case.reference == ""
        else:
            assert case.reference


def test_estimate_scales_with_question_count():
    assert estimate_calls(10) == (120, 200)


# ---- the guard that stops the eval from spending money by accident ----------

def test_eval_without_yes_runs_nothing_and_touches_no_api(monkeypatch, capsys):
    def explode(*args, **kwargs):
        raise AssertionError("the eval must not create an API client without --yes")

    monkeypatch.setattr("thairag.gemini.make_client", explode)
    monkeypatch.setattr(cli, "ask_sync", explode)
    monkeypatch.setattr(cli, "build_agents", explode)

    code = cli.main(["eval", "--limit", "5"])

    out = capsys.readouterr().out
    assert code == 2
    assert "5 questions" in out and "--yes" in out and "Nothing was run" in out
