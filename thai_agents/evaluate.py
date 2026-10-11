"""Score the agents on a fixed set of questions: right documents, right answer, honest abstention, tool use."""
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

from thairag.evaluate import JUDGE_SYSTEM, correctness_prompt, faithfulness_prompt, parse_judgement
from thairag.metrics import mean

from thai_agents.agents import NOT_FOUND
from thai_agents.runner import RunResult

Judge = Callable[[str], str]  # prompt in, JSON text out

# One question costs roughly this many Gemini calls: the coordinator, researcher and fact checker
# each take a few turns, plus one or two query embeddings and two judge calls.
CALLS_PER_QUESTION = (12, 20)


@dataclass(frozen=True)
class Case:
    id: str
    type: str  # single, comparison, unanswerable or partial
    question: str
    expected_docs: list[str]
    reference: str
    min_researcher_calls: int = 1


@dataclass
class CaseScore:
    id: str
    type: str
    question: str
    answer: str
    abstained: bool
    doc_recall: Optional[float]
    correct: Optional[bool]
    faithful: Optional[bool]
    cited: int
    unknown_citations: int
    researcher_calls: int
    decomposed: bool
    used_fact_checker: bool
    model_calls: int
    tokens: int
    seconds: float
    error: Optional[str]
    search_queries: list  # what the researcher searched for, in order
    sources: dict  # every passage collected, by reference id, for checking the judge by hand


def load_cases(path: Path) -> list[Case]:
    with open(path, encoding="utf-8") as f:
        return [Case(**json.loads(line)) for line in f if line.strip()]


def estimate_calls(n_questions: int) -> tuple[int, int]:
    low, high = CALLS_PER_QUESTION
    return n_questions * low, n_questions * high


def sources_text(result: RunResult) -> str:
    return "\n\n".join(
        f"[{s['ref']}] {s['title']}" + (f" — {s['section']}" if s["section"] else "") + f"\n{s['text']}" for s in result.cited
    )


def score_case(case: Case, result: RunResult, judge: Judge) -> CaseScore:
    answer = result.answer
    abstained = NOT_FOUND in answer
    trace = result.trace
    calls = trace.tool_calls() if trace else {}

    titles = {s["title"] for s in result.sources.values()}
    doc_recall = (
        sum(doc in titles for doc in case.expected_docs) / len(case.expected_docs) if case.expected_docs else None
    )

    if case.type == "unanswerable":
        # Right answer is to say so. Facts next to the refusal would mean it made something up.
        correct: Optional[bool] = abstained and not result.cited
    elif answer and not result.error:
        correct = parse_judgement(
            judge(correctness_prompt(case.question, case.reference, answer)), "correct"
        )
    else:
        correct = False
    if case.type in ("single", "comparison") and abstained:
        correct = False

    faithful = None
    if result.cited and not result.error:
        faithful = parse_judgement(judge(faithfulness_prompt(sources_text(result), answer)), "faithful")

    return CaseScore(
        id=case.id,
        type=case.type,
        question=case.question,
        answer=answer,
        abstained=abstained,
        doc_recall=doc_recall,
        correct=correct,
        faithful=faithful,
        cited=len(result.cited),
        unknown_citations=len(result.unknown_citations),
        researcher_calls=calls.get("researcher", 0),
        decomposed=calls.get("researcher", 0) >= case.min_researcher_calls,
        used_fact_checker=calls.get("fact_checker", 0) > 0,
        model_calls=sum(trace.model_calls.values()) if trace else 0,
        tokens=(trace.prompt_tokens + trace.output_tokens) if trace else 0,
        seconds=trace.summary()["seconds"] if trace else 0.0,
        error=result.error,
        search_queries=[str(sp.args.get("query", "")) for sp in (trace.spans if trace else []) if sp.tool == "search_thai_wikipedia"],
        sources=result.sources,
    )


def _rate(values: Sequence[Optional[bool]]) -> Optional[float]:
    known = [v for v in values if v is not None]
    return round(mean([float(v) for v in known]), 3) if known else None


def summarize(scores: Sequence[CaseScore]) -> dict:
    """Overall numbers plus one row per question type."""

    def block(rows: Sequence[CaseScore]) -> dict:
        recalls = [r.doc_recall for r in rows if r.doc_recall is not None]
        return {
            "questions": len(rows),
            "correct": _rate([r.correct for r in rows]),
            "faithful": _rate([r.faithful for r in rows]),
            "doc_recall": round(mean(recalls), 3) if recalls else None,
            "decomposed": _rate([r.decomposed for r in rows]),
            "used_fact_checker": _rate([r.used_fact_checker for r in rows]),
            "unknown_citations": sum(r.unknown_citations for r in rows),
            "errors": sum(1 for r in rows if r.error),
            "avg_model_calls": round(mean([r.model_calls for r in rows]), 1) if rows else 0,
            "avg_seconds": round(mean([r.seconds for r in rows]), 1) if rows else 0,
        }

    types = list(dict.fromkeys(r.type for r in scores))
    return {"overall": block(scores), **{t: block([r for r in scores if r.type == t]) for t in types}}


def summary_table(summary: dict) -> str:
    cols = ["questions", "correct", "faithful", "doc_recall", "decomposed", "used_fact_checker", "unknown_citations", "errors", "avg_model_calls", "avg_seconds"]
    lines = ["| group | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for name, row in summary.items():
        cells = ["–" if row[c] is None else str(row[c]) for c in cols]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def save_results(scores: Sequence[CaseScore], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.jsonl", "w", encoding="utf-8") as f:
        for score in scores:
            f.write(json.dumps(asdict(score), ensure_ascii=False) + "\n")
    summary = summarize(scores)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "summary.md").write_text(summary_table(summary) + "\n", encoding="utf-8")


def judge_from_llm(llm) -> Judge:
    """Adapt a thairag-style LLM (generate(prompt, system, json_mode)) to a Judge."""
    return lambda prompt: llm.generate(prompt, JUDGE_SYSTEM, json_mode=True)
