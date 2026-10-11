"""python -m thai_agents ask "คำถาม" [--trace]"""
import argparse
import sys
from pathlib import Path

from thai_agents.agents import build_agents
from thai_agents.retrieval import LazyKnowledgeBase
from thai_agents.runner import ask_sync


def cmd_eval(args) -> int:
    from thai_agents.evaluate import estimate_calls, judge_from_llm, load_cases, save_results, score_case, summarize, summary_table

    cases = load_cases(args.cases)[: args.limit]
    low, high = estimate_calls(len(cases))
    print(f"{len(cases)} questions: roughly {low}-{high} Gemini calls (generation, embeddings and judging).")
    if not args.yes:
        print("Nothing was run. These calls may be billed; re-run with --yes to start.")
        return 2

    from thairag.gemini import GeminiLLM, make_client

    judge = judge_from_llm(GeminiLLM(make_client(), temperature=0.0))
    agents = build_agents(LazyKnowledgeBase(), model=args.model)
    scores = []
    for case in cases:
        score = score_case(case, ask_sync(agents, case.question), judge)
        scores.append(score)
        flag = "ok " if score.correct else "MISS"
        print(f"{flag} {case.id} ({case.type}): researcher x{score.researcher_calls}, docs {score.doc_recall}")
    save_results(scores, args.out)
    print("\n" + summary_table(summarize(scores)))
    print(f"\nResults written to {args.out}/")
    return 0


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(prog="thai_agents", description="Ask the Thai Wikipedia agents a question")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("ask", help="answer one question")
    p.add_argument("question")
    p.add_argument("--trace", action="store_true", help="print which agent called which tool")
    p.add_argument("--model", default="gemini-2.5-flash")
    e = sub.add_parser("eval", help="score the agents on evals/cases.jsonl (calls the Gemini API)")
    e.add_argument("--cases", type=Path, default=Path("evals/cases.jsonl"))
    e.add_argument("--out", type=Path, default=Path("evals/results"))
    e.add_argument("--limit", type=int, default=1000, help="only run the first N questions")
    e.add_argument("--model", default="gemini-2.5-flash")
    e.add_argument("--yes", action="store_true", help="confirm the Gemini calls; without it nothing runs")
    args = parser.parse_args(argv)
    if args.command == "eval":
        return cmd_eval(args)

    result = ask_sync(build_agents(LazyKnowledgeBase(), model=args.model), args.question)

    if result.error:
        print(f"ERROR: {result.error}", file=sys.stderr)
    print(result.answer or "(no answer)")
    if result.cited:
        print("\nแหล่งที่มา:")
        for source in result.cited:
            where = source["title"] + (f" — {source['section']}" if source["section"] else "")
            print(f"[{source['ref']}] {where}")
    if result.unknown_citations:
        print(f"\nคำเตือน: อ้างอิงที่ไม่มีในผลค้น {', '.join(result.unknown_citations)}", file=sys.stderr)
    if args.trace and result.trace:
        print("\nTrace:\n" + result.trace.render())
        print("\nSummary:", result.trace.summary())
    return 1 if result.error else 0


if __name__ == "__main__":
    raise SystemExit(main())
