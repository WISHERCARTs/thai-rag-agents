"""The live "ask it yourself" demo: POST /ask runs the agents on one question and returns a RecordedRun.

Run it locally with `uvicorn thai_agents.server:main_app --factory --port 8081`. On Cloud Run, every model call and
query embedding goes through thai-llm-gateway (LLM_GATEWAY_URL, LLM_GATEWAY_KEY).
"""
import asyncio
import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from thai_agents.agents import build_agents, default_model
from thai_agents.gateway import RUN_ID, GatewayCallError, GatewayLlm
from thai_agents.limits import Counter, DemoLimits, MemoryCounter, UpstashCounter, visitor_id
from thai_agents.live import recorded_run
from thai_agents.retrieval import LazyKnowledgeBase
from thai_agents.runner import ask

log = logging.getLogger("thai_agents.server")

MIN_CHARS, MAX_CHARS = 3, 300


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def client_ip(request: Request) -> str:
    """The visitor's address. Cloud Run's front end appends it as the last X-Forwarded-For entry;
    anything before it was sent by the client and can be made up."""
    forwarded = request.headers.get("x-forwarded-for", "")
    last = forwarded.split(",")[-1].strip()
    return last or (request.client.host if request.client else "unknown")


def error(status: int, code: str, message: str, retry_after: Optional[int] = None) -> JSONResponse:
    headers = {"retry-after": str(retry_after)} if retry_after else None
    return JSONResponse({"error": {"code": code, "message": message}}, status_code=status, headers=headers)


def _failure_response(failure: BaseException) -> JSONResponse:
    if isinstance(failure, asyncio.TimeoutError):
        return error(504, "timeout", "The agents took too long on this question. Try a shorter one.")
    if isinstance(failure, GatewayCallError):
        if failure.code == "prompt_injection":
            return error(400, "blocked", "The gateway's guardrail blocked this question as a prompt-injection attempt.")
        if failure.status == 429:
            retry = int(failure.retry_after) if (failure.retry_after or "").isdigit() else None
            return error(429, failure.code, "The model gateway is at its limit right now. Try again later.", retry)
    return error(502, "agent_failed", "The agents could not answer this question. Try again in a minute.")


def create_app(
    *,
    root_agent=None,
    knowledge_base: Optional[LazyKnowledgeBase] = None,
    counter: Optional[Counter] = None,
    timeout: Optional[float] = None,
    preload: bool = True,
) -> FastAPI:
    kb = knowledge_base if knowledge_base is not None else LazyKnowledgeBase()
    http = httpx.AsyncClient(timeout=90.0)
    if root_agent is None:
        model = default_model(os.getenv("AGENT_MODEL", "gemini-2.5-flash"))
        if isinstance(model, GatewayLlm):
            model.use_client(http)
        root_agent = build_agents(kb, model=model)
    if counter is None:
        url, token = os.getenv("UPSTASH_REDIS_REST_URL", ""), os.getenv("UPSTASH_REDIS_REST_TOKEN", "")
        counter = UpstashCounter(url, token) if url and token else MemoryCounter()
    limits = DemoLimits(
        counter,
        per_hour=_int("ASK_PER_VISITOR_HOUR", 5),
        per_day=_int("ASK_PER_VISITOR_DAY", 15),
        site_per_day=_int("ASK_PER_DAY", 300),
    )
    salt = os.getenv("USER_HASH_SALT", "")
    run_timeout = timeout if timeout is not None else float(_int("RUN_TIMEOUT_SECONDS", 120))
    slots = asyncio.Semaphore(_int("MAX_CONCURRENT_RUNS", 4))

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if preload and hasattr(kb, "load"):
            # Load the index (and BM25) before the first visitor, not during their question.
            await asyncio.to_thread(kb.load)
        yield
        await http.aclose()
        if hasattr(counter, "aclose"):
            await counter.aclose()

    app = FastAPI(title="thai-rag-agents live demo", lifespan=lifespan)
    origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
    if origins:
        app.add_middleware(
            CORSMiddleware, allow_origins=origins, allow_methods=["POST"], allow_headers=["content-type"], max_age=3600
        )

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.post("/ask")
    async def ask_route(request: Request):
        try:
            body = await request.json()
        except ValueError:
            return error(400, "invalid_json", "Send JSON like {\"question\": \"...\"}")
        question = body.get("question") if isinstance(body, dict) else None
        if not isinstance(question, str) or not MIN_CHARS <= len(question.strip()) <= MAX_CHARS:
            return error(400, "invalid_question", f"The question must be {MIN_CHARS} to {MAX_CHARS} characters.")
        question = question.strip()

        try:
            refusal = await limits.take(visitor_id(client_ip(request), salt))
        except Exception:
            log.exception("rate-limit store failed")
            return error(503, "unavailable", "The demo is unavailable for a moment. Try again shortly.")
        if refusal:
            return error(429, "rate_limited", refusal.message, refusal.retry_after)

        if slots.locked():
            return error(503, "busy", "The demo is answering other questions. Try again in a minute.", 30)
        run_id = uuid.uuid4().hex
        async with slots:
            token = RUN_ID.set(run_id)
            try:
                result = await ask(root_agent, question, timeout=run_timeout)
            finally:
                RUN_ID.reset(token)
        if result.failure is not None or result.error:
            log.warning("run %s failed: %s", run_id, result.error)
            return _failure_response(result.failure or RuntimeError(result.error))
        return JSONResponse(recorded_run(result, run_id), headers={"x-run-id": run_id})

    return app


def main_app() -> FastAPI:
    from thai_agents.env import load_env

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    load_env()
    return create_app()

