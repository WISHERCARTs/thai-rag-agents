"""Run one question through the agents and collect the answer, the sources it cited and a trace."""
import asyncio
import re
from dataclasses import dataclass, field
from typing import Optional

from google.adk.agents import BaseAgent
from google.adk.apps import App
from google.adk.runners import InMemoryRunner
from google.genai import types

from thai_agents.tools import SOURCES_KEY
from thai_agents.trace import TracePlugin

APP_NAME = "thai_agents"
CITATION = re.compile(r"\[(S\d+)\]")


@dataclass
class RunResult:
    question: str
    answer: str
    sources: dict = field(default_factory=dict)  # every passage collected, by reference id
    trace: Optional[TracePlugin] = None
    error: Optional[str] = None

    @property
    def cited(self) -> list[dict]:
        """Sources the answer actually cites, in order of first mention."""
        refs = list(dict.fromkeys(CITATION.findall(self.answer)))
        return [{"ref": ref, **self.sources[ref]} for ref in refs if ref in self.sources]

    @property
    def unknown_citations(self) -> list[str]:
        """Reference ids in the answer that no search ever returned (made up by the model)."""
        return [ref for ref in dict.fromkeys(CITATION.findall(self.answer)) if ref not in self.sources]


async def ask(root_agent: BaseAgent, question: str, *, timeout: float = 180.0, user_id: str = "user") -> RunResult:
    trace = TracePlugin()
    runner = InMemoryRunner(app=App(name=APP_NAME, root_agent=root_agent, plugins=[trace]))
    session = await runner.session_service.create_session(app_name=APP_NAME, user_id=user_id)
    message = types.Content(role="user", parts=[types.Part(text=question)])

    texts: list[str] = []

    async def consume() -> None:
        async for event in runner.run_async(user_id=user_id, session_id=session.id, new_message=message):
            if event.is_final_response() and event.content and event.content.parts:
                texts.extend(p.text for p in event.content.parts if p.text and not getattr(p, "thought", False))

    error = None
    try:
        await asyncio.wait_for(consume(), timeout=timeout)
    except asyncio.TimeoutError:
        error = f"timed out after {timeout:.0f}s"
    except Exception as exc:  # surface model/API failures as a result, not a crash
        error = f"{type(exc).__name__}: {exc}"

    stored = await runner.session_service.get_session(app_name=APP_NAME, user_id=user_id, session_id=session.id)
    sources = dict(stored.state.get(SOURCES_KEY) or {}) if stored else {}
    return RunResult(question, "".join(texts).strip(), sources, trace, error)


def ask_sync(root_agent: BaseAgent, question: str, **kwargs) -> RunResult:
    return asyncio.run(ask(root_agent, question, **kwargs))
