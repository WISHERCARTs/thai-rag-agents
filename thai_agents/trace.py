"""A small ADK plugin that records who called what: model calls, tool calls and agent hand-offs."""
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

from google.adk.plugins import BasePlugin


@dataclass
class ToolSpan:
    tool: str
    args: dict
    depth: int
    started: float
    duration: Optional[float] = None
    result: Optional[str] = None
    error: Optional[str] = None


def preview_result(result: Any, limit: int = 80) -> str:
    """One short line describing a tool result."""
    if isinstance(result, dict):
        if isinstance(result.get("passages"), list):
            return f"{len(result['passages'])} passages"
        if isinstance(result.get("sources"), list):
            return f"{len(result['sources'])} sources"
        if "status" in result:
            return str(result.get("message") or result["status"])[:limit]
    text = str(result).replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "…"


class TracePlugin(BasePlugin):
    """Attach to a runner with `plugins=[TracePlugin()]`. Read `spans`, `summary()` and `render()` afterwards.

    An agent used as a tool (AgentTool) shows up as a tool span whose children are the tool
    calls its own agent made, so the output reads as a call tree.
    """

    def __init__(self) -> None:
        super().__init__(name="trace")
        self.spans: list[ToolSpan] = []
        self.model_calls: Counter[str] = Counter()
        self.prompt_tokens = 0
        self.output_tokens = 0
        self._open: list[ToolSpan] = []
        self._started = time.perf_counter()

    # ---- ADK callbacks (none of them change the run: they all return None) ----

    async def before_model_callback(self, *, callback_context, llm_request):
        self.model_calls[callback_context.agent_name] += 1
        return None

    async def after_model_callback(self, *, callback_context, llm_response):
        usage = getattr(llm_response, "usage_metadata", None)
        if usage:
            self.prompt_tokens += getattr(usage, "prompt_token_count", 0) or 0
            self.output_tokens += getattr(usage, "candidates_token_count", 0) or 0
        return None

    async def before_tool_callback(self, *, tool, tool_args, tool_context):
        span = ToolSpan(tool.name, dict(tool_args), depth=len(self._open), started=time.perf_counter())
        self.spans.append(span)
        self._open.append(span)
        return None

    async def after_tool_callback(self, *, tool, tool_args, tool_context, result):
        span = self._close(tool.name)
        if span:
            span.result = preview_result(result)
        return None

    async def on_tool_error_callback(self, *, tool, tool_args, tool_context, error):
        span = self._close(tool.name)
        if span:
            span.error = f"{type(error).__name__}: {error}"
        return None

    def _close(self, tool_name: str) -> Optional[ToolSpan]:
        # Close the innermost open span for this tool. Tools run one after another in this app.
        for index in range(len(self._open) - 1, -1, -1):
            span = self._open[index]
            if span.tool == tool_name:
                span.duration = time.perf_counter() - span.started
                del self._open[index:]
                return span
        return None

    # ---- reading the trace ----

    def tool_calls(self) -> Counter:
        return Counter(span.tool for span in self.spans)

    def summary(self) -> dict:
        return {
            "tool_calls": dict(self.tool_calls()),
            "model_calls": dict(self.model_calls),
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "seconds": round(time.perf_counter() - self._started, 2),
        }

    def render(self) -> str:
        lines = []
        for span in self.spans:
            arg = next(iter(span.args.values()), "") if span.args else ""
            arg = str(arg).replace("\n", " ")
            arg = arg if len(arg) <= 70 else arg[:69] + "…"
            took = f" {span.duration:.1f}s" if span.duration is not None else ""
            outcome = f"  ERROR {span.error}" if span.error else (f"  -> {span.result}" if span.result else "")
            lines.append(f"{'  ' * span.depth}- {span.tool}({arg!r}){took}{outcome}")
        return "\n".join(lines) if lines else "(no tool calls)"
