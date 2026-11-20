"""How many questions a visitor (and the whole site) may ask the live demo.

Each question costs about eight Gemini calls, so the demo counts questions, not model calls.
The gateway still applies its own per-client limit and daily budget behind this.
Counters live in Upstash Redis on Cloud Run (instances come and go); in memory for tests.
"""
import hashlib
import hmac
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Protocol

import httpx


class Counter(Protocol):
    async def incr(self, key: str, ttl_seconds: int) -> int: ...


class MemoryCounter:
    def __init__(self, clock=time.monotonic):
        self._data: dict[str, tuple[int, float]] = {}
        self._clock = clock

    async def incr(self, key: str, ttl_seconds: int) -> int:
        now = self._clock()
        value, expires = self._data.get(key, (0, now + ttl_seconds))
        if expires <= now:
            value, expires = 0, now + ttl_seconds
        self._data[key] = (value + 1, expires)
        return value + 1


class UpstashCounter:
    def __init__(self, url: str, token: str, client: Optional[httpx.AsyncClient] = None):
        self._url = url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {token}"}
        self._client = client or httpx.AsyncClient(timeout=5.0)

    async def incr(self, key: str, ttl_seconds: int) -> int:
        # SET NX gives a new key its TTL; INCR then counts. One round trip.
        cmds = [["SET", key, "0", "EX", str(ttl_seconds), "NX"], ["INCR", key]]
        resp = await self._client.post(f"{self._url}/pipeline", json=cmds, headers=self._headers)
        resp.raise_for_status()
        result = resp.json()[1]
        if "error" in result:
            raise RuntimeError(f"upstash: {result['error']}")
        return int(result["result"])

    async def aclose(self) -> None:
        await self._client.aclose()


@dataclass
class Refusal:
    message: str
    retry_after: int


def visitor_id(ip: str, salt: str) -> str:
    """Keyed hash of the visitor's IP: counters never hold the address itself."""
    return hmac.new(salt.encode(), ip.encode(), hashlib.sha256).hexdigest()[:16]


class DemoLimits:
    def __init__(self, counter: Counter, *, per_hour: int, per_day: int, site_per_day: int, clock=time.time):
        self.counter, self.per_hour, self.per_day, self.site_per_day = counter, per_hour, per_day, site_per_day
        self._clock = clock

    async def take(self, visitor: str) -> Optional[Refusal]:
        """Count one question. Returns why it is refused, or None when it may run."""
        now = self._clock()
        day = datetime.fromtimestamp(now, timezone.utc).strftime("%Y%m%d")
        to_midnight = 86_400 - int(now % 86_400)
        hour = int(now // 3600)
        if await self.counter.incr(f"rag:site:{day}", 90_000) > self.site_per_day:
            return Refusal("The demo has answered its questions for today. Try again after 07:00 Bangkok time.", to_midnight)
        if await self.counter.incr(f"rag:v:{visitor}:{hour}", 3_700) > self.per_hour:
            return Refusal(f"You can ask {self.per_hour} questions an hour. Try again later.", 3600 - int(now % 3600))
        if await self.counter.incr(f"rag:v:{visitor}:{day}", 90_000) > self.per_day:
            return Refusal(f"You can ask {self.per_day} questions a day. Try again tomorrow.", to_midnight)
        return None
