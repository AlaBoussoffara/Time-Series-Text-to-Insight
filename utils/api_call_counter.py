from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Optional
import threading
import uuid

_REQUEST_ID: ContextVar[Optional[str]] = ContextVar("api_call_request_id", default=None)
_GLOBAL_REQUEST_ID: Optional[str] = None
_REQUEST_COUNTS: dict[str, int] = {}
_REQUEST_AGENT_COUNTS: dict[str, dict[str, int]] = {}
_FALLBACK_COUNT = 0
_FALLBACK_AGENT_COUNTS: dict[str, int] = {}
_LOCK = threading.Lock()


def _resolve_request_id() -> Optional[str]:
    request_id = _REQUEST_ID.get()
    if request_id is not None:
        return request_id
    return _GLOBAL_REQUEST_ID


def reset_api_call_count() -> str:
    request_id = uuid.uuid4().hex
    _REQUEST_ID.set(request_id)
    with _LOCK:
        global _GLOBAL_REQUEST_ID
        _GLOBAL_REQUEST_ID = request_id
        _REQUEST_COUNTS[request_id] = 0
        _REQUEST_AGENT_COUNTS[request_id] = {}
        global _FALLBACK_COUNT
        _FALLBACK_COUNT = 0
        _FALLBACK_AGENT_COUNTS.clear()
    return request_id


def increment_api_call_count(delta: int = 1, agent_name: Optional[str] = None) -> None:
    if delta <= 0:
        return
    request_id = _resolve_request_id()
    with _LOCK:
        global _FALLBACK_COUNT
        if request_id is None:
            _FALLBACK_COUNT += delta
            if agent_name:
                _FALLBACK_AGENT_COUNTS[agent_name] = (
                    _FALLBACK_AGENT_COUNTS.get(agent_name, 0) + delta
                )
            return
        _REQUEST_COUNTS[request_id] = _REQUEST_COUNTS.get(request_id, 0) + delta
        if agent_name:
            agent_counts = _REQUEST_AGENT_COUNTS.setdefault(request_id, {})
            agent_counts[agent_name] = agent_counts.get(agent_name, 0) + delta


def get_api_call_count() -> int:
    request_id = _resolve_request_id()
    with _LOCK:
        if request_id is None:
            return _FALLBACK_COUNT
        return _REQUEST_COUNTS.get(request_id, 0)


def get_api_call_counts_by_agent() -> dict[str, int]:
    request_id = _resolve_request_id()
    with _LOCK:
        if request_id is None:
            return dict(_FALLBACK_AGENT_COUNTS)
        return dict(_REQUEST_AGENT_COUNTS.get(request_id, {}))


def format_api_call_count(prefix: str = "API calls") -> str:
    return f"{prefix}: {get_api_call_count()}"


def print_api_call_count(prefix: str = "API calls") -> None:
    print(format_api_call_count(prefix))


def format_api_call_breakdown(prefix: str = "API calls") -> str:
    counts = get_api_call_counts_by_agent()
    total = get_api_call_count()
    if not counts:
        return f"{prefix}: total={total}"
    sum_counts = sum(counts.values())
    if total > sum_counts:
        counts["Unattributed"] = total - sum_counts
    parts = [f"{name}={count}" for name, count in sorted(counts.items())]
    parts.append(f"total={total}")
    return f"{prefix}: " + " | ".join(parts)


def print_api_call_breakdown(prefix: str = "API calls") -> None:
    print(format_api_call_breakdown(prefix))


def append_api_call_log(
    prefix: str = "API calls",
    *,
    path: Optional[Path] = None,
) -> None:
    target = path or Path("reports") / "api_call_log.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    request_id = _resolve_request_id() or "unknown"
    timestamp = datetime.now().isoformat(timespec="seconds")
    line = f"{timestamp} request={request_id} {format_api_call_breakdown(prefix)}"
    with target.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


@contextmanager
def api_call_session(prefix: str = "API calls") -> Iterator[None]:
    reset_api_call_count()
    try:
        yield
    finally:
        print_api_call_breakdown(prefix)
        append_api_call_log(prefix)


class ApiCallCountingWrapper:
    def __init__(self, llm: Any, *, agent_name: Optional[str] = None):
        self._llm = llm
        self._agent_name = agent_name

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        increment_api_call_count(1, agent_name=self._agent_name)
        return self._llm.invoke(*args, **kwargs)

    async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
        increment_api_call_count(1, agent_name=self._agent_name)
        return await self._llm.ainvoke(*args, **kwargs)

    def stream(self, *args: Any, **kwargs: Any):
        increment_api_call_count(1, agent_name=self._agent_name)
        return self._llm.stream(*args, **kwargs)

    def with_structured_output(self, *args: Any, **kwargs: Any):
        wrapped = self._llm.with_structured_output(*args, **kwargs)
        return ApiCallCountingWrapper(wrapped, agent_name=self._agent_name)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._llm, name)


def wrap_llm_with_api_call_counter(
    llm: Any, *, agent_name: Optional[str] = None
) -> Any:
    if isinstance(llm, ApiCallCountingWrapper):
        if agent_name and llm._agent_name != agent_name:
            return ApiCallCountingWrapper(llm._llm, agent_name=agent_name)
        return llm
    return ApiCallCountingWrapper(llm, agent_name=agent_name)
