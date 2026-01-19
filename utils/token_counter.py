from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator, Optional
import threading
import uuid

_REQUEST_ID: ContextVar[Optional[str]] = ContextVar("token_request_id", default=None)
_GLOBAL_REQUEST_ID: Optional[str] = None
_REQUEST_USAGE: dict[str, dict[str, int]] = {}
_FALLBACK_USAGE: dict[str, int] = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
}
_LOCK = threading.Lock()


def _resolve_request_id() -> Optional[str]:
    request_id = _REQUEST_ID.get()
    if request_id is not None:
        return request_id
    return _GLOBAL_REQUEST_ID


def reset_token_usage() -> str:
    request_id = uuid.uuid4().hex
    _REQUEST_ID.set(request_id)
    with _LOCK:
        global _GLOBAL_REQUEST_ID
        _GLOBAL_REQUEST_ID = request_id
        _REQUEST_USAGE[request_id] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        _FALLBACK_USAGE.update(
            {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
        )
    return request_id


def _as_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def record_token_usage(
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: Optional[int] = None,
) -> None:
    prompt = _as_int(prompt_tokens)
    completion = _as_int(completion_tokens)
    total = _as_int(total_tokens) if total_tokens is not None else prompt + completion
    if prompt == 0 and completion == 0 and total == 0:
        return
    request_id = _resolve_request_id()
    with _LOCK:
        if request_id is None:
            _FALLBACK_USAGE["prompt_tokens"] += prompt
            _FALLBACK_USAGE["completion_tokens"] += completion
            _FALLBACK_USAGE["total_tokens"] += total
            return
        usage = _REQUEST_USAGE.setdefault(
            request_id,
            {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        )
        usage["prompt_tokens"] += prompt
        usage["completion_tokens"] += completion
        usage["total_tokens"] += total


def get_token_usage() -> dict[str, int]:
    request_id = _resolve_request_id()
    with _LOCK:
        if request_id is None:
            return dict(_FALLBACK_USAGE)
        return dict(_REQUEST_USAGE.get(request_id, _FALLBACK_USAGE))


def format_token_usage(prefix: str = "SQL tokens") -> str:
    usage = get_token_usage()
    return (
        f"{prefix}: prompt={usage['prompt_tokens']} "
        f"completion={usage['completion_tokens']} "
        f"total={usage['total_tokens']}"
    )


def print_token_usage(prefix: str = "SQL tokens") -> None:
    print(format_token_usage(prefix))


@contextmanager
def token_usage_session(prefix: str = "SQL tokens") -> Iterator[None]:
    reset_token_usage()
    try:
        yield
    finally:
        print_token_usage(prefix)


def _extract_usage_dict(metadata: dict) -> dict:
    if not isinstance(metadata, dict):
        return {}
    bedrock_metrics = metadata.get("amazon-bedrock-invocationMetrics")
    if isinstance(bedrock_metrics, dict):
        return bedrock_metrics
    usage = metadata.get("usage")
    if isinstance(usage, dict):
        return usage
    return {}


def _extract_token_usage_from_metadata(metadata: Any) -> tuple[int, int, int]:
    if not isinstance(metadata, dict):
        return 0, 0, 0
    usage = _extract_usage_dict(metadata)
    prompt = _as_int(
        usage.get("inputTokenCount")
        or usage.get("input_tokens")
    )
    completion = _as_int(
        usage.get("outputTokenCount")
        or usage.get("output_tokens")
    )
    total = _as_int(
        usage.get("totalTokenCount")
        or usage.get("total_tokens")
    )
    if total == 0 and (prompt or completion):
        total = prompt + completion
    return prompt, completion, total


def _record_usage_from_response(response: Any) -> None:
    if response is None:
        return
    if isinstance(response, dict) and "raw" in response:
        _record_usage_from_response(response.get("raw"))
        return
    metadata: Any = None
    if isinstance(response, dict):
        metadata = response.get("response_metadata") or response.get("metadata")
        if metadata is None and "usage" in response:
            metadata = response
    else:
        metadata = getattr(response, "response_metadata", None)
        if metadata is None:
            metadata = getattr(response, "metadata", None)
        if metadata is None:
            metadata = getattr(response, "additional_kwargs", None)
    prompt, completion, total = _extract_token_usage_from_metadata(metadata or {})
    record_token_usage(prompt, completion, total)


class StructuredOutputTokenCountingWrapper:
    def __init__(self, runnable: Any):
        self._runnable = runnable

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        result = self._runnable.invoke(*args, **kwargs)
        raw = result.get("raw") if isinstance(result, dict) else None
        parsed = result.get("parsed") if isinstance(result, dict) else None
        if raw is not None:
            _record_usage_from_response(raw)
        else:
            _record_usage_from_response(result)
        return parsed if parsed is not None else result

    async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
        result = await self._runnable.ainvoke(*args, **kwargs)
        raw = result.get("raw") if isinstance(result, dict) else None
        parsed = result.get("parsed") if isinstance(result, dict) else None
        if raw is not None:
            _record_usage_from_response(raw)
        else:
            _record_usage_from_response(result)
        return parsed if parsed is not None else result

    def stream(self, *args: Any, **kwargs: Any):
        return self._runnable.stream(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._runnable, name)


class TokenUsageCountingWrapper:
    def __init__(self, llm: Any):
        self._llm = llm

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        result = self._llm.invoke(*args, **kwargs)
        _record_usage_from_response(result)
        return result

    async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
        result = await self._llm.ainvoke(*args, **kwargs)
        _record_usage_from_response(result)
        return result

    def stream(self, *args: Any, **kwargs: Any):
        return self._llm.stream(*args, **kwargs)

    def with_structured_output(self, *args: Any, **kwargs: Any):
        kwargs = dict(kwargs)
        kwargs.setdefault("include_raw", True)
        runnable = self._llm.with_structured_output(*args, **kwargs)
        return StructuredOutputTokenCountingWrapper(runnable)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._llm, name)


def wrap_llm_with_token_counter(llm: Any) -> Any:
    if isinstance(llm, TokenUsageCountingWrapper):
        return llm
    return TokenUsageCountingWrapper(llm)
