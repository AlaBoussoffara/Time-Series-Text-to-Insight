from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Optional
import threading
import uuid

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser

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
    def __init__(
        self,
        llm: Any,
        *,
        agent_name: Optional[str] = None,
        schema: Optional[Any] = None,
        orig_llm: Optional[Any] = None,
    ):
        self._llm = llm
        self._agent_name = agent_name
        self._schema = schema
        # If this wrapper wraps a structured-output runnable, we keep a reference
        # to the original LLM so we can use it for fallback generation.
        self._orig_llm = orig_llm

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        increment_api_call_count(1, agent_name=self._agent_name)
        result = self._llm.invoke(*args, **kwargs)

        # Fallback Logic:
        # If the result is None and we have a schema (meaning we expected structured output),
        # try to generate valid JSON using a standard prompt + PydanticOutputParser.
        if result is None and self._schema and self._orig_llm:
            try:
                parser = PydanticOutputParser(pydantic_object=self._schema)
                format_instructions = parser.get_format_instructions()
                
                # We need to construct a new input for the fallback LLM.
                # args[0] is typically the input (list of messages or string).
                input_arg = args[0] if args else None
                fallback_messages = []

                if isinstance(input_arg, list):
                    # It's a list of messages. Copy them.
                    fallback_messages = list(input_arg)
                    # Append instructions to the last message if plain text,
                    # or add a new system/human message with instructions.
                    fallback_messages.append(
                        HumanMessage(content=f"Please provide your response in JSON format.\n{format_instructions}")
                    )
                elif isinstance(input_arg, str):
                    # It's a string prompt.
                    fallback_messages = [
                         HumanMessage(content=input_arg),
                         HumanMessage(content=f"Please provide your response in JSON format.\n{format_instructions}")
                    ]
                elif isinstance(input_arg, dict):
                     # LangGraph often passes a dict state; this might handle some cases
                     # but typically invoke takes messages. We'll skip if complex.
                     pass

                if fallback_messages:
                    print(f"[{self._agent_name}] Structured output returned None. Retrying with PydanticOutputParser fallback...")
                    # We invoke the ORIGINAL, non-structured LLM with textual instructions
                    fallback_response = self._orig_llm.invoke(fallback_messages)
                    # The response is usually an AIMessage
                    text_output = fallback_response.content if hasattr(fallback_response, "content") else str(fallback_response)
                    
                    # Robust extraction: Try to find a JSON block first
                    import re
                    json_match = re.search(r"```json\s*({.*?})\s*```", text_output, re.DOTALL)
                    if not json_match:
                        # Try finding just the first outer brace
                        json_match = re.search(r"({.*})", text_output, re.DOTALL)
                    
                    if json_match:
                        text_output = json_match.group(1)
                    
                    parsed_result = parser.parse(text_output)
                    return parsed_result
            except Exception as e:
                print(f"[{self._agent_name}] Fallback failed: {e}")
                # If fallback fails, return None (or raise) as before
                pass

        return result

    async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
        increment_api_call_count(1, agent_name=self._agent_name)
        return await self._llm.ainvoke(*args, **kwargs)

    def stream(self, *args: Any, **kwargs: Any):
        increment_api_call_count(1, agent_name=self._agent_name)
        return self._llm.stream(*args, **kwargs)

    def with_structured_output(self, schema: Any, *args: Any, **kwargs: Any):
        # We wrap the result of with_structured_output
        wrapped = self._llm.with_structured_output(schema, *args, **kwargs)
        # We pass 'self._llm' as 'orig_llm' assuming 'self._llm' is the base chat model.
        # If self._llm is already a wrapper, unwrap or use it. using self._llm is usually safe if it's a Runnable.
        return ApiCallCountingWrapper(
            wrapped,
            agent_name=self._agent_name,
            schema=schema,
            orig_llm=self._llm 
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._llm, name)


def wrap_llm_with_api_call_counter(
    llm: Any, *, agent_name: Optional[str] = None
) -> Any:
    if isinstance(llm, ApiCallCountingWrapper):
        if agent_name and llm._agent_name != agent_name:
            return ApiCallCountingWrapper(
                llm._llm,
                agent_name=agent_name,
                schema=llm._schema,
                orig_llm=llm._orig_llm
            )
        return llm
    return ApiCallCountingWrapper(llm, agent_name=agent_name)
