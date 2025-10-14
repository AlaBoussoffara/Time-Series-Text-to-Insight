from __future__ import annotations

import os
import textwrap
from pathlib import Path
from typing import Optional, Sequence

import chainlit as cl
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage, BaseMessage, HumanMessage
from langchain_community.chat_message_histories import SQLChatMessageHistory

from datalayer import SQLiteDataLayer
from supervisor import run_supervisor

try:  # Chainlit 1.0+
    from chainlit.data import get_data_layer
except ImportError:  # pragma: no cover - runtime fallback
    get_data_layer = None  # type: ignore

try:
    from chainlit.context import get_current_context
except ImportError:  # pragma: no cover - older Chainlit
    get_current_context = None  # type: ignore

load_dotenv()

PERSIST_DIR = Path(".chainlit_memory")
PERSIST_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_SCOPE = os.getenv("CHAT_MEMORY_SCOPE", "conversation").strip().lower()
HISTORY_OPTIONS = {"conversation", "user"}
if DEFAULT_SCOPE not in HISTORY_OPTIONS:
    DEFAULT_SCOPE = "conversation"

USERS = {"demo": "demo123"}
SCOPE_COMMAND_PREFIX = "scope:"


@cl.data_layer
def configure_data_layer() -> SQLiteDataLayer:
    return SQLiteDataLayer(PERSIST_DIR / "chat_data.db")


def _normalize_scope(value: Optional[str]) -> str:
    if isinstance(value, str) and value.lower() in HISTORY_OPTIONS:
        return value.lower()
    return DEFAULT_SCOPE


def _user_identifier() -> str:
    user = cl.user_session.get("user")
    return getattr(user, "identifier", "anonymous")


def _resolve_thread_id() -> Optional[str]:
    session_thread = cl.user_session.get("thread_id")
    if isinstance(session_thread, str):
        return session_thread
    if get_current_context:
        ctx = get_current_context()
        if ctx is not None:
            thread = getattr(ctx, "thread", None)
            ctx_thread = getattr(ctx, "thread_id", None)
            if thread is not None and getattr(thread, "id", None):
                return thread.id  # type: ignore[attr-defined]
            if isinstance(ctx_thread, str):
                return ctx_thread
    return None


def _memory_session_key(scope: str, owner: str, thread_id: Optional[str]) -> str:
    if scope == "user":
        return owner
    return f"{owner}:{thread_id or 'live'}"


def _build_memory(scope: str, owner: str, thread_id: Optional[str]) -> ConversationBufferMemory:
    if scope == "user":
        connection = f"sqlite:///{PERSIST_DIR / 'chat_history.db'}"
        history = SQLChatMessageHistory(session_id=owner, connection=connection)
        return ConversationBufferMemory(chat_memory=history, return_messages=True)
    return ConversationBufferMemory(return_messages=True)


def _current_scope() -> str:
    scope = _normalize_scope(cl.user_session.get("history_scope"))
    cl.user_session.set("history_scope", scope)
    return scope


def _history_from_memory(memory: ConversationBufferMemory) -> Sequence[BaseMessage]:
    data = memory.load_memory_variables({})
    history = data.get("history", [])
    if isinstance(history, list):
        return history
    return []


def _chunk_text(text: str, width: int = 200) -> list[str]:
    if not text:
        return []
    return textwrap.wrap(text, width=width, replace_whitespace=False)


def _step_to_message(step: dict) -> Optional[BaseMessage]:
    step_type = step.get("type")
    if step_type not in {"user_message", "assistant_message"}:
        return None
    content = step.get("output") or step.get("content") or step.get("input") or ""
    if not isinstance(content, str) or not content.strip():
        return None
    if step_type == "user_message":
        return HumanMessage(content=content)
    return AIMessage(content=content)


async def _load_thread_messages(thread_id: str) -> list[BaseMessage]:
    if not get_data_layer:
        return []
    data_layer = get_data_layer()
    if not data_layer:
        return []
    try:
        thread = await data_layer.get_thread(thread_id)
    except Exception:
        return []
    steps = thread.get("steps") if isinstance(thread, dict) else None
    messages: list[BaseMessage] = []
    if isinstance(steps, list):
        for step in steps:
            if isinstance(step, dict):
                msg = _step_to_message(step)
                if msg:
                    messages.append(msg)
    return messages


def _overwrite_memory(memory: ConversationBufferMemory, messages: Sequence[BaseMessage]) -> None:
    chat_memory = getattr(memory, "chat_memory", None)
    if chat_memory is None:
        return
    if hasattr(chat_memory, "messages"):
        chat_memory.messages = []
    if hasattr(chat_memory, "clear"):
        try:
            chat_memory.clear()
        except Exception:
            pass
    for message in messages:
        try:
            chat_memory.add_message(message)
        except AttributeError:
            if hasattr(chat_memory, "messages"):
                chat_memory.messages.append(message)


async def _ensure_memory(scope: str) -> ConversationBufferMemory:
    owner = _user_identifier()
    thread_id = _resolve_thread_id()
    session_key = _memory_session_key(scope, owner, thread_id)
    memory: Optional[ConversationBufferMemory] = cl.user_session.get("memory")
    active_key: Optional[str] = cl.user_session.get("memory_session_id")
    if memory is None or active_key != session_key:
        memory = _build_memory(scope, owner, thread_id)
        cl.user_session.set("memory", memory)
        cl.user_session.set("memory_session_id", session_key)
    if scope == "conversation":
        if not memory.chat_memory.messages and thread_id:
            history = await _load_thread_messages(thread_id)
            if history:
                _overwrite_memory(memory, history)
    else:
        memory.load_memory_variables({})
    return memory


def _parse_scope_command(content: str) -> Optional[str]:
    lowered = content.strip().lower()
    if not lowered.startswith(SCOPE_COMMAND_PREFIX):
        return None
    desired = lowered[len(SCOPE_COMMAND_PREFIX) :].strip()
    if desired in HISTORY_OPTIONS:
        return desired
    return ""


async def _stream_response(text: str, structured: dict) -> None:
    author = structured.get("output", "Supervisor") if structured else "Supervisor"
    metadata = {"structured": structured} if structured else None
    message = cl.Message(content="", author=author, metadata=metadata)
    await message.send()
    chunks = _chunk_text(text) or [""]
    for chunk in chunks:
        await message.stream_token(chunk)
    message.content = text
    await message.update()


@cl.password_auth_callback
def auth_callback(username: str, password: str):
    if USERS.get(username) != password:
        return None
    return cl.User(identifier=username)


@cl.on_chat_start
async def on_chat_start():
    scope = _current_scope()
    await _ensure_memory(scope)
    notice = (
        "Type `scope: conversation` to limit context to this chat or `scope: user` to reuse all your past chats."
    )
    await cl.Message(content=f"History scope set to `{scope}`.\n{notice}").send()


@cl.on_chat_resume
async def on_chat_resume(thread: dict):
    thread_id = thread.get("id")
    if isinstance(thread_id, str):
        cl.user_session.set("thread_id", thread_id)
    scope = _current_scope()
    if scope == "conversation" and isinstance(thread_id, str):
        memory = await _ensure_memory(scope)
        history = await _load_thread_messages(thread_id)
        if history:
            _overwrite_memory(memory, history)


@cl.on_message
async def main(message: cl.Message):
    scope = _current_scope()
    requested_scope = _parse_scope_command(message.content)
    if requested_scope is not None:
        if requested_scope:
            cl.user_session.set("history_scope", requested_scope)
            await _ensure_memory(requested_scope)
            await cl.Message(content=f"History scope switched to `{requested_scope}`.").send()
        else:
            options = ", ".join(sorted(HISTORY_OPTIONS))
            await cl.Message(content=f"Invalid scope. Choose one of: {options}.").send()
        return

    memory = await _ensure_memory(scope)
    history_messages = list(_history_from_memory(memory))

    supervisor = await cl.make_async(run_supervisor)(
        message.content,
        history=history_messages,
        log=True,
    )
    if supervisor is None:
        await cl.Message(content="Sorry, I couldn't generate a response.").send()
        return

    structured = supervisor.additional_kwargs.get("structured", {})
    response_text = structured.get("content") or supervisor.content or ""
    await _stream_response(response_text, structured)
    memory.save_context({"input": message.content}, {"output": response_text})
