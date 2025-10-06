from pathlib import Path
from typing import Optional, Sequence

from datetime import datetime

import chainlit as cl
from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage, BaseMessage, HumanMessage
from langchain_community.chat_message_histories import SQLChatMessageHistory
from dotenv import load_dotenv
import os
from llm import llm_from
from datalayer import SQLiteDataLayer

try:
    from chainlit.data import get_data_layer
    from chainlit.types import Pagination, ThreadFilter
except ImportError:  # pragma: no cover - handled by runtime environment
    get_data_layer = None  # type: ignore
    Pagination = None  # type: ignore
    ThreadFilter = None  # type: ignore

try:
    from chainlit.context import get_current_context, context as cl_context
except ImportError:  # Fallback for older Chainlit versions
    get_current_context = None
    cl_context = None

load_dotenv()

PERSIST_DIR = Path(".chainlit_memory")
PERSIST_DIR.mkdir(parents=True, exist_ok=True)


@cl.data_layer
def configure_data_layer() -> SQLiteDataLayer:
    return SQLiteDataLayer(PERSIST_DIR / "chat_data.db")

USERS = {"demo": "demo123"}


ALLOWED_MEMORY_SCOPES = {"conversation", "user", "all"}
MEMORY_SCOPE = os.getenv("CHAT_MEMORY_SCOPE", "conversation").strip().lower()
if MEMORY_SCOPE not in ALLOWED_MEMORY_SCOPES:
    MEMORY_SCOPE = "conversation"

PERSISTED_SCOPE = MEMORY_SCOPE in {"user", "all"}


def build_memory(session_id: str) -> ConversationBufferMemory:
    if PERSISTED_SCOPE:
        connection = f"sqlite:///{PERSIST_DIR / 'chat_history.db'}"
        history = SQLChatMessageHistory(
            session_id=session_id,
            connection=connection,
        )
        return ConversationBufferMemory(chat_memory=history, return_messages=True)
    return ConversationBufferMemory(return_messages=True)


def _parse_iso(timestamp: Optional[str]) -> float:
    if not timestamp:
        return 0.0
    try:
        return datetime.fromisoformat(timestamp.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def _step_to_message(step: dict) -> Optional[tuple[float, BaseMessage]]:
    step_type = step.get("type")
    if step_type not in {"user_message", "assistant_message"}:
        return None
    content = step.get("output") or step.get("content") or step.get("input") or ""
    if not isinstance(content, str) or not content.strip():
        return None
    ts = _parse_iso(step.get("createdAt"))
    if step_type == "user_message":
        return ts, HumanMessage(content=content)
    return ts, AIMessage(content=content)
async def _load_thread_messages(thread_id: str) -> list[BaseMessage]:
    if not get_data_layer:
        return []
    dl = get_data_layer()
    try:
        thread = await dl.get_thread(thread_id)
    except Exception:
        return []

    steps = thread.get("steps", []) if isinstance(thread, dict) else []
    collected: list[tuple[float, BaseMessage]] = []
    for step in steps:
        if isinstance(step, dict):
            converted = _step_to_message(step)
            if converted:
                collected.append(converted)

    collected.sort(key=lambda x: x[0])
    return [m for _, m in collected]


async def _load_user_message_history(user_identifier: str) -> Sequence[BaseMessage]:
    if not PERSISTED_SCOPE or not get_data_layer or not Pagination or not ThreadFilter:
        return []

    data_layer = get_data_layer()
    if not data_layer:
        return []

    try:
        persisted_user = await data_layer.get_user(user_identifier)
    except Exception:
        persisted_user = None
    if not persisted_user or not getattr(persisted_user, "id", None):
        return []

    try:
        response = await data_layer.list_threads(
            Pagination(first=500),
            ThreadFilter(userId=persisted_user.id),
        )
    except Exception:
        return []

    if hasattr(response, "data"):
        threads = response.data  # type: ignore[attr-defined]
    elif isinstance(response, dict):
        threads = response.get("data", [])
    else:
        threads = []
    collected: list[tuple[float, BaseMessage]] = []

    for thread in threads:
        thread_id = thread.get("id") if isinstance(thread, dict) else None
        if not thread_id:
            continue
        try:
            full_thread = await data_layer.get_thread(thread_id)
        except Exception:
            continue
        steps = (full_thread or {}).get("steps") if isinstance(full_thread, dict) else None
        if not isinstance(steps, list):
            continue
        for step in steps:
            if not isinstance(step, dict):
                continue
            converted = _step_to_message(step)
            if converted:
                collected.append(converted)

    collected.sort(key=lambda item: item[0])
    return [message for _, message in collected]


def _resolve_thread_id() -> Optional[str]:
    if cl_context is not None:
        try:
            thread_id = getattr(cl_context.session, "thread_id", None)
            if isinstance(thread_id, str):
                return thread_id
        except Exception:
            pass
    if get_current_context:
        ctx = get_current_context()
        if ctx is not None:
            thread = getattr(ctx, "thread", None)
            if thread is not None and getattr(thread, "id", None):
                return thread.id  # type: ignore[attr-defined]
            ctx_thread_id = getattr(ctx, "thread_id", None)
            if isinstance(ctx_thread_id, str):
                return ctx_thread_id
    thread_id = cl.user_session.get("thread_id")
    if isinstance(thread_id, str):
        return thread_id
    return None


def _current_session_identifier() -> str:
    user = cl.user_session.get("user")
    base_id = user.identifier if user else "anonymous"
    thread_id = _resolve_thread_id()
    if PERSISTED_SCOPE:
        return base_id
    return f"{base_id}:{thread_id}" if thread_id else base_id


def _get_or_create_memory(session_id: str) -> ConversationBufferMemory:
    memory: Optional[ConversationBufferMemory] = cl.user_session.get("memory")
    current_session = cl.user_session.get("memory_session_id")
    if memory is None or current_session != session_id:
        memory = build_memory(session_id)
        cl.user_session.set("memory", memory)
        cl.user_session.set("memory_session_id", session_id)
    return memory


@cl.password_auth_callback
def auth_callback(username: str, password: str):
    if USERS.get(username) != password:
        return None
    return cl.User(identifier=username)


@cl.on_chat_start
async def on_chat_start():
    session_id = _current_session_identifier()
    _get_or_create_memory(session_id)

USE_MODEL = os.getenv("USE_MODEL", "mistral-ollama")
llm = llm_from(USE_MODEL)

@cl.on_message
async def main(message: cl.Message):
    session_id = _current_session_identifier()
    memory = _get_or_create_memory(session_id)

    user_obj = cl.user_session.get("user")
    user_identifier = getattr(user_obj, "identifier", "anonymous")

    if MEMORY_SCOPE == "conversation":
        history_messages = list(memory.load_memory_variables({}).get("history", []))
    else:
        history_messages = list(await _load_user_message_history(user_identifier))

    if history_messages and isinstance(history_messages[-1], HumanMessage) and history_messages[-1].content == message.content:
        conversation = history_messages
    else:
        conversation = history_messages + [HumanMessage(content=message.content)]

    msg = await cl.Message(content="").send()
    response_chunks: list[str] = []

    async for chunk in llm.astream(conversation):
        token = chunk.content or ""
        if token:
            response_chunks.append(token)
            await msg.stream_token(token)

    await msg.update()

    memory.save_context({"input": message.content}, {"output": "".join(response_chunks)})
@cl.on_chat_resume
async def on_chat_resume(thread: dict):
    # 1) bind the chosen thread to this live session
    thread_id = thread.get("id")
    if isinstance(thread_id, str):
        cl.user_session.set("thread_id", thread_id)

    # 2) (re)build memory for this thread
    session_id = _current_session_identifier()  # will now include this thread id if MEMORY_SCOPE=="conversation"
    memory = _get_or_create_memory(session_id)

    # 3) seed memory with that thread’s messages
    msgs = await _load_thread_messages(thread_id)
    # clear old memory and load this thread history
    try:
        # LangChain ConversationBufferMemory API
        memory.chat_memory.messages = []
    except Exception:
        pass
    for m in msgs:
        memory.chat_memory.add_message(m)

