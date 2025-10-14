"""Compact SQLite-backed Chainlit data layer used to persist users, threads and steps."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

try:
    from chainlit.data.base import BaseDataLayer
    from chainlit.data.utils import queue_until_user_message
    from chainlit.types import (
        Feedback,
        PageInfo,
        PaginatedResponse,
        Pagination,
        ThreadDict,
        ThreadFilter,
    )
    from chainlit.user import PersistedUser, User
    from chainlit.step import StepDict
    from chainlit.element import Element, ElementDict
except ModuleNotFoundError:  # pragma: no cover - fallback for tooling
    BaseDataLayer = object  # type: ignore

    def queue_until_user_message():  # type: ignore
        def decorator(func):
            async def wrapper(*args, **kwargs):
                return await func(*args, **kwargs)

            return wrapper

        return decorator

    @dataclass
    class Pagination:  # type: ignore[override]
        first: int = 20
        cursor: Optional[str] = None

    @dataclass
    class ThreadFilter:  # type: ignore[override]
        userId: Optional[str] = None
        search: Optional[str] = None

    @dataclass
    class PageInfo:  # type: ignore[override]
        hasNextPage: bool
        startCursor: Optional[str]
        endCursor: Optional[str]

    @dataclass
    class PaginatedResponse:  # type: ignore[override]
        data: List[Dict[str, Any]]
        pageInfo: PageInfo

    ThreadDict = Dict[str, Any]  # type: ignore[assignment]
    Feedback = Dict[str, Any]  # type: ignore[assignment]
    StepDict = Dict[str, Any]  # type: ignore[assignment]
    Element = Any  # type: ignore[assignment]
    ElementDict = Dict[str, Any]  # type: ignore[assignment]

    @dataclass
    class User:  # type: ignore[override]
        identifier: str
        metadata: Dict[str, Any]

    @dataclass
    class PersistedUser:  # type: ignore[override]
        id: str
        identifier: str
        createdAt: str
        metadata: Dict[str, Any]


DEFAULT_DB_PATH = Path(".chainlit_memory") / "chat_data.db"


def _utc_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _serialize(value: Any) -> str:
    if value is None:
        value = {}
    return json.dumps(value, ensure_ascii=False)


def _deserialize(value: Optional[str]) -> Any:
    if not value:
        return {}
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return {}


class SQLiteDataLayer(BaseDataLayer):
    supports_updates = True

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_lock = asyncio.Lock()
        self._initialized = False

    async def _ensure_ready(self) -> None:
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            await self._run(self._create_tables)
            self._initialized = True

    async def _run(self, func, *args):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: func(*args))

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _create_tables(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    identifier TEXT UNIQUE NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS threads (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    user_identifier TEXT,
                    name TEXT,
                    overview TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata TEXT,
                    tags TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS steps (
                    id TEXT PRIMARY KEY,
                    thread_id TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback (
                    id TEXT PRIMARY KEY,
                    step_id TEXT NOT NULL,
                    name TEXT,
                    value REAL,
                    comment TEXT
                )
                """
            )

    # ------------------------------------------------------------------ Users
    async def get_user(self, identifier: str) -> Optional[PersistedUser]:  # type: ignore[override]
        await self._ensure_ready()
        row = await self._run(self._fetch_user_row, identifier)
        if not row:
            return None
        return PersistedUser(
            id=row["id"],
            identifier=row["identifier"],
            createdAt=row["created_at"],
            metadata=_deserialize(row["metadata"]),
        )

    def _fetch_user_row(self, identifier: str) -> Optional[sqlite3.Row]:
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT id, identifier, created_at, metadata FROM users WHERE identifier = ?",
                (identifier,),
            )
            return cur.fetchone()

    async def create_user(self, user: User) -> Optional[PersistedUser]:  # type: ignore[override]
        await self._ensure_ready()
        existing = await self._run(self._fetch_user_row, user.identifier)
        now = _utc_iso()
        payload = {
            "id": existing["id"] if existing else str(uuid4()),
            "identifier": user.identifier,
            "created_at": existing["created_at"] if existing else now,
            "metadata": _serialize(getattr(user, "metadata", {}) or {}),
        }
        await self._run(self._upsert_user_row, payload)
        return PersistedUser(
            id=payload["id"],
            identifier=payload["identifier"],
            createdAt=payload["created_at"],
            metadata=_deserialize(payload["metadata"]),
        )

    def _upsert_user_row(self, payload: Dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO users (id, identifier, created_at, metadata)
                VALUES (:id, :identifier, :created_at, :metadata)
                ON CONFLICT(identifier) DO UPDATE SET metadata=excluded.metadata
                """,
                payload,
            )

    # -------------------------------------------------------------- Feedback
    async def delete_feedback(self, feedback_id: str) -> bool:  # type: ignore[override]
        await self._ensure_ready()
        deleted = await self._run(self._delete_feedback_row, feedback_id)
        return bool(deleted)

    def _delete_feedback_row(self, feedback_id: str) -> int:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM feedback WHERE id = ?", (feedback_id,))
            return cur.rowcount

    async def upsert_feedback(self, feedback: Feedback) -> str:  # type: ignore[override]
        await self._ensure_ready()
        feedback_id = getattr(feedback, "id", None) or str(uuid4())
        payload = {
            "id": feedback_id,
            "step_id": getattr(feedback, "forId", None),
            "name": getattr(feedback, "name", "user_feedback"),
            "value": getattr(feedback, "value", None),
            "comment": getattr(feedback, "comment", None),
        }
        await self._run(self._upsert_feedback_row, payload)
        return feedback_id

    def _upsert_feedback_row(self, payload: Dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO feedback (id, step_id, name, value, comment)
                VALUES (:id, :step_id, :name, :value, :comment)
                ON CONFLICT(id) DO UPDATE SET
                    step_id=excluded.step_id,
                    name=excluded.name,
                    value=excluded.value,
                    comment=excluded.comment
                """,
                payload,
            )

    # ------------------------------------------------------------ Elements API
    @queue_until_user_message()
    async def create_element(self, element: Element):  # type: ignore[override]
        return None

    async def get_element(self, thread_id: str, element_id: str) -> Optional[ElementDict]:  # type: ignore[override]
        return None

    @queue_until_user_message()
    async def delete_element(self, element_id: str, thread_id: Optional[str] = None):  # type: ignore[override]
        return None

    # ------------------------------------------------------------- Steps CRUD
    def _ensure_thread_row(
        self,
        conn: sqlite3.Connection,
        thread_id: str,
        *,
        user_id: Optional[str],
        user_identifier: Optional[str],
        created_at: str,
    ) -> None:
        conn.execute(
            """
            INSERT INTO threads (id, user_id, user_identifier, name, overview, created_at, updated_at, metadata, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, '{}', '[]')
            ON CONFLICT(id) DO NOTHING
            """,
            (
                thread_id,
                user_id,
                user_identifier,
                "Conversation",
                "",
                created_at,
                created_at,
            ),
        )

    @queue_until_user_message()
    async def create_step(self, step: StepDict):  # type: ignore[override]
        await self._ensure_ready()
        await self._run(self._insert_step_row, step)

    def _insert_step_row(self, step: StepDict) -> None:
        thread_id = step.get("threadId") or str(uuid4())
        step_id = step.get("id") or str(uuid4())
        created_at = str(step.get("createdAt") or _utc_iso())
        content_preview = (
            step.get("output")
            or step.get("input")
            or step.get("name")
            or ""
        )
        step_type = step.get("type")
        with self._connect() as conn:
            self._ensure_thread_row(
                conn,
                thread_id,
                user_id=step.get("userId"),
                user_identifier=step.get("userIdentifier"),
                created_at=created_at,
            )
            conn.execute(
                """
                INSERT INTO steps (id, thread_id, payload, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET payload=excluded.payload, updated_at=excluded.updated_at
                """,
                (
                    step_id,
                    thread_id,
                    _serialize(step),
                    created_at,
                    created_at,
                ),
            )
            conn.execute(
                """
                UPDATE threads
                SET updated_at = ?,
                    overview = CASE WHEN TRIM(?) <> '' THEN ? ELSE overview END
                WHERE id = ?
                """,
                (created_at, content_preview, content_preview, thread_id),
            )
            if step_type == "user_message" and isinstance(content_preview, str) and content_preview.strip():
                conn.execute(
                    """
                    UPDATE threads
                    SET name = CASE
                        WHEN name IS NULL OR name = '' OR name = 'Conversation' THEN ?
                        ELSE name
                    END
                    WHERE id = ?
                    """,
                    (content_preview.strip()[:80], thread_id),
                )

    @queue_until_user_message()
    async def update_step(self, step: StepDict):  # type: ignore[override]
        await self._ensure_ready()
        await self._run(self._update_step_row, step)

    def _update_step_row(self, step: StepDict) -> None:
        step_id = step.get("id")
        if not step_id:
            return
        payload = _serialize(step)
        updated_at = str(step.get("updatedAt") or _utc_iso())
        with self._connect() as conn:
            conn.execute(
                "UPDATE steps SET payload = ?, updated_at = ? WHERE id = ?",
                (payload, updated_at, step_id),
            )

    @queue_until_user_message()
    async def delete_step(self, step_id: str):  # type: ignore[override]
        await self._ensure_ready()
        await self._run(self._delete_step_row, step_id)

    def _delete_step_row(self, step_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM feedback WHERE step_id = ?", (step_id,))
            conn.execute("DELETE FROM steps WHERE id = ?", (step_id,))

    # ------------------------------------------------------------- Threads API
    async def get_thread_author(self, thread_id: str) -> str:  # type: ignore[override]
        await self._ensure_ready()
        row = await self._run(self._fetch_thread_row, thread_id)
        return row["user_identifier"] if row else ""

    async def delete_thread(self, thread_id: str):  # type: ignore[override]
        await self._ensure_ready()
        await self._run(self._delete_thread_rows, thread_id)

    def _delete_thread_rows(self, thread_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM feedback WHERE step_id IN (SELECT id FROM steps WHERE thread_id = ?)",
                (thread_id,),
            )
            conn.execute("DELETE FROM steps WHERE thread_id = ?", (thread_id,))
            conn.execute("DELETE FROM threads WHERE id = ?", (thread_id,))

    async def list_threads(
        self,
        pagination: Pagination,
        filters: ThreadFilter,
    ) -> PaginatedResponse:  # type: ignore[override]
        await self._ensure_ready()
        first = getattr(pagination, "first", None) or 20
        cursor = getattr(pagination, "cursor", None)
        offset = int(cursor) if cursor else 0
        user_id = getattr(filters, "userId", None)
        search = getattr(filters, "search", None)

        rows = await self._run(
            self._list_thread_rows,
            first + 1,
            offset,
            user_id,
            search,
        )
        has_more = len(rows) > first
        page_rows = rows[:first]
        threads: List[ThreadDict] = [self._row_to_thread(row, include_steps=False) for row in page_rows]
        page_info = PageInfo(
            hasNextPage=has_more,
            startCursor=str(offset) if threads else None,
            endCursor=str(offset + len(threads) - 1) if threads else None,
        )
        next_cursor = str(offset + first) if has_more else None
        if hasattr(page_info, "nextCursor"):  # type: ignore[attr-defined]
            page_info.nextCursor = next_cursor  # pragma: no cover - backwards compat
        if hasattr(page_info, "endCursor") and not page_info.endCursor:
            page_info.endCursor = next_cursor
        response = PaginatedResponse(data=threads, pageInfo=page_info)
        if hasattr(response, "nextCursor"):  # pragma: no cover
            response.nextCursor = next_cursor
        return response

    def _list_thread_rows(
        self,
        limit: int,
        offset: int,
        user_id: Optional[str],
        search: Optional[str],
    ) -> List[sqlite3.Row]:
        query = [
            "SELECT id, user_id, user_identifier, name, overview, created_at, updated_at, metadata, tags",
            "FROM threads",
        ]
        clauses: List[str] = []
        params: List[Any] = []
        if user_id:
            clauses.append("user_id = ?")
            params.append(user_id)
        if search:
            clauses.append("(LOWER(name) LIKE ? OR LOWER(overview) LIKE ?)")
            needle = f"%{search.lower()}%"
            params.extend([needle, needle])
        if clauses:
            query.append("WHERE " + " AND ".join(clauses))
        query.append("ORDER BY updated_at DESC")
        query.append("LIMIT ? OFFSET ?")
        params.extend([limit, offset])
        sql = " ".join(query)
        with self._connect() as conn:
            cur = conn.execute(sql, params)
            return cur.fetchall()

    async def get_thread(self, thread_id: str) -> Optional[ThreadDict]:  # type: ignore[override]
        await self._ensure_ready()
        row = await self._run(self._fetch_thread_row, thread_id)
        if not row:
            return None
        steps = await self._run(self._list_steps_for_thread, thread_id)
        thread = self._row_to_thread(row, include_steps=False)
        thread["steps"] = steps
        return thread

    def _fetch_thread_row(self, thread_id: str) -> Optional[sqlite3.Row]:
        with self._connect() as conn:
            cur = conn.execute(
                """
                SELECT id, user_id, user_identifier, name, overview, created_at, updated_at, metadata, tags
                FROM threads
                WHERE id = ?
                """,
                (thread_id,),
            )
            return cur.fetchone()

    async def update_thread(
        self,
        thread_id: str,
        name: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ):  # type: ignore[override]
        await self._ensure_ready()
        await self._run(self._update_thread_row, thread_id, name, user_id, metadata, tags)

    def _update_thread_row(
        self,
        thread_id: str,
        name: Optional[str],
        user_id: Optional[str],
        metadata: Optional[Dict[str, Any]],
        tags: Optional[List[str]],
    ) -> None:
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT metadata, tags, user_identifier FROM threads WHERE id = ?",
                (thread_id,),
            ).fetchone()
            meta = _deserialize(existing["metadata"] if existing else None)
            if metadata:
                meta.update({k: v for k, v in metadata.items() if v is not None})
            current_tags = _deserialize(existing["tags"] if existing else None)
            new_tags = tags if tags is not None else (current_tags if isinstance(current_tags, list) else [])
            user_identifier = existing["user_identifier"] if existing else None
            if user_id:
                row = conn.execute("SELECT identifier FROM users WHERE id = ?", (user_id,)).fetchone()
                user_identifier = row["identifier"] if row else user_identifier
            conn.execute(
                """
                UPDATE threads
                SET name = COALESCE(?, name),
                    user_id = COALESCE(?, user_id),
                    user_identifier = COALESCE(?, user_identifier),
                    metadata = ?,
                    tags = ?
                WHERE id = ?
                """,
                (
                    name,
                    user_id,
                    user_identifier,
                    _serialize(meta),
                    _serialize(new_tags),
                    thread_id,
                ),
            )

    def _list_steps_for_thread(self, thread_id: str) -> List[StepDict]:
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT payload FROM steps WHERE thread_id = ? ORDER BY created_at ASC",
                (thread_id,),
            )
            rows = cur.fetchall()
        steps: List[StepDict] = []
        for row in rows:
            payload = _deserialize(row["payload"])
            if isinstance(payload, dict):
                steps.append(payload)
        return steps

    def _row_to_thread(self, row: sqlite3.Row, *, include_steps: bool) -> ThreadDict:
        thread: ThreadDict = {
            "id": row["id"],
            "userId": row["user_id"],
            "userIdentifier": row["user_identifier"],
            "name": row["name"],
            "overview": row["overview"],
            "createdAt": row["created_at"],
            "updatedAt": row["updated_at"],
            "metadata": _deserialize(row["metadata"]),
            "tags": _deserialize(row["tags"]),
            "steps": [],
            "elements": [],
        }
        if include_steps:
            thread["steps"] = self._list_steps_for_thread(row["id"])
        return thread

    # ------------------------------------------------------------------- Misc
    async def build_debug_url(self) -> str:  # type: ignore[override]
        return ""

    async def close(self) -> None:  # type: ignore[override]
        return None

