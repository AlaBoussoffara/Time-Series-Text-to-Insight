from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

try:
    import psycopg
except ImportError:  # pragma: no cover - psycopg is optional at runtime
    psycopg = None  # type: ignore

try:
    from dotenv import load_dotenv  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None  # type: ignore

if TYPE_CHECKING:  # pragma: no cover - typing only
    from psycopg import Connection

MAX_RESULT_ROWS = 1000


def connect_postgres(dsn: Optional[str] = None) -> "Connection":
    """
    Create a PostgreSQL connection using psycopg.

    Parameters
    ----------
    dsn:
        Optional connection string. Falls back to the POSTGRES_DSN environment
        variable when omitted.
    """
    if psycopg is None:
        raise RuntimeError(
            "psycopg is required for PostgreSQL support. Install it with "
            "`pip install psycopg[binary]`."
        )

    if dsn is None:
        if load_dotenv is not None:
            load_dotenv()
        dsn = os.environ.get("POSTGRES_DSN")
    if not dsn:
        raise RuntimeError("Provide a DSN via connect_postgres(dsn=...) or POSTGRES_DSN.")

    return psycopg.connect(dsn, autocommit=True)


def execute_sql_tool(conn: Any, query: str) -> Union[List[Dict[str, Any]], str]:
    """
    Execute the supplied SQL and return a list of dictionaries representing rows.
    Up to MAX_RESULT_ROWS rows are returned; a truncation notice is appended when
    additional rows exist. Errors are propagated as strings for the caller to log.
    """
    if not query or not query.strip():
        return "SQL execution error: empty query received."

    try:
        with conn.cursor() as cursor:
            cursor.execute(query)
            description = cursor.description
            if description is None:
                return []

            columns = [col[0] for col in description]
            rows = cursor.fetchmany(MAX_RESULT_ROWS + 1)
    except Exception as exc:  # pragma: no cover - surface DB errors
        return f"SQL execution error: {exc}"

    truncated = len(rows) > MAX_RESULT_ROWS
    rows = rows[:MAX_RESULT_ROWS]

    results = [dict(zip(columns, row)) for row in rows]
    if truncated:
        results.append({"__notice__": f"Result truncated to {MAX_RESULT_ROWS} rows."})
    return results


def test_postgres_connection(dsn: Optional[str] = None) -> bool:
    """
    Return True when a PostgreSQL connection can be opened and responds to SELECT 1.
    """
    conn: Optional["Connection"] = None
    try:
        conn = connect_postgres(dsn)
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1;")
            cursor.fetchone()
        return True
    except Exception:
        return False
    finally:
        if conn is not None:
            conn.close()


__all__ = [
    "connect_postgres",
    "execute_sql_tool",
    "test_postgres_connection",
]
