from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List

import json
import pandas as pd
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from utils.datastore import DATASTORE, DataStore
from utils.sql_utils import connect_postgres, execute_sql_tool
from utils.states import SQLState
from utils.messages import AgentMessage


def _load_schema_from_db_schema(db_schema_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    schema: Dict[str, List[Dict[str, Any]]] = {}
    if not db_schema_dir.exists() or not db_schema_dir.is_dir():
        return schema

    for json_path in sorted(db_schema_dir.glob("*.json")):
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        table = str(payload.get("table") or json_path.stem).strip()
        if not table:
            continue

        raw_columns = payload.get("columns")
        if not isinstance(raw_columns, list):
            continue

        columns: List[Dict[str, Any]] = []
        for raw_column in raw_columns:
            if not isinstance(raw_column, dict):
                continue
            name = str(raw_column.get("name", "")).strip()
            if not name:
                continue

            entry: Dict[str, Any] = {"name": name}
            col_type = raw_column.get("type")
            if col_type is not None:
                entry["type"] = str(col_type).strip()
            nullable = raw_column.get("nullable")
            if isinstance(nullable, bool):
                entry["nullable"] = nullable
            description = str(raw_column.get("description", "")).strip()
            if description:
                entry["description"] = description
            columns.append(entry)

        if columns:
            schema[table] = columns

    return schema


def _load_schema_from_information_schema() -> Dict[str, List[Dict[str, Any]]]:
    try:
        conn = connect_postgres()
    except Exception:
        return {}

    schema: Dict[str, List[Dict[str, Any]]] = {}
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT table_schema, table_name, column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_schema NOT IN ('pg_catalog','information_schema')
                ORDER BY table_schema, table_name, ordinal_position
                """
            )
            for schema_name, table_name, column_name, data_type, is_nullable in cur.fetchall():
                table = f"{schema_name}.{table_name}"
                schema.setdefault(table, []).append(
                    {
                        "name": column_name,
                        "type": data_type,
                        "nullable": is_nullable == "YES",
                    }
                )
    except Exception:
        return {}
    finally:
        try:
            conn.close()  # type: ignore[union-attr]
        except Exception:
            pass

    return schema


def _load_database_schema_snapshot() -> tuple[Dict[str, List[Dict[str, Any]]], str]:
    schema = _load_schema_from_db_schema(Path("DB_schema"))
    if schema:
        return schema, "DB_schema/*.json"

    schema = _load_schema_from_information_schema()
    if schema:
        return schema, "information_schema"

    return {}, "unavailable"


def _format_database_schema_summary(schema: Dict[str, List[Dict[str, Any]]]) -> str:
    if not schema:
        return "Database schema: unavailable."

    lines: List[str] = ["Database schema (authoritative):"]
    for table_name in sorted(schema.keys()):
        columns = schema.get(table_name) or []
        rendered_cols: List[str] = []
        for col in columns:
            name = str(col.get("name", "")).strip()
            col_type = str(col.get("type", "")).strip()
            if not name:
                continue
            rendered_cols.append(f"{name} {col_type}".strip())
        if rendered_cols:
            lines.append(f"- {table_name}: " + ", ".join(rendered_cols))

    raw_measurements = schema.get("public.raw_measurements") or []
    for col in raw_measurements:
        if str(col.get("name", "")).strip().lower() == "timestamp":
            lines.append('Note: `raw_measurements` has a column named `timestamp`; reference it as `"timestamp"` in SQL.')
            break

    return "\n".join(lines)

def create_sql_agent(llm):
    """
    Build the minimal SQL controller graph.
    """
    def initialize_state(state: SQLState) -> SQLState:
        messages: List[AnyMessage] = list(state.get("messages", []))
        if not messages:
            PROMPT_PATH = Path("prompts/sql_agent_prompt.txt")
            system_prompt = PROMPT_PATH.read_text(encoding="utf-8")
            messages.append(SystemMessage(system_prompt))
        instruction = state.get("instruction", "")
        if instruction:
            messages.append(HumanMessage(instruction))
        datastore = state.get("datastore")
        if not isinstance(datastore, DataStore):
            datastore = DATASTORE
        query_log = list(state.get("query_log", []))
        return {
            "messages": messages,
            "instruction": instruction,
            "datastore": datastore,
            "query_log": query_log,
        }

    def controller_node(state: SQLState) -> SQLState:
        response = llm.invoke(state["messages"])
        raw_structured = response.model_dump() if hasattr(response, "model_dump") else response
        structured = raw_structured if isinstance(raw_structured, dict) else {}
        output_type = structured.get("output_type", "")
        output_content = structured.get("output_content", "")
        agent_msg = AgentMessage(
            name="SQL Controller",
            structured_output=structured,
        )
        print(f"[SQL Controller] {output_type}: {output_content}")
        return {"messages": [agent_msg]}

    def route_controller(state: SQLState) -> str:
        last_message = state["messages"][-1]
        output_type: str | None = None
        if isinstance(last_message, AgentMessage):
            output_type = str(last_message.structured_output.get("output_type", ""))
        if output_type == "summarize_datastore_updates":
            return "summarize_datastore_updates"
        if output_type == "execute_sql":
            return "execute_sql"
        if output_type == "persist_dataset":
            return "persist_dataset"
        if output_type == "sql_agent_final_answer":
            return "record_final"
        if output_type == "no_hallucination":
            return "end"
        return "controller"

    def execute_sql_node(state: SQLState) -> SQLState:
        last_message = state["messages"][-1]
        structured_output = getattr(last_message, "structured_output", {}) or {}
        if not isinstance(structured_output, dict):
            structured_output = {}
        sql_query = str(structured_output.get("sql_query")).strip()
        if sql_query:
            print(f"[SQL Agent] execute_sql: {sql_query}")
        else:
            print("[SQL Agent] execute_sql: EMPTY QUERY")
        query_log = list(state.get("query_log", []))
        result: Any
        if sql_query:
            try:
                conn = connect_postgres()
            except Exception as exc:  # pragma: no cover - surface connection issues
                raise RuntimeError(
                    "Failed to connect to PostgreSQL. Ensure POSTGRES_DSN is set and reachable."
                ) from exc
            try:
                result = execute_sql_tool(conn, sql_query)
            finally:
                try:
                    conn.close()  # type: ignore[union-attr]
                except Exception:
                    pass
        else:
            result = "SQL execution error: empty query."

        if isinstance(result, str):
            rows: List[Dict[str, Any]] = []
            payload: Dict[str, Any] = {
                "output_type": "execute_sql_result",
                "output_content": f"SQL execution error: {result}",
                "sql_query": sql_query,
                "status": "error",
                "error_message": result,
                "row_count": 0,
            }
        else:
            notices: List[str] = []
            cleaned_rows: List[Dict[str, Any]] = []
            for row in result:
                if isinstance(row, dict) and "__notice__" in row:
                    notices.append(str(row.get("__notice__")))
                else:
                    cleaned_rows.append(row)
            rows = cleaned_rows
            payload = {
                "output_type": "execute_sql_result",
                "output_content": f"Returned {len(rows)} row(s).",
                "sql_query": sql_query,
                "status": "success",
                "row_count": len(rows),
                "rows": rows,
            }
            if notices:
                payload["notices"] = notices

        log_entry: Dict[str, Any] = {
            "entry_type": "sql_result",
            "sql_query": sql_query,
            "status": payload.get("status", "unknown"),
            "row_count": int(payload.get("row_count", 0)),
            "rows": rows,
            "error_message": payload.get("error_message"),
            "notices": payload.get("notices", []),
            "persisted": False,
            "reference_key": None,
            "description": None,
            "datastore_ref": None,
        }
        query_log.append(log_entry)

        message = AgentMessage(
            name="execute_sql_tool",
            structured_output=payload,
        )
        print(f"[SQL Agent] execute_sql_result: status={payload['status']} row_count={payload['row_count']}")
        return {
            "messages": [message],
            "query_log": query_log,
        }

    def summarize_datastore_updates_node(state: SQLState) -> SQLState:
        query_log = list(state.get("query_log", []))
        datastore_obj = state.get("datastore")
        schema_shared = bool(state.get("database_schema_shared"))

        datastore_summary_parts: List[str] = []
        if isinstance(datastore_obj, DataStore):
            try:
                stats = datastore_obj.stats()
                namespaces = stats.get("namespaces", [])
                namespace_text = ", ".join(namespaces) if namespaces else "none"
                datastore_summary_parts.append(
                    f"Datastore holds {stats.get('items', 0)} item(s) across namespaces: {namespace_text}."
                )
            except Exception:
                datastore_summary_parts.append("Unable to read datastore statistics.")
        else:
            datastore_summary_parts.append("No datastore instance available.")

        persisted_summaries: List[str] = []
        pending_summaries: List[str] = []
        for entry in query_log:
            entry_type = entry.get("entry_type")

            if entry_type == "persistence_summary":
                reference_key = str(entry.get("reference_key", "")).strip() or "unnamed_dataset"
                description = str(entry.get("description", "")).strip()
                row_count = int(entry.get("row_count", 0))
                note = str(entry.get("note", "")).strip()
                desc_text = f" – {description}" if description else ""
                note_text = f" Note: {note}" if note else ""
                persisted_summaries.append(
                    f"{reference_key} ({row_count} row(s){desc_text}).{note_text}"
                )
                continue

            if entry_type != "sql_result":
                continue

            sql_query = str(entry.get("sql_query", "") or "").strip()
            status = str(entry.get("status", "unknown"))
            row_count = int(entry.get("row_count", 0))
            error_message = entry.get("error_message")
            notices = entry.get("notices") or []

            if status == "error":
                reason = error_message or "unknown error"
                pending_summaries.append(
                    f"Error while running `{sql_query or 'EMPTY QUERY'}`: {reason}."
                )
                continue

            notice_text = f" Notice: {'; '.join(map(str, notices))}." if notices else ""
            truncated_query = (sql_query[:120] + "…") if len(sql_query) > 120 else sql_query
            pending_summaries.append(
                f"Pending result from `{truncated_query}` with {row_count} row(s).{notice_text}"
            )

        sections: List[str] = datastore_summary_parts
        if persisted_summaries:
            sections.append("Persisted datasets: " + " ".join(persisted_summaries))
        else:
            sections.append("Persisted datasets: none yet.")

        if pending_summaries:
            sections.append("Pending SQL results: " + " ".join(pending_summaries))
        else:
            sections.append("Pending SQL results: none.")

        summary_text = " ".join(section.strip() for section in sections if section).strip()
        if not summary_text:
            summary_text = "No SQL activity recorded yet."

        payload = {
            "output_type": "summarize_datastore_updates_tool",
            "output_content": summary_text,
        }
        state_updates: Dict[str, Any] = {}
        if not schema_shared:
            database_schema, schema_source = _load_database_schema_snapshot()
            schema_summary = _format_database_schema_summary(database_schema)
            payload["output_content"] = (
                f"{summary_text} Database schema snapshot included (source: {schema_source})."
            )
            payload.update(
                {
                    "database_schema_source": schema_source,
                    "database_schema_summary": schema_summary,
                    "database_schema": database_schema,
                }
            )
            state_updates.update(
                {
                    "database_schema_source": schema_source,
                    "database_schema_summary": schema_summary,
                    "database_schema": database_schema,
                    "database_schema_shared": True,
                }
            )

        message = AgentMessage(
            name="summarize datastore updates",
            structured_output=payload,
        )
        print(f"[SQL Agent] summarize_datastore_updates_tool: {summary_text}")
        return {"messages": [message], **state_updates}

    def persist_dataset_node(state: SQLState) -> SQLState:
        last_message = state["messages"][-1]
        structured_output = getattr(last_message, "structured_output", {}) or {}
        if not isinstance(structured_output, dict):
            structured_output = {}
        reference_key = str(structured_output.get("reference_key")).strip()
        description = str(structured_output.get("description")).strip()

        datastore: DataStore = state.get("datastore")
        if not isinstance(datastore, DataStore):
            datastore = DATASTORE
        query_log = list(state.get("query_log", []))
        last_entry = query_log[-1] if query_log else {}
        rows = list(last_entry.get("rows", []) or [])
        error_message = last_entry.get("error_message")
        status = str(last_entry.get("status", ""))

        entry_type = last_entry.get("entry_type")
        persisted = bool(reference_key) and status == "success" and entry_type == "sql_result"
        datastore_ref: str | None = None
        if persisted:
            df = pd.DataFrame(rows)
            datastore_ref = datastore.put(
                df,
                namespace="sql_agent",
                description=description,
                ref=reference_key,
                upsert=True,
            )
        warning: str | None
        if not reference_key:
            warning = "Missing reference key; dataset not persisted."
        elif status != "success":
            warning = error_message or "Latest SQL execution failed; persistence skipped."
        else:
            warning = None

        row_count = len(rows)
        summary_entry: Dict[str, Any] | None = None
        if persisted:
            if query_log:
                query_log.pop()
            summary_entry = {
                "entry_type": "persistence_summary",
                "sql_query": last_entry.get("sql_query"),
                "reference_key": reference_key or "",
                "description": description or "",
                "row_count": row_count,
                "datastore_ref": datastore_ref,
                "persisted": True,
                "note": (
                    f"Persisted {row_count} row(s) to `{reference_key}`."
                    if reference_key
                    else f"Persisted {row_count} row(s) to datastore."
                ),
            }
            query_log.append(summary_entry)
        elif query_log:
            updated_entry = dict(last_entry)
            updated_entry.update(
                {
                    "persisted": persisted,
                    "reference_key": reference_key or None,
                    "description": description or None,
                    "datastore_ref": datastore_ref,
                }
            )
            query_log[-1] = updated_entry

        payload = {
            "output_type": "persist_dataset_result",
            "output_content": f"Persisted rows: {row_count}." if persisted else "Persistence skipped.",
            "reference_key": reference_key,
            "description": description,
            "row_count": row_count,
            "persisted": persisted,
            "warning": warning,
            "datastore_ref": datastore_ref,
        }

        message = AgentMessage(
            name="persist_dataset_tool",
            structured_output=payload,
        )
        status_text = "success" if persisted else "skipped"
        print(f"[SQL Agent] persist_dataset_result: {status_text} reference_key={reference_key or 'None'}")

        return {
            "messages": [message],
            "datastore": datastore,
            "query_log": query_log,
        }

    def record_final_answer_node(state: SQLState) -> SQLState:
        last_message = state["messages"][-1]
        final_answer = ""
        if isinstance(last_message, AgentMessage):
            final_answer = str(last_message.structured_output.get("output_content", ""))
        return {"sql_agent_final_answer": final_answer}

    workflow = StateGraph(SQLState)
    workflow.add_node("initialize", initialize_state)
    workflow.add_node("controller", controller_node)
    workflow.add_node("summarize_datastore_updates", summarize_datastore_updates_node)
    workflow.add_node("execute_sql", execute_sql_node)
    workflow.add_node("persist_dataset", persist_dataset_node)
    workflow.add_node("record_final", record_final_answer_node)

    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "controller")
    workflow.add_edge("summarize_datastore_updates", "controller")
    workflow.add_edge("execute_sql", "controller")
    workflow.add_edge("persist_dataset", "controller")
    workflow.add_edge("record_final", "controller")
    workflow.add_conditional_edges(
        "controller",
        route_controller,
        {
            "summarize_datastore_updates": "summarize_datastore_updates",
            "execute_sql": "execute_sql",
            "persist_dataset": "persist_dataset",
            "record_final": "record_final",
            "end": END,
            "controller": "controller",
        },
    )

    sql_agent = workflow.compile()
    return sql_agent
