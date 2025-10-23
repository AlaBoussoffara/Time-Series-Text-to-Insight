from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from utils.datastore import DATASTORE, DataStore
from utils.output_basemodels import SQLAgentOutput
from utils.sql_utils import connect_postgres, execute_sql_tool
from utils.states import SQLState

PROMPT_PATH = Path("prompts/sql_agent_prompt.txt")
SQL_AGENT_PROMPT_TEXT = PROMPT_PATH.read_text(encoding="utf-8")
DATABASE_PROMPT_PATH = Path("prompts/database_prompt.txt")
DATABASE_CONTEXT = DATABASE_PROMPT_PATH.read_text(encoding="utf-8")


def create_sql_agent(llm):
    """
    Build the minimal SQL controller graph.
    """

    try:
        conn = connect_postgres()
    except Exception as exc:  # pragma: no cover - surface connection issues
        raise RuntimeError(
            "Failed to connect to PostgreSQL. Ensure POSTGRES_DSN is set and reachable."
        ) from exc

    controller_llm = llm.with_structured_output(SQLAgentOutput)
    system_prompt = SQL_AGENT_PROMPT_TEXT.replace("{database_context}", DATABASE_CONTEXT)

    def _get_last_structured(state: SQLState) -> Dict[str, Any]:
        last_message = state["messages"][-1]
        structured = last_message.additional_kwargs.get("structured", {})
        if not structured:
            try:
                structured = json.loads(last_message.content)
            except Exception:  # pragma: no cover - defensive
                structured = {}
        return structured

    def initialize_state(state: SQLState) -> SQLState:
        command = state.get("command", "")
        messages: List[Any] = [
            SystemMessage(system_prompt),
            HumanMessage(command),
        ]
        datastore = state.get("datastore")
        if not isinstance(datastore, DataStore):
            datastore = DATASTORE
        return {
            "messages": messages,
            "command": command,
            "sql_queries": list(state.get("sql_queries", [])),
            "datastore": datastore,
            "latest_payload": dict(state.get("latest_payload", {})),
            "datastore_updates": dict(state.get("datastore_updates", {})),
            "last_persist": dict(state.get("last_persist", {})),
            "final_answer": state.get("final_answer", ""),
        }

    def controller_node(state: SQLState) -> SQLState:
        response = controller_llm.invoke(state["messages"])
        ai_msg = AIMessage(
            content=response.model_dump_json(),
            additional_kwargs={"structured": response.model_dump()},
            name="SQL Controller",
        )
        structured = response.model_dump()
        output = structured.get("output", "")
        content = structured.get("content", "")
        if output:
            print(f"[SQL Controller] {output}: {content}")
        return {"messages": [ai_msg]}

    def route_controller(state: SQLState) -> str:
        structured = _get_last_structured(state)
        output = structured.get("output")
        if output == "execute_sql":
            return "execute_sql"
        if output == "persist_dataset":
            return "persist_dataset"
        if output == "final_answer":
            return "record_final"
        if output == "no_hallucination":
            return "complete"
        return "controller"

    def execute_sql_node(state: SQLState) -> SQLState:
        structured = _get_last_structured(state)
        sql_query = (structured.get("sql_query") or "").strip()
        sql_queries = list(state.get("sql_queries", []))
        if sql_query:
            sql_queries.append(sql_query)

        result = execute_sql_tool(conn, sql_query) if sql_query else "SQL execution error: empty query."

        if isinstance(result, str):
            rows: List[Dict[str, Any]] = []
            payload: Dict[str, Any] = {
                "output": "execute_sql_result",
                "sql_query": sql_query,
                "status": "error",
                "error_message": result,
                "row_count": 0,
            }
        else:
            rows = result
            payload = {
                "output": "execute_sql_result",
                "sql_query": sql_query,
                "status": "success",
                "row_count": len(rows),
                "rows": rows,
            }

        latest_payload = {
            "sql_query": sql_query,
            "rows": rows,
            "error_message": payload.get("error_message"),
        }

        message = AIMessage(
            content=json.dumps(payload),
            additional_kwargs={"structured": payload},
            name="execute_sql",
        )
        print(f"[SQL Agent] execute_sql_result: status={payload['status']} row_count={payload['row_count']}")
        return {
            "messages": [message],
            "sql_queries": sql_queries,
            "latest_payload": latest_payload,
        }

    def persist_dataset_node(state: SQLState) -> SQLState:
        structured = _get_last_structured(state)
        reference_key = (structured.get("reference_key") or "").strip()
        description = (structured.get("description") or "").strip()

        datastore: DataStore = state.get("datastore")  # type: ignore[assignment]
        latest_payload = dict(state.get("latest_payload", {}))
        rows = latest_payload.get("rows", []) or []
        error_message = latest_payload.get("error_message")

        persisted = bool(reference_key) and not error_message
        datastore_ref: str | None = None
        if persisted:
            df = pd.DataFrame(rows)
            datastore_ref = datastore.put_df(df, namespace="sql_agent")

        datastore_updates = dict(state.get("datastore_updates", {}))
        if reference_key:
            datastore_updates[reference_key] = {
                "description": description,
                "data": rows,
                "datastore_ref": datastore_ref,
            }
        last_persist = {
            "reference_key": reference_key,
            "description": description,
            "rows": rows,
            "datastore_ref": datastore_ref,
        }

        payload = {
            "output": "persist_dataset_result",
            "reference_key": reference_key,
            "description": description,
            "row_count": len(rows),
            "persisted": persisted,
            "warning": error_message if error_message else None,
            "datastore_ref": datastore_ref,
        }

        message = AIMessage(
            content=json.dumps(payload),
            additional_kwargs={"structured": payload},
            name="persist_dataset",
        )
        status = "success" if persisted else "skipped"
        print(f"[SQL Agent] persist_dataset_result: {status} reference_key={reference_key or 'None'}")

        return {
            "messages": [message],
            "datastore": datastore,
            "datastore_updates": datastore_updates,
            "last_persist": last_persist,
        }

    def record_final_answer_node(state: SQLState) -> SQLState:
        structured = _get_last_structured(state)
        final_answer = structured.get("content", "")
        return {"final_answer": final_answer}

    def complete_node(state: SQLState) -> SQLState:
        datastore_updates = dict(state.get("datastore_updates", {}))
        final_answer = state.get("final_answer", "")

        clean_datastore = {
            key: {
                "description": value.get("description", ""),
                "data": value.get("data", []),
                "datastore_ref": value.get("datastore_ref"),
            }
            for key, value in datastore_updates.items()
            if key
        }

        return {
            "datastore": clean_datastore,
            "final_answer": final_answer,
        }

    workflow = StateGraph(SQLState)
    workflow.add_node("initialize", initialize_state)
    workflow.add_node("controller", controller_node)
    workflow.add_node("execute_sql", execute_sql_node)
    workflow.add_node("persist_dataset", persist_dataset_node)
    workflow.add_node("record_final", record_final_answer_node)
    workflow.add_node("complete", complete_node)

    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "controller")
    workflow.add_edge("execute_sql", "controller")
    workflow.add_edge("persist_dataset", "controller")
    workflow.add_edge("record_final", "controller")
    workflow.add_conditional_edges(
        "controller",
        route_controller,
        {
            "execute_sql": "execute_sql",
            "persist_dataset": "persist_dataset",
            "record_final": "record_final",
            "complete": "complete",
            "controller": "controller",
        },
    )
    workflow.add_edge("complete", END)

    sql_agent = workflow.compile()
    return sql_agent
