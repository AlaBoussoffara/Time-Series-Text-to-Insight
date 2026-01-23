from pathlib import Path
from typing import Any, Optional, Sequence
import os
import json
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph


from agents.analysis_agent import create_analysis_agent
from agents.spider_sql_agent import PromptAgentAdapter
from agents.sql_agent import create_sql_agent
from agents.visualisation_agent import create_visualization_agent
from utils.general_helpers import llm_from, stream_graph
from utils.messages import AgentMessage
from utils.states import GlobalState
from utils.api_call_counter import (
    append_api_call_log,
    print_api_call_breakdown,
    reset_api_call_count,
)
from utils.token_counter import (
    print_token_usage,
    reset_token_usage,
    wrap_llm_with_token_counter,
)
from utils.output_basemodels import *
from utils.datastore import DATASTORE, DataStore


_SUPERVISOR_OUTPUT_TYPES = {
    "plan",
    "thought",
    "supervisor_final_answer",
    "hallucination",
    "no_hallucination",
    "SQL Agent",
    "Analysis Agent",
    "Visualization Agent",
}


def _coerce_supervisor_output(answer: Any) -> tuple[dict[str, Any], bool]:
    if answer is None:
        return {
            "output_type": "supervisor_final_answer",
            "output_content": "Sorry, I couldn't generate a response. Please try again.",
        }, True
    if hasattr(answer, "model_dump"):
        try:
            structured = answer.model_dump()
        except Exception:
            structured = {}
    elif isinstance(answer, dict):
        structured = dict(answer)
    else:
        content = getattr(answer, "content", None)
        structured = {
            "output_type": "supervisor_final_answer",
            "output_content": str(content or answer),
        }
    output_type = structured.get("output_type")
    output_content = structured.get("output_content")
    if output_type not in _SUPERVISOR_OUTPUT_TYPES:
        if not output_content:
            output_content = "Sorry, I couldn't generate a structured response. Please try again."
        return {
            "output_type": "supervisor_final_answer",
            "output_content": str(output_content),
        }, True
    if output_content is None:
        structured["output_content"] = ""
    return structured, False


def _flatten_history(messages: Sequence[BaseMessage]) -> str:
    lines: list[str] = []
    for msg in messages:
        role = getattr(msg, "type", msg.__class__.__name__)
        name = getattr(msg, "name", None)
        content = getattr(msg, "content", "")
        if not isinstance(content, str):
            content = str(content)
        label = role
        if name:
            label = f"{role} name={name}"
        lines.append(f"{label}: {content}")
    return "\n".join(lines)


def build_supervisor_graph() -> StateGraph:
    """Build and compile the Supervisor graph."""
    
    supervisor_llm = wrap_llm_with_token_counter(
        llm_from(
            agent_name="Supervisor",
        )
    )
    
    sql_llm = wrap_llm_with_token_counter(
        llm_from(
            agent_name="SQL Agent",
        )
    )
    analysis_llm = llm_from(
        agent_name="Analysis Agent",
    ).with_structured_output(AnalysisAgentOutput)
    visualization_llm = llm_from(
        agent_name="Visualization Agent",
    ).with_structured_output(VisualizationCodeOutput)
    visualization_agent = create_visualization_agent(visualization_llm)
    
    if os.getenv("SQL_AGENT_MODE") == "SPIDER" :
        sql_agent = PromptAgentAdapter()
    else :
        sql_agent = create_sql_agent(sql_llm)
    analysis_agent = create_analysis_agent(analysis_llm)

    def supervisor_node(state: GlobalState) -> GlobalState:
        history = list(state.get("global_messages_history") or [])
        system_content = ""
        non_system: list[BaseMessage] = []
        for msg in history:
            if isinstance(msg, SystemMessage) and not system_content:
                system_content = msg.content or ""
                continue
            non_system.append(msg)
        flattened = _flatten_history(non_system)
        prompt_messages: list[BaseMessage] = [
            SystemMessage(system_content),
            HumanMessage("Conversation history:\n" + flattened),
        ]
        
        # Anti-Loop Logic: Check if the last decision was a 'plan'
        last_message = history[-1] if history else None
        if isinstance(last_message, AgentMessage) and last_message.name == "Supervisor":
             structured = last_message.structured_output
             if structured.get("output_type") == "plan":
                 print("[Supervisor] Anti-loop: Plan detected. Injecting execution mandate.")
                 prompt_messages.append(
                     HumanMessage(
                         content=(
                             "CRITICAL INSTRUCTION: You have just created a plan. "
                             "DO NOT PLAN AGAIN. "
                             "You MUST now EXECUTE the first step of your plan. "
                             "Output the name of the agent to call (e.g., 'SQL Agent') or 'supervisor_final_answer'."
                         )
                     )
                 )

        response = supervisor_llm.invoke(prompt_messages)
        
        # Parse the raw AIMessage content using robust JSON decoding
        content = str(getattr(response, "content", str(response)))
        
        raw_structured = {}
        try:
            start_index = content.find("{")
            if start_index != -1:
                decoder = json.JSONDecoder()
                raw_structured, _ = decoder.raw_decode(content, start_index)
            else:
                raw_structured = {
                    "output_type": "thought",
                    "output_content": content.strip()
                }
        except Exception as e:
            raw_structured = {
                "output_type": "thought",
                "output_content": f"JSON Parsing Error: {e}. Raw content: {content[:100]}..."
            }
            
        # Ensure output_type is valid if possible
        if "output_type" not in raw_structured and "output_content" not in raw_structured:
             # Try to salvage if it's just a string returned
             raw_structured["output_type"] = "thought"
             raw_structured["output_content"] = content.strip()

        agent_msg = AgentMessage(
            name="Supervisor",
            structured_output=raw_structured,
        )
        return {"global_messages_history": [agent_msg]}

    def route_supervisor(state: GlobalState) -> str:
        last_message = state["global_messages_history"][-1]
        if isinstance(last_message, AgentMessage):
            structured = last_message.structured_output
            return str(structured.get("output_type", "thought"))
        return "thought"

    def sql_agent_node(state: GlobalState) -> GlobalState:
        last_message = state["global_messages_history"][-1]
        instruction = ""
        if isinstance(last_message, AgentMessage):
            instruction = str(last_message.structured_output.get("output_content", ""))
        if not instruction:
            instruction = str(getattr(last_message, "content", ""))
        datastore = state.get("datastore")
        if not isinstance(datastore, DataStore):
            datastore = DATASTORE
        sql_agent_messages_history = list(state["sql_agent_messages_history"])
        response = sql_agent.invoke(
            {
                "messages": sql_agent_messages_history,
                "instruction": instruction,
                "datastore": datastore,
            }
        )
        sql_final_answer = response.get("sql_agent_final_answer", "SQL agent completed the task.")
        datastore = response.get("datastore", datastore)
        datastore_snapshot = datastore.snapshot() if isinstance(datastore, DataStore) else {}
        trimmed_history: list[BaseMessage] = list(sql_agent_messages_history)
        if instruction:
            trimmed_history.append(HumanMessage(instruction))
        response_messages = response.get("messages", []) or []
        if len(response_messages) >= 2:
            trimmed_history.append(response_messages[-2])
        elif response_messages:
            trimmed_history.append(response_messages[-1])
            
        # Extract executed SQL queries from the query log
        query_log = response.get("query_log", [])
        executed_sqls = []
        if isinstance(query_log, list):
            for entry in query_log:
                if isinstance(entry, dict) and entry.get("entry_type") in (
                    "sql_result",
                    "persistence_summary",
                ):
                    sql = entry.get("sql_query")
                    if sql:
                        executed_sqls.append(sql)

        # Sanitize datastore_snapshot to ensure JSON serializability
        safe_snapshot = {}
        if isinstance(datastore_snapshot, dict):
            for key, val in datastore_snapshot.items():
                if isinstance(val, (str, int, float, bool, type(None))):
                    safe_snapshot[key] = val
                elif isinstance(val, (list, dict)):
                    # For lists and dicts, convert to string if they might contain complex objects
                    try:
                        json.dumps(val)
                        safe_snapshot[key] = val
                    except (TypeError, ValueError):
                        safe_snapshot[key] = str(val)
                else:
                    safe_snapshot[key] = str(val)
        
        sql_structured_output = {
            "output_type": "SQL Agent",
            "output_content": str(sql_final_answer),
            "datastore_summary": safe_snapshot,
            "sql_queries": [str(q) for q in executed_sqls],
        }
        return {
            "datastore": datastore,
            "global_messages_history": [
                AgentMessage(
                    name="SQL Agent",
                    structured_output=sql_structured_output,
                )
            ],
            "sql_agent_messages_history": trimmed_history
        }

    def analysis_agent_node(state: GlobalState) -> GlobalState:
        last_message = state["global_messages_history"][-1]
        instruction = ""
        if isinstance(last_message, AgentMessage):
            instruction = str(last_message.structured_output.get("output_content", ""))
        if not instruction:
            instruction = str(getattr(last_message, "content", ""))
        datastore = state.get("datastore")
        if not isinstance(datastore, DataStore):
            datastore = DATASTORE
        datastore_snapshot = datastore.snapshot()
        response = analysis_agent.invoke(
            {
                "instruction": instruction,
                "datastore": datastore_snapshot,
                "datastore_obj": datastore,
            }
        )
        analysis_final_answer = response.get(
            "analysis_agent_final_answer", "Analysis agent completed the task."
        )
        insights = response.get("insights", [])
        follow_ups = response.get("follow_up_questions", [])
        referenced_keys = response.get("referenced_keys", [])
        error_message = response.get("error_message")

        sections: list[str] = [analysis_final_answer]
        if insights:
            sections.append("Insights:\n" + "\n".join(f"- {item}" for item in insights))
        if follow_ups:
            sections.append("Follow-up suggestions:\n" + "\n".join(f"- {item}" for item in follow_ups))
        if error_message:
            sections.append(f"Warning: {error_message}")
        content = "\n\n".join(sections)
        analysis_structured_output = {
            "output_type": "Analysis Agent",
            "output_content": content,
            "insights": insights,
            "follow_up_questions": follow_ups,
            "referenced_keys": referenced_keys,
            "error_message": error_message,
        }
        return {
            "global_messages_history": [
                AgentMessage(
                    name="Analysis Agent",
                    structured_output=analysis_structured_output,
                )
            ],
            "datastore": datastore,
        }

    def visualization_agent_node(state: GlobalState) -> GlobalState:
        last_message = state["global_messages_history"][-1]
        instruction = ""
        if isinstance(last_message, AgentMessage):
            instruction = str(last_message.structured_output.get("output_content", ""))
        if not instruction:
            instruction = str(getattr(last_message, "content", ""))
        datastore = state.get("datastore")
        if not isinstance(datastore, DataStore):
            datastore = DATASTORE
        datastore_snapshot = datastore.snapshot()
        response = visualization_agent.invoke(
            {
                "instruction": instruction,
                "datastore": datastore_snapshot,
                "datastore_obj": datastore,
            }
        )
        visualization_final_answer = response.get(
            "visualization_agent_final_answer", "Visualization created successfully."
        )
        output_path = response.get("output_path")
        output_paths = response.get("output_paths") or ([] if not output_path else [output_path])
        visualizations = response.get("visualizations") or []
        warnings = response.get("warnings", [])
        error_message = response.get("error_message")
        visualization_structured_output = {
            "output_type": "Visualization Agent",
            "output_content": visualization_final_answer,
            "chart_path": output_path,
            "chart_paths": output_paths,
            "visualizations": visualizations,
            "warnings": warnings,
            "error_message": error_message,
        }
        return {
            "global_messages_history": [
                AgentMessage(
                    name="Visualization Agent",
                    structured_output=visualization_structured_output,
                )
            ],
            "datastore": datastore,
        }

    builder = StateGraph(GlobalState)
    builder.add_node("Supervisor", supervisor_node)
    builder.add_node("SQLAgent", sql_agent_node)
    builder.add_node("AnalysisAgent", analysis_agent_node)
    builder.add_node("VisualizationAgent", visualization_agent_node)

    builder.add_edge(START, "Supervisor")
    builder.add_conditional_edges(
        "Supervisor",
        route_supervisor,
        {
            "SQL Agent": "SQLAgent",
            "Analysis Agent": "AnalysisAgent",
            "Visualization Agent": "VisualizationAgent",
            "supervisor_final_answer": "Supervisor",
            "thought": "Supervisor",
            "plan": "Supervisor",
            "hallucination": "Supervisor",
            "no_hallucination": END,
        },
    )
    builder.add_edge("SQLAgent", "Supervisor")
    builder.add_edge("AnalysisAgent", "Supervisor")
    builder.add_edge("VisualizationAgent", "Supervisor")
    return builder.compile()


def run_supervisor(
    user_input: str,
    *,
    history: Optional[Sequence[BaseMessage]] = None,
    log: bool = True,
):
    """Execute the Supervisor graph and return the final message."""
    reset_api_call_count()
    reset_token_usage()
    try:
        compiled_graph = build_supervisor_graph()
        SUPERVISOR_PROMPT_TEXT = Path("prompts/supervisor_prompt.txt"
                                      ).read_text(encoding="utf-8")

        messages: list[BaseMessage] = [SystemMessage(SUPERVISOR_PROMPT_TEXT)]
        if history:
            for msg in history:
                if isinstance(msg, BaseMessage):
                    messages.append(msg)
        messages.append(HumanMessage(user_input))
        global_state = {
            "global_messages_history": messages,
            "datastore": DATASTORE,
            "database_schema": {},
        }

        final_messages = stream_graph(compiled_graph, global_state, log=log)
        visualization_artifacts: list[dict] = []
        for message in final_messages:
            if isinstance(message, AgentMessage) and getattr(message, "name", "") == "Visualization Agent":
                structured = message.structured_output
                artifacts = structured.get("visualizations") or []
                if artifacts:
                    for item in artifacts:
                        visualization_artifacts.append(
                            {
                                "chart_path": item.get("chart_path"),
                                "warnings": structured.get("warnings", []),
                                "summary": item.get("summary") or structured.get("output_content"),
                                "error_message": structured.get("error_message"),
                            }
                        )
                else:
                    chart_path = structured.get("chart_path")
                    if chart_path:
                        visualization_artifacts.append(
                            {
                                "chart_path": chart_path,
                                "warnings": structured.get("warnings", []),
                                "summary": structured.get("output_content"),
                                "error_message": structured.get("error_message"),
                            }
                        )
        final_response = final_messages[-2]
        if visualization_artifacts:
            setattr(final_response, "visualizations", visualization_artifacts)
        return final_response
    finally:
        print_api_call_breakdown("API calls for request")
        append_api_call_log("API calls for request")
        token_label = (
            "Spider tokens for request"
            if os.getenv("SQL_AGENT_MODE") == "SPIDER"
            else "SQL tokens for request"
        )
        print_token_usage(token_label)


if __name__ == "__main__":
    sample_input = (
        "extract insights from sensor data where temperature > 30 and humidity < 50, "
        "then visualize the trends over time."
    )
    run_supervisor(sample_input, log=True)
