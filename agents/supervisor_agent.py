from pathlib import Path
from typing import Optional, Sequence
import os
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph


from agents.analysis_agent import create_analysis_agent
from agents.spider_sql_agent import PromptAgentAdapter
from agents.sql_agent import create_sql_agent
from agents.visualisation_agent import create_visualization_agent
from utils.general_helpers import format_message, llm_from, stream_graph
from utils.messages import AgentMessage
from utils.states import GlobalState
from utils.output_basemodels import *
from utils.datastore import DATASTORE, DataStore


def build_supervisor_graph() -> StateGraph:
    """Build and compile the Supervisor graph."""
    
    supervisor_llm = llm_from("aws", "anthropic.claude-3-5-sonnet-20241022-v2:0").with_structured_output(SupervisorOutput)
    
    sql_llm = llm_from("aws", "anthropic.claude-3-5-sonnet-20241022-v2:0").with_structured_output(SQLAgentOutput)
    analysis_llm = llm_from("aws", "anthropic.claude-3-5-sonnet-20241022-v2:0").with_structured_output(AnalysisAgentOutput)
    visualization_llm = llm_from("aws", "anthropic.claude-3-haiku-20240307-v1:0").with_structured_output(VisualizationPlanOutput)
    
    if os.getenv("SQL_AGENT_MODE") == "SPIDER" :
        sql_agent = PromptAgentAdapter()
    else :
        sql_agent = create_sql_agent(sql_llm)
    analysis_agent = create_analysis_agent(analysis_llm)
    visualization_agent = create_visualization_agent(visualization_llm)
    
    def supervisor_node(state: GlobalState) -> GlobalState:
        answer = supervisor_llm.invoke(state["global_messages_history"])
        structured = answer.model_dump()
        agent_msg = AgentMessage(
            name="Supervisor",
            structured_output=structured,
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
        sql_structured_output = {
            "output_type": "SQL Agent",
            "output_content": sql_final_answer,
            "datastore_summary": datastore_snapshot,
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
        warnings = response.get("warnings", [])
        error_message = response.get("error_message")
        visualization_structured_output = {
            "output_type": "Visualization Agent",
            "output_content": visualization_final_answer,
            "chart_path": output_path,
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


if __name__ == "__main__":
    sample_input = (
        "extract insights from sensor data where temperature > 30 and humidity < 50, "
        "then visualize the trends over time."
    )
    run_supervisor(sample_input, log=True)
