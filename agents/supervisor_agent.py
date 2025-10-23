from pathlib import Path
from typing import Optional, Sequence
import dotenv
from langchain_core.messages import AIMessage, AnyMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from agents.sql_agent import create_sql_agent 
from utils.llm import llm_from 
from utils.states import OverallState
from utils.output_basemodels import SupervisorOutput

dotenv.load_dotenv()

def build_supervisor_graph() -> StateGraph:
    """Build and compile the supervisor graph using the Haiku model."""
    
    supervisor_llm = llm_from("aws", "anthropic.claude-3-5-sonnet-20241022-v2:0").with_structured_output(SupervisorOutput)
    
    sql_llm = llm_from("aws", "anthropic.claude-3-5-sonnet-20241022-v2:0")
    #analysis_llm = llm_from("aws", "anthropic.claude-3-5-sonnet-20241022-v2:0")
    #visualization_llm = llm_from("aws", "anthropic.claude-3-haiku-20240307-v1:0")
    
    sql_agent = create_sql_agent(sql_llm)
    #analysis_agent = create_analysis_agent(analysis_llm)
    #visualization_agent = create_visualization_agent(visualization_llm)
    
    def supervisor_node(state: OverallState) -> OverallState:
        answer = supervisor_llm.invoke(state["messages"])
        ai_msg = AIMessage(
            content=answer.model_dump_json(),
            additional_kwargs={"structured": answer.model_dump()},
            name="Supervisor",
        )
        return {"messages": [ai_msg]}

    def route_supervisor(state: OverallState) -> str:
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage):
            structured = last_message.additional_kwargs.get("structured", {})
            output = structured.get("output", "thought")
            if output in {"hallucination", "no_hallucination"}:
                return output
            return output
        return "thought"

    def sql_agent_node(state: OverallState) -> OverallState:
        res = sql_agent.invoke({"question": state["messages"][-1].additional_kwargs["structured"]["content"]})
        answer = res.get("answer", "SQL agent completed the task.")
        reference_key = res.get("reference_key")
        description = res.get("description", "")
        query_result = res.get("query_result", [])
        datastore_update = {}
        if reference_key:
            datastore_update[reference_key] = {
                "description": description,
                "data": query_result,
            }
        return {
            "datastore": datastore_update,
            "messages": [AIMessage(content=answer, name="SQL Agent")],
        }

    def analysis_agent_node(state: OverallState) -> OverallState:
        ai_msg = AIMessage(content="Data analysis completed successfully.", name="Analysis Agent")
        return {"messages": [ai_msg]}

    def visualization_agent_node(state: OverallState) -> OverallState:
        ai_msg = AIMessage(content="Visualization created successfully.", name="Visualization Agent")
        return {"messages": [ai_msg]}

    builder = StateGraph(OverallState)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("SQLAgent", sql_agent_node)
    builder.add_node("AnalysisAgent", analysis_agent_node)
    builder.add_node("VisualizationAgent", visualization_agent_node)

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "SQL Agent": "SQLAgent",
            "Analysis Agent": "AnalysisAgent",
            "Visualization Agent": "VisualizationAgent",
            "final_answer": "supervisor",
            "thought": "supervisor",
            "plan": "supervisor",
            "hallucination": "supervisor",
            "no_hallucination": END,
        },
    )
    builder.add_edge("SQLAgent", "supervisor")
    builder.add_edge("AnalysisAgent", "supervisor")
    builder.add_edge("VisualizationAgent", "supervisor")
    return builder.compile()


def _format_message(message: BaseMessage) -> tuple[Optional[str], str]:
    """Return a label/content pair"""
    content = getattr(message, "content", "")
    if isinstance(message, AIMessage):
        structured = message.additional_kwargs.get("structured")
        if message.name == "Supervisor" or structured:
            label = ""
            if structured:
                label = structured.get("output") or ""
                content = structured.get("content", content)
            if not label:
                label = message.name or getattr(message, "type", message.__class__.__name__)
            return label, content
        return None, content
    return None, content


def _stream_graph(compiled_graph, state, *, log: bool = True) -> list[AnyMessage]:
    seen_ids: set[str] = set()
    collected_messages: list[AnyMessage] = []
    final_messages: list[AnyMessage] = []
    for event in compiled_graph.stream(state):
        for node, payload in event.items():
            if node == "__end__":
                final_messages = payload.get("messages", final_messages)  # type: ignore[assignment]
                continue
            for message in payload.get("messages", []):
                msg_id = getattr(message, "id", None)
                key = msg_id or f"{node}-{id(message)}"
                if key in seen_ids:
                    continue
                seen_ids.add(key)
                collected_messages.append(message)
                if log:
                    label, content = _format_message(message)
                    if label:
                        print(f"[{node}] {label}: {content}")
                    else:
                        print(f"[{node}] {content}")
    return collected_messages


def run_supervisor(
    user_input: str,
    *,
    history: Optional[Sequence[BaseMessage]] = None,
    log: bool = True,
):
    """Execute the supervisor graph and return the final message."""
    compiled_graph = build_supervisor_graph()
    SUPERVISOR_PROMPT_TEXT = Path("prompts/supervisor_prompt.txt"
                                  ).read_text(encoding="utf-8")

    messages: list[AnyMessage] = [SystemMessage(SUPERVISOR_PROMPT_TEXT)]
    if history:
        for msg in history:
            if isinstance(msg, BaseMessage):
                messages.append(msg)
    messages.append(HumanMessage(user_input))
    state = {"messages": messages}

    final_messages = _stream_graph(compiled_graph, state, log=log)

    if not final_messages:
        return None

    for message in reversed(final_messages):
        if isinstance(message, AIMessage):
            structured = message.additional_kwargs.get("structured", {})
            if structured.get("output") == "final_answer":
                return message

    return final_messages[-1]


if __name__ == "__main__":
    sample_input = (
        "extract insights from sensor data where temperature > 30 and humidity < 50, "
        "then visualize the trends over time."
    )
    final_message = run_supervisor(sample_input, log=True)
    if final_message:
        label, content = _format_message(final_message)
        if label:
            print(f"Final message ({label}): {content}")
        else:
            print(f"Final message: {content}")
