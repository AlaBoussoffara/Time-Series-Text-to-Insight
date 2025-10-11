import os
from pathlib import Path
from typing import Annotated, Literal, Optional, Sequence, TypedDict

import dotenv
from langchain_core.messages import AIMessage, AnyMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from llm import llm_from

dotenv.load_dotenv()

PROMPT_PATH = Path("supervisor_prompt.txt")
SUPERVISOR_PROMPT_TEXT = PROMPT_PATH.read_text()


class Step(BaseModel):
    output: Literal["plan", "thought", "final_answer", "SQL Agent", "Analysis Agent", "Visualization Agent"] = Field(
        ..., description="Either 'plan', 'thought', 'final_answer', or an agent name."
    )
    content: str = Field(..., description="content for the chosen output.")


class OverallState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def _system_prompt() -> SystemMessage:
    return SystemMessage(SUPERVISOR_PROMPT_TEXT)


def load_llm(provider: Optional[str] = None):
    """Return an LLM instance for the given provider or the environment default."""
    use_model = provider or os.getenv("USE_MODEL", "ollama")
    return llm_from(use_model)


def build_supervisor_graph(llm) -> StateGraph:
    """Build and compile the supervisor graph for the provided LLM."""
    llm_supervisor = llm.with_structured_output(Step)

    def supervisor_node(state: OverallState) -> OverallState:
        step = llm_supervisor.invoke(state["messages"])
        ai_msg = AIMessage(
            content=step.model_dump_json(),
            additional_kwargs={"structured": step.model_dump()},
            name="Supervisor",
        )
        return {"messages": [ai_msg]}

    def route_supervisor(state: OverallState) -> str:
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage):
            structured = last_message.additional_kwargs.get("structured", {})
            return structured.get("output", "thought")
        return "thought"

    def sql_agent_node(state: OverallState) -> OverallState:
        ai_msg = AIMessage(content="SQL query executed successfully.", name="SQL Agent")
        return {"messages": [ai_msg]}

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
            "final_answer": END,
            "thought": "supervisor",
            "plan": "supervisor",
        },
    )
    builder.add_edge("SQLAgent", "supervisor")
    builder.add_edge("AnalysisAgent", "supervisor")
    builder.add_edge("VisualizationAgent", "supervisor")
    return builder.compile()


def _format_message(message: BaseMessage) -> tuple[str, str]:
    if isinstance(message, SystemMessage):
        return "System", message.content
    if isinstance(message, HumanMessage):
        label = message.name or "User"
        return label, message.content
    if isinstance(message, AIMessage):
        structured = message.additional_kwargs.get("structured")
        if structured:
            label = structured.get("output", message.name or "Supervisor")
            content = structured.get("content", message.content)
        else:
            label = message.name or "Agent"
            content = message.content
        return label, content
    label = getattr(message, "name", getattr(message, "type", message.__class__.__name__))
    content = getattr(message, "content", "")
    return label, content


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
                    print(f"[{node}] {label}: {content}")
    return final_messages or collected_messages


def run_supervisor(
    user_input: str,
    *,
    history: Optional[Sequence[BaseMessage]] = None,
    log: bool = True,
    llm_name: Optional[str] = None,
):
    """Execute the supervisor graph and return the final message."""
    llm = load_llm(llm_name)
    compiled_graph = build_supervisor_graph(llm)
    messages: list[AnyMessage] = [_system_prompt()]
    if history:
        for msg in history:
            if isinstance(msg, BaseMessage):
                messages.append(msg)
    messages.append(HumanMessage(user_input))
    state = {"messages": messages}

    if log:
        final_messages = _stream_graph(compiled_graph, state, log=log)
    else:
        result = compiled_graph.invoke(state)
        final_messages = result["messages"]

    return final_messages[-1] if final_messages else None


if __name__ == "__main__":
    sample_input = (
        "extract insights from sensor data where temperature > 30 and humidity < 50, "
        "then visualize the trends over time."
    )
    final_message = run_supervisor(sample_input, log=True)
    if final_message:
        label, content = _format_message(final_message)
        print(f"Final message ({label}): {content}")
