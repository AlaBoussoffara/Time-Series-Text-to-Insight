from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, TypedDict
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
ANALYSIS_SYSTEM_PROMPT = """You are a senior data analyst collaborating with a supervisor agent.
You receive:
- A natural language analysis request.
- A compact inventory of tabular datasets that were generated earlier in the conversation.
Your job is to synthesise insights that directly address the request, referencing the datastore keys
that were used. If no data is available, explain why and suggest next steps.
Return structured JSON with the following fields:
1. answer: A concise narrative (1-2 paragraphs, no bullet points) that summarises the insights for the supervisor.
   Mention the datastore keys, row counts, notable metrics, and caveats.
2. insights: Bullet-level insights (each entry is a short sentence) derived from the data.
3. follow_up_questions: Optional follow-up questions or actions that the supervisor could pursue next.
4. referenced_keys: Datastore keys that the analysis relied on (empty list if none).
"""
class AnalysisAgentOutput(BaseModel):
    """Structured response produced by the analysis agent."""
    answer: str = Field(
        ...,
        description="Narrative summary that references datastore keys, row counts, and caveats.",
    )
    insights: List[str] = Field(
        default_factory=list,
        description="Bullet-style insights derived from the provided data.",
    )
    follow_up_questions: List[str] = Field(
        default_factory=list,
        description="Suggested next questions or actions for the supervisor.",
    )
    referenced_keys: List[str] = Field(
        default_factory=list,
        description="Datastore keys that were essential for the analysis.",
    )
class AnalysisState(TypedDict, total=False):
    """State passed through the analysis workflow."""
    question: str
    datastore: Dict[str, Dict[str, Any]]
    datastore_summary: str
    referenced_keys: List[str]
    answer: str
    insights: List[str]
    follow_up_questions: List[str]
    error_message: Optional[str]
def _format_datastore_summary(datastore: Dict[str, Dict[str, Any]]) -> str:
    """Create a compact textual summary of datastore entries for the LLM."""
    if not datastore:
        return "No datastore entries are available for analysis."
    summary_lines: List[str] = []
    for key, payload in datastore.items():
        description = str(payload.get("description", "")).strip()
        data = payload.get("data")
        row_count = 0
        sample_preview = ""
        if isinstance(data, list):
            row_count = len(data)
            if data:
                sample_row = data[0]
                if isinstance(sample_row, (list, tuple)):
                    preview_values = [str(value) for value in sample_row[:5]]
                    sample_preview = ", ".join(preview_values)
                else:
                    sample_preview = str(sample_row)
        line = f"- key={key}; rows={row_count}"
        if description:
            line += f"; description={description}"
        if sample_preview:
            line += f"; sample_row={sample_preview}"
        summary_lines.append(line)
    return "\n".join(summary_lines)
def create_analysis_agent(
    llm,
    *,
    datastore_loader: Optional[Callable[[], Dict[str, Dict[str, Any]]]] = None,
):
    """
    Build a LangGraph-powered analysis agent that mirrors the SQL agent pattern.
    Args:
        llm: Base chat model that supports `.with_structured_output`.
        datastore_loader: Optional callable returning a datastore snapshot when the
            runtime state does not provide one.
    """
    def load_datastore_node(state: AnalysisState) -> AnalysisState:
        """Collect datastore context from the incoming state or a loader."""
        print("--- ?? STEP: Gather datastore context ---")
        datastore = state.get("datastore")
        if not datastore and datastore_loader is not None:
            try:
                datastore = datastore_loader() or {}
            except Exception as exc:  # pragma: no cover - defensive
                state["error_message"] = f"Failed to load datastore: {exc}"
                state["datastore_summary"] = "Unable to load datastore snapshot."
                state["datastore"] = {}
                state["referenced_keys"] = []
                return state
        if not isinstance(datastore, dict):
            state["datastore"] = {}
            state["referenced_keys"] = []
            state["datastore_summary"] = "No datastore entries are available for analysis."
            return state
        state["datastore"] = datastore
        state["referenced_keys"] = list(datastore.keys())
        state["datastore_summary"] = _format_datastore_summary(datastore)
        return state
    def generate_analysis_node(state: AnalysisState) -> AnalysisState:
        """Invoke the LLM to produce structured analytical insights."""
        print("--- ?? STEP: Generate analysis insights ---")
        question = state.get("question", "").strip()
        datastore_summary = state.get(
            "datastore_summary",
            "No datastore entries are available for analysis.",
        )
        structured_llm = llm.with_structured_output(AnalysisAgentOutput)
        try:
            response = structured_llm.invoke(
                [
                    ("system", ANALYSIS_SYSTEM_PROMPT),
                    (
                        "human",
                        f"Analysis request: {question or 'Summarise the available datasets.'}\n\n"
                        f"Datastore inventory:\n{datastore_summary}",
                    ),
                ]
            )
        except Exception as exc:  # pragma: no cover - defensive
            print(f"--- ?? Failed to generate analysis: {exc} ---")
            state["error_message"] = f"Failed to generate analysis: {exc}"
            state["answer"] = (
                "Analysis agent could not generate insights because the LLM request failed."
            )
            state["insights"] = []
            state["follow_up_questions"] = []
            state.setdefault("referenced_keys", [])
            return state
        state["answer"] = response.answer.strip()
        state["insights"] = [item.strip() for item in response.insights if item.strip()]
        state["follow_up_questions"] = [
            item.strip() for item in response.follow_up_questions if item.strip()
        ]
        state["referenced_keys"] = response.referenced_keys or state.get("referenced_keys", [])
        return state
    workflow = StateGraph(AnalysisState)
    workflow.add_node("load_datastore", load_datastore_node)
    workflow.add_node("generate_analysis", generate_analysis_node)
    workflow.add_edge(START, "load_datastore")
    workflow.add_edge("load_datastore", "generate_analysis")
    workflow.add_edge("generate_analysis", END)
    return workflow.compile()
__all__ = ["create_analysis_agent", "AnalysisState", "AnalysisAgentOutput"]