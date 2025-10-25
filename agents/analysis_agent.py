"""LangGraph-driven analysis agent utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from langgraph.graph import END, START, StateGraph

from utils.datastore import load_df, DataStore
from utils.output_basemodels import AnalysisAgentOutput
from utils.states import AnalysisDatastore, AnalysisState


DEFAULT_DATASTORE_MESSAGE = "No datastore entries are available for analysis."
DEFAULT_ANALYSIS_REQUEST = "Summarise the available datasets."
PROMPT_PATH = Path("prompts/analysis_agent_prompt.txt")
ANALYSIS_SYSTEM_PROMPT = PROMPT_PATH.read_text(encoding="utf-8")

Datastore = AnalysisDatastore


def _build_sample_preview(values: List[Any]) -> str:
    """Render up to five values from a row preview."""
    return ", ".join(str(value) for value in values[:5])


def _format_row_preview(row: Dict[str, Any]) -> str:
    preview_items = []
    for key, value in list(row.items())[:5]:
        preview_items.append(f"{key}={value}")
    return ", ".join(preview_items)


def _normalize_datastore_input(source: Any) -> AnalysisDatastore:
    if isinstance(source, DataStore):
        return source.snapshot()
    if isinstance(source, dict):
        return dict(source)
    return {}


def _resolve_datastore_entry(payload: Dict[str, Any]) -> Tuple[int, str]:
    """Return row count and a short preview for a datastore payload."""
    datastore_ref = payload.get("datastore_ref")
    missing_preview = ""
    if datastore_ref:
        try:
            df = load_df(str(datastore_ref))
            row_count = len(df)
            if row_count == 0:
                return 0, ""
            head = df.head(500)
            previews = []
            for _, row in head.iterrows():
                previews.append(_format_row_preview(row.to_dict()))
            sample_preview = " | ".join(previews)
            return row_count, sample_preview
        except Exception:  # pragma: no cover - defensive
            # Fall back to any serialized payload the state might still have.
            missing_preview = f"datastore_ref={datastore_ref} unavailable"

    data = payload.get("data")
    if isinstance(data, list):
        row_count = len(data)
        if not data:
            return row_count, ""
        previews: List[str] = []
        for raw_row in data[:500]:
            if isinstance(raw_row, dict):
                previews.append(_format_row_preview(raw_row))
            elif isinstance(raw_row, (list, tuple)):
                previews.append(_build_sample_preview(list(raw_row)))
            else:
                previews.append(str(raw_row))
        sample_preview = " | ".join(previews)
        return row_count, sample_preview

    return 0, missing_preview


def _format_datastore_summary(datastore: Datastore) -> str:
    """Create a compact textual summary of datastore entries for the LLM."""
    if not datastore:
        return DEFAULT_DATASTORE_MESSAGE
    summary_lines: List[str] = []
    for key, payload in datastore.items():
        description = str(payload.get("description", "")).strip()
        row_count, sample_preview = _resolve_datastore_entry(payload)
        line = f"- key={key}; rows={row_count}"
        if description:
            line += f"; description={description}"
        if sample_preview:
            line += f"; sample_row={sample_preview}"
        summary_lines.append(line)
    return "\n".join(summary_lines)


def _collect_datastore_context(
    state: AnalysisState,
    datastore_loader: Optional[Callable[[], Datastore]],
) -> tuple[Datastore, str, List[str], Optional[str]]:
    """Resolve datastore, summary, and referenced keys from the current state."""
    raw_datastore = state.get("datastore")
    datastore_obj = state.get("datastore_obj")

    datastore = _normalize_datastore_input(raw_datastore)
    if not datastore and isinstance(datastore_obj, DataStore):
        datastore = datastore_obj.snapshot()

    if not datastore and datastore_loader is not None:
        try:
            loaded = datastore_loader() or {}
            datastore = _normalize_datastore_input(loaded)
        except Exception as exc:  # pragma: no cover - defensive
            return {}, "Unable to load datastore snapshot.", [], f"Failed to load datastore: {exc}"

    if not datastore:
        return {}, DEFAULT_DATASTORE_MESSAGE, [], None

    return datastore, _format_datastore_summary(datastore), list(datastore.keys()), None


def _invoke_analysis_llm(
    llm,
    instruction: str,
    datastore_summary: str,
) -> AnalysisAgentOutput:
    """Call the LLM to obtain structured analysis output."""
    prompt_instruction = instruction or DEFAULT_ANALYSIS_REQUEST

    return llm.invoke(
        [
            ("system", ANALYSIS_SYSTEM_PROMPT),
            (
                "human",
                f"Analysis request: {prompt_instruction}\n\n"
                f"Datastore inventory:\n{datastore_summary}",
            ),
        ]
    )


def create_analysis_agent(
    llm,
    *,
    datastore_loader: Optional[Callable[[], Datastore]] = None,
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
        datastore, summary, referenced_keys, error = _collect_datastore_context(
            state, datastore_loader
        )
        if error:
            state["error_message"] = error
        state["datastore"] = datastore
        state["datastore_summary"] = summary
        state["referenced_keys"] = referenced_keys
        return state

    def generate_analysis_node(state: AnalysisState) -> AnalysisState:
        """Invoke the LLM to produce structured analytical insights."""
        print("--- ?? STEP: Generate analysis insights ---")
        instruction = state.get("instruction", "").strip()
        datastore_summary = state.get("datastore_summary", DEFAULT_DATASTORE_MESSAGE)
        try:
            response = _invoke_analysis_llm(llm, instruction, datastore_summary)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"--- ?? Failed to generate analysis: {exc} ---")
            state["error_message"] = f"Failed to generate analysis: {exc}"
            state["analysis_agent_final_answer"] = (
                "Analysis agent could not generate insights because the LLM request failed."
            )
            state["insights"] = []
            state["follow_up_questions"] = []
            state.setdefault("referenced_keys", [])
            return state
        state["analysis_agent_final_answer"] = response.analysis_agent_final_answer.strip()
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
