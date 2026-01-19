"""LangGraph workflow for a code-generation visualization agent."""

from __future__ import annotations

import builtins
import re
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from langgraph.graph import END, START, StateGraph

from utils.datastore import DATASTORE, DataStore
from utils.output_basemodels import VisualizationCodeOutput
from utils.states import VisualizationState

PROMPT_PATH = Path("prompts/visualization_codegen_prompt.txt")
VISUALIZATION_CODE_PROMPT = PROMPT_PATH.read_text(encoding="utf-8")
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")
MAX_CODEGEN_RETRIES = 2
MAX_ERROR_CHARS = 2000


def _normalize_datastore_input(source: Any) -> Dict[str, Dict[str, Any]]:
    if isinstance(source, DataStore):
        return source.snapshot()
    if isinstance(source, dict):
        return dict(source)
    return {}


def _summarize_datastore(datastore: Dict[str, Dict[str, Any]]) -> str:
    if not datastore:
        return "No datasets available."
    summary_lines: List[str] = []
    for key, payload in datastore.items():
        description = str(payload.get("description", "")).strip()
        row_count = payload.get("row_count")
        summary = f"- key={key}"
        if isinstance(row_count, int):
            summary += f"; rows={row_count}"
        if description:
            summary += f"; description={description}"
        summary_lines.append(summary)
    return "\n".join(summary_lines)


def _slugify(text: str, fallback: str = "visualization") -> str:
    tokens = TOKEN_PATTERN.findall(text.lower())
    return "-".join(tokens) or fallback


def _save_figure(fig, output_path: Path) -> tuple[Path, List[str]]:
    warnings: List[str] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.write_image(str(output_path))
    except Exception as exc:  # pragma: no cover - fallback for missing kaleido
        fallback = output_path.with_suffix(".html")
        fig.write_html(str(fallback))
        warnings.append(f"Image export failed ({exc}); chart saved as HTML at {fallback}.")
        output_path = fallback
    return output_path, warnings


def create_visualization_codegen_agent(llm):
    """
    Build a visualization agent that generates and executes Python code.
    """

    def request_code(
        instruction: str,
        datastore_summary: str,
        *,
        error_context: Optional[str] = None,
        previous_code: Optional[str] = None,
    ) -> VisualizationCodeOutput:
        human_parts = [
            f"Instruction: {instruction or 'Create helpful charts.'}",
            f"Datastore inventory:\n{datastore_summary}",
        ]
        if error_context:
            human_parts.append(f"Previous error:\n{error_context}")
        if previous_code:
            human_parts.append(f"Previous code:\n```python\n{previous_code}\n```")
        response = llm.invoke(
            [
                ("system", VISUALIZATION_CODE_PROMPT),
                ("human", "\n\n".join(human_parts)),
            ]
        )
        return response if isinstance(response, VisualizationCodeOutput) else VisualizationCodeOutput(**response)

    def load_context_node(state: VisualizationState) -> VisualizationState:
        datastore_obj = state.get("datastore_obj")
        if not isinstance(datastore_obj, DataStore):
            datastore_obj = DATASTORE
        state["datastore_obj"] = datastore_obj
        datastore = _normalize_datastore_input(state.get("datastore"))
        if not datastore:
            datastore = datastore_obj.snapshot()
        state["datastore"] = datastore
        state["datastore_summary"] = _summarize_datastore(datastore)
        state.setdefault("warnings", [])
        return state

    def generate_code_node(state: VisualizationState) -> VisualizationState:
        instruction = str(state.get("instruction", "")).strip()
        datastore_summary = state.get("datastore_summary", "No datasets available.")
        warnings = list(state.get("warnings", []))
        try:
            code_output = request_code(instruction, datastore_summary)
        except Exception as exc:  # pragma: no cover - defensive
            state["error_message"] = f"Visualization code generation failed: {exc}"
            state["visualization_agent_final_answer"] = (
                "Visualization agent could not generate code."
            )
            state["warnings"] = warnings
            return state

        state["visualization_code"] = code_output.code
        state["visualization_code_summary"] = code_output.summary
        if code_output.notes:
            warnings.append(code_output.notes)
        state["warnings"] = warnings
        state["visualization_codegen_attempts"] = 0
        print("[Visualization Code Agent] code generated")
        return state

    def execute_code_node(state: VisualizationState) -> VisualizationState:
        warnings = list(state.get("warnings", []))
        datastore_obj = state.get("datastore_obj")
        if not isinstance(datastore_obj, DataStore):
            datastore_obj = DATASTORE

        datastore_snapshot = state.get("datastore") or {}
        instruction = str(state.get("instruction", "")).strip()
        datastore_summary = state.get("datastore_summary", "No datasets available.")
        reports_dir = Path("reports")
        attempts = int(state.get("visualization_codegen_attempts", 0))
        code = str(state.get("visualization_code", "")).strip()
        last_error: Optional[str] = None

        def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
            raise ImportError("Imports are disabled in visualization code.")

        def safe_open(file, mode="r", *args, **kwargs):
            if "r" in mode or "+" in mode:
                raise PermissionError("File reads are disabled in visualization code.")
            if isinstance(file, (str, Path)):
                path = Path(file).expanduser().resolve()
                allowed_root = reports_dir.resolve()
                if not str(path).startswith(str(allowed_root)):
                    raise PermissionError("File writes outside reports are disabled.")
                return builtins.open(path, mode, *args, **kwargs)
            raise PermissionError("File access is disabled in visualization code.")

        for attempt in range(attempts, attempts + MAX_CODEGEN_RETRIES + 1):
            last_error = None
            outputs: List[Dict[str, Any]] = []
            print(f"[Visualization Code Agent] execute attempt {attempt + 1}")

            def list_datasets() -> List[str]:
                return list(datastore_snapshot.keys())

            def get_df(key: str) -> pd.DataFrame:
                return datastore_obj.get_df(key)

            def register_output(
                path: str,
                summary: Optional[str] = None,
                meta: Optional[Dict[str, Any]] = None,
            ) -> None:
                outputs.append(
                    {
                        "chart_path": str(path),
                        "summary": summary or "",
                        "meta": meta or {},
                    }
                )

            def save_figure(
                fig,
                filename: Optional[str] = None,
                summary: Optional[str] = None,
                meta: Optional[Dict[str, Any]] = None,
            ) -> str:
                reports_dir.mkdir(parents=True, exist_ok=True)
                file_name = filename or f"visualization-{len(outputs) + 1}.png"
                path = Path(file_name)
                suffix = path.suffix or ".png"
                base = _slugify(path.stem or "visualization")
                output_path = reports_dir / f"{base}{suffix}"
                saved_path, save_warnings = _save_figure(fig, output_path)
                warnings.extend(save_warnings)
                register_output(str(saved_path), summary=summary, meta=meta)
                return str(saved_path)

            safe_builtins = {
                "__import__": safe_import,
                "Exception": Exception,
                "ValueError": ValueError,
                "KeyError": KeyError,
                "TypeError": TypeError,
                "abs": abs,
                "all": all,
                "any": any,
                "bool": bool,
                "dict": dict,
                "enumerate": enumerate,
                "float": float,
                "int": int,
                "len": len,
                "list": list,
                "map": map,
                "max": max,
                "min": min,
                "print": print,
                "range": range,
                "set": set,
                "sorted": sorted,
                "str": str,
                "sum": sum,
                "tuple": tuple,
                "zip": zip,
                "open": safe_open,
            }

            exec_globals: Dict[str, Any] = {
                "__builtins__": safe_builtins,
                "pd": pd,
                "px": px,
                "go": go,
                "np": np,
                "datastore_snapshot": datastore_snapshot,
                "instruction": instruction,
                "list_datasets": list_datasets,
                "get_df": get_df,
                "save_figure": save_figure,
                "register_output": register_output,
            }
            exec_locals: Dict[str, Any] = {}

            if not code:
                last_error = "Visualization code was empty."
            else:
                try:
                    exec(code, exec_globals, exec_locals)
                except Exception as exc:
                    tb = traceback.format_exc()
                    if len(tb) > MAX_ERROR_CHARS:
                        tb = tb[-MAX_ERROR_CHARS:]
                    last_error = f"{exc}\n{tb}"
                else:
                    result = exec_locals.get("result") or exec_globals.get("result")
                    summary = ""
                    output_paths: List[str] = []
                    if isinstance(result, dict):
                        summary = str(result.get("summary", "")).strip()
                        output_paths = list(result.get("output_paths") or [])

                    if outputs:
                        visualizations = [
                            {
                                "chart_path": item.get("chart_path"),
                                "summary": item.get("summary"),
                                "meta": item.get("meta", {}),
                            }
                            for item in outputs
                        ]
                        output_paths = [
                            item.get("chart_path")
                            for item in visualizations
                            if item.get("chart_path")
                        ]
                    elif output_paths:
                        visualizations = [
                            {"chart_path": path, "summary": summary} for path in output_paths
                        ]
                    else:
                        last_error = (
                            "Code executed but produced no outputs. "
                            "Ensure save_figure/register_output is called and result is set."
                        )

                    if not last_error:
                        final_summary = summary or state.get("visualization_code_summary") or (
                            f"Generated {len(output_paths)} chart(s)."
                        )
                        state["visualizations"] = visualizations
                        state["output_paths"] = output_paths
                        state["output_path"] = output_paths[0] if output_paths else ""
                        state["visualization_agent_final_answer"] = final_summary
                        state["warnings"] = warnings
                        state["visualization_codegen_attempts"] = attempt
                        print(
                            f"[Visualization Code Agent] success: outputs={len(output_paths)}"
                        )
                        return state

            if attempt >= attempts + MAX_CODEGEN_RETRIES:
                break

            warnings.append(f"Code execution failed on attempt {attempt + 1}; retrying.")
            print(
                f"[Visualization Code Agent] retrying after error on attempt {attempt + 1}"
            )
            error_context = last_error or "Unknown error."
            try:
                code_output = request_code(
                    instruction,
                    datastore_summary,
                    error_context=error_context,
                    previous_code=code,
                )
            except Exception as exc:  # pragma: no cover - defensive
                state["error_message"] = f"Visualization code regeneration failed: {exc}"
                state["visualization_agent_final_answer"] = (
                    "Visualization agent could not regenerate code."
                )
                state["warnings"] = warnings
                return state
            code = code_output.code
            state["visualization_code"] = code
            if code_output.notes:
                warnings.append(code_output.notes)

        state["error_message"] = f"Visualization code execution failed: {last_error}"
        state["visualization_agent_final_answer"] = (
            "Visualization agent could not execute the generated code."
        )
        state["warnings"] = warnings
        return state

    workflow = StateGraph(VisualizationState)
    workflow.add_node("load_context", load_context_node)
    workflow.add_node("generate_code", generate_code_node)
    workflow.add_node("execute_code", execute_code_node)

    workflow.add_edge(START, "load_context")
    workflow.add_edge("load_context", "generate_code")
    workflow.add_edge("generate_code", "execute_code")
    workflow.add_edge("execute_code", END)

    return workflow.compile()


__all__ = ["create_visualization_codegen_agent"]
