"""LangGraph workflow for the visualization agent."""

from __future__ import annotations

import builtins
import re
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from langgraph.graph import END, START, StateGraph

from utils.datastore import DATASTORE, DataStore
from utils.output_basemodels import VisualizationCodeOutput
from utils.states import VisualizationState

PROMPT_PATH = Path("prompts/visualization_agent_prompt.txt")
VISUALIZATION_CODE_PROMPT = PROMPT_PATH.read_text(encoding="utf-8")
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")
MAX_CODEGEN_RETRIES = 2
MAX_ERROR_CHARS = 2000
MAX_COLUMN_PREVIEW = 12
DEFAULT_DATASET_KEY = "data"
TIME_HINTS = ("time", "timestamp", "date", "datetime", "ts")
SERIES_HINTS = ("axis", "variable", "metric", "channel", "sensor", "component")
VALUE_HINTS = ("value", "reading", "measurement", "avg", "mean")


def _auto_description(df: pd.DataFrame) -> str:
    cols = ", ".join(map(str, df.columns.tolist()))
    desc = f"Rows: {len(df)} | Columns: {cols}"
    if "ts" in df.columns:
        try:
            ts = pd.to_datetime(df["ts"], utc=True, errors="coerce")
            if ts.notna().any():
                desc += f" | ts range: {ts.min()} -> {ts.max()}"
        except Exception:
            pass
    return desc


def _build_metadata_from_df(
    df: pd.DataFrame, *, description: str = "", key: str = ""
) -> Dict[str, Any]:
    return {
        "description": description or _auto_description(df),
        "row_count": len(df),
        "datastore_ref": key,
        "columns": [str(col) for col in df.columns],
        "dtypes": {str(col): str(dtype) for col, dtype in df.dtypes.items()},
    }


def _normalize_datastore_input(
    source: Any,
) -> tuple[Dict[str, Dict[str, Any]], Dict[str, pd.DataFrame]]:
    if isinstance(source, DataStore):
        return source.snapshot(), {}
    if isinstance(source, pd.DataFrame):
        key = DEFAULT_DATASET_KEY
        return {key: _build_metadata_from_df(source, key=key)}, {key: source}
    if isinstance(source, (list, tuple)) and source and all(
        isinstance(item, pd.DataFrame) for item in source
    ):
        metadata: Dict[str, Dict[str, Any]] = {}
        frames: Dict[str, pd.DataFrame] = {}
        for idx, df in enumerate(source, start=1):
            key = f"{DEFAULT_DATASET_KEY}_{idx}"
            metadata[key] = _build_metadata_from_df(df, key=key)
            frames[key] = df
        return metadata, frames
    if isinstance(source, dict):
        metadata = {}
        frames = {}
        for key, value in source.items():
            if isinstance(value, pd.DataFrame):
                frames[key] = value
                metadata[key] = _build_metadata_from_df(value, key=key)
                continue
            if isinstance(value, dict) and isinstance(value.get("df"), pd.DataFrame):
                df = value["df"]
                frames[key] = df
                description = str(value.get("description", "")).strip()
                metadata[key] = _build_metadata_from_df(df, description=description, key=key)
                if "row_count" in value:
                    metadata[key]["row_count"] = value["row_count"]
                continue
            if isinstance(value, dict):
                metadata[key] = dict(value)
            else:
                metadata[key] = {"description": str(value)}
        return metadata, frames
    return {}, {}


def _format_columns(columns: List[Any], limit: int = MAX_COLUMN_PREVIEW) -> str:
    if not columns:
        return ""
    col_names = [str(col) for col in columns if col is not None]
    if not col_names:
        return ""
    if len(col_names) <= limit:
        return ", ".join(col_names)
    extra = len(col_names) - limit
    return ", ".join(col_names[:limit]) + f", ... (+{extra} more)"


def _format_column_types(
    columns: List[Any],
    dtypes: Dict[str, Any],
    limit: int = MAX_COLUMN_PREVIEW,
) -> str:
    if not columns or not dtypes:
        return ""
    pairs: List[str] = []
    for col in columns[:limit]:
        col_name = str(col)
        dtype = dtypes.get(col_name)
        if dtype:
            pairs.append(f"{col_name}:{dtype}")
        else:
            pairs.append(col_name)
    extra = len(columns) - limit
    suffix = f", ... (+{extra} more)" if extra > 0 else ""
    return ", ".join(pairs) + suffix


def _is_time_dtype(dtype: Any) -> bool:
    text = str(dtype).lower()
    return "datetime" in text or text.startswith("date")


def _is_numeric_dtype(dtype: Any) -> bool:
    text = str(dtype).lower()
    return any(token in text for token in ("int", "float", "double", "decimal", "number"))


def _infer_column_roles(
    columns: List[str],
    dtypes: Dict[str, Any],
) -> Dict[str, List[str]]:
    time_columns: List[str] = []
    numeric_columns: List[str] = []
    categorical_columns: List[str] = []
    series_columns: List[str] = []
    value_columns: List[str] = []
    for col in columns:
        name = str(col)
        name_lower = name.lower()
        dtype = dtypes.get(name, "")
        if any(hint in name_lower for hint in TIME_HINTS) or _is_time_dtype(dtype):
            time_columns.append(name)
            continue
        if _is_numeric_dtype(dtype):
            numeric_columns.append(name)
        else:
            categorical_columns.append(name)
        if any(hint in name_lower for hint in SERIES_HINTS):
            series_columns.append(name)
        if any(hint in name_lower for hint in VALUE_HINTS):
            value_columns.append(name)
    if not value_columns and len(numeric_columns) == 1:
        value_columns = [numeric_columns[0]]
    return {
        "time_columns": time_columns,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "series_columns": series_columns,
        "value_columns": value_columns,
    }


def _build_input_profile(
    datastore: Dict[str, Dict[str, Any]],
    frames: Optional[Dict[str, pd.DataFrame]] = None,
) -> List[Dict[str, Any]]:
    profiles: List[Dict[str, Any]] = []
    for key, payload in datastore.items():
        columns = payload.get("columns") or []
        dtypes = payload.get("dtypes") or {}
        if not columns and frames and key in frames:
            columns = list(frames[key].columns)
        if not dtypes and frames and key in frames:
            dtypes = {str(col): str(dtype) for col, dtype in frames[key].dtypes.items()}
        columns = [str(col) for col in columns]
        dtypes = {str(col): str(dtype) for col, dtype in dict(dtypes).items()}
        roles = _infer_column_roles(columns, dtypes) if columns else {}
        profiles.append(
            {
                "key": key,
                "row_count": payload.get("row_count"),
                "columns": columns,
                "dtypes": dtypes,
                "time_columns": roles.get("time_columns", []),
                "numeric_columns": roles.get("numeric_columns", []),
                "categorical_columns": roles.get("categorical_columns", []),
                "series_columns": roles.get("series_columns", []),
                "value_columns": roles.get("value_columns", []),
                "long_form": bool(roles.get("series_columns")) and bool(roles.get("value_columns")),
            }
        )
    return profiles


def _summarize_input_profile(profiles: List[Dict[str, Any]]) -> str:
    if not profiles:
        return "No input profile available."
    lines: List[str] = []
    for profile in profiles:
        key = profile.get("key", "")
        time_cols = _format_columns(profile.get("time_columns", []))
        numeric_cols = _format_columns(profile.get("numeric_columns", []))
        categorical_cols = _format_columns(profile.get("categorical_columns", []))
        series_cols = _format_columns(profile.get("series_columns", []))
        value_cols = _format_columns(profile.get("value_columns", []))
        layout = "long" if profile.get("long_form") else "wide"
        parts = [f"- key={key}", f"layout={layout}"]
        if time_cols:
            parts.append(f"time={time_cols}")
        if numeric_cols:
            parts.append(f"numeric={numeric_cols}")
        if categorical_cols:
            parts.append(f"categorical={categorical_cols}")
        if series_cols:
            parts.append(f"series={series_cols}")
        if value_cols:
            parts.append(f"values={value_cols}")
        lines.append("; ".join(parts))
    return "\n".join(lines)


def _summarize_datastore(
    datastore: Dict[str, Dict[str, Any]],
    frames: Optional[Dict[str, pd.DataFrame]] = None,
) -> str:
    if not datastore:
        return "No datasets available."
    summary_lines: List[str] = []
    for key, payload in datastore.items():
        description = str(payload.get("description", "")).strip()
        row_count = payload.get("row_count")
        columns = payload.get("columns")
        dtypes = payload.get("dtypes")
        if not columns and frames and key in frames:
            columns = list(frames[key].columns)
        if (not dtypes) and frames and key in frames:
            dtypes = {str(col): str(dtype) for col, dtype in frames[key].dtypes.items()}
        if columns and not isinstance(columns, (list, tuple)):
            columns = [columns]
        summary = f"- key={key}"
        if isinstance(row_count, int):
            summary += f"; rows={row_count}"
        if columns:
            summary += f"; columns={_format_columns(list(columns))}"
        if columns and isinstance(dtypes, dict):
            summary += f"; types={_format_column_types(list(columns), dtypes)}"
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
    if fig is None:
        fig = plt.gcf()
    elif fig is plt:
        fig = plt.gcf()
    elif hasattr(fig, "figure") and not hasattr(fig, "savefig"):
        fig = fig.figure
    if not hasattr(fig, "savefig"):
        raise ValueError("Unsupported figure type for saving.")
    try:
        fig.tight_layout()
    except Exception:
        pass
    fig.savefig(str(output_path), dpi=160, bbox_inches="tight")
    try:
        plt.close(fig)
    except Exception:
        pass
    return output_path, warnings


def _dedupe_preserve(items: List[str]) -> List[str]:
    seen = set()
    deduped: List[str] = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _short_error(message: Optional[str], limit: int = 240) -> str:
    if not message:
        return ""
    text = str(message)
    for line in text.splitlines():
        if line.strip():
            text = line.strip()
            break
    if len(text) > limit:
        text = text[: limit - 3] + "..."
    return text


def create_visualization_agent(llm):
    """
    Build a visualization agent that generates and executes Python code.
    """

    def request_code(
        instruction: str,
        datastore_summary: str,
        *,
        input_profile_summary: Optional[str] = None,
        error_context: Optional[str] = None,
        previous_code: Optional[str] = None,
    ) -> VisualizationCodeOutput:
        human_parts = [
            f"Instruction: {instruction or 'Create helpful charts.'}",
            f"Datastore inventory:\n{datastore_summary}",
        ]
        if input_profile_summary:
            human_parts.append(f"Input profile:\n{input_profile_summary}")
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
        if response is None:
            raise ValueError("Visualization LLM returned None")
        return response if isinstance(response, VisualizationCodeOutput) else VisualizationCodeOutput(**response)

    def load_context_node(state: VisualizationState) -> VisualizationState:
        datastore_source = state.get("datastore")
        datastore_obj = state.get("datastore_obj")
        if isinstance(datastore_source, DataStore):
            datastore_obj = datastore_source
        elif not isinstance(datastore_obj, DataStore):
            datastore_obj = DATASTORE
        datastore, datastore_frames = _normalize_datastore_input(datastore_source)
        if not datastore:
            datastore = datastore_obj.snapshot()
        state["datastore_obj"] = datastore_obj
        state["datastore"] = datastore
        state["datastore_frames"] = datastore_frames
        state["datastore_summary"] = _summarize_datastore(datastore, datastore_frames)
        input_profile = _build_input_profile(datastore, datastore_frames)
        state["input_profile"] = input_profile
        state["input_profile_summary"] = _summarize_input_profile(input_profile)
        state.setdefault("warnings", [])
        return state

    def generate_code_node(state: VisualizationState) -> VisualizationState:
        instruction = str(state.get("instruction", "")).strip()
        datastore_summary = state.get("datastore_summary", "No datasets available.")
        input_profile_summary = state.get("input_profile_summary", "")
        error_context = state.get("visualization_error_context") or state.get("error_message")
        warnings = list(state.get("warnings", []))
        try:
            code_output = request_code(
                instruction,
                datastore_summary,
                input_profile_summary=input_profile_summary,
                error_context=error_context,
            )
        except Exception as exc:  # pragma: no cover - defensive
            print(
                "[Visualization Agent] error: code generation failed "
                f"({type(exc).__name__}: {exc})"
            )
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
        print("[Visualization Agent] code generated")
        return state

    def execute_code_node(state: VisualizationState) -> VisualizationState:
        warnings = list(state.get("warnings", []))
        datastore_obj = state.get("datastore_obj")
        if not isinstance(datastore_obj, DataStore):
            datastore_obj = DATASTORE

        datastore_snapshot = state.get("datastore") or {}
        datastore_frames = state.get("datastore_frames") or {}
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

        previous_error = str(state.get("visualization_error_context") or "")
        for attempt in range(attempts, attempts + MAX_CODEGEN_RETRIES + 1):
            last_error = None
            outputs: List[Dict[str, Any]] = []
            print(f"[Visualization Agent] execute attempt {attempt + 1}")

            def list_datasets() -> List[str]:
                keys = list(datastore_snapshot.keys())
                if not keys and datastore_frames:
                    keys = list(datastore_frames.keys())
                return keys

            def get_df(key: str) -> pd.DataFrame:
                if key in datastore_frames:
                    return datastore_frames[key].copy()
                return datastore_obj.get_df(key)

            def get_all_dfs() -> Dict[str, pd.DataFrame]:
                return {key: get_df(key) for key in list_datasets()}

            def dataset_meta(key: str) -> Dict[str, Any]:
                payload = datastore_snapshot.get(key, {})
                return dict(payload) if isinstance(payload, dict) else {}

            def warn(message: str) -> None:
                if message:
                    warnings.append(str(message))

            def inspect_dataset(key: str) -> Dict[str, Any]:
                df = get_df(key)
                columns = [str(col) for col in df.columns]
                dtypes = {str(col): str(dtype) for col, dtype in df.dtypes.items()}
                time_columns: List[str] = []
                numeric_columns: List[str] = []
                categorical_columns: List[str] = []
                for col in df.columns:
                    name = str(col).lower()
                    series = df[col]
                    if pd.api.types.is_datetime64_any_dtype(series) or any(
                        hint in name for hint in TIME_HINTS
                    ):
                        time_columns.append(str(col))
                        continue
                    if pd.api.types.is_numeric_dtype(series):
                        numeric_columns.append(str(col))
                    else:
                        categorical_columns.append(str(col))
                return {
                    "key": key,
                    "row_count": len(df),
                    "columns": columns,
                    "dtypes": dtypes,
                    "time_columns": time_columns,
                    "numeric_columns": numeric_columns,
                    "categorical_columns": categorical_columns,
                }

            def inspect_inputs(max_datasets: Optional[int] = None) -> List[Dict[str, Any]]:
                keys = list_datasets()
                if isinstance(max_datasets, int) and max_datasets > 0:
                    keys = keys[:max_datasets]
                return [inspect_dataset(key) for key in keys]

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

            def make_subplots(
                rows: int,
                cols: int,
                *,
                shared_xaxes: bool = False,
                subplot_titles: Optional[List[str]] = None,
                figsize: Optional[tuple[float, float]] = None,
            ):
                fig, axes = plt.subplots(
                    rows,
                    cols,
                    sharex=shared_xaxes,
                    figsize=figsize,
                )
                if subplot_titles:
                    flat_axes = np.atleast_1d(axes).ravel()
                    for ax, title in zip(flat_axes, subplot_titles):
                        ax.set_title(title)
                return fig, axes

            dataset_keys = list_datasets()
            default_key = dataset_keys[0] if len(dataset_keys) == 1 else ""
            default_df = None
            if default_key:
                try:
                    default_df = get_df(default_key)
                except Exception as exc:
                    warnings.append(
                        f"Failed to load default dataset '{default_key}': {exc}"
                    )
                    default_key = ""

            def _coerce_str_list(value: Any) -> List[str]:
                if not value:
                    return []
                if isinstance(value, str):
                    return [value]
                if isinstance(value, (list, tuple, set)):
                    return [str(item) for item in value if item]
                return [str(value)]

            def _coerce_paths(result_payload: Dict[str, Any]) -> List[str]:
                if not isinstance(result_payload, dict):
                    return []
                paths: List[str] = []
                for key in ("output_paths", "chart_paths", "paths"):
                    value = result_payload.get(key)
                    if isinstance(value, (list, tuple, set)):
                        paths.extend(str(item) for item in value if item)
                    elif isinstance(value, str):
                        paths.append(value)
                for key in ("output_path", "chart_path", "path"):
                    value = result_payload.get(key)
                    if value:
                        paths.append(str(value))
                return _dedupe_preserve(paths)

            def _normalize_visualizations(items: Any) -> List[Dict[str, Any]]:
                if not items:
                    return []
                if isinstance(items, dict):
                    items = [items]
                if isinstance(items, str):
                    items = [items]
                if not isinstance(items, (list, tuple)):
                    return []
                normalized: List[Dict[str, Any]] = []
                for item in items:
                    if isinstance(item, dict):
                        path = item.get("chart_path") or item.get("path") or item.get("output_path")
                        if not path:
                            continue
                        normalized.append(
                            {
                                "chart_path": str(path),
                                "summary": str(item.get("summary", "")).strip(),
                                "meta": item.get("meta") or {},
                            }
                        )
                    elif isinstance(item, str):
                        normalized.append({"chart_path": item, "summary": "", "meta": {}})
                return normalized

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
                "isinstance": isinstance,
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
                "plt": plt,
                "matplotlib": matplotlib,
                "np": np,
                "make_subplots": make_subplots,
                "datastore_snapshot": datastore_snapshot,
                "instruction": instruction,
                "input_profile": state.get("input_profile", []),
                "input_profile_summary": state.get("input_profile_summary", ""),
                "previous_error": previous_error,
                "default_key": default_key,
                "default_df": default_df,
                "df": default_df if default_df is not None else None,
                "list_datasets": list_datasets,
                "get_df": get_df,
                "get_all_dfs": get_all_dfs,
                "dataset_meta": dataset_meta,
                "inspect_dataset": inspect_dataset,
                "inspect_inputs": inspect_inputs,
                "save_figure": save_figure,
                "register_output": register_output,
                "warn": warn,
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
                    print(
                        "[Visualization Agent] error: exec failed "
                        f"(attempt {attempt + 1}; {type(exc).__name__}: {exc})"
                    )
                else:
                    result = exec_locals.get("result") or exec_globals.get("result")
                    summary = ""
                    result_status = ""
                    result_warnings: List[str] = []
                    result_visualizations: List[Dict[str, Any]] = []
                    result_paths: List[str] = []

                    if isinstance(result, dict):
                        summary = str(result.get("summary", "")).strip()
                        result_status = str(result.get("status", "")).strip().lower()
                        result_warnings = _coerce_str_list(result.get("warnings"))
                        result_visualizations = _normalize_visualizations(
                            result.get("visualizations") or result.get("outputs")
                        )
                        result_paths = _coerce_paths(result)
                    elif isinstance(result, str):
                        summary = result.strip()

                    visualizations = _normalize_visualizations(outputs) + result_visualizations
                    if summary:
                        for item in visualizations:
                            if not item.get("summary"):
                                item["summary"] = summary

                    paths_from_outputs = [
                        item.get("chart_path")
                        for item in visualizations
                        if item.get("chart_path")
                    ]
                    output_paths = _dedupe_preserve(paths_from_outputs + result_paths)
                    if output_paths and not visualizations:
                        visualizations = [
                            {"chart_path": path, "summary": summary, "meta": {}}
                            for path in output_paths
                        ]

                    no_output_ok = result_status in {"no_data", "no_chart", "no_charts", "empty"} or not datastore_snapshot
                    if not output_paths:
                        if summary and no_output_ok:
                            final_summary = (
                                summary
                                or state.get("visualization_code_summary")
                                or "No charts generated."
                            )
                            warnings.extend(result_warnings)
                            warnings = _dedupe_preserve(warnings)
                            state["visualizations"] = []
                            state["output_paths"] = []
                            state["output_path"] = ""
                            state["visualization_agent_final_answer"] = final_summary
                            state["warnings"] = warnings
                            state["visualization_codegen_attempts"] = attempt
                            print("[Visualization Agent] no-output success")
                            return state
                        last_error = (
                            "Code executed but produced no outputs. "
                            "Ensure save_figure/register_output is called or set "
                            "result with status='no_data'."
                        )
                        print(
                            "[Visualization Agent] error: no outputs produced "
                            f"(attempt {attempt + 1}; { _short_error(last_error) })"
                        )
                    else:
                        final_summary = summary or state.get("visualization_code_summary") or (
                            f"Generated {len(output_paths)} chart(s)."
                        )
                        warnings.extend(result_warnings)
                        warnings = _dedupe_preserve(warnings)
                        state["visualizations"] = visualizations
                        state["output_paths"] = output_paths
                        state["output_path"] = output_paths[0] if output_paths else ""
                        state["visualization_agent_final_answer"] = final_summary
                        state["warnings"] = warnings
                        state["visualization_codegen_attempts"] = attempt
                        print(
                            f"[Visualization Agent] success: outputs={len(output_paths)}"
                        )
                        return state

            if attempt >= attempts + MAX_CODEGEN_RETRIES:
                break

            warnings.append(f"Code execution failed on attempt {attempt + 1}; retrying.")
            print(
                f"[Visualization Agent] retrying after error on attempt {attempt + 1}"
            )
            error_context = last_error or "Unknown error."
            state["visualization_error_context"] = error_context
            previous_error = error_context
            try:
                code_output = request_code(
                    instruction,
                    datastore_summary,
                    input_profile_summary=state.get("input_profile_summary", ""),
                    error_context=error_context,
                    previous_code=code,
                )
            except Exception as exc:  # pragma: no cover - defensive
                print(
                    "[Visualization Agent] error: code regeneration failed "
                    f"({type(exc).__name__}: {exc})"
                )
                state["error_message"] = f"Visualization code regeneration failed: {exc}"
                state["visualization_agent_final_answer"] = (
                    "Visualization agent could not regenerate code."
                )
                state["warnings"] = warnings
                return state
            code = code_output.code
            state["visualization_code"] = code
            state["visualization_code_summary"] = code_output.summary
            if code_output.notes:
                warnings.append(code_output.notes)

        print(
            "[Visualization Agent] error: execution failed "
            f"({ _short_error(last_error) })"
        )
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


__all__ = ["create_visualization_agent"]
