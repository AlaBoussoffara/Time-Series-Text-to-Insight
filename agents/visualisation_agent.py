"""LangGraph workflow for the visualization agent."""

from __future__ import annotations

import re
from datetime import timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import plotly.express as px
from langgraph.graph import END, START, StateGraph

from utils.datastore import DATASTORE, DataStore
from utils.output_basemodels import VisualizationPlanOutput
from utils.states import VisualizationState

PROMPT_PATH = Path("prompts/visualization_agent_prompt.txt")
VISUALIZATION_SYSTEM_PROMPT = PROMPT_PATH.read_text(encoding="utf-8")

TIME_COLUMN_CANDIDATES = [
    "timestamp",
    "ts",
    "time",
    "datetime",
    "date",
    "event_time",
    "recorded_at",
    "created_at",
]
SERIES_COLUMN_HINTS = ["gateway", "device", "metric", "alias", "name", "entity", "station"]
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


def _normalize_datastore(source: Any) -> Dict[str, Dict[str, Any]]:
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
            summary += f"; rows≈{row_count}"
        if description:
            summary += f"; description={description}"
        summary_lines.append(summary)
    return "\n".join(summary_lines)


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def _score_dataset(key: str, description: str, keywords: Sequence[str], tokens: Sequence[str]) -> float:
    haystack = f"{key} {description}".lower()
    score = 0.0
    for keyword in keywords:
        if not keyword:
            continue
        lowered = keyword.lower()
        if lowered in haystack:
            score += 3.0
    for token in tokens:
        if token and token in haystack:
            score += 0.5
    return score


def _rank_datasets(
    datastore: Dict[str, Dict[str, Any]],
    plan: VisualizationPlanOutput,
    instruction: str,
) -> List[tuple[str, str, float]]:
    if not datastore:
        return []
    combined_keywords = list(plan.dataset_hints) + list(plan.metric_keywords) + list(plan.entity_keywords)
    instruction_tokens = _tokenize(instruction + " " + plan.priority_notes)
    ranked: List[tuple[str, str, float]] = []
    for key, payload in datastore.items():
        description = str(payload.get("description", "")).strip()
        score = _score_dataset(key, description, combined_keywords, instruction_tokens)
        row_count = payload.get("row_count")
        if isinstance(row_count, int):
            score += min(row_count, 2000) / 2000.0
        ranked.append((key, description, score))
    ranked.sort(key=lambda item: item[2], reverse=True)
    return ranked


def _infer_time_column(df: pd.DataFrame) -> Optional[str]:
    for candidate in TIME_COLUMN_CANDIDATES:
        if candidate in df.columns:
            return candidate
    for column in df.columns:
        series = df[column]
        if pd.api.types.is_datetime64_any_dtype(series):
            return column
        if series.dtype == object:
            try:
                parsed = pd.to_datetime(series, errors="coerce")
            except Exception:
                continue
            if parsed.notna().mean() > 0.8:
                df[column] = parsed
                return column
    return None


def _infer_value_column(df: pd.DataFrame, time_column: Optional[str], metric_keywords: Sequence[str]) -> Optional[str]:
    metric_keywords = [kw.lower() for kw in metric_keywords if kw]
    numeric_columns = [
        col for col in df.columns if col != time_column and pd.api.types.is_numeric_dtype(df[col])
    ]
    if not numeric_columns:
        return None
    if metric_keywords:
        for keyword in metric_keywords:
            for col in numeric_columns:
                if keyword in col.lower():
                    return col
    for col in numeric_columns:
        if any(hint in col.lower() for hint in ["value", "reading", "metric", "measurement"]):
            return col
    return numeric_columns[0]


def _infer_series_column(df: pd.DataFrame, entity_keywords: Sequence[str]) -> Optional[str]:
    entity_keywords = [kw.lower() for kw in entity_keywords if kw]
    candidate_columns: List[str] = []
    for column in df.columns:
        series = df[column]
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(series):
            candidate_columns.append(column)
    for column in candidate_columns:
        name = column.lower()
        if any(hint in name for hint in SERIES_COLUMN_HINTS):
            return column
    for keyword in entity_keywords:
        for column in candidate_columns:
            if keyword in column.lower():
                return column
    return candidate_columns[0] if candidate_columns else None


def _ensure_time_column(df: pd.DataFrame) -> tuple[str, bool, List[str]]:
    """Return a column suitable for the x-axis; fallback to row index."""
    warnings: List[str] = []
    time_column = _infer_time_column(df)
    if time_column:
        return time_column, False, warnings
    synthetic_column = "__row_index"
    df[synthetic_column] = range(len(df))
    warnings.append("No timestamp column detected; using row order for the x-axis.")
    return synthetic_column, True, warnings


def _ensure_value_column(
    df: pd.DataFrame,
    time_column: str,
    metric_keywords: Sequence[str],
) -> tuple[str, List[str]]:
    warnings: List[str] = []
    column = _infer_value_column(df, time_column, metric_keywords)
    if column:
        return column, warnings
    for col in df.columns:
        if col == time_column:
            continue
        try:
            converted = pd.to_numeric(df[col], errors="coerce")
        except Exception:
            continue
        if converted.notna().mean() > 0.6:
            new_col = f"__numeric_{col}"
            df[new_col] = converted.fillna(method="ffill").fillna(method="bfill").fillna(0)
            warnings.append(f"Converted column '{col}' to numeric for plotting.")
            return new_col, warnings
    for col in df.columns:
        if col == time_column:
            continue
        series = df[col]
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(series):
            codes, _ = pd.factorize(series, sort=False)
            new_col = f"__encoded_{col}"
            df[new_col] = codes
            warnings.append(f"Encoded column '{col}' as numeric values for plotting.")
            return new_col, warnings
    fallback_col = "__row_value"
    df[fallback_col] = range(len(df))
    warnings.append("No numeric column detected; using row order as the metric.")
    return fallback_col, warnings


def _ensure_series_column(
    df: pd.DataFrame,
    entity_keywords: Sequence[str],
) -> tuple[Optional[str], List[str]]:
    warnings: List[str] = []
    column = _infer_series_column(df, entity_keywords)
    if column:
        return column, warnings
    if entity_keywords:
        obj_columns = [
            col
            for col in df.columns
            if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col])
        ]
        for keyword in entity_keywords:
            lowered = keyword.lower()
            for col in obj_columns:
                if lowered in col.lower():
                    warnings.append(
                        f"No direct entity column detected; using '{col}' as a proxy."
                    )
                    return col, warnings
    return None, warnings


def _apply_time_window_filter(
    df: pd.DataFrame,
    time_column: str,
    window_text: str,
    *,
    allow_filter: bool,
) -> tuple[pd.DataFrame, List[str], Optional[str]]:
    warnings: List[str] = []
    applied_window: Optional[str] = None
    if not window_text or not allow_filter:
        if window_text and not allow_filter:
            warnings.append("Time filter skipped because no timestamp column was detected.")
        return df, warnings, applied_window
    series = pd.to_datetime(df[time_column], errors="coerce")
    non_null_mask = series.notna()
    if not non_null_mask.any():
        warnings.append(f"Time filter skipped: column '{time_column}' is not parseable as datetime.")
        return df, warnings, applied_window
    df = df.loc[non_null_mask].copy()
    df[time_column] = series.loc[non_null_mask]
    lowered = window_text.lower().strip()
    now = pd.Timestamp.utcnow()
    start: Optional[pd.Timestamp] = None
    end: Optional[pd.Timestamp] = None
    match = re.search(r"last\s+(\d+)\s+(day|days|hour|hours|week|weeks)", lowered)
    if match:
        qty = int(match.group(1))
        unit = match.group(2)
        if "week" in unit:
            delta = timedelta(days=7 * qty)
        elif "day" in unit:
            delta = timedelta(days=qty)
        else:
            delta = timedelta(hours=qty)
        start = now - delta
        applied_window = f"last {qty} {unit}"
    elif lowered in {"today"}:
        start = pd.Timestamp(now.date(), tz=timezone.utc)
        end = start + timedelta(days=1)
        applied_window = "today"
    elif lowered in {"yesterday"}:
        end = pd.Timestamp(now.date(), tz=timezone.utc)
        start = end - timedelta(days=1)
        applied_window = "yesterday"
    elif lowered in {"last week"}:
        start = now - timedelta(days=7)
        applied_window = "last 7 days"
    elif lowered in {"last month"}:
        start = now - timedelta(days=30)
        applied_window = "last 30 days"
    elif lowered in {"last 24 hours"}:
        start = now - timedelta(hours=24)
        applied_window = "last 24 hours"
    else:
        warnings.append(f"Time window '{window_text}' not recognized; using full dataset.")
        return df, warnings, applied_window
    mask = pd.Series(True, index=df.index)
    if start is not None:
        mask &= df[time_column] >= start
    if end is not None:
        mask &= df[time_column] < end
    filtered = df.loc[mask]
    if filtered.empty:
        warnings.append("Time window filter removed all rows; using full dataset instead.")
        return df, warnings, applied_window
    return filtered, warnings, applied_window


def _apply_entity_filter(
    df: pd.DataFrame,
    series_column: Optional[str],
    entity_keywords: Sequence[str],
) -> tuple[pd.DataFrame, List[str], Optional[List[str]], Optional[str]]:
    warnings: List[str] = []
    if not entity_keywords:
        return df, warnings, None, series_column

    candidate_columns: List[Optional[str]] = []
    if series_column:
        candidate_columns.append(series_column)
    candidate_columns.extend(
        [
            col
            for col in df.columns
            if col not in candidate_columns
            and (pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]))
        ]
    )

    for column in candidate_columns:
        if not column:
            continue
        series = df[column].astype(str).str.lower()
        mask = pd.Series(False, index=df.index)
        matched_entities: List[str] = []
        for keyword in entity_keywords:
            lowered = keyword.strip().lower()
            if not lowered:
                continue
            cond = series.str.contains(re.escape(lowered), na=False)
            if cond.any():
                mask |= cond
                matched_entities.append(keyword)
        if matched_entities:
            filtered = df.loc[mask]
            if filtered.empty:
                continue
            resolved_column = column
            return filtered, warnings, matched_entities, resolved_column

    warnings.append(
        "Entity keywords provided but no matching columns were found; using all rows."
    )
    return df, warnings, None, series_column


def _slugify(text: str, fallback: str = "visualization") -> str:
    tokens = TOKEN_PATTERN.findall(text.lower())
    return "-".join(tokens) or fallback


def _build_output_path(dataset_key: str, plan: VisualizationPlanOutput) -> Path:
    slug_source = f"{dataset_key}-{plan.chart_type}-{plan.metric_keywords[:2]}"
    slug = _slugify(slug_source)
    return Path("reports") / f"{slug}.png"


def _render_chart(
    df: pd.DataFrame,
    chart_type: str,
    time_column: str,
    value_column: str,
    series_column: Optional[str],
    title: str,
):
    plot_kwargs: Dict[str, Any] = {
        "data_frame": df,
        "x": time_column,
        "y": value_column,
        "title": title or None,
    }
    if series_column:
        plot_kwargs["color"] = series_column
    if chart_type == "line":
        return px.line(**plot_kwargs)
    if chart_type == "bar":
        return px.bar(**plot_kwargs)
    if chart_type == "scatter":
        return px.scatter(**plot_kwargs)
    raise ValueError(f"Unsupported chart type: {chart_type}")


def _save_figure(fig, output_path: Path) -> tuple[Path, List[str]]:
    warnings: List[str] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    try:
        if suffix in {".png", ".jpg", ".jpeg", ".svg", ".pdf", ".webp"}:
            fig.write_image(str(output_path))
        else:
            fig.write_html(str(output_path))
    except Exception as exc:  # pragma: no cover - fallback for missing kaleido
        fallback = output_path.with_suffix(".html")
        try:
            fig.write_html(str(fallback))
        except Exception as html_exc:  # pragma: no cover
            raise RuntimeError(
                f"Failed to save chart to {output_path}: {html_exc}"
            ) from exc
        warnings.append(f"Image export failed ({exc}); chart saved as HTML at {fallback}.")
        output_path = fallback
    return output_path, warnings


def _attempt_visualization(
    dataset_key: str,
    dataset_reason: str,
    plan: VisualizationPlanOutput,
    datastore_obj: DataStore,
) -> dict:
    """
    Attempt to render a visualization for a single dataset.
    Returns a dictionary with keys:
        success (bool)
        warnings (List[str])
        summary (Optional[str])
        output_path (Optional[str])
        detected_columns (Dict[str, Any])
        reason (Optional[str]) -- failure reason when success == False
    """
    warnings: List[str] = []
    try:
        df = datastore_obj.get_df(dataset_key)
    except KeyError:
        return {
            "success": False,
            "warnings": warnings,
            "reason": "dataset missing from datastore",
        }

    if df.empty:
        return {
            "success": False,
            "warnings": warnings,
            "reason": "dataset empty",
        }

    working_df = df.reset_index(drop=True).copy()
    time_column, synthetic_time, time_warnings = _ensure_time_column(working_df)
    warnings.extend(time_warnings)
    value_column, value_warnings = _ensure_value_column(working_df, time_column, plan.metric_keywords)
    warnings.extend(value_warnings)
    series_column, series_warnings = _ensure_series_column(working_df, plan.entity_keywords)
    warnings.extend(series_warnings)

    filtered_df, time_filter_warnings, applied_window = _apply_time_window_filter(
        working_df,
        time_column,
        plan.time_window,
        allow_filter=not synthetic_time,
    )
    warnings.extend(time_filter_warnings)

    filtered_df, entity_warnings, matched_entities, resolved_series = _apply_entity_filter(
        filtered_df, series_column, plan.entity_keywords
    )
    warnings.extend(entity_warnings)

    if filtered_df.empty:
        return {
            "success": False,
            "warnings": warnings,
            "reason": "no rows available after filtering",
        }

    title = plan.chart_intent or f"{value_column} over time"
    detected_columns = {
        "time_column": time_column,
        "value_column": value_column,
        "series_column": resolved_series,
        "applied_window": applied_window,
        "matched_entities": matched_entities,
        "dataset_reason": dataset_reason,
        "synthetic_time": synthetic_time,
    }

    try:
        figure = _render_chart(
            filtered_df,
            plan.chart_type,
            time_column,
            value_column,
            resolved_series,
            title,
        )
    except Exception as exc:
        return {
            "success": False,
            "warnings": warnings,
            "reason": f"plotting failed: {exc}",
        }

    output_path = _build_output_path(dataset_key, plan)
    try:
        final_path, save_warnings = _save_figure(figure, output_path)
    except Exception as exc:  # pragma: no cover - defensive
        return {
            "success": False,
            "warnings": warnings,
            "reason": f"saving failed: {exc}",
        }

    warnings.extend(save_warnings)
    if plan.priority_notes:
        warnings.append(plan.priority_notes)

    window_text = applied_window or plan.time_window or "full dataset"
    entity_text = (
        f" filtered to {', '.join(matched_entities)}"
        if matched_entities
        else ""
    )
    summary = (
        f"Saved {plan.chart_type} chart of {value_column} vs {time_column}{entity_text} "
        f"from {dataset_key} ({window_text}) to {final_path}."
    )

    return {
        "success": True,
        "warnings": warnings,
        "summary": summary,
        "output_path": str(final_path),
        "detected_columns": detected_columns,
    }


def create_visualization_agent(llm):
    """
    Build and return the visualization agent workflow.
    """

    def load_context_node(state: VisualizationState) -> VisualizationState:
        datastore_obj = state.get("datastore_obj")
        if not isinstance(datastore_obj, DataStore):
            datastore_obj = DATASTORE
        state["datastore_obj"] = datastore_obj
        datastore = _normalize_datastore(state.get("datastore"))
        if not datastore:
            datastore = datastore_obj.snapshot()
        state["datastore"] = datastore
        state["datastore_summary"] = _summarize_datastore(datastore)
        state.setdefault("warnings", [])
        return state

    def plan_visualization_node(state: VisualizationState) -> VisualizationState:
        instruction = state.get("instruction", "").strip()
        datastore_summary = state.get("datastore_summary", "No datasets available.")
        try:
            response = llm.invoke(
                [
                    ("system", VISUALIZATION_SYSTEM_PROMPT),
                    (
                        "human",
                        f"Instruction: {instruction or 'Create a helpful chart.'}\n\n"
                        f"Datastore inventory:\n{datastore_summary}",
                    ),
                ]
            )
        except Exception as exc:  # pragma: no cover - defensive
            state["error_message"] = f"Failed to request visualization plan: {exc}"
            state["visualization_agent_final_answer"] = (
                "Visualization agent could not generate a plan."
            )
            return state
        plan = response if isinstance(response, VisualizationPlanOutput) else VisualizationPlanOutput(**response)
        state["chart_plan"] = plan.model_dump()
        return state

    def render_visualization_node(state: VisualizationState) -> VisualizationState:
        plan_dict = state.get("chart_plan") or {}
        try:
            plan = VisualizationPlanOutput(**plan_dict)
        except Exception as exc:
            state["error_message"] = f"Invalid visualization plan: {exc}"
            state["visualization_agent_final_answer"] = (
                "Visualization agent returned an invalid plan."
            )
            return state

        datastore_obj = state.get("datastore_obj")
        if not isinstance(datastore_obj, DataStore):
            datastore_obj = DATASTORE

        datastore_snapshot = state.get("datastore") or {}
        instruction = state.get("instruction", "")
        warnings = list(state.get("warnings", []))

        ranked_datasets = _rank_datasets(datastore_snapshot, plan, instruction)
        if not ranked_datasets:
            state["error_message"] = "No datasets available in datastore."
            state["visualization_agent_final_answer"] = (
                "Visualization skipped because no datasets were present."
            )
            state["warnings"] = warnings
            return state

        attempt_logs: List[str] = []
        for dataset_key, dataset_reason, _score in ranked_datasets:
            attempt = _attempt_visualization(dataset_key, dataset_reason, plan, datastore_obj)
            warnings.extend(attempt.get("warnings", []))
            if attempt.get("success"):
                state["selected_dataset"] = dataset_key
                detected_columns = attempt.get("detected_columns", {})
                if detected_columns:
                    state["detected_columns"] = detected_columns
                state["visualization_agent_final_answer"] = attempt.get(
                    "summary",
                    f"Visualization saved for dataset {dataset_key}.",
                )
                state["output_path"] = attempt.get("output_path", "")
                state["warnings"] = warnings
                print(f"[Visualization Agent] success: dataset={dataset_key} output={state['output_path']}")
                return state
            reason = attempt.get("reason") or "unknown issue"
            attempt_logs.append(f"{dataset_key}: {reason}")

        state["error_message"] = "Unable to create a visualization from available datasets."
        if attempt_logs:
            warnings.append("Tried datasets: " + "; ".join(attempt_logs))
        print(f"[Visualization Agent] failure: {state['error_message']} | {attempt_logs}")
        state["visualization_agent_final_answer"] = (
            "Visualization agent could not build a chart from the current datastore."
        )
        state["warnings"] = warnings
        return state

    workflow = StateGraph(VisualizationState)
    workflow.add_node("load_context", load_context_node)
    workflow.add_node("plan_visualization", plan_visualization_node)
    workflow.add_node("render_visualization", render_visualization_node)

    workflow.add_edge(START, "load_context")
    workflow.add_edge("load_context", "plan_visualization")
    workflow.add_edge("plan_visualization", "render_visualization")
    workflow.add_edge("render_visualization", END)

    return workflow.compile()


__all__ = ["create_visualization_agent"]
