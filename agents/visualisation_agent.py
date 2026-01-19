"""Minimal LangGraph workflow for the visualization agent."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
from langgraph.graph import END, START, StateGraph

from utils.datastore import DATASTORE, DataStore
from utils.output_basemodels import VisualizationFigurePlan, VisualizationPlanOutput
from utils.states import VisualizationState

PROMPT_PATH = Path("prompts/visualization_agent_prompt.txt")
VISUALIZATION_SYSTEM_PROMPT = PROMPT_PATH.read_text(encoding="utf-8")

TIME_HINTS = ("time", "timestamp", "date", "datetime", "ts", "month", "year", "week", "period")
VALUE_HINTS = ("value", "reading", "metric", "measurement", "count", "total", "avg", "mean")
ALLOWED_CHART_TYPES = (
    "line",
    "area",
    "bar",
    "stacked_bar",
    "scatter",
    "histogram",
    "box",
    "heatmap",
    "step",
    "rolling_line",
)
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")
SAMPLE_ROWS = 500
MAX_SERIES_CATEGORIES = 12
DEFAULT_MAX_FIGURES = 3
DEFAULT_MAX_METRICS = 3
DEFAULT_MAX_ENTITIES = 8
HEATMAP_MAX_DAYS = 31
ROLLING_WINDOW = 7
TARGET_POINTS = 800


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


def _invoke_plan(llm, instruction: str, datastore_summary: str) -> VisualizationPlanOutput:
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
    return response if isinstance(response, VisualizationPlanOutput) else VisualizationPlanOutput(**response)


def _tokenize(text: str) -> List[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text or "")]


def _column_tokens(name: Any) -> List[str]:
    raw = str(name)
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", raw)
    return _tokenize(spaced)


def _extract_axis_hints(text: str) -> List[str]:
    lowered = text.lower()
    axis_patterns = {
        "x": [
            r"\bx[-_ ]?axis\b",
            r"\baxis[-_ ]?x\b",
            r"\baccel(?:eration)?[-_ ]?x\b",
        ],
        "y": [
            r"\by[-_ ]?axis\b",
            r"\baxis[-_ ]?y\b",
            r"\baccel(?:eration)?[-_ ]?y\b",
        ],
        "z": [
            r"\bz[-_ ]?axis\b",
            r"\baxis[-_ ]?z\b",
            r"\baccel(?:eration)?[-_ ]?z\b",
        ],
    }
    hits: List[Tuple[int, str]] = []
    for axis, patterns in axis_patterns.items():
        positions = []
        for pattern in patterns:
            match = re.search(pattern, lowered)
            if match:
                positions.append(match.start())
        if positions:
            hits.append((min(positions), axis))
    hits.sort(key=lambda item: item[0])
    return [axis for _, axis in hits]


def _extract_axis_hint(text: str) -> Optional[str]:
    hints = _extract_axis_hints(text)
    return hints[0] if hints else None


def _score_dataset(text: str, hints: List[str], tokens: List[str]) -> float:
    lowered = text.lower()
    score = 0.0
    for hint in hints:
        if hint and hint.lower() in lowered:
            score += 2.0
    for token in tokens:
        if token and token in lowered:
            score += 0.25
    return score


def _select_dataset(
    datastore: Dict[str, Dict[str, Any]],
    plan: VisualizationFigurePlan,
    instruction: str,
    used_keys: Optional[set[str]] = None,
) -> Tuple[Optional[str], str]:
    if not datastore:
        return None, ""
    if len(datastore) == 1:
        key, payload = next(iter(datastore.items()))
        return key, str(payload.get("description", "")).strip()
    for hint in plan.dataset_hints:
        if not hint:
            continue
        for key, payload in datastore.items():
            if hint.strip().lower() == str(key).lower():
                return key, str(payload.get("description", "")).strip()
    hints = list(plan.dataset_hints) + list(plan.metric_keywords) + list(plan.entity_keywords)
    tokens = _tokenize(instruction + " " + plan.chart_intent + " " + plan.priority_notes)
    best_key = None
    best_score = -1.0
    best_description = ""
    for key, payload in datastore.items():
        description = str(payload.get("description", "")).strip()
        text = f"{key} {description}"
        score = _score_dataset(text, hints, tokens)
        row_count = payload.get("row_count")
        if isinstance(row_count, int):
            score += min(row_count, 2000) / 2000.0
        if score > best_score:
            best_score = score
            best_key = key
            best_description = description
    if used_keys and best_key in used_keys:
        for key, payload in datastore.items():
            if key == best_key:
                continue
            description = str(payload.get("description", "")).strip()
            text = f"{key} {description}"
            score = _score_dataset(text, hints, tokens)
            if score >= max(0.1, best_score - 0.2):
                return key, description
    return best_key, best_description


def _sample_series(series: pd.Series, max_rows: int = SAMPLE_ROWS) -> pd.Series:
    if len(series) <= max_rows:
        return series
    return series.head(max_rows)


def _find_time_column(df: pd.DataFrame) -> Optional[str]:
    for column in df.columns:
        name = str(column).lower()
        if any(hint in name for hint in TIME_HINTS):
            return column
    for column in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[column]):
            return column
    for column in df.columns:
        if pd.api.types.is_object_dtype(df[column]):
            sample = _sample_series(df[column].dropna())
            parsed = pd.to_datetime(sample, errors="coerce")
            if parsed.notna().mean() > 0.6:
                df[column] = pd.to_datetime(df[column], errors="coerce")
                return column
    return None


def _get_numeric_columns(df: pd.DataFrame, time_column: Optional[str]) -> List[str]:
    numeric_columns: List[str] = []
    for column in df.columns:
        if column == time_column:
            continue
        series = df[column]
        if pd.api.types.is_numeric_dtype(series):
            numeric_columns.append(column)
            continue
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            sample = _sample_series(series)
            converted_sample = pd.to_numeric(sample, errors="coerce")
            if converted_sample.notna().mean() > 0.6:
                df[column] = pd.to_numeric(df[column], errors="coerce")
                numeric_columns.append(column)
    return numeric_columns


def _rank_value_columns(
    columns: List[str],
    metric_keywords: List[str],
    context_text: str,
) -> List[str]:
    metric_keywords = [kw.lower() for kw in metric_keywords if kw]
    axis_hint = _extract_axis_hint(context_text)
    context_tokens = _tokenize(context_text)
    scored: List[Tuple[str, float]] = []
    axis_tokens = {"x", "y", "z"}
    for column in columns:
        name = str(column).lower()
        col_tokens = set(_column_tokens(column))
        score = 0.0
        for keyword in metric_keywords:
            keyword_tokens = _tokenize(keyword)
            if keyword_tokens and all(token in col_tokens for token in keyword_tokens):
                score += 2.0
            else:
                score += 0.5 * sum(1 for token in keyword_tokens if token in col_tokens)
        if any(hint in name for hint in VALUE_HINTS):
            score += 1.0
        if axis_hint:
            if axis_hint in col_tokens:
                score += 1.5
            if (axis_tokens - {axis_hint}) & col_tokens:
                score -= 0.5
        if context_tokens:
            score += 0.2 * sum(1 for token in context_tokens if token in col_tokens)
        scored.append((column, score))
    scored.sort(key=lambda item: item[1], reverse=True)
    return [item[0] for item in scored]


def _score_axis_column(
    column: str,
    metric_keywords: List[str],
    context_tokens: List[str],
    axis: str,
) -> float:
    name = str(column).lower()
    col_tokens = set(_column_tokens(column))
    score = 0.0
    if axis in col_tokens:
        score += 2.0
    for keyword in metric_keywords:
        keyword_tokens = _tokenize(keyword)
        if keyword_tokens and all(token in col_tokens for token in keyword_tokens):
            score += 1.5
        else:
            score += 0.4 * sum(1 for token in keyword_tokens if token in col_tokens)
    if "accel" in context_tokens and "accel" in col_tokens:
        score += 0.8
    if any(hint in name for hint in VALUE_HINTS):
        score += 0.5
    return score


def _pick_axis_columns(
    columns: List[str],
    metric_keywords: List[str],
    context_text: str,
    axis_hints: List[str],
) -> Tuple[List[str], List[str]]:
    warnings: List[str] = []
    context_tokens = _tokenize(context_text)
    selected: List[str] = []
    for axis in axis_hints:
        candidates = [col for col in columns if axis in _column_tokens(col)]
        if not candidates:
            warnings.append(f"No '{axis}' axis metric found.")
            continue
        best = max(
            candidates,
            key=lambda col: _score_axis_column(col, metric_keywords, context_tokens, axis),
        )
        if best not in selected:
            selected.append(best)
    return selected, warnings


def _select_value_columns(
    df: pd.DataFrame,
    time_column: Optional[str],
    metric_keywords: List[str],
    max_metrics: int,
    context_text: str,
) -> Tuple[List[str], List[str]]:
    warnings: List[str] = []
    numeric_columns = _get_numeric_columns(df, time_column)
    if not numeric_columns:
        fallback = "__row_value"
        df[fallback] = range(len(df))
        warnings.append("No numeric column detected; using row index for the y-axis.")
        return [fallback], warnings

    axis_hints = _extract_axis_hints(context_text)
    selected: List[str] = []
    if len(axis_hints) > 1:
        axis_selected, axis_warnings = _pick_axis_columns(
            numeric_columns,
            metric_keywords,
            context_text,
            axis_hints,
        )
        selected.extend(axis_selected)
        warnings.extend(axis_warnings)
        max_metrics = max(max_metrics, len(selected))

    ranked = _rank_value_columns(numeric_columns, metric_keywords, context_text)
    selected.extend([col for col in ranked if col not in selected])
    selected = [col for col in selected if col in numeric_columns][:max_metrics]
    if not selected:
        selected = numeric_columns[:max_metrics]
    if len(selected) > max_metrics:
        selected = selected[:max_metrics]
    axis_hint = _extract_axis_hint(context_text)
    if axis_hint and not any(axis_hint in _column_tokens(col) for col in selected):
        warnings.append(f"No '{axis_hint}' axis metric found; using best available columns.")
    if len(selected) < len(ranked):
        warnings.append(
            f"Limiting metrics to {len(selected)} column(s) to avoid overcrowding."
        )
    return selected, warnings


def _find_series_column(
    df: pd.DataFrame,
    time_column: Optional[str],
    value_columns: List[str],
    entity_keywords: List[str],
) -> Optional[str]:
    entity_keywords = [kw.lower() for kw in entity_keywords if kw]
    candidate_columns: List[str] = []
    for column in df.columns:
        if column in {time_column, *value_columns}:
            continue
        if pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_categorical_dtype(df[column]):
            candidate_columns.append(column)
    for keyword in entity_keywords:
        for column in candidate_columns:
            if keyword in str(column).lower():
                return column
    if entity_keywords:
        for column in candidate_columns:
            sample = _sample_series(df[column].astype(str).str.lower())
            for keyword in entity_keywords:
                if sample.str.contains(re.escape(keyword), na=False).any():
                    return column
    for column in candidate_columns:
        if df[column].nunique(dropna=True) <= MAX_SERIES_CATEGORIES:
            return column
    return None


def _resolve_chart_type(
    plan: VisualizationFigurePlan,
    instruction: str,
    time_column: Optional[str],
) -> str:
    if plan.chart_type in ALLOWED_CHART_TYPES:
        return plan.chart_type
    tokens = _tokenize(f"{instruction} {plan.chart_intent} {plan.priority_notes}")
    if any(token in {"smooth", "smoothed", "tsmooth", "rolling", "moving", "ema"} for token in tokens):
        return "rolling_line"
    if any(token in {"histogram", "distribution"} for token in tokens):
        return "histogram"
    if any(token in {"box", "boxplot"} for token in tokens):
        return "box"
    if any(token in {"heatmap", "density"} for token in tokens):
        return "heatmap"
    if time_column:
        return "line"
    return "bar"


def _slugify(text: str, fallback: str = "visualization") -> str:
    tokens = TOKEN_PATTERN.findall(text.lower())
    return "-".join(tokens) or fallback


def _build_output_path(dataset_key: str, chart_type: str, suffix: str) -> Path:
    slug = _slugify(f"{dataset_key}-{chart_type}-{suffix}")
    return Path("reports") / f"{slug}.png"


def _choose_resample_rule(span: pd.Timedelta, target_points: int) -> str:
    span_seconds = max(span.total_seconds(), 1.0)
    target_seconds = span_seconds / max(target_points, 1)
    candidates: List[Tuple[float, str]] = [
        (60, "1min"),
        (300, "5min"),
        (900, "15min"),
        (3600, "1h"),
        (21600, "6h"),
        (86400, "1d"),
        (604800, "1w"),
        (2592000, "1M"),
    ]
    for seconds, rule in candidates:
        if target_seconds <= seconds:
            return rule
    return "1M"


def _resample_time_series(
    df: pd.DataFrame,
    time_column: str,
    value_columns: List[str],
    series_column: Optional[str],
    target_points: int,
) -> Tuple[pd.DataFrame, Optional[str]]:
    if not pd.api.types.is_datetime64_any_dtype(df[time_column]):
        return df, None
    if len(df) <= target_points:
        return df, None
    sorted_df = df.sort_values(time_column)
    span = sorted_df[time_column].max() - sorted_df[time_column].min()
    rule = _choose_resample_rule(span, target_points)
    if series_column:
        grouped = (
            sorted_df.set_index(time_column)
            .groupby([pd.Grouper(freq=rule), series_column])[value_columns]
            .mean()
            .reset_index()
        )
    else:
        grouped = (
            sorted_df.set_index(time_column)[value_columns]
            .resample(rule)
            .mean()
            .reset_index()
        )
    return grouped.dropna(subset=value_columns, how="all"), rule


def _downsample_for_scatter(
    df: pd.DataFrame,
    time_column: str,
    target_points: int,
) -> Tuple[pd.DataFrame, bool]:
    if len(df) <= target_points:
        return df, False
    sorted_df = df.sort_values(time_column)
    step = max(len(sorted_df) // target_points, 1)
    return sorted_df.iloc[::step].copy(), True


def _apply_time_window_filter(
    df: pd.DataFrame,
    time_column: str,
    window_text: str,
    *,
    allow_filter: bool,
) -> Tuple[pd.DataFrame, List[str], Optional[str]]:
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
            delta = pd.Timedelta(days=7 * qty)
        elif "day" in unit:
            delta = pd.Timedelta(days=qty)
        else:
            delta = pd.Timedelta(hours=qty)
        start = now - delta
        applied_window = f"last {qty} {unit}"
    elif lowered in {"today"}:
        start = pd.Timestamp(now.date())
        end = start + pd.Timedelta(days=1)
        applied_window = "today"
    elif lowered in {"yesterday"}:
        end = pd.Timestamp(now.date())
        start = end - pd.Timedelta(days=1)
        applied_window = "yesterday"
    elif lowered in {"last week"}:
        start = now - pd.Timedelta(days=7)
        applied_window = "last 7 days"
    elif lowered in {"last month"}:
        start = now - pd.Timedelta(days=30)
        applied_window = "last 30 days"
    elif lowered in {"last 24 hours"}:
        start = now - pd.Timedelta(hours=24)
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
    entity_keywords: List[str],
    max_entities: int,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    warnings: List[str] = []
    matched_entities: List[str] = []
    if not series_column:
        return df, warnings, matched_entities

    series = df[series_column].astype(str)
    lowered = series.str.lower()
    keywords = [kw.strip().lower() for kw in entity_keywords if kw.strip()]
    if keywords:
        mask = pd.Series(False, index=df.index)
        for keyword in keywords:
            cond = lowered.str.contains(re.escape(keyword), na=False)
            if cond.any():
                mask |= cond
                matched_entities.append(keyword)
        filtered = df.loc[mask] if mask.any() else df
    else:
        filtered = df

    if filtered[series_column].nunique(dropna=True) > max_entities:
        top_entities = (
            filtered[series_column]
            .value_counts(dropna=True)
            .head(max_entities)
            .index.astype(str)
        )
        filtered = filtered[filtered[series_column].astype(str).isin(top_entities)]
        warnings.append(
            f"Limiting entities to top {len(top_entities)} values to avoid overcrowding."
        )
    return filtered, warnings, matched_entities


def _normalize_plan(plan: VisualizationPlanOutput) -> Tuple[List[VisualizationFigurePlan], Dict[str, int]]:
    figures = list(plan.figures) if plan.figures else []
    if not figures:
        figures = [
            VisualizationFigurePlan(
                chart_intent=plan.chart_intent or plan.overall_intent,
                chart_type=plan.chart_type,
                dataset_hints=plan.dataset_hints,
                metric_keywords=plan.metric_keywords,
                entity_keywords=plan.entity_keywords,
                time_window=plan.time_window,
                priority_notes=plan.priority_notes,
            )
        ]
    max_figures = max(1, plan.max_figures or DEFAULT_MAX_FIGURES)
    limits = {
        "max_figures": max_figures,
        "max_metrics": max(1, plan.max_metrics_per_figure or DEFAULT_MAX_METRICS),
        "max_entities": max(1, plan.max_entities_per_figure or DEFAULT_MAX_ENTITIES),
    }
    return figures[:max_figures], limits


def _render_chart(
    df: pd.DataFrame,
    chart_type: str,
    x_column: str,
    y_columns: List[str],
    series_column: Optional[str],
    title: str,
) -> Tuple[Any, List[str]]:
    warnings: List[str] = []
    if chart_type == "line":
        fig = px.line(
            df,
            x=x_column,
            y=y_columns,
            color=series_column,
            title=title or None,
        )
        return fig, warnings
    if chart_type == "area":
        fig = px.area(
            df,
            x=x_column,
            y=y_columns,
            color=series_column,
            title=title or None,
        )
        return fig, warnings
    if chart_type == "step":
        fig = px.line(
            df,
            x=x_column,
            y=y_columns,
            color=series_column,
            title=title or None,
            line_shape="hv",
        )
        return fig, warnings
    if chart_type == "rolling_line":
        rolling_df = df.sort_values(x_column)
        rolled_columns: List[str] = []
        for col in y_columns:
            rolled_name = f"__rolling_{col}"
            if series_column:
                rolling_df[rolled_name] = (
                    rolling_df.groupby(series_column)[col]
                    .transform(
                        lambda s: s.rolling(window=ROLLING_WINDOW, min_periods=max(2, ROLLING_WINDOW // 2)).mean()
                    )
                )
            else:
                rolling_df[rolled_name] = (
                    rolling_df[col]
                    .rolling(window=ROLLING_WINDOW, min_periods=max(2, ROLLING_WINDOW // 2))
                    .mean()
                )
            rolled_columns.append(rolled_name)
        fig = px.line(
            rolling_df,
            x=x_column,
            y=rolled_columns,
            color=series_column if len(rolled_columns) == 1 else None,
            title=title or None,
        )
        warnings.append(f"Applied rolling mean window={ROLLING_WINDOW}.")
        return fig, warnings
    if chart_type in {"bar", "stacked_bar"}:
        fig = px.bar(
            df,
            x=x_column,
            y=y_columns,
            color=series_column,
            title=title or None,
        )
        barmode = "stack" if chart_type == "stacked_bar" else "group"
        fig.update_layout(barmode=barmode)
        return fig, warnings
    if chart_type == "scatter":
        fig = px.scatter(
            df,
            x=x_column,
            y=y_columns[0],
            color=series_column,
            title=title or None,
        )
        return fig, warnings
    if chart_type == "histogram":
        fig = px.histogram(
            df,
            x=y_columns[0],
            color=series_column,
            title=title or None,
        )
        return fig, warnings
    if chart_type == "box":
        fig = px.box(
            df,
            x=series_column,
            y=y_columns[0],
            title=title or None,
        )
        return fig, warnings
    if chart_type == "heatmap":
        if not pd.api.types.is_datetime64_any_dtype(df[x_column]):
            raise ValueError("Heatmap requires a datetime x-axis.")
        heatmap_df = df.copy()
        heatmap_df["__viz_date"] = heatmap_df[x_column].dt.date
        heatmap_df["__viz_hour"] = heatmap_df[x_column].dt.hour
        dates = sorted(heatmap_df["__viz_date"].unique())
        if len(dates) > HEATMAP_MAX_DAYS:
            dates = dates[-HEATMAP_MAX_DAYS:]
            heatmap_df = heatmap_df[heatmap_df["__viz_date"].isin(dates)]
            warnings.append(
                f"Heatmap limited to last {HEATMAP_MAX_DAYS} days to stay readable."
            )
        pivot = heatmap_df.pivot_table(
            index="__viz_date",
            columns="__viz_hour",
            values=y_columns[0],
            aggfunc="mean",
        )
        fig = px.imshow(
            pivot,
            aspect="auto",
            title=title or None,
            labels={"x": "hour", "y": "date", "color": y_columns[0]},
        )
        return fig, warnings
    raise ValueError(f"Unsupported chart type: {chart_type}")


def _save_figure(fig, output_path: Path) -> Tuple[Path, List[str]]:
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


def create_visualization_agent(llm):
    """
    Build a minimal visualization agent workflow inspired by the SQL agent pattern.
    """

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

    def plan_visualization_node(state: VisualizationState) -> VisualizationState:
        instruction = str(state.get("instruction", "")).strip()
        datastore_summary = state.get("datastore_summary", "No datasets available.")
        warnings = list(state.get("warnings", []))
        try:
            plan = _invoke_plan(llm, instruction, datastore_summary)
        except Exception as exc:  # pragma: no cover - defensive
            warnings.append(f"LLM plan failed ({exc}); using default line chart plan.")
            plan = VisualizationPlanOutput(
                overall_intent=instruction or "Visualize the available data.",
                figures=[
                    VisualizationFigurePlan(
                        chart_intent=instruction or "Visualize the available data.",
                        chart_type="line",
                        dataset_hints=[],
                        metric_keywords=[],
                        entity_keywords=[],
                        time_window="",
                        priority_notes="Auto-generated fallback plan.",
                    )
                ],
                priority_notes="Auto-generated fallback plan.",
            )
        state["chart_plan"] = plan.model_dump()
        state["warnings"] = warnings
        return state

    def render_visualization_node(state: VisualizationState) -> VisualizationState:
        warnings = list(state.get("warnings", []))
        plan_dict = state.get("chart_plan") or {}
        try:
            plan = VisualizationPlanOutput(**plan_dict)
        except Exception as exc:
            state["error_message"] = f"Invalid visualization plan: {exc}"
            state["visualization_agent_final_answer"] = (
                "Visualization agent could not render a chart due to invalid plan."
            )
            state["warnings"] = warnings
            return state

        datastore_obj = state.get("datastore_obj")
        if not isinstance(datastore_obj, DataStore):
            datastore_obj = DATASTORE

        datastore_snapshot = state.get("datastore") or {}
        instruction = str(state.get("instruction", ""))
        figure_plans, limits = _normalize_plan(plan)
        if not figure_plans:
            state["error_message"] = "Visualization plan contained no figures."
            state["visualization_agent_final_answer"] = (
                "Visualization agent could not identify any figures to render."
            )
            state["warnings"] = warnings
            return state

        visualizations: List[Dict[str, Any]] = []
        output_paths: List[str] = []
        used_datasets: set[str] = set()

        for idx, figure_plan in enumerate(figure_plans, start=1):
            dataset_key, dataset_description = _select_dataset(
                datastore_snapshot,
                figure_plan,
                instruction,
                used_keys=used_datasets,
            )
            if not dataset_key:
                warnings.append("No datasets available in datastore for a figure.")
                continue
            used_datasets.add(dataset_key)

            try:
                df = datastore_obj.get_df(dataset_key)
            except KeyError:
                warnings.append(f"Dataset '{dataset_key}' missing from datastore.")
                continue

            if df.empty:
                warnings.append(f"Dataset '{dataset_key}' is empty.")
                continue

            working_df = df.reset_index(drop=True).copy()
            time_column = _find_time_column(working_df)
            synthetic_time = False
            if not time_column:
                time_column = "__row_index"
                working_df[time_column] = range(len(working_df))
                synthetic_time = True
                warnings.append("No time column detected; using row index for the x-axis.")

            filtered_df, time_warnings, applied_window = _apply_time_window_filter(
                working_df,
                time_column,
                figure_plan.time_window,
                allow_filter=not synthetic_time,
            )
            warnings.extend(time_warnings)

            context_text = " ".join(
                [
                    instruction,
                    figure_plan.chart_intent,
                    figure_plan.priority_notes,
                    " ".join(figure_plan.metric_keywords),
                ]
            ).strip()
            value_columns, value_warnings = _select_value_columns(
                filtered_df,
                time_column,
                figure_plan.metric_keywords,
                limits["max_metrics"],
                context_text,
            )
            warnings.extend(value_warnings)

            series_column = _find_series_column(
                filtered_df,
                time_column,
                value_columns,
                figure_plan.entity_keywords,
            )
            filtered_df, entity_warnings, matched_entities = _apply_entity_filter(
                filtered_df,
                series_column,
                figure_plan.entity_keywords,
                limits["max_entities"],
            )
            warnings.extend(entity_warnings)

            if filtered_df.empty:
                warnings.append(
                    f"Figure {idx} removed all rows after filtering; skipping."
                )
                continue

            if len(value_columns) > 1 and series_column:
                warnings.append(
                    "Multiple metrics detected; ignoring entity series to reduce clutter."
                )
                series_column = None

            chart_type = _resolve_chart_type(
                figure_plan,
                instruction,
                None if synthetic_time else time_column,
            )
            if chart_type == "heatmap" and synthetic_time:
                warnings.append("Heatmap requires a time column; skipping figure.")
                continue
            title = figure_plan.chart_intent or plan.overall_intent or f"{dataset_key} chart"

            plot_df = filtered_df
            resample_rule = None
            downsampled = False
            if chart_type in {"line", "area", "step", "rolling_line"} and not synthetic_time:
                plot_df, resample_rule = _resample_time_series(
                    filtered_df,
                    time_column,
                    value_columns,
                    series_column,
                    TARGET_POINTS,
                )
                if resample_rule:
                    warnings.append(
                        f"Resampled time series to {resample_rule} buckets for readability."
                    )
            if chart_type == "scatter" and not synthetic_time:
                plot_df, downsampled = _downsample_for_scatter(
                    filtered_df,
                    time_column,
                    TARGET_POINTS,
                )
                if downsampled:
                    warnings.append("Downsampled scatter points for readability.")

            try:
                figure, render_warnings = _render_chart(
                    plot_df,
                    chart_type,
                    time_column,
                    value_columns,
                    series_column,
                    title,
                )
            except Exception as exc:
                warnings.append(f"Plotting failed for figure {idx}: {exc}")
                continue

            warnings.extend(render_warnings)
            suffix = f"fig-{idx}"
            output_path = _build_output_path(dataset_key, chart_type, suffix)
            try:
                final_path, save_warnings = _save_figure(figure, output_path)
            except Exception as exc:  # pragma: no cover - defensive
                warnings.append(f"Saving failed for figure {idx}: {exc}")
                continue

            warnings.extend(save_warnings)
            summary_parts = [
                f"Saved {chart_type} chart for {dataset_key} to {final_path}."
            ]
            if applied_window:
                summary_parts.append(f"Window: {applied_window}.")
            if matched_entities:
                summary_parts.append(f"Entities: {', '.join(matched_entities)}.")
            if resample_rule:
                summary_parts.append(f"Resampled: {resample_rule}.")
            if downsampled:
                summary_parts.append("Downsampled points.")
            summary = " ".join(summary_parts)
            artifact = {
                "chart_path": str(final_path),
                "chart_type": chart_type,
                "dataset_key": dataset_key,
                "dataset_description": dataset_description,
                "time_column": time_column,
                "value_columns": value_columns,
                "series_column": series_column,
                "matched_entities": matched_entities,
                "applied_window": applied_window,
                "resample_rule": resample_rule,
                "downsampled": downsampled,
                "summary": summary,
            }
            visualizations.append(artifact)
            output_paths.append(str(final_path))
            print(
                f"[Visualization Agent] success: figure={idx} dataset={dataset_key} output={final_path}"
            )

        if not visualizations:
            state["error_message"] = "Unable to create any visualizations from available datasets."
            state["visualization_agent_final_answer"] = (
                "Visualization agent could not build charts from the current datastore."
            )
            state["warnings"] = warnings
            return state

        state["selected_dataset"] = visualizations[0]["dataset_key"]
        state["detected_columns"] = {
            "time_column": visualizations[0]["time_column"],
            "value_columns": visualizations[0]["value_columns"],
            "series_column": visualizations[0]["series_column"],
            "dataset_description": visualizations[0]["dataset_description"],
        }
        state["output_path"] = output_paths[0]
        state["output_paths"] = output_paths
        state["visualizations"] = visualizations
        state["visualization_agent_final_answer"] = (
            f"Generated {len(visualizations)} chart(s)."
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
