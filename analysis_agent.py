

from __future__ import annotations

import asyncio
import json
import math
import textwrap
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from datastore import load_description, load_df
from llm import llm_from

__all__ = [
    "AnalysisAgentConfig",
    "run_analysis_agent",
    "arun_analysis_agent",
]


@dataclass
class AnalysisAgentConfig:
    """Runtime controls for the analysis prompt construction."""

    max_table_rows: int = 60
    sample_head: int = 5
    sample_tail: int = 3
    max_numeric_columns: int = 6
    max_categorical_columns: int = 4
    round_precision: int = 4
    include_missing_stats: bool = True
    system_prompt: str = field(
        default_factory=lambda: textwrap.dedent(
            """
            You are the Analysis Agent for an infrastructure monitoring assistant.
            Study the datasets provided by the SQL Agent and produce concise, actionable insights.

            Guidelines:
            - Use only the supplied data; never invent measurements or time ranges.
            - Quantify trends, anomalies, threshold breaches, and comparisons between sensors.
            - Call out data limitations (gaps, stale coverage, insufficient columns) and recommend follow-up actions.
            - Reference dataset aliases (e.g., Dataset 1) when citing figures.
            - Respond with short paragraphs or bullet-like sentences that a supervisor can reuse directly.
            """
        ).strip()
    )


@dataclass
class LoadedDataset:
    """Container for datasets retrieved from the datastore or provided inline."""

    alias: str
    df: pd.DataFrame
    reference: Optional[str] = None
    description: str = ""
    context: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)


def run_analysis_agent(
    request: str,
    *,
    df_refs: Optional[Sequence[str]] = None,
    datasets: Optional[Dict[str, pd.DataFrame]] = None,
    config: Optional[AnalysisAgentConfig] = None,
) -> AIMessage:
    """Execute the analysis workflow synchronously."""

    if not request or not request.strip():
        raise ValueError("The analysis request must be a non-empty string.")

    config = config or AnalysisAgentConfig()
    refs = list(df_refs or [])
    inline = datasets or {}

    loaded, errors = _load_datasets(refs, inline)
    dataset_sections: List[str] = []

    for dataset in loaded:
        summary, payload = _build_dataset_summary(dataset.df, config=config)
        description = dataset.description or "No description provided."
        formatted = f"{dataset.alias} | {description}\n{summary}"
        dataset.context = formatted
        dataset.payload = payload
        dataset_sections.append(formatted)

    dataset_text = "\n\n".join(dataset_sections) if dataset_sections else "No datasets available."
    error_block = ""
    if errors:
        details = "\n".join(f"- {err['reference']}: {err['error']}" for err in errors)
        error_block = f"\n\nDataset load issues:\n{details}"

    human_prompt = textwrap.dedent(
        f"""
        User objective:
        {request.strip()}

        Dataset context:
        {dataset_text}{error_block}
        """
    ).strip()

    messages: List[BaseMessage] = [
        SystemMessage(content=config.system_prompt),
        HumanMessage(content=human_prompt),
    ]

    llm = llm_from()
    result = llm.invoke(messages)
    message = _ensure_ai_message(result)

    structured = {
        "agent": "Analysis Agent",
        "request": request.strip(),
        "datasets": [
            {
                "alias": ds.alias,
                "reference": ds.reference,
                "description": ds.description,
                "summary": ds.context,
                "payload": ds.payload,
            }
            for ds in loaded
        ],
        "errors": errors,
    }

    message.name = message.name or "Analysis Agent"
    message.additional_kwargs = {
        **message.additional_kwargs,
        "structured": {**message.additional_kwargs.get("structured", {}), **structured},
    }

    return message


async def arun_analysis_agent(
    request: str,
    *,
    df_refs: Optional[Sequence[str]] = None,
    datasets: Optional[Dict[str, pd.DataFrame]] = None,
    config: Optional[AnalysisAgentConfig] = None,
) -> AIMessage:
    """Async facade for contexts that prefer awaiting the analysis step."""

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        lambda: run_analysis_agent(
            request,
            df_refs=df_refs,
            datasets=datasets,
            config=config,
        ),
    )


# --------------------------------------------------------------------------- #
#  helpers
# --------------------------------------------------------------------------- #


def _ensure_ai_message(result: Any) -> AIMessage:
    """Normalise any LangChain response into an AIMessage instance."""

    if isinstance(result, AIMessage):
        return result
    if isinstance(result, BaseMessage):
        return AIMessage(content=result.content, additional_kwargs=result.additional_kwargs)
    if isinstance(result, str):
        return AIMessage(content=result)
    return AIMessage(content=json.dumps(result, default=str))


def _load_datasets(
    df_refs: Sequence[str],
    datasets: Dict[str, pd.DataFrame],
) -> Tuple[List[LoadedDataset], List[Dict[str, str]]]:
    """Materialise datasets from datastore references or inline arguments."""

    loaded: List[LoadedDataset] = []
    errors: List[Dict[str, str]] = []

    for index, ref in enumerate(df_refs, start=1):
        alias = f"Dataset {index}"
        try:
            df = load_df(ref)
            description = load_description(ref)
        except Exception as exc:  # broad to capture KeyError, ValueError, etc.
            errors.append({"reference": ref, "error": str(exc)})
            continue
        loaded.append(
            LoadedDataset(
                alias=alias,
                df=df,
                reference=ref,
                description=description,
            )
        )

    for alias, df in datasets.items():
        alias_str = str(alias)
        safe_alias = alias_str if alias_str.lower().startswith("dataset") else f"Dataset {alias_str}"
        loaded.append(
            LoadedDataset(alias=safe_alias, df=df, reference=None, description="Inline dataframe")
        )

    return loaded, errors


def _build_dataset_summary(
    df: pd.DataFrame,
    *,
    config: AnalysisAgentConfig,
) -> Tuple[str, Dict[str, Any]]:
    """Produce a textual and structured summary for a single dataframe."""

    if df.empty:
        return "Dataset is empty; request a fresh SQL pull.", {"rows": 0, "columns": [], "empty": True}

    rows, cols = df.shape
    truncated_df = _truncate_dataframe(df, max_rows=config.max_table_rows)
    column_info = _describe_columns(df)
    numeric_stats = _numeric_summary(df, config)
    categorical_stats = _categorical_summary(df, config)
    time_coverage = _time_coverage(df)
    missing_info = _missing_values(df) if config.include_missing_stats else {}
    head_records = _to_records(truncated_df.head(config.sample_head))
    tail_records = _to_records(truncated_df.tail(config.sample_tail)) if rows > config.sample_head else []

    descriptor_lines = [
        f"- shape: {rows} rows x {cols} columns",
        f"- columns: {', '.join(column_info)}",
    ]
    if time_coverage:
        descriptor_lines.append(
            f"- time coverage ({time_coverage['column']}): {time_coverage['start']} to {time_coverage['end']}"
        )
    if numeric_stats:
        descriptor_lines.append("- numeric summary: see payload.numeric")
    if categorical_stats:
        descriptor_lines.append("- categorical modes: see payload.categorical")
    if missing_info:
        descriptor_lines.append("- missing values: see payload.missing")

    summary = "\n".join(descriptor_lines)

    payload: Dict[str, Any] = {
        "rows": rows,
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "numeric": numeric_stats,
        "categorical": categorical_stats,
        "time_coverage": time_coverage,
        "missing": missing_info,
        "samples": {"head": head_records, "tail": tail_records},
    }

    return summary, payload


def _truncate_dataframe(df: pd.DataFrame, *, max_rows: int) -> pd.DataFrame:
    """Cap the dataframe length to avoid overwhelming the prompt."""

    if max_rows <= 0 or len(df) <= max_rows:
        return df
    head_count = max_rows // 2
    tail_count = max_rows - head_count
    head = df.head(head_count)
    tail = df.tail(tail_count)
    return pd.concat([head, tail])


def _describe_columns(df: pd.DataFrame) -> List[str]:
    """Return formatted column metadata."""

    return [f"{col} ({df[col].dtype})" for col in df.columns]


def _numeric_summary(df: pd.DataFrame, config: AnalysisAgentConfig) -> Dict[str, Dict[str, float]]:
    """Generate rounded descriptive statistics for numeric columns."""

    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.empty:
        return {}

    cols = numeric_df.columns[: config.max_numeric_columns]
    desc = numeric_df[cols].describe().transpose().round(config.round_precision)
    return desc.to_dict(orient="index")


def _categorical_summary(df: pd.DataFrame, config: AnalysisAgentConfig) -> Dict[str, Dict[str, Any]]:
    """Collect top categorical values with percentages."""

    categorical_cols = [
        column
        for column in df.columns
        if df[column].dtype == "object" or str(df[column].dtype).startswith("category")
    ][: config.max_categorical_columns]

    summary: Dict[str, Dict[str, Any]] = {}
    total_rows = max(len(df), 1)

    for column in categorical_cols:
        counts = df[column].value_counts(dropna=True).head(5)
        values = []
        for value, count in counts.items():
            percent = round(count / total_rows * 100, 2)
            values.append({"value": _to_serializable(value), "count": int(count), "percent": percent})
        summary[column] = {"top_values": values}

    return summary


def _time_coverage(df: pd.DataFrame) -> Dict[str, Any]:
    """Detect a likely timestamp column and return its coverage."""

    candidate_names = [col for col in df.columns if col.lower() in {"timestamp", "ts", "time", "datetime", "date"}]
    candidate_names.extend(df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist())

    for column in candidate_names:
        serie = pd.to_datetime(df[column], errors="coerce", utc=True)
        if serie.notna().any():
            start = serie.min().isoformat()
            end = serie.max().isoformat()
            return {"column": column, "start": start, "end": end}
    return {}


def _missing_values(df: pd.DataFrame) -> Dict[str, float]:
    """Compute the percentage of missing values per column."""

    missing = df.isna().mean()
    with_missing = missing[missing > 0]
    return {column: round(ratio * 100, 2) for column, ratio in with_missing.items()}


def _to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Convert a dataframe into serialisable row dictionaries."""

    records: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        record: Dict[str, Any] = {}
        for column, value in row.items():
            record[column] = _to_serializable(value)
        records.append(record)
    return records


def _to_serializable(value: Any) -> Any:
    """Convert values so they can be safely JSON serialised."""

    if pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, (pd.Timedelta, timedelta)):
        total_seconds = value.total_seconds()
        return total_seconds if not math.isnan(total_seconds) else None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    if isinstance(value, (set, tuple)):
        return [_to_serializable(v) for v in value]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value
