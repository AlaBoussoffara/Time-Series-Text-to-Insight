from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.messages import AIMessage

from agents.spider_sql_agent import PromptAgentAdapter
from agents.sql_agent import create_sql_agent
from utils.datastore import DataStore
from utils.general_helpers import llm_from
from utils.messages import AgentMessage
from utils.output_basemodels import SQLAgentOutput
from utils.states import SQLState


def _load_queries(path: Path, limit: Optional[int], start: int) -> List[str]:
    """Return a slice of benchmark queries from a JSON file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        queries = data.get("benchmark_queries") or []
    elif isinstance(data, list):
        queries = data
    else:
        queries = []
    if not queries:
        raise ValueError(f"No queries found in {path}")
    sliced = queries[start:]
    if limit is not None:
        sliced = sliced[:limit]
    return [str(q).strip() for q in sliced if str(q).strip()]


def _message_text(raw: Any) -> str:
    """Normalize different message types to plain text."""
    if isinstance(raw, AgentMessage):
        structured = raw.structured_output
        return str(structured.get("output_content") or raw.content or structured or "").strip()
    if isinstance(raw, AIMessage):
        return str(raw.content or "").strip()
    if isinstance(raw, dict):
        return str(raw.get("output_content") or raw.get("content") or "").strip()
    if raw is None:
        return ""
    return str(raw).strip()


def run_spider_agent(
    adapter: PromptAgentAdapter,
    instruction: str,
) -> Dict[str, Any]:
    """Execute the Spider-based agent and return runtime metadata."""
    start = time.perf_counter()
    datastore = DataStore()
    status = "ok"
    error: Optional[str] = None
    result: Dict[str, Any] = {}
    try:
        result = adapter.invoke(
            {
                "instruction": instruction,
                "datastore": datastore,
            }
        )
    except Exception as exc:  # pragma: no cover - runtime guardrail
        status = "error"
        error = f"{exc}"
    duration_ms = round((time.perf_counter() - start) * 1000, 1)
    final_msg = result.get("sql_agent_final_answer") if isinstance(result, dict) else None
    messages = result.get("messages") if isinstance(result, dict) else []
    return {
        "status": status,
        "duration_ms": duration_ms,
        "final_answer": _message_text(final_msg),
        "message_count": len(messages) if isinstance(messages, list) else 0,
        "error": error,
    }


def run_sql_agent(
    compiled_graph,
    instruction: str,
    datastore: Optional[DataStore] = None,
) -> Dict[str, Any]:
    """Execute the LangGraph SQL agent and capture execution metadata."""
    start = time.perf_counter()
    status = "ok"
    error: Optional[str] = None
    result: Dict[str, Any] = {}
    ds = datastore or DataStore()
    try:
        state: SQLState = {
            "messages": [],
            "instruction": instruction,
            "datastore": ds,
        }
        result = compiled_graph.invoke(state)
    except Exception as exc:  # pragma: no cover - runtime guardrail
        status = "error"
        error = f"{exc}"
    duration_ms = round((time.perf_counter() - start) * 1000, 1)
    final_msg = result.get("sql_agent_final_answer") if isinstance(result, dict) else None
    query_log = result.get("query_log") if isinstance(result, dict) else []
    sql_footprints: List[Dict[str, Any]] = []
    if isinstance(query_log, list):
        for entry in query_log:
            if not isinstance(entry, dict):
                continue
            sql = entry.get("sql_query")
            if not sql:
                continue
            sql_footprints.append(
                {
                    "sql": str(sql),
                    "status": entry.get("status"),
                    "row_count": entry.get("row_count"),
                    "error": entry.get("error_message"),
                }
            )
    return {
        "status": status,
        "duration_ms": duration_ms,
        "final_answer": _message_text(final_msg),
        "sql_queries": sql_footprints,
        "error": error,
    }


def benchmark(
    queries: List[str],
    provider: str,
    sql_model: str,
    spider_model: Optional[str],
) -> Dict[str, Any]:
    """Run both agents against provided queries and collect results."""
    if spider_model:
        os.environ["USE_MODEL"] = spider_model
    sql_llm = llm_from(
        provider,
        sql_model,
        agent_name="SQL Agent",
    ).with_structured_output(SQLAgentOutput)
    sql_agent = create_sql_agent(sql_llm)
    spider_agent = PromptAgentAdapter()

    runs: List[Dict[str, Any]] = []
    for idx, query in enumerate(queries, start=1):
        spider_result = run_spider_agent(spider_agent, query)
        sql_result = run_sql_agent(sql_agent, query)
        runs.append(
            {
                "query_id": idx,
                "instruction": query,
                "spider_agent": spider_result,
                "sql_agent": sql_result,
            }
        )
        print(
            f"[{idx}] Spider={spider_result['status']} ({spider_result['duration_ms']} ms) "
            f"| SQL={sql_result['status']} ({sql_result['duration_ms']} ms)"
        )
    return {
        "provider": provider,
        "sql_model": sql_model,
        "spider_model": spider_model or os.getenv("USE_MODEL"),
        "total_requests": len(queries),
        "runs": runs,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Spider and SQL agents on a set of natural language requests."
    )
    parser.add_argument(
        "--benchmark-file",
        type=Path,
        default=Path("benchmark.json"),
        help="Path to a JSON file containing a 'benchmark_queries' array.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of queries to run from the benchmark file.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index offset inside the benchmark file.",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=os.getenv("USE_PROVIDER", "aws"),
        help="LLM provider for the SQL agent (default: USE_PROVIDER env var).",
    )
    parser.add_argument(
        "--sql-model",
        type=str,
        default=os.getenv("USE_MODEL", "anthropic.claude-3-5-sonnet-20241022-v2:0"),
        help="Model identifier for the SQL agent.",
    )
    parser.add_argument(
        "--spider-model",
        type=str,
        default=None,
        help="Optional model identifier for the Spider agent (overrides USE_MODEL).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/agent_benchmark_results.json"),
        help="Where to write the benchmark results as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    queries = _load_queries(args.benchmark_file, args.limit, args.start)
    results = benchmark(
        queries=queries,
        provider=args.provider,
        sql_model=args.sql_model,
        spider_model=args.spider_model,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote results to {args.output}")


if __name__ == "__main__":
    main()
