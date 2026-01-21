from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _coerce_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _load_entries(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [entry for entry in payload if isinstance(entry, dict)]
    raise ValueError(f"Expected a JSON list in {path}")


def _summarize(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    totals_by_agent: Dict[str, int] = {}
    total_calls = 0
    entries_with_calls = 0

    for entry in entries:
        api_calls = entry.get("api_calls_for_request")
        if not isinstance(api_calls, dict):
            continue
        entries_with_calls += 1
        total = _coerce_int(api_calls.get("total"))
        total_calls += total

        by_agent = api_calls.get("by_agent")
        sum_agents = 0
        if isinstance(by_agent, dict):
            for agent, count in by_agent.items():
                count_int = _coerce_int(count)
                totals_by_agent[agent] = totals_by_agent.get(agent, 0) + count_int
                sum_agents += count_int

        if total > sum_agents:
            totals_by_agent["Unattributed"] = totals_by_agent.get("Unattributed", 0) + (
                total - sum_agents
            )

    avg_by_agent = {
        agent: (total / entries_with_calls if entries_with_calls else 0.0)
        for agent, total in totals_by_agent.items()
    }
    avg_total = total_calls / entries_with_calls if entries_with_calls else 0.0

    return {
        "entries_total": len(entries),
        "entries_with_calls": entries_with_calls,
        "total_calls": total_calls,
        "total_by_agent": totals_by_agent,
        "avg_total": avg_total,
        "avg_by_agent": avg_by_agent,
    }


def _print_summary(label: str, path: Path, summary: Dict[str, Any]) -> None:
    print(f"{label} ({path})")
    print(
        f"Requests with api_calls_for_request: "
        f"{summary['entries_with_calls']} / {summary['entries_total']}"
    )
    print(f"Avg total API calls/request: {summary['avg_total']:.2f}")
    print("Avg calls by agent:")
    for agent in sorted(summary["avg_by_agent"]):
        avg = summary["avg_by_agent"][agent]
        print(f"  {agent}: {avg:.2f}")


def _print_comparison(custom: Dict[str, Any], spider: Dict[str, Any]) -> None:
    agents = set(custom["avg_by_agent"]).union(spider["avg_by_agent"])
    if not agents:
        print("No agent API calls found to compare.")
        return
    print("Comparison (avg API calls per request)")
    for agent in sorted(agents):
        custom_avg = custom["avg_by_agent"].get(agent, 0.0)
        spider_avg = spider["avg_by_agent"].get(agent, 0.0)
        delta = custom_avg - spider_avg
        print(
            f"  {agent}: custom={custom_avg:.2f} spider={spider_avg:.2f} delta={delta:+.2f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize average API calls per agent for benchmark results."
    )
    parser.add_argument(
        "--custom",
        type=Path,
        default=Path("benchmark_results_CUSTOM.json"),
        help="Path to the custom-agent benchmark results JSON.",
    )
    parser.add_argument(
        "--spider",
        type=Path,
        default=Path("benchmark_results_SPIDER.json"),
        help="Path to the spider-agent benchmark results JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON file to write the summaries.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    custom_entries = _load_entries(args.custom)
    spider_entries = _load_entries(args.spider)

    custom_summary = _summarize(custom_entries)
    spider_summary = _summarize(spider_entries)

    _print_summary("CUSTOM", args.custom, custom_summary)
    print()
    _print_summary("SPIDER", args.spider, spider_summary)
    print()
    _print_comparison(custom_summary, spider_summary)

    if args.output:
        payload = {
            "custom": custom_summary,
            "spider": spider_summary,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
