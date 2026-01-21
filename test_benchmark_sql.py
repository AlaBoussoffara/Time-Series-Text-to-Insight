
import json
import csv
import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from agents.supervisor_agent import build_supervisor_graph
from utils.api_call_counter import (
    append_api_call_log,
    get_api_call_count,
    get_api_call_counts_by_agent,
    print_api_call_breakdown,
    reset_api_call_count,
)
from utils.token_counter import get_token_usage, print_token_usage, reset_token_usage
from utils.datastore import DATASTORE
from utils.messages import AgentMessage
from utils.general_helpers import stream_graph

def _load_queries(path: Path, limit: Optional[int], start: int) -> List[Dict[str, str]]:
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
    
    return sliced

def _message_to_dict(message: BaseMessage) -> Dict[str, Any]:
    name = getattr(message, "name", None) or getattr(message, "type", message.__class__.__name__)
    entry: Dict[str, Any] = {
        "name": name,
        "message_type": message.__class__.__name__,
    }
    content = getattr(message, "content", None)
    if content is not None:
        entry["content"] = content
    structured: Optional[Dict[str, Any]] = None
    if isinstance(message, AgentMessage):
        structured = message.structured_output
    else:
        raw_structured = getattr(message, "structured_output", None)
        if isinstance(raw_structured, dict):
            structured = dict(raw_structured)
    if structured is not None:
        entry["structured_output"] = structured
        output_type = structured.get("output_type")
        if output_type:
            entry["output_type"] = output_type
        if "output_content" in structured:
            entry["output_content"] = structured.get("output_content")
    return entry


def _append_json_result(path: Path, entry: Dict[str, Any]) -> None:
    existing: list = []
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                existing = payload
        except Exception:
            existing = []
    existing.append(entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(existing, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


# def extract_sql_query(history: List[BaseMessage]) -> str:
#     """Extracts the generated SQL query from the message history."""
#     for message in reversed(history):
#         if isinstance(message, AgentMessage) and message.name == "SQL Agent":
#             structured = message.structured_output

#     return "N/A"

def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit number of queries")
    parser.add_argument("--start", type=int, default=0, help="Start index")
    parser.add_argument("--output", type=str, default="benchmark_results.csv", help="Output CSV file")
    parser.add_argument("--output-json", type=str, default="benchmark_results.json", help="Output JSON file")
    args = parser.parse_args()

    benchmark_file = Path("benchmark_sql.json")
    queries_data = _load_queries(benchmark_file, args.limit, args.start)

    # Prepare CSV
    file_exists = os.path.isfile(args.output)
    
    with open(args.output, mode='a' if file_exists else 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'Question',
            'Benchmark SQL',
            'Generated SQL',
            'Generated SQLs',
            'Supervisor Answer',
            'SQL Prompt Tokens',
            'SQL Completion Tokens',
            'SQL Total Tokens',
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        compiled_graph = build_supervisor_graph()

        for i, item in enumerate(queries_data):
            question = item.get("question")
            benchmark_sql = item.get("sql")
            role = item.get("role")
            
            print(f"[{i+args.start+1}] Processing: {question}")
            request_label = f"API calls for request {i+args.start+1}"
            reset_api_call_count()
            reset_token_usage()
            DATASTORE.clear()
            
            try:
                SUPERVISOR_PROMPT_TEXT = Path("prompts/supervisor_prompt.txt").read_text(encoding="utf-8")
                messages = [SystemMessage(SUPERVISOR_PROMPT_TEXT), HumanMessage(question)]
                
                global_state = {
                    "global_messages_history": messages,
                    "datastore": DATASTORE, # Using the global one imported
                    "database_schema": {},
                }
                
                history = stream_graph(compiled_graph, global_state, log=True) 
                
                # Extract Answer
                supervisor_answer = ""
                final_msg = history[-2]
                if isinstance(final_msg, AgentMessage):
                     supervisor_answer = final_msg.structured_output.get("output_content", "")
                elif isinstance(final_msg, BaseMessage):
                     supervisor_answer = final_msg.content
                
                # Extract SQL
                # Trying to find it in the history
                generated_sql = "N/A"
                generated_sqls: List[str] = []
                for msg in history:
                     if isinstance(msg, AgentMessage) and msg.name == "SQL Agent":
                          # Look for a new field I put in, or just "N/A" for now
                          # I will update supervisor_agent.py next to include sql_queries in the output
                          queries = msg.structured_output.get("sql_queries", [])
                          if isinstance(queries, list) and queries:
                               generated_sqls.extend([str(q).strip() for q in queries if str(q).strip()])
                          elif "sql_query" in msg.structured_output:
                               sql_query = str(msg.structured_output["sql_query"]).strip()
                               if sql_query:
                                    generated_sqls.append(sql_query)
                
                if generated_sqls:
                     generated_sql = generated_sqls[-1]
                
                token_usage = get_token_usage()
                if args.output_json:
                    json_entry = {
                        "index": i + args.start + 1,
                        "role": role,
                        "question": question,
                        "benchmark_sql": benchmark_sql,
                        "generated_sql": generated_sql,
                        "generated_sql_all": generated_sqls,
                        "supervisor_answer": supervisor_answer,
                        "discussion": [_message_to_dict(msg) for msg in history],
                        "api_calls_for_request": {
                            "total": get_api_call_count(),
                            "by_agent": get_api_call_counts_by_agent(),
                        },
                        "sql_tokens": {
                            "prompt": token_usage.get("prompt_tokens", 0),
                            "completion": token_usage.get("completion_tokens", 0),
                            "total": token_usage.get("total_tokens", 0),
                        },
                    }
                    _append_json_result(Path(args.output_json), json_entry)
                writer.writerow({
                    'Question': question,
                    'Benchmark SQL': benchmark_sql,
                    'Generated SQL': generated_sql,
                    'Generated SQLs': ";; ".join(generated_sqls),
                    'Supervisor Answer': supervisor_answer,
                    'SQL Prompt Tokens': token_usage.get('prompt_tokens', 0),
                    'SQL Completion Tokens': token_usage.get('completion_tokens', 0),
                    'SQL Total Tokens': token_usage.get('total_tokens', 0),
                })
                csvfile.flush() # Ensure write
                
            except Exception as e:
                print(f"Error processing query: {e}")
                token_usage = get_token_usage()
                if args.output_json:
                    json_entry = {
                        "index": i + args.start + 1,
                        "role": role,
                        "question": question,
                        "benchmark_sql": benchmark_sql,
                        "generated_sql": "ERROR",
                        "generated_sql_all": [],
                        "supervisor_answer": str(e),
                        "discussion": [_message_to_dict(msg) for msg in history] if "history" in locals() else [],
                        "api_calls_for_request": {
                            "total": get_api_call_count(),
                            "by_agent": get_api_call_counts_by_agent(),
                        },
                        "sql_tokens": {
                            "prompt": token_usage.get("prompt_tokens", 0),
                            "completion": token_usage.get("completion_tokens", 0),
                            "total": token_usage.get("total_tokens", 0),
                        },
                    }
                    _append_json_result(Path(args.output_json), json_entry)
                writer.writerow({
                    'Question': question,
                    'Benchmark SQL': benchmark_sql,
                    'Generated SQL': "ERROR",
                    'Generated SQLs': "",
                    'Supervisor Answer': str(e),
                    'SQL Prompt Tokens': token_usage.get('prompt_tokens', 0),
                    'SQL Completion Tokens': token_usage.get('completion_tokens', 0),
                    'SQL Total Tokens': token_usage.get('total_tokens', 0),
                })
            finally:
                print_api_call_breakdown(request_label)
                append_api_call_log(request_label)
                token_label = (
                    f"Spider tokens for request {i+args.start+1}"
                    if os.getenv("SQL_AGENT_MODE") == "SPIDER"
                    else f"SQL tokens for request {i+args.start+1}"
                )
                print_token_usage(token_label)

if __name__ == "__main__":
    main()
