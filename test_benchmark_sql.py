
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
    args = parser.parse_args()

    benchmark_file = Path("benchmark_sql.json")
    queries_data = _load_queries(benchmark_file, args.limit, args.start)

    # Prepare CSV
    file_exists = os.path.isfile(args.output)
    
    with open(args.output, mode='a' if file_exists else 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Question', 'Benchmark SQL', 'Generated SQL', 'Supervisor Answer']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        compiled_graph = build_supervisor_graph()

        for i, item in enumerate(queries_data):
            question = item.get("question")
            benchmark_sql = item.get("sql")
            
            print(f"[{i+args.start+1}] Processing: {question}")
            
            try:
                # Reset Datastore? usage in supervisor_agent uses global DATASTORE or passes it.
                # `run_supervisor` creates a new state.
                SUPERVISOR_PROMPT_TEXT = Path("prompts/supervisor_prompt.txt").read_text(encoding="utf-8")
                messages = [SystemMessage(SUPERVISOR_PROMPT_TEXT), HumanMessage(question)]
                
                global_state = {
                    "global_messages_history": messages,
                    "datastore": DATASTORE, # Using the global one imported
                    "database_schema": {},
                }
                
                history = stream_graph(compiled_graph, global_state, log=False) # log=False to keep stdout clean
                
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
                for msg in history:
                     if isinstance(msg, AgentMessage) and msg.name == "SQL Agent":
                          # Look for a new field I put in, or just "N/A" for now
                          # I will update supervisor_agent.py next to include sql_queries in the output
                          queries = msg.structured_output.get("sql_queries", [])
                          if queries:
                               generated_sql = "; ".join(queries)
                          elif "sql_query" in msg.structured_output:
                               generated_sql = msg.structured_output["sql_query"]
                
                writer.writerow({
                    'Question': question,
                    'Benchmark SQL': benchmark_sql,
                    'Generated SQL': generated_sql,
                    'Supervisor Answer': supervisor_answer
                })
                csvfile.flush() # Ensure write
                
            except Exception as e:
                print(f"Error processing query: {e}")
                writer.writerow({
                    'Question': question,
                    'Benchmark SQL': benchmark_sql,
                    'Generated SQL': "ERROR",
                    'Supervisor Answer': str(e)
                })

if __name__ == "__main__":
    main()
