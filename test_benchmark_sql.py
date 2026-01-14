import json
import os
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from agents.supervisor_agent import build_supervisor_graph
from utils.datastore import DATASTORE
from utils.messages import AgentMessage
from utils.general_helpers import stream_graph
from utils.llm_judge import BenchmarkJudge

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

def run_agent_for_query(question: str, mode: str) -> Dict[str, Any]:
    """
    Runs the supervisor agent for a given question and mode (SPIDER or CUSTOM).
    Returns a dictionary containing the Answer, Generated SQL, and any error info.
    """
    # Set the environment variable for the agent mode
    os.environ["SQL_AGENT_MODE"] = mode
    
    # Re-build graph to pick up the env var change
    compiled_graph = build_supervisor_graph()

    SUPERVISOR_PROMPT_TEXT = Path("prompts/supervisor_prompt.txt").read_text(encoding="utf-8")
    messages = [SystemMessage(SUPERVISOR_PROMPT_TEXT), HumanMessage(question)]
    
    # Create valid global state structure
    global_state = {
        "global_messages_history": messages,
        "datastore": DATASTORE, 
        "database_schema": {},
        # Ensure sub-agent histories are initialized if the graph expects them,
        # though usually they are optional or initialized by the nodes.
        "sql_agent_messages_history": [], 
        "analysis_agent_messages_history": [],
        "visualization_agent_messages_history": []
    }
    
    try:
        # Run the graph
        history = stream_graph(compiled_graph, global_state, log=False)
        
        # Extract Supervisor Answer
        supervisor_answer = "N/A"
        if history:
            final_msg = history[-2] if len(history) >= 2 else history[-1]
            if isinstance(final_msg, AgentMessage):
                supervisor_answer = final_msg.structured_output.get("output_content", "")
            elif isinstance(final_msg, BaseMessage):
                supervisor_answer = getattr(final_msg, 'content', str(final_msg))

        # Extract "Last" Generated SQL
        generated_sql = "N/A"
        # We look through history for SQL Agent messages
        # We want the LAST one that has a valid SQL query
        for msg in history:
            if isinstance(msg, AgentMessage) and msg.name == "SQL Agent":
                structured = msg.structured_output
                queries = structured.get("sql_queries", [])
                if queries:
                    # Take the last executed query from this step
                    generated_sql = queries[-1]
                elif "sql_query" in structured:
                    generated_sql = structured["sql_query"]
        
        return {
            "answer": supervisor_answer,
            "sql": generated_sql,
            "error": None
        }

    except Exception as e:
        return {
            "answer": str(e),
            "sql": "ERROR",
            "error": str(e)
        }

def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit number of queries")
    parser.add_argument("--start", type=int, default=0, help="Start index")
    parser.add_argument("--output", type=str, default="benchmark_results.xlsx", help="Output Excel file")
    args = parser.parse_args()

    benchmark_file = Path("benchmark_sql.json")
    queries_data = _load_queries(benchmark_file, args.limit, args.start)

    # Initialize Judge
    judge = BenchmarkJudge()
    
    results = []
    
    # We will save incrementally to avoid data loss, but since we need to write a valid XLSX,
    # we'll write the full dataframe each time.
    
    for i, item in enumerate(queries_data):
        question = item.get("question")
        benchmark_sql = item.get("sql")
        
        print(f"[{i+args.start+1}/{len(queries_data)+args.start}] Processing: {question}")
        
        # --- Run Spider Agent ---
        print("   Running Spider Agent...")
        spider_res = run_agent_for_query(question, mode="SPIDER")
        
        # --- Run Custom Agent ---
        print("   Running Custom Agent...")
        custom_res = run_agent_for_query(question, mode="CUSTOM") # Or whatever default mode is
        
        # --- Judging ---
        print("   Judging Spider Results...")
        spider_sql_judge = judge.judge_sql(question, benchmark_sql, spider_res["sql"])
        spider_ans_judge = judge.judge_answer(question, benchmark_sql, spider_res["answer"])
        
        print("   Judging Custom Results...")
        custom_sql_judge = judge.judge_sql(question, benchmark_sql, custom_res["sql"])
        custom_ans_judge = judge.judge_answer(question, benchmark_sql, custom_res["answer"])
        
        # Collect Data
        row = {
            "Question": question,
            "Benchmark SQL": benchmark_sql,
            
            # Spider Columns
            "Spider Generated SQL": spider_res["sql"],
            "Spider Supervisor Answer": spider_res["answer"],
            "Spider SQL Score": spider_sql_judge.score,
            "Spider SQL Reasoning": spider_sql_judge.reasoning,
            "Spider Answer Score": spider_ans_judge.score,
            "Spider Answer Reasoning": spider_ans_judge.reasoning,
            
            # Custom Columns
            "Custom Generated SQL": custom_res["sql"],
            "Custom Supervisor Answer": custom_res["answer"],
            "Custom SQL Score": custom_sql_judge.score,
            "Custom SQL Reasoning": custom_sql_judge.reasoning,
            "Custom Answer Score": custom_ans_judge.score,
            "Custom Answer Reasoning": custom_ans_judge.reasoning
        }
        
        results.append(row)
        
        # Save progress
        df = pd.DataFrame(results)
        if args.output.endswith('.csv'):
            df.to_csv(args.output, index=False)
        else:
            df.to_excel(args.output, index=False)
        print(f"   Saved progress to {args.output}")

    print("Benchmark complete!")

if __name__ == "__main__":
    main()
