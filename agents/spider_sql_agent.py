from __future__ import annotations

from langchain_core.messages import AIMessage
from agents.spider_agent.agents import PromptAgent 
from utils.sql_utils import connect_postgres, execute_sql_tool
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage, AnyMessage
from langgraph.graph import END, START, StateGraph
import os
from utils.datastore import DATASTORE, DataStore
from utils.sql_utils import connect_postgres, execute_sql_tool
import subprocess
from utils.messages import AgentMessage



class PromptAgentAdapter:
    """
    Wraps the PromptAgent to make it compatible with LangGraph's .invoke()
    """
    def __init__(self):
        # Initialize the legacy agent with its preferred hyperparameters
        self.internal_agent = PromptAgent(
            model=os.getenv("USE_MODEL"),
            max_steps=15,
            use_plan=False
        )

    def invoke(self, state: dict) -> dict:
        """
        This signature matches LangGraph's requirements.
        Input: State Dictionary
        Output: State Dictionary updates
        """
        
        # 1. UNPACK: Get data from the Supervisor's state
        instruction = state.get("instruction", "")
        datastore = state.get("datastore")

        # 2. SETUP: Create the mock environment with current data
        # This binds the legacy agent to the current request
        query_log: List[Dict[str, Any]] = []
        mock_env = MockSpiderEnv(instruction, datastore, query_log=query_log)
        self.internal_agent.set_env_and_task(mock_env)

        # 3. EXECUTE: Run the legacy loop
        # We block here until the agent finishes its 'while' loop
        done = False
        try:
            done, final_result_string = self.internal_agent.run()
        except Exception as e:
            final_result_string = f"Spider Agent Crashed: {str(e)}"
        print(done)
        print(final_result_string)

        # 4. TRANSFORM HISTORY (Optional but recommended for comparison)
        # Convert legacy self.thoughts/self.actions into LangChain messages
        # so the Supervisor can see what happened.
        converted_messages = []
        for i in range(len(self.internal_agent.observations)):
             obs = self.internal_agent.observations[i]
             thought = self.internal_agent.thoughts[i]
             msg = AIMessage(content=f"Thought: {thought}\nObs: {obs}")
             print(msg)
             converted_messages.append(msg)

        # 5. REPACK: Return the exact keys the Supervisor expects
        if not converted_messages:
            # If no observations were generated (e.g. crash or immediate return),
            # ensure we have at least one message to return as the final answer.
            converted_messages.append(AIMessage(content=final_result_string or "No result generated."))
        return {
            "sql_agent_final_answer": converted_messages[-1],
            "messages": converted_messages, # The trace
            "datastore": datastore, # Pass back the datastore
            "query_log": query_log,
        }
        
class MockSpiderEnv:
    """
    This class simulates the 'Spider_Agent_Env' the legacy agent expects.
    It redirects actions to your current project's SQL utilities.
    """
    def __init__(self, instruction: str, datastore: any, query_log: Optional[List[Dict[str, Any]]] = None):
        # 1. Mimic the config structure the agent expects
        self.task_config = {
            'question': instruction,
            'type': 'Postgres' # Forces the agent to use Postgres_EXEC_SQL logic
        }
        self.datastore = datastore
        self.conn = connect_postgres() # Use your existing connection tool
        self.query_log = query_log if isinstance(query_log, list) else []

    def step(self, action):
        """
        The legacy agent calls this with an Action object.
        We execute it using our NEW tools and return the result string.
        """
        observation = ""
        done = False

        # 2. Intercept SQL Actions
        if type(action).__name__ == 'POSTGRES_EXEC_SQL': # or whatever the legacy action class is
            sql_query = action.sql_query # Extract SQL from the legacy action object
            if sql_query:
                print(f"[Spider Agent] execute_sql: {sql_query}")
            else:
                print("[Spider Agent] execute_sql: EMPTY QUERY")
            
            result_rows = None
            error_message = None
            try:
                # Delegate to your EXISTING sql_utils
                result_rows = execute_sql_tool(self.conn, sql_query)
                
                # Handle persistence if requested
                if getattr(action, 'is_save', False):
                    try:
                        df = pd.DataFrame(result_rows)
                        # Use save_path as reference key if provided, otherwise let datastore generate one
                        ref_key = action.save_path if action.save_path else None
                        saved_ref = self.datastore.put(
                            df, 
                            description=f"Result of query: {sql_query}",
                            ref=ref_key,
                            upsert=True
                        )
                        observation = f"Success. Rows returned: {len(result_rows)}. Data saved to datastore with ref: {saved_ref}"
                    except Exception as e:
                        observation = f"Success. Rows returned: {len(result_rows)}. Warning: Failed to save to datastore: {str(e)}"
                else:
                    observation = f"Success. Rows returned: {result_rows}"
            except Exception as e:
                error_message = str(e)
                observation = f"SQL Error: {error_message}"

            if sql_query:
                log_entry: Dict[str, Any] = {
                    "entry_type": "sql_result",
                    "sql_query": sql_query,
                }
                if error_message is not None:
                    log_entry.update(
                        {
                            "status": "error",
                            "row_count": 0,
                            "error_message": error_message,
                        }
                    )
                elif isinstance(result_rows, list):
                    log_entry.update(
                        {
                            "status": "success",
                            "row_count": len(result_rows),
                        }
                    )
                else:
                    log_entry.update(
                        {
                            "status": "error",
                            "row_count": 0,
                            "error_message": str(result_rows) if result_rows is not None else None,
                        }
                    )
                self.query_log.append(log_entry)

        # 3. Intercept Bash Actions
        elif type(action).__name__ == 'Bash':
            try:
                # Execute the bash command
                result = subprocess.run(action.code, shell=True, capture_output=True, text=True, timeout=30)
                observation = ""
                if result.stdout:
                    observation += f"Stdout: {result.stdout}"
                if result.stderr:
                    if observation:
                        observation += "\n"
                    observation += f"Stderr: {result.stderr}"
                if not observation:
                    observation = "Command executed successfully with no output."
            except Exception as e:
                observation = f"Bash Error: {str(e)}"

        # 3. Intercept Termination
        elif type(action).__name__ == 'Terminate':
            done = True
            observation = action.output # The final answer text

        # 4. Default/Fallbacks
        else:
            observation = f"Action {type(action).__name__} executed (simulated)"

        return observation, done
