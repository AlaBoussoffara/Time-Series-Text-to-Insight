from pathlib import Path

from langgraph.graph import END, START, StateGraph

from utils.output_basemodels import SQLAgentOutput
from utils.sql_utils import connect_postgres, execute_sql_tool
from utils.states import SQLState

PROMPT_PATH = Path("prompts/sql_agent_prompt.txt")
SQL_AGENT_PROMPT_TEXT = PROMPT_PATH.read_text(encoding="utf-8")
DATABASE_PROMPT_PATH = Path("prompts/database_prompt.txt")
DATABASE_CONTEXT = DATABASE_PROMPT_PATH.read_text(encoding="utf-8")


def create_sql_agent(llm):

    def generate_sql_and_summary_node(state: SQLState):
        """Generate the SQL query plus reference, description, and summary in one pass."""
        print("--- 🧠 STEP: Generate SQL and summary ---")
        question = state['question']

        system_prompt = SQL_AGENT_PROMPT_TEXT.replace("{database_context}", DATABASE_CONTEXT)
        structured_llm = llm.with_structured_output(SQLAgentOutput)
        try:
            response = structured_llm.invoke([
                ("system", system_prompt),
                ("human", f"Question: {question}")
            ])
        except Exception as exc:
            print(f"--- 🚨 Failed to generate structured output: {exc} ---")
            state['sql_query'] = ""
            state['reference_key'] = ""
            state['description'] = "Structured generation failed."
            state['answer'] = "SQL agent could not generate a query."
            return state

        sql_query = response.sql_query.strip()
        if sql_query.startswith("```sql"):
            sql_query = sql_query[5:]
        if sql_query.endswith("```"):
            sql_query = sql_query[:-3]

        state['sql_query'] = sql_query.strip()
        state['reference_key'] = response.reference_key.strip()
        state['description'] = response.description.strip()
        state['answer'] = response.answer.strip()
        return state

    def execute_sql_node(state: SQLState):
        """Execute the SQL query and capture the result or error."""
        print("--- 🧠 STEP: Execute SQL query ---")
        sql_query = state['sql_query']

        if not sql_query:
            print("--- ⚠️ WARNING: No SQL query was generated. Skipping execution. ---")
            state['error_message'] = "No SQL query generated."
            state['query_result'] = []
            return state
        
        result = execute_sql_tool(conn, sql_query)
        
        if isinstance(result, str):
            print("--- 🚨 SQL execution error ---")
            state['error_message'] = result
            state['query_result'] = []
        else:
            print("--- ✅ SQL execution succeeded ---")
            state['query_result'] = result
            state['error_message'] = None
            
        return state

    def save_to_datastore_node(state: SQLState):
        """Persist the result and description in the in-memory datastore."""
        print("--- 🧠 STEP: Save result to datastore ---")
        reference_key = state['reference_key']
        description = state['description']
        data = state['query_result']
        
        if not reference_key:
            print("--- ⚠️ WARNING: Missing reference key, skipping datastore save. ---")
            return state
            
        DATASTORE[reference_key] = {
            "description": description,
            "data": data
        }
        print(f"--- ✅ Saved data under key: '{reference_key}' ---")
        return state


    DATASTORE={}
    try:
        conn = connect_postgres()
    except Exception as exc:
        raise RuntimeError(
            "Failed to connect to PostgreSQL. Ensure POSTGRES_DSN is set and reachable."
        ) from exc


    workflow = StateGraph(SQLState)

    workflow.add_node("generate_sql_and_summary", generate_sql_and_summary_node)
    workflow.add_node("execute_sql",execute_sql_node)
    workflow.add_node("save_to_datastore", save_to_datastore_node)

    workflow.add_edge(START, "generate_sql_and_summary")
    workflow.add_edge("generate_sql_and_summary","execute_sql")
    workflow.add_edge("execute_sql", "save_to_datastore")
    workflow.add_edge("save_to_datastore", END)

    sql_agent = workflow.compile()

    return sql_agent
