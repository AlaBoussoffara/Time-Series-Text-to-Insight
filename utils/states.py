from typing import Annotated, Any, Dict, List, Optional, TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class GlobalState(TypedDict):
    global_messages_history: Annotated[list[AnyMessage], add_messages]
    sql_agent_messages_history: Annotated[list[AnyMessage], add_messages]
    analysis_agent_messages_history: Annotated[list[AnyMessage], add_messages]
    visualization_agent_messages_history: Annotated[list[AnyMessage], add_messages]
    datastore: Any
    database_schema: dict



class SQLState(TypedDict, total=False):
    """
    Minimal controller state for the SQL agent.

    Attributes:
        messages: Running conversation history between controller and tools.
        instruction: Instruction received from the supervisor.
        query_log: Chronological record of executed SQL queries and their outcomes.
        datastore: Persistent datastore object shared across steps.
        sql_agent_final_answer: Text response prepared for the supervisor.
    """

    messages: Annotated[List[AnyMessage], add_messages]
    instruction: str
    query_log: List[Dict[str, Any]]
    datastore: Any
    sql_agent_final_answer: str



AnalysisDatastore = Dict[str, Dict[str, Any]]


class AnalysisState(TypedDict, total=False):
    """State passed through the analysis workflow."""

    instruction: str
    datastore: Any
    datastore_summary: str
    referenced_keys: List[str]
    analysis_agent_final_answer: str
    insights: List[str]
    follow_up_questions: List[str]
    error_message: Optional[str]
    datastore_obj: Any


class VisualizationState(TypedDict, total=False):
    """State passed through the visualization workflow."""

    instruction: str
    datastore: Any
    datastore_summary: str
    datastore_obj: Any
    chart_plan: Dict[str, Any]
    selected_dataset: str
    detected_columns: Dict[str, Any]
    output_path: str
    warnings: List[str]
    error_message: Optional[str]
    visualization_agent_final_answer: str
    generated_code: Optional[str]
