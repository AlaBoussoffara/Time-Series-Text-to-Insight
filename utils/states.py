from typing import TypedDict, List, Dict, Any, Annotated
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class OverallState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    datastore: dict
    database_schema: dict

class SQLState(TypedDict, total=False):
    """
    Minimal controller state for the SQL agent.

    Attributes:
        messages: Running conversation history between controller and tools.
        command: Instruction received from the supervisor.
        sql_queries: Ordered list of every SQL query executed (exploratory and final).
        datastore: Persistent datastore object shared across steps.
        latest_payload: Result of the most recent SQL execution (rows and metadata).
        datastore_updates: Cached view of persisted datasets keyed by reference.
        last_persist: Details about the most recent persistence action.
        final_answer: Text response prepared for the supervisor.
    """

    messages: Annotated[List[AnyMessage], add_messages]
    command: str
    sql_queries: List[str]
    datastore: Any
    latest_payload: Dict[str, Any]
    datastore_updates: Dict[str, Any]
    last_persist: Dict[str, Any]
    final_answer: str
    answer: str
    reference_key: str
    description: str
    query_result: List[Any]
    error_message: str
