from typing import Literal
from pydantic import BaseModel, Field


class SupervisorOutput(BaseModel):
    output: Literal[
        "plan",
        "thought",
        "final_answer",
        "hallucination",
        "no_hallucination",
        "SQL Agent",
        "Analysis Agent",
        "Visualization Agent",
    ] = Field(
        ...,
        description="Supervisor control signal: planning/thinking steps, agent names, 'final_answer', or hallucination audit outputs.",
    )
    content: str = Field(..., description="Content corresponding to the chosen output.")


class SQLAgentOutput(BaseModel):
    """Structured response from the SQL agent."""
    sql_query: str = Field(description="Executable SQL query (no markdown fences or commentary) that answers the question.")
    reference_key: str = Field(description="Snake_case identifier used to store the query result, e.g. sensor_summary_001.")
    description: str = Field(description="Readable description of what the stored data represents in the datastore.")
    answer: str = Field(description="English summary that enumerates the actions taken (schema review, SQL generation, execution, datastore save) and specifies the data provided to the supervisor.")
