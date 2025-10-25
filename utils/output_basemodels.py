from typing import List, Literal
from pydantic import BaseModel, Field


class SupervisorOutput(BaseModel):
    output_type: Literal[
        "plan",
        "thought",
        "supervisor_final_answer",
        "hallucination",
        "no_hallucination",
        "SQL Agent",
        "Analysis Agent",
        "Visualization Agent",
    ] = Field(
        ...,
        description="Supervisor control signal: planning/thinking steps, agent names, 'supervisor_final_answer', or hallucination audit outputs.",
    )
    output_content: str = Field(..., description="Content corresponding to the chosen output.")


class SQLAgentOutput(BaseModel):
    """
    Structured response for the minimal SQL controller loop.
    """

    output_type: Literal[
        "plan",
        "thought",
        "summarize_datastore_updates",
        "execute_sql",
        "persist_dataset",
        "sql_agent_final_answer",
        "hallucination",
        "no_hallucination",
    ] = Field(
        ...,
        description="Control signal emitted by the SQL controller.",
    )
    output_content: str = Field(..., description="Reasoning or instruction associated with the current step.")
    sql_query: str | None = Field(
        default=None,
        description="SQL to run when output_type == 'execute_sql'.",
    )
    reference_key: str | None = Field(
        default=None,
        description="Datastore key to use when output_type == 'persist_dataset'.",
    )
    description: str | None = Field(
        default=None,
        description="Datastore description to use when output_type == 'persist_dataset'.",
    )


class AnalysisAgentOutput(BaseModel):
    """Structured response produced by the analysis agent."""

    analysis_agent_final_answer: str = Field(
        ...,
        description="Narrative summary that references datastore keys, row counts, and caveats.",
    )
    insights: List[str] = Field(
        default_factory=list,
        description="Bullet-style insights derived from the provided data.",
    )
    follow_up_questions: List[str] = Field(
        default_factory=list,
        description="Suggested next questions or actions for the supervisor.",
    )
    referenced_keys: List[str] = Field(
        default_factory=list,
        description="Datastore keys that were essential for the analysis.",
    )

__all__ = ["SupervisorOutput", "SQLAgentOutput", "AnalysisAgentOutput"]
