from typing import List, Literal
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
    """
    Structured response for the minimal SQL controller loop.
    """

    output: Literal[
        "plan",
        "thought",
        "execute_sql",
        "persist_dataset",
        "final_answer",
        "hallucination",
        "no_hallucination",
    ] = Field(
        ...,
        description="Control signal emitted by the SQL controller.",
    )
    content: str = Field(..., description="Reasoning or instruction associated with the current step.")
    sql_query: str | None = Field(
        default=None,
        description="SQL to run when output == 'execute_sql'.",
    )
    reference_key: str | None = Field(
        default=None,
        description="Datastore key to use when output == 'persist_dataset'.",
    )
    description: str | None = Field(
        default=None,
        description="Datastore description to use when output == 'persist_dataset'.",
    )


class AnalysisAgentOutput(BaseModel):
    """Structured response produced by the analysis agent."""

    answer: str = Field(
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
