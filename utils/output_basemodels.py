from typing import Literal
from pydantic import BaseModel, Field


class SupervisorOutput(BaseModel):
    output: Literal["plan", "thought", "final_answer", "SQL Agent", "Analysis Agent", "Visualization Agent"] = Field(
        ..., description="Either 'plan', 'thought', 'final_answer', or an agent name."
    )
    content: str = Field(..., description="content for the chosen output.")
class SQLAgentOutput(BaseModel):
        """Un modèle pour contenir une clé de référence et sa description."""
        reference_key: str = Field(description="L'identifiant unique pour la référence, ex: releve_max_sensor_123")
        description: str = Field(description="Une description lisible de ce à quoi la référence se rapporte.")
        answer : str = Field(description="Une phrase qui commence par 'Tâche réussie' et qui répond à la question posée à l'aide du résultat de la requête")