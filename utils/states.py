from typing import TypedDict, List, Any, Annotated
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class OverallState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    datastore: dict
    database_schema: dict

class SQLState(TypedDict):
        """
        Représente l'état de notre graphe.

        Attributs:
            question: La question en langage naturel de l'utilisateur.
            db_schema: Le schéma de la base de données.
            sql_query: La requête SQL générée par le LLM.
            query_result: Le résultat de l'exécution de la requête SQL.
            description: Une description textuelle des données récupérées.
            reference_key: La clé unique pour accéder aux données dans le datastore.
            error_message: Un message d'erreur en cas de problème.
        """
        question: str
        db_schema: str
        sql_query: str
        query_result: List[Any]
        description: str
        answer: str #réponse à donner au superviseur
        reference_key: str
        error_message: str