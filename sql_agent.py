from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from llm import llm_from
import os
import sqlite3
import json
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser


from typing import TypedDict, List, Any
from typing import TypedDict, List, Any

class GraphState(TypedDict):
    """
    Représente l'état de notre graphe (mis à jour pour le datastore).

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
    reference_key: str
    error_message: str


# Pour cet exemple, nous utilisons une base de données SQLite en mémoire.
# Remplacez ceci par la connexion à votre base de données réelle.
conn = sqlite3.connect(':memory:')
cursor = conn.cursor()

# Créer des tables de démo
cursor.execute("""
CREATE TABLE sensors (
    id TEXT PRIMARY KEY,
    location TEXT NOT NULL,
    type TEXT
);
""")
cursor.execute("""
CREATE TABLE sensor_readings (
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    sensor_id TEXT,
    value REAL,
    FOREIGN KEY (sensor_id) REFERENCES sensors(id)
);
""")
conn.commit()

def get_db_schema(connexion):
    """Récupère le schéma de la base de données (instructions CREATE)."""
    schema = ""
    cursor = connexion.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    for table in tables:
        cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table[0]}'")
        schema += cursor.fetchone()[0] + "\n\n"
    return schema

# Test de la fonction
print(get_db_schema(conn))


def retrieve_schema_node(state: GraphState):
    """
    Nœud qui récupère le schéma de la base de données et l'ajoute à l'état.
    """
    print("--- 🧠 ÉTAPE : RÉCUPÉRATION DU SCHÉMA ---")
    state['db_schema'] = get_db_schema(conn)
    return state


def execute_sql_tool(connexion, sql_query: str):
    """
    Exécute une requête SQL et retourne le résultat ou une erreur.
    """
    try:
        cursor = connexion.cursor()
        cursor.execute(sql_query)
        result = cursor.fetchall()
        connexion.commit()
        return result
    except sqlite3.Error as e:
        # En cas d'erreur, nous retournons le message d'erreur.
        # Cela sera utile plus tard pour l'auto-correction.
        return f"Erreur d'exécution SQL: {e}"

# Initialisez votre modèle LLM
USE_MODEL = os.getenv("USE_MODEL", "mistral-ollama")
llm = llm_from(USE_MODEL)

def generate_sql_node(state: GraphState):
    """
    Nœud qui génère la requête SQL en utilisant le LLM.
    """
    print("--- 🧠 ÉTAPE : GÉNÉRATION DE LA REQUÊTE SQL ---")
    question = state['question']
    db_schema = state['db_schema']

    # Le prompt système est la clé du succès !
    system_prompt = f"""
    Vous êtes un expert en SQL. Votre unique tâche est de convertir une question en langage naturel en une requête SQL syntaxiquement correcte pour une base de données SQLite.

    Règles importantes :
    1.  Basez-vous **uniquement** sur le schéma de base de données fourni ci-dessous. N'inventez pas de colonnes ou de tables.
    2.  Ne retournez **que** le code SQL, sans aucune explication, phrase d'introduction ou formatage de type markdown. La sortie doit être directement exécutable.
    3.  Analysez attentivement la question pour comprendre les colonnes, les agrégations (COUNT, AVG, MAX), et les conditions (WHERE) nécessaires.

    Voici le schéma de la base de données :
    ---
    {db_schema}
    ---
    """

    user_prompt = f"Question: {question}"

    # Appel du LLM
    response = llm.invoke([
        ("system", system_prompt),
        ("human", user_prompt)
    ])
    
    sql_query = response.content.strip()
    
    # Nettoyage pour supprimer les ```sql et ``` potentiels
    if sql_query.startswith("```sql"):
        sql_query = sql_query[5:]
    if sql_query.endswith("```"):
        sql_query = sql_query[:-3]
    
    state['sql_query'] = sql_query.strip()
    return state

def execute_sql_node(state: GraphState):
    """
    Nœud qui exécute la requête SQL et stocke le résultat.
    """
    print("--- 🧠 ÉTAPE : EXÉCUTION DE LA REQUÊTE SQL ---")
    sql_query = state['sql_query']
    
    # Appel de notre outil d'exécution
    result = execute_sql_tool(conn, sql_query)
    
    if isinstance(result, str) and "Erreur" in result:
        # Si une erreur est retournée, on la stocke dans error_message
        print(f"--- 🚨 ERREUR D'EXÉCUTION ---")
        state['error_message'] = result
        state['query_result'] = []
    else:
        # Sinon, on stocke le résultat
        print(f"--- ✅ SUCCÈS DE L'EXÉCUTION ---")
        state['query_result'] = result
        state['error_message'] = None # On s'assure qu'il n'y a pas d'ancien message d'erreur
        
    return state

#### Création de la référerence et description ####
from pydantic import BaseModel, Field

class ReferenceOutput(BaseModel):
    """Un modèle pour contenir une clé de référence et sa description."""
    reference_key: str = Field(description="L'identifiant unique pour la référence, ex: releve_max_sensor_123")
    description: str = Field(description="Une description lisible de ce à quoi la référence se rapporte.")

def generate_summary_node(state: GraphState):
    """
    Nœud qui génère une description et une clé de référence
    en utilisant la sortie structurée du LLM.
    """
    print("--- 🧠 ÉTAPE : GÉNÉRATION DE LA DESCRIPTION ET DE LA CLÉ ---")
    
    # Configure le LLM pour qu'il retourne directement un objet Pydantic
    structured_llm = llm.with_structured_output(ReferenceOutput)
    question = state['question']
    
    # Le prompt peut être simplifié, car LangChain ajoute ses propres instructions
    # pour garantir le format de sortie.
    prompt = f"""
    Analyse la question suivante et extrais-en une clé de référence et une description.

    Question: "{question}"
    """
    
    try:
        # La réponse est déjà un objet Pydantic, pas besoin de parser !
        response = structured_llm.invoke(prompt)
        
        # On accède directement aux attributs de l'objet
        state['reference_key'] = response.reference_key
        state['description'] = response.description

    except Exception as e:
        # Gère les erreurs si le LLM ne parvient pas à se conformer au schéma
        print(f"--- 🚨 ERREUR : Le LLM n'a pas pu générer une sortie structurée valide. Erreur: {e} ---")
        state['reference_key'] = "erreur_generation_cle"
        state['description'] = "Erreur lors de la génération de la description."
        
    return state

def save_to_datastore_node(state: GraphState):
    """
    Nœud qui sauvegarde le résultat et sa description dans le datastore
    en utilisant la clé de référence.
    """
    print("--- 🧠 ÉTAPE : SAUVEGARDE DANS LE DATASTORE ---")
    reference_key = state['reference_key']
    description = state['description']
    data = state['query_result']
    
    if not reference_key or "erreur" in reference_key:
        print("--- ⚠️ AVERTISSEMENT : Clé invalide, sauvegarde annulée. ---")
        return state
        
    # Structure demandée : { "description": str, "data": resultat }
    DATASTORE[reference_key] = {
        "description": description,
        "data": data
    }
    print(f"--- ✅ Données sauvegardées sous la clé : '{reference_key}' ---")
    return state


#### Création du workflow ####

# Notre datastore en mémoire pour stocker les résultats
DATASTORE = {}

#### Création du workflow ####

# Créer une nouvelle instance de graphe
workflow = StateGraph(GraphState)

# Ajouter les nœuds au graphe
workflow.add_node("retrieve_schema", retrieve_schema_node)
workflow.add_node("generate_sql", generate_sql_node)
workflow.add_node("execute_sql",execute_sql_node)
workflow.add_node("generate_summary", generate_summary_node)
workflow.add_node("save_to_datastore", save_to_datastore_node)

# Définir le point de départ
workflow.set_entry_point("retrieve_schema")

# Définir les arêtes (le flux)
workflow.add_edge("retrieve_schema", "generate_sql")
workflow.add_edge("generate_sql","execute_sql")
workflow.add_edge("execute_sql", "generate_summary")
workflow.add_edge("generate_summary", "save_to_datastore")
workflow.add_edge("save_to_datastore", END) # Et on termine

# Compiler le graphe
sql_agent_mvp = workflow.compile()

# # Question de l'utilisateur
# user_question = "Quel est le relevé maximum pour le capteur 'sensor_123' ?"

# # Lancer le graphe avec l'entrée initiale
# initial_state = {"question": user_question}
# final_state = sql_agent_mvp.invoke(initial_state)

# # Afficher le résultat
# print("\n--- ✅ RÉSULTAT FINAL ---")
# print(f"Question : {final_state['question']}")
# print(f"Requête SQL générée :\n{final_state['sql_query']}")

# # Test avec une autre question
# user_question_2 = "Combien y a-t-il de capteurs au total ?"
# final_state_2 = sql_agent_mvp.invoke({"question": user_question_2})

# print("\n--- ✅ RÉSULTAT FINAL 2 ---")
# print(f"Question : {final_state_2['question']}")
# print(f"Requête SQL générée :\n{final_state_2['sql_query']}")

# (Assurez-vous que le code de l'étape 1 du MVP est présent : connexion DB, 
#  définition de GraphState, et les fonctions retrieve_schema_node et generate_sql_node)

# Question de l'utilisateur
# user_question = "Montre-moi tous les capteurs qui se trouvent à Paris. Donne-moi leur ID et leur type."

# # Pour rendre l'exemple intéressant, ajoutons une donnée
# cursor.execute("INSERT INTO sensors (id, location, type) VALUES ('sensor_789', 'Paris', 'Température');")
# conn.commit()

# # Lancer le graphe avec l'entrée initiale
# initial_state = {"question": user_question, "query_result": []} # Initialiser query_result
# final_state = sql_agent_mvp.invoke(initial_state)

# # Afficher le résultat complet
# print("\n--- ✅ RÉSULTAT FINAL COMPLET ---")
# print(f"Question : {final_state['question']}")
# print(f"Requête SQL générée :\n{final_state['sql_query']}")
# print(f"Résultat de la requête : {final_state['query_result']}")

# (Assurez-vous que le code des étapes précédentes est présent et exécuté)

# Question de l'utilisateur
user_question_3 = "Combien y a-t-il de relevés pour chaque capteur ?"

# Lancer le graphe
initial_state = {"question": user_question_3}
final_state = sql_agent_mvp.invoke(initial_state)

# Afficher le contenu du datastore pour vérifier
print("\n--- 🗄️ CONTENU DU DATASTORE ---")
import pprint
pprint.pprint(DATASTORE)