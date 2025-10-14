from langgraph.graph import StateGraph, END, START
from utils.sql_utils import create_test_database, get_db_schema, execute_sql_tool
from utils.states import SQLState
from utils.output_basemodels import SQLAgentOutput
from pathlib import Path

def create_sql_agent(llm):

    def retrieve_schema_node(state: SQLState):
        """
        Nœud qui récupère le schéma de la base de données et l'ajoute à l'état.
        """
        print("--- 🧠 ÉTAPE : RÉCUPÉRATION DU SCHÉMA ---")
        state['db_schema'] = get_db_schema(conn)
        return state
  
    def generate_sql_node(state: SQLState):
        """
        Nœud qui génère la requête SQL en utilisant le LLM.
        """
        print("--- 🧠 ÉTAPE : GÉNÉRATION DE LA REQUÊTE SQL ---")
        question = state['question']
        db_schema = state['db_schema']
        PROMPT_PATH = Path("prompts/sql_agent_prompt.txt")
        SQLAGENTPROMPT = PROMPT_PATH.read_text(encoding="utf-8")

        # Le prompt système est la clé du succès !
        system_prompt = SQLAGENTPROMPT.format(db_schema=db_schema)
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

    def execute_sql_node(state: SQLState):
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

    #### Création de la référerence et description ###
    def generate_summary_node(state: SQLState):
        """
        Nœud qui génère une description et une clé de référence
        en utilisant la sortie structurée du LLM.
        """
        print("--- 🧠 ÉTAPE : GÉNÉRATION DE LA DESCRIPTION ET DE LA CLÉ ---")
        
        # Configure le LLM pour qu'il retourne directement un objet Pydantic
        structured_llm = llm.with_structured_output(SQLAgentOutput)
        question = state['question']
        
        # Le prompt peut être simplifié, car LangChain ajoute ses propres instructions
        # pour garantir le format de sortie.
        prompt = f"""
        Analyse la question suivante et extrais-en une clé de référence, une description et une réponse chiffrée à la question avec les données obtenues.

        Question: "{question}"
        """
        
        try:
            # La réponse est déjà un objet Pydantic, pas besoin de parser !
            response = structured_llm.invoke(prompt)
            
            # On accède directement aux attributs de l'objet
            state['reference_key'] = response.reference_key
            state['description'] = response.description
            state['answer'] = response.answer
        except Exception as e:
            # Gère les erreurs si le LLM ne parvient pas à se conformer au schéma
            print(f"--- 🚨 ERREUR : Le LLM n'a pas pu générer une sortie structurée valide. Erreur: {e} ---")
            state['reference_key'] = "erreur_generation_cle"
            state['description'] = "Erreur lors de la génération de la description."
            
        return state

    def save_to_datastore_node(state: SQLState):
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
    DATASTORE={}  # Notre datastore en mémoire simple
    conn = create_test_database()


    # Créer une nouvelle instance de graphe
    workflow = StateGraph(SQLState)

    # Ajouter les nœuds au graphe
    workflow.add_node("retrieve_schema", retrieve_schema_node)
    workflow.add_node("generate_sql", generate_sql_node)
    workflow.add_node("execute_sql",execute_sql_node)
    workflow.add_node("generate_summary", generate_summary_node)
    workflow.add_node("save_to_datastore", save_to_datastore_node)



    # Définir les arêtes (le flux)
    workflow.add_edge(START, "retrieve_schema")
    workflow.add_edge("retrieve_schema", "generate_sql")
    workflow.add_edge("generate_sql","execute_sql")
    workflow.add_edge("execute_sql", "generate_summary")
    workflow.add_edge("generate_summary", "save_to_datastore")
    workflow.add_edge("save_to_datastore", END) # Et on termine

    # Compiler le graphe
    sql_agent = workflow.compile()

    return sql_agent