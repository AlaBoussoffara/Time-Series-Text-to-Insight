import pandas as pd
import json
from typing import List, Optional, Dict, Any
from tqdm import tqdm
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage
from utils.general_helpers import llm_from

# --- 1. IMPORT DU NOUVEAU SUPERVISEUR ---
# Assurez-vous que le code du superviseur fourni est bien dans agents/supervisor_agent.py
# et que les utils sont accessibles.
from agents.supervisor_agent import run_supervisor

# --- 2. CONFIGURATION DU JUGE (LLM) ---
# Le juge reste le même, mais son prompt sera adapté au nouveau format de sortie
JUDGE_LLM = llm_from("aws", "anthropic.claude-3-5-sonnet-20241022-v2:0")

# Structure de notation détaillée
class DetailedGrade(BaseModel):
    expected: str = Field(description="Description de ce que la consigne exigeait (type de graph, données, filtres).")
    actual: str = Field(description="Description de ce que l'agent a produit (basé sur le résumé et le chemin du fichier).")
    score: int = Field(description="1 si le résultat semble pertinent et généré avec succès, 0 sinon.")
    reasoning: str = Field(description="Explication du verdict.")

# --- 3. LE BENCHMARK (Inchangé) ---
VISUALIZATION_BENCHMARK = [
    {"id": 1, "level": "easy", "query": "Génère une série de boxplots (6 subplots) incluant : basic, notched, outliers modifiés, horizontal, et longueur de moustaches personnalisée."},
    {"id": 2, "level": "easy", "query": "Crée une grille 3x3 de subplots partageant les axes X (colonne) et Y (ligne), affichant différentes fonctions (z, z^2, z^3) avec des couleurs distinctes."},
    {"id": 3, "level": "easy", "query": "Trace un Violin plot comparatif : un tracé par défaut à gauche et un personnalisé (sans moyenne/médiane, corps bleu, bords noirs) à droite."},
    {"id": 4, "level": "easy", "query": "Affiche un Scatter plot avec 3 ellipses de confiance superposées (1 sigma, 2 sigma, 3 sigma) et marqueur rouge au centre (1,1)."},
    {"id": 5, "level": "easy", "query": "Crée une figure combinant un Pie chart (répartition fruits) relié par des lignes à un Stacked Bar chart (détail par âge) sur le côté."},
    {"id": 6, "level": "easy", "query": "Génère un 'Nested Pie Plot' (camembert imbriqué) utilisant la méthode bar en coordonnées polaires pour visualiser deux niveaux de données."},
    {"id": 7, "level": "easy", "query": "Fais un Scatter plot en projection polaire où la surface des points est proportionnelle au carré de la distance radiale."},
    {"id": 8, "level": "medium", "query": "Crée un graphique avec 4 histogrammes de distributions Beta dans un style 'bmh', disposés proprement sur une même figure."},
    {"id": 9, "level": "medium", "query": "Trace la courbe (z-4)*(z-6)*(z-8)+90 avec une zone ombrée (fill_between) entre x=3 et x=10, et ajoute la formule en annotation."},
    {"id": 10, "level": "medium", "query": "Anatomie d'une figure : Crée un plot complexe avec grille personnalisée, tiques mineures/majeures, et annotations explicatives (flèches, cercles) sur les éléments."},
    {"id": 11, "level": "medium", "query": "Génère un 'Packed Bubble Chart' où la taille des bulles représente la popularité de langages de programmation (Python, Java, C++, etc.)."},
    {"id": 12, "level": "medium", "query": "Crée un graphique en barres 3D avec 4 ensembles de données (couleurs violet, orange, gris, rose) sur le plan y=k avec transparence."},
    {"id": 13, "level": "medium", "query": "Affiche une Heatmap des températures mensuelles par ville avec annotations des valeurs et un code couleur divergent (bleu-rouge)."},
    {"id": 14, "level": "medium", "query": "Trace deux courbes sur le même graphe utilisant deux axes Y différents (twinx) : une exponentielle à gauche et un sinus à droite."},
    {"id": 15, "level": "hard", "query": "Génère un plot 3D de voxels représentant 3 cuboïdes distincts (jaune, bleu, vert) reliés par des liens voxels violets."},
    {"id": 16, "level": "hard", "query": "Implémente une projection géographique personnalisée (Aitoff-Hammer) via une classe Python et affiche une grille sur ce globe."},
    {"id": 17, "level": "hard", "query": "Crée un Diagramme de Sankey complexe ('Flow Diagram') avec des boucles et chaînes latérales, titré 'This might seem unnecessary'."},
    {"id": 18, "level": "hard", "query": "Visualise un attracteur de Rossler en 3D (trajectoire continue) calculé via équations différentielles."},
    {"id": 19, "level": "hard", "query": "Génère deux diagrammes ternaires (phase liquide-liquide) : un en triangle équilatéral et l'autre en triangle rectangle, côte à côte."},
    {"id": 20, "level": "hard", "query": "Crée un graphique à 'axes brisés' (broken axis) avec des marqueurs de coupure diagonaux pour visualiser des outliers lointains."}
]

# --- 4. FONCTION D'ÉVALUATION ADAPTÉE AU NOUVEAU FORMAT ---
def evaluate_response_detailed(query: str, agent_result: Dict[str, Any]) -> DetailedGrade:
    """
    Évalue la réponse basée sur le dictionnaire 'visualizations' retourné par le nouveau superviseur.
    """
    system_prompt = """Tu es un expert en Data Visualization. Tu agis en tant que juge impartial.
    TACHE : Comparer la demande utilisateur avec le résultat de l'exécution d'un Agent de Visualisation.
    
    CONTEXTE : L'agent génère du code Python utilisant Plotly Express.
    Tu dois juger la qualité et la pertinence du code généré par rapport à la demande.
    Le fichier graphique (chart_path) est une conséquence du code, mais c'est le CODE qui est le plus important ici.
    
    INSTRUCTIONS :
    1. Analyse la DEMANDE.
    2. Analyse le CODE GÉNÉRÉ (s'il existe).
    3. Si le code semble correct pour répondre à la demande (bon type de graph, bonnes variables, bons filtres), note 1.
    4. Si le code est absent, incohérent, ou utilise le mauvais type de graphique, note 0.
    
    Note : Sois indulgent sur les détails stylistiques (couleurs précises non supportées par l'agent simple), 
    mais strict sur la logique de visualisation (ex: Pie chart vs Bar chart)."""
    
    # Construction d'une représentation textuelle de l'objet résultat pour le LLM
    output_description = f"""
    --- RÉSULTAT DE L'AGENT ---
    Success: {not bool(agent_result.get('error_message'))}
    Chart Path: {agent_result.get('chart_path', 'N/A')}
    Generated Code: {agent_result.get('generated_code', 'N/A')}
    Summary: {agent_result.get('summary', 'N/A')}
    Warnings: {agent_result.get('warnings', [])}
    Error: {agent_result.get('error_message', 'None')}
    """
    
    user_prompt_template = """
    --- DEMANDE UTILISATEUR ---
    "{query}"
    
    {output_description}
    """
    
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("user", user_prompt_template)])
    return (prompt | JUDGE_LLM.with_structured_output(DetailedGrade)).invoke({
        "query": query,
        "output_description": output_description
    })

# --- 5. EXECUTION ---
def run_benchmark_20():
    results = []
    print(f"Démarrage du Benchmark 'New Supervisor' ({len(VISUALIZATION_BENCHMARK)} questions)...")
    
    for item in tqdm(VISUALIZATION_BENCHMARK):
        try:
            # 1. Appel du Nouveau Superviseur
            # Note : Le nouveau run_supervisor renvoie un objet message avec un attribut 'visualizations'
            response_msg = run_supervisor(item["query"], log=False)
            
            # Extraction des artefacts de visualisation
            viz_artifacts = getattr(response_msg, "visualizations", [])
            
            # Préparation des données pour le juge
            agent_result = {}
            if viz_artifacts:
                # On prend la dernière visualisation générée s'il y en a plusieurs
                agent_result = viz_artifacts[-1]
                print(f"\n[Q{item['id']}] Graphique généré : {agent_result.get('chart_path')}")
            else:
                # Pas de visualisation, on regarde le contenu textuel (erreur potentielle ou réponse simple)
                agent_result = {
                    "chart_path": None,
                    "generated_code": None,
                    "summary": response_msg.content,
                    "error_message": "Aucune visualisation détectée dans la réponse du superviseur."
                }
                print(f"\n[Q{item['id']}] Pas de graphique. Réponse : {response_msg.content[:100]}...")

            # 2. Notation
            grade = evaluate_response_detailed(item["query"], agent_result)
            
            results.append({
                "id": item["id"],
                "level": item["level"],
                "query": item["query"],
                "score": grade.score, 
                "expected": grade.expected,
                "actual": grade.actual,
                "reason": grade.reasoning,
                "file_generated": agent_result.get('chart_path'),
                "code_generated": agent_result.get('generated_code')
            })

            # Sauvegarde incrémentale
            df_temp = pd.DataFrame(results)
            df_temp.to_csv("benchmark_new_architecture_results.csv", index=False)

        except Exception as e:
            print(f"Erreur Critique Q{item['id']}: {e}")
            results.append({
                "id": item["id"], 
                "level": item["level"], 
                "query": item["query"], 
                "score": 0, 
                "expected": "N/A",
                "actual": "Crash du script python",
                "reason": str(e),
                "file_generated": None,
                "code_generated": None
            })
            
            # Sauvegarde incrémentale même en cas d'erreur
            df_temp = pd.DataFrame(results)
            df_temp.to_csv("benchmark_new_architecture_results.csv", index=False)

    # --- 6. RÉSULTATS FINAUX ---
    df = pd.DataFrame(results)
    final_score = df["score"].sum()
    
    print("\n" + "="*60)
    print(f"      NOTE FINALE : {final_score} / 20      ")
    print("="*60)
    
    print("\nDétail par difficulté :")
    print(df.groupby("level")["score"].agg(['count', 'sum', 'mean']))
    
    print(f"\nRapport final sauvegardé dans 'benchmark_new_architecture_results.csv'")

if __name__ == "__main__":  
    # Attention : Ce benchmark suppose que le DATASTORE est peuplé. 
    # Si le datastore est vide, l'agent échouera logiquement à produire des graphs.
    run_benchmark_20()