import pandas as pd
from typing import List, Optional
from tqdm import tqdm
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from llm import llm_from 

# --- 1. IMPORT DU SUPERVISEUR ---
from agents.supervisor_agent import run_supervisor

# --- 2. CONFIGURATION DU JUGE (LLM) ---
JUDGE_LLM = llm_from("aws", "us.anthropic.claude-sonnet-4-5-20250929-v1:0")

# Structure détaillée
class DetailedGrade(BaseModel):
    expected: str = Field(description="Description de ce que la consigne exigeait spécifiquement (type de graph, axes, filtres).")
    actual: str = Field(description="Description de ce que l'agent a réellement produit d'après le code ou le HTML fourni.")
    score: int = Field(description="1 si la réponse est satisfaisante, 0 si elle est incorrecte.")
    reasoning: str = Field(description="Explication du verdict final.")


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

# --- 4. FONCTION D'ÉVALUATION ---
def evaluate_response_detailed(query: str, agent_output: str) -> DetailedGrade:
    system_prompt = """Tu es un expert en Data Visualization et en code Python/Plotly. Tu agis en tant que juge impartial.
    TACHE : Comparer la demande utilisateur avec la sortie générée par un agent AI.
    INSTRUCTIONS :
    1. Analyse la DEMANDE pour identifier le type de graphique, les données et les contraintes.
    2. Analyse la RÉPONSE (Code JSON ou HTML) pour voir ce qui a été techniquement généré.
    3. Remplis les champs 'expected' et 'actual' de manière factuelle.
    4. Attribue 1 point (Succès) ou 0 point (Échec)."""
    
    user_prompt = f"""
    --- DEMANDE UTILISATEUR ---
    "{query}"
    --- RÉPONSE DE L'AGENT ---
    {agent_output}
    """
    
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("user", user_prompt)])
    return (prompt | JUDGE_LLM.with_structured_output(DetailedGrade)).invoke({})

# --- 5. EXECUTION ---
def run_benchmark_20():
    results = []
    print(f"Démarrage du Benchmark Détaillé ({len(VISUALIZATION_BENCHMARK)} questions)...")
    
    for item in tqdm(VISUALIZATION_BENCHMARK):
        try:
            # 1. Appel Agent
            response_msg = run_supervisor(item["query"], log=False)
            
            # Extraction propre
            content = ""
            if hasattr(response_msg, "additional_kwargs"):
                structured = response_msg.additional_kwargs.get("structured", {})
                content = str(structured.get("plot_json") or structured.get("plot_html") or structured.get("content") or response_msg.content)
            else:
                content = str(response_msg)

            # === AJOUT ICI : Affichage de la réponse ===
            print(f"\n\n--- SORTIE AGENT (Q{item['id']}) ---")
            print(content) 
            print("-" * 50)
            # ===========================================

            # 2. Notation Détaillée
            grade = evaluate_response_detailed(item["query"], content)
            
            results.append({
                "id": item["id"],
                "level": item["level"],
                "query": item["query"],
                "score": grade.score, 
                "expected": grade.expected,
                "actual": grade.actual,
                "reason": grade.reasoning
            })

        except Exception as e:
            print(f"Erreur Q{item['id']}: {e}")
            results.append({
                "id": item["id"], 
                "level": item["level"], 
                "query": item["query"], 
                "score": 0, 
                "expected": "N/A",
                "actual": "Crash de l'agent",
                "reason": str(e)
            })

    # --- 6. RÉSULTATS ---
    df = pd.DataFrame(results)
    final_score = df["score"].sum()
    
    print("\n" + "="*60)
    print(f"      NOTE FINALE : {final_score} / 20      ")
    print("="*60)
    
    print("\nDétail par difficulté :")
    print(df.groupby("level")["score"].agg(['count', 'sum', 'mean']))
    
    df.to_csv("benchmark_detailed_results.csv", index=False)
    print(f"\nRapport détaillé sauvegardé dans 'benchmark_detailed_results.csv'")

if __name__ == "__main__":  
    run_benchmark_20()


#avant marchait avant avec prompt en dessous code


import pandas as pd
import re
from typing import List, Optional
from tqdm import tqdm
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from llm import llm_from 

# --- 1. IMPORT DU SUPERVISEUR ---
# On garde l'import qui fonctionne chez toi
from agents.supervisor_agent import run_supervisor

# --- 2. CONFIGURATION DU JUGE (LLM) ---
JUDGE_LLM = llm_from("aws", "us.anthropic.claude-sonnet-4-5-20250929-v1:0")

# --- 3. UTILITAIRE D'EXTRACTION DE CODE ---
def extract_python_code(text: str) -> str:
    """
    Extrait uniquement le bloc de code Python d'une réponse textuelle.
    Cherche les balises markdown ```python ... ```.
    """
    if not text:
        return ""
    
    # Regex pour capturer le contenu entre ```python et ``` (ou juste ```)
    # re.DOTALL permet au . de matcher les retours à la ligne
    pattern = r"```(?:python)?(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    
    if matches:
        # On retourne le code trouvé (s'il y en a plusieurs, on les joint)
        return "\n\n# --- AUTRE BLOC ---\n".join([m.strip() for m in matches])
    
    # Si pas de balises, on renvoie le texte brut (l'agent a peut-être oublié le formatage)
    return text

# --- 4. MODÈLE DE NOTATION (CODE REVIEW) ---
class CodeGrade(BaseModel):
    syntax_valid: bool = Field(description="Le code semble-t-il syntaxiquement correct (imports présents, variables définies) ?")
    libraries_used: List[str] = Field(description="Liste des bibliothèques utilisées (ex: matplotlib, plotly, seaborn).")
    logic_assessment: str = Field(description="Analyse de la logique : le code génère-t-il bien les données et le type de graphique demandé ?")
    score: int = Field(description="1 si le CODE est fonctionnel et répond à la demande, 0 sinon.")
    reasoning: str = Field(description="Explication technique du verdict.")

# --- 5. LE BENCHMARK ---
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

# --- 6. FONCTION D'ÉVALUATION (Code Quality) ---
def evaluate_code_quality(query: str, code_snippet: str) -> CodeGrade:
    system_prompt = """Tu es un Senior Python Developer spécialisé en Data Visualization.
    
    TACHE : Effectuer une revue de code (Code Review) sur le snippet fourni par un agent AI.
    
    CRITÈRES DE SUCCÈS (Score 1) :
    1. Le code importe les bonnes librairies (matplotlib, plotly, seaborn, numpy...).
    2. La logique génère les données nécessaires (si elles ne sont pas fournies).
    3. Les fonctions de tracé utilisées correspondent à la demande (ex: utiliser `ax.voxels` pour des voxels 3D).
    4. Le code semble complet et exécutable.

    CRITÈRES D'ÉCHEC (Score 0) :
    1. Code halluciné ou incomplet (ex: variables non définies).
    2. Mauvais type de graphique.
    3. Pas de code fourni.
    """
    
    user_prompt = f"""
    --- CONSIGNE UTILISATEUR ---
    "{query}"

    --- CODE GÉNÉRÉ ---
    {code_snippet}
    """
    
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("user", user_prompt)])
    return (prompt | JUDGE_LLM.with_structured_output(CodeGrade)).invoke({})

# --- 7. EXECUTION ---
def run_benchmark_20():
    results = []
    print(f"Démarrage du Benchmark CODE REVIEW ({len(VISUALIZATION_BENCHMARK)} questions)...")
    
    for item in tqdm(VISUALIZATION_BENCHMARK):
        try:
            # 1. Appel Agent
            response_msg = run_supervisor(item["query"], log=False)
            
            # 2. Récupération du contenu textuel complet
            full_content = getattr(response_msg, "content", str(response_msg))
            
            # 3. Extraction du CODE uniquement
            extracted_code = extract_python_code(full_content)

            # === AFFICHAGE DU CODE ===
            print(f"\n\n--- CODE EXTRAIT (Q{item['id']}) ---")
            # On tronque pour l'affichage si c'est immense, mais on garde assez pour lire
            print(extracted_code[:2000] + ("\n... [Code tronqué] ..." if len(extracted_code) > 2000 else "")) 
            print("-" * 50)
            # =========================

            # 4. Notation sur le CODE
            grade = evaluate_code_quality(item["query"], extracted_code)
            
            results.append({
                "id": item["id"],
                "level": item["level"],
                "query": item["query"],
                "score": grade.score, 
                "syntax_valid": grade.syntax_valid,
                "libraries": ", ".join(grade.libraries_used),
                "logic": grade.logic_assessment,
                "reason": grade.reasoning
            })

        except Exception as e:
            print(f"Erreur Q{item['id']}: {e}")
            results.append({
                "id": item["id"], 
                "level": item["level"], 
                "query": item["query"], 
                "score": 0, 
                "syntax_valid": False,
                "libraries": "N/A",
                "logic": "Crash de l'agent ou erreur extraction",
                "reason": str(e)
            })

    # --- 8. RÉSULTATS ---
    df = pd.DataFrame(results)
    final_score = df["score"].sum()
    
    print("\n" + "="*60)
    print(f"      NOTE QUALITÉ CODE : {final_score} / 20      ")
    print("="*60)
    
    print("\nDétail par difficulté :")
    print(df.groupby("level")["score"].agg(['count', 'sum', 'mean']))
    
    # Sauvegarde
    filename = "benchmark_code_quality_results.csv"
    df.to_csv(filename, index=False)
    print(f"\nRapport Code Review sauvegardé dans '{filename}'")

if __name__ == "__main__":
    run_benchmark_20()



from pathlib import Path
import pandas as pd
import re
from typing import List, Optional
from tqdm import tqdm
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from llm import llm_from 

# --- 1. IMPORT DU SUPERVISEUR ---
from agents.supervisor_agent import run_supervisor

# --- 2. CONFIGURATION DU JUGE (LLM) ---
JUDGE_LLM = llm_from("aws", "us.anthropic.claude-sonnet-4-5-20250929-v1:0")

# --- 3. UTILITAIRE D'EXTRACTION DE CODE ---
def extract_python_code(text: str) -> str:
    """
    Extrait le code contenu dans les blocs markdown ```python ... ```
    Si aucun bloc n'est trouvé, renvoie le texte brut (au cas où l'agent donne le code sans formatage).
    """
    if not text:
        return ""
    # Regex pour capturer le contenu entre ```python et ```
    pattern = r"```python(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    
    if matches:
        # On retourne tous les blocs de code trouvés, séparés par des sauts de ligne
        return "\n\n# --- BLOC DE CODE ---\n".join([m.strip() for m in matches])
    
    # Fallback : si pas de balises, on suppose que tout le texte est pertinent pour l'analyse
    return text

# Structure de notation orientée "Code Review"
class CodeGrade(BaseModel):
    syntax_valid: bool = Field(description="Le code semble-t-il syntaxiquement correct ?")
    libraries_used: List[str] = Field(description="Liste des bibliothèques principales utilisées (ex: ['matplotlib', 'pandas']).")
    logic_assessment: str = Field(description="Analyse de la logique : les données sont-elles bien générées/traitées pour répondre à la question ?")
    score: int = Field(description="1 si le CODE est capable de générer le graphique demandé, 0 sinon.")
    reasoning: str = Field(description="Explication technique du verdict.")

# --- 4. LE BENCHMARK ---
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

# --- 5. FONCTION D'ÉVALUATION DU CODE ---
def evaluate_code_quality(query: str, code_snippet: str) -> CodeGrade:
    system_prompt = """Tu es un Senior Python Developer spécialisé en Data Viz.
    TACHE : Faire une revue de code (Code Review) sur le snippet fourni par un agent.
    
    CRITÈRES DE SUCCÈS (Score 1) :
    1. Le code utilise les bonnes bibliothèques (matplotlib, plotly, seaborn, numpy) pour la demande.
    2. La logique de génération de données (si nécessaire) est cohérente avec la demande (ex: distribution Beta, équation spécifique).
    3. Les fonctions de traçage (ex: boxplot, scatter, plot_surface) correspondent au type de graphique demandé.
    4. Le code semble exécutable sans erreur de syntaxe majeure.

    CRITÈRES D'ÉCHEC (Score 0) :
    1. Code halluciné ou incomplet (manque les imports, variables non définies).
    2. Mauvais type de graphique (ex: Bar chart au lieu de Scatter).
    3. Pas de code fourni.
    """
    
    user_prompt = f"""
    --- CONSIGNE UTILISATEUR ---
    "{query}"

    --- CODE PROPOSÉ PAR L'AGENT ---
    {code_snippet}
    """
    
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("user", user_prompt)])
    return (prompt | JUDGE_LLM.with_structured_output(CodeGrade)).invoke({})

# --- 6. EXECUTION ---
def run_benchmark_20():
    results = []
    print(f"Démarrage du Benchmark CODE REVIEW ({len(VISUALIZATION_BENCHMARK)} questions)...")
    
    for item in tqdm(VISUALIZATION_BENCHMARK):
        try:
            # 1. Appel Agent
            response_msg = run_supervisor(item["query"], log=False)
            
            # 2. Récupération du contenu textuel complet
            full_content = getattr(response_msg, "content", str(response_msg))
            
            # 3. Extraction spécifique du CODE
            extracted_code = extract_python_code(full_content)

            # === AJOUT ICI : Affichage du CODE uniquement ===
            print(f"\n\n--- CODE EXTRAIT (Q{item['id']}) ---")
            print(extracted_code[:1500] + ("\n... [Code tronqué] ..." if len(extracted_code) > 1500 else "")) 
            print("-" * 50)
            # ===============================================

            # 4. Notation du Code
            grade = evaluate_code_quality(item["query"], extracted_code)
            
            results.append({
                "id": item["id"],
                "level": item["level"],
                "query": item["query"],
                "score": grade.score, 
                "syntax_valid": grade.syntax_valid,
                "libraries": ", ".join(grade.libraries_used),
                "logic": grade.logic_assessment,
                "reason": grade.reasoning
            })

        except Exception as e:
            print(f"Erreur Q{item['id']}: {e}")
            results.append({
                "id": item["id"], 
                "level": item["level"], 
                "query": item["query"], 
                "score": 0, 
                "syntax_valid": False,
                "libraries": "N/A",
                "logic": "Crash",
                "reason": str(e)
            })

    # --- 7. RÉSULTATS ---
    df = pd.DataFrame(results)
    final_score = df["score"].sum()
    
    print("\n" + "="*60)
    print(f"      NOTE CODE QUALITY : {final_score} / 20      ")
    print("="*60)
    
    print("\nDétail par difficulté :")
    print(df.groupby("level")["score"].agg(['count', 'sum', 'mean']))
    
    df.to_csv("benchmark_code_quality_results.csv", index=False)
    print(f"\nRapport détaillé sauvegardé dans 'benchmark_code_quality_results.csv'")

if __name__ == "__main__":
    run_benchmark_20()