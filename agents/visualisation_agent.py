from langgraph.graph import StateGraph, START, END
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import plotly.express as px
import base64
import json
import io

# === Définition du State (état partagé entre les nœuds) ===
class PlotState(Dict[str, Any]):
    """
    Structure d'état pour l'agent de visualisation.
    Contient :
      - 'data' : les données d'entrée (liste de dicts ou JSON)
      - 'question' : la consigne ou question utilisateur
      - 'chart_type' : le type de graphique choisi (bar, line, pie, etc.)
      - 'plot_json' : graphique final en JSON (pour stockage)
      - 'plot_html' : HTML embarquable du graphique (pour affichage direct en conversation)
      - 'plot_png_b64' : image PNG encodée base64 (fallback si nécessaire)
      - 'error_message' : message d'erreur si échec
      - 'display_in_conversation' : bool -> indique que le frontend doit afficher directement en conversation
    """
    pass


def create_plot_agent(llm):
    """
    Crée un agent LangGraph capable de :
      1. Analyser la question et les données
      2. Choisir un type de graphique pertinent
      3. Générer un graphique Plotly
      4. Retourner la figure JSON et un HTML embarquable pour affichage direct en conversation
    """

    def choose_chart_type_node(state: PlotState):
        # Choisit un type de graphique simple si non fourni
        chart_type = state.get("chart_type")
        if not chart_type:
            # heuristique minimale : si question contient "trend" -> line, "distribution" -> histogram, else bar
            q = (state.get("question") or "").lower()
            if "trend" in q or "trend" in q or "over time" in q:
                chart_type = "line"
            elif "distribution" in q or "hist" in q:
                chart_type = "histogram"
            else:
                chart_type = "bar"
        state["chart_type"] = chart_type
        return state

    def generate_plot_node(state: PlotState):
        print("--- DEBUG: generate_plot_node start ---")
        data = state.get("data", [])
        chart_type = state.get("chart_type", "bar")

        # Préparer structure de sortie
        viz = {"plot_json": None, "plot_html": None, "plot_png_b64": None, "error_message": None, "display_in_conversation": False}

        try:
            df = pd.DataFrame(data)
            if df.empty:
                raise ValueError("Les données sont vides ou invalides pour la visualisation.")

            x_col = df.columns[0]
            y_col = df.columns[1] if len(df.columns) > 1 else None

            if chart_type == "bar":
                fig = px.bar(df, x=x_col, y=y_col, title="Visualisation des données")
            elif chart_type == "line":
                fig = px.line(df, x=x_col, y=y_col, title="Visualisation des données")
            elif chart_type == "scatter":
                fig = px.scatter(df, x=x_col, y=y_col, title="Visualisation des données")
            elif chart_type == "pie" and y_col:
                fig = px.pie(df, names=x_col, values=y_col, title="Répartition des données")
            else:
                fig = px.histogram(df, x=x_col, title="Distribution des valeurs")

            # Stockage JSON (fig.to_json)
            try:
                viz["plot_json"] = fig.to_json()
            except Exception as e:
                print(f"--- DEBUG: to_json failed: {e} ---")
                viz["plot_json"] = None

            # Fragment HTML embarquable (full_html=False) -> préférable pour conversation
            try:
                plot_html = fig.to_html(full_html=False, include_plotlyjs="cdn")
                viz["plot_html"] = plot_html
            except Exception as e:
                print(f"--- DEBUG: to_html failed: {e} ---")
                viz["plot_html"] = None

            # Fallback image PNG base64 (nécessite kaleido)
            try:
                img_bytes = fig.to_image(format="png")
                viz["plot_png_b64"] = base64.b64encode(img_bytes).decode("ascii")
            except Exception as e:
                print(f"--- DEBUG: to_image failed (kaleido?): {e} ---")
                viz["plot_png_b64"] = None

            viz["display_in_conversation"] = True
            print("--- DEBUG: plot generated ---")

        except Exception as e:
            viz["error_message"] = str(e)
            viz["display_in_conversation"] = False
            print(f"--- DEBUG: generate_plot_node error: {e} ---")

        # Stocker de façon fiable dans le datastore attendu par le superviseur
        datastore = state.get("datastore")
        if datastore is None or not isinstance(datastore, dict):
            state["datastore"] = {}
            datastore = state["datastore"]
        datastore["visualization"] = viz

        # logs pour debug rapide
        if viz["plot_html"]:
            print(f"--- DEBUG plot_html length: {len(viz['plot_html'])} ---")
        if viz["plot_png_b64"]:
            print(f"--- DEBUG plot_png_b64 length: {len(viz['plot_png_b64'])} ---")

        return state

    workflow = StateGraph(PlotState)
    workflow.add_node("choose_chart_type", choose_chart_type_node)
    workflow.add_node("generate_plot", generate_plot_node)
    workflow.add_edge(START, "choose_chart_type")
    workflow.add_edge("choose_chart_type", "generate_plot")
    workflow.add_edge("generate_plot", END)

    plot_agent = workflow.compile()
    return plot_agent
