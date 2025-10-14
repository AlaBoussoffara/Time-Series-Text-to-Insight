import sqlite3


def create_test_database():
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
    return conn

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