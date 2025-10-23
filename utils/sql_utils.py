import sqlite3
import random
from datetime import datetime, timedelta

def create_test_database():
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()

    # Créer les tables
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

    # === Insérer deux capteurs ===
    cursor.execute("INSERT INTO sensors (id, location, type) VALUES (?, ?, ?)", 
                   ("sensor_1", "Living Room", "temperature"))
    cursor.execute("INSERT INTO sensors (id, location, type) VALUES (?, ?, ?)", 
                   ("sensor_2", "Kitchen", "humidity"))

    # === Générer des valeurs aléatoires ===
    now = datetime.now()
    for i in range(10):  # 10 lectures par capteur
        ts = now - timedelta(minutes=i*5)  # toutes les 5 minutes
        val1 = round(random.uniform(18, 25), 2)  # temp en °C
        val2 = round(random.uniform(30, 60), 2)  # humidité en %
        cursor.execute("INSERT INTO sensor_readings (timestamp, sensor_id, value) VALUES (?, ?, ?)", 
                       (ts, "S1", val1))
        cursor.execute("INSERT INTO sensor_readings (timestamp, sensor_id, value) VALUES (?, ?, ?)", 
                       (ts, "S2", val2))

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
        return f"Erreur d'exécution SQL: {e}"


# === Exemple d'utilisation ===
if __name__ == "__main__":
    conn = create_test_database()
    print(get_db_schema(conn))
    print(execute_sql_tool(conn, "SELECT * FROM sensors;"))
    print(execute_sql_tool(conn, "SELECT * FROM sensor_readings LIMIT 5;"))
