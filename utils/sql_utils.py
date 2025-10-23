import csv
import sqlite3
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

DEFAULT_DATABASE_DIR = Path("database")
DEFAULT_INSTRUCTION_FILE = DEFAULT_DATABASE_DIR / "requetesSQL_depuisBTM-PROD.txt"


def create_test_database() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE sensors (
            id TEXT PRIMARY KEY,
            location TEXT NOT NULL,
            type TEXT
        );
        """
    )
    cursor.execute(
        """
        CREATE TABLE sensor_readings (
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            sensor_id TEXT,
            value REAL,
            FOREIGN KEY (sensor_id) REFERENCES sensors(id)
        );
        """
    )
    conn.commit()
    return conn


def _detect_delimiter(sample: str) -> str:
    comma = sample.count(",")
    semicolon = sample.count(";")
    return ";" if semicolon > comma else ","


def _sanitize_identifier(name: str, fallback: str) -> str:
    cleaned = "".join(char if char.isalnum() or char == "_" else "_" for char in name.strip())
    cleaned = cleaned.strip("_")
    if not cleaned:
        cleaned = fallback
    if cleaned[0].isdigit():
        cleaned = f"{fallback}_{cleaned}"
    return cleaned.lower()


def _read_csv(csv_path: Path) -> Tuple[List[str], List[List[str]]]:
    encodings = ("utf-8-sig", "utf-8", "latin-1")
    last_error: Exception | None = None

    for encoding in encodings:
        try:
            with csv_path.open("r", encoding=encoding, newline="") as handle:
                sample = handle.read(4096)
                handle.seek(0)
                delimiter = _detect_delimiter(sample)
                reader = csv.reader(handle, delimiter=delimiter)
                rows = list(reader)
                if not rows:
                    return [], []
                header, data_rows = rows[0], rows[1:]
                return header, data_rows
        except UnicodeDecodeError as exc:
            last_error = exc
            continue

    if last_error:
        raise last_error
    raise ValueError(f"Unable to decode CSV file: {csv_path}")


def _load_csv_into_table(
    cursor: sqlite3.Cursor,
    csv_path: Path,
) -> Tuple[str, List[str], List[str]]:
    header, rows = _read_csv(csv_path)
    if not header:
        raise ValueError(f"CSV file {csv_path} is empty or missing a header.")

    base_table_name = _sanitize_identifier(csv_path.stem, "table")
    table_name = base_table_name
    suffix = 1
    while True:
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        )
        if cursor.fetchone() is None:
            break
        suffix += 1
        table_name = f"{base_table_name}_{suffix}"

    sanitized_columns: List[str] = []
    seen: Dict[str, int] = {}
    for idx, column in enumerate(header):
        base_column = _sanitize_identifier(column, f"column_{idx}")
        counter = seen.get(base_column, 0)
        name = base_column if counter == 0 else f"{base_column}_{counter}"
        while name in seen:
            counter += 1
            name = f"{base_column}_{counter}"
        seen[base_column] = counter + 1
        sanitized_columns.append(name)

    cursor.execute(f'DROP TABLE IF EXISTS "{table_name}"')
    column_defs = ", ".join(f'"{col}" TEXT' for col in sanitized_columns)
    cursor.execute(f'CREATE TABLE "{table_name}" ({column_defs})')

    placeholders = ", ".join("?" for _ in sanitized_columns)
    for row in rows:
        values = list(row[: len(sanitized_columns)])
        if len(values) < len(sanitized_columns):
            values.extend([""] * (len(sanitized_columns) - len(values)))
        cursor.execute(
            f'INSERT INTO "{table_name}" VALUES ({placeholders})',
            values,
        )

    return table_name, sanitized_columns, header


def _normalize_label(label: str) -> str:
    return label.strip().lower()


def _iter_instruction_blocks(path: Path) -> Iterator[Tuple[str, str]]:
    if not path.exists():
        return

    content = path.read_text(encoding="utf-8")
    for block in content.split("###########"):
        block = block.strip()
        if not block:
            continue

        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue

        label = lines[0]
        if label.startswith("##"):
            label = label.lstrip("#").strip()

        query = "\n".join(lines[1:]).strip()
        if query:
            yield label, query


def _rows_to_dicts(
    rows: Sequence[Sequence[str]],
    column_names: Sequence[str],
    *,
    aliases: Sequence[str] | None = None,
) -> List[Dict[str, str]]:
    keys = list(aliases) if aliases else list(column_names)
    result: List[Dict[str, str]] = []
    for row in rows:
        entry = {keys[idx]: row[idx] for idx in range(min(len(keys), len(row)))}
        result.append(entry)
    return result


def _resolve_table_name(label_lookup: Dict[str, str], key: str) -> Optional[str]:
    return label_lookup.get(_normalize_label(key))


def _custom_instruction_sql(
    normalized_label: str,
    label_lookup: Dict[str, str],
) -> Optional[Tuple[str, List[str]]]:
    site_filters = [
        "MAT",
        "MAT_Avois%",
        "MAT_LA%",
        "MAT_Topo%",
        "MAT_Sono%",
        "MAT_Corps%",
    ]

    projects_table = _resolve_table_name(label_lookup, "projects_sites.csv")
    gateways_table = _resolve_table_name(label_lookup, "gateways_configs_sensors.csv")
    metrics_table = _resolve_table_name(
        label_lookup, "variables_metrics_raw_data-1760453175763.csv"
    )

    if normalized_label in {"projects_sites.csv", "projects_sites"}:
        if not projects_table:
            return None
        conditions = " OR ".join("ps.name_2 LIKE ?" for _ in site_filters)
        sql_text = f"""
            SELECT ps.*
            FROM "{projects_table}" AS ps
            WHERE ps.name LIKE ?
              AND ({conditions})
        """
        params = ["%M3%"] + site_filters
        return sql_text, params

    if normalized_label in {"gateways_configs_sensors.csv", "gateways_configs_sensors"}:
        if not (gateways_table and projects_table):
            return None
        conditions = " OR ".join("ps.name_2 LIKE ?" for _ in site_filters)
        sql_text = f"""
            SELECT gcs.*
            FROM "{gateways_table}" AS gcs
            WHERE gcs.project_id IN (
                SELECT ps.project_id
                FROM "{projects_table}" AS ps
                WHERE ps.name LIKE ?
                  AND ({conditions})
            )
        """
        params = ["%M3%"] + site_filters
        return sql_text, params

    if normalized_label in {
        "variables_metrics_raw_data-1760453175763.csv",
        "variables_metrics_raw_data-1760453175763",
    }:
        if not (metrics_table and gateways_table and projects_table):
            return None
        sql_text = f"""
            SELECT vm.*
            FROM "{metrics_table}" AS vm
            JOIN "{gateways_table}" AS gcs
              ON gcs.gateway_name = vm.gateway_name
            JOIN "{projects_table}" AS ps
              ON ps.project_id = gcs.project_id
            WHERE ps.name LIKE ?
              AND ps.name_2 LIKE ?
              AND vm."timestamp" BETWEEN ? AND ?
              AND vm.gateway_name IN (?, ?, ?)
        """
        params = [
            "%M3%",
            "MAT",
            "2024-11-20 10:00:00.000 +0200",
            "2025-05-20 10:00:00.000 +0200",
            "Gateway-Tisséo Ligne A",
            "MAT_INCLINO",
            "MAT_LA_STA1_Custom",
        ]
        return sql_text, params

    return None


def load_db(
    data_dir: str | Path = DEFAULT_DATABASE_DIR,
    instruction_file: str | Path = DEFAULT_INSTRUCTION_FILE,
) -> Tuple[sqlite3.Connection, Dict[str, Dict[str, object]]]:
    """
    Load the CSV data stored under `database/` into an in-memory SQLite database
    and execute the reference SQL queries listed in the instruction file.

    Returns the SQLite connection plus a dictionary describing the outcome of
    each executed query.
    """
    data_path = Path(data_dir)
    instructions_path = Path(instruction_file)

    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()

    table_metadata: Dict[str, Dict[str, object]] = {}
    label_lookup: Dict[str, str] = {}

    for csv_path in sorted(data_path.glob("*.csv")):
        table_name, columns, original_columns = _load_csv_into_table(cursor, csv_path)
        table_metadata[table_name] = {
            "columns": columns,
            "original_columns": original_columns,
            "source": csv_path,
        }

        label_lookup[_normalize_label(csv_path.name)] = table_name
        label_lookup[_normalize_label(csv_path.stem)] = table_name

    executed_queries: Dict[str, Dict[str, object]] = {}

    if instructions_path.exists():
        for label, query in _iter_instruction_blocks(instructions_path):
            normalized_label = _normalize_label(label)
            custom_sql = _custom_instruction_sql(normalized_label, label_lookup)
            custom_error: Optional[str] = None

            if custom_sql:
                sql_text, params = custom_sql
                try:
                    cursor.execute(sql_text, params)
                    raw_rows = cursor.fetchall()
                    column_names = (
                        [description[0] for description in cursor.description]
                        if cursor.description
                        else []
                    )
                    rows = _rows_to_dicts(raw_rows, column_names)
                    executed_queries[label] = {
                        "query": query,
                        "effective_query": sql_text,
                        "rows": rows,
                        "columns": column_names,
                        "success": True,
                        "error": None,
                        "used_fallback": False,
                    }
                    continue
                except sqlite3.Error as exc:
                    custom_error = str(exc)

            try:
                cursor.execute(query)
                raw_rows = cursor.fetchall()
                column_names = (
                    [description[0] for description in cursor.description]
                    if cursor.description
                    else []
                )
                rows = _rows_to_dicts(raw_rows, column_names)
                executed_queries[label] = {
                    "query": query,
                    "effective_query": query,
                    "rows": rows,
                    "columns": column_names,
                    "success": True,
                    "error": custom_error,
                    "used_fallback": False,
                }
                if custom_error:
                    executed_queries[label]["custom_error"] = custom_error
            except sqlite3.Error as exc:
                fallback_table = (
                    label_lookup.get(normalized_label)
                    or label_lookup.get(_normalize_label(Path(label).stem))
                )

                fallback_rows: List[Dict[str, str]] = []
                fallback_columns: List[str] = []
                effective_query = query

                if fallback_table:
                    metadata = table_metadata.get(fallback_table, {})
                    original_columns = metadata.get("original_columns", [])
                    columns = metadata.get("columns", [])

                    cursor.execute(f'SELECT * FROM "{fallback_table}"')
                    raw_rows = cursor.fetchall()
                    fallback_rows = _rows_to_dicts(
                        raw_rows,
                        columns,
                        aliases=original_columns if original_columns else None,
                    )
                    fallback_columns = list(original_columns) if original_columns else list(columns)
                    effective_query = f'SELECT * FROM "{fallback_table}"'

                if fallback_table:
                    executed_queries[label] = {
                        "query": query,
                        "effective_query": effective_query,
                        "rows": fallback_rows,
                        "columns": fallback_columns,
                        "success": True,
                        "error": str(exc),
                        "used_fallback": True,
                        "fallback_table": fallback_table,
                    }
                else:
                    executed_queries[label] = {
                        "query": query,
                        "effective_query": query,
                        "rows": [],
                        "columns": [],
                        "success": False,
                        "error": str(exc),
                        "used_fallback": False,
                        "fallback_table": None,
                    }
                if custom_error:
                    executed_queries[label]["custom_error"] = custom_error

    conn.commit()
    return conn, executed_queries


def get_db_schema(connexion: sqlite3.Connection) -> str:
    """Récupère le schéma de la base de données (instructions CREATE)."""
    schema = ""
    cursor = connexion.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    for table in tables:
        cursor.execute(
            f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table[0]}'"
        )
        definition = cursor.fetchone()
        if definition and definition[0]:
            schema += definition[0] + "\n\n"
    return schema


def execute_sql_tool(connexion: sqlite3.Connection, sql_query: str):
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


def selftest_load_db(
    data_dir: str | Path = DEFAULT_DATABASE_DIR,
    instruction_file: str | Path = DEFAULT_INSTRUCTION_FILE,
) -> Dict[str, object]:
    """
    Basic smoke test ensuring the database loads and instruction queries run.

    Returns a dictionary with:
        - connection: sqlite3.Connection
        - tables: list of available table names
        - query_stats: summary of executed instruction blocks
    """
    conn, results = load_db(data_dir=data_dir, instruction_file=instruction_file)

    tables = [
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    ]

    query_stats: Dict[str, Dict[str, object]] = {}
    for label, info in results.items():
        if not isinstance(info, dict):
            continue
        query_stats[label] = {
            "success": info.get("success", False),
            "used_fallback": info.get("used_fallback", False),
            "row_count": len(info.get("rows", [])),
            "columns": info.get("columns", []),
            "error": info.get("error"),
            "fallback_table": info.get("fallback_table"),
        }

    return {
        "connection": conn,
        "tables": tables,
        "query_stats": query_stats,
    }
from pprint import pprint

summary = selftest_load_db()
pprint(summary["tables"])
pprint(summary["query_stats"])