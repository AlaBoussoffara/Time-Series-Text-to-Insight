from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import psycopg
import csv
try:
    from dotenv import load_dotenv  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None  # type: ignore

DATA_DIR = Path(__file__).resolve().parent.parent / "database"

PROJECT_COLUMNS = [
    "project_id",
    "name",
    "country",
    "client_company_name",
    "time_zone",
    "city",
    "start_date",
    "description",
]

SITE_COLUMNS = [
    "site_id",
    "name",
    "type",
    "extent",
    "start_date",
    "previsional_end",
    "project_id",
    "created_at",
    "updated_at",
    "deleted",
    "operating_rate",
    "main_site",
]

GATEWAY_COLUMNS = [
    "gateway_id",
    "name",
    "serial_number",
    "transfer_protocol",
    "power_supply",
    "installation_date",
    "operating_team",
    "x",
    "y",
    "z",
    "time_zone",
    "geom",
    "created_at",
    "updated_at",
    "provider",
]

CONFIG_COLUMNS = [
    "config_id",
    "file_name",
    "last_treatment",
    "ftp",
    "ftp_ip",
    "ftp_user",
    "ftp_password",
    "ftp_directory",
    "payload",
    "gateway_id",
    "parsing_id",
    "file_id",
    "to_move",
    "regex_variables",
    "created_at",
    "updated_at",
    "active",
    "error_message",
    "last_modified",
    "keep_folder",
]

SITE_GATEWAY_COLUMNS = ["site_id", "gateway_id"]

RAW_MEASUREMENT_COLUMNS = [
    "gateway_name",
    "variable_name",
    "variable_alias",
    "sensor_id",
    "value",
    "timestamp",
    "variable_id",
    "unit",
    "metric",
]

RAW_MEASUREMENT_ORDER = ["gateway_name", "timestamp", "variable_id", "sensor_id"]

EXPECTED_SCHEMA: Dict[str, List[Tuple[str, str]]] = {
    "projects": [
        ("project_id", "integer"),
        ("name", "text"),
        ("country", "text"),
        ("client_company_name", "text"),
        ("time_zone", "text"),
        ("city", "text"),
        ("start_date", "text"),
        ("description", "text"),
    ],
    "sites": [
        ("site_id", "integer"),
        ("name", "text"),
        ("type", "text"),
        ("extent", "text"),
        ("start_date", "text"),
        ("previsional_end", "text"),
        ("project_id", "integer"),
        ("created_at", "text"),
        ("updated_at", "text"),
        ("deleted", "boolean"),
        ("operating_rate", "double precision"),
        ("main_site", "integer"),
    ],
    "gateways": [
        ("gateway_id", "integer"),
        ("name", "text"),
        ("serial_number", "text"),
        ("transfer_protocol", "text"),
        ("power_supply", "text"),
        ("installation_date", "text"),
        ("operating_team", "text"),
        ("x", "double precision"),
        ("y", "double precision"),
        ("z", "double precision"),
        ("time_zone", "text"),
        ("geom", "text"),
        ("created_at", "text"),
        ("updated_at", "text"),
        ("provider", "integer"),
    ],
    "configs": [
        ("config_id", "integer"),
        ("file_name", "text"),
        ("last_treatment", "text"),
        ("ftp", "text"),
        ("ftp_ip", "text"),
        ("ftp_user", "text"),
        ("ftp_password", "text"),
        ("ftp_directory", "text"),
        ("payload", "text"),
        ("gateway_id", "integer"),
        ("parsing_id", "integer"),
        ("file_id", "integer"),
        ("to_move", "text"),
        ("regex_variables", "text"),
        ("created_at", "text"),
        ("updated_at", "text"),
        ("active", "boolean"),
        ("error_message", "text"),
        ("last_modified", "text"),
        ("keep_folder", "boolean"),
    ],
    "site_gateways": [
        ("site_id", "integer"),
        ("gateway_id", "integer"),
    ],
    "raw_measurements": [
        ("gateway_name", "text"),
        ("variable_name", "text"),
        ("variable_alias", "text"),
        ("sensor_id", "integer"),
        ("value", "double precision"),
        ("timestamp", "text"),
        ("variable_id", "integer"),
        ("unit", "text"),
        ("metric", "text"),
    ],
}

ROW_ORDER = {
    "projects": ["project_id"],
    "sites": ["site_id"],
    "gateways": ["gateway_id"],
    "configs": ["config_id"],
    "site_gateways": ["site_id", "gateway_id"],
}


@dataclass
class ExpectedTable:
    columns: Sequence[str]
    rows: Optional[List[Tuple]] = None
    checksum: Optional[str] = None
    checksum_count: Optional[int] = None

    def expected_count(self) -> int:
        if self.rows is not None:
            return len(self.rows)
        if self.checksum_count is not None:
            return self.checksum_count
        raise ValueError("No expected row information recorded.")


def main() -> None:
    if load_dotenv is not None:
        load_dotenv()
    dsn = os.environ.get("POSTGRES_DSN")
    if not dsn:
        raise RuntimeError("Set POSTGRES_DSN to point at your PostgreSQL database.")

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Expected CSV directory at {DATA_DIR}")

    expected: Dict[str, ExpectedTable] = {}

    with psycopg.connect(dsn) as conn:
        conn.execute("SET client_encoding = 'UTF8';")
        create_tables(conn)
        truncate_tables(conn)

        expected.update(load_projects_and_sites(conn))
        expected.update(load_gateways_configs_sites(conn))
        expected.update(load_raw_measurements(conn))
        conn.commit()

        create_indexes(conn)
        conn.commit()

        verify_schema(conn)
        verify_counts(conn, expected)
        verify_rows(conn, expected)
        verify_raw_measurements(conn, expected)

    print("✅ PostgreSQL database seeded and verified against CSV extracts.")


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #


def load_projects_and_sites(conn: psycopg.Connection) -> Dict[str, ExpectedTable]:
    path = DATA_DIR / "projects_sites.csv"
    if not path.exists():
        raise FileNotFoundError(path)

    projects: Dict[int, Tuple] = {}
    sites: Dict[int, Tuple] = {}

    for record in iter_csv(path, delimiter=";", encoding="latin-1"):
        project_id = to_int(record["id"])
        site_id = to_int(record["id_1"])

        if project_id is not None and project_id not in projects:
            projects[project_id] = (
                project_id,
                clean(record["name"]),
                clean(record["country"]),
                clean(record["client_company_name"]),
                clean(record["time_zone"]),
                clean(record["city"]),
                clean(record["start_date"]),
                clean(record["description"]),
            )

        if site_id is not None and site_id not in sites:
            sites[site_id] = (
                site_id,
                clean(record["name_2"]),
                clean(record["type"]),
                clean(record["extent"]),
                clean(record["start_date_3"]),
                clean(record["previsional_end"]),
                to_int(record["project_id"]),
                clean(record["created_at"]),
                clean(record["updated_at"]),
                to_bool(record["deleted"]),
                to_float(record["operating_rate"]),
                to_int(record["main_site"]),
            )

    with conn.cursor() as cur:
        if projects:
            cur.executemany(
                """
                INSERT INTO projects (
                    project_id, name, country, client_company_name,
                    time_zone, city, start_date, description
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
                """,
                projects.values(),
            )

        if sites:
            cur.executemany(
                """
                INSERT INTO sites (
                    site_id, name, type, extent, start_date, previsional_end,
                    project_id, created_at, updated_at, deleted,
                    operating_rate, main_site
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
                """,
                sites.values(),
            )

    project_rows = sort_rows(list(projects.values()), ROW_ORDER["projects"], PROJECT_COLUMNS)
    site_rows = sort_rows(list(sites.values()), ROW_ORDER["sites"], SITE_COLUMNS)

    return {
        "projects": ExpectedTable(columns=PROJECT_COLUMNS, rows=project_rows),
        "sites": ExpectedTable(columns=SITE_COLUMNS, rows=site_rows),
    }


def load_gateways_configs_sites(conn: psycopg.Connection) -> Dict[str, ExpectedTable]:
    path = DATA_DIR / "gateways_configs_sensors.csv"
    if not path.exists():
        raise FileNotFoundError(path)

    gateways: Dict[int, Tuple] = {}
    configs: Dict[int, Tuple] = {}
    site_links: set[Tuple[int, int]] = set()

    for record in iter_csv(path, delimiter=",", encoding="utf-8"):
        gateway_id = to_int(record["id"])
        config_id = to_int(record["id_2"])
        site_id = to_int(record["id_3"])

        if gateway_id is not None and gateway_id not in gateways:
            gateways[gateway_id] = (
                gateway_id,
                clean(record["gateway_name"]),
                clean(record["serial_number"]),
                clean(record["transfer_protocol"]),
                clean(record["power_supply"]),
                clean(record["installation_date"]),
                clean(record["operating_team"]),
                to_float(record["x"]),
                to_float(record["y"]),
                to_float(record["z"]),
                clean(record["time_zone"]),
                clean(record["geom"]),
                clean(record["created_at"]),
                clean(record["updated_at"]),
                to_int(record["provider"]),
            )

        if config_id is not None and config_id not in configs:
            configs[config_id] = (
                config_id,
                clean(record["file_name"]),
                clean(record["last_treatment"]),
                clean(record["ftp"]),
                clean(record["ftp_ip"]),
                clean(record["ftp_user"]),
                clean(record["ftp_password"]),
                clean(record["ftp_directory"]),
                clean(record["config"]),
                to_int(record["gateway_id"]),
                to_int(record["parsing_id"]),
                to_int(record["file_id"]),
                clean(record["to_move"]),
                clean(record["regex_variables"]),
                clean(record["created_at_2"]),
                clean(record["updated_at_2"]),
                to_bool(record["active"]),
                clean(record["error_message"]),
                clean(record["last_modified"]),
                to_bool(record["keep_folder"]),
            )

        if site_id is not None and gateway_id is not None:
            site_links.add((site_id, gateway_id))

    with conn.cursor() as cur:
        if gateways:
            cur.executemany(
                """
                INSERT INTO gateways (
                    gateway_id, name, serial_number, transfer_protocol,
                    power_supply, installation_date, operating_team, x, y, z,
                    time_zone, geom, created_at, updated_at, provider
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
                """,
                gateways.values(),
            )

        if configs:
            cur.executemany(
                """
                INSERT INTO configs (
                    config_id, file_name, last_treatment, ftp, ftp_ip,
                    ftp_user, ftp_password, ftp_directory, payload,
                    gateway_id, parsing_id, file_id, to_move, regex_variables,
                    created_at, updated_at, active, error_message,
                    last_modified, keep_folder
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
                """,
                configs.values(),
            )

        if site_links:
            cur.executemany(
                """
                INSERT INTO site_gateways (site_id, gateway_id)
                VALUES (%s, %s);
                """,
                list(site_links),
            )

    gateway_rows = sort_rows(list(gateways.values()), ROW_ORDER["gateways"], GATEWAY_COLUMNS)
    config_rows = sort_rows(list(configs.values()), ROW_ORDER["configs"], CONFIG_COLUMNS)
    link_rows = sort_rows(list(site_links), ROW_ORDER["site_gateways"], SITE_GATEWAY_COLUMNS)

    return {
        "gateways": ExpectedTable(columns=GATEWAY_COLUMNS, rows=gateway_rows),
        "configs": ExpectedTable(columns=CONFIG_COLUMNS, rows=config_rows),
        "site_gateways": ExpectedTable(columns=SITE_GATEWAY_COLUMNS, rows=link_rows),
    }


def load_raw_measurements(conn: psycopg.Connection) -> Dict[str, ExpectedTable]:
    path = next(DATA_DIR.glob("variables_metrics_raw_data-*.csv"), None)
    if not path:
        raise FileNotFoundError("variables_metrics_raw_data-*.csv not found in /database.")

    checksum = hashlib.sha256()
    row_batch: List[Tuple] = []
    expected_rows: List[Tuple] = []

    with conn.cursor() as cur:
        for record in iter_csv(path, delimiter=",", encoding="utf-8"):
            parsed = (
                clean(record["gateway_name"]),
                clean(record["variable_name"]),
                clean(record["variable_alias"]),
                to_int(record["sensor_id"]),
                to_float(record["value"]),
                clean(record["timestamp"]),
                to_int(record["variable_id"]),
                clean(record["unit"]),
                clean(record["metric"]),
            )

            row_batch.append(parsed)
            expected_rows.append(parsed)

            if len(row_batch) >= 5000:
                cur.executemany(
                    """
                    INSERT INTO raw_measurements (
                        gateway_name, variable_name, variable_alias, sensor_id,
                        value, timestamp, variable_id, unit, metric
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
                    """,
                    row_batch,
                )
                row_batch.clear()

        if row_batch:
            cur.executemany(
                """
                INSERT INTO raw_measurements (
                    gateway_name, variable_name, variable_alias, sensor_id,
                    value, timestamp, variable_id, unit, metric
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
                """,
                    row_batch,
            )

    sort_indexes = [RAW_MEASUREMENT_COLUMNS.index(col) for col in RAW_MEASUREMENT_ORDER]
    expected_rows.sort(key=lambda row: tuple(row[idx] for idx in sort_indexes))
    for row in expected_rows:
        checksum.update(canonical_row(row))

    return {
        "raw_measurements": ExpectedTable(
            columns=RAW_MEASUREMENT_COLUMNS,
            checksum=checksum.hexdigest(),
            checksum_count=len(expected_rows),
        )
    }


# --------------------------------------------------------------------------- #
# Verification helpers
# --------------------------------------------------------------------------- #


def verify_schema(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        for table, expected_columns in EXPECTED_SCHEMA.items():
            cur.execute(
                """
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = %s
                ORDER BY ordinal_position;
                """,
                (table,),
            )
            actual = cur.fetchall()
            if actual != expected_columns:
                raise RuntimeError(
                    f"Schema mismatch for table '{table}'. "
                    f"Expected {expected_columns}, found {actual}."
                )


def verify_counts(conn: psycopg.Connection, expected: Dict[str, ExpectedTable]) -> None:
    with conn.cursor() as cur:
        for table, table_meta in expected.items():
            cur.execute(f"SELECT COUNT(*) FROM {table};")
            actual = cur.fetchone()[0]
            expected_count = table_meta.expected_count()
            if actual != expected_count:
                raise RuntimeError(
                    f"Row count mismatch for '{table}': expected {expected_count}, found {actual}."
                )


def verify_rows(conn: psycopg.Connection, expected: Dict[str, ExpectedTable]) -> None:
    for table, table_meta in expected.items():
        if table_meta.rows is None:
            continue

        order_cols = ROW_ORDER[table]
        columns = table_meta.columns
        order_clause = ", ".join(order_cols)
        select_clause = ", ".join(columns)

        with conn.cursor() as cur:
            cur.execute(
                f"SELECT {select_clause} FROM {table} ORDER BY {order_clause};"
            )
            actual = [tuple(row) for row in cur.fetchall()]

        expected_rows = table_meta.rows
        if actual != expected_rows:
            raise RuntimeError(
                f"Data mismatch for '{table}'. Verification aborted."
            )


def verify_raw_measurements(conn: psycopg.Connection, expected: Dict[str, ExpectedTable]) -> None:
    table_meta = expected.get("raw_measurements")
    if table_meta is None or table_meta.checksum is None:
        return

    select_clause = ", ".join(RAW_MEASUREMENT_COLUMNS)
    order_clause = ", ".join(RAW_MEASUREMENT_ORDER)

    digest = hashlib.sha256()
    row_count = 0

    with conn.cursor(name="raw_measurements_verify") as cur:
        cur.itersize = 10000
        cur.execute(
            f"SELECT {select_clause} FROM raw_measurements ORDER BY {order_clause};"
        )
        for row in cur:
            digest.update(canonical_row(tuple(row)))
            row_count += 1

    expected_count = table_meta.expected_count()
    if row_count != expected_count:
        raise RuntimeError(
            f"Row count mismatch for 'raw_measurements': expected {expected_count}, found {row_count}."
        )

    checksum = digest.hexdigest()
    if checksum != table_meta.checksum:
        raise RuntimeError(
            "Checksum mismatch for 'raw_measurements'. "
            "Database contents do not match the CSV extract."
        )


# --------------------------------------------------------------------------- #
# Schema + utility helpers
# --------------------------------------------------------------------------- #


def create_tables(conn: psycopg.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS projects (
            project_id integer PRIMARY KEY,
            name text,
            country text,
            client_company_name text,
            time_zone text,
            city text,
            start_date text,
            description text
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sites (
            site_id integer PRIMARY KEY,
            name text,
            type text,
            extent text,
            start_date text,
            previsional_end text,
            project_id integer REFERENCES projects(project_id),
            created_at text,
            updated_at text,
            deleted boolean,
            operating_rate double precision,
            main_site integer
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS gateways (
            gateway_id integer PRIMARY KEY,
            name text,
            serial_number text,
            transfer_protocol text,
            power_supply text,
            installation_date text,
            operating_team text,
            x double precision,
            y double precision,
            z double precision,
            time_zone text,
            geom text,
            created_at text,
            updated_at text,
            provider integer
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS configs (
            config_id integer PRIMARY KEY,
            file_name text,
            last_treatment text,
            ftp text,
            ftp_ip text,
            ftp_user text,
            ftp_password text,
            ftp_directory text,
            payload text,
            gateway_id integer REFERENCES gateways(gateway_id),
            parsing_id integer,
            file_id integer,
            to_move text,
            regex_variables text,
            created_at text,
            updated_at text,
            active boolean,
            error_message text,
            last_modified text,
            keep_folder boolean
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS site_gateways (
            site_id integer REFERENCES sites(site_id),
            gateway_id integer REFERENCES gateways(gateway_id),
            PRIMARY KEY (site_id, gateway_id)
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS raw_measurements (
            gateway_name text,
            variable_name text,
            variable_alias text,
            sensor_id integer,
            value double precision,
            timestamp text,
            variable_id integer,
            unit text,
            metric text
        );
        """
    )


def truncate_tables(conn: psycopg.Connection) -> None:
    conn.execute(
        """
        TRUNCATE TABLE
            raw_measurements,
            site_gateways,
            configs,
            gateways,
            sites,
            projects
        RESTART IDENTITY CASCADE;
        """
    )


def create_indexes(conn: psycopg.Connection) -> None:
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sites_project_id ON sites(project_id);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_configs_gateway_id ON configs(gateway_id);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_site_gateways_gateway ON site_gateways(gateway_id);")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_raw_measurements_sensor_time "
        "ON raw_measurements(sensor_id, timestamp);"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_raw_measurements_gateway "
        "ON raw_measurements(gateway_name);"
    )


# --------------------------------------------------------------------------- #
# CSV + transformation helpers
# --------------------------------------------------------------------------- #


def iter_csv(
    path: Path,
    *,
    delimiter: str,
    encoding: str,
) -> Iterable[Dict[str, str]]:
    with path.open("r", newline="", encoding=encoding) as handle:
        reader = iter(csv.reader(handle, delimiter=delimiter))
        headers = next(reader)
        unique_headers = _unique_headers(headers)

        for row in reader:
            if not row:
                continue
            # Pad row length in case of trailing delimiters
            if len(row) < len(unique_headers):
                row += [""] * (len(unique_headers) - len(row))
            yield dict(zip(unique_headers, row))


def _unique_headers(headers: Sequence[str]) -> List[str]:
    seen: Dict[str, int] = {}
    unique: List[str] = []
    for header in headers:
        header = header.strip()
        count = seen.get(header, 0)
        if count == 0:
            unique.append(header)
        else:
            unique.append(f"{header}_{count + 1}")
        seen[header] = count + 1
    return unique


def sort_rows(rows: List[Tuple], order_cols: Sequence[str], columns: Sequence[str]) -> List[Tuple]:
    if not rows:
        return []
    index_map = [columns.index(col) for col in order_cols]
    return sorted(rows, key=lambda row: tuple(row[idx] for idx in index_map))


def clean(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = value.strip()
    return value or None


def to_int(value: Optional[str]) -> Optional[int]:
    value = clean(value)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        try:
            return int(float(value.replace(",", ".")))
        except (ValueError, AttributeError):
            return None


def to_float(value: Optional[str]) -> Optional[float]:
    value = clean(value)
    if value is None:
        return None
    try:
        return float(value.replace(",", "."))
    except (ValueError, AttributeError):
        return None


def to_bool(value: Optional[str]) -> Optional[bool]:
    value = clean(value)
    if value is None:
        return None
    lowered = value.lower()
    truthy = {"true", "t", "1", "yes", "y", "vrai", "oui"}
    falsy = {"false", "f", "0", "no", "n", "faux", "non"}
    if lowered in truthy:
        return True
    if lowered in falsy:
        return False
    return None


def canonical_row(row: Sequence[object]) -> bytes:
    canonical_fields = []
    for value in row:
        if value is None:
            canonical_fields.append("null")
        elif isinstance(value, float):
            canonical_fields.append(format(value, ".15g"))
        else:
            canonical_fields.append(str(value))
    return "|".join(canonical_fields).encode("utf-8")


if __name__ == "__main__":
    main()
