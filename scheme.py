import os, json, csv
from pathlib import Path
import psycopg  # pip install psycopg[binary]

DSN = os.environ["POSTGRES_DSN"]
OUT = Path("DB_schema")
OUT.mkdir(exist_ok=True)

with psycopg.connect(DSN) as conn, conn.cursor() as cur:
    cur.execute("""
        SELECT table_schema, table_name
        FROM information_schema.tables
        WHERE table_schema NOT IN ('pg_catalog','information_schema')
        ORDER BY table_schema, table_name
    """)
    tables = cur.fetchall()

    ddl_rows = []
    for schema, table in tables:
        qualified = f"{schema}.{table}"
        cur.execute("""
            SELECT column_name, data_type, is_nullable,
                   coalesce(col_description((table_schema||'.'||table_name)::regclass::oid,
                                            ordinal_position),'') AS description
            FROM information_schema.columns
            WHERE table_schema=%s AND table_name=%s
            ORDER BY ordinal_position
        """, (schema, table))
        cols = cur.fetchall()
        col_defs = [f'"{c[0]}" {c[1]} {"NULL" if c[2]=="YES" else "NOT NULL"}' for c in cols]
        ddl = f"CREATE TABLE {qualified} (\n  " + ",\n  ".join(col_defs) + "\n);"
        ddl_rows.append((qualified, ddl))

        cur.execute(f'SELECT * FROM {qualified} LIMIT 10;')
        headers = [d[0] for d in cur.description]
        sample = [dict(zip(headers, row)) for row in cur.fetchall()]
        json_payload = {
            "table": qualified,
            "columns": [
                {"name": c[0], "type": c[1], "nullable": c[2]=="YES", "description": c[3]}
                for c in cols
            ],
            "sample_rows": sample,
        }
        json_path = OUT / f"{schema}.{table}.json"
        json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

with (OUT / "DDL.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["table", "ddl"])
    writer.writerows(ddl_rows)
print("Schema dumped to DB_schema/")
