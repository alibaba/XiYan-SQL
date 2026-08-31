"""
Random Baseline Database Population.

Populates database schemas using type-matched Faker generators with primary key
and foreign key constraint satisfaction, but WITHOUT semantic guidance from the
schema metadata. This serves as a controlled comparison to isolate the effect
of BridgeSQL's code-driven, LLM-guided population.

Population strategy:
    - INTEGER PK: auto-increment sequence
    - INTEGER FK: random sample from parent table's referenced column
    - INTEGER: random_int(0, 10000)
    - REAL/FLOAT: random uniform(0, 1000)
    - TEXT: guess Faker provider by column name (name/email/date etc.),
            otherwise use random word
    - Respects PK uniqueness and FK referential integrity
    - Does NOT respect semantic plausibility (e.g., age may be negative,
      correlations between columns are absent)

Input:
    output/data_synthesis/
    ├── populated_databases.json    (or valid_databases.json)
    └── {db_id}/schema/mschema.json

Output:
    {output_root}/{db_id}/
    └── {db_id}.sqlite

Usage:
    python data_synthesis/random_baseline.py \
        --data_root output/data_synthesis/ \
        --output_root output/random_baseline/ \
        --n_rows 1000
"""

import argparse
import json
import os
import random
import re
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Optional

from faker import Faker
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from schema_engine import Schema

# ---------------------------------------------------------------------------
# Faker column-name heuristics (type-matched, no semantic guidance)
# ---------------------------------------------------------------------------

FAKER_COLUMN_PATTERNS = [
    (r"(^|_)name($|_)", "name"),
    (r"(^|_)first.?name", "first_name"),
    (r"(^|_)last.?name", "last_name"),
    (r"(^|_)email($|_)", "email"),
    (r"(^|_)phone($|_|num)", "phone_number"),
    (r"(^|_)address($|_)", "address"),
    (r"(^|_)city($|_)", "city"),
    (r"(^|_)state($|_)", "state"),
    (r"(^|_)country($|_)", "country"),
    (r"(^|_)zip($|_|code)", "zipcode"),
    (r"(^|_)url($|_)|website", "url"),
    (r"(^|_)date($|_)|_at$|timestamp", "date"),
    (r"(^|_)year($|_)", "year"),
    (r"(^|_)title($|_)", "sentence"),
    (r"(^|_)desc($|_)|description", "sentence"),
    (r"(^|_)company($|_)|publisher|organization|affiliation", "company"),
    (r"(^|_)text($|_)|abstract|content|body|comment|note", "paragraph"),
]


def _guess_faker_provider(col_name: str) -> Optional[str]:
    col_lower = col_name.lower()
    for pattern, provider in FAKER_COLUMN_PATTERNS:
        if re.search(pattern, col_lower):
            return provider
    return None


# ---------------------------------------------------------------------------
# Value generation (type-matched, no semantic awareness)
# ---------------------------------------------------------------------------

def generate_value(fake: Faker, col_name: str, col_type: str, is_pk: bool,
                   row_idx: int, fk_values: Optional[List] = None):
    """Generate a single random value for a column."""
    col_type_upper = (col_type or "TEXT").upper()

    if fk_values is not None:
        if fk_values:
            return random.choice(fk_values)
        return None

    if is_pk:
        if "INT" in col_type_upper:
            return row_idx + 1
        else:
            return fake.uuid4()

    if "INT" in col_type_upper:
        if re.search(r"year", col_name.lower()):
            return random.randint(1950, 2025)
        if re.search(r"(^|_)(is_|has_|flag|active|enabled|status$)", col_name.lower()):
            return random.randint(0, 1)
        return random.randint(0, 10000)

    if any(t in col_type_upper for t in ("REAL", "FLOAT", "DOUBLE", "NUMERIC", "DECIMAL")):
        return round(random.uniform(0, 1000), 2)

    if "BOOL" in col_type_upper:
        return random.choice([0, 1])

    provider = _guess_faker_provider(col_name)
    if provider:
        try:
            return getattr(fake, provider)()
        except Exception:
            pass
    return fake.word()


# ---------------------------------------------------------------------------
# Per-database population
# ---------------------------------------------------------------------------

def populate_database(db_id: str, mschema: Schema, output_dir: str,
                      n_rows: int, seed: int) -> Dict:
    """Populate a single database with random data.

    Reads schema from mschema, creates a new SQLite database, and fills
    tables in topological order with type-matched random values.
    """
    output_path = os.path.join(output_dir, f"{db_id}.sqlite")
    if os.path.exists(output_path):
        os.remove(output_path)

    topo = mschema.topo
    if not topo:
        return {"db_id": db_id, "status": "skip", "msg": "no topological order"}

    conn = sqlite3.connect(output_path)
    cursor = conn.cursor()
    cursor.execute("PRAGMA foreign_keys = OFF;")

    # Create all tables
    for table_name in topo:
        ddl = mschema.tables[table_name].get("ddl")
        if ddl:
            try:
                cursor.execute(ddl)
            except sqlite3.Error as e:
                conn.close()
                os.remove(output_path)
                return {"db_id": db_id, "status": "error",
                        "msg": f"DDL {table_name}: {e}"}
    conn.commit()

    fake = Faker()
    Faker.seed(seed)
    random.seed(seed)

    # Build FK lookup: {(table, col) -> (ref_table, ref_col)}
    fk_map = {}
    for fk in mschema.foreign_keys:
        src_table, src_col, _, ref_table, ref_col = fk
        fk_map[(src_table, src_col)] = (ref_table, ref_col)

    total_rows = 0

    for table_name in topo:
        fields = mschema.tables[table_name]["fields"]
        col_names = list(fields.keys())
        if not col_names:
            continue

        # Preload FK parent values
        fk_parent_values = {}
        for col_name in col_names:
            key = (table_name, col_name)
            if key in fk_map:
                ref_table, ref_col = fk_map[key]
                if ref_table == table_name:
                    fk_parent_values[col_name] = "self"
                else:
                    try:
                        rows = cursor.execute(
                            f'SELECT "{ref_col}" FROM "{ref_table}"'
                        ).fetchall()
                        fk_parent_values[col_name] = [
                            r[0] for r in rows if r[0] is not None
                        ]
                    except sqlite3.Error:
                        fk_parent_values[col_name] = []

        # Generate rows
        col_str = ", ".join(f'"{c}"' for c in col_names)
        placeholders = ", ".join("?" for _ in col_names)
        insert_sql = (
            f'INSERT OR IGNORE INTO "{table_name}" ({col_str}) '
            f'VALUES ({placeholders})'
        )

        generated_pks = {}
        batch = []

        for i in range(n_rows):
            row = []
            for col_name in col_names:
                info = fields[col_name]
                is_pk = info.get("primary_key", False)
                col_type = info.get("type", "TEXT")

                fk_vals = fk_parent_values.get(col_name)
                if fk_vals == "self":
                    pk_col = [c for c in col_names
                              if fields[c].get("primary_key")]
                    if pk_col and pk_col[0] in generated_pks:
                        existing = generated_pks[pk_col[0]]
                        fk_vals = existing[:i] if existing[:i] else None
                    else:
                        fk_vals = None

                val = generate_value(fake, col_name, col_type, is_pk, i,
                                     fk_vals)

                if is_pk:
                    generated_pks.setdefault(col_name, []).append(val)

                row.append(val)
            batch.append(tuple(row))

        try:
            cursor.executemany(insert_sql, batch)
            conn.commit()
            actual = cursor.execute(
                f'SELECT COUNT(*) FROM "{table_name}"'
            ).fetchone()[0]
            total_rows += actual
        except sqlite3.Error as e:
            conn.rollback()
            conn.close()
            if os.path.exists(output_path):
                os.remove(output_path)
            return {"db_id": db_id, "status": "error",
                    "msg": f"insert {table_name}: {e}"}

    conn.close()
    return {"db_id": db_id, "status": "ok",
            "tables": len(topo), "total_rows": total_rows}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Random Baseline Database Population "
                    "(type-matched Faker, no LLM semantic guidance)"
    )
    parser.add_argument(
        "--data_root", type=str, required=True,
        help="Data synthesis output directory containing {db_id}/schema/mschema.json",
    )
    parser.add_argument(
        "--output_root", type=str, required=True,
        help="Output directory for random baseline databases",
    )
    parser.add_argument(
        "--n_rows", type=int, default=1000,
        help="Number of rows per table (default: 1000)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--db_ids", type=str, nargs="*", default=None,
        help="Specific database IDs to populate (default: all)",
    )
    parser.add_argument(
        "--db_list", type=str, default=None,
        help="JSON file containing a list of db_ids",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_root = Path(args.output_root)

    # Determine database list
    if args.db_ids:
        db_ids = args.db_ids
    elif args.db_list:
        with open(args.db_list, "r") as f:
            db_ids = json.load(f)
    else:
        db_ids = sorted([
            d for d in os.listdir(data_root)
            if (data_root / d / "schema" / "mschema.json").exists()
        ])

    print(f"Data root:      {data_root}")
    print(f"Output root:    {output_root}")
    print(f"Databases:      {len(db_ids)}")
    print(f"Rows per table: {args.n_rows}")
    print(f"Seed:           {args.seed}")
    print()

    output_root.mkdir(parents=True, exist_ok=True)

    success, failed, skipped = 0, 0, 0

    for i, db_id in enumerate(tqdm(db_ids, desc="Populating")):
        schema_path = data_root / db_id / "schema" / "mschema.json"
        if not schema_path.exists():
            skipped += 1
            continue

        mschema = Schema(db_id=db_id)
        mschema.load(str(schema_path))

        db_output_dir = output_root / db_id
        db_output_dir.mkdir(parents=True, exist_ok=True)

        result = populate_database(
            db_id, mschema, str(db_output_dir),
            args.n_rows, args.seed + i,
        )

        if result["status"] == "ok":
            success += 1
        elif result["status"] == "skip":
            skipped += 1
        else:
            failed += 1
            print(f"  [FAIL] {result['db_id']}: {result.get('msg', '')}")

    print(f"\nDone: {success} success, {failed} failed, {skipped} skipped")
    print(f"Output: {output_root}")


if __name__ == "__main__":
    main()
