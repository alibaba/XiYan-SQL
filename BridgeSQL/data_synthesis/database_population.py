"""
Code-driven Database Population.

For each validated database, use LLM-generated Python code to populate tables
with constraint-satisfying synthetic data.

Pipeline (per database, tables processed in topological order):
    1. Generate Python mock-data code via LLM (with retry)
    2. Execute the code N times to produce N rows
    3. Validate rows against SQLite schema (column-level + row-level)
    4. Fix constraint conflicts (PK dedup, FK reference alignment)
    5. Assemble final SQLite database with all tables and data

Input:
    output/data_synthesis/
    ├── valid_databases.json
    └── {db_id}/schema/mschema.json

Output (added to existing structure):
    output/data_synthesis/{db_id}/
    ├── schema/mock_code.json     (cached LLM-generated code per table)
    ├── data/{table_name}.pkl     (intermediate per-table data)
    └── {db_id}.sqlite            (final populated database)

Usage:
    python data_synthesis/database_population.py \
        --output_path output/data_synthesis/ \
        --n_rows 1000 \
        --db_root databases/
"""

import argparse
import io
import json
import os
import pickle
import random
import re
import shutil
import signal
import sqlite3
import string
import sys
from contextlib import redirect_stdout
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_synthesis.llm_utils import LLMClient
from schema_engine import Schema

# ---------------------------------------------------------------------------
# Prompt template for mock-data code generation
# ---------------------------------------------------------------------------

MOCK_CODE_PROMPT = '''**Task Overview:**
You need to analyze and mock a random data entry with python and practical packages according to the table schema.

**Instructions**
1. The python code should be surrounded by ```python and ``` and the result variable should be `row`.
2. For each column, the random data generation should be appropriate and reasonable in the scenario.

Here is a demonstration.
**Schema**
CREATE TABLE student_scores (
    pid INTEGER PRIMARY KEY, -- primary key record id
    student_name TEXT, -- full name of the student
    age INTEGER, -- student age
    score NUMERIC(10,2), -- examination score, examples: [85.50, 92.00]
    evaluation VARCHAR(10), -- evaluation of the daily performance ('excellent', 'good', 'pass', 'fail')
    is_checked BOOLEAN, -- score checked or not
    code VARCHAR(10),
    created_date DATE, -- record date
    created_ts TIMESTAMP -- record timestamp
);

**Data Mock**
```python
# Initialize the row dictionary with None
columns = ["pid", "student_name", "age", "score", "evaluation", "is_checked", "code", "created_date", "created_ts"]
row = {{col_name: None for col_name in columns}}

# Primary key pid could be a random int
import random
row['pid'] = random.randint(0, 10000)

# We use faker package for student_name
from faker import Faker
fake = Faker(locale='en_US')
row['student_name'] = fake.name()

# student age could be a random integer between 18 and 21
row['age'] = random.randint(18, 21)

# score of the examination could be normal distributed with mean 80 and std 15, with range 0-100
row['score'] = max(0, min(100, random.gauss(80, 15)))

# evaluation is the from a list of options
row['evaluation'] = random.choice(['excellent', 'good', 'pass', 'fail'])

# is_checked could be a random boolean
row['is_checked'] = random.choice([True, False])

# code may be a random letter string
row['code'] = fake.pystr(min_chars=5, max_chars=8)

# create_date and create_ts seem to represent the same time of one entry.
# Hypothesize the time of database is between 2020-1-1 and 2025-12-31
from datetime import datetime
rand_date_time = fake.date_time_between_dates(datetime_start=datetime(2020, 1, 1, 0, 0, 0), datetime_end=datetime(2025, 12, 31, 23, 59, 59))
row['created_date'] = rand_date_time.strftime('%Y-%m-%d')
row['created_ts'] = rand_date_time.strftime('%Y-%m-%d %H:%M:%S')
```

Now mock new data by the following schema.
**Schema**
{schema}

**Data Mock**
'''

CODE_EXEC_TIMEOUT = 2  # seconds


# ---------------------------------------------------------------------------
# Code execution
# ---------------------------------------------------------------------------

def _timeout_handler(signum, frame):
    raise TimeoutError("Code execution exceeded timeout")


def run_mock_code(mock_code: str, num: int):
    """Execute mock code `num` times and collect generated rows."""
    rows = []
    for _ in range(num):
        try:
            local_vars = {}
            string_io = io.StringIO()
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(CODE_EXEC_TIMEOUT)
            with redirect_stdout(string_io):
                exec(mock_code, local_vars, local_vars)
            signal.alarm(0)
            row = local_vars.get("row")
            if row is not None:
                rows.append(row)
        except Exception:
            signal.alarm(0)
            continue
    return rows


# ---------------------------------------------------------------------------
# LLM-based code generation
# ---------------------------------------------------------------------------

def generate_mock_code(mschema: Schema, table_name: str, llm: LLMClient,
                       max_retries: int = 3):
    """Generate Python mock-data code for a table via LLM.

    Returns (success: bool, code: str | None).
    """
    schema_str = mschema.single_table_omnischema(table_name)
    prompt = MOCK_CODE_PROMPT.format(schema=schema_str)

    for attempt in range(max_retries):
        model = random.choice(llm.chat_models)
        response = llm.call(model, prompt, temperature=0.7)
        if not response:
            continue

        match = re.search(r"```python(.*?)```", response, re.DOTALL)
        if not match:
            continue
        code = match.group(1).strip()

        test_rows = run_mock_code(code, 1)
        if not test_rows:
            continue

        row = test_rows[0]
        expected_cols = set(mschema.tables[table_name]["fields"].keys())
        if not isinstance(row, dict) or set(row.keys()) != expected_cols:
            continue

        return True, code

    return False, None


# ---------------------------------------------------------------------------
# Data validation and constraint fixing
# ---------------------------------------------------------------------------

def generate_unique_set(sample_data, num):
    """Generate a set of unique values matching the type of sample_data."""
    if isinstance(sample_data, str):
        chars = string.ascii_letters + string.digits
        unique = set()
        while len(unique) < num:
            unique.add("".join(random.choices(chars, k=10)))
        return list(unique)
    elif isinstance(sample_data, (int, float)):
        return list(range(num))
    else:
        raise ValueError(f"Unsupported type: {type(sample_data)}")


def init_memory_table(mschema: Schema, table_name: str):
    """Create an in-memory SQLite table (without FK constraints) for validation."""
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    ddl = mschema.tables[table_name]["ddl"]
    lines = ddl.split("\n")
    for i, line in enumerate(lines):
        if "UNIQUE NOT NULL" in line:
            lines[i] = line.replace("UNIQUE NOT NULL", "")
    lines = [l for l in lines if "FOREIGN KEY" not in l and "CONSTRAINT" not in l]
    if len(lines) > 2 and lines[-2].strip().endswith(","):
        lines[-2] = lines[-2].strip()[:-1]
    cursor.execute("\n".join(lines))
    return conn, cursor


def validate_rows(mschema: Schema, table_name: str, table_rows: list):
    """Validate rows against the table schema.

    Returns (valid_columns, valid_rows).
    """
    if not table_rows:
        return [], []

    instance = table_rows[0]
    conn, cursor = init_memory_table(mschema, table_name)
    table_info = mschema.tables[table_name]
    pks = [f for f, info in table_info["fields"].items() if info["primary_key"]]
    col_types = {f: info["type"] for f, info in table_info["fields"].items()}

    valid_columns = []

    # Validate primary keys first
    for col in pks:
        val = instance[col]
        try:
            cursor.execute(f'INSERT INTO "{table_name}" ("{col}") VALUES (?)', (val,))
            valid_columns.append(col)
            cursor.execute(f'DELETE FROM "{table_name}";')
        except sqlite3.Error:
            conn, cursor = init_memory_table(mschema, table_name)
            new_vals = generate_unique_set(val, len(table_rows))
            for i in range(len(table_rows)):
                table_rows[i][col] = new_vals[i]
            valid_columns.append(col)

    # Validate other columns
    for col, val in instance.items():
        if col in pks:
            continue
        try:
            cursor.execute(
                f'INSERT INTO "{table_name}" ("{pks[0]}", "{col}") VALUES (?, ?)',
                (instance[pks[0]],
                 json.dumps(val) if col_types.get(col) in ("JSON", "JSONB") else val),
            )
            valid_columns.append(col)
            cursor.execute(f'DELETE FROM "{table_name}";')
        except sqlite3.Error:
            cursor.close()
            conn.close()
            conn, cursor = init_memory_table(mschema, table_name)

    # Validate full rows
    valid_rows = []
    for row in table_rows:
        cols = [c for c in row if c in valid_columns]
        vals = tuple(
            json.dumps(row[c]) if col_types.get(c) in ("JSON", "JSONB") else row[c]
            for c in cols
        )
        col_str = ", ".join(f'"{c}"' for c in cols)
        placeholders = ", ".join("?" for _ in vals)
        try:
            cursor.execute(
                f'INSERT INTO "{table_name}" ({col_str}) VALUES ({placeholders})',
                vals,
            )
            valid_rows.append(row)
        except Exception:
            conn, cursor = init_memory_table(mschema, table_name)

    cursor.close()
    conn.close()
    return valid_columns, valid_rows


def fix_constraints(table_name: str, mschema: Schema, table_rows: list,
                    valid_columns: list, data_dir: str):
    """Fix PK uniqueness and FK reference constraints.

    Returns a column-oriented dict: {col_name: [values...]}.
    """
    row_num = len(table_rows)
    data = {}
    for row in table_rows:
        for k, v in row.items():
            data.setdefault(k, []).append(v)

    for col in data:
        if col not in valid_columns:
            data[col] = [None] * row_num

    # Deduplicate primary keys
    pk_list = [f for f, info in mschema.tables[table_name]["fields"].items()
               if info["primary_key"]]
    pk_tuples = set(tuple(data[pk][i] for pk in pk_list) for i in range(row_num))
    real_len = len(pk_tuples)
    if real_len != row_num:
        pk_list_data = list(pk_tuples)
        for j, col in enumerate(pk_list):
            data[col] = [t[j] for t in pk_list_data]
        for col in data:
            if col not in pk_list:
                data[col] = data[col][:real_len]

    # Deduplicate columns referenced as unique by foreign keys
    unique_cols = []
    for fk in mschema.foreign_keys:
        if table_name == fk[3] and fk[4] not in pk_list:
            unique_cols.append(fk[4])
    for col in unique_cols:
        seen = {}
        for i, val in enumerate(data[col]):
            if val not in seen:
                seen[val] = i
        remain = list(seen.values())
        if len(remain) < len(data[col]):
            for c in data:
                data[c] = [data[c][i] for i in remain]

    # Fix foreign key references
    fk_pairs = [fk for fk in mschema.foreign_keys if fk[0] == table_name]
    for fk in fk_pairs:
        _, src_col, _, ref_table, ref_col = fk
        src_vals = data[src_col]

        if ref_table == table_name:
            ref_vals = data[ref_col]
            for i in range(len(src_vals)):
                if src_vals[i] is None:
                    continue
                if i == 0:
                    src_vals[i] = None
                elif src_vals[i] not in ref_vals[:i]:
                    src_vals[i] = random.choice(ref_vals[:i])
        else:
            pkl_path = os.path.join(data_dir, f"{ref_table}.pkl")
            if os.path.exists(pkl_path):
                ref_data = pickle.load(open(pkl_path, "rb"))
                ref_vals = ref_data[ref_col]
                ref_set = set(ref_vals)
                for i in range(len(src_vals)):
                    if type(src_vals[i]) != type(ref_vals[0]) or src_vals[i] not in ref_set:
                        src_vals[i] = random.choice(ref_vals)

    return data


# ---------------------------------------------------------------------------
# Database assembly
# ---------------------------------------------------------------------------

def assemble_database(mschema: Schema, db_dir: str):
    """Create the final SQLite database from per-table pickle files.

    Returns True on success.
    """
    db_path = os.path.join(db_dir, f"{mschema.db_id}.sqlite")
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("PRAGMA foreign_keys = ON;")

    try:
        for table in mschema.topo:
            cursor.execute(mschema.tables[table]["ddl"])

        for table in mschema.topo:
            pkl_path = os.path.join(db_dir, "data", f"{table}.pkl")
            if not os.path.exists(pkl_path):
                continue
            data = pickle.load(open(pkl_path, "rb"))
            col_types = {f: info["type"]
                         for f, info in mschema.tables[table]["fields"].items()}

            for col in data:
                if col_types.get(col) in ("JSON", "JSONB"):
                    data[col] = [json.dumps(d) if d is not None else None
                                 for d in data[col]]

            columns = list(data.keys())
            rows = list(zip(*(data[c] for c in columns)))
            col_str = ", ".join(f'"{c}"' for c in columns)
            placeholders = ", ".join("?" for _ in columns)
            cursor.executemany(
                f'INSERT INTO "{table}" ({col_str}) VALUES ({placeholders})',
                rows,
            )

        conn.commit()
        return True
    except Exception as e:
        print(f"  [ERROR] Assembly failed: {e}")
        return False
    finally:
        cursor.close()
        conn.close()


# ---------------------------------------------------------------------------
# Per-database processing
# ---------------------------------------------------------------------------

def process_database(db_id: str, output_path: Path, llm: LLMClient,
                     n_rows: int = 1000):
    """Process a single database: generate data for all tables and assemble.

    Returns True on success.
    """
    db_dir = output_path / db_id
    schema_dir = db_dir / "schema"
    data_dir = db_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    mschema = Schema(db_id=db_id)
    mschema.load(str(schema_dir / "mschema.json"))

    # Load or init cached mock code
    mock_code_path = schema_dir / "mock_code.json"
    mock_code_dict = {}
    if mock_code_path.exists():
        mock_code_dict = json.load(open(mock_code_path, "r", encoding="utf-8"))

    topo = mschema.topo
    if not topo:
        print(f"  [SKIP] {db_id}: no topological order")
        return False

    for table_name in topo:
        pkl_path = data_dir / f"{table_name}.pkl"
        if pkl_path.exists():
            continue

        # Get or generate mock code
        if table_name in mock_code_dict and mock_code_dict[table_name]:
            code = mock_code_dict[table_name]
        else:
            ok, code = generate_mock_code(mschema, table_name, llm)
            if not ok:
                print(f"  [FAIL] {db_id}/{table_name}: mock code generation failed")
                return False
            mock_code_dict[table_name] = code
            with open(mock_code_path, "w", encoding="utf-8") as f:
                json.dump(mock_code_dict, f, ensure_ascii=False, indent=2)

        # Generate rows
        rows = run_mock_code(code, n_rows)
        if not rows:
            print(f"  [FAIL] {db_id}/{table_name}: code execution produced no rows")
            return False

        # Validate and fix
        valid_cols, valid_rows = validate_rows(mschema, table_name, rows)
        data_dict = fix_constraints(
            table_name, mschema, valid_rows, valid_cols, str(data_dir),
        )

        with open(pkl_path, "wb") as f:
            pickle.dump(data_dict, f)

    # Assemble final SQLite
    return assemble_database(mschema, str(db_dir))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Code-driven Database Population"
    )
    parser.add_argument(
        "--output_path", type=str, required=True,
        help="Output directory (same as filter_databases.py output)",
    )
    parser.add_argument(
        "--n_rows", type=int, default=1000,
        help="Number of rows to generate per table (default: 1000)",
    )
    parser.add_argument(
        "--db_root", type=str, default=None,
        help="If set, copy populated .sqlite files to {db_root}/bridgesql/{db_id}/",
    )
    args = parser.parse_args()

    output_path = Path(args.output_path)
    valid_db_file = output_path / "valid_databases.json"

    if not valid_db_file.exists():
        print(f"[ERROR] {valid_db_file} not found. Run filter_databases.py first.")
        return

    llm = LLMClient()
    print(f"Chat models: {llm.chat_models}")

    valid_dbs = json.load(open(valid_db_file, "r", encoding="utf-8"))

    # Track progress
    success_file = output_path / "populated_databases.json"
    failed_file = output_path / "population_failed.json"
    success_dbs = json.load(open(success_file)) if success_file.exists() else []
    failed_dbs = json.load(open(failed_file)) if failed_file.exists() else []
    done_set = set(success_dbs + failed_dbs)

    print(f"Total: {len(valid_dbs)}, Already done: {len(done_set)}")

    for db_id in tqdm(valid_dbs, desc="Populating"):
        if db_id in done_set:
            continue

        ok = process_database(db_id, output_path, llm, args.n_rows)
        if ok:
            success_dbs.append(db_id)
            with open(success_file, "w") as f:
                json.dump(success_dbs, f, ensure_ascii=False, indent=2)
        else:
            failed_dbs.append(db_id)
            with open(failed_file, "w") as f:
                json.dump(failed_dbs, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Success: {len(success_dbs)}, Failed: {len(failed_dbs)}")

    # Copy populated databases to unified db_root for RLVR training
    if args.db_root:
        db_root = Path(args.db_root) / "bridgesql"
        db_root.mkdir(parents=True, exist_ok=True)
        copied = 0
        for db_id in success_dbs:
            src = output_path / db_id / f"{db_id}.sqlite"
            if src.exists():
                dst_dir = db_root / db_id
                dst_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst_dir / f"{db_id}.sqlite")
                copied += 1
        print(f"Copied {copied} databases to {db_root}")


if __name__ == "__main__":
    main()
