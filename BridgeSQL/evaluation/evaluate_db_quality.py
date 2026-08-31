"""Evaluate populated databases with the eight reported quality metrics.

The metrics cover Data Realism (4 metrics) and SQL Execution Quality
(4 metrics).  This entry point evaluates one population method at a time.  The
fixed three-system comparison in ``docs/database_quality.md`` additionally pairs
the systems on their joint common support; here the corresponding single-method
sample size is ``min(1000, local support)``.

All axes use a larger-is-more orientation for reporting; no single proxy is a
universal measure of semantic correctness.

Data Realism:
    1. Non-key Inter-column Correlation — tables with a non-trivial numeric association
    2. Strict Numeric Range Validity    — values satisfying schema-implied numeric ranges
    3. Categorical Realism              — repeated low-cardinality text columns
    4. Entropy Profile Diversity        — IQR of normalized column entropies

SQL Execution Quality:
    5. Non-empty Result Rate       — fraction of SQL queries returning >= 1 row
    6. Result Distinctiveness      — fraction of queries producing unique result sets
    7. WHERE Clause Reasonableness — fraction of WHERE queries filtering a proper subset
    8. JOIN Non-empty Rate         — fraction of JOIN queries returning >= 1 row

Input:
    Populated databases:
        {db_root}/{db_id}/{db_id}.sqlite

    SQL queries (one of the following):
        {data_root}/{db_id}/questions/verified.json
        {data_root}/{db_id}/questions/peer_reviews_with_cot_verified.json
        {data_root}/{db_id}/questions/questions.json

Output:
    Prints per-database and aggregate metrics to stdout.
    Optionally saves results to a JSON file.

Usage:
    # Evaluate BridgeSQL populated databases
    python evaluation/evaluate_db_quality.py \
        --db_root output/data_synthesis/ \
        --data_root output/data_synthesis/ \
        --output results/db_quality.json

    # Evaluate with separate db and data directories
    python evaluation/evaluate_db_quality.py \
        --db_root databases/bridgesql/ \
        --data_root output/data_synthesis/ \
        --db_ids db1 db2 db3

    # Evaluate only Data Realism (no SQL needed)
    python evaluation/evaluate_db_quality.py \
        --db_root output/data_synthesis/ \
        --semantic_only
"""

import argparse
import json
import math
import os
import re
import sqlite3
import sys
import time
from collections import Counter

import numpy as np


# ---------------------------------------------------------------------------
# Shared schema and sampling helpers
# ---------------------------------------------------------------------------

SAMPLE_LIMIT = 1000
MIN_SUPPORT = 10
NUMERIC_TYPE_MARKERS = ("INT", "REAL", "NUM", "DEC", "DOUBLE", "FLOAT")
TEXT_TYPE_MARKERS = ("TEXT", "CHAR", "CLOB")


def quote_identifier(name):
    """Quote one SQLite identifier."""
    return '"' + name.replace('"', '""') + '"'


def user_tables(conn):
    """Return user-defined table names in deterministic order."""
    return [
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
    ]


def table_columns(conn, table):
    """Return ``PRAGMA table_info`` rows for one table."""
    return list(conn.execute(f"PRAGMA table_info({quote_identifier(table)})"))


def key_columns(conn, table, columns=None):
    """Return declared PK and child-side FK columns, matched case-insensitively."""
    columns = columns if columns is not None else table_columns(conn, table)
    keys = {row[1].casefold() for row in columns if row[5]}
    keys.update(
        row[3].casefold()
        for row in conn.execute(f"PRAGMA foreign_key_list({quote_identifier(table)})")
    )
    return keys


def has_declared_type(declared_type, markers):
    declared = (declared_type or "").upper()
    return any(marker in declared for marker in markers)


def first_non_null_values(conn, table, column, sample_limit=SAMPLE_LIMIT):
    """Read the first non-null values in deterministic SQLite rowid order."""
    query = (
        f"SELECT {quote_identifier(column)} FROM {quote_identifier(table)} "
        f"WHERE {quote_identifier(column)} IS NOT NULL "
        "ORDER BY rowid LIMIT ?"
    )
    return [row[0] for row in conn.execute(query, (sample_limit,))]


def numeric_or_nan(value):
    """Convert a SQLite numeric value to float; treat other values as missing."""
    if value is None or isinstance(value, (bytes, bytearray)):
        return np.nan
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return np.nan
    return number if math.isfinite(number) else np.nan


def _typed_value(value):
    """Return a stable, JSON-serializable SQLite value representation."""
    if value is None:
        return ("null", None)
    if isinstance(value, bytes):
        return ("bytes", value.hex())
    return (type(value).__name__, repr(value))


# ---------------------------------------------------------------------------
# Metric 1: Non-key Inter-column Correlation
# ---------------------------------------------------------------------------

def compute_non_key_inter_column_correlation(conn, sample_limit=SAMPLE_LIMIT):
    """Fraction of eligible tables containing a numeric pair with ``|r| > .3``.

    Declared primary keys and child-side foreign keys are excluded.  Each table
    uses its first ``min(sample_limit, row_count)`` rows.  A table with no finite
    Pearson pair remains in the denominator with indicator zero.
    """
    eligible_tables = 0
    correlated_tables = 0

    for table in user_tables(conn):
        columns = table_columns(conn, table)
        excluded = key_columns(conn, table, columns)
        numeric_columns = [
            row[1]
            for row in columns
            if row[1].casefold() not in excluded
            and has_declared_type(row[2], NUMERIC_TYPE_MARKERS)
        ]
        if len(numeric_columns) < 2:
            continue

        row_count = conn.execute(
            f"SELECT COUNT(*) FROM {quote_identifier(table)}"
        ).fetchone()[0]
        paired_size = min(sample_limit, row_count)
        if paired_size < MIN_SUPPORT:
            continue

        selected = ", ".join(quote_identifier(name) for name in numeric_columns)
        try:
            rows = conn.execute(
                f"SELECT {selected} FROM {quote_identifier(table)} "
                "ORDER BY rowid LIMIT ?",
                (paired_size,),
            ).fetchall()
        except sqlite3.Error:
            continue

        eligible_tables += 1
        data = np.asarray(
            [[numeric_or_nan(value) for value in row] for row in rows],
            dtype=float,
        )
        has_significant_pair = False
        for left in range(len(numeric_columns)):
            for right in range(left + 1, len(numeric_columns)):
                left_values = data[:, left]
                right_values = data[:, right]
                mask = np.isfinite(left_values) & np.isfinite(right_values)
                if int(mask.sum()) < 2:
                    continue
                left_valid = left_values[mask]
                right_valid = right_values[mask]
                if np.ptp(left_valid) == 0 or np.ptp(right_valid) == 0:
                    continue
                correlation = float(np.corrcoef(left_valid, right_valid)[0, 1])
                if math.isfinite(correlation) and abs(correlation) > 0.3:
                    has_significant_pair = True
                    break
            if has_significant_pair:
                break
        correlated_tables += int(has_significant_pair)

    if eligible_tables == 0:
        return None
    return correlated_tables / eligible_tables


# ---------------------------------------------------------------------------
# Metric 2: Strict Numeric Range Validity
# ---------------------------------------------------------------------------

PLAIN_DECIMAL_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")
SEASON_YEAR_RE = re.compile(r"^(\d{4})(?:\s*[-/]\s*(\d{2}|\d{4}))?$")
BOOLEAN_COLUMN_NAMES = {
    "injured",
    "personal_best",
    "affiliated_to_cens",
    "active",
    "enabled",
    "disabled",
}
COUNT_COLUMN_NAMES = {
    "capacity",
    "count",
    "quantity",
    "quantity_available",
    "attendance",
    "citations",
    "views",
    "clicks",
    "conversions",
    "fleet_size",
    "seats",
    "number_of_participants",
    "number_of_doors",
    "helpful_votes",
    "unhelpful_votes",
    "baggage_count",
    "stock_level",
    "low_stock_threshold",
    "reorder_point",
    "seating_capacity",
}
DURATION_COLUMN_NAMES = {
    "duration_months",
    "delay_duration",
    "warranty_period",
    "contract_length",
}
AMOUNT_COLUMN_NAMES = {
    "cost",
    "entry_fee",
    "contract_value",
    "msrp",
    "revenue",
    "compensation",
}
COUNT_SUFFIX_RE = re.compile(
    r"_(?:count|quantity|attendance|citations|seats|views|clicks|conversions|votes)$"
)


def normalized_column_name(name):
    return re.sub(r"[^a-z0-9]+", "_", name.casefold()).strip("_")


def infer_numeric_range_rule(column_name, declared_type):
    """Infer the fixed range rule for a declared column."""
    name = normalized_column_name(column_name)
    tokens = re.split(r"[^a-z0-9]+", column_name.casefold())
    range_numeric_type = has_declared_type(declared_type, ("INT", "REAL", "NUM"))

    if name.startswith(("is_", "has_", "can_", "should_")):
        return "boolean"
    if name in BOOLEAN_COLUMN_NAMES:
        return "boolean"

    if (
        name.endswith("_pct")
        or name.startswith("pct_")
        or {"percent", "percentage"} & set(tokens)
    ):
        return "proportion"
    if name.endswith("_rate") and not name.endswith("_rating"):
        return "rate"
    if (
        (name == "year" and range_numeric_type)
        or (name.endswith("_year") and not name.endswith("_years"))
        or name.startswith("year_of_")
    ):
        return "year"
    if name == "age" or name.endswith("_age"):
        return "age"
    if name == "month":
        return "month"
    if name == "day":
        return "day"
    if name == "latitude" or name.endswith("_latitude"):
        return "latitude"
    if name == "longitude" or name.endswith("_longitude"):
        return "longitude"
    if name != "time" and name.endswith("_time") and range_numeric_type:
        return "clock"
    if name in COUNT_COLUMN_NAMES or COUNT_SUFFIX_RE.search(name):
        return "nonnegative_count"
    if name.endswith("_capacity"):
        if has_declared_type(declared_type, ("INT",)):
            return "nonnegative_count"
        return "nonnegative_measurement"
    if name in DURATION_COLUMN_NAMES:
        return "nonnegative_duration"
    if "date" in tokens or "time" in tokens:
        return None
    if (
        ("price" in tokens and "difference" not in tokens)
        or name.endswith("_amount")
        or name in AMOUNT_COLUMN_NAMES
    ):
        return "nonnegative_amount"
    return None


def parse_plain_number(value):
    """Parse SQLite numbers and unformatted plain-decimal strings only."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        return number if math.isfinite(number) else None
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not PLAIN_DECIMAL_RE.fullmatch(stripped):
        return None
    try:
        number = float(stripped)
    except (ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def valid_year(value):
    """Validate scalar years and the documented consecutive season-year form."""
    if isinstance(value, str):
        match = SEASON_YEAR_RE.fullmatch(value.strip())
        if match:
            start = int(match.group(1))
            end_text = match.group(2)
            if end_text is None:
                return 1800 <= start <= 2100
            end = int(end_text)
            if len(end_text) == 2:
                end = (start // 100) * 100 + end
            return 1800 <= start <= 2100 and 1800 <= end <= 2100 and end in {
                start,
                start + 1,
            }
    number = parse_plain_number(value)
    return number is not None and number.is_integer() and 1800 <= number <= 2100


def value_satisfies_range(value, rule):
    if rule == "year":
        return valid_year(value)

    number = parse_plain_number(value)
    if number is None:
        return False
    if rule == "month":
        return number.is_integer() and 1 <= number <= 12
    if rule == "day":
        return number.is_integer() and 1 <= number <= 31
    if rule == "age":
        return 0 <= number <= 130
    if rule == "boolean":
        return number in {0.0, 1.0}
    if rule == "clock":
        return (
            number.is_integer()
            and 0 <= number <= 2359
            and int(number) % 100 < 60
        )
    if rule == "proportion":
        return 0 <= number <= 1
    if rule == "rate":
        return 0 <= number <= 100
    if rule == "latitude":
        return -90 <= number <= 90
    if rule == "longitude":
        return -180 <= number <= 180
    if rule == "nonnegative_count":
        return number.is_integer() and number >= 0
    if rule in {
        "nonnegative_measurement",
        "nonnegative_duration",
        "nonnegative_amount",
    }:
        return number >= 0
    raise ValueError(f"Unknown numeric range rule: {rule}")


def compute_strict_numeric_range_validity(conn, sample_limit=SAMPLE_LIMIT):
    """Mean per-column validity under conservative schema-implied ranges."""
    column_scores = []
    for table in user_tables(conn):
        columns = table_columns(conn, table)
        excluded = key_columns(conn, table, columns)
        for column in columns:
            name, declared_type = column[1], column[2]
            if name.casefold() in excluded:
                continue
            rule = infer_numeric_range_rule(name, declared_type)
            if rule is None:
                continue
            try:
                values = first_non_null_values(conn, table, name, sample_limit)
            except sqlite3.Error:
                continue
            if len(values) < MIN_SUPPORT:
                continue
            valid = sum(value_satisfies_range(value, rule) for value in values)
            column_scores.append(valid / len(values))

    return float(np.mean(column_scores)) if column_scores else None


# ---------------------------------------------------------------------------
# Metric 3: Categorical Realism
# ---------------------------------------------------------------------------

def compute_categorical_realism(conn, sample_limit=SAMPLE_LIMIT):
    """Fraction of eligible text columns with cardinality ratio below ``0.1``."""
    indicators = []
    for table in user_tables(conn):
        columns = table_columns(conn, table)
        excluded = key_columns(conn, table, columns)
        for column in columns:
            name, declared_type = column[1], column[2]
            if name.casefold() in excluded:
                continue
            if not has_declared_type(declared_type, TEXT_TYPE_MARKERS):
                continue
            try:
                values = first_non_null_values(conn, table, name, sample_limit)
            except sqlite3.Error:
                continue
            if len(values) < MIN_SUPPORT:
                continue
            distinct = len({_typed_value(value) for value in values})
            indicators.append(float(distinct / len(values) < 0.1))

    return float(np.mean(indicators)) if indicators else None


# ---------------------------------------------------------------------------
# Metric 4: Entropy Profile Diversity
# ---------------------------------------------------------------------------

def compute_entropy_profile_diversity(conn, sample_limit=SAMPLE_LIMIT):
    """IQR of non-key column entropies normalized by ``log2(sample size)``."""
    normalized_entropies = []
    for table in user_tables(conn):
        columns = table_columns(conn, table)
        excluded = key_columns(conn, table, columns)
        for column in columns:
            name = column[1]
            if name.casefold() in excluded:
                continue
            try:
                values = first_non_null_values(conn, table, name, sample_limit)
            except sqlite3.Error:
                continue
            if len(values) < MIN_SUPPORT:
                continue
            counts = Counter(_typed_value(value) for value in values)
            probabilities = [count / len(values) for count in counts.values()]
            entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)
            normalized_entropies.append(entropy / math.log2(len(values)))

    if not normalized_entropies:
        return None
    lower, upper = np.quantile(
        np.asarray(normalized_entropies, dtype=float),
        [0.25, 0.75],
        method="linear",
    )
    return float(upper - lower)


# ---------------------------------------------------------------------------
# SQL Execution helpers
# ---------------------------------------------------------------------------

SQL_EXEC_TIMEOUT = 3  # seconds


def execute_sql(conn, sql, timeout=SQL_EXEC_TIMEOUT):
    """Execute SQL with timeout. Returns (rows, error)."""
    started = time.monotonic()

    def interrupt():
        return int(time.monotonic() - started > timeout)

    try:
        conn.set_progress_handler(interrupt, 1000)
        return conn.execute(sql).fetchall(), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    finally:
        conn.set_progress_handler(None, 0)


def get_main_table(sql):
    """Extract the first simple unquoted or quoted identifier after ``FROM``."""
    match = re.search(
        r"\bFROM\s+(?:\[([A-Za-z_]\w*)\]|\"([A-Za-z_]\w*)\"|"
        r"`([A-Za-z_]\w*)`|([A-Za-z_]\w*))",
        sql,
        re.IGNORECASE,
    )
    if not match:
        return None
    return next(group for group in match.groups() if group is not None)


def has_where(sql):
    return bool(re.search(r'\bWHERE\b', sql, re.IGNORECASE))


def has_join(sql):
    return bool(re.search(r'\bJOIN\b', sql, re.IGNORECASE))


def canonical_result_set(rows):
    """Return a comparable row-set key, ignoring row order and duplicates."""
    unique_rows = {
        tuple(_typed_value(value) for value in row)
        for row in rows
    }
    return tuple(sorted(unique_rows))


def compute_sql_execution_metrics(conn, sqls, timeout=SQL_EXEC_TIMEOUT):
    """Compute all four SQL metrics in one pass with fixed denominators.

    Query classes are determined before execution.  Errors and timeouts remain
    in every applicable denominator and therefore count as failures.
    """
    sqls = list(sqls)
    total_queries = len(sqls)
    nonempty = 0
    distinct_result_sets = set()
    where_total = 0
    where_reasonable = 0
    join_total = 0
    join_nonempty = 0

    table_counts = {}
    for table in user_tables(conn):
        table_counts[table.casefold()] = conn.execute(
            f"SELECT COUNT(*) FROM {quote_identifier(table)}"
        ).fetchone()[0]

    for sql in sqls:
        query_has_where = has_where(sql)
        query_has_join = has_join(sql)
        where_total += int(query_has_where)
        join_total += int(query_has_join)

        rows, error = execute_sql(conn, sql, timeout=timeout)
        if error is not None:
            continue

        if rows:
            nonempty += 1
        distinct_result_sets.add(canonical_result_set(rows))
        join_nonempty += int(query_has_join and bool(rows))

        if query_has_where:
            main_table = get_main_table(sql)
            source_size = table_counts.get(main_table.casefold(), 0) if main_table else 0
            if source_size:
                ratio = len(rows) / source_size
                where_reasonable += int(0 < ratio < 1)

    return {
        "nonempty_result_rate": (
            nonempty / total_queries if total_queries else None
        ),
        "result_distinctiveness": (
            len(distinct_result_sets) / total_queries if total_queries else None
        ),
        "where_reasonableness": (
            where_reasonable / where_total if where_total else None
        ),
        "join_nonempty_rate": (
            join_nonempty / join_total if join_total else None
        ),
    }


# Backward-compatible single-metric helpers.  ``evaluate_database`` uses the
# one-pass function above so a query is not executed four times.
def compute_nonempty_rate(conn, sqls):
    return compute_sql_execution_metrics(conn, sqls)["nonempty_result_rate"]


def compute_result_distinctiveness(conn, sqls):
    return compute_sql_execution_metrics(conn, sqls)["result_distinctiveness"]


def compute_where_reasonableness(conn, sqls):
    return compute_sql_execution_metrics(conn, sqls)["where_reasonableness"]


def compute_join_nonempty_rate(conn, sqls):
    return compute_sql_execution_metrics(conn, sqls)["join_nonempty_rate"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

QUESTION_FILES = [
    "questions/verified.json",
    "questions/peer_reviews_with_cot_verified.json",
    "questions/questions.json",
]


def load_sqls(data_dir, db_id, accepted_only=True):
    """Load SQL queries from the first available question file.

    If accepted_only=True (default), only load queries with
    review_status='accepted' when that field is present.
    """
    for qf in QUESTION_FILES:
        path = os.path.join(data_dir, db_id, qf)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            sqls = []
            for item in data:
                if accepted_only and "review_status" in item:
                    if item["review_status"] != "accepted":
                        continue
                sql = item.get("sql", "").strip()
                if sql:
                    sqls.append(sql)
            return sqls, os.path.basename(qf)
    return [], None


def find_sqlite(db_root, db_id):
    """Find the .sqlite file for a database."""
    candidates = [
        os.path.join(db_root, db_id, f"{db_id}.sqlite"),
        os.path.join(db_root, db_id, "database.sqlite"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c

    db_dir = os.path.join(db_root, db_id)
    if os.path.isdir(db_dir):
        for f in os.listdir(db_dir):
            if f.endswith(".sqlite"):
                return os.path.join(db_dir, f)
    return None


# ---------------------------------------------------------------------------
# Per-database evaluation
# ---------------------------------------------------------------------------

def evaluate_database(db_root, data_root, db_id, semantic_only=False,
                      accepted_only=True):
    """Evaluate all 8 metrics for a single database.

    Returns a dict of metric_name -> value (or None if not applicable).
    """
    sqlite_path = find_sqlite(db_root, db_id)
    if not sqlite_path:
        return None

    conn = sqlite3.connect(sqlite_path)
    results = {}

    # Data Realism metrics
    results["non_key_inter_column_correlation"] = (
        compute_non_key_inter_column_correlation(conn)
    )
    results["strict_numeric_range_validity"] = compute_strict_numeric_range_validity(conn)
    results["categorical_realism"] = compute_categorical_realism(conn)
    results["entropy_profile_diversity"] = compute_entropy_profile_diversity(conn)

    # SQL Execution Quality metrics
    if not semantic_only:
        data_dir = data_root or db_root
        sqls, source_file = load_sqls(data_dir, db_id, accepted_only)

        if sqls:
            results["_sql_count"] = len(sqls)
            results["_sql_source"] = source_file
            results.update(compute_sql_execution_metrics(conn, sqls))
        else:
            results["_sql_count"] = 0
            results["_sql_source"] = None

    conn.close()
    return results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

METRIC_NAMES = [
    "non_key_inter_column_correlation",
    "strict_numeric_range_validity",
    "categorical_realism",
    "entropy_profile_diversity",
    "nonempty_result_rate",
    "result_distinctiveness",
    "where_reasonableness",
    "join_nonempty_rate",
]

METRIC_LABELS = {
    "non_key_inter_column_correlation": "Non-key Inter-column Correlation",
    "strict_numeric_range_validity": "Strict Numeric Range Validity",
    "categorical_realism": "Categorical Realism",
    "entropy_profile_diversity": "Entropy Profile Diversity",
    "nonempty_result_rate": "Non-empty Result Rate",
    "result_distinctiveness": "Result Distinctiveness",
    "where_reasonableness": "WHERE Reasonableness",
    "join_nonempty_rate": "JOIN Non-empty Rate",
}


def aggregate_results(all_results):
    """Compute aggregate statistics across all databases."""
    agg = {}
    for metric in METRIC_NAMES:
        values = [r[metric] for r in all_results.values()
                  if r and metric in r and r[metric] is not None]
        if values:
            agg[metric] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "median": np.median(values),
                "min": np.min(values),
                "max": np.max(values),
                "count": len(values),
            }
    return agg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate populated database quality (8 metrics)"
    )
    parser.add_argument(
        "--db_root", type=str, required=True,
        help="Root directory containing {db_id}/{db_id}.sqlite files",
    )
    parser.add_argument(
        "--data_root", type=str, default=None,
        help="Root directory containing {db_id}/questions/*.json "
             "(defaults to --db_root)",
    )
    parser.add_argument(
        "--db_ids", type=str, nargs="*", default=None,
        help="Specific database IDs to evaluate (default: all in db_root)",
    )
    parser.add_argument(
        "--db_list", type=str, default=None,
        help="JSON file containing a list of db_ids to evaluate",
    )
    parser.add_argument(
        "--semantic_only", action="store_true",
        help="Only compute Data Realism metrics (no SQL needed)",
    )
    parser.add_argument(
        "--all_sqls", action="store_true",
        help="Include all SQLs (not just accepted ones) for execution metrics",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Save results to JSON file",
    )
    args = parser.parse_args()

    db_root = args.db_root
    data_root = args.data_root or db_root

    # Determine which databases to evaluate
    if args.db_ids:
        db_ids = args.db_ids
    elif args.db_list:
        with open(args.db_list, "r") as f:
            db_ids = json.load(f)
    else:
        db_ids = sorted([
            d for d in os.listdir(db_root)
            if os.path.isdir(os.path.join(db_root, d))
            and not d.startswith(".")
        ])

    print(f"Database root:  {db_root}")
    print(f"Data root:      {data_root}")
    print(f"Databases:      {len(db_ids)}")
    print(f"Semantic only:  {args.semantic_only}")
    print(f"Accepted only:  {not args.all_sqls}")
    print()

    # Evaluate each database
    all_results = {}
    skipped = 0

    for i, db_id in enumerate(db_ids):
        sys.stdout.write(f"\r  Evaluating [{i+1}/{len(db_ids)}] {db_id[:50]:<50}")
        sys.stdout.flush()

        result = evaluate_database(
            db_root, data_root, db_id,
            semantic_only=args.semantic_only,
            accepted_only=not args.all_sqls,
        )

        if result is None:
            skipped += 1
            continue

        all_results[db_id] = result

    print(f"\r  Evaluated {len(all_results)} databases, skipped {skipped}    ")

    # Aggregate
    agg = aggregate_results(all_results)

    # Print report
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  Database Quality Report  ({len(all_results)} databases)")
    print(sep)

    print(f"\n  {'Metric':<28} {'Mean':>8} {'Std':>8} {'Med':>8} {'N':>5}")
    print(f"  {'-'*28} {'-'*8} {'-'*8} {'-'*8} {'-'*5}")

    for i, metric in enumerate(METRIC_NAMES):
        if args.semantic_only and i >= 4:
            break
        label = METRIC_LABELS[metric]
        if metric in agg:
            a = agg[metric]
            print(f"  {label:<28} {a['mean']:>8.4f} {a['std']:>8.4f} "
                  f"{a['median']:>8.4f} {a['count']:>5}")
        else:
            print(f"  {label:<28} {'N/A':>8}")

        if i == 3:
            print(f"  {'-'*28} {'-'*8} {'-'*8} {'-'*8} {'-'*5}")

    # Total SQL stats
    if not args.semantic_only:
        total_sqls = sum(
            r.get("_sql_count", 0) for r in all_results.values() if r
        )
        dbs_with_sql = sum(
            1 for r in all_results.values()
            if r and r.get("_sql_count", 0) > 0
        )
        print(f"\n  Total SQL queries: {total_sqls} across {dbs_with_sql} databases")

    print(f"\n{sep}\n")

    # Save results
    if args.output:
        output_data = {
            "config": {
                "db_root": db_root,
                "data_root": data_root,
                "n_databases": len(all_results),
                "semantic_only": args.semantic_only,
                "accepted_only": not args.all_sqls,
            },
            "aggregate": {
                k: {sk: round(sv, 6) for sk, sv in v.items()}
                for k, v in agg.items()
            },
            "per_database": {
                db_id: {
                    k: round(v, 6) if isinstance(v, float) else v
                    for k, v in result.items()
                }
                for db_id, result in all_results.items()
            },
        }
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"  Results saved to {args.output}\n")


if __name__ == "__main__":
    main()
