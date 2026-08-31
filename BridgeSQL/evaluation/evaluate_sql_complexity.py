"""
Evaluate SQL complexity metrics for curated datasets.

Measures SQL complexity across multiple dimensions to assess whether
peer review preserves syntactic diversity. All metrics are computed
via regex pattern matching on the SQL string.

Metrics:
    - Mean Length        : Average character count of SQL queries
    - JOIN               : Queries containing any JOIN clause
    - GROUP BY           : Queries with aggregation grouping
    - Subquery           : Queries containing nested SELECT
    - HAVING             : Queries with post-aggregation filtering
    - CTE                : Queries using Common Table Expressions
    - Window Function    : Queries using window/analytic functions

Input:
    SQL queries from one of:
        {data_root}/{db_id}/questions/verified.json
        {data_root}/{db_id}/questions/peer_reviews_with_cot_verified.json
        {data_root}/{db_id}/questions/questions.json

Output:
    Prints complexity statistics to stdout.
    Optionally saves results to a JSON file.

Usage:
    # Analyze BridgeSQL curated data (accepted only)
    python evaluation/evaluate_sql_complexity.py \
        --data_root output/data_synthesis/

    # Analyze all SQLs (before filtering)
    python evaluation/evaluate_sql_complexity.py \
        --data_root output/data_synthesis/ \
        --all_sqls

    # Compare multiple datasets side by side
    python evaluation/evaluate_sql_complexity.py \
        --data_root output/data_synthesis/ \
        --compare path/to/other/data_root/

    # Analyze specific databases
    python evaluation/evaluate_sql_complexity.py \
        --data_root output/data_synthesis/ \
        --db_list output/data_synthesis/populated_databases.json
"""

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path


# ---------------------------------------------------------------------------
# SQL complexity patterns (case-insensitive regex)
# ---------------------------------------------------------------------------

COMPLEXITY_PATTERNS = {
    "JOIN":             r"\bJOIN\b",
    "GROUP BY":         r"\bGROUP\s+BY\b",
    "Subquery":         r"\(\s*SELECT\b",
    "HAVING":           r"\bHAVING\b",
    "CTE":              r"\bWITH\b(?!\s+ROLLUP)",
    "Window Function":  r"\bOVER\s*\(",
}

COMPILED_PATTERNS = {k: re.compile(v, re.IGNORECASE)
                     for k, v in COMPLEXITY_PATTERNS.items()}

# Extended patterns for detailed analysis
EXTENDED_PATTERNS = {
    "ORDER BY":         r"\bORDER\s+BY\b",
    "LIMIT":            r"\bLIMIT\b",
    "DISTINCT":         r"\bDISTINCT\b",
    "UNION":            r"\bUNION\b",
    "INTERSECT":        r"\bINTERSECT\b",
    "EXCEPT":           r"\bEXCEPT\b",
    "EXISTS":           r"\bEXISTS\s*\(",
    "CASE":             r"\bCASE\b",
    "LIKE":             r"\bLIKE\b",
    "BETWEEN":          r"\bBETWEEN\b",
    "IN (subquery)":    r"\bIN\s*\(\s*SELECT\b",
    "LEFT JOIN":        r"\bLEFT\s+(?:OUTER\s+)?JOIN\b",
    "COUNT":            r"\bCOUNT\s*\(",
    "SUM":              r"\bSUM\s*\(",
    "AVG":              r"\bAVG\s*\(",
    "MAX":              r"\bMAX\s*\(",
    "MIN":              r"\bMIN\s*\(",
}

COMPILED_EXTENDED = {k: re.compile(v, re.IGNORECASE)
                     for k, v in EXTENDED_PATTERNS.items()}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

QUESTION_FILES = [
    "questions/verified.json",
    "questions/peer_reviews_with_cot_verified.json",
    "questions/questions.json",
]


def load_sqls(data_root, db_ids=None, accepted_only=True):
    """Load SQL queries from all databases under data_root.

    Returns list of (db_id, sql) tuples.
    """
    if db_ids is None:
        db_ids = sorted([
            d for d in os.listdir(data_root)
            if os.path.isdir(os.path.join(data_root, d))
            and not d.startswith(".")
        ])

    all_sqls = []
    for db_id in db_ids:
        for qf in QUESTION_FILES:
            path = os.path.join(data_root, db_id, qf)
            if not os.path.exists(path):
                continue

            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for item in data:
                if accepted_only and "review_status" in item:
                    if item["review_status"] != "accepted":
                        continue
                sql = item.get("sql", "").strip()
                if sql:
                    all_sqls.append((db_id, sql))
            break

    return all_sqls


# ---------------------------------------------------------------------------
# Complexity analysis
# ---------------------------------------------------------------------------

def analyze_complexity(sqls, label="Dataset"):
    """Analyze SQL complexity for a list of (db_id, sql) tuples.

    Returns a dict of metric results.
    """
    n = len(sqls)
    if n == 0:
        return {"label": label, "count": 0}

    sql_texts = [s[1] for s in sqls]

    # Mean length
    lengths = [len(sql) for sql in sql_texts]
    mean_len = sum(lengths) / n

    # Core pattern matching
    core_counts = {}
    for name, pattern in COMPILED_PATTERNS.items():
        count = sum(1 for sql in sql_texts if pattern.search(sql))
        core_counts[name] = count

    # Extended pattern matching
    ext_counts = {}
    for name, pattern in COMPILED_EXTENDED.items():
        count = sum(1 for sql in sql_texts if pattern.search(sql))
        ext_counts[name] = count

    # JOIN count distribution
    join_counts = [len(re.findall(r"\bJOIN\b", sql, re.IGNORECASE))
                   for sql in sql_texts]
    subq_counts = [len(re.findall(r"\(\s*SELECT\b", sql, re.IGNORECASE))
                   for sql in sql_texts]

    # SQL source distribution (if available)
    source_counter = Counter()
    for db_id, sql in sqls:
        source_counter[db_id] += 1

    return {
        "label": label,
        "count": n,
        "n_databases": len(source_counter),
        "mean_length": mean_len,
        "length_stats": {
            "min": min(lengths),
            "p25": sorted(lengths)[n // 4],
            "median": sorted(lengths)[n // 2],
            "p75": sorted(lengths)[3 * n // 4],
            "max": max(lengths),
        },
        "core_metrics": {
            name: {"count": count, "rate": count / n}
            for name, count in core_counts.items()
        },
        "extended_metrics": {
            name: {"count": count, "rate": count / n}
            for name, count in ext_counts.items()
        },
        "join_count_dist": dict(Counter(join_counts).most_common()),
        "subquery_count_dist": dict(Counter(subq_counts).most_common()),
    }


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def print_report(result, detailed=False):
    """Print a formatted complexity report."""
    label = result["label"]
    n = result["count"]

    if n == 0:
        print(f"  {label}: No SQL queries found.")
        return

    print(f"  {label}")
    print(f"  {'─' * 60}")
    print(f"  Total SQLs:   {n:,}")
    print(f"  Databases:    {result['n_databases']}")
    print(f"  Mean Length:  {result['mean_length']:.0f} chars")

    ls = result["length_stats"]
    print(f"  Length Stats:  min={ls['min']}  p25={ls['p25']}  "
          f"median={ls['median']}  p75={ls['p75']}  max={ls['max']}")

    print(f"\n  {'Feature':<20} {'Count':>8} {'Rate':>8}")
    print(f"  {'─' * 20} {'─' * 8} {'─' * 8}")

    for name in COMPLEXITY_PATTERNS:
        m = result["core_metrics"][name]
        print(f"  {name:<20} {m['count']:>8} {m['rate']:>8.1%}")

    if detailed:
        print(f"\n  {'Extended Feature':<20} {'Count':>8} {'Rate':>8}")
        print(f"  {'─' * 20} {'─' * 8} {'─' * 8}")
        for name in EXTENDED_PATTERNS:
            m = result["extended_metrics"][name]
            if m["count"] > 0:
                print(f"  {name:<20} {m['count']:>8} {m['rate']:>8.1%}")

    print()


def print_comparison(results):
    """Print a side-by-side comparison table."""
    labels = [r["label"] for r in results]
    counts = [r["count"] for r in results]

    col_width = max(12, max(len(l) for l in labels) + 2)

    # Header
    header = f"  {'Metric':<20}"
    for label in labels:
        header += f" {label:>{col_width}}"
    print(header)
    print(f"  {'─' * 20}" + f" {'─' * col_width}" * len(labels))

    # Count
    row = f"  {'n':20}"
    for r in results:
        row += f" {r['count']:>{col_width},}"
    print(row)

    # Mean length
    row = f"  {'Mean Length':20}"
    for r in results:
        if r["count"] > 0:
            row += f" {r['mean_length']:>{col_width}.0f}"
        else:
            row += f" {'N/A':>{col_width}}"
    print(row)

    # Core metrics
    for name in COMPLEXITY_PATTERNS:
        row = f"  {name:20}"
        for r in results:
            if r["count"] > 0:
                rate = r["core_metrics"][name]["rate"]
                row += f" {rate:>{col_width}.1%}"
            else:
                row += f" {'N/A':>{col_width}}"
        print(row)

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SQL complexity metrics"
    )
    parser.add_argument(
        "--data_root", type=str, required=True,
        help="Root directory containing {db_id}/questions/*.json",
    )
    parser.add_argument(
        "--db_ids", type=str, nargs="*", default=None,
        help="Specific database IDs to analyze",
    )
    parser.add_argument(
        "--db_list", type=str, default=None,
        help="JSON file containing a list of db_ids",
    )
    parser.add_argument(
        "--all_sqls", action="store_true",
        help="Include all SQLs (not just accepted ones)",
    )
    parser.add_argument(
        "--compare", type=str, nargs="*", default=None,
        help="Additional data_root directories to compare against",
    )
    parser.add_argument(
        "--compare_labels", type=str, nargs="*", default=None,
        help="Labels for comparison datasets (default: directory names)",
    )
    parser.add_argument(
        "--detailed", action="store_true",
        help="Show extended keyword coverage",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Save results to JSON file",
    )
    args = parser.parse_args()

    # Determine db_ids
    db_ids = args.db_ids
    if db_ids is None and args.db_list:
        with open(args.db_list, "r") as f:
            db_ids = json.load(f)

    accepted_only = not args.all_sqls

    # Load primary dataset
    print(f"Loading SQLs from {args.data_root} ...")
    primary_sqls = load_sqls(args.data_root, db_ids, accepted_only)
    primary_label = Path(args.data_root).name or "Primary"
    primary_result = analyze_complexity(primary_sqls, primary_label)

    all_results = [primary_result]

    # Load comparison datasets
    if args.compare:
        for i, comp_root in enumerate(args.compare):
            print(f"Loading SQLs from {comp_root} ...")
            comp_sqls = load_sqls(comp_root, db_ids, accepted_only)
            if args.compare_labels and i < len(args.compare_labels):
                label = args.compare_labels[i]
            else:
                label = Path(comp_root).name or f"Compare-{i+1}"
            all_results.append(analyze_complexity(comp_sqls, label))

    # Print report
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  SQL Complexity Report")
    print(f"  Accepted only: {accepted_only}")
    print(sep)

    if len(all_results) == 1:
        print()
        print_report(primary_result, detailed=args.detailed)
    else:
        print()
        print_comparison(all_results)
        if args.detailed:
            for r in all_results:
                print_report(r, detailed=True)

    print(sep)

    # Save results
    if args.output:
        output_data = {
            "config": {
                "data_root": args.data_root,
                "accepted_only": accepted_only,
                "n_databases": primary_result.get("n_databases", 0),
            },
            "results": [
                {
                    "label": r["label"],
                    "count": r["count"],
                    "mean_length": round(r.get("mean_length", 0), 1),
                    "core_metrics": {
                        k: round(v["rate"], 4)
                        for k, v in r.get("core_metrics", {}).items()
                    },
                }
                for r in all_results
            ],
        }
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\n  Results saved to {args.output}\n")


if __name__ == "__main__":
    main()
