# SQL Complexity Metrics

We measure SQL complexity across the following dimensions to assess whether peer review preserves syntactic diversity. All metrics are computed via regex pattern matching on the SQL string:

| Metric | Detection Pattern | Description |
|--------|------------------|-------------|
| Mean Length | `len(sql)` | Average character count of SQL queries |
| JOIN | `\bJOIN\b` | Queries containing any JOIN clause |
| GROUP BY | `\bGROUP\s+BY\b` | Queries with aggregation grouping |
| Subquery | `\(\s*SELECT\b` | Queries containing nested SELECT |
| HAVING | `\bHAVING\b` | Queries with post-aggregation filtering |
| CTE | `\bWITH\b(?!\s+ROLLUP)` | Queries using Common Table Expressions |
| Window Function | `\bOVER\s*\(` | Queries using window/analytic functions |

All patterns are case-insensitive. A query is counted as containing a feature if the pattern matches anywhere in the SQL string.

## Results

| Dataset | Mean Len. | JOIN | GROUP BY | Subquery | HAVING | CTE | Win. Func. |
|---|---|---|---|---|---|---|---|
| BIRD Train | 170 | 76.5% | 10.7% | 7.8% | 1.5% | 0.0% | 0.1% |
| Spider Train | 123 | 44.0% | 22.6% | 15.6% | 5.2% | 0.0% | 0.0% |
| SynSQL | 581 | 90.7% | 65.3% | 50.5% | 19.7% | 42.0% | 24.3% |
| BridgeSQL | 342 | 82.5% | 50.2% | 29.5% | 15.4% | 19.3% | 10.9% |

## Running SQL Complexity Evaluation

```bash
# Analyze curated dataset (accepted SQLs only)
python evaluation/evaluate_sql_complexity.py \
    --data_root output/data_synthesis/

# Analyze all SQLs (before peer review filtering)
python evaluation/evaluate_sql_complexity.py \
    --data_root output/data_synthesis/ \
    --all_sqls

# Compare multiple datasets side by side
python evaluation/evaluate_sql_complexity.py \
    --data_root output/data_synthesis/ \
    --compare path/to/synsql/data/ \
    --compare_labels "BridgeSQL" "SynSQL"

# Show extended keyword coverage
python evaluation/evaluate_sql_complexity.py \
    --data_root output/data_synthesis/ \
    --detailed
```

See [`evaluate_sql_complexity.py`](../evaluation/evaluate_sql_complexity.py) for full options.
