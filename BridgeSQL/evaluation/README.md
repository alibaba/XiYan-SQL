# Evaluation

This directory provides entry points for model evaluation, database population
quality evaluation, SQL-complexity analysis, and peer-review retention
analysis. Detailed metric definitions, fixed protocols, and result tables live
in [`docs/`](../docs/README.md).

Run all commands from the repository root.

## Model Evaluation

Prepare the benchmark data and SQLite databases as described in the main
[README](../README.md#step-3-evaluation), start the SQLite execution server,
then evaluate a checkpoint on all seven benchmarks:

```bash
python training/reward_utils/sqlite_server.py \
    --db_dir "$DB_ROOT" --port 8000 --workers 200

bash evaluation/run_eval.sh --model_path <checkpoint_path>
```

The main entry points are [`run_eval.sh`](run_eval.sh) and
[`evaluate.py`](evaluate.py). Benchmark preparation utilities are
[`prepare_databases.py`](prepare_databases.py) and
[`prepare_eval_dataset.py`](prepare_eval_dataset.py).

## Database Population Quality

Evaluate one populated database collection with the eight quality metrics:

```bash
python evaluation/evaluate_db_quality.py \
    --db_root output/data_synthesis/ \
    --data_root output/data_synthesis/ \
    --output output/analysis/database_quality.json
```

See [Database Population Quality Evaluation](../docs/database_quality.md) for
the fixed three-system protocol, complete metric definitions, and reported
results. The implementation entry point is
[`evaluate_db_quality.py`](evaluate_db_quality.py).

## SQL Complexity

Analyze the accepted SQLs in a curated dataset:

```bash
python evaluation/evaluate_sql_complexity.py \
    --data_root output/data_synthesis/
```

See [SQL Complexity Metrics](../docs/sql_complexity.md) for the metric
definitions, result table, and comparison commands. The implementation entry
point is [`evaluate_sql_complexity.py`](evaluate_sql_complexity.py).

## Peer-review Retention

Analyze which upstream SQL examples are retained by execution-grounded peer
review:

```bash
python evaluation/evaluate_peer_review_retention.py \
    --data_root output/data_synthesis/ \
    --db_list output/data_synthesis/valid_databases.json \
    --source_label BridgeSQL-48k \
    --output output/analysis/peer_review_retention.json
```

See [Peer-review Retention Analysis](../docs/peer_review_retention.md) for the
paper-snapshot accounting, complete feature breakdowns, strict audit command,
and output formats. The implementation entry point is
[`evaluate_peer_review_retention.py`](evaluate_peer_review_retention.py).
