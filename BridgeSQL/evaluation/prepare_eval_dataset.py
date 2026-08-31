"""
Generate unified evaluation datasets from raw benchmark data and SQLite databases.

Extracts M-Schema directly from SQLite files (no separate schema extraction step needed).

Input:
    - Raw benchmark JSONs from data/ (various field formats per dataset)
    - SQLite databases from $DB_ROOT/{dataset_name}/{db_name}/{db_name}.sqlite

Output:
    - eval/{dataset_name}.json  (unified format)

Each output item:
    {
        "id": int,
        "dataset": str,
        "db_id": str,
        "question": str,
        "gt_sql": str,
        "messages": [{"role": "user", "content": str}],
    }

Usage:
    python evaluation/prepare_eval_dataset.py \
        --data_dir data/ \
        --db_dir databases/ \
        --output_dir eval/
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from schema_engine import build_schema_engine

PROMPT_TEMPLATE = (
    "As a data analyst, you are provided with a SQLite database schema. "
    "Your task is to answer a natural language question and generate a valid "
    "SQL query with given information. Enclose the SQL with ```sql and ```.\n\n"
    "# Database Schema:\n"
    "{schema}\n\n"
    "# Question:\n"
    "{question}"
)

DATASET_CONFIGS = {
    "spider_test": {
        "source_file": "spider/test.json",
        "question_key": "question",
        "sql_key": "query",
    },
    "bird_dev": {
        "source_file": "bird/dev_20240627/dev.json",
        "question_key": "question",
        "sql_key": "SQL",
        "extra_keys": ["evidence", "difficulty"],
    },
    "ehrsql": {
        "source_file": "EHRSQL/dev.json",
        "question_key": "question",
        "sql_key": "query",
    },
    "sciencebenchmark": {
        "source_file": "sciencebenchmark/dev.json",
        "question_key": "question",
        "sql_key": "query",
    },
    "spider-dk": {
        "source_file": "Spider-DK/Spider-DK.json",
        "question_key": "question",
        "sql_key": "query",
    },
    "spider-syn": {
        "source_file": "Spider-Syn/dev.json",
        "question_key": "SpiderSynQuestion",
        "sql_key": "query",
    },
    "spider-realistic": {
        "source_file": "spider-realistic/spider-realistic.json",
        "question_key": "question",
        "sql_key": "query",
    },
}


def extract_schema_string(db_dir: Path, dataset_name: str, db_id: str) -> str:
    """Extract M-Schema string directly from SQLite database."""
    db_path_dir = db_dir / dataset_name / db_id
    sqlite_file = db_path_dir / f"{db_id}.sqlite"

    if not sqlite_file.exists():
        candidates = list(db_path_dir.glob("*.sqlite"))
        if candidates:
            sqlite_file = candidates[0]
        else:
            raise FileNotFoundError(f"No .sqlite file in {db_path_dir}")

    se = build_schema_engine(str(sqlite_file), db_id)
    schema_str = se.mschema.to_mschema()
    se.dispose()
    return schema_str


def build_question_text(question: str, evidence: str = "") -> str:
    if evidence and evidence.strip():
        return f"{evidence.strip()}\n{question}"
    return question


def process_dataset(data_dir: Path, db_dir: Path, output_dir: Path,
                    dataset_name: str, config: dict):
    source_path = data_dir / config["source_file"]
    if not source_path.exists():
        print(f"[ERROR] Source file not found: {source_path}")
        return

    db_ds_dir = db_dir / dataset_name
    if not db_ds_dir.exists():
        print(f"[ERROR] Database directory not found: {db_ds_dir}")
        return

    with open(source_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    print(f"\n[{dataset_name}] Processing {len(raw_data)} items...")

    question_key = config["question_key"]
    sql_key = config["sql_key"]
    extra_keys = config.get("extra_keys", [])

    results = []
    skipped = 0
    schema_cache = {}

    for idx, item in enumerate(raw_data):
        db_id = item["db_id"]
        question = item[question_key]
        gt_sql = item[sql_key]
        evidence = item.get("evidence", "")

        if db_id not in schema_cache:
            try:
                schema_cache[db_id] = extract_schema_string(db_dir, dataset_name, db_id)
            except Exception as e:
                print(f"  [SKIP] {db_id}: {e}")
                skipped += 1
                continue

        schema_str = schema_cache[db_id]
        question_text = build_question_text(question, evidence)
        prompt = PROMPT_TEMPLATE.format(schema=schema_str, question=question_text)

        entry = {
            "id": idx,
            "dataset": dataset_name,
            "db_id": db_id,
            "question": question,
            "gt_sql": gt_sql,
            "messages": [{"role": "user", "content": prompt}],
        }

        for key in extra_keys:
            if key in item:
                entry[key] = item[key]

        results.append(entry)

    output_file = output_dir / f"{dataset_name}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"[{dataset_name}] Done: {len(results)} items saved, {skipped} skipped")


def main():
    parser = argparse.ArgumentParser(
        description="Generate unified evaluation datasets"
    )
    parser.add_argument(
        "--data_dir", type=str, default="data/",
        help="Root directory of downloaded data (default: data/)",
    )
    parser.add_argument(
        "--db_dir", type=str,
        default=os.environ.get("DB_ROOT", "databases/"),
        help="Root directory of organized databases (default: $DB_ROOT or databases/)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="eval/",
        help="Output directory for evaluation JSONs (default: eval/)",
    )
    parser.add_argument(
        "--datasets", type=str, nargs="*", default=None,
        help="Specific datasets to process (default: all)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    db_dir = Path(args.db_dir)
    output_dir = Path(args.output_dir)

    if not data_dir.exists():
        print(f"[ERROR] Data directory not found: {data_dir}")
        return

    if not db_dir.exists():
        print(f"[ERROR] Database directory not found: {db_dir}")
        print("Run evaluation/prepare_databases.py first.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = args.datasets or list(DATASET_CONFIGS.keys())
    print(f"Data directory:     {data_dir}")
    print(f"Database directory: {db_dir}")
    print(f"Output directory:   {output_dir}")

    for dataset_name in datasets:
        if dataset_name not in DATASET_CONFIGS:
            print(f"[WARN] Unknown dataset: {dataset_name}, skipping")
            continue
        process_dataset(data_dir, db_dir, output_dir,
                        dataset_name, DATASET_CONFIGS[dataset_name])

    print("\nDone. Evaluation datasets saved to:")
    for dataset_name in datasets:
        out_file = output_dir / f"{dataset_name}.json"
        if out_file.exists():
            with open(out_file) as f:
                count = len(json.load(f))
            print(f"  {out_file}  ({count} items)")


if __name__ == "__main__":
    main()
