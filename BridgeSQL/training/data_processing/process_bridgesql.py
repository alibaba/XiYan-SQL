"""
Process verified BridgeSQL data into SFT and RLVR training formats.

Reads the peer-review verified Q-SQL pairs from each database's
questions/verified.json, constructs prompts with M-Schema, and produces:

  - SFT dataset: train/dev split by database (95%/5%), includes CoT response
  - RLVR dataset: all databases (no split), prompt-only for online generation

Input:
    output/data_synthesis/
    ├── populated_databases.json
    └── {db_id}/
        ├── schema/mschema.json
        └── questions/verified.json

Output:
    output/training_data/
    ├── bridgesql_sft_train.json
    ├── bridgesql_sft_dev.json
    └── bridgesql_rl.json

Usage:
    python training/data_processing/process_bridgesql.py \
        --output_path output/data_synthesis/ \
        --save_dir output/training_data/
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from schema_engine import Schema

# ---------------------------------------------------------------------------
# Prompt template (shared by SFT and RL)
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = '''As a data analyst, you are provided with a SQLite database schema. Your task is to answer a natural language question and generate a valid SQL query with given information. Enclose the SQL with ```sql and ```.

# Database Schema:
{schema}

# Question:
{question}
'''


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def build_question_text(entry: dict) -> str:
    """Combine external_knowledge and question into the full question text."""
    question = entry["question"]
    knowledge = entry.get("external_knowledge", "").strip()
    if knowledge:
        return knowledge + "\n" + question
    return question


def process_database_sft(db_id: str, output_path: Path) -> list:
    """Process one database into SFT format items.

    Each item has messages: [user prompt, assistant CoT response].
    """
    schema_path = output_path / db_id / "schema" / "mschema.json"
    verified_path = output_path / db_id / "questions" / "verified.json"

    if not verified_path.exists():
        return []

    mschema = Schema(db_id=db_id)
    mschema.load(str(schema_path))
    schema_str = mschema.to_mschema()

    verified = json.load(open(verified_path, "r", encoding="utf-8"))
    items = []

    for entry in verified:
        if entry.get("review_status") != "accepted":
            continue

        full_question = build_question_text(entry)
        prompt = PROMPT_TEMPLATE.format(schema=schema_str, question=full_question)

        items.append({
            "db_id": db_id,
            "question": entry["question"],
            "external_knowledge": entry.get("external_knowledge", ""),
            "SQL": entry["sql"],
            "cot": entry.get("cot", ""),
            "sql_source": entry.get("sql_source", ""),
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": entry.get("cot", "")},
            ],
        })

    return items


def process_database_rl(db_id: str, output_path: Path) -> list:
    """Process one database into RLVR format items.

    Each item has messages: [user prompt] + metadata for reward computation.
    """
    schema_path = output_path / db_id / "schema" / "mschema.json"
    verified_path = output_path / db_id / "questions" / "verified.json"

    if not verified_path.exists():
        return []

    mschema = Schema(db_id=db_id)
    mschema.load(str(schema_path))
    schema_str = mschema.to_mschema()

    verified = json.load(open(verified_path, "r", encoding="utf-8"))
    items = []

    for entry in verified:
        if entry.get("review_status") != "accepted":
            continue

        full_question = build_question_text(entry)
        prompt = PROMPT_TEMPLATE.format(schema=schema_str, question=full_question)

        items.append({
            "dataset": "bridgesql",
            "database": db_id,
            "question": full_question,
            "gt_sql": entry["sql"],
            "messages": [
                {"role": "user", "content": prompt},
            ],
        })

    return items


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Process BridgeSQL verified data into training formats"
    )
    parser.add_argument(
        "--output_path", type=str, required=True,
        help="Synthesis output directory (contains populated_databases.json)",
    )
    parser.add_argument(
        "--save_dir", type=str, default="output/training_data",
        help="Directory to save processed datasets (default: output/training_data/)",
    )
    parser.add_argument(
        "--dev_ratio", type=float, default=0.05,
        help="Fraction of databases for SFT dev set (default: 0.05)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for train/dev split (default: 42)",
    )
    args = parser.parse_args()

    output_path = Path(args.output_path)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load database list
    populated_file = output_path / "populated_databases.json"
    if not populated_file.exists():
        print(f"[ERROR] {populated_file} not found.")
        return

    db_list = json.load(open(populated_file, "r", encoding="utf-8"))
    print(f"Total databases: {len(db_list)}")

    # Split databases for SFT: train / dev
    random.seed(args.seed)
    shuffled = db_list.copy()
    random.shuffle(shuffled)
    dev_size = max(1, int(len(shuffled) * args.dev_ratio))
    dev_dbs = shuffled[:dev_size]
    train_dbs = shuffled[dev_size:]
    print(f"SFT split: {len(train_dbs)} train, {len(dev_dbs)} dev")

    # Process SFT
    sft_train, sft_dev = [], []
    for db_id in tqdm(train_dbs, desc="SFT train"):
        sft_train.extend(process_database_sft(db_id, output_path))
    for db_id in tqdm(dev_dbs, desc="SFT dev"):
        sft_dev.extend(process_database_sft(db_id, output_path))

    # Process RL (all databases, no split)
    rl_data = []
    for db_id in tqdm(db_list, desc="RLVR"):
        rl_data.extend(process_database_rl(db_id, output_path))

    # Save
    sft_train_path = save_dir / "bridgesql_sft_train.json"
    sft_dev_path = save_dir / "bridgesql_sft_dev.json"
    rl_path = save_dir / "bridgesql_rl.json"

    with open(sft_train_path, "w", encoding="utf-8") as f:
        json.dump(sft_train, f, ensure_ascii=False, indent=2)
    with open(sft_dev_path, "w", encoding="utf-8") as f:
        json.dump(sft_dev, f, ensure_ascii=False, indent=2)
    with open(rl_path, "w", encoding="utf-8") as f:
        json.dump(rl_data, f, ensure_ascii=False, indent=2)

    print(f"\nSummary:")
    print(f"  SFT train: {len(sft_train)} items -> {sft_train_path}")
    print(f"  SFT dev:   {len(sft_dev)} items -> {sft_dev_path}")
    print(f"  RLVR:      {len(rl_data)} items -> {rl_path}")


if __name__ == "__main__":
    main()
