"""
Organize benchmark SQLite databases into a unified directory structure:
    databases/{dataset_name}/{db_name}/{db_name}.sqlite

Usage:
    python evaluation/prepare_databases.py --data_dir data/ --output_dir databases/
"""

import argparse
import os
import shutil
from pathlib import Path

DATASET_NAMES = [
    "spider_test",
    "bird_dev",
    "sciencebenchmark",
    "ehrsql",
    "spider-dk",
    "spider-syn",
    "spider-realistic",
]

# Mapping: dataset_name -> source directory containing {db_name}/{db_name}.sqlite
SOURCE_DB_DIRS = {
    "spider_test": "spider/test_database",
    "bird_dev": "bird/dev_20240627/dev_databases",
    "sciencebenchmark": "sciencebenchmark/databases",
    "ehrsql": "EHRSQL/database",
    "spider-dk": "Spider-DK/database",
    "spider-syn": "spider/database",
    "spider-realistic": "spider/database",
}


def copy_database(src_db_dir: Path, dst_db_dir: Path, db_name: str):
    """Copy a single database's .sqlite file into the target structure."""
    src_sqlite = src_db_dir / db_name / f"{db_name}.sqlite"
    if not src_sqlite.exists():
        candidates = list((src_db_dir / db_name).glob("*.sqlite"))
        if candidates:
            src_sqlite = candidates[0]
        else:
            print(f"  [SKIP] {db_name}: no .sqlite file found in {src_db_dir / db_name}")
            return False

    dst_dir = dst_db_dir / db_name
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_sqlite, dst_dir / f"{db_name}.sqlite")
    return True


def process_dataset(data_dir: Path, output_dir: Path, dataset_name: str):
    """Copy all databases from source directory to target structure."""
    src_dir = data_dir / SOURCE_DB_DIRS[dataset_name]
    dst_dir = output_dir / dataset_name

    if not src_dir.exists():
        print(f"[ERROR] Source directory not found: {src_dir}")
        return

    db_names = sorted([d.name for d in src_dir.iterdir() if d.is_dir()])
    print(f"\n[{dataset_name}] Found {len(db_names)} databases in {src_dir}")

    copied = 0
    for db_name in db_names:
        if copy_database(src_dir, dst_dir, db_name):
            copied += 1

    print(f"[{dataset_name}] Copied {copied}/{len(db_names)} databases")


def main():
    parser = argparse.ArgumentParser(
        description="Organize benchmark databases into unified structure"
    )
    parser.add_argument(
        "--data_dir", type=str, default="data/",
        help="Root directory of downloaded data",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default=os.environ.get("DB_ROOT", "databases/"),
        help="Output directory for organized databases (default: $DB_ROOT or databases/)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    if not data_dir.exists():
        print(f"[ERROR] Data directory not found: {data_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")

    for dataset_name in DATASET_NAMES:
        process_dataset(data_dir, output_dir, dataset_name)

    print("\nDone. Final structure:")
    for dataset_name in DATASET_NAMES:
        ds_dir = output_dir / dataset_name
        if ds_dir.exists():
            count = sum(1 for d in ds_dir.iterdir() if d.is_dir())
            print(f"  databases/{dataset_name}/  ({count} databases)")


if __name__ == "__main__":
    main()
