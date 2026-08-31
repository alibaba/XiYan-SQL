"""
Schema Selection and Validation from SynSQL-2.5M.

Pipeline:
    1. List all databases from SynSQL-2.5M/databases/
    2. Compute text embeddings for database names (DashScope API)
    3. K-Means clustering → select 1 representative per cluster
    4. Validate each selected database (SchemaEngine: FK + topology)
    5. Enrich column descriptions from tables.json
    6. Save M-Schema JSONs for valid databases

Input:
    data/SynSQL-2.5M/
    ├── databases/{db_id}/{db_id}.sqlite
    └── tables.json

Output:
    output/
    ├── selected_databases.json          (1,200 selected db_ids)
    ├── valid_databases.json             (1,022 validated db_ids)
    └── {db_id}/schema/mschema.json      (one per valid database)

Usage:
    python data_synthesis/filter_databases.py \
        --source_path data/SynSQL-2.5M \
        --output_path output/ \
        --n_clusters 1200
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_synthesis.llm_utils import LLMClient
from schema_engine import build_schema_engine


# ---------------------------------------------------------------------------
# Step 1: Text Embedding
# ---------------------------------------------------------------------------

def compute_embeddings(db_names, llm: LLMClient, cache_file=None):
    """Compute text embeddings for all database names.

    Supports caching to avoid redundant API calls.
    """
    if cache_file and os.path.exists(cache_file):
        print(f"Loading cached embeddings from {cache_file}")
        data = np.load(cache_file)
        if len(data) == len(db_names):
            return data
        print(f"  Cache size mismatch ({len(data)} vs {len(db_names)}), recomputing")

    print(f"Computing embeddings for {len(db_names)} databases (model: {llm.embedding_model})...")
    embeddings = []
    for name in tqdm(db_names, desc="Embedding"):
        while True:
            vec = llm.embed(name)
            if vec is not None:
                embeddings.append(vec)
                break
            time.sleep(2)

    embeddings = np.array(embeddings, dtype=np.float32)

    if cache_file:
        os.makedirs(os.path.dirname(cache_file) or ".", exist_ok=True)
        np.save(cache_file, embeddings)
        print(f"Embeddings cached to {cache_file}")

    return embeddings


# ---------------------------------------------------------------------------
# Step 2: K-Means Clustering
# ---------------------------------------------------------------------------

def select_representatives(embeddings, db_names, n_clusters):
    """Select representative databases via K-Means clustering.

    For each cluster, pick the database closest to the cluster center.
    """
    print(f"Running K-Means clustering (k={n_clusters})...")
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters, random_state=42, batch_size=1000,
    )
    labels = kmeans.fit_predict(embeddings)
    centers = kmeans.cluster_centers_

    selected = []
    for cid in tqdm(range(n_clusters), desc="Selecting representatives"):
        mask = labels == cid
        if not mask.any():
            continue
        cluster_embeddings = embeddings[mask]
        cluster_names = [db_names[i] for i in np.where(mask)[0]]
        distances = cdist(cluster_embeddings, centers[cid:cid+1], "cosine").flatten()
        selected.append(cluster_names[distances.argmin()])

    print(f"Selected {len(selected)} representative databases")
    return selected


# ---------------------------------------------------------------------------
# Step 3: Schema Validation and Enrichment
# ---------------------------------------------------------------------------

def load_tables_lookup(tables_json_path):
    """Load tables.json and build a lookup by db_id."""
    print(f"Loading tables.json...")
    with open(tables_json_path, "r", encoding="utf-8") as f:
        tables = json.load(f)
    lookup = {t["db_id"]: t for t in tables}
    print(f"  {len(lookup)} database schemas loaded")
    return lookup


def enrich_column_descriptions(schema_engine, table_meta):
    """Add column descriptions from tables.json into SchemaEngine."""
    column_names = table_meta.get("column_names", [])
    column_names_original = table_meta.get("column_names_original", [])
    table_names_original = table_meta.get("table_names_original", [])

    for i in range(len(column_names_original)):
        table_idx = column_names_original[i][0]
        if table_idx == -1:
            continue
        if table_idx >= len(table_names_original):
            continue
        table_name = table_names_original[table_idx]
        col_name = column_names_original[i][1]
        col_desc = column_names[i][1] if i < len(column_names) else ""
        schema_engine.mschema.set_column_property(table_name, col_name, "comment", col_desc)


def validate_and_save(db_id, source_path, tables_lookup, output_path):
    """Validate a database and save its M-Schema if valid.

    Returns True if the database passes validation.
    """
    sqlite_file = source_path / "databases" / db_id / f"{db_id}.sqlite"
    if not sqlite_file.exists():
        print(f"  [SKIP] {db_id}: SQLite file not found")
        return False

    try:
        se = build_schema_engine(str(sqlite_file), db_id)
    except ValueError as e:
        print(f"  [FAIL] {db_id}: {e}")
        return False
    except Exception as e:
        print(f"  [FAIL] {db_id}: {e}")
        return False

    if db_id in tables_lookup:
        enrich_column_descriptions(se, tables_lookup[db_id])

    schema_dir = output_path / db_id / "schema"
    schema_dir.mkdir(parents=True, exist_ok=True)
    se.mschema.save(str(schema_dir / "mschema.json"))
    se.dispose()
    return True


def validate_selected_databases(selected_dbs, source_path, tables_lookup, output_path):
    """Validate all selected databases and save valid schemas."""
    print(f"\nValidating {len(selected_dbs)} selected databases...")
    valid = []
    for db_id in tqdm(selected_dbs, desc="Validating"):
        if validate_and_save(db_id, source_path, tables_lookup, output_path):
            valid.append(db_id)

    print(f"Validation complete: {len(valid)}/{len(selected_dbs)} passed")
    return valid


# ---------------------------------------------------------------------------
# Step 4 (optional): Extract Candidate Questions
# ---------------------------------------------------------------------------

def extract_candidate_questions(data_json_path, valid_dbs, output_path):
    """Extract candidate questions for valid databases from data.json.

    Streams the large JSON array to avoid loading the full file into memory.
    """
    valid_set = set(valid_dbs)

    print(f"\nExtracting candidate questions from {data_json_path}...")
    print(f"  (this may take a while for large files)")

    filtered = {}
    total = 0

    try:
        import ijson
        with open(data_json_path, "rb") as f:
            for item in tqdm(ijson.items(f, "item"), desc="Scanning"):
                db_id = item.get("db_id", "")
                if db_id in valid_set:
                    filtered.setdefault(db_id, []).append(item)
                    total += 1
    except ImportError:
        print("  ijson not available, loading full file into memory...")
        with open(data_json_path, "r", encoding="utf-8") as f:
            all_data = json.load(f)
        for item in tqdm(all_data, desc="Filtering"):
            db_id = item.get("db_id", "")
            if db_id in valid_set:
                filtered.setdefault(db_id, []).append(item)
                total += 1

    for db_id, questions in filtered.items():
        q_dir = output_path / db_id / "questions"
        q_dir.mkdir(parents=True, exist_ok=True)
        with open(q_dir / "questions.json", "w", encoding="utf-8") as f:
            json.dump(questions, f, ensure_ascii=False, indent=2)

    print(f"Extracted {total} candidate questions across {len(filtered)} databases")
    return total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Select and validate database schemas from SynSQL-2.5M"
    )
    parser.add_argument(
        "--source_path", type=str, required=True,
        help="Path to SynSQL-2.5M directory (contains databases/ and tables.json)",
    )
    parser.add_argument(
        "--output_path", type=str, required=True,
        help="Output directory for validated schemas",
    )
    parser.add_argument(
        "--n_clusters", type=int, default=1200,
        help="Number of K-Means clusters (default: 1200)",
    )
    parser.add_argument(
        "--embedding_cache", type=str, default=None,
        help="Path to cache embeddings (.npy file) to avoid recomputation",
    )
    parser.add_argument(
        "--extract_questions", action="store_true",
        help="Also extract candidate questions from data.json for valid databases",
    )
    args = parser.parse_args()

    source_path = Path(args.source_path)
    output_path = Path(args.output_path)

    if not source_path.exists():
        print(f"[ERROR] Source path not found: {source_path}")
        return

    db_dir = source_path / "databases"
    if not db_dir.exists():
        print(f"[ERROR] databases/ directory not found in {source_path}")
        return

    llm = LLMClient()
    output_path.mkdir(parents=True, exist_ok=True)

    # Step 1: List all databases
    db_names = sorted([
        d.name for d in db_dir.iterdir()
        if d.is_dir() and (d / f"{d.name}.sqlite").exists()
    ])
    print(f"Found {len(db_names)} databases in {db_dir}")

    # Step 2: Compute embeddings
    embeddings = compute_embeddings(
        db_names, llm, cache_file=args.embedding_cache,
    )

    # Step 3: K-Means clustering
    selected = select_representatives(embeddings, db_names, args.n_clusters)

    selected_file = output_path / "selected_databases.json"
    with open(selected_file, "w", encoding="utf-8") as f:
        json.dump(selected, f, ensure_ascii=False, indent=2)
    print(f"Selected database list saved to {selected_file}")

    # Step 4: Validate and enrich schemas
    tables_json = source_path / "tables.json"
    tables_lookup = load_tables_lookup(str(tables_json)) if tables_json.exists() else {}

    valid = validate_selected_databases(selected, source_path, tables_lookup, output_path)

    valid_file = output_path / "valid_databases.json"
    with open(valid_file, "w", encoding="utf-8") as f:
        json.dump(valid, f, ensure_ascii=False, indent=2)
    print(f"Valid database list saved to {valid_file}")

    # Step 5 (optional): Extract candidate questions
    if args.extract_questions:
        data_json = source_path / "data.json"
        if data_json.exists():
            extract_candidate_questions(data_json, valid, output_path)
        else:
            print(f"[WARN] data.json not found at {data_json}, skipping question extraction")

    print(f"\nSummary:")
    print(f"  Total databases:    {len(db_names)}")
    print(f"  Selected (k={args.n_clusters}): {len(selected)}")
    print(f"  Validated:          {len(valid)}")


if __name__ == "__main__":
    main()
