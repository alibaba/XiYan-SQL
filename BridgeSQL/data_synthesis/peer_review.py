"""
Execution-Grounded Peer Review.

For each populated database, collect SQL candidates from multiple LLM
reviewers, execute them against the populated SQLite database, and retain
only Question-SQL pairs where a sufficient number of candidates produce
identical execution results.

Pipeline (per database):
    1. Load candidate questions from questions/questions.json
    2. For each question, call N reviewer models to generate SQL answers
    3. Execute all SQL candidates (original + reviewer-generated) against
       the populated SQLite database
    4. Group candidates by execution result (set equality)
    5. Accept the pair if the largest consensus group >= threshold

Input:
    output/data_synthesis/{db_id}/
    ├── schema/mschema.json
    ├── {db_id}.sqlite
    └── questions/questions.json

Output (added to existing structure):
    output/data_synthesis/{db_id}/questions/
    ├── peer_reviews.json             (all candidates with reviewer SQLs)
    └── verified.json                 (verified Q-SQL pairs with status)

Usage:
    python data_synthesis/peer_review.py \
        --output_path output/data_synthesis/ \
        --threshold 0.6
"""

import argparse
import json
import os
import random
import re
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_synthesis.llm_utils import LLMClient
from schema_engine import Schema

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

SQL_REVIEW_PROMPT = '''As a data analyst, you are provided with a SQLite database schema. Your task is to answer a natural language question by generating a valid SQL query.
The SQL query should be surrounded by ```sql and ```.

**Database Schema**
{schema}

**Question**
{question}
'''

SQL_EXEC_TIMEOUT = 10  # seconds per query


# ---------------------------------------------------------------------------
# Phase 1: Generate SQL candidates from reviewer models
# ---------------------------------------------------------------------------

def call_reviewer(question: str, schema_str: str, model: str,
                  llm: LLMClient) -> tuple:
    """Call a single reviewer model to generate SQL for a question.

    Returns (sql, full_response) or (None, None) on failure.
    """
    prompt = SQL_REVIEW_PROMPT.format(schema=schema_str, question=question)
    response = llm.call(model, prompt, temperature=0.7)
    if not response:
        return None, None

    match = re.search(r"```sql(.*?)```", response, re.DOTALL)
    if not match:
        return None, None

    return match.group(1).strip(), response


def generate_reviews(db_id: str, db_dir: Path, llm: LLMClient):
    """Generate SQL candidates from all reviewer models for a database.

    Reads questions/questions.json, calls reviewer models in parallel,
    and saves questions/peer_reviews.json.
    """
    reviews_path = db_dir / "questions" / "peer_reviews.json"
    if reviews_path.exists():
        return True

    questions_path = db_dir / "questions" / "questions.json"
    if not questions_path.exists():
        print(f"  [SKIP] {db_id}: no questions.json")
        return False

    mschema = Schema(db_id=db_id)
    mschema.load(str(db_dir / "schema" / "mschema.json"))
    schema_str = mschema.to_mschema()

    questions = json.load(open(questions_path, "r", encoding="utf-8"))
    models = llm.chat_models

    # Build all (question_idx, model) requests
    requests = [
        (i, q, m)
        for i, q in enumerate(questions)
        for m in models
    ]

    # Parallel LLM calls
    results = {}
    with ThreadPoolExecutor(max_workers=min(40, len(requests))) as pool:
        futures = {}
        for idx, q, model in requests:
            question_text = q.get("external_knowledge", "")
            if question_text:
                question_text += "\n"
            question_text += q["question"]
            fut = pool.submit(call_reviewer, question_text, schema_str,
                              model, llm)
            futures[fut] = (idx, model)

        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc=f"Reviewing {db_id}"):
            idx, model = futures[fut]
            try:
                sql, cot = fut.result(timeout=60)
                results.setdefault(idx, {})[model] = (sql, cot)
            except Exception:
                pass

    # Assemble results
    reviewed = []
    for i, q in enumerate(questions):
        sql_candidates = {"synsql": q["sql"]}
        cot_candidates = {"synsql": q.get("cot", "")}

        if i in results:
            for model, (sql, cot) in results[i].items():
                if sql and cot:
                    sql_candidates[model] = sql
                    cot_candidates[model] = cot

        entry = q.copy()
        entry["sql_candidates"] = sql_candidates
        entry["cot_candidates"] = cot_candidates
        reviewed.append(entry)

    with open(reviews_path, "w", encoding="utf-8") as f:
        json.dump(reviewed, f, ensure_ascii=False, indent=2)

    return True


# ---------------------------------------------------------------------------
# Phase 2: Execute and verify via consensus voting
# ---------------------------------------------------------------------------

class _SQLTimeout(Exception):
    pass


def _execute_sql(db_path: str, sql: str, timeout: int = SQL_EXEC_TIMEOUT):
    """Execute a SQL query against a SQLite database with timeout.

    Returns the result set (list of tuples) or None on error/timeout.
    """
    conn = None
    try:
        conn = sqlite3.connect(db_path, timeout=30.0)
        start = [None]

        def progress():
            if start[0] is None:
                start[0] = time.time()
            elif time.time() - start[0] > timeout:
                raise _SQLTimeout()
            return 0

        conn.set_progress_handler(progress, 100)
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        cursor.close()
        return rows if rows else None
    except Exception:
        return None
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass


def verify_question(question_entry: dict, db_path: str,
                    threshold: float) -> dict:
    """Verify a single question by executing all SQL candidates and voting.

    Returns a dict with review_status, consensus_count, selected sql/cot.
    """
    sql_candidates = question_entry.get("sql_candidates", {})
    cot_candidates = question_entry.get("cot_candidates", {})

    names = list(sql_candidates.keys())
    sqls = [sql_candidates[n] for n in names]

    # Execute all candidates
    exec_results = [_execute_sql(db_path, sql) for sql in sqls]

    # Group by execution result (set equality)
    groups = []
    grouped = set()
    for i in range(len(exec_results)):
        if i in grouped or exec_results[i] is None:
            grouped.add(i)
            continue

        same = [i]
        set_i = set(exec_results[i])
        for j in range(i + 1, len(exec_results)):
            if j not in grouped and exec_results[j] is not None:
                if set_i == set(exec_results[j]):
                    same.append(j)
                    grouped.add(j)
        grouped.add(i)
        groups.append(same)

    # Find largest consensus group
    best = max(groups, key=len) if groups else []
    total = len(names)
    consensus = len(best)

    result = {
        "question": question_entry["question"],
        "external_knowledge": question_entry.get("external_knowledge", ""),
    }

    if total > 0 and consensus / total >= threshold:
        result["review_status"] = "accepted"
        result["consensus_count"] = consensus
        pick = random.choice(best)
        result["sql"] = sqls[pick]
        result["sql_source"] = names[pick]
        result["cot"] = cot_candidates.get(names[pick], "")
    else:
        result["review_status"] = "rejected"
        result["consensus_count"] = consensus
        result["sql"] = sqls[0] if sqls else ""
        result["sql_source"] = names[0] if names else "synsql"
        result["cot"] = cot_candidates.get(names[0] if names else "synsql", "")

    return result


def verify_database(db_id: str, db_dir: Path, threshold: float):
    """Verify all questions for a database via execution-based consensus.

    Reads questions/peer_reviews.json, writes questions/verified.json.
    Returns (n_accepted, n_total).
    """
    verified_path = db_dir / "questions" / "verified.json"
    if verified_path.exists():
        data = json.load(open(verified_path, "r", encoding="utf-8"))
        accepted = sum(1 for d in data if d.get("review_status") == "accepted")
        return accepted, len(data)

    reviews_path = db_dir / "questions" / "peer_reviews.json"
    if not reviews_path.exists():
        print(f"  [SKIP] {db_id}: no peer_reviews.json")
        return 0, 0

    db_path = str(db_dir / f"{db_id}.sqlite")
    if not os.path.exists(db_path):
        print(f"  [SKIP] {db_id}: no SQLite database")
        return 0, 0

    reviews = json.load(open(reviews_path, "r", encoding="utf-8"))

    verified = []
    for entry in tqdm(reviews, desc=f"Verifying {db_id}", leave=False):
        verified.append(verify_question(entry, db_path, threshold))

    with open(verified_path, "w", encoding="utf-8") as f:
        json.dump(verified, f, ensure_ascii=False, indent=2)

    accepted = sum(1 for v in verified if v["review_status"] == "accepted")
    return accepted, len(verified)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Execution-Grounded Peer Review"
    )
    parser.add_argument(
        "--output_path", type=str, required=True,
        help="Output directory (same as database_population.py output)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.6,
        help="Consensus threshold for acceptance (default: 0.6)",
    )
    parser.add_argument(
        "--skip_generate", action="store_true",
        help="Skip review generation, only run verification",
    )
    args = parser.parse_args()

    output_path = Path(args.output_path)
    populated_file = output_path / "populated_databases.json"

    if not populated_file.exists():
        print(f"[ERROR] {populated_file} not found. "
              f"Run database_population.py first.")
        return

    db_ids = json.load(open(populated_file, "r", encoding="utf-8"))
    print(f"Databases to review: {len(db_ids)}")

    llm = LLMClient()
    print(f"Reviewer models: {llm.chat_models}")

    # Phase 1: Generate reviews
    if not args.skip_generate:
        print(f"\n=== Phase 1: Generating SQL candidates ===")
        for db_id in tqdm(db_ids, desc="Generating"):
            db_dir = output_path / db_id
            generate_reviews(db_id, db_dir, llm)

    # Phase 2: Verify via execution
    print(f"\n=== Phase 2: Verifying via execution (threshold={args.threshold}) ===")
    total_accepted = 0
    total_questions = 0

    for db_id in tqdm(db_ids, desc="Verifying"):
        db_dir = output_path / db_id
        accepted, total = verify_database(db_id, db_dir, args.threshold)
        total_accepted += accepted
        total_questions += total

    print(f"\nDone.")
    print(f"  Total questions: {total_questions}")
    print(f"  Accepted: {total_accepted}")
    if total_questions > 0:
        print(f"  Acceptance rate: {total_accepted / total_questions * 100:.1f}%")


if __name__ == "__main__":
    main()
