"""
SQLite Execution Server for RLVR reward computation.

A Flask server that executes predicted and ground-truth SQL queries against
SQLite databases and returns execution accuracy scores. Designed for high
concurrency during GRPO training.

The server expects databases organized as:
    {db_dir}/{dataset_name}/{db_name}/{db_name}.sqlite

Endpoints:
    GET  /health    — Service health and worker status
    POST /execute   — Execute a single SQL and return results
    POST /score     — Compare pred_sql vs gt_sql execution results

Usage:
    python training/reward_utils/sqlite_server.py \
        --db_dir databases/ \
        --port 8000 \
        --workers 200
"""

import argparse
import atexit
import logging
import os
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from threading import Semaphore

from flask import Flask, request, jsonify, copy_current_request_context

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Global configuration (set in main)
# ---------------------------------------------------------------------------

DB_DIR = ""
MAX_WORKERS = 200
semaphore = Semaphore(MAX_WORKERS)
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="SQLWorker")


# ---------------------------------------------------------------------------
# SQL execution with timeout via progress_handler
# ---------------------------------------------------------------------------

def execute_sql(db_path: str, sql: str, timeout: int = 10):
    """Execute a SQL query with progress-handler-based timeout.

    Returns (rows: list[tuple], "OK") or (error_msg: str, "ERROR").
    """
    conn = None
    cursor = None
    start_time = time.time()

    def progress_handler():
        if time.time() - start_time > timeout:
            return 1  # non-zero aborts the query
        return 0

    try:
        conn = sqlite3.connect(db_path, timeout=timeout)
        conn.set_progress_handler(progress_handler, 1000)
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        return rows, "OK"
    except sqlite3.OperationalError as e:
        err = str(e).lower()
        if "interrupted" in err or "abort" in err:
            return f"Query execution exceeded {timeout} seconds", "ERROR"
        return f"sqlite3.OperationalError: {e}", "ERROR"
    except Exception as e:
        return f"{type(e).__name__}: {e}", "ERROR"
    finally:
        if conn is not None:
            try:
                conn.set_progress_handler(None, 0)
            except Exception:
                pass
        if cursor is not None:
            try:
                cursor.close()
            except Exception:
                pass
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def run_query(sql: str, dataset_name: str, db_name: str,
              timeout: int = 10, return_string: bool = False):
    """Execute a query and return structured result dict."""
    db_path = os.path.join(DB_DIR, dataset_name, db_name, f"{db_name}.sqlite")
    try:
        result, status = execute_sql(db_path, sql, timeout=timeout)
        if status == "OK" and return_string:
            result_str = str(result)
            if len(result_str) > 2048:
                result_str = result_str[:2048] + "\n... (truncated)"
            result = result_str
        return {"SQL": sql, "RESULT": result, "STATUS": status}
    except Exception as e:
        return {"SQL": sql, "RESULT": f"Error: {e}", "STATUS": "ERROR"}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint for monitoring."""
    available = semaphore._value
    active = MAX_WORKERS - available
    return jsonify({
        "status": "healthy" if active < MAX_WORKERS * 0.95 else "degraded",
        "max_workers": MAX_WORKERS,
        "active_workers": active,
        "available_workers": available,
    })


@app.route("/execute", methods=["POST"])
def execute():
    """Execute a single SQL query and return the result."""
    if not request.is_json:
        return jsonify({"error": "JSON expected"}), 400
    data = request.get_json()

    @copy_current_request_context
    def task():
        try:
            return jsonify(run_query(
                data["sql"],
                data["dataset_name"],
                data["database_name"],
                timeout=data.get("timeout", 10),
                return_string=True,
            ))
        except Exception as e:
            return jsonify({"SQL": data.get("sql", ""),
                           "RESULT": f"Error: {e}", "STATUS": "ERROR"})
        finally:
            semaphore.release()

    acquired = False
    try:
        semaphore.acquire()
        acquired = True
        future = executor.submit(task)
        request_timeout = data.get("timeout", 10) + 5
        try:
            return future.result(timeout=request_timeout)
        except FutureTimeoutError:
            return jsonify({"SQL": data.get("sql", ""),
                           "RESULT": "Request timeout", "STATUS": "ERROR"})
    except Exception as e:
        if acquired:
            semaphore.release()
        return jsonify({"SQL": data.get("sql", ""),
                       "RESULT": f"Error: {e}", "STATUS": "ERROR"})


@app.route("/score", methods=["POST"])
def score():
    """Compare pred_sql vs gt_sql by execution result (set equality).

    Request JSON:
        dataset_name, database_name, gt_sql, pred_sql, timeout (optional)

    Response JSON:
        VALID (0|1), SCORE (0.0|1.0), EXECUTE (0|1)
    """
    if not request.is_json:
        return jsonify({"error": "JSON expected"}), 400
    data = request.get_json()

    @copy_current_request_context
    def task():
        try:
            timeout = data.get("timeout", 10)
            gt_result = run_query(
                data["gt_sql"], data["dataset_name"],
                data["database_name"], timeout=timeout,
            )
            if gt_result["STATUS"] != "OK":
                return jsonify({"VALID": 0, "SCORE": 0.0, "EXECUTE": 0})

            pred_result = run_query(
                data["pred_sql"], data["dataset_name"],
                data["database_name"], timeout=timeout,
            )
            if pred_result["STATUS"] != "OK":
                return jsonify({"VALID": 1, "SCORE": 0.0, "EXECUTE": 0})

            gt_res = gt_result["RESULT"]
            pred_res = pred_result["RESULT"]
            match = 1.0 if set(gt_res) == set(pred_res) else 0.0
            return jsonify({"VALID": 1, "SCORE": match, "EXECUTE": 1})
        except Exception as e:
            return jsonify({"VALID": 0, "SCORE": 0.0, "EXECUTE": 0,
                           "ERROR": str(e)})
        finally:
            semaphore.release()

    acquired = False
    try:
        semaphore.acquire()
        acquired = True
        future = executor.submit(task)
        request_timeout = data.get("timeout", 10) * 2 + 10
        try:
            return future.result(timeout=request_timeout)
        except FutureTimeoutError:
            return jsonify({"VALID": 0, "SCORE": 0.0, "EXECUTE": 0,
                           "ERROR": "Request timeout"})
    except Exception as e:
        if acquired:
            semaphore.release()
        return jsonify({"VALID": 0, "SCORE": 0.0, "EXECUTE": 0,
                       "ERROR": str(e)})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global DB_DIR, MAX_WORKERS, semaphore, executor

    parser = argparse.ArgumentParser(
        description="SQLite Execution Server for RLVR reward computation"
    )
    parser.add_argument(
        "--db_dir", type=str, required=True,
        help="Root directory: {db_dir}/{dataset}/{db_name}/{db_name}.sqlite",
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help="Port to listen on (default: 8000)",
    )
    parser.add_argument(
        "--workers", type=int, default=200,
        help="Max concurrent query workers (default: 200)",
    )
    args = parser.parse_args()

    DB_DIR = args.db_dir
    MAX_WORKERS = args.workers
    semaphore = Semaphore(MAX_WORKERS)
    executor = ThreadPoolExecutor(
        max_workers=MAX_WORKERS, thread_name_prefix="SQLWorker"
    )

    def cleanup():
        logger.info("Shutting down executor...")
        executor.shutdown(wait=True, cancel_futures=True)

    atexit.register(cleanup)

    print(f"Database root: {DB_DIR}")
    print(f"Max workers: {MAX_WORKERS}")
    print(f"Starting server on port {args.port}...")
    app.run(host="0.0.0.0", port=args.port, threaded=True)


if __name__ == "__main__":
    main()
