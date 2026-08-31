"""
ms-swift ORM reward plugin for RLVR training.

Defines the execution reward used during GRPO training. ``sql_acc`` sends the
predicted SQL to the SQLite execution server and compares its execution result
against the ground-truth SQL.

Usage (in swift rlhf command):
    --external_plugins training/reward_utils/plugin.py
    --reward_funcs sql_acc

Environment variable:
    SQLITE_SERVER_URL: URL of the SQLite execution server
                       (default: http://localhost:8000)
"""

import os
import re
from typing import List

import requests
from swift.plugin import ORM, orms

SQLITE_SERVER_URL = os.environ.get("SQLITE_SERVER_URL", "http://localhost:8000")


class SQLAccuracy(ORM):
    """Reward = 1.0 if predicted SQL produces the same result as ground-truth.

    Extracts SQL from the ```sql ... ``` block, sends it to the SQLite
    execution server along with the ground-truth SQL, and returns the
    execution accuracy score.

    Expected dataset columns: dataset, database, gt_sql
    """

    def __call__(self, completions, dataset, database, gt_sql,
                 **kwargs) -> List[float]:
        rewards = []
        for completion, dataset_name, db_name, sql in zip(
            completions, dataset, database, gt_sql
        ):
            try:
                match = re.search(r"```sql(.*?)```", completion, re.DOTALL)
                if match is None:
                    rewards.append(0.0)
                    continue

                pred_sql = match.group(1).strip()
                resp = requests.post(
                    f"{SQLITE_SERVER_URL}/score",
                    json={
                        "dataset_name": dataset_name,
                        "database_name": db_name,
                        "gt_sql": sql,
                        "pred_sql": pred_sql,
                        "timeout": 10,
                    },
                    timeout=30,
                )
                result = resp.json()
                rewards.append(result["SCORE"] if result.get("VALID") else 0.0)
            except Exception:
                rewards.append(0.0)

        return rewards


orms["sql_acc"] = SQLAccuracy
