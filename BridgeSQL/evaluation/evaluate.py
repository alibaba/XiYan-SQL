"""
Evaluate a model on Text-to-SQL benchmarks using vLLM and execution accuracy.

For each benchmark dataset:
    1. Load evaluation data (with pre-built prompts)
    2. Run batch inference via vLLM
    3. Extract SQL from model responses
    4. Score against ground truth via the SQLite execution server
    5. Report Execution Accuracy (EX)

Usage:
    python evaluation/evaluate.py \
        --model_path <checkpoint_path> \
        --eval_dir eval/ \
        --db_dir databases/ \
        --output_dir output/eval_results/

Prerequisites:
    - SQLite execution server running (training/reward_utils/sqlite_server.py)
    - Evaluation datasets prepared (evaluation/prepare_eval_dataset.py)
"""

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import requests
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

DATASET_NAMES = [
    "spider_test",
    "bird_dev",
    "sciencebenchmark",
    "ehrsql",
    "spider-dk",
    "spider-syn",
    "spider-realistic",
]


def extract_sql(response: str) -> str:
    """Extract SQL from model response."""
    if "</think>" in response:
        response = response.split("</think>", 1)[1]
    match = re.search(r"```sql(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response.strip()


def score_sql(server_url: str, dataset: str, db_name: str,
              gt_sql: str, pred_sql: str, timeout: int = 30) -> float:
    """Score a predicted SQL against ground truth via execution server."""
    try:
        resp = requests.post(
            f"{server_url}/score",
            json={
                "dataset_name": dataset,
                "database_name": db_name,
                "gt_sql": gt_sql,
                "pred_sql": pred_sql,
                "timeout": timeout,
            },
            timeout=timeout + 10,
        )
        result = resp.json()
        return result.get("SCORE", 0.0) if result.get("VALID") == 1 else 0.0
    except Exception:
        return 0.0


def evaluate_dataset(
    dataset_name: str,
    data_path: str,
    llm,
    tokenizer,
    sampling_params,
    server_url: str,
    max_workers: int = 20,
) -> dict:
    """Evaluate a single dataset. Returns metrics dict."""
    with open(data_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    prompts = []
    for sample in test_data:
        text = tokenizer.apply_chat_template(
            sample["messages"], tokenize=False, add_generation_prompt=True
        )
        prompts.append(text)

    outputs = llm.generate(prompts, sampling_params)

    scores = [0.0] * len(outputs)
    tasks = []
    for i, output in enumerate(outputs):
        response = output.outputs[0].text
        pred_sql = extract_sql(response)
        tasks.append({
            "index": i,
            "dataset": test_data[i]["dataset"],
            "db_id": test_data[i]["db_id"],
            "gt_sql": test_data[i]["gt_sql"],
            "pred_sql": pred_sql,
            "response": response,
            "question": test_data[i]["question"],
        })

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                score_sql, server_url,
                t["dataset"], t["db_id"], t["gt_sql"], t["pred_sql"],
            ): t["index"]
            for t in tasks
        }
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc=f"  Scoring {dataset_name}"):
            idx = futures[fut]
            try:
                scores[idx] = fut.result()
            except Exception:
                scores[idx] = 0.0

    correct = sum(1 for s in scores if s >= 1.0)
    total = len(scores)
    ex = correct / total if total > 0 else 0.0

    details = []
    for i, t in enumerate(tasks):
        details.append({
            "db_id": t["db_id"],
            "question": t["question"],
            "gt_sql": t["gt_sql"],
            "pred_sql": t["pred_sql"],
            "score": scores[i],
        })

    return {
        "dataset": dataset_name,
        "total": total,
        "correct": correct,
        "ex": ex,
        "details": details,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Text-to-SQL model")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model or checkpoint")
    parser.add_argument("--eval_dir", type=str, default="eval/",
                        help="Directory containing evaluation datasets")
    parser.add_argument("--output_dir", type=str, default="output/eval_results/",
                        help="Directory to save results")
    parser.add_argument("--server_url", type=str, default=None,
                        help="SQLite server URL (default: $SQLITE_SERVER_URL)")
    parser.add_argument("--datasets", type=str, nargs="+", default=None,
                        help="Specific datasets to evaluate (default: all)")
    parser.add_argument("--tensor_parallel_size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--max_workers", type=int, default=20)
    args = parser.parse_args()

    server_url = args.server_url or os.environ.get(
        "SQLITE_SERVER_URL", "http://localhost:8000"
    )

    datasets = args.datasets or DATASET_NAMES
    for d in datasets:
        if d not in DATASET_NAMES:
            print(f"[ERROR] Unknown dataset: {d}")
            print(f"Available: {DATASET_NAMES}")
            sys.exit(1)

    # Load model
    print(f"Loading model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=0.8,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=0.9,
        max_tokens=args.max_tokens,
    )

    # Evaluate each dataset
    model_name = Path(args.model_path).name
    output_dir = os.path.join(args.output_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)

    summary = {}
    for dataset_name in datasets:
        data_path = os.path.join(args.eval_dir, f"{dataset_name}.json")
        if not os.path.exists(data_path):
            print(f"  [SKIP] {data_path} not found")
            continue

        print(f"\nEvaluating {dataset_name}...")
        result = evaluate_dataset(
            dataset_name, data_path, llm, tokenizer, sampling_params,
            server_url, args.max_workers,
        )

        # Save detailed results
        detail_path = os.path.join(output_dir, f"{dataset_name}.json")
        with open(detail_path, "w", encoding="utf-8") as f:
            json.dump(result["details"], f, ensure_ascii=False, indent=2)

        summary[dataset_name] = {
            "total": result["total"],
            "correct": result["correct"],
            "ex": result["ex"],
        }
        print(f"  {dataset_name}: EX = {result['ex']:.4f} "
              f"({result['correct']}/{result['total']})")

    # Print summary table
    print(f"\n{'='*60}")
    print(f"Model: {args.model_path}")
    print(f"{'='*60}")
    print(f"{'Dataset':<20} {'EX':>8} {'Correct':>8} {'Total':>8}")
    print(f"{'-'*20} {'-'*8} {'-'*8} {'-'*8}")
    for name, m in summary.items():
        print(f"{name:<20} {m['ex']:>8.4f} {m['correct']:>8} {m['total']:>8}")
    if summary:
        avg_ex = sum(m["ex"] for m in summary.values()) / len(summary)
        print(f"{'-'*20} {'-'*8} {'-'*8} {'-'*8}")
        print(f"{'Average':<20} {avg_ex:>8.4f}")
    print(f"{'='*60}")

    # Save summary
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": args.model_path,
            "eval_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "results": summary,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
