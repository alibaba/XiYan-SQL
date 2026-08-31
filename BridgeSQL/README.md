# BridgeSQL

**Bridging the Synthesis-Execution Gap in Text-to-SQL with Execution-Grounded Data Curation**

This repository contains the code, data processing scripts, and training configurations for reproducing the results presented in the paper.

## Overview

BridgeSQL is a pipeline-agnostic, execution-grounded framework for data curation that operates downstream of any from-scratch Text-to-SQL synthesis pipeline. It addresses the Synthesis-Execution Gap: existing pipelines produce structurally rich schemas and candidate queries but lack populated databases, leaving the synthesized data without executable environments for quality verification or execution-based training.

The framework introduces two core modules:

1. **Code-driven Database Population**: An LLM-guided technique that populates empty database schemas with large-scale, constraint-satisfying data by generating intermediate Python code, enabling execution-based feedback.
2. **Execution-Grounded Peer Review**: A multi-model consensus mechanism that validates Question-SQL pairs by executing candidate queries against populated databases, ensuring logical correctness and semantic alignment.

Together, these modules enable a two-stage **SFT + RLVR** training paradigm for Small Language Models (SLMs).

## Dataset

### Source Data

Our framework builds upon schemas and candidate questions from the [SynSQL-2.5M](https://huggingface.co/datasets/seeklhy/SynSQL-2.5M) corpus as the upstream synthesis pipeline.

| Stage | Description |
|---|---|
| Schema Selection | 1,200 databases selected from SynSQL-2.5M (~16,500 databases) via K-Means clustering using text embeddings |
| Schema Validation | Foreign key and topology checks retain 1,022 valid databases |
| Database Population | 1,014 databases successfully populated (99.2% success rate), initialized with 1,000 records per table |
| Candidate Questions | 156,820 questions from 1,022 valid databases; 155,584 from the 1,014 populated databases enter peer review |
| Peer Review | 48,320 verified Question-SQL pairs retained (31.1% of the evaluated pool; 30.8% of all generated candidates) |

### Synthesized Dataset

| Dataset | Databases | Question-SQL Pairs | Usage |
|---|---|---|---|
| BridgeSQL-48k | 1,014 | 48,320 | Full training set |
| BridgeSQL-5k | 98 | 5,204 | Ablation study |

BridgeSQL-5k is produced by running the same pipeline with `--n_clusters 100` in the schema selection step. After validation, 98 databases remain, yielding 5,204 verified pairs under the same peer review threshold (θ=0.6).

### Data Format

Intermediate curated records produced by the synthesis pipeline contain:
- `db_id`: Database identifier
- `question`: Natural language question
- `SQL`: Verified SQL query
- `schema`: Database schema in structured format (M-Schema)
- `cot` (optional): Chain-of-Thought reasoning path

Released SFT and RLVR snapshots use training-specific schemas; see the README
inside each dataset directory for their exact fields. Populated SQLite databases
are generated in Step 1.2 and are not included in this code package. The larger
BridgeSQL-48k/5k data and database artifacts will be released separately.

## Repository Structure

| Directory | Description |
|---|---|
| `data_synthesis/` | Schema selection, database population, and peer review |
| `schema_engine/` | Database schema utilities (M-Schema extraction) |
| `training/` | SFT and RLVR training scripts, reward server, data preprocessing |
| `evaluation/` | Benchmark preparation, evaluation, and peer-review retention analysis |
| `datasets/` | Released BridgeSQL dataset snapshots used in the paper |
| `docs/` | Detailed evaluation protocols, analyses, result tables, and cost accounting |
| `data/` | Downloaded external data (Step 0) |
| `output/` | All generated outputs (data synthesis, training data, checkpoints, eval results) |

Detailed protocols, supplementary result tables, and accounting notes are
indexed in the [`docs` README](docs/README.md). Direct entries include the
[cost analysis](docs/cost_analysis.md), complete
[ablation results](docs/ablation_results.md), and
[peer-review retention analysis](docs/peer_review_retention.md). Concise
evaluation commands are collected in the
[`evaluation` README](evaluation/README.md).

## Requirements

### Environment

- Python >= 3.10
- CUDA >= 12.1
- 4× NVIDIA A100 (80GB) per node (2 nodes for 7B RLVR training)

### Dependencies

```bash
pip install -r requirements.txt
```

Core packages (see `requirements.txt` for full list):

| Package | Version | Purpose |
|---|---|---|
| ms-swift | 3.11.0 | SFT & RLVR training framework |
| torch | 2.8.0 | Deep learning backend |
| vllm | 0.11.0 | Fast inference (RLVR generation & evaluation) |
| deepspeed | 0.18.1 | Distributed training (ZeRO-3) |
| transformers | 4.57.1 | Model loading & tokenization |
| flash-attn | 2.8.1 | Memory-efficient attention |
| accelerate | 1.11.0 | Training utilities |
| faker | — | Synthetic data generation |
| flask | — | SQLite execution server |

## Reproduction Guide

### Step 0: Data Preparation

All external data (upstream SynSQL-2.5M and evaluation benchmarks) is bundled in a single download from [OmniSQL-datasets](https://huggingface.co/datasets/seeklhy/OmniSQL-datasets).

```bash
# Download the dataset (~22.2 GB)
pip install -U huggingface_hub
huggingface-cli download seeklhy/OmniSQL-datasets --repo-type dataset --local-dir downloads/

# Unzip into project root (creates data/ directory)
unzip downloads/data.zip -d .

# Clean up
rm -rf downloads/
```



**Note:** Due to the inherent randomness of database population (random seed, Faker library) and non-determinism of LLM outputs, the exact synthesized data will differ across runs. The overall statistics (population success rate, acceptance rate) should be comparable.

### Step 1: Data Synthesis

#### 1.1 Schema Selection and Validation

Select 1,200 representative databases from SynSQL-2.5M via K-Means clustering, then validate foreign key and topology constraints (1,022 databases pass validation). Use `--extract_questions` to also extract candidate Question-SQL pairs from `data.json` into per-database directories for peer review:

```bash
python data_synthesis/filter_databases.py \
    --source_path data/SynSQL-2.5M \
    --output_path output/data_synthesis/ \
    --n_clusters 1200 \
    --extract_questions
```

#### 1.2 Code-driven Database Population

Before running, configure your LLM and embedding endpoints in `data_synthesis/llm_utils.py` by editing the `MODEL_CONFIG` dictionary (OpenAI-compatible API):

```python
MODEL_CONFIG = {
    "text-embedding-v4": {
        "base_url": "https://your-embedding-endpoint/v1",
        "api_key": "your-api-key",
    },
    "kimi-k2.5": {
        "base_url": "https://your-chat-endpoint/v1",
        "api_key": "your-api-key",
    },
}
```

Populate the validated schemas with synthetic data (1,000 rows per table). This step uses LLM-generated Python code to create semantically plausible records while enforcing primary key, foreign key, and type constraints. Use `--db_root` to automatically copy the populated databases into the unified database root (`$DB_ROOT`) for RLVR training:

```bash
python data_synthesis/database_population.py \
    --output_path output/data_synthesis/ \
    --n_rows 1000 \
    --db_root $DB_ROOT
```

#### 1.3 Execution-Grounded Peer Review

Run multi-model consensus voting on candidate Question-SQL pairs. Each question is answered by all reviewer models configured in `llm_utils.py`; their SQL outputs are executed against the populated database and compared. A pair is accepted when the fraction of candidates producing identical results meets the threshold (default θ=0.6):

```bash
python data_synthesis/peer_review.py \
    --output_path output/data_synthesis/ \
    --threshold 0.6
```

### Step 2: Training

#### 2.1 Data Preprocessing

Convert the verified Q-SQL pairs into training formats. This produces three files: SFT train/dev (databases split 95%/5%) and RLVR (all databases, no split):

```bash
python training/data_processing/process_bridgesql.py \
    --output_path output/data_synthesis/ \
    --save_dir output/training_data/
```

Output:
- `output/training_data/bridgesql_sft_train.json` — SFT training set with CoT responses
- `output/training_data/bridgesql_sft_dev.json` — SFT validation set
- `output/training_data/bridgesql_rl.json` — RLVR dataset (prompt-only, with `gt_sql` for reward)

#### 2.2 Supervised Fine-Tuning (SFT)

Edit `training/config.sh` to set your model paths and data paths. The hyperparameters in the training scripts match the paper's reported settings; adjust batch sizes, gradient accumulation, and number of GPUs/nodes according to your specific hardware:

```bash
# Train 7B model (also supports 0.5b, 1.5b)
MODEL_SIZE=7b bash training/sft/train.sh
```

#### 2.3 Reinforcement Learning with Verifiable Rewards (RLVR)

Start the SQLite execution server for reward computation. We recommend running the server on a separate machine within the same LAN that has sufficient CPU and memory resources, so that SQL execution does not compete with GPU training workloads. The training machine itself also works if resources permit:

```bash
# On the execution server machine (copy databases/ to this machine first)
python training/reward_utils/sqlite_server.py \
    --db_dir databases/ \
    --port 8000 \
    --workers 200
```

Then launch RLVR training, pointing to the SFT checkpoint and the execution server.

For 0.5B/1.5B (single node, 4× A100):

```bash
export SQLITE_SERVER_URL=http://<server_ip>:8000

MODEL_SIZE=1.5b \
SFT_CKPT=output/checkpoints/sft/1.5b/checkpoint-400 \
bash training/rlvr/train.sh
```

For 7B (2 nodes × 4× A100, run on each node):

```bash
export SQLITE_SERVER_URL=http://<server_ip>:8000

# Node 0 (master)
MODEL_SIZE=7b \
SFT_CKPT=output/checkpoints/sft/7b/checkpoint-280 \
NNODES=2 NODE_RANK=0 MASTER_ADDR=<node0_ip> \
bash training/rlvr/train.sh

# Node 1 (worker) — run on the second machine
MODEL_SIZE=7b \
SFT_CKPT=output/checkpoints/sft/7b/checkpoint-280 \
NNODES=2 NODE_RANK=1 MASTER_ADDR=<node0_ip> \
bash training/rlvr/train.sh
```

### Step 3: Evaluation

#### 3.1 Organize Benchmark Databases

Reorganize all benchmark SQLite databases into a unified structure. This step should be completed **before** RLVR training (Step 2.3), as the execution server also serves from `$DB_ROOT`. You can set `DB_ROOT` to place the databases on a fast storage device (e.g., SSD or tmpfs) for better I/O performance during evaluation and RLVR training:

```bash
# Default: databases/ in project root
export DB_ROOT=databases/

# Or use a custom path, e.g.:
# export DB_ROOT=/dev/shm/bridge_sql_databases/

python evaluation/prepare_databases.py \
    --data_dir data/ \
    --output_dir $DB_ROOT
```

#### 3.2 Generate Evaluation Datasets

Convert raw benchmark data into unified evaluation format. M-Schema is extracted directly from the SQLite databases (no separate schema extraction step needed):

```bash
python evaluation/prepare_eval_dataset.py \
    --data_dir data/ \
    --db_dir $DB_ROOT \
    --output_dir eval/
```

#### 3.3 Run Evaluation

Ensure the SQLite execution server is running (same server as RLVR training), then evaluate on all seven benchmarks:

```bash
# Start the execution server (if not already running)
python training/reward_utils/sqlite_server.py \
    --db_dir $DB_ROOT --port 8000 --workers 200

# Evaluate a checkpoint
bash evaluation/run_eval.sh --model_path <checkpoint_path>

# Evaluate on specific datasets only
bash evaluation/run_eval.sh --model_path <checkpoint_path> \
    --datasets spider_test bird_dev

# Custom tensor parallel size (default: 4)
bash evaluation/run_eval.sh --model_path <checkpoint_path> --tp 2
```

Results are saved to `output/eval_results/<model_name>/` with per-dataset detail files and a `summary.json`.

## Training Configurations

### SFT Hyperparameters

| Parameter | 0.5B / 1.5B | 7B |
|---|---|---|
| Batch Size | 256 | 256 |
| Learning Rate | 3e-5 | 2e-5 |
| Warmup Ratio | 3% | 3% |
| LR Schedule | Cosine | Cosine |
| Weight Decay | 0.1 | 0.1 |
| Optimizer | AdamW (ZeRO-3) | AdamW (ZeRO-3) |

### RLVR Hyperparameters (GRPO)

| Parameter | 0.5B / 1.5B | 7B |
|---|---|---|
| Learning Rate | 5e-6 | 1e-6 |
| Batch Size | 64 prompts | 64 prompts |
| Generation Size | 8 | 8 |
| Temperature | 1.0 | 1.0 |
| Top-p | 0.9 | 0.9 |
| KL Coefficient | 0.001 | 0.001 |
| Max Prompt Length | 25,000 | 25,000 |
| Max Completion Length | 2,048 | 2,048 |
| Resample Iterations | 3 | 3 |

## Benchmarks

We evaluate on seven cross-domain benchmarks using Execution Accuracy (EX):

| Benchmark | # Examples | Source |
|---|---|---|
| Spider (Test) | 2,147 | [Link](https://yale-lily.github.io/spider) |
| BIRD (Dev) | 1,534 | [Link](https://bird-bench.github.io/) |
| Science Benchmark | 299 | [Link](https://github.com/AlanFeder/ScienceBenchmark) |
| EHRSQL | 1,008 | [Link](https://github.com/glee4810/EHRSQL) |
| Spider-DK | 535 | [Link](https://github.com/ygan/Spider-DK) |
| Spider-Syn | 1,034 | [Link](https://github.com/ygan/Spider-Syn) |
| Spider-Realistic | 508 | [Link](https://github.com/ygan/Spider-Realistic) |

## Results

Main results: Execution Accuracy (%) across seven benchmarks. Models are Qwen2.5-Coder at three scales, trained on different synthetic datasets.

### Qwen2.5-Coder-0.5B-Instruct

| Train Data | Training | Spider | BIRD | Science | EHRSQL | Spider-DK | Spider-Syn | Spider-Real. | Avg |
|---|---|---|---|---|---|---|---|---|---|
| - (base) | - | 34.0 | 7.4 | 10.4 | 5.4 | 32.3 | 26.0 | 26.8 | 20.3 |
| SynSQL-50k | SFT | 50.8 | 23.9 | 19.1 | 12.9 | 45.6 | 38.1 | 42.9 | 33.3 |
| SingSQL-34k | SFT | 40.5 | 11.5 | 9.4 | 9.2 | 34.0 | 24.1 | 31.1 | 22.8 |
| BridgeSQL-5k | SFT+RLVR | 55.3 | 26.8 | 20.1 | 18.7 | 46.7 | 39.8 | 46.1 | 36.2 |
| BridgeSQL-48k | SFT | 59.2 | 27.6 | 21.7 | 18.7 | 49.3 | 42.6 | 50.8 | 38.6 |
| **BridgeSQL-48k** | **SFT+RLVR** | **61.1** | **32.4** | **23.1** | **21.1** | **52.3** | **49.7** | **56.7** | **42.4** |

### Qwen2.5-Coder-1.5B-Instruct

| Train Data | Training | Spider | BIRD | Science | EHRSQL | Spider-DK | Spider-Syn | Spider-Real. | Avg |
|---|---|---|---|---|---|---|---|---|---|
| - (base) | - | 51.1 | 24.3 | 17.7 | 12.6 | 43.7 | 36.4 | 39.8 | 32.2 |
| SynSQL-50k | SFT | 68.5 | 41.3 | 30.1 | 27.4 | 56.3 | 52.9 | 60.2 | 48.1 |
| SingSQL-34k | SFT | 60.9 | 28.3 | 22.7 | 14.0 | 49.3 | 40.4 | 57.5 | 39.0 |
| BridgeSQL-5k | SFT+RLVR | 67.9 | 41.7 | 24.4 | 31.4 | 59.8 | 53.6 | 64.2 | 49.0 |
| BridgeSQL-48k | SFT | **71.0** | 45.1 | 31.4 | 28.1 | 61.1 | 59.4 | 67.5 | 51.9 |
| **BridgeSQL-48k** | **SFT+RLVR** | **71.0** | **45.4** | **36.8** | **36.4** | **63.6** | **63.3** | **69.1** | **55.1** |

### Qwen2.5-Coder-7B-Instruct

| Train Data | Training | Spider | BIRD | Science | EHRSQL | Spider-DK | Spider-Syn | Spider-Real. | Avg |
|---|---|---|---|---|---|---|---|---|---|
| - (base) | - | 74.9 | 48.9 | 41.1 | 20.6 | 63.2 | 63.2 | 67.3 | 54.2 |
| SynSQL-50k | SFT | 75.7 | 53.6 | 37.8 | 34.6 | 63.0 | 65.6 | 73.6 | 57.7 |
| SynSQL-2.5M | SFT | 76.7 | 56.0 | 37.1 | 32.3 | 64.3 | 66.3 | 74.2 | 58.2 |
| SingSQL-34k | SFT | 71.2 | 45.4 | 33.1 | 21.6 | 58.9 | 55.0 | 69.7 | 50.7 |
| BridgeSQL-5k | SFT+RLVR | **77.7** | 58.6 | 40.1 | 34.4 | **68.0** | **70.1** | 75.6 | 60.7 |
| BridgeSQL-48k | SFT | 75.1 | 57.0 | 41.8 | 34.3 | 64.9 | 69.4 | **78.0** | 60.1 |
| **BridgeSQL-48k** | **SFT+RLVR** | 77.1 | **58.7** | **46.5** | **38.0** | 65.6 | 69.0 | 77.2 | **61.7** |

The complete seven-benchmark ablation results are available in
[`docs/ablation_results.md`](docs/ablation_results.md).

### Reproduce

```bash
# Ensure the execution server is running
python training/reward_utils/sqlite_server.py \
    --db_dir $DB_ROOT --port 8000 --workers 200

# Evaluate a trained checkpoint on all 7 benchmarks
bash evaluation/run_eval.sh --model_path <checkpoint_path>
```

Results are saved to `output/eval_results/<model_name>/summary.json`.

## License

The source code in this repository is licensed under the
[Apache License 2.0](LICENSE). Dataset artifacts are not covered by this code
license and remain subject to their respective source licenses. The paper is
governed by the applicable IEEE publishing agreement.
