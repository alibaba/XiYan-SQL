# BridgeSQL Ablation Results

This document expands the macro-average ablation results reported in the paper
into complete per-benchmark execution accuracy (EX) results. It follows the
current two-panel interpretation of the ablation study and does not introduce
additional experimental conditions.

All results use Qwen2.5-Coder-Instruct models and the same seven-benchmark
evaluation protocol: Spider Test, BIRD Dev, Science Benchmark, EHRSQL,
Spider-DK, Spider-Syn, and Spider-Realistic. `Avg` is the unweighted macro
average over these seven benchmarks. Averages are computed from the underlying
unrounded scores, so they may differ slightly from an average recomputed from
the displayed one-decimal values. The reported controls use one training run
per configuration.

## A. Database Population and RLVR

This panel compares SFT with SFT+RLVR under each data source. Database
population supplies the executable environments required by RLVR. The rows are
end-to-end training configurations; they do not independently isolate database
population from the RLVR optimization stage.

| Model | Training data | Pairs | Training | Spider | BIRD | Science | EHRSQL | Spider-DK | Spider-Syn | Spider-Real. | Avg |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-Coder-1.5B-Instruct | SynSQL-15k | 15.4k | SFT | 64.5 | 34.5 | 28.8 | 21.1 | 52.7 | 47.6 | 51.4 | 42.9 |
| Qwen2.5-Coder-1.5B-Instruct | SynSQL-15k | 15.4k | SFT+RLVR | 69.9 | 38.8 | 28.8 | 31.0 | 56.6 | 50.4 | 62.4 | 48.3 |
| Qwen2.5-Coder-1.5B-Instruct | BridgeSQL-5k | 5.2k | SFT | 64.6 | 34.7 | 27.1 | 23.5 | 52.9 | 50.0 | 56.3 | 44.1 |
| Qwen2.5-Coder-1.5B-Instruct | BridgeSQL-5k | 5.2k | SFT+RLVR | 67.9 | 41.7 | 24.4 | 31.4 | 59.8 | 53.6 | 64.2 | 49.0 |
| Qwen2.5-Coder-7B-Instruct | SynSQL-15k | 15.4k | SFT | 75.4 | 53.1 | 36.5 | 29.3 | 62.1 | 65.2 | 73.4 | 56.4 |
| Qwen2.5-Coder-7B-Instruct | SynSQL-15k | 15.4k | SFT+RLVR | 78.2 | 55.2 | 39.1 | 32.0 | 66.0 | 67.3 | 75.8 | 59.1 |
| Qwen2.5-Coder-7B-Instruct | BridgeSQL-5k | 5.2k | SFT | 76.4 | 54.4 | 37.8 | 28.9 | 64.3 | 67.5 | 75.6 | 57.8 |
| Qwen2.5-Coder-7B-Instruct | BridgeSQL-5k | 5.2k | SFT+RLVR | 77.7 | 58.6 | 40.1 | 34.4 | 68.0 | 70.1 | 75.6 | 60.7 |

Adding RLVR raises the average EX of SynSQL-15k from 42.9 to 48.3 at
1.5B and from 56.4 to 59.1 at 7B. BridgeSQL-5k improves from 44.1 to 49.0
and from 57.8 to 60.7, respectively.

## B. Execution-Grounded Peer Review

All configurations below use SFT, contain 5,204 training pairs, and share the
same training and evaluation setup. They vary along two dimensions:

- **Question selection**: `Random` uses a fixed-seed, database-matched random
  sample from the upstream pool, while `Peer Review` uses the questions retained
  by execution-grounded review.
- **Supervision source**: `Original SynSQL` uses the upstream CoT and SQL;
  `Peer Review` uses the consensus-supported CoT and SQL selected by the review
  procedure.

| Model | Question selection | Supervision source | Training | Spider | BIRD | Science | EHRSQL | Spider-DK | Spider-Syn | Spider-Real. | Avg |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-Coder-1.5B-Instruct | Random | Original SynSQL | SFT | 61.6 | 32.5 | 22.1 | 19.6 | 51.8 | 43.9 | 53.0 | 40.6 |
| Qwen2.5-Coder-1.5B-Instruct | Peer Review | Original SynSQL | SFT | 61.0 | 32.7 | 24.4 | 20.8 | 52.0 | 43.3 | 54.5 | 41.3 |
| Qwen2.5-Coder-1.5B-Instruct | Peer Review | Peer Review | SFT | 64.6 | 34.7 | 27.1 | 23.5 | 52.9 | 50.0 | 56.3 | 44.1 |
| Qwen2.5-Coder-7B-Instruct | Random | Original SynSQL | SFT | 73.4 | 51.7 | 37.8 | 28.5 | 62.1 | 62.3 | 69.9 | 55.1 |
| Qwen2.5-Coder-7B-Instruct | Peer Review | Original SynSQL | SFT | 74.1 | 52.7 | 38.1 | 28.6 | 60.6 | 63.5 | 70.7 | 55.5 |
| Qwen2.5-Coder-7B-Instruct | Peer Review | Peer Review | SFT | 76.4 | 54.4 | 37.8 | 28.9 | 64.3 | 67.5 | 75.6 | 57.8 |

With the supervision source fixed to the original SynSQL trajectories, Peer
Review question selection yields modest single-run differences: average EX
changes from 40.6 to 41.3 at 1.5B and from 55.1 to 55.5 at 7B. With the
selected questions fixed, replacing the original supervision with Peer Review
supervision raises average EX from 41.3 to 44.1 and from 55.5 to 57.8. This
comparison evaluates the complete consensus-supported target-construction step;
it does not isolate reviewer candidate quality from the execution-based
consensus restriction.
