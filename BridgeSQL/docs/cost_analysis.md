# BridgeSQL Cost Analysis

This document records the pricing snapshot and calculation scope used for the
BridgeSQL-48k cost estimate reported in the paper. API prices were collected in
May 2026 and are expressed in USD per one million tokens. The reported values
are estimates rather than provider billing records.

## API Pricing Snapshot

| Candidate source / model | Input ($/1M tokens) | Output ($/1M tokens) |
|---|---:|---:|
| GPT-4o | 2.50 | 15.00 |
| Kimi-K2.5 | 0.58 | 3.07 |
| DeepSeek-V3.2 | 0.44 | 0.88 |
| Qwen3-Coder-480B-A35B | 0.88 | 3.51 |
| Arithmetic mean | 1.10 | 5.62 |

The exact mean output price is 5.615 per million tokens and is displayed as
$5.62 after rounding. The peer-review estimate uses the arithmetic mean of the
four candidate-source prices shown above.

The original SynSQL candidate was generated upstream with GPT-4o and is reused
by BridgeSQL; BridgeSQL does not issue an additional GPT-4o request for this
candidate during peer review. The GPT-4o row is retained in the historical
four-source accounting convention used by the paper's estimate.

## BridgeSQL-48k Estimate

| Module | Input (M tokens) | Output (M tokens) | API cost (USD) | End-to-end wall time |
|---|---:|---:|---:|---:|
| DB Population | 10.22 | 4.09 | 18.53 | ~4 h |
| Peer Review | 251.47 | 44.62 | 527.14 | ~40 h |
| **Total** | **261.69** | **48.71** | **545.67** | **~44 h** |

For a module with model-level token totals, its estimated API cost is computed
as:

```text
cost = sum_model(input_tokens_model * input_price_model
               + output_tokens_model * output_price_model) / 1,000,000
```

For the paper's peer-review aggregate estimate:

```text
251.472 * 1.10 + 44.616 * 5.615 = 527.138 USD, reported as 527.14 USD
```

The table displays the corresponding token totals after rounding to two decimal
places.

The total estimated API cost per retained pair is:

```text
545.67 / 48,320 = 0.01129 USD, reported as 0.011 USD per pair
```

## Scope and Runtime

- Wall-clock time covers the full data-construction pipeline, including LLM
  calls, SQL execution, and retries.
- Queries run over tables populated at roughly the 1,000-row scale. At this
  scale, local SQL execution time is negligible relative to LLM API latency and
  is therefore not itemized separately.
- Monetary estimates cover LLM API usage only. They exclude model training,
  GPU/CPU hardware, storage, and human labor.
- Training hardware requirements and distributed settings are documented in
  the repository [README](../README.md#requirements) and
  [`training/config.sh`](../training/config.sh).
