# Peer-review Retention Analysis

This analysis asks a narrower question than the
[SQL-complexity comparison](sql_complexity.md): **which properties of an
upstream SQL example are associated with its being retained by peer review?**
To avoid post-selection leakage, every difficulty label and SQL feature is
extracted from the original `questions/questions.json` record. The
reviewer-selected SQL in the verified file is used only for the separate
replacement analysis.

## Accounting and consensus

The source snapshot contains 156,820 generated candidates across 1,022 question-bearing databases. Of these, 155,584 candidates from the 1,014 populated databases have complete original, raw-review, and verified files and therefore enter this analysis. The remaining 1,236 candidates from eight databases were not peer reviewed and are excluded from both the numerator and denominator.

| Outcome | Count | Share of evaluated pool |
|---|---:|---:|
| Evaluated candidates | 155,584 | 100.00% |
| Accepted | 48,320 | 31.06% |
| Rejected | 107,264 | 68.94% |

| Largest identical-result group (`consensus_count`) | Count | Share |
|---------------------------------------------------:|---:|---:|
|                                                  0 | 31,676 | 20.36% |
|                                                  1 | 40,574 | 26.08% |
|                                                  2 | 35,014 | 22.50% |
|                                                  3 | 23,405 | 15.04% |
|                                                  4 | 24,915 | 16.01% |

All 155,584 records in this cached run contain four non-empty candidate SQLs. With four candidates, the configured threshold `theta=0.6` requires a consensus group of at least three; the cached `review_status` agrees with this rule for every record. `consensus_count=0` should not be interpreted as a single failure type: the stored verified file does not reliably separate execution errors, empty results, timeouts, and other cases that fail to form a valid result group.

The pooled acceptance rate is 31.06%. Across databases, the unweighted mean is 31.28% and the median is 31.12%; the aggregate JSON also reports per-database macro statistics for every group below.

## Retention by upstream difficulty

`Relative retention` is the within-group acceptance rate divided by the overall 31.06% rate. It is descriptive rather than a causal effect.

| Upstream difficulty | Candidates | Accepted | Rejected | Acceptance rate | Relative retention |
|---|---:|---:|---:|---:|---:|
| Simple | 14,689 | 8,811 | 5,878 | 59.98% | 1.93x |
| Moderate | 47,001 | 16,720 | 30,281 | 35.57% | 1.15x |
| Complex | 53,958 | 14,516 | 39,442 | 26.90% | 0.87x |
| Highly Complex | 39,936 | 8,273 | 31,663 | 20.72% | 0.67x |

## Retention by structural depth

| JOIN count | Candidates | Accepted | Rejected | Acceptance rate | Relative retention |
|---|---:|---:|---:|---:|---:|
| 0 | 15,634 | 9,659 | 5,975 | 61.78% | 1.99x |
| 1 | 39,907 | 15,927 | 23,980 | 39.91% | 1.29x |
| 2 | 46,485 | 13,815 | 32,670 | 29.72% | 0.96x |
| 3 | 28,031 | 5,954 | 22,077 | 21.24% | 0.68x |
| 4+ | 25,527 | 2,965 | 22,562 | 11.62% | 0.37x |

| SELECT-block count | Candidates | Accepted | Rejected | Acceptance rate | Relative retention |
|---|---:|---:|---:|---:|---:|
| 1 | 78,159 | 31,736 | 46,423 | 40.60% | 1.31x |
| 2 | 25,782 | 8,331 | 17,451 | 32.31% | 1.04x |
| 3 | 26,670 | 5,392 | 21,278 | 20.22% | 0.65x |
| 4+ | 24,973 | 2,861 | 22,112 | 11.46% | 0.37x |

## Retention by SQL feature

Features overlap, so rows must not be summed. Detection is case-insensitive after SQL comments, string literals, and quoted identifiers are masked. The table reports features of the upstream SQL, even when a different reviewer SQL is selected later.

| Upstream SQL feature | Candidates | Accepted | Rejected | Acceptance rate | Relative retention |
|---|---:|---:|---:|---:|---:|
| JOIN | 139,950 | 38,661 | 101,289 | 27.62% | 0.89x |
| Multiple JOINs | 100,043 | 22,734 | 77,309 | 22.72% | 0.73x |
| JOIN >= 3 | 53,558 | 8,919 | 44,639 | 16.65% | 0.54x |
| GROUP BY | 101,227 | 24,999 | 76,228 | 24.70% | 0.80x |
| HAVING | 31,110 | 8,014 | 23,096 | 25.76% | 0.83x |
| Aggregation | 115,542 | 34,060 | 81,482 | 29.48% | 0.95x |
| Subquery | 77,292 | 16,536 | 60,756 | 21.39% | 0.69x |
| Multiple Subqueries | 51,379 | 8,155 | 43,224 | 15.87% | 0.51x |
| CTE | 66,178 | 12,113 | 54,065 | 18.30% | 0.59x |
| Recursive CTE | 695 | 33 | 662 | 4.75% | 0.15x |
| Window Function | 38,019 | 6,580 | 31,439 | 17.31% | 0.56x |
| ROW_NUMBER | 23,254 | 4,451 | 18,803 | 19.14% | 0.62x |
| RANK / DENSE_RANK | 10,824 | 1,769 | 9,055 | 16.34% | 0.53x |
| LAG / LEAD | 964 | 64 | 900 | 6.64% | 0.21x |
| Window Aggregate | 4,317 | 361 | 3,956 | 8.36% | 0.27x |
| Set Operation | 1,987 | 307 | 1,680 | 15.45% | 0.50x |
| CASE | 7,526 | 807 | 6,719 | 10.72% | 0.35x |
| DISTINCT | 13,531 | 4,436 | 9,095 | 32.78% | 1.06x |
| CTE + Window | 36,782 | 6,333 | 30,449 | 17.22% | 0.55x |
| JOIN + Subquery | 72,327 | 14,167 | 58,160 | 19.59% | 0.63x |
| Aggregate + Window | 26,411 | 3,927 | 22,484 | 14.87% | 0.48x |

## Retention by question style

| Question style | Candidates | Accepted | Rejected | Acceptance rate | Relative retention |
|---|---:|---:|---:|---:|---:|
| Colloquial | 17,161 | 5,065 | 12,096 | 29.51% | 0.95x |
| Concise | 17,368 | 5,895 | 11,473 | 33.94% | 1.09x |
| Descriptive | 17,451 | 5,848 | 11,603 | 33.51% | 1.08x |
| Formal | 17,315 | 5,916 | 11,399 | 34.17% | 1.10x |
| Imperative | 17,288 | 5,628 | 11,660 | 32.55% | 1.05x |
| Interrogative | 17,418 | 5,579 | 11,839 | 32.03% | 1.03x |
| Metaphorical | 17,169 | 4,289 | 12,880 | 24.98% | 0.80x |
| Multi-turn Dialogue | 17,109 | 5,266 | 11,843 | 30.78% | 0.99x |
| Vague | 17,305 | 4,834 | 12,471 | 27.93% | 0.90x |

## Candidate completeness and selected-SQL replacement

The script records the number of non-empty candidate SQLs per example so that API-call failures can be audited. There is no availability variation in this snapshot: every analyzed example has all four configured candidates, so lower retention for complex structures cannot be attributed to missing candidates here.

Among the 48,320 accepted examples, the selected source is distributed as follows:

| Selected source | Count | Share of accepted set |
|---|---:|---:|
| `qwen3-coder-480b-a35b-instruct` | 13,212 | 27.34% |
| `kimi-k2.5` | 13,081 | 27.07% |
| `deepseek-v3.2` | 11,699 | 24.21% |
| `synsql` | 10,328 | 21.37% |

The selected source differs from the upstream `synsql` slot in 37,992 accepted examples (78.63%). Selected SQL text differs exactly from the upstream SQL in 36,754 examples (76.06%), or in 34,757 examples (71.93%) after case/whitespace normalization that preserves literal and quoted-identifier contents. These are replacement diagnostics, not additional retention rates; different candidate sources can still contain textually identical SQL.

## Reproducing the analysis

The strict command below audits the paper snapshot, which uses four configured candidate slots and therefore requires a consensus count of three. The model names in `llm_utils.py` are configuration examples: set the reviewer slots to match the run being analyzed. For a custom run with a different number of slots, adjust or omit the two `--expected_*` arguments.

```bash
python evaluation/evaluate_peer_review_retention.py \
    --data_root output/data_synthesis/ \
    --db_list output/data_synthesis/valid_databases.json \
    --source_label BridgeSQL-48k \
    --expected_acceptance_consensus 3 \
    --expected_candidate_slots 4 \
    --strict \
    --detailed \
    --output output/analysis/peer_review_retention.json \
    --csv_output output/analysis/peer_review_retention.csv
```

The aggregate JSON includes raw counts, pooled and per-database acceptance rates, consensus distributions, recorded-slot and non-empty-candidate counts, all feature groups, database-level summaries, and sanity checks. The aggregate CSV flattens the group tables. The optional `--records_output` argument produces a roughly 34 MB CSV for this snapshot, using the stable key `<db_id>#<zero-based-index>` because question text is not unique; omit it if only aggregate statistics are needed. `--strict` returns a non-zero exit code if a pairing or review invariant fails. See [`evaluate_peer_review_retention.py`](../evaluation/evaluate_peer_review_retention.py) for the exact feature definitions.
