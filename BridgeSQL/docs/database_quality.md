# Database Population Quality Evaluation

We use **8 complementary metrics** to evaluate database population from two
perspectives: four query-independent proxies for data realism and four
held-out SQL-execution metrics. All axes use a larger-is-more orientation for
display. This is not a claim that any single proxy, especially correlation,
entropy-profile spread, or distinctiveness, is universally monotonic with
semantic correctness.

## Fixed comparison protocol

The results below use the same fixed set of 10 database schemas and the
corresponding population produced by each of three systems:

- **BridgeSQL**: the code-driven population produced without evaluation SQLs.
- **Random**: type-matched [Faker](https://faker.readthedocs.io/) population
  with PK/FK constraint satisfaction but no schema-semantic guidance.
- **ParSEval-heldout**: ParSEval v0.0.6 (`0a2df9b9`) population generated from
  the raw SynSQL workload after withholding the first 30 queries of every
  database. These 300 withheld queries are used only for evaluation. Of 1,494
  original queries, 1,194 are population candidates and 1,192 are successfully
  instantiated and merged; no query with an index below 30 enters the merge.

<details>
<summary>Fixed database IDs (10)</summary>

- `academic_research_and_publication_management`
- `air_travel_and_flight_information_management`
- `american_football_player_statistics_and_performance_tracking`
- `apparel_size_and_measurement_management`
- `apparel_sizing_and_measurements`
- `athletic_competition_results_management`
- `automotive_data_management_and_analysis`
- `automotive_sales_and_inventory_management`
- `basketball_player_performance_statistics_and_analytics`
- `biological_data_management_and_analysis`

</details>

The following rules are shared by the four data-realism metrics:

1. Declared primary-key columns and child-side foreign-key columns are
   excluded. Identifier values characterize identity and referential structure,
   and ParSEval's physical merge necessarily remaps many of them.
2. Schema-level candidates are determined from the shared declared schema, not
   separately from each system's realized values. Final inclusion additionally
   requires the joint common-support rule below; this output-dependent test is
   applied symmetrically to all three systems.
3. For a column $c$, let $N_{s,c}$ be the number of non-null values produced
   by system $s$. The paired sample size is

   $$
   m_c = \min(1000, N_{\mathrm{Bridge},c}, N_{\mathrm{Random},c},
   N_{\mathrm{ParSEval},c}).
   $$

   Columns with $m_c < 10$ are skipped jointly. Otherwise, each system uses
   its first $m_c$ non-null values in deterministic `rowid` order. Correlation
   uses the table-level rule defined in its subsection.
4. Each metric is computed per database first and then macro-averaged over the
   10 databases. Large databases therefore do not dominate the headline value.
   Every database has at least one eligible unit for every reported metric, so
   all 10 databases contribute to every macro-average.

The SQL metrics use each system's corresponding populated database and the
same 30 held-out queries per schema. Query errors and timeouts remain in the
relevant denominator and count as failures. SQL is classified with
case-insensitive word-boundary matching for `WHERE` and `JOIN`, including
line-broken queries. The timeout is 3 seconds per query.

## Data realism (4 metrics)

### 1. Non-key Inter-column Correlation

This metric measures the fraction of eligible tables containing at least one
non-trivial linear association between non-key numeric columns. For system
$s$, database $d$, and eligible table set $T_d$,

$$
C_{s,d} = \frac{M_{s,d}}{|T_d|}
$$

where $M_{s,d}$ is the number of tables in $T_d$ that contain at least one
pair of non-key numeric columns whose absolute Pearson correlation is greater
than 0.3. A column is numeric when its shared declared type contains `INT`,
`REAL`, `NUM`, `DEC`, `DOUBLE`, or `FLOAT`. For each table $t$, let $n_{s,t}$
be its row count under system $s$ and define

$$
m_t = \min(1000,n_{\mathrm{Bridge},t},n_{\mathrm{Random},t},
n_{\mathrm{ParSEval},t}).
$$

A table enters all three evaluations when it has at least two shared non-key
numeric columns and $m_t\ge 10$. Each system uses the first $m_t$ table rows
in deterministic `rowid` order. Pearson correlation uses
pairwise-complete observations when a selected value is null. Undefined or
non-finite correlations, including constant-column pairs, are ignored; a table
with no finite pair receives indicator 0 rather than being removed. The final
score is the mean of $C_{s,d}$ across databases. The fixed evaluation set
contains 31 eligible table-database instances.

The metric captures the presence of cross-column statistical structure; it
does not assert that every detected correlation is semantically correct.

### 2. Strict Numeric Range Validity

This metric applies predefined conservative column-name rules, with declared
type constraints where appropriate, to identify bounded or non-negative
numeric domains. For eligible column $c$,

$$
V_{s,c} = \frac{1}{m_c}\sum_{x\in X_{s,c}}
\mathbf{1}[\operatorname{parse}(x)\in R_c],
\qquad
V_{s,d} = \frac{1}{|\mathcal{C}_d|}\sum_{c\in \mathcal{C}_d}V_{s,c}.
$$

`parse` accepts SQLite numeric values or trimmed, plain decimal strings. It
does not remove currency symbols, currency codes, commas, percent signs, or
unit suffixes; for example, `68y` is not treated as a scalar age. The only
structured-string exception is a season-year form such as `2024-25` or
`2024/2025`. A two-digit end is expanded within the start year's century
(`2024-25` becomes `2025`), after which the end must be the same as or
immediately follow the start.

| Rule family | Accepted range |
|---|---|
| Year | integer in `[1800, 2100]` |
| Month / day | month in `[1, 12]`; day in `[1, 31]` |
| Age | numeric value in `[0, 130]` |
| Boolean | strict scalar `0` or `1`, including plain decimal strings; text such as `true/false` is rejected |
| Numeric clock time | valid HHMM from `0000` to `2359` with minutes `< 60` |
| Percentage / rate | `*_pct` in `[0, 1]`; `*_rate` in `[0, 100]` |
| Latitude / longitude | latitude in `[-90, 90]`; longitude in `[-180, 180]` |
| Non-negative quantities | schema-implied counts, capacities, durations, prices, costs, and amounts must be `>= 0` |

Boolean rules take precedence for names beginning with `is_`, `has_`, `can_`,
or `should_`, and for the frozen exact-name set `injured`, `personal_best`,
`affiliated_to_cens`, `active`, `enabled`, and `disabled`. Ambiguous signed
quantities such as `price_difference`, delays, and `plus_minus` are excluded.
The schema rules identify 87 candidate columns; 86 have common support
$m_c \ge 10$ and enter the reported score. The biological database contributes
only one eligible column, so that column determines one tenth of this
database-macro score; this is a coverage limitation of the proxy.

### 3. Categorical Realism

An eligible text column has a shared declared type containing `TEXT`, `CHAR`,
or `CLOB` (so `VARCHAR` is included); `BLOB` and empty declared types are not
included. For each such non-key column $c$, let $K_{s,c}$ be the number of
distinct values in its paired, non-null sample. The column is treated as
categorical when its cardinality ratio is strictly below 0.1:

$$
A_{s,d} = \frac{1}{|\mathcal{C}_d|}\sum_{c\in \mathcal{C}_d}
\mathbf{1}\!\left[\frac{K_{s,c}}{m_c}<0.1\right].
$$

This measures whether the population produces repeated low-cardinality fields
such as status, department, or category instead of making every text value
nearly unique. The common-support evaluation contains 392 eligible
column-database instances.

### 4. Entropy Profile Diversity

We avoid a fixed moderate-entropy threshold because normalizing by the observed
support size can reward a solver that repeatedly draws from a small artificial
value set. For each eligible non-key column, let
$p_{s,c}(v)=\operatorname{count}(v)/m_c$ in its paired, non-null sample. Then

$$
H_{s,c} = -\sum_v p_{s,c}(v)\log_2p_{s,c}(v),
\qquad
h_{s,c} = \frac{H_{s,c}}{\log_2m_c}.
$$

The database-level score is the interquartile range of its column entropies:

$$
E_{s,d}=Q_{0.75}(\{h_{s,c}\})-Q_{0.25}(\{h_{s,c}\}).
$$

Quantiles use linear interpolation. Constant and binary columns are retained.
The metric therefore captures whether a populated relational database contains
a heterogeneous mix of repeated categorical fields and high-cardinality
attributes. It is an entropy-profile statistic, not a judgment about the
meaning of individual values. The common-support evaluation contains 572
column-database instances.

## Held-out SQL execution quality (4 metrics)

Let $Q_d$ denote the 30 held-out SQL queries for database $d$.

### 5. Non-empty Result Rate

$$
N_{s,d}=\frac{\#\{q\in Q_d:\ q\text{ executes and returns at least one row}\}}
{|Q_d|}.
$$

The denominator is always 30. Empty results, errors, and timeouts are failures.

### 6. Result Distinctiveness

Each successful result is canonicalized under **row-set semantics**: rows are
type-tagged, duplicate rows are removed, and row order is ignored. The empty
set has one stable canonical representation.

$$
D_{s,d}=\frac{\#\{\text{distinct successful canonical row sets for }Q_d\}}
{|Q_d|}.
$$

Failed queries produce no result set but remain in the denominator.

### 7. WHERE Reasonableness

For each held-out query containing `WHERE`, the result row count is compared
with the number of rows in the first table matched after `FROM`. The table is
extracted by case-insensitively matching the first simple identifier
immediately after `FROM`; unquoted and square-bracketed SQLite identifiers are
supported. A query is reasonable when it executes successfully, the table is
resolved and non-empty, and the query returns a proper subset:

$$
W_{s,d}=\frac{\#\{q\in Q_d:\ q\text{ contains WHERE and }
0<|q(s,d)|/|\operatorname{FROM}(q,d)|<1\}}
{\#\{q\in Q_d:\ q\text{ contains WHERE}\}}.
$$

The query class is determined before execution, so an error or timeout counts
as a failure rather than disappearing from the denominator.

### 8. JOIN Non-empty Rate

$$
J_{s,d}=\frac{\#\{q\in Q_d:\ q\text{ contains JOIN, executes, and is non-empty}\}}
{\#\{q\in Q_d:\ q\text{ contains JOIN}\}}.
$$

The held-out workload contains 300 queries in total, including 209 `WHERE`
queries and 266 `JOIN` queries. Headline results remain per-database macro
averages rather than pooled query-level ratios.

## Results

All values are unweighted macro-averages over the same 10 databases.

| Category | Metric | BridgeSQL | Random | ParSEval-heldout |
|---|---|---:|---:|---:|
| Data realism | Non-key Inter-column Correlation | **37.0%** | 3.3% | 15.7% |
| Data realism | Strict Numeric Range Validity | **90.0%** | 70.7% | 49.1% |
| Data realism | Categorical Realism | **36.6%** | 0.6% | 0.0% |
| Data realism | Entropy Profile Diversity | **57.3%** | 2.4% | 10.4% |
| SQL execution | Non-empty Result Rate | **78.7%** | 58.3% | 75.0% |
| SQL execution | Result Distinctiveness | **78.3%** | 59.3% | 77.0% |
| SQL execution | WHERE Reasonableness | **69.9%** | 41.8% | 63.6% |
| SQL execution | JOIN Non-empty Rate | **78.1%** | 57.4% | 73.3% |

BridgeSQL has the largest value on all eight reported axes. Under these
proxies, ParSEval remains competitive on held-out query execution, especially
non-empty results and result distinctiveness, but exhibits substantially less
categorical and entropy-profile structure. Random population satisfies many
basic non-negative numeric checks but provides little cross-column or
categorical structure under the corresponding proxies.

The single-population entry point is
[`evaluation/evaluate_db_quality.py`](../evaluation/evaluate_db_quality.py).
