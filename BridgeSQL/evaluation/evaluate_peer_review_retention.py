"""Analyze which upstream SQL patterns are retained by peer review.

The verified record may contain a reviewer-selected SQL query.  Using that
query to measure retention would conflate question selection with supervision
replacement.  This script therefore extracts every SQL feature from the
original ``questions/questions.json`` record and uses the aligned verified
record only for review status, consensus count, selected source, and selected
SQL.

Supported verified filenames, in priority order:

* ``questions/verified.json`` (public pipeline output)
* ``questions/peer_reviews_with_cot_verified.json`` (paper cache)

Example:

    # Paper snapshot: four configured candidate slots, accept at three votes.
    # Adjust or omit the two --expected_* checks for a custom configuration.

    python evaluation/evaluate_peer_review_retention.py \
        --data_root output/data_synthesis/ \
        --db_list output/data_synthesis/valid_databases.json \
        --source_label BridgeSQL-48k \
        --expected_acceptance_consensus 3 \
        --expected_candidate_slots 4 \
        --strict \
        --detailed \
        --output output/analysis/peer_review_retention.json \
        --csv_output output/analysis/peer_review_retention.csv \
        --records_output output/analysis/peer_review_retention_records.csv

Only the Python standard library is required.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


VERIFIED_FILENAMES = (
    "verified.json",
    "peer_reviews_with_cot_verified.json",
)

RAW_REVIEW_FILENAMES = (
    "peer_reviews.json",
    "peer_reviews_with_cot.json",
)

# Features intentionally overlap.  For example, a query with ROW_NUMBER also
# contributes to the broader Window Function bucket.
FEATURE_PATTERNS = {
    "JOIN": r"\bJOIN\b",
    "GROUP BY": r"\bGROUP\s+BY\b",
    "ORDER BY": r"\bORDER\s+BY\b",
    "HAVING": r"\bHAVING\b",
    "Subquery": r"\(\s*SELECT\b",
    "CTE": r"\bWITH\b(?!\s+ROLLUP)",
    "Recursive CTE": r"\bWITH\s+RECURSIVE\b",
    "Window Function": r"\bOVER\s*\(",
    "ROW_NUMBER": r"\bROW_NUMBER\s*\(",
    "RANK": r"\bRANK\s*\(",
    "DENSE_RANK": r"\bDENSE_RANK\s*\(",
    "NTILE": r"\bNTILE\s*\(",
    "LAG": r"\bLAG\s*\(",
    "LEAD": r"\bLEAD\s*\(",
    "UNION": r"\bUNION\b",
    "INTERSECT": r"\bINTERSECT\b",
    "EXCEPT": r"\bEXCEPT\b",
    "CASE": r"\bCASE\b",
    "EXISTS": r"\bEXISTS\s*\(",
    "IN Subquery": r"\bIN\s*\(\s*SELECT\b",
    "DISTINCT": r"\bDISTINCT\b",
    "Aggregation": r"\b(?:COUNT|SUM|AVG|MIN|MAX)\s*\(",
    "LIMIT": r"\bLIMIT\b",
}

DERIVED_FEATURES = (
    "Multiple JOINs",
    "JOIN >= 3",
    "Multiple Subqueries",
    "RANK/DENSE_RANK",
    "LAG/LEAD",
    "Set Operation",
    "Window Aggregate",
    "CTE + Window",
    "CTE + Subquery",
    "JOIN + Subquery",
    "Aggregate + Window",
)

ALL_FEATURES = tuple(FEATURE_PATTERNS) + DERIVED_FEATURES

COMPILED_FEATURES = {
    name: re.compile(pattern, re.IGNORECASE)
    for name, pattern in FEATURE_PATTERNS.items()
}

SQL_NON_CODE = re.compile(
    r"--[^\n]*|/\*[\s\S]*?\*/|'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"|"
    r"`(?:``|[^`])*`|\[(?:\]\]|[^\]])*\]",
    re.MULTILINE,
)

AGGREGATE_CALL = re.compile(
    r"\b(?:COUNT|SUM|AVG|MIN|MAX)\s*\(",
    re.IGNORECASE,
)

OVER_CLAUSE = re.compile(r"\s*OVER\s*\(", re.IGNORECASE)
FILTER_CLAUSE = re.compile(r"\s*FILTER\s*\(", re.IGNORECASE)

README_FEATURES = (
    "JOIN",
    "Subquery",
    "CTE",
    "Recursive CTE",
    "Window Function",
    "ROW_NUMBER",
    "RANK/DENSE_RANK",
    "LAG/LEAD",
)

DIFFICULTY_ORDER = (
    "Simple",
    "Moderate",
    "Complex",
    "Highly Complex",
    "Unspecified",
)


def _load_json(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list: {path}")
    return data


def _text(record: dict, key: str) -> str:
    value = record.get(key, "")
    return value.strip() if isinstance(value, str) else ""


def _sql(record: dict) -> str:
    return _text(record, "sql") or _text(record, "SQL")


def _candidate_sqls(record: dict) -> dict[str, str]:
    candidates = record.get("sql_candidates")
    if not isinstance(candidates, dict):
        return {}
    return {
        str(source): value.strip() if isinstance(value, str) else ""
        for source, value in candidates.items()
    }


def _record_key(record: dict) -> tuple[str, str]:
    return (_text(record, "question"), _text(record, "external_knowledge"))


def _normalize_sql(sql: str) -> str:
    """Normalize SQL formatting while preserving literal/identifier contents."""
    protected: list[str] = []

    def mask(match: re.Match) -> str:
        token = match.group(0)
        if token.startswith("--") or token.startswith("/*"):
            return " "
        index = len(protected)
        protected.append(token)
        return f"\ue000{index}\ue001"

    masked = SQL_NON_CODE.sub(mask, sql)
    normalized = masked.strip().rstrip(";").strip().casefold()
    normalized = re.sub(r"\s+", " ", normalized)
    for index, token in enumerate(protected):
        normalized = normalized.replace(f"\ue000{index}\ue001", token)
    return normalized


def _sql_code(sql: str) -> str:
    """Mask comments, string literals, and quoted identifiers before regexes."""
    return SQL_NON_CODE.sub(" ", sql)


def _matching_parenthesis(code: str, opening: int) -> int | None:
    depth = 0
    for index in range(opening, len(code)):
        if code[index] == "(":
            depth += 1
        elif code[index] == ")":
            depth -= 1
            if depth == 0:
                return index
    return None


def _has_window_aggregate(code: str) -> bool:
    """Detect aggregate calls followed by optional FILTER and inline OVER."""
    for match in AGGREGATE_CALL.finditer(code):
        opening = code.rfind("(", match.start(), match.end())
        closing = _matching_parenthesis(code, opening)
        if closing is None:
            continue
        cursor = closing + 1
        filter_match = FILTER_CLAUSE.match(code, cursor)
        if filter_match:
            filter_opening = code.rfind("(", filter_match.start(), filter_match.end())
            filter_closing = _matching_parenthesis(code, filter_opening)
            if filter_closing is None:
                continue
            cursor = filter_closing + 1
        if OVER_CLAUSE.match(code, cursor):
            return True
    return False


def _detect_features(sql: str) -> tuple[set[str], int, int]:
    code = _sql_code(sql)
    features = {
        name for name, pattern in COMPILED_FEATURES.items() if pattern.search(code)
    }
    join_count = len(re.findall(r"\bJOIN\b", code, re.IGNORECASE))
    subquery_count = len(re.findall(r"\(\s*SELECT\b", code, re.IGNORECASE))
    select_block_count = len(re.findall(r"\bSELECT\b", code, re.IGNORECASE))

    if join_count >= 2:
        features.add("Multiple JOINs")
    if join_count >= 3:
        features.add("JOIN >= 3")
    if subquery_count >= 2:
        features.add("Multiple Subqueries")
    if {"RANK", "DENSE_RANK"} & features:
        features.add("RANK/DENSE_RANK")
    if {"LAG", "LEAD"} & features:
        features.add("LAG/LEAD")
    if {"UNION", "INTERSECT", "EXCEPT"} & features:
        features.add("Set Operation")
    if _has_window_aggregate(code):
        features.add("Window Aggregate")
    if {"CTE", "Window Function"} <= features:
        features.add("CTE + Window")
    if {"CTE", "Subquery"} <= features:
        features.add("CTE + Subquery")
    if {"JOIN", "Subquery"} <= features:
        features.add("JOIN + Subquery")
    if {"Aggregation", "Window Function"} <= features:
        features.add("Aggregate + Window")
    return features, join_count, select_block_count


def _normalize_label(value: str, default: str = "Unspecified") -> str:
    value = re.sub(r"[_-]+", " ", value.strip())
    value = re.sub(r"\s+", " ", value)
    if not value:
        return default
    lookup = {label.casefold(): label for label in DIFFICULTY_ORDER}
    return lookup.get(value.casefold(), value.title())


def _find_verified_file(question_dir: Path) -> Path | None:
    for filename in VERIFIED_FILENAMES:
        candidate = question_dir / filename
        if candidate.exists():
            return candidate
    return None


def _find_raw_review_file(question_dir: Path) -> Path | None:
    for filename in RAW_REVIEW_FILENAMES:
        candidate = question_dir / filename
        if candidate.exists():
            return candidate
    return None


def _align_records(
    originals: list[dict], verified: list[dict], db_id: str
) -> tuple[list[tuple[dict, dict]], str]:
    """Validate positional alignment; question text is not a unique key."""
    if len(originals) != len(verified):
        raise ValueError(
            f"{db_id}: array length mismatch "
            f"(original={len(originals)}, verified={len(verified)})"
        )
    mismatches = [
        index
        for index, (left, right) in enumerate(zip(originals, verified))
        if _record_key(left) != _record_key(right)
    ]
    if mismatches:
        raise ValueError(
            f"{db_id}: positional question mismatch at indices {mismatches[:5]}"
        )
    return list(zip(originals, verified)), "position"


def _new_bucket() -> dict:
    return {
        "total": 0,
        "accepted": 0,
        "rejected": 0,
        "consensus": Counter(),
        "candidate_slot_count": Counter(),
        "candidate_nonempty_count": Counter(),
    }


def _update_bucket(
    bucket: dict,
    accepted: bool,
    consensus: int,
    candidate_nonempty_count: int | None = None,
    candidate_slot_count: int | None = None,
) -> None:
    bucket["total"] += 1
    bucket["accepted" if accepted else "rejected"] += 1
    bucket["consensus"][str(consensus)] += 1
    if candidate_slot_count is not None:
        bucket["candidate_slot_count"][str(candidate_slot_count)] += 1
    if candidate_nonempty_count is not None:
        bucket["candidate_nonempty_count"][str(candidate_nonempty_count)] += 1


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _length_stats(values: list[int]) -> dict:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "mean": round(statistics.fmean(values), 3),
        "p25": round(_percentile(values, 0.25), 3),
        "median": round(_percentile(values, 0.50), 3),
        "p75": round(_percentile(values, 0.75), 3),
        "p90": round(_percentile(values, 0.90), 3),
        "p95": round(_percentile(values, 0.95), 3),
        "max": max(values),
    }


def _finalize_bucket(
    bucket: dict,
    overall_total: int,
    overall_accepted: int,
    overall_acceptance_rate: float,
    database_rates: list[float] | None = None,
) -> dict:
    total = bucket["total"]
    accepted = bucket["accepted"]
    rejected = bucket["rejected"]
    acceptance_rate = accepted / total if total else 0.0
    candidate_prevalence = total / overall_total if overall_total else 0.0
    accepted_prevalence = accepted / overall_accepted if overall_accepted else 0.0
    relative_retention = (
        acceptance_rate / overall_acceptance_rate
        if overall_acceptance_rate
        else 0.0
    )
    consensus_counts = {
        str(key): bucket["consensus"].get(str(key), 0)
        for key in sorted(int(k) for k in bucket["consensus"])
    }
    candidate_slot_counts = {
        str(key): bucket["candidate_slot_count"].get(str(key), 0)
        for key in sorted(int(k) for k in bucket["candidate_slot_count"])
    }
    candidate_nonempty_counts = {
        str(key): bucket["candidate_nonempty_count"].get(str(key), 0)
        for key in sorted(int(k) for k in bucket["candidate_nonempty_count"])
    }
    macro = {"databases_with_group": 0}
    if database_rates:
        macro = {
            "databases_with_group": len(database_rates),
            "mean": round(statistics.fmean(database_rates), 8),
            "median": round(_percentile(database_rates, 0.50), 8),
            "p25": round(_percentile(database_rates, 0.25), 8),
            "p75": round(_percentile(database_rates, 0.75), 8),
        }
    return {
        "total": total,
        "accepted": accepted,
        "rejected": rejected,
        "acceptance_rate": round(acceptance_rate, 8),
        "candidate_prevalence": round(candidate_prevalence, 8),
        "accepted_prevalence": round(accepted_prevalence, 8),
        "relative_retention": round(relative_retention, 8),
        "consensus_count_distribution": consensus_counts,
        "candidate_slot_count_distribution": candidate_slot_counts,
        "candidate_nonempty_count_distribution": candidate_nonempty_counts,
        "per_database_acceptance_rate": macro,
    }


def _sort_labels(labels: Iterable[str], preferred: tuple[str, ...]) -> list[str]:
    labels = set(labels)
    ordered = [label for label in preferred if label in labels]
    ordered.extend(sorted(labels - set(ordered)))
    return ordered


def analyze(
    data_root: Path,
    db_ids: list[str] | None = None,
    expected_acceptance_consensus: int | None = None,
    expected_candidate_slots: int | None = None,
    source_label: str = "Dataset",
    collect_records: bool = False,
) -> tuple[dict, list[dict]]:
    if db_ids is None:
        db_ids = sorted(
            path.name
            for path in data_root.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        )

    feature_buckets = {name: _new_bucket() for name in ALL_FEATURES}
    difficulty_buckets: dict[str, dict] = defaultdict(_new_bucket)
    style_buckets: dict[str, dict] = defaultdict(_new_bucket)
    join_count_buckets: dict[str, dict] = defaultdict(_new_bucket)
    select_block_buckets: dict[str, dict] = defaultdict(_new_bucket)
    slot_count_buckets: dict[str, dict] = defaultdict(_new_bucket)
    nonempty_count_buckets: dict[str, dict] = defaultdict(_new_bucket)
    overall = _new_bucket()
    selected_sources = Counter()
    verified_filenames = Counter()
    raw_review_filenames = Counter()
    candidate_slots_seen = Counter()
    candidate_slots_available = Counter()
    pairing_modes = Counter()
    upstream_lengths = {"all": [], "accepted": [], "rejected": []}
    database_rows = []
    record_rows: list[dict] = []
    feature_database_rates: dict[str, list[float]] = defaultdict(list)
    difficulty_database_rates: dict[str, list[float]] = defaultdict(list)
    style_database_rates: dict[str, list[float]] = defaultdict(list)
    join_count_database_rates: dict[str, list[float]] = defaultdict(list)
    select_block_database_rates: dict[str, list[float]] = defaultdict(list)
    slot_count_database_rates: dict[str, list[float]] = defaultdict(list)
    nonempty_count_database_rates: dict[str, list[float]] = defaultdict(list)
    selected_sql_changed_exact = 0
    selected_sql_changed_format_normalized = 0
    selected_source_differs_from_synsql = 0
    selected_sql_compared = 0
    selected_sql_missing = 0
    missing_upstream_sql = 0
    status_consensus_mismatches = 0
    negative_consensus_counts = 0
    raw_review_databases = 0
    raw_review_records = 0
    missing_raw_review_database_ids: list[str] = []
    raw_original_sql_mismatches = 0
    raw_synsql_candidate_mismatches = 0
    original_db_id_mismatches = 0
    consensus_exceeds_candidate_nonempty_count = 0
    candidate_slot_count_mismatches = 0
    selected_source_missing_from_candidates = 0
    selected_sql_candidate_mismatches = 0
    skipped_database_ids: list[str] = []
    missing_original_database_ids: list[str] = []
    missing_verified_database_ids: list[str] = []
    question_files_discovered = 0

    for db_id in db_ids:
        question_dir = data_root / db_id / "questions"
        original_path = question_dir / "questions.json"
        verified_path = _find_verified_file(question_dir)
        if not original_path.exists():
            skipped_database_ids.append(db_id)
            missing_original_database_ids.append(db_id)
            continue
        question_files_discovered += 1
        if verified_path is None:
            skipped_database_ids.append(db_id)
            missing_verified_database_ids.append(db_id)
            continue

        originals = _load_json(original_path)
        verified = _load_json(verified_path)
        pairs, pairing_mode = _align_records(originals, verified, db_id)
        pairing_modes[pairing_mode] += 1
        verified_filenames[verified_path.name] += 1

        raw_by_index: list[dict] | None = None
        raw_path = _find_raw_review_file(question_dir)
        if raw_path is not None:
            raw_reviews = _load_json(raw_path)
            raw_pairs, _ = _align_records(originals, raw_reviews, db_id)
            raw_by_index = [raw for _, raw in raw_pairs]
            raw_review_databases += 1
            raw_review_records += len(raw_by_index)
            raw_review_filenames[raw_path.name] += 1
        else:
            missing_raw_review_database_ids.append(db_id)

        db_total = 0
        db_accepted = 0
        db_feature_buckets: dict[str, dict] = defaultdict(_new_bucket)
        db_difficulty_buckets: dict[str, dict] = defaultdict(_new_bucket)
        db_style_buckets: dict[str, dict] = defaultdict(_new_bucket)
        db_join_count_buckets: dict[str, dict] = defaultdict(_new_bucket)
        db_select_block_buckets: dict[str, dict] = defaultdict(_new_bucket)
        db_slot_count_buckets: dict[str, dict] = defaultdict(_new_bucket)
        db_nonempty_count_buckets: dict[str, dict] = defaultdict(_new_bucket)

        for index, (original, review) in enumerate(pairs):
            if _text(original, "db_id") != db_id:
                original_db_id_mismatches += 1
            status = _text(review, "review_status").casefold()
            if status not in {"accepted", "rejected"}:
                raise ValueError(
                    f"{db_id}: unexpected review_status={review.get('review_status')!r}"
                )
            accepted = status == "accepted"
            try:
                consensus = int(review.get("consensus_count"))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{db_id}: invalid consensus_count="
                    f"{review.get('consensus_count')!r}"
                ) from exc
            if consensus < 0:
                negative_consensus_counts += 1

            if expected_acceptance_consensus is not None:
                expected_status = consensus >= expected_acceptance_consensus
                if accepted != expected_status:
                    status_consensus_mismatches += 1

            upstream_sql = _sql(original)
            if not upstream_sql:
                missing_upstream_sql += 1

            raw_record = raw_by_index[index] if raw_by_index is not None else None
            candidates = _candidate_sqls(raw_record) if raw_record is not None else {}
            candidate_slot_count = len(candidates) if raw_record is not None else None
            candidate_nonempty_count = (
                sum(bool(value) for value in candidates.values())
                if raw_record is not None
                else None
            )
            if (
                candidate_nonempty_count is not None
                and consensus > candidate_nonempty_count
            ):
                consensus_exceeds_candidate_nonempty_count += 1
            if (
                expected_candidate_slots is not None
                and candidate_slot_count is not None
                and candidate_slot_count != expected_candidate_slots
            ):
                candidate_slot_count_mismatches += 1
            for source, candidate_sql in candidates.items():
                candidate_slots_seen[source] += 1
                if candidate_sql:
                    candidate_slots_available[source] += 1

            if raw_record is not None:
                if _sql(raw_record) != upstream_sql:
                    raw_original_sql_mismatches += 1
                synsql = candidates.get("synsql", "")
                if synsql != upstream_sql:
                    raw_synsql_candidate_mismatches += 1

            sql_length = len(upstream_sql)
            upstream_lengths["all"].append(sql_length)
            upstream_lengths["accepted" if accepted else "rejected"].append(
                sql_length
            )

            _update_bucket(
                overall,
                accepted,
                consensus,
                candidate_nonempty_count,
                candidate_slot_count,
            )
            db_total += 1
            db_accepted += int(accepted)

            difficulty = _normalize_label(_text(original, "sql_complexity"))
            style = _normalize_label(
                _text(original, "question_style"), default="Unspecified"
            )
            features, join_count, select_block_count = _detect_features(upstream_sql)
            join_count_label = str(join_count) if join_count < 4 else "4+"
            select_block_label = (
                str(select_block_count) if select_block_count < 4 else "4+"
            )

            _update_bucket(
                difficulty_buckets[difficulty],
                accepted,
                consensus,
                candidate_nonempty_count,
                candidate_slot_count,
            )
            _update_bucket(
                db_difficulty_buckets[difficulty],
                accepted,
                consensus,
                candidate_nonempty_count,
                candidate_slot_count,
            )
            _update_bucket(
                style_buckets[style],
                accepted,
                consensus,
                candidate_nonempty_count,
                candidate_slot_count,
            )
            _update_bucket(
                db_style_buckets[style],
                accepted,
                consensus,
                candidate_nonempty_count,
                candidate_slot_count,
            )
            _update_bucket(
                join_count_buckets[join_count_label],
                accepted,
                consensus,
                candidate_nonempty_count,
                candidate_slot_count,
            )
            _update_bucket(
                db_join_count_buckets[join_count_label],
                accepted,
                consensus,
                candidate_nonempty_count,
                candidate_slot_count,
            )
            _update_bucket(
                select_block_buckets[select_block_label],
                accepted,
                consensus,
                candidate_nonempty_count,
                candidate_slot_count,
            )
            _update_bucket(
                db_select_block_buckets[select_block_label],
                accepted,
                consensus,
                candidate_nonempty_count,
                candidate_slot_count,
            )
            if candidate_slot_count is not None:
                slot_count_label = str(candidate_slot_count)
                _update_bucket(
                    slot_count_buckets[slot_count_label],
                    accepted,
                    consensus,
                    candidate_nonempty_count,
                    candidate_slot_count,
                )
                _update_bucket(
                    db_slot_count_buckets[slot_count_label],
                    accepted,
                    consensus,
                    candidate_nonempty_count,
                    candidate_slot_count,
                )
            if candidate_nonempty_count is not None:
                nonempty_count_label = str(candidate_nonempty_count)
                _update_bucket(
                    nonempty_count_buckets[nonempty_count_label],
                    accepted,
                    consensus,
                    candidate_nonempty_count,
                    candidate_slot_count,
                )
                _update_bucket(
                    db_nonempty_count_buckets[nonempty_count_label],
                    accepted,
                    consensus,
                    candidate_nonempty_count,
                    candidate_slot_count,
                )

            for name in features:
                _update_bucket(
                    feature_buckets[name],
                    accepted,
                    consensus,
                    candidate_nonempty_count,
                    candidate_slot_count,
                )
                _update_bucket(
                    db_feature_buckets[name],
                    accepted,
                    consensus,
                    candidate_nonempty_count,
                    candidate_slot_count,
                )

            exact_changed: bool | None = None
            format_normalized_changed: bool | None = None
            selected_source = ""
            if accepted:
                selected_source = _text(review, "sql_source")
                selected_sources[selected_source or "unspecified"] += 1
                selected_source_differs_from_synsql += int(
                    selected_source.casefold() != "synsql"
                )
                selected_sql = _sql(review)
                if selected_sql:
                    selected_sql_compared += 1
                    exact_changed = selected_sql.strip() != upstream_sql.strip()
                    format_normalized_changed = (
                        _normalize_sql(selected_sql) != _normalize_sql(upstream_sql)
                    )
                    selected_sql_changed_exact += int(exact_changed)
                    selected_sql_changed_format_normalized += int(
                        format_normalized_changed
                    )
                else:
                    selected_sql_missing += 1

                if raw_record is not None:
                    selected_candidate = candidates.get(selected_source)
                    if not selected_source or not selected_candidate:
                        selected_source_missing_from_candidates += 1
                    elif selected_sql.strip() != selected_candidate.strip():
                        selected_sql_candidate_mismatches += 1

            if collect_records:
                record_rows.append(
                    {
                        "sample_id": f"{db_id}#{index}",
                        "db_id": db_id,
                        "index": index,
                        "difficulty": difficulty,
                        "question_style": style,
                        "review_status": status,
                        "consensus_count": consensus,
                        "candidate_slot_count": (
                            candidate_slot_count
                            if candidate_slot_count is not None
                            else ""
                        ),
                        "candidate_nonempty_count": (
                            candidate_nonempty_count
                            if candidate_nonempty_count is not None
                            else ""
                        ),
                        "selected_source": selected_source,
                        "selected_sql_changed_exact": (
                            int(exact_changed) if exact_changed is not None else ""
                        ),
                        "selected_sql_changed_format_normalized": (
                            int(format_normalized_changed)
                            if format_normalized_changed is not None
                            else ""
                        ),
                        "upstream_sql_length": sql_length,
                        "join_count": join_count,
                        "select_block_count": select_block_count,
                        "features": ";".join(sorted(features)),
                    }
                )

        for name, bucket in db_feature_buckets.items():
            feature_database_rates[name].append(bucket["accepted"] / bucket["total"])
        for name, bucket in db_difficulty_buckets.items():
            difficulty_database_rates[name].append(
                bucket["accepted"] / bucket["total"]
            )
        for name, bucket in db_style_buckets.items():
            style_database_rates[name].append(bucket["accepted"] / bucket["total"])
        for name, bucket in db_join_count_buckets.items():
            join_count_database_rates[name].append(
                bucket["accepted"] / bucket["total"]
            )
        for name, bucket in db_select_block_buckets.items():
            select_block_database_rates[name].append(
                bucket["accepted"] / bucket["total"]
            )
        for name, bucket in db_slot_count_buckets.items():
            slot_count_database_rates[name].append(
                bucket["accepted"] / bucket["total"]
            )
        for name, bucket in db_nonempty_count_buckets.items():
            nonempty_count_database_rates[name].append(
                bucket["accepted"] / bucket["total"]
            )

        database_rows.append(
            {
                "db_id": db_id,
                "total": db_total,
                "accepted": db_accepted,
                "rejected": db_total - db_accepted,
                "acceptance_rate": round(db_accepted / db_total, 8)
                if db_total
                else 0.0,
            }
        )

    overall_total = overall["total"]
    overall_accepted = overall["accepted"]
    overall_acceptance_rate = (
        overall_accepted / overall_total if overall_total else 0.0
    )
    if overall_total == 0:
        raise ValueError(f"No aligned peer-review records found under {data_root}")

    acceptance_rates = [row["acceptance_rate"] for row in database_rows]
    result = {
        "schema_version": 2,
        "config": {
            "source_label": source_label,
            "sql_feature_source": "questions/questions.json (upstream SQL)",
            "grouping_source": "aligned verified review_status",
            "verified_filenames": dict(sorted(verified_filenames.items())),
            "raw_review_filenames": dict(sorted(raw_review_filenames.items())),
            "pairing_modes": dict(sorted(pairing_modes.items())),
            "expected_acceptance_consensus": expected_acceptance_consensus,
            "expected_candidate_slots": expected_candidate_slots,
            "feature_detection": (
                "case-insensitive regular expressions after masking comments, "
                "string literals, and quoted identifiers"
            ),
            "feature_patterns": FEATURE_PATTERNS,
            "derived_features": list(DERIVED_FEATURES),
            "window_aggregate_detection": (
                "balanced aggregate-function call, optional FILTER clause, and "
                "inline OVER clause"
            ),
            "sql_non_code_mask_pattern": SQL_NON_CODE.pattern,
            "overlapping_features": True,
        },
        "summary": {
            "databases": len(database_rows),
            "database_directories_scanned": len(db_ids),
            "database_directories_skipped": len(skipped_database_ids),
            "question_files_discovered": question_files_discovered,
            "total": overall_total,
            "accepted": overall_accepted,
            "rejected": overall["rejected"],
            "acceptance_rate": round(overall_acceptance_rate, 8),
            "consensus_count_distribution": dict(
                sorted(overall["consensus"].items(), key=lambda item: int(item[0]))
            ),
            "candidate_slot_count_distribution": dict(
                sorted(
                    overall["candidate_slot_count"].items(),
                    key=lambda item: int(item[0]),
                )
            ),
            "candidate_nonempty_count_distribution": dict(
                sorted(
                    overall["candidate_nonempty_count"].items(),
                    key=lambda item: int(item[0]),
                )
            ),
            "candidate_slots": {
                source: {
                    "records_with_slot": candidate_slots_seen[source],
                    "records_with_nonempty_sql": candidate_slots_available[source],
                    "slot_presence_rate": round(
                        candidate_slots_seen[source] / raw_review_records,
                        8,
                    )
                    if raw_review_records
                    else 0.0,
                    "nonempty_rate": round(
                        candidate_slots_available[source] / raw_review_records,
                        8,
                    )
                    if raw_review_records
                    else 0.0,
                }
                for source in sorted(
                    set(candidate_slots_seen) | set(candidate_slots_available)
                )
            },
            "selected_sql_compared": selected_sql_compared,
            "selected_sql_missing": selected_sql_missing,
            "selected_sql_changed_exact": selected_sql_changed_exact,
            "selected_sql_changed_exact_rate": round(
                selected_sql_changed_exact / selected_sql_compared, 8
            )
            if selected_sql_compared
            else 0.0,
            "selected_sql_changed_format_normalized": (
                selected_sql_changed_format_normalized
            ),
            "selected_sql_changed_format_normalized_rate": round(
                selected_sql_changed_format_normalized / selected_sql_compared, 8
            )
            if selected_sql_compared
            else 0.0,
            "selected_source_differs_from_synsql": selected_source_differs_from_synsql,
            "selected_source_differs_from_synsql_rate": round(
                selected_source_differs_from_synsql / overall_accepted, 8
            )
            if overall_accepted
            else 0.0,
            "selected_sql_sources": dict(selected_sources.most_common()),
        },
        "sql_length": {
            name: _length_stats(values)
            for name, values in upstream_lengths.items()
        },
        "database_acceptance_rate": {
            "mean": round(statistics.fmean(acceptance_rates), 8),
            "p25": round(_percentile(acceptance_rates, 0.25), 8),
            "median": round(_percentile(acceptance_rates, 0.50), 8),
            "p75": round(_percentile(acceptance_rates, 0.75), 8),
            "min": round(min(acceptance_rates), 8),
            "max": round(max(acceptance_rates), 8),
        },
        "features": {
            name: _finalize_bucket(
                feature_buckets[name],
                overall_total,
                overall_accepted,
                overall_acceptance_rate,
                feature_database_rates[name],
            )
            for name in ALL_FEATURES
        },
        "difficulty": {
            name: _finalize_bucket(
                difficulty_buckets[name],
                overall_total,
                overall_accepted,
                overall_acceptance_rate,
                difficulty_database_rates[name],
            )
            for name in _sort_labels(difficulty_buckets, DIFFICULTY_ORDER)
        },
        "question_style": {
            name: _finalize_bucket(
                style_buckets[name],
                overall_total,
                overall_accepted,
                overall_acceptance_rate,
                style_database_rates[name],
            )
            for name in sorted(style_buckets)
        },
        "join_count": {
            name: _finalize_bucket(
                join_count_buckets[name],
                overall_total,
                overall_accepted,
                overall_acceptance_rate,
                join_count_database_rates[name],
            )
            for name in _sort_labels(join_count_buckets, ("0", "1", "2", "3", "4+"))
        },
        "select_block_count": {
            name: _finalize_bucket(
                select_block_buckets[name],
                overall_total,
                overall_accepted,
                overall_acceptance_rate,
                select_block_database_rates[name],
            )
            for name in _sort_labels(
                select_block_buckets, ("0", "1", "2", "3", "4+")
            )
        },
        "candidate_slot_count": {
            name: _finalize_bucket(
                slot_count_buckets[name],
                overall_total,
                overall_accepted,
                overall_acceptance_rate,
                slot_count_database_rates[name],
            )
            for name in sorted(slot_count_buckets, key=int)
        },
        "candidate_nonempty_count": {
            name: _finalize_bucket(
                nonempty_count_buckets[name],
                overall_total,
                overall_accepted,
                overall_acceptance_rate,
                nonempty_count_database_rates[name],
            )
            for name in sorted(nonempty_count_buckets, key=int)
        },
        "database": sorted(database_rows, key=lambda row: row["db_id"]),
        "sanity_checks": {
            "missing_upstream_sql": missing_upstream_sql,
            "status_consensus_mismatches": status_consensus_mismatches,
            "negative_consensus_counts": negative_consensus_counts,
            "raw_review_databases": raw_review_databases,
            "raw_review_records": raw_review_records,
            "records_without_raw_review": overall_total - raw_review_records,
            "missing_raw_review_database_ids": missing_raw_review_database_ids,
            "raw_original_sql_mismatches": raw_original_sql_mismatches,
            "raw_synsql_candidate_mismatches": raw_synsql_candidate_mismatches,
            "original_db_id_mismatches": original_db_id_mismatches,
            "consensus_exceeds_candidate_nonempty_count": (
                consensus_exceeds_candidate_nonempty_count
            ),
            "candidate_slot_count_mismatches": candidate_slot_count_mismatches,
            "selected_source_missing_from_candidates": (
                selected_source_missing_from_candidates
            ),
            "selected_sql_candidate_mismatches": selected_sql_candidate_mismatches,
            "skipped_database_ids": skipped_database_ids,
            "missing_original_database_ids": missing_original_database_ids,
            "missing_verified_database_ids": missing_verified_database_ids,
        },
    }
    return result, record_rows


def _print_bucket_table(title: str, buckets: dict, names: Iterable[str]) -> None:
    print(f"\n{title}")
    print(
        f"  {'Group':<24} {'Total':>9} {'Accepted':>9} {'Rejected':>9} "
        f"{'Accept':>9} {'Rel.Ret.':>9}"
    )
    print(f"  {'-' * 24} {'-' * 9} {'-' * 9} {'-' * 9} {'-' * 9} {'-' * 9}")
    for name in names:
        bucket = buckets.get(name)
        if not bucket:
            continue
        print(
            f"  {name:<24} {bucket['total']:>9,} {bucket['accepted']:>9,} "
            f"{bucket['rejected']:>9,} {bucket['acceptance_rate']:>8.2%} "
            f"{bucket['relative_retention']:>9.3f}"
        )


def print_report(result: dict, detailed: bool = False) -> None:
    summary = result["summary"]
    print("=" * 84)
    print(f"Peer-review Retention Analysis: {result['config']['source_label']}")
    print("=" * 84)
    print(f"Databases:       {summary['databases']:,}")
    print(f"Candidates:      {summary['total']:,}")
    print(f"Accepted:        {summary['accepted']:,}")
    print(f"Rejected:        {summary['rejected']:,}")
    print(f"Acceptance rate: {summary['acceptance_rate']:.2%}")
    print(
        "Consensus count: "
        + ", ".join(
            f"{key}={value:,}"
            for key, value in summary["consensus_count_distribution"].items()
        )
    )
    print(
        f"Selected SQL changed (exact text): "
        f"{summary['selected_sql_changed_exact']:,}/"
        f"{summary['selected_sql_compared']:,} "
        f"({summary['selected_sql_changed_exact_rate']:.2%})"
    )
    print(
        f"Selected SQL changed (format-normalized, literals preserved): "
        f"{summary['selected_sql_changed_format_normalized']:,}/"
        f"{summary['selected_sql_compared']:,} "
        f"({summary['selected_sql_changed_format_normalized_rate']:.2%})"
    )

    _print_bucket_table(
        "Difficulty (from upstream metadata)",
        result["difficulty"],
        result["difficulty"].keys(),
    )
    feature_names = result["features"].keys() if detailed else README_FEATURES
    _print_bucket_table(
        "Upstream SQL features", result["features"], feature_names
    )

    if detailed:
        _print_bucket_table(
            "Recorded candidate slots",
            result["candidate_slot_count"],
            result["candidate_slot_count"].keys(),
        )
        _print_bucket_table(
            "Non-empty candidate SQLs",
            result["candidate_nonempty_count"],
            result["candidate_nonempty_count"].keys(),
        )
        _print_bucket_table(
            "JOIN count", result["join_count"], result["join_count"].keys()
        )
        _print_bucket_table(
            "SELECT-block count",
            result["select_block_count"],
            result["select_block_count"].keys(),
        )
        _print_bucket_table(
            "Question styles",
            result["question_style"],
            result["question_style"].keys(),
        )
        print("\nSelected SQL source (accepted records)")
        for source, count in result["summary"]["selected_sql_sources"].items():
            print(f"  {source:<40} {count:>9,}")

    checks = result["sanity_checks"]
    print("\nSanity checks")
    print(
        "  Directories scanned/analyzed: "
        f"{summary['database_directories_scanned']:,}/{summary['databases']:,}"
    )
    print(
        "  Question files/skipped dirs:  "
        f"{summary['question_files_discovered']:,}/"
        f"{summary['database_directories_skipped']:,}"
    )
    print(
        "  Raw review records:           "
        f"{checks['raw_review_records']:,}"
    )
    print(f"  Missing upstream SQL:          {checks['missing_upstream_sql']:,}")
    print(
        "  Status/consensus mismatches:  "
        f"{checks['status_consensus_mismatches']:,}"
    )
    print(
        "  Negative consensus counts:    "
        f"{checks['negative_consensus_counts']:,}"
    )
    print(
        "  Consensus exceeds non-empty:  "
        f"{checks['consensus_exceeds_candidate_nonempty_count']:,}"
    )
    print(
        "  Candidate-slot mismatches:    "
        f"{checks['candidate_slot_count_mismatches']:,}"
    )
    print(
        "  Upstream/raw SQL mismatches:  "
        f"{checks['raw_original_sql_mismatches']:,}"
    )
    print(
        "  Upstream/SynSQL mismatches:   "
        f"{checks['raw_synsql_candidate_mismatches']:,}"
    )
    print(
        "  Selected source missing:      "
        f"{checks['selected_source_missing_from_candidates']:,}"
    )
    print(
        "  Selected/candidate mismatch:  "
        f"{checks['selected_sql_candidate_mismatches']:,}"
    )
    print(
        "  DB-ID mismatches:              "
        f"{checks['original_db_id_mismatches']:,}"
    )
    print(
        "  Records without raw review:   "
        f"{checks['records_without_raw_review']:,}"
    )
    print("=" * 84)


def _write_csv(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    consensus_keys = sorted(
        result["summary"]["consensus_count_distribution"], key=int
    )
    slot_count_keys = sorted(
        result["summary"]["candidate_slot_count_distribution"], key=int
    )
    nonempty_count_keys = sorted(
        result["summary"]["candidate_nonempty_count_distribution"], key=int
    )
    fields = [
        "category",
        "name",
        "total",
        "accepted",
        "rejected",
        "acceptance_rate",
        "candidate_prevalence",
        "accepted_prevalence",
        "relative_retention",
        "macro_database_count",
        "macro_acceptance_mean",
        "macro_acceptance_median",
        "macro_acceptance_p25",
        "macro_acceptance_p75",
    ] + [f"consensus_{key}" for key in consensus_keys] + [
        f"candidate_slot_count_{key}" for key in slot_count_keys
    ] + [
        f"candidate_nonempty_count_{key}" for key in nonempty_count_keys
    ]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for category in (
            "features",
            "difficulty",
            "question_style",
            "join_count",
            "select_block_count",
            "candidate_slot_count",
            "candidate_nonempty_count",
        ):
            for name, bucket in result[category].items():
                row = {
                    "category": category,
                    "name": name,
                    **{
                        key: bucket[key]
                        for key in (
                            "total",
                            "accepted",
                            "rejected",
                            "acceptance_rate",
                            "candidate_prevalence",
                            "accepted_prevalence",
                            "relative_retention",
                        )
                    },
                }
                macro = bucket["per_database_acceptance_rate"]
                row.update(
                    {
                        "macro_database_count": macro.get(
                            "databases_with_group", 0
                        ),
                        "macro_acceptance_mean": macro.get("mean", ""),
                        "macro_acceptance_median": macro.get("median", ""),
                        "macro_acceptance_p25": macro.get("p25", ""),
                        "macro_acceptance_p75": macro.get("p75", ""),
                    }
                )
                for key in consensus_keys:
                    row[f"consensus_{key}"] = bucket[
                        "consensus_count_distribution"
                    ].get(key, 0)
                for key in slot_count_keys:
                    row[f"candidate_slot_count_{key}"] = bucket[
                        "candidate_slot_count_distribution"
                    ].get(key, 0)
                for key in nonempty_count_keys:
                    row[f"candidate_nonempty_count_{key}"] = bucket[
                        "candidate_nonempty_count_distribution"
                    ].get(key, 0)
                writer.writerow(row)


def _write_records_csv(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_id",
        "db_id",
        "index",
        "difficulty",
        "question_style",
        "review_status",
        "consensus_count",
        "candidate_slot_count",
        "candidate_nonempty_count",
        "selected_source",
        "selected_sql_changed_exact",
        "selected_sql_changed_format_normalized",
        "upstream_sql_length",
        "join_count",
        "select_block_count",
        "features",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def _critical_sanity_failures(result: dict) -> dict[str, int]:
    checks = result["sanity_checks"]
    failures = {
        key: int(checks[key])
        for key in (
            "missing_upstream_sql",
            "status_consensus_mismatches",
            "negative_consensus_counts",
            "records_without_raw_review",
            "raw_original_sql_mismatches",
            "raw_synsql_candidate_mismatches",
            "original_db_id_mismatches",
            "consensus_exceeds_candidate_nonempty_count",
            "candidate_slot_count_mismatches",
            "selected_source_missing_from_candidates",
            "selected_sql_candidate_mismatches",
        )
        if checks[key]
    }
    if result["summary"]["selected_sql_missing"]:
        failures["selected_sql_missing"] = result["summary"]["selected_sql_missing"]
    missing_originals = len(checks["missing_original_database_ids"])
    if missing_originals:
        failures["missing_original_database_ids"] = missing_originals
    missing_raw_databases = len(checks["missing_raw_review_database_ids"])
    if missing_raw_databases:
        failures["missing_raw_review_database_ids"] = missing_raw_databases
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze peer-review retention using upstream SQL features"
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        required=True,
        help="Root containing {db_id}/questions/questions.json and verified output",
    )
    parser.add_argument(
        "--db_ids", nargs="*", default=None, help="Optional database IDs"
    )
    parser.add_argument(
        "--db_list", type=Path, default=None, help="Optional JSON list of database IDs"
    )
    parser.add_argument(
        "--source_label", default="Dataset", help="Portable label stored in output"
    )
    parser.add_argument(
        "--expected_acceptance_consensus",
        type=int,
        default=None,
        help="If set, validate accepted iff consensus_count is at least this value",
    )
    parser.add_argument(
        "--expected_candidate_slots",
        type=int,
        default=None,
        help="If set, validate the recorded candidate-slot count for every record",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return a non-zero exit code when a critical sanity check fails",
    )
    parser.add_argument(
        "--detailed", action="store_true", help="Print all features and question styles"
    )
    parser.add_argument("--output", type=Path, default=None, help="Write JSON result")
    parser.add_argument(
        "--csv_output", type=Path, default=None, help="Write flat CSV tables"
    )
    parser.add_argument(
        "--records_output",
        type=Path,
        default=None,
        help="Write one audit row per aligned question as CSV",
    )
    args = parser.parse_args()

    if not args.data_root.is_dir():
        parser.error(f"--data_root is not a directory: {args.data_root}")

    db_ids = args.db_ids
    if args.db_list is not None:
        if db_ids:
            parser.error("Use only one of --db_ids or --db_list")
        db_ids = _load_json(args.db_list)
        if not all(isinstance(item, str) for item in db_ids):
            parser.error("--db_list must contain a JSON list of strings")

    try:
        result, records = analyze(
            data_root=args.data_root,
            db_ids=db_ids,
            expected_acceptance_consensus=args.expected_acceptance_consensus,
            expected_candidate_slots=args.expected_candidate_slots,
            source_label=args.source_label,
            collect_records=args.records_output is not None,
        )
    except (OSError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    print_report(result, detailed=args.detailed)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        print(f"JSON saved to {args.output}")

    if args.csv_output is not None:
        _write_csv(args.csv_output, result)
        print(f"CSV saved to {args.csv_output}")

    if args.records_output is not None:
        _write_records_csv(args.records_output, records)
        print(f"Record-level CSV saved to {args.records_output}")

    failures = _critical_sanity_failures(result)
    if failures:
        message = ", ".join(f"{key}={value}" for key, value in failures.items())
        level = "ERROR" if args.strict else "WARNING"
        print(f"[{level}] Critical sanity checks failed: {message}", file=sys.stderr)
        if args.strict:
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
