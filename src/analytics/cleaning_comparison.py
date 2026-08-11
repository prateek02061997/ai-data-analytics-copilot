"""Before and after cleaning comparison metrics."""

from __future__ import annotations

from dataclasses import dataclass

from src.analytics.profiling_engine import ProfileReport
from src.cleaning import CleaningSummary


@dataclass(frozen=True)
class CleaningComparisonMetric:
    """One before/after metric for the cleaning comparison view."""

    metric: str
    before: int | str
    after: int | str
    change: int | str


@dataclass(frozen=True)
class CleaningComparisonReport:
    """Cleaning comparison summary and downloadable report content."""

    metrics: list[CleaningComparisonMetric]
    health_score_before: int
    health_score_after: int
    health_score_improvement: int
    percentage_improvement: float
    markdown: str


def build_cleaning_comparison(
    before_profile: ProfileReport,
    after_profile: ProfileReport,
    cleaning_summary: CleaningSummary,
) -> CleaningComparisonReport:
    """Compare original and cleaned dataset quality."""
    metrics = [
        _metric("Rows", before_profile.rows, after_profile.rows),
        _metric("Columns", before_profile.columns, after_profile.columns),
        _metric("Missing Values", before_profile.missing_values, after_profile.missing_values),
        _metric("Duplicate Records", before_profile.duplicate_records, after_profile.duplicate_records),
        _metric("Outliers", _issue_count(before_profile, "Outliers"), _issue_count(after_profile, "Outliers")),
        _metric("Health Score", f"{before_profile.health_score}%", f"{after_profile.health_score}%"),
        _metric("Cleaning Actions", 0, len(cleaning_summary.actions)),
    ]
    health_score_improvement = after_profile.health_score - before_profile.health_score
    percentage_improvement = _percentage_improvement(before_profile.health_score, after_profile.health_score)
    markdown = _build_markdown(metrics, cleaning_summary, health_score_improvement, percentage_improvement)
    return CleaningComparisonReport(
        metrics=metrics,
        health_score_before=before_profile.health_score,
        health_score_after=after_profile.health_score,
        health_score_improvement=health_score_improvement,
        percentage_improvement=percentage_improvement,
        markdown=markdown,
    )


def _metric(metric: str, before: int | str, after: int | str) -> CleaningComparisonMetric:
    if isinstance(before, int) and isinstance(after, int):
        change: int | str = after - before
    else:
        change = ""
    return CleaningComparisonMetric(metric=metric, before=before, after=after, change=change)


def _issue_count(profile: ProfileReport, category: str) -> int:
    return sum(issue.affected_rows for issue in profile.issues if issue.category == category)


def _percentage_improvement(before_score: int, after_score: int) -> float:
    remaining_gap = max(100 - before_score, 1)
    return max(0.0, ((after_score - before_score) / remaining_gap) * 100)


def _build_markdown(
    metrics: list[CleaningComparisonMetric],
    cleaning_summary: CleaningSummary,
    health_score_improvement: int,
    percentage_improvement: float,
) -> str:
    rows = [
        "# Cleaning Comparison Report",
        "",
        "## Quality Improvement",
        f"- Health score change: {health_score_improvement:+d} points",
        f"- Gap closed: {percentage_improvement:.1f}%",
        "",
        "## Before vs After",
        "| Metric | Before | After | Change |",
        "| --- | ---: | ---: | ---: |",
    ]
    rows.extend(f"| {item.metric} | {item.before} | {item.after} | {item.change} |" for item in metrics)
    rows.extend(["", "## Cleaning Actions"])
    if cleaning_summary.actions:
        rows.extend(
            f"- {action.action}: {action.records_affected:,} record(s)"
            + (f" in {action.field}" if action.field else "")
            + (f". {action.details}" if action.details else "")
            for action in cleaning_summary.actions
        )
    else:
        rows.append("- No cleaning changes were required.")
    return "\n".join(rows) + "\n"