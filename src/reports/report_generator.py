"""Executive report generation for cleaned and saved datasets."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from typing import Any

import pandas as pd

from src.analytics import ProfileReport, build_dashboard_report


@dataclass(frozen=True)
class ExecutiveReport:
    """Downloadable executive report assets."""

    title: str
    html: str
    markdown: str
    html_file_name: str
    markdown_file_name: str


def build_executive_report(
    dataframe: pd.DataFrame,
    dataset_name: str,
    profile_report: ProfileReport,
    cleaning_summary: Any | None = None,
) -> ExecutiveReport:
    """Build an executive summary report from the current dataset state."""
    if dataframe.empty:
        raise ValueError("Cannot generate an executive report for an empty dataset.")

    safe_name = _safe_file_name(dataset_name)
    title = f"Executive Report: {dataset_name}"
    dashboard = build_dashboard_report(dataframe)
    numeric_columns = [str(column) for column in dataframe.columns if pd.api.types.is_numeric_dtype(dataframe[column])]
    text_columns = [str(column) for column in dataframe.columns if pd.api.types.is_object_dtype(dataframe[column])]
    top_issues = profile_report.issues[:5]
    recommendations = profile_report.recommendations[:5]

    markdown = _build_markdown(
        title=title,
        dataframe=dataframe,
        profile_report=profile_report,
        cleaning_summary=cleaning_summary,
        numeric_columns=numeric_columns,
        text_columns=text_columns,
        kpis={
            "Revenue": _format_optional_currency(dashboard.kpis.revenue),
            "Growth": _format_optional_percent(dashboard.kpis.growth_percent),
            "Profit": _format_optional_currency(dashboard.kpis.profit),
            "Customers": _format_optional_integer(dashboard.kpis.customers),
        },
        top_issues=top_issues,
        recommendations=recommendations,
    )
    html = _markdown_to_html(markdown, title=title)
    return ExecutiveReport(
        title=title,
        html=html,
        markdown=markdown,
        html_file_name=f"{safe_name}_executive_report.html",
        markdown_file_name=f"{safe_name}_executive_report.md",
    )


def _build_markdown(
    title: str,
    dataframe: pd.DataFrame,
    profile_report: ProfileReport,
    cleaning_summary: Any | None,
    numeric_columns: list[str],
    text_columns: list[str],
    kpis: dict[str, str],
    top_issues: list[Any],
    recommendations: list[str],
) -> str:
    rows = [
        f"# {title}",
        "",
        "## Dataset Summary",
        f"- Rows: {len(dataframe):,}",
        f"- Columns: {len(dataframe.columns):,}",
        f"- Numeric fields: {len(numeric_columns):,}",
        f"- Text fields: {len(text_columns):,}",
        f"- Missing values: {int(dataframe.isna().sum().sum()):,}",
        "",
        "## Data Quality",
        f"- Health score: {profile_report.health_score}%",
        f"- Profiling issues: {len(profile_report.issues):,}",
        f"- Duplicate records: {profile_report.duplicate_records:,}",
        "",
        "## Business KPIs",
    ]

    rows.extend(f"- {name}: {value}" for name, value in kpis.items())
    rows.extend(["", "## Key Issues"])
    if top_issues:
        rows.extend(f"- {issue.category}: {issue.message}" for issue in top_issues)
    else:
        rows.append("- No major profiling issues detected.")

    rows.extend(["", "## Recommendations"])
    rows.extend(f"- {recommendation}" for recommendation in recommendations)

    if cleaning_summary is not None:
        rows.extend(
            [
                "",
                "## Cleaning Summary",
                f"- Rows processed: {cleaning_summary.records_processed:,}",
                f"- Rows modified: {cleaning_summary.rows_modified:,}",
                f"- Cells modified: {cleaning_summary.cells_modified:,}",
                f"- Rows removed: {cleaning_summary.records_removed:,}",
                f"- Missing values filled: {cleaning_summary.missing_values_filled:,}",
                f"- Duplicates removed: {cleaning_summary.duplicates_removed:,}",
                f"- Outliers flagged: {cleaning_summary.outliers_flagged:,}",
                f"- Formats standardized: {cleaning_summary.formats_standardized:,}",
            ]
        )

    rows.extend(["", "## Columns"])
    rows.extend(f"- {column}: {dataframe[column].dtype}" for column in dataframe.columns)
    return "\n".join(rows) + "\n"


def _markdown_to_html(markdown: str, title: str) -> str:
    lines = []
    in_list = False
    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        if not line:
            if in_list:
                lines.append("</ul>")
                in_list = False
            continue
        if line.startswith("# "):
            if in_list:
                lines.append("</ul>")
                in_list = False
            lines.append(f"<h1>{escape(line[2:])}</h1>")
        elif line.startswith("## "):
            if in_list:
                lines.append("</ul>")
                in_list = False
            lines.append(f"<h2>{escape(line[3:])}</h2>")
        elif line.startswith("- "):
            if not in_list:
                lines.append("<ul>")
                in_list = True
            lines.append(f"<li>{escape(line[2:])}</li>")
        else:
            if in_list:
                lines.append("</ul>")
                in_list = False
            lines.append(f"<p>{escape(line)}</p>")
    if in_list:
        lines.append("</ul>")

    body = "\n".join(lines)
    return (
        "<!doctype html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '  <meta charset="utf-8">\n'
        f"  <title>{escape(title)}</title>\n"
        "  <style>body{font-family:Segoe UI,Arial,sans-serif;max-width:960px;margin:40px auto;line-height:1.5;color:#202124}"
        "h1,h2{color:#12355b}li{margin:6px 0}</style>\n"
        "</head>\n"
        f"<body>\n{body}\n</body>\n"
        "</html>\n"
    )


def _safe_file_name(dataset_name: str) -> str:
    safe_name = "".join(character if character.isalnum() else "_" for character in dataset_name.lower()).strip("_")
    return safe_name or "dataset"


def _format_optional_currency(value: float | None) -> str:
    return "N/A" if value is None else f"${value:,.0f}"


def _format_optional_percent(value: float | None) -> str:
    return "N/A" if value is None else f"{value:,.1f}%"


def _format_optional_integer(value: int | None) -> str:
    return "N/A" if value is None else f"{value:,}"