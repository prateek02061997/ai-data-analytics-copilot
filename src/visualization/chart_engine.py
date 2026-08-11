"""Automatic Plotly chart recommendation and generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd
import plotly.express as px
from pandas.api import types as pd_types
from plotly.graph_objects import Figure, Heatmap


ChartType = Literal["line", "bar", "histogram", "scatter", "box", "pie", "heatmap", "table", "area", "treemap", "sunburst"]
MAX_CHART_ROWS = 10_000


@dataclass(frozen=True)
class ChartRecommendation:
    """Recommended chart configuration for a DataFrame."""

    chart_type: ChartType
    title: str
    x_column: str | None
    y_column: str | None
    reason: str


@dataclass(frozen=True)
class ChartResult:
    """Generated chart plus its recommendation metadata."""

    recommendation: ChartRecommendation
    figure: Figure
    html: str


@dataclass(frozen=True)
class ChartSuiteItem:
    """One automatically generated interactive chart."""

    chart_type: ChartType
    title: str
    figure: Figure
    html: str
    reason: str


def convert_chart_type(dataframe: pd.DataFrame, x_col: str | None, y_col: str | None, target_type: str, title: str = "") -> Figure:
    """Convert dataset variables dynamically into a target chart rendering type."""
    if dataframe.empty or not x_col:
        fig = px.scatter(title=title or "No data available")
        fig.add_annotation(text="No data available for chart conversion", showarrow=False, x=0.5, y=0.5)
        fig.update_layout(template="plotly_white", margin={"l": 24, "r": 24, "t": 56, "b": 24})
        return fig

    chart_frame = _sample_for_charts(dataframe)
    title_text = title or f"{target_type.title()} View of {x_col}" + (f" by {y_col}" if y_col else "")

    if target_type == "bar":
        if y_col and pd_types.is_numeric_dtype(chart_frame[y_col]):
            grouped = chart_frame.groupby(x_col, dropna=False, as_index=False)[y_col].sum().head(25)
            fig = px.bar(grouped, x=x_col, y=y_col, title=title_text)
        else:
            counts = chart_frame[x_col].astype(str).value_counts().head(25).reset_index()
            counts.columns = [x_col, "Count"]
            fig = px.bar(counts, x=x_col, y="Count", title=title_text)
    elif target_type in {"line", "area"}:
        if y_col and pd_types.is_numeric_dtype(chart_frame[y_col]):
            grouped = chart_frame.groupby(x_col, dropna=False, as_index=False)[y_col].mean().head(30)
            fig = px.area(grouped, x=x_col, y=y_col, title=title_text) if target_type == "area" else px.line(grouped, x=x_col, y=y_col, markers=True, title=title_text)
        else:
            counts = chart_frame[x_col].astype(str).value_counts().head(30).reset_index()
            counts.columns = [x_col, "Count"]
            fig = px.area(counts, x=x_col, y="Count", title=title_text) if target_type == "area" else px.line(counts, x=x_col, y="Count", markers=True, title=title_text)
    elif target_type == "pie":
        counts = chart_frame[x_col].astype(str).value_counts().head(12).reset_index()
        counts.columns = [x_col, "Count"]
        fig = px.pie(counts, names=x_col, values="Count", title=title_text)
    elif target_type == "scatter":
        if y_col and pd_types.is_numeric_dtype(chart_frame[y_col]):
            fig = px.scatter(chart_frame, x=x_col, y=y_col, title=title_text)
        else:
            counts = chart_frame[x_col].astype(str).value_counts().reset_index()
            counts.columns = [x_col, "Count"]
            fig = px.scatter(counts, x=x_col, y="Count", title=title_text)
    elif target_type == "histogram":
        fig = px.histogram(chart_frame, x=x_col, title=title_text)
    elif target_type == "box":
        if y_col and pd_types.is_numeric_dtype(chart_frame[y_col]):
            fig = px.box(chart_frame, x=x_col, y=y_col, title=title_text)
        else:
            fig = px.box(chart_frame, y=x_col, title=title_text)
    elif target_type == "treemap":
        counts = chart_frame[x_col].astype(str).value_counts().head(20).reset_index()
        counts.columns = [x_col, "Count"]
        fig = px.treemap(counts, path=[x_col], values="Count", title=title_text)
    else:
        fig = px.bar(chart_frame.head(20), x=x_col, title=title_text)

    fig.update_layout(template="plotly_white", margin={"l": 24, "r": 24, "t": 56, "b": 24})
    return fig


def recommend_chart(dataframe: pd.DataFrame, intent: str = "") -> ChartRecommendation:
    """Choose a practical chart type from dataset shape and optional intent."""
    if dataframe.empty:
        return ChartRecommendation("table", "No data available", None, None, "The result has no rows to chart.")

    numeric_columns = [str(column) for column in dataframe.columns if pd_types.is_numeric_dtype(dataframe[column])]
    datetime_columns = [str(column) for column in dataframe.columns if pd_types.is_datetime64_any_dtype(dataframe[column])]
    category_columns = [
        str(column)
        for column in dataframe.columns
        if (pd_types.is_object_dtype(dataframe[column]) or pd_types.is_string_dtype(dataframe[column]))
        and dataframe[column].nunique(dropna=True) <= max(20, int(len(dataframe) * 0.3))
    ]

    lower_intent = intent.casefold()
    if datetime_columns and numeric_columns:
        return ChartRecommendation(
            "line",
            f"{numeric_columns[0]} over {datetime_columns[0]}",
            datetime_columns[0],
            numeric_columns[0],
            "A datetime field and numeric measure are available, which is best suited to a trend line.",
        )
    if category_columns and numeric_columns:
        return ChartRecommendation(
            "bar",
            f"{numeric_columns[0]} by {category_columns[0]}",
            category_columns[0],
            numeric_columns[0],
            "A categorical field and numeric measure are available, which is best suited to comparison.",
        )
    if "distribution" in lower_intent or (numeric_columns and len(dataframe.columns) == 1):
        return ChartRecommendation(
            "histogram",
            f"Distribution of {numeric_columns[0]}",
            numeric_columns[0] if numeric_columns else None,
            None,
            "A single numeric field is best reviewed as a distribution.",
        )
    if len(numeric_columns) >= 2:
        return ChartRecommendation(
            "scatter",
            f"{numeric_columns[1]} vs {numeric_columns[0]}",
            numeric_columns[0],
            numeric_columns[1],
            "Two numeric measures are available, which is useful for relationship analysis.",
        )
    return ChartRecommendation("table", "Table Preview", None, None, "No reliable chart mapping was detected.")


def build_chart(dataframe: pd.DataFrame, intent: str = "") -> ChartResult:
    """Build an interactive Plotly chart from a DataFrame."""
    recommendation = recommend_chart(dataframe, intent=intent)
    chart_frame = dataframe.copy()

    if recommendation.chart_type == "line" and recommendation.x_column and recommendation.y_column:
        chart_frame = chart_frame.sort_values(recommendation.x_column)
        figure = px.line(chart_frame, x=recommendation.x_column, y=recommendation.y_column, title=recommendation.title, markers=True)
    elif recommendation.chart_type == "bar" and recommendation.x_column and recommendation.y_column:
        chart_frame = chart_frame.groupby(recommendation.x_column, dropna=False, as_index=False)[recommendation.y_column].sum()
        chart_frame = chart_frame.sort_values(recommendation.y_column, ascending=False).head(30)
        figure = px.bar(chart_frame, x=recommendation.x_column, y=recommendation.y_column, title=recommendation.title)
    elif recommendation.chart_type == "histogram" and recommendation.x_column:
        figure = px.histogram(chart_frame, x=recommendation.x_column, title=recommendation.title)
    elif recommendation.chart_type == "scatter" and recommendation.x_column and recommendation.y_column:
        figure = px.scatter(chart_frame, x=recommendation.x_column, y=recommendation.y_column, title=recommendation.title)
    else:
        figure = px.scatter(title=recommendation.title)
        figure.add_annotation(text="No chart available for this result shape", showarrow=False, x=0.5, y=0.5)

    figure.update_layout(template="plotly_white", margin={"l": 24, "r": 24, "t": 56, "b": 24})
    html = figure.to_html(full_html=True, include_plotlyjs="cdn")
    return ChartResult(recommendation=recommendation, figure=figure, html=html)


def build_chart_suite(dataframe: pd.DataFrame) -> list[ChartSuiteItem]:
    """Build a comprehensive suite of interactive charts for the available dataset columns."""
    if dataframe.empty:
        return []

    chart_frame = _sample_for_charts(dataframe)
    numeric_columns = [str(column) for column in chart_frame.columns if pd_types.is_numeric_dtype(chart_frame[column])]
    datetime_columns = [str(column) for column in chart_frame.columns if pd_types.is_datetime64_any_dtype(chart_frame[column])]
    category_columns = _category_columns(chart_frame)
    charts: list[ChartSuiteItem] = []

    if numeric_columns:
        charts.append(
            _chart_item(
                "histogram",
                f"Distribution of {numeric_columns[0]}",
                px.histogram(chart_frame, x=numeric_columns[0], title=f"Distribution of {numeric_columns[0]}"),
                "Numeric fields are best checked first with distribution charts.",
            )
        )
        charts.append(
            _chart_item(
                "box",
                f"Outlier View: {numeric_columns[0]}",
                px.box(chart_frame, y=numeric_columns[0], title=f"Outlier View: {numeric_columns[0]}"),
                "Box plots make spread and potential outliers visible.",
            )
        )

    if category_columns:
        category_counts = chart_frame[category_columns[0]].fillna("Missing").astype(str).value_counts().head(15).reset_index()
        category_counts.columns = [category_columns[0], "Count"]
        charts.append(
            _chart_item(
                "bar",
                f"Top {category_columns[0]} Categories",
                px.bar(category_counts, x=category_columns[0], y="Count", title=f"Top {category_columns[0]} Categories"),
                "Categorical fields are summarized by their most common values.",
            )
        )
        charts.append(
            _chart_item(
                "pie",
                f"Share of {category_columns[0]}",
                px.pie(category_counts, names=category_columns[0], values="Count", title=f"Share of {category_columns[0]}"),
                "Pie charts show the category mix for low-cardinality fields.",
            )
        )
        charts.append(
            _chart_item(
                "treemap",
                f"Treemap: {category_columns[0]} Breakdown",
                px.treemap(category_counts, path=[category_columns[0]], values="Count", title=f"Treemap: {category_columns[0]} Breakdown"),
                "Treemaps visualize proportion hierarchy effectively.",
            )
        )

    if len(category_columns) >= 2:
        top_cats = chart_frame.groupby(category_columns[:2]).size().reset_index(name="Count").head(20)
        charts.append(
            _chart_item(
                "sunburst",
                f"Sunburst: {category_columns[0]} & {category_columns[1]}",
                px.sunburst(top_cats, path=[category_columns[0], category_columns[1]], values="Count", title=f"Sunburst: {category_columns[0]} & {category_columns[1]}"),
                "Sunburst charts illustrate nested category relationships.",
            )
        )

    if datetime_columns and numeric_columns:
        trend_frame = chart_frame[[datetime_columns[0], numeric_columns[0]]].dropna().copy()
        if not trend_frame.empty:
            trend_frame[datetime_columns[0]] = pd.to_datetime(trend_frame[datetime_columns[0]], errors="coerce")
            trend_frame[numeric_columns[0]] = pd.to_numeric(trend_frame[numeric_columns[0]], errors="coerce")
            trend_frame = trend_frame.dropna().sort_values(datetime_columns[0])
            if not trend_frame.empty:
                trend_frame = trend_frame.set_index(datetime_columns[0]).resample("ME")[numeric_columns[0]].mean().reset_index()
                charts.append(
                    _chart_item(
                        "line",
                        f"Monthly Trend: {numeric_columns[0]}",
                        px.line(trend_frame, x=datetime_columns[0], y=numeric_columns[0], markers=True, title=f"Monthly Trend: {numeric_columns[0]}"),
                        "Date and numeric fields are summarized as a monthly trend.",
                    )
                )
                charts.append(
                    _chart_item(
                        "area",
                        f"Monthly Area: {numeric_columns[0]}",
                        px.area(trend_frame, x=datetime_columns[0], y=numeric_columns[0], title=f"Monthly Area: {numeric_columns[0]}"),
                        "Area charts emphasize total volume accumulation over time.",
                    )
                )

    if len(numeric_columns) >= 2:
        charts.append(
            _chart_item(
                "scatter",
                f"{numeric_columns[1]} vs {numeric_columns[0]}",
                px.scatter(chart_frame, x=numeric_columns[0], y=numeric_columns[1], title=f"{numeric_columns[1]} vs {numeric_columns[0]}"),
                "Scatter plots help inspect relationships between numeric fields.",
            )
        )
        correlation = chart_frame[numeric_columns].corr(numeric_only=True)
        charts.append(
            _chart_item(
                "heatmap",
                "Correlation Heatmap",
                Figure(data=Heatmap(z=correlation.values, x=correlation.columns.tolist(), y=correlation.index.tolist(), colorscale="RdBu", zmid=0)),
                "Correlation heatmaps show how numeric fields move together.",
            )
        )

    return charts


def _sample_for_charts(dataframe: pd.DataFrame) -> pd.DataFrame:
    if len(dataframe) <= MAX_CHART_ROWS:
        return dataframe.copy()
    return dataframe.sample(n=MAX_CHART_ROWS, random_state=42).copy()


def _category_columns(dataframe: pd.DataFrame) -> list[str]:
    return [
        str(column)
        for column in dataframe.columns
        if (pd_types.is_object_dtype(dataframe[column]) or pd_types.is_string_dtype(dataframe[column]) or pd_types.is_bool_dtype(dataframe[column]))
        and 1 < dataframe[column].nunique(dropna=True) <= 30
    ]


def _chart_item(chart_type: ChartType, title: str, figure: Figure, reason: str) -> ChartSuiteItem:
    figure.update_layout(template="plotly_white", margin={"l": 24, "r": 24, "t": 56, "b": 24})
    return ChartSuiteItem(
        chart_type=chart_type,
        title=title,
        figure=figure,
        html=figure.to_html(full_html=True, include_plotlyjs="cdn"),
        reason=reason,
    )


def _sample_for_charts(dataframe: pd.DataFrame) -> pd.DataFrame:
    if len(dataframe) <= MAX_CHART_ROWS:
        return dataframe.copy()
    return dataframe.sample(n=MAX_CHART_ROWS, random_state=42).copy()


def _category_columns(dataframe: pd.DataFrame) -> list[str]:
    return [
        str(column)
        for column in dataframe.columns
        if (pd_types.is_object_dtype(dataframe[column]) or pd_types.is_string_dtype(dataframe[column]) or pd_types.is_bool_dtype(dataframe[column]))
        and 1 < dataframe[column].nunique(dropna=True) <= 30
    ]


def _chart_item(chart_type: ChartType, title: str, figure: Figure, reason: str) -> ChartSuiteItem:
    figure.update_layout(template="plotly_white", margin={"l": 24, "r": 24, "t": 56, "b": 24})
    return ChartSuiteItem(
        chart_type=chart_type,
        title=title,
        figure=figure,
        html=figure.to_html(full_html=True, include_plotlyjs="cdn"),
        reason=reason,
    )