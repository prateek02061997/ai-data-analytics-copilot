import pandas as pd

from src.visualization import build_chart, build_chart_suite, convert_chart_type, recommend_chart


def test_recommend_chart_uses_line_for_datetime_trend() -> None:
    dataframe = pd.DataFrame({"Date": pd.to_datetime(["2026-01-01", "2026-01-02"]), "Revenue": [100, 120]})

    recommendation = recommend_chart(dataframe)

    assert recommendation.chart_type == "line"
    assert recommendation.x_column == "Date"
    assert recommendation.y_column == "Revenue"


def test_recommend_chart_uses_bar_for_category_comparison() -> None:
    dataframe = pd.DataFrame({"Region": ["Auckland", "Wellington"], "Revenue": [100, 120]})

    recommendation = recommend_chart(dataframe)

    assert recommendation.chart_type == "bar"
    assert recommendation.x_column == "Region"
    assert recommendation.y_column == "Revenue"


def test_recommend_chart_uses_histogram_for_distribution_intent() -> None:
    dataframe = pd.DataFrame({"Revenue": [100, 120, 140]})

    recommendation = recommend_chart(dataframe, intent="Show distribution")

    assert recommendation.chart_type == "histogram"


def test_recommend_chart_uses_scatter_for_two_numeric_measures() -> None:
    dataframe = pd.DataFrame({"Revenue": [100, 120, 140], "Profit": [20, 30, 35]})

    recommendation = recommend_chart(dataframe)

    assert recommendation.chart_type == "scatter"


def test_build_chart_returns_downloadable_html() -> None:
    dataframe = pd.DataFrame({"Region": ["Auckland", "Wellington"], "Revenue": [100, 120]})

    result = build_chart(dataframe)

    assert result.recommendation.chart_type == "bar"
    assert "<html>" in result.html.lower()


def test_build_chart_suite_generates_multiple_interactive_charts() -> None:
    dataframe = pd.DataFrame(
        {
            "Visit Date": pd.to_datetime(["2026-01-01", "2026-01-15", "2026-02-01", "2026-02-15"]),
            "Age": [30, 40, 50, 60],
            "Cholesterol": [180, 220, 200, 240],
            "Diagnosis": ["Flu", "Cold", "Flu", "Asthma"],
        }
    )

    charts = build_chart_suite(dataframe)
    chart_types = {chart.chart_type for chart in charts}

    assert {"histogram", "box", "bar", "pie", "line", "scatter", "heatmap"}.issubset(chart_types)
    assert all("<html>" in chart.html.lower() for chart in charts)


def test_build_chart_suite_returns_empty_for_empty_dataframe() -> None:
    assert build_chart_suite(pd.DataFrame()) == []


def test_convert_chart_type_generates_figure() -> None:
    dataframe = pd.DataFrame({"Region": ["Auckland", "Wellington"], "Revenue": [100, 120]})
    fig = convert_chart_type(dataframe, x_col="Region", y_col="Revenue", target_type="bar")
    assert fig is not None
    assert hasattr(fig, "to_html")