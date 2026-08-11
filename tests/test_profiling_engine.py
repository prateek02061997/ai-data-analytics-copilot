import pandas as pd

from src.analytics import generate_profile_report


def test_generate_profile_report_detects_missing_values_and_duplicates() -> None:
    dataframe = pd.DataFrame(
        {
            "Customer ID": ["C001", None, "C001"],
            "Revenue": [100, 200, 100],
            "Region": ["Auckland", "Wellington", "Auckland"],
        }
    )

    report = generate_profile_report(dataframe)

    assert report.rows == 3
    assert report.columns == 3
    assert report.missing_values == 1
    assert report.duplicate_records == 1
    assert report.health_score < 100
    assert {issue.category for issue in report.issues} >= {"Missing values", "Duplicate records"}


def test_generate_profile_report_detects_outliers() -> None:
    dataframe = pd.DataFrame({"Revenue": [100, 102, 98, 101, 10_000]})

    report = generate_profile_report(dataframe)

    outlier_issues = [issue for issue in report.issues if issue.category == "Outliers"]

    assert len(outlier_issues) == 1
    assert outlier_issues[0].field == "Revenue"
    assert outlier_issues[0].affected_rows == 1


def test_generate_profile_report_detects_wrong_data_types_and_invalid_formats() -> None:
    dataframe = pd.DataFrame(
        {
            "Revenue": ["100", "250", "300", "400"],
            "Order Date": ["2026-01-01", "not-a-date", "2026-01-03", "bad-date"],
        }
    )

    report = generate_profile_report(dataframe)
    messages = [issue.message for issue in report.issues]

    assert any("Revenue appears numeric" in message for message in messages)
    assert any("Order Date contains invalid" in message for message in messages)


def test_generate_profile_report_detects_inconsistent_categories() -> None:
    dataframe = pd.DataFrame({"City": ["Auckland", "auckland", " Auckland ", "Wellington"]})

    report = generate_profile_report(dataframe)

    assert any(issue.category == "Inconsistent categories" and issue.field == "City" for issue in report.issues)


def test_generate_profile_report_returns_ready_recommendation_for_clean_data() -> None:
    dataframe = pd.DataFrame({"Revenue": [100, 120, 130], "Customer": ["A", "B", "C"]})

    report = generate_profile_report(dataframe)

    assert report.health_score == 100
    assert report.issues == []
    assert report.recommendations == ["Dataset is ready for initial analysis."]


def test_generate_profile_report_includes_numerical_and_time_stats() -> None:
    dataframe = pd.DataFrame(
        {
            "Revenue": [100, 200, 300, 400, 500],
            "Category": ["Electronics", "Electronics", "Books", "Books", "Toys"],
            "Order Date": ["2026-01-01", "2026-01-15", "2026-02-01", "2026-02-15", "2026-03-01"],
        }
    )

    report = generate_profile_report(dataframe)

    assert report.completion_percentage == 100.0
    assert len(report.numerical_stats) == 1
    assert report.numerical_stats[0].mean == 300.0
    assert report.numerical_stats[0].median == 300.0
    assert report.numerical_stats[0].min_val == 100.0
    assert report.numerical_stats[0].max_val == 500.0

    assert len(report.category_stats) >= 1
    cat_stat = next(cs for cs in report.category_stats if cs.field == "Category")
    assert cat_stat.unique_count == 3

    assert len(report.time_analysis) == 1
    assert report.time_analysis[0].field == "Order Date"
    assert len(report.time_analysis[0].monthly_trend) >= 1