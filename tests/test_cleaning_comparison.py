import pandas as pd

from src.analytics import build_cleaning_comparison, generate_profile_report
from src.cleaning import clean_dataframe


def test_build_cleaning_comparison_reports_before_after_metrics() -> None:
    dataframe = pd.DataFrame(
        {
            "Customer": ["A", "A", None],
            "Revenue": [100.0, 100.0, 300.0],
        }
    )
    before_profile = generate_profile_report(dataframe)
    cleaning_result = clean_dataframe(dataframe)
    after_profile = generate_profile_report(cleaning_result.cleaned_dataframe)

    comparison = build_cleaning_comparison(before_profile, after_profile, cleaning_result.summary)

    metrics = {item.metric: item for item in comparison.metrics}
    assert metrics["Rows"].before == 3
    assert metrics["Rows"].after == 2
    assert metrics["Missing Values"].before == 1
    assert metrics["Missing Values"].after == 0
    assert metrics["Duplicate Records"].before == 1
    assert metrics["Duplicate Records"].after == 0
    assert comparison.health_score_after >= comparison.health_score_before
    assert "Cleaning Comparison Report" in comparison.markdown
    assert "Handle missing values" in comparison.markdown