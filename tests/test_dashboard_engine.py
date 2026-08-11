import pandas as pd

from src.analytics import build_dashboard_report


def test_build_dashboard_report_detects_core_kpis() -> None:
    dataframe = pd.DataFrame(
        {
            "Order Date": pd.to_datetime(["2026-01-01", "2026-02-01", "2026-02-15"]),
            "Revenue": [100.0, 150.0, 50.0],
            "Profit": [20.0, 30.0, 10.0],
            "Customer": ["A", "B", "A"],
            "Region": ["Auckland", "Wellington", "Auckland"],
            "Product": ["Widget", "Widget", "Service"],
        }
    )

    report = build_dashboard_report(dataframe)

    assert report.kpis.revenue == 300.0
    assert report.kpis.profit == 60.0
    assert report.kpis.customers == 2
    assert report.kpis.growth_percent == 100.0
    assert report.dataset_type == "Sales"
    assert [kpi.label for kpi in report.dynamic_kpis] == ["Revenue", "Profit", "Orders", "Customers"]
    assert report.detected_fields["revenue"] == "Revenue"
    assert report.detected_fields["date"] == "Order Date"


def test_build_dashboard_report_creates_expected_figures() -> None:
    dataframe = pd.DataFrame(
        {
            "Order Date": pd.to_datetime(["2026-01-01", "2026-02-01"]),
            "Sales Amount": [100.0, 150.0],
            "Region": ["Auckland", "Wellington"],
            "Product Category": ["Hardware", "Services"],
        }
    )

    report = build_dashboard_report(dataframe)

    assert report.figures.sales_trend is not None
    assert report.figures.regional_analysis is not None
    assert report.figures.product_analysis is not None


def test_build_dashboard_report_handles_missing_business_fields() -> None:
    dataframe = pd.DataFrame({"Name": ["A", "B"], "Status": ["Open", "Closed"]})

    report = build_dashboard_report(dataframe)

    assert report.kpis.revenue is None
    assert report.kpis.growth_percent is None
    assert report.kpis.profit is None
    assert report.kpis.customers is None
    assert report.dataset_type == "General"
    assert [kpi.label for kpi in report.dynamic_kpis] == ["Rows", "Columns", "Numeric Fields", "Data Quality Score"]
    assert report.figures.sales_trend is None


def test_build_dashboard_report_creates_healthcare_kpis_without_sales_fields() -> None:
    dataframe = pd.DataFrame(
        {
            "Patient ID": ["P1", "P2", "P3"],
            "Age": [30, 40, 50],
            "Treatment Cost": [1000.0, 1500.0, 1200.0],
            "Diagnosis": ["Flu", "Cold", "Flu"],
        }
    )

    report = build_dashboard_report(dataframe)

    assert report.dataset_type == "Healthcare"
    assert {kpi.label: kpi.value for kpi in report.dynamic_kpis} == {
        "Total Patients": "3",
        "Average Age": "40.0",
        "Average Treatment Cost": "$1,233",
        "Unique Diagnoses": "2",
    }


def test_build_dashboard_report_creates_inventory_kpis() -> None:
    dataframe = pd.DataFrame(
        {
            "SKU": ["A", "B", "C"],
            "Stock Quantity": [10, 20, 30],
            "Supplier": ["S1", "S1", "S2"],
            "Warehouse": ["North", "South", "North"],
        }
    )

    report = build_dashboard_report(dataframe)

    assert report.dataset_type == "Inventory"
    assert {kpi.label: kpi.value for kpi in report.dynamic_kpis} == {
        "Total Stock": "60.0",
        "Items": "3",
        "Suppliers": "2",
        "Warehouses": "2",
    }