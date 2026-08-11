"""Executive dashboard metrics and chart data preparation."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import plotly.express as px
from pandas.api import types as pd_types
from plotly.graph_objects import Figure


@dataclass(frozen=True)
class DashboardKpis:
    """Executive KPI values detected from a business dataset."""

    revenue: float | None
    growth_percent: float | None
    profit: float | None
    customers: int | None


@dataclass(frozen=True)
class DashboardKpiItem:
    """One display-ready dashboard KPI card."""

    label: str
    value: str
    detail: str


@dataclass(frozen=True)
class DashboardFigures:
    """Executive dashboard chart figures."""

    sales_trend: Figure | None
    regional_analysis: Figure | None
    product_analysis: Figure | None


@dataclass(frozen=True)
class DashboardReport:
    """Prepared executive dashboard assets."""

    kpis: DashboardKpis
    dynamic_kpis: list[DashboardKpiItem]
    dataset_type: str
    figures: DashboardFigures
    detected_fields: dict[str, str | None]


def build_dashboard_report(dataframe: pd.DataFrame) -> DashboardReport:
    """Build KPI values and dashboard charts from a cleaned dataset."""
    fields = _detect_dashboard_fields(dataframe)
    revenue_column = fields["revenue"]
    profit_column = fields["profit"]
    customer_column = fields["customer"]
    date_column = fields["date"]
    region_column = fields["region"]
    product_column = fields["product"]
    dataset_type = _detect_dataset_type(dataframe)

    kpis = DashboardKpis(
        revenue=_sum_column(dataframe, revenue_column),
        growth_percent=_calculate_growth(dataframe, date_column, revenue_column),
        profit=_sum_column(dataframe, profit_column),
        customers=_count_customers(dataframe, customer_column),
    )
    figures = DashboardFigures(
        sales_trend=_build_sales_trend(dataframe, date_column, revenue_column),
        regional_analysis=_build_category_analysis(dataframe, region_column, revenue_column, "Regional Revenue"),
        product_analysis=_build_category_analysis(dataframe, product_column, revenue_column, "Product Revenue"),
    )
    return DashboardReport(
        kpis=kpis,
        dynamic_kpis=_build_dynamic_kpis(dataframe, fields, dataset_type, kpis),
        dataset_type=dataset_type,
        figures=figures,
        detected_fields=fields,
    )


def _detect_dataset_type(dataframe: pd.DataFrame) -> str:
    normalized_columns = " ".join(str(column).casefold() for column in dataframe.columns)
    if any(keyword in normalized_columns for keyword in ["patient", "diagnosis", "treatment", "hospital", "medical", "doctor"]):
        return "Healthcare"
    if any(keyword in normalized_columns for keyword in ["stock", "inventory", "warehouse", "supplier", "sku", "quantity"]):
        return "Inventory"
    if any(keyword in normalized_columns for keyword in ["employee", "salary", "department", "hire", "attrition", "job"]):
        return "HR"
    if any(keyword in normalized_columns for keyword in ["revenue", "sales", "profit", "order", "customer"]):
        return "Sales"
    if any(keyword in normalized_columns for keyword in ["amount", "balance", "account", "transaction", "payment", "cost"]):
        return "Finance"
    return "General"


def _build_dynamic_kpis(
    dataframe: pd.DataFrame,
    fields: dict[str, str | None],
    dataset_type: str,
    sales_kpis: DashboardKpis,
) -> list[DashboardKpiItem]:
    if dataset_type == "Healthcare":
        return _healthcare_kpis(dataframe)
    if dataset_type == "Inventory":
        return _inventory_kpis(dataframe)
    if dataset_type == "Sales":
        return _sales_kpis(dataframe, fields, sales_kpis)
    return _generic_kpis(dataframe)


def _healthcare_kpis(dataframe: pd.DataFrame) -> list[DashboardKpiItem]:
    patient_column = _find_column([str(column) for column in dataframe.columns], ["patient", "id"], [str(column) for column in dataframe.columns])
    age_column = _first_matching_numeric(dataframe, ["age"])
    cost_column = _first_matching_numeric(dataframe, ["cost", "charge", "amount", "bill"])
    diagnosis_column = _find_column([str(column) for column in dataframe.columns], ["diagnosis", "condition", "disease"], [str(column) for column in dataframe.columns])
    return [
        DashboardKpiItem("Total Patients", _format_integer(_unique_or_rows(dataframe, patient_column)), "Detected from patient identifier or row count."),
        DashboardKpiItem("Average Age", _format_number(_mean_column(dataframe, age_column)), f"Column: {age_column}" if age_column else "No age column detected."),
        DashboardKpiItem("Average Treatment Cost", _format_currency(_mean_column(dataframe, cost_column)), f"Column: {cost_column}" if cost_column else "No cost column detected."),
        DashboardKpiItem("Unique Diagnoses", _format_integer(_count_unique(dataframe, diagnosis_column)), f"Column: {diagnosis_column}" if diagnosis_column else "No diagnosis column detected."),
    ]


def _inventory_kpis(dataframe: pd.DataFrame) -> list[DashboardKpiItem]:
    stock_column = _first_matching_numeric(dataframe, ["stock", "quantity", "qty", "units"])
    item_column = _find_column([str(column) for column in dataframe.columns], ["item", "product", "sku"], [str(column) for column in dataframe.columns])
    supplier_column = _find_column([str(column) for column in dataframe.columns], ["supplier", "vendor"], [str(column) for column in dataframe.columns])
    warehouse_column = _find_column([str(column) for column in dataframe.columns], ["warehouse", "location"], [str(column) for column in dataframe.columns])
    return [
        DashboardKpiItem("Total Stock", _format_number(_sum_column(dataframe, stock_column)), f"Column: {stock_column}" if stock_column else "No stock column detected."),
        DashboardKpiItem("Items", _format_integer(_count_unique(dataframe, item_column)), f"Column: {item_column}" if item_column else "No item column detected."),
        DashboardKpiItem("Suppliers", _format_integer(_count_unique(dataframe, supplier_column)), f"Column: {supplier_column}" if supplier_column else "No supplier column detected."),
        DashboardKpiItem("Warehouses", _format_integer(_count_unique(dataframe, warehouse_column)), f"Column: {warehouse_column}" if warehouse_column else "No warehouse column detected."),
    ]


def _sales_kpis(dataframe: pd.DataFrame, fields: dict[str, str | None], sales_kpis: DashboardKpis) -> list[DashboardKpiItem]:
    order_column = _find_column([str(column) for column in dataframe.columns], ["order", "invoice", "transaction"], [str(column) for column in dataframe.columns])
    return [
        DashboardKpiItem("Revenue", _format_currency(sales_kpis.revenue), f"Column: {fields['revenue']}" if fields["revenue"] else "Revenue-like column not detected."),
        DashboardKpiItem("Profit", _format_currency(sales_kpis.profit), f"Column: {fields['profit']}" if fields["profit"] else "Profit-like column not detected."),
        DashboardKpiItem("Orders", _format_integer(_unique_or_rows(dataframe, order_column)), f"Column: {order_column}" if order_column else "Using row count as order count."),
        DashboardKpiItem("Customers", _format_integer(sales_kpis.customers), f"Column: {fields['customer']}" if fields["customer"] else "Customer-like column not detected."),
    ]


def _generic_kpis(dataframe: pd.DataFrame) -> list[DashboardKpiItem]:
    numeric_columns = [str(column) for column in dataframe.columns if pd_types.is_numeric_dtype(dataframe[column])]
    category_columns = [
        str(column)
        for column in dataframe.columns
        if pd_types.is_object_dtype(dataframe[column]) or pd_types.is_string_dtype(dataframe[column])
    ]
    missing_values = int(dataframe.isna().sum().sum())
    total_cells = max(len(dataframe) * max(len(dataframe.columns), 1), 1)
    quality_score = round((1 - (missing_values / total_cells)) * 100)
    return [
        DashboardKpiItem("Rows", _format_integer(len(dataframe)), "Total records available."),
        DashboardKpiItem("Columns", _format_integer(len(dataframe.columns)), "Total fields available."),
        DashboardKpiItem("Numeric Fields", _format_integer(len(numeric_columns)), "Detected numeric columns."),
        DashboardKpiItem("Data Quality Score", f"{quality_score}%", f"Based on {missing_values:,} missing value(s)."),
    ]


def _detect_dashboard_fields(dataframe: pd.DataFrame) -> dict[str, str | None]:
    columns = [str(column) for column in dataframe.columns]
    numeric_columns = [column for column in columns if pd_types.is_numeric_dtype(dataframe[column])]
    datetime_columns = [column for column in columns if pd_types.is_datetime64_any_dtype(dataframe[column])]

    return {
        "revenue": _find_column(columns, ["revenue", "sales", "amount", "turnover", "income"], numeric_columns),
        "profit": _find_column(columns, ["profit", "margin", "earnings"], numeric_columns),
        "customer": _find_column(columns, ["customer", "client", "account", "buyer"], columns),
        "date": _find_column(columns, ["date", "month", "period", "time"], datetime_columns) or (datetime_columns[0] if datetime_columns else None),
        "region": _find_column(columns, ["region", "city", "country", "state", "market", "location"], columns),
        "product": _find_column(columns, ["product", "item", "sku", "category", "service"], columns),
    }


def _find_column(columns: list[str], keywords: list[str], candidates: list[str]) -> str | None:
    candidate_set = set(candidates)
    for keyword in keywords:
        for column in columns:
            if column in candidate_set and keyword in column.casefold():
                return column
    return None


def _first_matching_numeric(dataframe: pd.DataFrame, keywords: list[str]) -> str | None:
    columns = [str(column) for column in dataframe.columns]
    numeric_columns = [column for column in columns if pd_types.is_numeric_dtype(dataframe[column])]
    return _find_column(columns, keywords, numeric_columns)


def _sum_column(dataframe: pd.DataFrame, column: str | None) -> float | None:
    if not column:
        return None
    return float(pd.to_numeric(dataframe[column], errors="coerce").fillna(0).sum())


def _mean_column(dataframe: pd.DataFrame, column: str | None) -> float | None:
    if not column:
        return None
    values = pd.to_numeric(dataframe[column], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.mean())


def _count_unique(dataframe: pd.DataFrame, column: str | None) -> int | None:
    if not column:
        return None
    return int(dataframe[column].nunique(dropna=True))


def _unique_or_rows(dataframe: pd.DataFrame, column: str | None) -> int:
    if not column:
        return len(dataframe)
    return int(dataframe[column].nunique(dropna=True))


def _format_currency(value: float | None) -> str:
    if value is None:
        return "Not detected"
    return f"${value:,.0f}"


def _format_number(value: float | None) -> str:
    if value is None:
        return "Not detected"
    return f"{value:,.1f}"


def _format_integer(value: int | None) -> str:
    if value is None:
        return "Not detected"
    return f"{value:,}"


def _count_customers(dataframe: pd.DataFrame, column: str | None) -> int | None:
    if not column:
        return None
    return int(dataframe[column].nunique(dropna=True))


def _calculate_growth(dataframe: pd.DataFrame, date_column: str | None, revenue_column: str | None) -> float | None:
    if not date_column or not revenue_column:
        return None
    working = dataframe[[date_column, revenue_column]].dropna().copy()
    if working.empty:
        return None
    working[date_column] = pd.to_datetime(working[date_column], errors="coerce")
    working[revenue_column] = pd.to_numeric(working[revenue_column], errors="coerce")
    working = working.dropna()
    if working.empty:
        return None

    trend = working.groupby(pd.Grouper(key=date_column, freq="ME"))[revenue_column].sum()
    trend = trend[trend > 0]
    if len(trend) < 2:
        return None
    previous = float(trend.iloc[-2])
    current = float(trend.iloc[-1])
    if previous == 0:
        return None
    return ((current - previous) / previous) * 100


def _build_sales_trend(dataframe: pd.DataFrame, date_column: str | None, revenue_column: str | None) -> Figure | None:
    if not date_column or not revenue_column:
        return None
    working = dataframe[[date_column, revenue_column]].dropna().copy()
    if working.empty:
        return None
    working[date_column] = pd.to_datetime(working[date_column], errors="coerce")
    working[revenue_column] = pd.to_numeric(working[revenue_column], errors="coerce")
    working = working.dropna()
    if working.empty:
        return None
    trend = working.groupby(pd.Grouper(key=date_column, freq="ME"))[revenue_column].sum().reset_index()
    figure = px.line(trend, x=date_column, y=revenue_column, markers=True, title="Sales Trend")
    return _style_figure(figure)


def _build_category_analysis(dataframe: pd.DataFrame, category_column: str | None, revenue_column: str | None, title: str) -> Figure | None:
    if not category_column or not revenue_column:
        return None
    working = dataframe[[category_column, revenue_column]].dropna().copy()
    if working.empty:
        return None
    working[revenue_column] = pd.to_numeric(working[revenue_column], errors="coerce")
    working = working.dropna()
    if working.empty:
        return None
    grouped = working.groupby(category_column, as_index=False)[revenue_column].sum()
    grouped = grouped.sort_values(revenue_column, ascending=False).head(10)
    figure = px.bar(grouped, x=category_column, y=revenue_column, title=title)
    return _style_figure(figure)


def _style_figure(figure: Figure) -> Figure:
    figure.update_layout(template="plotly_white", margin={"l": 24, "r": 24, "t": 56, "b": 24})
    return figure