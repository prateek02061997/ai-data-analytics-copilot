"""Power BI export preparation for saved analytics datasets."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from io import StringIO
from typing import Any

import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_integer_dtype, is_numeric_dtype


@dataclass(frozen=True)
class PowerBIField:
    """Column metadata prepared for Power BI import."""

    name: str
    pandas_dtype: str
    power_bi_type: str
    nullable: bool


@dataclass(frozen=True)
class PowerBIExport:
    """Downloadable Power BI preparation assets."""

    table_name: str
    csv_file_name: str
    schema_file_name: str
    power_query_file_name: str
    csv_bytes: bytes
    schema_json: str
    power_query_m: str
    fields: list[PowerBIField]


def build_power_bi_export(dataframe: pd.DataFrame, table_name: str) -> PowerBIExport:
    """Build CSV, schema JSON, and Power Query M assets for Power BI Desktop."""
    if dataframe.empty:
        raise ValueError("Cannot prepare Power BI export for an empty dataset.")

    safe_name = _safe_table_name(table_name)
    fields = [_field_from_series(name=str(column), series=dataframe[column]) for column in dataframe.columns]
    csv_bytes = _dataframe_to_csv_bytes(dataframe)
    schema_json = json.dumps(
        {
            "table_name": safe_name,
            "source_table": table_name,
            "row_count": int(len(dataframe)),
            "column_count": int(len(dataframe.columns)),
            "fields": [asdict(field) for field in fields],
        },
        indent=2,
        sort_keys=True,
    )

    return PowerBIExport(
        table_name=safe_name,
        csv_file_name=f"{safe_name}.csv",
        schema_file_name=f"{safe_name}_schema.json",
        power_query_file_name=f"{safe_name}_power_query.m",
        csv_bytes=csv_bytes,
        schema_json=schema_json,
        power_query_m=_build_power_query_m(safe_name=safe_name, fields=fields),
        fields=fields,
    )


def _dataframe_to_csv_bytes(dataframe: pd.DataFrame) -> bytes:
    buffer = StringIO()
    dataframe.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8-sig")


def dataframe_to_excel_bytes(dataframe: pd.DataFrame) -> bytes:
    """Convert a DataFrame into Excel .xlsx binary format, removing timezones for openpyxl compatibility."""
    from io import BytesIO

    clean_df = dataframe.copy()
    for col in clean_df.columns:
        if is_datetime64_any_dtype(clean_df[col]):
            try:
                if getattr(clean_df[col].dt, "tz", None) is not None:
                    clean_df[col] = clean_df[col].dt.tz_localize(None)
            except Exception:
                clean_df[col] = clean_df[col].astype(str)

    buffer = BytesIO()
    try:
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            clean_df.to_excel(writer, index=False, sheet_name="Data")
    except Exception:
        # Fallback: convert any troublesome columns to strings for guaranteed export
        for col in clean_df.columns:
            if is_datetime64_any_dtype(clean_df[col]):
                clean_df[col] = clean_df[col].astype(str)
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            clean_df.to_excel(writer, index=False, sheet_name="Data")

    return buffer.getvalue()


def _field_from_series(name: str, series: pd.Series) -> PowerBIField:
    return PowerBIField(
        name=name,
        pandas_dtype=str(series.dtype),
        power_bi_type=_power_bi_type(series),
        nullable=bool(series.isna().any()),
    )


def _power_bi_type(series: pd.Series) -> str:
    if is_bool_dtype(series):
        return "type logical"
    if is_integer_dtype(series):
        return "Int64.Type"
    if is_numeric_dtype(series):
        return "type number"
    if is_datetime64_any_dtype(series):
        return "type datetime"
    return "type text"


def _build_power_query_m(safe_name: str, fields: list[PowerBIField]) -> str:
    type_pairs = ", ".join(f'{{"{_escape_m(field.name)}", {field.power_bi_type}}}' for field in fields)
    csv_file = f"{safe_name}.csv"
    return (
        "let\n"
        f'    Source = Csv.Document(File.Contents("C:\\\\path\\\\to\\\\{csv_file}"), '
        '[Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.Csv]),\n'
        "    PromotedHeaders = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),\n"
        f"    TypedColumns = Table.TransformColumnTypes(PromotedHeaders, {{{type_pairs}}})\n"
        "in\n"
        "    TypedColumns\n"
    )


def _safe_table_name(table_name: str) -> str:
    safe_name = re.sub(r"[^A-Za-z0-9_]+", "_", table_name).strip("_")
    return safe_name or "power_bi_dataset"


def _escape_m(value: Any) -> str:
    return str(value).replace('"', '""')