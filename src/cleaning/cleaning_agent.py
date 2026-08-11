"""Deterministic data cleaning agent for analytics preparation."""

from __future__ import annotations

from dataclasses import dataclass
from re import sub

import pandas as pd
from pandas.api import types as pd_types


NUMERIC_PARSE_THRESHOLD = 0.75
DATETIME_PARSE_THRESHOLD = 0.8
OUTLIER_MIN_VALUES = 4

LOCATION_ALIASES = {
    "akl": "Auckland",
    "auckland": "Auckland",
    "wlg": "Wellington",
    "wellington": "Wellington",
    "chc": "Christchurch",
    "christchurch": "Christchurch",
}


@dataclass(frozen=True)
class CleaningAction:
    """One operation performed by the cleaning agent."""

    action: str
    records_affected: int
    field: str | None = None
    details: str | None = None


@dataclass(frozen=True)
class CleaningSummary:
    """Summary of changes made by the cleaning agent."""

    records_processed: int
    records_removed: int
    records_fixed: int
    rows_modified: int
    cells_modified: int
    duplicates_removed: int
    missing_values_filled: int
    formats_standardized: int
    text_values_standardized: int
    numeric_values_converted: int
    date_values_fixed: int
    outliers_flagged: int
    anomalies_detected: int
    invalid_records_removed: int
    actions: list[CleaningAction]


@dataclass(frozen=True)
class CleaningResult:
    """Cleaned DataFrame and its audit summary."""

    cleaned_dataframe: pd.DataFrame
    summary: CleaningSummary


def clean_dataframe(dataframe: pd.DataFrame) -> CleaningResult:
    """Clean a DataFrame using transparent analytics preparation rules."""
    cleaned = dataframe.copy()
    records_processed = len(cleaned)
    actions: list[CleaningAction] = []
    modified_row_indexes: set[int] = set()

    empty_rows_removed = int(cleaned.isna().all(axis=1).sum())
    if empty_rows_removed:
        cleaned = cleaned.dropna(how="all")
        actions.append(CleaningAction("Remove empty records", empty_rows_removed))

    duplicates_removed = int(cleaned.duplicated().sum())
    if duplicates_removed:
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)
        actions.append(CleaningAction("Remove duplicates", duplicates_removed))

    text_values_standardized, text_row_indexes = _standardize_text_columns(cleaned, actions)
    invalid_numeric_rows, numeric_values_converted, numeric_row_indexes = _convert_numeric_text_columns(cleaned, actions)
    invalid_date_rows, date_values_fixed, date_row_indexes = _fix_date_columns(cleaned, actions)
    modified_row_indexes.update(text_row_indexes)
    modified_row_indexes.update(numeric_row_indexes)
    modified_row_indexes.update(date_row_indexes)

    invalid_row_indexes = invalid_numeric_rows.union(invalid_date_rows)
    invalid_records_removed = len(invalid_row_indexes)
    if invalid_records_removed:
        cleaned = cleaned.drop(index=invalid_row_indexes).reset_index(drop=True)
        actions.append(CleaningAction("Remove invalid records", invalid_records_removed))

    missing_values_filled, missing_row_indexes = _fill_missing_values(cleaned, actions)
    modified_row_indexes.update(missing_row_indexes)
    standardized_duplicates_removed = int(cleaned.duplicated().sum())
    if standardized_duplicates_removed:
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)
        duplicates_removed += standardized_duplicates_removed
        actions.append(CleaningAction("Remove duplicates after cleaning", standardized_duplicates_removed))

    anomalies_detected, anomaly_row_indexes = _detect_anomalies(cleaned, actions)

    records_removed = records_processed - len(cleaned)
    rows_modified = len(modified_row_indexes)
    formats_standardized = text_values_standardized + numeric_values_converted + date_values_fixed
    cells_modified = (
        + missing_values_filled
        + text_values_standardized
        + numeric_values_converted
        + date_values_fixed
    )
    records_fixed = rows_modified

    summary = CleaningSummary(
        records_processed=records_processed,
        records_removed=records_removed,
        records_fixed=records_fixed,
        rows_modified=rows_modified,
        cells_modified=cells_modified,
        duplicates_removed=duplicates_removed,
        missing_values_filled=missing_values_filled,
        formats_standardized=formats_standardized,
        text_values_standardized=text_values_standardized,
        numeric_values_converted=numeric_values_converted,
        date_values_fixed=date_values_fixed,
        outliers_flagged=len(anomaly_row_indexes),
        anomalies_detected=anomalies_detected,
        invalid_records_removed=invalid_records_removed,
        actions=actions,
    )
    return CleaningResult(cleaned_dataframe=cleaned, summary=summary)


def _standardize_text_columns(dataframe: pd.DataFrame, actions: list[CleaningAction]) -> tuple[int, set[int]]:
    values_changed = 0
    row_indexes: set[int] = set()
    for column in dataframe.columns:
        series = dataframe[column]
        if not (pd_types.is_object_dtype(series) or pd_types.is_string_dtype(series)):
            continue

        original = series.copy()
        standardized = series.map(lambda value: _standardize_text_value(value, str(column)))
        changed_mask = (original != standardized) & ~(original.isna() & standardized.isna())
        changed = int(changed_mask.sum())
        if changed:
            dataframe[column] = standardized
            values_changed += changed
            row_indexes.update(dataframe.index[changed_mask].tolist())
            actions.append(CleaningAction("Standardise text", changed, field=str(column)))
    return values_changed, row_indexes


def _convert_numeric_text_columns(dataframe: pd.DataFrame, actions: list[CleaningAction]) -> tuple[set[int], int, set[int]]:
    invalid_indexes: set[int] = set()
    converted_indexes: set[int] = set()
    converted_values = 0
    for column in dataframe.columns:
        series = dataframe[column]
        if not (pd_types.is_object_dtype(series) or pd_types.is_string_dtype(series)):
            continue

        non_null = series.dropna().astype(str).str.strip()
        if non_null.empty:
            continue

        parsed = pd.to_numeric(series, errors="coerce")
        parse_ratio = parsed.loc[non_null.index].notna().mean()
        if parse_ratio < NUMERIC_PARSE_THRESHOLD:
            continue

        invalid_mask = series.notna() & parsed.isna()
        converted_mask = series.notna() & parsed.notna()
        invalid_indexes.update(dataframe.index[invalid_mask].tolist())
        converted_indexes.update(dataframe.index[converted_mask].tolist())
        converted_values += int(converted_mask.sum())
        dataframe[column] = parsed
        actions.append(CleaningAction("Convert numeric text", int(converted_mask.sum()), field=str(column)))
    return invalid_indexes, converted_values, converted_indexes


def _fix_date_columns(dataframe: pd.DataFrame, actions: list[CleaningAction]) -> tuple[set[int], int, set[int]]:
    invalid_indexes: set[int] = set()
    fixed_indexes: set[int] = set()
    total_fixed = 0
    for column in dataframe.columns:
        series = dataframe[column]
        if pd_types.is_datetime64_any_dtype(series):
            continue
        if not _is_date_candidate(str(column), series):
            continue

        parsed = pd.to_datetime(series, errors="coerce", format="mixed")
        non_null_mask = series.notna() & (series.astype(str).str.strip() != "")
        valid_fixed = int((non_null_mask & parsed.notna()).sum())
        fixed_mask = non_null_mask & parsed.notna()
        invalid_mask = non_null_mask & parsed.isna()
        invalid_indexes.update(dataframe.index[invalid_mask].tolist())
        fixed_indexes.update(dataframe.index[fixed_mask].tolist())
        dataframe[column] = parsed
        if valid_fixed:
            total_fixed += valid_fixed
            actions.append(CleaningAction("Fix date formats", valid_fixed, field=str(column)))
    return invalid_indexes, total_fixed, fixed_indexes


def _fill_missing_values(dataframe: pd.DataFrame, actions: list[CleaningAction]) -> tuple[int, set[int]]:
    filled_values = 0
    row_indexes: set[int] = set()
    for column in dataframe.columns:
        missing_count = int(dataframe[column].isna().sum())
        if not missing_count:
            continue

        if pd_types.is_numeric_dtype(dataframe[column]):
            fill_value = dataframe[column].median()
        elif pd_types.is_datetime64_any_dtype(dataframe[column]):
            mode = dataframe[column].mode(dropna=True)
            fill_value = mode.iloc[0] if not mode.empty else pd.Timestamp("1970-01-01")
        else:
            fill_value = "Unknown"

        missing_mask = dataframe[column].isna()
        dataframe[column] = dataframe[column].fillna(fill_value)
        filled_values += missing_count
        row_indexes.update(dataframe.index[missing_mask].tolist())
        actions.append(CleaningAction("Handle missing values", missing_count, field=str(column)))
    return filled_values, row_indexes


def _detect_anomalies(dataframe: pd.DataFrame, actions: list[CleaningAction]) -> tuple[int, set[int]]:
    anomalies_detected = 0
    anomaly_indexes: set[int] = set()
    for column in dataframe.columns:
        series = dataframe[column]
        if not pd_types.is_numeric_dtype(series):
            continue

        values = series.dropna()
        if len(values) < OUTLIER_MIN_VALUES:
            continue

        first_quartile = values.quantile(0.25)
        third_quartile = values.quantile(0.75)
        interquartile_range = third_quartile - first_quartile
        if interquartile_range == 0:
            continue

        lower_bound = first_quartile - (1.5 * interquartile_range)
        upper_bound = third_quartile + (1.5 * interquartile_range)
        outlier_mask = (values < lower_bound) | (values > upper_bound)
        outlier_count = int(outlier_mask.sum())
        if outlier_count:
            anomalies_detected += outlier_count
            anomaly_indexes.update(values.index[outlier_mask].tolist())
            actions.append(
                CleaningAction(
                    "Detect anomalies",
                    outlier_count,
                    field=str(column),
                    details="Outliers flagged for review, not removed automatically.",
                )
            )
    return anomalies_detected, anomaly_indexes


def _standardize_text_value(value: object, field_name: str) -> object:
    if pd.isna(value):
        return value

    cleaned = sub(r"\s+", " ", str(value).strip())
    normalized = cleaned.casefold()
    if _looks_like_location_field(field_name) and normalized in LOCATION_ALIASES:
        return LOCATION_ALIASES[normalized]
    if _looks_like_identifier_field(field_name):
        return cleaned
    return cleaned.title()


def _is_date_candidate(field_name: str, series: pd.Series) -> bool:
    lower_name = field_name.lower()
    if "date" in lower_name or "time" in lower_name:
        return True
    if not (pd_types.is_object_dtype(series) or pd_types.is_string_dtype(series)):
        return False

    non_null = series.dropna().astype(str).str.strip()
    if non_null.empty:
        return False
    parsed = pd.to_datetime(non_null, errors="coerce", format="mixed")
    return bool(parsed.notna().mean() >= DATETIME_PARSE_THRESHOLD)


def _looks_like_location_field(field_name: str) -> bool:
    lower_name = field_name.lower()
    return any(token in lower_name for token in ("city", "region", "location", "market", "area"))


def _looks_like_identifier_field(field_name: str) -> bool:
    lower_name = field_name.lower()
    return any(token in lower_name for token in ("id", "code", "sku", "email", "phone"))