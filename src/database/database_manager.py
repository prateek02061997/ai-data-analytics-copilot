"""SQLite-first analytics database layer."""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd


METADATA_TABLE = "uploaded_datasets"
COLUMNS_TABLE = "uploaded_dataset_columns"


class DatabaseError(RuntimeError):
    """Raised when database operations fail."""


@dataclass(frozen=True)
class StoredDatasetMetadata:
    """Metadata for a dataset persisted to the analytics database."""

    dataset_id: str
    source_file: str
    table_name: str
    row_count: int
    column_count: int
    created_at: str
    profile_health_score: int | None
    cleaning_summary: dict[str, int | str | None]


@dataclass(frozen=True)
class QueryResult:
    """Result returned from a database query."""

    dataframe: pd.DataFrame
    row_count: int


class DatabaseManager:
    """Persist cleaned datasets and metadata to SQLite."""

    def __init__(self, database_url: str) -> None:
        self.database_url = database_url
        self.database_path = _sqlite_path_from_url(database_url)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def initialize(self) -> None:
        """Create metadata tables if they do not exist."""
        with self._connect() as connection:
            connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {METADATA_TABLE} (
                    dataset_id TEXT PRIMARY KEY,
                    source_file TEXT NOT NULL,
                    table_name TEXT NOT NULL UNIQUE,
                    row_count INTEGER NOT NULL,
                    column_count INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    profile_health_score INTEGER,
                    cleaning_summary TEXT NOT NULL
                )
                """
            )
            connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {COLUMNS_TABLE} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_id TEXT NOT NULL,
                    column_name TEXT NOT NULL,
                    detected_type TEXT NOT NULL,
                    missing_values INTEGER NOT NULL,
                    FOREIGN KEY (dataset_id) REFERENCES {METADATA_TABLE} (dataset_id)
                )
                """
            )

    def save_dataset(
        self,
        dataframe: pd.DataFrame,
        source_file: str,
        column_metadata: list[dict[str, int | str]],
        profile_health_score: int | None = None,
        cleaning_summary: dict[str, int | str | None] | None = None,
    ) -> StoredDatasetMetadata:
        """Store a DataFrame as an analytics table and save metadata."""
        if dataframe.empty:
            raise DatabaseError("Cannot store an empty dataset.")

        dataset_id = uuid4().hex
        table_name = _build_table_name(source_file, dataset_id)
        created_at = datetime.now(UTC).isoformat()
        summary = cleaning_summary or {}

        try:
            with self._connect() as connection:
                dataframe.to_sql(table_name, connection, if_exists="fail", index=False)
                connection.execute(
                    f"""
                    INSERT INTO {METADATA_TABLE} (
                        dataset_id,
                        source_file,
                        table_name,
                        row_count,
                        column_count,
                        created_at,
                        profile_health_score,
                        cleaning_summary
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        dataset_id,
                        source_file,
                        table_name,
                        len(dataframe),
                        len(dataframe.columns),
                        created_at,
                        profile_health_score,
                        json.dumps(summary, sort_keys=True),
                    ),
                )
                connection.executemany(
                    f"""
                    INSERT INTO {COLUMNS_TABLE} (dataset_id, column_name, detected_type, missing_values)
                    VALUES (?, ?, ?, ?)
                    """,
                    [
                        (
                            dataset_id,
                            str(column["column_name"]),
                            str(column["detected_type"]),
                            int(column["missing_values"]),
                        )
                        for column in column_metadata
                    ],
                )
        except sqlite3.Error as exc:
            raise DatabaseError(f"Failed to save dataset: {exc}") from exc

        return StoredDatasetMetadata(
            dataset_id=dataset_id,
            source_file=source_file,
            table_name=table_name,
            row_count=len(dataframe),
            column_count=len(dataframe.columns),
            created_at=created_at,
            profile_health_score=profile_health_score,
            cleaning_summary=summary,
        )

    def list_datasets(self) -> list[StoredDatasetMetadata]:
        """Return saved dataset metadata from newest to oldest."""
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT dataset_id, source_file, table_name, row_count, column_count,
                       created_at, profile_health_score, cleaning_summary
                FROM {METADATA_TABLE}
                ORDER BY created_at DESC
                """
            ).fetchall()

        return [_metadata_from_row(row) for row in rows]

    def get_column_metadata(self, dataset_id: str) -> pd.DataFrame:
        """Return stored column metadata for a dataset."""
        with self._connect() as connection:
            return pd.read_sql_query(
                f"""
                SELECT column_name, detected_type, missing_values
                FROM {COLUMNS_TABLE}
                WHERE dataset_id = ?
                ORDER BY id ASC
                """,
                connection,
                params=(dataset_id,),
            )

    def query_table(self, table_name: str, limit: int = 100) -> QueryResult:
        """Read rows from a stored dataset table."""
        if limit <= 0 or limit > 1000:
            raise DatabaseError("Query limit must be between 1 and 1000.")
        if not self._table_exists(table_name):
            raise DatabaseError(f"Unknown table: {table_name}")

        quoted_table = _quote_identifier(table_name)
        with self._connect() as connection:
            dataframe = pd.read_sql_query(f"SELECT * FROM {quoted_table} LIMIT ?", connection, params=(limit,))

        return QueryResult(dataframe=dataframe, row_count=len(dataframe))

    def execute_select_query(self, sql: str) -> QueryResult:
        """Execute a previously validated SELECT query."""
        try:
            with self._connect() as connection:
                dataframe = pd.read_sql_query(sql, connection)
        except sqlite3.Error as exc:
            raise DatabaseError(f"Failed to execute query: {exc}") from exc
        return QueryResult(dataframe=dataframe, row_count=len(dataframe))

    def _table_exists(self, table_name: str) -> bool:
        with self._connect() as connection:
            row = connection.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?", (table_name,)).fetchone()
        return row is not None

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.execute("PRAGMA foreign_keys = ON")
        return connection


def _sqlite_path_from_url(database_url: str) -> Path:
    if database_url.startswith("postgresql://"):
        raise DatabaseError("PostgreSQL support is planned; use a sqlite:/// DATABASE_URL for the current phase.")
    if not database_url.startswith("sqlite:///"):
        raise DatabaseError("DATABASE_URL must start with sqlite:/// for the current database layer.")
    raw_path = database_url.removeprefix("sqlite:///")
    if not raw_path:
        raise DatabaseError("SQLite database path cannot be empty.")
    return Path(raw_path)


def _build_table_name(source_file: str, dataset_id: str) -> str:
    stem = Path(source_file).stem.lower()
    sanitized_stem = re.sub(r"[^a-z0-9]+", "_", stem).strip("_") or "dataset"
    compact_stem = sanitized_stem[:40].strip("_") or "dataset"
    return f"dataset_{compact_stem}_{dataset_id[:8]}"


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _metadata_from_row(row: sqlite3.Row | tuple[object, ...]) -> StoredDatasetMetadata:
    dataset_id, source_file, table_name, row_count, column_count, created_at, health_score, cleaning_summary = row
    return StoredDatasetMetadata(
        dataset_id=str(dataset_id),
        source_file=str(source_file),
        table_name=str(table_name),
        row_count=int(row_count),
        column_count=int(column_count),
        created_at=str(created_at),
        profile_health_score=int(health_score) if health_score is not None else None,
        cleaning_summary=json.loads(str(cleaning_summary)),
    )