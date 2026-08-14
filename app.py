"""AI BI Copilot – Executive Business Intelligence Platform."""

from __future__ import annotations

from io import BytesIO
import re

import pandas as pd
import plotly.express as px
import streamlit as st

from src.ai import AIConfigurationError, AIResponseError, NaturalLanguageSQLAgent, SQLSafetyError, BusinessAnalystAgent
from src.analytics import build_cleaning_comparison, build_dashboard_report, generate_profile_report
from src.cleaning import clean_dataframe
from src.config import build_health_report
from src.config.logging_config import configure_logging, get_logger
from src.config.settings import SettingsError, get_settings
from src.data_upload import UploadValidationError, load_uploaded_dataset
from src.database import DatabaseError, DatabaseManager
from src.powerbi import build_power_bi_export, dataframe_to_excel_bytes
from src.reports import build_executive_report
from src.visualization import build_chart, build_chart_suite, convert_chart_type


configure_logging()
logger = get_logger(__name__)

MAX_AUTOMATIC_ANALYSIS_ROWS = 50_000


def main() -> None:
    """Render the initial application shell."""
    st.set_page_config(
        page_title="AI BI Copilot",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom UI styling for executive aesthetics
    st.markdown(
        """
        <style>
        .stMetric {
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            padding: 12px 16px;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        .stMetric label {
            font-size: 0.85rem !important;
            color: #6c757d !important;
            font-weight: 600 !important;
        }
        .stMetric .metric-value {
            font-size: 1.6rem !important;
            font-weight: 700 !important;
            color: #111827 !important;
        }
        .main-header {
            font-size: 2.2rem;
            font-weight: 800;
            color: #1e293b;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            font-size: 1.05rem;
            color: #64748b;
            margin-bottom: 1.5rem;
        }
        .card-box {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    try:
        settings = get_settings()
    except SettingsError as exc:
        logger.exception("Application configuration failed")
        st.error(f"Configuration error: {exc}")
        st.stop()

    ai_provider, ai_key, ai_model = _resolve_ai_provider(settings)

    st.markdown('<div class="main-header">AI BI Copilot</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Executive Data Intelligence, Auto-Cleaning, Visualizations & Grounded AI Analyst</div>', unsafe_allow_html=True)

    with st.sidebar:
        st.header("⚙️ Workspace Status")
        st.write(f"**Environment:** `{settings.app_env.title()}`")
        st.write(f"**Database:** `{settings.database_url}`")
        _render_health_status(settings)

    try:
        database = DatabaseManager(settings.database_url)
    except DatabaseError as exc:
        st.error(str(exc))
        return

    st.subheader("📁 Upload Your Dataset")
    uploader_col1, uploader_col2 = st.columns([3, 1])
    with uploader_col1:
        uploaded_file = st.file_uploader(
            "Drop your CSV or Excel (.xlsx) business file here",
            type=["csv", "xlsx"],
            accept_multiple_files=False,
            help=f"Maximum file size limit: {settings.max_upload_mb} MB",
        )
    with uploader_col2:
        st.write("")
        st.write("")
        use_sample = st.button("🚀 Load Sample Dataset", help="Test the AI Copilot instantly with sample sales data.")

    if uploaded_file is None and not use_sample and not st.session_state.get("use_sample_active", False):
        st.info("💡 **What should I do next?** Upload a dataset above or click **Load Sample Dataset** to test the Executive Dashboard, Visualizations, and AI Assistant instantly.")

        st.markdown("---")
        st.subheader("💬 AI Copilot Assistant")
        st.caption("Ask questions or get guidance on how to use AI BI Copilot.")
        landing_prompt = st.chat_input("Ask a question about AI BI Copilot...")
        if landing_prompt:
            with st.chat_message("user"):
                st.write(landing_prompt)
            with st.chat_message("assistant"):
                st.write(
                    "Welcome! I am your AI Business Analyst. Upload your dataset above or click **Load Sample Dataset** "
                    "to unlock automated data cleaning, executive dashboards, interactive charts, and grounded natural language insights."
                )
        return

    if use_sample:
        st.session_state.use_sample_active = True

    if uploaded_file is None and st.session_state.get("use_sample_active", False):
        # Generate in-memory sample dataset
        sample_df = pd.DataFrame(
            {
                "Order Date": pd.date_range(start="2026-01-01", periods=100, freq="D").strftime("%Y-%m-%d"),
                "Customer": [f"Customer {i % 10 + 1}" for i in range(100)],
                "Region": ["Auckland", "Wellington", "Christchurch", "Hamilton"][::1] * 25,
                "Product": ["Laptop", "Smartphone", "Tablet", "Monitor"][::1] * 25,
                "Units Sold": [i % 5 + 1 for i in range(100)],
                "Revenue": [(i % 5 + 1) * 250.0 + (i * 10) for i in range(100)],
                "Profit": [(i % 5 + 1) * 75.0 for i in range(100)],
            }
        )
        file_bytes = sample_df.to_csv(index=False).encode("utf-8")
        file_stream = BytesIO(file_bytes)
        file_name = "Sample_Business_Sales_Dataset.csv"
        file_size_bytes = len(file_bytes)
    else:
        file_stream = uploaded_file
        file_name = uploaded_file.name
        file_size_bytes = uploaded_file.size

    try:
        progress_bar = st.progress(0, text="Loading dataset...")
        progress_bar.progress(0.25, text="Reading dataset contents...")
        dataset = _load_uploaded_dataset_cached(
            file_name=file_name,
            file_size_bytes=file_size_bytes,
            max_upload_mb=settings.max_upload_mb,
            _file_stream=file_stream,
        )
        progress_bar.progress(1.0, text="Dataset loaded successfully.")
    except Exception as exc:
        logger.exception("Upload failed for %s: %s", file_name, exc)
        st.error(f"Unable to load file `{file_name}`: {exc}")
        return

    upload_key = f"{dataset.file_name}:{file_size_bytes}"
    _ensure_dashboard_state(upload_key)

    try:
        analysis_dataframe, analysis_is_sampled = _analysis_dataframe(dataset.dataframe)
        if analysis_is_sampled:
            st.warning(
                f"⚡ **Large Dataset Mode Active:** Automatically sampling {len(analysis_dataframe):,} representative rows "
                f"out of {len(dataset.dataframe):,} total rows to ensure ultra-fast interactive UI responsiveness."
            )

        with st.spinner("Processing automated analysis & data quality checks..."):
            profile, cleaning_result, cleaned_profile, comparison = _prepare_analysis_cached(analysis_dataframe)

        cleaning_summary = cleaning_result.summary
        dashboard_dataframe = _apply_chat_dashboard_state(cleaning_result.cleaned_dataframe)
        dashboard = build_dashboard_report(dashboard_dataframe)

        st.success(f"✅ Successfully loaded **{dataset.file_name}** ({len(dataset.dataframe):,} total rows)")
    except Exception as exc:
        logger.exception("Analysis pipeline failed for %s: %s", file_name, exc)
        st.error(f"Error processing dataset analysis: {exc}")
        return

    # ----------------------------------------------------
    # Simplified Navigation Tabs (Business-Friendly Labels)
    # ----------------------------------------------------
    tabs = st.tabs([
        "📊 Executive Dashboard",
        "🔍 Understand Your Data",
        "🛠️ Improve Data Quality",
        "📈 Interactive Charts",
        "📋 Excel Data Explorer",
        "💬 AI Analyst Chat",
        "📄 Export & Reports",
    ])

    # ----------------------------------------------------
    # TAB 1: EXECUTIVE DASHBOARD
    # ----------------------------------------------------
    with tabs[0]:
        st.caption(f"Showing {len(dashboard_dataframe):,} rows | Dataset Domain: **{dashboard.dataset_type}**")

        active_filters = _dashboard_state_summary()
        if active_filters:
            st.info(f"🎯 {active_filters}")

        # High-Impact Executive KPI Cards
        kpi_cols = st.columns(6)
        kpi_cols[0].metric("Total Records", f"{len(dashboard_dataframe):,}")
        kpi_cols[1].metric("Total Fields", f"{len(dashboard_dataframe.columns):,}")
        kpi_cols[2].metric("Data Quality Score", f"{profile.health_score}%")
        kpi_cols[3].metric("Completion Rate", f"{profile.completion_percentage}%")
        kpi_cols[4].metric("Missing Values", f"{profile.missing_values:,}")
        kpi_cols[5].metric("Duplicates", f"{profile.duplicate_records:,}")

        st.markdown("---")
        _render_dashboard_kpis(dashboard.dynamic_kpis)

        requested_chart = _build_requested_chart(dashboard_dataframe)
        if requested_chart is not None:
            st.plotly_chart(requested_chart, use_container_width=True)

        dash_subtabs = st.tabs(["📈 Sales & Performance Trend", "🌍 Regional & Location Analysis", "📦 Category & Product Breakdown"])
        with dash_subtabs[0]:
            if dashboard.figures.sales_trend:
                st.plotly_chart(dashboard.figures.sales_trend, use_container_width=True)
            else:
                st.info("A trend line chart will automatically appear when a date field and numeric measure exist in your data.")
        with dash_subtabs[1]:
            if dashboard.figures.regional_analysis:
                st.plotly_chart(dashboard.figures.regional_analysis, use_container_width=True)
            else:
                st.info("A location chart will automatically appear when a region/city field and numeric measure exist in your data.")
        with dash_subtabs[2]:
            if dashboard.figures.product_analysis:
                st.plotly_chart(dashboard.figures.product_analysis, use_container_width=True)
            else:
                st.info("A category chart will automatically appear when a category field and numeric measure exist in your data.")

        st.markdown("---")
        with st.expander("💬 Ask AI Analyst (Interactive Copilot & Dashboard Controller)", expanded=True):
            st.caption("Ask questions or control dashboard filters directly using natural language.")

            for msg in st.session_state.ai_analyst_messages:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])

            dash_prompt = st.text_input("Type your question or dashboard command:", key="dash_quick_chat_input")
            if st.button("Ask AI Analyst", key="dash_quick_chat_btn") and dash_prompt:
                st.session_state.ai_analyst_messages.append({"role": "user", "content": dash_prompt})
                with st.status("Analyzing dataset & processing query...", expanded=False):
                    response, dashboard_changed = _handle_dashboard_chat_command(
                        prompt=dash_prompt,
                        base_dataframe=cleaning_result.cleaned_dataframe,
                        current_dataframe=dashboard_dataframe,
                        profile_report=profile,
                        cleaning_summary=cleaning_summary,
                        ai_provider=ai_provider,
                        ai_key=ai_key,
                        ai_model=ai_model,
                    )
                st.session_state.ai_analyst_messages.append({"role": "assistant", "content": response})
                st.rerun()

    # ----------------------------------------------------
    # TAB 2: UNDERSTAND YOUR DATA (AUTOMATIC STATISTICAL ANALYSIS)
    # ----------------------------------------------------
    with tabs[1]:
        st.subheader("🔍 Understand Your Data (Automated Analysis)")
        st.caption("Comprehensive data breakdown, numerical distributions, categorical frequencies, and time trends.")

        st.markdown("#### 📌 Overview & File Metadata")
        meta_cols = st.columns(4)
        meta_cols[0].metric("File Name", dataset.file_name)
        meta_cols[1].metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")
        meta_cols[2].metric("Quality Score", f"{profile.health_score}%")
        meta_cols[3].metric("Data Completion", f"{profile.completion_percentage}%")

        st.markdown("---")
        st.markdown("#### 🔢 Numerical Breakdown (Averages, Spreads & Quantiles)")
        if profile.numerical_stats:
            num_df = pd.DataFrame(
                [
                    {
                        "Field": stat.field,
                        "Count": f"{stat.count:,}",
                        "Mean (Average)": stat.mean,
                        "Median": stat.median,
                        "Mode": stat.mode,
                        "Std Dev": stat.std,
                        "Minimum": stat.min_val,
                        "Maximum": stat.max_val,
                        "Q1 (25%)": stat.q1,
                        "Q3 (75%)": stat.q3,
                        "IQR": stat.iqr,
                    }
                    for stat in profile.numerical_stats
                ]
            )
            st.dataframe(num_df, use_container_width=True, hide_index=True)
        else:
            st.info("No numeric fields were detected in this dataset.")

        st.markdown("---")
        st.markdown("#### 🏷️ Category Breakdown (Top Values & Frequencies)")
        if profile.category_stats:
            for cat_stat in profile.category_stats[:4]:
                with st.expander(f"Field: **{cat_stat.field}** ({cat_stat.unique_count:,} unique values)", expanded=True):
                    cat_df = pd.DataFrame(
                        [
                            {
                                "Category Label": val.label,
                                "Count": f"{val.count:,}",
                                "Share (%)": f"{val.percentage}%",
                            }
                            for val in cat_stat.top_values
                        ]
                    )
                    st.dataframe(cat_df, use_container_width=True, hide_index=True)
        else:
            st.info("No categorical text fields were detected.")

        if profile.time_analysis:
            st.markdown("---")
            st.markdown("#### 📅 Automated Time-Series Analysis")
            for t_item in profile.time_analysis:
                st.write(f"**Date Field:** `{t_item.field}`")
                t_cols = st.columns(3)
                with t_cols[0]:
                    st.caption("Monthly Trend (Recent 24 months)")
                    m_df = pd.DataFrame([{"Month": p.period, "Records": p.count} for p in t_item.monthly_trend])
                    st.dataframe(m_df, use_container_width=True, hide_index=True)
                with t_cols[1]:
                    st.caption("Quarterly Trend")
                    q_df = pd.DataFrame([{"Quarter": p.period, "Records": p.count} for p in t_item.quarterly_trend])
                    st.dataframe(q_df, use_container_width=True, hide_index=True)
                with t_cols[2]:
                    st.caption("Weekday Breakdown")
                    w_df = pd.DataFrame([{"Day": p.period, "Records": p.count} for p in t_item.weekday_trend])
                    st.dataframe(w_df, use_container_width=True, hide_index=True)

    # ----------------------------------------------------
    # TAB 3: IMPROVE DATA QUALITY (DATA CLEANING AGENT)
    # ----------------------------------------------------
    with tabs[2]:
        st.subheader("🛠️ Improve Data Quality (Auto-Cleaned Results)")
        st.caption("Deterministic cleaning rules standardise text, fill missing values, drop duplicates, and flag outliers.")

        clean_cols = st.columns(4)
        clean_cols[0].metric("Rows Processed", f"{cleaning_summary.records_processed:,}")
        clean_cols[1].metric("Rows Modified", f"{cleaning_summary.rows_modified:,}")
        clean_cols[2].metric("Cells Modified", f"{cleaning_summary.cells_modified:,}")
        clean_cols[3].metric("Rows Removed", f"{cleaning_summary.records_removed:,}")

        clean_detail = st.columns(4)
        clean_detail[0].metric("Missing Values Filled", f"{cleaning_summary.missing_values_filled:,}")
        clean_detail[1].metric("Duplicates Removed", f"{cleaning_summary.duplicates_removed:,}")
        clean_detail[2].metric("Outliers Flagged", f"{cleaning_summary.outliers_flagged:,}")
        clean_detail[3].metric("Formats Standardized", f"{cleaning_summary.formats_standardized:,}")

        _render_cleaning_comparison(comparison)

    # ----------------------------------------------------
    # TAB 4: INTERACTIVE CHARTS & CHART TYPE SWITCHER
    # ----------------------------------------------------
    with tabs[3]:
        st.subheader("📈 Interactive Chart Suite & Visual Switcher")
        st.caption("Explore automatically generated charts and dynamically convert chart rendering types.")

        _render_chart_suite(dashboard_dataframe, key_prefix="suite_view")

    # ----------------------------------------------------
    # TAB 5: EXCEL DATA EXPLORER
    # ----------------------------------------------------
    with tabs[4]:
        st.subheader("📋 Excel Data Explorer")
        st.caption("Search, filter, sort, and export full dataset records directly to CSV or Excel.")

        explorer_dataframe = _render_data_explorer(dashboard_dataframe, key_prefix="excel_explorer")

    # ----------------------------------------------------
    # TAB 6: AI ANALYST CHAT (GROUNDED & CONTROL CENTER)
    # ----------------------------------------------------
    with tabs[5]:
        st.subheader("💬 AI Analyst Chat & Dashboard Control Center")
        ai_provider, ai_key, ai_model = _resolve_ai_provider(settings)
        chat_prompt = _render_ai_chat_shell(dashboard.dataset_type)

        if chat_prompt:
            st.session_state.ai_analyst_messages.append({"role": "user", "content": chat_prompt})
            with st.status("Analyzing dataset & processing query...", expanded=False):
                response, dashboard_changed = _handle_dashboard_chat_command(
                    prompt=chat_prompt,
                    base_dataframe=cleaning_result.cleaned_dataframe,
                    current_dataframe=dashboard_dataframe,
                    profile_report=profile,
                    cleaning_summary=cleaning_summary,
                    ai_provider=ai_provider,
                    ai_key=ai_key,
                    ai_model=ai_model,
                )
            st.session_state.ai_analyst_messages.append({"role": "assistant", "content": response})
            if dashboard_changed:
                st.rerun()

    # ----------------------------------------------------
    # TAB 7: EXPORT & REPORTS
    # ----------------------------------------------------
    with tabs[6]:
        st.subheader("📄 Export & Reports")
        st.caption("Download executive reports, Power BI templates, and database tables.")

        _render_executive_report(
            dataframe=cleaning_result.cleaned_dataframe,
            dataset_name=dataset.file_name,
            profile_report=profile,
            key_prefix="export_tab",
            cleaning_summary=cleaning_summary,
        )

        st.markdown("---")
        _render_power_bi_export(explorer_dataframe, dataset.file_name, key_prefix="export_tab")

        st.markdown("---")
        st.subheader("🗄️ Analytics Database Storage")
        if analysis_is_sampled:
            st.info("Saving is disabled in large dataset sample mode to avoid storing a sampled dataset as full.")
        if st.button("Save Cleaned Dataset to SQLite Database", type="primary", disabled=analysis_is_sampled):
            try:
                stored_dataset = database.save_dataset(
                    dataframe=cleaning_result.cleaned_dataframe,
                    source_file=dataset.file_name,
                    column_metadata=[
                        {
                            "column_name": field.name,
                            "detected_type": field.detected_type,
                            "missing_values": field.missing_values,
                        }
                        for field in summary.fields
                    ],
                    profile_health_score=profile.health_score,
                    cleaning_summary={
                        "records_processed": cleaning_summary.records_processed,
                        "rows_modified": cleaning_summary.rows_modified,
                        "cells_modified": cleaning_summary.cells_modified,
                        "records_fixed": cleaning_summary.records_fixed,
                        "records_removed": cleaning_summary.records_removed,
                        "duplicates_removed": cleaning_summary.duplicates_removed,
                        "missing_values_filled": cleaning_summary.missing_values_filled,
                        "formats_standardized": cleaning_summary.formats_standardized,
                        "outliers_flagged": cleaning_summary.outliers_flagged,
                        "invalid_records_removed": cleaning_summary.invalid_records_removed,
                        "anomalies_detected": cleaning_summary.anomalies_detected,
                    },
                )
            except DatabaseError as exc:
                logger.exception("Failed to save dataset")
                st.error(str(exc))
            else:
                st.success(f"Saved cleaned dataset to table `{stored_dataset.table_name}`")


# ----------------------------------------------------
# Helper Rendering & Processing Functions
# ----------------------------------------------------

def _resolve_ai_provider(settings) -> tuple[str, str, str]:
    if settings.ai_provider == "gemini" or (settings.ai_provider == "auto" and settings.gemini_api_key):
        return "gemini", settings.gemini_api_key, settings.gemini_model
    if settings.ai_provider == "claude" or (settings.ai_provider == "auto" and settings.claude_api_key):
        return "claude", settings.claude_api_key, settings.claude_model
    return settings.ai_provider, "", ""


@st.cache_resource(show_spinner=False)
def _load_uploaded_dataset_cached(file_name: str, file_size_bytes: int, max_upload_mb: int, _file_stream: Any):
    return load_uploaded_dataset(
        file=_file_stream,
        file_name=file_name,
        file_size_bytes=file_size_bytes,
        max_upload_mb=max_upload_mb,
    )


@st.cache_resource(show_spinner=False)
def _prepare_analysis_cached(dataframe: pd.DataFrame):
    profile = generate_profile_report(dataframe)
    cleaning_result = clean_dataframe(dataframe)
    cleaned_profile = generate_profile_report(cleaning_result.cleaned_dataframe)
    comparison = build_cleaning_comparison(profile, cleaned_profile, cleaning_result.summary)
    return profile, cleaning_result, cleaned_profile, comparison


def _analysis_dataframe(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    if len(dataframe) <= MAX_AUTOMATIC_ANALYSIS_ROWS:
        return dataframe, False
    return dataframe.sample(n=MAX_AUTOMATIC_ANALYSIS_ROWS, random_state=42).reset_index(drop=True), True


def _ensure_dashboard_state(upload_key: str) -> None:
    if st.session_state.get("active_upload_key") != upload_key:
        st.session_state.active_upload_key = upload_key
        st.session_state.dashboard_filters = []
        st.session_state.show_missing_only = False
        st.session_state.drop_missing_rows = False
        st.session_state.show_outliers_only = False
        st.session_state.requested_chart = None
        st.session_state.ai_analyst_messages = []


def _apply_chat_dashboard_state(dataframe: pd.DataFrame) -> pd.DataFrame:
    filtered = dataframe
    if st.session_state.get("drop_missing_rows", False):
        filtered = filtered.dropna()
    if st.session_state.get("show_missing_only", False):
        filtered = filtered[filtered.isna().any(axis=1)]
    if st.session_state.get("show_outliers_only", False):
        outlier_mask = _outlier_mask(filtered)
        filtered = filtered[outlier_mask]
    for dashboard_filter in st.session_state.get("dashboard_filters", []):
        column = dashboard_filter["column"]
        operator = dashboard_filter["operator"]
        value = dashboard_filter["value"]
        if column not in filtered.columns:
            continue
        series = filtered[column]
        if operator == "contains":
            filtered = filtered[series.astype(str).str.contains(str(value), case=False, na=False)]
        elif operator == "in":
            selected_values = {str(item).casefold() for item in value}
            filtered = filtered[series.astype(str).str.casefold().isin(selected_values)]
        elif operator in {">", ">=", "<", "<=", "="}:
            numeric_series = pd.to_numeric(series, errors="coerce")
            numeric_value = float(value)
            if operator == ">":
                filtered = filtered[numeric_series > numeric_value]
            elif operator == ">=":
                filtered = filtered[numeric_series >= numeric_value]
            elif operator == "<":
                filtered = filtered[numeric_series < numeric_value]
            elif operator == "<=":
                filtered = filtered[numeric_series <= numeric_value]
            else:
                filtered = filtered[numeric_series == numeric_value]
    return filtered


def _render_ai_chat_shell(dataset_type: str) -> str | None:
    st.caption(
        "Type questions or natural language commands like `show Toyota only`, `filter price above 20000`, `compare petrol and diesel`, `create price vs mileage chart`, or `reset filters`."
    )
    controls = st.columns([1, 1, 4])
    if controls[0].button("Clear Chat", key="clear_ai_conversation"):
        st.session_state.ai_analyst_messages = []
        st.session_state.chat_suggestion = None
        st.rerun()
    if controls[1].button("Reset Filters", key="chat_reset_dashboard"):
        st.session_state.dashboard_filters = []
        st.session_state.show_missing_only = False
        st.session_state.drop_missing_rows = False
        st.session_state.show_outliers_only = False
        st.session_state.requested_chart = None
        st.session_state.ai_analyst_messages.append({"role": "assistant", "content": "Dashboard reset to original cleaned dataset."})
        st.rerun()

    suggestions = _suggested_questions(dataset_type)
    suggestion_columns = st.columns(len(suggestions))
    for index, suggestion in enumerate(suggestions):
        if suggestion_columns[index].button(suggestion, key=f"suggestion_{index}"):
            st.session_state.chat_suggestion = suggestion
            st.rerun()

    for message in st.session_state.ai_analyst_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    pending_suggestion = st.session_state.pop("chat_suggestion", None)
    chat_prompt = st.chat_input("Ask a question or command the dashboard...")
    return pending_suggestion or chat_prompt


def _handle_dashboard_chat_command(
    prompt: str,
    base_dataframe: pd.DataFrame,
    current_dataframe: pd.DataFrame,
    profile_report,
    cleaning_summary,
    ai_provider: str,
    ai_key: str,
    ai_model: str,
) -> tuple[str, bool]:
    command_response = _apply_dashboard_command(prompt, base_dataframe, current_dataframe)
    if command_response is not None:
        return command_response, True

    if not ai_key:
        return "I couldn't find that information in the uploaded dataset.", False

    try:
        analyst = BusinessAnalystAgent(api_key=ai_key, model=ai_model, provider=ai_provider)
        analysis = analyst.answer_question(
            question=prompt,
            dataframe=current_dataframe,
            profile_report=profile_report,
            cleaning_summary=cleaning_summary,
            business_context="Use only the currently uploaded dataset and active filter state.",
            history=st.session_state.ai_analyst_messages,
        )
    except (AIConfigurationError, AIResponseError, ValueError) as exc:
        logger.warning("AI analyst request failed: %s", exc)
        return "I couldn't find that information in the uploaded dataset.", False

    return _format_ai_analysis(analysis.summary, analysis.key_findings, analysis.business_recommendations), False


def _apply_dashboard_command(prompt: str, base_dataframe: pd.DataFrame, current_dataframe: pd.DataFrame) -> str | None:
    normalized = prompt.casefold().strip()
    if normalized in {"reset", "reset dashboard", "clear filters", "show all"}:
        st.session_state.dashboard_filters = []
        st.session_state.show_missing_only = False
        st.session_state.drop_missing_rows = False
        st.session_state.show_outliers_only = False
        st.session_state.requested_chart = None
        return "Dashboard reset to the uploaded dataset."

    if "missing" in normalized and any(word in normalized for word in ["remove", "drop", "exclude"]):
        st.session_state.drop_missing_rows = True
        st.session_state.show_missing_only = False
        return "Applied filter: rows with missing values are excluded from view."

    if "missing" in normalized:
        st.session_state.show_missing_only = True
        return "Dashboard updated to show rows with missing values."

    if "outlier" in normalized:
        st.session_state.show_outliers_only = True
        return "Dashboard updated to show rows with potential numeric outliers."

    requested_chart = _parse_chart_request(normalized, base_dataframe)
    if requested_chart:
        st.session_state.requested_chart = requested_chart
        return f"Created chart request: {requested_chart['y']} vs {requested_chart['x']}. Dashboard updated."

    comparison = _parse_comparison_request(normalized, current_dataframe)
    if comparison:
        column, left_value, right_value = comparison
        st.session_state.dashboard_filters.append({"column": column, "operator": "in", "value": [left_value, right_value]})
        filtered = current_dataframe[current_dataframe[column].astype(str).str.casefold().isin([left_value, right_value])]
        counts = filtered[column].astype(str).value_counts()
        lines = [f"Comparison for {column}:"]
        lines.extend(f"- {index}: {count:,} row(s)" for index, count in counts.items())
        return "\n".join(lines)

    numeric_filter = _parse_numeric_filter(normalized, base_dataframe)
    if numeric_filter:
        st.session_state.dashboard_filters.append(numeric_filter)
        return f"Dashboard filtered: {numeric_filter['column']} {numeric_filter['operator']} {numeric_filter['value']}."

    value_filter = _parse_value_filter(normalized, base_dataframe)
    if value_filter:
        st.session_state.dashboard_filters.append(value_filter)
        return f"Dashboard filtered to rows where {value_filter['column']} contains `{value_filter['value']}`."

    top_category = _parse_top_category(normalized, current_dataframe)
    if top_category:
        column, limit = top_category
        counts = current_dataframe[column].astype(str).value_counts().head(limit)
        lines = [f"Top {limit} values in {column}:"]
        lines.extend(f"- {index}: {count:,}" for index, count in counts.items())
        return "\n".join(lines)

    return None


def _parse_numeric_filter(prompt: str, dataframe: pd.DataFrame) -> dict[str, object] | None:
    natural_match = re.search(r"(?:filter|show)?\s*([a-z0-9 _-]+?)\s+(?:above|over|greater than|more than|after)\s+([0-9]+(?:\.[0-9]+)?)", prompt)
    if natural_match:
        field_text, value = natural_match.groups()
        column = _find_prompt_column(field_text, dataframe, numeric_only=True)
        if column:
            return {"column": column, "operator": ">", "value": float(value)}
    below_match = re.search(r"(?:filter|show)?\s*([a-z0-9 _-]+?)\s+(?:below|under|less than|before)\s+([0-9]+(?:\.[0-9]+)?)", prompt)
    if below_match:
        field_text, value = below_match.groups()
        column = _find_prompt_column(field_text, dataframe, numeric_only=True)
        if column:
            return {"column": column, "operator": "<", "value": float(value)}
    match = re.search(r"([a-z0-9 _-]+?)\s*(>=|<=|>|<|=)\s*([0-9]+(?:\.[0-9]+)?)", prompt)
    if not match:
        return None
    field_text, operator, value = match.groups()
    column = _find_prompt_column(field_text, dataframe, numeric_only=True)
    if not column:
        return None
    return {"column": column, "operator": operator, "value": float(value)}


def _parse_chart_request(prompt: str, dataframe: pd.DataFrame) -> dict[str, str] | None:
    match = re.search(r"(?:create|show|make|build).*?([a-z0-9 _-]+?)\s+vs\s+([a-z0-9 _-]+)", prompt)
    if not match:
        return None
    left_text, right_text = match.groups()
    x_column = _find_prompt_column(right_text, dataframe, numeric_only=False)
    y_column = _find_prompt_column(left_text, dataframe, numeric_only=False)
    if not x_column or not y_column:
        return None
    return {"x": x_column, "y": y_column}


def _parse_comparison_request(prompt: str, dataframe: pd.DataFrame) -> tuple[str, str, str] | None:
    match = re.search(r"compare\s+(.+?)\s+(?:and|vs|versus)\s+(.+)", prompt)
    if not match:
        return None
    left_text, right_text = match.groups()
    for column in dataframe.columns:
        series = dataframe[column]
        if not (series.dtype == "object" or pd.api.types.is_string_dtype(series)):
            continue
        values = {str(value).casefold(): str(value) for value in series.dropna().astype(str).unique().tolist()[:500]}
        left_match = next((value for value in values if value in left_text), None)
        right_match = next((value for value in values if value in right_text), None)
        if left_match and right_match:
            return str(column), left_match, right_match
    return None


def _parse_value_filter(prompt: str, dataframe: pd.DataFrame) -> dict[str, object] | None:
    for column in dataframe.columns:
        series = dataframe[column]
        if not (series.dtype == "object" or pd.api.types.is_string_dtype(series)):
            continue
        values = series.dropna().astype(str).unique().tolist()
        for value in values[:500]:
            if str(value).casefold() in prompt:
                return {"column": column, "operator": "contains", "value": str(value)}
    return None


def _parse_top_category(prompt: str, dataframe: pd.DataFrame) -> tuple[str, int] | None:
    match = re.search(r"top\s+([0-9]+)\s+([a-z0-9 _-]+)", prompt)
    if not match:
        return None
    limit_text, field_text = match.groups()
    column = _find_prompt_column(field_text, dataframe, numeric_only=False)
    if not column:
        return None
    return column, min(max(int(limit_text), 1), 25)


def _find_prompt_column(text: str, dataframe: pd.DataFrame, numeric_only: bool) -> str | None:
    normalized_text = text.casefold().strip()
    candidates = [column for column in dataframe.columns if not numeric_only or pd.api.types.is_numeric_dtype(dataframe[column])]
    for column in candidates:
        normalized_column = str(column).casefold()
        singular_column = normalized_column.rstrip("s")
        if normalized_column in normalized_text or singular_column in normalized_text:
            return column
    return None


def _dashboard_state_summary() -> str:
    filters = st.session_state.get("dashboard_filters", [])
    parts = []
    if st.session_state.get("show_missing_only", False):
        parts.append("showing missing-value rows")
    if st.session_state.get("drop_missing_rows", False):
        parts.append("missing rows excluded")
    if st.session_state.get("show_outliers_only", False):
        parts.append("showing potential outlier rows")
    parts.extend(f"{item['column']} {item['operator']} {item['value']}" for item in filters)
    if not parts:
        return ""
    return "Active Dashboard Filters: " + "; ".join(parts)


def _build_requested_chart(dataframe: pd.DataFrame):
    requested_chart = st.session_state.get("requested_chart")
    if not requested_chart:
        return None
    x_column = requested_chart.get("x")
    y_column = requested_chart.get("y")
    if x_column not in dataframe.columns or y_column not in dataframe.columns:
        return None
    chart_frame = dataframe[[x_column, y_column]].dropna()
    if chart_frame.empty:
        return None
    if pd.api.types.is_numeric_dtype(chart_frame[x_column]) and pd.api.types.is_numeric_dtype(chart_frame[y_column]):
        figure = px.scatter(chart_frame, x=x_column, y=y_column, title=f"{y_column} vs {x_column}")
    elif pd.api.types.is_numeric_dtype(chart_frame[y_column]):
        grouped = chart_frame.groupby(x_column, dropna=False, as_index=False)[y_column].mean().head(30)
        figure = px.bar(grouped, x=x_column, y=y_column, title=f"Average {y_column} by {x_column}")
    else:
        counts = chart_frame[x_column].astype(str).value_counts().head(30).reset_index()
        counts.columns = [x_column, "Rows"]
        figure = px.bar(counts, x=x_column, y="Rows", title=f"Rows by {x_column}")
    figure.update_layout(template="plotly_white", margin={"l": 24, "r": 24, "t": 56, "b": 24})
    return figure


def _outlier_mask(dataframe: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=dataframe.index)
    numeric_columns = [column for column in dataframe.columns if pd.api.types.is_numeric_dtype(dataframe[column])]
    for column in numeric_columns:
        values = dataframe[column].dropna()
        if len(values) < 4:
            continue
        first_quartile = values.quantile(0.25)
        third_quartile = values.quantile(0.75)
        interquartile_range = third_quartile - first_quartile
        if interquartile_range == 0:
            continue
        lower_bound = first_quartile - (1.5 * interquartile_range)
        upper_bound = third_quartile + (1.5 * interquartile_range)
        mask.loc[values.index] = mask.loc[values.index] | (values < lower_bound) | (values > upper_bound)
    return mask


def _suggested_questions(dataset_type: str) -> list[str]:
    if dataset_type == "Healthcare":
        return ["Show missing values", "Top 10 diagnoses", "Show outliers"]
    if dataset_type == "Sales":
        return ["Top 10 customers", "Show missing values", "Reset filters"]
    if dataset_type == "Inventory":
        return ["Top 10 suppliers", "Show outliers", "Reset filters"]
    if dataset_type == "Vehicle Listings":
        return ["Show Toyota only", "Filter year after 2020", "Create price vs mileage chart"]
    return ["Show missing values", "Show outliers", "Reset filters"]


def _render_health_status(settings) -> None:
    report = build_health_report(settings)
    st.subheader("Deployment Readiness")
    status_label = "🟢 Ready" if report.status == "ready" else "🟡 Action Needed"
    st.write(status_label)
    for check in report.checks:
        if check.status == "ok":
            st.success(f"{check.name}: {check.message}")
        elif check.status == "warning":
            st.warning(f"{check.name}: {check.message}")
        else:
            st.error(f"{check.name}: {check.message}")


def _render_dashboard_kpis(kpis) -> None:
    if not kpis:
        st.info("No primary business KPIs detected for this dataset shape.")
        return
    columns = st.columns(min(len(kpis), 4))
    for index, kpi in enumerate(kpis):
        columns[index % len(columns)].metric(kpi.label, kpi.value, help=kpi.detail)


def _render_cleaning_comparison(comparison) -> None:
    st.subheader("Data Cleaning Comparison")
    score_columns = st.columns(3)
    score_columns[0].metric("Original Quality Score", f"{comparison.health_score_before}%")
    score_columns[1].metric("Cleaned Quality Score", f"{comparison.health_score_after}%", delta=f"{comparison.health_score_improvement:+d}")
    score_columns[2].metric("Quality Gap Closed", f"{comparison.percentage_improvement:.1f}%")
    st.dataframe(
        [
            {
                "Metric": item.metric,
                "Before Cleaning": item.before,
                "After Cleaning": item.after,
                "Change": item.change,
            }
            for item in comparison.metrics
        ],
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        "Download Quality Comparison Report (.md)",
        data=comparison.markdown,
        file_name="cleaning_comparison_report.md",
        mime="text/markdown",
        key="cleaning_comparison_report_download",
    )


def _render_chart_suite(dataframe, key_prefix: str) -> None:
    charts = build_chart_suite(dataframe)
    if not charts:
        st.info("No interactive charts could be generated for this dataset shape.")
        return

    numeric_columns = [str(c) for c in dataframe.columns if pd.api.types.is_numeric_dtype(dataframe[c])]
    categorical_columns = [str(c) for c in dataframe.columns if not pd.api.types.is_numeric_dtype(dataframe[c])]

    st.markdown("#### 🔄 Custom Chart Builder & Converter")
    conv_cols = st.columns(4)
    x_var = conv_cols[0].selectbox("X Axis (Field)", dataframe.columns, key=f"{key_prefix}_x_var")
    y_var = conv_cols[1].selectbox("Y Axis (Measure)", [None] + list(numeric_columns), key=f"{key_prefix}_y_var")
    chart_type_sel = conv_cols[2].selectbox("Chart Rendering Type", ["bar", "line", "area", "pie", "scatter", "histogram", "box", "treemap"], key=f"{key_prefix}_target_type")
    
    custom_fig = convert_chart_type(dataframe, x_col=x_var, y_col=y_var, target_type=chart_type_sel)
    st.plotly_chart(custom_fig, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 📊 Auto-Generated Chart Suite")
    tabs = st.tabs([chart.title[:35] for chart in charts])
    for index, chart in enumerate(charts):
        with tabs[index]:
            st.caption(chart.reason)
            st.plotly_chart(chart.figure, use_container_width=True)
            st.download_button(
                "Download Interactive HTML Chart",
                data=chart.html,
                file_name=f"{key_prefix}_{chart.chart_type}_{index + 1}.html",
                mime="text/html",
                key=f"{key_prefix}_{chart.chart_type}_{index}_download",
            )


def _render_data_explorer(dataframe, key_prefix: str):
    filtered = dataframe.copy()

    # Top search bar
    search_text = st.text_input("🔍 Global Quick Search (searches all columns)", key=f"{key_prefix}_search")
    if search_text:
        text_columns = [column for column in filtered.columns]
        search_mask = filtered[text_columns].astype(str).apply(
            lambda column: column.str.contains(search_text, case=False, na=False)
        ).any(axis=1)
        filtered = filtered[search_mask]

    categorical_columns = [
        column
        for column in dataframe.columns
        if (dataframe[column].dtype == "object" or pd.api.types.is_string_dtype(dataframe[column])) and 1 < dataframe[column].nunique(dropna=True) <= 30
    ][:4]

    if categorical_columns:
        st.write("Column Filters:")
        filter_columns = st.columns(len(categorical_columns))
        for index, column in enumerate(categorical_columns):
            options = sorted(dataframe[column].dropna().astype(str).unique().tolist())
            selected = filter_columns[index].multiselect(str(column), options, key=f"{key_prefix}_{column}_filter")
            if selected:
                filtered = filtered[filtered[column].astype(str).isin(selected)]

    sort_cols = st.columns([2, 1])
    sort_column = sort_cols[0].selectbox("Sort Data By", dataframe.columns, key=f"{key_prefix}_sort_col")
    sort_ascending = sort_cols[1].checkbox("Ascending Sort", value=True, key=f"{key_prefix}_sort_asc")

    if sort_column in filtered.columns:
        filtered = filtered.sort_values(by=sort_column, ascending=sort_ascending)

    st.caption(f"Displaying **{len(filtered):,}** of **{len(dataframe):,}** total cleaned records")
    st.dataframe(filtered, use_container_width=True)

    # Multi-Format Downloads
    dl_cols = st.columns(2)
    dl_cols[0].download_button(
        "📥 Export Cleaned Dataset (CSV)",
        data=filtered.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"{key_prefix}_cleaned_data.csv",
        mime="text/csv",
        key=f"{key_prefix}_cleaned_csv_download",
    )
    
    excel_bytes = dataframe_to_excel_bytes(filtered)
    dl_cols[1].download_button(
        "📥 Export Cleaned Dataset (Excel .xlsx)",
        data=excel_bytes,
        file_name=f"{key_prefix}_cleaned_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key=f"{key_prefix}_cleaned_excel_download",
    )
    return filtered


def _render_executive_report(dataframe, dataset_name: str, profile_report, key_prefix: str, cleaning_summary=None) -> None:
    st.subheader("📄 Executive Summary Report")
    try:
        report = build_executive_report(
            dataframe=dataframe,
            dataset_name=dataset_name,
            profile_report=profile_report,
            cleaning_summary=cleaning_summary,
        )
    except ValueError as exc:
        st.info(str(exc))
        return

    report_columns = st.columns(2)
    report_columns[0].download_button(
        "📥 Download Executive Report (HTML)",
        data=report.html,
        file_name=report.html_file_name,
        mime="text/html",
        key=f"{key_prefix}_report_html",
    )
    report_columns[1].download_button(
        "📥 Download Executive Report (Markdown)",
        data=report.markdown,
        file_name=report.markdown_file_name,
        mime="text/markdown",
        key=f"{key_prefix}_report_markdown",
    )


def _render_power_bi_export(dataframe, table_name: str, key_prefix: str) -> None:
    st.subheader("🟡 Power BI Integration Bundle")
    try:
        export = build_power_bi_export(dataframe, table_name=table_name)
    except ValueError as exc:
        st.info(str(exc))
        return

    download_columns = st.columns(3)
    download_columns[0].download_button(
        "Download Power BI CSV",
        data=export.csv_bytes,
        file_name=export.csv_file_name,
        mime="text/csv",
        key=f"{key_prefix}_powerbi_csv",
    )
    download_columns[1].download_button(
        "Download Schema (JSON)",
        data=export.schema_json,
        file_name=export.schema_file_name,
        mime="application/json",
        key=f"{key_prefix}_powerbi_schema",
    )
    download_columns[2].download_button(
        "Download Power Query Script (.m)",
        data=export.power_query_m,
        file_name=export.power_query_file_name,
        mime="text/plain",
        key=f"{key_prefix}_powerbi_m",
    )


def _format_ai_analysis(summary: str, key_findings: list[str], business_recommendations: list[str]) -> str:
    findings = "\n".join(f"{index}. {finding}" for index, finding in enumerate(key_findings, start=1))
    recommendations = "\n".join(
        f"{index}. {recommendation}" for index, recommendation in enumerate(business_recommendations, start=1)
    )
    return (
        f"**Summary**\n\n{summary}\n\n"
        f"**Key Findings**\n\n{findings}\n\n"
        f"**Business Recommendations**\n\n{recommendations}"
    )


if __name__ == "__main__":
    main()