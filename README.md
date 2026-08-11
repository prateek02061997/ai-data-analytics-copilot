# AI BI Copilot

AI BI Copilot is a business intelligence platform that will help users upload business data, profile and clean datasets, ask natural language questions, generate visualisations, prepare analytics storage, integrate with Power BI, and create executive reports.

This repository is being built in controlled phases. The current implementation contains **Phase 12: Operational Readiness**.

## Current Phase

Completed foundation features:

- Professional Python project structure
- Streamlit application entry point
- Environment variable loading
- Centralised settings management
- API key placeholder management
- Logging configuration
- Error handling for invalid configuration
- Docker-ready baseline
- Streamlit local config for repeatable headless startup
- Initial validation test

Phase 2 adds:

- CSV and XLSX drag-and-drop upload through Streamlit
- File type validation
- Upload size validation using `MAX_UPLOAD_MB`
- First 100 row preview
- Row and column count summary
- Field detection for numeric, datetime, boolean, text, and empty columns
- Missing value count per uploaded field
- Upload error handling with user-safe messages

Phase 3 adds:

- Automatic dataset profiling after upload
- Dataset health score
- Missing value issue detection
- Duplicate record detection
- Wrong data type detection
- Outlier detection for numeric fields
- Invalid date and numeric format detection
- Inconsistent category label detection
- Business-friendly recommendations

Phase 4 adds:

- Deterministic data cleaning agent for analytics preparation
- Duplicate record removal
- Missing value handling
- Text standardisation
- Date format repair
- Numeric text conversion
- Invalid record removal
- Anomaly detection with review flags
- Cleaning summary with records processed and records fixed

Phase 5 adds:

- SQLite analytics database layer
- Automatic table creation for cleaned datasets
- Upload metadata storage
- Column metadata storage
- Previous upload history
- Saved table query preview
- PostgreSQL-ready adapter boundary

Phase 6 adds:

- Configurable Gemini or Claude AI business analyst agent
- Dataset-aware business question answering
- Business context input
- Structured AI response format
- Summary, key findings, and business recommendations
- API key gating so no fake AI responses are shown

Phase 7 adds:

- Natural language to SQL agent for saved datasets
- Claude-generated SQLite SELECT statements
- SQL safety checks before execution
- Query result table
- SQL explanation
- Chart recommendation

Phase 8 adds:

- Automatic Plotly chart recommendation
- Interactive line, bar, histogram, and scatter charts
- Cleaned dataset chart preview
- SQL result chart preview
- Downloadable chart HTML

Phase 9 adds:

- Executive dashboard after cleaning
- KPI detection for revenue, growth, profit, and customers
- Sales trend chart
- Regional analysis chart
- Product analysis chart
- Graceful fallback when expected business fields are missing

Phase 10 adds:

- Power BI-ready export preparation for saved datasets
- Downloadable CSV files with UTF-8 encoding
- Downloadable schema JSON with detected Power BI field types
- Downloadable Power Query M template for Power BI Desktop

Phase 11 adds:

- Executive report generation for cleaned uploads and saved datasets
- Downloadable HTML executive reports
- Downloadable Markdown executive reports
- Dataset summary, quality findings, KPI summary, recommendations, and cleaning audit sections

Phase 12 adds:

- Runtime health checks for database, AI provider configuration, and upload limits
- Deployment status panel in the Streamlit sidebar
- Docker container health check for Streamlit readiness
- Docker build ignore file for local secrets, virtual environments, caches, and local databases

## Planned Architecture

```text
app.py                      Streamlit UI entry point
src/config/                 Settings, environment, and logging
src/database/               SQLite first, PostgreSQL-ready data access layer
src/ai/                     Claude API integration and AI agents
src/cleaning/               Data cleaning pipelines
src/analytics/              Profiling, metrics, and business analysis
src/data_upload/            CSV/XLSX upload validation and summary engine
src/visualization/          Plotly chart generation
src/powerbi/                Power BI export preparation
src/reports/                Executive report generation
tests/                      Automated tests
data/                       Local development data storage
notebooks/                  Exploration notebooks
```

## Setup

1. Create and activate a virtual environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies.

```powershell
pip install -r requirements.txt
```

3. Copy the example environment file.

```powershell
Copy-Item .env.example .env
```

4. Update `.env` with real values when integrations are enabled.

## Run Locally

```powershell
streamlit run app.py
```

## Test

```powershell
python -m compileall app.py src tests
pytest
```

## Current Usage

Run the Streamlit app and upload a `.csv` or `.xlsx` file. The app validates the file, loads it into a DataFrame, displays the total rows and columns, detects field types, counts missing values, previews the first 100 rows, generates a profiling report with a dataset health score, creates a cleaned data preview with a cleaning audit summary, builds an executive dashboard, recommends interactive Plotly charts, supports configurable AI-powered business analysis, saves the cleaned dataset to SQLite, can answer natural language SQL questions against saved tables, prepares saved datasets for Power BI Desktop, generates downloadable executive reports, and shows deployment readiness status in the sidebar.

## Environment Variables

| Variable | Purpose | Default |
| --- | --- | --- |
| `APP_NAME` | Application display name | `AI BI Copilot` |
| `APP_ENV` | Runtime environment | `development` |
| `LOG_LEVEL` | Python logging level | `INFO` |
| `DATABASE_URL` | Database connection string | `sqlite:///data/ai_bi_copilot.db` |
| `AI_PROVIDER` | AI provider selection: `auto`, `gemini`, or `claude` | `auto` |
| `GEMINI_API_KEY` | Google AI Studio API key for Gemini | Empty |
| `GEMINI_MODEL` | Gemini model used by AI agents | `gemini-1.5-flash` |
| `CLAUDE_API_KEY` | Claude API key fallback | Empty |
| `CLAUDE_MODEL` | Claude model mode or explicit model ID for AI agents | `auto` |
| `MAX_UPLOAD_MB` | Future upload size limit | `100` |

## Development Workflow

Each module is built, tested, reviewed, and improved before moving to the next phase. Phase 13 will add user workflow polish and navigation improvements.