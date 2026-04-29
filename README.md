# FOMC Intelligence Pipeline

**Data engineering and NLP infrastructure by Adam.** This repository contains the full data collection, storage, analysis API, and live inference pipeline for FOMC (Federal Open Market Committee) text analysis. The machine-learning layer is developed separately by the team — this README is written specifically to give ML teammates everything they need to understand the data, its structure, its limitations, and how to extend the models.

---

## Project Goals

The demo has two distinct objectives:

### 1. Macro direction forecast
Given the full corpus of recent Fed speeches, minutes, press conferences, and statements, the pipeline should output the **expected future direction of the federal funds rate** over the coming weeks — specifically whether the Fed is likely to `raise`, `maintain`, or `lower` at its next meeting.

### 2. Live speech inference
Given a Fed speech text streaming in real time (even a partial transcript), the model should **converge on the correct rate decision faster than a human analyst** by detecting hawkish/dovish language patterns as early as the opening paragraphs.

---

## Architecture Overview

```
federalreserve.gov ──┐
BIS / CNBC / Fox ────┤  scraping/fed_official.py
                     ├──► rebuild/  ──► data/parquet/  (canonical store)
FRED API ────────────┤              ──► data/audit.sqlite
FedNLP seed data ────┘
                                          │
                              core/ingest.py  (load_fomc / load_speaker)
                                          │
                        ┌─────────────────┼─────────────────┐
                        ▼                 ▼                   ▼
              core/analysis_pipeline   scripts/train_model  tools/mcp_server.py
              (TextBlob + LM + TF-IDF  (vectorizer.joblib   (12 MCP tools via
               + summarizer + predict)  + model.joblib)      FastMCP stdio)
                        │
               api/app.py (Flask REST)     streamlit_app.py (UI)
                        │
               tools/scheduler.py (APScheduler — hourly/4h/daily/weekly jobs)
```

### Key components

| Component | File | Role |
|---|---|---|
| Corpus rebuild | `scripts/rebuild_database.py` | One-command full rebuild of all Parquet + SQLite |
| FOMC scraper | `scraping/fed_official.py` | Pulls statements, minutes, press conferences, speeches from federalreserve.gov |
| FRED integration | `core/fred_api.py` + `rebuild/fred_rates.py` | Fetches rate labels and economic series; SQLite cache |
| News fetcher | `core/news_fetcher.py` | Alpha Vantage + NewsAPI polling; stores to `news` SQLite table |
| Scheduler | `tools/scheduler.py` | Long-running APScheduler process; hourly speech check, 4h news, daily FRED, weekly rebuild |
| Data loader | `core/ingest.py` | `load_fomc()` / `load_speaker()` — returns legacy-compatible DataFrames from Parquet |
| NLP pipeline | `core/analysis_pipeline.py` | Orchestrates all analysis: sentiment, summarization, prediction |
| Classifier | `core/predict.py` | TF-IDF vectorizer + Logistic Regression (or XGBoost) inference |
| Sentiment | `core/sentiment_lm.py` | Loughran-McDonald financial lexicon scoring (or keyword fallback) |
| Flask API | `api/app.py` | REST endpoints: `POST /api/v1/analyze`, `GET /api/v1/sample` |
| Streamlit UI | `streamlit_app.py` | Browser demo: corpus EDA, live text analysis tab, Flask API tab |
| AI Fed Analyst | `core/agentic_fed.py` | Optional Claude/Portkey agent that chooses project tools and writes plain-English Fed briefings |
| MCP server | `tools/mcp_server.py` | 12-tool MCP server for AI assistant integration |
| Audit | `rebuild/audit.py` | Writes per-document quality flags to `document_audit` SQLite table |

---

## Data Storage

### Canonical store — Parquet + SQLite

The runtime path uses Parquet + SQLite. Legacy `.pkl` backups may be present in
the packaged `data/` folder for compatibility/archive reasons, but
`core/ingest.py` reads the partitioned Parquet store and does not use pickle
files as its production fallback.

| Location | Content |
|---|---|
| `data/parquet/fomc/year=YYYY/` | FOMC corpus partitioned by year |
| `data/parquet/speaker/year=YYYY/` | Speaker corpus partitioned by year |
| `data/audit.sqlite` | All audit, cache, news, and scheduler tables (see below) |
| `data/fomc_doc_original.parquet` | Cached FedNLP public seed (FOMC) |
| `data/speaker_doc_original.parquet` | Cached FedNLP public seed (speaker) |
| `data/*.pkl` | Legacy backup/cache artifacts; not the preferred runtime store |

### Parquet schema

Both corpora share these rich metadata columns written by `rebuild/`:

| Column | Type | Description |
|---|---|---|
| `date` | str (YYYYMMDD) | Document date |
| `document_type` | str | `statement`, `minutes`, `press-conference`, `speech`, `testimony` |
| `decision` | str | Rate decision label: `maintain`, `raise`, `lower` |
| `high` | float | Fed funds target range upper bound at that meeting |
| `low` | float | Fed funds target range lower bound |
| `text_content` | str | Full parsed document text |
| `source_url` | str | Canonical URL |
| `ingested_at` | str ISO | Timestamp when row was written |
| `content_hash` | str | SHA-256 of text — used for change detection |
| `quality_score` | float | 0–1 quality estimate |
| `quality_flags` | list[str] | `no_text`, `pdf_unreadable`, `stub`, etc. |
| `parser_version` | str | Parser version that produced this row |
| `labels_from_fred` | bool | Whether decision/high/low came from FRED fallback |
| `domain` | str | (speaker only) Source domain |
| `participant` | str | (speaker only) Speaker name |
| `fomc_ref_date` | str | (speaker only) FOMC meeting this speech is aligned to |

### SQLite tables in `data/audit.sqlite`

| Table | Content |
|---|---|
| `ingestion_log` | One row per rebuild run — timestamps, counts, errors |
| `document_audit` | Per-document quality audit: URL, hash, flags, parse status |
| `fetch_errors` | HTTP errors during scraping — URL, status code, message |
| `fred_cache` | FRED series observations cached locally — `series_id`, `obs_date`, `value` |
| `news` | Financial news articles — `source`, `title`, `url`, `published_at`, `sentiment_score`, `sentiment_label` |
| `scheduler_log` | APScheduler job history — job name, status, records added, errors |
| `_feed_hashes` | Hash of Fed speeches JSON feed — used to detect new speeches cheaply |

### Ingest projection layer

`core/ingest.py` exposes a **legacy-compatible** DataFrame view over the Parquet store. Callers always receive these column names regardless of the underlying Parquet schema:

- **FOMC:** `date`, `type`, `decision`, `high`, `low`, `document`, `word_count`
- **Speaker:** `fomc-ref-date`, `date`, `decision`, `high`, `low`, `domain`, `participant`, `document`, `word_count`

The `document` column maps to `text_content` in Parquet. Do not break this contract when adding columns — extend Parquet and update the projection in `_fomc_from_parquet()` / `_speaker_from_parquet()`.

---

## Live Web Scraping

### How it works

The scraping stack has two layers:

**1. On-demand rebuild** (`scripts/rebuild_database.py`)

Running `python scripts/rebuild_database.py` executes the full pipeline in order:
1. Fetches the Fed speeches JSON feed from `federalreserve.gov/feeds/speeches.json`
2. Crawls official speech and testimony pages with `scraping/fed_official.py` (rate-limited, HTML cached under `data/.cache/fed_html/`)
3. Fetches FOMC statements, minutes, and press-conference PDFs for each known meeting date
4. Downloads the public FedNLP seed data on first run (cached as Parquet — no re-download unless `--refresh-public` is passed)
5. Merges BIS gap-filler speeches where official Fed coverage is missing
6. Merges a curated accessible supplement (CNBC / Fox Business Fed interviews)
7. Writes canonical Parquet + audit SQLite
8. Logs ingestion run to `ingestion_log`

HTML responses are disk-cached in `data/.cache/fed_html/`. Pass `--force-refresh-stubs` to re-fetch only URLs flagged with bad quality, or `--force-refresh-all` to bypass the cache entirely.

**2. Continuous scheduler** (`tools/scheduler.py`)

When running as a long-lived process (`python tools/scheduler.py`), four APScheduler jobs fire automatically:

| Job | Frequency | What it does |
|---|---|---|
| `check_fed_speeches` | Every 60 min | Hashes the Fed speeches JSON feed; logs a change if the feed differs from the last known hash |
| `fetch_news` | Every 4 hours | Calls `fetch_all_news()` — queries Alpha Vantage + NewsAPI for Fed-related articles, deduplicates by URL, inserts new rows into `news` table |
| `refresh_fred` | Daily at 03:00 UTC | Re-fetches all `FRED_DEFAULT_SERIES` and upserts `fred_cache` |
| `full_rebuild` | Every Monday 06:00 UTC | Runs `scripts/rebuild_database.py` as a subprocess — full corpus refresh |

The scheduler also runs `fetch_news` and `refresh_fred` immediately on startup so the DB is populated without waiting for the first scheduled window.

### Document parsing

`scraping/document_parser.py` handles HTML and PDF extraction:
- **HTML:** BeautifulSoup, removes boilerplate navigation/footer, extracts main content
- **PDF:** PyMuPDF-first (better layout reconstruction), fallback to pypdf
- Failed extractions write a `quality_flags` list (`no_text`, `pdf_unreadable`) rather than inventing placeholder text — this means you can query for stubs and re-fetch them after parser improvements

### Speaker label alignment — blackout-aware

This is one of the most important design decisions in the pipeline. Fed governors observe a communication blackout starting ~10 days before each FOMC meeting. A speech given during the blackout window should be labeled to the *upcoming* meeting, not the most recent past one. `rebuild/meetings.py` implements:

- If the speech date falls within the pre-meeting blackout window → label to **that upcoming meeting**
- Otherwise → label to the **most recent prior meeting**

This is more accurate than naive nearest-date joins used in most public Fed NLP datasets.

---

## API Keys

All keys are optional — the pipeline degrades gracefully when absent. Configure in `.env` (copy from `.env.example`).

| Key | Variable | Source | Required for | Free tier |
|---|---|---|---|---|
| FRED API | `FRED_API_KEY` | [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html) | Faster/cleaner rate label fetch; falls back to public CSV | Yes — free |
| NewsAPI | `NEWS_API_KEY` | [newsapi.org/register](https://newsapi.org/register) | Financial news fetch (secondary source) | Yes — 100 req/day |
| Alpha Vantage | `ALPHA_VANTAGE_KEY` | [alphavantage.co](https://www.alphavantage.co/support/#api-key) | Financial news + sentiment (primary news source) | Yes — 500 calls/day |
| OpenAI | `OPENAI_API_KEY` | [platform.openai.com](https://platform.openai.com) | Abstractive summary in `/api/v1/analyze`; falls back to TextRank | No — paid |
| Flask auth | `API_KEY` | Generate with `python -c "import secrets; print(secrets.token_hex(32))"` | Secures `/api/v1/*` endpoints in production | N/A |

**Without any key:** the pipeline still runs fully — FRED uses CSV download, news fetching is skipped with a warning, OpenAI summary returns null, Flask routes are unauthenticated.

### Loughran-McDonald dictionary (optional, recommended)
Set `LM_DICT_PATH` to the path of the tab-separated [LM Master Dictionary](https://sraf.nd.edu/loughranmcdonald-master-dictionary/) file. Without it, a small keyword fallback is used for `score_lm_style()`. The LM dictionary dramatically improves the quality of financial sentiment scoring. Download the CSV from Notre Dame SRAF, convert to TSV, set the path.

---

## FRED Default Series

These series are fetched daily by the scheduler and cached in `fred_cache`:

| Series ID | Description |
|---|---|
| `DFEDTARL` | Fed funds target range lower bound |
| `DFEDTARU` | Fed funds target range upper bound |
| `FEDFUNDS` | Effective federal funds rate |
| `T10Y2Y` | 10-year minus 2-year Treasury spread (recession indicator) |
| `UNRATE` | Unemployment rate |
| `CPIAUCSL` | CPI all urban consumers |

---

## MCP Server

`tools/mcp_server.py` exposes 12 tools via the [MCP](https://github.com/anthropics/model-context-protocol) stdio protocol. Run with `python tools/mcp_server.py` and connect any MCP-compatible AI assistant.

| Tool | Description |
|---|---|
| `analyze_fed_text` | Full NLP pipeline on raw text — returns prediction, sentiment, summary, word cloud |
| `random_corpus_document` | Random document from combined FOMC + speaker Parquet corpus |
| `trigger_scrape_fed` | Live scrape of federalreserve.gov — checks speeches feed and FOMC calendar |
| `trigger_scrape_news` | Fetch news from Alpha Vantage + NewsAPI for a given query |
| `query_database` | Safe read-only SQL SELECT on `audit.sqlite` — up to 500 rows |
| `get_latest_documents` | Newest N documents from Parquet corpus |
| `search_corpus` | Keyword substring search across all document text |
| `get_fred_series` | FRED series observations (cache-first) |
| `get_fomc_calendar` | Upcoming FOMC meeting dates scraped from federalreserve.gov |
| `get_corpus_stats` | Row counts, year range, decision distribution, news counts, FRED cache list |
| `get_audit_report` | Recent ingestion runs, quality flag distribution, fetch errors |
| `get_scheduler_status` | Last 20 APScheduler job run records |

---

## AI Fed Analyst Agent

The Streamlit app includes an optional **AI Fed Analyst** explainer directly
under ML result panels. After a user runs the early-warning speech model,
custom-text analysis, or Flask API analysis, the UI shows an
**Explain these ML results** button and a follow-up question box. With NYU
Portkey configured, Claude receives the specific model output, can choose
which project tools to call, reads the tool outputs, and returns a
plain-English explanation of the prediction.

The agent does not replace the existing ML pipeline. It sits on top of it and
can call these local tools:

| Agent tool | What it does |
|---|---|
| `analyze_fed_text` | Runs the local NLP, sentiment, summary, and policy prediction pipeline |
| `run_policy_early_warning` | Runs the online-prefix model and reports how the prediction changes as the speech unfolds |
| `macro_direction_forecast` | Aggregates recent FOMC and speech documents into a corpus-level cut / hold / hike lean |
| `search_fed_corpus` | Searches local FOMC and speaker text for evidence snippets |
| `get_fred_snapshot` | Reads recent cached FRED macro/rate observations from SQLite |
| `get_model_diagnostics` | Reads the latest saved tuning metrics and model leaderboard |

Enable the Portkey-backed agent locally:

```bash
pip install -r requirements.txt
export PORTKEY_API_KEY="your-key-here"
streamlit run streamlit_app.py
```

Important: keep the real API key in your local shell, private `.env` file, or
the Streamlit password field shown beside the result. Do not hardcode it in
Python files, README text, notebooks, or commits. If no Portkey key is
configured, the result panel still offers a local explanation mode so the demo
remains usable.

---

## Current Model (ML teammates: read this)

### What is trained

`scripts/train_model.py` trains a **TF-IDF + Logistic Regression** classifier (optionally XGBoost via `USE_XGBOOST=1`):

- **Input:** `document` text column (full document text)
- **Target:** `decision` column — three classes: `maintain`, `raise`, `lower`
- **Vectorizer:** TF-IDF, 50,000 features, unigrams + bigrams, `sublinear_tf=True`, `min_df=2`, `max_df=0.95`
- **Model:** Logistic Regression with `class_weight="balanced"`, `solver="saga"`, `max_iter=10000`
- **XGBoost variant:** `multi:softprob`, 200 estimators, depth 6, manual per-sample class weights

Artifacts saved to `artifacts/`: `vectorizer.joblib`, `model.joblib`, `model_meta.joblib`

### Early policy-signal model

The newer product-facing ML workflow is `scripts/tune_policy_signal_models.py`.
It is designed for the live-speech question rather than only full-document
classification.

- **Stage 1:** a rate-relevance classifier predicts whether the speech has any
  usable interest-rate signal or should be labeled `no_rate_signal`.
- **Stage 2:** a direction classifier predicts `lower`, `maintain`, or `raise`
  only when the speech appears rate-relevant.
- **Feature engineering:** TF-IDF unigrams/bigrams plus unsupervised SVD topics,
  NMF topics, KMeans cluster indicators, prefix percent, policy-keyword density,
  hawkish/dovish term counts, FRED macro/rate context at the document date
  (`FEDFUNDS`, target range, `T10Y2Y`, `UNRATE`, `CPIAUCSL`), 90/180-day
  rate-cycle deltas, CPI year-over-year change, meeting-alignment features
  (`blackout_pre`, `post_meeting`, or `no_prior_meeting`), extraction-quality
  flags, optional cached Claude teacher features, and a simple speaker-authority
  tier.
- **Training augmentation:** each training document is expanded into partial
  prefixes such as 20%, 35%, 50%, 75%, and 100%, so the model sees the same kind
  of partial text it receives during the demo.
- **Evaluation split:** the product model defaults to a chronological
  70% train / 10% threshold-tuning / 20% held-out test split. Thresholds are
  selected on the middle split and reported metrics come from the final test
  split.
- **Models compared:** Dummy baseline, Logistic Regression, SGDClassifier, and
  RidgeClassifier by default. Add `--include-neural` for a slower sklearn MLP
  baseline, or `--include-calibrated-svc` for a slower calibrated LinearSVC
  comparison.
- **Threshold tuning:** the lower-class override grid is now family-specific by
  default. Ridge can search lower/cut recovery thresholds, while Logistic
  Regression and SGD keep that override disabled unless explicitly requested.
- **Saved outputs:** every trial is saved under
  `artifacts/policy_signal_tuning/runs/<run_id>/`, with the current best copied
  to `artifacts/policy_signal_tuning/latest/`. The tuner removes empty failed
  run directories and keeps the newest complete runs by default so local
  artifacts do not grow without bound.

Recommended tuning command:

```bash
python scripts/tune_policy_signal_models.py
```

Clean/prune local tuning run history without retraining:

```bash
python scripts/tune_policy_signal_models.py --cleanup-runs-only
```

The Streamlit **Policy signal ML** tab loads
`artifacts/policy_signal_tuning/latest/best_policy_signal_model.joblib` when
that local artifact has been generated or restored. Joblib model binaries are
gitignored, so a fresh clone should run `python scripts/tune_policy_signal_models.py`
or obtain the artifact from the project team before using the pasted-text
early-warning path. Corpus dropdowns and macro aggregation also require local
`data/parquet/` files.

### Train / test split

There are two training paths:

| Path | Purpose | Split behavior |
|---|---|---|
| `scripts/train_model.py` | Legacy full-document classifier used as a fallback artifact | Uses `core.ingest.build_train_test_split`, which now sorts speaker rows by date before its 70/30 speaker holdout. |
| `scripts/tune_policy_signal_models.py` | Main product model for early speech inference | Defaults to chronological 70/10/20 train/threshold-test/held-out-test splitting across the policy-signal frame. |

The Streamlit demo and `core.predict.predict_decision` prefer the tuned
policy-signal artifact when it exists, then fall back to the legacy
`artifacts/model.joblib` path if needed.

### Running inference

```python
from core.predict import predict_decision
from core.text_clean import clean_for_ml
from core.config import MAX_TEXT_CHARS_FOR_VECTOR

result = predict_decision(clean_for_ml(text, MAX_TEXT_CHARS_FOR_VECTOR))
# result = {
#   "prediction": "no_rate_signal" | "lower" | "maintain" | "raise" | "uncertain_rate_signal",
#   "probabilities": {"no_rate_signal": 0.05, "lower": 0.42, "maintain": 0.35, "raise": 0.18},
#   "rate_relevance": 0.95,
#   "error": None
# }
```

When the tuned policy-signal artifact is available, inference returns the
early-warning policy call and probability timeline summary. When only the
legacy logistic model is available, `top_features` contains the TF-IDF ×
coefficient contribution for each token.

---

## Strengths and Weaknesses

### Strengths

| Strength | Detail |
|---|---|
| Ground-truth labels | `decision` comes from FRED actual target range — not market estimates |
| Blackout-aware alignment | Speaker speeches are aligned to meetings using the pre-meeting blackout window, not naive date proximity |
| Document type diversity | Statements, minutes, press conferences, speeches, testimony — all in one corpus |
| Quality flags | Failed extractions are flagged (`no_text`, `pdf_unreadable`) instead of silently inserted as blank rows — the model never trains on empty stubs |
| Fully reproducible | One command (`rebuild_database.py`) rebuilds everything from scratch; Parquet snapshots are backed up |
| Portable storage | Primary runtime data is Parquet plus SQLite; legacy pickle backups may be present but are not the preferred storage path |
| Continuous ingestion | Scheduler keeps the corpus live: new speeches appear within 60 minutes of publication |
| Dissent detection | Because the model is trained on full statement text, it picks up dissenting vote language (e.g. "preferred to lower the target range") — this is genuine dovish pressure signal |

### Weaknesses

| Weakness | Detail | Fix |
|---|---|---|
| Class imbalance | `lower` and `raise` are rarer than `maintain`, and recent chronological splits may contain no recent `raise` examples. | Add more historical cycles and report per-class metrics, not only accuracy |
| Bag-of-words architecture | TF-IDF has no concept of word position or discourse structure. "Inflation is elevated" in paragraph 1 vs 12 looks identical. | Replace or augment with FinBERT / sentence embeddings |
| Limited sequence modeling | The product model now trains on prefixes, but each prefix is still represented as bag-of-words plus engineered numeric features. | Add a carefully evaluated sequence model after the sklearn baseline is stable |
| Legacy artifact path | `scripts/train_model.py` remains a simpler full-document model for fallback compatibility. | Prefer the tuned policy-signal artifact for demo and API inference |
| Short speeches | Powell media interviews (400–650 words) are far below the ~4,000-word training average. Predictions from short texts are noisy. | Filter by word count or apply a confidence threshold based on document length |
| Partial macro context | The model now uses FRED levels and simple deltas, but not market expectations or real-time vintages. | Add market-implied paths, real-time macro vintages, and calibration |
| No calibration | Probabilities are not Platt-calibrated. 57% "lower" does not mean 57% real-world probability. | Add `CalibratedClassifierCV` after training |

---

## Demo Readiness Assessment

### Goal 1 — Macro direction forecast

**Current state:** Implemented in the Streamlit **Speech Early-Warning ML** tab
as a simple aggregation over recent corpus speeches. The app runs the tuned
policy-signal model on each recent document, averages final `lower`,
`maintain`, and `raise` probabilities, and displays the most likely direction.
When corpus rows include dates, FRED context is attached at the speech date.

**What is needed to improve this:**
- Build an ensemble over document types weighted by recency and document authority (statement > minutes > speech)
- Add market-implied policy paths and real-time data vintages, not just latest-prior FRED levels/deltas
- Calibrate probabilities with `CalibratedClassifierCV`

### Goal 2 — Live speech inference faster than a human

**Current state:** Implemented as an sklearn MVP in
`core/policy_signal_ml.py` and `scripts/tune_policy_signal_models.py`. The
model is trained on partial-document prefixes and the Streamlit UI can show a
word-progress timeline for pasted text or selected corpus speeches.

Interpret the results carefully: relevance detection is the strongest part of
the system, while `lower`/`raise` direction remains harder because labels are
imbalanced and Fed language changes across rate regimes. The model now includes
FRED rate-cycle context and speaker-tier features, but it is still an sklearn
MVP rather than a calibrated trading model. The optional neural model is an
sklearn MLP behind `--include-neural`, not a full LSTM/transformer.

---

## Setup

```bash
cd FED-rate-forecasting-main
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m nltk.downloader punkt
python -m textblob.download_corpora
cp .env.example .env   # add your API keys
```

Use the virtual environment for tests and scripts. The project pins
`numpy>=1.24,<2` because NumPy 2.x can break binary pandas/pyarrow/matplotlib
wheels in the system Anaconda environment. If you rename or move the project
folder, recreate the venv because broken shebangs in `.venv/bin/*` can cause
import errors.

---

## Running the Demo

```bash
streamlit run streamlit_app.py
```

The Streamlit app supports two demo modes:

| Mode | Requires |
|---|---|
| Paste-your-own text policy-signal inference | Local tuned artifact at `artifacts/policy_signal_tuning/latest/best_policy_signal_model.joblib`; generate with `python scripts/tune_policy_signal_models.py` if absent |
| Corpus dropdowns, corpus charts, and macro aggregation | Local corpus files under `data/parquet/` |

If `data/parquet/` is unavailable, the pasted-text early-warning demo still
works as long as the tuned artifact exists locally. Rebuild corpus data with:

```bash
python scripts/rebuild_database.py
```

---

## Key Commands

```bash
# Rebuild the full corpus from scratch
python scripts/rebuild_database.py

# Rebuild with specific overrides
python scripts/rebuild_database.py --offline                 # use local cache only
python scripts/rebuild_database.py --no-fred                 # skip FRED label fallback
python scripts/rebuild_database.py --force-refresh-stubs     # re-fetch flagged bad URLs
python scripts/rebuild_database.py --force-refresh-all       # bypass all HTML cache

# Check coverage after rebuild
python scripts/audit_completeness.py --min-year 2020 --format md

# Restore previous Parquet snapshot
python scripts/restore_dataset_backup.py

# Train the classifier
python scripts/train_model.py                  # Logistic Regression (default)
USE_XGBOOST=1 python scripts/train_model.py   # XGBoost variant

# Train/tune the early policy-signal product model
python scripts/tune_policy_signal_models.py

# Run Flask API
python api/app.py                              # or: flask run --port 5000

# Run Streamlit UI
streamlit run streamlit_app.py

# Optional: enable the embedded AI result explainer through NYU Portkey
export PORTKEY_API_KEY="your-key-here"
streamlit run streamlit_app.py

# Run continuous scheduler (long-lived process)
python tools/scheduler.py

# Run MCP server (stdio, for AI assistant integration)
python tools/mcp_server.py

# Run tests
pytest
```

---

## Versioned Content

| What | Git status |
|---|---|
| Code, `requirements.txt`, this README | Commit |
| `data/parquet/` | Optional — large, can reconstruct with `rebuild_database.py` |
| `data/audit.sqlite` | Optional — can reconstruct |
| `.venv/` | Never commit |
| `.env` | Never commit — use `.env.example` only |
| `**/*.joblib` model artifacts | Do not commit — reconstruct with `train_model.py` or `scripts/tune_policy_signal_models.py` |
| `data/.cache/` | Never commit — local HTTP cache, gitignored |

---

## First GitHub Push

1. Create or choose the target repository.
2. From this repository root after the first commit:

```bash
git remote add origin git@github.com:YOUR_USER/YOUR_REPO.git
git switch -c tl/policy-signal-tuning-update
git push -u origin tl/policy-signal-tuning-update
```

For a collaboration repository, open a pull request from the pushed branch into
`main` rather than force-pushing directly to `main`.
