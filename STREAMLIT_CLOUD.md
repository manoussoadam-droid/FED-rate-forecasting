# Deploy on Streamlit Community Cloud

This repo is ready for [Streamlit Community Cloud](https://share.streamlit.io): link GitHub, pick this repo and branch, set the main file to **`streamlit_app.py`** (repository root).

## 1. Push this branch to GitHub

Branch used for production demos: **`final-version`**.

## 2. Create the app on share.streamlit.io

1. Sign in with GitHub.
2. **New app** → select **Repository**: `manoussoadam-droid/FED-rate-forecasting` (or your fork).
3. **Branch**: `final-version`.
4. **Main file path**: `streamlit_app.py`.
5. **Deploy**.

## 3. Secrets (optional but recommended)

In the app’s **Settings → Secrets**, add only what you need, for example:

```toml
# Optional — OpenAI summaries
# OPENAI_API_KEY = "sk-..."

# Optional — Claude / Portkey (AI analyst & global assistant)
# PORTKEY_API_KEY = "..."
# PORTKEY_BASE_URL = "https://ai-gateway.apps.cloud.rt.nyu.edu/v1"
# PORTKEY_MODEL = "@vertexai/anthropic.claude-opus-4-6"

# Optional — data APIs
# FRED_API_KEY = "..."
# NEWS_API_KEY = "..."
# ALPHA_VANTAGE_KEY = "..."
```

Without keys, many features still run (see README): FRED may use CSV fallback; news agents may skip; AI analyst falls back to local text.

## 4. Data & corpus on Cloud

This repository **does not** ship `data/parquet/` or `data/*.sqlite` (they are in `.gitignore` for size). On Streamlit Cloud:

- **Speech Early-Warning ML** / **macro forecast** / corpus tabs may show warnings until you provide data (e.g. release a dataset artifact, or mount storage — advanced).
- Tabs that only need the **bundled policy artifact** and pasted text can still work.

To widen what works in the cloud, commit a minimal `data/` subset only if licensing and size allow, or document a post-deploy download step.

## 5. Limits

- **Resource limits**: Community Cloud has CPU/RAM limits; very large models or huge DataFrames may need a paid tier or another host.
- **Sleep**: Free apps spin down when idle; first load after sleep can take a minute.

## 6. Your public URL

After deploy, Streamlit shows a URL like `https://<app-name>.streamlit.app`. Share that link with your class.
