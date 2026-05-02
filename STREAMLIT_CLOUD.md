# Deploy on Streamlit Community Cloud — IMPORTANT

> **The single most important step** is selecting **Python 3.10** in
> **Advanced settings** when creating the app. Streamlit Cloud now defaults to
> Python 3.14, which has no PyArrow wheels and breaks the build.

## 1. Push this branch to GitHub

Branch used for production demos: **`final-version`**.

## 2. Create the app on share.streamlit.io

1. Sign in with GitHub.
2. **New app** → select **Repository**: `manoussoadam-droid/FED-rate-forecasting`.
3. **Branch**: `final-version`.
4. **Main file path**: `streamlit_app.py`.
5. ⚠️ **Click "Advanced settings" BEFORE clicking Deploy.**
6. **Python version** dropdown → choose **3.10** (3.11 or 3.12 also OK).
7. Add Secrets here too (see step 3) if you have them ready.
8. **Deploy**.

> The `runtime.txt` and `.python-version` files in the repo are kept as a
> belt-and-suspenders signal, but Streamlit Cloud reads the dropdown first.

## 3. Secrets (optional but recommended)

In **Settings → Secrets**, paste TOML like:

```toml
# OPENAI_API_KEY = "sk-..."

# PORTKEY_API_KEY = "..."
# PORTKEY_BASE_URL = "https://ai-gateway.apps.cloud.rt.nyu.edu/v1"
# PORTKEY_MODEL = "@vertexai/anthropic.claude-opus-4-6"

# FRED_API_KEY = "..."
# NEWS_API_KEY = "..."
# ALPHA_VANTAGE_KEY = "..."
```

Without keys, many features still run (FRED CSV fallback, AI analyst falls back to local text, etc.).

## 4. Data & corpus on Cloud

`data/parquet/` and `data/*.sqlite` are in `.gitignore` for size. On Streamlit Cloud:

- **Speech Early-Warning ML / macro forecast** corpus tabs may show warnings until you ship a small dataset sample in the repo.
- Tabs based on the bundled policy artifact + pasted text work without data.

## 5. If the build still fails

- **Delete and recreate the app** so the Python 3.10 selection is applied from scratch.
- Check **Manage app → Logs** for the actual `pip` failure.
- Common pitfalls:
  - Python version is still 3.14 → you forgot the Advanced settings step.
  - PyArrow source build → wrong Python version (no wheel for cp314).

## 6. Your public URL

After deploy, Streamlit gives you `https://<app-name>.streamlit.app`. Share that link with your class.
