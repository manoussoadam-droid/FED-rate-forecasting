# FOMC text tools

Python stack for exploring FOMC-related documents: pandas/sklearn pipeline, Flask API, optional Streamlit UI.

## Data

Place `fomc_doc.pkl` and `speaker_doc.pkl` in the `data/` folder (default), or set `DATA_DIR` to another directory. Expected columns:

- **fomc**: `date`, `decision`, `high`, `low`, `document`, `word_count` (optional legacy column `type` is still accepted if present)
- **speaker**: `fomc-ref-date`, `date`, `decision`, `high`, `low`, `domain`, `participant`, `document`, `word_count` (no `type`; for combined EDA, `domain` is exposed as `type`)

**Current canonical corpus (through 2025):** `fomc_doc.pkl` is the full FOMC table (statements through **2025**). `speaker_doc.pkl` is the **Fed-extended** speaker set (regional Feds + `federalreserve.gov` speeches through 2025) **plus** the legacy **non-Fed** rows (e.g. Bloomberg, press) from the original multi-source corpus so training stays comparable. If you only run `extend_speaker_federalreserve.py`, you get Fed-only output; use `scripts/merge_speaker_corpora.py` to re-attach non-Fed rows from a saved legacy pickle.

**Speaker labels (`decision`, `high`, `low`) — extended Fed rows:** for rows built by `scripts/extend_speaker_federalreserve.py`, each speech is mapped to an FOMC meeting date (`fomc-ref-date` = next meeting in the official calendar list, with an end-of-horizon carry-forward). Labels for that meeting come from (1) one row per `fomc-ref-date` in the **original** `speaker_doc.pkl` when the meeting already existed, or (2) otherwise from the official **FOMC press release** HTML at `federalreserve.gov/newsevents/pressreleases/monetary{YYYYMMDD}a.htm`, parsed in `scraping/fed_official.py` (`parse_monetary_statement`: verb maintain/lower/raise + target range text). Nothing is invented by the model; parsing errors are logged when a page does not match the expected wording.

**Retraining notes:** (1) `build_train_test_split` puts the **first 70% of speaker rows in file order** (+ all FOMC) in train and the **last 30%** in test, then shuffles. If you sort or rebuild the speaker DataFrame, train/test composition changes even with the same `random_state`. (2) After adding years of data, class counts for `raise` / `lower` change; read the printed `classification_report` when retraining — rare classes can score worse or show zeros until you have enough examples.

## Setup

```bash
cd fomc_tools
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m nltk.downloader punkt
python -m textblob.download_corpora
cp .env.example .env   # optional
```

If you **rename or move** this project folder, either recreate the venv (remove `.venv` and run the commands above again) or expect broken shebangs in `.venv/bin/*` until you do.

## Train classifier (`decision`: maintain / raise / lower)

```bash
python scripts/train_model.py
```

Artifacts: `artifacts/vectorizer.joblib`, `artifacts/model.joblib`, `artifacts/model_meta.joblib`.

## API (Flask)

```bash
export FLASK_APP=api.app:create_app
flask run --host 127.0.0.1 --port 5000
```

Or: `python api/app.py`

- `GET /health`
- `POST /api/v1/analyze` JSON `{"text": "..."}`
- `GET /api/v1/sample` random row from combined corpus

## Streamlit

```bash
streamlit run streamlit_app.py
```

## MCP server (optional)

```bash
python tools/mcp_server.py
```

## Scraping

See `scraping/README.md` (example only; respect robots.txt and site terms).

### Extend Fed-system speaker data

Reproducible pipeline (JSON index + HTML on **federalreserve.gov**, BeautifulSoup, regex on FOMC press releases, pandas) that:

- Optionally copies `data/speaker_doc.pkl` to `data/speaker_doc_backup.pkl` once if missing (`--no-backup` to skip).
- Keeps only **Fed System** rows from the current speaker pickle, then appends new speeches from `https://www.federalreserve.gov/json/ne-speeches.json` through **2025** (defaults).
- Writes **`data/speaker_doc_fed_extended.pkl`** (Fed-only). To match the full multi-source corpus, merge with non-Fed rows from a legacy pickle:

```bash
python scripts/extend_speaker_federalreserve.py
python scripts/merge_speaker_corpora.py \
  --fed data/speaker_doc_fed_extended.pkl \
  --legacy path/to/speaker_doc_legacy.pkl \
  --out data/speaker_doc.pkl
```

HTML responses are cached under `data/.cache/fed_html/` (gitignored). Options: `--min-date`, `--max-date`, `--max-fomc-ref`, `--delay`, `--limit` (debug), `--no-cache`, `--no-backup`.

## OpenAI (optional)

Set `OPENAI_API_KEY` in `.env` for abstractive summary in `/api/v1/analyze`. Without a key, extractive TextRank still runs.

## Loughran–McDonald dictionary (optional)

Set `LM_DICT_PATH` to a tab-separated **Master Dictionary** file with a `Word` column and sentiment columns. Without it, a small keyword fallback is used (see code).

---

## À faire (équipe — fin du projet)

Pistes concrètes ; cochez dans les Issues GitHub ou un tableau partagé.

1. **Modèle** : améliorer `raise` / `lower` (rappel faible sur le jeu de test) — stratified split, `class_weight`, ou sur-échantillonnage ; documenter les choix dans le rapport.
2. **Données** : si vous régénérez le corpus Fed (`extend_speaker_federalreserve.py`), refaire la fusion avec `merge_speaker_corpora.py` en gardant une copie **legacy** non-Fed si besoin.
3. **Produit / cours** : finaliser la démo (Streamlit ou Flask), texte d’intro, captures ; optionnel : déploiement (Vercel / hébergeur Python) selon consignes du cours.
4. **Qualité** : lancer `pytest`, noter les limites (scraping, clés API optionnelles) et citer les sources Fed dans le README du rendu.
5. **Secrets** : ne jamais committer `.env` ; seulement `.env.example`.

## Publier sur GitHub (premier push)

1. Créer un **nouveau dépôt vide** sur [github.com/new](https://github.com/new) (sans README s’il existe déjà localement).
2. Dans ce dossier, après le premier commit :

```bash
cd fomc_tools
git remote add origin git@github.com:VOTRE_USER/VOTRE_REPO.git
git branch -M main
git push -u origin main
```

(Remplacez par l’URL HTTPS si vous préférez : `https://github.com/VOTRE_USER/VOTRE_REPO.git`.)

**Contenu versionné :** code, `requirements.txt`, `data/*.pkl` (~31 Mo), docs. **Exclus :** `.venv/`, `.env`, `artifacts/*.joblib`, `data/.cache/`.
