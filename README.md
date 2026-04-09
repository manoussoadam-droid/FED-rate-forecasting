# FOMC text tools

Python stack for exploring FOMC-related documents: pandas/sklearn pipeline, Flask API, optional Streamlit UI.

## Data

Place the dataset files in `data/` (default), or set `DATA_DIR` to another directory. Expected columns:

- **fomc**: `date`, `decision`, `high`, `low`, `document`, `word_count` (optional legacy column `type` is still accepted if present)
- **speaker**: `fomc-ref-date`, `date`, `decision`, `high`, `low`, `domain`, `participant`, `document`, `word_count` (no `type`; for combined EDA, `domain` is exposed as `type`)

**Current data layout:** the app and training use `data/fomc_doc.pkl` and `data/speaker_doc.pkl` as the **fullest working pair**. The folder also keeps `data/fomc_doc_original.pkl` and `data/speaker_doc_original.pkl` as the **original public FedNLP pair** for reference / comparison.

**Speaker labels (`decision`, `high`, `low`):** the refresh pipeline maps each speech to the nearest FOMC meeting date and uses the resulting meeting labels in the final speaker table. Fed-side rows now come from `federalreserve.gov` speeches **and** annual Board testimony pages; the working speaker corpus is then enriched with non-Fed legacy rows, a conservative BIS gap-filler layer, and the curated accessible supplement.

**Retraining notes:** (1) `build_train_test_split` puts the **first 70% of speaker rows in file order** (+ all FOMC) in train and the **last 30%** in test, then shuffles. If you sort or rebuild the speaker DataFrame, train/test composition changes even with the same `random_state`. (2) After adding years of data, class counts for `raise` / `lower` change; read the printed `classification_report` when retraining — rare classes can score worse or show zeros until you have enough examples.

## What Changed

Recent dataset and pipeline work by **Adam**:

- Rebuilt the missing dataset layer into a reproducible 4-file structure: `fomc_doc.pkl`, `speaker_doc.pkl`, `fomc_doc_original.pkl`, and `speaker_doc_original.pkl`.
- Extended post-2020 coverage by adding live official FOMC `statement`, `minutes`, and `press-conference` documents to the canonical corpus.
- Expanded the speaker dataset beyond the original speech feed by ingesting official Board `speech` and `testimony` pages and aligning them automatically to the nearest FOMC meeting.
- Reworked the refresh process into one clear command, `python scripts/rebuild_database.py`, with backup/restore support for safer dataset updates.
- Added controlled supplemental enrichment from legacy public FedNLP data, BIS gap-fillers, and accessible non-Fed media sources while preserving official-source precedence.
- Improved dataset audit output so refresh runs now report coverage, document-type balance, and label distribution more clearly for retraining and maintenance.

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

### Refresh the datasets

Use one command to refresh **all four** dataset files:

```bash
python scripts/rebuild_database.py
```

What it does:

- rebuilds the current **Fed-side** corpus from the local HTML cache plus optional network fetches
- extends the live FOMC table with official `statement`, `minutes`, and `press-conference` documents where available
- extends the live Board speaker table with official `speech` and `testimony` pages
- refreshes the public **original** FedNLP pair into `data/fomc_doc_original.pkl` and `data/speaker_doc_original.pkl`
- merges non-Fed legacy rows back into the working speaker corpus
- adds BIS rows only when they still fill a gap after official Board/Fed ingestion
- adds the curated, currently accessible CNBC / Fox Business supplement
- writes the final working pair to `data/fomc_doc.pkl` and `data/speaker_doc.pkl`
- stores a backup of the previous working pair in `artifacts/data_backups/`

Useful options:

- `--offline` : use only local cache / already-downloaded originals
- `--refresh-public` : re-download the public original FedNLP pair
- `--skip-bis-supplement` : skip the conservative BIS Fed speech gap-filler
- `--skip-accessible-supplement` : skip the curated CNBC / Fox Business add-on
- `--no-fred` : skip FRED target-range fallback

To restore the previous working pair:

```bash
python scripts/restore_dataset_backup.py
```

HTML responses from the Fed are cached under `data/.cache/fed_html/` (gitignored).

## OpenAI (optional)

Set `OPENAI_API_KEY` in `.env` for abstractive summary in `/api/v1/analyze`. Without a key, extractive TextRank still runs.

## Loughran–McDonald dictionary (optional)

Set `LM_DICT_PATH` to a tab-separated **Master Dictionary** file with a `Word` column and sentiment columns. Without it, a small keyword fallback is used (see code).

---

## À faire (équipe — fin du projet)

Pistes concrètes ; cochez dans les Issues GitHub ou un tableau partagé.

1. **Modèle** : améliorer `raise` / `lower` (rappel faible sur le jeu de test) — stratified split, `class_weight`, ou sur-échantillonnage ; documenter les choix dans le rapport.
2. **Données** : si vous régénérez le corpus, utilisez `python scripts/rebuild_database.py` pour recréer proprement les quatre fichiers de données et `python scripts/restore_dataset_backup.py` pour revenir au dernier jeu canonique sauvegardé.
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
