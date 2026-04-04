"""Flask API + minimal Jinja UI."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root on path when running as script
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from flask import Flask, jsonify, render_template, request

from core.analysis_pipeline import analyze_text
from core.ingest import load_fomc, load_speaker, random_corpus_row

_fomc_cache = None
_speaker_cache = None


def _get_corpus():
    global _fomc_cache, _speaker_cache
    if _fomc_cache is None:
        _fomc_cache = load_fomc()
        _speaker_cache = load_speaker()
    return _fomc_cache, _speaker_cache


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
        static_url_path="/static",
    )

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.post("/api/v1/analyze")
    def api_analyze():
        data = request.get_json(silent=True) or {}
        text = data.get("text") or ""
        if not str(text).strip():
            return jsonify({"error": "missing or empty 'text'"}), 400
        result = analyze_text(str(text))
        return jsonify(result)

    @app.get("/api/v1/sample")
    def api_sample():
        try:
            fomc, speaker = _get_corpus()
            row = random_corpus_row(fomc, speaker)
            return jsonify(row)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.post("/analyze-form")
    def analyze_form():
        text = request.form.get("text") or ""
        result = analyze_text(text) if text.strip() else None
        return render_template("index.html", text=text, result=result)

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
