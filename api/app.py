"""Flask API + minimal Jinja UI."""

from __future__ import annotations

import functools
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from flask import Flask, jsonify, render_template, request

from core.analysis_pipeline import analyze_text
from core.config import API_KEY
from core.ingest import load_fomc, load_speaker, random_corpus_row
from core.logging_config import setup_logging

setup_logging()

log = logging.getLogger(__name__)

_fomc_cache = None
_speaker_cache = None


def _get_corpus():
    global _fomc_cache, _speaker_cache
    if _fomc_cache is None:
        _fomc_cache = load_fomc()
        _speaker_cache = load_speaker()
    return _fomc_cache, _speaker_cache


def _require_api_key(fn):
    """Decorator: enforce X-API-Key header when API_KEY env var is configured."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if API_KEY:
            provided = request.headers.get("X-API-Key", "")
            if provided != API_KEY:
                return jsonify({"error": "Unauthorized — valid X-API-Key header required"}), 401
        return fn(*args, **kwargs)
    return wrapper


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
        static_url_path="/static",
    )

    if not API_KEY:
        log.warning(
            "API_KEY is not set. /api/v1/* endpoints are unauthenticated. "
            "Set API_KEY in your .env file before deploying."
        )

    # ------------------------------------------------------------------ #
    # Rate limiting (requires flask-limiter)                              #
    # ------------------------------------------------------------------ #
    try:
        from flask_limiter import Limiter
        from flask_limiter.util import get_remote_address

        limiter = Limiter(
            key_func=get_remote_address,
            app=app,
            default_limits=["200 per hour", "60 per minute"],
            storage_uri="memory://",
        )

        def _api_limit(fn):
            return limiter.limit("30 per minute")(fn)

    except ImportError:
        log.warning("flask-limiter not installed — rate limiting disabled. pip install flask-limiter")

        def _api_limit(fn):
            return fn

    # ------------------------------------------------------------------ #
    # Routes                                                              #
    # ------------------------------------------------------------------ #

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.post("/api/v1/analyze")
    @_require_api_key
    @_api_limit
    def api_analyze():
        data = request.get_json(silent=True) or {}
        text = data.get("text") or ""
        if not str(text).strip():
            return jsonify({"error": "missing or empty 'text'"}), 400
        if len(str(text)) > 500_000:
            return jsonify({"error": "'text' exceeds 500 000 character limit"}), 413
        result = analyze_text(str(text))
        return jsonify(result)

    @app.get("/api/v1/sample")
    @_require_api_key
    @_api_limit
    def api_sample():
        try:
            fomc, speaker = _get_corpus()
            row = random_corpus_row(fomc, speaker)
            return jsonify(row)
        except FileNotFoundError as exc:
            log.error("Corpus not found: %s", exc)
            return jsonify({"error": str(exc)}), 503
        except Exception as exc:
            log.exception("Unexpected error in /api/v1/sample")
            return jsonify({"error": "internal server error"}), 500

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.post("/analyze-form")
    def analyze_form():
        text = request.form.get("text") or ""
        result = analyze_text(text) if text.strip() else None
        return render_template("index.html", text=text, result=result)

    # ------------------------------------------------------------------ #
    # Error handlers                                                      #
    # ------------------------------------------------------------------ #

    @app.errorhandler(404)
    def not_found(exc):
        return jsonify({"error": "not found"}), 404

    @app.errorhandler(405)
    def method_not_allowed(exc):
        return jsonify({"error": "method not allowed"}), 405

    @app.errorhandler(429)
    def rate_limit_exceeded(exc):
        return jsonify({"error": "rate limit exceeded — slow down"}), 429

    @app.errorhandler(500)
    def internal_error(exc):
        log.exception("Unhandled 500 error")
        return jsonify({"error": "internal server error"}), 500

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
