#!/usr/bin/env python3
"""Continuous data-ingestion scheduler for the FOMC analysis pipeline.

Run from project root with venv activated:
    python tools/scheduler.py

Jobs:
  - Every 60 min : Check federalreserve.gov for new speeches (lightweight)
  - Every 4 hours: Fetch financial news (Alpha Vantage + NewsAPI)
  - Every 24 hours: Refresh FRED economic series into fred_cache
  - Every Monday 06:00 UTC: Full corpus rebuild (scripts/rebuild_database.py)

All job runs are logged to the ``scheduler_log`` SQLite table so the MCP
``get_scheduler_status`` tool can report history without the scheduler running.

Press Ctrl-C to stop gracefully.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.triggers.interval import IntervalTrigger
except ImportError:
    print(
        "APScheduler not installed. Run: pip install apscheduler",
        file=sys.stderr,
    )
    sys.exit(1)

from core.config import FRED_DEFAULT_SERIES
from core.db_queries import log_job_finish, log_job_start, insert_news_articles
from core.logging_config import setup_logging

setup_logging()
log = logging.getLogger("scheduler")

# ---------------------------------------------------------------------------
# Job: Hourly Fed speech check
# ---------------------------------------------------------------------------


def job_check_fed_speeches() -> None:
    """Lightweight check of the Fed speeches JSON feed for new entries."""
    run_id = log_job_start("check_fed_speeches")
    new_count = 0
    error: str | None = None
    try:
        import hashlib
        import json
        import sqlite3
        from scraping.fed_official import FedHttpSession, fetch_speeches_metadata

        session = FedHttpSession()
        speeches = fetch_speeches_metadata(session)
        feed_hash = hashlib.sha256(
            json.dumps(speeches, sort_keys=True).encode()
        ).hexdigest()

        from core.config import AUDIT_DB
        conn = sqlite3.connect(AUDIT_DB)
        try:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS _feed_hashes "
                "(feed TEXT PRIMARY KEY, hash TEXT, checked_at TEXT)"
            )
            existing = conn.execute(
                "SELECT hash FROM _feed_hashes WHERE feed='speeches_json'"
            ).fetchone()
            if existing and existing[0] == feed_hash:
                log.info("[check_fed_speeches] No new speeches detected.")
            else:
                from datetime import datetime, timezone
                conn.execute(
                    "INSERT OR REPLACE INTO _feed_hashes VALUES (?, ?, ?)",
                    ("speeches_json", feed_hash, datetime.now(timezone.utc).isoformat()),
                )
                conn.commit()
                new_count = 1
                log.info(
                    "[check_fed_speeches] Feed changed — %d entries in feed.", len(speeches)
                )
        finally:
            conn.close()
    except Exception as exc:
        log.error("[check_fed_speeches] Error: %s", exc)
        error = str(exc)
    finally:
        log_job_finish(run_id, status="ok" if error is None else "error",
                       records_added=new_count, error_msg=error)


# ---------------------------------------------------------------------------
# Job: 4-hourly news fetch
# ---------------------------------------------------------------------------


def job_fetch_news() -> None:
    """Fetch financial news from all configured sources."""
    run_id = log_job_start("fetch_news")
    inserted = 0
    error: str | None = None
    try:
        from core.news_fetcher import fetch_all_news

        articles = fetch_all_news(days_back=1)
        inserted = insert_news_articles(articles)
        log.info("[fetch_news] Fetched %d, inserted %d new articles.", len(articles), inserted)
    except Exception as exc:
        log.error("[fetch_news] Error: %s", exc)
        error = str(exc)
    finally:
        log_job_finish(run_id, status="ok" if error is None else "error",
                       records_added=inserted, error_msg=error)


# ---------------------------------------------------------------------------
# Job: Daily FRED refresh
# ---------------------------------------------------------------------------


def job_refresh_fred() -> None:
    """Refresh all default FRED series into the fred_cache table."""
    run_id = log_job_start("refresh_fred")
    total_rows = 0
    error: str | None = None
    try:
        from core.fred_api import refresh_series_list

        results = refresh_series_list(FRED_DEFAULT_SERIES)
        total_rows = sum(v for v in results.values() if v > 0)
        failures = {k: v for k, v in results.items() if v < 0}
        if failures:
            log.warning("[refresh_fred] Failed series: %s", failures)
        log.info("[refresh_fred] Refreshed %d total observations across %d series.",
                 total_rows, len(results))
    except Exception as exc:
        log.error("[refresh_fred] Error: %s", exc)
        error = str(exc)
    finally:
        log_job_finish(run_id, status="ok" if error is None else "error",
                       records_added=total_rows, error_msg=error)


# ---------------------------------------------------------------------------
# Job: Weekly full corpus rebuild
# ---------------------------------------------------------------------------


def job_full_rebuild() -> None:
    """Run the full rebuild_database.py pipeline (same as the GitHub Action)."""
    run_id = log_job_start("full_rebuild")
    error: str | None = None
    try:
        rebuild_script = ROOT / "scripts" / "rebuild_database.py"
        log.info("[full_rebuild] Starting rebuild: %s", rebuild_script)
        result = subprocess.run(
            [sys.executable, str(rebuild_script)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour hard timeout
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"rebuild_database.py exited {result.returncode}\n"
                f"stderr: {result.stderr[-2000:]}"
            )
        log.info("[full_rebuild] Completed successfully.")
        log.debug("[full_rebuild] stdout tail: %s", result.stdout[-1000:])
    except Exception as exc:
        log.error("[full_rebuild] Error: %s", exc)
        error = str(exc)
    finally:
        log_job_finish(run_id, status="ok" if error is None else "error",
                       records_added=0, error_msg=error)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    log.info("Starting FOMC Intelligence Scheduler (Ctrl-C to stop)...")

    scheduler = BlockingScheduler(timezone="UTC")

    # Every 60 minutes: lightweight Fed feed check
    scheduler.add_job(
        job_check_fed_speeches,
        trigger=IntervalTrigger(minutes=60),
        id="check_fed_speeches",
        name="Fed speeches feed check",
        replace_existing=True,
        misfire_grace_time=300,
    )

    # Every 4 hours: full news fetch
    scheduler.add_job(
        job_fetch_news,
        trigger=IntervalTrigger(hours=4),
        id="fetch_news",
        name="Financial news fetch",
        replace_existing=True,
        misfire_grace_time=600,
    )

    # Every 24 hours: FRED series refresh (staggered to avoid overlapping rebuild)
    scheduler.add_job(
        job_refresh_fred,
        trigger=CronTrigger(hour=3, minute=0),  # 03:00 UTC daily
        id="refresh_fred",
        name="FRED series refresh",
        replace_existing=True,
        misfire_grace_time=1800,
    )

    # Every Monday at 06:00 UTC: full rebuild
    scheduler.add_job(
        job_full_rebuild,
        trigger=CronTrigger(day_of_week="mon", hour=6, minute=0),
        id="full_rebuild",
        name="Full corpus rebuild",
        replace_existing=True,
        misfire_grace_time=3600,
    )

    log.info("Scheduled jobs:")
    for job in scheduler.get_jobs():
        log.info("  %-30s next run: %s", job.name, job.next_run_time)

    # Run an immediate news fetch and FRED check on startup so the DB is
    # populated right away rather than waiting for the first scheduled window.
    log.info("Running startup jobs (news + FRED)...")
    job_fetch_news()
    job_refresh_fred()

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        log.info("Scheduler stopped.")


if __name__ == "__main__":
    main()
