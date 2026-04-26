"""Centralised logging configuration for the FOMC tools pipeline.

Call :func:`setup_logging` once at the top of every entry-point module
(``tools/scheduler.py``, ``api/app.py``, ``scripts/rebuild_database.py``,
``tools/mcp_server.py``) before any other imports that may log.

Log format
----------
Plain text by default; set ``LOG_FORMAT=json`` in the environment to emit
newline-delimited JSON (suitable for log aggregators like Datadog, Loki,
or Cloud Logging).

Log level
---------
Controlled by the ``LOG_LEVEL`` environment variable (default ``INFO``).
Valid values: ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``, ``CRITICAL``.

Log file
--------
If ``LOG_FILE`` env var is set, logs are tee'd to that path in addition to
stderr.  The file handler uses a rotating policy (10 MB, 5 backups) so it
never fills the disk on a long-running scheduler process.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sys
from datetime import datetime, timezone
from typing import Any


# ---------------------------------------------------------------------------
# JSON formatter
# ---------------------------------------------------------------------------


class _JsonFormatter(logging.Formatter):
    """Emit one JSON object per log record (newline-delimited)."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)
        extra_skip = {
            "name", "msg", "args", "levelname", "levelno", "pathname",
            "filename", "module", "exc_info", "exc_text", "stack_info",
            "lineno", "funcName", "created", "msecs", "relativeCreated",
            "thread", "threadName", "processName", "process", "taskName",
            "message",
        }
        for key, val in record.__dict__.items():
            if key not in extra_skip:
                try:
                    json.dumps(val)
                    payload[key] = val
                except (TypeError, ValueError):
                    payload[key] = str(val)
        return json.dumps(payload, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def setup_logging(
    *,
    level: str | None = None,
    fmt: str | None = None,
    log_file: str | None = None,
) -> None:
    """Configure root logger.  Safe to call multiple times (idempotent).

    Parameters
    ----------
    level:
        Override ``LOG_LEVEL`` env var.  Accepts Python level names.
    fmt:
        ``"json"`` for JSON output, anything else for plain text.
        Defaults to ``LOG_FORMAT`` env var, then ``"text"``.
    log_file:
        Path to a rotating log file.  Defaults to ``LOG_FILE`` env var.
    """
    root = logging.getLogger()
    if root.handlers:
        return  # already configured

    level_name = (level or os.environ.get("LOG_LEVEL", "INFO")).upper()
    numeric_level = getattr(logging, level_name, logging.INFO)
    root.setLevel(numeric_level)

    use_json = (fmt or os.environ.get("LOG_FORMAT", "text")).lower() == "json"

    if use_json:
        formatter: logging.Formatter = _JsonFormatter()
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(formatter)
    root.addHandler(stderr_handler)

    file_path = log_file or os.environ.get("LOG_FILE", "")
    if file_path:
        rotating = logging.handlers.RotatingFileHandler(
            file_path,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding="utf-8",
        )
        rotating.setFormatter(formatter)
        root.addHandler(rotating)

    # Silence chatty third-party loggers at WARNING unless debug mode.
    if numeric_level > logging.DEBUG:
        for noisy in ("urllib3", "httpcore", "httpx", "playwright", "asyncio"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
