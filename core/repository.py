"""Data access layer for the FOMC / speaker corpora.

Canonical storage is Parquet (partitioned by ``year``) plus a small SQLite
audit database.

Schemas
-------

``fomc`` (partition column = ``year``)::

    date           string  YYYYMMDD
    year           int32   (partition)
    document_type  string  statement | minutes | press-conference
    speaker        string  nullable (minutes/press-conf)
    text_content   string
    decision       string  maintain | raise | lower
    high           string  (fed convention: high=lower bound)
    low            string  (fed convention: low=upper bound)
    source_url     string
    ingested_at    timestamp[us, tz=UTC]
    content_hash   string  sha256(text_content)
    quality_score  float64 in [0, 1]
    quality_flags  list<string>
    parser_version string
    labels_from_fred bool

``speaker`` extends the above with::

    participant     string
    domain          string
    fomc_ref_date   string  YYYYMMDD
    alignment_rule  string  blackout_pre | post_meeting | no_prior_meeting

SQLite tables
-------------

- ``ingestion_log(run_id, started_at, finished_at, rows_fomc, rows_speaker, errors_csv_path)``
- ``document_audit(content_hash PK, date, document_type, source_url,
  quality_flags, last_seen_at, parser_version)``
- ``fetch_errors(run_id, url, http_status, message, occurred_at)``
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
import uuid
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from core.config import (
    AUDIT_DB,
    FOMC_PARQUET_DIR,
    PARQUET_DIR,
    SPEAKER_PARQUET_DIR,
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


def _utc_ts_type() -> pa.DataType:
    return pa.timestamp("us", tz="UTC")


FOMC_SCHEMA = pa.schema(
    [
        pa.field("date", pa.string()),
        pa.field("year", pa.int32()),
        pa.field("document_type", pa.string()),
        pa.field("speaker", pa.string()),
        pa.field("text_content", pa.string()),
        pa.field("decision", pa.string()),
        pa.field("high", pa.string()),
        pa.field("low", pa.string()),
        pa.field("source_url", pa.string()),
        pa.field("ingested_at", _utc_ts_type()),
        pa.field("content_hash", pa.string()),
        pa.field("quality_score", pa.float64()),
        pa.field("quality_flags", pa.list_(pa.string())),
        pa.field("parser_version", pa.string()),
        pa.field("labels_from_fred", pa.bool_()),
    ]
)

SPEAKER_SCHEMA = pa.schema(
    [
        pa.field("date", pa.string()),
        pa.field("year", pa.int32()),
        pa.field("document_type", pa.string()),
        pa.field("participant", pa.string()),
        pa.field("domain", pa.string()),
        pa.field("text_content", pa.string()),
        pa.field("decision", pa.string()),
        pa.field("high", pa.string()),
        pa.field("low", pa.string()),
        pa.field("fomc_ref_date", pa.string()),
        pa.field("alignment_rule", pa.string()),
        pa.field("source_url", pa.string()),
        pa.field("ingested_at", _utc_ts_type()),
        pa.field("content_hash", pa.string()),
        pa.field("quality_score", pa.float64()),
        pa.field("quality_flags", pa.list_(pa.string())),
        pa.field("parser_version", pa.string()),
    ]
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def content_hash(text: str) -> str:
    """Stable sha256 over normalized text (used as Parquet upsert key)."""
    payload = (text or "").strip()
    return hashlib.sha256(payload.encode("utf-8", errors="replace")).hexdigest()


def compute_quality_score(word_count: int, flags: Iterable[str]) -> float:
    """Transparent linear scorer; matches the formula documented in the plan."""
    f = set(flags or [])
    score = 1.0
    if "no_text" in f:
        score -= 0.4
    if "pdf_unreadable" in f:
        score -= 0.2
    if "html_structure_unknown" in f:
        score -= 0.2
    if "http_error" in f or "http_404" in f:
        score -= 0.1
    if (word_count or 0) < 120:
        score -= 0.1
    return max(0.0, min(1.0, round(score, 3)))


def _ensure_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, tuple):
        return [str(v) for v in value]
    if isinstance(value, str):
        if not value.strip():
            return []
        try:
            loaded = json.loads(value)
            if isinstance(loaded, list):
                return [str(v) for v in loaded]
        except Exception:
            pass
        return [value]
    return []


def _derive_year(ymd: str) -> int:
    try:
        return int(str(ymd)[:4])
    except Exception:
        return 0


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Normalization: DataFrame -> canonical Parquet rows
# ---------------------------------------------------------------------------


def normalize_fomc_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce an assemble-output FOMC DataFrame to the canonical Parquet schema."""
    if df is None or df.empty:
        return pd.DataFrame({f.name: pd.Series(dtype="object") for f in FOMC_SCHEMA})
    out = df.copy()
    out["date"] = out["date"].astype(str).str.replace(r"\.0$", "", regex=True)
    out["year"] = out["date"].map(_derive_year).astype("int32")
    doc_type_source = out["type"] if "type" in out.columns else "statement"
    out["document_type"] = pd.Series(doc_type_source, index=out.index).astype(str).fillna("statement")
    out["document_type"] = out["document_type"].replace({"": "statement"})
    if "document" in out.columns and "text_content" not in out.columns:
        out["text_content"] = out["document"].astype(str)
    out["text_content"] = out.get("text_content", pd.Series([""] * len(out), index=out.index)).fillna("").astype(str)
    out["speaker"] = out.get("speaker", pd.Series([None] * len(out), index=out.index))
    if "decision" not in out.columns:
        out["decision"] = ""
    out["decision"] = out["decision"].astype(str)
    if "high" not in out.columns:
        out["high"] = ""
    out["high"] = out["high"].astype(str)
    if "low" not in out.columns:
        out["low"] = ""
    out["low"] = out["low"].astype(str)
    if "source_url" not in out.columns:
        out["source_url"] = ""
    out["source_url"] = out["source_url"].astype(str)
    if "ingested_at" in out.columns:
        out["ingested_at"] = pd.to_datetime(out["ingested_at"], utc=True, errors="coerce").fillna(
            pd.Timestamp(_now())
        )
    else:
        out["ingested_at"] = pd.Timestamp(_now())
    flags_series = out.get("quality_flags", pd.Series([[]] * len(out), index=out.index))
    out["quality_flags"] = flags_series.map(_ensure_list)
    if "word_count" not in out.columns:
        out["word_count"] = out["text_content"].map(lambda t: len(str(t).split()))
    out["content_hash"] = out["text_content"].map(content_hash)
    out["quality_score"] = [
        compute_quality_score(int(wc or 0), flags)
        for wc, flags in zip(out["word_count"].tolist(), out["quality_flags"].tolist(), strict=False)
    ]
    if "parser_version" not in out.columns:
        out["parser_version"] = ""
    out["parser_version"] = out["parser_version"].astype(str)
    if "labels_from_fred" not in out.columns:
        out["labels_from_fred"] = False
    out["labels_from_fred"] = out["labels_from_fred"].astype(bool)
    cols = [f.name for f in FOMC_SCHEMA]
    for c in cols:
        if c not in out.columns:
            out[c] = None
    return out[cols]


def normalize_speaker_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame({f.name: pd.Series(dtype="object") for f in SPEAKER_SCHEMA})
    out = df.copy()
    out["date"] = out["date"].astype(str).str.replace(r"\.0$", "", regex=True)
    out["year"] = out["date"].map(_derive_year).astype("int32")
    out["document_type"] = "speech"
    if "document" in out.columns and "text_content" not in out.columns:
        out["text_content"] = out["document"].astype(str)
    out["text_content"] = out.get("text_content", pd.Series([""] * len(out), index=out.index)).fillna("").astype(str)
    if "participant" not in out.columns:
        out["participant"] = ""
    out["participant"] = out["participant"].astype(str)
    if "domain" not in out.columns:
        out["domain"] = ""
    out["domain"] = out["domain"].astype(str)
    if "fomc_ref_date" not in out.columns and "fomc-ref-date" in out.columns:
        out["fomc_ref_date"] = out["fomc-ref-date"].astype(str)
    if "fomc_ref_date" not in out.columns:
        out["fomc_ref_date"] = ""
    out["fomc_ref_date"] = out["fomc_ref_date"].astype(str)
    if "alignment_rule" not in out.columns:
        out["alignment_rule"] = ""
    out["alignment_rule"] = out["alignment_rule"].astype(str)
    if "decision" not in out.columns:
        out["decision"] = ""
    out["decision"] = out["decision"].astype(str)
    if "high" not in out.columns:
        out["high"] = ""
    out["high"] = out["high"].astype(str)
    if "low" not in out.columns:
        out["low"] = ""
    out["low"] = out["low"].astype(str)
    if "source_url" not in out.columns:
        out["source_url"] = ""
    out["source_url"] = out["source_url"].astype(str)
    if "ingested_at" in out.columns:
        out["ingested_at"] = pd.to_datetime(out["ingested_at"], utc=True, errors="coerce").fillna(
            pd.Timestamp(_now())
        )
    else:
        out["ingested_at"] = pd.Timestamp(_now())
    flags_series = out.get("quality_flags", pd.Series([[]] * len(out), index=out.index))
    out["quality_flags"] = flags_series.map(_ensure_list)
    if "word_count" not in out.columns:
        out["word_count"] = out["text_content"].map(lambda t: len(str(t).split()))
    out["content_hash"] = out["text_content"].map(content_hash)
    out["quality_score"] = [
        compute_quality_score(int(wc or 0), flags)
        for wc, flags in zip(out["word_count"].tolist(), out["quality_flags"].tolist(), strict=False)
    ]
    if "parser_version" not in out.columns:
        out["parser_version"] = ""
    out["parser_version"] = out["parser_version"].astype(str)
    cols = [f.name for f in SPEAKER_SCHEMA]
    for c in cols:
        if c not in out.columns:
            out[c] = None
    return out[cols]


# ---------------------------------------------------------------------------
# SQLite audit
# ---------------------------------------------------------------------------


_AUDIT_DDL = [
    """CREATE TABLE IF NOT EXISTS ingestion_log (
        run_id          TEXT PRIMARY KEY,
        started_at      TEXT NOT NULL,
        finished_at     TEXT,
        rows_fomc       INTEGER,
        rows_speaker    INTEGER,
        errors_csv_path TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS document_audit (
        content_hash    TEXT PRIMARY KEY,
        date            TEXT,
        document_type   TEXT,
        source_url      TEXT,
        quality_flags   TEXT,
        last_seen_at    TEXT,
        parser_version  TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS fetch_errors (
        run_id        TEXT,
        url           TEXT,
        http_status   INTEGER,
        message       TEXT,
        occurred_at   TEXT
    )""",
    "CREATE INDEX IF NOT EXISTS idx_audit_date ON document_audit(date)",
    "CREATE INDEX IF NOT EXISTS idx_audit_flags ON document_audit(quality_flags)",
]


# ---------------------------------------------------------------------------
# Repository
# ---------------------------------------------------------------------------


class DocumentRepository:
    """Parquet + SQLite data access layer.

    Writes are idempotent on ``content_hash`` per partition. Reads return
    pandas DataFrames with the full canonical schema; helper functions in
    :mod:`core.ingest` project these back to the legacy column set for
    existing callers.
    """

    def __init__(
        self,
        *,
        fomc_dir: Path | None = None,
        speaker_dir: Path | None = None,
        audit_db: Path | None = None,
    ) -> None:
        self.fomc_dir = Path(fomc_dir) if fomc_dir is not None else FOMC_PARQUET_DIR
        self.speaker_dir = Path(speaker_dir) if speaker_dir is not None else SPEAKER_PARQUET_DIR
        self.audit_db = Path(audit_db) if audit_db is not None else AUDIT_DB
        PARQUET_DIR.mkdir(parents=True, exist_ok=True)
        self.fomc_dir.mkdir(parents=True, exist_ok=True)
        self.speaker_dir.mkdir(parents=True, exist_ok=True)
        self._init_audit()

    # -- SQLite -----------------------------------------------------------
    def _init_audit(self) -> None:
        self.audit_db.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.audit_db)) as cx:
            cx.execute("PRAGMA journal_mode=WAL")
            cx.execute("PRAGMA synchronous=NORMAL")
            cx.execute("PRAGMA foreign_keys=ON")
            for stmt in _AUDIT_DDL:
                cx.execute(stmt)
            cx.commit()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        cx = sqlite3.connect(self.audit_db, check_same_thread=False)
        cx.execute("PRAGMA journal_mode=WAL")
        cx.execute("PRAGMA synchronous=NORMAL")
        cx.execute("PRAGMA foreign_keys=ON")
        try:
            yield cx
            cx.commit()
        finally:
            cx.close()

    # -- Parquet write ---------------------------------------------------
    def _has_existing(self, parquet_dir: Path) -> bool:
        return any(parquet_dir.rglob("*.parquet"))

    def _read_table(self, parquet_dir: Path, schema: pa.Schema) -> pa.Table | None:
        if not self._has_existing(parquet_dir):
            return None
        dataset = ds.dataset(parquet_dir, format="parquet", partitioning="hive", schema=schema)
        return dataset.to_table()

    def _write_partitioned(self, df: pd.DataFrame, parquet_dir: Path, schema: pa.Schema) -> None:
        if df.empty:
            return
        table = pa.Table.from_pandas(df, schema=schema, preserve_index=False, safe=False)
        pq.write_to_dataset(
            table,
            root_path=str(parquet_dir),
            partition_cols=["year"],
            existing_data_behavior="overwrite_or_ignore",
        )

    def _merge_and_write(
        self,
        df_new: pd.DataFrame,
        *,
        parquet_dir: Path,
        schema: pa.Schema,
        dedupe_keys: list[str],
    ) -> int:
        """Upsert semantics via content_hash (+ optional extra keys)."""
        if df_new is None or df_new.empty:
            return 0
        existing = self._read_table(parquet_dir, schema)
        frames: list[pd.DataFrame] = []
        if existing is not None and existing.num_rows > 0:
            frames.append(existing.to_pandas(types_mapper=pd.ArrowDtype))
        frames.append(df_new)
        merged = pd.concat(frames, axis=0, ignore_index=True)
        merged = merged.drop_duplicates(subset=dedupe_keys, keep="last")
        # Rewrite all partitions touched by the merged frame.
        years_to_write = sorted({int(y) for y in merged["year"].tolist() if y is not None})
        for year in years_to_write:
            part = merged[merged["year"] == year]
            # Remove old files in this partition, then write fresh.
            part_dir = parquet_dir / f"year={year}"
            if part_dir.exists():
                for f in part_dir.glob("*.parquet"):
                    f.unlink()
            self._write_partitioned(part.reset_index(drop=True), parquet_dir, schema)
        return int(len(df_new))

    def write_fomc(self, df: pd.DataFrame) -> int:
        normalized = normalize_fomc_frame(df)
        rows = self._merge_and_write(
            normalized,
            parquet_dir=self.fomc_dir,
            schema=FOMC_SCHEMA,
            dedupe_keys=["date", "document_type", "content_hash"],
        )
        self._record_audit(normalized)
        return rows

    def write_speaker(self, df: pd.DataFrame) -> int:
        normalized = normalize_speaker_frame(df)
        rows = self._merge_and_write(
            normalized,
            parquet_dir=self.speaker_dir,
            schema=SPEAKER_SCHEMA,
            dedupe_keys=["date", "participant", "domain", "content_hash"],
        )
        self._record_audit(normalized)
        return rows

    # -- Parquet read -----------------------------------------------------
    def read_fomc(self, *, year_range: tuple[int, int] | None = None) -> pd.DataFrame:
        return self._read_partitioned(self.fomc_dir, FOMC_SCHEMA, year_range=year_range)

    def read_speaker(self, *, year_range: tuple[int, int] | None = None) -> pd.DataFrame:
        return self._read_partitioned(self.speaker_dir, SPEAKER_SCHEMA, year_range=year_range)

    def _read_partitioned(
        self,
        parquet_dir: Path,
        schema: pa.Schema,
        *,
        year_range: tuple[int, int] | None,
    ) -> pd.DataFrame:
        if not self._has_existing(parquet_dir):
            return pd.DataFrame({f.name: pd.Series(dtype="object") for f in schema})
        dataset = ds.dataset(parquet_dir, format="parquet", partitioning="hive", schema=schema)
        filt = None
        if year_range is not None:
            lo, hi = year_range
            filt = (ds.field("year") >= lo) & (ds.field("year") <= hi)
        table = dataset.to_table(filter=filt)
        df = table.to_pandas(types_mapper=None)
        if "quality_flags" in df.columns:
            df["quality_flags"] = df["quality_flags"].map(lambda v: list(v) if v is not None else [])
        return df

    # -- Audit tables ----------------------------------------------------
    def _record_audit(self, df: pd.DataFrame) -> None:
        if df is None or df.empty:
            return
        now_iso = _now().isoformat()
        rows = []
        for _, r in df.iterrows():
            rows.append(
                (
                    str(r.get("content_hash", "")),
                    str(r.get("date", "")),
                    str(r.get("document_type", "")),
                    str(r.get("source_url", "")),
                    json.dumps(list(r.get("quality_flags", []) or [])),
                    now_iso,
                    str(r.get("parser_version", "")),
                )
            )
        with self._conn() as cx:
            cx.executemany(
                """INSERT INTO document_audit
                   (content_hash, date, document_type, source_url, quality_flags,
                    last_seen_at, parser_version)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(content_hash) DO UPDATE SET
                       date=excluded.date,
                       document_type=excluded.document_type,
                       source_url=excluded.source_url,
                       quality_flags=excluded.quality_flags,
                       last_seen_at=excluded.last_seen_at,
                       parser_version=excluded.parser_version""",
                rows,
            )

    def start_run(self) -> str:
        run_id = uuid.uuid4().hex
        with self._conn() as cx:
            cx.execute(
                "INSERT INTO ingestion_log (run_id, started_at) VALUES (?, ?)",
                (run_id, _now().isoformat()),
            )
        return run_id

    def finish_run(
        self,
        run_id: str,
        *,
        rows_fomc: int,
        rows_speaker: int,
        errors_csv_path: str = "",
    ) -> None:
        with self._conn() as cx:
            cx.execute(
                """UPDATE ingestion_log
                   SET finished_at=?, rows_fomc=?, rows_speaker=?, errors_csv_path=?
                   WHERE run_id=?""",
                (_now().isoformat(), int(rows_fomc), int(rows_speaker), str(errors_csv_path or ""), run_id),
            )

    def log_fetch_error(self, run_id: str, url: str, http_status: int | None, message: str) -> None:
        with self._conn() as cx:
            cx.execute(
                "INSERT INTO fetch_errors (run_id, url, http_status, message, occurred_at) VALUES (?, ?, ?, ?, ?)",
                (run_id, url, int(http_status) if http_status is not None else None, message, _now().isoformat()),
            )

    def stub_urls(
        self,
        *,
        flags: tuple[str, ...] = ("no_text", "pdf_unreadable", "html_structure_unknown"),
    ) -> list[tuple[str, str]]:
        """Return ``[(source_url, content_hash), ...]`` for audit rows matching any given flag.

        Used by the ``--force-refresh-stubs`` path in ``rebuild_database.py``.
        """
        out: list[tuple[str, str]] = []
        like_clauses = " OR ".join(["quality_flags LIKE ?" for _ in flags])
        params = [f'%"{f}"%' for f in flags]
        with closing(sqlite3.connect(self.audit_db, check_same_thread=False)) as cx:
            cx.execute("PRAGMA journal_mode=WAL")
            cur = cx.execute(
                f"SELECT source_url, content_hash FROM document_audit WHERE {like_clauses}",
                params,
            )
            for url, h in cur.fetchall():
                if url:
                    out.append((str(url), str(h)))
        return out


__all__ = [
    "DocumentRepository",
    "FOMC_SCHEMA",
    "SPEAKER_SCHEMA",
    "normalize_fomc_frame",
    "normalize_speaker_frame",
    "content_hash",
    "compute_quality_score",
]


if __name__ == "__main__":  # pragma: no cover - quick manual check
    repo = DocumentRepository()
    print(f"fomc parquet dir: {repo.fomc_dir}", file=sys.stderr)
    print(f"speaker parquet dir: {repo.speaker_dir}", file=sys.stderr)
    print(f"audit db: {repo.audit_db}", file=sys.stderr)
