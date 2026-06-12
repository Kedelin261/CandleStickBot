"""
Sprint 13 — Historical CSV Data Loader
========================================
Loads EURUSD Daily (or any OHLCV) data from CSV files into
List[CandleData] ready for PipelineRunner.run().

Supported CSV formats
---------------------
Minimum required columns (case-insensitive):

    date, open, high, low, close

Optional columns (silently ignored if absent):

    volume, tick_volume, spread

Any extra columns are ignored.

Date column formats accepted
----------------------------
    YYYY-MM-DD
    YYYY.MM.DD
    YYYY/MM/DD
    DD-MM-YYYY
    DD/MM/YYYY
    MM/DD/YYYY
    YYYY-MM-DD HH:MM  (date + time in same cell)
    YYYY-MM-DD HH:MM:SS

Output ordering
---------------
Candles are returned oldest-first regardless of source ordering.

Data quality
------------
Every load produces a DataQualityReport documenting:
  - total_candles loaded
  - duplicate timestamps removed
  - rows with invalid OHLC skipped
  - rows that could not be parsed skipped
  - date range (first_date → last_date)
"""

from __future__ import annotations

import csv
import io
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple, Union

from src.data.types import CandleData

logger = logging.getLogger("candlestickbot.backtesting.data_loader")

# ---------------------------------------------------------------------------
# Date format strings tried in order
# ---------------------------------------------------------------------------
_DATE_FORMATS: List[str] = [
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
    "%Y.%m.%d %H:%M:%S",
    "%Y.%m.%d %H:%M",
    "%Y.%m.%d",
    "%Y/%m/%d %H:%M:%S",
    "%Y/%m/%d %H:%M",
    "%Y/%m/%d",
    "%d-%m-%Y %H:%M:%S",
    "%d-%m-%Y %H:%M",
    "%d-%m-%Y",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M",
    "%d/%m/%Y",
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M",
    "%m/%d/%Y",
]

# Canonical column aliases (lower-cased key → field name)
_COL_ALIASES = {
    # date
    "date": "date", "datetime": "date", "time": "date",
    "timestamp": "date",
    # open
    "open": "open", "o": "open",
    # high
    "high": "high", "h": "high",
    # low
    "low": "low", "l": "low",
    # close
    "close": "close", "c": "close",
    # volume
    "volume": "volume", "vol": "volume", "v": "volume",
    "tick_volume": "volume", "tickvolume": "volume",
}

_REQUIRED_FIELDS = {"date", "open", "high", "low", "close"}


# ---------------------------------------------------------------------------
# DataQualityReport
# ---------------------------------------------------------------------------

@dataclass
class DataQualityReport:
    """
    Summary of data quality produced by every load_candles_from_csv() call.

    Fields
    ------
    symbol          : Instrument label (passed by caller, not in CSV).
    timeframe       : Timeframe label  (passed by caller, not in CSV).
    total_candles   : Number of candles in the returned list.
    duplicate_candles: Rows removed because their timestamp already existed.
    invalid_rows    : Rows skipped due to bad OHLC or unparseable values.
    missing_candles : Estimated missing business-day gaps (0 if timeframe ≠ 'D1').
    start_date      : Timestamp of the oldest candle returned.
    end_date        : Timestamp of the newest candle returned.
    source_path     : File path or '<string>' if loaded from a string buffer.
    warnings        : Non-fatal advisory messages.
    """
    symbol:           str
    timeframe:        str
    total_candles:    int              = 0
    duplicate_candles: int             = 0
    invalid_rows:     int              = 0
    missing_candles:  int              = 0
    start_date:       Optional[datetime] = None
    end_date:         Optional[datetime] = None
    source_path:      str              = "<unknown>"
    warnings:         List[str]        = field(default_factory=list)

    # ---- derived helpers ----

    @property
    def is_clean(self) -> bool:
        """True when no duplicates, invalids, or missings were found."""
        return (
            self.duplicate_candles == 0
            and self.invalid_rows == 0
            and self.missing_candles == 0
        )

    @property
    def coverage_days(self) -> int:
        """Calendar days between first and last candle (0 if < 2 candles)."""
        if self.start_date and self.end_date:
            return (self.end_date - self.start_date).days
        return 0

    def summary(self) -> str:
        """Return a compact one-block text summary."""
        sep = "─" * 50
        lines = [
            sep,
            f"  DATA QUALITY REPORT — {self.symbol} {self.timeframe}",
            sep,
            f"  Source       : {self.source_path}",
            f"  Total candles: {self.total_candles}",
            f"  Date range   : {_fmt_dt(self.start_date)} → {_fmt_dt(self.end_date)}",
            f"  Coverage     : {self.coverage_days} calendar days",
            f"  Duplicates   : {self.duplicate_candles}",
            f"  Invalid rows : {self.invalid_rows}",
            f"  Missing gaps : {self.missing_candles}",
            f"  Status       : {'✅ CLEAN' if self.is_clean else '⚠️  ISSUES FOUND'}",
        ]
        if self.warnings:
            lines.append("  Warnings     :")
            for w in self.warnings:
                lines.append(f"    • {w}")
        lines.append(sep)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_candles_from_csv(
    source:    Union[str, Path, io.StringIO],
    symbol:    str = "EURUSD",
    timeframe: str = "D1",
    start_date: Optional[datetime] = None,
    end_date:   Optional[datetime] = None,
) -> Tuple[List[CandleData], DataQualityReport]:
    """
    Load OHLCV candles from a CSV file (or in-memory StringIO buffer).

    Parameters
    ----------
    source    : File path (str or Path) or io.StringIO buffer.
    symbol    : Instrument label attached to every CandleData row.
    timeframe : Timeframe label attached to every CandleData row.
    start_date: If given, only candles with timestamp >= start_date are kept.
    end_date  : If given, only candles with timestamp <= end_date are kept.

    Returns
    -------
    (candles, report) where candles is sorted oldest-first and report
    documents all data-quality issues encountered.

    Raises
    ------
    ValueError  : If the CSV has no rows, cannot be read, or is missing
                  required columns (date/open/high/low/close).
    """
    report = DataQualityReport(symbol=symbol, timeframe=timeframe)

    # ---- open source ----
    if isinstance(source, io.StringIO):
        report.source_path = "<string>"
        reader, raw_rows = _read_stringio(source)
    else:
        path = Path(source)
        report.source_path = str(path)
        if not path.exists():
            raise ValueError(f"CSV file not found: {path}")
        raw_rows = _read_file(path)

    if not raw_rows:
        raise ValueError(f"CSV source is empty: {report.source_path}")

    # ---- map header → canonical field names ----
    header = raw_rows[0]
    col_map = _build_col_map(header)
    missing_req = _REQUIRED_FIELDS - set(col_map.keys())
    if missing_req:
        raise ValueError(
            f"CSV is missing required columns: {missing_req}. "
            f"Found: {list(header)}"
        )

    # ---- parse rows ----
    candles: List[CandleData] = []
    seen_timestamps: dict = {}   # ts → row index, for duplicate detection
    invalid_rows = 0

    for row_idx, row in enumerate(raw_rows[1:], start=2):
        try:
            ts    = _parse_date(row[col_map["date"]])
            o     = float(row[col_map["open"]])
            h     = float(row[col_map["high"]])
            l     = float(row[col_map["low"]])
            c     = float(row[col_map["close"]])
            vol   = float(row[col_map["volume"]]) if "volume" in col_map else 0.0
        except (ValueError, KeyError, IndexError) as exc:
            logger.debug("CSV row %d skipped: %s", row_idx, exc)
            invalid_rows += 1
            continue

        # OHLC sanity checks
        if not _ohlc_valid(o, h, l, c):
            logger.debug("CSV row %d invalid OHLC: O=%s H=%s L=%s C=%s",
                         row_idx, o, h, l, c)
            invalid_rows += 1
            continue

        # Date range filter
        if start_date and ts < start_date:
            continue
        if end_date and ts > end_date:
            continue

        # Duplicate detection
        if ts in seen_timestamps:
            report.duplicate_candles += 1
            logger.debug("CSV row %d duplicate timestamp %s (first at row %d)",
                         row_idx, ts, seen_timestamps[ts])
            continue
        seen_timestamps[ts] = row_idx

        candles.append(CandleData(
            timestamp=ts,
            open=o, high=h, low=l, close=c,
            volume=vol,
            symbol=symbol.upper(),
            timeframe=timeframe.upper(),
        ))

    report.invalid_rows = invalid_rows

    if not candles:
        report.total_candles = 0
        return candles, report

    # Sort oldest-first
    candles.sort(key=lambda cd: cd.timestamp)

    report.total_candles = len(candles)
    report.start_date    = candles[0].timestamp
    report.end_date      = candles[-1].timestamp

    # Estimate missing gaps for D1 data
    if timeframe.upper() == "D1":
        report.missing_candles = _estimate_missing_d1(candles)
        if report.missing_candles > 0:
            report.warnings.append(
                f"{report.missing_candles} estimated missing trading-day gap(s) detected"
            )

    logger.info(
        "Loaded %d candles for %s %s from %s (dupes=%d invalid=%d missing=%d)",
        report.total_candles, symbol, timeframe, report.source_path,
        report.duplicate_candles, report.invalid_rows, report.missing_candles,
    )
    return candles, report


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _read_file(path: Path) -> List[List[str]]:
    """Read CSV file → list of string rows (including header)."""
    rows: List[List[str]] = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if any(cell.strip() for cell in row):   # skip blank lines
                rows.append([c.strip() for c in row])
    return rows


def _read_stringio(buf: io.StringIO) -> Tuple[None, List[List[str]]]:
    """Read StringIO buffer → list of string rows (including header)."""
    buf.seek(0)
    rows: List[List[str]] = []
    reader = csv.reader(buf)
    for row in reader:
        if any(cell.strip() for cell in row):
            rows.append([c.strip() for c in row])
    return None, rows


def _build_col_map(header: List[str]) -> dict:
    """
    Map column indices by matching header names to canonical field names.

    Returns {canonical_name: column_index}.
    """
    col_map: dict = {}
    for idx, name in enumerate(header):
        canonical = _COL_ALIASES.get(name.strip().lower())
        if canonical and canonical not in col_map:
            col_map[canonical] = idx
    return col_map


def _parse_date(raw: str) -> datetime:
    """Try all date formats and return a UTC-aware datetime. Raises ValueError if none match."""
    raw = raw.strip()
    for fmt in _DATE_FORMATS:
        try:
            dt = datetime.strptime(raw, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse date: {raw!r}")


def _ohlc_valid(o: float, h: float, l: float, c: float) -> bool:
    """Return True when OHLC values are internally consistent and positive."""
    if any(v <= 0 for v in (o, h, l, c)):
        return False
    if h < o or h < c:
        return False
    if l > o or l > c:
        return False
    if l > h:
        return False
    return True


def _estimate_missing_d1(candles: List[CandleData]) -> int:
    """
    Count date gaps larger than 3 calendar days between consecutive candles
    (allowing for weekends + typical holidays) as probable missing data.
    Only meaningful for D1 data.
    """
    if len(candles) < 2:
        return 0
    missing = 0
    for a, b in zip(candles, candles[1:]):
        gap = (b.timestamp - a.timestamp).days
        # >3 days suggests a missing candle (weekend = 2, holiday = 3)
        if gap > 3:
            missing += gap // 5   # crude business-day estimate
    return missing


def _fmt_dt(dt: Optional[datetime]) -> str:
    if dt is None:
        return "N/A"
    return dt.strftime("%Y-%m-%d")
