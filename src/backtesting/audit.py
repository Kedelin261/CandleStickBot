"""
Sprint 14 — Data Quality Audit
================================
Provides DataQualityAuditReport: a richer, file-level audit of EURUSD D1
CSV data that goes beyond the per-load DataQualityReport produced by
data_loader.py.

Differences from DataQualityReport
------------------------------------
DataQualityReport is produced at load time and tracks per-candle issues
(dupes, invalid OHLC, estimated missing gaps).

DataQualityAuditReport is produced at audit time and adds:
  - raw file row count (before any filtering)
  - weekend-gap count (expected non-trading days vs actual gaps)
  - chronological-order check
  - required-column presence check
  - OHLC consistency pass rate
  - per-field null counts
  - explicit pass/fail gate (valid_rows_pct >= 99%)

Public API
----------
audit_csv(path, symbol, timeframe) -> DataQualityAuditReport
save_audit_report(report, path)     -> None
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from src.backtesting.data_loader import (
    _build_col_map,
    _ohlc_valid,
    _parse_date,
    _REQUIRED_FIELDS,
)


# ---------------------------------------------------------------------------
# DataQualityAuditReport
# ---------------------------------------------------------------------------

@dataclass
class DataQualityAuditReport:
    """
    Full file-level data quality audit for a EURUSD D1 CSV.

    Fields
    ------
    file_name          : Basename of the audited file.
    symbol             : Instrument label (caller-supplied).
    timeframe          : Timeframe label (caller-supplied).
    total_rows         : All data rows in the file (excluding header).
    valid_rows         : Rows that passed every quality check.
    invalid_rows       : Rows that failed at least one check.
    duplicate_rows     : Rows with a repeated timestamp (counted in invalid_rows).
    missing_dates      : Estimated missing business-day gaps (D1 only).
    weekend_gaps       : Number of Saturday/Sunday gap transitions detected.
    date_range         : (start_date, end_date) of valid candles.
    has_required_cols  : True when all five required columns are present.
    chronological      : True when timestamps are strictly ordered.
    ohlc_pass_rate     : Fraction of rows that pass OHLC validation.
    null_counts        : Dict of column → count of null/empty cells.
    warnings           : Non-fatal advisory messages.
    errors             : Fatal structural issues (empty file, missing cols …).
    """

    file_name:         str
    symbol:            str
    timeframe:         str
    total_rows:        int                           = 0
    valid_rows:        int                           = 0
    invalid_rows:      int                           = 0
    duplicate_rows:    int                           = 0
    missing_dates:     int                           = 0
    weekend_gaps:      int                           = 0
    date_range:        tuple                         = (None, None)
    has_required_cols: bool                          = False
    chronological:     bool                          = True
    ohlc_pass_rate:    float                         = 0.0
    null_counts:       dict                          = field(default_factory=dict)
    warnings:          List[str]                     = field(default_factory=list)
    errors:            List[str]                     = field(default_factory=list)

    # ---- derived ----

    @property
    def valid_rows_pct(self) -> float:
        """Percentage of rows that are valid (0–100)."""
        return (self.valid_rows / self.total_rows * 100.0) if self.total_rows else 0.0

    @property
    def invalid_rows_pct(self) -> float:
        return 100.0 - self.valid_rows_pct

    @property
    def passes_quality_gate(self) -> bool:
        """
        Backtest may proceed only when:
          - valid_rows_pct >= 99%
          - invalid_rows_pct <= 1%
          - no fatal errors
          - has_required_cols is True
        """
        return (
            self.has_required_cols
            and len(self.errors) == 0
            and self.valid_rows_pct >= 99.0
        )

    @property
    def coverage_years(self) -> float:
        start, end = self.date_range
        if start and end:
            return (end - start).days / 365.25
        return 0.0

    def summary(self) -> str:
        sep = "═" * 60
        lines = [
            sep,
            f"  DATA QUALITY AUDIT REPORT — {self.symbol} {self.timeframe}",
            sep,
            f"  File           : {self.file_name}",
            f"  Symbol         : {self.symbol}",
            f"  Timeframe      : {self.timeframe}",
            f"  Date range     : {_fmt_dt(self.date_range[0])} → {_fmt_dt(self.date_range[1])}",
            f"  Coverage       : {self.coverage_years:.2f} years",
            sep,
            f"  Total rows     : {self.total_rows}",
            f"  Valid rows     : {self.valid_rows}  ({self.valid_rows_pct:.2f}%)",
            f"  Invalid rows   : {self.invalid_rows}  ({self.invalid_rows_pct:.2f}%)",
            f"  Duplicate rows : {self.duplicate_rows}",
            f"  Missing dates  : {self.missing_dates} estimated gaps",
            f"  Weekend gaps   : {self.weekend_gaps}",
            sep,
            f"  Has req cols   : {'✅ YES' if self.has_required_cols else '❌ NO'}",
            f"  Chronological  : {'✅ YES' if self.chronological else '❌ OUT OF ORDER'}",
            f"  OHLC pass rate : {self.ohlc_pass_rate * 100:.2f}%",
            sep,
        ]

        if self.null_counts:
            lines.append("  Null counts    :")
            for col, cnt in self.null_counts.items():
                lines.append(f"    {col:10s} : {cnt}")
            lines.append(sep)

        if self.warnings:
            lines.append("  Warnings       :")
            for w in self.warnings:
                lines.append(f"    ⚠️  {w}")
            lines.append(sep)

        if self.errors:
            lines.append("  Errors         :")
            for e in self.errors:
                lines.append(f"    ❌ {e}")
            lines.append(sep)

        gate_status = "✅ PASS — backtesting may proceed" if self.passes_quality_gate \
                      else "❌ FAIL — data quality insufficient for backtesting"
        lines += [
            f"  Quality gate   : {gate_status}",
            sep,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def audit_csv(
    path:      str | Path,
    symbol:    str = "EURUSD",
    timeframe: str = "D1",
) -> DataQualityAuditReport:
    """
    Perform a full data quality audit on a CSV file.

    Returns a DataQualityAuditReport. Errors are accumulated in
    report.errors; they do not raise exceptions (so callers can inspect
    the full report before deciding whether to proceed).
    """
    path = Path(path)
    report = DataQualityAuditReport(
        file_name=path.name,
        symbol=symbol.upper(),
        timeframe=timeframe.upper(),
    )

    # ---- read file ----
    if not path.exists():
        report.errors.append(f"File not found: {path}")
        return report

    try:
        raw_rows = _read_csv_rows(path)
    except Exception as exc:
        report.errors.append(f"Cannot read file: {exc}")
        return report

    if len(raw_rows) < 2:
        report.errors.append("File is empty or contains only a header row.")
        return report

    header  = raw_rows[0]
    data_rows = raw_rows[1:]
    report.total_rows = len(data_rows)

    # ---- column presence ----
    col_map = _build_col_map(header)
    missing = _REQUIRED_FIELDS - set(col_map.keys())
    if missing:
        report.has_required_cols = False
        report.errors.append(
            f"Missing required columns: {sorted(missing)}. "
            f"Found: {list(header)}"
        )
        return report
    report.has_required_cols = True

    # ---- per-row analysis ----
    null_counts: dict[str, int] = {k: 0 for k in col_map}
    valid_timestamps: list[datetime] = []
    seen_ts: set[datetime] = set()
    ohlc_ok = 0
    ohlc_total = 0
    invalid = 0
    prev_ts: Optional[datetime] = None
    is_chrono = True

    for row in data_rows:
        # null / empty cells per canonical column
        for canon, idx in col_map.items():
            if idx >= len(row) or not row[idx].strip():
                null_counts[canon] = null_counts.get(canon, 0) + 1

        # parse date
        try:
            date_raw = row[col_map["date"]] if col_map["date"] < len(row) else ""
            ts = _parse_date(date_raw.strip())
        except (ValueError, IndexError):
            invalid += 1
            continue

        # parse OHLC
        try:
            o = float(row[col_map["open"]])
            h = float(row[col_map["high"]])
            l = float(row[col_map["low"]])
            c = float(row[col_map["close"]])
        except (ValueError, IndexError):
            invalid += 1
            continue

        # duplicate detection
        if ts in seen_ts:
            report.duplicate_rows += 1
            invalid += 1
            continue
        seen_ts.add(ts)

        # chronological order
        if prev_ts and ts < prev_ts:
            is_chrono = False
        prev_ts = ts

        # OHLC validation
        ohlc_total += 1
        if _ohlc_valid(o, h, l, c):
            ohlc_ok += 1
        else:
            invalid += 1
            continue

        valid_timestamps.append(ts)

    report.invalid_rows    = invalid
    report.valid_rows      = max(0, report.total_rows - invalid)
    report.chronological   = is_chrono
    report.ohlc_pass_rate  = ohlc_ok / ohlc_total if ohlc_total else 0.0
    report.null_counts     = {k: v for k, v in null_counts.items() if v > 0}

    if valid_timestamps:
        valid_timestamps.sort()
        report.date_range = (valid_timestamps[0], valid_timestamps[-1])

        # Weekend gap count (gaps of exactly 2 days = Fri→Mon, expected)
        # and missing-date estimation for D1
        report.weekend_gaps  = _count_weekend_gaps(valid_timestamps)
        if timeframe.upper() == "D1":
            report.missing_dates = _estimate_missing_d1_gaps(valid_timestamps)
            if report.missing_dates > 0:
                report.warnings.append(
                    f"{report.missing_dates} estimated missing trading-day gap(s) detected"
                )

    if not is_chrono:
        report.warnings.append("Timestamps are not in chronological order")

    if report.duplicate_rows > 0:
        report.warnings.append(f"{report.duplicate_rows} duplicate timestamp(s) removed")

    if report.valid_rows_pct < 99.0:
        report.warnings.append(
            f"Valid row percentage ({report.valid_rows_pct:.2f}%) is below 99% threshold"
        )

    return report


def save_audit_report(report: DataQualityAuditReport, path: str | Path) -> None:
    """Write the audit report summary to a text file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.summary(), encoding="utf-8")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _read_csv_rows(path: Path) -> list[list[str]]:
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if any(cell.strip() for cell in row):
                rows.append([c.strip() for c in row])
    return rows


def _count_weekend_gaps(timestamps: list[datetime]) -> int:
    """Count gaps of exactly 2–3 calendar days (Fri→Mon or holiday bridges)."""
    count = 0
    for a, b in zip(timestamps, timestamps[1:]):
        gap = (b - a).days
        if 2 <= gap <= 3:
            count += 1
    return count


def _estimate_missing_d1_gaps(timestamps: list[datetime]) -> int:
    """
    Count gaps > 3 calendar days as probable missing candles.
    Excludes normal weekend gaps (2 days) and extended holiday bridges (3 days).
    Returns a rough business-day estimate.
    """
    if len(timestamps) < 2:
        return 0
    missing = 0
    for a, b in zip(timestamps, timestamps[1:]):
        gap = (b - a).days
        if gap > 3:
            missing += max(0, gap // 5)
    return missing


def _fmt_dt(dt: Optional[datetime]) -> str:
    if dt is None:
        return "N/A"
    return dt.strftime("%Y-%m-%d")
