"""
Sprint 16 — Data Authenticity Gate
====================================
Permanent, tested gate that classifies a EURUSD D1 CSV dataset as:

    VERIFIED_REAL         — passes all hard-fail checks; safe to use for verdicts
    SYNTHETIC_SUSPECT     — one or more hard-fail signals triggered
    MIXED_RECONSTRUCTED   — no hard-fail but multiple coincident soft warnings

The gate is called BEFORE any backtest run that will produce a verdict.
Runs that use the legacy file for Sprint 14/15 reproducibility must pass
``override_authenticity=True`` (loudly labelled in the run report).

Hard-fail signals (any ONE → SYNTHETIC_SUSPECT)
------------------------------------------------
H1  Exact doji rate ≥ 2 %  (open == close to 5-decimal precision)
H2  Calendar-impossible bars: Saturday, Sunday, or named fixed holidays
    (Christmas Day Dec 25, New Year's Day Jan 1)
H3  Future-dated bars relative to ``run_date`` (defaults to today)
H4  Non-chronological rows, duplicate timestamps, or any OHLC-inconsistent
    rows (these are already caught by the existing loader/audit; we re-check
    here for completeness)
H5  Spot-check failure: ≥ 8 reference closes provided AND more than 2 of
    them deviate by > 30 pips from the file's close on the same date

Soft signals (warn + record, never auto-fail)
---------------------------------------------
S1  Zero or absent volume column (suspicious only in combination)
S2  Perfectly uniform day-of-week distribution  (χ² p-value proxy: all
    DOW counts within ±2 of the mean)
S3  Fewer than ~3 holiday gaps per year (real feeds have ~8–12)
    — computed as: expected_holidays_per_year = (missing_non-weekend_gaps / years)

Output
------
DataAuthenticityReport  — dataclass with all metrics + classification
classify_dataset()      — main public function
save_authenticity_report() — writes .txt to disk
run_authenticity_check()   — convenience: loads CSV, classifies, saves report

Tests
-----
See tests/backtesting/test_sprint16_data_authenticity.py
"""

from __future__ import annotations

import csv
import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from src.backtesting.data_loader import _build_col_map, _ohlc_valid, _parse_date
from src.data.types import CandleData

logger = logging.getLogger("candlestickbot.backtesting.data_authenticity")

# ---------------------------------------------------------------------------
# Classification constants
# ---------------------------------------------------------------------------

VERIFIED_REAL       = "VERIFIED_REAL"
SYNTHETIC_SUSPECT   = "SYNTHETIC_SUSPECT"
MIXED_RECONSTRUCTED = "MIXED_RECONSTRUCTED"

# Hard-fail thresholds
_DOJI_RATE_THRESHOLD   = 0.02   # 2% — any more than this → H1 fail
_SPOT_CHECK_PIP_TOL    = 30     # pips — deviation tolerance for spot checks
_SPOT_CHECK_MAX_FAILS  = 2      # more than this many spot-check fails → H5 fail
_PIP_SIZE              = 0.0001 # EURUSD pip

# Fixed market holidays that should NEVER appear as trading bars
# (date-part only: month, day)
_HARD_HOLIDAYS: set[tuple[int, int]] = {
    (12, 25),  # Christmas Day
    (1, 1),    # New Year's Day
}

# Soft-signal uniformity: all DOW counts within this absolute tolerance of mean
_DOW_UNIFORM_TOLERANCE = 2


# ---------------------------------------------------------------------------
# DataAuthenticityReport
# ---------------------------------------------------------------------------

@dataclass
class DataAuthenticityReport:
    """
    Full authenticity classification for a EURUSD D1 CSV dataset.

    Fields
    ------
    file_name          : Basename of the audited file.
    classification     : VERIFIED_REAL | SYNTHETIC_SUSPECT | MIXED_RECONSTRUCTED
    run_date           : The date used for future-bar detection.

    ── Hard-fail signals ───────────────────────────────────────────────────────
    h1_doji_rate        : Fraction of rows where open == close (5-dp equality).
    h1_triggered        : True when h1_doji_rate >= 2%.
    h2_calendar_bars    : List of date strings for impossible calendar bars.
    h2_triggered        : True when any calendar-impossible bars found.
    h3_future_bars      : List of date strings for future-dated bars.
    h3_triggered        : True when any future-dated bars found.
    h4_structural       : True when non-chrono / duplicate / OHLC errors found.
    h4_triggered        : Same as h4_structural (H4 = structural integrity).
    h5_spot_check_fails : Number of spot-check references that deviate > 30 pips.
    h5_triggered        : True when spot-checks provided AND > 2 fail.
    hard_fails          : List of triggered hard-fail codes (e.g. ["H1","H2"]).

    ── Soft signals ────────────────────────────────────────────────────────────
    s1_zero_volume_pct  : Fraction of rows with volume == 0 (or absent).
    s1_triggered        : True when s1_zero_volume_pct == 100%.
    s2_dow_distribution : Dict day_name → count.
    s2_triggered        : True when all DOW counts within ±2 of mean (too uniform).
    s3_holiday_gaps_per_year : Estimated non-weekend gaps per year.
    s3_triggered        : True when < 3 per year (too few holiday gaps).
    soft_warnings       : List of triggered soft-signal descriptions.

    ── Summary statistics ──────────────────────────────────────────────────────
    total_rows          : Total data rows analysed.
    date_range          : (start_date, end_date) as date objects.
    coverage_years      : Float coverage of the dataset.
    spot_checks_run     : Number of spot-check references provided.
    """

    file_name:             str
    classification:        str                  = SYNTHETIC_SUSPECT
    run_date:              date                 = field(default_factory=date.today)

    # Hard-fail signals
    h1_doji_rate:          float                = 0.0
    h1_triggered:          bool                 = False
    h2_calendar_bars:      List[str]            = field(default_factory=list)
    h2_triggered:          bool                 = False
    h3_future_bars:        List[str]            = field(default_factory=list)
    h3_triggered:          bool                 = False
    h4_structural:         bool                 = False
    h4_triggered:          bool                 = False
    h5_spot_check_fails:   int                  = 0
    h5_triggered:          bool                 = False
    hard_fails:            List[str]            = field(default_factory=list)

    # Soft signals
    s1_zero_volume_pct:    float                = 0.0
    s1_triggered:          bool                 = False
    s2_dow_distribution:   Dict[str, int]       = field(default_factory=dict)
    s2_triggered:          bool                 = False
    s3_holiday_gaps_per_year: float             = 0.0
    s3_triggered:          bool                 = False
    soft_warnings:         List[str]            = field(default_factory=list)

    # Summary stats
    total_rows:            int                  = 0
    date_range:            Tuple                = (None, None)
    coverage_years:        float                = 0.0
    spot_checks_run:       int                  = 0

    def summary(self) -> str:
        """Return a human-readable summary of the authenticity report."""
        sep  = "═" * 68
        sep2 = "─" * 68
        lines = [
            sep,
            f"  DATA AUTHENTICITY REPORT — Sprint 16",
            sep,
            f"  File           : {self.file_name}",
            f"  Run date       : {self.run_date}",
            f"  Total rows     : {self.total_rows}",
            f"  Date range     : {_fmt_date(self.date_range[0])} → "
            f"{_fmt_date(self.date_range[1])}",
            f"  Coverage       : {self.coverage_years:.2f} years",
            sep2,
            "  ── HARD-FAIL SIGNALS ─────────────────────────────────────────",
            sep2,
            f"  H1 Doji rate   : {self.h1_doji_rate*100:.2f}%  "
            f"(threshold ≥ 2%)  "
            f"{'❌ TRIGGERED' if self.h1_triggered else '✅ OK'}",
            f"  H2 Calendar    : {len(self.h2_calendar_bars)} impossible bar(s)  "
            f"{'❌ TRIGGERED' if self.h2_triggered else '✅ OK'}",
        ]
        if self.h2_calendar_bars:
            lines.append(f"    → {', '.join(self.h2_calendar_bars[:5])}"
                         + (" …" if len(self.h2_calendar_bars) > 5 else ""))
        lines += [
            f"  H3 Future bars : {len(self.h3_future_bars)} bar(s)  "
            f"{'❌ TRIGGERED' if self.h3_triggered else '✅ OK'}",
        ]
        if self.h3_future_bars:
            lines.append(f"    → {', '.join(self.h3_future_bars[:3])}")
        lines += [
            f"  H4 Structural  : {'❌ TRIGGERED' if self.h4_triggered else '✅ OK'}  "
            f"(chrono/dupe/OHLC errors)",
            f"  H5 Spot-checks : {self.h5_spot_check_fails}/{self.spot_checks_run} "
            f"fail(s)  (threshold > 2)  "
            f"{'❌ TRIGGERED' if self.h5_triggered else '✅ OK' if self.spot_checks_run > 0 else 'N/A (no refs)'}",
            sep2,
            "  ── SOFT SIGNALS ──────────────────────────────────────────────",
            sep2,
            f"  S1 Zero volume : {self.s1_zero_volume_pct*100:.1f}%  "
            f"{'⚠️  WARN' if self.s1_triggered else '✅ OK'}",
            f"  S2 DOW uniform : {'⚠️  WARN (suspiciously uniform)' if self.s2_triggered else '✅ OK'}",
        ]
        if self.s2_dow_distribution:
            dow_str = "  ".join(
                f"{k[:3]}={v}" for k, v in sorted(self.s2_dow_distribution.items())
            )
            lines.append(f"    Distribution  : {dow_str}")
        lines += [
            f"  S3 Holiday gaps: {self.s3_holiday_gaps_per_year:.1f}/yr  "
            f"(expect ~6–12)  "
            f"{'⚠️  WARN' if self.s3_triggered else '✅ OK'}",
            sep2,
        ]
        if self.soft_warnings:
            lines.append("  Soft warnings  :")
            for w in self.soft_warnings:
                lines.append(f"    ⚠️  {w}")
            lines.append(sep2)

        # Verdict box
        verdict_icon = {
            VERIFIED_REAL:       "✅",
            SYNTHETIC_SUSPECT:   "❌",
            MIXED_RECONSTRUCTED: "⚠️ ",
        }.get(self.classification, "?")
        lines += [
            f"  CLASSIFICATION : {verdict_icon}  {self.classification}",
        ]
        if self.hard_fails:
            lines.append(f"  Hard fails     : {', '.join(self.hard_fails)}")
        lines += [
            sep,
            "",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def classify_dataset(
    candles: List[CandleData],
    *,
    file_name:    str  = "<unknown>",
    run_date:     Optional[date] = None,
    spot_checks:  Optional[Sequence[Tuple[str, float]]] = None,
    has_volume:   bool = True,
) -> DataAuthenticityReport:
    """
    Classify a loaded candle list for data authenticity.

    Parameters
    ----------
    candles      : List[CandleData] already loaded via load_candles_from_csv().
    file_name    : Basename label for the report.
    run_date     : Reference date for future-bar detection (default: today).
    spot_checks  : Optional sequence of (date_str, reference_close) pairs.
                   date_str format: 'YYYY-MM-DD'. reference_close in price units.
                   If provided (and ≥ 8 pairs), H5 check activates.
    has_volume   : Whether the source CSV had a volume column. If False,
                   S1 is noted as absent-volume rather than zero-volume.

    Returns
    -------
    DataAuthenticityReport with classification and all component results.
    """
    rpt = DataAuthenticityReport(
        file_name=file_name,
        run_date=run_date or date.today(),
    )

    if not candles:
        rpt.hard_fails.append("EMPTY")
        rpt.classification = SYNTHETIC_SUSPECT
        return rpt

    rpt.total_rows = len(candles)
    start_dt = candles[0].timestamp.date()
    end_dt   = candles[-1].timestamp.date()
    rpt.date_range = (start_dt, end_dt)
    rpt.coverage_years = (end_dt - start_dt).days / 365.25

    # ── H1: Exact doji rate ─────────────────────────────────────────────────
    exact_dojis = sum(
        1 for c in candles if _is_exact_doji(c.open, c.close)
    )
    rpt.h1_doji_rate = exact_dojis / rpt.total_rows
    if rpt.h1_doji_rate >= _DOJI_RATE_THRESHOLD:
        rpt.h1_triggered = True
        rpt.hard_fails.append("H1")

    # ── H2: Calendar-impossible bars ────────────────────────────────────────
    impossible = []
    for c in candles:
        d = c.timestamp.date()
        if d.weekday() >= 5:   # Saturday=5, Sunday=6
            impossible.append(str(d))
        elif (d.month, d.day) in _HARD_HOLIDAYS:
            impossible.append(str(d))
    rpt.h2_calendar_bars = impossible
    if impossible:
        rpt.h2_triggered = True
        rpt.hard_fails.append("H2")

    # ── H3: Future-dated bars ───────────────────────────────────────────────
    future = [
        str(c.timestamp.date())
        for c in candles
        if c.timestamp.date() > rpt.run_date
    ]
    rpt.h3_future_bars = future
    if future:
        rpt.h3_triggered = True
        rpt.hard_fails.append("H3")

    # ── H4: Structural integrity ─────────────────────────────────────────────
    h4_fail = False
    seen_ts: set = set()
    prev_ts: Optional[datetime] = None
    for c in candles:
        ts = c.timestamp
        # duplicate
        if ts in seen_ts:
            h4_fail = True
            break
        seen_ts.add(ts)
        # non-chronological
        if prev_ts and ts < prev_ts:
            h4_fail = True
            break
        prev_ts = ts
        # OHLC consistency
        if not _ohlc_valid(c.open, c.high, c.low, c.close):
            h4_fail = True
            break
    rpt.h4_structural = h4_fail
    rpt.h4_triggered  = h4_fail
    if h4_fail:
        rpt.hard_fails.append("H4")

    # ── H5: Spot-check reference closes ─────────────────────────────────────
    if spot_checks:
        rpt.spot_checks_run = len(spot_checks)
        date_map = {c.timestamp.date(): c for c in candles}
        fails = 0
        for date_str, ref_close in spot_checks:
            try:
                d = datetime.strptime(date_str, "%Y-%m-%d").date()
            except ValueError:
                continue
            if d not in date_map:
                continue  # date not in dataset — skip (not a fail)
            candle = date_map[d]
            deviation_pips = abs(candle.close - ref_close) / _PIP_SIZE
            if deviation_pips > _SPOT_CHECK_PIP_TOL:
                fails += 1
        rpt.h5_spot_check_fails = fails
        # Activate H5 only when we have ≥ 8 reference points AND > 2 fail
        if rpt.spot_checks_run >= 8 and fails > _SPOT_CHECK_MAX_FAILS:
            rpt.h5_triggered = True
            rpt.hard_fails.append("H5")

    # ── S1: Volume ──────────────────────────────────────────────────────────
    if not has_volume:
        rpt.s1_zero_volume_pct = 1.0
        rpt.s1_triggered = True
        rpt.soft_warnings.append(
            "Volume column absent — cannot verify tick/dollar activity"
        )
    else:
        zero_vol = sum(1 for c in candles if c.volume == 0.0)
        rpt.s1_zero_volume_pct = zero_vol / rpt.total_rows
        if rpt.s1_zero_volume_pct == 1.0:
            rpt.s1_triggered = True
            rpt.soft_warnings.append(
                f"100% zero volume — definitive indicator of synthetic/reconstructed data"
            )
        elif rpt.s1_zero_volume_pct > 0.5:
            rpt.soft_warnings.append(
                f"{rpt.s1_zero_volume_pct*100:.1f}% zero volume rows (>50%)"
            )

    # ── S2: DOW distribution ────────────────────────────────────────────────
    dow_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    dow_counts: Dict[str, int] = {d: 0 for d in dow_names}
    for c in candles:
        name = c.timestamp.strftime("%A")
        if name in dow_counts:
            dow_counts[name] += 1
    rpt.s2_dow_distribution = {k: dow_counts[k] for k in dow_names}

    counts = [dow_counts[d] for d in dow_names if dow_counts[d] > 0]
    if counts:
        mean_count = sum(counts) / len(counts)
        max_deviation = max(abs(v - mean_count) for v in counts)
        if max_deviation <= _DOW_UNIFORM_TOLERANCE:
            rpt.s2_triggered = True
            rpt.soft_warnings.append(
                f"Suspiciously uniform DOW distribution (max deviation "
                f"{max_deviation:.1f} from mean {mean_count:.1f}) — "
                f"real feeds show non-uniform patterns from holiday adjustments"
            )

    # ── S3: Holiday gaps per year ────────────────────────────────────────────
    # Count inter-bar gaps > 3 days AND not a full weekend (> 5 days would skip
    # a week).  Gaps of exactly 4 or 5 days = Mon–Fri holiday bridge.
    holiday_gap_count = 0
    for a, b in zip(candles, candles[1:]):
        gap = (b.timestamp - a.timestamp).days
        if 4 <= gap <= 7:   # extended weekend through a holiday
            holiday_gap_count += 1
    rpt.s3_holiday_gaps_per_year = (
        holiday_gap_count / rpt.coverage_years if rpt.coverage_years > 0 else 0.0
    )
    if rpt.s3_holiday_gaps_per_year < 3.0 and rpt.coverage_years >= 1.0:
        rpt.s3_triggered = True
        rpt.soft_warnings.append(
            f"Only {rpt.s3_holiday_gaps_per_year:.1f} holiday gaps/year detected "
            f"(expect ~6–12) — real feeds have gaps for major market holidays"
        )

    # ── Final classification ─────────────────────────────────────────────────
    if rpt.hard_fails:
        rpt.classification = SYNTHETIC_SUSPECT
    elif len(rpt.soft_warnings) >= 3:
        # Multiple coincident soft warnings → MIXED at best
        rpt.classification = MIXED_RECONSTRUCTED
    else:
        rpt.classification = VERIFIED_REAL

    return rpt


# ---------------------------------------------------------------------------
# CSV-based convenience wrapper
# ---------------------------------------------------------------------------

def classify_csv(
    path:        Union[str, Path],
    *,
    run_date:    Optional[date]                        = None,
    spot_checks: Optional[Sequence[Tuple[str, float]]] = None,
) -> DataAuthenticityReport:
    """
    Load a CSV file with the existing data_loader and classify it.

    Volume column presence is inferred from the CSV header.
    """
    from src.backtesting.data_loader import load_candles_from_csv
    path = Path(path)

    # Detect whether volume column exists before loading
    has_volume = _csv_has_volume_column(path)

    candles, _ = load_candles_from_csv(path)
    return classify_dataset(
        candles,
        file_name=path.name,
        run_date=run_date,
        spot_checks=spot_checks,
        has_volume=has_volume,
    )


def save_authenticity_report(
    report: DataAuthenticityReport,
    path:   Union[str, Path],
) -> None:
    """Persist the report summary to a text file."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report.summary(), encoding="utf-8")
    logger.info("Authenticity report saved to %s", out)


def run_authenticity_check(
    csv_path:    Union[str, Path],
    report_path: Union[str, Path],
    *,
    run_date:    Optional[date]                        = None,
    spot_checks: Optional[Sequence[Tuple[str, float]]] = None,
) -> DataAuthenticityReport:
    """
    Convenience: classify a CSV and save the report.

    Returns the DataAuthenticityReport so callers can inspect it.
    """
    rpt = classify_csv(csv_path, run_date=run_date, spot_checks=spot_checks)
    save_authenticity_report(rpt, report_path)
    return rpt


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _is_exact_doji(open_: float, close: float) -> bool:
    """True when open == close at 5-decimal precision."""
    return round(open_, 5) == round(close, 5)


def _csv_has_volume_column(path: Path) -> bool:
    """Peek at the CSV header to detect if a volume column is present."""
    try:
        with open(path, newline="", encoding="utf-8-sig") as fh:
            reader = csv.reader(fh)
            header = next(reader, None)
        if header is None:
            return False
        from src.backtesting.data_loader import _COL_ALIASES
        for name in header:
            if _COL_ALIASES.get(name.strip().lower()) == "volume":
                return True
    except Exception:
        pass
    return False


def _fmt_date(d: Optional[date]) -> str:
    if d is None:
        return "N/A"
    return str(d)
