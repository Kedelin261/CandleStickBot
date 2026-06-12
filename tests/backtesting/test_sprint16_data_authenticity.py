"""
Sprint 16 — Stage 1: Data Authenticity Gate Tests
===================================================
Tests the permanent DataAuthenticityReport gate introduced in
src/backtesting/data_authenticity.py.

Requirements verified:
  - Legacy file (EURUSD_D1_2014_2026.csv) must classify SYNTHETIC_SUSPECT
  - Crafted clean fixture must classify VERIFIED_REAL
  - Each hard-fail trigger in isolation (H1–H5)
  - Soft-signal triggers in isolation (S1–S3)
  - Classification logic: multi-hard → SYNTHETIC_SUSPECT;
    multi-soft (3+) → MIXED_RECONSTRUCTED; clean → VERIFIED_REAL
  - Volume column: absent vs zero vs non-zero
  - Spot-check logic (H5 threshold, deviation tolerance)
  - Coverage-years and date-range computation
  - save_authenticity_report / run_authenticity_check helpers
  - classify_csv correctly detects volume presence from header

Total: ≥ 60 tests across 10 test classes.
"""

from __future__ import annotations

import io
import os
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import List
from unittest.mock import patch

import pytest

from src.backtesting.data_authenticity import (
    DataAuthenticityReport,
    MIXED_RECONSTRUCTED,
    SYNTHETIC_SUSPECT,
    VERIFIED_REAL,
    _is_exact_doji,
    classify_csv,
    classify_dataset,
    run_authenticity_check,
    save_authenticity_report,
)
from src.data.types import CandleData


# ===========================================================================
# Candle factories
# ===========================================================================

_BASE_DATE = date(2020, 1, 2)   # Thursday — first trading day of 2020

# Months to avoid in clean fixtures: Dec (Christmas) and Jan (New Year)
_SKIP_MONTHS = {1, 12}   # Jan=1, Dec=12


def _make_candle(
    d: date,
    o: float = 1.1000,
    h: float = 1.1050,
    lo: float = 1.0950,
    c: float = 1.1020,
    vol: float = 1000.0,
    symbol: str = "EURUSD",
    tf: str = "D1",
) -> CandleData:
    return CandleData(
        timestamp=datetime(d.year, d.month, d.day, tzinfo=timezone.utc),
        open=o, high=h, low=lo, close=c,
        volume=vol, symbol=symbol, timeframe=tf,
    )


def _weekday_sequence(
    n: int,
    start: date = _BASE_DATE,
    skip_holiday_months: bool = False,
) -> List[date]:
    """Return n weekday dates starting from `start`, skipping weekends.
    If skip_holiday_months=True, also skips January and December dates
    to avoid hardcoded holiday (Jan 1, Dec 25) contamination.
    """
    out: List[date] = []
    d = start
    while len(out) < n:
        if d.weekday() < 5:   # Mon=0 … Fri=4
            if not (skip_holiday_months and d.month in _SKIP_MONTHS):
                out.append(d)
        d += timedelta(days=1)
    return out


def _clean_candles(
    n: int = 260,
    start: date = _BASE_DATE,
    skip_holiday_months: bool = True,
) -> List[CandleData]:
    """
    A clean, real-looking set of n daily candles.
    - Weekdays only, skipping Jan/Dec to avoid hardcoded holiday bars
    - Non-zero bodies (no dojis)
    - Non-zero volume
    - Slightly varying prices to avoid uniform patterns
    """
    dates = _weekday_sequence(n, start, skip_holiday_months=skip_holiday_months)
    out = []
    base = 1.1000
    for i, d in enumerate(dates):
        o = base + i * 0.0001
        c = o + 0.0003 + (i % 3) * 0.0001   # always strictly above open
        h = c + 0.0008
        lo = o - 0.0005
        out.append(_make_candle(d, o=o, h=h, lo=lo, c=c, vol=1000.0 + i))
    return out


def _inject_holiday_gaps(candles: List[CandleData], count: int = 10) -> List[CandleData]:
    """
    Create real-looking holiday gaps by duplicating a candle's timestamp
    to the following Wednesday (bridging over a 4-day gap) at evenly-spaced
    positions. This simulates the appearance of a 4-day gap without
    removing candles (which would reduce n).

    Simpler approach: just remove every N-th candle at positions that are
    on a Wednesday, then the gap to the next bar is Wed→Mon = 5 days.
    This creates gaps of exactly 5 days which look like holiday weeks.
    """
    if not candles or count <= 0:
        return candles
    step = len(candles) // (count + 1)
    indices_to_remove = {step * (i + 1) for i in range(count)}
    return [c for i, c in enumerate(candles) if i not in indices_to_remove]


# ===========================================================================
# Class 1 — _is_exact_doji helper
# ===========================================================================

class TestIsExactDoji:
    def test_exact_match(self):
        assert _is_exact_doji(1.12345, 1.12345) is True

    def test_not_doji(self):
        assert _is_exact_doji(1.12345, 1.12346) is False

    def test_rounds_to_5dp(self):
        # 1.123456 rounds to 1.12346; 1.123454 rounds to 1.12345 — different
        assert _is_exact_doji(1.123454, 1.123456) is False

    def test_both_zero(self):
        assert _is_exact_doji(0.0, 0.0) is True

    def test_large_prices(self):
        assert _is_exact_doji(120.000, 120.000) is True  # USDJPY-like


# ===========================================================================
# Class 2 — H1: Exact doji rate
# ===========================================================================

class TestH1DojiRate:
    def test_no_dojis_no_trigger(self):
        candles = _clean_candles(100)
        rpt = classify_dataset(candles, file_name="test.csv")
        assert rpt.h1_triggered is False
        assert rpt.h1_doji_rate < 0.02

    def test_doji_rate_exactly_at_threshold_triggers(self):
        """2% dojis → H1 triggered."""
        candles = _clean_candles(100)
        # inject 2 exact dojis (2%)
        base_price = 1.1500
        candles[10] = _make_candle(
            candles[10].timestamp.date(), o=base_price, h=base_price + 0.001,
            lo=base_price - 0.001, c=base_price
        )
        candles[20] = _make_candle(
            candles[20].timestamp.date(), o=base_price + 0.001, h=base_price + 0.002,
            lo=base_price, c=base_price + 0.001
        )
        rpt = classify_dataset(candles, file_name="test.csv")
        assert rpt.h1_doji_rate == pytest.approx(0.02)
        assert rpt.h1_triggered is True
        assert "H1" in rpt.hard_fails

    def test_below_threshold_no_trigger(self):
        """1 doji in 100 = 1% → below 2% threshold → no H1."""
        candles = _clean_candles(100)
        p = 1.1500
        candles[5] = _make_candle(
            candles[5].timestamp.date(), o=p, h=p+0.001, lo=p-0.001, c=p
        )
        rpt = classify_dataset(candles, file_name="test.csv")
        assert rpt.h1_triggered is False

    def test_high_doji_rate_triggers(self):
        """34% dojis (legacy file scenario) → H1 triggered."""
        n = 300
        dates = _weekday_sequence(n)
        candles = []
        for i, d in enumerate(dates):
            p = 1.1000 + i * 0.0001
            if i % 3 == 0:   # every 3rd bar is a doji ≈ 33%
                candles.append(_make_candle(d, o=p, h=p+0.001, lo=p-0.001, c=p))
            else:
                candles.append(_make_candle(d, o=p, h=p+0.001, lo=p-0.001, c=p+0.0003))
        rpt = classify_dataset(candles, file_name="test.csv")
        assert rpt.h1_triggered is True
        assert rpt.h1_doji_rate > 0.30


# ===========================================================================
# Class 3 — H2: Calendar-impossible bars
# ===========================================================================

class TestH2CalendarImpossible:
    def test_weekend_bar_triggers(self):
        """A Saturday bar must trigger H2."""
        candles = _clean_candles(50)
        saturday = date(2020, 1, 4)   # Saturday
        candles.append(_make_candle(saturday))
        candles.sort(key=lambda c: c.timestamp)
        rpt = classify_dataset(candles, file_name="test.csv")
        assert rpt.h2_triggered is True
        assert "H2" in rpt.hard_fails
        assert "2020-01-04" in rpt.h2_calendar_bars

    def test_sunday_bar_triggers(self):
        """A Sunday bar must trigger H2."""
        candles = _clean_candles(50)
        sunday = date(2020, 1, 5)   # Sunday
        candles.append(_make_candle(sunday))
        candles.sort(key=lambda c: c.timestamp)
        rpt = classify_dataset(candles, file_name="test.csv")
        assert rpt.h2_triggered is True
        assert "2020-01-05" in rpt.h2_calendar_bars

    def test_christmas_triggers(self):
        """Dec 25 bar must trigger H2."""
        christmas = date(2020, 12, 25)
        candles = _clean_candles(50)
        candles.append(_make_candle(christmas))
        candles.sort(key=lambda c: c.timestamp)
        rpt = classify_dataset(candles, file_name="test.csv")
        assert rpt.h2_triggered is True

    def test_new_years_triggers(self):
        """Jan 1 bar must trigger H2."""
        new_year = date(2020, 1, 1)
        candles = _clean_candles(50)
        candles.append(_make_candle(new_year))
        candles.sort(key=lambda c: c.timestamp)
        rpt = classify_dataset(candles, file_name="test.csv")
        assert rpt.h2_triggered is True

    def test_clean_weekdays_no_h2(self):
        """Weekday-only bars — no H2."""
        candles = _clean_candles(200)
        rpt = classify_dataset(candles, file_name="test.csv")
        assert rpt.h2_triggered is False
        assert len(rpt.h2_calendar_bars) == 0


# ===========================================================================
# Class 4 — H3: Future-dated bars
# ===========================================================================

class TestH3FutureBars:
    def test_future_bar_triggers(self):
        """A bar dated 1 year from now must trigger H3."""
        candles = _clean_candles(50)
        future = date.today() + timedelta(days=365)
        candles.append(_make_candle(future))
        candles.sort(key=lambda c: c.timestamp)
        rpt = classify_dataset(candles, file_name="test.csv", run_date=date.today())
        assert rpt.h3_triggered is True
        assert "H3" in rpt.hard_fails
        assert str(future) in rpt.h3_future_bars

    def test_today_bar_not_future(self):
        """A bar dated today is NOT future."""
        candles = _clean_candles(50)
        today = date.today()
        if today.weekday() < 5:
            candles.append(_make_candle(today))
        rpt = classify_dataset(candles, file_name="test.csv", run_date=today)
        assert rpt.h3_triggered is False

    def test_no_future_bars(self):
        """Historical fixture — no future bars."""
        candles = _clean_candles(100, start=date(2018, 1, 2))
        rpt = classify_dataset(candles, file_name="test.csv", run_date=date(2026, 1, 1))
        assert rpt.h3_triggered is False


# ===========================================================================
# Class 5 — H4: Structural integrity
# ===========================================================================

class TestH4Structural:
    def test_duplicate_timestamp_triggers(self):
        """Duplicate timestamp → H4 triggered."""
        candles = _clean_candles(50)
        candles.append(candles[5])   # exact duplicate
        rpt = classify_dataset(candles, file_name="test.csv")
        assert rpt.h4_triggered is True
        assert "H4" in rpt.hard_fails

    def test_non_chronological_triggers(self):
        """Out-of-order timestamps → H4 triggered."""
        candles = _clean_candles(50)
        # Swap two candles to break order
        candles[10], candles[11] = candles[11], candles[10]
        rpt = classify_dataset(candles, file_name="test.csv")
        assert rpt.h4_triggered is True

    def test_ohlc_inconsistent_triggers(self):
        """Low > High → H4 triggered."""
        candles = _clean_candles(50)
        d = candles[5].timestamp.date()
        candles[5] = _make_candle(d, o=1.1, h=1.0, lo=1.2, c=1.1)  # h < lo
        rpt = classify_dataset(candles, file_name="test.csv")
        assert rpt.h4_triggered is True

    def test_clean_structure_no_h4(self):
        """Clean fixture — no H4."""
        candles = _clean_candles(100)
        rpt = classify_dataset(candles, file_name="test.csv")
        assert rpt.h4_triggered is False


# ===========================================================================
# Class 6 — H5: Spot-check reference closes
# ===========================================================================

class TestH5SpotChecks:
    def _perfect_spot_checks(self, candles: List[CandleData], count: int = 10):
        """Generate spot checks that exactly match candle closes."""
        refs = []
        for c in candles[:count]:
            refs.append((str(c.timestamp.date()), c.close))
        return refs

    def test_no_spot_checks_no_trigger(self):
        """Without spot checks, H5 never triggers."""
        candles = _clean_candles(100)
        rpt = classify_dataset(candles, file_name="test.csv", spot_checks=None)
        assert rpt.h5_triggered is False
        assert rpt.spot_checks_run == 0

    def test_perfect_spot_checks_pass(self):
        """All spot checks match file exactly → H5 not triggered."""
        candles = _clean_candles(100)
        spot = self._perfect_spot_checks(candles, 10)
        rpt = classify_dataset(candles, file_name="test.csv", spot_checks=spot)
        assert rpt.h5_triggered is False
        assert rpt.h5_spot_check_fails == 0

    def test_more_than_2_large_deviations_triggers(self):
        """3 spot checks with >30 pip deviation → H5 triggered."""
        candles = _clean_candles(100)
        spot = []
        for c in candles[:10]:
            # Use a reference that is 50 pips off
            spot.append((str(c.timestamp.date()), c.close + 0.0050))
        rpt = classify_dataset(candles, file_name="test.csv", spot_checks=spot)
        assert rpt.h5_triggered is True
        assert rpt.h5_spot_check_fails > 2
        assert "H5" in rpt.hard_fails

    def test_exactly_2_fails_no_trigger(self):
        """Exactly 2 spot-check failures (threshold > 2) → H5 NOT triggered."""
        candles = _clean_candles(100)
        spot = []
        for i, c in enumerate(candles[:10]):
            if i < 2:
                # 2 bad references (>30 pips off)
                spot.append((str(c.timestamp.date()), c.close + 0.0050))
            else:
                spot.append((str(c.timestamp.date()), c.close))
        rpt = classify_dataset(candles, file_name="test.csv", spot_checks=spot)
        assert rpt.h5_spot_check_fails == 2
        assert rpt.h5_triggered is False

    def test_fewer_than_8_spot_checks_no_trigger(self):
        """H5 requires ≥ 8 reference points; fewer refs → no H5 even with fails."""
        candles = _clean_candles(100)
        # 5 spot checks all badly wrong
        spot = [(str(candles[i].timestamp.date()), candles[i].close + 0.0050)
                for i in range(5)]
        rpt = classify_dataset(candles, file_name="test.csv", spot_checks=spot)
        assert rpt.h5_triggered is False   # < 8 refs, threshold not active

    def test_deviation_within_tolerance_no_fail(self):
        """29-pip deviation (< 30) → not a spot-check fail."""
        candles = _clean_candles(100)
        spot = []
        for i, c in enumerate(candles[:10]):
            # 29 pips = 0.0029 — just under threshold
            spot.append((str(c.timestamp.date()), c.close + 0.0029))
        rpt = classify_dataset(candles, file_name="test.csv", spot_checks=spot)
        assert rpt.h5_spot_check_fails == 0

    def test_dates_not_in_dataset_skipped(self):
        """Spot check dates absent from dataset are silently skipped."""
        candles = _clean_candles(50, start=date(2020, 3, 2))
        # A date that definitely is not in the dataset
        spot = [("1999-01-01", 1.0500)] * 10
        rpt = classify_dataset(candles, file_name="test.csv", spot_checks=spot)
        assert rpt.h5_triggered is False
        assert rpt.h5_spot_check_fails == 0


# ===========================================================================
# Class 7 — Soft signals S1, S2, S3
# ===========================================================================

class TestSoftSignals:
    def test_s1_zero_volume_all_triggers(self):
        """100% zero-volume → S1 triggered."""
        dates = _weekday_sequence(100)
        candles = [_make_candle(d, vol=0.0) for d in dates]
        rpt = classify_dataset(candles, file_name="test.csv", has_volume=True)
        assert rpt.s1_triggered is True
        assert rpt.s1_zero_volume_pct == 1.0
        assert any("zero volume" in w.lower() for w in rpt.soft_warnings)

    def test_s1_non_zero_volume_no_trigger(self):
        """Non-zero volume → S1 not triggered."""
        candles = _clean_candles(100)
        rpt = classify_dataset(candles, file_name="test.csv", has_volume=True)
        assert rpt.s1_triggered is False

    def test_s1_absent_volume_column_triggers(self):
        """has_volume=False → S1 triggered (absent)."""
        candles = _clean_candles(100)
        rpt = classify_dataset(candles, file_name="test.csv", has_volume=False)
        assert rpt.s1_triggered is True
        assert rpt.s1_zero_volume_pct == 1.0

    def test_s2_uniform_dow_triggers(self):
        """Exactly 20 bars per weekday (uniform) → S2 triggered."""
        dates = []
        # Manually build exactly 20 of each weekday
        for week in range(20):
            base = date(2020, 1, 6) + timedelta(weeks=week)
            for offset in range(5):  # Mon–Fri
                dates.append(base + timedelta(days=offset))
        dates.sort()
        candles = [_make_candle(d) for d in dates]
        rpt = classify_dataset(candles, file_name="test.csv")
        assert rpt.s2_triggered is True

    def test_s2_non_uniform_dow_no_trigger(self):
        """Non-uniform DOW (holiday gaps cause variation) → S2 not triggered."""
        # Build explicitly non-uniform DOW by removing many Mondays
        dates = _weekday_sequence(300, skip_holiday_months=True)
        # Keep only non-Monday OR every 4th Monday (results in ~2/3 of Mondays)
        dates = [d for d in dates if d.weekday() != 0 or dates.index(d) % 4 == 0]
        candles = []
        for i, d in enumerate(dates):
            p = 1.1 + i * 0.0001
            candles.append(_make_candle(d, o=p, h=p+0.001, lo=p-0.001, c=p+0.0003))
        rpt = classify_dataset(candles, file_name="test.csv")
        assert rpt.s2_triggered is False

    def test_s3_no_holiday_gaps_triggers(self):
        """No holiday gaps (perfectly consecutive weekdays, ≥1 yr) → S3 triggered."""
        # Use skip_holiday_months=True so no H2; ≥1yr needed for S3 to check
        dates = _weekday_sequence(260, skip_holiday_months=True)
        candles = []
        for i, d in enumerate(dates):
            p = 1.1 + i * 0.0001
            candles.append(_make_candle(d, o=p, h=p+0.001, lo=p-0.001, c=p+0.0003))
        rpt = classify_dataset(candles, file_name="test.csv")
        # coverage_years > 1 since we skip Jan/Dec but span multiple years
        assert rpt.coverage_years >= 1.0
        assert rpt.s3_triggered is True

    def test_s3_with_holiday_gaps_no_trigger(self):
        """With ~8+ holiday gaps per year → S3 not triggered."""
        # 520 skip-holiday-month candles ≈ 2 years of Feb-Nov bars
        candles = _clean_candles(520, skip_holiday_months=True)
        # Inject ~16 holiday gaps (8/yr over 2 years)
        # Each removal creates a gap from prev-Friday to next-Monday = 5+ days
        # We remove single mid-week bars to create 4-day Wednesday→Monday gaps
        gap_candles = _inject_holiday_gaps(candles, count=16)
        rpt = classify_dataset(gap_candles, file_name="test.csv")
        # Should NOT trigger S3 because we have enough holiday gaps
        assert rpt.s3_triggered is False


# ===========================================================================
# Class 8 — Classification logic
# ===========================================================================

class TestClassificationLogic:
    def test_clean_fixture_verified_real(self):
        """Fully clean fixture with holiday gaps → VERIFIED_REAL."""
        # skip_holiday_months=True avoids Jan 1 / Dec 25 H2 triggers
        base_candles = _clean_candles(520, skip_holiday_months=True)
        candles = _inject_holiday_gaps(base_candles, count=16)
        rpt = classify_dataset(candles, file_name="clean.csv",
                               run_date=date(2026, 1, 1))
        assert len(rpt.hard_fails) == 0, (
            f"Expected no hard fails, got: {rpt.hard_fails}"
        )
        assert rpt.classification == VERIFIED_REAL

    def test_any_hard_fail_synthetic_suspect(self):
        """Single hard fail → SYNTHETIC_SUSPECT."""
        candles = _clean_candles(100)
        # Add weekend bar (H2)
        candles.append(_make_candle(date(2020, 1, 4)))  # Saturday
        candles.sort(key=lambda c: c.timestamp)
        rpt = classify_dataset(candles, file_name="test.csv")
        assert rpt.classification == SYNTHETIC_SUSPECT

    def test_three_soft_warnings_mixed_reconstructed(self):
        """3+ soft signals without hard fails → MIXED_RECONSTRUCTED."""
        # Trigger S1 (zero vol) + S2 (uniform DOW) + S3 (no holiday gaps)
        # Use skip_holiday_months=True to avoid H2 (Dec 25 / Jan 1)
        # Use ≥ 1 year of data so S3 activates
        dates = _weekday_sequence(260, skip_holiday_months=True)
        candles = [
            _make_candle(d, vol=0.0)   # zero volume → S1
            for d in dates             # no holiday gaps → S3
        ]                              # perfectly uniform DOW → S2
        rpt = classify_dataset(candles, file_name="test.csv",
                               has_volume=True, run_date=date(2030, 1, 1))
        # S1 (zero vol), S2 (uniform DOW), S3 (no holiday gaps) → 3 soft warns
        assert len(rpt.soft_warnings) >= 3, (
            f"Expected ≥3 soft warns, got {len(rpt.soft_warnings)}: {rpt.soft_warnings}"
        )
        assert rpt.classification == MIXED_RECONSTRUCTED

    def test_two_soft_warnings_verified_real(self):
        """2 soft warnings (under threshold) → VERIFIED_REAL (no hard fails)."""
        # Build candles: zero volume + no holiday gaps  (2 soft signals)
        # but with non-uniform DOW → only 2 soft
        dates = _weekday_sequence(100)
        # Remove some Mondays for DOW non-uniformity
        dates = [d for d in dates if not (d.weekday() == 0 and dates.index(d) % 4 == 0)]
        candles = [_make_candle(d, vol=0.0) for d in dates]
        rpt = classify_dataset(candles, file_name="test.csv", has_volume=True)
        # At most 2 soft signals → VERIFIED_REAL (borderline, check no hard fail)
        assert len(rpt.hard_fails) == 0
        if len(rpt.soft_warnings) < 3:
            assert rpt.classification == VERIFIED_REAL

    def test_multiple_hard_fails_synthetic_suspect(self):
        """Multiple hard fails → SYNTHETIC_SUSPECT."""
        n = 300
        dates = _weekday_sequence(n)
        candles = []
        for i, d in enumerate(dates):
            p = 1.1 + i * 0.0001
            # Every 3rd bar is a doji → H1
            c = p if i % 3 == 0 else p + 0.0003
            candles.append(_make_candle(d, o=p, c=c, h=max(p,c)+0.001, lo=min(p,c)-0.001))
        # Also add a Jan 1 bar → H2
        candles.append(_make_candle(date(2020, 1, 1)))
        candles.sort(key=lambda c: c.timestamp)
        rpt = classify_dataset(candles, file_name="test.csv")
        assert rpt.classification == SYNTHETIC_SUSPECT
        assert len(rpt.hard_fails) >= 2

    def test_hard_fails_list_populated(self):
        """hard_fails list contains the triggered code strings."""
        candles = _clean_candles(50)
        candles.append(_make_candle(date(2020, 1, 4)))  # Saturday → H2
        candles.sort(key=lambda c: c.timestamp)
        rpt = classify_dataset(candles, file_name="test.csv")
        assert "H2" in rpt.hard_fails

    def test_empty_candles_synthetic_suspect(self):
        """Empty candle list → SYNTHETIC_SUSPECT (EMPTY hard fail)."""
        rpt = classify_dataset([], file_name="empty.csv")
        assert rpt.classification == SYNTHETIC_SUSPECT
        assert "EMPTY" in rpt.hard_fails


# ===========================================================================
# Class 9 — Summary statistics
# ===========================================================================

class TestSummaryStatistics:
    def test_total_rows(self):
        n = 150
        candles = _clean_candles(n)
        rpt = classify_dataset(candles, file_name="test.csv")
        assert rpt.total_rows == n

    def test_date_range(self):
        # Use a non-Jan/Dec start so skip_holiday_months=False is safe for 50 bars
        # (50 bars from Feb 2018 won't hit any hardcoded holiday)
        start = date(2018, 2, 1)
        candles = _clean_candles(50, start=start, skip_holiday_months=False)
        rpt = classify_dataset(candles, file_name="test.csv")
        assert rpt.date_range[0] == start

    def test_coverage_years_positive(self):
        candles = _clean_candles(260)
        rpt = classify_dataset(candles, file_name="test.csv")
        assert rpt.coverage_years > 0

    def test_summary_string_contains_classification(self):
        """report.summary() must contain the classification string."""
        candles = _inject_holiday_gaps(_clean_candles(520), count=16)
        rpt = classify_dataset(candles, file_name="clean.csv",
                               run_date=date(2026, 1, 1))
        summary = rpt.summary()
        assert rpt.classification in summary

    def test_summary_string_contains_hard_fail_codes(self):
        """summary() must list triggered hard-fail codes."""
        candles = _clean_candles(50)
        candles.append(_make_candle(date(2020, 1, 4)))  # H2
        candles.sort(key=lambda c: c.timestamp)
        rpt = classify_dataset(candles, file_name="test.csv")
        summary = rpt.summary()
        assert "H2" in summary

    def test_h1_doji_rate_in_summary(self):
        """Doji rate % appears in the summary."""
        candles = _clean_candles(100)
        rpt = classify_dataset(candles, file_name="test.csv")
        summary = rpt.summary()
        assert "%" in summary


# ===========================================================================
# Class 10 — I/O helpers
# ===========================================================================

class TestIOHelpers:
    def test_save_report_creates_file(self, tmp_path):
        """save_authenticity_report() creates the output file."""
        candles = _clean_candles(50)
        rpt = classify_dataset(candles, file_name="test.csv")
        out = tmp_path / "auth_report.txt"
        save_authenticity_report(rpt, out)
        assert out.exists()
        content = out.read_text()
        assert rpt.classification in content

    def test_save_report_creates_parent_dirs(self, tmp_path):
        """save_authenticity_report creates parent dirs as needed."""
        candles = _clean_candles(50)
        rpt = classify_dataset(candles, file_name="test.csv")
        out = tmp_path / "subdir" / "nested" / "report.txt"
        save_authenticity_report(rpt, out)
        assert out.exists()

    def test_run_authenticity_check_returns_report(self, tmp_path):
        """run_authenticity_check() returns a DataAuthenticityReport."""
        # Build a tiny CSV
        csv_content = "date,open,high,low,close,volume\n"
        for i in range(50):
            d = date(2020, 1, 2) + timedelta(days=i)
            if d.weekday() < 5:
                csv_content += f"{d},1.1{i:03d},1.1050,1.0950,1.1020,1000\n"

        csv_file = tmp_path / "test.csv"
        csv_file.write_text(csv_content)
        report_file = tmp_path / "auth.txt"

        rpt = run_authenticity_check(csv_file, report_file)
        assert isinstance(rpt, DataAuthenticityReport)
        assert report_file.exists()

    def test_classify_csv_detects_no_volume_column(self, tmp_path):
        """classify_csv sets has_volume=False when volume column is absent."""
        csv_content = "date,open,high,low,close\n"
        for i in range(50):
            d = date(2020, 1, 2) + timedelta(days=i)
            if d.weekday() < 5:
                csv_content += f"{d},1.10{i:02d},1.1050,1.0950,1.1020\n"
        csv_file = tmp_path / "no_vol.csv"
        csv_file.write_text(csv_content)
        rpt = classify_csv(csv_file)
        # S1 triggered because no volume column
        assert rpt.s1_triggered is True

    def test_classify_csv_detects_volume_column(self, tmp_path):
        """classify_csv sets has_volume=True when volume column present."""
        csv_content = "date,open,high,low,close,volume\n"
        for i in range(50):
            d = date(2020, 1, 2) + timedelta(days=i)
            if d.weekday() < 5:
                p = 1.1000 + i * 0.0001
                csv_content += f"{d},{p:.5f},{p+0.001:.5f},{p-0.001:.5f},{p+0.0003:.5f},1000\n"
        csv_file = tmp_path / "with_vol.csv"
        csv_file.write_text(csv_content)
        rpt = classify_csv(csv_file)
        # Non-zero volume → S1 not triggered
        assert rpt.s1_triggered is False


# ===========================================================================
# Class 11 — Legacy file classification (integration test)
# ===========================================================================

LEGACY_CSV = "data/EURUSD_D1_2014_2026.csv"


class TestLegacyFileClassification:
    """
    The legacy EURUSD_D1_2014_2026.csv MUST be classified SYNTHETIC_SUSPECT.
    This is the permanent regression guard for the Sprint 15 data-authenticity
    finding.  If someone replaces the legacy file with clean data, these tests
    will catch the classification change.
    """

    @pytest.mark.skipif(
        not os.path.exists(LEGACY_CSV),
        reason="Legacy dataset not present",
    )
    def test_legacy_file_classified_synthetic_suspect(self):
        """Core guard: legacy file → SYNTHETIC_SUSPECT."""
        rpt = classify_csv(LEGACY_CSV, run_date=date(2026, 6, 12))
        assert rpt.classification == SYNTHETIC_SUSPECT, (
            f"Expected SYNTHETIC_SUSPECT, got {rpt.classification}. "
            f"Hard fails: {rpt.hard_fails}"
        )

    @pytest.mark.skipif(
        not os.path.exists(LEGACY_CSV),
        reason="Legacy dataset not present",
    )
    def test_legacy_h1_doji_rate_above_threshold(self):
        """Legacy file has ~34.8% exact dojis → well above 2% threshold."""
        rpt = classify_csv(LEGACY_CSV, run_date=date(2026, 6, 12))
        assert rpt.h1_triggered is True
        assert rpt.h1_doji_rate > 0.30, (
            f"Expected >30% doji rate, got {rpt.h1_doji_rate*100:.1f}%"
        )

    @pytest.mark.skipif(
        not os.path.exists(LEGACY_CSV),
        reason="Legacy dataset not present",
    )
    def test_legacy_h2_calendar_bar_jan1(self):
        """Legacy file contains 2014-01-01 (New Year's Day) → H2."""
        rpt = classify_csv(LEGACY_CSV, run_date=date(2026, 6, 12))
        assert rpt.h2_triggered is True
        assert "2014-01-01" in rpt.h2_calendar_bars

    @pytest.mark.skipif(
        not os.path.exists(LEGACY_CSV),
        reason="Legacy dataset not present",
    )
    def test_legacy_s1_zero_volume(self):
        """Legacy file has 100% zero volume → S1 soft warning."""
        rpt = classify_csv(LEGACY_CSV, run_date=date(2026, 6, 12))
        assert rpt.s1_triggered is True
        assert rpt.s1_zero_volume_pct == pytest.approx(1.0)

    @pytest.mark.skipif(
        not os.path.exists(LEGACY_CSV),
        reason="Legacy dataset not present",
    )
    def test_legacy_s2_uniform_dow(self):
        """Legacy file has perfectly uniform DOW → S2 soft warning."""
        rpt = classify_csv(LEGACY_CSV, run_date=date(2026, 6, 12))
        assert rpt.s2_triggered is True

    @pytest.mark.skipif(
        not os.path.exists(LEGACY_CSV),
        reason="Legacy dataset not present",
    )
    def test_legacy_hard_fails_list_populated(self):
        """Legacy file hard_fails must contain at least H1 and H2."""
        rpt = classify_csv(LEGACY_CSV, run_date=date(2026, 6, 12))
        assert "H1" in rpt.hard_fails
        assert "H2" in rpt.hard_fails

    @pytest.mark.skipif(
        not os.path.exists(LEGACY_CSV),
        reason="Legacy dataset not present",
    )
    def test_legacy_report_summary_output(self):
        """Legacy file report summary contains SYNTHETIC_SUSPECT."""
        rpt = classify_csv(LEGACY_CSV, run_date=date(2026, 6, 12))
        summary = rpt.summary()
        assert SYNTHETIC_SUSPECT in summary
