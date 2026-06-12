"""
tests/backtesting/test_data_loader.py
======================================
Sprint 13 — ≥30 tests for src/backtesting/data_loader.py

Coverage areas
--------------
1.  Basic CSV loading — standard header, correct candle count / fields
2.  DataQualityReport fields populated correctly
3.  Date format variants (all 18 patterns)
4.  Column alias variants (datetime/time/timestamp, vol/v, o/h/l/c)
5.  OHLC validity gating
6.  Duplicate timestamp detection / removal
7.  Missing-candle (D1 gap) estimation
8.  Date range filtering (start_date / end_date)
9.  Candles returned sorted oldest-first regardless of CSV order
10. StringIO input vs file-path input
11. Empty-CSV and header-only-CSV error cases
12. Missing required column error case
13. Completely non-numeric price data (invalid_rows accumulation)
14. DataQualityReport properties: is_clean, coverage_days, summary()
15. Volume column optional — defaults to 0.0
16. Symbol and timeframe upper-cased on every CandleData
17. File-not-found error
18. Candles with identical OHLC (flat bar) are valid
19. Non-D1 timeframe — missing_candles stays 0
"""

from __future__ import annotations

import io
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent

import pytest

from src.backtesting.data_loader import (
    DataQualityReport,
    _build_col_map,
    _ohlc_valid,
    _parse_date,
    _estimate_missing_d1,
    load_candles_from_csv,
)
from src.data.types import CandleData


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _csv(*rows: str) -> io.StringIO:
    """Wrap lines in a StringIO, first line is header."""
    return io.StringIO("\n".join(rows))


BASIC_HEADER = "date,open,high,low,close,volume"
FIVE_ROWS = [
    "2024-01-02,1.09500,1.09800,1.09200,1.09650,12345",
    "2024-01-03,1.09650,1.10000,1.09400,1.09850,13210",
    "2024-01-04,1.09850,1.10200,1.09600,1.09700,11987",
    "2024-01-05,1.09700,1.10100,1.09500,1.09950,14500",
    "2024-01-08,1.09950,1.10300,1.09700,1.10100,15000",
]


def _basic_csv(extra_rows: list[str] | None = None) -> io.StringIO:
    rows = [BASIC_HEADER] + FIVE_ROWS + (extra_rows or [])
    return io.StringIO("\n".join(rows))


# ---------------------------------------------------------------------------
# Class 1 — Basic loading
# ---------------------------------------------------------------------------

class TestBasicLoading:
    def test_returns_tuple(self):
        buf = _basic_csv()
        result = load_candles_from_csv(buf)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_candle_count(self):
        candles, _ = load_candles_from_csv(_basic_csv())
        assert len(candles) == 5

    def test_candle_type(self):
        candles, _ = load_candles_from_csv(_basic_csv())
        for c in candles:
            assert isinstance(c, CandleData)

    def test_ohlcv_values(self):
        candles, _ = load_candles_from_csv(_basic_csv())
        c = candles[0]
        assert c.open  == pytest.approx(1.09500)
        assert c.high  == pytest.approx(1.09800)
        assert c.low   == pytest.approx(1.09200)
        assert c.close == pytest.approx(1.09650)
        assert c.volume == pytest.approx(12345.0)

    def test_sorted_oldest_first(self):
        # Reverse the rows so CSV is newest-first
        rows = [BASIC_HEADER] + list(reversed(FIVE_ROWS))
        candles, _ = load_candles_from_csv(io.StringIO("\n".join(rows)))
        timestamps = [c.timestamp for c in candles]
        assert timestamps == sorted(timestamps)

    def test_symbol_attached(self):
        candles, _ = load_candles_from_csv(_basic_csv(), symbol="EURUSD")
        assert all(c.symbol == "EURUSD" for c in candles)

    def test_symbol_uppercased(self):
        candles, _ = load_candles_from_csv(_basic_csv(), symbol="eurusd")
        assert all(c.symbol == "EURUSD" for c in candles)

    def test_timeframe_attached(self):
        candles, _ = load_candles_from_csv(_basic_csv(), timeframe="D1")
        assert all(c.timeframe == "D1" for c in candles)

    def test_timeframe_uppercased(self):
        candles, _ = load_candles_from_csv(_basic_csv(), timeframe="d1")
        assert all(c.timeframe == "D1" for c in candles)

    def test_timestamp_utc_aware(self):
        candles, _ = load_candles_from_csv(_basic_csv())
        for c in candles:
            assert c.timestamp.tzinfo is not None


# ---------------------------------------------------------------------------
# Class 2 — DataQualityReport fields
# ---------------------------------------------------------------------------

class TestDataQualityReport:
    def test_report_type(self):
        _, report = load_candles_from_csv(_basic_csv())
        assert isinstance(report, DataQualityReport)

    def test_total_candles(self):
        _, report = load_candles_from_csv(_basic_csv())
        assert report.total_candles == 5

    def test_symbol_field(self):
        _, report = load_candles_from_csv(_basic_csv(), symbol="GBPUSD")
        assert report.symbol == "GBPUSD"

    def test_timeframe_field(self):
        _, report = load_candles_from_csv(_basic_csv(), timeframe="H4")
        assert report.timeframe == "H4"

    def test_start_date(self):
        _, report = load_candles_from_csv(_basic_csv())
        assert report.start_date == datetime(2024, 1, 2, tzinfo=timezone.utc)

    def test_end_date(self):
        _, report = load_candles_from_csv(_basic_csv())
        assert report.end_date == datetime(2024, 1, 8, tzinfo=timezone.utc)

    def test_source_path_string(self):
        _, report = load_candles_from_csv(_basic_csv())
        assert report.source_path == "<string>"

    def test_no_duplicates_by_default(self):
        _, report = load_candles_from_csv(_basic_csv())
        assert report.duplicate_candles == 0

    def test_no_invalid_rows_by_default(self):
        _, report = load_candles_from_csv(_basic_csv())
        assert report.invalid_rows == 0

    def test_is_clean_true(self):
        _, report = load_candles_from_csv(_basic_csv())
        assert report.is_clean is True

    def test_coverage_days(self):
        # 2024-01-02 → 2024-01-08 = 6 calendar days
        _, report = load_candles_from_csv(_basic_csv())
        assert report.coverage_days == 6

    def test_coverage_days_single_candle(self):
        buf = _csv(BASIC_HEADER, "2024-01-02,1.0950,1.0980,1.0920,1.0965,100")
        _, report = load_candles_from_csv(buf)
        assert report.coverage_days == 0

    def test_summary_returns_str(self):
        _, report = load_candles_from_csv(_basic_csv())
        s = report.summary()
        assert isinstance(s, str)
        assert "EURUSD" in s

    def test_summary_clean_status(self):
        _, report = load_candles_from_csv(_basic_csv())
        assert "CLEAN" in report.summary()

    def test_summary_contains_source(self):
        _, report = load_candles_from_csv(_basic_csv())
        assert "<string>" in report.summary()


# ---------------------------------------------------------------------------
# Class 3 — Date format variants
# ---------------------------------------------------------------------------

class TestDateFormats:
    """Verify _parse_date accepts all supported patterns."""

    @pytest.mark.parametrize("raw,expected_year", [
        ("2024-01-15",              2024),
        ("2024.01.15",              2024),
        ("2024/01/15",              2024),
        ("15-01-2024",              2024),
        ("15/01/2024",              2024),
        ("2024-01-15 09:30:00",     2024),
        ("2024-01-15 09:30",        2024),
        ("2024.01.15 09:30:00",     2024),
        ("2024.01.15 09:30",        2024),
        ("2024/01/15 09:30:00",     2024),
        ("2024/01/15 09:30",        2024),
        ("15-01-2024 09:30:00",     2024),
        ("15-01-2024 09:30",        2024),
        ("15/01/2024 09:30:00",     2024),
        ("15/01/2024 09:30",        2024),
    ])
    def test_date_format(self, raw, expected_year):
        dt = _parse_date(raw)
        assert dt.year == expected_year

    def test_parse_date_returns_utc(self):
        dt = _parse_date("2024-01-15")
        assert dt.tzinfo == timezone.utc

    def test_parse_date_invalid_raises(self):
        with pytest.raises(ValueError):
            _parse_date("not-a-date")

    def test_csv_dot_separator(self):
        buf = _csv(
            BASIC_HEADER,
            "2024.03.10,1.0800,1.0850,1.0750,1.0820,5000",
        )
        candles, _ = load_candles_from_csv(buf)
        assert candles[0].timestamp.month == 3

    def test_csv_slash_separator(self):
        buf = _csv(
            BASIC_HEADER,
            "2024/04/01,1.0800,1.0850,1.0750,1.0820,5000",
        )
        candles, _ = load_candles_from_csv(buf)
        assert candles[0].timestamp.month == 4


# ---------------------------------------------------------------------------
# Class 4 — Column alias variants
# ---------------------------------------------------------------------------

class TestColumnAliases:
    def test_datetime_alias(self):
        buf = _csv(
            "datetime,open,high,low,close,volume",
            "2024-01-02,1.09500,1.09800,1.09200,1.09650,100",
        )
        candles, _ = load_candles_from_csv(buf)
        assert len(candles) == 1

    def test_timestamp_alias(self):
        buf = _csv(
            "timestamp,open,high,low,close,volume",
            "2024-01-02,1.09500,1.09800,1.09200,1.09650,100",
        )
        candles, _ = load_candles_from_csv(buf)
        assert len(candles) == 1

    def test_time_alias(self):
        buf = _csv(
            "time,open,high,low,close",
            "2024-01-02,1.09500,1.09800,1.09200,1.09650",
        )
        candles, _ = load_candles_from_csv(buf)
        assert len(candles) == 1

    def test_short_ohcl_aliases(self):
        # o, h, l, c instead of open, high, low, close
        buf = _csv(
            "date,o,h,l,c",
            "2024-01-02,1.09500,1.09800,1.09200,1.09650",
        )
        candles, _ = load_candles_from_csv(buf)
        assert len(candles) == 1

    def test_vol_alias(self):
        buf = _csv(
            "date,open,high,low,close,vol",
            "2024-01-02,1.09500,1.09800,1.09200,1.09650,9999",
        )
        candles, _ = load_candles_from_csv(buf)
        assert candles[0].volume == pytest.approx(9999.0)

    def test_tickvolume_alias(self):
        buf = _csv(
            "date,open,high,low,close,tickvolume",
            "2024-01-02,1.09500,1.09800,1.09200,1.09650,8888",
        )
        candles, _ = load_candles_from_csv(buf)
        assert candles[0].volume == pytest.approx(8888.0)

    def test_volume_optional_defaults_to_zero(self):
        buf = _csv(
            "date,open,high,low,close",
            "2024-01-02,1.09500,1.09800,1.09200,1.09650",
        )
        candles, _ = load_candles_from_csv(buf)
        assert candles[0].volume == pytest.approx(0.0)

    def test_extra_columns_ignored(self):
        buf = _csv(
            "date,open,high,low,close,volume,spread,extra_field",
            "2024-01-02,1.09500,1.09800,1.09200,1.09650,100,0.0002,ignored",
        )
        candles, _ = load_candles_from_csv(buf)
        assert len(candles) == 1


# ---------------------------------------------------------------------------
# Class 5 — OHLC validation
# ---------------------------------------------------------------------------

class TestOhlcValidation:
    def test_valid_ohlc(self):
        assert _ohlc_valid(1.09, 1.10, 1.08, 1.095) is True

    def test_high_below_open_invalid(self):
        assert _ohlc_valid(1.09, 1.08, 1.07, 1.085) is False

    def test_high_below_close_invalid(self):
        assert _ohlc_valid(1.09, 1.08, 1.07, 1.095) is False

    def test_low_above_open_invalid(self):
        assert _ohlc_valid(1.09, 1.10, 1.10, 1.095) is False

    def test_low_above_close_invalid(self):
        assert _ohlc_valid(1.09, 1.10, 1.095, 1.09) is False

    def test_zero_price_invalid(self):
        assert _ohlc_valid(0.0, 1.10, 1.08, 1.09) is False

    def test_negative_price_invalid(self):
        assert _ohlc_valid(-1.0, 1.10, 1.08, 1.09) is False

    def test_low_above_high_invalid(self):
        assert _ohlc_valid(1.09, 1.07, 1.10, 1.09) is False

    def test_flat_candle_valid(self):
        # Open == High == Low == Close
        assert _ohlc_valid(1.09, 1.09, 1.09, 1.09) is True

    def test_invalid_rows_counted(self):
        buf = _csv(
            BASIC_HEADER,
            "2024-01-02,1.09500,1.09800,1.09200,1.09650,100",
            "2024-01-03,1.09,1.07,1.10,1.09,100",   # low > high → invalid
        )
        candles, report = load_candles_from_csv(buf)
        assert len(candles) == 1
        assert report.invalid_rows == 1


# ---------------------------------------------------------------------------
# Class 6 — Duplicate detection
# ---------------------------------------------------------------------------

class TestDuplicateDetection:
    def test_duplicate_removed(self):
        buf = _csv(
            BASIC_HEADER,
            "2024-01-02,1.09500,1.09800,1.09200,1.09650,100",
            "2024-01-02,1.09600,1.09900,1.09300,1.09750,200",  # dup
        )
        candles, report = load_candles_from_csv(buf)
        assert len(candles) == 1
        assert report.duplicate_candles == 1

    def test_is_clean_false_on_duplicate(self):
        buf = _csv(
            BASIC_HEADER,
            "2024-01-02,1.09500,1.09800,1.09200,1.09650,100",
            "2024-01-02,1.09600,1.09900,1.09300,1.09750,200",
        )
        _, report = load_candles_from_csv(buf)
        assert report.is_clean is False

    def test_warnings_present_when_issues(self):
        # Insert a large gap to trigger missing warning
        buf = _csv(
            BASIC_HEADER,
            "2024-01-02,1.09500,1.09800,1.09200,1.09650,100",
            "2024-02-15,1.09600,1.09900,1.09300,1.09750,200",  # ~44 day gap
        )
        _, report = load_candles_from_csv(buf, timeframe="D1")
        assert len(report.warnings) > 0


# ---------------------------------------------------------------------------
# Class 7 — Missing candle estimation
# ---------------------------------------------------------------------------

class TestMissingCandleEstimation:
    def test_no_missing_for_weekly_gaps(self):
        # Weekend gap (2 days) should not flag missing
        buf = _csv(
            BASIC_HEADER,
            "2024-01-05,1.09500,1.09800,1.09200,1.09650,100",  # Fri
            "2024-01-08,1.09600,1.09900,1.09300,1.09750,200",  # Mon
        )
        _, report = load_candles_from_csv(buf, timeframe="D1")
        assert report.missing_candles == 0

    def test_large_gap_detected(self):
        buf = _csv(
            BASIC_HEADER,
            "2024-01-02,1.09500,1.09800,1.09200,1.09650,100",
            "2024-01-22,1.09600,1.09900,1.09300,1.09750,200",  # 20 day gap
        )
        _, report = load_candles_from_csv(buf, timeframe="D1")
        assert report.missing_candles > 0

    def test_non_d1_skips_estimation(self):
        buf = _csv(
            BASIC_HEADER,
            "2024-01-02,1.09500,1.09800,1.09200,1.09650,100",
            "2024-01-22,1.09600,1.09900,1.09300,1.09750,200",
        )
        _, report = load_candles_from_csv(buf, timeframe="H4")
        assert report.missing_candles == 0

    def test_estimate_missing_d1_single_candle(self):
        from src.data.types import CandleData
        c = CandleData(
            timestamp=datetime(2024, 1, 2, tzinfo=timezone.utc),
            open=1.09, high=1.10, low=1.08, close=1.095,
            volume=100, symbol="EURUSD", timeframe="D1",
        )
        assert _estimate_missing_d1([c]) == 0

    def test_is_clean_false_when_missing(self):
        buf = _csv(
            BASIC_HEADER,
            "2024-01-02,1.09500,1.09800,1.09200,1.09650,100",
            "2024-02-10,1.09600,1.09900,1.09300,1.09750,200",
        )
        _, report = load_candles_from_csv(buf, timeframe="D1")
        assert report.is_clean is False


# ---------------------------------------------------------------------------
# Class 8 — Date range filtering
# ---------------------------------------------------------------------------

class TestDateRangeFiltering:
    def test_start_date_filter(self):
        candles, _ = load_candles_from_csv(
            _basic_csv(),
            start_date=datetime(2024, 1, 4, tzinfo=timezone.utc),
        )
        assert all(c.timestamp >= datetime(2024, 1, 4, tzinfo=timezone.utc)
                   for c in candles)
        assert len(candles) == 3  # Jan 4, 5, 8

    def test_end_date_filter(self):
        candles, _ = load_candles_from_csv(
            _basic_csv(),
            end_date=datetime(2024, 1, 4, tzinfo=timezone.utc),
        )
        assert all(c.timestamp <= datetime(2024, 1, 4, tzinfo=timezone.utc)
                   for c in candles)
        assert len(candles) == 3  # Jan 2, 3, 4

    def test_both_filters(self):
        candles, _ = load_candles_from_csv(
            _basic_csv(),
            start_date=datetime(2024, 1, 3, tzinfo=timezone.utc),
            end_date=datetime(2024, 1, 5, tzinfo=timezone.utc),
        )
        assert len(candles) == 3  # Jan 3, 4, 5

    def test_filter_excludes_all_returns_empty(self):
        candles, _ = load_candles_from_csv(
            _basic_csv(),
            start_date=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )
        assert len(candles) == 0

    def test_total_candles_in_report_reflects_filter(self):
        _, report = load_candles_from_csv(
            _basic_csv(),
            start_date=datetime(2024, 1, 4, tzinfo=timezone.utc),
        )
        assert report.total_candles == 3


# ---------------------------------------------------------------------------
# Class 9 — Error cases
# ---------------------------------------------------------------------------

class TestErrorCases:
    def test_empty_csv_raises(self):
        buf = io.StringIO("")
        with pytest.raises(ValueError, match="empty"):
            load_candles_from_csv(buf)

    def test_header_only_returns_empty_candles(self):
        buf = _csv(BASIC_HEADER)
        candles, report = load_candles_from_csv(buf)
        assert len(candles) == 0
        assert report.total_candles == 0

    def test_missing_required_column_raises(self):
        buf = _csv(
            "date,open,high,low",           # missing 'close'
            "2024-01-02,1.09500,1.09800,1.09200",
        )
        with pytest.raises(ValueError, match="missing required"):
            load_candles_from_csv(buf)

    def test_file_not_found_raises(self):
        with pytest.raises(ValueError, match="not found"):
            load_candles_from_csv("/tmp/does_not_exist_sprint13_xyz.csv")

    def test_entirely_invalid_rows_returns_empty(self):
        buf = _csv(
            BASIC_HEADER,
            "2024-01-02,not_a_number,1.09800,1.09200,1.09650,100",
            "2024-01-03,1.09650,not_a_number,1.09400,1.09850,100",
        )
        candles, report = load_candles_from_csv(buf)
        assert len(candles) == 0
        assert report.invalid_rows == 2

    def test_unparseable_date_row_skipped(self):
        buf = _csv(
            BASIC_HEADER,
            "BADDATE,1.09500,1.09800,1.09200,1.09650,100",
            "2024-01-03,1.09650,1.10000,1.09400,1.09850,100",
        )
        candles, report = load_candles_from_csv(buf)
        assert len(candles) == 1
        assert report.invalid_rows == 1


# ---------------------------------------------------------------------------
# Class 10 — File path input
# ---------------------------------------------------------------------------

class TestFilePathInput:
    def test_load_from_str_path(self, tmp_path):
        csv_file = tmp_path / "eurusd.csv"
        csv_file.write_text(
            BASIC_HEADER + "\n" + "\n".join(FIVE_ROWS),
            encoding="utf-8",
        )
        candles, report = load_candles_from_csv(str(csv_file))
        assert len(candles) == 5
        assert report.source_path == str(csv_file)

    def test_load_from_path_object(self, tmp_path):
        csv_file = tmp_path / "eurusd.csv"
        csv_file.write_text(
            BASIC_HEADER + "\n" + "\n".join(FIVE_ROWS),
            encoding="utf-8",
        )
        candles, report = load_candles_from_csv(Path(csv_file))
        assert len(candles) == 5

    def test_bom_utf8_file(self, tmp_path):
        csv_file = tmp_path / "bom.csv"
        content = (BASIC_HEADER + "\n" + FIVE_ROWS[0]).encode("utf-8-sig")
        csv_file.write_bytes(content)
        candles, _ = load_candles_from_csv(csv_file)
        assert len(candles) == 1


# ---------------------------------------------------------------------------
# Class 11 — _build_col_map unit tests
# ---------------------------------------------------------------------------

class TestBuildColMap:
    def test_standard_header(self):
        col_map = _build_col_map(["date", "open", "high", "low", "close"])
        assert col_map["date"]  == 0
        assert col_map["open"]  == 1
        assert col_map["high"]  == 2
        assert col_map["low"]   == 3
        assert col_map["close"] == 4

    def test_case_insensitive(self):
        col_map = _build_col_map(["DATE", "OPEN", "HIGH", "LOW", "CLOSE"])
        assert "date"  in col_map
        assert "close" in col_map

    def test_first_alias_wins(self):
        # If two columns map to 'date' the first one should win
        col_map = _build_col_map(["date", "datetime", "open", "high", "low", "close"])
        assert col_map["date"] == 0  # 'date' comes first

    def test_volume_alias_mapped(self):
        col_map = _build_col_map(["date", "open", "high", "low", "close", "vol"])
        assert "volume" in col_map

    def test_unknown_column_ignored(self):
        col_map = _build_col_map(["date", "open", "high", "low", "close", "mystery"])
        assert "mystery" not in col_map


# ---------------------------------------------------------------------------
# Class 12 — DataQualityReport dataclass directly
# ---------------------------------------------------------------------------

class TestDataQualityReportDirect:
    def test_defaults(self):
        r = DataQualityReport(symbol="TEST", timeframe="D1")
        assert r.total_candles == 0
        assert r.duplicate_candles == 0
        assert r.invalid_rows == 0
        assert r.missing_candles == 0
        assert r.start_date is None
        assert r.end_date is None
        assert r.warnings == []

    def test_is_clean_all_zero(self):
        r = DataQualityReport(symbol="X", timeframe="D1")
        assert r.is_clean is True

    def test_is_clean_duplicate_contaminates(self):
        r = DataQualityReport(symbol="X", timeframe="D1", duplicate_candles=1)
        assert r.is_clean is False

    def test_is_clean_invalid_contaminates(self):
        r = DataQualityReport(symbol="X", timeframe="D1", invalid_rows=2)
        assert r.is_clean is False

    def test_is_clean_missing_contaminates(self):
        r = DataQualityReport(symbol="X", timeframe="D1", missing_candles=3)
        assert r.is_clean is False

    def test_coverage_days_none_dates(self):
        r = DataQualityReport(symbol="X", timeframe="D1")
        assert r.coverage_days == 0

    def test_coverage_days_calculation(self):
        r = DataQualityReport(
            symbol="X", timeframe="D1",
            start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2024, 1, 31, tzinfo=timezone.utc),
        )
        assert r.coverage_days == 30

    def test_summary_includes_issues_found(self):
        r = DataQualityReport(symbol="X", timeframe="D1", invalid_rows=3)
        assert "ISSUES" in r.summary()

    def test_summary_includes_warnings(self):
        r = DataQualityReport(symbol="X", timeframe="D1",
                              warnings=["something went wrong"])
        assert "something went wrong" in r.summary()
