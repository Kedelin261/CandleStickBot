"""
Sprint 14 — Real Data Baseline Validation Tests
================================================
≥ 50 tests covering:
  - DataQualityAuditReport dataclass fields and properties
  - audit_csv() function (valid data, invalid data, edge cases)
  - save_audit_report() function
  - Report file generation (scorecard, comparison, validation)
  - BacktestResult.passes_baseline logic
  - StrategyValidationLab with real data
  - Root cause investigation findings (SR / Trend Gate behaviour)
"""

from __future__ import annotations

import csv
import io
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.backtesting.audit import (
    DataQualityAuditReport,
    audit_csv,
    save_audit_report,
    _count_weekend_gaps,
    _estimate_missing_d1_gaps,
)
from src.backtesting.backtest_runner import (
    BacktestConfig,
    BacktestResult,
    BacktestRunner,
    StrategyStats,
    StrategyValidationLab,
    ValidationReport,
)
from src.backtesting.reports import (
    generate_scorecard,
    generate_comparison_report,
    generate_validation_report,
)

# ---------------------------------------------------------------------------
# Constants / helpers
# ---------------------------------------------------------------------------
REAL_DATA = ROOT / "data" / "EURUSD_D1_2014_2026.csv"
REPORTS_DIR = ROOT / "reports"

_MINIMAL_CSV = (
    "date,open,high,low,close,volume\n"
    "2024-01-02,1.10000,1.11000,1.09000,1.10500,1000\n"
    "2024-01-03,1.10500,1.11500,1.10000,1.11000,900\n"
    "2024-01-04,1.11000,1.12000,1.10500,1.11500,800\n"
)

_INVALID_OHLC_CSV = (
    "date,open,high,low,close,volume\n"
    "2024-01-02,1.10000,1.09000,1.11000,1.10500,1000\n"  # high < low  → invalid
    "2024-01-03,1.10500,1.11500,1.10000,1.11000,900\n"
)

_DUPE_CSV = (
    "date,open,high,low,close,volume\n"
    "2024-01-02,1.10000,1.11000,1.09000,1.10500,1000\n"
    "2024-01-02,1.10000,1.11000,1.09000,1.10500,1000\n"  # duplicate
    "2024-01-03,1.10500,1.11500,1.10000,1.11000,900\n"
)

_MISSING_COL_CSV = (
    "date,open,high,low,volume\n"  # missing 'close'
    "2024-01-02,1.10000,1.11000,1.09000,1000\n"
)


def _write_tmp(content: str, suffix: str = ".csv") -> Path:
    """Write content to a temp file and return its Path."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "w") as f:
        f.write(content)
    return Path(path)


def _make_result(**kwargs) -> BacktestResult:
    """Build a minimal BacktestResult for testing."""
    defaults = dict(
        symbol="EURUSD",
        timeframe="D1",
        strategy_mode="combined",
        data_source="test",
        date_range=(None, None),
        candles_processed=0,
        trades_generated=0,
        trades_approved=0,
        trades_rejected=0,
        trades_executed=0,
        wins=0,
        losses=0,
        initial_balance=10_000.0,
        final_balance=10_000.0,
        gross_profit=0.0,
        gross_loss=0.0,
        max_drawdown_pct=0.0,
        total_candles_loaded=0,
        duplicate_candles=0,
        missing_candles=0,
        invalid_rows=0,
    )
    defaults.update(kwargs)
    return BacktestResult(**defaults)


# ===========================================================================
# 1. DataQualityAuditReport — dataclass & properties
# ===========================================================================

class TestDataQualityAuditReportFields:
    """Test DataQualityAuditReport dataclass construction and defaults."""

    def test_required_fields_present(self):
        r = DataQualityAuditReport(file_name="f.csv", symbol="EURUSD", timeframe="D1")
        assert r.file_name == "f.csv"
        assert r.symbol == "EURUSD"
        assert r.timeframe == "D1"

    def test_defaults_are_zero_or_empty(self):
        r = DataQualityAuditReport(file_name="f.csv", symbol="EURUSD", timeframe="D1")
        assert r.total_rows == 0
        assert r.valid_rows == 0
        assert r.invalid_rows == 0
        assert r.duplicate_rows == 0
        assert r.missing_dates == 0
        assert r.weekend_gaps == 0
        assert r.date_range == (None, None)
        assert r.has_required_cols is False
        assert r.chronological is True
        assert r.ohlc_pass_rate == 0.0
        assert r.null_counts == {}
        assert r.warnings == []
        assert r.errors == []

    def test_valid_rows_pct_no_rows(self):
        r = DataQualityAuditReport(file_name="f.csv", symbol="EURUSD", timeframe="D1")
        assert r.valid_rows_pct == 0.0

    def test_valid_rows_pct_all_valid(self):
        r = DataQualityAuditReport(
            file_name="f.csv", symbol="EURUSD", timeframe="D1",
            total_rows=100, valid_rows=100
        )
        assert r.valid_rows_pct == 100.0

    def test_valid_rows_pct_partial(self):
        r = DataQualityAuditReport(
            file_name="f.csv", symbol="EURUSD", timeframe="D1",
            total_rows=200, valid_rows=196
        )
        assert r.valid_rows_pct == pytest.approx(98.0)

    def test_invalid_rows_pct(self):
        r = DataQualityAuditReport(
            file_name="f.csv", symbol="EURUSD", timeframe="D1",
            total_rows=100, valid_rows=99
        )
        assert r.invalid_rows_pct == pytest.approx(1.0)

    def test_passes_quality_gate_all_good(self):
        r = DataQualityAuditReport(
            file_name="f.csv", symbol="EURUSD", timeframe="D1",
            total_rows=100, valid_rows=100,
            has_required_cols=True,
        )
        assert r.passes_quality_gate is True

    def test_fails_gate_missing_cols(self):
        r = DataQualityAuditReport(
            file_name="f.csv", symbol="EURUSD", timeframe="D1",
            total_rows=100, valid_rows=100,
            has_required_cols=False,
        )
        assert r.passes_quality_gate is False

    def test_fails_gate_errors_present(self):
        r = DataQualityAuditReport(
            file_name="f.csv", symbol="EURUSD", timeframe="D1",
            total_rows=100, valid_rows=100,
            has_required_cols=True,
            errors=["Something went wrong"],
        )
        assert r.passes_quality_gate is False

    def test_fails_gate_below_99pct(self):
        r = DataQualityAuditReport(
            file_name="f.csv", symbol="EURUSD", timeframe="D1",
            total_rows=100, valid_rows=98,
            has_required_cols=True,
        )
        assert r.passes_quality_gate is False

    def test_passes_gate_exactly_99pct(self):
        r = DataQualityAuditReport(
            file_name="f.csv", symbol="EURUSD", timeframe="D1",
            total_rows=100, valid_rows=99,
            has_required_cols=True,
        )
        assert r.passes_quality_gate is True

    def test_coverage_years_no_dates(self):
        r = DataQualityAuditReport(file_name="f.csv", symbol="EURUSD", timeframe="D1")
        assert r.coverage_years == 0.0

    def test_coverage_years_one_year(self):
        start = datetime(2023, 1, 1, tzinfo=timezone.utc)
        end   = datetime(2024, 1, 1, tzinfo=timezone.utc)
        r = DataQualityAuditReport(
            file_name="f.csv", symbol="EURUSD", timeframe="D1",
            date_range=(start, end),
        )
        assert r.coverage_years == pytest.approx(365 / 365.25, abs=0.01)

    def test_summary_returns_string(self):
        r = DataQualityAuditReport(file_name="f.csv", symbol="EURUSD", timeframe="D1")
        s = r.summary()
        assert isinstance(s, str)
        assert "DATA QUALITY AUDIT REPORT" in s
        assert "EURUSD" in s

    def test_summary_contains_quality_gate_status(self):
        r = DataQualityAuditReport(
            file_name="f.csv", symbol="EURUSD", timeframe="D1",
            total_rows=100, valid_rows=100, has_required_cols=True,
        )
        s = r.summary()
        assert "PASS" in s

    def test_summary_fail_shows_fail(self):
        r = DataQualityAuditReport(file_name="f.csv", symbol="EURUSD", timeframe="D1")
        s = r.summary()
        assert "FAIL" in s


# ===========================================================================
# 2. audit_csv() — with temp files
# ===========================================================================

class TestAuditCsvValid:
    """audit_csv() on a minimal valid CSV."""

    def test_returns_dataqualityauditreport(self, tmp_path):
        f = tmp_path / "test.csv"
        f.write_text(_MINIMAL_CSV)
        r = audit_csv(f)
        assert isinstance(r, DataQualityAuditReport)

    def test_file_name_set_correctly(self, tmp_path):
        f = tmp_path / "mydata.csv"
        f.write_text(_MINIMAL_CSV)
        r = audit_csv(f)
        assert r.file_name == "mydata.csv"

    def test_total_rows(self, tmp_path):
        f = tmp_path / "test.csv"
        f.write_text(_MINIMAL_CSV)
        r = audit_csv(f)
        assert r.total_rows == 3

    def test_valid_rows_all_pass(self, tmp_path):
        f = tmp_path / "test.csv"
        f.write_text(_MINIMAL_CSV)
        r = audit_csv(f)
        assert r.valid_rows == 3

    def test_invalid_rows_zero(self, tmp_path):
        f = tmp_path / "test.csv"
        f.write_text(_MINIMAL_CSV)
        r = audit_csv(f)
        assert r.invalid_rows == 0

    def test_has_required_cols_true(self, tmp_path):
        f = tmp_path / "test.csv"
        f.write_text(_MINIMAL_CSV)
        r = audit_csv(f)
        assert r.has_required_cols is True

    def test_passes_quality_gate(self, tmp_path):
        f = tmp_path / "test.csv"
        f.write_text(_MINIMAL_CSV)
        r = audit_csv(f)
        assert r.passes_quality_gate is True

    def test_date_range_set(self, tmp_path):
        f = tmp_path / "test.csv"
        f.write_text(_MINIMAL_CSV)
        r = audit_csv(f)
        start, end = r.date_range
        assert start is not None
        assert end is not None
        assert start <= end

    def test_chronological_true(self, tmp_path):
        f = tmp_path / "test.csv"
        f.write_text(_MINIMAL_CSV)
        r = audit_csv(f)
        assert r.chronological is True

    def test_ohlc_pass_rate_1(self, tmp_path):
        f = tmp_path / "test.csv"
        f.write_text(_MINIMAL_CSV)
        r = audit_csv(f)
        assert r.ohlc_pass_rate == pytest.approx(1.0)


class TestAuditCsvInvalid:
    """audit_csv() on files with quality issues."""

    def test_invalid_ohlc_detected(self, tmp_path):
        f = tmp_path / "bad_ohlc.csv"
        f.write_text(_INVALID_OHLC_CSV)
        r = audit_csv(f)
        assert r.invalid_rows >= 1

    def test_invalid_ohlc_reduces_valid_rows(self, tmp_path):
        f = tmp_path / "bad_ohlc.csv"
        f.write_text(_INVALID_OHLC_CSV)
        r = audit_csv(f)
        assert r.valid_rows < r.total_rows

    def test_duplicate_rows_detected(self, tmp_path):
        f = tmp_path / "dupe.csv"
        f.write_text(_DUPE_CSV)
        r = audit_csv(f)
        assert r.duplicate_rows >= 1

    def test_missing_col_sets_error(self, tmp_path):
        f = tmp_path / "no_close.csv"
        f.write_text(_MISSING_COL_CSV)
        r = audit_csv(f)
        assert len(r.errors) > 0

    def test_missing_col_fails_gate(self, tmp_path):
        f = tmp_path / "no_close.csv"
        f.write_text(_MISSING_COL_CSV)
        r = audit_csv(f)
        assert r.passes_quality_gate is False

    def test_nonexistent_file_returns_error(self, tmp_path):
        r = audit_csv(tmp_path / "does_not_exist.csv")
        assert len(r.errors) > 0
        assert r.passes_quality_gate is False

    def test_empty_file_returns_error(self, tmp_path):
        f = tmp_path / "empty.csv"
        f.write_text("date,open,high,low,close,volume\n")
        r = audit_csv(f)
        assert len(r.errors) > 0


# ===========================================================================
# 3. save_audit_report() — file I/O
# ===========================================================================

class TestSaveAuditReport:
    """save_audit_report() writes the report to disk."""

    def test_creates_file(self, tmp_path):
        r = DataQualityAuditReport(
            file_name="f.csv", symbol="EURUSD", timeframe="D1",
            total_rows=3, valid_rows=3, has_required_cols=True,
        )
        out = tmp_path / "report.txt"
        save_audit_report(r, out)
        assert out.exists()

    def test_file_not_empty(self, tmp_path):
        r = DataQualityAuditReport(file_name="f.csv", symbol="EURUSD", timeframe="D1")
        out = tmp_path / "report.txt"
        save_audit_report(r, out)
        assert out.stat().st_size > 0

    def test_file_contains_symbol(self, tmp_path):
        r = DataQualityAuditReport(file_name="f.csv", symbol="EURUSD", timeframe="D1")
        out = tmp_path / "report.txt"
        save_audit_report(r, out)
        content = out.read_text()
        assert "EURUSD" in content

    def test_creates_parent_dirs(self, tmp_path):
        r = DataQualityAuditReport(file_name="f.csv", symbol="EURUSD", timeframe="D1")
        out = tmp_path / "nested" / "deep" / "report.txt"
        save_audit_report(r, out)
        assert out.exists()

    def test_accepts_string_path(self, tmp_path):
        r = DataQualityAuditReport(file_name="f.csv", symbol="EURUSD", timeframe="D1")
        out = str(tmp_path / "report.txt")
        save_audit_report(r, out)
        assert Path(out).exists()


# ===========================================================================
# 4. Real data quality report (if EURUSD_D1_2014_2026.csv exists)
# ===========================================================================

@pytest.mark.skipif(not REAL_DATA.exists(), reason="real data file not present")
class TestRealDataAudit:
    """Validate the already-generated data quality report expectations."""

    def test_real_data_passes_quality_gate(self):
        r = audit_csv(REAL_DATA)
        assert r.passes_quality_gate is True

    def test_real_data_total_rows_gte_3000(self):
        r = audit_csv(REAL_DATA)
        assert r.total_rows >= 3000

    def test_real_data_valid_rows_pct_gte_99(self):
        r = audit_csv(REAL_DATA)
        assert r.valid_rows_pct >= 99.0

    def test_real_data_has_required_cols(self):
        r = audit_csv(REAL_DATA)
        assert r.has_required_cols is True

    def test_real_data_no_errors(self):
        r = audit_csv(REAL_DATA)
        assert r.errors == []

    def test_real_data_chronological(self):
        r = audit_csv(REAL_DATA)
        assert r.chronological is True

    def test_real_data_date_range_starts_before_2015(self):
        r = audit_csv(REAL_DATA)
        start, _ = r.date_range
        assert start is not None
        assert start.year <= 2015

    def test_real_data_coverage_gt_5_years(self):
        r = audit_csv(REAL_DATA)
        assert r.coverage_years >= 5.0


# ===========================================================================
# 5. Saved report files (generated in run_sprint14_backtests.py)
# ===========================================================================

@pytest.mark.skipif(
    not (REPORTS_DIR / "data_quality_report.txt").exists(),
    reason="data_quality_report.txt not generated yet"
)
class TestSavedReportFiles:
    """Verify the Sprint 14 report files exist and contain expected content."""

    def test_data_quality_report_exists(self):
        assert (REPORTS_DIR / "data_quality_report.txt").exists()

    def test_data_quality_report_contains_pass(self):
        text = (REPORTS_DIR / "data_quality_report.txt").read_text()
        assert "PASS" in text

    def test_pin_bar_scorecard_exists(self):
        assert (REPORTS_DIR / "pin_bar_scorecard.txt").exists()

    def test_engulfing_scorecard_exists(self):
        assert (REPORTS_DIR / "engulfing_scorecard.txt").exists()

    def test_combined_scorecard_exists(self):
        assert (REPORTS_DIR / "combined_scorecard.txt").exists()

    def test_validation_lab_report_exists(self):
        assert (REPORTS_DIR / "validation_lab_report.txt").exists()

    def test_pin_bar_scorecard_contains_eurusd(self):
        text = (REPORTS_DIR / "pin_bar_scorecard.txt").read_text()
        assert "EURUSD" in text

    def test_combined_scorecard_baseline_fail(self):
        text = (REPORTS_DIR / "combined_scorecard.txt").read_text()
        assert "NO" in text  # Baseline pass: ❌ NO


# ===========================================================================
# 6. BacktestResult.passes_baseline logic
# ===========================================================================

class TestPassesBaseline:
    """passes_baseline property on BacktestResult."""

    def test_no_trades_fails_baseline(self):
        r = _make_result(
            wins=0, losses=0, trades_executed=0,
            gross_profit=0.0, gross_loss=0.0,
            max_drawdown_pct=0.0,
        )
        assert r.passes_baseline is False

    def test_passing_baseline(self):
        r = _make_result(
            wins=10, losses=5, trades_executed=15,
            gross_profit=15.0, gross_loss=5.0,
            profit_factor=3.0,    # PF = 3.0
            win_rate=10/15,       # ~0.667
            max_drawdown_pct=5.0,
        )
        assert r.passes_baseline is True

    def test_high_drawdown_fails(self):
        r = _make_result(
            wins=10, losses=5, trades_executed=15,
            gross_profit=15.0, gross_loss=5.0,
            max_drawdown_pct=25.0,   # > 20%
        )
        assert r.passes_baseline is False

    def test_low_profit_factor_fails(self):
        r = _make_result(
            wins=5, losses=10, trades_executed=15,
            gross_profit=5.0, gross_loss=10.0,  # PF = 0.5
            max_drawdown_pct=5.0,
        )
        assert r.passes_baseline is False

    def test_passes_baseline_exactly_threshold(self):
        # PF = 1.1, win_rate = 0.40, DD = 20%, trades = 10
        r = _make_result(
            wins=4, losses=6, trades_executed=10,
            gross_profit=11.0, gross_loss=10.0,
            profit_factor=1.1,   # exactly 1.1
            win_rate=0.40,       # exactly 0.40
            max_drawdown_pct=20.0,
        )
        assert r.passes_baseline is True


# ===========================================================================
# 7. generate_scorecard / generate_comparison_report / generate_validation_report
# ===========================================================================

class TestGenerateScorecard:
    """generate_scorecard() output format tests."""

    def test_returns_string(self):
        r = _make_result()
        s = generate_scorecard(r)
        assert isinstance(s, str)

    def test_contains_symbol(self):
        r = _make_result(symbol="EURUSD")
        s = generate_scorecard(r)
        assert "EURUSD" in s

    def test_contains_trade_stats_section(self):
        r = _make_result()
        s = generate_scorecard(r)
        assert "TRADE STATISTICS" in s

    def test_contains_performance_section(self):
        r = _make_result()
        s = generate_scorecard(r)
        assert "PERFORMANCE METRICS" in s

    def test_baseline_no_shown_in_no_trades_result(self):
        r = _make_result()
        s = generate_scorecard(r)
        assert "NO" in s

    def test_error_message_included_in_output(self):
        r = _make_result(error_message="Test error message")
        s = generate_scorecard(r)
        assert "Test error message" in s


class TestGenerateComparisonReport:
    """generate_comparison_report() multi-result output."""

    def test_single_result(self):
        r = _make_result(strategy_mode="pin_bar_only")
        s = generate_comparison_report([r])
        assert isinstance(s, str)
        assert "EURUSD" in s

    def test_three_results(self):
        r1 = _make_result(strategy_mode="pin_bar_only")
        r2 = _make_result(strategy_mode="engulfing_only")
        r3 = _make_result(strategy_mode="combined")
        s = generate_comparison_report([r1, r2, r3])
        assert isinstance(s, str)
        assert "COMPARISON" in s.upper()

    def test_empty_list(self):
        s = generate_comparison_report([])
        assert isinstance(s, str)


class TestGenerateValidationReport:
    """generate_validation_report() with a ValidationReport."""

    def test_returns_string(self):
        pb  = _make_result(strategy_mode="pin_bar_only")
        eng = _make_result(strategy_mode="engulfing_only")
        com = _make_result(strategy_mode="combined")
        vr  = ValidationReport(pin_bar_result=pb, engulfing_result=eng, combined_result=com)
        s   = generate_validation_report(vr)
        assert isinstance(s, str)

    def test_contains_eurusd(self):
        pb  = _make_result()
        eng = _make_result()
        com = _make_result()
        vr  = ValidationReport(pin_bar_result=pb, engulfing_result=eng, combined_result=com)
        s   = generate_validation_report(vr)
        assert "EURUSD" in s


# ===========================================================================
# 8. StrategyValidationLab with real data
# ===========================================================================

@pytest.mark.skipif(not REAL_DATA.exists(), reason="real data file not present")
class TestStrategyValidationLabRealData:
    """Sprint 14: validation lab produces well-formed output on real EURUSD data."""

    def test_lab_run_returns_validation_report(self):
        lab    = StrategyValidationLab(BacktestConfig())
        report = lab.run(str(REAL_DATA))
        assert isinstance(report, ValidationReport)

    def test_lab_report_has_all_three_results(self):
        lab    = StrategyValidationLab(BacktestConfig())
        report = lab.run(str(REAL_DATA))
        assert report.pin_bar_result is not None
        assert report.engulfing_result is not None
        assert report.combined_result is not None

    def test_lab_candles_processed_matches_data(self):
        lab    = StrategyValidationLab(BacktestConfig())
        report = lab.run(str(REAL_DATA))
        # All modes should have processed all candles
        assert report.combined_result.candles_processed >= 3000

    def test_lab_data_source_set(self):
        lab    = StrategyValidationLab(BacktestConfig())
        report = lab.run(str(REAL_DATA))
        assert report.pin_bar_result.data_source != ""

    def test_lab_invalid_rows_zero(self):
        """Clean OHLC data should yield 0 invalid rows reported."""
        lab    = StrategyValidationLab(BacktestConfig())
        report = lab.run(str(REAL_DATA))
        assert report.combined_result.invalid_rows == 0

    def test_lab_generate_validation_report_string(self):
        lab    = StrategyValidationLab(BacktestConfig())
        report = lab.run(str(REAL_DATA))
        s      = lab.generate_validation_report(report)
        assert isinstance(s, str)
        assert len(s) > 100


# ===========================================================================
# 9. Private helper functions
# ===========================================================================

class TestCountWeekendGaps:
    """_count_weekend_gaps() helper."""

    def test_no_timestamps(self):
        assert _count_weekend_gaps([]) == 0

    def test_single_timestamp(self):
        ts = [datetime(2024, 1, 2, tzinfo=timezone.utc)]
        assert _count_weekend_gaps(ts) == 0

    def test_consecutive_days_no_gap(self):
        ts = [
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 1, 2, tzinfo=timezone.utc),
            datetime(2024, 1, 3, tzinfo=timezone.utc),
        ]
        assert _count_weekend_gaps(ts) == 0

    def test_friday_monday_gap_counted(self):
        # Fri→Mon = 3 days gap (gap = 3 calendar days)
        ts = [
            datetime(2024, 1, 5, tzinfo=timezone.utc),  # Fri
            datetime(2024, 1, 8, tzinfo=timezone.utc),  # Mon
        ]
        result = _count_weekend_gaps(ts)
        assert result == 1


class TestEstimateMissingD1Gaps:
    """_estimate_missing_d1_gaps() helper."""

    def test_no_timestamps_returns_zero(self):
        assert _estimate_missing_d1_gaps([]) == 0

    def test_single_timestamp_returns_zero(self):
        ts = [datetime(2024, 1, 2, tzinfo=timezone.utc)]
        assert _estimate_missing_d1_gaps(ts) == 0

    def test_normal_gaps_no_missing(self):
        # Daily gaps of 1-3 days → not missing
        ts = [
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 1, 2, tzinfo=timezone.utc),
            datetime(2024, 1, 5, tzinfo=timezone.utc),  # weekend = 3 days
        ]
        assert _estimate_missing_d1_gaps(ts) == 0

    def test_large_gap_detected_as_missing(self):
        # 10-day gap → should detect missing
        ts = [
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 1, 11, tzinfo=timezone.utc),  # 10 calendar days
        ]
        result = _estimate_missing_d1_gaps(ts)
        assert result >= 1
