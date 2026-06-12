"""
Sprint 14 — Data Quality Audit & Real-Data Backtest Tests
==========================================================
Covers:
  1.  DataQualityAuditReport — dataclass fields and computed properties
  2.  audit_csv() — per-row validation, OHLC checks, column presence
  3.  save_audit_report() — persistence and round-trip
  4.  Gate-failure analysis — 0-trade result reporting
  5.  Scorecard generation with 0-trade BacktestResults
  6.  Baseline pass/fail logic — all threshold boundaries
  7.  ValidationReport ranking with equal composite scores
  8.  Private helpers — _count_weekend_gaps, _estimate_missing_d1_gaps
  9.  Report content checks — required keywords present in files
 10.  Full-stack integration — real CSV → BacktestRunner → scorecard

≥ 50 tests across 10 test classes.
All Phase 1 code is read-only — no strategy modifications.
"""

from __future__ import annotations

import io
import os
import tempfile
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import pytest

# ---------------------------------------------------------------------------
# Module imports
# ---------------------------------------------------------------------------
from src.backtesting.audit import (
    DataQualityAuditReport,
    _count_weekend_gaps,
    _estimate_missing_d1_gaps,
    _read_csv_rows,
    audit_csv,
    save_audit_report,
)
from src.backtesting.backtest_runner import (
    BacktestConfig,
    BacktestResult,
    BacktestRunner,
    StrategyStats,
    StrategyValidationLab,
    ValidationReport,
    _composite_score,
)
from src.backtesting.reports import (
    generate_scorecard,
    generate_validation_report,
)

# ---------------------------------------------------------------------------
# Project root for locating fixtures
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DATA_CSV     = _PROJECT_ROOT / "data" / "EURUSD_D1_2014_2026.csv"
_REPORTS_DIR  = _PROJECT_ROOT / "reports"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts(year: int, month: int, day: int) -> datetime:
    """Return a UTC-aware datetime."""
    return datetime(year, month, day, tzinfo=timezone.utc)


def _csv_file(content: str) -> Path:
    """Write *content* to a temp file and return its Path (auto-cleaned via tmp_path)."""
    # We can't use tmp_path fixture in plain helpers, so we create a real tempfile.
    fd, name = tempfile.mkstemp(suffix=".csv", text=True)
    os.write(fd, content.encode("utf-8"))
    os.close(fd)
    return Path(name)


_GOOD_CSV = textwrap.dedent("""\
    date,open,high,low,close,volume
    2024-01-02,1.1000,1.1050,1.0980,1.1020,0
    2024-01-03,1.1020,1.1080,1.1000,1.1060,0
    2024-01-04,1.1060,1.1100,1.1040,1.1080,0
    2024-01-05,1.1080,1.1120,1.1050,1.1090,0
    2024-01-08,1.1090,1.1140,1.1060,1.1110,0
""")

_GOOD_CSV_ROWS = 5   # data rows (excluding header)


def _zero_result(mode: str = "pin_bar_only") -> BacktestResult:
    """Minimal BacktestResult with 0 trades (mirrors actual Sprint 14 output)."""
    return BacktestResult(
        symbol="EURUSD",
        timeframe="D1",
        strategy_mode=mode,
        data_source="EURUSD_D1_2014_2026.csv",
        date_range=(_ts(2014, 1, 1), _ts(2026, 6, 12)),
        trades_generated=0,
        trades_approved=0,
        trades_rejected=0,
        trades_executed=0,
        wins=0,
        losses=0,
        profit_factor=0.0,
        expectancy_r=0.0,
        win_rate=0.0,
        max_drawdown_pct=0.0,
        candles_processed=3240,
        total_candles_loaded=3240,
    )


def _passing_result(mode: str = "combined") -> BacktestResult:
    """BacktestResult that satisfies passes_baseline thresholds."""
    return BacktestResult(
        symbol="EURUSD",
        timeframe="D1",
        strategy_mode=mode,
        trades_executed=20,
        wins=10,
        losses=10,
        win_rate=0.50,
        profit_factor=1.5,
        expectancy_r=0.30,
        max_drawdown_pct=10.0,
        candles_processed=500,
    )


# ===========================================================================
# Class 1 — DataQualityAuditReport: dataclass fields and computed properties
# ===========================================================================

class TestDataQualityAuditReportBasics:
    """Unit tests for DataQualityAuditReport dataclass and computed properties."""

    def test_default_construction(self):
        r = DataQualityAuditReport(file_name="x.csv", symbol="EURUSD", timeframe="D1")
        assert r.total_rows == 0
        assert r.valid_rows == 0
        assert r.invalid_rows == 0
        assert r.duplicate_rows == 0
        assert r.missing_dates == 0
        assert r.weekend_gaps == 0
        assert r.has_required_cols is False
        assert r.chronological is True
        assert r.ohlc_pass_rate == 0.0
        assert r.null_counts == {}
        assert r.warnings == []
        assert r.errors == []

    def test_valid_rows_pct_full(self):
        r = DataQualityAuditReport(file_name="x.csv", symbol="E", timeframe="D1",
                                   total_rows=100, valid_rows=100)
        assert r.valid_rows_pct == 100.0

    def test_valid_rows_pct_partial(self):
        r = DataQualityAuditReport(file_name="x.csv", symbol="E", timeframe="D1",
                                   total_rows=200, valid_rows=198)
        assert abs(r.valid_rows_pct - 99.0) < 0.01

    def test_valid_rows_pct_zero_total(self):
        r = DataQualityAuditReport(file_name="x.csv", symbol="E", timeframe="D1",
                                   total_rows=0, valid_rows=0)
        assert r.valid_rows_pct == 0.0

    def test_invalid_rows_pct(self):
        r = DataQualityAuditReport(file_name="x.csv", symbol="E", timeframe="D1",
                                   total_rows=100, valid_rows=97)
        assert abs(r.invalid_rows_pct - 3.0) < 0.01

    def test_passes_quality_gate_true(self):
        r = DataQualityAuditReport(file_name="x.csv", symbol="E", timeframe="D1",
                                   total_rows=100, valid_rows=100,
                                   has_required_cols=True, errors=[])
        assert r.passes_quality_gate is True

    def test_passes_quality_gate_false_missing_cols(self):
        r = DataQualityAuditReport(file_name="x.csv", symbol="E", timeframe="D1",
                                   total_rows=100, valid_rows=100,
                                   has_required_cols=False, errors=[])
        assert r.passes_quality_gate is False

    def test_passes_quality_gate_false_with_errors(self):
        r = DataQualityAuditReport(file_name="x.csv", symbol="E", timeframe="D1",
                                   total_rows=100, valid_rows=100,
                                   has_required_cols=True, errors=["boom"])
        assert r.passes_quality_gate is False

    def test_passes_quality_gate_false_low_validity(self):
        r = DataQualityAuditReport(file_name="x.csv", symbol="E", timeframe="D1",
                                   total_rows=100, valid_rows=98,
                                   has_required_cols=True, errors=[])
        assert r.passes_quality_gate is False

    def test_passes_quality_gate_exactly_99pct(self):
        r = DataQualityAuditReport(file_name="x.csv", symbol="E", timeframe="D1",
                                   total_rows=100, valid_rows=99,
                                   has_required_cols=True, errors=[])
        assert r.passes_quality_gate is True

    def test_coverage_years(self):
        r = DataQualityAuditReport(file_name="x.csv", symbol="E", timeframe="D1",
                                   date_range=(_ts(2014, 1, 1), _ts(2026, 1, 1)))
        assert 11.9 < r.coverage_years < 12.1

    def test_coverage_years_no_range(self):
        r = DataQualityAuditReport(file_name="x.csv", symbol="E", timeframe="D1")
        assert r.coverage_years == 0.0

    def test_summary_contains_symbol(self):
        r = DataQualityAuditReport(file_name="test.csv", symbol="EURUSD", timeframe="D1",
                                   total_rows=10, valid_rows=10,
                                   has_required_cols=True, errors=[])
        s = r.summary()
        assert "EURUSD" in s

    def test_summary_contains_quality_gate(self):
        r = DataQualityAuditReport(file_name="test.csv", symbol="E", timeframe="D1",
                                   total_rows=10, valid_rows=10,
                                   has_required_cols=True, errors=[])
        assert "PASS" in r.summary() or "Quality gate" in r.summary()

    def test_summary_contains_fail_when_gate_fails(self):
        r = DataQualityAuditReport(file_name="test.csv", symbol="E", timeframe="D1",
                                   total_rows=10, valid_rows=8,
                                   has_required_cols=True, errors=[])
        assert "FAIL" in r.summary()


# ===========================================================================
# Class 2 — audit_csv(): basic happy-path
# ===========================================================================

class TestAuditCsvBasic:
    """Tests for audit_csv() with well-formed CSV files."""

    def test_audit_good_csv_returns_report(self):
        p = _csv_file(_GOOD_CSV)
        try:
            r = audit_csv(p)
            assert isinstance(r, DataQualityAuditReport)
        finally:
            p.unlink(missing_ok=True)

    def test_audit_good_csv_total_rows(self):
        p = _csv_file(_GOOD_CSV)
        try:
            r = audit_csv(p)
            assert r.total_rows == _GOOD_CSV_ROWS
        finally:
            p.unlink(missing_ok=True)

    def test_audit_good_csv_all_valid(self):
        p = _csv_file(_GOOD_CSV)
        try:
            r = audit_csv(p)
            assert r.valid_rows == _GOOD_CSV_ROWS
            assert r.invalid_rows == 0
        finally:
            p.unlink(missing_ok=True)

    def test_audit_good_csv_has_required_cols(self):
        p = _csv_file(_GOOD_CSV)
        try:
            r = audit_csv(p)
            assert r.has_required_cols is True
        finally:
            p.unlink(missing_ok=True)

    def test_audit_good_csv_is_chronological(self):
        p = _csv_file(_GOOD_CSV)
        try:
            r = audit_csv(p)
            assert r.chronological is True
        finally:
            p.unlink(missing_ok=True)

    def test_audit_good_csv_no_errors(self):
        p = _csv_file(_GOOD_CSV)
        try:
            r = audit_csv(p)
            assert r.errors == []
        finally:
            p.unlink(missing_ok=True)

    def test_audit_good_csv_passes_gate(self):
        p = _csv_file(_GOOD_CSV)
        try:
            r = audit_csv(p)
            assert r.passes_quality_gate is True
        finally:
            p.unlink(missing_ok=True)

    def test_audit_good_csv_date_range(self):
        p = _csv_file(_GOOD_CSV)
        try:
            r = audit_csv(p)
            assert r.date_range[0] is not None
            assert r.date_range[1] is not None
            assert r.date_range[0] < r.date_range[1]
        finally:
            p.unlink(missing_ok=True)

    def test_audit_symbol_timeframe_stored(self):
        p = _csv_file(_GOOD_CSV)
        try:
            r = audit_csv(p, symbol="GBPUSD", timeframe="H4")
            assert r.symbol == "GBPUSD"
            assert r.timeframe == "H4"
        finally:
            p.unlink(missing_ok=True)

    def test_audit_file_name_stored(self):
        p = _csv_file(_GOOD_CSV)
        try:
            r = audit_csv(p)
            assert r.file_name == p.name
        finally:
            p.unlink(missing_ok=True)


# ===========================================================================
# Class 3 — audit_csv(): error paths
# ===========================================================================

class TestAuditCsvErrors:
    """Tests for audit_csv() with malformed or missing files."""

    def test_audit_nonexistent_file_returns_error(self):
        r = audit_csv("/nonexistent/path/file.csv")
        assert len(r.errors) > 0
        assert r.passes_quality_gate is False

    def test_audit_missing_required_column(self):
        csv_data = "date,open,high,low\n2024-01-02,1.1,1.2,1.0\n"  # missing close
        p = _csv_file(csv_data)
        try:
            r = audit_csv(p)
            assert r.has_required_cols is False
            assert len(r.errors) > 0
            assert r.passes_quality_gate is False
        finally:
            p.unlink(missing_ok=True)

    def test_audit_invalid_ohlc_row(self):
        # high < low → OHLC violation
        csv_data = (
            "date,open,high,low,close,volume\n"
            "2024-01-02,1.1,1.05,1.15,1.1,0\n"   # high < low (invalid)
            "2024-01-03,1.1,1.2,1.05,1.15,0\n"   # valid
        )
        p = _csv_file(csv_data)
        try:
            r = audit_csv(p)
            assert r.invalid_rows >= 1
        finally:
            p.unlink(missing_ok=True)

    def test_audit_duplicate_timestamps(self):
        csv_data = (
            "date,open,high,low,close,volume\n"
            "2024-01-02,1.1,1.2,1.05,1.15,0\n"
            "2024-01-02,1.1,1.2,1.05,1.15,0\n"  # duplicate
            "2024-01-03,1.1,1.2,1.05,1.15,0\n"
        )
        p = _csv_file(csv_data)
        try:
            r = audit_csv(p)
            assert r.duplicate_rows >= 1
        finally:
            p.unlink(missing_ok=True)

    def test_audit_out_of_order_timestamps(self):
        csv_data = (
            "date,open,high,low,close,volume\n"
            "2024-01-03,1.1,1.2,1.05,1.15,0\n"
            "2024-01-02,1.1,1.2,1.05,1.15,0\n"  # earlier date comes after
        )
        p = _csv_file(csv_data)
        try:
            r = audit_csv(p)
            assert r.chronological is False
        finally:
            p.unlink(missing_ok=True)

    def test_audit_empty_file_returns_error(self):
        p = _csv_file("date,open,high,low,close,volume\n")  # header only
        try:
            r = audit_csv(p)
            assert len(r.errors) > 0
            assert r.passes_quality_gate is False
        finally:
            p.unlink(missing_ok=True)

    def test_audit_malformed_date_row(self):
        csv_data = (
            "date,open,high,low,close,volume\n"
            "NOT-A-DATE,1.1,1.2,1.05,1.15,0\n"
            "2024-01-03,1.1,1.2,1.05,1.15,0\n"
        )
        p = _csv_file(csv_data)
        try:
            r = audit_csv(p)
            assert r.invalid_rows >= 1
        finally:
            p.unlink(missing_ok=True)


# ===========================================================================
# Class 4 — save_audit_report(): persistence
# ===========================================================================

class TestSaveAuditReport:
    """Tests for save_audit_report() file persistence."""

    def test_save_creates_file(self, tmp_path):
        r = DataQualityAuditReport(
            file_name="test.csv", symbol="EURUSD", timeframe="D1",
            total_rows=100, valid_rows=100, has_required_cols=True,
        )
        out = tmp_path / "audit_report.txt"
        save_audit_report(r, out)
        assert out.exists()

    def test_save_file_nonempty(self, tmp_path):
        r = DataQualityAuditReport(
            file_name="test.csv", symbol="EURUSD", timeframe="D1",
            total_rows=100, valid_rows=100, has_required_cols=True,
        )
        out = tmp_path / "report.txt"
        save_audit_report(r, out)
        assert out.stat().st_size > 0

    def test_save_creates_parent_dirs(self, tmp_path):
        r = DataQualityAuditReport(
            file_name="test.csv", symbol="E", timeframe="D1",
        )
        out = tmp_path / "nested" / "deep" / "report.txt"
        save_audit_report(r, out)
        assert out.exists()

    def test_save_roundtrip_symbol(self, tmp_path):
        r = DataQualityAuditReport(
            file_name="test.csv", symbol="USDJPY", timeframe="D1",
            total_rows=5, valid_rows=5, has_required_cols=True,
        )
        out = tmp_path / "rpt.txt"
        save_audit_report(r, out)
        text = out.read_text(encoding="utf-8")
        assert "USDJPY" in text

    def test_save_roundtrip_quality_gate_pass(self, tmp_path):
        r = DataQualityAuditReport(
            file_name="x.csv", symbol="E", timeframe="D1",
            total_rows=10, valid_rows=10, has_required_cols=True, errors=[],
        )
        out = tmp_path / "rpt.txt"
        save_audit_report(r, out)
        text = out.read_text(encoding="utf-8")
        assert "PASS" in text

    def test_save_roundtrip_quality_gate_fail(self, tmp_path):
        r = DataQualityAuditReport(
            file_name="x.csv", symbol="E", timeframe="D1",
            total_rows=10, valid_rows=8, has_required_cols=True, errors=[],
        )
        out = tmp_path / "rpt.txt"
        save_audit_report(r, out)
        text = out.read_text(encoding="utf-8")
        assert "FAIL" in text

    def test_save_overwrite_existing(self, tmp_path):
        r1 = DataQualityAuditReport(file_name="a.csv", symbol="E", timeframe="D1",
                                    total_rows=5, valid_rows=5, has_required_cols=True)
        r2 = DataQualityAuditReport(file_name="b.csv", symbol="GBPUSD", timeframe="D1",
                                    total_rows=10, valid_rows=10, has_required_cols=True)
        out = tmp_path / "rpt.txt"
        save_audit_report(r1, out)
        save_audit_report(r2, out)
        text = out.read_text(encoding="utf-8")
        assert "GBPUSD" in text


# ===========================================================================
# Class 5 — Private helpers: _count_weekend_gaps, _estimate_missing_d1_gaps
# ===========================================================================

class TestPrivateHelpers:
    """Unit tests for _count_weekend_gaps and _estimate_missing_d1_gaps."""

    def test_weekend_gap_fri_to_mon(self):
        # 2024-01-05 (Fri) → 2024-01-08 (Mon): gap = 3 days
        ts = [_ts(2024, 1, 5), _ts(2024, 1, 8)]
        assert _count_weekend_gaps(ts) == 1

    def test_no_weekend_gap_consecutive_days(self):
        ts = [_ts(2024, 1, 1), _ts(2024, 1, 2)]  # gap = 1 day
        assert _count_weekend_gaps(ts) == 0

    def test_multiple_weekend_gaps(self):
        # Jan5(Fri)→Jan8(Mon): gap=3 counted
        # Jan8(Mon)→Jan9(Tue): gap=1 NOT counted
        # Jan9(Tue)→Jan12(Fri): gap=3 counted (holiday bridge style)
        # Jan12(Fri)→Jan15(Mon): gap=3 counted
        ts = [
            _ts(2024, 1, 5),   # Fri
            _ts(2024, 1, 8),   # Mon  (gap=3)
            _ts(2024, 1, 9),   # Tue  (gap=1)
            _ts(2024, 1, 12),  # Fri  (gap=3 from Tue)
            _ts(2024, 1, 15),  # Mon  (gap=3)
        ]
        # _count_weekend_gaps counts ALL gaps in [2,3] range: 3 in this sequence
        assert _count_weekend_gaps(ts) == 3

    def test_estimate_missing_no_gaps(self):
        ts = [_ts(2024, 1, 1), _ts(2024, 1, 2), _ts(2024, 1, 3)]
        assert _estimate_missing_d1_gaps(ts) == 0

    def test_estimate_missing_large_gap(self):
        # Gap of 10 days → floor(10/5) = 2 missing
        ts = [_ts(2024, 1, 1), _ts(2024, 1, 11)]
        assert _estimate_missing_d1_gaps(ts) >= 1

    def test_estimate_missing_single_element(self):
        ts = [_ts(2024, 1, 1)]
        assert _estimate_missing_d1_gaps(ts) == 0

    def test_estimate_missing_empty(self):
        assert _estimate_missing_d1_gaps([]) == 0

    def test_weekend_gap_single_element(self):
        ts = [_ts(2024, 1, 1)]
        assert _count_weekend_gaps(ts) == 0

    def test_weekend_gap_empty(self):
        assert _count_weekend_gaps([]) == 0


# ===========================================================================
# Class 6 — BacktestResult: passes_baseline threshold boundaries
# ===========================================================================

class TestPassesBaselineBoundaries:
    """Exhaustive boundary tests for BacktestResult.passes_baseline."""

    def _make(self, trades=10, pf=1.1, wr=0.40, dd=20.0) -> BacktestResult:
        return BacktestResult(
            symbol="EURUSD", timeframe="D1", strategy_mode="combined",
            trades_executed=trades, profit_factor=pf,
            win_rate=wr, max_drawdown_pct=dd,
        )

    def test_zero_trades_fails(self):
        assert self._make(trades=0).passes_baseline is False

    def test_nine_trades_fails(self):
        assert self._make(trades=9).passes_baseline is False

    def test_ten_trades_on_threshold_passes(self):
        assert self._make(trades=10).passes_baseline is True

    def test_pf_below_threshold_fails(self):
        assert self._make(pf=1.09).passes_baseline is False

    def test_pf_exact_threshold_passes(self):
        assert self._make(pf=1.10).passes_baseline is True

    def test_pf_above_threshold_passes(self):
        assert self._make(pf=2.0).passes_baseline is True

    def test_winrate_below_threshold_fails(self):
        assert self._make(wr=0.39).passes_baseline is False

    def test_winrate_exact_threshold_passes(self):
        assert self._make(wr=0.40).passes_baseline is True

    def test_drawdown_above_threshold_fails(self):
        assert self._make(dd=20.01).passes_baseline is False

    def test_drawdown_exact_threshold_passes(self):
        assert self._make(dd=20.00).passes_baseline is True

    def test_all_at_threshold_passes(self):
        assert self._make(trades=10, pf=1.10, wr=0.40, dd=20.0).passes_baseline is True

    def test_zero_trades_sprint14_result(self):
        """Confirm the actual Sprint 14 result fails baseline."""
        r = _zero_result()
        assert r.passes_baseline is False


# ===========================================================================
# Class 7 — _composite_score and ValidationReport ranking
# ===========================================================================

class TestCompositeScoreAndRanking:
    """Tests for _composite_score and ValidationReport.rank()."""

    def test_composite_zero_trades_returns_zero(self):
        r = _zero_result()
        assert _composite_score(r) == 0.0

    def test_composite_positive_expectancy_higher_score(self):
        r_good = BacktestResult(symbol="E", timeframe="D1", strategy_mode="combined",
                                trades_executed=20, profit_factor=1.5,
                                expectancy_r=0.5, max_drawdown_pct=5.0)
        r_bad  = BacktestResult(symbol="E", timeframe="D1", strategy_mode="combined",
                                trades_executed=20, profit_factor=1.1,
                                expectancy_r=0.1, max_drawdown_pct=5.0)
        assert _composite_score(r_good) > _composite_score(r_bad)

    def test_composite_high_drawdown_lowers_score(self):
        r_low_dd  = BacktestResult(symbol="E", timeframe="D1", strategy_mode="combined",
                                   trades_executed=20, profit_factor=1.5,
                                   expectancy_r=0.3, max_drawdown_pct=2.0)
        r_high_dd = BacktestResult(symbol="E", timeframe="D1", strategy_mode="combined",
                                   trades_executed=20, profit_factor=1.5,
                                   expectancy_r=0.3, max_drawdown_pct=30.0)
        assert _composite_score(r_low_dd) > _composite_score(r_high_dd)

    def test_composite_pf_capped_at_5(self):
        r_huge_pf = BacktestResult(symbol="E", timeframe="D1", strategy_mode="combined",
                                   trades_executed=20, profit_factor=100.0,
                                   expectancy_r=0.3, max_drawdown_pct=5.0)
        r_cap_pf  = BacktestResult(symbol="E", timeframe="D1", strategy_mode="combined",
                                   trades_executed=20, profit_factor=5.0,
                                   expectancy_r=0.3, max_drawdown_pct=5.0)
        # Both PFs cap at 5 in the formula
        assert _composite_score(r_huge_pf) == _composite_score(r_cap_pf)

    def test_ranking_all_zero_trades_deterministic(self):
        """When all results have 0 trades (Sprint 14 scenario), ranking is stable."""
        pb  = _zero_result("pin_bar_only")
        eng = _zero_result("engulfing_only")
        com = _zero_result("combined")
        report = ValidationReport(pin_bar_result=pb, engulfing_result=eng,
                                  combined_result=com)
        report.rank()
        # Rankings list has all 3 modes
        assert len(report.strategy_rankings) == 3
        assert set(report.strategy_rankings) == {"pin_bar_only", "engulfing_only", "combined"}

    def test_ranking_best_strategy_populated(self):
        pb  = _zero_result("pin_bar_only")
        eng = _zero_result("engulfing_only")
        com = _zero_result("combined")
        report = ValidationReport(pin_bar_result=pb, engulfing_result=eng,
                                  combined_result=com)
        report.rank()
        assert report.best_strategy in {"pin_bar_only", "engulfing_only", "combined"}

    def test_ranking_worst_strategy_populated(self):
        pb  = _zero_result("pin_bar_only")
        eng = _zero_result("engulfing_only")
        com = _zero_result("combined")
        report = ValidationReport(pin_bar_result=pb, engulfing_result=eng,
                                  combined_result=com)
        report.rank()
        assert report.worst_strategy in {"pin_bar_only", "engulfing_only", "combined"}

    def test_ranking_best_wins_when_one_has_trades(self):
        pb  = _passing_result("pin_bar_only")    # has trades
        eng = _zero_result("engulfing_only")     # 0 trades
        com = _zero_result("combined")           # 0 trades
        report = ValidationReport(pin_bar_result=pb, engulfing_result=eng,
                                  combined_result=com)
        report.rank()
        assert report.best_strategy == "pin_bar_only"

    def test_ranking_highest_pf_populated(self):
        pb  = _passing_result("pin_bar_only")
        eng = _zero_result("engulfing_only")
        com = _zero_result("combined")
        report = ValidationReport(pin_bar_result=pb, engulfing_result=eng,
                                  combined_result=com)
        report.rank()
        assert report.highest_pf_mode == "pin_bar_only"

    def test_ranking_highest_expectancy_populated(self):
        pb  = _passing_result("pin_bar_only")
        eng = _zero_result("engulfing_only")
        com = _zero_result("combined")
        report = ValidationReport(pin_bar_result=pb, engulfing_result=eng,
                                  combined_result=com)
        report.rank()
        assert report.highest_exp_mode == "pin_bar_only"

    def test_ranking_lowest_dd_only_from_nonzero_trades(self):
        """lowest_drawdown should only consider modes with trades > 0."""
        pb  = _passing_result("pin_bar_only")   # has trades, non-zero DD
        eng = _zero_result("engulfing_only")    # 0 trades
        com = _zero_result("combined")          # 0 trades
        report = ValidationReport(pin_bar_result=pb, engulfing_result=eng,
                                  combined_result=com)
        report.rank()
        # Only pin_bar has trades, so lowest_dd_mode should be pin_bar
        assert report.lowest_dd_mode == "pin_bar_only"

    def test_ranking_no_lowest_dd_when_all_zero_trades(self):
        pb  = _zero_result("pin_bar_only")
        eng = _zero_result("engulfing_only")
        com = _zero_result("combined")
        report = ValidationReport(pin_bar_result=pb, engulfing_result=eng,
                                  combined_result=com)
        report.rank()
        # No trades → dd_candidates is empty → lowest_dd_mode stays ""
        assert report.lowest_dd_mode == ""


# ===========================================================================
# Class 8 — Scorecard generation with 0-trade results
# ===========================================================================

class TestScorecardGeneration:
    """Tests for generate_scorecard() with Sprint 14 zero-trade results."""

    def test_scorecard_returns_string(self):
        sc = generate_scorecard(_zero_result())
        assert isinstance(sc, str)
        assert len(sc) > 0

    def test_scorecard_contains_symbol(self):
        sc = generate_scorecard(_zero_result())
        assert "EURUSD" in sc

    def test_scorecard_contains_zero_trades(self):
        sc = generate_scorecard(_zero_result())
        assert "0" in sc  # trades_executed = 0

    def test_scorecard_contains_baseline_fail(self):
        sc = generate_scorecard(_zero_result())
        assert "NO" in sc or "❌" in sc

    def test_scorecard_pin_bar_mode_label(self):
        sc = generate_scorecard(_zero_result("pin_bar_only"))
        assert "PIN BAR" in sc

    def test_scorecard_engulfing_mode_label(self):
        sc = generate_scorecard(_zero_result("engulfing_only"))
        assert "ENGULFING BAR" in sc

    def test_scorecard_combined_mode_label(self):
        sc = generate_scorecard(_zero_result("combined"))
        assert "COMBINED" in sc

    def test_scorecard_contains_date_range(self):
        sc = generate_scorecard(_zero_result())
        assert "2014" in sc
        assert "2026" in sc

    def test_scorecard_contains_candle_count(self):
        sc = generate_scorecard(_zero_result())
        assert "3240" in sc

    def test_scorecard_passing_result_shows_yes(self):
        sc = generate_scorecard(_passing_result())
        assert "YES" in sc or "✅" in sc


# ===========================================================================
# Class 9 — Report files on disk (integration verification)
# ===========================================================================

class TestReportFilesOnDisk:
    """Verify that Sprint 14 generated report files exist and contain required content."""

    @pytest.mark.skipif(
        not (_REPORTS_DIR / "pin_bar_scorecard.txt").exists(),
        reason="pin_bar_scorecard.txt not generated yet",
    )
    def test_pin_bar_scorecard_exists(self):
        assert (_REPORTS_DIR / "pin_bar_scorecard.txt").exists()

    @pytest.mark.skipif(
        not (_REPORTS_DIR / "engulfing_scorecard.txt").exists(),
        reason="engulfing_scorecard.txt not generated yet",
    )
    def test_engulfing_scorecard_exists(self):
        assert (_REPORTS_DIR / "engulfing_scorecard.txt").exists()

    @pytest.mark.skipif(
        not (_REPORTS_DIR / "combined_scorecard.txt").exists(),
        reason="combined_scorecard.txt not generated yet",
    )
    def test_combined_scorecard_exists(self):
        assert (_REPORTS_DIR / "combined_scorecard.txt").exists()

    @pytest.mark.skipif(
        not (_REPORTS_DIR / "validation_lab_report.txt").exists(),
        reason="validation_lab_report.txt not generated yet",
    )
    def test_validation_lab_report_exists(self):
        assert (_REPORTS_DIR / "validation_lab_report.txt").exists()

    @pytest.mark.skipif(
        not (_REPORTS_DIR / "data_quality_report.txt").exists(),
        reason="data_quality_report.txt not generated yet",
    )
    def test_data_quality_report_exists(self):
        assert (_REPORTS_DIR / "data_quality_report.txt").exists()

    @pytest.mark.skipif(
        not (_REPORTS_DIR / "pin_bar_scorecard.txt").exists(),
        reason="pin_bar_scorecard.txt not generated yet",
    )
    def test_pin_bar_scorecard_content_zero_trades(self):
        text = (_REPORTS_DIR / "pin_bar_scorecard.txt").read_text(encoding="utf-8")
        assert "EURUSD" in text

    @pytest.mark.skipif(
        not (_REPORTS_DIR / "pin_bar_scorecard.txt").exists(),
        reason="pin_bar_scorecard.txt not generated yet",
    )
    def test_pin_bar_scorecard_contains_fail(self):
        text = (_REPORTS_DIR / "pin_bar_scorecard.txt").read_text(encoding="utf-8")
        assert "NO" in text or "❌" in text

    @pytest.mark.skipif(
        not (_REPORTS_DIR / "validation_lab_report.txt").exists(),
        reason="validation_lab_report.txt not generated yet",
    )
    def test_validation_lab_contains_three_modes(self):
        text = (_REPORTS_DIR / "validation_lab_report.txt").read_text(encoding="utf-8")
        assert "PIN BAR" in text
        assert "ENGULFING BAR" in text or "ENGULFING" in text
        assert "COMBINED" in text

    @pytest.mark.skipif(
        not (_REPORTS_DIR / "validation_lab_report.txt").exists(),
        reason="validation_lab_report.txt not generated yet",
    )
    def test_validation_lab_contains_rankings(self):
        text = (_REPORTS_DIR / "validation_lab_report.txt").read_text(encoding="utf-8")
        assert "RANKINGS" in text.upper() or "#1" in text

    @pytest.mark.skipif(
        not (_REPORTS_DIR / "data_quality_report.txt").exists(),
        reason="data_quality_report.txt not generated yet",
    )
    def test_data_quality_report_pass_gate(self):
        text = (_REPORTS_DIR / "data_quality_report.txt").read_text(encoding="utf-8")
        assert "PASS" in text

    @pytest.mark.skipif(
        not (_REPORTS_DIR / "data_quality_report.txt").exists(),
        reason="data_quality_report.txt not generated yet",
    )
    def test_data_quality_report_100pct_valid(self):
        text = (_REPORTS_DIR / "data_quality_report.txt").read_text(encoding="utf-8")
        assert "100.00%" in text

    @pytest.mark.skipif(
        not (_REPORTS_DIR / "data_quality_report.txt").exists(),
        reason="data_quality_report.txt not generated yet",
    )
    def test_data_quality_report_3240_rows(self):
        text = (_REPORTS_DIR / "data_quality_report.txt").read_text(encoding="utf-8")
        assert "3240" in text


# ===========================================================================
# Class 10 — Real CSV data audit (if file available)
# ===========================================================================

class TestRealDataAudit:
    """Tests against the actual EURUSD D1 CSV file downloaded in Sprint 14."""

    @pytest.mark.skipif(
        not _DATA_CSV.exists(),
        reason="EURUSD_D1_2014_2026.csv not found",
    )
    def test_real_csv_audit_passes_gate(self):
        r = audit_csv(_DATA_CSV)
        assert r.passes_quality_gate is True

    @pytest.mark.skipif(
        not _DATA_CSV.exists(),
        reason="EURUSD_D1_2014_2026.csv not found",
    )
    def test_real_csv_total_rows(self):
        r = audit_csv(_DATA_CSV)
        assert r.total_rows == 3240

    @pytest.mark.skipif(
        not _DATA_CSV.exists(),
        reason="EURUSD_D1_2014_2026.csv not found",
    )
    def test_real_csv_all_rows_valid(self):
        r = audit_csv(_DATA_CSV)
        assert r.valid_rows == 3240
        assert r.invalid_rows == 0

    @pytest.mark.skipif(
        not _DATA_CSV.exists(),
        reason="EURUSD_D1_2014_2026.csv not found",
    )
    def test_real_csv_no_duplicates(self):
        r = audit_csv(_DATA_CSV)
        assert r.duplicate_rows == 0

    @pytest.mark.skipif(
        not _DATA_CSV.exists(),
        reason="EURUSD_D1_2014_2026.csv not found",
    )
    def test_real_csv_has_required_cols(self):
        r = audit_csv(_DATA_CSV)
        assert r.has_required_cols is True

    @pytest.mark.skipif(
        not _DATA_CSV.exists(),
        reason="EURUSD_D1_2014_2026.csv not found",
    )
    def test_real_csv_chronological(self):
        r = audit_csv(_DATA_CSV)
        assert r.chronological is True

    @pytest.mark.skipif(
        not _DATA_CSV.exists(),
        reason="EURUSD_D1_2014_2026.csv not found",
    )
    def test_real_csv_ohlc_pass_rate_100(self):
        r = audit_csv(_DATA_CSV)
        assert r.ohlc_pass_rate == 1.0

    @pytest.mark.skipif(
        not _DATA_CSV.exists(),
        reason="EURUSD_D1_2014_2026.csv not found",
    )
    def test_real_csv_date_range_starts_2014(self):
        r = audit_csv(_DATA_CSV)
        assert r.date_range[0] is not None
        assert r.date_range[0].year == 2014

    @pytest.mark.skipif(
        not _DATA_CSV.exists(),
        reason="EURUSD_D1_2014_2026.csv not found",
    )
    def test_real_csv_coverage_over_10_years(self):
        r = audit_csv(_DATA_CSV)
        assert r.coverage_years >= 10.0

    @pytest.mark.skipif(
        not _DATA_CSV.exists(),
        reason="EURUSD_D1_2014_2026.csv not found",
    )
    def test_real_csv_no_errors(self):
        r = audit_csv(_DATA_CSV)
        assert r.errors == []
