"""
Sprint 14 — Comprehensive Test Suite
=====================================
≥ 50 tests covering:
  - DataQualityAuditReport dataclass fields and properties
  - audit_csv() against the real EURUSD D1 file and synthetic CSVs
  - save_audit_report() file output
  - passes_quality_gate logic (all combinations)
  - BacktestRunner + StrategyValidationLab with real data
  - Scorecard and report generation
  - Baseline pass/fail criteria
  - Sprint 14 diagnostic findings (0-trade root cause)

Classes
-------
TestDataQualityAuditReportFields       — 12 tests
TestDataQualityAuditReportProperties   — 10 tests
TestAuditCsvRealFile                   — 10 tests
TestAuditCsvSyntheticFiles             — 10 tests
TestSaveAuditReport                    —  5 tests
TestBacktestRunnerRealData             —  6 tests
TestStrategyValidationLabRealData      —  5 tests
TestScorecardGeneration                —  5 tests
TestBaselinePassCriteria               —  5 tests
TestDiagnosticFindings                 —  5 tests
                                  Total: 73 tests
"""
from __future__ import annotations

import csv
import io
import os
import textwrap
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import pytest

from src.backtesting.audit import (
    DataQualityAuditReport,
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
)
from src.backtesting.reports import (
    generate_scorecard,
    generate_comparison_report,
    generate_validation_report,
)

# ── Path constants ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.parent
REAL_CSV     = PROJECT_ROOT / "data" / "EURUSD_D1_2014_2026.csv"
REPORTS_DIR  = PROJECT_ROOT / "reports"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_csv(rows: list[list], tmp_path: Path, name: str = "test.csv") -> Path:
    """Write a list-of-lists CSV to a temp file and return its path."""
    p = tmp_path / name
    with open(p, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    return p


_VALID_HEADER = [["date", "open", "high", "low", "close", "volume"]]
_VALID_ROW_1  = ["2020-01-02", "1.1200", "1.1250", "1.1180", "1.1220", "1000"]
_VALID_ROW_2  = ["2020-01-03", "1.1220", "1.1260", "1.1200", "1.1240", "1100"]
_VALID_ROW_3  = ["2020-01-06", "1.1240", "1.1280", "1.1220", "1.1260", "1200"]


# ═══════════════════════════════════════════════════════════════════════════════
# 1. TestDataQualityAuditReportFields  (12 tests)
# ═══════════════════════════════════════════════════════════════════════════════

class TestDataQualityAuditReportFields:
    """Verify DataQualityAuditReport stores each required field correctly."""

    def test_file_name_stored(self):
        r = DataQualityAuditReport(file_name="test.csv", symbol="EURUSD", timeframe="D1")
        assert r.file_name == "test.csv"

    def test_symbol_stored(self):
        r = DataQualityAuditReport(file_name="f.csv", symbol="GBPUSD", timeframe="H1")
        assert r.symbol == "GBPUSD"

    def test_timeframe_stored(self):
        r = DataQualityAuditReport(file_name="f.csv", symbol="EURUSD", timeframe="H4")
        assert r.timeframe == "H4"

    def test_total_rows_default_zero(self):
        r = DataQualityAuditReport(file_name="f.csv", symbol="EURUSD", timeframe="D1")
        assert r.total_rows == 0

    def test_valid_rows_default_zero(self):
        r = DataQualityAuditReport(file_name="f.csv", symbol="EURUSD", timeframe="D1")
        assert r.valid_rows == 0

    def test_invalid_rows_default_zero(self):
        r = DataQualityAuditReport(file_name="f.csv", symbol="EURUSD", timeframe="D1")
        assert r.invalid_rows == 0

    def test_duplicate_rows_default_zero(self):
        r = DataQualityAuditReport(file_name="f.csv", symbol="EURUSD", timeframe="D1")
        assert r.duplicate_rows == 0

    def test_missing_dates_default_zero(self):
        r = DataQualityAuditReport(file_name="f.csv", symbol="EURUSD", timeframe="D1")
        assert r.missing_dates == 0

    def test_weekend_gaps_default_zero(self):
        r = DataQualityAuditReport(file_name="f.csv", symbol="EURUSD", timeframe="D1")
        assert r.weekend_gaps == 0

    def test_date_range_default_none(self):
        r = DataQualityAuditReport(file_name="f.csv", symbol="EURUSD", timeframe="D1")
        assert r.date_range == (None, None)

    def test_warnings_default_empty_list(self):
        r = DataQualityAuditReport(file_name="f.csv", symbol="EURUSD", timeframe="D1")
        assert r.warnings == []

    def test_errors_default_empty_list(self):
        r = DataQualityAuditReport(file_name="f.csv", symbol="EURUSD", timeframe="D1")
        assert r.errors == []


# ═══════════════════════════════════════════════════════════════════════════════
# 2. TestDataQualityAuditReportProperties  (10 tests)
# ═══════════════════════════════════════════════════════════════════════════════

class TestDataQualityAuditReportProperties:
    """Test computed properties: valid_rows_pct, passes_quality_gate, etc."""

    def _perfect(self) -> DataQualityAuditReport:
        now = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 12, 31, tzinfo=timezone.utc)
        return DataQualityAuditReport(
            file_name="ok.csv",
            symbol="EURUSD",
            timeframe="D1",
            total_rows=1000,
            valid_rows=1000,
            invalid_rows=0,
            has_required_cols=True,
            date_range=(now, end),
        )

    def test_valid_rows_pct_100(self):
        r = self._perfect()
        assert r.valid_rows_pct == pytest.approx(100.0)

    def test_valid_rows_pct_partial(self):
        r = self._perfect()
        r.total_rows = 200
        r.valid_rows = 198
        assert r.valid_rows_pct == pytest.approx(99.0)

    def test_valid_rows_pct_zero_total(self):
        r = DataQualityAuditReport(file_name="f.csv", symbol="EURUSD", timeframe="D1")
        assert r.valid_rows_pct == 0.0

    def test_invalid_rows_pct_complement(self):
        r = self._perfect()
        r.total_rows = 100
        r.valid_rows = 95
        assert r.invalid_rows_pct == pytest.approx(5.0)

    def test_passes_quality_gate_true_when_100pct(self):
        r = self._perfect()
        assert r.passes_quality_gate is True

    def test_passes_quality_gate_false_when_errors(self):
        r = self._perfect()
        r.errors = ["fatal error"]
        assert r.passes_quality_gate is False

    def test_passes_quality_gate_false_when_missing_cols(self):
        r = self._perfect()
        r.has_required_cols = False
        assert r.passes_quality_gate is False

    def test_passes_quality_gate_false_when_below_99pct(self):
        r = self._perfect()
        r.total_rows = 100
        r.valid_rows = 98
        assert r.passes_quality_gate is False

    def test_passes_quality_gate_exactly_99pct(self):
        r = self._perfect()
        r.total_rows = 100
        r.valid_rows = 99
        assert r.passes_quality_gate is True

    def test_coverage_years_computed(self):
        start = datetime(2020, 1, 1, tzinfo=timezone.utc)
        end   = datetime(2022, 1, 1, tzinfo=timezone.utc)
        r = DataQualityAuditReport(
            file_name="f.csv", symbol="EURUSD", timeframe="D1",
            date_range=(start, end)
        )
        assert r.coverage_years > 1.9
        assert r.coverage_years < 2.1


# ═══════════════════════════════════════════════════════════════════════════════
# 3. TestAuditCsvRealFile  (10 tests)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(
    not REAL_CSV.exists(),
    reason="Real EURUSD data file not present"
)
class TestAuditCsvRealFile:
    """Audit the actual EURUSD_D1_2014_2026.csv file."""

    @pytest.fixture(scope="class")
    def report(self):
        return audit_csv(REAL_CSV, symbol="EURUSD", timeframe="D1")

    def test_file_name_correct(self, report):
        assert report.file_name == "EURUSD_D1_2014_2026.csv"

    def test_symbol_uppercase(self, report):
        assert report.symbol == "EURUSD"

    def test_timeframe_uppercase(self, report):
        assert report.timeframe == "D1"

    def test_total_rows_3240(self, report):
        assert report.total_rows == 3240

    def test_zero_invalid_rows(self, report):
        assert report.invalid_rows == 0

    def test_zero_duplicate_rows(self, report):
        assert report.duplicate_rows == 0

    def test_has_required_cols(self, report):
        assert report.has_required_cols is True

    def test_valid_rows_pct_100(self, report):
        assert report.valid_rows_pct == pytest.approx(100.0)

    def test_passes_quality_gate(self, report):
        assert report.passes_quality_gate is True

    def test_date_range_starts_2014(self, report):
        start, end = report.date_range
        assert start is not None
        assert start.year == 2014

    def test_coverage_at_least_10_years(self, report):
        assert report.coverage_years >= 10.0

    def test_weekend_gaps_detected(self, report):
        # 12 years of data → lots of Fri→Mon gaps
        assert report.weekend_gaps > 500

    def test_no_fatal_errors(self, report):
        assert report.errors == []

    def test_chronological_order(self, report):
        assert report.chronological is True

    def test_ohlc_pass_rate_is_one(self, report):
        assert report.ohlc_pass_rate == pytest.approx(1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. TestAuditCsvSyntheticFiles  (10 tests)
# ═══════════════════════════════════════════════════════════════════════════════

class TestAuditCsvSyntheticFiles:
    """Test audit_csv() against various synthetic CSV inputs."""

    def test_perfect_file_passes_gate(self, tmp_path):
        rows = _VALID_HEADER + [_VALID_ROW_1, _VALID_ROW_2, _VALID_ROW_3]
        p = _make_csv(rows, tmp_path)
        r = audit_csv(p)
        assert r.passes_quality_gate is True

    def test_missing_file_returns_error(self, tmp_path):
        r = audit_csv(tmp_path / "nonexistent.csv")
        assert len(r.errors) > 0

    def test_empty_file_returns_error(self, tmp_path):
        p = tmp_path / "empty.csv"
        p.write_text("")
        r = audit_csv(p)
        assert len(r.errors) > 0

    def test_header_only_returns_error(self, tmp_path):
        p = _make_csv(_VALID_HEADER, tmp_path)
        r = audit_csv(p)
        assert len(r.errors) > 0

    def test_missing_required_column_fails_gate(self, tmp_path):
        rows = [["date", "open", "high", "low"],  # missing close, volume
                ["2020-01-02", "1.12", "1.13", "1.11"]]
        p = _make_csv(rows, tmp_path)
        r = audit_csv(p)
        assert r.passes_quality_gate is False
        assert r.has_required_cols is False

    def test_duplicate_timestamps_counted(self, tmp_path):
        rows = _VALID_HEADER + [_VALID_ROW_1, _VALID_ROW_1, _VALID_ROW_2]
        p = _make_csv(rows, tmp_path)
        r = audit_csv(p)
        assert r.duplicate_rows >= 1

    def test_invalid_ohlc_row_counted(self, tmp_path):
        bad_row = ["2020-01-02", "1.12", "1.10", "1.13", "1.11", "1000"]  # high < low
        rows = _VALID_HEADER + [bad_row, _VALID_ROW_2]
        p = _make_csv(rows, tmp_path)
        r = audit_csv(p)
        assert r.invalid_rows >= 1

    def test_symbol_uppercased(self, tmp_path):
        rows = _VALID_HEADER + [_VALID_ROW_1]
        p = _make_csv(rows, tmp_path)
        r = audit_csv(p, symbol="eurusd")
        assert r.symbol == "EURUSD"

    def test_timeframe_uppercased(self, tmp_path):
        rows = _VALID_HEADER + [_VALID_ROW_1]
        p = _make_csv(rows, tmp_path)
        r = audit_csv(p, timeframe="d1")
        assert r.timeframe == "D1"

    def test_total_rows_matches_data(self, tmp_path):
        rows = _VALID_HEADER + [_VALID_ROW_1, _VALID_ROW_2, _VALID_ROW_3]
        p = _make_csv(rows, tmp_path)
        r = audit_csv(p)
        assert r.total_rows == 3


# ═══════════════════════════════════════════════════════════════════════════════
# 5. TestSaveAuditReport  (5 tests)
# ═══════════════════════════════════════════════════════════════════════════════

class TestSaveAuditReport:
    """Test save_audit_report() file writing."""

    def _minimal_report(self) -> DataQualityAuditReport:
        start = datetime(2020, 1, 1, tzinfo=timezone.utc)
        end   = datetime(2024, 12, 31, tzinfo=timezone.utc)
        return DataQualityAuditReport(
            file_name="test.csv",
            symbol="EURUSD",
            timeframe="D1",
            total_rows=500,
            valid_rows=500,
            has_required_cols=True,
            date_range=(start, end),
        )

    def test_creates_file(self, tmp_path):
        p = tmp_path / "report.txt"
        r = self._minimal_report()
        save_audit_report(r, p)
        assert p.exists()

    def test_file_contains_symbol(self, tmp_path):
        p = tmp_path / "report.txt"
        r = self._minimal_report()
        save_audit_report(r, p)
        assert "EURUSD" in p.read_text()

    def test_file_contains_total_rows(self, tmp_path):
        p = tmp_path / "report.txt"
        r = self._minimal_report()
        save_audit_report(r, p)
        assert "500" in p.read_text()

    def test_file_contains_quality_gate_status(self, tmp_path):
        p = tmp_path / "report.txt"
        r = self._minimal_report()
        save_audit_report(r, p)
        content = p.read_text()
        assert "PASS" in content or "FAIL" in content

    def test_creates_parent_directories(self, tmp_path):
        p = tmp_path / "nested" / "dir" / "report.txt"
        r = self._minimal_report()
        save_audit_report(r, p)
        assert p.exists()


# ═══════════════════════════════════════════════════════════════════════════════
# 6. TestBacktestRunnerRealData  (6 tests)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(
    not REAL_CSV.exists(),
    reason="Real EURUSD data file not present"
)
class TestBacktestRunnerRealData:
    """Run all 3 modes on real EURUSD D1 data and verify result shape."""

    @pytest.fixture(scope="class")
    def pin_result(self):
        cfg = BacktestConfig(lookback_window=50, minimum_tqs=0.0)
        runner = BacktestRunner(cfg)
        return runner.run_pin_bar_only(str(REAL_CSV))

    @pytest.fixture(scope="class")
    def eng_result(self):
        cfg = BacktestConfig(lookback_window=50, minimum_tqs=0.0)
        runner = BacktestRunner(cfg)
        return runner.run_engulfing_only(str(REAL_CSV))

    @pytest.fixture(scope="class")
    def com_result(self):
        cfg = BacktestConfig(lookback_window=50, minimum_tqs=0.0)
        runner = BacktestRunner(cfg)
        return runner.run_combined(str(REAL_CSV))

    def test_pin_bar_result_type(self, pin_result):
        assert isinstance(pin_result, BacktestResult)

    def test_engulfing_result_type(self, eng_result):
        assert isinstance(eng_result, BacktestResult)

    def test_combined_result_type(self, com_result):
        assert isinstance(com_result, BacktestResult)

    def test_pin_bar_candles_processed(self, pin_result):
        assert pin_result.candles_processed == 3240

    def test_engulfing_candles_processed(self, eng_result):
        assert eng_result.candles_processed == 3240

    def test_combined_candles_processed(self, com_result):
        assert com_result.candles_processed == 3240


# ═══════════════════════════════════════════════════════════════════════════════
# 7. TestStrategyValidationLabRealData  (5 tests)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(
    not REAL_CSV.exists(),
    reason="Real EURUSD data file not present"
)
class TestStrategyValidationLabRealData:
    """StrategyValidationLab produces a valid ValidationReport."""

    @pytest.fixture(scope="class")
    def lab_report(self):
        cfg = BacktestConfig(lookback_window=50, minimum_tqs=0.0)
        lab = StrategyValidationLab(cfg)
        return lab.run(str(REAL_CSV))

    def test_returns_validation_report(self, lab_report):
        assert isinstance(lab_report, ValidationReport)

    def test_has_pin_bar_result(self, lab_report):
        assert isinstance(lab_report.pin_bar_result, BacktestResult)

    def test_has_engulfing_result(self, lab_report):
        assert isinstance(lab_report.engulfing_result, BacktestResult)

    def test_has_combined_result(self, lab_report):
        assert isinstance(lab_report.combined_result, BacktestResult)

    def test_ranked_list_has_three_entries(self, lab_report):
        assert len(lab_report.strategy_rankings) == 3


# ═══════════════════════════════════════════════════════════════════════════════
# 8. TestScorecardGeneration  (5 tests)
# ═══════════════════════════════════════════════════════════════════════════════

class TestScorecardGeneration:
    """generate_scorecard() produces correct string output."""

    @pytest.fixture
    def zero_result(self):
        """A minimal BacktestResult with 0 trades."""
        return BacktestResult(
            symbol="EURUSD",
            timeframe="D1",
            strategy_mode="pin_bar_only",
            candles_processed=3240,
        )

    def test_scorecard_returns_string(self, zero_result):
        card = generate_scorecard(zero_result)
        assert isinstance(card, str)

    def test_scorecard_contains_symbol(self, zero_result):
        card = generate_scorecard(zero_result)
        assert "EURUSD" in card

    def test_scorecard_contains_mode(self, zero_result):
        card = generate_scorecard(zero_result)
        assert "pin_bar" in card.lower() or "PIN BAR" in card or "pin" in card.lower()

    def test_scorecard_contains_baseline_pass(self, zero_result):
        card = generate_scorecard(zero_result)
        assert "Baseline" in card or "BASELINE" in card or "baseline" in card

    def test_scorecard_contains_trade_count(self, zero_result):
        card = generate_scorecard(zero_result)
        assert "0" in card  # 0 trades should appear somewhere


# ═══════════════════════════════════════════════════════════════════════════════
# 9. TestBaselinePassCriteria  (5 tests)
# ═══════════════════════════════════════════════════════════════════════════════

class TestBaselinePassCriteria:
    """BacktestResult.passes_baseline enforces all 4 criteria."""

    def _result(self, trades=0, pf=0.0, wr=0.0, dd=0.0) -> BacktestResult:
        r = BacktestResult(symbol="EURUSD", timeframe="D1", strategy_mode="combined")
        r.trades_executed = trades
        r.profit_factor   = pf
        r.win_rate        = wr
        r.max_drawdown_pct = dd
        return r

    def test_zero_trades_fails_baseline(self):
        r = self._result(trades=0, pf=1.5, wr=0.5, dd=10.0)
        assert r.passes_baseline is False

    def test_insufficient_pf_fails_baseline(self):
        r = self._result(trades=20, pf=1.05, wr=0.5, dd=10.0)
        assert r.passes_baseline is False

    def test_insufficient_wr_fails_baseline(self):
        r = self._result(trades=20, pf=1.5, wr=0.35, dd=10.0)
        assert r.passes_baseline is False

    def test_high_drawdown_fails_baseline(self):
        r = self._result(trades=20, pf=1.5, wr=0.5, dd=25.0)
        assert r.passes_baseline is False

    def test_all_criteria_pass(self):
        r = self._result(trades=20, pf=1.2, wr=0.45, dd=15.0)
        assert r.passes_baseline is True


# ═══════════════════════════════════════════════════════════════════════════════
# 10. TestDiagnosticFindings  (5 tests)
# ═══════════════════════════════════════════════════════════════════════════════

class TestDiagnosticFindings:
    """
    Validate Sprint 14 diagnostic: SR engine produces levels when
    swing points are correctly supplied (M03→M05 wiring verification).
    """

    @pytest.fixture(scope="class")
    def fifty_candles(self):
        """Load first 50 candles from real CSV."""
        if not REAL_CSV.exists():
            pytest.skip("Real EURUSD data file not present")
        from src.backtesting.data_loader import load_candles_from_csv
        candles, _ = load_candles_from_csv(str(REAL_CSV))
        return candles[:50]

    @pytest.fixture(scope="class")
    def hundred_candles(self):
        """Load first 100 candles from real CSV."""
        if not REAL_CSV.exists():
            pytest.skip("Real EURUSD data file not present")
        from src.backtesting.data_loader import load_candles_from_csv
        candles, _ = load_candles_from_csv(str(REAL_CSV))
        return candles[:100]

    def test_sr_engine_without_swings_produces_no_levels(self, fifty_candles):
        """Confirms the pipeline bug: no swings → no SR levels."""
        from src.analysis.sr_engine import SREngine
        sr = SREngine(pip_size=0.0001)
        result = sr.analyze(fifty_candles)
        # Without swing points, SR produces 0 levels
        total = len(result.support_levels) + len(result.resistance_levels)
        assert total == 0, "Without swing points, SR must produce 0 levels"

    def test_sr_engine_with_swings_produces_levels(self, hundred_candles):
        """Confirms fix works: with swings → SR levels appear."""
        from src.analysis.market_structure import MarketStructureAnalyzer
        from src.analysis.sr_engine import SREngine
        structure = MarketStructureAnalyzer(lookback=5, pip_size=0.0001)
        sr = SREngine(pip_size=0.0001)
        struct_a = structure.analyze(hundred_candles)
        sr_result = sr.analyze(
            hundred_candles,
            swing_highs=struct_a.swing_highs,
            swing_lows=struct_a.swing_lows,
        )
        total = len(sr_result.support_levels) + len(sr_result.resistance_levels)
        assert total > 0, "With swing points, SR must produce ≥1 level"

    def test_market_structure_finds_swing_points(self, hundred_candles):
        """M03 MarketStructureAnalyzer finds swing highs and lows."""
        from src.analysis.market_structure import MarketStructureAnalyzer
        structure = MarketStructureAnalyzer(lookback=5, pip_size=0.0001)
        result = structure.analyze(hundred_candles)
        total_swings = len(result.swing_highs) + len(result.swing_lows)
        assert total_swings > 0, "M03 should detect swing points in 100 candles"

    def test_reports_dir_has_all_sprint14_files(self):
        """All 4 required Sprint 14 report files must exist."""
        required = [
            "data_quality_report.txt",
            "pin_bar_scorecard.txt",
            "engulfing_scorecard.txt",
            "combined_scorecard.txt",
            "validation_lab_report.txt",
        ]
        missing = [f for f in required if not (REPORTS_DIR / f).exists()]
        assert missing == [], f"Missing Sprint 14 reports: {missing}"

    def test_validation_lab_report_contains_diagnostic(self):
        """validation_lab_report.txt should contain root cause analysis."""
        val_path = REPORTS_DIR / "validation_lab_report.txt"
        if not val_path.exists():
            pytest.skip("validation_lab_report.txt not yet generated")
        content = val_path.read_text()
        # Should mention the SR wiring issue
        assert (
            "SR" in content
            or "swing" in content.lower()
            or "Level Gate" in content
            or "0 trades" in content.lower()
            or "ROOT CAUSE" in content
        ), "Validation report should document the SR wiring root cause"
