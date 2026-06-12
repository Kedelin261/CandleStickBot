"""
Sprint 13 — Tests for reports.py
==================================
Covers:
  - generate_scorecard() output structure and content
  - generate_scorecard() with error result
  - generate_scorecard() with no trades
  - generate_scorecard() with winning strategies
  - generate_scorecard() passes_baseline display
  - generate_scorecard() pin_bar and engulfing breakdown sections
  - generate_scorecard() review analysis section (M19)
  - generate_scorecard() data quality section
  - generate_comparison_report() with 1 result
  - generate_comparison_report() with 2 results
  - generate_comparison_report() with 3 results
  - generate_comparison_report() empty list
  - generate_comparison_report() all metrics present
  - generate_comparison_report() columns align with modes
  - generate_validation_report() rankings section
  - generate_validation_report() best/worst callout
  - generate_validation_report() recommendations section
  - generate_validation_report() includes comparison table
  - Private helpers: _mode_label, _fmt_pf, _fmt_dt, _fmt_range, _row, _get_result
  - DTO compatibility (to_dict round-trips)
  - BacktestRunner.generate_scorecard() delegation
  - BacktestRunner.generate_comparison_report() delegation
  - StrategyValidationLab.generate_validation_report() delegation

Total tests in this file: 40+
Combined with test_data_loader.py (66) + test_backtest_runner.py (79) → well over 100
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

import pytest

from src.backtesting.backtest_runner import (
    BacktestConfig,
    BacktestResult,
    BacktestRunner,
    StrategyStats,
    StrategyValidationLab,
    ValidationReport,
)
from src.backtesting.reports import (
    _fmt_dt,
    _fmt_pf,
    _fmt_range,
    _get_result,
    _mode_label,
    _row,
    generate_comparison_report,
    generate_scorecard,
    generate_validation_report,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_TS_NOW = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
_TS_START = datetime(2024, 1, 2, tzinfo=timezone.utc)
_TS_END   = datetime(2024, 12, 31, tzinfo=timezone.utc)


def _make_result(
    mode: str = "combined",
    trades: int = 15,
    wins: int = 9,
    losses: int = 6,
    pf: float = 1.5,
    wr: float = 0.60,
    dd: float = 8.0,
    exp: float = 0.35,
    net_pnl: float = 850.0,
    error: Optional[str] = None,
) -> BacktestResult:
    return BacktestResult(
        symbol="EURUSD",
        timeframe="D1",
        strategy_mode=mode,
        data_source="test_data.csv",
        date_range=(_TS_START, _TS_END),
        started_at=_TS_NOW,
        completed_at=_TS_NOW,
        trades_generated=trades + 5,
        trades_approved=trades + 2,
        trades_rejected=3,
        trades_executed=trades,
        wins=wins,
        losses=losses,
        initial_balance=10_000.0,
        final_balance=10_000.0 + net_pnl,
        net_profit_usd=net_pnl,
        gross_profit=9.0,
        gross_loss=6.0,
        profit_factor=pf,
        expectancy_r=exp,
        average_r=exp * 0.9,
        win_rate=wr,
        max_drawdown_pct=dd,
        max_consecutive_wins=4,
        max_consecutive_losses=2,
        candles_processed=500,
        pin_bar=StrategyStats(
            "pin_bar", trades=8, wins=5, losses=3,
            profit_factor=1.67, expectancy_r=0.40,
            avg_winner_r=2.1, avg_loser_r=-1.0,
            max_consecutive_wins=3, max_consecutive_losses=2,
            max_drawdown_pct=6.5,
        ),
        engulfing=StrategyStats(
            "engulfing_bar", trades=7, wins=4, losses=3,
            profit_factor=1.33, expectancy_r=0.25,
            avg_winner_r=1.8, avg_loser_r=-1.0,
            max_consecutive_wins=2, max_consecutive_losses=2,
            max_drawdown_pct=7.0,
        ),
        bad_signal=1,
        bad_regime=2,
        bad_level=1,
        bad_execution=0,
        normal_statistical=2,
        total_candles_loaded=600,
        duplicate_candles=2,
        missing_candles=3,
        invalid_rows=1,
        error_message=error,
    )


def _make_validation_report() -> ValidationReport:
    pb  = _make_result("pin_bar_only",  trades=12, pf=1.5, exp=0.35, dd=8.0)
    eng = _make_result("engulfing_only", trades=10, pf=1.3, exp=0.20, dd=12.0)
    com = _make_result("combined",       trades=20, pf=1.7, exp=0.45, dd=9.0)
    report = ValidationReport(
        pin_bar_result=pb,
        engulfing_result=eng,
        combined_result=com,
    )
    report.rank()
    return report


# ---------------------------------------------------------------------------
# 1. generate_scorecard() — structure
# ---------------------------------------------------------------------------

class TestGenerateScorecard:

    def test_returns_string(self):
        result = _make_result()
        output = generate_scorecard(result)
        assert isinstance(output, str)

    def test_non_empty_output(self):
        result = _make_result()
        output = generate_scorecard(result)
        assert len(output) > 200

    def test_header_contains_strategy_mode(self):
        result = _make_result(mode="pin_bar_only")
        output = generate_scorecard(result)
        assert "PIN BAR" in output

    def test_header_contains_engulfing(self):
        result = _make_result(mode="engulfing_only")
        output = generate_scorecard(result)
        assert "ENGULFING" in output

    def test_header_contains_combined(self):
        result = _make_result(mode="combined")
        output = generate_scorecard(result)
        assert "COMBINED" in output

    def test_symbol_in_output(self):
        output = generate_scorecard(_make_result())
        assert "EURUSD" in output

    def test_timeframe_in_output(self):
        output = generate_scorecard(_make_result())
        assert "D1" in output

    def test_trade_count_in_output(self):
        output = generate_scorecard(_make_result(trades=15))
        assert "15" in output

    def test_win_rate_in_output(self):
        output = generate_scorecard(_make_result(wr=0.60))
        assert "60.0%" in output

    def test_profit_factor_in_output(self):
        output = generate_scorecard(_make_result(pf=1.50))
        assert "1.50" in output

    def test_net_pnl_in_output(self):
        output = generate_scorecard(_make_result(net_pnl=850.0))
        assert "850" in output

    def test_max_drawdown_in_output(self):
        output = generate_scorecard(_make_result(dd=8.0))
        assert "8.00%" in output

    def test_passes_baseline_yes(self):
        # Sprint 15 FIX-4: N>=30 now required; updated trades=15 → 30.
        # exp=0.35 is the _make_result default, satisfying Expectancy > 0.
        result = _make_result(trades=30, pf=1.5, wr=0.55, dd=8.0)
        output = generate_scorecard(result)
        assert "YES" in output or "✅" in output

    def test_passes_baseline_no(self):
        result = _make_result(trades=5, pf=0.8, wr=0.30, dd=30.0)
        output = generate_scorecard(result)
        assert "NO" in output or "❌" in output

    def test_review_section_present(self):
        output = generate_scorecard(_make_result())
        assert "REVIEW" in output or "M19" in output

    def test_bad_signal_count_in_output(self):
        result = _make_result()
        output = generate_scorecard(result)
        assert "BAD_SIGNAL" in output

    def test_data_quality_section_present(self):
        output = generate_scorecard(_make_result())
        assert "DATA QUALITY" in output or "Candles loaded" in output

    def test_strategy_breakdown_section_present(self):
        output = generate_scorecard(_make_result())
        assert "BREAKDOWN" in output

    def test_pin_bar_breakdown_in_output(self):
        output = generate_scorecard(_make_result())
        assert "PIN_BAR" in output or "pin_bar" in output.lower()

    def test_engulfing_breakdown_in_output(self):
        output = generate_scorecard(_make_result())
        assert "ENGULFING_BAR" in output or "engulfing" in output.lower()


class TestGenerateScorecardErrorResult:

    def test_error_result_shows_error_message(self):
        result = _make_result(error="CSV file not found: /bad/path.csv")
        output = generate_scorecard(result)
        assert "ERROR" in output
        assert "CSV file not found" in output

    def test_error_result_still_shows_symbol(self):
        result = _make_result(error="load failed")
        output = generate_scorecard(result)
        assert "EURUSD" in output

    def test_error_result_shows_source(self):
        result = _make_result(error="load failed")
        output = generate_scorecard(result)
        assert "test_data.csv" in output


class TestGenerateScorecardNoTrades:

    def test_no_trades_still_produces_output(self):
        result = _make_result(trades=0, wins=0, losses=0, pf=0.0, wr=0.0)
        output = generate_scorecard(result)
        assert isinstance(output, str)
        assert len(output) > 100

    def test_no_trades_strategy_breakdown_shows_no_trades(self):
        result = BacktestResult(
            symbol="EURUSD", timeframe="D1", strategy_mode="combined",
            pin_bar=StrategyStats("pin_bar"),
            engulfing=StrategyStats("engulfing_bar"),
        )
        output = generate_scorecard(result)
        assert "no trades" in output.lower()


# ---------------------------------------------------------------------------
# 2. generate_comparison_report()
# ---------------------------------------------------------------------------

class TestGenerateComparisonReport:

    def test_empty_list_returns_string(self):
        output = generate_comparison_report([])
        assert isinstance(output, str)
        assert "No results" in output

    def test_single_result_returns_string(self):
        output = generate_comparison_report([_make_result()])
        assert isinstance(output, str)
        assert len(output) > 100

    def test_two_results_returns_string(self):
        r1 = _make_result("pin_bar_only")
        r2 = _make_result("engulfing_only")
        output = generate_comparison_report([r1, r2])
        assert isinstance(output, str)

    def test_three_results_returns_string(self):
        r1 = _make_result("pin_bar_only")
        r2 = _make_result("engulfing_only")
        r3 = _make_result("combined")
        output = generate_comparison_report([r1, r2, r3])
        assert isinstance(output, str)

    def test_pin_bar_label_in_output(self):
        r1 = _make_result("pin_bar_only")
        r2 = _make_result("engulfing_only")
        output = generate_comparison_report([r1, r2])
        assert "PIN BAR" in output

    def test_engulfing_label_in_output(self):
        r1 = _make_result("pin_bar_only")
        r2 = _make_result("engulfing_only")
        output = generate_comparison_report([r1, r2])
        assert "ENGULFING" in output

    def test_profit_factor_row_present(self):
        output = generate_comparison_report([_make_result()])
        assert "Profit Factor" in output or "Factor" in output

    def test_win_rate_row_present(self):
        output = generate_comparison_report([_make_result()])
        assert "Win Rate" in output

    def test_drawdown_row_present(self):
        output = generate_comparison_report([_make_result()])
        assert "Drawdown" in output or "drawdown" in output.lower()

    def test_trades_exec_row_present(self):
        output = generate_comparison_report([_make_result(trades=15)])
        assert "15" in output


# ---------------------------------------------------------------------------
# 3. generate_validation_report()
# ---------------------------------------------------------------------------

class TestGenerateValidationReport:

    def test_returns_string(self):
        report = _make_validation_report()
        output = generate_validation_report(report)
        assert isinstance(output, str)

    def test_non_empty(self):
        report = _make_validation_report()
        output = generate_validation_report(report)
        assert len(output) > 500

    def test_rankings_section_present(self):
        report = _make_validation_report()
        output = generate_validation_report(report)
        assert "RANKING" in output

    def test_best_strategy_shown(self):
        report = _make_validation_report()
        output = generate_validation_report(report)
        assert "Best strategy" in output or "best_strategy" in output.lower()

    def test_recommendations_section_present(self):
        report = _make_validation_report()
        output = generate_validation_report(report)
        assert "RECOMMENDATION" in output

    def test_comparison_table_embedded(self):
        report = _make_validation_report()
        output = generate_validation_report(report)
        assert "COMPARISON" in output

    def test_phase_1_header_present(self):
        report = _make_validation_report()
        output = generate_validation_report(report)
        assert "PHASE 1" in output or "VALIDATION" in output

    def test_symbol_in_output(self):
        report = _make_validation_report()
        output = generate_validation_report(report)
        assert "EURUSD" in output


# ---------------------------------------------------------------------------
# 4. Private helpers
# ---------------------------------------------------------------------------

class TestModeLabelHelper:

    def test_pin_bar_only(self):
        assert _mode_label("pin_bar_only") == "PIN BAR"

    def test_engulfing_only(self):
        assert _mode_label("engulfing_only") == "ENGULFING BAR"

    def test_combined(self):
        assert _mode_label("combined") == "COMBINED"

    def test_unknown_mode_uppercased(self):
        result = _mode_label("unknown_mode")
        assert result == "UNKNOWN_MODE"


class TestFmtPfHelper:

    def test_normal_pf(self):
        assert "1.50" in _fmt_pf(1.5)

    def test_infinity(self):
        assert "∞" in _fmt_pf(float("inf"))

    def test_zero(self):
        assert "0.00" in _fmt_pf(0.0)


class TestFmtDtHelper:

    def test_none_returns_na(self):
        assert _fmt_dt(None) == "N/A"

    def test_datetime_formats_correctly(self):
        dt = datetime(2024, 6, 15, 10, 30, tzinfo=timezone.utc)
        result = _fmt_dt(dt)
        assert "2024-06-15" in result


class TestFmtRangeHelper:

    def test_both_dates_present(self):
        r = _fmt_range((_TS_START, _TS_END))
        assert "2024-01-02" in r
        assert "2024-12-31" in r

    def test_none_dates_show_na(self):
        r = _fmt_range((None, None))
        assert "N/A" in r


class TestRowHelper:

    def test_returns_string(self):
        result = _row("Label", ["A", "B", "C"], 12)
        assert isinstance(result, str)

    def test_label_included(self):
        result = _row("MyLabel", ["x"], 10)
        assert "MyLabel" in result


class TestGetResultHelper:

    def test_pin_bar_result(self):
        pb  = BacktestResult(symbol="EURUSD", timeframe="D1", strategy_mode="pin_bar_only")
        eng = BacktestResult(symbol="EURUSD", timeframe="D1", strategy_mode="engulfing_only")
        com = BacktestResult(symbol="EURUSD", timeframe="D1", strategy_mode="combined")
        report = ValidationReport(pin_bar_result=pb, engulfing_result=eng, combined_result=com)
        assert _get_result(report, "pin_bar_only") is pb

    def test_engulfing_result(self):
        pb  = BacktestResult(symbol="EURUSD", timeframe="D1", strategy_mode="pin_bar_only")
        eng = BacktestResult(symbol="EURUSD", timeframe="D1", strategy_mode="engulfing_only")
        com = BacktestResult(symbol="EURUSD", timeframe="D1", strategy_mode="combined")
        report = ValidationReport(pin_bar_result=pb, engulfing_result=eng, combined_result=com)
        assert _get_result(report, "engulfing_only") is eng

    def test_combined_result(self):
        pb  = BacktestResult(symbol="EURUSD", timeframe="D1", strategy_mode="pin_bar_only")
        eng = BacktestResult(symbol="EURUSD", timeframe="D1", strategy_mode="engulfing_only")
        com = BacktestResult(symbol="EURUSD", timeframe="D1", strategy_mode="combined")
        report = ValidationReport(pin_bar_result=pb, engulfing_result=eng, combined_result=com)
        assert _get_result(report, "combined") is com


# ---------------------------------------------------------------------------
# 5. DTO compatibility
# ---------------------------------------------------------------------------

class TestDTOCompatibility:

    def test_backtest_result_to_dict_is_serializable(self):
        import json
        result = _make_result()
        d = result.to_dict()
        # All values must be JSON-serializable primitives
        json_str = json.dumps(d)
        assert isinstance(json_str, str)

    def test_strategy_stats_to_dict_is_serializable(self):
        import json
        s = StrategyStats("pin_bar", trades=10, wins=6, losses=4, profit_factor=1.5, expectancy_r=0.3)
        d = s.to_dict()
        json_str = json.dumps(d)
        assert isinstance(json_str, str)

    def test_backtest_result_to_dict_required_keys(self):
        result = _make_result()
        d = result.to_dict()
        for key in ("symbol", "timeframe", "strategy_mode",
                    "trades_executed", "wins", "losses",
                    "win_rate", "profit_factor", "expectancy_r",
                    "max_drawdown_pct", "net_profit_usd", "passes_baseline"):
            assert key in d, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# 6. BacktestRunner delegation tests
# ---------------------------------------------------------------------------

class TestRunnerReportDelegation:

    def test_runner_generate_scorecard_delegates(self):
        runner = BacktestRunner()
        result = _make_result()
        output = runner.generate_scorecard(result)
        assert isinstance(output, str)
        assert "EURUSD" in output

    def test_runner_generate_comparison_report_delegates(self):
        runner = BacktestRunner()
        r1 = _make_result("pin_bar_only")
        r2 = _make_result("engulfing_only")
        output = runner.generate_comparison_report([r1, r2])
        assert isinstance(output, str)
        assert "PIN BAR" in output

    def test_lab_generate_validation_report_delegates(self):
        lab = StrategyValidationLab()
        report = _make_validation_report()
        output = lab.generate_validation_report(report)
        assert isinstance(output, str)
        assert "VALIDATION" in output
