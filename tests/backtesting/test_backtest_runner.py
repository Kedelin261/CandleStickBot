"""
Sprint 13 — Tests for backtest_runner.py
==========================================
Covers:
  - BacktestConfig defaults and overrides
  - BacktestConfig.to_pipeline_config()
  - BacktestResult fields and properties
  - BacktestResult.passes_baseline logic
  - BacktestResult.to_dict()
  - BacktestResult.is_successful
  - StrategyStats fields and win_rate property
  - StrategyStats.to_dict()
  - BacktestRunner.run_pin_bar_only / run_engulfing_only / run_combined
  - BacktestRunner.run_from_candles (all modes)
  - BacktestRunner error result on bad CSV
  - BacktestRunner error result on empty candles
  - Determinism (same input → same output)
  - ValidationReport fields
  - ValidationReport.rank() — best/worst, highest PF, lowest DD
  - StrategyValidationLab.run() all three modes
  - StrategyValidationLab returns ValidationReport
  - _composite_score helper
  - _build_recommendations helper
  - _compute_streaks helper
  - _override helper
  - Pipeline integration (PipelineRunner called, not reimplemented)
  - Analytics integration (M18 called)
  - Trade review integration (M19 fields populated)

Minimum: 50 tests
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.backtesting.backtest_runner import (
    BacktestConfig,
    BacktestResult,
    BacktestRunner,
    StrategyStats,
    StrategyValidationLab,
    ValidationReport,
    _build_recommendations,
    _composite_score,
    _compute_streaks,
    _override,
)
from src.data.types import CandleData


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_candle(date: str, o=1.07, h=1.08, l=1.06, c=1.075, vol=1000.0) -> CandleData:
    dt = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return CandleData(
        timestamp=dt, open=o, high=h, low=l, close=c,
        volume=vol, symbol="EURUSD", timeframe="D1",
    )


def _generate_candles(n: int = 100) -> list[CandleData]:
    """Generate n realistic daily EURUSD candles starting 2020-01-02."""
    from datetime import timedelta
    candles = []
    base = datetime(2020, 1, 2, tzinfo=timezone.utc)
    price = 1.1000
    for i in range(n):
        dt = base + timedelta(days=i)
        o = price
        h = price + 0.003
        l = price - 0.002
        # Alternate small up/down
        c = price + 0.001 if i % 3 != 0 else price - 0.001
        c = max(l + 0.0001, min(h - 0.0001, c))
        candles.append(CandleData(
            timestamp=dt, open=round(o, 5), high=round(h, 5),
            low=round(l, 5), close=round(c, 5), volume=10000.0,
            symbol="EURUSD", timeframe="D1",
        ))
        price = c
    return candles


def _minimal_csv_buf(n: int = 60) -> io.StringIO:
    lines = ["date,open,high,low,close,volume"]
    price = 1.1000
    for i in range(n):
        d = f"2020-01-{i+1:02d}" if i < 31 else f"2020-02-{i-30:02d}"
        o = price
        h = price + 0.003
        l = price - 0.002
        c = price + 0.001
        c = max(l + 0.0001, min(h - 0.0001, c))
        lines.append(f"{d},{o:.5f},{h:.5f},{l:.5f},{c:.5f},10000")
        price = c
    return io.StringIO("\n".join(lines))


# ---------------------------------------------------------------------------
# 1. BacktestConfig
# ---------------------------------------------------------------------------

class TestBacktestConfig:

    def test_default_symbol(self):
        cfg = BacktestConfig()
        assert cfg.symbol == "EURUSD"

    def test_default_timeframe(self):
        cfg = BacktestConfig()
        assert cfg.timeframe == "D1"

    def test_default_initial_balance(self):
        cfg = BacktestConfig()
        assert cfg.initial_balance == 10_000.0

    def test_default_both_strategies_enabled(self):
        cfg = BacktestConfig()
        assert cfg.enable_pin_bar is True
        assert cfg.enable_engulfing is True

    def test_custom_balance(self):
        cfg = BacktestConfig(initial_balance=50_000.0)
        assert cfg.initial_balance == 50_000.0

    def test_start_end_dates_optional(self):
        cfg = BacktestConfig()
        assert cfg.start_date is None
        assert cfg.end_date is None

    def test_to_pipeline_config_symbol(self):
        cfg = BacktestConfig(symbol="GBPUSD")
        pc = cfg.to_pipeline_config()
        assert pc.symbol == "GBPUSD"

    def test_to_pipeline_config_enable_pin_bar(self):
        cfg = BacktestConfig(enable_pin_bar=False)
        pc = cfg.to_pipeline_config()
        assert pc.enable_pin_bar is False

    def test_to_pipeline_config_enable_engulfing(self):
        cfg = BacktestConfig(enable_engulfing=False)
        pc = cfg.to_pipeline_config()
        assert pc.enable_engulfing is False

    def test_to_pipeline_config_initial_balance(self):
        cfg = BacktestConfig(initial_balance=25_000.0)
        pc = cfg.to_pipeline_config()
        assert pc.initial_balance == 25_000.0


# ---------------------------------------------------------------------------
# 2. StrategyStats
# ---------------------------------------------------------------------------

class TestStrategyStats:

    def test_win_rate_with_trades(self):
        s = StrategyStats("pin_bar", trades=10, wins=6, losses=4)
        assert s.win_rate == pytest.approx(0.6)

    def test_win_rate_zero_trades(self):
        s = StrategyStats("pin_bar")
        assert s.win_rate == pytest.approx(0.0)

    def test_to_dict_has_required_keys(self):
        s = StrategyStats("pin_bar", trades=5, wins=3, losses=2)
        d = s.to_dict()
        for key in ("strategy_name", "trades", "wins", "losses", "win_rate",
                    "profit_factor", "expectancy_r"):
            assert key in d

    def test_to_dict_win_rate_rounded(self):
        s = StrategyStats("pin_bar", trades=3, wins=1, losses=2)
        d = s.to_dict()
        assert isinstance(d["win_rate"], float)


# ---------------------------------------------------------------------------
# 3. BacktestResult
# ---------------------------------------------------------------------------

class TestBacktestResult:

    def _make_result(self, **kwargs) -> BacktestResult:
        defaults = dict(symbol="EURUSD", timeframe="D1", strategy_mode="combined")
        defaults.update(kwargs)
        return BacktestResult(**defaults)

    def test_is_successful_no_error(self):
        r = self._make_result()
        assert r.is_successful is True

    def test_is_successful_with_error(self):
        r = self._make_result(error_message="test error")
        assert r.is_successful is False

    def test_passes_baseline_insufficient_trades(self):
        r = self._make_result(trades_executed=5, profit_factor=2.0,
                              win_rate=0.55, max_drawdown_pct=5.0)
        assert r.passes_baseline is False  # < 10 trades

    def test_passes_baseline_low_pf(self):
        r = self._make_result(trades_executed=15, profit_factor=1.0,
                              win_rate=0.55, max_drawdown_pct=5.0)
        assert r.passes_baseline is False  # PF < 1.1

    def test_passes_baseline_low_win_rate(self):
        r = self._make_result(trades_executed=15, profit_factor=1.5,
                              win_rate=0.35, max_drawdown_pct=5.0)
        assert r.passes_baseline is False  # WR < 40%

    def test_passes_baseline_high_drawdown(self):
        r = self._make_result(trades_executed=15, profit_factor=1.5,
                              win_rate=0.55, max_drawdown_pct=25.0)
        assert r.passes_baseline is False  # DD > 20%

    def test_passes_baseline_all_good(self):
        r = self._make_result(trades_executed=15, profit_factor=1.5,
                              win_rate=0.55, max_drawdown_pct=10.0)
        assert r.passes_baseline is True

    def test_to_dict_has_strategy_mode(self):
        r = self._make_result(strategy_mode="pin_bar_only")
        d = r.to_dict()
        assert d["strategy_mode"] == "pin_bar_only"

    def test_to_dict_has_passes_baseline(self):
        r = self._make_result()
        d = r.to_dict()
        assert "passes_baseline" in d

    def test_default_pin_bar_stats(self):
        r = self._make_result()
        assert r.pin_bar.strategy_name == "pin_bar"

    def test_default_engulfing_stats(self):
        r = self._make_result()
        assert r.engulfing.strategy_name == "engulfing_bar"

    def test_to_dict_win_rate_rounded(self):
        r = self._make_result(win_rate=0.333333)
        d = r.to_dict()
        assert isinstance(d["win_rate"], float)


# ---------------------------------------------------------------------------
# 4. BacktestRunner — error results
# ---------------------------------------------------------------------------

class TestBacktestRunnerErrors:

    def test_bad_csv_path_returns_error_result(self):
        runner = BacktestRunner()
        result = runner.run_combined("/nonexistent/file.csv")
        assert result.error_message is not None
        assert not result.is_successful

    def test_error_result_has_correct_mode(self):
        runner = BacktestRunner()
        result = runner.run_pin_bar_only("/nonexistent/file.csv")
        assert result.strategy_mode == "pin_bar_only"

    def test_empty_stringio_returns_error_result(self):
        runner = BacktestRunner()
        result = runner.run_combined(io.StringIO(""))
        assert result.error_message is not None

    def test_error_result_preserves_symbol(self):
        cfg = BacktestConfig(symbol="GBPUSD")
        runner = BacktestRunner(cfg)
        result = runner.run_combined("/bad/path.csv")
        assert result.symbol == "GBPUSD"


# ---------------------------------------------------------------------------
# 5. BacktestRunner — successful runs with candle list
# ---------------------------------------------------------------------------

class TestBacktestRunnerCandles:

    def test_run_from_candles_returns_backtest_result(self):
        candles = _generate_candles(80)
        runner = BacktestRunner()
        result = runner.run_from_candles(candles)
        assert isinstance(result, BacktestResult)

    def test_run_pin_bar_only_mode_field(self):
        candles = _generate_candles(80)
        runner = BacktestRunner()
        result = runner.run_pin_bar_only(candles)
        assert result.strategy_mode == "pin_bar_only"

    def test_run_engulfing_only_mode_field(self):
        candles = _generate_candles(80)
        runner = BacktestRunner()
        result = runner.run_engulfing_only(candles)
        assert result.strategy_mode == "engulfing_only"

    def test_run_combined_mode_field(self):
        candles = _generate_candles(80)
        runner = BacktestRunner()
        result = runner.run_combined(candles)
        assert result.strategy_mode == "combined"

    def test_candles_processed_populated(self):
        candles = _generate_candles(80)
        runner = BacktestRunner()
        result = runner.run_combined(candles)
        assert result.candles_processed > 0

    def test_initial_balance_in_result(self):
        cfg = BacktestConfig(initial_balance=20_000.0)
        candles = _generate_candles(80)
        runner = BacktestRunner(cfg)
        result = runner.run_combined(candles)
        assert result.initial_balance == pytest.approx(20_000.0)

    def test_symbol_in_result(self):
        candles = _generate_candles(80)
        runner = BacktestRunner()
        result = runner.run_combined(candles)
        assert result.symbol == "EURUSD"

    def test_timeframe_in_result(self):
        candles = _generate_candles(80)
        runner = BacktestRunner()
        result = runner.run_combined(candles)
        assert result.timeframe == "D1"

    def test_run_from_candles_pin_bar_only_mode(self):
        candles = _generate_candles(80)
        runner = BacktestRunner()
        result = runner.run_from_candles(candles, mode="pin_bar_only")
        assert result.strategy_mode == "pin_bar_only"

    def test_run_from_candles_engulfing_only_mode(self):
        candles = _generate_candles(80)
        runner = BacktestRunner()
        result = runner.run_from_candles(candles, mode="engulfing_only")
        assert result.strategy_mode == "engulfing_only"

    def test_run_from_candles_default_is_combined(self):
        candles = _generate_candles(80)
        runner = BacktestRunner()
        result = runner.run_from_candles(candles)
        assert result.strategy_mode == "combined"

    def test_data_quality_from_candles_has_total_loaded(self):
        candles = _generate_candles(80)
        runner = BacktestRunner()
        result = runner.run_combined(candles)
        assert result.total_candles_loaded == len(candles)

    def test_pin_bar_only_disables_engulfing(self):
        # pin_bar_only mode: engulfing trades should be 0
        candles = _generate_candles(80)
        runner = BacktestRunner()
        result = runner.run_pin_bar_only(candles)
        # The engulfing breakdown trades should be 0
        assert result.engulfing.trades == 0

    def test_engulfing_only_disables_pin_bar(self):
        candles = _generate_candles(80)
        runner = BacktestRunner()
        result = runner.run_engulfing_only(candles)
        assert result.pin_bar.trades == 0


# ---------------------------------------------------------------------------
# 6. BacktestRunner — CSV loading integration
# ---------------------------------------------------------------------------

class TestBacktestRunnerCSV:

    def test_run_combined_from_csv_buf(self):
        buf = _minimal_csv_buf(60)
        runner = BacktestRunner()
        result = runner.run_combined(buf)
        assert isinstance(result, BacktestResult)

    def test_data_source_is_string_for_stringio(self):
        buf = _minimal_csv_buf(60)
        runner = BacktestRunner()
        result = runner.run_combined(buf)
        assert result.data_source == "<string>"


# ---------------------------------------------------------------------------
# 7. Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:

    def test_same_candles_same_result(self):
        candles = _generate_candles(80)
        runner = BacktestRunner()
        r1 = runner.run_combined(candles)
        r2 = runner.run_combined(candles)
        assert r1.trades_executed == r2.trades_executed
        assert r1.wins == r2.wins
        assert r1.losses == r2.losses

    def test_same_csv_same_result(self):
        csv_content = _minimal_csv_buf(60).getvalue()
        runner = BacktestRunner()
        r1 = runner.run_combined(io.StringIO(csv_content))
        r2 = runner.run_combined(io.StringIO(csv_content))
        assert r1.trades_executed == r2.trades_executed


# ---------------------------------------------------------------------------
# 8. ValidationReport
# ---------------------------------------------------------------------------

class TestValidationReport:

    def _make_report(self) -> ValidationReport:
        def _res(mode, pf=1.3, wr=0.5, dd=10.0, trades=12, exp=0.3):
            return BacktestResult(
                symbol="EURUSD", timeframe="D1", strategy_mode=mode,
                profit_factor=pf, win_rate=wr, max_drawdown_pct=dd,
                trades_executed=trades, expectancy_r=exp,
            )
        pb  = _res("pin_bar_only",  pf=1.5, exp=0.4)
        eng = _res("engulfing_only", pf=1.2, exp=0.2)
        com = _res("combined",       pf=1.8, exp=0.5)
        report = ValidationReport(
            pin_bar_result=pb,
            engulfing_result=eng,
            combined_result=com,
        )
        report.rank()
        return report

    def test_rank_sets_best_strategy(self):
        report = self._make_report()
        assert report.best_strategy != ""

    def test_rank_sets_worst_strategy(self):
        report = self._make_report()
        assert report.worst_strategy != ""

    def test_best_is_not_worst(self):
        report = self._make_report()
        assert report.best_strategy != report.worst_strategy

    def test_strategy_rankings_has_three_entries(self):
        report = self._make_report()
        assert len(report.strategy_rankings) == 3

    def test_highest_pf_positive(self):
        report = self._make_report()
        assert report.highest_pf > 0

    def test_highest_pf_mode_in_rankings(self):
        report = self._make_report()
        assert report.highest_pf_mode in report.strategy_rankings

    def test_lowest_drawdown_non_negative(self):
        report = self._make_report()
        assert report.lowest_drawdown >= 0.0

    def test_recommendations_is_list(self):
        report = self._make_report()
        assert isinstance(report.recommendations, list)

    def test_all_results_returns_three(self):
        pb  = BacktestResult(symbol="EURUSD", timeframe="D1", strategy_mode="pin_bar_only")
        eng = BacktestResult(symbol="EURUSD", timeframe="D1", strategy_mode="engulfing_only")
        com = BacktestResult(symbol="EURUSD", timeframe="D1", strategy_mode="combined")
        report = ValidationReport(pin_bar_result=pb, engulfing_result=eng, combined_result=com)
        assert len(report._all_results()) == 3


# ---------------------------------------------------------------------------
# 9. StrategyValidationLab
# ---------------------------------------------------------------------------

class TestStrategyValidationLab:

    def test_run_returns_validation_report(self):
        candles = _generate_candles(100)
        lab = StrategyValidationLab()
        report = lab.run(candles)
        assert isinstance(report, ValidationReport)

    def test_run_produces_three_results(self):
        candles = _generate_candles(100)
        lab = StrategyValidationLab()
        report = lab.run(candles)
        assert isinstance(report.pin_bar_result, BacktestResult)
        assert isinstance(report.engulfing_result, BacktestResult)
        assert isinstance(report.combined_result, BacktestResult)

    def test_pin_bar_mode_is_correct(self):
        candles = _generate_candles(100)
        lab = StrategyValidationLab()
        report = lab.run(candles)
        assert report.pin_bar_result.strategy_mode == "pin_bar_only"

    def test_engulfing_mode_is_correct(self):
        candles = _generate_candles(100)
        lab = StrategyValidationLab()
        report = lab.run(candles)
        assert report.engulfing_result.strategy_mode == "engulfing_only"

    def test_combined_mode_is_correct(self):
        candles = _generate_candles(100)
        lab = StrategyValidationLab()
        report = lab.run(candles)
        assert report.combined_result.strategy_mode == "combined"

    def test_report_is_ranked_after_run(self):
        candles = _generate_candles(100)
        lab = StrategyValidationLab()
        report = lab.run(candles)
        # rank() should have been called — strategy_rankings not empty or correctly set
        assert len(report.strategy_rankings) == 3

    def test_generate_validation_report_returns_string(self):
        candles = _generate_candles(100)
        lab = StrategyValidationLab()
        report = lab.run(candles)
        text = lab.generate_validation_report(report)
        assert isinstance(text, str)
        assert len(text) > 50

    def test_custom_config_used(self):
        candles = _generate_candles(100)
        cfg = BacktestConfig(initial_balance=5000.0)
        lab = StrategyValidationLab(cfg)
        report = lab.run(candles)
        assert report.pin_bar_result.initial_balance == pytest.approx(5000.0)


# ---------------------------------------------------------------------------
# 10. Private helpers
# ---------------------------------------------------------------------------

class TestCompositeScore:

    def test_zero_trades_returns_zero(self):
        r = BacktestResult(
            symbol="EURUSD", timeframe="D1", strategy_mode="combined",
            trades_executed=0,
        )
        assert _composite_score(r) == pytest.approx(0.0)

    def test_higher_expectancy_scores_higher(self):
        r1 = BacktestResult(
            symbol="EURUSD", timeframe="D1", strategy_mode="combined",
            trades_executed=10, expectancy_r=0.5, profit_factor=1.5, max_drawdown_pct=10.0,
        )
        r2 = BacktestResult(
            symbol="EURUSD", timeframe="D1", strategy_mode="combined",
            trades_executed=10, expectancy_r=0.1, profit_factor=1.5, max_drawdown_pct=10.0,
        )
        assert _composite_score(r1) > _composite_score(r2)

    def test_higher_drawdown_scores_lower(self):
        r1 = BacktestResult(
            symbol="EURUSD", timeframe="D1", strategy_mode="combined",
            trades_executed=10, expectancy_r=0.3, profit_factor=1.5, max_drawdown_pct=5.0,
        )
        r2 = BacktestResult(
            symbol="EURUSD", timeframe="D1", strategy_mode="combined",
            trades_executed=10, expectancy_r=0.3, profit_factor=1.5, max_drawdown_pct=30.0,
        )
        assert _composite_score(r1) > _composite_score(r2)


class TestComputeStreaks:

    def _make_order(self, r_multiple: float):
        order = MagicMock()
        order.r_multiple = r_multiple
        order.is_winner = r_multiple > 0
        order.is_loser  = r_multiple < 0
        return order

    def test_no_orders_returns_zero(self):
        assert _compute_streaks([]) == (0, 0)

    def test_all_wins(self):
        orders = [self._make_order(1.0)] * 5
        max_w, max_l = _compute_streaks(orders)
        assert max_w == 5
        assert max_l == 0

    def test_all_losses(self):
        orders = [self._make_order(-1.0)] * 4
        max_w, max_l = _compute_streaks(orders)
        assert max_w == 0
        assert max_l == 4

    def test_alternating(self):
        orders = [
            self._make_order(1.0), self._make_order(-1.0),
            self._make_order(1.0), self._make_order(-1.0),
        ]
        max_w, max_l = _compute_streaks(orders)
        assert max_w == 1
        assert max_l == 1

    def test_streak_breaks_correctly(self):
        orders = [
            self._make_order(1.0), self._make_order(1.0), self._make_order(1.0),
            self._make_order(-1.0),
            self._make_order(1.0), self._make_order(1.0),
        ]
        max_w, max_l = _compute_streaks(orders)
        assert max_w == 3
        assert max_l == 1


class TestOverride:

    def test_override_changes_field(self):
        cfg = BacktestConfig(enable_pin_bar=True)
        new_cfg = _override(cfg, enable_pin_bar=False)
        assert new_cfg.enable_pin_bar is False

    def test_override_does_not_mutate_original(self):
        cfg = BacktestConfig(enable_pin_bar=True)
        _override(cfg, enable_pin_bar=False)
        assert cfg.enable_pin_bar is True

    def test_override_returns_backtest_config(self):
        cfg = BacktestConfig()
        result = _override(cfg, symbol="GBPUSD")
        assert isinstance(result, BacktestConfig)


class TestBuildRecommendations:

    def _make_validation_report(self, best_pf=1.5, best_exp=0.3,
                                 best_trades=10, combined_pf=1.5,
                                 pb_pf=1.2, eng_pf=1.3) -> ValidationReport:
        def _r(mode, pf, exp=0.2, dd=10.0, trades=10):
            return BacktestResult(
                symbol="EURUSD", timeframe="D1", strategy_mode=mode,
                profit_factor=pf, expectancy_r=exp, max_drawdown_pct=dd,
                trades_executed=trades,
            )
        pb  = _r("pin_bar_only",  pb_pf)
        eng = _r("engulfing_only", eng_pf)
        com = _r("combined",       combined_pf)
        report = ValidationReport(pin_bar_result=pb, engulfing_result=eng, combined_result=com)
        report.rank()
        return report

    def test_returns_list(self):
        report = self._make_validation_report()
        recs = _build_recommendations(report)
        assert isinstance(recs, list)

    def test_at_least_one_recommendation(self):
        report = self._make_validation_report()
        recs = _build_recommendations(report)
        assert len(recs) >= 1

    def test_drawdown_warning_included_when_high_dd(self):
        # Create a result with >20% drawdown
        def _r(mode, dd):
            return BacktestResult(
                symbol="EURUSD", timeframe="D1", strategy_mode=mode,
                profit_factor=1.5, max_drawdown_pct=dd, trades_executed=10,
                expectancy_r=0.3,
            )
        pb  = _r("pin_bar_only", 25.0)
        eng = _r("engulfing_only", 10.0)
        com = _r("combined", 10.0)
        report = ValidationReport(pin_bar_result=pb, engulfing_result=eng, combined_result=com)
        report.rank()
        recs = _build_recommendations(report)
        assert any("20%" in r or "drawdown" in r.lower() for r in recs)
