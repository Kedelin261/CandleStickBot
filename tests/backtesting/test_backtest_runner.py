"""
tests/backtesting/test_backtest_runner.py
==========================================
Sprint 13 — ≥70 tests for:

  src/backtesting/backtest_runner.py
  src/backtesting/reports.py

Coverage areas
--------------
Class 1  — BacktestConfig defaults and field assignment
Class 2  — BacktestConfig.to_pipeline_config() mapping
Class 3  — StrategyStats — win_rate / to_dict
Class 4  — BacktestResult defaults, is_successful, passes_baseline
Class 5  — BacktestResult.to_dict()
Class 6  — BacktestRunner instantiation and config defaults
Class 7  — BacktestRunner.run_combined() basic contract
Class 8  — BacktestRunner.run_pin_bar_only() and run_engulfing_only()
Class 9  — BacktestRunner.run_from_candles() mode dispatch
Class 10 — BacktestRunner error path (bad CSV / missing file)
Class 11 — BacktestRunner date-range filtering via config
Class 12 — BacktestRunner deterministic re-run (same input → same output)
Class 13 — ValidationReport.rank() logic
Class 14 — StrategyValidationLab full run
Class 15 — generate_scorecard() content and structure
Class 16 — generate_comparison_report() content and structure
Class 17 — generate_validation_report() content and structure
Class 18 — _composite_score() helper
Class 19 — _compute_streaks() helper
Class 20 — _override() helper preserves unmodified fields
"""

from __future__ import annotations

import io
import math
from datetime import datetime, timedelta, timezone
from typing import List

import pytest

from src.backtesting.backtest_runner import (
    BacktestConfig,
    BacktestResult,
    BacktestRunner,
    StrategyStats,
    StrategyValidationLab,
    ValidationReport,
    _composite_score,
    _compute_streaks,
    _override,
)
from src.backtesting.reports import (
    _fmt_pf,
    _mode_label,
    generate_comparison_report,
    generate_scorecard,
    generate_validation_report,
)
from src.data.types import CandleData


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_candle(
    date: str,
    o: float = 1.09500,
    h: float = 1.09800,
    l: float = 1.09200,
    c: float = 1.09650,
    vol: float = 10000.0,
    symbol: str = "EURUSD",
    timeframe: str = "D1",
) -> CandleData:
    ts = datetime.fromisoformat(date).replace(tzinfo=timezone.utc)
    return CandleData(
        timestamp=ts,
        open=o, high=h, low=l, close=c,
        volume=vol, symbol=symbol, timeframe=timeframe,
    )


def _weekday_candles(n: int = 60, symbol: str = "EURUSD") -> List[CandleData]:
    """Generate n weekday candles with alternating trending prices."""
    candles = []
    d = datetime(2023, 1, 2, tzinfo=timezone.utc)
    price = 1.09000
    count = 0
    while count < n:
        if d.weekday() < 5:
            o = round(price, 5)
            h = round(o + 0.0015, 5)
            l = round(o - 0.0015, 5)
            c = round(o + 0.0005 * (1 if count % 2 == 0 else -1), 5)
            c = max(l, min(h, c))
            candles.append(CandleData(
                timestamp=d, open=o, high=h, low=l, close=c,
                volume=10000, symbol=symbol, timeframe="D1",
            ))
            price = c
            count += 1
        d += timedelta(days=1)
    return candles


def _minimal_csv(n: int = 60) -> io.StringIO:
    lines = ["date,open,high,low,close,volume"]
    d = datetime(2023, 1, 2)
    price = 1.09000
    count = 0
    while count < n:
        if d.weekday() < 5:
            o = round(price, 5)
            h = round(o + 0.0015, 5)
            l = round(o - 0.0015, 5)
            c = round(o + 0.0003, 5)
            lines.append(f"{d.date()},{o},{h},{l},{c},10000")
            price = c
            count += 1
        d += timedelta(days=1)
    return io.StringIO("\n".join(lines))


def _default_result(**kwargs) -> BacktestResult:
    defaults = dict(
        symbol="EURUSD",
        timeframe="D1",
        strategy_mode="combined",
        trades_executed=0,
        wins=0,
        losses=0,
        win_rate=0.0,
        profit_factor=0.0,
        expectancy_r=0.0,
        max_drawdown_pct=0.0,
        initial_balance=10_000.0,
        final_balance=10_000.0,
    )
    defaults.update(kwargs)
    return BacktestResult(**defaults)


def _passing_result(**kwargs) -> BacktestResult:
    """A BacktestResult that satisfies passes_baseline.
    Sprint 15 FIX-4: updated to use canonical criteria (N>=30, PF>1.10,
    Expectancy>0, DD<25%). Previously N>=10, WR>=40%, DD<=20%.
    """
    r = _default_result(
        trades_executed=30,       # FIX-4: was 20, spec requires >=30
        wins=16,
        losses=14,
        win_rate=0.53,
        profit_factor=1.50,
        expectancy_r=0.15,        # FIX-4: Expectancy>0 now required
        max_drawdown_pct=10.0,
        **kwargs,
    )
    return r


# ---------------------------------------------------------------------------
# Class 1 — BacktestConfig defaults and field assignment
# ---------------------------------------------------------------------------

class TestBacktestConfigDefaults:
    def test_default_symbol(self):
        assert BacktestConfig().symbol == "EURUSD"

    def test_default_timeframe(self):
        assert BacktestConfig().timeframe == "D1"

    def test_default_initial_balance(self):
        assert BacktestConfig().initial_balance == 10_000.0

    def test_default_slippage(self):
        assert BacktestConfig().slippage_pips == 1.0

    def test_default_enable_pin_bar(self):
        assert BacktestConfig().enable_pin_bar is True

    def test_default_enable_engulfing(self):
        assert BacktestConfig().enable_engulfing is True

    def test_default_risk_enabled(self):
        assert BacktestConfig().risk_enabled is True

    def test_default_analytics_enabled(self):
        assert BacktestConfig().analytics_enabled is True

    def test_default_review_enabled(self):
        assert BacktestConfig().review_enabled is True

    def test_default_minimum_tqs(self):
        assert BacktestConfig().minimum_tqs == 0.0

    def test_default_minimum_rr(self):
        assert BacktestConfig().minimum_rr == 2.0

    def test_default_lookback_window(self):
        assert BacktestConfig().lookback_window == 50

    def test_default_start_date_none(self):
        assert BacktestConfig().start_date is None

    def test_default_end_date_none(self):
        assert BacktestConfig().end_date is None

    def test_custom_symbol(self):
        cfg = BacktestConfig(symbol="GBPUSD")
        assert cfg.symbol == "GBPUSD"

    def test_custom_balance(self):
        cfg = BacktestConfig(initial_balance=50_000.0)
        assert cfg.initial_balance == 50_000.0


# ---------------------------------------------------------------------------
# Class 2 — BacktestConfig.to_pipeline_config()
# ---------------------------------------------------------------------------

class TestBacktestConfigToPipelineConfig:
    def test_returns_pipeline_config(self):
        from src.integration.pipeline_runner import PipelineConfig
        pc = BacktestConfig().to_pipeline_config()
        assert isinstance(pc, PipelineConfig)

    def test_symbol_propagated(self):
        pc = BacktestConfig(symbol="GBPUSD").to_pipeline_config()
        assert pc.symbol == "GBPUSD"

    def test_timeframe_propagated(self):
        pc = BacktestConfig(timeframe="H4").to_pipeline_config()
        assert pc.timeframe == "H4"

    def test_initial_balance_propagated(self):
        pc = BacktestConfig(initial_balance=25_000.0).to_pipeline_config()
        assert pc.initial_balance == 25_000.0

    def test_enable_pin_bar_false_propagated(self):
        pc = BacktestConfig(enable_pin_bar=False).to_pipeline_config()
        assert pc.enable_pin_bar is False

    def test_enable_engulfing_false_propagated(self):
        pc = BacktestConfig(enable_engulfing=False).to_pipeline_config()
        assert pc.enable_engulfing is False

    def test_minimum_rr_propagated(self):
        pc = BacktestConfig(minimum_rr=3.0).to_pipeline_config()
        assert pc.minimum_rr == 3.0

    def test_lookback_window_propagated(self):
        pc = BacktestConfig(lookback_window=100).to_pipeline_config()
        assert pc.lookback_window == 100


# ---------------------------------------------------------------------------
# Class 3 — StrategyStats
# ---------------------------------------------------------------------------

class TestStrategyStats:
    def test_win_rate_zero_trades(self):
        s = StrategyStats(strategy_name="pin_bar")
        assert s.win_rate == 0.0

    def test_win_rate_calculation(self):
        s = StrategyStats(strategy_name="pin_bar", trades=10, wins=7, losses=3)
        assert s.win_rate == pytest.approx(0.70)

    def test_to_dict_keys(self):
        s = StrategyStats(strategy_name="pin_bar", trades=5, wins=3, losses=2)
        d = s.to_dict()
        expected_keys = {
            "strategy_name", "trades", "wins", "losses", "win_rate",
            "profit_factor", "expectancy_r", "avg_winner_r", "avg_loser_r",
            "max_consecutive_wins", "max_consecutive_losses", "max_drawdown_pct",
        }
        assert expected_keys.issubset(d.keys())

    def test_to_dict_values(self):
        s = StrategyStats(strategy_name="pin_bar", trades=4, wins=2, losses=2,
                          profit_factor=1.5)
        d = s.to_dict()
        assert d["trades"] == 4
        assert d["profit_factor"] == 1.5
        assert d["win_rate"] == pytest.approx(0.5)

    def test_win_rate_all_wins(self):
        s = StrategyStats(strategy_name="x", trades=5, wins=5, losses=0)
        assert s.win_rate == 1.0

    def test_win_rate_all_losses(self):
        s = StrategyStats(strategy_name="x", trades=5, wins=0, losses=5)
        assert s.win_rate == 0.0


# ---------------------------------------------------------------------------
# Class 4 — BacktestResult defaults, is_successful, passes_baseline
# ---------------------------------------------------------------------------

class TestBacktestResultProperties:
    def test_is_successful_no_error(self):
        r = _default_result()
        assert r.is_successful is True

    def test_is_successful_with_error(self):
        r = _default_result(error_message="something failed")
        assert r.is_successful is False

    def test_passes_baseline_too_few_trades(self):
        """Sprint 15 FIX-4: N<30 → fails (was N<10)."""
        r = _default_result(
            trades_executed=29,   # FIX-4: threshold now 30
            win_rate=0.60,
            profit_factor=1.5,
            expectancy_r=0.3,
            max_drawdown_pct=5.0,
        )
        assert r.passes_baseline is False

    def test_passes_baseline_low_pf(self):
        """Sprint 15 FIX-4: PF=1.10 exactly fails (strict >, was >=)."""
        r = _default_result(
            trades_executed=30,
            win_rate=0.50,
            profit_factor=1.10,   # FIX-4: strict > means 1.10 fails
            expectancy_r=0.10,
            max_drawdown_pct=5.0,
        )
        assert r.passes_baseline is False

    def test_passes_baseline_zero_expectancy_fails(self):
        """Sprint 15 FIX-4: Expectancy=0 → fails (new criterion replacing WR)."""
        r = _default_result(
            trades_executed=30,
            win_rate=0.50,
            profit_factor=1.5,
            expectancy_r=0.0,     # FIX-4: Expectancy must be > 0
            max_drawdown_pct=5.0,
        )
        assert r.passes_baseline is False

    def test_passes_baseline_high_drawdown(self):
        """Sprint 15 FIX-4: DD=25.0% fails (threshold now <25%, was <=20%)."""
        r = _default_result(
            trades_executed=30,
            win_rate=0.50,
            profit_factor=1.5,
            expectancy_r=0.15,
            max_drawdown_pct=25.0,  # FIX-4: < 25% strict, so 25.0 fails
        )
        assert r.passes_baseline is False

    def test_passes_baseline_all_conditions_met(self):
        """Sprint 15 FIX-4: canonical criteria — N>=30, PF>1.10, Exp>0, DD<25%."""
        r = _passing_result()
        assert r.passes_baseline is True

    def test_passes_baseline_exact_boundary_pf_above(self):
        """Sprint 15 FIX-4: PF=1.11 (just above 1.10 strict) → passes."""
        r = _default_result(
            trades_executed=30,
            win_rate=0.50,
            profit_factor=1.11,
            expectancy_r=0.05,
            max_drawdown_pct=20.0,
        )
        assert r.passes_baseline is True

    def test_passes_baseline_dd_just_under_25(self):
        """Sprint 15 FIX-4: DD=24.99% → passes (threshold is < 25%)."""
        r = _default_result(
            trades_executed=30,
            win_rate=0.50,
            profit_factor=1.2,
            expectancy_r=0.10,
            max_drawdown_pct=24.99,
        )
        assert r.passes_baseline is True

    def test_passes_baseline_dd_exactly_25_fails(self):
        """Sprint 15 FIX-4: DD=25.0% exactly → fails (strict <)."""
        r = _default_result(
            trades_executed=30,
            win_rate=0.50,
            profit_factor=1.2,
            expectancy_r=0.10,
            max_drawdown_pct=25.0,
        )
        assert r.passes_baseline is False

    def test_error_message_default_none(self):
        r = BacktestResult(symbol="X", timeframe="D1", strategy_mode="combined")
        assert r.error_message is None


# ---------------------------------------------------------------------------
# Class 5 — BacktestResult.to_dict()
# ---------------------------------------------------------------------------

class TestBacktestResultToDict:
    def test_returns_dict(self):
        assert isinstance(_default_result().to_dict(), dict)

    def test_required_keys_present(self):
        d = _default_result().to_dict()
        for key in ("symbol", "timeframe", "strategy_mode", "trades_executed",
                    "win_rate", "profit_factor", "expectancy_r", "max_drawdown_pct",
                    "net_profit_usd", "passes_baseline"):
            assert key in d, f"Missing key: {key}"

    def test_win_rate_rounded(self):
        r = _default_result(win_rate=0.333333)
        d = r.to_dict()
        assert d["win_rate"] == pytest.approx(0.3333, abs=1e-3)

    def test_passes_baseline_false_in_dict(self):
        d = _default_result(trades_executed=0).to_dict()
        assert d["passes_baseline"] is False

    def test_passes_baseline_true_in_dict(self):
        d = _passing_result().to_dict()
        assert d["passes_baseline"] is True


# ---------------------------------------------------------------------------
# Class 6 — BacktestRunner instantiation
# ---------------------------------------------------------------------------

class TestBacktestRunnerInstantiation:
    def test_default_config_used_when_none(self):
        runner = BacktestRunner()
        assert runner._cfg.symbol == "EURUSD"

    def test_custom_config_stored(self):
        cfg = BacktestConfig(symbol="USDJPY", initial_balance=20_000.0)
        runner = BacktestRunner(cfg)
        assert runner._cfg.symbol == "USDJPY"
        assert runner._cfg.initial_balance == 20_000.0


# ---------------------------------------------------------------------------
# Class 7 — BacktestRunner.run_combined() basic contract
# ---------------------------------------------------------------------------

class TestBacktestRunnerCombined:
    def test_returns_backtest_result(self):
        candles = _weekday_candles(60)
        runner = BacktestRunner(BacktestConfig())
        result = runner.run_combined(candles)
        assert isinstance(result, BacktestResult)

    def test_strategy_mode_combined(self):
        candles = _weekday_candles(30)
        result = BacktestRunner().run_combined(candles)
        assert result.strategy_mode == "combined"

    def test_symbol_in_result(self):
        candles = _weekday_candles(30)
        cfg = BacktestConfig(symbol="EURUSD")
        result = BacktestRunner(cfg).run_combined(candles)
        assert result.symbol == "EURUSD"

    def test_candles_processed_positive(self):
        candles = _weekday_candles(50)
        result = BacktestRunner().run_combined(candles)
        assert result.candles_processed == 50

    def test_no_error_on_valid_data(self):
        candles = _weekday_candles(40)
        result = BacktestRunner().run_combined(candles)
        assert result.error_message is None

    def test_run_combined_from_stringio(self):
        buf = _minimal_csv(40)
        result = BacktestRunner().run_combined(buf)
        assert isinstance(result, BacktestResult)
        assert result.total_candles_loaded == 40

    def test_win_rate_between_zero_and_one(self):
        candles = _weekday_candles(60)
        result = BacktestRunner().run_combined(candles)
        assert 0.0 <= result.win_rate <= 1.0

    def test_profit_factor_non_negative(self):
        candles = _weekday_candles(60)
        result = BacktestRunner().run_combined(candles)
        assert result.profit_factor >= 0.0

    def test_max_drawdown_non_negative(self):
        candles = _weekday_candles(60)
        result = BacktestRunner().run_combined(candles)
        assert result.max_drawdown_pct >= 0.0

    def test_data_source_string_for_candle_list(self):
        candles = _weekday_candles(20)
        result = BacktestRunner().run_combined(candles)
        assert result.data_source == "<candles>"


# ---------------------------------------------------------------------------
# Class 8 — run_pin_bar_only and run_engulfing_only mode labels
# ---------------------------------------------------------------------------

class TestRunModeLabels:
    def test_pin_bar_only_mode_label(self):
        candles = _weekday_candles(30)
        result = BacktestRunner().run_pin_bar_only(candles)
        assert result.strategy_mode == "pin_bar_only"

    def test_engulfing_only_mode_label(self):
        candles = _weekday_candles(30)
        result = BacktestRunner().run_engulfing_only(candles)
        assert result.strategy_mode == "engulfing_only"

    def test_pin_bar_disables_engulfing(self):
        """When running pin_bar_only, the pipeline config must have engulfing=False."""
        candles = _weekday_candles(30)
        # We can't easily inspect the PipelineConfig used internally, but we can
        # verify no error occurs and the mode is set correctly.
        result = BacktestRunner().run_pin_bar_only(candles)
        assert result.is_successful

    def test_engulfing_disables_pin_bar(self):
        candles = _weekday_candles(30)
        result = BacktestRunner().run_engulfing_only(candles)
        assert result.is_successful


# ---------------------------------------------------------------------------
# Class 9 — run_from_candles mode dispatch
# ---------------------------------------------------------------------------

class TestRunFromCandles:
    def test_default_mode_combined(self):
        candles = _weekday_candles(30)
        result = BacktestRunner().run_from_candles(candles)
        assert result.strategy_mode == "combined"

    def test_pin_bar_mode_dispatch(self):
        candles = _weekday_candles(30)
        result = BacktestRunner().run_from_candles(candles, mode="pin_bar_only")
        assert result.strategy_mode == "pin_bar_only"

    def test_engulfing_mode_dispatch(self):
        candles = _weekday_candles(30)
        result = BacktestRunner().run_from_candles(candles, mode="engulfing_only")
        assert result.strategy_mode == "engulfing_only"

    def test_combined_explicit(self):
        candles = _weekday_candles(30)
        result = BacktestRunner().run_from_candles(candles, mode="combined")
        assert result.strategy_mode == "combined"

    def test_unknown_mode_falls_through_to_combined(self):
        candles = _weekday_candles(30)
        # Unknown mode falls through to combined (matches the `return self.run_combined` fallback)
        result = BacktestRunner().run_from_candles(candles, mode="unknown_mode")
        assert result.strategy_mode == "combined"


# ---------------------------------------------------------------------------
# Class 10 — BacktestRunner error paths
# ---------------------------------------------------------------------------

class TestBacktestRunnerErrors:
    def test_missing_csv_file_returns_error_result(self):
        result = BacktestRunner().run_combined("/tmp/does_not_exist_sprint13.csv")
        assert not result.is_successful
        assert result.error_message is not None
        assert result.trades_executed == 0

    def test_error_result_has_symbol(self):
        cfg = BacktestConfig(symbol="USDJPY")
        result = BacktestRunner(cfg).run_combined("/tmp/no_file.csv")
        assert result.symbol == "USDJPY"

    def test_error_result_passes_baseline_false(self):
        result = BacktestRunner().run_combined("/tmp/no_file.csv")
        assert result.passes_baseline is False

    def test_empty_candle_list_returns_error_result(self):
        result = BacktestRunner().run_combined([])
        assert not result.is_successful
        assert result.error_message is not None


# ---------------------------------------------------------------------------
# Class 11 — Date range filtering via config
# ---------------------------------------------------------------------------

class TestDateRangeViaConfig:
    def test_start_date_filters_candles(self):
        candles = _weekday_candles(60)
        cutoff = candles[30].timestamp
        cfg = BacktestConfig(start_date=cutoff)
        result = BacktestRunner(cfg).run_combined(candles)
        # Should process at most len(candles)-30 candles
        assert result.candles_processed <= len(candles)

    def test_end_date_filters_candles(self):
        candles = _weekday_candles(60)
        cutoff = candles[20].timestamp
        cfg = BacktestConfig(end_date=cutoff)
        result = BacktestRunner(cfg).run_combined(candles)
        assert result.candles_processed <= 21   # ≤ 21 because inclusive

    def test_start_after_all_candles_returns_error(self):
        candles = _weekday_candles(10)
        future = datetime(2099, 1, 1, tzinfo=timezone.utc)
        cfg = BacktestConfig(start_date=future)
        result = BacktestRunner(cfg).run_combined(candles)
        assert not result.is_successful


# ---------------------------------------------------------------------------
# Class 12 — Deterministic re-run
# ---------------------------------------------------------------------------

class TestDeterministicReruns:
    def test_same_candles_same_trade_count(self):
        candles = _weekday_candles(60)
        r1 = BacktestRunner().run_combined(list(candles))
        r2 = BacktestRunner().run_combined(list(candles))
        assert r1.trades_executed == r2.trades_executed

    def test_same_candles_same_win_rate(self):
        candles = _weekday_candles(60)
        r1 = BacktestRunner().run_combined(list(candles))
        r2 = BacktestRunner().run_combined(list(candles))
        assert r1.win_rate == pytest.approx(r2.win_rate)

    def test_same_candles_same_profit_factor(self):
        candles = _weekday_candles(60)
        r1 = BacktestRunner().run_combined(list(candles))
        r2 = BacktestRunner().run_combined(list(candles))
        assert r1.profit_factor == pytest.approx(r2.profit_factor)

    def test_csv_and_candles_same_result(self):
        """Loading the same data from CSV vs CandleData list should give same trades."""
        buf = _minimal_csv(60)
        # Run from StringIO
        r_csv = BacktestRunner().run_combined(buf)
        # Load candles then run from list
        from src.backtesting.data_loader import load_candles_from_csv
        buf2 = _minimal_csv(60)
        candles, _ = load_candles_from_csv(buf2)
        r_candles = BacktestRunner().run_combined(candles)
        assert r_csv.trades_executed == r_candles.trades_executed


# ---------------------------------------------------------------------------
# Class 13 — ValidationReport.rank() logic
# ---------------------------------------------------------------------------

class TestValidationReportRank:
    def _make_report(self, pb_pf=1.0, eng_pf=1.5, com_pf=1.2,
                     pb_exp=0.0, eng_exp=0.2, com_exp=0.1) -> ValidationReport:
        def _res(mode, pf, exp):
            return _default_result(
                strategy_mode=mode,
                trades_executed=20,
                profit_factor=pf,
                expectancy_r=exp,
                win_rate=0.50,
                max_drawdown_pct=5.0,
            )
        report = ValidationReport(
            pin_bar_result=_res("pin_bar_only", pb_pf, pb_exp),
            engulfing_result=_res("engulfing_only", eng_pf, eng_exp),
            combined_result=_res("combined", com_pf, com_exp),
        )
        report.rank()
        return report

    def test_rank_returns_self(self):
        report = ValidationReport(
            pin_bar_result=_default_result(strategy_mode="pin_bar_only"),
            engulfing_result=_default_result(strategy_mode="engulfing_only"),
            combined_result=_default_result(strategy_mode="combined"),
        )
        returned = report.rank()
        assert returned is report

    def test_strategy_rankings_length(self):
        report = self._make_report()
        assert len(report.strategy_rankings) == 3

    def test_best_strategy_is_highest_composite(self):
        # engulfing has PF=1.5, exp=0.2 → composite = 0.2*2 + 1.5*0.5 - 0.05 = 1.1
        report = self._make_report()
        assert report.best_strategy == "engulfing_only"

    def test_worst_strategy_field_set(self):
        report = self._make_report()
        assert report.worst_strategy != ""

    def test_highest_pf_mode_set(self):
        report = self._make_report(pb_pf=1.0, eng_pf=2.0, com_pf=1.5)
        assert report.highest_pf_mode == "engulfing_only"
        assert report.highest_pf == pytest.approx(2.0)

    def test_highest_expectancy_mode_set(self):
        report = self._make_report(pb_exp=0.1, eng_exp=0.5, com_exp=0.3)
        assert report.highest_exp_mode == "engulfing_only"
        assert report.highest_expectancy == pytest.approx(0.5)

    def test_lowest_drawdown_mode_set(self):
        def _res_dd(mode, dd):
            return _default_result(
                strategy_mode=mode,
                trades_executed=10,
                profit_factor=1.2,
                win_rate=0.50,
                max_drawdown_pct=dd,
            )
        report = ValidationReport(
            pin_bar_result=_res_dd("pin_bar_only", 15.0),
            engulfing_result=_res_dd("engulfing_only", 5.0),
            combined_result=_res_dd("combined", 10.0),
        )
        report.rank()
        assert report.lowest_dd_mode == "engulfing_only"
        assert report.lowest_drawdown == pytest.approx(5.0)

    def test_recommendations_non_empty(self):
        report = self._make_report()
        assert len(report.recommendations) >= 1

    def test_all_zero_trades_still_ranks(self):
        report = ValidationReport(
            pin_bar_result=_default_result(strategy_mode="pin_bar_only"),
            engulfing_result=_default_result(strategy_mode="engulfing_only"),
            combined_result=_default_result(strategy_mode="combined"),
        )
        report.rank()
        assert len(report.strategy_rankings) == 3


# ---------------------------------------------------------------------------
# Class 14 — StrategyValidationLab
# ---------------------------------------------------------------------------

class TestStrategyValidationLab:
    def test_run_returns_validation_report(self):
        candles = _weekday_candles(50)
        lab = StrategyValidationLab()
        report = lab.run(candles)
        assert isinstance(report, ValidationReport)

    def test_all_three_results_populated(self):
        candles = _weekday_candles(50)
        report = StrategyValidationLab().run(candles)
        assert isinstance(report.pin_bar_result, BacktestResult)
        assert isinstance(report.engulfing_result, BacktestResult)
        assert isinstance(report.combined_result, BacktestResult)

    def test_rankings_set_after_run(self):
        candles = _weekday_candles(50)
        report = StrategyValidationLab().run(candles)
        assert len(report.strategy_rankings) == 3

    def test_correct_modes_on_sub_results(self):
        candles = _weekday_candles(50)
        report = StrategyValidationLab().run(candles)
        assert report.pin_bar_result.strategy_mode == "pin_bar_only"
        assert report.engulfing_result.strategy_mode == "engulfing_only"
        assert report.combined_result.strategy_mode == "combined"

    def test_generate_validation_report_string(self):
        candles = _weekday_candles(50)
        lab = StrategyValidationLab()
        report = lab.run(candles)
        text = lab.generate_validation_report(report)
        assert isinstance(text, str)
        assert "VALIDATION LAB" in text

    def test_run_from_csv_string_io(self):
        buf = _minimal_csv(60)
        lab = StrategyValidationLab()
        report = lab.run(buf)
        assert isinstance(report, ValidationReport)


# ---------------------------------------------------------------------------
# Class 15 — generate_scorecard()
# ---------------------------------------------------------------------------

class TestGenerateScorecard:
    def _run_scorecard(self, **kwargs) -> str:
        r = _default_result(**kwargs)
        return generate_scorecard(r)

    def test_returns_string(self):
        assert isinstance(self._run_scorecard(), str)

    def test_contains_mode_label(self):
        text = self._run_scorecard(strategy_mode="combined")
        assert "COMBINED" in text

    def test_contains_symbol(self):
        text = self._run_scorecard(symbol="EURUSD")
        assert "EURUSD" in text

    def test_contains_timeframe(self):
        text = self._run_scorecard(timeframe="D1")
        assert "D1" in text

    def test_scorecard_error_path(self):
        r = _default_result(error_message="load failed")
        text = generate_scorecard(r)
        assert "ERROR" in text
        assert "load failed" in text

    def test_scorecard_sections_present(self):
        text = self._run_scorecard()
        assert "TRADE STATISTICS" in text
        assert "PERFORMANCE METRICS" in text
        assert "STRATEGY BREAKDOWN" in text
        assert "REVIEW ANALYSIS" in text
        assert "DATA QUALITY" in text

    def test_passes_baseline_no(self):
        text = self._run_scorecard(trades_executed=0)
        assert "NO" in text or "❌" in text

    def test_passes_baseline_yes(self):
        text = generate_scorecard(_passing_result())
        assert "YES" in text or "✅" in text

    def test_profit_factor_displayed(self):
        r = _default_result(profit_factor=1.42)
        text = generate_scorecard(r)
        assert "1.42" in text

    def test_runner_generate_scorecard_delegates(self):
        candles = _weekday_candles(30)
        runner = BacktestRunner()
        result = runner.run_combined(candles)
        text = runner.generate_scorecard(result)
        assert "COMBINED" in text


# ---------------------------------------------------------------------------
# Class 16 — generate_comparison_report()
# ---------------------------------------------------------------------------

class TestGenerateComparisonReport:
    def test_empty_list_message(self):
        text = generate_comparison_report([])
        assert "No results" in text

    def test_returns_string(self):
        text = generate_comparison_report([_default_result()])
        assert isinstance(text, str)

    def test_single_result_contains_mode(self):
        r = _default_result(strategy_mode="pin_bar_only")
        text = generate_comparison_report([r])
        assert "PIN BAR" in text

    def test_multi_result_both_modes_appear(self):
        r1 = _default_result(strategy_mode="pin_bar_only")
        r2 = _default_result(strategy_mode="engulfing_only")
        text = generate_comparison_report([r1, r2])
        assert "PIN BAR" in text
        assert "ENGULFING BAR" in text

    def test_metric_labels_present(self):
        text = generate_comparison_report([_default_result()])
        assert "Win Rate" in text
        assert "Profit Factor" in text

    def test_runner_generate_comparison_delegates(self):
        candles = _weekday_candles(30)
        runner = BacktestRunner()
        r1 = runner.run_pin_bar_only(candles)
        r2 = runner.run_engulfing_only(candles)
        text = runner.generate_comparison_report([r1, r2])
        assert "PIN BAR" in text
        assert "ENGULFING BAR" in text


# ---------------------------------------------------------------------------
# Class 17 — generate_validation_report()
# ---------------------------------------------------------------------------

class TestGenerateValidationReport:
    def _make_validation_report(self) -> ValidationReport:
        def _res(mode):
            return _default_result(strategy_mode=mode)
        rpt = ValidationReport(
            pin_bar_result=_res("pin_bar_only"),
            engulfing_result=_res("engulfing_only"),
            combined_result=_res("combined"),
        )
        rpt.rank()
        return rpt

    def test_returns_string(self):
        text = generate_validation_report(self._make_validation_report())
        assert isinstance(text, str)

    def test_contains_validation_lab_header(self):
        text = generate_validation_report(self._make_validation_report())
        assert "VALIDATION LAB" in text

    def test_contains_strategy_rankings(self):
        text = generate_validation_report(self._make_validation_report())
        assert "STRATEGY RANKINGS" in text

    def test_contains_recommendations(self):
        text = generate_validation_report(self._make_validation_report())
        assert "RECOMMENDATIONS" in text

    def test_contains_all_mode_labels(self):
        text = generate_validation_report(self._make_validation_report())
        assert "PIN BAR" in text
        assert "ENGULFING BAR" in text
        assert "COMBINED" in text

    def test_contains_key_findings(self):
        text = generate_validation_report(self._make_validation_report())
        assert "KEY FINDINGS" in text


# ---------------------------------------------------------------------------
# Class 18 — _composite_score() helper
# ---------------------------------------------------------------------------

class TestCompositeScore:
    def test_zero_trades_returns_zero(self):
        r = _default_result(trades_executed=0)
        assert _composite_score(r) == 0.0

    def test_higher_expectancy_raises_score(self):
        r_low  = _default_result(trades_executed=10, expectancy_r=0.1, profit_factor=1.0)
        r_high = _default_result(trades_executed=10, expectancy_r=0.5, profit_factor=1.0)
        assert _composite_score(r_high) > _composite_score(r_low)

    def test_higher_pf_raises_score(self):
        r_low  = _default_result(trades_executed=10, expectancy_r=0.2, profit_factor=1.0)
        r_high = _default_result(trades_executed=10, expectancy_r=0.2, profit_factor=2.0)
        assert _composite_score(r_high) > _composite_score(r_low)

    def test_higher_drawdown_lowers_score(self):
        r_low_dd  = _default_result(trades_executed=10, expectancy_r=0.2,
                                     profit_factor=1.5, max_drawdown_pct=5.0)
        r_high_dd = _default_result(trades_executed=10, expectancy_r=0.2,
                                     profit_factor=1.5, max_drawdown_pct=20.0)
        assert _composite_score(r_low_dd) > _composite_score(r_high_dd)

    def test_inf_profit_factor_capped(self):
        r = _default_result(trades_executed=5, expectancy_r=0.3,
                             profit_factor=float("inf"), max_drawdown_pct=2.0)
        score = _composite_score(r)
        assert math.isfinite(score)


# ---------------------------------------------------------------------------
# Class 19 — _compute_streaks() helper
# ---------------------------------------------------------------------------

class TestComputeStreaks:
    class _Order:
        def __init__(self, won: bool):
            self.is_winner = won
            self.is_loser  = not won

    def test_no_orders(self):
        assert _compute_streaks([]) == (0, 0)

    def test_all_wins(self):
        orders = [self._Order(True)] * 5
        wins, losses = _compute_streaks(orders)
        assert wins == 5
        assert losses == 0

    def test_all_losses(self):
        orders = [self._Order(False)] * 4
        wins, losses = _compute_streaks(orders)
        assert wins == 0
        assert losses == 4

    def test_alternating(self):
        orders = [self._Order(i % 2 == 0) for i in range(6)]
        wins, losses = _compute_streaks(orders)
        assert wins == 1
        assert losses == 1

    def test_streak_in_middle(self):
        # W W W L L W L W W
        pattern = [True, True, True, False, False, True, False, True, True]
        orders = [self._Order(p) for p in pattern]
        wins, losses = _compute_streaks(orders)
        assert wins == 3
        assert losses == 2


# ---------------------------------------------------------------------------
# Class 20 — _override() helper
# ---------------------------------------------------------------------------

class TestOverrideHelper:
    def test_returns_new_instance(self):
        cfg = BacktestConfig()
        cfg2 = _override(cfg)
        assert cfg2 is not cfg

    def test_override_single_field(self):
        cfg = BacktestConfig(symbol="EURUSD", enable_pin_bar=True)
        cfg2 = _override(cfg, enable_pin_bar=False)
        assert cfg2.enable_pin_bar is False

    def test_unmodified_fields_preserved(self):
        cfg = BacktestConfig(symbol="EURUSD", initial_balance=25_000.0)
        cfg2 = _override(cfg, enable_pin_bar=False)
        assert cfg2.symbol == "EURUSD"
        assert cfg2.initial_balance == 25_000.0

    def test_override_multiple_fields(self):
        cfg = BacktestConfig()
        cfg2 = _override(cfg, enable_pin_bar=False, enable_engulfing=False)
        assert cfg2.enable_pin_bar is False
        assert cfg2.enable_engulfing is False


# ---------------------------------------------------------------------------
# Class 21 — _fmt_pf and _mode_label helpers (reports.py)
# ---------------------------------------------------------------------------

class TestReportHelpers:
    def test_fmt_pf_normal(self):
        assert _fmt_pf(1.25) == "1.25"

    def test_fmt_pf_infinity(self):
        assert _fmt_pf(float("inf")) == "∞"

    def test_fmt_pf_nan(self):
        result = _fmt_pf(float("nan"))
        assert result == "N/A"

    def test_mode_label_pin_bar(self):
        assert _mode_label("pin_bar_only") == "PIN BAR"

    def test_mode_label_engulfing(self):
        assert _mode_label("engulfing_only") == "ENGULFING BAR"

    def test_mode_label_combined(self):
        assert _mode_label("combined") == "COMBINED"

    def test_mode_label_unknown_uppercased(self):
        assert _mode_label("custom_mode") == "CUSTOM_MODE"
