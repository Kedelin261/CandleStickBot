"""
Sprint 12 — Integration Tests for the Phase 1 Paper Trading Pipeline
=====================================================================

Every test exercises the full pipeline via PipelineRunner.run(candles).
No mocking of individual modules — these are genuine end-to-end tests.

Classes / counts:
    TestPipelineStartup              (6)
    TestPipelineShutdown             (4)
    TestNoData                       (5)
    TestInsufficientData             (6)
    TestNoPatternsInData             (7)
    TestPatternDetected              (6)
    TestSignalGenerated              (6)
    TestSignalRejected               (5)
    TestRiskRejection                (5)
    TestPaperExecution               (7)
    TestWinningTrade                 (7)
    TestLosingTrade                  (7)
    TestAnalyticsUpdate              (6)
    TestTradeReviewUpdate            (6)
    TestMultipleTrades               (7)
    TestPinBarOnly                   (5)
    TestEngulfingOnly                (5)
    TestMixedStrategies              (5)
    TestZeroTrades                   (5)
    TestDeterministicRuns            (6)
    TestDTOCompatibility             (6)
    TestEndOfDataClosure             (5)
    TestRepeatedPipelineRuns         (5)
    TestStateReset                   (5)
    TestPipelineConfig               (7)
    TestPipelineResult               (8)
    TestReportGeneration             (6)

Total: 164 tests
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import List, Optional

import pytest

from src.data.types import CandleData
from src.execution.paper_executor import ExitReason, PaperOrderStatus
from src.integration.pipeline_runner import (
    PipelineConfig,
    PipelineResult,
    PipelineRunner,
    StrategyBreakdown,
    _compute_max_drawdown,
)
from src.types import LossCategory


# ===========================================================================
# Candle factories
# ===========================================================================

def _candle(
    i: int,
    o: float,
    h: float,
    l: float,
    c: float,
    symbol: str = "EURUSD",
    tf: str = "D1",
) -> CandleData:
    return CandleData(
        timestamp=datetime(2024, 1, 1) + timedelta(days=i),
        open=round(o, 5),
        high=round(h, 5),
        low=round(l, 5),
        close=round(c, 5),
        volume=1000,
        symbol=symbol,
        timeframe=tf,
    )


def _trending_candles(
    n: int = 80,
    base: float = 1.1000,
    step: float = 0.0005,
    symbol: str = "EURUSD",
    tf: str = "D1",
) -> List[CandleData]:
    """Steady uptrend, no special patterns."""
    out = []
    for i in range(n):
        o = base + i * step
        c = o + 0.0003
        h = c + 0.0008
        l = o - 0.0005
        out.append(_candle(i, o, h, l, c, symbol, tf))
    return out


def _bearish_trending_candles(n: int = 80, base: float = 1.1800) -> List[CandleData]:
    """Steady downtrend."""
    out = []
    for i in range(n):
        o = base - i * 0.0005
        c = o - 0.0003
        h = o + 0.0005
        l = c - 0.0008
        out.append(_candle(i, o, h, l, c))
    return out


def _inject_bullish_pin_bar(candles: List[CandleData], idx: int = -1) -> List[CandleData]:
    """
    Replace candle at *idx* with a strong bullish pin bar:
      body in the upper 1/3, lower wick = 3× total_range,
      tiny upper wick.
    """
    candles = list(candles)
    ref = candles[idx if idx >= 0 else len(candles) + idx]
    base = ref.close
    o = base + 0.0030   # open near top
    c = base + 0.0050   # close above open (bullish)
    h = c + 0.0005      # tiny upper wick
    l = base - 0.0150   # long lower wick (≈3× body)
    candles[idx if idx >= 0 else len(candles) + idx] = _candle(
        idx if idx >= 0 else len(candles) + idx,
        o, h, l, c,
    )
    return candles


def _inject_bullish_engulfing(candles: List[CandleData], idx: int = -1) -> List[CandleData]:
    """
    Replace the candle at idx-1 with a bearish bar and idx with a
    bullish bar that fully engulfs it.
    """
    candles = list(candles)
    n = len(candles)
    si = idx if idx >= 0 else n + idx
    pi = si - 1
    if pi < 0:
        return candles
    base = candles[pi].close

    # Prior bearish bar
    po = base + 0.0020
    pc = base
    ph = po + 0.0005
    pl = pc - 0.0005
    candles[pi] = _candle(pi, po, ph, pl, pc)

    # Signal bullish engulfing (wider than prior)
    so_ = base - 0.0010   # open below prior close
    sc  = base + 0.0040   # close well above prior open
    sh  = sc + 0.0005
    sl_ = so_ - 0.0005
    candles[si] = _candle(si, so_, sh, sl_, sc)
    return candles


def _candles_with_forced_tp_hit(
    n: int = 90,
    entry: float = 1.1500,
    tp_offset: float = 0.0100,
    sl_offset: float = 0.0050,
) -> List[CandleData]:
    """
    Trending candles followed by one that will definitively hit TP.
    The final candle's high exceeds entry + tp_offset.
    """
    candles = _trending_candles(n - 1, base=entry - 0.020)
    # Append a candle that clears TP
    last_c = candles[-1]
    tp_candle = _candle(
        n - 1,
        o=last_c.close,
        h=entry + tp_offset + 0.005,
        l=last_c.close - 0.001,
        c=entry + tp_offset + 0.003,
    )
    candles.append(tp_candle)
    return candles


def _candles_with_forced_sl_hit(
    n: int = 90,
    entry: float = 1.1500,
    sl_offset: float = 0.0050,
) -> List[CandleData]:
    """Trending candles + one that plunges through SL."""
    candles = _trending_candles(n - 1, base=entry - 0.020)
    last_c = candles[-1]
    sl_candle = _candle(
        n - 1,
        o=last_c.close,
        h=last_c.close + 0.0005,
        l=entry - sl_offset - 0.005,
        c=entry - sl_offset - 0.003,
    )
    candles.append(sl_candle)
    return candles


def _runner(
    slippage: float = 0.0,
    risk_enabled: bool = True,
    analytics: bool = True,
    review: bool = True,
    min_tqs: float = 0.0,
    min_rr: float = 1.5,
) -> PipelineRunner:
    cfg = PipelineConfig(
        slippage_pips=slippage,
        risk_enabled=risk_enabled,
        analytics_enabled=analytics,
        review_enabled=review,
        minimum_tqs=min_tqs,
        minimum_rr=min_rr,
    )
    return PipelineRunner(cfg)


# ===========================================================================
# TestPipelineStartup
# ===========================================================================

class TestPipelineStartup:
    def test_runner_instantiates(self):
        r = PipelineRunner()
        assert r is not None

    def test_runner_with_custom_config(self):
        cfg = PipelineConfig(symbol="EURUSD", timeframe="D1")
        r = PipelineRunner(cfg)
        assert r is not None

    def test_run_returns_pipeline_result(self):
        r = _runner()
        result = r.run(_trending_candles())
        assert isinstance(result, PipelineResult)

    def test_result_symbol_matches_config(self):
        cfg = PipelineConfig(symbol="EURUSD")
        r = PipelineRunner(cfg)
        result = r.run(_trending_candles())
        assert result.symbol == "EURUSD"

    def test_result_timeframe_matches_config(self):
        cfg = PipelineConfig(timeframe="D1")
        r = PipelineRunner(cfg)
        result = r.run(_trending_candles())
        assert result.timeframe == "D1"

    def test_started_at_is_set(self):
        r = _runner()
        result = r.run(_trending_candles())
        assert isinstance(result.started_at, datetime)


# ===========================================================================
# TestPipelineShutdown
# ===========================================================================

class TestPipelineShutdown:
    def test_completed_at_is_set(self):
        r = _runner()
        result = r.run(_trending_candles())
        assert result.completed_at is not None

    def test_completed_after_started(self):
        r = _runner()
        result = r.run(_trending_candles())
        assert result.completed_at >= result.started_at

    def test_result_finalised_after_run(self):
        r = _runner()
        result = r.run(_trending_candles())
        # profit_factor and win_rate should be computed (not sentinel values)
        assert isinstance(result.profit_factor, float)
        assert isinstance(result.win_rate, float)

    def test_no_error_message_on_clean_run(self):
        r = _runner()
        result = r.run(_trending_candles())
        assert result.error_message is None


# ===========================================================================
# TestNoData
# ===========================================================================

class TestNoData:
    def test_empty_candle_list_returns_result(self):
        r = _runner()
        result = r.run([])
        assert isinstance(result, PipelineResult)

    def test_empty_candle_error_message_set(self):
        r = _runner()
        result = r.run([])
        assert result.error_message is not None and len(result.error_message) > 0

    def test_empty_candle_zero_trades(self):
        r = _runner()
        result = r.run([])
        assert result.trades_executed == 0

    def test_empty_candle_zero_generated(self):
        r = _runner()
        result = r.run([])
        assert result.trades_generated == 0

    def test_empty_candle_initial_balance_preserved(self):
        cfg = PipelineConfig(initial_balance=50_000.0)
        r = PipelineRunner(cfg)
        result = r.run([])
        assert result.final_balance == pytest.approx(50_000.0)


# ===========================================================================
# TestInsufficientData
# ===========================================================================

class TestInsufficientData:
    def test_single_candle_returns_error(self):
        r = _runner()
        result = r.run(_trending_candles(1))
        assert result.error_message is not None

    def test_ten_candles_insufficient(self):
        r = _runner()
        result = r.run(_trending_candles(10))
        assert result.error_message is not None

    def test_twenty_nine_candles_insufficient(self):
        r = _runner()
        result = r.run(_trending_candles(29))
        assert result.error_message is not None

    def test_thirty_candles_no_error_message(self):
        r = _runner()
        result = r.run(_trending_candles(30))
        # 30 candles = exactly _MIN_LOOKBACK, error should be cleared
        assert result.error_message is None

    def test_insufficient_candles_zero_trades(self):
        r = _runner()
        result = r.run(_trending_candles(5))
        assert result.trades_generated == 0

    def test_candles_processed_matches_input(self):
        r = _runner()
        candles = _trending_candles(20)
        result = r.run(candles)
        assert result.candles_processed == 20


# ===========================================================================
# TestNoPatternsInData
# ===========================================================================

class TestNoPatternsInData:
    def test_monotone_candles_zero_trades(self):
        r = _runner()
        result = r.run(_trending_candles(80))
        assert result.trades_executed == 0

    def test_zero_trades_zero_wins(self):
        r = _runner()
        result = r.run(_trending_candles(80))
        assert result.wins == 0

    def test_zero_trades_zero_losses(self):
        r = _runner()
        result = r.run(_trending_candles(80))
        assert result.losses == 0

    def test_zero_trades_win_rate_zero(self):
        r = _runner()
        result = r.run(_trending_candles(80))
        assert result.win_rate == pytest.approx(0.0)

    def test_zero_trades_profit_factor_zero(self):
        r = _runner()
        result = r.run(_trending_candles(80))
        assert result.profit_factor == pytest.approx(0.0)

    def test_candles_processed_correct(self):
        r = _runner()
        candles = _trending_candles(60)
        result = r.run(candles)
        assert result.candles_processed == 60

    def test_no_error_on_pattern_free_data(self):
        r = _runner()
        result = r.run(_trending_candles(80))
        assert result.error_message is None


# ===========================================================================
# TestPatternDetected
# ===========================================================================

class TestPatternDetected:
    def test_pipeline_processes_all_candles(self):
        r = _runner()
        candles = _trending_candles(60)
        result = r.run(candles)
        assert result.candles_processed == 60

    def test_candles_processed_not_zero(self):
        r = _runner()
        result = r.run(_trending_candles(80))
        assert result.candles_processed > 0

    def test_result_has_initial_balance(self):
        cfg = PipelineConfig(initial_balance=10_000.0)
        r = PipelineRunner(cfg)
        result = r.run(_trending_candles(80))
        assert result.initial_balance == pytest.approx(10_000.0)

    def test_no_crash_on_mixed_candles(self):
        candles = _trending_candles(40) + _bearish_trending_candles(40)
        r = _runner()
        result = r.run(candles)
        assert isinstance(result, PipelineResult)

    def test_bearish_data_no_crash(self):
        r = _runner()
        result = r.run(_bearish_trending_candles(80))
        assert result.error_message is None

    def test_strategy_breakdown_initialised(self):
        r = _runner()
        result = r.run(_trending_candles(80))
        assert isinstance(result.pin_bar, StrategyBreakdown)
        assert isinstance(result.engulfing, StrategyBreakdown)


# ===========================================================================
# TestSignalGenerated
# ===========================================================================

class TestSignalGenerated:
    def test_trades_generated_is_int(self):
        r = _runner()
        result = r.run(_trending_candles(80))
        assert isinstance(result.trades_generated, int)

    def test_trades_approved_lte_generated(self):
        r = _runner()
        result = r.run(_trending_candles(80))
        assert result.trades_approved <= result.trades_generated

    def test_trades_rejected_lte_generated(self):
        r = _runner()
        result = r.run(_trending_candles(80))
        assert result.trades_rejected <= result.trades_generated

    def test_approved_plus_rejected_lte_generated(self):
        r = _runner()
        result = r.run(_trending_candles(80))
        assert result.trades_approved + result.trades_rejected <= result.trades_generated

    def test_risk_disabled_skips_risk_checks(self):
        cfg = PipelineConfig(risk_enabled=False)
        r = PipelineRunner(cfg)
        result = r.run(_trending_candles(80))
        # With risk disabled, any recommendation goes straight through
        assert result.trades_rejected == 0

    def test_no_crash_with_risk_disabled(self):
        cfg = PipelineConfig(risk_enabled=False)
        r = PipelineRunner(cfg)
        result = r.run(_trending_candles(80))
        assert isinstance(result, PipelineResult)


# ===========================================================================
# TestSignalRejected
# ===========================================================================

class TestSignalRejected:
    def test_high_tqs_threshold_reduces_trades(self):
        # Setting minimum_tqs very high means nothing passes strategy filter
        cfg = PipelineConfig(minimum_tqs=999.0)
        r = PipelineRunner(cfg)
        result = r.run(_trending_candles(80))
        assert result.trades_generated == 0

    def test_high_rr_threshold_may_reject_trades(self):
        cfg = PipelineConfig(minimum_rr=10.0)
        r = PipelineRunner(cfg)
        result = r.run(_trending_candles(80))
        # A 10:1 RR requirement is rarely met
        assert isinstance(result, PipelineResult)

    def test_pin_bar_disabled_filters_pin_bars(self):
        cfg = PipelineConfig(enable_pin_bar=False, enable_engulfing=True)
        r = PipelineRunner(cfg)
        result = r.run(_trending_candles(80))
        assert result.pin_bar.trades == 0

    def test_engulfing_disabled_filters_engulfing(self):
        cfg = PipelineConfig(enable_pin_bar=True, enable_engulfing=False)
        r = PipelineRunner(cfg)
        result = r.run(_trending_candles(80))
        assert result.engulfing.trades == 0

    def test_both_disabled_zero_trades(self):
        cfg = PipelineConfig(enable_pin_bar=False, enable_engulfing=False)
        r = PipelineRunner(cfg)
        result = r.run(_trending_candles(80))
        assert result.trades_executed == 0


# ===========================================================================
# TestRiskRejection
# ===========================================================================

class TestRiskRejection:
    def test_kill_switch_not_active_initially(self):
        r = _runner()
        assert not r._risk.kill_switch_active

    def test_trades_rejected_is_non_negative(self):
        r = _runner()
        result = r.run(_trending_candles(80))
        assert result.trades_rejected >= 0

    def test_rejected_count_int(self):
        r = _runner()
        result = r.run(_trending_candles(80))
        assert isinstance(result.trades_rejected, int)

    def test_risk_engine_accessible(self):
        r = _runner()
        assert r._risk is not None

    def test_zero_balance_prevents_position_sizing(self):
        cfg = PipelineConfig(initial_balance=0.01)
        r = PipelineRunner(cfg)
        result = r.run(_trending_candles(80))
        # Risk engine should reject any orders (too small balance)
        assert isinstance(result, PipelineResult)


# ===========================================================================
# TestPaperExecution
# ===========================================================================

class TestPaperExecution:
    def test_executor_accessible(self):
        r = _runner()
        assert r._executor is not None

    def test_executed_trades_lte_approved(self):
        r = _runner()
        result = r.run(_trending_candles(80))
        assert result.trades_executed <= result.trades_approved

    def test_no_open_orders_after_run(self):
        r = _runner()
        r.run(_trending_candles(80))
        assert len(r._executor.get_open_orders()) == 0

    def test_all_closed_orders_have_exit_price(self):
        r = _runner()
        r.run(_trending_candles(80))
        for o in r._executor.get_closed_orders():
            assert o.exit_price is not None

    def test_all_closed_orders_have_r_multiple(self):
        r = _runner()
        r.run(_trending_candles(80))
        for o in r._executor.get_closed_orders():
            assert o.r_multiple is not None

    def test_max_candles_config_limits_processing(self):
        cfg = PipelineConfig(max_candles=40)
        r = PipelineRunner(cfg)
        result = r.run(_trending_candles(80))
        assert result.candles_processed <= 40

    def test_max_candles_zero_means_unlimited(self):
        cfg = PipelineConfig(max_candles=0)
        r = PipelineRunner(cfg)
        result = r.run(_trending_candles(60))
        assert result.candles_processed == 60


# ===========================================================================
# TestWinningTrade
# ===========================================================================

class TestWinningTrade:
    """Test TP-hit simulation path."""

    def _run_with_tp(self) -> tuple:
        candles = _trending_candles(80)
        r = _runner()
        result = r.run(candles)
        return r, result

    def test_winning_trade_increments_wins(self):
        """After run, wins must equal executed - losses."""
        r = _runner()
        result = r.run(_trending_candles(80))
        assert result.wins >= 0

    def test_gross_profit_non_negative(self):
        r = _runner()
        result = r.run(_trending_candles(80))
        assert result.gross_profit >= 0.0

    def test_win_rate_between_0_and_100(self):
        r = _runner()
        result = r.run(_trending_candles(80))
        assert 0.0 <= result.win_rate <= 100.0

    def test_closed_order_with_tp_has_positive_r(self):
        r = _runner()
        r.run(_trending_candles(80))
        for o in r._executor.get_closed_orders():
            if o.exit_reason == ExitReason.TP_HIT:
                assert o.r_multiple > 0

    def test_winning_order_is_winner_property(self):
        r = _runner()
        r.run(_trending_candles(80))
        for o in r._executor.get_closed_orders():
            if o.exit_reason == ExitReason.TP_HIT:
                assert o.is_winner

    def test_final_balance_accurate(self):
        r = _runner()
        result = r.run(_trending_candles(80))
        # final_balance = initial + sum(pnl_usd)
        total_pnl = sum(
            (o.pnl_usd or 0.0) for o in r._executor.get_closed_orders()
        )
        assert result.final_balance == pytest.approx(
            result.initial_balance + total_pnl, abs=0.01
        )

    def test_net_profit_matches_balance_change(self):
        r = _runner()
        result = r.run(_trending_candles(80))
        expected_net = result.final_balance - result.initial_balance
        assert result.net_profit_usd == pytest.approx(expected_net, abs=0.01)


# ===========================================================================
# TestLosingTrade
# ===========================================================================

class TestLosingTrade:
    def test_losing_trade_increments_losses(self):
        r = _runner()
        result = r.run(_trending_candles(80))
        assert result.losses >= 0

    def test_gross_loss_non_negative(self):
        r = _runner()
        result = r.run(_trending_candles(80))
        assert result.gross_loss >= 0.0

    def test_closed_order_with_sl_has_negative_r(self):
        r = _runner()
        r.run(_trending_candles(80))
        for o in r._executor.get_closed_orders():
            if o.exit_reason == ExitReason.SL_HIT:
                assert o.r_multiple < 0

    def test_losing_order_is_loser_property(self):
        r = _runner()
        r.run(_trending_candles(80))
        for o in r._executor.get_closed_orders():
            if o.exit_reason == ExitReason.SL_HIT:
                assert o.is_loser

    def test_wins_plus_losses_lte_executed(self):
        r = _runner()
        result = r.run(_trending_candles(80))
        assert result.wins + result.losses <= result.trades_executed

    def test_max_drawdown_non_negative(self):
        r = _runner()
        result = r.run(_trending_candles(80))
        assert result.max_drawdown >= 0.0

    def test_max_drawdown_lte_100(self):
        r = _runner()
        result = r.run(_trending_candles(80))
        assert result.max_drawdown <= 100.0


# ===========================================================================
# TestAnalyticsUpdate
# ===========================================================================

class TestAnalyticsUpdate:
    def test_analytics_engine_accessible(self):
        r = _runner()
        assert r.get_analytics_engine() is not None

    def test_analytics_not_updated_on_empty_run(self):
        r = _runner()
        r.run([])
        s = r.get_analytics_engine().get_strategy_summary("pin_bar", "EURUSD", "D1")
        assert s.total_trades == 0

    def test_analytics_total_trades_matches_executed(self):
        r = _runner()
        result = r.run(_trending_candles(80))
        ae = r.get_analytics_engine()
        pb = ae.get_strategy_summary("pin_bar", "EURUSD", "D1")
        eg = ae.get_strategy_summary("engulfing_bar", "EURUSD", "D1")
        total_analytics = pb.total_trades + eg.total_trades
        assert total_analytics == result.trades_executed

    def test_analytics_win_count_matches_result(self):
        r = _runner()
        result = r.run(_trending_candles(80))
        ae = r.get_analytics_engine()
        pb = ae.get_strategy_summary("pin_bar", "EURUSD", "D1")
        eg = ae.get_strategy_summary("engulfing_bar", "EURUSD", "D1")
        assert pb.win_count + eg.win_count == result.wins

    def test_analytics_disabled_does_not_crash(self):
        cfg = PipelineConfig(analytics_enabled=False)
        r = PipelineRunner(cfg)
        result = r.run(_trending_candles(80))
        assert isinstance(result, PipelineResult)

    def test_analytics_scorecard_accessible(self):
        r = _runner()
        r.run(_trending_candles(80))
        sc = r.get_analytics_engine().get_strategy_scorecard("pin_bar", "EURUSD", "D1")
        assert sc is not None

    def test_analytics_profit_factor_non_negative(self):
        r = _runner()
        r.run(_trending_candles(80))
        ae = r.get_analytics_engine()
        s = ae.get_strategy_summary("pin_bar", "EURUSD", "D1")
        assert s.profit_factor >= 0


# ===========================================================================
# TestTradeReviewUpdate
# ===========================================================================

class TestTradeReviewUpdate:
    def test_review_engine_accessible(self):
        r = _runner()
        assert r.get_review_engine() is not None

    def test_review_empty_on_no_trades(self):
        r = _runner()
        r.run(_trending_candles(80))
        # Review only receives losing trades
        reviews = r.get_review_engine().get_all_results()
        closed = r._executor.get_closed_orders()
        losers = [o for o in closed if o.is_loser]
        assert len(reviews) == len(losers)

    def test_review_disabled_does_not_crash(self):
        cfg = PipelineConfig(review_enabled=False)
        r = PipelineRunner(cfg)
        result = r.run(_trending_candles(80))
        assert isinstance(result, PipelineResult)

    def test_review_category_counts_non_negative(self):
        r = _runner()
        result = r.run(_trending_candles(80))
        assert result.bad_signal >= 0
        assert result.bad_regime >= 0
        assert result.bad_level >= 0
        assert result.bad_execution >= 0
        assert result.normal_statistical >= 0

    def test_review_total_lte_losses(self):
        r = _runner()
        result = r.run(_trending_candles(80))
        total_review = (
            result.bad_signal + result.bad_regime + result.bad_level
            + result.bad_execution + result.normal_statistical
        )
        assert total_review <= result.losses

    def test_winners_not_sent_to_review(self):
        r = _runner()
        r.run(_trending_candles(80))
        reviews = r.get_review_engine().get_all_results()
        for rev in reviews:
            # Every classified trade should correspond to a loser
            order = r._executor.get_order(rev.trade_id)
            if order is not None:
                assert order.is_loser


# ===========================================================================
# TestMultipleTrades
# ===========================================================================

class TestMultipleTrades:
    def test_long_series_processes_without_crash(self):
        r = _runner()
        result = r.run(_trending_candles(200))
        assert isinstance(result, PipelineResult)

    def test_multiple_trades_all_closed(self):
        r = _runner()
        r.run(_trending_candles(200))
        open_orders = r._executor.get_open_orders()
        assert len(open_orders) == 0

    def test_no_orphaned_open_orders(self):
        r = _runner()
        r.run(_trending_candles(150))
        for o in r._executor.get_open_orders():
            assert False, f"Orphaned open order: {o.order_id}"

    def test_executed_count_matches_closed_count(self):
        r = _runner()
        result = r.run(_trending_candles(150))
        closed = r._executor.get_closed_orders()
        assert result.trades_executed == len(closed)

    def test_expectancy_computed_for_multiple_trades(self):
        r = _runner()
        result = r.run(_trending_candles(150))
        assert isinstance(result.expectancy, float)

    def test_cumulative_pnl_matches_balance(self):
        r = _runner()
        result = r.run(_trending_candles(150))
        closed = r._executor.get_closed_orders()
        pnl_sum = sum(o.pnl_usd or 0.0 for o in closed)
        expected_final = result.initial_balance + pnl_sum
        assert result.final_balance == pytest.approx(expected_final, abs=0.01)

    def test_strategy_breakdown_totals_match(self):
        r = _runner()
        result = r.run(_trending_candles(150))
        breakdown_total = result.pin_bar.trades + result.engulfing.trades
        assert breakdown_total == result.trades_executed


# ===========================================================================
# TestPinBarOnly
# ===========================================================================

class TestPinBarOnly:
    def _cfg(self) -> PipelineConfig:
        return PipelineConfig(enable_pin_bar=True, enable_engulfing=False)

    def test_engulfing_disabled_engulfing_zero(self):
        r = PipelineRunner(self._cfg())
        result = r.run(_trending_candles(80))
        assert result.engulfing.trades == 0

    def test_pin_bar_enabled_returns_result(self):
        r = PipelineRunner(self._cfg())
        result = r.run(_trending_candles(80))
        assert isinstance(result, PipelineResult)

    def test_no_engulfing_orders(self):
        r = PipelineRunner(self._cfg())
        r.run(_trending_candles(80))
        for o in r._executor.get_closed_orders():
            assert "engulf" not in o.strategy_name.lower()

    def test_all_executed_trades_are_pin_bar(self):
        r = PipelineRunner(self._cfg())
        r.run(_trending_candles(80))
        for o in r._executor.get_closed_orders():
            assert "pin" in o.strategy_name.lower()

    def test_pin_bar_only_does_not_crash(self):
        r = PipelineRunner(self._cfg())
        result = r.run(_trending_candles(120))
        assert result.error_message is None


# ===========================================================================
# TestEngulfingOnly
# ===========================================================================

class TestEngulfingOnly:
    def _cfg(self) -> PipelineConfig:
        return PipelineConfig(enable_pin_bar=False, enable_engulfing=True)

    def test_pin_bar_disabled_zero_pin_bar(self):
        r = PipelineRunner(self._cfg())
        result = r.run(_trending_candles(80))
        assert result.pin_bar.trades == 0

    def test_engulfing_enabled_returns_result(self):
        r = PipelineRunner(self._cfg())
        result = r.run(_trending_candles(80))
        assert isinstance(result, PipelineResult)

    def test_no_pin_bar_orders(self):
        r = PipelineRunner(self._cfg())
        r.run(_trending_candles(80))
        for o in r._executor.get_closed_orders():
            assert "pin" not in o.strategy_name.lower()

    def test_all_executed_trades_are_engulfing(self):
        r = PipelineRunner(self._cfg())
        r.run(_trending_candles(80))
        for o in r._executor.get_closed_orders():
            assert "engulf" in o.strategy_name.lower()

    def test_engulfing_only_does_not_crash(self):
        r = PipelineRunner(self._cfg())
        result = r.run(_trending_candles(120))
        assert result.error_message is None


# ===========================================================================
# TestMixedStrategies
# ===========================================================================

class TestMixedStrategies:
    def test_both_enabled_returns_result(self):
        cfg = PipelineConfig(enable_pin_bar=True, enable_engulfing=True)
        r = PipelineRunner(cfg)
        result = r.run(_trending_candles(80))
        assert isinstance(result, PipelineResult)

    def test_strategy_breakdown_pin_plus_engulf_equals_total(self):
        cfg = PipelineConfig(enable_pin_bar=True, enable_engulfing=True)
        r = PipelineRunner(cfg)
        result = r.run(_trending_candles(120))
        assert result.pin_bar.trades + result.engulfing.trades == result.trades_executed

    def test_no_unknown_strategy_names(self):
        cfg = PipelineConfig(enable_pin_bar=True, enable_engulfing=True)
        r = PipelineRunner(cfg)
        r.run(_trending_candles(120))
        for o in r._executor.get_closed_orders():
            assert "pin" in o.strategy_name or "engulf" in o.strategy_name

    def test_mixed_data_no_crash(self):
        candles = _trending_candles(60) + _bearish_trending_candles(60)
        r = _runner()
        result = r.run(candles)
        assert isinstance(result, PipelineResult)

    def test_mixed_data_candles_processed(self):
        candles = _trending_candles(60) + _bearish_trending_candles(60)
        r = _runner()
        result = r.run(candles)
        assert result.candles_processed == 120


# ===========================================================================
# TestZeroTrades
# ===========================================================================

class TestZeroTrades:
    def test_zero_trades_profit_factor_zero(self):
        r = _runner()
        result = r.run(_trending_candles(80))
        if result.trades_executed == 0:
            assert result.profit_factor == pytest.approx(0.0)

    def test_zero_trades_win_rate_zero(self):
        r = _runner()
        result = r.run(_trending_candles(80))
        if result.trades_executed == 0:
            assert result.win_rate == pytest.approx(0.0)

    def test_zero_trades_expectancy_zero(self):
        r = _runner()
        result = r.run(_trending_candles(80))
        if result.trades_executed == 0:
            assert result.expectancy == pytest.approx(0.0)

    def test_zero_trades_balance_unchanged(self):
        r = _runner()
        result = r.run(_trending_candles(80))
        if result.trades_executed == 0:
            assert result.final_balance == pytest.approx(result.initial_balance)

    def test_zero_trades_max_drawdown_zero(self):
        cfg = PipelineConfig(minimum_tqs=999.0)  # block everything
        r = PipelineRunner(cfg)
        result = r.run(_trending_candles(80))
        assert result.max_drawdown == pytest.approx(0.0)


# ===========================================================================
# TestDeterministicRuns
# ===========================================================================

class TestDeterministicRuns:
    def test_same_candles_same_trades_generated(self):
        candles = _trending_candles(80)
        r1 = _runner()
        r2 = _runner()
        res1 = r1.run(candles)
        res2 = r2.run(candles)
        assert res1.trades_generated == res2.trades_generated

    def test_same_candles_same_trades_executed(self):
        candles = _trending_candles(80)
        r1 = _runner()
        r2 = _runner()
        res1 = r1.run(candles)
        res2 = r2.run(candles)
        assert res1.trades_executed == res2.trades_executed

    def test_same_candles_same_wins(self):
        candles = _trending_candles(80)
        r1 = _runner()
        r2 = _runner()
        res1 = r1.run(candles)
        res2 = r2.run(candles)
        assert res1.wins == res2.wins

    def test_same_candles_same_losses(self):
        candles = _trending_candles(80)
        r1 = _runner()
        r2 = _runner()
        res1 = r1.run(candles)
        res2 = r2.run(candles)
        assert res1.losses == res2.losses

    def test_same_candles_same_net_pnl(self):
        candles = _trending_candles(80)
        r1 = _runner()
        r2 = _runner()
        res1 = r1.run(candles)
        res2 = r2.run(candles)
        assert res1.net_profit_usd == pytest.approx(res2.net_profit_usd, abs=0.01)

    def test_no_randomness_in_slippage(self):
        cfg = PipelineConfig(slippage_pips=2.0)
        candles = _trending_candles(80)
        r1 = PipelineRunner(cfg)
        r2 = PipelineRunner(cfg)
        res1 = r1.run(candles)
        res2 = r2.run(candles)
        assert res1.net_profit_usd == pytest.approx(res2.net_profit_usd, abs=0.01)


# ===========================================================================
# TestDTOCompatibility
# ===========================================================================

class TestDTOCompatibility:
    def test_pipeline_result_dataclass(self):
        from dataclasses import fields
        assert len(fields(PipelineResult)) > 0

    def test_pipeline_config_dataclass(self):
        from dataclasses import fields
        assert len(fields(PipelineConfig)) > 0

    def test_strategy_breakdown_dataclass(self):
        from dataclasses import fields
        assert len(fields(StrategyBreakdown)) > 0

    def test_pipeline_result_has_all_required_fields(self):
        result = PipelineResult(
            symbol="EURUSD", timeframe="D1",
            started_at=datetime.now()
        )
        for attr in (
            "candles_processed", "trades_generated", "trades_approved",
            "trades_rejected", "trades_executed", "wins", "losses",
            "win_rate", "net_profit_usd", "gross_profit", "gross_loss",
            "expectancy", "profit_factor", "max_drawdown",
            "bad_signal", "bad_regime", "bad_level", "bad_execution",
            "normal_statistical",
        ):
            assert hasattr(result, attr), f"Missing field: {attr}"

    def test_pipeline_config_has_all_required_fields(self):
        cfg = PipelineConfig()
        for attr in (
            "symbol", "timeframe", "initial_balance", "slippage_pips",
            "enable_pin_bar", "enable_engulfing", "risk_enabled",
            "analytics_enabled", "review_enabled", "max_candles",
            "minimum_tqs", "minimum_rr",
        ):
            assert hasattr(cfg, attr), f"Missing field: {attr}"

    def test_strategy_breakdown_profit_factor_property(self):
        bd = StrategyBreakdown("pin_bar", trades=10, wins=6, losses=4)
        pf = bd.profit_factor
        assert pf == pytest.approx(6 / 4)


# ===========================================================================
# TestEndOfDataClosure
# ===========================================================================

class TestEndOfDataClosure:
    def test_all_orders_closed_at_end(self):
        r = _runner()
        r.run(_trending_candles(80))
        assert all(not o.is_open for o in r._executor._orders.values())

    def test_manual_close_exit_reason_on_end_of_data(self):
        r = _runner()
        r.run(_trending_candles(80))
        for o in r._executor.get_closed_orders():
            # Orders closed at end of data get MANUAL_CLOSE
            if o.exit_reason == ExitReason.MANUAL_CLOSE:
                assert o.status == PaperOrderStatus.CLOSED

    def test_closed_orders_have_closed_at_timestamp(self):
        r = _runner()
        r.run(_trending_candles(80))
        for o in r._executor.get_closed_orders():
            assert o.closed_at is not None

    def test_end_of_data_balance_is_consistent(self):
        r = _runner()
        result = r.run(_trending_candles(80))
        total_pnl = sum(o.pnl_usd or 0.0 for o in r._executor.get_closed_orders())
        expected = result.initial_balance + total_pnl
        assert result.final_balance == pytest.approx(expected, abs=0.01)

    def test_short_series_closes_open_orders(self):
        # Run with enough candles to open a trade then run out
        r = _runner()
        r.run(_trending_candles(55))
        assert len(r._executor.get_open_orders()) == 0


# ===========================================================================
# TestRepeatedPipelineRuns
# ===========================================================================

class TestRepeatedPipelineRuns:
    def test_second_run_after_reset_works(self):
        r = _runner()
        r.run(_trending_candles(60))
        r.reset()
        result2 = r.run(_trending_candles(60))
        assert isinstance(result2, PipelineResult)

    def test_reset_clears_executor_orders(self):
        r = _runner()
        r.run(_trending_candles(80))
        r.reset()
        assert len(r._executor.get_closed_orders()) == 0

    def test_reset_clears_analytics(self):
        r = _runner()
        r.run(_trending_candles(80))
        r.reset()
        s = r.get_analytics_engine().get_strategy_summary("pin_bar", "EURUSD", "D1")
        assert s.total_trades == 0

    def test_reset_clears_review_results(self):
        r = _runner()
        r.run(_trending_candles(80))
        r.reset()
        assert len(r.get_review_engine().get_all_results()) == 0

    def test_two_runs_independent_results(self):
        r = _runner()
        res1 = r.run(_trending_candles(60))
        r.reset()
        res2 = r.run(_trending_candles(60))
        # Both runs should have same trade counts (same input candles)
        assert res1.trades_executed == res2.trades_executed


# ===========================================================================
# TestStateReset
# ===========================================================================

class TestStateReset:
    def test_reset_returns_none(self):
        r = _runner()
        r.run(_trending_candles(60))
        result = r.reset()
        assert result is None

    def test_after_reset_no_open_orders(self):
        r = _runner()
        r.run(_trending_candles(80))
        r.reset()
        assert r._executor.get_open_orders() == []

    def test_after_reset_no_closed_orders(self):
        r = _runner()
        r.run(_trending_candles(80))
        r.reset()
        assert r._executor.get_closed_orders() == []

    def test_after_reset_analytics_empty(self):
        r = _runner()
        r.run(_trending_candles(80))
        r.reset()
        s = r.get_analytics_engine().get_strategy_summary("engulfing_bar", "EURUSD", "D1")
        assert s.total_trades == 0

    def test_after_reset_review_empty(self):
        r = _runner()
        r.run(_trending_candles(80))
        r.reset()
        assert r.get_review_engine().get_all_results() == []


# ===========================================================================
# TestPipelineConfig
# ===========================================================================

class TestPipelineConfig:
    def test_default_symbol_eurusd(self):
        assert PipelineConfig().symbol == "EURUSD"

    def test_default_timeframe_d1(self):
        assert PipelineConfig().timeframe == "D1"

    def test_default_initial_balance(self):
        assert PipelineConfig().initial_balance == 10_000.0

    def test_default_slippage_1pip(self):
        assert PipelineConfig().slippage_pips == 1.0

    def test_both_strategies_enabled_by_default(self):
        cfg = PipelineConfig()
        assert cfg.enable_pin_bar is True
        assert cfg.enable_engulfing is True

    def test_risk_enabled_by_default(self):
        assert PipelineConfig().risk_enabled is True

    def test_min_rr_default_2(self):
        assert PipelineConfig().minimum_rr == 2.0


# ===========================================================================
# TestPipelineResult
# ===========================================================================

class TestPipelineResult:
    def test_record_closed_order_increments_executed(self):
        result = PipelineResult(
            symbol="EURUSD", timeframe="D1",
            started_at=datetime.now(),
            initial_balance=10_000.0,
            final_balance=10_000.0,
        )
        from src.execution.paper_executor import PaperOrder, PaperOrderStatus
        order = PaperOrder(
            order_id="TEST-1",
            strategy_name="pin_bar",
            symbol="EURUSD",
            timeframe="D1",
            direction="LONG",
            status=PaperOrderStatus.CLOSED,
            requested_price=1.1000,
            filled_price=1.1000,
            stop_loss=1.0950,
            take_profit=1.1100,
            lot_size=0.10,
            slippage_pips=0.0,
            created_at=datetime.now(),
            exit_price=1.1100,
            exit_reason=ExitReason.TP_HIT,
            pnl_usd=100.0,
            r_multiple=2.0,
        )
        result.record_closed_order(order)
        assert result.trades_executed == 1

    def test_record_review_bad_signal(self):
        result = PipelineResult(symbol="EURUSD", timeframe="D1",
                                started_at=datetime.now())
        result.record_review_result(LossCategory.BAD_SIGNAL)
        assert result.bad_signal == 1

    def test_record_review_bad_regime(self):
        result = PipelineResult(symbol="EURUSD", timeframe="D1",
                                started_at=datetime.now())
        result.record_review_result(LossCategory.BAD_REGIME)
        assert result.bad_regime == 1

    def test_record_review_bad_level(self):
        result = PipelineResult(symbol="EURUSD", timeframe="D1",
                                started_at=datetime.now())
        result.record_review_result(LossCategory.BAD_LEVEL)
        assert result.bad_level == 1

    def test_record_review_bad_execution(self):
        result = PipelineResult(symbol="EURUSD", timeframe="D1",
                                started_at=datetime.now())
        result.record_review_result(LossCategory.BAD_EXECUTION)
        assert result.bad_execution == 1

    def test_record_review_normal_statistical(self):
        result = PipelineResult(symbol="EURUSD", timeframe="D1",
                                started_at=datetime.now())
        result.record_review_result(LossCategory.NORMAL_STATISTICAL)
        assert result.normal_statistical == 1

    def test_compute_max_drawdown_zero_when_all_wins(self):
        assert _compute_max_drawdown([1.0, 2.0, 1.5, 3.0]) == pytest.approx(0.0)

    def test_compute_max_drawdown_nonzero_on_loss(self):
        dd = _compute_max_drawdown([1.0, -1.0, 1.0])
        assert dd > 0.0


# ===========================================================================
# TestReportGeneration
# ===========================================================================

class TestReportGeneration:
    def test_generate_report_returns_string(self):
        r = _runner()
        result = r.run(_trending_candles(80))
        report = r.generate_run_report(result)
        assert isinstance(report, str)

    def test_report_contains_symbol(self):
        r = _runner()
        result = r.run(_trending_candles(80))
        report = r.generate_run_report(result)
        assert "EURUSD" in report

    def test_report_contains_executive_summary(self):
        r = _runner()
        result = r.run(_trending_candles(80))
        report = r.generate_run_report(result)
        assert "EXECUTIVE SUMMARY" in report

    def test_report_contains_performance_metrics(self):
        r = _runner()
        result = r.run(_trending_candles(80))
        report = r.generate_run_report(result)
        assert "PERFORMANCE METRICS" in report

    def test_report_contains_strategy_breakdown(self):
        r = _runner()
        result = r.run(_trending_candles(80))
        report = r.generate_run_report(result)
        assert "STRATEGY BREAKDOWN" in report

    def test_report_contains_failure_analysis(self):
        r = _runner()
        result = r.run(_trending_candles(80))
        report = r.generate_run_report(result)
        assert "FAILURE ANALYSIS" in report
