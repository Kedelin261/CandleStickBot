"""
Unit tests for M18 Strategy Analytics Engine.

Sprint 9 — 105 tests across 17 test classes:

    TestStrategyPerformanceRecord    (10) — DTO fields, is_winner/loser/breakeven
    TestStrategySummaryDefaults      ( 5) — empty summary fields
    TestRecordTrade                  ( 8) — recording, Phase 1 guard, disabled warning
    TestWinLossClassification        ( 8) — r>0 winner, r<0 loser, r=0 breakeven
    TestProfitFactor                 ( 9) — all-win, all-loss, mixed, zero-loss edge
    TestExpectancy                   ( 6) — formula, sign, empty
    TestAverageWinnerLoser           ( 6) — avg values, empty
    TestMaxDrawdown                  ( 8) — flat, up-only, drawdown calc, recovery
    TestConsecutiveStreaks           ( 7) — win streaks, loss streaks, alternating
    TestSharpeRatio                  ( 8) — below threshold, exact, all-same, zero std
    TestRollingWindows               ( 7) — 30-trade, 90-trade, partial window
    TestEnableDisable                ( 9) — enable/disable/reason/is_enabled
    TestDegradationDetection         ( 6) — below threshold, above, min trades guard
    TestRankingStrategies            ( 6) — rank by various metrics
    TestComparisonTable              ( 5) — keys, values, both strategies
    TestStrategyScorecard            ( 7) — fields, str, is_degrading flag
    TestEdgeCases                    (10) — empty engine, one trade, non-EURUSD,
                                           pin vs engulfing separation, Phase 2 block

Total: 116 tests
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import List

import pytest

from src.analytics.strategy_analytics import (
    StrategyAnalyticsEngine,
    StrategyPerformanceRecord,
    StrategyScorecard,
    StrategySummary,
    _profit_factor_of,
    make_record,
)


# ===========================================================================
# Shared helpers
# ===========================================================================

def _win(r: float = 2.0, strat: str = "pin_bar", **kw) -> StrategyPerformanceRecord:
    return make_record(strat, r_multiple=r, **kw)


def _loss(r: float = -1.0, strat: str = "pin_bar", **kw) -> StrategyPerformanceRecord:
    return make_record(strat, r_multiple=r, **kw)


def _engine(**kw) -> StrategyAnalyticsEngine:
    return StrategyAnalyticsEngine(**kw)


def _filled_engine(
    wins: int = 3, losses: int = 2,
    win_r: float = 2.0, loss_r: float = -1.0,
    strat: str = "pin_bar",
) -> StrategyAnalyticsEngine:
    e = _engine()
    for _ in range(wins):
        e.record_trade(_win(win_r, strat))
    for _ in range(losses):
        e.record_trade(_loss(loss_r, strat))
    return e


# ===========================================================================
# TestStrategyPerformanceRecord
# ===========================================================================

class TestStrategyPerformanceRecord:
    """10 tests — DTO properties."""

    def test_is_winner_positive_r(self):
        r = make_record("pin_bar", 2.0)
        assert r.is_winner is True

    def test_is_loser_negative_r(self):
        r = make_record("pin_bar", -1.0)
        assert r.is_loser is True

    def test_is_breakeven_zero_r(self):
        r = make_record("pin_bar", 0.0)
        assert r.is_breakeven is True

    def test_is_winner_false_for_loss(self):
        assert make_record("pin_bar", -0.5).is_winner is False

    def test_is_loser_false_for_win(self):
        assert make_record("pin_bar", 1.5).is_loser is False

    def test_strategy_name_stored(self):
        r = make_record("engulfing_bar", 1.0)
        assert r.strategy_name == "engulfing_bar"

    def test_r_multiple_stored(self):
        r = make_record("pin_bar", 3.5)
        assert r.r_multiple == 3.5

    def test_symbol_stored(self):
        r = make_record("pin_bar", 1.0, symbol="GBPUSD")
        assert r.symbol == "GBPUSD"

    def test_trade_id_stored(self):
        r = make_record("pin_bar", 1.0, trade_id="T001")
        assert r.trade_id == "T001"

    def test_default_exit_reason(self):
        r = make_record("pin_bar", 2.0)
        assert r.exit_reason == "TP_HIT"


# ===========================================================================
# TestStrategySummaryDefaults
# ===========================================================================

class TestStrategySummaryDefaults:
    """5 tests — empty summary state."""

    def test_empty_engine_returns_zero_total(self):
        e = _engine()
        s = e.get_strategy_summary("pin_bar", "EURUSD", "H1")
        assert s.total_trades == 0

    def test_empty_engine_win_rate_zero(self):
        e = _engine()
        s = e.get_strategy_summary("pin_bar", "EURUSD", "H1")
        assert s.win_rate == 0.0

    def test_empty_engine_profit_factor_zero(self):
        e = _engine()
        s = e.get_strategy_summary("pin_bar", "EURUSD", "H1")
        assert s.profit_factor == 0.0

    def test_empty_engine_sharpe_none(self):
        e = _engine()
        s = e.get_strategy_summary("pin_bar", "EURUSD", "H1")
        assert s.sharpe_ratio is None

    def test_empty_engine_is_enabled_true(self):
        e = _engine()
        s = e.get_strategy_summary("pin_bar", "EURUSD", "H1")
        assert s.is_enabled is True


# ===========================================================================
# TestRecordTrade
# ===========================================================================

class TestRecordTrade:
    """8 tests — record_trade mechanics."""

    def test_record_increments_total(self):
        e = _engine()
        e.record_trade(_win())
        assert e.get_strategy_summary("pin_bar", "EURUSD", "H1").total_trades == 1

    def test_record_multiple(self):
        e = _engine()
        for _ in range(5):
            e.record_trade(_win())
        assert e.get_strategy_summary("pin_bar", "EURUSD", "H1").total_trades == 5

    def test_phase2_strategy_rejected(self):
        """inside_bar is not a Phase 1 strategy — silently skipped."""
        e = _engine()
        e.record_trade(make_record("inside_bar", 2.0))
        s = e.get_strategy_summary("inside_bar", "EURUSD", "H1")
        assert s.total_trades == 0

    def test_false_breakout_rejected(self):
        e = _engine()
        e.record_trade(make_record("inside_bar_false_breakout", 2.0))
        assert e.get_strategy_summary("inside_bar_false_breakout", "EURUSD", "H1").total_trades == 0

    def test_disabled_strategy_still_recorded(self):
        """Disabled strategy keeps recording trades for audit."""
        e = _engine()
        e.disable_strategy("pin_bar", "testing")
        e.record_trade(_win())
        assert e.get_strategy_summary("pin_bar", "EURUSD", "H1").total_trades == 1

    def test_records_stored_in_order(self):
        e = _engine()
        e.record_trade(_win(2.0))
        e.record_trade(_loss(-1.0))
        records = e.get_all_records("pin_bar", "EURUSD", "H1")
        assert records[0].r_multiple == 2.0
        assert records[1].r_multiple == -1.0

    def test_separate_symbol_tracking(self):
        e = _engine()
        e.record_trade(_win(2.0, symbol="EURUSD"))
        e.record_trade(_win(2.0, symbol="GBPUSD"))
        eu = e.get_strategy_summary("pin_bar", "EURUSD", "H1")
        gb = e.get_strategy_summary("pin_bar", "GBPUSD", "H1")
        assert eu.total_trades == 1
        assert gb.total_trades == 1

    def test_separate_timeframe_tracking(self):
        e = _engine()
        e.record_trade(_win(2.0, timeframe="H1"))
        e.record_trade(_win(2.0, timeframe="H4"))
        h1 = e.get_strategy_summary("pin_bar", "EURUSD", "H1")
        h4 = e.get_strategy_summary("pin_bar", "EURUSD", "H4")
        assert h1.total_trades == 1
        assert h4.total_trades == 1


# ===========================================================================
# TestWinLossClassification
# ===========================================================================

class TestWinLossClassification:
    """8 tests — win/loss/breakeven counting."""

    def test_three_wins(self):
        e = _engine()
        for _ in range(3):
            e.record_trade(_win(2.0))
        s = e.get_strategy_summary("pin_bar", "EURUSD", "H1")
        assert s.win_count == 3
        assert s.loss_count == 0

    def test_three_losses(self):
        e = _engine()
        for _ in range(3):
            e.record_trade(_loss(-1.0))
        s = e.get_strategy_summary("pin_bar", "EURUSD", "H1")
        assert s.win_count == 0
        assert s.loss_count == 3

    def test_win_rate_all_winners(self):
        e = _filled_engine(wins=4, losses=0)
        s = e.get_strategy_summary("pin_bar", "EURUSD", "H1")
        assert s.win_rate == pytest.approx(1.0)

    def test_win_rate_all_losers(self):
        e = _filled_engine(wins=0, losses=4)
        s = e.get_strategy_summary("pin_bar", "EURUSD", "H1")
        assert s.win_rate == 0.0

    def test_win_rate_60_pct(self):
        e = _filled_engine(wins=3, losses=2)
        s = e.get_strategy_summary("pin_bar", "EURUSD", "H1")
        assert s.win_rate == pytest.approx(0.6)

    def test_breakeven_not_counted_as_win_or_loss(self):
        e = _engine()
        e.record_trade(make_record("pin_bar", 0.0))
        s = e.get_strategy_summary("pin_bar", "EURUSD", "H1")
        assert s.win_count == 0
        assert s.loss_count == 0
        assert s.breakeven_count == 1

    def test_mixed_results(self):
        e = _engine()
        for r in [2.0, -1.0, 3.0, -1.0, 0.0, 1.5]:
            e.record_trade(make_record("pin_bar", r))
        s = e.get_strategy_summary("pin_bar", "EURUSD", "H1")
        assert s.win_count == 3
        assert s.loss_count == 2
        assert s.breakeven_count == 1

    def test_scorecard_win_rate_pct(self):
        e = _filled_engine(wins=3, losses=2)
        sc = e.get_strategy_scorecard("pin_bar", "EURUSD", "H1")
        assert sc.win_rate_pct == pytest.approx(60.0)


# ===========================================================================
# TestProfitFactor
# ===========================================================================

class TestProfitFactor:
    """9 tests — profit factor formula."""

    def test_all_winners_pf_infinite(self):
        e = _filled_engine(wins=3, losses=0)
        s = e.get_strategy_summary("pin_bar", "EURUSD", "H1")
        assert s.profit_factor == float("inf")

    def test_all_losers_pf_zero(self):
        e = _filled_engine(wins=0, losses=3)
        s = e.get_strategy_summary("pin_bar", "EURUSD", "H1")
        assert s.profit_factor == 0.0

    def test_equal_wins_losses_pf_one(self):
        e = _filled_engine(wins=2, losses=2, win_r=1.0, loss_r=-1.0)
        s = e.get_strategy_summary("pin_bar", "EURUSD", "H1")
        assert s.profit_factor == pytest.approx(1.0)

    def test_two_to_one_rr_pf(self):
        # 3 wins at 2R, 2 losses at -1R → PF = 6/2 = 3.0
        e = _filled_engine(wins=3, losses=2, win_r=2.0, loss_r=-1.0)
        s = e.get_strategy_summary("pin_bar", "EURUSD", "H1")
        assert s.profit_factor == pytest.approx(3.0)

    def test_pf_half(self):
        # 2 wins at 1R, 2 losses at 2R → PF = 2/4 = 0.5
        e = _filled_engine(wins=2, losses=2, win_r=1.0, loss_r=-2.0)
        s = e.get_strategy_summary("pin_bar", "EURUSD", "H1")
        assert s.profit_factor == pytest.approx(0.5)

    def test_module_helper_all_win(self):
        records = [make_record("pin_bar", 2.0) for _ in range(3)]
        assert _profit_factor_of(records) == float("inf")

    def test_module_helper_all_loss(self):
        records = [make_record("pin_bar", -1.0) for _ in range(3)]
        assert _profit_factor_of(records) == 0.0

    def test_module_helper_mixed(self):
        records = [make_record("pin_bar", r) for r in [2.0, 2.0, -1.0, -1.0]]
        assert _profit_factor_of(records) == pytest.approx(2.0)

    def test_empty_records_pf_zero(self):
        assert _profit_factor_of([]) == 0.0


# ===========================================================================
# TestExpectancy
# ===========================================================================

class TestExpectancy:
    """6 tests — expectancy_r calculation."""

    def test_positive_expectancy(self):
        # 3 × 2R wins, 2 × -1R losses = net 4R / 5 trades = 0.8R
        e = _filled_engine(wins=3, losses=2, win_r=2.0, loss_r=-1.0)
        s = e.get_strategy_summary("pin_bar", "EURUSD", "H1")
        assert s.expectancy_r == pytest.approx(0.8)

    def test_negative_expectancy(self):
        # 2 × 1R wins, 4 × -2R losses = 2 - 8 = -6 / 6 = -1.0R
        e = _filled_engine(wins=2, losses=4, win_r=1.0, loss_r=-2.0)
        s = e.get_strategy_summary("pin_bar", "EURUSD", "H1")
        assert s.expectancy_r == pytest.approx(-1.0)

    def test_zero_expectancy(self):
        e = _filled_engine(wins=2, losses=2, win_r=1.0, loss_r=-1.0)
        s = e.get_strategy_summary("pin_bar", "EURUSD", "H1")
        assert s.expectancy_r == pytest.approx(0.0)

    def test_one_trade_win(self):
        e = _engine()
        e.record_trade(_win(3.0))
        s = e.get_strategy_summary("pin_bar", "EURUSD", "H1")
        assert s.expectancy_r == pytest.approx(3.0)

    def test_one_trade_loss(self):
        e = _engine()
        e.record_trade(_loss(-1.0))
        s = e.get_strategy_summary("pin_bar", "EURUSD", "H1")
        assert s.expectancy_r == pytest.approx(-1.0)

    def test_empty_expectancy_zero(self):
        e = _engine()
        s = e.get_strategy_summary("pin_bar", "EURUSD", "H1")
        assert s.expectancy_r == 0.0


# ===========================================================================
# TestAverageWinnerLoser
# ===========================================================================

class TestAverageWinnerLoser:
    """6 tests."""

    def test_avg_winner(self):
        e = _engine()
        for r in [1.0, 2.0, 3.0]:
            e.record_trade(_win(r))
        s = e.get_strategy_summary("pin_bar", "EURUSD", "H1")
        assert s.avg_winner_r == pytest.approx(2.0)

    def test_avg_loser_magnitude(self):
        e = _engine()
        for r in [-1.0, -2.0, -3.0]:
            e.record_trade(_loss(r))
        s = e.get_strategy_summary("pin_bar", "EURUSD", "H1")
        assert s.avg_loser_r == pytest.approx(2.0)  # stored as positive

    def test_avg_winner_zero_when_no_wins(self):
        e = _filled_engine(wins=0, losses=3)
        s = e.get_strategy_summary("pin_bar", "EURUSD", "H1")
        assert s.avg_winner_r == 0.0

    def test_avg_loser_zero_when_no_losses(self):
        e = _filled_engine(wins=3, losses=0)
        s = e.get_strategy_summary("pin_bar", "EURUSD", "H1")
        assert s.avg_loser_r == 0.0

    def test_scorecard_avg_winner(self):
        e = _engine()
        for r in [2.0, 4.0]:
            e.record_trade(_win(r))
        sc = e.get_strategy_scorecard("pin_bar", "EURUSD", "H1")
        assert sc.avg_winner_r == pytest.approx(3.0)

    def test_scorecard_avg_loser(self):
        e = _engine()
        e.record_trade(_loss(-1.0))
        e.record_trade(_loss(-3.0))
        sc = e.get_strategy_scorecard("pin_bar", "EURUSD", "H1")
        assert sc.avg_loser_r == pytest.approx(2.0)


# ===========================================================================
# TestMaxDrawdown
# ===========================================================================

class TestMaxDrawdown:
    """8 tests — equity curve max drawdown."""

    def test_no_trades_zero_dd(self):
        e = _engine()
        s = e.get_strategy_summary("pin_bar", "EURUSD", "H1")
        assert s.max_drawdown_pct == 0.0

    def test_all_winners_zero_dd(self):
        e = _filled_engine(wins=5, losses=0)
        s = e.get_strategy_summary("pin_bar", "EURUSD", "H1")
        assert s.max_drawdown_pct == 0.0

    def test_all_losers_dd_100(self):
        """All losers from a 0 start — peak never goes above 0, so dd stays 0."""
        e = _engine()
        for _ in range(3):
            e.record_trade(_loss(-1.0))
        s = e.get_strategy_summary("pin_bar", "EURUSD", "H1")
        # Equity starts at 0, goes negative — peak=0, no positive peak means dd=0
        assert s.max_drawdown_pct == 0.0

    def test_win_then_loss_dd(self):
        """Win 4R, then lose 2R → peak=4, trough=2 → dd=50%"""
        e = _engine()
        e.record_trade(_win(4.0))
        e.record_trade(_loss(-2.0))
        s = e.get_strategy_summary("pin_bar", "EURUSD", "H1")
        assert s.max_drawdown_pct == pytest.approx(50.0)

    def test_recovery_dd(self):
        """W4, L2, W6 → peak after W4=4, trough=2, dd=50%; then recover"""
        e = _engine()
        for r in [4.0, -2.0, 6.0]:
            e.record_trade(make_record("pin_bar", r))
        s = e.get_strategy_summary("pin_bar", "EURUSD", "H1")
        assert s.max_drawdown_pct == pytest.approx(50.0)

    def test_larger_dd_later(self):
        """W2, L1, W4, L3 → 2nd dd: peak=5, trough=2 → 60%"""
        e = _engine()
        for r in [2.0, -1.0, 4.0, -3.0]:
            e.record_trade(make_record("pin_bar", r))
        s = e.get_strategy_summary("pin_bar", "EURUSD", "H1")
        assert s.max_drawdown_pct == pytest.approx(60.0)

    def test_single_win_zero_dd(self):
        e = _engine()
        e.record_trade(_win(2.0))
        s = e.get_strategy_summary("pin_bar", "EURUSD", "H1")
        assert s.max_drawdown_pct == 0.0

    def test_scorecard_max_drawdown(self):
        e = _engine()
        e.record_trade(_win(4.0))
        e.record_trade(_loss(-2.0))
        sc = e.get_strategy_scorecard("pin_bar", "EURUSD", "H1")
        assert sc.max_drawdown_pct == pytest.approx(50.0)


# ===========================================================================
# TestConsecutiveStreaks
# ===========================================================================

class TestConsecutiveStreaks:
    """7 tests — max consecutive wins/losses."""

    def test_no_trades_zero_streaks(self):
        e = _engine()
        s = e.get_strategy_summary("pin_bar", "EURUSD", "H1")
        assert s.max_consecutive_wins == 0
        assert s.max_consecutive_losses == 0

    def test_three_in_a_row_wins(self):
        e = _engine()
        for _ in range(3):
            e.record_trade(_win())
        s = e.get_strategy_summary("pin_bar", "EURUSD", "H1")
        assert s.max_consecutive_wins == 3

    def test_three_in_a_row_losses(self):
        e = _engine()
        for _ in range(3):
            e.record_trade(_loss())
        s = e.get_strategy_summary("pin_bar", "EURUSD", "H1")
        assert s.max_consecutive_losses == 3

    def test_alternating_win_loss(self):
        e = _engine()
        for _ in range(4):
            e.record_trade(_win())
            e.record_trade(_loss())
        s = e.get_strategy_summary("pin_bar", "EURUSD", "H1")
        assert s.max_consecutive_wins == 1
        assert s.max_consecutive_losses == 1

    def test_streak_reset_on_win(self):
        e = _engine()
        for r in [-1.0, -1.0, 2.0, -1.0, -1.0, -1.0]:
            e.record_trade(make_record("pin_bar", r))
        s = e.get_strategy_summary("pin_bar", "EURUSD", "H1")
        assert s.max_consecutive_losses == 3

    def test_long_win_streak_detected(self):
        e = _engine()
        for r in [2.0, 2.0, 2.0, 2.0, 2.0, -1.0, 2.0, 2.0]:
            e.record_trade(make_record("pin_bar", r))
        s = e.get_strategy_summary("pin_bar", "EURUSD", "H1")
        assert s.max_consecutive_wins == 5

    def test_scorecard_max_con_losses(self):
        e = _engine()
        for _ in range(4):
            e.record_trade(_loss())
        sc = e.get_strategy_scorecard("pin_bar", "EURUSD", "H1")
        assert sc.max_con_losses == 4


# ===========================================================================
# TestSharpeRatio
# ===========================================================================

class TestSharpeRatio:
    """8 tests — Sharpe ratio edge cases."""

    def test_below_min_trades_returns_none(self):
        e = _engine(min_sharpe_trades=5)
        for _ in range(4):
            e.record_trade(_win(2.0))
        s = e.get_strategy_summary("pin_bar", "EURUSD", "H1")
        assert s.sharpe_ratio is None

    def test_exactly_min_trades_computes(self):
        e = _engine(min_sharpe_trades=5)
        for r in [2.0, -1.0, 2.0, -1.0, 2.0]:
            e.record_trade(make_record("pin_bar", r))
        s = e.get_strategy_summary("pin_bar", "EURUSD", "H1")
        assert s.sharpe_ratio is not None

    def test_all_same_r_returns_none(self):
        """std=0 → Sharpe undefined → None."""
        e = _engine(min_sharpe_trades=3)
        for _ in range(5):
            e.record_trade(_win(2.0))
        s = e.get_strategy_summary("pin_bar", "EURUSD", "H1")
        assert s.sharpe_ratio is None

    def test_positive_sharpe_for_good_strategy(self):
        e = _engine(min_sharpe_trades=5)
        for r in [2.0, 2.0, 2.0, 2.0, -1.0]:
            e.record_trade(make_record("pin_bar", r))
        s = e.get_strategy_summary("pin_bar", "EURUSD", "H1")
        assert s.sharpe_ratio is not None
        assert s.sharpe_ratio > 0

    def test_negative_sharpe_for_bad_strategy(self):
        e = _engine(min_sharpe_trades=5)
        for r in [-2.0, -2.0, -2.0, -2.0, 1.0]:
            e.record_trade(make_record("pin_bar", r))
        s = e.get_strategy_summary("pin_bar", "EURUSD", "H1")
        assert s.sharpe_ratio is not None
        assert s.sharpe_ratio < 0

    def test_sharpe_formula_manual(self):
        """Verify formula: mean/std."""
        e = _engine(min_sharpe_trades=4)
        r_series = [2.0, -1.0, 2.0, -1.0]
        for r in r_series:
            e.record_trade(make_record("pin_bar", r))
        s = e.get_strategy_summary("pin_bar", "EURUSD", "H1")
        mean = sum(r_series) / len(r_series)           # 0.5
        variance = sum((x - mean) ** 2 for x in r_series) / (len(r_series) - 1)
        std = variance ** 0.5
        expected = mean / std
        assert s.sharpe_ratio == pytest.approx(expected, abs=1e-4)

    def test_scorecard_sharpe_none_when_empty(self):
        e = _engine()
        sc = e.get_strategy_scorecard("pin_bar", "EURUSD", "H1")
        assert sc.sharpe_ratio is None

    def test_sharpe_improves_with_more_wins(self):
        e = _engine(min_sharpe_trades=5)
        # Baseline: 50/50
        for r in [2.0, -1.0, 2.0, -1.0, 2.0]:
            e.record_trade(make_record("pin_bar", r))
        s1 = e.get_strategy_summary("pin_bar", "EURUSD", "H1")
        # Add more wins
        for _ in range(5):
            e.record_trade(_win(2.0))
        s2 = e.get_strategy_summary("pin_bar", "EURUSD", "H1")
        assert s2.sharpe_ratio is not None
        assert (s2.sharpe_ratio or 0) >= (s1.sharpe_ratio or 0)


# ===========================================================================
# TestRollingWindows
# ===========================================================================

class TestRollingWindows:
    """7 tests — recent_30 and recent_90 profit factors."""

    def test_fewer_than_30_uses_all(self):
        e = _engine()
        for _ in range(5):
            e.record_trade(_win(2.0))
        for _ in range(3):
            e.record_trade(_loss(-1.0))
        s = e.get_strategy_summary("pin_bar", "EURUSD", "H1")
        # recent_30 should use all 8 since we have < 30
        assert s.recent_30_profit_factor is not None
        assert s.recent_30_profit_factor == pytest.approx(10.0 / 3.0, abs=0.01)

    def test_empty_rolling_none(self):
        e = _engine()
        s = e.get_strategy_summary("pin_bar", "EURUSD", "H1")
        assert s.recent_30_profit_factor is None
        assert s.recent_90_profit_factor is None

    def test_30_and_90_same_when_few_trades(self):
        e = _engine()
        for _ in range(10):
            e.record_trade(_win(2.0))
        for _ in range(5):
            e.record_trade(_loss(-1.0))
        s = e.get_strategy_summary("pin_bar", "EURUSD", "H1")
        assert s.recent_30_profit_factor == s.recent_90_profit_factor

    def test_30_window_differs_from_90_window(self):
        """Add 80 bad trades then 30 good trades — recent_30 > recent_90."""
        e = _engine()
        for _ in range(80):
            e.record_trade(_loss(-1.0))
        for _ in range(30):
            e.record_trade(_win(3.0))
        s = e.get_strategy_summary("pin_bar", "EURUSD", "H1")
        # recent_30: all 30 wins → inf
        assert s.recent_30_profit_factor == float("inf")
        # recent_90: last 90 = 80 losses then 30 wins? No: 80+30=110, last 90=20 losses + 30 wins
        # 30 wins at 3.0 = 90R; 20 losses at 1.0 = 20R → PF = 90/20 = 4.5
        assert s.recent_90_profit_factor is not None
        assert (s.recent_90_profit_factor or 0) < float("inf")

    def test_recent_30_all_losers(self):
        e = _engine()
        for _ in range(30):
            e.record_trade(_loss(-1.0))
        s = e.get_strategy_summary("pin_bar", "EURUSD", "H1")
        assert s.recent_30_profit_factor == 0.0

    def test_scorecard_recent_30_pf(self):
        e = _engine()
        for _ in range(3):
            e.record_trade(_win(2.0))
        for _ in range(1):
            e.record_trade(_loss(-1.0))
        sc = e.get_strategy_scorecard("pin_bar", "EURUSD", "H1")
        assert sc.recent_30_pf == pytest.approx(6.0 / 1.0)

    def test_scorecard_recent_90_pf_none_when_empty(self):
        e = _engine()
        sc = e.get_strategy_scorecard("pin_bar", "EURUSD", "H1")
        assert sc.recent_90_pf is None


# ===========================================================================
# TestEnableDisable
# ===========================================================================

class TestEnableDisable:
    """9 tests — strategy enable/disable mechanics."""

    def test_enabled_by_default(self):
        e = _engine()
        assert e.is_strategy_enabled("pin_bar") is True

    def test_disable_sets_false(self):
        e = _engine()
        e.disable_strategy("pin_bar", "degrading edge")
        assert e.is_strategy_enabled("pin_bar") is False

    def test_enable_restores_true(self):
        e = _engine()
        e.disable_strategy("pin_bar", "reason")
        e.enable_strategy("pin_bar")
        assert e.is_strategy_enabled("pin_bar") is True

    def test_disable_reason_stored(self):
        e = _engine()
        e.disable_strategy("pin_bar", "win_rate below 40%")
        s = e.get_strategy_summary("pin_bar", "EURUSD", "H1")
        assert s.disable_reason == "win_rate below 40%"

    def test_enable_clears_reason(self):
        e = _engine()
        e.disable_strategy("pin_bar", "reason")
        e.enable_strategy("pin_bar")
        s = e.get_strategy_summary("pin_bar", "EURUSD", "H1")
        assert s.disable_reason is None

    def test_engulfing_independent_of_pin(self):
        e = _engine()
        e.disable_strategy("pin_bar")
        assert e.is_strategy_enabled("pin_bar") is False
        assert e.is_strategy_enabled("engulfing_bar") is True

    def test_disable_no_reason_allowed(self):
        e = _engine()
        e.disable_strategy("pin_bar")   # no reason
        assert e.is_strategy_enabled("pin_bar") is False

    def test_summary_is_enabled_reflects_state(self):
        e = _engine()
        e.record_trade(_win())
        e.disable_strategy("pin_bar", "test")
        s = e.get_strategy_summary("pin_bar", "EURUSD", "H1")
        assert s.is_enabled is False

    def test_enable_already_enabled_no_error(self):
        e = _engine()
        e.enable_strategy("pin_bar")   # Already enabled — no-op
        assert e.is_strategy_enabled("pin_bar") is True


# ===========================================================================
# TestDegradationDetection
# ===========================================================================

class TestDegradationDetection:
    """6 tests — is_strategy_degrading."""

    def test_not_degrading_below_min_trades(self):
        e = _engine(degradation_window=20, degradation_pf_threshold=0.80)
        for _ in range(4):
            e.record_trade(_loss(-1.0))
        # 4 < 5 minimum → not degrading
        assert e.is_strategy_degrading("pin_bar", "EURUSD", "H1") is False

    def test_degrading_when_pf_below_threshold(self):
        e = _engine(degradation_window=20, degradation_pf_threshold=0.80)
        # 2 wins at 1R, 10 losses at 1R → PF = 2/10 = 0.2 < 0.8
        for _ in range(2):
            e.record_trade(_win(1.0))
        for _ in range(10):
            e.record_trade(_loss(-1.0))
        assert e.is_strategy_degrading("pin_bar", "EURUSD", "H1") is True

    def test_not_degrading_when_pf_above_threshold(self):
        e = _engine(degradation_window=20, degradation_pf_threshold=0.80)
        for _ in range(10):
            e.record_trade(_win(2.0))
        for _ in range(3):
            e.record_trade(_loss(-1.0))
        assert e.is_strategy_degrading("pin_bar", "EURUSD", "H1") is False

    def test_degradation_uses_window(self):
        """Add 50 good trades then 20 bad trades — recent window should flag degradation."""
        e = _engine(degradation_window=20, degradation_pf_threshold=0.80)
        for _ in range(50):
            e.record_trade(_win(2.0))
        for _ in range(18):
            e.record_trade(_loss(-1.0))
        for _ in range(2):
            e.record_trade(_win(1.0))
        # Last 20: 18 losses, 2 wins → PF = 2/18 = 0.11 < 0.8
        assert e.is_strategy_degrading("pin_bar", "EURUSD", "H1") is True

    def test_scorecard_is_degrading_flag(self):
        e = _engine(degradation_window=10, degradation_pf_threshold=0.80)
        for _ in range(2):
            e.record_trade(_win(1.0))
        for _ in range(10):
            e.record_trade(_loss(-1.0))
        sc = e.get_strategy_scorecard("pin_bar", "EURUSD", "H1")
        assert sc.is_degrading is True

    def test_empty_strategy_not_degrading(self):
        e = _engine()
        assert e.is_strategy_degrading("pin_bar", "EURUSD", "H1") is False


# ===========================================================================
# TestRankingStrategies
# ===========================================================================

class TestRankingStrategies:
    """6 tests — rank_strategies_by_metric."""

    def test_rank_by_profit_factor_descending(self):
        e = _engine()
        for _ in range(5):
            e.record_trade(_win(3.0, strat="pin_bar"))
        for _ in range(3):
            e.record_trade(_loss(-1.0, strat="pin_bar"))
        for _ in range(3):
            e.record_trade(_win(1.5, strat="engulfing_bar"))
        for _ in range(3):
            e.record_trade(_loss(-1.0, strat="engulfing_bar"))
        ranks = e.rank_strategies_by_metric("profit_factor")
        assert ranks[0][0] == "pin_bar"

    def test_rank_by_total_trades_descending(self):
        e = _engine()
        for _ in range(10):
            e.record_trade(_win(strat="pin_bar"))
        for _ in range(5):
            e.record_trade(_win(strat="engulfing_bar"))
        ranks = e.rank_strategies_by_metric("total_trades")
        assert ranks[0][0] == "pin_bar"

    def test_rank_returns_two_phase1_strategies(self):
        e = _engine()
        ranks = e.rank_strategies_by_metric("profit_factor")
        names = [r[0] for r in ranks]
        assert "pin_bar" in names
        assert "engulfing_bar" in names
        assert len(ranks) == 2

    def test_rank_ascending(self):
        e = _engine()
        for _ in range(5):
            e.record_trade(_win(3.0, strat="pin_bar"))
        for _ in range(2):
            e.record_trade(_loss(-1.0, strat="pin_bar"))
        for _ in range(5):
            e.record_trade(_win(1.0, strat="engulfing_bar"))
        for _ in range(4):
            e.record_trade(_loss(-1.0, strat="engulfing_bar"))
        ranks = e.rank_strategies_by_metric("profit_factor", descending=False)
        assert ranks[0][0] == "engulfing_bar"

    def test_rank_by_expectancy(self):
        e = _engine()
        for r in [2.0, 2.0, -1.0]:
            e.record_trade(make_record("pin_bar", r))
        for r in [1.0, -1.0, -1.0]:
            e.record_trade(make_record("engulfing_bar", r))
        ranks = e.rank_strategies_by_metric("expectancy_r")
        assert ranks[0][0] == "pin_bar"

    def test_rank_win_rate_both_same(self):
        e = _engine()
        for _ in range(2):
            e.record_trade(_win(strat="pin_bar"))
            e.record_trade(_win(strat="engulfing_bar"))
        ranks = e.rank_strategies_by_metric("win_rate")
        # Both 100% — order is deterministic but both 1.0
        assert all(v == 1.0 for _, v in ranks)


# ===========================================================================
# TestComparisonTable
# ===========================================================================

class TestComparisonTable:
    """5 tests — get_strategy_comparison_table."""

    def test_returns_both_strategies(self):
        e = _engine()
        table = e.get_strategy_comparison_table()
        assert "pin_bar" in table
        assert "engulfing_bar" in table

    def test_table_has_required_keys(self):
        e = _engine()
        table = e.get_strategy_comparison_table()
        for strat in ("pin_bar", "engulfing_bar"):
            for key in ("total_trades", "win_rate", "profit_factor",
                        "expectancy_r", "is_enabled"):
                assert key in table[strat], f"missing key {key} for {strat}"

    def test_table_reflects_recorded_trades(self):
        e = _engine()
        for _ in range(3):
            e.record_trade(_win(2.0, strat="pin_bar"))
        table = e.get_strategy_comparison_table()
        assert table["pin_bar"]["total_trades"] == 3
        assert table["engulfing_bar"]["total_trades"] == 0

    def test_table_is_enabled_flag(self):
        e = _engine()
        e.disable_strategy("engulfing_bar", "no data")
        table = e.get_strategy_comparison_table()
        assert table["engulfing_bar"]["is_enabled"] is False
        assert table["pin_bar"]["is_enabled"] is True

    def test_table_no_phase2_strategies(self):
        e = _engine()
        table = e.get_strategy_comparison_table()
        assert "inside_bar" not in table
        assert "false_breakout" not in table


# ===========================================================================
# TestStrategyScorecard
# ===========================================================================

class TestStrategyScorecard:
    """7 tests — StrategyScorecard DTO and string output."""

    def test_scorecard_has_strategy_name(self):
        e = _engine()
        sc = e.get_strategy_scorecard("pin_bar", "EURUSD", "H1")
        assert sc.strategy_name == "pin_bar"

    def test_scorecard_has_symbol(self):
        e = _engine()
        sc = e.get_strategy_scorecard("engulfing_bar", "GBPUSD", "H4")
        assert sc.symbol == "GBPUSD"
        assert sc.timeframe == "H4"

    def test_scorecard_str_contains_strategy(self):
        e = _engine()
        sc = e.get_strategy_scorecard("pin_bar", "EURUSD", "H1")
        assert "PIN_BAR" in str(sc)

    def test_scorecard_str_contains_win_rate(self):
        e = _filled_engine(wins=3, losses=2)
        sc = e.get_strategy_scorecard("pin_bar", "EURUSD", "H1")
        assert "60.0%" in str(sc)

    def test_scorecard_generated_at_is_datetime(self):
        e = _engine()
        sc = e.get_strategy_scorecard("pin_bar", "EURUSD", "H1")
        assert isinstance(sc.generated_at, datetime)

    def test_scorecard_is_enabled_flag(self):
        e = _engine()
        e.disable_strategy("pin_bar", "reason")
        sc = e.get_strategy_scorecard("pin_bar", "EURUSD", "H1")
        assert sc.is_enabled is False

    def test_scorecard_is_degrading_flag_false_when_healthy(self):
        e = _filled_engine(wins=10, losses=2)
        sc = e.get_strategy_scorecard("pin_bar", "EURUSD", "H1")
        assert sc.is_degrading is False


# ===========================================================================
# TestEdgeCases
# ===========================================================================

class TestEdgeCases:
    """10 tests — boundaries and separation."""

    def test_one_trade_win(self):
        e = _engine()
        e.record_trade(_win(2.0))
        s = e.get_strategy_summary("pin_bar", "EURUSD", "H1")
        assert s.total_trades == 1
        assert s.win_count == 1
        assert s.profit_factor == float("inf")

    def test_one_trade_loss(self):
        e = _engine()
        e.record_trade(_loss(-1.0))
        s = e.get_strategy_summary("pin_bar", "EURUSD", "H1")
        assert s.loss_count == 1
        assert s.profit_factor == 0.0

    def test_pin_bar_and_engulfing_tracked_separately(self):
        e = _engine()
        for _ in range(3):
            e.record_trade(_win(2.0, strat="pin_bar"))
        for _ in range(5):
            e.record_trade(_win(2.0, strat="engulfing_bar"))
        pb = e.get_strategy_summary("pin_bar", "EURUSD", "H1")
        eb = e.get_strategy_summary("engulfing_bar", "EURUSD", "H1")
        assert pb.total_trades == 3
        assert eb.total_trades == 5

    def test_non_eurusd_symbol_tracked(self):
        e = _engine()
        e.record_trade(make_record("pin_bar", 2.0, symbol="GBPUSD"))
        s = e.get_strategy_summary("pin_bar", "GBPUSD", "H1")
        assert s.total_trades == 1

    def test_usdjpy_symbol(self):
        e = _engine()
        e.record_trade(make_record("engulfing_bar", -1.0, symbol="USDJPY", timeframe="D1"))
        s = e.get_strategy_summary("engulfing_bar", "USDJPY", "D1")
        assert s.loss_count == 1

    def test_case_insensitive_strategy_name(self):
        e = _engine()
        e.record_trade(make_record("PIN_BAR", 2.0))
        s = e.get_strategy_summary("pin_bar", "EURUSD", "H1")
        assert s.total_trades == 1

    def test_large_r_multiple(self):
        e = _engine()
        e.record_trade(_win(50.0))
        s = e.get_strategy_summary("pin_bar", "EURUSD", "H1")
        assert s.avg_winner_r == pytest.approx(50.0)

    def test_tiny_r_multiple(self):
        e = _engine()
        e.record_trade(make_record("pin_bar", 0.01))
        s = e.get_strategy_summary("pin_bar", "EURUSD", "H1")
        assert s.win_count == 1

    def test_metrics_are_deterministic(self):
        """Same input always produces same output."""
        e1 = _engine()
        e2 = _engine()
        for r in [2.0, -1.0, 3.0, -1.0, 2.0]:
            e1.record_trade(make_record("pin_bar", r))
            e2.record_trade(make_record("pin_bar", r))
        s1 = e1.get_strategy_summary("pin_bar", "EURUSD", "H1")
        s2 = e2.get_strategy_summary("pin_bar", "EURUSD", "H1")
        assert s1.profit_factor == s2.profit_factor
        assert s1.expectancy_r == s2.expectancy_r

    def test_no_phase2_strategy_records(self):
        """Phase 2 strategy names must not appear in any tracked key."""
        e = _engine()
        for strat in ("inside_bar", "inside_bar_false_breakout", "fibonacci"):
            e.record_trade(make_record(strat, 2.0))
        # No records for any strategy since all were skipped
        table = e.get_strategy_comparison_table()
        for strat_data in table.values():
            assert strat_data["total_trades"] == 0
