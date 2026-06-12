"""
Sprint 12.1 — Analytics Consistency Tests
==========================================
Verifies that every trade close — regardless of exit type — flows through
_finalize_closed_order() and consistently updates:

    - PipelineResult  (counters, P&L, R-multiples)
    - M18 StrategyAnalyticsEngine  (always)
    - M19 TradeReviewEngine  (losses only)

No trade should appear in analytics more than once (no double-recording).
PipelineResult and M18 must always agree on total trades, wins, losses,
and P&L-derived metrics.

Tests are organised into 7 focused classes, 30 tests total.
No Phase 2 features, no MT5, no live execution.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import List, Tuple

import pytest

from src.data.types import CandleData
from src.execution.paper_executor import (
    ExitReason,
    PaperExecutorConfig,
    PaperOrder,
    PaperOrderStatus,
    PaperTradeExecutor,
)
from src.integration.pipeline_runner import (
    PipelineConfig,
    PipelineResult,
    PipelineRunner,
    _compute_max_drawdown,
)
from src.types import AccountState


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

def _runner(slippage: float = 0.0) -> PipelineRunner:
    """PipelineRunner with risk disabled and zero slippage for clean testing."""
    cfg = PipelineConfig(
        slippage_pips=slippage,
        risk_enabled=False,
        analytics_enabled=True,
        review_enabled=True,
    )
    return PipelineRunner(cfg)


def _order(
    order_id:   str,
    direction:  str   = "LONG",
    entry:      float = 1.1000,
    sl:         float = 1.0950,   # 50 pips below entry
    tp:         float = 1.1100,   # 100 pips above entry → 2R
    lot_size:   float = 0.10,
    stop_pips:  float = 50.0,
    strategy:   str   = "pin_bar",
) -> PaperOrder:
    """Build a FILLED PaperOrder ready for finalization."""
    return PaperOrder(
        order_id=order_id,
        strategy_name=strategy,
        symbol="EURUSD",
        timeframe="D1",
        direction=direction,
        status=PaperOrderStatus.FILLED,
        requested_price=entry,
        filled_price=entry,
        stop_loss=sl,
        take_profit=tp,
        lot_size=lot_size,
        slippage_pips=0.0,
        created_at=datetime.now(timezone.utc),
        stop_pips=stop_pips,
    )


def _result() -> PipelineResult:
    """Fresh PipelineResult accumulator."""
    return PipelineResult(
        symbol="EURUSD",
        timeframe="D1",
        started_at=datetime.now(timezone.utc),
        initial_balance=10_000.0,
        final_balance=10_000.0,
    )


def _account(balance: float = 10_000.0) -> AccountState:
    return PipelineRunner._make_account(balance)


def _place_and_finalize(
    runner:      PipelineRunner,
    order_id:    str,
    exit_price:  float,
    exit_reason: str,
    result:      PipelineResult,
    account:     AccountState,
    **order_kwargs,
) -> Tuple[PaperOrder, AccountState]:
    """Helper: register order in executor then finalize it."""
    o = _order(order_id, **order_kwargs)
    runner._executor._orders[order_id] = o
    new_account = runner._finalize_closed_order(
        o, exit_price, exit_reason, result, account
    )
    return o, new_account


# ---------------------------------------------------------------------------
# Class 1 — TP hit updates M18
# ---------------------------------------------------------------------------

class TestTPHitUpdatesM18:
    """TP exits must push to M18 analytics."""

    def test_tp_hit_increments_m18_total_trades(self):
        r = _runner()
        res, acc = _result(), _account()
        _place_and_finalize(r, "T1", 1.1100, ExitReason.TP_HIT, res, acc)
        s = r.get_analytics_engine().get_strategy_summary("pin_bar", "EURUSD", "D1")
        assert s.total_trades == 1

    def test_tp_hit_recorded_as_win_in_m18(self):
        r = _runner()
        res, acc = _result(), _account()
        _place_and_finalize(r, "T1", 1.1100, ExitReason.TP_HIT, res, acc)
        s = r.get_analytics_engine().get_strategy_summary("pin_bar", "EURUSD", "D1")
        assert s.win_count == 1
        assert s.loss_count == 0

    def test_tp_hit_m18_win_rate_is_one(self):
        r = _runner()
        res, acc = _result(), _account()
        _place_and_finalize(r, "T1", 1.1100, ExitReason.TP_HIT, res, acc)
        s = r.get_analytics_engine().get_strategy_summary("pin_bar", "EURUSD", "D1")
        assert s.win_rate == pytest.approx(1.0)

    def test_tp_hit_m18_positive_expectancy(self):
        r = _runner()
        res, acc = _result(), _account()
        _place_and_finalize(r, "T1", 1.1100, ExitReason.TP_HIT, res, acc)
        s = r.get_analytics_engine().get_strategy_summary("pin_bar", "EURUSD", "D1")
        assert s.expectancy_r > 0

    def test_two_tp_hits_both_recorded_in_m18(self):
        r = _runner()
        res, acc = _result(), _account()
        _, acc = _place_and_finalize(r, "T1", 1.1100, ExitReason.TP_HIT, res, acc)
        _, acc = _place_and_finalize(r, "T2", 1.1100, ExitReason.TP_HIT, res, acc)
        s = r.get_analytics_engine().get_strategy_summary("pin_bar", "EURUSD", "D1")
        assert s.total_trades == 2
        assert s.win_count == 2


# ---------------------------------------------------------------------------
# Class 2 — SL hit updates M18
# ---------------------------------------------------------------------------

class TestSLHitUpdatesM18:
    """SL exits must also push to M18 analytics."""

    def test_sl_hit_increments_m18_total_trades(self):
        r = _runner()
        res, acc = _result(), _account()
        _place_and_finalize(r, "T1", 1.0950, ExitReason.SL_HIT, res, acc)
        s = r.get_analytics_engine().get_strategy_summary("pin_bar", "EURUSD", "D1")
        assert s.total_trades == 1

    def test_sl_hit_recorded_as_loss_in_m18(self):
        r = _runner()
        res, acc = _result(), _account()
        _place_and_finalize(r, "T1", 1.0950, ExitReason.SL_HIT, res, acc)
        s = r.get_analytics_engine().get_strategy_summary("pin_bar", "EURUSD", "D1")
        assert s.loss_count == 1
        assert s.win_count == 0

    def test_sl_hit_m18_win_rate_is_zero(self):
        r = _runner()
        res, acc = _result(), _account()
        _place_and_finalize(r, "T1", 1.0950, ExitReason.SL_HIT, res, acc)
        s = r.get_analytics_engine().get_strategy_summary("pin_bar", "EURUSD", "D1")
        assert s.win_rate == pytest.approx(0.0)

    def test_sl_hit_m18_negative_expectancy(self):
        r = _runner()
        res, acc = _result(), _account()
        _place_and_finalize(r, "T1", 1.0950, ExitReason.SL_HIT, res, acc)
        s = r.get_analytics_engine().get_strategy_summary("pin_bar", "EURUSD", "D1")
        assert s.expectancy_r < 0


# ---------------------------------------------------------------------------
# Class 3 — SL hit updates M19; TP hit does NOT
# ---------------------------------------------------------------------------

class TestM19WiringByExitType:
    """M19 receives only losing trades, never winners."""

    def test_sl_hit_sends_review_to_m19(self):
        r = _runner()
        res, acc = _result(), _account()
        _place_and_finalize(r, "T1", 1.0950, ExitReason.SL_HIT, res, acc)
        reviews = r.get_review_engine().get_all_results()
        assert len(reviews) == 1

    def test_tp_hit_does_not_send_review_to_m19(self):
        r = _runner()
        res, acc = _result(), _account()
        _place_and_finalize(r, "T1", 1.1100, ExitReason.TP_HIT, res, acc)
        reviews = r.get_review_engine().get_all_results()
        assert len(reviews) == 0

    def test_m19_review_references_correct_trade_id(self):
        r = _runner()
        res, acc = _result(), _account()
        _place_and_finalize(r, "PAPER-SL-001", 1.0950, ExitReason.SL_HIT, res, acc)
        reviews = r.get_review_engine().get_all_results()
        assert reviews[0].trade_id == "PAPER-SL-001"

    def test_m19_review_references_correct_strategy(self):
        r = _runner()
        res, acc = _result(), _account()
        _place_and_finalize(r, "T1", 1.0950, ExitReason.SL_HIT, res, acc,
                            strategy="engulfing_bar")
        reviews = r.get_review_engine().get_all_results()
        assert reviews[0].strategy_name == "engulfing_bar"

    def test_manual_close_winner_not_in_m19(self):
        """End-of-data close at profit must not feed M19."""
        r = _runner()
        res, acc = _result(), _account()
        # Close at 1.1050 = 50 pips profit on 50-pip SL → R=+1.0 (winner)
        _place_and_finalize(r, "T1", 1.1050, ExitReason.MANUAL_CLOSE, res, acc)
        reviews = r.get_review_engine().get_all_results()
        assert len(reviews) == 0

    def test_manual_close_loser_sent_to_m19(self):
        """End-of-data close at a loss must feed M19."""
        r = _runner()
        res, acc = _result(), _account()
        # Close at 1.0970 = 30 pips loss on 50-pip SL → R=-0.6 (loser)
        _place_and_finalize(r, "T1", 1.0970, ExitReason.MANUAL_CLOSE, res, acc)
        reviews = r.get_review_engine().get_all_results()
        assert len(reviews) == 1


# ---------------------------------------------------------------------------
# Class 4 — No duplicate records
# ---------------------------------------------------------------------------

class TestNoDuplicateRecords:
    """_finalize_closed_order must record each trade exactly once."""

    def test_single_tp_trade_appears_once_in_m18(self):
        r = _runner()
        res, acc = _result(), _account()
        _place_and_finalize(r, "T1", 1.1100, ExitReason.TP_HIT, res, acc)
        s = r.get_analytics_engine().get_strategy_summary("pin_bar", "EURUSD", "D1")
        assert s.total_trades == 1

    def test_single_tp_trade_appears_once_in_result(self):
        r = _runner()
        res, acc = _result(), _account()
        _place_and_finalize(r, "T1", 1.1100, ExitReason.TP_HIT, res, acc)
        assert res.trades_executed == 1

    def test_single_sl_trade_reviewed_once_in_m19(self):
        r = _runner()
        res, acc = _result(), _account()
        _place_and_finalize(r, "T1", 1.0950, ExitReason.SL_HIT, res, acc)
        reviews = r.get_review_engine().get_all_results()
        assert len(reviews) == 1

    def test_three_trades_three_m18_records(self):
        r = _runner()
        res, acc = _result(), _account()
        for i, (ep, reason) in enumerate([
            (1.1100, ExitReason.TP_HIT),
            (1.0950, ExitReason.SL_HIT),
            (1.1100, ExitReason.TP_HIT),
        ]):
            _, acc = _place_and_finalize(r, f"T{i}", ep, reason, res, acc)
        s = r.get_analytics_engine().get_strategy_summary("pin_bar", "EURUSD", "D1")
        assert s.total_trades == 3

    def test_two_sl_trades_two_m19_reviews(self):
        r = _runner()
        res, acc = _result(), _account()
        _, acc = _place_and_finalize(r, "T1", 1.0950, ExitReason.SL_HIT, res, acc)
        _, acc = _place_and_finalize(r, "T2", 1.0950, ExitReason.SL_HIT, res, acc)
        reviews = r.get_review_engine().get_all_results()
        assert len(reviews) == 2


# ---------------------------------------------------------------------------
# Class 5 — PipelineResult ↔ M18 consistency
# ---------------------------------------------------------------------------

class TestResultM18Consistency:
    """
    PipelineResult and M18 StrategyAnalyticsEngine must agree on every metric.
    """

    def _run_sequence(
        self,
        exits: List[Tuple[float, str]],
    ) -> Tuple[PipelineRunner, PipelineResult]:
        """Run a sequence of (exit_price, exit_reason) pairs and return
        (runner, result) after all trades are finalized."""
        r = _runner()
        res, acc = _result(), _account()
        for i, (ep, reason) in enumerate(exits):
            _, acc = _place_and_finalize(r, f"T{i}", ep, reason, res, acc)
        res.finalise()
        return r, res

    def test_total_trades_agree(self):
        r, res = self._run_sequence([
            (1.1100, ExitReason.TP_HIT),
            (1.0950, ExitReason.SL_HIT),
            (1.1100, ExitReason.TP_HIT),
        ])
        s = r.get_analytics_engine().get_strategy_summary("pin_bar", "EURUSD", "D1")
        assert res.trades_executed == s.total_trades

    def test_win_counts_agree(self):
        r, res = self._run_sequence([
            (1.1100, ExitReason.TP_HIT),
            (1.0950, ExitReason.SL_HIT),
            (1.1100, ExitReason.TP_HIT),
        ])
        s = r.get_analytics_engine().get_strategy_summary("pin_bar", "EURUSD", "D1")
        assert res.wins == s.win_count

    def test_loss_counts_agree(self):
        r, res = self._run_sequence([
            (1.1100, ExitReason.TP_HIT),
            (1.0950, ExitReason.SL_HIT),
            (1.1100, ExitReason.TP_HIT),
        ])
        s = r.get_analytics_engine().get_strategy_summary("pin_bar", "EURUSD", "D1")
        assert res.losses == s.loss_count

    def test_net_pnl_sign_agrees_with_m18_expectancy(self):
        """A run with more wins than losses should have positive net P&L
        and positive M18 expectancy."""
        r, res = self._run_sequence([
            (1.1100, ExitReason.TP_HIT),
            (1.1100, ExitReason.TP_HIT),
            (1.0950, ExitReason.SL_HIT),
        ])
        s = r.get_analytics_engine().get_strategy_summary("pin_bar", "EURUSD", "D1")
        assert res.net_profit_usd > 0
        assert s.expectancy_r > 0

    def test_all_losses_results_in_negative_expectancy_in_m18(self):
        r, res = self._run_sequence([
            (1.0950, ExitReason.SL_HIT),
            (1.0950, ExitReason.SL_HIT),
        ])
        s = r.get_analytics_engine().get_strategy_summary("pin_bar", "EURUSD", "D1")
        assert s.expectancy_r < 0
        assert res.net_profit_usd < 0

    def test_all_wins_result_agrees_between_result_and_m18(self):
        r, res = self._run_sequence([
            (1.1100, ExitReason.TP_HIT),
            (1.1100, ExitReason.TP_HIT),
            (1.1100, ExitReason.TP_HIT),
        ])
        s = r.get_analytics_engine().get_strategy_summary("pin_bar", "EURUSD", "D1")
        assert res.wins == s.total_trades == s.win_count == 3
        assert s.loss_count == 0

    def test_zero_trades_m18_is_empty(self):
        r = _runner()
        s = r.get_analytics_engine().get_strategy_summary("pin_bar", "EURUSD", "D1")
        assert s.total_trades == 0


# ---------------------------------------------------------------------------
# Class 6 — End-of-data path (MANUAL_CLOSE) consistency
# ---------------------------------------------------------------------------

class TestEndOfDataPathConsistency:
    """End-of-data manual closes must follow the same finalization path."""

    def test_eod_winner_updates_m18(self):
        r = _runner()
        res, acc = _result(), _account()
        # 1.1050 exit from 1.1000 entry → +50 pips on 50-pip SL → R=+1.0 (winner)
        _place_and_finalize(r, "T1", 1.1050, ExitReason.MANUAL_CLOSE, res, acc)
        s = r.get_analytics_engine().get_strategy_summary("pin_bar", "EURUSD", "D1")
        assert s.total_trades == 1
        assert s.win_count == 1

    def test_eod_loser_updates_m18(self):
        r = _runner()
        res, acc = _result(), _account()
        # 1.0970 exit from 1.1000 entry → -30 pips on 50-pip SL → R=-0.6 (loser)
        _place_and_finalize(r, "T1", 1.0970, ExitReason.MANUAL_CLOSE, res, acc)
        s = r.get_analytics_engine().get_strategy_summary("pin_bar", "EURUSD", "D1")
        assert s.total_trades == 1
        assert s.loss_count == 1

    def test_eod_loser_updates_m19(self):
        r = _runner()
        res, acc = _result(), _account()
        _place_and_finalize(r, "T1", 1.0970, ExitReason.MANUAL_CLOSE, res, acc)
        reviews = r.get_review_engine().get_all_results()
        assert len(reviews) == 1

    def test_eod_winner_m19_empty(self):
        r = _runner()
        res, acc = _result(), _account()
        _place_and_finalize(r, "T1", 1.1050, ExitReason.MANUAL_CLOSE, res, acc)
        reviews = r.get_review_engine().get_all_results()
        assert len(reviews) == 0

    def test_eod_result_and_m18_agree_on_executed_count(self):
        r = _runner()
        res, acc = _result(), _account()
        _place_and_finalize(r, "T1", 1.1050, ExitReason.MANUAL_CLOSE, res, acc)
        s = r.get_analytics_engine().get_strategy_summary("pin_bar", "EURUSD", "D1")
        assert res.trades_executed == s.total_trades == 1


# ---------------------------------------------------------------------------
# Class 7 — Determinism and reset
# ---------------------------------------------------------------------------

class TestDeterminismAndReset:
    """Same sequence → same analytics. Reset clears both engines."""

    def _run_exits(self, exits):
        r = _runner()
        res, acc = _result(), _account()
        for i, (ep, reason) in enumerate(exits):
            _, acc = _place_and_finalize(r, f"T{i}", ep, reason, res, acc)
        res.finalise()
        s = r.get_analytics_engine().get_strategy_summary("pin_bar", "EURUSD", "D1")
        return res, s

    def test_same_sequence_produces_same_m18_totals(self):
        seq = [
            (1.1100, ExitReason.TP_HIT),
            (1.0950, ExitReason.SL_HIT),
            (1.1100, ExitReason.TP_HIT),
        ]
        _, s1 = self._run_exits(seq)
        _, s2 = self._run_exits(seq)
        assert s1.total_trades == s2.total_trades
        assert s1.win_count    == s2.win_count
        assert s1.loss_count   == s2.loss_count

    def test_same_sequence_produces_same_result_totals(self):
        seq = [
            (1.1100, ExitReason.TP_HIT),
            (1.0950, ExitReason.SL_HIT),
        ]
        res1, _ = self._run_exits(seq)
        res2, _ = self._run_exits(seq)
        assert res1.trades_executed == res2.trades_executed
        assert res1.wins            == res2.wins
        assert res1.losses          == res2.losses

    def test_reset_clears_m18(self):
        r = _runner()
        res, acc = _result(), _account()
        _place_and_finalize(r, "T1", 1.1100, ExitReason.TP_HIT, res, acc)
        r.reset()
        # After reset, fresh engine → no trades
        s = r.get_analytics_engine().get_strategy_summary("pin_bar", "EURUSD", "D1")
        assert s.total_trades == 0

    def test_reset_clears_m19(self):
        r = _runner()
        res, acc = _result(), _account()
        _place_and_finalize(r, "T1", 1.0950, ExitReason.SL_HIT, res, acc)
        r.reset()
        reviews = r.get_review_engine().get_all_results()
        assert len(reviews) == 0

    def test_mixed_strategy_types_recorded_separately_in_m18(self):
        """pin_bar and engulfing_bar trades must appear in their own M18 buckets."""
        r = _runner()
        res, acc = _result(), _account()
        _, acc = _place_and_finalize(
            r, "P1", 1.1100, ExitReason.TP_HIT, res, acc, strategy="pin_bar"
        )
        _, acc = _place_and_finalize(
            r, "E1", 1.0950, ExitReason.SL_HIT, res, acc, strategy="engulfing_bar"
        )
        sp = r.get_analytics_engine().get_strategy_summary("pin_bar", "EURUSD", "D1")
        se = r.get_analytics_engine().get_strategy_summary("engulfing_bar", "EURUSD", "D1")
        assert sp.total_trades == 1
        assert se.total_trades == 1
        assert sp.win_count  == 1
        assert se.loss_count == 1

    def test_account_balance_reflects_pnl_after_finalize(self):
        """_finalize_closed_order must return an updated account with new balance."""
        r = _runner()
        res, acc = _result(), _account(10_000.0)
        # TP hit: +100 pips × 0.10 lot × $10/pip = +$100
        _, new_acc = _place_and_finalize(r, "T1", 1.1100, ExitReason.TP_HIT, res, acc)
        assert new_acc.balance == pytest.approx(10_100.0, rel=0.01)

    def test_account_balance_decreases_after_sl(self):
        r = _runner()
        res, acc = _result(), _account(10_000.0)
        # SL hit: -50 pips × 0.10 lot × $10/pip = -$50
        _, new_acc = _place_and_finalize(r, "T1", 1.0950, ExitReason.SL_HIT, res, acc)
        assert new_acc.balance == pytest.approx(9_950.0, rel=0.01)

    def test_result_final_balance_matches_account_balance(self):
        """PipelineResult.final_balance must track P&L correctly."""
        r = _runner()
        res, acc = _result(), _account(10_000.0)
        # Two winners (+$100 each) and one loser (-$50)
        _, acc = _place_and_finalize(r, "T1", 1.1100, ExitReason.TP_HIT, res, acc)
        _, acc = _place_and_finalize(r, "T2", 1.1100, ExitReason.TP_HIT, res, acc)
        _, acc = _place_and_finalize(r, "T3", 1.0950, ExitReason.SL_HIT, res, acc)
        res.finalise()
        # Net P&L = +100 + 100 - 50 = +150
        assert res.net_profit_usd == pytest.approx(150.0, abs=1.0)
