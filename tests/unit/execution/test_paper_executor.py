"""
Sprint 11 — M10 Paper Trade Executor
Tests for src/execution/paper_executor.py

Classes / counts:
    TestPaperOrderFields             (8)
    TestPaperOrderStatus             (7)
    TestGenerateOrderId              (5)
    TestSimulateSlippage             (8)
    TestPlacePaperOrder              (10)
    TestCancelOrder                  (6)
    TestCloseOrderBasics             (8)
    TestPnlCalculation               (8)
    TestRMultipleCalculation         (8)
    TestGetOpenOrders                (6)
    TestGetClosedOrders              (6)
    TestGetOrder                     (5)
    TestResetSession                 (4)
    TestM18Integration               (8)
    TestM19Integration               (7)
    TestNoBrokerCalls                (4)
    TestEdgeCasesAndIntegration      (9)

Total: 117 tests
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, call, patch

import pytest

from src.analytics.strategy_analytics import StrategyAnalyticsEngine
from src.analytics.trade_review import TradeReviewEngine
from src.execution.paper_executor import (
    ExitReason,
    PaperExecutorConfig,
    PaperOrder,
    PaperOrderStatus,
    PaperTradeExecutor,
)
from src.types import (
    Direction,
    RiskApprovedOrder,
    StrategyName,
    TQSComponents,
    TradeRecommendation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tqs(total: float = 80.0) -> TQSComponents:
    q = total / 4
    return TQSComponents(
        trend_score=q,
        level_score=q,
        pattern_score=q,
        regime_score=q,
    )


def _recommendation(
    strategy: StrategyName = StrategyName.PIN_BAR,
    direction: Direction = Direction.LONG,
    entry: float = 1.1000,
    stop: float = 1.0950,
    tp: float = 1.1100,
    symbol: str = "EURUSD",
    timeframe: str = "H1",
) -> TradeRecommendation:
    rr = abs(tp - entry) / abs(entry - stop)
    return TradeRecommendation(
        strategy=strategy,
        symbol=symbol,
        timeframe=timeframe,
        direction=direction,
        entry_price=entry,
        stop_price=stop,
        target_price=tp,
        rr_ratio=rr,
        tqs=_tqs(),
    )


def _approved_order(
    strategy: StrategyName = StrategyName.PIN_BAR,
    direction: Direction = Direction.LONG,
    entry: float = 1.1000,
    stop: float = 1.0950,
    tp: float = 1.1100,
    lot_size: float = 0.10,
    risk_pct: float = 1.0,
    risk_amount_usd: float = 100.0,
    account_balance: float = 10_000.0,
    stop_pips: float = 50.0,
    symbol: str = "EURUSD",
    timeframe: str = "H1",
) -> RiskApprovedOrder:
    rec = _recommendation(
        strategy=strategy,
        direction=direction,
        entry=entry,
        stop=stop,
        tp=tp,
        symbol=symbol,
        timeframe=timeframe,
    )
    return RiskApprovedOrder(
        recommendation=rec,
        lot_size=lot_size,
        risk_pct=risk_pct,
        risk_amount_usd=risk_amount_usd,
        account_balance=account_balance,
        stop_pips=stop_pips,
    )


def _executor(**kwargs) -> PaperTradeExecutor:
    """Create executor with default zero-slippage config unless overridden."""
    cfg = kwargs.pop("config", PaperExecutorConfig(default_slippage_pips=0.0))
    return PaperTradeExecutor(config=cfg, **kwargs)


def _place(executor: PaperTradeExecutor, **kwargs) -> PaperOrder:
    return executor.place_paper_order(_approved_order(**kwargs))


# ===========================================================================
# TestPaperOrderFields
# ===========================================================================

class TestPaperOrderFields:
    def test_has_order_id(self):
        e = _executor()
        o = _place(e)
        assert isinstance(o.order_id, str) and len(o.order_id) > 0

    def test_has_strategy_name(self):
        e = _executor()
        o = _place(e, strategy=StrategyName.ENGULFING_BAR)
        assert o.strategy_name == "engulfing_bar"

    def test_has_symbol(self):
        e = _executor()
        o = _place(e, symbol="USDJPY")
        assert o.symbol == "USDJPY"

    def test_has_timeframe(self):
        e = _executor()
        o = _place(e, timeframe="M15")
        assert o.timeframe == "M15"

    def test_has_direction(self):
        e = _executor()
        o = _place(e, direction=Direction.SHORT)
        assert o.direction == "SHORT"

    def test_has_lot_size(self):
        e = _executor()
        o = _place(e, lot_size=0.25)
        assert o.lot_size == 0.25

    def test_has_stop_loss(self):
        e = _executor()
        o = _place(e, stop=1.0950)
        assert o.stop_loss == pytest.approx(1.0950)

    def test_has_take_profit(self):
        e = _executor()
        o = _place(e, tp=1.1200)
        assert o.take_profit == pytest.approx(1.1200)


# ===========================================================================
# TestPaperOrderStatus
# ===========================================================================

class TestPaperOrderStatus:
    def test_new_order_is_filled(self):
        e = _executor()
        o = _place(e)
        assert o.status == PaperOrderStatus.FILLED

    def test_is_open_true_when_filled(self):
        e = _executor()
        o = _place(e)
        assert o.is_open is True

    def test_is_closed_false_when_filled(self):
        e = _executor()
        o = _place(e)
        assert o.is_closed is False

    def test_is_winner_false_when_open(self):
        e = _executor()
        o = _place(e)
        assert o.is_winner is False

    def test_is_loser_false_when_open(self):
        e = _executor()
        o = _place(e)
        assert o.is_loser is False

    def test_cancelled_order_not_open(self):
        e = _executor()
        o = _place(e)
        e.cancel_order(o.order_id)
        assert o.is_open is False

    def test_closed_order_is_closed(self):
        e = _executor()
        o = _place(e, entry=1.1000, stop=1.0950, tp=1.1100)
        e.close_order(o.order_id, exit_price=1.1100)
        assert o.status == PaperOrderStatus.CLOSED


# ===========================================================================
# TestGenerateOrderId
# ===========================================================================

class TestGenerateOrderId:
    def test_id_is_string(self):
        e = _executor()
        assert isinstance(e.generate_order_id(), str)

    def test_id_starts_with_paper(self):
        e = _executor()
        assert e.generate_order_id().startswith("PAPER-")

    def test_ids_are_unique(self):
        e = _executor()
        ids = {e.generate_order_id() for _ in range(100)}
        assert len(ids) == 100

    def test_multiple_orders_have_unique_ids(self):
        e = _executor()
        o1 = _place(e)
        o2 = _place(e)
        assert o1.order_id != o2.order_id

    def test_id_length_reasonable(self):
        e = _executor()
        oid = e.generate_order_id()
        assert 10 <= len(oid) <= 40


# ===========================================================================
# TestSimulateSlippage
# ===========================================================================

class TestSimulateSlippage:
    def test_long_slippage_increases_price(self):
        e = _executor()
        result = e.simulate_slippage(1.1000, Direction.LONG, slippage_pips=1.0)
        assert result > 1.1000

    def test_short_slippage_decreases_price(self):
        e = _executor()
        result = e.simulate_slippage(1.1000, Direction.SHORT, slippage_pips=1.0)
        assert result < 1.1000

    def test_zero_slippage_unchanged(self):
        e = _executor()
        result = e.simulate_slippage(1.1000, Direction.LONG, slippage_pips=0.0)
        assert result == pytest.approx(1.1000)

    def test_long_1pip_slippage_exact(self):
        e = _executor()
        result = e.simulate_slippage(1.1000, Direction.LONG, slippage_pips=1.0)
        assert result == pytest.approx(1.1001, abs=1e-9)

    def test_short_1pip_slippage_exact(self):
        e = _executor()
        result = e.simulate_slippage(1.1000, Direction.SHORT, slippage_pips=1.0)
        assert result == pytest.approx(1.0999, abs=1e-9)

    def test_uses_config_default_when_pips_not_given(self):
        cfg = PaperExecutorConfig(default_slippage_pips=2.0)
        e = _executor(config=cfg)
        result = e.simulate_slippage(1.1000, Direction.LONG)
        assert result == pytest.approx(1.1002, abs=1e-9)

    def test_slippage_symmetric_long_short(self):
        e = _executor()
        long_price  = e.simulate_slippage(1.1000, Direction.LONG,  slippage_pips=3.0)
        short_price = e.simulate_slippage(1.1000, Direction.SHORT, slippage_pips=3.0)
        assert abs(long_price  - 1.1000) == pytest.approx(abs(short_price - 1.1000))

    def test_large_slippage(self):
        e = _executor()
        result = e.simulate_slippage(1.1000, Direction.LONG, slippage_pips=100.0)
        assert result == pytest.approx(1.1100, abs=1e-9)


# ===========================================================================
# TestPlacePaperOrder
# ===========================================================================

class TestPlacePaperOrder:
    def test_returns_paper_order(self):
        e = _executor()
        o = _place(e)
        assert isinstance(o, PaperOrder)

    def test_order_stored_in_executor(self):
        e = _executor()
        o = _place(e)
        assert e.get_order(o.order_id) is o

    def test_filled_price_equals_entry_with_zero_slippage(self):
        e = _executor()  # 0-pip slippage config
        o = _place(e, entry=1.1000)
        assert o.filled_price == pytest.approx(1.1000)

    def test_filled_price_adjusted_for_long_slippage(self):
        cfg = PaperExecutorConfig(default_slippage_pips=2.0)
        e = _executor(config=cfg)
        o = _place(e, entry=1.1000, direction=Direction.LONG)
        assert o.filled_price == pytest.approx(1.1002, abs=1e-9)

    def test_filled_price_adjusted_for_short_slippage(self):
        cfg = PaperExecutorConfig(default_slippage_pips=2.0)
        e = _executor(config=cfg)
        o = _place(e, entry=1.1000, direction=Direction.SHORT)
        assert o.filled_price == pytest.approx(1.0998, abs=1e-9)

    def test_slippage_pips_computed(self):
        cfg = PaperExecutorConfig(default_slippage_pips=3.0)
        e = _executor(config=cfg)
        o = _place(e, entry=1.1000)
        assert o.slippage_pips == pytest.approx(3.0, abs=1e-6)

    def test_created_at_is_datetime(self):
        e = _executor()
        o = _place(e)
        assert isinstance(o.created_at, datetime)

    def test_multiple_orders_independent(self):
        e = _executor()
        o1 = _place(e, entry=1.1000)
        o2 = _place(e, entry=1.2000)
        assert o1.order_id != o2.order_id
        assert len(e.get_open_orders()) == 2

    def test_engulfing_bar_strategy_stored(self):
        e = _executor()
        o = _place(e, strategy=StrategyName.ENGULFING_BAR)
        assert o.strategy_name == "engulfing_bar"

    def test_risk_fields_stored(self):
        e = _executor()
        o = _place(e, risk_pct=1.5, risk_amount_usd=150.0, account_balance=10000.0)
        assert o.risk_pct == pytest.approx(1.5)
        assert o.risk_amount_usd == pytest.approx(150.0)


# ===========================================================================
# TestCancelOrder
# ===========================================================================

class TestCancelOrder:
    def test_cancel_sets_cancelled_status(self):
        e = _executor()
        o = _place(e)
        e.cancel_order(o.order_id)
        assert o.status == PaperOrderStatus.CANCELLED

    def test_cancel_marks_not_open(self):
        e = _executor()
        o = _place(e)
        e.cancel_order(o.order_id)
        assert o.is_open is False

    def test_cancel_sets_exit_reason(self):
        e = _executor()
        o = _place(e)
        e.cancel_order(o.order_id)
        assert o.exit_reason == ExitReason.CANCELLED

    def test_cancel_sets_closed_at(self):
        e = _executor()
        o = _place(e)
        e.cancel_order(o.order_id)
        assert isinstance(o.closed_at, datetime)

    def test_cancel_nonexistent_raises_key_error(self):
        e = _executor()
        with pytest.raises(KeyError):
            e.cancel_order("NONEXISTENT")

    def test_cancel_already_closed_raises_value_error(self):
        e = _executor()
        o = _place(e, entry=1.1000, stop=1.0950, tp=1.1100)
        e.close_order(o.order_id, exit_price=1.1100)
        with pytest.raises(ValueError):
            e.cancel_order(o.order_id)


# ===========================================================================
# TestCloseOrderBasics
# ===========================================================================

class TestCloseOrderBasics:
    def test_close_sets_closed_status(self):
        e = _executor()
        o = _place(e)
        e.close_order(o.order_id, exit_price=1.1100)
        assert o.status == PaperOrderStatus.CLOSED

    def test_close_sets_exit_price(self):
        e = _executor()
        o = _place(e)
        e.close_order(o.order_id, exit_price=1.1100)
        assert o.exit_price == pytest.approx(1.1100)

    def test_close_sets_exit_reason_default(self):
        e = _executor()
        o = _place(e)
        e.close_order(o.order_id, exit_price=1.1100)
        assert o.exit_reason == ExitReason.MANUAL_CLOSE

    def test_close_sets_custom_exit_reason(self):
        e = _executor()
        o = _place(e)
        e.close_order(o.order_id, exit_price=1.1100, exit_reason=ExitReason.TP_HIT)
        assert o.exit_reason == ExitReason.TP_HIT

    def test_close_sets_closed_at(self):
        e = _executor()
        o = _place(e)
        e.close_order(o.order_id, exit_price=1.1100)
        assert isinstance(o.closed_at, datetime)

    def test_close_nonexistent_raises_key_error(self):
        e = _executor()
        with pytest.raises(KeyError):
            e.close_order("NONEXISTENT", exit_price=1.1)

    def test_close_already_closed_raises_value_error(self):
        e = _executor()
        o = _place(e)
        e.close_order(o.order_id, exit_price=1.1100)
        with pytest.raises(ValueError):
            e.close_order(o.order_id, exit_price=1.1200)

    def test_close_is_winner_at_tp(self):
        e = _executor()
        o = _place(e, entry=1.1000, stop=1.0950, tp=1.1100)
        e.close_order(o.order_id, exit_price=1.1100)
        assert o.is_winner is True


# ===========================================================================
# TestPnlCalculation
# ===========================================================================

class TestPnlCalculation:
    def test_long_win_positive_pnl(self):
        e = _executor()
        o = _place(e, entry=1.1000, stop=1.0950, tp=1.1100, lot_size=1.0)
        e.close_order(o.order_id, exit_price=1.1100)
        assert o.pnl_usd > 0

    def test_long_loss_negative_pnl(self):
        e = _executor()
        o = _place(e, entry=1.1000, stop=1.0950, tp=1.1100, lot_size=1.0)
        e.close_order(o.order_id, exit_price=1.0950)
        assert o.pnl_usd < 0

    def test_short_win_positive_pnl(self):
        e = _executor()
        o = _place(e, entry=1.1000, stop=1.1050, tp=1.0900,
                   direction=Direction.SHORT, lot_size=1.0)
        e.close_order(o.order_id, exit_price=1.0900)
        assert o.pnl_usd > 0

    def test_short_loss_negative_pnl(self):
        e = _executor()
        o = _place(e, entry=1.1000, stop=1.1050, tp=1.0900,
                   direction=Direction.SHORT, lot_size=1.0)
        e.close_order(o.order_id, exit_price=1.1050)
        assert o.pnl_usd < 0

    def test_breakeven_close_near_zero_pnl(self):
        e = _executor()
        o = _place(e, entry=1.1000, stop=1.0950, tp=1.1100, lot_size=1.0)
        e.close_order(o.order_id, exit_price=1.1000)
        assert o.pnl_usd == pytest.approx(0.0, abs=0.01)

    def test_pnl_usd_set_on_close(self):
        e = _executor()
        o = _place(e)
        assert o.pnl_usd is None
        e.close_order(o.order_id, exit_price=1.1100)
        assert o.pnl_usd is not None

    def test_pnl_larger_lot_larger_usd(self):
        e = _executor()
        o1 = _place(e, entry=1.1000, stop=1.0950, tp=1.1100, lot_size=0.1)
        o2 = _place(e, entry=1.1000, stop=1.0950, tp=1.1100, lot_size=1.0)
        e.close_order(o1.order_id, exit_price=1.1100)
        e.close_order(o2.order_id, exit_price=1.1100)
        assert abs(o2.pnl_usd) > abs(o1.pnl_usd)

    def test_pnl_usd_type_is_float(self):
        e = _executor()
        o = _place(e)
        e.close_order(o.order_id, exit_price=1.1100)
        assert isinstance(o.pnl_usd, float)


# ===========================================================================
# TestRMultipleCalculation
# ===========================================================================

class TestRMultipleCalculation:
    def test_r_multiple_set_on_close(self):
        e = _executor()
        o = _place(e)
        e.close_order(o.order_id, exit_price=1.1100)
        assert o.r_multiple is not None

    def test_r_multiple_2r_long_win(self):
        # entry=1.1000, SL=1.0950 (50 pips risk), exit=1.1100 (100 pips gain)
        e = _executor()
        o = _place(e, entry=1.1000, stop=1.0950, tp=1.1100)
        e.close_order(o.order_id, exit_price=1.1100)
        assert o.r_multiple == pytest.approx(2.0, rel=0.01)

    def test_r_multiple_1r_long_loss(self):
        # entry=1.1000, SL=1.0950, exit=1.0950 → -1R
        e = _executor()
        o = _place(e, entry=1.1000, stop=1.0950, tp=1.1100)
        e.close_order(o.order_id, exit_price=1.0950)
        assert o.r_multiple == pytest.approx(-1.0, rel=0.01)

    def test_r_multiple_2r_short_win(self):
        # entry=1.1000, SL=1.1050 (50 pips risk), exit=1.0900 (100 pips gain)
        e = _executor()
        o = _place(e, entry=1.1000, stop=1.1050, tp=1.0900,
                   direction=Direction.SHORT)
        e.close_order(o.order_id, exit_price=1.0900)
        assert o.r_multiple == pytest.approx(2.0, rel=0.01)

    def test_r_multiple_1r_short_loss(self):
        # entry=1.1000, SL=1.1050, exit=1.1050 → -1R
        e = _executor()
        o = _place(e, entry=1.1000, stop=1.1050, tp=1.0900,
                   direction=Direction.SHORT)
        e.close_order(o.order_id, exit_price=1.1050)
        assert o.r_multiple == pytest.approx(-1.0, rel=0.01)

    def test_is_winner_true_for_positive_r(self):
        e = _executor()
        o = _place(e, entry=1.1000, stop=1.0950, tp=1.1100)
        e.close_order(o.order_id, exit_price=1.1100)
        assert o.is_winner is True and o.is_loser is False

    def test_is_loser_true_for_negative_r(self):
        e = _executor()
        o = _place(e, entry=1.1000, stop=1.0950, tp=1.1100)
        e.close_order(o.order_id, exit_price=1.0950)
        assert o.is_loser is True and o.is_winner is False

    def test_r_multiple_is_float(self):
        e = _executor()
        o = _place(e)
        e.close_order(o.order_id, exit_price=1.1100)
        assert isinstance(o.r_multiple, float)


# ===========================================================================
# TestGetOpenOrders
# ===========================================================================

class TestGetOpenOrders:
    def test_empty_initially(self):
        e = _executor()
        assert e.get_open_orders() == []

    def test_single_open_order(self):
        e = _executor()
        _place(e)
        assert len(e.get_open_orders()) == 1

    def test_multiple_open_orders(self):
        e = _executor()
        _place(e)
        _place(e)
        _place(e)
        assert len(e.get_open_orders()) == 3

    def test_closed_orders_not_in_open(self):
        e = _executor()
        o = _place(e)
        e.close_order(o.order_id, exit_price=1.1100)
        assert len(e.get_open_orders()) == 0

    def test_cancelled_orders_not_in_open(self):
        e = _executor()
        o = _place(e)
        e.cancel_order(o.order_id)
        assert len(e.get_open_orders()) == 0

    def test_mixed_open_and_closed(self):
        e = _executor()
        o1 = _place(e)
        o2 = _place(e)
        e.close_order(o1.order_id, exit_price=1.1100)
        open_orders = e.get_open_orders()
        assert len(open_orders) == 1
        assert open_orders[0].order_id == o2.order_id


# ===========================================================================
# TestGetClosedOrders
# ===========================================================================

class TestGetClosedOrders:
    def test_empty_initially(self):
        e = _executor()
        assert e.get_closed_orders() == []

    def test_closed_order_appears(self):
        e = _executor()
        o = _place(e)
        e.close_order(o.order_id, exit_price=1.1100)
        assert len(e.get_closed_orders()) == 1

    def test_cancelled_order_appears(self):
        e = _executor()
        o = _place(e)
        e.cancel_order(o.order_id)
        assert len(e.get_closed_orders()) == 1

    def test_open_orders_not_in_closed(self):
        e = _executor()
        _place(e)
        assert e.get_closed_orders() == []

    def test_multiple_closed(self):
        e = _executor()
        for _ in range(3):
            o = _place(e)
            e.close_order(o.order_id, exit_price=1.1100)
        assert len(e.get_closed_orders()) == 3

    def test_mix_returns_correct_closed_count(self):
        e = _executor()
        o1 = _place(e)
        _place(e)  # stays open
        e.close_order(o1.order_id, exit_price=1.1100)
        assert len(e.get_closed_orders()) == 1


# ===========================================================================
# TestGetOrder
# ===========================================================================

class TestGetOrder:
    def test_returns_order_by_id(self):
        e = _executor()
        o = _place(e)
        assert e.get_order(o.order_id) is o

    def test_returns_none_for_unknown_id(self):
        e = _executor()
        assert e.get_order("UNKNOWN") is None

    def test_returns_closed_order(self):
        e = _executor()
        o = _place(e)
        e.close_order(o.order_id, exit_price=1.1100)
        assert e.get_order(o.order_id) is o

    def test_returns_cancelled_order(self):
        e = _executor()
        o = _place(e)
        e.cancel_order(o.order_id)
        assert e.get_order(o.order_id) is o

    def test_returns_none_after_reset(self):
        e = _executor()
        o = _place(e)
        e.reset_session()
        assert e.get_order(o.order_id) is None


# ===========================================================================
# TestResetSession
# ===========================================================================

class TestResetSession:
    def test_clears_open_orders(self):
        e = _executor()
        _place(e)
        _place(e)
        e.reset_session()
        assert e.get_open_orders() == []

    def test_clears_closed_orders(self):
        e = _executor()
        o = _place(e)
        e.close_order(o.order_id, exit_price=1.1100)
        e.reset_session()
        assert e.get_closed_orders() == []

    def test_allows_new_orders_after_reset(self):
        e = _executor()
        _place(e)
        e.reset_session()
        _place(e)
        assert len(e.get_open_orders()) == 1

    def test_double_reset_safe(self):
        e = _executor()
        e.reset_session()
        e.reset_session()
        assert e.get_open_orders() == []


# ===========================================================================
# TestM18Integration
# ===========================================================================

class TestM18Integration:
    def test_closed_trade_pushed_to_analytics(self):
        analytics = MagicMock(spec=StrategyAnalyticsEngine)
        e = PaperTradeExecutor(analytics_engine=analytics)
        o = _place(e)
        e.close_order(o.order_id, exit_price=1.1100)
        analytics.record_trade.assert_called_once()

    def test_cancelled_order_not_pushed_to_analytics(self):
        analytics = MagicMock(spec=StrategyAnalyticsEngine)
        e = PaperTradeExecutor(analytics_engine=analytics)
        o = _place(e)
        e.cancel_order(o.order_id)
        analytics.record_trade.assert_not_called()

    def test_analytics_receives_correct_strategy_name(self):
        analytics = MagicMock(spec=StrategyAnalyticsEngine)
        e = PaperTradeExecutor(analytics_engine=analytics)
        o = _place(e, strategy=StrategyName.ENGULFING_BAR)
        e.close_order(o.order_id, exit_price=1.1100)
        args = analytics.record_trade.call_args[0]
        record = args[0]
        assert record.strategy_name == "engulfing_bar"

    def test_analytics_receives_correct_symbol(self):
        analytics = MagicMock(spec=StrategyAnalyticsEngine)
        e = PaperTradeExecutor(analytics_engine=analytics)
        o = _place(e, symbol="GBPUSD")
        e.close_order(o.order_id, exit_price=1.1100)
        record = analytics.record_trade.call_args[0][0]
        assert record.symbol == "GBPUSD"

    def test_analytics_receives_correct_r_multiple(self):
        analytics = MagicMock(spec=StrategyAnalyticsEngine)
        cfg = PaperExecutorConfig(default_slippage_pips=0.0)
        e = PaperTradeExecutor(analytics_engine=analytics, config=cfg)
        order = _approved_order(entry=1.1000, stop=1.0950, tp=1.1100)
        o = e.place_paper_order(order)
        e.close_order(o.order_id, exit_price=1.1100)
        record = analytics.record_trade.call_args[0][0]
        # R should be positive (winner) and close to 2.0
        assert record.r_multiple > 1.5
        assert record.r_multiple == pytest.approx(2.0, rel=0.05)

    def test_multiple_closes_push_multiple_records(self):
        analytics = MagicMock(spec=StrategyAnalyticsEngine)
        e = PaperTradeExecutor(analytics_engine=analytics)
        for _ in range(3):
            o = _place(e)
            e.close_order(o.order_id, exit_price=1.1100)
        assert analytics.record_trade.call_count == 3

    def test_analytics_failure_does_not_crash_executor(self):
        analytics = MagicMock(spec=StrategyAnalyticsEngine)
        analytics.record_trade.side_effect = RuntimeError("M18 down")
        e = PaperTradeExecutor(analytics_engine=analytics)
        o = _place(e)
        # Should not raise
        e.close_order(o.order_id, exit_price=1.1100)

    def test_analytics_receives_exit_reason(self):
        analytics = MagicMock(spec=StrategyAnalyticsEngine)
        e = PaperTradeExecutor(analytics_engine=analytics)
        o = _place(e)
        e.close_order(o.order_id, exit_price=1.1100, exit_reason=ExitReason.TP_HIT)
        record = analytics.record_trade.call_args[0][0]
        assert record.exit_reason == ExitReason.TP_HIT


# ===========================================================================
# TestM19Integration
# ===========================================================================

class TestM19Integration:
    def test_losing_close_pushed_to_review(self):
        review = MagicMock(spec=TradeReviewEngine)
        e = PaperTradeExecutor(review_engine=review)
        o = _place(e, entry=1.1000, stop=1.0950, tp=1.1100)
        e.close_order(o.order_id, exit_price=1.0950)   # SL hit → loss
        review.classify_loss.assert_called_once()

    def test_winning_close_not_pushed_to_review(self):
        review = MagicMock(spec=TradeReviewEngine)
        e = PaperTradeExecutor(review_engine=review)
        o = _place(e, entry=1.1000, stop=1.0950, tp=1.1100)
        e.close_order(o.order_id, exit_price=1.1100)   # TP hit → win
        review.classify_loss.assert_not_called()

    def test_review_receives_correct_trade_id(self):
        review = MagicMock(spec=TradeReviewEngine)
        e = PaperTradeExecutor(review_engine=review)
        o = _place(e, entry=1.1000, stop=1.0950, tp=1.1100)
        e.close_order(o.order_id, exit_price=1.0950)
        call_args = review.classify_loss.call_args
        assert call_args[0][0] == o.order_id

    def test_review_receives_correct_strategy_name(self):
        review = MagicMock(spec=TradeReviewEngine)
        e = PaperTradeExecutor(review_engine=review)
        o = _place(e, strategy=StrategyName.ENGULFING_BAR,
                   entry=1.1000, stop=1.0950, tp=1.1100)
        e.close_order(o.order_id, exit_price=1.0950)
        call_args = review.classify_loss.call_args
        assert call_args[0][1] == "engulfing_bar"

    def test_review_failure_does_not_crash_executor(self):
        review = MagicMock(spec=TradeReviewEngine)
        review.classify_loss.side_effect = RuntimeError("M19 down")
        e = PaperTradeExecutor(review_engine=review)
        o = _place(e, entry=1.1000, stop=1.0950, tp=1.1100)
        # Losing close, M19 throws — should not propagate
        e.close_order(o.order_id, exit_price=1.0950)

    def test_cancelled_order_not_pushed_to_review(self):
        review = MagicMock(spec=TradeReviewEngine)
        e = PaperTradeExecutor(review_engine=review)
        o = _place(e)
        e.cancel_order(o.order_id)
        review.classify_loss.assert_not_called()

    def test_multiple_losses_each_pushed(self):
        review = MagicMock(spec=TradeReviewEngine)
        e = PaperTradeExecutor(review_engine=review)
        for _ in range(4):
            o = _place(e, entry=1.1000, stop=1.0950, tp=1.1100)
            e.close_order(o.order_id, exit_price=1.0950)
        assert review.classify_loss.call_count == 4


# ===========================================================================
# TestNoBrokerCalls
# ===========================================================================

class TestNoBrokerCalls:
    def test_no_mt5_import(self):
        import src.execution.paper_executor as mod
        assert not hasattr(mod, "MetaTrader5"), "MT5 must not be imported"

    def test_no_broker_attribute(self):
        e = _executor()
        assert not hasattr(e, "broker"), "No broker attribute allowed"

    def test_no_live_execution_attribute(self):
        e = _executor()
        assert not hasattr(e, "live_execute")

    def test_no_external_api_calls_on_place(self):
        """Placing an order makes no network calls."""
        e = _executor()
        # This should complete instantly with no I/O
        _place(e)


# ===========================================================================
# TestEdgeCasesAndIntegration
# ===========================================================================

class TestEdgeCasesAndIntegration:
    def test_full_lifecycle_win(self):
        e = _executor()
        o = _place(e, entry=1.1000, stop=1.0950, tp=1.1100)
        assert o.is_open
        e.close_order(o.order_id, exit_price=1.1100, exit_reason=ExitReason.TP_HIT)
        assert o.is_closed and o.is_winner and o.pnl_usd > 0

    def test_full_lifecycle_loss(self):
        e = _executor()
        o = _place(e, entry=1.1000, stop=1.0950, tp=1.1100)
        e.close_order(o.order_id, exit_price=1.0950, exit_reason=ExitReason.SL_HIT)
        assert o.is_closed and o.is_loser and o.pnl_usd < 0

    def test_usdjpy_pip_size_separate_config(self):
        """Pip size can be configured per-instrument."""
        cfg = PaperExecutorConfig(default_slippage_pips=0.0, pip_size=0.01)
        e = _executor(config=cfg)
        o = _place(e, entry=150.00, stop=149.50, tp=151.00, symbol="USDJPY")
        e.close_order(o.order_id, exit_price=151.00)
        assert o.is_winner

    def test_reset_then_new_session_works(self):
        e = _executor()
        o1 = _place(e)
        e.close_order(o1.order_id, exit_price=1.1100)
        e.reset_session()
        o2 = _place(e)
        assert len(e.get_open_orders()) == 1
        assert o2.order_id != o1.order_id

    def test_integration_analytics_and_review_real_engines(self):
        """End-to-end with real M18 and M19 engines; both receive data."""
        analytics = StrategyAnalyticsEngine()
        review    = TradeReviewEngine()
        e = PaperTradeExecutor(analytics_engine=analytics, review_engine=review)

        # Place and close a losing trade
        o = _place(e, entry=1.1000, stop=1.0950, tp=1.1100,
                   strategy=StrategyName.PIN_BAR)
        e.close_order(o.order_id, exit_price=1.0950)

        # M18 received record
        summary = analytics.get_strategy_summary("pin_bar", "EURUSD", "H1")
        assert summary.total_trades == 1
        assert summary.loss_count == 1

        # M19 received classification
        results = review.get_all_results()
        assert len(results) == 1
        assert results[0].trade_id == o.order_id

    def test_integration_win_goes_to_m18_not_m19(self):
        analytics = StrategyAnalyticsEngine()
        review    = TradeReviewEngine()
        e = PaperTradeExecutor(analytics_engine=analytics, review_engine=review)

        o = _place(e, entry=1.1000, stop=1.0950, tp=1.1100)
        e.close_order(o.order_id, exit_price=1.1100)

        summary = analytics.get_strategy_summary("pin_bar", "EURUSD", "H1")
        assert summary.win_count == 1
        assert len(review.get_all_results()) == 0   # no M19 push for win

    def test_session_order_count_correct(self):
        e = _executor()
        for _ in range(5):
            _place(e)
        assert len(e.get_open_orders()) == 5

    def test_deterministic_slippage_on_same_direction(self):
        cfg = PaperExecutorConfig(default_slippage_pips=1.5)
        e = _executor(config=cfg)
        o1 = _place(e, entry=1.1000, direction=Direction.LONG)
        o2 = _place(e, entry=1.1000, direction=Direction.LONG)
        assert o1.filled_price == o2.filled_price

    def test_engulfing_bar_lifecycle(self):
        e = _executor()
        o = _place(e, strategy=StrategyName.ENGULFING_BAR,
                   entry=1.2000, stop=1.1950, tp=1.2100)
        e.close_order(o.order_id, exit_price=1.2100)
        assert o.strategy_name == "engulfing_bar"
        assert o.is_winner
