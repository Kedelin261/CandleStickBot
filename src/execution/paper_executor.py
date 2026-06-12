"""
M10 — Paper Trade Executor  (Phase 1)
======================================
Consumes RiskApprovedOrder objects from M09 and simulates trade fills
entirely in-memory.  No broker, MT5, or live-execution code of any kind.

Integration wiring:
    - Closed trades → M18 StrategyAnalyticsEngine.record_trade()
    - Losing closed trades → M19 TradeReviewEngine.classify_loss()

Public API:
    place_paper_order(order)     → PaperOrder
    cancel_order(order_id)       → PaperOrder
    close_order(order_id, ...)   → PaperOrder
    get_open_orders()            → List[PaperOrder]
    get_closed_orders()          → List[PaperOrder]
    get_order(order_id)          → Optional[PaperOrder]
    simulate_slippage(price, direction, pips) → float
    generate_order_id()          → str
    reset_session()              → None
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from src.analytics.strategy_analytics import (
    StrategyAnalyticsEngine,
    StrategyPerformanceRecord,
)
from src.analytics.trade_review import TradeContext, TradeReviewEngine
from src.types import Direction, LossCategory, RiskApprovedOrder

logger = logging.getLogger("candlestickbot.execution.paper_executor")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class PaperOrderStatus:
    OPEN      = "OPEN"
    FILLED    = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED  = "REJECTED"
    CLOSED    = "CLOSED"

    ALL = frozenset({OPEN, FILLED, CANCELLED, REJECTED, CLOSED})


class ExitReason:
    TP_HIT       = "TP_HIT"
    SL_HIT       = "SL_HIT"
    MANUAL_CLOSE = "MANUAL_CLOSE"
    CANCELLED    = "CANCELLED"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PaperExecutorConfig:
    """
    Configuration for PaperTradeExecutor.

    default_slippage_pips : slippage added to every fill (per side)
    pip_size              : one pip in price terms (e.g. 0.0001 for EURUSD)
    """
    default_slippage_pips: float = 1.0
    pip_size:              float = 0.0001
    # Pass-through thresholds to M19 TradeContext
    pattern_quality_default:  float = 70.0
    level_strength_default:   float = 70.0


# ---------------------------------------------------------------------------
# PaperOrder DTO
# ---------------------------------------------------------------------------

@dataclass
class PaperOrder:
    """
    Represents one paper trade from creation through close.
    """
    order_id:       str
    strategy_name:  str
    symbol:         str
    timeframe:      str
    direction:      str            # "LONG" | "SHORT"
    status:         str            # PaperOrderStatus member
    requested_price: float
    filled_price:   float
    stop_loss:      float
    take_profit:    float
    lot_size:       float
    slippage_pips:  float
    created_at:     datetime
    risk_pct:       float   = 0.0
    risk_amount_usd: float  = 0.0
    account_balance: float  = 0.0
    stop_pips:       float  = 0.0

    # Populated on close
    closed_at:     Optional[datetime] = None
    exit_price:    Optional[float]    = None
    exit_reason:   Optional[str]      = None
    pnl_usd:       Optional[float]    = None
    r_multiple:    Optional[float]    = None

    @property
    def is_open(self) -> bool:
        return self.status in (PaperOrderStatus.OPEN, PaperOrderStatus.FILLED)

    @property
    def is_closed(self) -> bool:
        return self.status == PaperOrderStatus.CLOSED

    @property
    def is_winner(self) -> bool:
        return self.r_multiple is not None and self.r_multiple > 0

    @property
    def is_loser(self) -> bool:
        return self.r_multiple is not None and self.r_multiple < 0


# ---------------------------------------------------------------------------
# Main executor
# ---------------------------------------------------------------------------

class PaperTradeExecutor:
    """
    M10 Paper Trade Executor.

    Simulates the entire trade lifecycle in-memory.  On close it pushes
    records into M18 (always) and M19 (losing trades only).

    Parameters
    ----------
    analytics_engine : StrategyAnalyticsEngine
        M18 engine to receive StrategyPerformanceRecord on each close.
    review_engine : TradeReviewEngine
        M19 engine to receive classify_loss on each losing close.
    config : PaperExecutorConfig
        Slippage, pip size, and default quality scores.
    """

    def __init__(
        self,
        analytics_engine: Optional[StrategyAnalyticsEngine] = None,
        review_engine:    Optional[TradeReviewEngine]       = None,
        config:           Optional[PaperExecutorConfig]     = None,
    ) -> None:
        self._analytics = analytics_engine or StrategyAnalyticsEngine()
        self._review    = review_engine    or TradeReviewEngine()
        self._config    = config           or PaperExecutorConfig()

        # order_id → PaperOrder
        self._orders: Dict[str, PaperOrder] = {}

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def place_paper_order(self, order: RiskApprovedOrder) -> PaperOrder:
        """
        Accept a RiskApprovedOrder from M09 and create a paper fill.

        Slippage is applied in the direction adverse to the trade:
            LONG  → filled_price = entry + slippage
            SHORT → filled_price = entry - slippage
        """
        rec = order.recommendation
        direction_str = rec.direction.value  # "LONG" | "SHORT"

        filled_price = self.simulate_slippage(
            price=rec.entry_price,
            direction=rec.direction,
            slippage_pips=self._config.default_slippage_pips,
        )
        slippage_pips = abs(filled_price - rec.entry_price) / self._config.pip_size

        order_id = self.generate_order_id()

        paper_order = PaperOrder(
            order_id=order_id,
            strategy_name=rec.strategy.value,
            symbol=rec.symbol,
            timeframe=rec.timeframe,
            direction=direction_str,
            status=PaperOrderStatus.FILLED,
            requested_price=rec.entry_price,
            filled_price=filled_price,
            stop_loss=rec.stop_price,
            take_profit=rec.target_price,
            lot_size=order.lot_size,
            slippage_pips=slippage_pips,
            created_at=_now_utc(),
            risk_pct=order.risk_pct,
            risk_amount_usd=order.risk_amount_usd,
            account_balance=order.account_balance,
            stop_pips=order.stop_pips,
        )

        self._orders[order_id] = paper_order
        logger.debug(
            "M10: paper order %s placed (%s %s @ %.5f, SL=%.5f, TP=%.5f)",
            order_id,
            direction_str,
            rec.symbol,
            filled_price,
            rec.stop_price,
            rec.target_price,
        )
        return paper_order

    def cancel_order(self, order_id: str) -> PaperOrder:
        """
        Cancel an open order.  Raises KeyError if not found,
        ValueError if already closed/cancelled.
        """
        order = self._get_or_raise(order_id)
        if not order.is_open:
            raise ValueError(
                f"Order {order_id} cannot be cancelled (status={order.status})"
            )
        order.status     = PaperOrderStatus.CANCELLED
        order.exit_reason = ExitReason.CANCELLED
        order.closed_at  = _now_utc()
        logger.debug("M10: order %s cancelled", order_id)
        return order

    def close_order(
        self,
        order_id:    str,
        exit_price:  float,
        exit_reason: str = ExitReason.MANUAL_CLOSE,
    ) -> PaperOrder:
        """
        Close an open paper trade.

        P&L and R-multiple are computed from the *filled* (not requested)
        entry price to the provided exit_price.

        After close:
            - Record pushed to M18 StrategyAnalyticsEngine.
            - If losing, also pushed to M19 TradeReviewEngine.
        """
        order = self._get_or_raise(order_id)
        if not order.is_open:
            raise ValueError(
                f"Order {order_id} cannot be closed (status={order.status})"
            )

        order.exit_price  = exit_price
        order.exit_reason = exit_reason
        order.closed_at   = _now_utc()
        order.status      = PaperOrderStatus.CLOSED

        # Compute P&L and R-multiple
        pnl_pips, r_multiple = self._compute_trade_metrics(order, exit_price)
        pip_value_per_lot = 10.0   # standard EURUSD; close enough for paper
        pnl_usd = pnl_pips * order.lot_size * pip_value_per_lot

        order.pnl_usd    = pnl_usd
        order.r_multiple = r_multiple

        logger.debug(
            "M10: order %s closed @ %.5f — R=%.2f PnL=$%.2f (%s)",
            order_id,
            exit_price,
            r_multiple,
            pnl_usd,
            exit_reason,
        )

        # --- M18 integration ---
        self._push_to_analytics(order)

        # --- M19 integration (losers only) ---
        if order.is_loser:
            self._push_to_review(order)

        return order

    def get_open_orders(self) -> List[PaperOrder]:
        """Return all currently open (FILLED/OPEN status) orders."""
        return [o for o in self._orders.values() if o.is_open]

    def get_closed_orders(self) -> List[PaperOrder]:
        """Return all closed orders (includes CLOSED, CANCELLED, REJECTED)."""
        return [o for o in self._orders.values() if not o.is_open]

    def get_order(self, order_id: str) -> Optional[PaperOrder]:
        """Return the PaperOrder for *order_id*, or None if not found."""
        return self._orders.get(order_id)

    def simulate_slippage(
        self,
        price:          float,
        direction:      Direction,
        slippage_pips:  Optional[float] = None,
    ) -> float:
        """
        Return the expected fill price after applying adverse slippage.

        LONG  → entry goes up   (we pay more)
        SHORT → entry goes down (we receive less)
        """
        pips = (
            slippage_pips
            if slippage_pips is not None
            else self._config.default_slippage_pips
        )
        pip_move = pips * self._config.pip_size
        if direction == Direction.LONG:
            return price + pip_move
        return price - pip_move

    def generate_order_id(self) -> str:
        """Generate a unique paper order ID."""
        return f"PAPER-{uuid.uuid4().hex[:12].upper()}"

    def reset_session(self) -> None:
        """Clear all orders from in-memory storage."""
        self._orders.clear()
        logger.debug("M10: paper session reset")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_or_raise(self, order_id: str) -> PaperOrder:
        order = self._orders.get(order_id)
        if order is None:
            raise KeyError(f"Order {order_id!r} not found")
        return order

    @staticmethod
    def _compute_trade_metrics(
        order: PaperOrder,
        exit_price: float,
    ):
        """Return (pnl_pips, r_multiple)."""
        entry = order.filled_price
        sl    = order.stop_loss

        if order.direction == Direction.LONG.value:
            pnl_pips  = (exit_price - entry) / 0.0001
            risk_pips = (entry - sl)         / 0.0001
        else:  # SHORT
            pnl_pips  = (entry - exit_price) / 0.0001
            risk_pips = (sl - entry)         / 0.0001

        risk_pips = abs(risk_pips)
        r_multiple = pnl_pips / risk_pips if risk_pips > 0 else 0.0
        return pnl_pips, r_multiple

    def _push_to_analytics(self, order: PaperOrder) -> None:
        """Push closed trade to M18 StrategyAnalyticsEngine."""
        entry_ts = order.created_at
        exit_ts  = order.closed_at or _now_utc()

        record = StrategyPerformanceRecord(
            trade_id=order.order_id,
            strategy_name=order.strategy_name,
            symbol=order.symbol,
            timeframe=order.timeframe,
            direction=order.direction,
            entry_timestamp=entry_ts,
            exit_timestamp=exit_ts,
            entry_price=order.filled_price,
            exit_price=order.exit_price or order.filled_price,
            stop_loss=order.stop_loss,
            take_profit=order.take_profit,
            pnl_pips=_pnl_pips_from_r(order),
            pnl_usd=order.pnl_usd or 0.0,
            r_multiple=order.r_multiple or 0.0,
            exit_reason=order.exit_reason or "MANUAL_CLOSE",
        )
        try:
            self._analytics.record_trade(record)
        except Exception as exc:
            logger.warning("M10: M18 push failed for %s: %s", order.order_id, exc)

    def _push_to_review(self, order: PaperOrder) -> None:
        """Push losing trade to M19 TradeReviewEngine."""
        ctx = TradeContext(
            pattern_quality_score=self._config.pattern_quality_default,
            level_strength_score=self._config.level_strength_default,
            regime="TRENDING",   # conservative default for paper trades
            fill_slippage_pips=order.slippage_pips,
            stop_distance_pips=order.stop_pips,
            was_overridden=False,
        )
        try:
            self._review.classify_loss(order.order_id, order.strategy_name, ctx)
        except Exception as exc:
            logger.warning("M10: M19 push failed for %s: %s", order.order_id, exc)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _pnl_pips_from_r(order: PaperOrder) -> float:
    """Back-compute pnl_pips from r_multiple × stop_pips."""
    if order.r_multiple is None:
        return 0.0
    return order.r_multiple * order.stop_pips
