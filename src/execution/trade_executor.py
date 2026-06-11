"""
M10 — Trade Executor
Manages trade lifecycle: open, monitor, close, and record.
The Candlestick Trading Bible: Execute flawlessly — let the system work.

MT5 Hybrid Architecture:
  - Python (this module): Signal generation, risk checks, order parameters
  - MT5 Expert Advisor: Actual order placement and position management
  - Communication: Via shared file or named pipe (Phase 1: file-based)

Trade Lifecycle:
  1. Receive RiskApprovedOrder from M09
  2. In Backtest mode: simulate fill using historical candles
  3. In Paper mode: log order (no actual execution)
  4. In Live mode: send order to MT5 EA via IPC
  5. Monitor position: check SL/TP hit using candle data
  6. On close: calculate P&L, R-multiple, update risk state

Phase 1: Backtest and Paper modes only. Live mode is Phase 2.
Status: STUB — Phase 0 scaffold. Full implementation in Phase 1 Sprint 3.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional

from src.types import RiskApprovedOrder, AccountState

logger = logging.getLogger("candlestickbot.execution.trade_executor")


class TradeStatus(str, Enum):
    """Current status of a managed trade."""
    PENDING = "PENDING"             # Order created, not yet filled
    OPEN = "OPEN"                   # Order filled, position active
    CLOSED_SL = "CLOSED_SL"        # Closed by stop loss hit
    CLOSED_TP = "CLOSED_TP"        # Closed by take profit hit
    CLOSED_MANUAL = "CLOSED_MANUAL"  # Manually closed
    CANCELLED = "CANCELLED"        # Order cancelled before fill
    EXPIRED = "EXPIRED"            # Order expired (time limit)


class ExecutionMode(str, Enum):
    """Trading execution mode."""
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"


@dataclass
class Trade:
    """
    A managed trade record. Mirrors the Trade ORM model in db/models.py.
    Used in-memory during trade lifecycle before persistence.
    """
    trade_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str = ""
    timeframe: str = ""
    direction: str = ""         # "LONG" or "SHORT"
    strategy: str = ""
    status: TradeStatus = TradeStatus.PENDING

    # Order parameters
    entry_price: float = 0.0
    stop_price: float = 0.0
    target_price: float = 0.0
    lot_size: float = 0.0
    risk_pct: float = 0.0

    # Fill data
    fill_price: Optional[float] = None
    fill_time: Optional[datetime] = None
    slippage_pips: float = 0.0

    # Close data
    close_price: Optional[float] = None
    close_time: Optional[datetime] = None
    pnl_pips: float = 0.0
    pnl_usd: float = 0.0
    pnl_r: float = 0.0          # P&L in R-multiples

    # TQS context
    tqs_total: int = 0
    tqs_tier: str = ""

    # MT5 order ID (for paper/live modes)
    mt5_order_id: Optional[int] = None
    mt5_ticket: Optional[int] = None

    @property
    def is_open(self) -> bool:
        return self.status == TradeStatus.OPEN

    @property
    def is_closed(self) -> bool:
        return self.status in (
            TradeStatus.CLOSED_SL,
            TradeStatus.CLOSED_TP,
            TradeStatus.CLOSED_MANUAL,
        )

    @property
    def risk_distance(self) -> float:
        return abs(self.entry_price - self.stop_price)

    @property
    def rr_ratio(self) -> float:
        if self.risk_distance <= 0:
            return 0.0
        return abs(self.target_price - self.entry_price) / self.risk_distance


@dataclass
class BacktestFill:
    """Simulated fill result for backtesting."""
    filled: bool
    fill_price: float
    fill_time: datetime
    slippage_pips: float = 0.0
    reason: str = ""


class TradeExecutor:
    """
    M10 — Trade Executor.

    Phase 1 modes:
    - BACKTEST: Simulate fills using historical candle data
    - PAPER: Log orders with full P&L tracking, no real execution

    Phase 2 (deferred):
    - LIVE: Full MT5 EA integration via IPC

    Key Interfaces:
    - submit_order(RiskApprovedOrder) → Trade (PENDING state)
    - update_position(Trade, List[CandleData]) → Trade (OPEN or PENDING)
    - check_exit(Trade, candle) → Trade (CLOSED or OPEN)
    - record_trade(Trade, db_session) → Trade ORM row
    """

    # Default backtest slippage simulation (pips)
    DEFAULT_SLIPPAGE_PIPS = 0.5

    def __init__(
        self,
        execution_mode: ExecutionMode = ExecutionMode.BACKTEST,
        pip_size: float = 0.0001,
        pip_value_per_lot: float = 10.0,
        slippage_pips: float = 0.5,
        commission_per_lot: float = 7.0,  # Round-trip commission in USD
        audit_logger=None,
        db_session=None,
    ):
        self.execution_mode = execution_mode
        self.pip_size = pip_size
        self.pip_value_per_lot = pip_value_per_lot
        self.slippage_pips = slippage_pips
        self.commission_per_lot = commission_per_lot
        self.audit_logger = audit_logger
        self.db_session = db_session
        self._open_trades: List[Trade] = []

    @property
    def open_trades(self) -> List[Trade]:
        return list(self._open_trades)

    def submit_order(self, approved_order: RiskApprovedOrder) -> Trade:
        """
        Submit an approved order. Creates a PENDING trade.

        In BACKTEST mode: trade waits for price to reach entry
        In PAPER mode: immediately logs as pending

        Args:
            approved_order: Risk-approved order from M09

        Returns:
            Trade in PENDING status.
        """
        rec = approved_order.recommendation
        trade = Trade(
            symbol=rec.strategy,  # Fixed in Sprint 3: rec.symbol
            direction="LONG" if rec.entry_price > rec.stop_price else "SHORT",
            strategy=rec.strategy,
            entry_price=rec.entry_price,
            stop_price=rec.stop_price,
            target_price=rec.target_price,
            lot_size=approved_order.lot_size,
            risk_pct=approved_order.risk_pct,
            tqs_total=rec.tqs.total if rec.tqs else 0,
            tqs_tier=rec.tqs.tier if rec.tqs else "",
        )

        logger.info(
            f"Order submitted: {trade.trade_id} | "
            f"{trade.strategy} {trade.direction} @ {trade.entry_price}"
        )

        if self.audit_logger:
            self.audit_logger.log_order_event(
                event_type="ORDER_PLACED",
                order_id=trade.trade_id,
                symbol=trade.symbol,
                direction=trade.direction,
                lots=trade.lot_size,
                entry=trade.entry_price,
                sl=trade.stop_price,
                tp=trade.target_price,
                trade_id=trade.trade_id,
            )

        # TODO: Full implementation in Phase 1 Sprint 3
        logger.warning("TradeExecutor.submit_order() — STUB")
        return trade

    def simulate_fill(
        self,
        trade: Trade,
        next_candle_open: float,
        next_candle_time: datetime,
    ) -> BacktestFill:
        """
        Simulate order fill for backtesting.

        For limit orders (entry orders), fill occurs when price reaches entry level.
        In simplified D1 backtesting, we check next candle's range.

        Args:
            trade: Pending trade
            next_candle_open: Open price of next candle
            next_candle_time: Open time of next candle

        Returns:
            BacktestFill with simulated fill details.
        """
        slippage = self.slippage_pips * self.pip_size

        if trade.direction == "LONG":
            fill_price = trade.entry_price + slippage
        else:
            fill_price = trade.entry_price - slippage

        return BacktestFill(
            filled=True,
            fill_price=fill_price,
            fill_time=next_candle_time,
            slippage_pips=self.slippage_pips,
            reason="Simulated fill at entry + slippage",
        )

    def check_exit_backtest(
        self,
        trade: Trade,
        candle,
    ) -> Optional[TradeStatus]:
        """
        Check if a candle triggers SL or TP for a backtested trade.

        For LONG trades:
            - SL hit: candle.low <= stop_price
            - TP hit: candle.high >= target_price

        For SHORT trades:
            - SL hit: candle.high >= stop_price
            - TP hit: candle.low <= target_price

        Conservative assumption: if both SL and TP are hit in same candle,
        SL takes precedence (worst-case for backtesting integrity).

        Args:
            trade: Open trade
            candle: Current candle to check

        Returns:
            TradeStatus if exit triggered, None if still open.
        """
        if trade.direction == "LONG":
            sl_hit = candle.low <= trade.stop_price
            tp_hit = candle.high >= trade.target_price
        else:
            sl_hit = candle.high >= trade.stop_price
            tp_hit = candle.low <= trade.target_price

        if sl_hit:
            return TradeStatus.CLOSED_SL
        if tp_hit:
            return TradeStatus.CLOSED_TP
        return None

    def calculate_pnl(
        self,
        trade: Trade,
        close_price: float,
        lot_size: float,
    ) -> tuple:
        """
        Calculate trade P&L.

        Returns:
            Tuple of (pnl_pips, pnl_usd, pnl_r)
        """
        if trade.fill_price is None:
            return 0.0, 0.0, 0.0

        if trade.direction == "LONG":
            pnl_pips = (close_price - trade.fill_price) / self.pip_size
        else:
            pnl_pips = (trade.fill_price - close_price) / self.pip_size

        pnl_usd = pnl_pips * self.pip_value_per_lot * lot_size
        pnl_usd -= self.commission_per_lot * lot_size  # Deduct commission

        risk_pips = abs(trade.fill_price - trade.stop_price) / self.pip_size
        pnl_r = pnl_pips / risk_pips if risk_pips > 0 else 0.0

        return round(pnl_pips, 1), round(pnl_usd, 2), round(pnl_r, 2)
