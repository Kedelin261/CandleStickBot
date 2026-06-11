"""
M02 — Candle Storage: SQLAlchemy ORM Models
Full database schema for CandleStickBot Phase 0.
Covers: candles, trades, signals, strategy performance, audit events.
Version: 3.1 (Phase 0)
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum as PyEnum

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    Enum,
    UniqueConstraint,
    event,
)
from sqlalchemy.orm import DeclarativeBase, relationship
from sqlalchemy.sql import func


# ===========================================================================
# BASE MODEL
# ===========================================================================

class Base(DeclarativeBase):
    """SQLAlchemy declarative base for all CandleStickBot models."""
    pass


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ===========================================================================
# ENUMERATIONS (mirrored in Python Enum for type safety)
# ===========================================================================

class TradeDirectionEnum(str, PyEnum):
    LONG = "LONG"
    SHORT = "SHORT"


class TradeTierEnum(str, PyEnum):
    REJECT = "REJECT"
    STANDARD = "STANDARD"
    PREMIUM = "PREMIUM"


class RegimeTypeEnum(str, PyEnum):
    TRENDING = "TRENDING"
    RANGING = "RANGING"
    VOLATILE = "VOLATILE"
    QUIET = "QUIET"
    CHOPPY = "CHOPPY"
    UNKNOWN = "UNKNOWN"


class TrendDirectionEnum(str, PyEnum):
    UP = "UP"
    DOWN = "DOWN"
    RANGING = "RANGING"
    CHOPPY = "CHOPPY"
    NONE = "NONE"


class ExitReasonEnum(str, PyEnum):
    TP_HIT = "TP_HIT"
    SL_HIT = "SL_HIT"
    MANUAL = "MANUAL"
    EXPIRED = "EXPIRED"
    KILL_SWITCH = "KILL_SWITCH"


class LossCategory(str, PyEnum):
    BAD_SIGNAL = "BAD_SIGNAL"
    BAD_REGIME = "BAD_REGIME"
    BAD_LEVEL = "BAD_LEVEL"
    BAD_EXECUTION = "BAD_EXECUTION"
    NORMAL_STATISTICAL_LOSS = "NORMAL_STATISTICAL_LOSS"
    OVERRIDDEN = "OVERRIDDEN"
    UNCLASSIFIED = "UNCLASSIFIED"


class SignalStatusEnum(str, PyEnum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED_MANUAL = "REJECTED_MANUAL"
    EXPIRED = "EXPIRED"
    EXECUTED = "EXECUTED"


class ExecutionModeEnum(str, PyEnum):
    BACKTEST = "backtest"
    PAPER = "paper"
    DEMO = "demo"
    LIVE = "live"


# ===========================================================================
# M02 — CANDLE STORAGE
# ===========================================================================

class Candle(Base):
    """
    OHLCV candle data.
    Core data contract: {timestamp, open, high, low, close, volume, spread, symbol, timeframe}
    INDEX ON (symbol, timeframe, timestamp) for fast lookup.
    """
    __tablename__ = "candles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False)
    timeframe = Column(String(10), nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False, default=0.0)
    spread = Column(Float, nullable=True)               # Bid/Ask spread in pips
    tick_volume = Column(Float, nullable=True)          # Tick count (MT5)
    created_at = Column(DateTime(timezone=True), default=utcnow)

    # Constraints
    __table_args__ = (
        UniqueConstraint("symbol", "timeframe", "timestamp", name="uq_candle"),
        Index("idx_candle_lookup", "symbol", "timeframe", "timestamp"),
        Index("idx_candle_symbol_tf", "symbol", "timeframe"),
    )

    def __repr__(self) -> str:
        return (
            f"<Candle {self.symbol} {self.timeframe} "
            f"{self.timestamp} O={self.open} H={self.high} "
            f"L={self.low} C={self.close}>"
        )


# ===========================================================================
# M03 / M04 — MARKET STRUCTURE & TREND
# ===========================================================================

class SwingPoint(Base):
    """
    Detected swing highs and lows for market structure analysis.
    Used by M03 Market Structure Engine.
    """
    __tablename__ = "swing_points"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False)
    timeframe = Column(String(10), nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    price = Column(Float, nullable=False)
    swing_type = Column(String(5), nullable=False)      # HIGH | LOW
    lookback = Column(Integer, nullable=False, default=5)
    created_at = Column(DateTime(timezone=True), default=utcnow)

    __table_args__ = (
        Index("idx_swing_lookup", "symbol", "timeframe", "timestamp"),
        Index("idx_swing_type", "symbol", "timeframe", "swing_type"),
    )


class SRLevel(Base):
    """
    Support / Resistance levels detected by M05 S/R Engine.
    Includes strength scoring and zone boundaries.
    """
    __tablename__ = "sr_levels"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False)
    timeframe = Column(String(10), nullable=False)
    price = Column(Float, nullable=False)
    level_type = Column(String(30), nullable=False)     # SWING_SR | SUPPLY_DEMAND | FIB_618 | FIB_50 | SMA_21
    direction = Column(String(10), nullable=False)      # SUPPORT | RESISTANCE | BOTH
    strength_score = Column(Integer, nullable=False)    # 1-10
    touch_count = Column(Integer, nullable=False, default=1)
    zone_high = Column(Float, nullable=False)
    zone_low = Column(Float, nullable=False)
    first_seen = Column(DateTime(timezone=True), nullable=False)
    last_tested = Column(DateTime(timezone=True), nullable=True)
    is_active = Column(Boolean, nullable=False, default=True)
    is_rts = Column(Boolean, nullable=False, default=False)   # Resistance-Turned-Support
    age_bars = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime(timezone=True), default=utcnow)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    __table_args__ = (
        Index("idx_sr_level_lookup", "symbol", "timeframe", "is_active"),
    )


# ===========================================================================
# M07 — PATTERN ENGINE
# ===========================================================================

class PatternDetection(Base):
    """
    Record of all detected and rejected candlestick patterns.
    Every pattern evaluation is logged (M13 integration).
    """
    __tablename__ = "pattern_detections"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False)
    timeframe = Column(String(10), nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False)   # Candle close timestamp
    pattern_type = Column(String(40), nullable=False)             # PIN_BAR_BULLISH | ENGULFING_BULLISH | etc.
    direction = Column(Enum(TradeDirectionEnum), nullable=True)
    quality_score = Column(Integer, nullable=True)                # 1-10
    detected = Column(Boolean, nullable=False, default=True)
    rejection_reason = Column(String(200), nullable=True)

    # Pattern math (stored for audit)
    body_size = Column(Float, nullable=True)
    upper_wick = Column(Float, nullable=True)
    lower_wick = Column(Float, nullable=True)
    total_range = Column(Float, nullable=True)
    tail_ratio = Column(Float, nullable=True)                     # For pin bar

    entry_price = Column(Float, nullable=True)
    stop_price = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow)

    __table_args__ = (
        Index("idx_pattern_lookup", "symbol", "timeframe", "timestamp"),
        Index("idx_pattern_type", "pattern_type", "detected"),
    )


# ===========================================================================
# M08 — STRATEGY ENGINE (Trade Signals)
# ===========================================================================

class TradeSignal(Base):
    """
    Full trade recommendation from M08 Strategy Engine.
    Includes TQS components, tier classification, and entry parameters.
    """
    __tablename__ = "trade_signals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    signal_id = Column(String(50), unique=True, nullable=False)   # UUID
    symbol = Column(String(20), nullable=False)
    timeframe = Column(String(10), nullable=False)
    strategy = Column(String(40), nullable=False)                 # PIN_BAR | ENGULFING | etc.
    direction = Column(Enum(TradeDirectionEnum), nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False)   # Signal generation time

    # Trade parameters
    entry_price = Column(Float, nullable=False)
    stop_price = Column(Float, nullable=False)
    target_price = Column(Float, nullable=False)
    rr_ratio = Column(Float, nullable=False)

    # TQS scoring
    tqs_total = Column(Integer, nullable=False)                   # 0-100
    tqs_trend = Column(Integer, nullable=False)                   # 0-25
    tqs_level = Column(Integer, nullable=False)                   # 0-25
    tqs_pattern = Column(Integer, nullable=False)                 # 0-25
    tqs_regime = Column(Integer, nullable=False)                  # 0-25
    trade_tier = Column(Enum(TradeTierEnum), nullable=False)

    # Market context at signal time
    regime = Column(Enum(RegimeTypeEnum), nullable=False)
    trend_direction = Column(Enum(TrendDirectionEnum), nullable=False)

    # Execution tracking
    status = Column(Enum(SignalStatusEnum), nullable=False, default=SignalStatusEnum.PENDING)
    lots = Column(Float, nullable=True)
    risk_amount = Column(Float, nullable=True)
    execution_mode = Column(Enum(ExecutionModeEnum), nullable=False)

    # Approval (manual mode)
    approved_at = Column(DateTime(timezone=True), nullable=True)
    approved_by = Column(String(100), nullable=True)
    rejection_reason = Column(String(500), nullable=True)
    expiry_at = Column(DateTime(timezone=True), nullable=True)

    created_at = Column(DateTime(timezone=True), default=utcnow)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    # Relationship to trade
    trade = relationship("Trade", back_populates="signal", uselist=False)

    __table_args__ = (
        Index("idx_signal_status", "status", "execution_mode"),
        Index("idx_signal_symbol", "symbol", "timestamp"),
    )


# ===========================================================================
# M10 — TRADE EXECUTION ENGINE
# ===========================================================================

class Trade(Base):
    """
    Executed trade record. Links to signal, tracks lifecycle from entry to close.
    Core table for M18 Strategy Analytics and M19 Trade Review.
    """
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    trade_id = Column(String(50), unique=True, nullable=False)    # UUID
    signal_id = Column(String(50), ForeignKey("trade_signals.signal_id"), nullable=True)

    symbol = Column(String(20), nullable=False)
    timeframe = Column(String(10), nullable=False)
    strategy = Column(String(40), nullable=False)
    direction = Column(Enum(TradeDirectionEnum), nullable=False)
    execution_mode = Column(Enum(ExecutionModeEnum), nullable=False)

    # Entry details
    timestamp_entry = Column(DateTime(timezone=True), nullable=False)
    entry_price = Column(Float, nullable=False)
    fill_price = Column(Float, nullable=True)                     # Actual fill (may differ from entry)
    entry_slippage_pips = Column(Float, nullable=True)
    lot_size = Column(Float, nullable=False)
    risk_amount = Column(Float, nullable=False)

    # Stop / Target
    sl_price = Column(Float, nullable=False)
    tp_price = Column(Float, nullable=False)
    sl_distance_pips = Column(Float, nullable=True)
    rr_ratio = Column(Float, nullable=False)

    # TQS at entry
    tqs_total = Column(Integer, nullable=True)
    trade_tier = Column(Enum(TradeTierEnum), nullable=True)
    regime_at_entry = Column(Enum(RegimeTypeEnum), nullable=True)
    market_structure_at_entry = Column(String(20), nullable=True)

    # Exit details
    timestamp_exit = Column(DateTime(timezone=True), nullable=True)
    exit_price = Column(Float, nullable=True)
    exit_slippage_pips = Column(Float, nullable=True)
    exit_reason = Column(Enum(ExitReasonEnum), nullable=True)

    # P&L
    pnl_pips = Column(Float, nullable=True)
    pnl_usd = Column(Float, nullable=True)
    r_multiple = Column(Float, nullable=True)                     # e.g., 2.0 = 2R winner

    # State
    is_open = Column(Boolean, nullable=False, default=True)
    is_winner = Column(Boolean, nullable=True)

    # M19 Loss classification
    loss_category = Column(Enum(LossCategory), nullable=True)
    loss_classified_at = Column(DateTime(timezone=True), nullable=True)

    # Broker order tracking
    broker_order_id = Column(String(100), nullable=True)
    broker_fill_price = Column(Float, nullable=True)

    created_at = Column(DateTime(timezone=True), default=utcnow)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    # Relationships
    signal = relationship("TradeSignal", back_populates="trade")

    __table_args__ = (
        Index("idx_trade_open", "is_open", "execution_mode"),
        Index("idx_trade_symbol", "symbol", "timestamp_entry"),
        Index("idx_trade_strategy", "strategy", "execution_mode"),
    )

    def __repr__(self) -> str:
        return (
            f"<Trade {self.trade_id} {self.symbol} {self.direction.value} "
            f"{'OPEN' if self.is_open else f'{self.r_multiple}R'}>"
        )


# ===========================================================================
# M18 — STRATEGY ANALYTICS ENGINE
# ===========================================================================

class StrategyPerformance(Base):
    """
    Per-strategy performance summary.
    Updated after every trade close.
    Feeds the Strategy Scorecard (M18).
    """
    __tablename__ = "strategy_performance"

    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy_name = Column(String(40), nullable=False)
    symbol = Column(String(20), nullable=False)
    timeframe = Column(String(10), nullable=False)
    execution_mode = Column(Enum(ExecutionModeEnum), nullable=False)

    # Aggregate statistics
    total_trades = Column(Integer, nullable=False, default=0)
    win_count = Column(Integer, nullable=False, default=0)
    loss_count = Column(Integer, nullable=False, default=0)
    win_rate = Column(Float, nullable=True)                       # 0.0-1.0
    profit_factor = Column(Float, nullable=True)
    expectancy_r = Column(Float, nullable=True)                   # R per trade
    avg_winner_r = Column(Float, nullable=True)
    avg_loser_r = Column(Float, nullable=True)
    max_drawdown_pct = Column(Float, nullable=True)
    sharpe_ratio = Column(Float, nullable=True)
    sortino_ratio = Column(Float, nullable=True)

    # Rolling windows
    recent_30d_pf = Column(Float, nullable=True)
    recent_90d_pf = Column(Float, nullable=True)

    # State
    is_enabled = Column(Boolean, nullable=False, default=True)
    disable_reason = Column(String(200), nullable=True)

    last_updated = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)
    created_at = Column(DateTime(timezone=True), default=utcnow)

    __table_args__ = (
        UniqueConstraint(
            "strategy_name", "symbol", "timeframe", "execution_mode",
            name="uq_strategy_perf"
        ),
        Index("idx_strategy_perf", "strategy_name", "execution_mode"),
    )


# ===========================================================================
# M19 — TRADE REVIEW ENGINE
# ===========================================================================

class MonthlyFailureReport(Base):
    """
    Monthly failure analysis report (M19 Trade Review Engine).
    Tracks loss categories and generates parameter improvement suggestions.
    """
    __tablename__ = "monthly_failure_reports"

    id = Column(Integer, primary_key=True, autoincrement=True)
    report_month = Column(String(7), nullable=False)              # "2024-01"
    execution_mode = Column(Enum(ExecutionModeEnum), nullable=False)

    # Loss category counts
    total_losses = Column(Integer, nullable=False, default=0)
    bad_signal_count = Column(Integer, nullable=False, default=0)
    bad_regime_count = Column(Integer, nullable=False, default=0)
    bad_level_count = Column(Integer, nullable=False, default=0)
    bad_execution_count = Column(Integer, nullable=False, default=0)
    normal_statistical_count = Column(Integer, nullable=False, default=0)
    overridden_count = Column(Integer, nullable=False, default=0)

    # Derived percentages
    bad_signal_pct = Column(Float, nullable=True)
    bad_regime_pct = Column(Float, nullable=True)
    bad_level_pct = Column(Float, nullable=True)
    bad_execution_pct = Column(Float, nullable=True)
    normal_statistical_pct = Column(Float, nullable=True)

    # Recommendations
    top_issue = Column(String(40), nullable=True)
    recommended_action = Column(Text, nullable=True)
    has_systematic_issue = Column(Boolean, nullable=False, default=False)

    created_at = Column(DateTime(timezone=True), default=utcnow)

    __table_args__ = (
        UniqueConstraint("report_month", "execution_mode", name="uq_monthly_report"),
    )


# ===========================================================================
# M13 — AUDIT EVENTS TABLE
# ===========================================================================

class AuditEvent(Base):
    """
    Database-persisted audit log.
    Every decision logged here in addition to file-based structlog output.
    Schema: {timestamp, module, event_type, payload, trade_id?}
    """
    __tablename__ = "audit_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    module = Column(String(10), nullable=False)                   # M01-M19
    event_type = Column(String(50), nullable=False)
    log_level = Column(String(10), nullable=False, default="INFO")
    payload = Column(Text, nullable=True)                         # JSON string
    trade_id = Column(String(50), nullable=True)
    symbol = Column(String(20), nullable=True)

    __table_args__ = (
        Index("idx_audit_timestamp", "timestamp"),
        Index("idx_audit_event_type", "event_type"),
        Index("idx_audit_trade", "trade_id"),
    )


# ===========================================================================
# BOT STATE / KILL SWITCH
# ===========================================================================

class BotState(Base):
    """
    Persistent bot state tracking.
    Kill switch state, current mode, daily/weekly loss tracking.
    """
    __tablename__ = "bot_state"

    id = Column(Integer, primary_key=True, default=1)              # Singleton row
    execution_mode = Column(Enum(ExecutionModeEnum), nullable=False, default=ExecutionModeEnum.BACKTEST)
    is_halted = Column(Boolean, nullable=False, default=False)     # Kill switch state
    halt_reason = Column(String(500), nullable=True)
    halted_at = Column(DateTime(timezone=True), nullable=True)

    # Daily tracking
    day_open_balance = Column(Float, nullable=True)
    day_realized_loss = Column(Float, nullable=False, default=0.0)

    # Weekly tracking
    week_open_balance = Column(Float, nullable=True)
    week_realized_loss = Column(Float, nullable=False, default=0.0)

    # Drawdown tracking
    peak_equity = Column(Float, nullable=True)
    current_drawdown_pct = Column(Float, nullable=False, default=0.0)

    # Consecutive losses
    consecutive_losses = Column(Integer, nullable=False, default=0)

    last_updated = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)
