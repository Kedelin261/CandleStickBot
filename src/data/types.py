"""
Shared data types / domain objects used across all modules.
These are pure Python dataclasses — no DB dependency.
Version: 3.1 (Phase 0)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional


# ===========================================================================
# ENUMERATIONS
# ===========================================================================

class Direction(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class TrendDirection(str, Enum):
    UP = "UP"
    DOWN = "DOWN"
    RANGING = "RANGING"
    CHOPPY = "CHOPPY"
    NONE = "NONE"


class RegimeType(str, Enum):
    TRENDING = "TRENDING"
    RANGING = "RANGING"
    VOLATILE = "VOLATILE"
    QUIET = "QUIET"
    CHOPPY = "CHOPPY"
    UNKNOWN = "UNKNOWN"


class PatternType(str, Enum):
    PIN_BAR_BULLISH = "PIN_BAR_BULLISH"
    PIN_BAR_BEARISH = "PIN_BAR_BEARISH"
    ENGULFING_BULLISH = "ENGULFING_BULLISH"
    ENGULFING_BEARISH = "ENGULFING_BEARISH"
    INSIDE_BAR = "INSIDE_BAR"
    INSIDE_BAR_BREAKOUT_BULL = "INSIDE_BAR_BREAKOUT_BULL"
    INSIDE_BAR_BREAKOUT_BEAR = "INSIDE_BAR_BREAKOUT_BEAR"
    FALSE_BREAKOUT_BULL = "FALSE_BREAKOUT_BULL"
    FALSE_BREAKOUT_BEAR = "FALSE_BREAKOUT_BEAR"


class StrategyType(str, Enum):
    PIN_BAR = "PIN_BAR"
    ENGULFING = "ENGULFING"
    INSIDE_BAR_BREAKOUT = "INSIDE_BAR_BREAKOUT"
    INSIDE_BAR_FALSE_BREAKOUT = "INSIDE_BAR_FALSE_BREAKOUT"


class TradeTier(str, Enum):
    REJECT = "REJECT"       # TQS 0-59
    STANDARD = "STANDARD"   # TQS 60-79
    PREMIUM = "PREMIUM"     # TQS 80-100


class LevelType(str, Enum):
    SWING_SR = "SWING_SR"
    SUPPLY_DEMAND = "SUPPLY_DEMAND"
    FIB_618 = "FIB_618"
    FIB_50 = "FIB_50"
    SMA_21 = "SMA_21"
    MINOR_SWING = "MINOR_SWING"


class LevelDirection(str, Enum):
    SUPPORT = "SUPPORT"
    RESISTANCE = "RESISTANCE"
    BOTH = "BOTH"


# ===========================================================================
# CANDLE DATA CONTRACT
# ===========================================================================

@dataclass
class CandleData:
    """
    Canonical candle object — data contract for all modules.
    Every candle = {timestamp, open, high, low, close, volume, spread, symbol, timeframe}
    """
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str
    timeframe: str
    spread: Optional[float] = None
    tick_volume: Optional[float] = None

    # Derived properties
    @property
    def body_size(self) -> float:
        return abs(self.close - self.open)

    @property
    def total_range(self) -> float:
        return self.high - self.low

    @property
    def upper_wick(self) -> float:
        return self.high - max(self.open, self.close)

    @property
    def lower_wick(self) -> float:
        return min(self.open, self.close) - self.low

    @property
    def is_bullish(self) -> bool:
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
        return self.close < self.open

    @property
    def is_doji(self) -> bool:
        return self.body_size == 0.0 or (
            self.total_range > 0 and self.body_size / self.total_range < 0.05
        )

    @property
    def midpoint(self) -> float:
        return (self.high + self.low) / 2.0

    @property
    def body_center(self) -> float:
        return (self.open + self.close) / 2.0

    def __repr__(self) -> str:
        direction = "▲" if self.is_bullish else "▼"
        return (
            f"<Candle {direction} {self.symbol} {self.timeframe} "
            f"{self.timestamp.strftime('%Y-%m-%d')} "
            f"O={self.open:.5f} H={self.high:.5f} L={self.low:.5f} C={self.close:.5f}>"
        )


# ===========================================================================
# MARKET STRUCTURE
# ===========================================================================

@dataclass
class SwingPointData:
    """Detected swing high or swing low."""
    timestamp: datetime
    price: float
    swing_type: str         # "HIGH" or "LOW"
    lookback: int = 5


@dataclass
class MarketStructure:
    """
    Output of M03 Market Structure Engine.
    Contains swing points and derived market structure.
    """
    regime: TrendDirection
    swing_highs: List[SwingPointData] = field(default_factory=list)
    swing_lows: List[SwingPointData] = field(default_factory=list)
    last_hh: Optional[float] = None     # Last Higher High price
    last_hl: Optional[float] = None     # Last Higher Low price
    last_lh: Optional[float] = None     # Last Lower High price
    last_ll: Optional[float] = None     # Last Lower Low price

    @property
    def has_sufficient_points(self) -> bool:
        """At least 2 swing highs and 2 swing lows confirmed."""
        return len(self.swing_highs) >= 2 and len(self.swing_lows) >= 2


# ===========================================================================
# TREND SIGNAL
# ===========================================================================

@dataclass
class TrendSignal:
    """
    Output of M04 Trend Detection Engine.
    Governs GATE 1 (Trend Gate) for all strategies.
    """
    direction: TrendDirection
    strength: float             # 0.0-1.0 (weak to strong)
    tradeable: bool
    reason: str
    ma_value: float             # 21 SMA value at evaluation time
    adx: Optional[float] = None
    price_vs_ma: str = "ABOVE"  # "ABOVE" | "BELOW" | "AT"

    # TQS contribution (0-25 pts) — computed by M08
    tqs_trend_score: int = 0


# ===========================================================================
# MARKET REGIME
# ===========================================================================

@dataclass
class RegimeSignal:
    """
    Output of M16 Market Regime Engine.
    Controls which strategies are allowed and risk multiplier.
    """
    regime: RegimeType
    confidence: float           # 0.0-1.0
    allowed_strategies: List[StrategyType]
    risk_multiplier: float      # 0.0 = no trades; 1.0 = full risk

    # Indicator values at classification
    adx: float = 0.0
    atr: float = 0.0
    atr_ma: float = 0.0
    bb_width: float = 0.0
    bb_width_ma: float = 0.0
    choppiness_index: float = 0.0

    # TQS contribution (0-25 pts) — computed by M08
    tqs_regime_score: int = 0

    @property
    def is_tradeable(self) -> bool:
        """VOLATILE and CHOPPY regimes are never tradeable."""
        return self.risk_multiplier > 0.0


# ===========================================================================
# SUPPORT / RESISTANCE
# ===========================================================================

@dataclass
class SRLevelData:
    """
    Support / Resistance level from M05 S/R Engine.
    """
    price: float
    level_type: LevelType
    direction: LevelDirection
    strength_score: int         # 1-10
    touch_count: int
    zone_high: float
    zone_low: float
    last_tested: Optional[datetime] = None
    is_rts: bool = False        # Resistance-turned-support (or vice versa)
    age_bars: int = 0

    @property
    def is_qualified(self) -> bool:
        """Level qualifies if strength_score >= 3 (configurable)."""
        return self.strength_score >= 3

    @property
    def zone_width_pips(self) -> float:
        return (self.zone_high - self.zone_low) * 10000  # Approx for 5-digit broker


# ===========================================================================
# PATTERN SIGNAL
# ===========================================================================

@dataclass
class PatternSignal:
    """
    Output of M07 Candlestick Pattern Engine.
    Contains detection result and entry/stop parameters.
    """
    pattern_type: PatternType
    direction: Direction
    quality_score: int          # 1-10
    entry_price: float
    stop_price: float
    invalidation_price: float

    # Pattern math (for audit log)
    body_size: float = 0.0
    upper_wick: float = 0.0
    lower_wick: float = 0.0
    total_range: float = 0.0
    tail_ratio: Optional[float] = None      # Pin bar specific
    engulf_pct: Optional[float] = None      # Engulfing specific

    # Candle timestamps
    signal_candle_ts: Optional[datetime] = None
    prior_candle_ts: Optional[datetime] = None


# ===========================================================================
# TRADE QUALITY SCORE
# ===========================================================================

@dataclass
class TQSResult:
    """
    Trade Quality Score computation result (Section 2a).
    Four components of equal weight (0-25 each) = 0-100 total.
    """
    trend_score: int = 0        # 0-25
    level_score: int = 0        # 0-25
    pattern_score: int = 0      # 0-25
    regime_score: int = 0       # 0-25

    @property
    def total(self) -> int:
        return self.trend_score + self.level_score + self.pattern_score + self.regime_score

    @property
    def tier(self) -> TradeTier:
        if self.total >= 80:
            return TradeTier.PREMIUM
        elif self.total >= 60:
            return TradeTier.STANDARD
        else:
            return TradeTier.REJECT

    @property
    def is_tradeable(self) -> bool:
        """Auto-reject if regime_score == 0 (VOLATILE/CHOPPY)."""
        return self.total >= 60 and self.regime_score > 0

    def __repr__(self) -> str:
        return (
            f"<TQS {self.total}/100 [{self.tier.value}] "
            f"T={self.trend_score} L={self.level_score} "
            f"P={self.pattern_score} R={self.regime_score}>"
        )


# ===========================================================================
# TRADE RECOMMENDATION
# ===========================================================================

@dataclass
class TradeRecommendation:
    """
    Output of M08 Strategy Engine.
    Full trade recommendation passed to M17 Portfolio Engine → M09 Risk Engine.
    """
    strategy: StrategyType
    symbol: str
    timeframe: str
    direction: Direction
    entry_price: float
    stop_price: float
    target_price: float
    rr_ratio: float
    tqs: TQSResult
    trade_tier: TradeTier
    regime: RegimeType
    trend_direction: TrendDirection
    signal: PatternSignal
    levels_used: List[SRLevelData] = field(default_factory=list)
    signal_id: Optional[str] = None
    timestamp: Optional[datetime] = None

    @property
    def stop_distance_pips(self) -> float:
        return abs(self.entry_price - self.stop_price) * 10000  # 5-digit broker

    @property
    def target_distance_pips(self) -> float:
        return abs(self.target_price - self.entry_price) * 10000


# ===========================================================================
# RISK APPROVED ORDER
# ===========================================================================

@dataclass
class RiskApprovedOrder:
    """
    Output of M09 Risk Engine — approved trade ready for execution.
    """
    recommendation: TradeRecommendation
    lots: float
    risk_amount: float
    risk_pct: float
    approved: bool
    rejection_reason: Optional[str] = None

    @property
    def is_approved(self) -> bool:
        return self.approved


@dataclass
class RiskRejection:
    """Risk engine rejection result with full reason."""
    approved: bool = False
    rejection_reason: str = ""
    gate_failed: str = ""       # DAILY_LIMIT | WEEKLY_LIMIT | KILL_SWITCH | RR | etc.
