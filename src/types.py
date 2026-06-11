"""
Shared domain types and data contracts for CandleStickBot.
These are the primary data transfer objects (DTOs) passed between modules.
Version: 3.1 (Phase 0)
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


# ===========================================================================
# ENUMERATIONS
# ===========================================================================

class Direction(str, enum.Enum):
    LONG  = "LONG"
    SHORT = "SHORT"


class RegimeType(str, enum.Enum):
    TRENDING = "TRENDING"
    RANGING  = "RANGING"
    VOLATILE = "VOLATILE"
    QUIET    = "QUIET"
    CHOPPY   = "CHOPPY"
    UNKNOWN  = "UNKNOWN"


class TrendDirection(str, enum.Enum):
    UP   = "UP"
    DOWN = "DOWN"
    NONE = "NONE"   # Choppy / no trend


class TradeTier(str, enum.Enum):
    REJECT   = "REJECT"    # TQS 0-59
    STANDARD = "STANDARD"  # TQS 60-79
    PREMIUM  = "PREMIUM"   # TQS 80-100


class PatternType(str, enum.Enum):
    PIN_BAR_BULLISH           = "PIN_BAR_BULLISH"
    PIN_BAR_BEARISH           = "PIN_BAR_BEARISH"
    ENGULFING_BULLISH         = "ENGULFING_BULLISH"
    ENGULFING_BEARISH         = "ENGULFING_BEARISH"
    INSIDE_BAR                = "INSIDE_BAR"
    INSIDE_BAR_BREAKOUT_BULL  = "INSIDE_BAR_BREAKOUT_BULL"
    INSIDE_BAR_BREAKOUT_BEAR  = "INSIDE_BAR_BREAKOUT_BEAR"
    FALSE_BREAKOUT_BULLISH    = "FALSE_BREAKOUT_BULLISH"
    FALSE_BREAKOUT_BEARISH    = "FALSE_BREAKOUT_BEARISH"


class LevelType(str, enum.Enum):
    SWING_SR            = "SWING_SR"
    SMA_21              = "SMA_21"
    SUPPLY_DEMAND_ZONE  = "SUPPLY_DEMAND_ZONE"
    FIBONACCI_618       = "FIBONACCI_618"
    FIBONACCI_50        = "FIBONACCI_50"
    MINOR_SWING         = "MINOR_SWING"


class StrategyName(str, enum.Enum):
    PIN_BAR         = "pin_bar"
    ENGULFING_BAR   = "engulfing_bar"
    INSIDE_BAR      = "inside_bar"
    FALSE_BREAKOUT  = "inside_bar_false_breakout"


class LossCategory(str, enum.Enum):
    BAD_SIGNAL          = "BAD_SIGNAL"
    BAD_REGIME          = "BAD_REGIME"
    BAD_LEVEL           = "BAD_LEVEL"
    BAD_EXECUTION       = "BAD_EXECUTION"
    NORMAL_STATISTICAL  = "NORMAL_STATISTICAL_LOSS"
    OVERRIDDEN          = "OVERRIDDEN"
    UNCATEGORIZED       = "UNCATEGORIZED"


# ===========================================================================
# DATA CONTRACTS (DTOs)
# ===========================================================================

@dataclass
class CandleData:
    """
    Canonical candle representation used across all analysis modules.
    Matches M01 data contract: {timestamp, open, high, low, close, volume, spread, symbol, timeframe}
    """
    symbol:    str
    timeframe: str
    timestamp: datetime
    open:      float
    high:      float
    low:       float
    close:     float
    volume:    float = 0.0
    spread:    Optional[float] = None

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
    def midpoint(self) -> float:
        return (self.high + self.low) / 2.0

    @property
    def body_center(self) -> float:
        return (self.open + self.close) / 2.0

    def __repr__(self) -> str:
        direction = "▲" if self.is_bullish else "▼"
        return (
            f"<Candle {self.symbol}/{self.timeframe} {direction} "
            f"{self.timestamp.strftime('%Y-%m-%d')} "
            f"O={self.open:.5f} H={self.high:.5f} L={self.low:.5f} C={self.close:.5f}>"
        )


@dataclass
class SwingPointData:
    """Identified swing high or swing low from M03 Market Structure Engine."""
    timestamp:  datetime
    price:      float
    swing_type: str          # "HIGH" or "LOW"
    symbol:     str
    timeframe:  str
    lookback:   int = 5


@dataclass
class MarketStructure:
    """
    Output of M03 Market Structure Engine.
    Contains all identified swing points and derived structure.
    """
    symbol:      str
    timeframe:   str
    timestamp:   datetime
    swing_highs: List[SwingPointData] = field(default_factory=list)
    swing_lows:  List[SwingPointData] = field(default_factory=list)
    # Derived properties (populated by classify_regime)
    last_hh:     Optional[float] = None   # Last Higher High price
    last_hl:     Optional[float] = None   # Last Higher Low price
    last_lh:     Optional[float] = None   # Last Lower High price
    last_ll:     Optional[float] = None   # Last Lower Low price
    regime:      TrendDirection = TrendDirection.NONE

    @property
    def has_enough_swings(self) -> bool:
        return len(self.swing_highs) >= 2 and len(self.swing_lows) >= 2


@dataclass
class TrendSignal:
    """
    Output of M04 Trend Detection Engine.
    Determines if a tradeable trend exists and in which direction.
    """
    symbol:     str
    timeframe:  str
    timestamp:  datetime
    direction:  TrendDirection
    sma21:      float                   # Current 21 SMA value
    tradeable:  bool                    # Is the trend strong enough to trade?
    reason:     str = ""                # Why it is/isn't tradeable
    adx:        Optional[float] = None  # ADX value if computed
    strength:   float = 0.0             # 0-1 normalized trend strength

    @property
    def is_bullish(self) -> bool:
        return self.direction == TrendDirection.UP and self.tradeable

    @property
    def is_bearish(self) -> bool:
        return self.direction == TrendDirection.DOWN and self.tradeable


@dataclass
class RegimeSignal:
    """
    Output of M16 Market Regime Engine.
    Classifies market conditions and controls strategy allowance.
    """
    symbol:             str
    timeframe:          str
    timestamp:          datetime
    regime:             RegimeType
    confidence:         float = 1.0              # 0-1 confidence in classification
    allowed_strategies: List[str] = field(default_factory=list)
    risk_multiplier:    float = 1.0              # Applied to position size
    adx:                Optional[float] = None
    atr:                Optional[float] = None
    atr_ma:             Optional[float] = None
    bb_width:           Optional[float] = None
    choppiness_index:   Optional[float] = None

    @property
    def is_tradeable(self) -> bool:
        """Returns True if any strategies are allowed in this regime."""
        return len(self.allowed_strategies) > 0 and self.risk_multiplier > 0.0

    def is_strategy_allowed(self, strategy_name: str) -> bool:
        """Check if a specific strategy is allowed in this regime."""
        return strategy_name in self.allowed_strategies


@dataclass
class LevelData:
    """
    A qualified Support/Resistance level from M05 S/R Engine.
    """
    price:                    float
    level_type:               LevelType
    strength_score:           float         # 1-10
    touch_count:              int
    zone_high:                float
    zone_low:                 float
    last_tested:              Optional[datetime] = None
    is_resistance_turned_support: bool = False

    def contains_price(self, price: float) -> bool:
        """Returns True if price is within the level zone."""
        return self.zone_low <= price <= self.zone_high

    def distance_pips(self, price: float, pip_size: float = 0.0001) -> float:
        """Distance from price to level center in pips."""
        return abs(price - self.price) / pip_size

    def __repr__(self) -> str:
        return (
            f"<Level {self.level_type.value} @ {self.price:.5f} "
            f"strength={self.strength_score:.1f} touches={self.touch_count}>"
        )


@dataclass
class PatternSignal:
    """
    Output of M07 Candlestick Pattern Engine.
    Contains pattern detection result with quality scoring.
    """
    pattern_type:        PatternType
    direction:           Direction
    quality_score:       float           # 1-10
    candle_timestamp:    datetime
    symbol:              str
    timeframe:           str
    # Entry / stop prices suggested by pattern
    suggested_entry:     float
    suggested_stop:      float
    invalidation_price:  float           # Pattern invalid if price crosses this
    # Optional context
    details:             Dict = field(default_factory=dict)
    # e.g. tail_ratio, body_size, engulf_pct, mother_bar prices

    @property
    def is_bullish(self) -> bool:
        return self.direction == Direction.LONG

    @property
    def is_bearish(self) -> bool:
        return self.direction == Direction.SHORT


@dataclass
class TQSComponents:
    """
    Trade Quality Score component breakdown.
    Each component: 0-25 points. Total: 0-100.
    See Section 2a of planning document.
    """
    trend_score:   float = 0.0   # 0-25: Trend Strength
    level_score:   float = 0.0   # 0-25: Level Strength
    pattern_score: float = 0.0   # 0-25: Pattern Quality
    regime_score:  float = 0.0   # 0-25: Market Regime

    @property
    def total(self) -> float:
        return self.trend_score + self.level_score + self.pattern_score + self.regime_score

    @property
    def tier(self) -> TradeTier:
        t = self.total
        if t < 60:
            return TradeTier.REJECT
        elif t < 80:
            return TradeTier.STANDARD
        else:
            return TradeTier.PREMIUM

    @property
    def is_auto_rejected(self) -> bool:
        """
        Automatic rejection conditions (Section 2a):
        - Regime score = 0 (VOLATILE or CHOPPY) → always REJECT
        - Total < 60 → REJECT
        """
        return self.regime_score == 0.0 or self.total < 60.0

    @property
    def all_components_above_threshold(self, threshold: float = 15.0) -> bool:
        """
        Check if all components >= threshold.
        Required for PREMIUM risk increase (Section 2a).
        """
        return (
            self.trend_score >= threshold
            and self.level_score >= threshold
            and self.pattern_score >= threshold
            and self.regime_score >= threshold
        )

    def as_dict(self) -> Dict[str, float]:
        return {
            "trend":   self.trend_score,
            "level":   self.level_score,
            "pattern": self.pattern_score,
            "regime":  self.regime_score,
            "total":   self.total,
        }


@dataclass
class TradeRecommendation:
    """
    Output of M08 Strategy Engine.
    Complete trade specification ready for risk engine evaluation.
    """
    strategy:       StrategyName
    symbol:         str
    timeframe:      str
    direction:      Direction
    entry_price:    float
    stop_price:     float
    target_price:   float
    rr_ratio:       float
    tqs:            TQSComponents
    # Context
    pattern:        PatternSignal = field(default=None)
    trend:          TrendSignal   = field(default=None)
    regime:         RegimeSignal  = field(default=None)
    level:          LevelData     = field(default=None)
    timestamp:      datetime      = field(default_factory=datetime.utcnow)

    @property
    def trade_tier(self) -> TradeTier:
        return self.tqs.tier

    @property
    def tqs_total(self) -> float:
        return self.tqs.total

    def __repr__(self) -> str:
        return (
            f"<TradeRec {self.strategy.value} {self.direction.value} "
            f"{self.symbol} TQS={self.tqs_total:.0f} [{self.trade_tier.value}] "
            f"E={self.entry_price:.5f} SL={self.stop_price:.5f} "
            f"TP={self.target_price:.5f} RR={self.rr_ratio:.1f}>"
        )


@dataclass
class RiskApprovedOrder:
    """
    Output of M09 Risk Engine — order approved for execution.
    """
    recommendation:  TradeRecommendation
    lot_size:        float
    risk_pct:        float
    risk_amount_usd: float
    account_balance: float
    stop_pips:       float
    approved_at:     datetime = field(default_factory=datetime.utcnow)


@dataclass
class RiskRejection:
    """
    Output of M09 Risk Engine — order rejected with reason.
    """
    recommendation: TradeRecommendation
    reason:         str
    check_type:     str     # KILL_SWITCH | DAILY_LIMIT | WEEKLY_LIMIT |
                            # MAX_TRADES | RR_RATIO | POSITION_SIZE | SPREAD | SESSION
    rejected_at:    datetime = field(default_factory=datetime.utcnow)


@dataclass
class OrderResult:
    """
    Output of M10 Trade Execution Engine.
    """
    order_id:      str
    status:        str          # FILLED | PENDING | REJECTED | CANCELLED
    filled_price:  Optional[float] = None
    lot_size:      Optional[float] = None
    slippage_pips: Optional[float] = None
    timestamp:     datetime = field(default_factory=datetime.utcnow)
    broker_ref:    Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class AccountState:
    """
    Current broker account state used by M09 Risk Engine.
    """
    balance:          float
    equity:           float
    margin:           float
    free_margin:      float
    open_pnl:         float
    peak_equity:      float
    day_open_balance: float
    week_open_balance: float
    open_trades:      int
    kill_switch_active: bool = False
    timestamp:        datetime = field(default_factory=datetime.utcnow)

    @property
    def drawdown_from_peak_pct(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return ((self.peak_equity - self.equity) / self.peak_equity) * 100.0

    @property
    def daily_pnl_pct(self) -> float:
        if self.day_open_balance <= 0:
            return 0.0
        return ((self.balance - self.day_open_balance) / self.day_open_balance) * 100.0

    @property
    def weekly_pnl_pct(self) -> float:
        if self.week_open_balance <= 0:
            return 0.0
        return ((self.balance - self.week_open_balance) / self.week_open_balance) * 100.0
