"""
M08 — Strategy Engine
Coordinates all analysis modules to produce Trade Recommendations.
The Candlestick Trading Bible: The strategy is the integration of all signals.

Pipeline (per candle evaluation):
  1. Fetch latest analysis results from M03, M04, M05, M16
  2. Run active pattern detectors (M07)
  3. If pattern detected → calculate TQS (0-100)
  4. Gate against TQS thresholds (REJECT/STANDARD/PREMIUM tiers)
  5. If TQS >= 60 → create TradeRecommendation
  6. Log all decisions (M13)

TQS Components (total 100 points):
  - Trend score (M04):    0-25 pts
  - Level score (M05):    0-25 pts
  - Pattern score (M07):  0-25 pts
  - Regime score (M16):   0-25 pts

TQS Tiers:
  - REJECT:   TQS < 60  → no trade
  - STANDARD: TQS 60-79 → trade at 1.0% risk
  - PREMIUM:  TQS >= 80 → eligible for 1.5% risk (if enabled, default: off)

Phase 1: EURUSD D1, Pin Bar + Engulfing Bar only.
Status: STUB — Phase 0 scaffold. Full implementation in Phase 1 Sprint 3.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from src.types import (
    CandleData,
    PatternSignal,
    TQSComponents,
    TradeRecommendation,
    RiskRejection,
)

logger = logging.getLogger("candlestickbot.strategy.engine")


class StrategyName(str, Enum):
    """Supported strategy names."""
    PIN_BAR = "PIN_BAR"
    ENGULFING_BAR = "ENGULFING_BAR"
    INSIDE_BAR = "INSIDE_BAR"           # Phase 2 only
    FALSE_BREAKOUT = "FALSE_BREAKOUT"   # Phase 2 only


class TQSTier(str, Enum):
    """Trade Quality Score tier classification."""
    REJECT = "REJECT"       # TQS < 60 — no trade
    STANDARD = "STANDARD"  # TQS 60-79 — trade at standard risk
    PREMIUM = "PREMIUM"    # TQS >= 80 — eligible for premium risk


def get_tqs_tier(score: int) -> TQSTier:
    """Return TQS tier for a given score."""
    if score < 60:
        return TQSTier.REJECT
    if score < 80:
        return TQSTier.STANDARD
    return TQSTier.PREMIUM


@dataclass
class EvaluationResult:
    """
    Complete evaluation result for a single candle bar.

    Contains all intermediate analysis results and final recommendation.
    """
    symbol: str
    timeframe: str
    candle: CandleData

    # Analysis inputs
    trend_direction: str = "UNDEFINED"
    trend_tradeable: bool = False
    regime: str = "UNDEFINED"
    regime_tradeable: bool = False

    # Pattern detection results
    patterns_checked: List[str] = field(default_factory=list)
    pattern_detected: Optional[str] = None
    pattern_direction: Optional[str] = None
    pattern_quality: int = 0

    # TQS
    tqs_total: int = 0
    tqs_trend: int = 0
    tqs_level: int = 0
    tqs_pattern: int = 0
    tqs_regime: int = 0
    tqs_tier: TQSTier = TQSTier.REJECT

    # Final output
    recommendation: Optional[TradeRecommendation] = None
    rejection_reason: Optional[str] = None
    rejection_gate: str = ""  # Which gate rejected (TREND, REGIME, TQS, etc.)

    @property
    def is_recommended(self) -> bool:
        return self.recommendation is not None


@dataclass
class StrategyEngineConfig:
    """Configuration for strategy engine operation."""
    min_tqs_score: int = 60
    premium_tqs_threshold: int = 80
    min_rr_ratio: float = 2.0
    active_strategies: List[str] = field(
        default_factory=lambda: ["PIN_BAR", "ENGULFING_BAR"]
    )
    trend_gate_enabled: bool = True
    regime_gate_enabled: bool = True
    tqs_gate_enabled: bool = True


class StrategyEngine:
    """
    M08 — Strategy Engine.

    Responsibilities:
    - Coordinate analysis modules (M03, M04, M05, M16)
    - Run pattern detectors (M07)
    - Calculate TQS for each pattern
    - Gate trades through quality filters
    - Produce TradeRecommendation or rejection

    Gate chain (each gate can reject and stop evaluation):
    1. TREND GATE:  Is there a tradeable trend? (M04)
    2. REGIME GATE: Is the regime suitable? (M16)
    3. PATTERN GATE: Does a pattern exist on current bar? (M07)
    4. LEVEL GATE:  Is pattern at a key level? (M05)
    5. TQS GATE:   Is total TQS >= minimum threshold?
    6. RR GATE:    Is R:R ratio >= minimum (2.0)?

    Only trades passing ALL gates are recommended.
    """

    def __init__(
        self,
        config: Optional[StrategyEngineConfig] = None,
        trend_detector=None,       # M04 TrendDetector instance
        sr_engine=None,            # M05 SREngine instance
        regime_engine=None,        # M16 RegimeEngine instance
        pin_bar_detector=None,     # M07 PinBarDetector instance
        engulfing_detector=None,   # M07 EngulfingDetector instance
        audit_logger=None,         # M13 AuditLogger instance
    ):
        self.config = config or StrategyEngineConfig()
        self.trend_detector = trend_detector
        self.sr_engine = sr_engine
        self.regime_engine = regime_engine
        self.pin_bar_detector = pin_bar_detector
        self.engulfing_detector = engulfing_detector
        self.audit_logger = audit_logger

    def evaluate(
        self,
        symbol: str,
        timeframe: str,
        candles: List[CandleData],
    ) -> EvaluationResult:
        """
        Evaluate latest candle for trade opportunities.

        Args:
            symbol: Symbol being analyzed
            timeframe: Timeframe of candles
            candles: Full candle series, ascending order (oldest first)
                     Must have at least 50 candles for analysis

        Returns:
            EvaluationResult with recommendation or rejection details.
        """
        if not candles:
            return EvaluationResult(
                symbol=symbol,
                timeframe=timeframe,
                candle=CandleData(symbol, timeframe, None, 0, 0, 0, 0, 0, 0),
                rejection_reason="No candles provided",
                rejection_gate="DATA",
            )

        current = candles[-1]
        result = EvaluationResult(
            symbol=symbol,
            timeframe=timeframe,
            candle=current,
        )

        # TODO: Full implementation in Phase 1 Sprint 3
        # Step 1: Trend gate (M04)
        # trend = self.trend_detector.analyze(candles)
        # if self.config.trend_gate_enabled and not trend.tradeable:
        #     result.trend_direction = trend.direction
        #     result.rejection_reason = f"Trend not tradeable: {trend.reason}"
        #     result.rejection_gate = "TREND"
        #     return result
        # ...

        logger.warning("StrategyEngine.evaluate() — STUB")
        result.rejection_reason = "STUB — not yet implemented"
        result.rejection_gate = "STUB"
        return result

    def calculate_tqs(
        self,
        trend_score: int,
        level_score: int,
        pattern_score: int,
        regime_score: int,
    ) -> TQSComponents:
        """
        Calculate Trade Quality Score from component scores.

        Args:
            trend_score: 0-25 from M04
            level_score: 0-25 from M05
            pattern_score: 0-25 from M07
            regime_score: 0-25 from M16

        Returns:
            TQSComponents with total and tier.
        """
        return TQSComponents(
            trend_score=trend_score,
            level_score=level_score,
            pattern_score=pattern_score,
            regime_score=regime_score,
        )

    def _calculate_pattern_tqs_score(
        self,
        pattern_quality: int,
        pattern_type: str,
    ) -> int:
        """
        Convert pattern quality (1-10) to TQS pattern score (0-25).

        Mapping:
          - Quality 9-10: 25 pts (exceptional pattern)
          - Quality 7-8:  20 pts (strong pattern)
          - Quality 5-6:  15 pts (valid pattern)
          - Quality 3-4:  10 pts (marginal pattern)
          - Quality 1-2:   5 pts (weak pattern — still above TQS floor)
          - Quality 0:     0 pts (not detected)
        """
        if pattern_quality <= 0:
            return 0
        if pattern_quality >= 9:
            return 25
        if pattern_quality >= 7:
            return 20
        if pattern_quality >= 5:
            return 15
        if pattern_quality >= 3:
            return 10
        return 5

    def _build_trade_recommendation(
        self,
        symbol: str,
        timeframe: str,
        strategy: str,
        direction: str,
        entry: float,
        stop: float,
        tqs: TQSComponents,
    ) -> Optional[TradeRecommendation]:
        """
        Build a TradeRecommendation if R:R ratio is acceptable.

        Target is calculated as:
            entry + (entry - stop) * min_rr_ratio  (for LONG)
            entry - (stop - entry) * min_rr_ratio  (for SHORT)

        Returns None if R:R < min_rr_ratio.
        """
        risk_distance = abs(entry - stop)
        if risk_distance <= 0:
            return None

        if direction == "LONG":
            target = entry + risk_distance * self.config.min_rr_ratio
        else:
            target = entry - risk_distance * self.config.min_rr_ratio

        rr_ratio = abs(target - entry) / risk_distance

        if rr_ratio < self.config.min_rr_ratio:
            return None

        return TradeRecommendation(
            strategy=strategy,
            direction=direction,
            entry_price=entry,
            stop_price=stop,
            target_price=target,
            rr_ratio=rr_ratio,
            tqs=tqs,
        )
