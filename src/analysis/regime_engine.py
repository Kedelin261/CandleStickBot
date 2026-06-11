"""
M16 — Market Regime Engine
Classifies market regime to select appropriate strategies and risk.
The Candlestick Trading Bible: Different market conditions require different approaches.

Regime Classification:
  - TRENDING: ADX > 25, directional bias clear → Pin Bar + Engulfing optimal
  - RANGING: ADX < 20, price oscillating between levels → lower quality, reduce risk
  - VOLATILE: ATR > 1.5x average, choppy → no trade (skip)
  - LOW_VOLATILITY: ATR < 0.5x average → wait for expansion

Risk Multiplier by Regime:
  - TRENDING: 1.0x (full risk)
  - RANGING: 0.75x (reduced risk)
  - VOLATILE: 0.0x (no trade)
  - LOW_VOLATILITY: 0.0x (no trade)

Strategy Allowances:
  - TRENDING: PIN_BAR, ENGULFING_BAR
  - RANGING: (none in Phase 1 — inside bar deferred to Phase 2)
  - VOLATILE: none
  - LOW_VOLATILITY: none

Status: STUB — Phase 0 scaffold. Full implementation in Phase 1 Sprint 2.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from src.types import CandleData, RegimeSignal

logger = logging.getLogger("candlestickbot.analysis.regime_engine")


class MarketRegime(str, Enum):
    """Market regime classifications."""
    TRENDING = "TRENDING"
    RANGING = "RANGING"
    VOLATILE = "VOLATILE"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    UNDEFINED = "UNDEFINED"


# Risk multiplier per regime (multiplied against base risk%)
REGIME_RISK_MULTIPLIER: Dict[MarketRegime, float] = {
    MarketRegime.TRENDING: 1.0,
    MarketRegime.RANGING: 0.75,
    MarketRegime.VOLATILE: 0.0,
    MarketRegime.LOW_VOLATILITY: 0.0,
    MarketRegime.UNDEFINED: 0.0,
}

# Allowed strategies per regime (Phase 1 strategies only)
REGIME_ALLOWED_STRATEGIES: Dict[MarketRegime, List[str]] = {
    MarketRegime.TRENDING: ["PIN_BAR", "ENGULFING_BAR"],
    MarketRegime.RANGING: [],  # Phase 2: ["INSIDE_BAR"]
    MarketRegime.VOLATILE: [],
    MarketRegime.LOW_VOLATILITY: [],
    MarketRegime.UNDEFINED: [],
}


@dataclass
class RegimeAnalysis:
    """
    Full regime analysis result from M16.

    Consumed by M08 Strategy Engine to gate strategy selection.
    """
    regime: MarketRegime
    allowed_strategies: List[str]
    risk_multiplier: float
    confidence: float           # 0.0-1.0 confidence in classification
    reason: str                 # Human-readable explanation

    # Indicator values used in classification
    adx: float = 0.0
    atr: float = 0.0
    atr_average: float = 0.0
    bb_width: float = 0.0          # Bollinger Band width
    choppiness_index: float = 0.0  # Choppiness Index (100=choppy, 0=trending)

    # TQS component
    tqs_regime_score: int = 0

    @property
    def is_tradeable(self) -> bool:
        """True if any strategies are allowed in this regime."""
        return len(self.allowed_strategies) > 0

    def to_regime_signal(self) -> RegimeSignal:
        """Convert to shared RegimeSignal DTO."""
        return RegimeSignal(
            regime=self.regime.value,
            allowed_strategies=self.allowed_strategies,
            risk_multiplier=self.risk_multiplier,
            adx=self.adx,
            atr=self.atr,
            bb_width=self.bb_width,
            choppiness_index=self.choppiness_index,
        )


class RegimeEngine:
    """
    M16 — Market Regime Engine.

    Algorithm:
    1. Calculate ADX (trend strength indicator)
    2. Calculate ATR (volatility measure)
    3. Calculate ATR ratio vs N-period average ATR
    4. Calculate Bollinger Band width (volatility proxy)
    5. Calculate Choppiness Index (61.8 = boundary between trending/ranging)
    6. Combine signals to classify regime
    7. Apply risk multiplier and strategy allowances
    8. Score TQS regime component

    Thresholds (from config):
      - ADX > trending_adx_threshold (25): Trending
      - ADX < ranging_adx_threshold (20): Ranging
      - ATR ratio > volatile_atr_multiplier (1.5): Volatile
      - ATR ratio < low_vol_atr_multiplier (0.5): Low volatility
      - Choppiness > 61.8: Ranging/choppy

    TQS Scoring for regime component (0-25 points):
      - TRENDING regime, pattern aligned: 25
      - TRENDING regime, pattern exists: 20
      - RANGING regime: 10 (reduced — Phase 1 no ranging trades)
      - VOLATILE / LOW_VOL: 0
    """

    # Choppiness Index boundary (above = choppy, below = trending)
    CHOPPINESS_THRESHOLD = 61.8

    def __init__(
        self,
        adx_period: int = 14,
        atr_period: int = 14,
        atr_avg_period: int = 50,
        bb_period: int = 20,
        bb_std_dev: float = 2.0,
        trending_adx_threshold: float = 25.0,
        ranging_adx_threshold: float = 20.0,
        volatile_atr_multiplier: float = 1.5,
        low_vol_atr_multiplier: float = 0.5,
        choppiness_period: int = 14,
    ):
        self.adx_period = adx_period
        self.atr_period = atr_period
        self.atr_avg_period = atr_avg_period
        self.bb_period = bb_period
        self.bb_std_dev = bb_std_dev
        self.trending_adx_threshold = trending_adx_threshold
        self.ranging_adx_threshold = ranging_adx_threshold
        self.volatile_atr_multiplier = volatile_atr_multiplier
        self.low_vol_atr_multiplier = low_vol_atr_multiplier
        self.choppiness_period = choppiness_period

    def analyze(self, candles: List[CandleData]) -> RegimeAnalysis:
        """
        Classify current market regime.

        Args:
            candles: Candle series (ascending, oldest first).
                     Minimum: max(adx_period*2, atr_avg_period) candles.

        Returns:
            RegimeAnalysis with regime classification and TQS score.
        """
        min_required = max(self.adx_period * 2, self.atr_avg_period, self.bb_period)
        if len(candles) < min_required:
            logger.warning(
                f"Insufficient candles for regime analysis: "
                f"{len(candles)} < {min_required} required"
            )
            return RegimeAnalysis(
                regime=MarketRegime.UNDEFINED,
                allowed_strategies=[],
                risk_multiplier=0.0,
                confidence=0.0,
                reason=f"Insufficient data: {len(candles)} < {min_required} candles",
                tqs_regime_score=0,
            )

        # TODO: Full implementation in Phase 1 Sprint 2
        # adx = self._calculate_adx(candles)
        # atr = self._calculate_atr(candles, self.atr_period)
        # atr_avg = self._calculate_atr(candles, self.atr_avg_period)
        # atr_ratio = atr / atr_avg if atr_avg > 0 else 1.0
        # bb_width = self._calculate_bb_width(candles)
        # choppiness = self._calculate_choppiness(candles)
        # regime = self._classify(adx, atr_ratio, choppiness)
        # ...

        logger.warning("RegimeEngine.analyze() — STUB")
        return RegimeAnalysis(
            regime=MarketRegime.UNDEFINED,
            allowed_strategies=[],
            risk_multiplier=0.0,
            confidence=0.0,
            reason="STUB — not yet implemented",
            tqs_regime_score=0,
        )

    def _calculate_atr(self, candles: List[CandleData], period: int) -> float:
        """
        Calculate Average True Range using Wilder's smoothing.

        True Range = max(H-L, |H-pC|, |L-pC|)
        ATR = Wilder's EMA of TR with period 'period'
        """
        if len(candles) < period + 1:
            return 0.0

        # Calculate True Range series
        tr_list = []
        for i in range(1, len(candles)):
            high = candles[i].high
            low = candles[i].low
            prev_close = candles[i - 1].close
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close),
            )
            tr_list.append(tr)

        if len(tr_list) < period:
            return sum(tr_list) / len(tr_list)

        # Wilder's smoothing: initial ATR = simple average of first 'period' TRs
        atr = sum(tr_list[:period]) / period
        alpha = 1.0 / period  # Wilder's smoothing factor

        for tr in tr_list[period:]:
            atr = atr * (1 - alpha) + tr * alpha

        return atr

    def _calculate_bb_width(self, candles: List[CandleData]) -> float:
        """
        Calculate Bollinger Band Width = (Upper - Lower) / Middle * 100.

        A higher BB width indicates higher volatility.
        """
        import math
        if len(candles) < self.bb_period:
            return 0.0

        closes = [c.close for c in candles[-self.bb_period:]]
        mean = sum(closes) / len(closes)
        variance = sum((c - mean) ** 2 for c in closes) / len(closes)
        std_dev = math.sqrt(variance)

        upper = mean + self.bb_std_dev * std_dev
        lower = mean - self.bb_std_dev * std_dev

        return (upper - lower) / mean * 100.0 if mean > 0 else 0.0

    def _calculate_choppiness(self, candles: List[CandleData]) -> float:
        """
        Calculate Choppiness Index.

        Formula: 100 * log10(sum(ATR_1) / (HH - LL)) / log10(period)
        Where:
          - sum(ATR_1) = sum of single-candle true ranges over period
          - HH = highest high over period
          - LL = lowest low over period

        Range: 100 (perfectly choppy) to 0 (perfectly trending)
        Threshold: 61.8 (golden ratio) — above=choppy, below=trending
        """
        import math
        n = self.choppiness_period

        if len(candles) < n + 1:
            return 61.8  # Default to threshold (neutral)

        period_candles = candles[-(n + 1):]
        hh = max(c.high for c in period_candles)
        ll = min(c.low for c in period_candles)
        price_range = hh - ll

        if price_range <= 0:
            return 100.0

        # Sum of single-period TRs
        tr_sum = 0.0
        for i in range(1, len(period_candles)):
            high = period_candles[i].high
            low = period_candles[i].low
            prev_close = period_candles[i - 1].close
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            tr_sum += tr

        choppiness = 100.0 * math.log10(tr_sum / price_range) / math.log10(n)
        return max(0.0, min(100.0, choppiness))

    def _classify_regime(
        self,
        adx: float,
        atr_ratio: float,
        choppiness: float,
    ) -> MarketRegime:
        """
        Classify regime from indicator combination.

        Priority order (most specific wins):
        1. Volatile: ATR ratio > 1.5x
        2. Low volatility: ATR ratio < 0.5x
        3. Trending: ADX > 25 AND choppiness < 61.8
        4. Ranging: ADX < 20 OR choppiness > 61.8
        5. Undefined (intermediate zone)
        """
        if atr_ratio > self.volatile_atr_multiplier:
            return MarketRegime.VOLATILE

        if atr_ratio < self.low_vol_atr_multiplier:
            return MarketRegime.LOW_VOLATILITY

        if adx >= self.trending_adx_threshold and choppiness < self.CHOPPINESS_THRESHOLD:
            return MarketRegime.TRENDING

        if adx < self.ranging_adx_threshold or choppiness > self.CHOPPINESS_THRESHOLD:
            return MarketRegime.RANGING

        return MarketRegime.UNDEFINED

    def calculate_tqs_regime_score(
        self,
        regime: MarketRegime,
        strategy: str,
        adx: float,
    ) -> int:
        """
        Calculate TQS regime component score (0-25 points).

        Phase 1 scoring:
          - TRENDING + valid strategy + ADX > 30: 25
          - TRENDING + valid strategy + ADX 25-30: 20
          - TRENDING + valid strategy + ADX 20-25: 15
          - RANGING (Phase 1 no-trade): 0
          - VOLATILE: 0
          - LOW_VOL: 0
        """
        if regime != MarketRegime.TRENDING:
            return 0

        if strategy not in REGIME_ALLOWED_STRATEGIES[MarketRegime.TRENDING]:
            return 0

        if adx >= 30:
            return 25
        if adx >= 25:
            return 20
        if adx >= 20:
            return 15
        return 0
