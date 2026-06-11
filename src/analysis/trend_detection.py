"""
M04 — Trend Detection Module
Classifies trend using 21 SMA position + market structure (M03).
The Candlestick Trading Bible: Only trade in the direction of the trend.

Rules (from spec):
  - UPTREND:   Price above 21 SMA AND structure shows HH+HL
  - DOWNTREND: Price below 21 SMA AND structure shows LH+LL
  - RANGING:   Price within ±0.5% of 21 SMA OR mixed structure
  - Tradeable: Trend confirmed AND ADX >= 20 (strength filter)

ADX thresholds:
  - < 20: Weak / ranging — DO NOT trade
  - 20–25: Developing trend — trade cautiously
  - 25–40: Strong trend — optimal
  - > 40: Very strong trend — watch for exhaustion

Phase 1 scope: EURUSD D1 only.
Status: STUB — Phase 0 scaffold. Full implementation in Phase 1 Sprint 2.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from src.types import CandleData, MarketStructure, TrendSignal

logger = logging.getLogger("candlestickbot.analysis.trend_detection")


class TrendStrength(str, Enum):
    """ADX-based trend strength classification."""
    WEAK = "WEAK"           # ADX < 20
    DEVELOPING = "DEVELOPING"  # ADX 20-25
    STRONG = "STRONG"       # ADX 25-40
    VERY_STRONG = "VERY_STRONG"  # ADX > 40


@dataclass
class TrendAnalysis:
    """
    Full trend analysis result from M04.

    Consumed by M08 (Strategy Engine) to gate trade direction.
    """
    direction: str              # "UP", "DOWN", "RANGING", "UNDEFINED"
    tradeable: bool             # True if trend is strong enough to trade
    reason: str                 # Human-readable explanation
    sma21: float               # 21-period SMA value at analysis time
    sma21_slope: float         # SMA slope (positive = rising, negative = falling)
    price_vs_sma: float        # Current price relative to SMA (percentage)
    adx: Optional[float]       # ADX value (None if insufficient data)
    adx_strength: TrendStrength
    structure_direction: str   # Direction from M03 market structure

    # TQS component: trend score (0-25)
    tqs_trend_score: int = 0

    def to_trend_signal(self) -> TrendSignal:
        """Convert to shared TrendSignal DTO."""
        return TrendSignal(
            direction=self.direction,
            sma21=self.sma21,
            tradeable=self.tradeable,
            reason=self.reason,
            adx=self.adx,
            strength=self.adx_strength.value,
        )


class TrendDetector:
    """
    M04 — Trend Detector.

    Algorithm:
    1. Calculate 21-period Simple Moving Average
    2. Determine price position relative to 21 SMA
    3. Read market structure direction from M03
    4. Calculate ADX for trend strength
    5. Classify final trend: UP/DOWN/RANGING
    6. Set tradeable flag based on ADX threshold

    TQS Scoring (0-25 points for trend component):
      - 25: Strong trend aligned (price clearly above/below SMA, ADX>25)
      - 20: Moderate trend aligned
      - 15: Weak trend aligned
      - 0: Ranging or counter-trend
    """

    ADX_MIN_TRADEABLE = 20.0      # Minimum ADX to consider trend tradeable
    ADX_STRONG = 25.0             # ADX threshold for STRONG classification
    ADX_VERY_STRONG = 40.0        # ADX threshold for VERY_STRONG
    SMA_BAND_PCT = 0.005          # ±0.5% band around SMA = ranging zone
    SMA_PERIOD = 21               # Fixed per spec

    def __init__(
        self,
        sma_period: int = 21,
        adx_period: int = 14,
        adx_min_tradeable: float = 20.0,
        sma_band_pct: float = 0.005,
    ):
        self.sma_period = sma_period
        self.adx_period = adx_period
        self.adx_min_tradeable = adx_min_tradeable
        self.sma_band_pct = sma_band_pct

    def analyze(
        self,
        candles: List[CandleData],
        market_structure: Optional[MarketStructure] = None,
    ) -> TrendAnalysis:
        """
        Perform full trend analysis.

        Args:
            candles: Candle series (ascending, oldest first).
                     Minimum: sma_period candles for SMA, sma_period + adx_period for ADX
            market_structure: Output from M03 MarketStructureAnalyzer

        Returns:
            TrendAnalysis with direction, tradeable flag, and TQS score.
        """
        min_required = self.sma_period + self.adx_period
        if len(candles) < self.sma_period:
            return TrendAnalysis(
                direction="UNDEFINED",
                tradeable=False,
                reason=f"Insufficient data: {len(candles)} candles, need {self.sma_period}",
                sma21=0.0,
                sma21_slope=0.0,
                price_vs_sma=0.0,
                adx=None,
                adx_strength=TrendStrength.WEAK,
                structure_direction="UNDEFINED",
                tqs_trend_score=0,
            )

        # TODO: Full implementation in Phase 1 Sprint 2
        # sma21 = self._calculate_sma(candles, self.sma_period)
        # sma21_prev = self._calculate_sma(candles[:-1], self.sma_period)
        # sma_slope = sma21 - sma21_prev
        # current_price = candles[-1].close
        # price_vs_sma_pct = (current_price - sma21) / sma21
        # adx = self._calculate_adx(candles, self.adx_period) if len(candles) >= min_required else None
        # ...

        logger.warning("TrendDetector.analyze() — STUB")
        return TrendAnalysis(
            direction="UNDEFINED",
            tradeable=False,
            reason="STUB — not yet implemented",
            sma21=0.0,
            sma21_slope=0.0,
            price_vs_sma=0.0,
            adx=None,
            adx_strength=TrendStrength.WEAK,
            structure_direction="UNDEFINED",
            tqs_trend_score=0,
        )

    def _calculate_sma(self, candles: List[CandleData], period: int) -> float:
        """
        Calculate Simple Moving Average of closing prices.

        Args:
            candles: Candle series (at least 'period' candles)
            period: SMA period

        Returns:
            SMA value using last 'period' closing prices.
        """
        if len(candles) < period:
            raise ValueError(f"Need {period} candles for SMA, got {len(candles)}")
        closes = [c.close for c in candles[-period:]]
        return sum(closes) / period

    def _calculate_sma_series(self, candles: List[CandleData], period: int) -> List[float]:
        """
        Calculate full SMA series for all candles.

        Returns list of SMA values. First (period-1) elements are None/0.
        """
        closes = [c.close for c in candles]
        sma_series = [0.0] * len(closes)
        for i in range(period - 1, len(closes)):
            sma_series[i] = sum(closes[i - period + 1:i + 1]) / period
        return sma_series

    def _calculate_adx(self, candles: List[CandleData], period: int = 14) -> float:
        """
        Calculate Average Directional Index (ADX).

        ADX measures trend strength regardless of direction.
        Uses Wilder's smoothing method (same as MT5).

        Algorithm:
        1. Calculate True Range (TR) = max(H-L, |H-pC|, |L-pC|)
        2. Calculate +DM and -DM for each period
        3. Smooth with Wilder's EMA (alpha = 1/period)
        4. Calculate +DI and -DI from smoothed values
        5. DX = 100 * |+DI - -DI| / (+DI + -DI)
        6. ADX = Wilder's EMA of DX
        """
        # Requires at least 2*period candles for reliable ADX
        if len(candles) < period * 2:
            return 0.0

        # TODO: Full Wilder's ADX implementation in Phase 1 Sprint 2
        # tr_list, plus_dm_list, minus_dm_list = [], [], []
        # for i in range(1, len(candles)):
        #     ...
        return 0.0

    def _classify_adx_strength(self, adx: Optional[float]) -> TrendStrength:
        """Classify ADX value into trend strength category."""
        if adx is None or adx < self.ADX_MIN_TRADEABLE:
            return TrendStrength.WEAK
        if adx < self.ADX_STRONG:
            return TrendStrength.DEVELOPING
        if adx < self.ADX_VERY_STRONG:
            return TrendStrength.STRONG
        return TrendStrength.VERY_STRONG

    def _calculate_tqs_trend_score(
        self,
        direction: str,
        tradeable: bool,
        adx: Optional[float],
        price_vs_sma_pct: float,
        structure_direction: str,
    ) -> int:
        """
        Calculate TQS trend component score (0-25 points).

        Scoring rubric:
          - Structure aligned with SMA direction: +10
          - ADX > 25 (strong): +10
          - ADX 20-25 (developing): +7
          - Price clearly above/below SMA (>1%): +5
          - Not tradeable (ranging/no trend): 0

        Maximum: 25 points
        """
        if not tradeable or direction == "RANGING":
            return 0

        score = 0

        # Structure alignment bonus
        if structure_direction == direction:
            score += 10

        # ADX bonus
        if adx is not None:
            if adx >= self.ADX_STRONG:
                score += 10
            elif adx >= self.ADX_MIN_TRADEABLE:
                score += 7

        # SMA separation bonus
        if abs(price_vs_sma_pct) > 0.01:  # >1% separation
            score += 5

        return min(score, 25)
