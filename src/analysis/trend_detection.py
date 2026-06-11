"""
M04 — Trend Detection Engine
Classifies trend using 21-period SMA position + M03 Market Structure output.

The Candlestick Trading Bible: Only trade in the direction of the trend.

Trend Confirmation Rules:
  BULLISH: M03 direction is UP AND latest close is ABOVE 21 SMA
  BEARISH: M03 direction is DOWN AND latest close is BELOW 21 SMA
  RANGING: M03 structure is_ranging=True (sideways consolidation)
  NONE:    M03 direction is NONE / insufficient data / close conflicts with structure

Confidence Scoring (0–100):
  40 pts — market structure agrees with SMA direction
  25 pts — price is on the correct side of 21 SMA
  15 pts — SMA slope agrees with direction
  10 pts — close-to-SMA distance is meaningful but not overextended
  10 pts — M03 structure confidence contributes positively

Tradeability:
  tradeable=True only when:
    - direction is UP or DOWN
    - close agrees with SMA side
    - confidence >= tradeable_threshold (default 60)
    - sufficient candles for SMA period
    - market is not ranging / choppy / undefined

Phase 1 Sprint 3: Full implementation replacing Phase 0 stub.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

from src.types import (
    CandleData,
    MarketStructure,
    TrendDirection,
    TrendSignal,
)

logger = logging.getLogger("candlestickbot.analysis.trend_detection")

# ---------------------------------------------------------------------------
# Internal sentinel — MarketStructure DTO uses TrendDirection (UP/DOWN/NONE)
# but we also need to distinguish a RANGING market explicitly.
# We carry this through StructureAnalysis.is_ranging if M03 object is passed
# directly, or detect it from the MarketStructure DTO regime field.
# ---------------------------------------------------------------------------
_RANGING_LABEL = "RANGING"

# ---------------------------------------------------------------------------
# SMA distance thresholds
# ---------------------------------------------------------------------------
_SMA_MEANINGFUL_DIST_PCT = 0.0010   # 0.10%  — minimum "meaningful" separation
_SMA_OVEREXTENDED_PCT    = 0.0150   # 1.50%  — price too far from SMA (exhaustion risk)


class TrendStrength(str, Enum):
    """Trend strength classification (retained from Phase 0 stub for compatibility)."""
    WEAK         = "WEAK"
    DEVELOPING   = "DEVELOPING"
    STRONG       = "STRONG"
    VERY_STRONG  = "VERY_STRONG"


# ---------------------------------------------------------------------------
# Internal result dataclass
# ---------------------------------------------------------------------------

@dataclass
class TrendAnalysis:
    """
    Full trend analysis result from M04 TrendDetector.

    Consumed by M08 Strategy Engine to gate trade direction.

    Fields:
        direction        — 'UP', 'DOWN', 'RANGING', or 'UNDEFINED'
        tradeable        — True if trend is confirmed and strong enough to trade
        reason           — Human-readable explanation
        sma21            — 21-period SMA value at time of analysis
        sma21_slope      — SMA slope: latest SMA minus previous SMA (positive=rising)
        price_vs_sma     — (close - SMA) / SMA  (fractional, signed)
        adx              — ADX value if computed, else None
        adx_strength     — TrendStrength enum classification
        structure_direction — Direction string from M03 ('UP'/'DOWN'/'RANGING'/'NONE')
        confidence_score — 0–100 composite score
        tqs_trend_score  — 0–25 TQS component for downstream scoring
    """
    direction:           str
    tradeable:           bool
    reason:              str
    sma21:               float
    sma21_slope:         float
    price_vs_sma:        float
    adx:                 Optional[float]
    adx_strength:        TrendStrength
    structure_direction: str
    confidence_score:    float = 0.0       # 0–100 composite
    tqs_trend_score:     int   = 0         # 0–25 TQS component
    structure_confidence: float = 0.0      # M03 confidence (0.0–1.0), 0 if not available
    sma_series:          List[float] = field(default_factory=list, repr=False)

    def to_trend_signal(
        self,
        symbol: str = "",
        timeframe: str = "",
        timestamp: Optional[datetime] = None,
    ) -> TrendSignal:
        """
        Convert to shared TrendSignal DTO for cross-module consumption.

        Args:
            symbol:    Instrument symbol (e.g. 'EURUSD')
            timeframe: Chart timeframe (e.g. 'D1')
            timestamp: Analysis timestamp; defaults to utcnow if omitted
        """
        ts = timestamp or datetime.now(timezone.utc)
        # Map internal direction string to TrendDirection enum
        direction_map = {
            "UP":        TrendDirection.UP,
            "DOWN":      TrendDirection.DOWN,
            "RANGING":   TrendDirection.NONE,
            "UNDEFINED": TrendDirection.NONE,
        }
        direction_enum = direction_map.get(self.direction, TrendDirection.NONE)

        return TrendSignal(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=ts,
            direction=direction_enum,
            sma21=self.sma21,
            tradeable=self.tradeable,
            reason=self.reason,
            adx=self.adx,
            strength=round(self.confidence_score / 100.0, 4),
        )


# ---------------------------------------------------------------------------
# Module-level stateless helpers (required by spec)
# ---------------------------------------------------------------------------

def compute_sma(candles: List[CandleData], period: int = 21) -> List[float]:
    """
    Compute the full SMA series for the given candle list.

    Returns a list of the same length as `candles`.  The first (period-1)
    entries are 0.0 because there are not yet enough bars to form a full window.

    Args:
        candles: Candle series (oldest first).
        period:  Lookback window (default 21).

    Returns:
        List[float] of length len(candles).  Positions 0..period-2 are 0.0.
    """
    if period < 1:
        raise ValueError(f"SMA period must be >= 1, got {period}")
    n = len(candles)
    result = [0.0] * n
    for i in range(period - 1, n):
        window = candles[i - period + 1 : i + 1]
        result[i] = sum(c.close for c in window) / period
    return result


def get_current_sma(candles: List[CandleData], period: int = 21) -> Optional[float]:
    """
    Return the SMA value for the LATEST candle.

    Returns None when fewer than `period` candles are available.

    Args:
        candles: Candle series (oldest first).
        period:  Lookback window (default 21).
    """
    if len(candles) < period:
        return None
    return sum(c.close for c in candles[-period:]) / period


def is_price_above_sma(price: float, sma: float) -> bool:
    """Return True when price is strictly above the SMA."""
    return price > sma


def is_price_below_sma(price: float, sma: float) -> bool:
    """Return True when price is strictly below the SMA."""
    return price < sma


def determine_trend_direction(
    market_structure: Optional[Union[MarketStructure, "StructureAnalysis"]],
    candles: List[CandleData],
    sma_period: int = 21,
) -> Tuple[str, str]:
    """
    Determine the trend direction by combining M03 structure with SMA position.

    Accepts either a ``MarketStructure`` DTO (shared type) or a
    ``StructureAnalysis`` object (M03 internal, if passed directly).

    Returns:
        Tuple (direction_str, reason_str)
        direction_str is one of: 'UP', 'DOWN', 'RANGING', 'UNDEFINED'
    """
    # ---- Guard: not enough candles for SMA
    if len(candles) < sma_period:
        return (
            "UNDEFINED",
            f"Insufficient candles for SMA{sma_period}: {len(candles)} < {sma_period}",
        )

    sma = get_current_sma(candles, sma_period)
    if sma is None or sma == 0.0:
        return "UNDEFINED", "SMA could not be computed"

    close = candles[-1].close

    # ---- Unpack structure info
    struct_dir, is_ranging = _extract_structure_info(market_structure)

    # ---- Ranging: structure says sideways regardless of SMA
    if is_ranging:
        return _RANGING_LABEL, "Market structure is ranging (consolidation)"

    # ---- No structure info available
    if struct_dir == "UNDEFINED" or market_structure is None:
        return "UNDEFINED", "No valid market structure provided"

    # ---- NONE: choppy / insufficient structure
    if struct_dir == "NONE":
        return "UNDEFINED", "Market structure is choppy or undefined"

    # ---- BULLISH: structure UP + price above SMA
    if struct_dir == "UP":
        if is_price_above_sma(close, sma):
            return "UP", "Structure UP and price above 21 SMA — bullish trend confirmed"
        return "UNDEFINED", "Structure UP but price below 21 SMA — trend not confirmed"

    # ---- BEARISH: structure DOWN + price below SMA
    if struct_dir == "DOWN":
        if is_price_below_sma(close, sma):
            return "DOWN", "Structure DOWN and price below 21 SMA — bearish trend confirmed"
        return "UNDEFINED", "Structure DOWN but price above 21 SMA — trend not confirmed"

    return "UNDEFINED", f"Unrecognized structure direction: {struct_dir}"


def calculate_trend_strength(
    market_structure: Optional[Union[MarketStructure, "StructureAnalysis"]],
    candles: List[CandleData],
    sma: Optional[float],
    sma_period: int = 21,
) -> float:
    """
    Compute a 0–100 composite trend confidence score.

    Scoring breakdown:
        40 pts — market structure direction agrees with SMA side
        25 pts — price is on the correct side of 21 SMA
        15 pts — SMA slope agrees with direction
        10 pts — close-to-SMA distance is meaningful but not overextended
        10 pts — M03 structure confidence contributes positively

    Returns:
        Float in [0.0, 100.0].  Returns 0.0 if sma is None/zero or no candles.
    """
    if not candles or sma is None or sma == 0.0:
        return 0.0

    close = candles[-1].close
    struct_dir, is_ranging = _extract_structure_info(market_structure)
    struct_confidence = _get_structure_confidence(market_structure)

    # Determine the effective trend direction from SMA position alone
    sma_direction = "UP" if close > sma else ("DOWN" if close < sma else "FLAT")

    score = 0.0

    # ── 40 pts: structure agrees with SMA direction ──────────────────────────
    if not is_ranging and struct_dir in ("UP", "DOWN") and struct_dir == sma_direction:
        score += 40.0

    # ── 25 pts: price is clearly on the correct SMA side ────────────────────
    price_vs_sma_pct = (close - sma) / sma if sma != 0.0 else 0.0
    if abs(price_vs_sma_pct) > 0.0:       # any side counts
        score += 25.0

    # ── 15 pts: SMA slope agrees with direction ──────────────────────────────
    sma_slope = _compute_sma_slope(candles, sma_period)
    if sma_direction == "UP" and sma_slope > 0:
        score += 15.0
    elif sma_direction == "DOWN" and sma_slope < 0:
        score += 15.0

    # ── 10 pts: distance meaningful but not overextended ─────────────────────
    dist_pct = abs(price_vs_sma_pct)
    if _SMA_MEANINGFUL_DIST_PCT <= dist_pct <= _SMA_OVEREXTENDED_PCT:
        score += 10.0

    # ── 10 pts: M03 structure confidence contribution ─────────────────────────
    # Full 10 pts when struct_confidence >= 0.6; scaled below that
    score += min(10.0, struct_confidence * 10.0 / 0.6) if struct_confidence > 0 else 0.0

    return min(100.0, round(score, 2))


def is_trend_tradeable(
    trend_analysis: TrendAnalysis,
    tradeable_threshold: float = 60.0,
) -> bool:
    """
    Return True when all tradeability conditions are met.

    Conditions:
      1. direction is 'UP' or 'DOWN'  (not RANGING / UNDEFINED)
      2. close is on the correct side of 21 SMA (price_vs_sma sign matches direction)
      3. confidence_score >= tradeable_threshold
      4. sma21 was computed (sma21 > 0)
    """
    if trend_analysis.direction not in ("UP", "DOWN"):
        return False
    if trend_analysis.sma21 <= 0.0:
        return False
    # price_vs_sma is (close - SMA) / SMA — positive means above, negative means below
    if trend_analysis.direction == "UP" and trend_analysis.price_vs_sma <= 0.0:
        return False
    if trend_analysis.direction == "DOWN" and trend_analysis.price_vs_sma >= 0.0:
        return False
    if trend_analysis.confidence_score < tradeable_threshold:
        return False
    return True


def summarize_trend(trend_analysis: TrendAnalysis) -> Dict:
    """
    Return a human-readable summary dict of the TrendAnalysis.

    Keys:
        direction, tradeable, confidence_score, sma21, sma21_slope,
        price_vs_sma_pct, price_vs_sma_side, adx, adx_strength,
        structure_direction, structure_confidence, reason, tqs_trend_score
    """
    pct = trend_analysis.price_vs_sma
    if pct > 0:
        side = "ABOVE"
    elif pct < 0:
        side = "BELOW"
    else:
        side = "AT"

    return {
        "direction":            trend_analysis.direction,
        "tradeable":            trend_analysis.tradeable,
        "confidence_score":     round(trend_analysis.confidence_score, 2),
        "sma21":                round(trend_analysis.sma21, 6),
        "sma21_slope":          round(trend_analysis.sma21_slope, 6),
        "price_vs_sma_pct":     round(pct * 100, 4),   # convert to %
        "price_vs_sma_side":    side,
        "adx":                  trend_analysis.adx,
        "adx_strength":         trend_analysis.adx_strength.value,
        "structure_direction":  trend_analysis.structure_direction,
        "structure_confidence": round(trend_analysis.structure_confidence, 4),
        "reason":               trend_analysis.reason,
        "tqs_trend_score":      trend_analysis.tqs_trend_score,
    }


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class TrendDetector:
    """
    M04 — Trend Detector.

    Algorithm:
      1. Compute 21-period SMA and its slope
      2. Read market structure from M03 (StructureAnalysis or MarketStructure DTO)
      3. Determine trend direction: needs BOTH structure agreement AND SMA confirmation
      4. Score confidence on a 0–100 scale
      5. Set tradeable flag when confidence >= threshold AND direction is clear
      6. Return TrendAnalysis (internal) — call .to_trend_signal() for shared DTO

    Accepts:
      - StructureAnalysis objects (M03 internal — richer, has .is_ranging flag)
      - MarketStructure DTOs (shared type — works without M03 import cycle)
      - None (falls back to SMA-only analysis, returns UNDEFINED if no structure)

    TQS Scoring (0–25 pts):
      25 — Strong confirmed trend, confidence >= 80
      20 — Moderate confirmed trend, confidence 60–79
      15 — Weak confirmed trend, confidence 40–59
       0 — Ranging / undefined / not tradeable
    """

    SMA_PERIOD:          int   = 21
    ADX_MIN_TRADEABLE:   float = 20.0
    ADX_STRONG:          float = 25.0
    ADX_VERY_STRONG:     float = 40.0
    DEFAULT_THRESHOLD:   float = 60.0

    def __init__(
        self,
        sma_period: int   = 21,
        adx_period: int   = 14,
        adx_min_tradeable: float = 20.0,
        sma_band_pct: float = 0.005,
        tradeable_threshold: float = 60.0,
    ):
        if sma_period < 1:
            raise ValueError(f"sma_period must be >= 1, got {sma_period}")
        self.sma_period          = sma_period
        self.adx_period          = adx_period
        self.adx_min_tradeable   = adx_min_tradeable
        self.sma_band_pct        = sma_band_pct
        self.tradeable_threshold = tradeable_threshold

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def analyze(
        self,
        candles: List[CandleData],
        market_structure: Optional[Union[MarketStructure, object]] = None,
    ) -> TrendAnalysis:
        """
        Perform full trend analysis.

        Args:
            candles:          Candle series (ascending, oldest first).
                              Minimum ``sma_period`` candles required for SMA.
            market_structure: M03 output — either a StructureAnalysis or
                              MarketStructure DTO.  If None, returns UNDEFINED.

        Returns:
            TrendAnalysis — never raises on empty / short input.
        """
        # ── Guard: insufficient candles ──────────────────────────────────────
        if len(candles) < self.sma_period:
            reason = (
                f"Insufficient candles: {len(candles)} < {self.sma_period} "
                f"required for SMA{self.sma_period}"
            )
            logger.debug("M04 analyze: %s", reason)
            return TrendAnalysis(
                direction="UNDEFINED",
                tradeable=False,
                reason=reason,
                sma21=0.0,
                sma21_slope=0.0,
                price_vs_sma=0.0,
                adx=None,
                adx_strength=TrendStrength.WEAK,
                structure_direction="UNDEFINED",
                confidence_score=0.0,
                tqs_trend_score=0,
                structure_confidence=0.0,
            )

        # ── SMA ───────────────────────────────────────────────────────────────
        sma21      = self._calculate_sma(candles, self.sma_period)
        sma_slope  = self._compute_sma_slope(candles)
        sma_series = self._calculate_sma_series(candles, self.sma_period)
        close      = candles[-1].close
        price_vs_sma = (close - sma21) / sma21 if sma21 != 0.0 else 0.0

        # ── Structure extraction ──────────────────────────────────────────────
        struct_dir, is_ranging   = _extract_structure_info(market_structure)
        struct_confidence        = _get_structure_confidence(market_structure)

        # ── Direction determination ───────────────────────────────────────────
        direction, reason = determine_trend_direction(
            market_structure, candles, self.sma_period
        )

        # ── Confidence score ──────────────────────────────────────────────────
        confidence = calculate_trend_strength(
            market_structure, candles, sma21, self.sma_period
        )
        if direction in ("RANGING", "UNDEFINED"):
            confidence = 0.0

        # ── Tradeable ─────────────────────────────────────────────────────────
        # Build a temporary analysis to run the check, then decide
        temp = TrendAnalysis(
            direction=direction,
            tradeable=False,
            reason=reason,
            sma21=sma21,
            sma21_slope=sma_slope,
            price_vs_sma=price_vs_sma,
            adx=None,
            adx_strength=TrendStrength.WEAK,
            structure_direction=struct_dir,
            confidence_score=confidence,
            tqs_trend_score=0,
            structure_confidence=struct_confidence,
            sma_series=sma_series,
        )
        tradeable = is_trend_tradeable(temp, self.tradeable_threshold)

        # ── TQS trend score (0–25) ────────────────────────────────────────────
        tqs = self._calculate_tqs_trend_score(direction, tradeable, confidence)

        # ── ADX (stub — Wilder's ADX is Phase 2+) ────────────────────────────
        adx = self._calculate_adx(candles, self.adx_period)
        adx_strength = self._classify_adx_strength(adx)

        logger.debug(
            "M04 analyze: direction=%s tradeable=%s confidence=%.1f sma=%.5f slope=%.6f",
            direction, tradeable, confidence, sma21, sma_slope,
        )

        return TrendAnalysis(
            direction=direction,
            tradeable=tradeable,
            reason=reason,
            sma21=sma21,
            sma21_slope=sma_slope,
            price_vs_sma=price_vs_sma,
            adx=adx,
            adx_strength=adx_strength,
            structure_direction=struct_dir,
            confidence_score=confidence,
            tqs_trend_score=tqs,
            structure_confidence=struct_confidence,
            sma_series=sma_series,
        )

    # ------------------------------------------------------------------
    # PRIVATE HELPERS
    # ------------------------------------------------------------------

    def _calculate_sma(self, candles: List[CandleData], period: int) -> float:
        """Compute SMA from last `period` closes. Raises if insufficient candles."""
        if len(candles) < period:
            raise ValueError(f"Need {period} candles for SMA, got {len(candles)}")
        return sum(c.close for c in candles[-period:]) / period

    def _calculate_sma_series(
        self, candles: List[CandleData], period: int
    ) -> List[float]:
        """Full SMA series; first (period-1) entries are 0.0."""
        return compute_sma(candles, period)

    def _compute_sma_slope(self, candles: List[CandleData]) -> float:
        """SMA slope = latest SMA minus the SMA one bar earlier. Returns 0.0 if < period+1 candles."""
        return _compute_sma_slope(candles, self.sma_period)

    def _calculate_adx(self, candles: List[CandleData], period: int = 14) -> Optional[float]:
        """
        ADX placeholder — Wilder's full ADX implementation is Phase 2+.
        Returns None when insufficient data, 0.0 otherwise (stub).
        """
        if len(candles) < period * 2:
            return None
        return None   # Intentional: no fake ADX values in Phase 1

    def _classify_adx_strength(self, adx: Optional[float]) -> TrendStrength:
        """Map ADX float to TrendStrength enum."""
        if adx is None or adx < self.adx_min_tradeable:
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
        confidence: float,
    ) -> int:
        """
        Compute TQS trend component (0–25 pts).

        Rubric:
          25 — tradeable and confidence >= 80
          20 — tradeable and confidence >= 60
          15 — direction confirmed but confidence 40–59
           0 — not tradeable or direction is RANGING/UNDEFINED
        """
        if direction in ("RANGING", "UNDEFINED") or not tradeable:
            if direction in ("UP", "DOWN") and confidence >= 40.0:
                return 15
            return 0
        if confidence >= 80.0:
            return 25
        if confidence >= 60.0:
            return 20
        return 15


# ---------------------------------------------------------------------------
# Private module-level helpers
# ---------------------------------------------------------------------------

def _extract_structure_info(
    market_structure: Optional[object],
) -> Tuple[str, bool]:
    """
    Normalise M03 output into (structure_direction_str, is_ranging) regardless
    of whether a StructureAnalysis or MarketStructure DTO was passed.

    Returns:
        (direction_str, is_ranging_bool)
        direction_str is 'UP', 'DOWN', 'NONE', 'RANGING', or 'UNDEFINED'
    """
    if market_structure is None:
        return "UNDEFINED", False

    # ---- StructureAnalysis (M03 internal object — has is_ranging + structure_type)
    if hasattr(market_structure, "is_ranging") and hasattr(market_structure, "structure_type"):
        sa = market_structure
        is_ranging: bool = bool(sa.is_ranging)
        if is_ranging:
            return _RANGING_LABEL, True
        direction = sa.direction  # TrendDirection enum
        if hasattr(direction, "value"):
            return direction.value, False    # 'UP', 'DOWN', 'NONE'
        return str(direction), False

    # ---- MarketStructure DTO (shared types.py object — has .regime field)
    if hasattr(market_structure, "regime"):
        ms = market_structure
        regime = ms.regime
        if hasattr(regime, "value"):
            dir_str = regime.value   # 'UP', 'DOWN', 'NONE'
        else:
            dir_str = str(regime)
        # MarketStructure DTO does not carry is_ranging; treat NONE as NONE
        return dir_str, False

    return "UNDEFINED", False


def _get_structure_confidence(market_structure: Optional[object]) -> float:
    """
    Extract M03 confidence (0.0–1.0) from structure object.

    Returns 0.0 if unavailable (e.g. MarketStructure DTO has no confidence field).
    """
    if market_structure is None:
        return 0.0
    if hasattr(market_structure, "confidence"):
        conf = market_structure.confidence
        try:
            return float(conf)
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def _compute_sma_slope(candles: List[CandleData], period: int = 21) -> float:
    """
    SMA slope = SMA(latest) - SMA(one-bar-earlier).

    Requires at least period+1 candles.  Returns 0.0 if insufficient.
    """
    if len(candles) < period + 1:
        return 0.0
    sma_latest = sum(c.close for c in candles[-period:]) / period
    sma_prev   = sum(c.close for c in candles[-(period + 1):-1]) / period
    return sma_latest - sma_prev
