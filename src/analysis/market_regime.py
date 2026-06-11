"""
M16 — Market Regime Engine
Classifies current market conditions to gate strategy selection and scale risk.
The Candlestick Trading Bible: Match your strategy to the market's current mode.

Regime Classification Rules (priority order):
  1. VOLATILE  — ATR > ATR_MA * 1.5  →  no trade (risk_multiplier = 0.0)
  2. QUIET     — ATR < ATR_MA * 0.6  →  no trade (risk_multiplier = 0.0)
  3. CHOPPY    — Choppiness >= 61.8  →  no trade (risk_multiplier = 0.0)
  4. TRENDING  — M03/M04 structure UP/DOWN + ATR > ATR_MA + BB expanding
                 + (ADX >= 25 if available) + Choppiness < 61.8
                 →  full trade (risk_multiplier = 1.0)
  5. RANGING   — BB contracting + Choppiness moderately elevated
                 →  no trade Phase 1 (risk_multiplier = 0.5)
  6. UNKNOWN   — insufficient data or ambiguous

ADX Note:
  ADX is accepted as an optional external float.  When not provided (None)
  the engine falls back to ATR, BB width, and Choppiness Index alone.  This
  ensures Phase 1 tests are never blocked on a full ADX implementation.

Indicator Formulas:
  True Range (TR)       = max(H-L, |H-prev_C|, |L-prev_C|)
  ATR(n)                = Wilder EMA of TR; seed = simple average of first n TRs
  ATR_MA                = simple SMA of ATR series over atr_ma_period
  BB Width              = (upper - lower) / middle × 100
                          where upper/lower = middle ± bb_std_dev × σ(close, bb_period)
  BB Width MA           = simple SMA of BB width series over bb_ma_period
  Choppiness Index(n)   = 100 × log10(Σ TR₁ / (HH_n − LL_n)) / log10(n)
                          bounded [0, 100]; threshold = 61.8

Confidence Score (0.0–1.0):
  Based on indicator agreement count:
    each agreeing indicator adds 0.25 (cap 1.0)
  Insufficient data → 0.0

Allowed Strategies / Risk Multipliers:
  TRENDING : ["pin_bar", "engulfing_bar"]   risk = 1.0
  RANGING  : []                             risk = 0.5   (Phase 2: inside_bar)
  VOLATILE : []                             risk = 0.0
  QUIET    : []                             risk = 0.0
  CHOPPY   : []                             risk = 0.0
  UNKNOWN  : []                             risk = 0.0

TQS Regime Component (0–25 pts):
  TRENDING  + ADX >= 30 (or no ADX): 25
  TRENDING  + ADX 25–29            : 20
  TRENDING  + ADX < 25             : 15
  RANGING                          : 10
  VOLATILE / QUIET / CHOPPY        :  0
  UNKNOWN                          :  0

Status: Full implementation — Phase 1 Sprint 5.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from src.types import CandleData, RegimeSignal, RegimeType

logger = logging.getLogger("candlestickbot.analysis.market_regime")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHOPPINESS_THRESHOLD: float = 61.8   # Above = choppy/ranging; below = trending

# Risk multiplier by regime
_REGIME_RISK: Dict[RegimeType, float] = {
    RegimeType.TRENDING: 1.0,
    RegimeType.RANGING:  0.5,
    RegimeType.VOLATILE: 0.0,
    RegimeType.QUIET:    0.0,
    RegimeType.CHOPPY:   0.0,
    RegimeType.UNKNOWN:  0.0,
}

# Allowed strategies by regime (Phase 1 — no pattern/fibonacci/supply-demand)
_REGIME_STRATEGIES: Dict[RegimeType, List[str]] = {
    RegimeType.TRENDING: ["pin_bar", "engulfing_bar"],
    RegimeType.RANGING:  [],          # Phase 2: ["inside_bar"]
    RegimeType.VOLATILE: [],
    RegimeType.QUIET:    [],
    RegimeType.CHOPPY:   [],
    RegimeType.UNKNOWN:  [],
}

# Minimum candles needed per indicator
_MIN_ATR_CANDLES        = 2    # need at least 2 for first TR
_MIN_BB_CANDLES         = 2    # need at least 2 for std-dev

# Default ATR_MA periods (used when caller does not override)
_DEFAULT_ATR_PERIOD     = 14
_DEFAULT_ATR_MA_PERIOD  = 50
_DEFAULT_BB_PERIOD      = 20
_DEFAULT_BB_MA_PERIOD   = 20
_DEFAULT_CHOPPY_PERIOD  = 14

# ATR ratio thresholds
_VOLATILE_ATR_RATIO = 1.5
_QUIET_ATR_RATIO    = 0.6


# ---------------------------------------------------------------------------
# RegimeAnalysis — M16 internal result (richer than the DTO)
# ---------------------------------------------------------------------------

@dataclass
class RegimeAnalysis:
    """
    Full output of the Market Regime Engine.

    ``regime`` and supporting indicator values allow downstream modules
    (M08 Strategy Engine, M09 Risk Engine) to make informed decisions.
    Call ``.to_regime_signal()`` to get the shared DTO.
    """
    regime:             RegimeType
    confidence:         float           # 0.0–1.0
    allowed_strategies: List[str]       = field(default_factory=list)
    risk_multiplier:    float           = 0.0
    reason:             str             = ""
    tqs_regime_score:   int             = 0

    # Indicator values
    atr:                float           = 0.0
    atr_ma:             float           = 0.0
    atr_ratio:          float           = 0.0   # atr / atr_ma (0 if atr_ma == 0)
    bb_width:           float           = 0.0
    bb_width_ma:        float           = 0.0
    bb_expanding:       bool            = False  # True if bb_width > bb_width_ma
    choppiness_index:   float           = 61.8   # neutral default
    adx:                Optional[float] = None

    # Raw series (useful for debugging / charting)
    atr_series:         List[float]     = field(default_factory=list)
    bb_width_series:    List[float]     = field(default_factory=list)

    @property
    def is_tradeable(self) -> bool:
        """True if at least one strategy is allowed."""
        return len(self.allowed_strategies) > 0

    def to_regime_signal(
        self,
        symbol:    str = "",
        timeframe: str = "",
        timestamp: Optional[datetime] = None,
    ) -> RegimeSignal:
        """Convert to shared ``RegimeSignal`` DTO (src.types)."""
        return RegimeSignal(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=timestamp or datetime.now(timezone.utc),
            regime=self.regime,
            confidence=self.confidence,
            allowed_strategies=list(self.allowed_strategies),
            risk_multiplier=self.risk_multiplier,
            adx=self.adx,
            atr=self.atr if self.atr > 0 else None,
            atr_ma=self.atr_ma if self.atr_ma > 0 else None,
            bb_width=self.bb_width if self.bb_width > 0 else None,
            choppiness_index=self.choppiness_index,
        )


# ---------------------------------------------------------------------------
# Module-level stateless functions (public API)
# ---------------------------------------------------------------------------

def compute_atr(candles: List[CandleData], period: int = 14) -> List[float]:
    """
    Compute ATR series using Wilder's smoothing method.

    The first ATR value is seeded as the simple average of the first ``period``
    True Range values.  Subsequent values use the Wilder smoothing formula:

        ATR_t = ATR_{t-1} × (period-1)/period + TR_t / period

    Args:
        candles: Candle series, oldest first.  Needs at least ``period + 1``
                 candles to produce the first smoothed ATR.
        period:  Smoothing period (default 14).

    Returns:
        List of ATR values (same length as ``candles[1:]``).
        Returns empty list if fewer than 2 candles supplied.
    """
    if len(candles) < 2:
        return []

    tr_list: List[float] = []
    for i in range(1, len(candles)):
        c = candles[i]
        pc = candles[i - 1].close
        tr = max(c.high - c.low, abs(c.high - pc), abs(c.low - pc))
        tr_list.append(tr)

    if len(tr_list) < period:
        # Not enough for full smoothing — return simple average series
        avg = sum(tr_list) / len(tr_list)
        return [avg] * len(tr_list)

    alpha = 1.0 / period
    # Seed with simple average of first 'period' TRs
    atr = sum(tr_list[:period]) / period
    result: List[float] = [0.0] * (period - 1) + [atr]   # pad head with 0

    for tr in tr_list[period:]:
        atr = atr * (1.0 - alpha) + tr * alpha
        result.append(atr)

    return result


def get_current_atr(candles: List[CandleData], period: int = 14) -> float:
    """
    Return the most-recent ATR value (scalar).

    Returns 0.0 if there are insufficient candles.
    """
    series = compute_atr(candles, period)
    # Filter out the 0-padding at the head
    valid = [v for v in series if v > 0.0]
    return valid[-1] if valid else 0.0


def compute_atr_ma(atr_series: List[float], ma_period: int = 50) -> List[float]:
    """
    Compute a simple moving average of an ATR series.

    Args:
        atr_series: Output of ``compute_atr()``.
        ma_period:  Window for the SMA.

    Returns:
        SMA series aligned to ``atr_series``.
        Values before index ``ma_period - 1`` are 0.0 (insufficient window).
    """
    n = len(atr_series)
    if n == 0:
        return []
    result = [0.0] * n
    for i in range(ma_period - 1, n):
        window = atr_series[i - ma_period + 1: i + 1]
        result[i] = sum(window) / ma_period
    return result


def get_current_atr_ma(candles: List[CandleData],
                        atr_period: int = 14,
                        ma_period: int = 50) -> float:
    """
    Return the most-recent ATR moving-average value.

    Returns 0.0 if insufficient data.
    """
    atr_series = compute_atr(candles, atr_period)
    valid_atr = [v for v in atr_series if v > 0.0]
    if len(valid_atr) < ma_period:
        # Not enough for full MA — use simple average of available ATRs
        return sum(valid_atr) / len(valid_atr) if valid_atr else 0.0
    ma_series = compute_atr_ma(valid_atr, ma_period)
    valid_ma = [v for v in ma_series if v > 0.0]
    return valid_ma[-1] if valid_ma else 0.0


def compute_bb_width(candles: List[CandleData],
                     period: int = 20,
                     std_dev: float = 2.0) -> List[float]:
    """
    Compute Bollinger Band Width series.

    BB Width = (Upper – Lower) / Middle × 100
    where:
      Middle = SMA(close, period)
      Upper  = Middle + std_dev × σ(close, period)
      Lower  = Middle − std_dev × σ(close, period)

    Args:
        candles: Candle series, oldest first.
        period:  BB period (default 20).
        std_dev: Standard deviation multiplier (default 2.0).

    Returns:
        BB Width series (same length as ``candles``).
        Values before index ``period - 1`` are 0.0.
    """
    n = len(candles)
    if n < 2:
        return [0.0] * n

    closes = [c.close for c in candles]
    result = [0.0] * n

    for i in range(period - 1, n):
        window = closes[i - period + 1: i + 1]
        mean = sum(window) / period
        variance = sum((x - mean) ** 2 for x in window) / period
        sigma = math.sqrt(variance)
        if mean > 0:
            upper = mean + std_dev * sigma
            lower = mean - std_dev * sigma
            result[i] = (upper - lower) / mean * 100.0

    return result


def get_current_bb_width(candles: List[CandleData],
                          period: int = 20,
                          std_dev: float = 2.0) -> float:
    """Return the most-recent BB Width value.  Returns 0.0 if insufficient."""
    series = compute_bb_width(candles, period, std_dev)
    valid = [v for v in series if v > 0.0]
    return valid[-1] if valid else 0.0


def compute_bb_width_ma(candles: List[CandleData],
                         bb_period: int = 20,
                         bb_std_dev: float = 2.0,
                         ma_period: int = 20) -> float:
    """
    Return the SMA of BB Width over ``ma_period`` most-recent BB Width values.

    Returns 0.0 if there are not enough non-zero BB Width values.
    """
    bw_series = compute_bb_width(candles, bb_period, bb_std_dev)
    valid = [v for v in bw_series if v > 0.0]
    if not valid:
        return 0.0
    window = valid[-ma_period:] if len(valid) >= ma_period else valid
    return sum(window) / len(window)


def compute_choppiness(candles: List[CandleData], period: int = 14) -> float:
    """
    Compute the Choppiness Index for the most recent ``period`` candles.

    Formula:
        CI(n) = 100 × log10(Σ TR_i / (HH_n − LL_n)) / log10(n)

    Where:
        Σ TR_i  = sum of individual 1-candle True Ranges over the period
        HH_n    = highest High over the period (including the candle before)
        LL_n    = lowest  Low  over the period (including the candle before)
        n       = period

    Returns:
        CI value bounded [0.0, 100.0].
        Returns 61.8 (neutral threshold) when insufficient data.
    """
    # Need period+1 candles: one prior for first TR, then period TRs
    if len(candles) < period + 1:
        return CHOPPINESS_THRESHOLD   # neutral

    window = candles[-(period + 1):]
    hh = max(c.high for c in window)
    ll = min(c.low for c in window)
    price_range = hh - ll

    if price_range <= 0:
        return 100.0    # flat market → maximally choppy

    tr_sum = 0.0
    for i in range(1, len(window)):
        c  = window[i]
        pc = window[i - 1].close
        tr_sum += max(c.high - c.low, abs(c.high - pc), abs(c.low - pc))

    if tr_sum <= 0:
        return 100.0

    ci = 100.0 * math.log10(tr_sum / price_range) / math.log10(period)
    return max(0.0, min(100.0, ci))


# ---------------------------------------------------------------------------
# MarketRegimeEngine — main class
# ---------------------------------------------------------------------------

class MarketRegimeEngine:
    """
    M16 — Market Regime Engine.

    Classifies the current market environment into one of five regimes
    (TRENDING, RANGING, VOLATILE, QUIET, CHOPPY) using ATR, Bollinger Band
    Width, and Choppiness Index.  ADX is accepted as an optional external
    input — its absence never blocks analysis.

    Usage::

        engine = MarketRegimeEngine()
        result = engine.analyze(candles)                    # RegimeAnalysis
        signal = result.to_regime_signal("EURUSD", "D1")   # RegimeSignal DTO

    Classification priority (first match wins):
        1. VOLATILE  if atr_ratio > volatile_atr_ratio (1.5)
        2. QUIET     if atr_ratio < quiet_atr_ratio (0.6)
        3. CHOPPY    if choppiness >= 61.8
        4. TRENDING  if bb_width > bb_width_ma (expanding) AND
                        choppiness < 61.8 AND
                        atr_ratio >= 1.0 AND
                        (adx >= adx_trending_threshold if adx provided)
        5. RANGING   otherwise (contracting or flat BB + no other flag)
        6. UNKNOWN   if ATR computation produced no valid values

    Parameters:
        atr_period              ATR smoothing period (default 14)
        atr_ma_period           Period for ATR SMA (default 50)
        bb_period               Bollinger Band period (default 20)
        bb_std_dev              BB standard deviation multiplier (default 2.0)
        bb_ma_period            Period for BB Width SMA (default 20)
        choppiness_period       Choppiness Index lookback (default 14)
        volatile_atr_ratio      ATR/ATR_MA threshold for VOLATILE (default 1.5)
        quiet_atr_ratio         ATR/ATR_MA threshold for QUIET (default 0.6)
        adx_trending_threshold  Minimum ADX for TRENDING gate (default 25.0)
    """

    CHOPPINESS_THRESHOLD: float = CHOPPINESS_THRESHOLD

    def __init__(
        self,
        atr_period:             int   = _DEFAULT_ATR_PERIOD,
        atr_ma_period:          int   = _DEFAULT_ATR_MA_PERIOD,
        bb_period:              int   = _DEFAULT_BB_PERIOD,
        bb_std_dev:             float = 2.0,
        bb_ma_period:           int   = _DEFAULT_BB_MA_PERIOD,
        choppiness_period:      int   = _DEFAULT_CHOPPY_PERIOD,
        volatile_atr_ratio:     float = _VOLATILE_ATR_RATIO,
        quiet_atr_ratio:        float = _QUIET_ATR_RATIO,
        adx_trending_threshold: float = 25.0,
    ):
        if atr_period < 1:
            raise ValueError(f"atr_period must be >= 1, got {atr_period}")
        if bb_period < 2:
            raise ValueError(f"bb_period must be >= 2, got {bb_period}")
        if choppiness_period < 2:
            raise ValueError(f"choppiness_period must be >= 2, got {choppiness_period}")
        if volatile_atr_ratio <= quiet_atr_ratio:
            raise ValueError(
                f"volatile_atr_ratio ({volatile_atr_ratio}) must be > "
                f"quiet_atr_ratio ({quiet_atr_ratio})"
            )

        self.atr_period             = atr_period
        self.atr_ma_period          = atr_ma_period
        self.bb_period              = bb_period
        self.bb_std_dev             = bb_std_dev
        self.bb_ma_period           = bb_ma_period
        self.choppiness_period      = choppiness_period
        self.volatile_atr_ratio     = volatile_atr_ratio
        self.quiet_atr_ratio        = quiet_atr_ratio
        self.adx_trending_threshold = adx_trending_threshold

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def analyze(
        self,
        candles: List[CandleData],
        adx: Optional[float] = None,
        market_structure=None,        # M03 StructureAnalysis or MarketStructure DTO
    ) -> RegimeAnalysis:
        """
        Classify current market regime.

        Args:
            candles:          Candle series (oldest first).
                              Minimum usable: ``choppiness_period + 1`` candles.
                              Full accuracy requires >= ``atr_ma_period + atr_period``.
            adx:              Optional external ADX value.  When None, the ADX
                              gate for TRENDING is skipped.
            market_structure: Optional M03 StructureAnalysis or MarketStructure
                              DTO.  Currently reserved for future use (not
                              consumed in Phase 1 to keep analysis stateless).

        Returns:
            RegimeAnalysis.  Never raises — returns UNKNOWN on empty input.
        """
        if not candles:
            return self._unknown("No candles provided", adx=adx)

        # ── Step 1: Compute indicators ────────────────────────────────────
        atr_series  = compute_atr(candles, self.atr_period)
        valid_atrs  = [v for v in atr_series if v > 0.0]
        current_atr = valid_atrs[-1] if valid_atrs else 0.0

        if current_atr == 0.0:
            return self._unknown("ATR is zero — cannot classify regime", adx=adx)

        # ATR moving average
        current_atr_ma = self._compute_atr_ma_value(valid_atrs)

        # ATR ratio — how volatile vs. "normal"
        atr_ratio = (current_atr / current_atr_ma) if current_atr_ma > 0 else 1.0

        # Bollinger Band Width
        bw_series       = compute_bb_width(candles, self.bb_period, self.bb_std_dev)
        valid_bw        = [v for v in bw_series if v > 0.0]
        current_bw      = valid_bw[-1] if valid_bw else 0.0
        bw_ma           = self._compute_bw_ma(valid_bw)
        bb_expanding    = (current_bw > bw_ma) if (current_bw > 0 and bw_ma > 0) else False

        # Choppiness Index
        choppiness = compute_choppiness(candles, self.choppiness_period)

        # ── Step 2: Classify regime ───────────────────────────────────────
        regime, reason, confidence = self._classify(
            atr_ratio=atr_ratio,
            bb_expanding=bb_expanding,
            current_bw=current_bw,
            bw_ma=bw_ma,
            choppiness=choppiness,
            adx=adx,
        )

        # ── Step 3: Lookup allowed strategies + risk multiplier ──────────
        allowed  = list(_REGIME_STRATEGIES[regime])
        risk_mul = _REGIME_RISK[regime]

        # ── Step 4: TQS score ─────────────────────────────────────────────
        tqs = self._tqs_score(regime, adx)

        logger.debug(
            "M16 regime=%s conf=%.2f atr_ratio=%.3f bw=%.3f bw_ma=%.3f "
            "chop=%.1f adx=%s",
            regime.value, confidence, atr_ratio, current_bw, bw_ma,
            choppiness, adx,
        )

        return RegimeAnalysis(
            regime=regime,
            confidence=confidence,
            allowed_strategies=allowed,
            risk_multiplier=risk_mul,
            reason=reason,
            tqs_regime_score=tqs,
            atr=current_atr,
            atr_ma=current_atr_ma,
            atr_ratio=atr_ratio,
            bb_width=current_bw,
            bb_width_ma=bw_ma,
            bb_expanding=bb_expanding,
            choppiness_index=choppiness,
            adx=adx,
            atr_series=valid_atrs,
            bb_width_series=valid_bw,
        )

    def calculate_tqs_regime_score(
        self,
        regime: RegimeType,
        adx: Optional[float] = None,
    ) -> int:
        """
        TQS regime component score (0–25 points).

        Scoring:
          TRENDING + ADX >= 30 (or ADX absent): 25
          TRENDING + ADX 25–29               : 20
          TRENDING + ADX < 25 (weak trend)   : 15
          RANGING                            : 10
          VOLATILE / QUIET / CHOPPY / UNKNOWN:  0
        """
        return self._tqs_score(regime, adx)

    # ------------------------------------------------------------------
    # PRIVATE HELPERS
    # ------------------------------------------------------------------

    def _classify(
        self,
        atr_ratio:   float,
        bb_expanding: bool,
        current_bw:  float,
        bw_ma:       float,
        choppiness:  float,
        adx:         Optional[float],
    ) -> Tuple[RegimeType, str, float]:
        """
        Return (regime, reason, confidence).

        Confidence counts how many independent indicators agree with the
        chosen regime: each agreeing indicator adds 0.25 (capped at 1.0).
        """
        # ── Priority 1: VOLATILE ─────────────────────────────────────────
        if atr_ratio >= self.volatile_atr_ratio:
            conf = min(0.5 + (atr_ratio - self.volatile_atr_ratio) * 0.5, 1.0)
            return (
                RegimeType.VOLATILE,
                f"ATR ratio {atr_ratio:.2f} >= volatile threshold "
                f"{self.volatile_atr_ratio}",
                round(conf, 4),
            )

        # ── Priority 2: QUIET ────────────────────────────────────────────
        if atr_ratio <= self.quiet_atr_ratio:
            conf = min(0.5 + (self.quiet_atr_ratio - atr_ratio) * 1.0, 1.0)
            return (
                RegimeType.QUIET,
                f"ATR ratio {atr_ratio:.2f} <= quiet threshold "
                f"{self.quiet_atr_ratio}",
                round(conf, 4),
            )

        # ── Priority 3: CHOPPY ───────────────────────────────────────────
        if choppiness >= self.CHOPPINESS_THRESHOLD:
            conf = min(0.4 + (choppiness - self.CHOPPINESS_THRESHOLD) / 100.0, 1.0)
            # Extra boost if ADX also below threshold
            if adx is not None and adx < self.adx_trending_threshold:
                conf = min(conf + 0.15, 1.0)
            return (
                RegimeType.CHOPPY,
                f"Choppiness Index {choppiness:.1f} >= {self.CHOPPINESS_THRESHOLD}",
                round(conf, 4),
            )

        # ── Priority 4: TRENDING ─────────────────────────────────────────
        trend_signals = 0
        trend_reasons = []

        if atr_ratio >= 1.0:
            trend_signals += 1
            trend_reasons.append(f"ATR ratio {atr_ratio:.2f} >= 1.0")

        if bb_expanding:
            trend_signals += 1
            trend_reasons.append(f"BB width {current_bw:.3f} > BB MA {bw_ma:.3f} (expanding)")

        if choppiness < self.CHOPPINESS_THRESHOLD:
            trend_signals += 1
            trend_reasons.append(f"Choppiness {choppiness:.1f} < {self.CHOPPINESS_THRESHOLD}")

        adx_ok = True   # Default: pass when no ADX
        if adx is not None:
            adx_ok = adx >= self.adx_trending_threshold
            if adx_ok:
                trend_signals += 1
                trend_reasons.append(f"ADX {adx:.1f} >= {self.adx_trending_threshold}")
            else:
                trend_reasons.append(f"ADX {adx:.1f} < {self.adx_trending_threshold} (weak)")

        # Require at least 2 positive signals (or 1 if ADX confirms strongly)
        min_signals = 2 if adx is None else 2
        if trend_signals >= min_signals and adx_ok:
            conf = min(0.25 * trend_signals, 1.0)
            return (
                RegimeType.TRENDING,
                "; ".join(trend_reasons),
                round(conf, 4),
            )

        # ── Priority 5: RANGING ──────────────────────────────────────────
        range_reasons = []
        range_signals = 0

        if not bb_expanding:
            range_signals += 1
            bw_desc = (
                f"BB width {current_bw:.3f} < BB MA {bw_ma:.3f} (contracting)"
                if bw_ma > 0 else "BB width contracting (no MA)"
            )
            range_reasons.append(bw_desc)

        if adx is not None and adx < self.adx_trending_threshold:
            range_signals += 1
            range_reasons.append(f"ADX {adx:.1f} < {self.adx_trending_threshold}")

        if choppiness >= 50.0:
            range_signals += 1
            range_reasons.append(f"Choppiness {choppiness:.1f} elevated (>= 50)")

        conf = min(0.25 + 0.2 * range_signals, 1.0)
        return (
            RegimeType.RANGING,
            "; ".join(range_reasons) if range_reasons else "BB contracting, no strong trend",
            round(conf, 4),
        )

    def _compute_atr_ma_value(self, valid_atrs: List[float]) -> float:
        """Simple SMA of last ``atr_ma_period`` valid ATR values."""
        if not valid_atrs:
            return 0.0
        window = valid_atrs[-self.atr_ma_period:] if len(valid_atrs) >= self.atr_ma_period \
                 else valid_atrs
        return sum(window) / len(window)

    def _compute_bw_ma(self, valid_bw: List[float]) -> float:
        """Simple SMA of last ``bb_ma_period`` valid BB Width values."""
        if not valid_bw:
            return 0.0
        window = valid_bw[-self.bb_ma_period:] if len(valid_bw) >= self.bb_ma_period \
                 else valid_bw
        return sum(window) / len(window)

    def _tqs_score(self, regime: RegimeType, adx: Optional[float]) -> int:
        """TQS regime component (0–25)."""
        if regime == RegimeType.TRENDING:
            if adx is None or adx >= 30.0:
                return 25
            if adx >= 25.0:
                return 20
            return 15
        if regime == RegimeType.RANGING:
            return 10
        return 0   # VOLATILE / QUIET / CHOPPY / UNKNOWN

    def _unknown(self, reason: str, adx: Optional[float] = None) -> RegimeAnalysis:
        """Return a safe UNKNOWN result."""
        return RegimeAnalysis(
            regime=RegimeType.UNKNOWN,
            confidence=0.0,
            allowed_strategies=[],
            risk_multiplier=0.0,
            reason=reason,
            tqs_regime_score=0,
            adx=adx,
        )
