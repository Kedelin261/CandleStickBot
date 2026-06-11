"""
M07 — Pattern Recognition Engine (Phase 1 MVP)
Detects Pin Bar and Engulfing Bar patterns from CandleData.
The Candlestick Trading Bible: Trade patterns that show clear price rejection.

Phase 1 patterns:
  - Bullish Pin Bar    (long lower tail rejecting support)
  - Bearish Pin Bar    (long upper tail rejecting resistance)
  - Bullish Engulfing  (current bullish body engulfs prior bearish body)
  - Bearish Engulfing  (current bearish body engulfs prior bullish body)

Deferred to Phase 2:
  - Inside Bar / Inside Bar Breakout / Inside Bar False Breakout
  - Morning Star / Evening Star / Doji patterns

=== ANATOMY HELPERS (module-level, stateless) ===

  body_size(c)        = abs(close - open)
  total_range(c)      = high - low
  upper_wick(c)       = high - max(open, close)
  lower_wick(c)       = min(open, close) - low
  is_bullish(c)       = close > open
  is_bearish(c)       = close < open
  midpoint(c)         = (high + low) / 2
  close_location(c)   = (close - low) / (high - low)   ∈ [0, 1]
                        0 = close at the low, 1 = close at the high

=== PIN BAR RULES ===

Bullish Pin Bar (all must hold):
  1. lower_wick >= body_size * min_tail_ratio   (tail dominates body)
  2. lower_wick >= total_range * 0.60           (tail dominates range)
  3. upper_wick <= body_size * 0.50             (tiny nose wick)
  4. close >= midpoint                          (closes in upper half)
  5. body_size > 0                              (not a doji)
  6. body_size <= total_range * 0.35            (small body)

Bearish Pin Bar (all must hold):
  1. upper_wick >= body_size * min_tail_ratio
  2. upper_wick >= total_range * 0.60
  3. lower_wick <= body_size * 0.50
  4. close <= midpoint
  5. body_size > 0
  6. body_size <= total_range * 0.35

Default min_tail_ratio = 2.0

=== ENGULFING RULES ===

Bullish Engulfing (all must hold):
  1. previous candle is bearish
  2. current candle is bullish
  3. current open  < previous close   (gaps down or at prior close)
  4. current close > previous open    (closes beyond prior open)
  5. current body_size > previous body_size

Bearish Engulfing (all must hold):
  1. previous candle is bullish
  2. current candle is bearish
  3. current open  > previous close
  4. current close < previous open
  5. current body_size > previous body_size

Strict mode (optional, default False):
  +  current high  > previous high
  +  current low   < previous low

=== QUALITY SCORING (1-10) ===

Pin Bar:
  Base valid pattern = 5
  tail_ratio >= 3.0  = +1
  close near tip in trade direction (>= 75% through range from tail end) = +1
  tail price pierces an optional level within pip_size * 5  = +2
  cap at 10

Engulfing:
  Base valid pattern = 5
  current body >= 1.5x previous body  = +1
  current body >= 2.0x previous body  = +1 additional (i.e., +2 total from size)
  strict full-range engulf             = +1
  close beyond previous body extremes = +1
  cap at 10

=== OUTPUT ===

PatternResult dataclass:
  - pattern_type  : str (from PatternType enum value)
  - direction     : str ("LONG" or "SHORT")
  - timestamp     : datetime
  - symbol        : str
  - timeframe     : str
  - quality_score : int 1-10
  - reason        : str   (human-readable explanation)
  - entry_reference  : float (high for bullish pin, low for bearish pin, etc.)
  - stop_reference   : float (low for bullish pin, high for bearish pin)
  - body_size     : float
  - total_range   : float
  - upper_wick    : float
  - lower_wick    : float
  - tail_ratio    : float (pin bar only, else 0.0)
  - engulf_ratio  : float (engulfing only, else 0.0)
  - close_loc     : float (close_location value)
  - strict_engulf : bool  (engulfing only)
  - to_pattern_signal() → PatternSignal DTO

PatternEngine.analyze(candles) → List[PatternResult]
  Scans the full candle series and returns one result per detected pattern
  (no duplicate patterns at the same candle index).

Status: Full implementation — Phase 1 Sprint 6.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from src.types import (
    CandleData,
    Direction,
    PatternSignal,
    PatternType,
)

logger = logging.getLogger("candlestickbot.patterns.pattern_engine")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MIN_TAIL_RATIO:        float = 2.0    # default min_tail_ratio for pin bars
_TAIL_PCT_OF_RANGE:     float = 0.60   # lower_wick (or upper_wick) / total_range
_MAX_BODY_PCT_OF_RANGE: float = 0.35   # body_size / total_range
_MAX_NOSE_WICK_RATIO:   float = 0.50   # nose_wick / body_size

_QUALITY_TAIL_RATIO_BONUS:   float = 3.0   # tail_ratio >= this gets +1
_QUALITY_CLOSE_THRESHOLD:    float = 0.75  # for close-near-tip bonus (75% from tail)
_QUALITY_LEVEL_PROXIMITY:    int   = 5     # pips within which tail counts as level touch

_ENGULF_LARGE_RATIO:  float = 2.0   # body >= 2x prior body for second size bonus
_ENGULF_MEDIUM_RATIO: float = 1.5   # body >= 1.5x prior body for first size bonus


# ---------------------------------------------------------------------------
# PatternResult — unified output for both pattern types
# ---------------------------------------------------------------------------

@dataclass
class PatternResult:
    """
    Detected pattern result from M07 Pattern Recognition Engine.

    Contains all anatomy data plus quality score and trade-reference levels.
    Patterns make no trade decisions — they only describe what was observed.
    """
    pattern_type:    str            # PatternType enum value string
    direction:       str            # "LONG" or "SHORT"
    timestamp:       datetime
    symbol:          str
    timeframe:       str
    quality_score:   int            # 1–10
    reason:          str            # Human-readable detection summary
    entry_reference: float          # Suggested entry reference price
    stop_reference:  float          # Suggested stop-loss reference price

    # Anatomy fields
    body_size:       float = 0.0
    total_range_val: float = 0.0
    upper_wick_val:  float = 0.0
    lower_wick_val:  float = 0.0
    close_loc:       float = 0.0    # 0 = at low, 1 = at high

    # Pattern-specific
    tail_ratio:      float = 0.0    # pin bar: lower_wick (or upper_wick) / body
    engulf_ratio:    float = 0.0    # engulfing: curr body / prev body
    strict_engulf:   bool  = False  # True if wicks also engulf

    def to_pattern_signal(self) -> PatternSignal:
        """Convert to shared ``PatternSignal`` DTO (src.types)."""
        pt_map = {
            "PIN_BAR_BULLISH":    PatternType.PIN_BAR_BULLISH,
            "PIN_BAR_BEARISH":    PatternType.PIN_BAR_BEARISH,
            "ENGULFING_BULLISH":  PatternType.ENGULFING_BULLISH,
            "ENGULFING_BEARISH":  PatternType.ENGULFING_BEARISH,
        }
        dir_map = {"LONG": Direction.LONG, "SHORT": Direction.SHORT}
        return PatternSignal(
            pattern_type=pt_map[self.pattern_type],
            direction=dir_map[self.direction],
            quality_score=float(self.quality_score),
            candle_timestamp=self.timestamp,
            symbol=self.symbol,
            timeframe=self.timeframe,
            suggested_entry=self.entry_reference,
            suggested_stop=self.stop_reference,
            invalidation_price=self.stop_reference,
            details={
                "reason":       self.reason,
                "body_size":    self.body_size,
                "total_range":  self.total_range_val,
                "upper_wick":   self.upper_wick_val,
                "lower_wick":   self.lower_wick_val,
                "close_loc":    self.close_loc,
                "tail_ratio":   self.tail_ratio,
                "engulf_ratio": self.engulf_ratio,
                "strict_engulf": self.strict_engulf,
            },
        )


# ---------------------------------------------------------------------------
# Candle anatomy helpers (module-level, stateless)
# ---------------------------------------------------------------------------

def body_size(candle: CandleData) -> float:
    """Absolute size of the candle body: abs(close - open)."""
    return abs(candle.close - candle.open)


def total_range(candle: CandleData) -> float:
    """Full candle range: high - low."""
    return candle.high - candle.low


def upper_wick(candle: CandleData) -> float:
    """Upper shadow: high - max(open, close)."""
    return candle.high - max(candle.open, candle.close)


def lower_wick(candle: CandleData) -> float:
    """Lower shadow: min(open, close) - low."""
    return min(candle.open, candle.close) - candle.low


def is_bullish(candle: CandleData) -> bool:
    """True when close > open (bullish / green candle)."""
    return candle.close > candle.open


def is_bearish(candle: CandleData) -> bool:
    """True when close < open (bearish / red candle)."""
    return candle.close < candle.open


def midpoint(candle: CandleData) -> float:
    """Midpoint of the candle's full range: (high + low) / 2."""
    return (candle.high + candle.low) / 2.0


def close_location(candle: CandleData) -> float:
    """
    Where the close sits within [low, high], normalised to [0, 1].

    0.0 = close exactly at the low
    1.0 = close exactly at the high
    Returns 0.5 when high == low (zero-range candle).
    """
    rng = total_range(candle)
    if rng <= 0:
        return 0.5
    return (candle.close - candle.low) / rng


# ---------------------------------------------------------------------------
# Pin Bar detection
# ---------------------------------------------------------------------------

def detect_bullish_pin_bar(
    candle: CandleData,
    min_tail_ratio: float = _MIN_TAIL_RATIO,
    level: Optional[float] = None,
    pip_size: float = 0.0001,
) -> Optional[PatternResult]:
    """
    Detect a Bullish Pin Bar on a single candle.

    Rules (all must be satisfied):
      1. lower_wick >= body_size * min_tail_ratio
      2. lower_wick >= total_range * 0.60
      3. upper_wick <= body_size * 0.50
      4. close >= midpoint
      5. body_size > 0
      6. body_size <= total_range * 0.35

    Args:
        candle:         Single CandleData to test.
        min_tail_ratio: Minimum lower_wick / body_size ratio (default 2.0).
        level:          Optional price level; if candle's low is within
                        ``pip_size * 5`` pips, adds +2 to quality score.
        pip_size:       Pip size for level-proximity check.

    Returns:
        PatternResult if detected, else None.
    """
    rng  = total_range(candle)
    bs   = body_size(candle)
    lw   = lower_wick(candle)
    uw   = upper_wick(candle)
    mid  = midpoint(candle)

    # Rule 5: body must exist
    if bs <= 0:
        return None

    # Rule 6: small body
    if bs > rng * _MAX_BODY_PCT_OF_RANGE:
        return None

    # Rule 2: lower wick dominates range
    if rng <= 0 or lw < rng * _TAIL_PCT_OF_RANGE:
        return None

    # Rule 1: lower wick vs body
    if lw < bs * min_tail_ratio:
        return None

    # Rule 3: tiny upper wick (nose)
    if uw > bs * _MAX_NOSE_WICK_RATIO:
        return None

    # Rule 4: close in upper half
    if candle.close < mid:
        return None

    # All rules passed → compute quality
    tail_r = lw / bs
    quality, reason = _pin_quality_and_reason(
        candle=candle,
        tail_ratio=tail_r,
        tail_wick=lw,
        direction="LONG",
        level=level,
        pip_size=pip_size,
    )

    return PatternResult(
        pattern_type="PIN_BAR_BULLISH",
        direction="LONG",
        timestamp=candle.timestamp,
        symbol=candle.symbol,
        timeframe=candle.timeframe,
        quality_score=quality,
        reason=reason,
        entry_reference=candle.high,
        stop_reference=candle.low,
        body_size=bs,
        total_range_val=rng,
        upper_wick_val=uw,
        lower_wick_val=lw,
        close_loc=close_location(candle),
        tail_ratio=tail_r,
    )


def detect_bearish_pin_bar(
    candle: CandleData,
    min_tail_ratio: float = _MIN_TAIL_RATIO,
    level: Optional[float] = None,
    pip_size: float = 0.0001,
) -> Optional[PatternResult]:
    """
    Detect a Bearish Pin Bar on a single candle.

    Rules (all must be satisfied):
      1. upper_wick >= body_size * min_tail_ratio
      2. upper_wick >= total_range * 0.60
      3. lower_wick <= body_size * 0.50
      4. close <= midpoint
      5. body_size > 0
      6. body_size <= total_range * 0.35

    Returns:
        PatternResult if detected, else None.
    """
    rng  = total_range(candle)
    bs   = body_size(candle)
    uw   = upper_wick(candle)
    lw   = lower_wick(candle)
    mid  = midpoint(candle)

    if bs <= 0:
        return None

    if bs > rng * _MAX_BODY_PCT_OF_RANGE:
        return None

    if rng <= 0 or uw < rng * _TAIL_PCT_OF_RANGE:
        return None

    if uw < bs * min_tail_ratio:
        return None

    if lw > bs * _MAX_NOSE_WICK_RATIO:
        return None

    if candle.close > mid:
        return None

    tail_r = uw / bs
    quality, reason = _pin_quality_and_reason(
        candle=candle,
        tail_ratio=tail_r,
        tail_wick=uw,
        direction="SHORT",
        level=level,
        pip_size=pip_size,
    )

    return PatternResult(
        pattern_type="PIN_BAR_BEARISH",
        direction="SHORT",
        timestamp=candle.timestamp,
        symbol=candle.symbol,
        timeframe=candle.timeframe,
        quality_score=quality,
        reason=reason,
        entry_reference=candle.low,
        stop_reference=candle.high,
        body_size=bs,
        total_range_val=rng,
        upper_wick_val=uw,
        lower_wick_val=lw,
        close_loc=close_location(candle),
        tail_ratio=tail_r,
    )


def detect_pin_bar(
    candle: CandleData,
    min_tail_ratio: float = _MIN_TAIL_RATIO,
    level: Optional[float] = None,
    pip_size: float = 0.0001,
) -> Optional[PatternResult]:
    """
    Detect either a Bullish or Bearish Pin Bar.

    Tries bullish first (lower wick dominant), then bearish (upper wick).
    Returns the first match, or None if neither qualifies.
    """
    result = detect_bullish_pin_bar(candle, min_tail_ratio, level, pip_size)
    if result:
        return result
    return detect_bearish_pin_bar(candle, min_tail_ratio, level, pip_size)


# ---------------------------------------------------------------------------
# Engulfing detection
# ---------------------------------------------------------------------------

def detect_bullish_engulfing(
    prev_candle: CandleData,
    curr_candle: CandleData,
    strict: bool = False,
) -> Optional[PatternResult]:
    """
    Detect a Bullish Engulfing pattern from a consecutive candle pair.

    Rules (all must hold):
      1. prev_candle is bearish (close < open)
      2. curr_candle is bullish (close > open)
      3. curr_candle.open  < prev_candle.close
      4. curr_candle.close > prev_candle.open
      5. body_size(curr)   > body_size(prev)

    Strict mode adds:
      6. curr_candle.high  > prev_candle.high
      7. curr_candle.low   < prev_candle.low

    Args:
        prev_candle: The candle immediately before curr_candle.
        curr_candle: The most-recent candle (the potential engulfer).
        strict:      Require full-range engulf (wicks too) if True.

    Returns:
        PatternResult if detected, else None.
    """
    if not is_bearish(prev_candle):
        return None
    if not is_bullish(curr_candle):
        return None

    # Body engulf rules
    if curr_candle.open >= prev_candle.close:
        return None
    if curr_candle.close <= prev_candle.open:
        return None

    bs_curr = body_size(curr_candle)
    bs_prev = body_size(prev_candle)

    if bs_curr <= bs_prev:
        return None

    # Strict mode: wicks must also engulf
    full_engulf = curr_candle.high > prev_candle.high and curr_candle.low < prev_candle.low
    if strict and not full_engulf:
        return None

    engulf_r = (bs_curr / bs_prev) if bs_prev > 0 else float("inf")
    # Close-beyond-previous-body: close > prev body high (prev.open for bearish)
    close_beyond = curr_candle.close > prev_candle.open

    quality, reason = _engulf_quality_and_reason(
        engulf_ratio=engulf_r,
        full_engulf=full_engulf,
        close_beyond=close_beyond,
        direction="LONG",
    )

    return PatternResult(
        pattern_type="ENGULFING_BULLISH",
        direction="LONG",
        timestamp=curr_candle.timestamp,
        symbol=curr_candle.symbol,
        timeframe=curr_candle.timeframe,
        quality_score=quality,
        reason=reason,
        entry_reference=curr_candle.high,
        stop_reference=curr_candle.low,
        body_size=bs_curr,
        total_range_val=total_range(curr_candle),
        upper_wick_val=upper_wick(curr_candle),
        lower_wick_val=lower_wick(curr_candle),
        close_loc=close_location(curr_candle),
        engulf_ratio=engulf_r,
        strict_engulf=full_engulf,
    )


def detect_bearish_engulfing(
    prev_candle: CandleData,
    curr_candle: CandleData,
    strict: bool = False,
) -> Optional[PatternResult]:
    """
    Detect a Bearish Engulfing pattern from a consecutive candle pair.

    Rules (all must hold):
      1. prev_candle is bullish (close > open)
      2. curr_candle is bearish (close < open)
      3. curr_candle.open  > prev_candle.close
      4. curr_candle.close < prev_candle.open
      5. body_size(curr)   > body_size(prev)

    Strict mode adds:
      6. curr_candle.high  > prev_candle.high
      7. curr_candle.low   < prev_candle.low

    Returns:
        PatternResult if detected, else None.
    """
    if not is_bullish(prev_candle):
        return None
    if not is_bearish(curr_candle):
        return None

    if curr_candle.open <= prev_candle.close:
        return None
    if curr_candle.close >= prev_candle.open:
        return None

    bs_curr = body_size(curr_candle)
    bs_prev = body_size(prev_candle)

    if bs_curr <= bs_prev:
        return None

    full_engulf = curr_candle.high > prev_candle.high and curr_candle.low < prev_candle.low
    if strict and not full_engulf:
        return None

    engulf_r = (bs_curr / bs_prev) if bs_prev > 0 else float("inf")
    close_beyond = curr_candle.close < prev_candle.open

    quality, reason = _engulf_quality_and_reason(
        engulf_ratio=engulf_r,
        full_engulf=full_engulf,
        close_beyond=close_beyond,
        direction="SHORT",
    )

    return PatternResult(
        pattern_type="ENGULFING_BEARISH",
        direction="SHORT",
        timestamp=curr_candle.timestamp,
        symbol=curr_candle.symbol,
        timeframe=curr_candle.timeframe,
        quality_score=quality,
        reason=reason,
        entry_reference=curr_candle.low,
        stop_reference=curr_candle.high,
        body_size=bs_curr,
        total_range_val=total_range(curr_candle),
        upper_wick_val=upper_wick(curr_candle),
        lower_wick_val=lower_wick(curr_candle),
        close_loc=close_location(curr_candle),
        engulf_ratio=engulf_r,
        strict_engulf=full_engulf,
    )


def detect_engulfing_bar(
    prev_candle: CandleData,
    curr_candle: CandleData,
    strict: bool = False,
) -> Optional[PatternResult]:
    """
    Detect either a Bullish or Bearish Engulfing Bar.

    Returns:
        PatternResult if detected, else None.
    """
    result = detect_bullish_engulfing(prev_candle, curr_candle, strict)
    if result:
        return result
    return detect_bearish_engulfing(prev_candle, curr_candle, strict)


# ---------------------------------------------------------------------------
# Multi-candle scanner
# ---------------------------------------------------------------------------

def detect_patterns(
    candles: List[CandleData],
    min_tail_ratio: float = _MIN_TAIL_RATIO,
    strict_engulfing: bool = False,
    level: Optional[float] = None,
    pip_size: float = 0.0001,
) -> List[PatternResult]:
    """
    Scan a candle series and return all detected Phase 1 patterns.

    Pin bars are detected on each individual candle.
    Engulfing bars are detected on each consecutive pair.

    No duplicate signals: at most one pattern result per candle index.
    If both a pin bar and engulfing are triggered at the same candle,
    the higher-quality pattern wins (tie → pin bar takes priority).

    Args:
        candles:          Candle series (oldest first, ascending).
        min_tail_ratio:   Minimum tail/body ratio for pin bar (default 2.0).
        strict_engulfing: Require full-range engulf for engulfing patterns.
        level:            Optional S/R level price for pin bar quality bonus.
        pip_size:         Pip size for level-proximity check.

    Returns:
        List of PatternResult, one per candle index (highest-quality per bar).
        Empty if fewer than 1 candle supplied.
    """
    if not candles:
        return []

    results: List[PatternResult] = []
    seen_indices: Dict[int, PatternResult] = {}   # index → best result so far

    def _register(idx: int, result: Optional[PatternResult]) -> None:
        if result is None:
            return
        existing = seen_indices.get(idx)
        if existing is None or result.quality_score > existing.quality_score:
            seen_indices[idx] = result

    for i, candle in enumerate(candles):
        # Pin bar on this candle
        pb = detect_pin_bar(candle, min_tail_ratio, level, pip_size)
        _register(i, pb)

        # Engulfing on pair (prev, curr)
        if i > 0:
            eg = detect_engulfing_bar(candles[i - 1], candle, strict_engulfing)
            _register(i, eg)

    # Collect in index order, no duplicates
    for i in sorted(seen_indices):
        results.append(seen_indices[i])

    logger.debug("M07 detect_patterns: %d candles → %d patterns", len(candles), len(results))
    return results


# ---------------------------------------------------------------------------
# Private scoring helpers
# ---------------------------------------------------------------------------

def _pin_quality_and_reason(
    candle: CandleData,
    tail_ratio: float,
    tail_wick: float,
    direction: str,
    level: Optional[float],
    pip_size: float,
) -> Tuple[int, str]:
    """
    Compute quality score (1-10) and reason string for a pin bar.

    Scoring:
      Base  = 5  (valid pattern passes all rules)
      +1 if tail_ratio >= 3.0
      +1 if close is >= 75% through range from tail end
      +2 if tail's extreme (low for bullish, high for bearish) is within
            pip_size * 5 of an optional level
      Cap at 10.
    """
    score = 5
    notes: List[str] = ["valid pin bar"]

    if tail_ratio >= _QUALITY_TAIL_RATIO_BONUS:
        score += 1
        notes.append(f"tail_ratio {tail_ratio:.2f} >= {_QUALITY_TAIL_RATIO_BONUS}")

    rng = total_range(candle)
    cl  = close_location(candle)
    if direction == "LONG":
        # Bullish: good close is near the high (close_location >= 0.75)
        if cl >= _QUALITY_CLOSE_THRESHOLD:
            score += 1
            notes.append(f"close location {cl:.2f} >= {_QUALITY_CLOSE_THRESHOLD}")
        tail_extreme = candle.low
    else:
        # Bearish: good close is near the low (close_location <= 0.25)
        if cl <= (1.0 - _QUALITY_CLOSE_THRESHOLD):
            score += 1
            notes.append(f"close location {cl:.2f} <= {1.0-_QUALITY_CLOSE_THRESHOLD}")
        tail_extreme = candle.high

    if level is not None:
        proximity = abs(tail_extreme - level)
        if proximity <= pip_size * _QUALITY_LEVEL_PROXIMITY:
            score += 2
            notes.append(
                f"tail extreme {tail_extreme:.5f} within {_QUALITY_LEVEL_PROXIMITY} pips "
                f"of level {level:.5f}"
            )

    score = min(score, 10)
    return score, "; ".join(notes)


def _engulf_quality_and_reason(
    engulf_ratio: float,
    full_engulf:  bool,
    close_beyond: bool,
    direction:    str,
) -> Tuple[int, str]:
    """
    Compute quality score (1-10) and reason string for an engulfing bar.

    Scoring:
      Base  = 5  (valid pattern)
      +1 if current body >= 1.5x previous body
      +1 additional (total +2 from size) if >= 2.0x previous body
      +1 if strict full-range engulf (wicks also engulf)
      +1 if close is beyond previous body extreme
      Cap at 10.
    """
    score = 5
    notes: List[str] = [f"valid {direction.lower()} engulfing"]

    if engulf_ratio == float("inf") or engulf_ratio >= _ENGULF_LARGE_RATIO:
        score += 2   # Both size bonuses at once
        notes.append(f"body ratio {engulf_ratio:.2f} >= {_ENGULF_LARGE_RATIO}x (+2)")
    elif engulf_ratio >= _ENGULF_MEDIUM_RATIO:
        score += 1
        notes.append(f"body ratio {engulf_ratio:.2f} >= {_ENGULF_MEDIUM_RATIO}x (+1)")

    if full_engulf:
        score += 1
        notes.append("strict full-range engulf (+1)")

    if close_beyond:
        score += 1
        notes.append("close beyond previous body extreme (+1)")

    score = min(score, 10)
    return score, "; ".join(notes)


# ---------------------------------------------------------------------------
# PatternEngine class (stateful configuration wrapper)
# ---------------------------------------------------------------------------

class PatternEngine:
    """
    M07 — Pattern Recognition Engine.

    Phase 1 patterns: Pin Bar (bullish/bearish) + Engulfing Bar (bullish/bearish).

    Configuration is fixed at construction time; the engine is stateless
    between ``analyze()`` calls.

    Usage::

        engine = PatternEngine()
        patterns = engine.analyze(candles)
        for p in patterns:
            signal = p.to_pattern_signal()
    """

    def __init__(
        self,
        min_tail_ratio:   float = _MIN_TAIL_RATIO,
        strict_engulfing: bool  = False,
        pip_size:         float = 0.0001,
    ):
        if min_tail_ratio <= 0:
            raise ValueError(f"min_tail_ratio must be > 0, got {min_tail_ratio}")
        if pip_size <= 0:
            raise ValueError(f"pip_size must be > 0, got {pip_size}")
        self.min_tail_ratio   = min_tail_ratio
        self.strict_engulfing = strict_engulfing
        self.pip_size         = pip_size

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def analyze(
        self,
        candles: List[CandleData],
        level: Optional[float] = None,
    ) -> List[PatternResult]:
        """
        Detect all Phase 1 patterns in a candle series.

        Args:
            candles: Candle series (oldest first, ascending).
            level:   Optional price level for pin bar quality scoring.

        Returns:
            List of PatternResult (one per candle, highest-quality pattern).
            Always returns a list; empty if no patterns found.
        """
        return detect_patterns(
            candles=candles,
            min_tail_ratio=self.min_tail_ratio,
            strict_engulfing=self.strict_engulfing,
            level=level,
            pip_size=self.pip_size,
        )

    def scan_pin_bars(
        self,
        candles: List[CandleData],
        level: Optional[float] = None,
    ) -> List[PatternResult]:
        """Detect only pin bar patterns (bullish and bearish)."""
        results = []
        for c in candles:
            r = detect_pin_bar(c, self.min_tail_ratio, level, self.pip_size)
            if r:
                results.append(r)
        return results

    def scan_engulfing(
        self,
        candles: List[CandleData],
    ) -> List[PatternResult]:
        """Detect only engulfing bar patterns (bullish and bearish)."""
        results = []
        for i in range(1, len(candles)):
            r = detect_engulfing_bar(candles[i - 1], candles[i], self.strict_engulfing)
            if r:
                results.append(r)
        return results

    def calculate_tqs_pattern_score(self, result: PatternResult) -> int:
        """
        Map a PatternResult quality score (1-10) to a TQS pattern component
        score (0-25).

        Mapping:
          quality 8-10 → 25
          quality 6-7  → 20
          quality 5    → 15
          quality 1-4  → 10
        """
        q = result.quality_score
        if q >= 8:
            return 25
        if q >= 6:
            return 20
        if q >= 5:
            return 15
        return 10
