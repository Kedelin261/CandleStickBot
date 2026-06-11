"""
M07 — Pattern Detection: Pin Bar
The Candlestick Trading Bible: Pin bars are the clearest rejection candles.

Pin Bar Definition:
  - Long tail (wick) rejecting a key level
  - Small body at one end of the candle
  - Tail is >= 2/3 of total candle range

Bullish Pin Bar (BPB) — at support:
  - Long lower tail (rejection of lower prices)
  - Close in upper 1/3 of total range
  - Open and close in upper 1/3 of range
  - Signal: price rejected support → expect LONG

Bearish Pin Bar (BPB) — at resistance:
  - Long upper tail (rejection of higher prices)
  - Close in lower 1/3 of total range
  - Open and close in lower 1/3 of range
  - Signal: price rejected resistance → expect SHORT

Quality scoring (1-10):
  - 10: Perfect tail ratio (>4x), close near high/low, tiny body
  - 7-9: Strong tail ratio (>3x), clean body placement
  - 5-6: Acceptable tail ratio (≥2x), passes minimum criteria
  - <5: Fails minimum criteria (not detected)

Phase 1: EURUSD D1 only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from src.types import CandleData, PatternSignal

logger = logging.getLogger("candlestickbot.patterns.pin_bar")


class PinBarType(str, Enum):
    """Pin bar directional classification."""
    BULLISH = "PIN_BAR_BULLISH"
    BEARISH = "PIN_BAR_BEARISH"


@dataclass
class PinBarResult:
    """
    Result of pin bar detection.

    If detected=False, all other fields have their default/None values.
    """
    detected: bool
    pin_bar_type: Optional[PinBarType] = None
    quality_score: int = 0             # 1-10
    tail_ratio: float = 0.0            # Tail / Body ratio
    tail_pct_of_range: float = 0.0     # Tail as % of total range
    body_position: float = 0.0         # 0=at bottom, 1=at top of range
    suggested_entry: Optional[float] = None
    suggested_stop: Optional[float] = None
    reject_reason: Optional[str] = None

    def to_pattern_signal(self, candle: CandleData) -> Optional[PatternSignal]:
        """Convert to shared PatternSignal DTO."""
        if not self.detected or self.pin_bar_type is None:
            return None
        return PatternSignal(
            pattern_type=self.pin_bar_type.value,
            direction="LONG" if self.pin_bar_type == PinBarType.BULLISH else "SHORT",
            quality_score=self.quality_score,
            suggested_entry=self.suggested_entry,
            suggested_stop=self.suggested_stop,
            invalidation_price=self.suggested_stop,
        )


class PinBarDetector:
    """
    M07 Pin Bar Pattern Detector.

    Parameters (from config/default_config.yaml):
        min_tail_ratio: Minimum tail/body ratio (default: 2.0)
        min_tail_pct_of_range: Minimum tail as % of total range (default: 0.66)
        max_body_pct_of_range: Maximum body size as % of range (default: 0.33)
        quality_min_score: Minimum quality score to signal (default: 5)
        buffer_pips: Entry buffer above/below nose (default: 2.0 pips)
        pip_size: Pip size for symbol (default: 0.0001 for EURUSD)

    Detection Algorithm:
    1. Calculate total range (high - low)
    2. Identify the tail and nose based on candl structure
    3. Calculate tail/body ratio
    4. Verify tail is >= 2/3 of total range
    5. Verify body is in upper 1/3 (bullish) or lower 1/3 (bearish)
    6. Score quality based on tail length, body size, and position
    """

    def __init__(
        self,
        min_tail_ratio: float = 2.0,
        min_tail_pct_of_range: float = 0.66,
        max_body_pct_of_range: float = 0.33,
        quality_min_score: int = 5,
        buffer_pips: float = 2.0,
        pip_size: float = 0.0001,
    ):
        self.min_tail_ratio = min_tail_ratio
        self.min_tail_pct_of_range = min_tail_pct_of_range
        self.max_body_pct_of_range = max_body_pct_of_range
        self.quality_min_score = quality_min_score
        self.buffer_pips = buffer_pips
        self.pip_size = pip_size
        self.buffer = buffer_pips * pip_size

    def detect(self, candle: CandleData) -> PinBarResult:
        """
        Detect if a candle is a pin bar pattern.

        Args:
            candle: Single CandleData to analyze

        Returns:
            PinBarResult with detection status and quality score.
        """
        total_range = candle.total_range
        body_size = candle.body_size

        # Skip doji (no meaningful range)
        if total_range < self.pip_size * 5:
            return PinBarResult(
                detected=False,
                reject_reason="Doji or near-doji (range too small)",
            )

        # Calculate wick components
        upper_wick = candle.upper_wick
        lower_wick = candle.lower_wick

        # Determine if this is a bullish or bearish pin bar
        # Bullish: lower wick is the tail (long lower wick)
        # Bearish: upper wick is the tail (long upper wick)

        if lower_wick > upper_wick:
            # Potential bullish pin bar (long lower tail)
            tail = lower_wick
            nose_wick = upper_wick
            pin_type = PinBarType.BULLISH
            # Body must be in upper 1/3 of total range
            body_high = max(candle.open, candle.close)
            body_position = (body_high - candle.low) / total_range
        else:
            # Potential bearish pin bar (long upper tail)
            tail = upper_wick
            nose_wick = lower_wick
            pin_type = PinBarType.BEARISH
            # Body must be in lower 1/3 of total range
            body_low = min(candle.open, candle.close)
            body_position = 1.0 - (candle.high - body_low) / total_range

        # Check minimum tail length as % of range
        tail_pct_of_range = tail / total_range
        if tail_pct_of_range < self.min_tail_pct_of_range:
            return PinBarResult(
                detected=False,
                reject_reason=(
                    f"Tail {tail_pct_of_range:.1%} < "
                    f"minimum {self.min_tail_pct_of_range:.1%} of range"
                ),
            )

        # Check tail/body ratio
        if body_size < self.pip_size * 2:
            tail_ratio = float("inf")  # Doji body — effectively infinite ratio
        else:
            tail_ratio = tail / body_size

        if tail_ratio < self.min_tail_ratio:
            return PinBarResult(
                detected=False,
                reject_reason=(
                    f"Tail ratio {tail_ratio:.1f} < minimum {self.min_tail_ratio}"
                ),
            )

        # Check body position — must be in correct 1/3 of range
        correct_zone = body_position >= (2 / 3)
        if not correct_zone:
            return PinBarResult(
                detected=False,
                reject_reason=(
                    f"Body position {body_position:.1%} not in correct zone "
                    f"(need >= 66% for {'bullish' if pin_type == PinBarType.BULLISH else 'bearish'})"
                ),
            )

        # Calculate quality score
        quality = self._calculate_quality(
            tail_pct_of_range=tail_pct_of_range,
            tail_ratio=tail_ratio,
            body_position=body_position,
            body_size=body_size,
            total_range=total_range,
        )

        if quality < self.quality_min_score:
            return PinBarResult(
                detected=False,
                reject_reason=f"Quality score {quality} < minimum {self.quality_min_score}",
            )

        # Calculate entry and stop levels
        if pin_type == PinBarType.BULLISH:
            entry = candle.high + self.buffer    # Buy above the pin bar high
            stop = candle.low - self.buffer      # Stop below the pin bar low (tail)
        else:
            entry = candle.low - self.buffer     # Sell below the pin bar low
            stop = candle.high + self.buffer     # Stop above the pin bar high (tail)

        return PinBarResult(
            detected=True,
            pin_bar_type=pin_type,
            quality_score=quality,
            tail_ratio=tail_ratio,
            tail_pct_of_range=tail_pct_of_range,
            body_position=body_position,
            suggested_entry=entry,
            suggested_stop=stop,
        )

    def _calculate_quality(
        self,
        tail_pct_of_range: float,
        tail_ratio: float,
        body_position: float,
        body_size: float,
        total_range: float,
    ) -> int:
        """
        Score pin bar quality on a 1-10 scale.

        Scoring criteria:
        - Tail % of range: 66-75% → 3pts, 75-85% → 4pts, >85% → 5pts
        - Tail/body ratio: 2-3x → 1pt, 3-4x → 2pts, >4x → 3pts
        - Body position: 66-75% in zone → 1pt, >75% → 2pts
        - Small body: body < 10% of range → +1pt bonus

        Maximum: 10 points (after capping at 10)
        """
        score = 0

        # Tail dominance (max 5 points)
        if tail_pct_of_range >= 0.85:
            score += 5
        elif tail_pct_of_range >= 0.75:
            score += 4
        elif tail_pct_of_range >= 0.66:
            score += 3

        # Tail ratio (max 3 points)
        if tail_ratio >= 4.0 or tail_ratio == float("inf"):
            score += 3
        elif tail_ratio >= 3.0:
            score += 2
        elif tail_ratio >= 2.0:
            score += 1

        # Body positioning (max 2 points)
        if body_position >= 0.80:
            score += 2
        elif body_position >= 0.66:
            score += 1

        # Small body bonus (max 1 point)
        if total_range > 0 and body_size / total_range <= 0.10:
            score += 1

        return min(score, 10)

    def is_inside_context(
        self,
        pin_candle: CandleData,
        context_candles: list,
    ) -> bool:
        """
        Check if pin bar is forming at a relevant structure point.
        A pin bar at a level is stronger than one in open space.

        Args:
            pin_candle: The pin bar candle
            context_candles: Recent candles for context

        Returns:
            True if pin bar is forming at an identifiable structure point.
        """
        # Simplified: check if pin bar's tail extends to prior swing area
        # Full implementation uses M05 S/R levels in Phase 1 Sprint 3
        return True  # Stub — context check deferred to M08 integration
