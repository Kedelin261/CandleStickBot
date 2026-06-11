"""
M07 — Pattern Detection: Engulfing Bar
The Candlestick Trading Bible: Engulfing bars show decisive momentum shifts.

Engulfing Bar Definition:
  - Current candle's body completely engulfs the previous candle's body
  - Opposite directions (bullish engulfs a bearish, bearish engulfs a bullish)
  - Indicates institutional-level position change

Bullish Engulfing (at support):
  - Previous candle is bearish (red)
  - Current candle is bullish (green)
  - Current body open <= previous body close
  - Current body close >= previous body open
  - Signal: buyers overwhelmed sellers → expect LONG

Bearish Engulfing (at resistance):
  - Previous candle is bullish (green)
  - Current candle is bearish (red)
  - Current body open >= previous body close
  - Current body close <= previous body open
  - Signal: sellers overwhelmed buyers → expect SHORT

Strict Mode (optional):
  - Current high > previous high AND current low < previous low
  - Even stronger signal when wicks also engulf

Quality scoring (1-10):
  - Relative size of engulfing body vs engulfed body
  - Candle body as % of total range (small wicks = better)
  - Momentum: strong close vs weak close of prior candle

Phase 1: EURUSD D1 only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from src.types import CandleData, PatternSignal

logger = logging.getLogger("candlestickbot.patterns.engulfing")


class EngulfingType(str, Enum):
    """Engulfing bar directional classification."""
    BULLISH = "ENGULFING_BULLISH"
    BEARISH = "ENGULFING_BEARISH"


@dataclass
class EngulfingResult:
    """
    Result of engulfing bar detection for a candle pair.
    """
    detected: bool
    engulfing_type: Optional[EngulfingType] = None
    quality_score: int = 0             # 1-10
    engulfing_ratio: float = 0.0       # Current body / previous body (>1 = engulfs)
    full_engulf: bool = False          # True if wicks also engulfed (strict mode)
    suggested_entry: Optional[float] = None
    suggested_stop: Optional[float] = None
    reject_reason: Optional[str] = None

    def to_pattern_signal(self) -> Optional[PatternSignal]:
        """Convert to shared PatternSignal DTO."""
        if not self.detected or self.engulfing_type is None:
            return None
        return PatternSignal(
            pattern_type=self.engulfing_type.value,
            direction="LONG" if self.engulfing_type == EngulfingType.BULLISH else "SHORT",
            quality_score=self.quality_score,
            suggested_entry=self.suggested_entry,
            suggested_stop=self.suggested_stop,
            invalidation_price=self.suggested_stop,
        )


class EngulfingDetector:
    """
    M07 Engulfing Bar Pattern Detector.

    Parameters (from config/default_config.yaml):
        strict_mode: Require wicks to engulf as well (default: False)
        min_body_ratio: Minimum current/previous body ratio (default: 1.1)
        min_body_pct_of_range: Minimum body as % of range (default: 0.5)
        quality_min_score: Minimum quality score to signal (default: 5)
        buffer_pips: Entry buffer beyond engulfer open (default: 2.0 pips)
        pip_size: Pip size for symbol (default: 0.0001 for EURUSD)

    Detection Algorithm:
    1. Check that current and previous candles are opposite colors
    2. Check body engulfment: current body fully contains previous body
    3. Optionally check wick engulfment (strict mode)
    4. Calculate engulfing ratio (current body / previous body)
    5. Score quality based on relative sizes and body dominance
    """

    def __init__(
        self,
        strict_mode: bool = False,
        min_body_ratio: float = 1.1,
        min_body_pct_of_range: float = 0.5,
        quality_min_score: int = 5,
        buffer_pips: float = 2.0,
        pip_size: float = 0.0001,
    ):
        self.strict_mode = strict_mode
        self.min_body_ratio = min_body_ratio
        self.min_body_pct_of_range = min_body_pct_of_range
        self.quality_min_score = quality_min_score
        self.buffer_pips = buffer_pips
        self.pip_size = pip_size
        self.buffer = buffer_pips * pip_size

    def detect(self, current: CandleData, previous: CandleData) -> EngulfingResult:
        """
        Detect if the current+previous candle pair forms an engulfing pattern.

        Args:
            current: Most recent candle (the engulfer)
            previous: Prior candle (the engulfed)

        Returns:
            EngulfingResult with detection status and quality score.
        """
        # Both candles must have meaningful bodies
        if current.body_size < self.pip_size * 3:
            return EngulfingResult(
                detected=False,
                reject_reason="Current candle body too small (near doji)",
            )

        if previous.body_size < self.pip_size * 2:
            return EngulfingResult(
                detected=False,
                reject_reason="Previous candle body too small (near doji)",
            )

        # Determine direction — must be opposite colors
        # is_bullish: close > open
        curr_bullish = current.is_bullish
        prev_bullish = previous.is_bullish

        if curr_bullish == prev_bullish:
            return EngulfingResult(
                detected=False,
                reject_reason="Same direction candles — not an engulfing pattern",
            )

        # Check body engulfment
        curr_body_high = max(current.open, current.close)
        curr_body_low = min(current.open, current.close)
        prev_body_high = max(previous.open, previous.close)
        prev_body_low = min(previous.open, previous.close)

        body_engulfs = curr_body_high >= prev_body_high and curr_body_low <= prev_body_low

        if not body_engulfs:
            return EngulfingResult(
                detected=False,
                reject_reason="Current body does not engulf previous body",
            )

        # Strict mode: check wick engulfment too
        full_engulf = False
        if self.strict_mode:
            wick_engulfs = current.high >= previous.high and current.low <= previous.low
            if not wick_engulfs:
                return EngulfingResult(
                    detected=False,
                    reject_reason="Strict mode: wicks do not engulf previous candle",
                )
            full_engulf = True
        else:
            # Non-strict: check if wicks also engulf (for quality bonus)
            full_engulf = current.high >= previous.high and current.low <= previous.low

        # Calculate engulfing ratio
        if previous.body_size < self.pip_size:
            engulfing_ratio = float("inf")
        else:
            engulfing_ratio = current.body_size / previous.body_size

        if engulfing_ratio < self.min_body_ratio:
            return EngulfingResult(
                detected=False,
                reject_reason=(
                    f"Body ratio {engulfing_ratio:.2f} < minimum {self.min_body_ratio}"
                ),
            )

        # Check body size relative to current candle's range
        body_pct_of_range = (
            current.body_size / current.total_range if current.total_range > 0 else 0
        )
        if body_pct_of_range < self.min_body_pct_of_range:
            return EngulfingResult(
                detected=False,
                reject_reason=(
                    f"Engulfer body {body_pct_of_range:.1%} of range < "
                    f"minimum {self.min_body_pct_of_range:.1%}"
                ),
            )

        # Determine type
        eng_type = EngulfingType.BULLISH if curr_bullish else EngulfingType.BEARISH

        # Calculate quality score
        quality = self._calculate_quality(
            engulfing_ratio=engulfing_ratio,
            body_pct_of_range=body_pct_of_range,
            full_engulf=full_engulf,
        )

        if quality < self.quality_min_score:
            return EngulfingResult(
                detected=False,
                reject_reason=f"Quality score {quality} < minimum {self.quality_min_score}",
            )

        # Entry and stop levels
        if eng_type == EngulfingType.BULLISH:
            entry = current.high + self.buffer     # Buy above engulfer high
            stop = current.low - self.buffer       # Stop below engulfer low
        else:
            entry = current.low - self.buffer      # Sell below engulfer low
            stop = current.high + self.buffer      # Stop above engulfer high

        return EngulfingResult(
            detected=True,
            engulfing_type=eng_type,
            quality_score=quality,
            engulfing_ratio=engulfing_ratio,
            full_engulf=full_engulf,
            suggested_entry=entry,
            suggested_stop=stop,
        )

    def _calculate_quality(
        self,
        engulfing_ratio: float,
        body_pct_of_range: float,
        full_engulf: bool,
    ) -> int:
        """
        Score engulfing bar quality on a 1-10 scale.

        Scoring criteria:
        - Engulfing ratio: 1.1-2x → 2pts, 2-3x → 4pts, >3x → 5pts
        - Body dominance: 50-65% → 2pts, 65-80% → 3pts, >80% → 4pts
        - Full engulf (wicks too): +1pt bonus
        - Very large ratio (>4x): additional +1pt

        Maximum: 10 points
        """
        score = 0

        # Engulfing ratio (max 5 points)
        if engulfing_ratio >= 3.0 or engulfing_ratio == float("inf"):
            score += 5
        elif engulfing_ratio >= 2.0:
            score += 4
        elif engulfing_ratio >= 1.5:
            score += 3
        elif engulfing_ratio >= 1.1:
            score += 2

        # Body dominance relative to range (max 4 points)
        if body_pct_of_range >= 0.80:
            score += 4
        elif body_pct_of_range >= 0.65:
            score += 3
        elif body_pct_of_range >= 0.50:
            score += 2

        # Full engulf bonus (1 point)
        if full_engulf:
            score += 1

        return min(score, 10)
