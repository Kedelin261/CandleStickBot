"""
M07 — Pattern Detection: False Breakout (Inside Bar False Breakout)
The Candlestick Trading Bible: False breakouts trap breakout traders.

**Phase 2 Feature — DISABLED in Phase 1**

False Breakout / Inside Bar False Breakout Setup:
  - Price breaks out of an inside bar (above mother bar high or below low)
  - Fails to sustain the breakout and closes back inside
  - Traps breakout traders → triggers opposite momentum
  - Strongest setups occur at key S/R levels

Phase 2 activation: same as Inside Bar (both features together).

Status: STUB — Phase 2 deferred. Class exists for import compatibility only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from src.types import CandleData, PatternSignal

logger = logging.getLogger("candlestickbot.patterns.false_breakout")


class FalseBreakoutType(str, Enum):
    """False breakout directional classification."""
    BULLISH_TRAP = "FALSE_BREAKOUT_BULLISH"  # Bearish reversal after failed upside break
    BEARISH_TRAP = "FALSE_BREAKOUT_BEARISH"  # Bullish reversal after failed downside break


@dataclass
class FalseBreakoutResult:
    """Result of false breakout detection."""
    detected: bool
    breakout_type: Optional[FalseBreakoutType] = None
    quality_score: int = 0
    breakout_extension_pips: float = 0.0  # How far price went before reversing
    suggested_entry: Optional[float] = None
    suggested_stop: Optional[float] = None
    reject_reason: Optional[str] = None

    def to_pattern_signal(self) -> Optional[PatternSignal]:
        if not self.detected or self.breakout_type is None:
            return None
        direction = "LONG" if self.breakout_type == FalseBreakoutType.BEARISH_TRAP else "SHORT"
        return PatternSignal(
            pattern_type=self.breakout_type.value,
            direction=direction,
            quality_score=self.quality_score,
            suggested_entry=self.suggested_entry,
            suggested_stop=self.suggested_stop,
            invalidation_price=self.suggested_stop,
        )


class FalseBreakoutDetector:
    """
    M07 False Breakout / Inside Bar False Breakout Detector — Phase 2 Feature.

    PHASE 1: This detector is disabled. The detect() method always returns
    FalseBreakoutResult(detected=False, reject_reason="Phase 2 feature — disabled")

    Phase 2: Full implementation including:
    - Detection of failed inside bar breakouts
    - Detection of failed S/R level breakouts (any candle pattern)
    - Trap candle identification (pin bar closing back into range)
    - Quality scoring based on breakout extension and reversal strength
    """

    def __init__(
        self,
        enabled: bool = False,  # Phase 1: MUST be False
        min_breakout_pips: float = 3.0,
        quality_min_score: int = 5,
        buffer_pips: float = 2.0,
        pip_size: float = 0.0001,
    ):
        self.enabled = enabled
        self.min_breakout_pips = min_breakout_pips
        self.quality_min_score = quality_min_score
        self.buffer_pips = buffer_pips
        self.pip_size = pip_size
        self.buffer = buffer_pips * pip_size

        if self.enabled:
            logger.warning(
                "FalseBreakoutDetector enabled — Phase 2 feature! "
                "Ensure phase==2 before enabling."
            )

    def detect(
        self,
        current: CandleData,
        previous: CandleData,
        mother_bar: Optional[CandleData] = None,
    ) -> FalseBreakoutResult:
        """
        Detect false breakout pattern.

        Phase 1: Returns NOT detected always.
        Phase 2: Full detection with inside bar context.

        Args:
            current: The candle that attempts to break out and fails
            previous: Prior candle (inside bar)
            mother_bar: The mother bar (if inside bar setup)
        """
        if not self.enabled:
            return FalseBreakoutResult(
                detected=False,
                reject_reason="Phase 2 feature — disabled in Phase 1",
            )

        # TODO: Phase 2 implementation
        return FalseBreakoutResult(
            detected=False,
            reject_reason="Phase 2 — not yet implemented",
        )
