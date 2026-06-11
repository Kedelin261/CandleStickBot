"""
M07 — Pattern Detection: Inside Bar
The Candlestick Trading Bible: Inside bars represent consolidation before continuation.

**Phase 2 Feature — DISABLED in Phase 1**

Inside Bar Definition:
  - Current candle's high AND low are within the previous candle's range
  - Represents a period of consolidation or indecision
  - Breakout from inside bar signals continuation or reversal

Phase 2 activation criteria:
  - After 50 completed trades AND 3 calendar months in Paper mode
  - User explicitly enables via config: strategies.inside_bar.enabled = true
  - Must pass baseline performance check (PF >= 1.1)

Status: STUB — Phase 2 deferred. Class exists for import compatibility only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from src.types import CandleData, PatternSignal

logger = logging.getLogger("candlestickbot.patterns.inside_bar")


class InsideBarType(str, Enum):
    """Inside bar directional classification."""
    BULLISH_BREAKOUT = "INSIDE_BAR_BULLISH"
    BEARISH_BREAKOUT = "INSIDE_BAR_BEARISH"
    UNRESOLVED = "INSIDE_BAR_UNRESOLVED"


@dataclass
class InsideBarResult:
    """Result of inside bar detection."""
    detected: bool
    inside_bar_type: Optional[InsideBarType] = None
    quality_score: int = 0
    containment_ratio: float = 0.0  # Inside bar range / mother bar range
    suggested_entry: Optional[float] = None
    suggested_stop: Optional[float] = None
    reject_reason: Optional[str] = None

    def to_pattern_signal(self) -> Optional[PatternSignal]:
        if not self.detected or self.inside_bar_type is None:
            return None
        direction = "LONG" if self.inside_bar_type == InsideBarType.BULLISH_BREAKOUT else "SHORT"
        return PatternSignal(
            pattern_type=self.inside_bar_type.value,
            direction=direction,
            quality_score=self.quality_score,
            suggested_entry=self.suggested_entry,
            suggested_stop=self.suggested_stop,
            invalidation_price=self.suggested_stop,
        )


class InsideBarDetector:
    """
    M07 Inside Bar Pattern Detector — Phase 2 Feature.

    PHASE 1: This detector is disabled. The detect() method always returns
    InsideBarResult(detected=False, reject_reason="Phase 2 feature — disabled")

    Phase 2: Full implementation including:
    - Multi-candle inside bar detection (IB within IB)
    - Breakout direction confirmation
    - Failed breakout (false breakout setup)
    - Quality scoring based on mother bar quality and position
    """

    def __init__(
        self,
        enabled: bool = False,  # Phase 1: MUST be False
        min_containment_ratio: float = 0.0,
        max_containment_ratio: float = 0.8,
        quality_min_score: int = 5,
        buffer_pips: float = 2.0,
        pip_size: float = 0.0001,
    ):
        self.enabled = enabled
        self.min_containment_ratio = min_containment_ratio
        self.max_containment_ratio = max_containment_ratio
        self.quality_min_score = quality_min_score
        self.buffer_pips = buffer_pips
        self.pip_size = pip_size
        self.buffer = buffer_pips * pip_size

        if self.enabled:
            logger.warning(
                "InsideBarDetector enabled — Phase 2 feature! "
                "Ensure phase==2 before enabling."
            )

    def detect(self, current: CandleData, previous: CandleData) -> InsideBarResult:
        """
        Detect if the current candle is an inside bar relative to previous.

        Phase 1: Returns NOT detected always (feature disabled).
        Phase 2: Full inside bar detection logic.
        """
        if not self.enabled:
            return InsideBarResult(
                detected=False,
                reject_reason="Phase 2 feature — disabled in Phase 1",
            )

        # TODO: Phase 2 implementation
        # Check containment: current.high < previous.high AND current.low > previous.low
        # if not (current.high < previous.high and current.low > previous.low):
        #     return InsideBarResult(detected=False, reject_reason="Not contained")
        # ...

        return InsideBarResult(
            detected=False,
            reject_reason="Phase 2 — not yet implemented",
        )
