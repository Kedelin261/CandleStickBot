"""
M03 — Market Structure Module
Detects swing highs/lows and classifies market structure (trending/ranging).
The Candlestick Trading Bible: Structure is the foundation — trade WITH structure.

Key Concepts:
  - Higher Highs (HH) + Higher Lows (HL) = Uptrend (look for LONG)
  - Lower Highs (LH) + Lower Lows (LL) = Downtrend (look for SHORT)
  - Mixed = Ranging / Consolidation
  - Swing pivot detection uses configurable lookback window

Phase 1 scope: EURUSD D1 only.
Status: STUB — Phase 0 scaffold. Full implementation in Phase 1 Sprint 2.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

from src.types import CandleData, MarketStructure

logger = logging.getLogger("candlestickbot.analysis.market_structure")


class TrendDirection(str, Enum):
    """Market trend directions."""
    UP = "UP"
    DOWN = "DOWN"
    RANGING = "RANGING"
    UNDEFINED = "UNDEFINED"


class SwingType(str, Enum):
    """Swing point types."""
    HIGH = "HIGH"
    LOW = "LOW"


@dataclass
class SwingPoint:
    """
    A confirmed swing high or low.

    Confirmed when a pivot has 'lookback' candles on each side
    that are all lower (for HIGH) or higher (for LOW).
    """
    index: int           # Index in the candle series (0 = oldest)
    price: float         # Pivot price (high for SH, low for SL)
    swing_type: SwingType
    candle: CandleData   # The candle that formed this swing
    confirmed: bool = True
    strength: int = 1    # How many candles confirm on each side


@dataclass
class StructureAnalysis:
    """
    Full market structure analysis result.

    Used by M04 (trend detection) and M05 (S/R engine) as input.
    """
    direction: TrendDirection
    swing_highs: List[SwingPoint] = field(default_factory=list)
    swing_lows: List[SwingPoint] = field(default_factory=list)

    # Most recent labeled pivots
    last_hh: Optional[float] = None   # Last higher high price
    last_hl: Optional[float] = None   # Last higher low price
    last_lh: Optional[float] = None   # Last lower high price
    last_ll: Optional[float] = None   # Last lower low price

    # Structure break flags
    structure_broken_up: bool = False    # Broke above last resistance
    structure_broken_down: bool = False  # Broke below last support

    # Analysis metadata
    candles_analyzed: int = 0
    lookback_used: int = 5
    confidence: float = 0.0  # 0.0–1.0

    @property
    def is_trending(self) -> bool:
        return self.direction in (TrendDirection.UP, TrendDirection.DOWN)

    @property
    def is_ranging(self) -> bool:
        return self.direction == TrendDirection.RANGING

    def to_market_structure(self) -> MarketStructure:
        """Convert to shared MarketStructure DTO for cross-module use."""
        return MarketStructure(
            swing_highs=[sp.price for sp in self.swing_highs[-10:]],
            swing_lows=[sp.price for sp in self.swing_lows[-10:]],
            last_hh=self.last_hh,
            last_hl=self.last_hl,
            last_lh=self.last_lh,
            last_ll=self.last_ll,
            regime=self.direction.value,
        )


class MarketStructureAnalyzer:
    """
    M03 — Market Structure Analyzer.

    Algorithm:
    1. Scan candle series for swing pivots using N-bar lookback
    2. Classify each swing as HH, HL, LH, or LL relative to prior swing
    3. Determine trend direction from last 2-3 swing pairs
    4. Detect structure breaks (BOS — Break of Structure)

    Parameters (from config):
        lookback: N candles each side to confirm a swing (default: 5)
        min_swing_size_pips: Minimum distance for a valid swing (default: 20)
        trend_min_swings: Minimum swing pairs to confirm trend (default: 2)
    """

    def __init__(
        self,
        lookback: int = 5,
        min_swing_size_pips: float = 20.0,
        pip_size: float = 0.0001,
        trend_min_swings: int = 2,
    ):
        self.lookback = lookback
        self.min_swing_size_pips = min_swing_size_pips
        self.pip_size = pip_size
        self.min_swing_size = min_swing_size_pips * pip_size
        self.trend_min_swings = trend_min_swings

    def analyze(self, candles: List[CandleData]) -> StructureAnalysis:
        """
        Analyze market structure from a candle series.

        Args:
            candles: List of CandleData in ascending timestamp order (oldest first)
                     Minimum length: 2 * lookback + 1

        Returns:
            StructureAnalysis with trend direction and swing points.

        Note:
            Requires at least (2 * lookback + 1) candles for any swing detection.
            Returns UNDEFINED direction if insufficient data.
        """
        min_candles = 2 * self.lookback + 1
        if len(candles) < min_candles:
            logger.warning(
                f"Insufficient candles for structure analysis: "
                f"{len(candles)} < {min_candles} required"
            )
            return StructureAnalysis(
                direction=TrendDirection.UNDEFINED,
                candles_analyzed=len(candles),
                lookback_used=self.lookback,
            )

        # TODO: Full implementation in Phase 1 Sprint 2
        # swing_highs = self._detect_swing_highs(candles)
        # swing_lows = self._detect_swing_lows(candles)
        # classified_highs = self._classify_swing_highs(swing_highs)
        # classified_lows = self._classify_swing_lows(swing_lows)
        # direction = self._determine_direction(classified_highs, classified_lows)
        # ...
        logger.warning("MarketStructureAnalyzer.analyze() — STUB")
        return StructureAnalysis(
            direction=TrendDirection.UNDEFINED,
            candles_analyzed=len(candles),
            lookback_used=self.lookback,
        )

    def _detect_swing_highs(self, candles: List[CandleData]) -> List[SwingPoint]:
        """
        Find all swing highs: candles whose high is higher than all
        surrounding candles within the lookback window on both sides.

        A swing high at index i is confirmed when:
            candles[i].high > max(candles[i-lookback:i].high)
            AND candles[i].high > max(candles[i+1:i+lookback+1].high)
        """
        swing_highs = []
        n = len(candles)
        lb = self.lookback

        for i in range(lb, n - lb):
            candidate_high = candles[i].high
            left_highs = [c.high for c in candles[i - lb:i]]
            right_highs = [c.high for c in candles[i + 1:i + lb + 1]]

            if candidate_high > max(left_highs) and candidate_high > max(right_highs):
                # Check minimum swing size vs previous swing low
                sp = SwingPoint(
                    index=i,
                    price=candidate_high,
                    swing_type=SwingType.HIGH,
                    candle=candles[i],
                    confirmed=True,
                    strength=lb,
                )
                swing_highs.append(sp)

        return swing_highs

    def _detect_swing_lows(self, candles: List[CandleData]) -> List[SwingPoint]:
        """
        Find all swing lows: candles whose low is lower than all
        surrounding candles within the lookback window on both sides.
        """
        swing_lows = []
        n = len(candles)
        lb = self.lookback

        for i in range(lb, n - lb):
            candidate_low = candles[i].low
            left_lows = [c.low for c in candles[i - lb:i]]
            right_lows = [c.low for c in candles[i + 1:i + lb + 1]]

            if candidate_low < min(left_lows) and candidate_low < min(right_lows):
                sp = SwingPoint(
                    index=i,
                    price=candidate_low,
                    swing_type=SwingType.LOW,
                    candle=candles[i],
                    confirmed=True,
                    strength=lb,
                )
                swing_lows.append(sp)

        return swing_lows

    def _classify_swing_highs(self, swing_highs: List[SwingPoint]) -> List[Tuple[SwingPoint, str]]:
        """
        Label each swing high as HH (higher high) or LH (lower high)
        relative to the previous swing high.
        Returns list of (SwingPoint, label) tuples.
        """
        if len(swing_highs) < 2:
            return [(sh, "UNDEFINED") for sh in swing_highs]

        result = [(swing_highs[0], "UNDEFINED")]
        for i in range(1, len(swing_highs)):
            label = "HH" if swing_highs[i].price > swing_highs[i - 1].price else "LH"
            result.append((swing_highs[i], label))
        return result

    def _classify_swing_lows(self, swing_lows: List[SwingPoint]) -> List[Tuple[SwingPoint, str]]:
        """
        Label each swing low as HL (higher low) or LL (lower low)
        relative to the previous swing low.
        """
        if len(swing_lows) < 2:
            return [(sl, "UNDEFINED") for sl in swing_lows]

        result = [(swing_lows[0], "UNDEFINED")]
        for i in range(1, len(swing_lows)):
            label = "HL" if swing_lows[i].price > swing_lows[i - 1].price else "LL"
            result.append((swing_lows[i], label))
        return result

    def _determine_direction(
        self,
        classified_highs: List[Tuple[SwingPoint, str]],
        classified_lows: List[Tuple[SwingPoint, str]],
    ) -> TrendDirection:
        """
        Determine trend direction from classified swings.

        Uptrend: last N swing pairs show HH + HL pattern
        Downtrend: last N swing pairs show LH + LL pattern
        Ranging: mixed or insufficient data
        """
        min_swings = self.trend_min_swings

        recent_high_labels = [label for _, label in classified_highs[-min_swings:]]
        recent_low_labels = [label for _, label in classified_lows[-min_swings:]]

        hh_count = recent_high_labels.count("HH")
        hl_count = recent_low_labels.count("HL")
        lh_count = recent_high_labels.count("LH")
        ll_count = recent_low_labels.count("LL")

        # Uptrend requires HH AND HL dominance
        if hh_count >= min_swings and hl_count >= min_swings:
            return TrendDirection.UP

        # Downtrend requires LH AND LL dominance
        if lh_count >= min_swings and ll_count >= min_swings:
            return TrendDirection.DOWN

        # Mixed signals = ranging
        return TrendDirection.RANGING

    def detect_structure_break(
        self,
        candles: List[CandleData],
        analysis: StructureAnalysis,
    ) -> Tuple[bool, bool]:
        """
        Detect Break of Structure (BOS).

        A bullish BOS occurs when price closes above last swing high.
        A bearish BOS occurs when price closes below last swing low.

        Returns:
            Tuple of (broke_up, broke_down)
        """
        if not candles or not analysis.swing_highs or not analysis.swing_lows:
            return False, False

        last_close = candles[-1].close
        last_sh = analysis.swing_highs[-1].price if analysis.swing_highs else None
        last_sl = analysis.swing_lows[-1].price if analysis.swing_lows else None

        broke_up = last_sh is not None and last_close > last_sh
        broke_down = last_sl is not None and last_close < last_sl

        return broke_up, broke_down
