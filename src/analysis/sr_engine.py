"""
M05 — Support & Resistance Engine
Identifies key S/R levels from swing points for trade entry confluence.
The Candlestick Trading Bible: Trade at key levels for high-probability setups.

Phase 1 methods:
  - Swing S/R: Derived from M03 swing highs/lows (primary)
  - 21 SMA Dynamic S/R: Moving average as support/resistance

Phase 2 (deferred):
  - Fibonacci retracements (M06 — disabled in Phase 1)
  - Supply & Demand zones (disabled in Phase 1)

Level strength scoring (0-10):
  - Touch count: Each test adds +2 points
  - Zone width: Tighter = stronger
  - Recency: Recent tests score higher
  - Role reversal: Old resistance becomes support (R→S) = strongest type

Status: STUB — Phase 0 scaffold. Full implementation in Phase 1 Sprint 2.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

from src.types import CandleData, LevelData

logger = logging.getLogger("candlestickbot.analysis.sr_engine")


class LevelType(str, Enum):
    """Types of S/R levels."""
    SUPPORT = "SUPPORT"
    RESISTANCE = "RESISTANCE"
    RESISTANCE_TURNED_SUPPORT = "RESISTANCE_TURNED_SUPPORT"
    SUPPORT_TURNED_RESISTANCE = "SUPPORT_TURNED_RESISTANCE"
    SMA21 = "SMA21"
    FIBONACCI = "FIBONACCI"  # Phase 2 only


class LevelStrength(str, Enum):
    """Qualitative strength classification."""
    WEAK = "WEAK"       # 1-3 touches, no role reversal
    MODERATE = "MODERATE"  # 3-4 touches or recent test
    STRONG = "STRONG"   # 5+ touches or role reversal confirmed


@dataclass
class SRLevel:
    """
    A detected Support/Resistance level.

    Used by M08 Strategy Engine for confluence scoring.
    """
    price: float
    level_type: LevelType
    strength_score: float      # 0-10 composite score
    touch_count: int           # Number of times price tested this level
    zone_high: float           # Upper boundary of zone
    zone_low: float            # Lower boundary of zone
    is_resistance_turned_support: bool = False
    last_tested_index: int = 0  # Index in candle series when last tested
    formed_index: int = 0       # Index when level was first identified

    @property
    def strength(self) -> LevelStrength:
        if self.strength_score >= 7:
            return LevelStrength.STRONG
        if self.strength_score >= 4:
            return LevelStrength.MODERATE
        return LevelStrength.WEAK

    @property
    def zone_midpoint(self) -> float:
        return (self.zone_high + self.zone_low) / 2.0

    @property
    def zone_width(self) -> float:
        return self.zone_high - self.zone_low

    def to_level_data(self) -> LevelData:
        """Convert to shared LevelData DTO."""
        return LevelData(
            price=self.price,
            level_type=self.level_type.value,
            strength_score=self.strength_score,
            touch_count=self.touch_count,
            zone_high=self.zone_high,
            zone_low=self.zone_low,
        )


@dataclass
class SRAnalysis:
    """
    Complete S/R analysis result from M05.

    Nearest support/resistance levels to current price are the most
    relevant for entry timing and TQS scoring.
    """
    levels: List[SRLevel] = field(default_factory=list)
    sma21_level: Optional[SRLevel] = None  # Dynamic S/R from 21 SMA
    nearest_support: Optional[SRLevel] = None
    nearest_resistance: Optional[SRLevel] = None
    current_price: float = 0.0
    candles_analyzed: int = 0

    @property
    def support_levels(self) -> List[SRLevel]:
        return [l for l in self.levels if l.level_type in (
            LevelType.SUPPORT, LevelType.RESISTANCE_TURNED_SUPPORT
        )]

    @property
    def resistance_levels(self) -> List[SRLevel]:
        return [l for l in self.levels if l.level_type in (
            LevelType.RESISTANCE, LevelType.SUPPORT_TURNED_RESISTANCE
        )]

    @property
    def strong_levels(self) -> List[SRLevel]:
        return [l for l in self.levels if l.strength == LevelStrength.STRONG]


class SREngine:
    """
    M05 — Support & Resistance Engine.

    Phase 1 algorithm:
    1. Extract swing highs from M03 → potential resistance levels
    2. Extract swing lows from M03 → potential support levels
    3. Merge nearby levels within zone_merge_pips into single zones
    4. Score each zone by touch count, recency, role reversals
    5. Calculate 21 SMA as dynamic support/resistance
    6. Return ranked levels for M08 confluence check

    TQS Scoring for level component (0-25 points):
      - Trading at STRONG level zone: 25
      - Trading at MODERATE level: 18
      - Trading at WEAK level: 10
      - No nearby level: 0
    """

    ZONE_WIDTH_PIPS = 10.0      # Default zone width around a level (±10 pips)
    MERGE_PIPS = 15.0            # Merge levels within 15 pips of each other
    MAX_LEVELS = 10              # Maximum levels to track per side
    NEARBY_THRESHOLD_PIPS = 30.0  # "Near" a level = within 30 pips

    def __init__(
        self,
        zone_width_pips: float = 10.0,
        merge_pips: float = 15.0,
        pip_size: float = 0.0001,
        max_levels: int = 10,
        nearby_threshold_pips: float = 30.0,
    ):
        self.zone_width_pips = zone_width_pips
        self.merge_pips = merge_pips
        self.pip_size = pip_size
        self.zone_width = zone_width_pips * pip_size
        self.merge_threshold = merge_pips * pip_size
        self.max_levels = max_levels
        self.nearby_threshold = nearby_threshold_pips * pip_size

    def analyze(
        self,
        candles: List[CandleData],
        swing_highs: Optional[List[float]] = None,
        swing_lows: Optional[List[float]] = None,
        sma21: Optional[float] = None,
    ) -> SRAnalysis:
        """
        Identify and score all S/R levels.

        Args:
            candles: Full candle series for touch-count analysis
            swing_highs: Prices of swing highs from M03 (optional)
            swing_lows: Prices of swing lows from M03 (optional)
            sma21: Current 21 SMA value (optional, for dynamic S/R)

        Returns:
            SRAnalysis with all detected levels and nearest to current price.
        """
        # TODO: Full implementation in Phase 1 Sprint 2
        logger.warning("SREngine.analyze() — STUB")

        current_price = candles[-1].close if candles else 0.0
        return SRAnalysis(
            levels=[],
            sma21_level=None,
            nearest_support=None,
            nearest_resistance=None,
            current_price=current_price,
            candles_analyzed=len(candles),
        )

    def _build_levels_from_swings(
        self,
        swing_prices: List[float],
        level_type: LevelType,
        candles: List[CandleData],
    ) -> List[SRLevel]:
        """
        Convert swing prices into SRLevel objects with zones and touch counts.

        Steps:
        1. Create zone (±zone_width/2) around each swing price
        2. Count how many candle closes/highs/lows touched each zone
        3. Merge nearby zones within merge_threshold
        4. Score each zone
        """
        if not swing_prices:
            return []

        levels = []
        for price in swing_prices:
            zone_high = price + self.zone_width / 2
            zone_low = price - self.zone_width / 2
            touches = self._count_touches(candles, zone_low, zone_high, level_type)
            score = self._calculate_strength_score(touches, price, candles)

            levels.append(SRLevel(
                price=price,
                level_type=level_type,
                strength_score=score,
                touch_count=touches,
                zone_high=zone_high,
                zone_low=zone_low,
            ))

        # Merge nearby levels
        levels = self._merge_nearby_levels(levels)

        # Sort by strength (strongest first), keep top N
        levels.sort(key=lambda l: l.strength_score, reverse=True)
        return levels[:self.max_levels]

    def _count_touches(
        self,
        candles: List[CandleData],
        zone_low: float,
        zone_high: float,
        level_type: LevelType,
    ) -> int:
        """
        Count how many candles touched this zone.

        For resistance: count candles whose high entered the zone.
        For support: count candles whose low entered the zone.
        """
        count = 0
        for c in candles:
            if level_type in (LevelType.RESISTANCE, LevelType.SUPPORT_TURNED_RESISTANCE):
                # Resistance: candle high enters the zone
                if c.high >= zone_low and c.high <= zone_high:
                    count += 1
                elif c.open >= zone_low and c.open <= zone_high:
                    count += 1
            else:
                # Support: candle low enters the zone
                if c.low <= zone_high and c.low >= zone_low:
                    count += 1
                elif c.close >= zone_low and c.close <= zone_high:
                    count += 1
        return count

    def _calculate_strength_score(
        self,
        touch_count: int,
        price: float,
        candles: List[CandleData],
    ) -> float:
        """
        Calculate 0-10 strength score for a level.

        Scoring:
          - Base: touch_count * 2.0 (max 6 for 3+ touches)
          - Recency bonus: +2 if tested in last 20 candles
          - Role reversal: +2 if level changed role (detected externally)

        Maximum: 10.0
        """
        score = min(touch_count * 2.0, 6.0)

        # Recency bonus: check if price was near level in last 20 candles
        recent = candles[-20:] if len(candles) >= 20 else candles
        zone_w = self.zone_width
        for c in recent:
            if abs(c.close - price) <= zone_w or abs(c.low - price) <= zone_w:
                score += 2.0
                break

        return min(score, 10.0)

    def _merge_nearby_levels(self, levels: List[SRLevel]) -> List[SRLevel]:
        """
        Merge levels within merge_threshold of each other.
        The merged level takes the average price and combined touch count.
        """
        if len(levels) <= 1:
            return levels

        levels_sorted = sorted(levels, key=lambda l: l.price)
        merged = [levels_sorted[0]]

        for level in levels_sorted[1:]:
            prev = merged[-1]
            if abs(level.price - prev.price) <= self.merge_threshold:
                # Merge: take average price, sum touches, max score
                new_price = (prev.price + level.price) / 2
                new_touches = prev.touch_count + level.touch_count
                new_score = max(prev.strength_score, level.strength_score)
                merged[-1] = SRLevel(
                    price=new_price,
                    level_type=prev.level_type,
                    strength_score=min(new_score + 1.0, 10.0),
                    touch_count=new_touches,
                    zone_high=max(prev.zone_high, level.zone_high),
                    zone_low=min(prev.zone_low, level.zone_low),
                    is_resistance_turned_support=prev.is_resistance_turned_support,
                )
            else:
                merged.append(level)

        return merged

    def _find_nearest(
        self,
        levels: List[SRLevel],
        current_price: float,
        above: bool,
    ) -> Optional[SRLevel]:
        """
        Find the nearest level above or below current price.

        Args:
            levels: All detected levels
            current_price: Current market price
            above: True = find nearest above, False = find nearest below
        """
        candidates = [
            l for l in levels
            if (above and l.price > current_price) or (not above and l.price < current_price)
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda l: abs(l.price - current_price))

    def calculate_tqs_level_score(
        self,
        candle: CandleData,
        nearest_support: Optional[SRLevel],
        nearest_resistance: Optional[SRLevel],
        direction: str,
    ) -> int:
        """
        Calculate TQS level component score (0-25 points).

        Scoring logic:
          - Pattern at STRONG level in trade direction: 25
          - Pattern at MODERATE level: 18
          - Pattern at WEAK level: 10
          - Pattern NOT at a key level: 5
          - Pattern at level against trade direction: 0

        Args:
            candle: The pattern candle (for proximity check)
            nearest_support: Nearest support level
            nearest_resistance: Nearest resistance level
            direction: Trade direction ("LONG" or "SHORT")
        """
        # Determine which level is relevant for the trade direction
        relevant_level = nearest_support if direction == "LONG" else nearest_resistance

        if relevant_level is None:
            return 5  # No level identified

        # Check if pattern candle is within the level zone
        candle_price = candle.low if direction == "LONG" else candle.high
        at_level = relevant_level.zone_low <= candle_price <= relevant_level.zone_high

        if not at_level:
            # Check if within nearby threshold
            distance = abs(candle_price - relevant_level.price)
            if distance > self.nearby_threshold:
                return 5  # Too far from any level

        # Score based on level strength
        strength = relevant_level.strength
        if relevant_level.is_resistance_turned_support:
            return 25  # Role reversal — strongest signal
        if strength == LevelStrength.STRONG:
            return 22
        if strength == LevelStrength.MODERATE:
            return 18
        return 12  # Weak level
