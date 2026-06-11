"""
M03 — Market Structure Engine
Detects swing highs/lows and classifies market structure (trending/ranging).
The Candlestick Trading Bible: Structure is the foundation — trade WITH structure.

Key Concepts:
  - Higher Highs (HH) + Higher Lows (HL) = Uptrend (look for LONG)
  - Lower Highs (LH) + Lower Lows (LL) = Downtrend (look for SHORT)
  - Mixed = Ranging / Consolidation
  - Swing pivot detection uses configurable lookback window

Phase 1 scope: EURUSD D1 only. Architecture supports any symbol/timeframe.
Status: COMPLETE — Phase 1 Sprint 2.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from src.types import CandleData, MarketStructure, SwingPointData, TrendDirection

logger = logging.getLogger("candlestickbot.analysis.market_structure")


# ---------------------------------------------------------------------------
# LOCAL ENUMS (module-internal — do not bleed into DTOs)
# ---------------------------------------------------------------------------

class SwingType:
    """String constants for swing point type (not enum — avoids conflicts with ORM Enum)."""
    HIGH = "HIGH"
    LOW = "LOW"


# ---------------------------------------------------------------------------
# MODULE-LEVEL DATA CLASSES
# ---------------------------------------------------------------------------

@dataclass
class SwingPoint:
    """
    A confirmed swing high or low detected in a candle series.

    Confirmed when candle[i] has 'lookback' candles on each side that are
    all lower-high (for HIGH) or higher-low (for LOW) respectively.

    Attributes:
        index:      Position in the original candle list (0 = oldest)
        price:      Pivot price — candle.high for SH, candle.low for SL
        swing_type: "HIGH" or "LOW"  (SwingType constants)
        candle:     The CandleData that formed this swing
        confirmed:  Always True for completed pivots
        strength:   Lookback size used to confirm this point
    """
    index: int
    price: float
    swing_type: str          # SwingType.HIGH | SwingType.LOW
    candle: CandleData
    confirmed: bool = True
    strength: int = 5

    @property
    def timestamp(self) -> datetime:
        return self.candle.timestamp

    @property
    def symbol(self) -> str:
        return self.candle.symbol

    @property
    def timeframe(self) -> str:
        return self.candle.timeframe

    def to_dto(self) -> SwingPointData:
        """Convert to shared SwingPointData DTO for cross-module use."""
        return SwingPointData(
            timestamp=self.candle.timestamp,
            price=self.price,
            swing_type=self.swing_type,
            symbol=self.candle.symbol,
            timeframe=self.candle.timeframe,
            lookback=self.strength,
        )


@dataclass
class StructureAnalysis:
    """
    Full market structure analysis result returned by MarketStructureAnalyzer.

    This is M03's internal result object. Downstream modules (M04, M05, M08)
    should call `.to_market_structure()` to get the shared DTO.

    Classification labels (stored on swing points):
        HH = Higher High   (swing high above previous swing high)
        LH = Lower High    (swing high below previous swing high)
        HL = Higher Low    (swing low above previous swing low)
        LL = Lower Low     (swing low below previous swing low)

    Trend Rules (direction field uses TrendDirection enum from src.types):
        UP      — last 2+ swing highs are HH AND last 2+ swing lows are HL
                  → direction = TrendDirection.UP
        DOWN    — last 2+ swing highs are LH AND last 2+ swing lows are LL
                  → direction = TrendDirection.DOWN
        RANGING — highs and lows are approximately equal within tolerance
                  → direction = TrendDirection.NONE, is_ranging = True
        CHOPPY/UNDEFINED — insufficient data or contradictory structure
                  → direction = TrendDirection.NONE, is_ranging = False

    Note: TrendDirection (src.types) has only UP / DOWN / NONE.
          Ranging is distinguished from pure NONE via the `is_ranging` flag.
    """
    direction: TrendDirection
    is_ranging: bool = False           # True when structure is sideways/consolidating
    swing_highs: List[SwingPoint] = field(default_factory=list)
    swing_lows: List[SwingPoint] = field(default_factory=list)

    # Most recent labeled pivots (populated after classification)
    last_hh: Optional[float] = None    # Last higher high price
    last_hl: Optional[float] = None    # Last higher low price
    last_lh: Optional[float] = None    # Last lower high price
    last_ll: Optional[float] = None    # Last lower low price

    # Structure break flags
    structure_broken_up: bool = False    # Last close > last swing high
    structure_broken_down: bool = False  # Last close < last swing low

    # Classification labels (parallel list to swing_highs / swing_lows)
    # Each entry is "HH", "LH", "HL", "LL", or "UNDEFINED"
    high_labels: List[str] = field(default_factory=list)
    low_labels: List[str] = field(default_factory=list)

    # Analysis metadata
    candles_analyzed: int = 0
    lookback_used: int = 5
    confidence: float = 0.0   # 0.0–1.0 quality proxy
    reason: str = ""           # Human-readable explanation

    @property
    def all_swing_points(self) -> List[SwingPoint]:
        """All swing points merged and sorted ascending by index."""
        merged = self.swing_highs + self.swing_lows
        return sorted(merged, key=lambda sp: sp.index)

    @property
    def latest_swing_high(self) -> Optional[SwingPoint]:
        return self.swing_highs[-1] if self.swing_highs else None

    @property
    def latest_swing_low(self) -> Optional[SwingPoint]:
        return self.swing_lows[-1] if self.swing_lows else None

    @property
    def is_trending(self) -> bool:
        return self.direction in (TrendDirection.UP, TrendDirection.DOWN)

    @property
    def structure_type(self) -> str:
        """Human-readable structure type: 'UP', 'DOWN', 'RANGING', or 'NONE'."""
        if self.direction == TrendDirection.UP:
            return "UP"
        if self.direction == TrendDirection.DOWN:
            return "DOWN"
        if self.is_ranging:
            return "RANGING"
        return "NONE"

    @property
    def is_defined(self) -> bool:
        return self.direction != TrendDirection.NONE or self.is_ranging

    @property
    def structure_label(self) -> str:
        """Alias for structure_type — 'UP', 'DOWN', 'RANGING', or 'NONE'."""
        return self.structure_type

    def to_market_structure(self) -> MarketStructure:
        """
        Convert to shared MarketStructure DTO for cross-module consumption.

        Uses the first available candle's symbol/timeframe. If no swings
        were found, symbol and timeframe are empty strings.
        """
        ref = (
            self.swing_highs[0].candle
            if self.swing_highs
            else (self.swing_lows[0].candle if self.swing_lows else None)
        )
        symbol = ref.symbol if ref else ""
        timeframe = ref.timeframe if ref else ""
        ts = ref.timestamp if ref else datetime.now(timezone.utc)

        return MarketStructure(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=ts,
            swing_highs=[sp.to_dto() for sp in self.swing_highs[-10:]],
            swing_lows=[sp.to_dto() for sp in self.swing_lows[-10:]],
            last_hh=self.last_hh,
            last_hl=self.last_hl,
            last_lh=self.last_lh,
            last_ll=self.last_ll,
            regime=self.direction,
        )


# ---------------------------------------------------------------------------
# MODULE-LEVEL COMPARISON HELPERS (stateless, importable individually)
# ---------------------------------------------------------------------------

def is_higher_high(current: SwingPoint, previous: SwingPoint) -> bool:
    """Return True if current swing high is above previous swing high."""
    return current.price > previous.price


def is_higher_low(current: SwingPoint, previous: SwingPoint) -> bool:
    """Return True if current swing low is above previous swing low."""
    return current.price > previous.price


def is_lower_high(current: SwingPoint, previous: SwingPoint) -> bool:
    """Return True if current swing high is below previous swing high."""
    return current.price < previous.price


def is_lower_low(current: SwingPoint, previous: SwingPoint) -> bool:
    """Return True if current swing low is below previous swing low."""
    return current.price < previous.price


# ---------------------------------------------------------------------------
# MODULE-LEVEL DETECTION FUNCTIONS
# ---------------------------------------------------------------------------

def identify_swing_highs(
    candles: List[CandleData],
    lookback: int = 5,
) -> List[SwingPoint]:
    """
    Identify all swing highs in a candle series.

    A candle at index i is a swing high when:
        candles[i].high > max(candles[i-lookback : i].high)   (left side)
        candles[i].high > max(candles[i+1 : i+lookback+1].high)  (right side)

    Args:
        candles:  Candle series in ascending timestamp order (oldest first).
        lookback: Number of candles required on each side (default 5).

    Returns:
        List of SwingPoint (HIGH type), sorted ascending by index.
        Empty list if fewer than (2 * lookback + 1) candles provided.
    """
    if lookback < 1:
        raise ValueError(f"lookback must be >= 1, got {lookback}")
    if not candles:
        return []

    swing_highs: List[SwingPoint] = []
    n = len(candles)

    for i in range(lookback, n - lookback):
        candidate = candles[i].high
        left_max = max(c.high for c in candles[i - lookback: i])
        right_max = max(c.high for c in candles[i + 1: i + lookback + 1])

        if candidate > left_max and candidate > right_max:
            swing_highs.append(SwingPoint(
                index=i,
                price=candidate,
                swing_type=SwingType.HIGH,
                candle=candles[i],
                confirmed=True,
                strength=lookback,
            ))

    return swing_highs


def identify_swing_lows(
    candles: List[CandleData],
    lookback: int = 5,
) -> List[SwingPoint]:
    """
    Identify all swing lows in a candle series.

    A candle at index i is a swing low when:
        candles[i].low < min(candles[i-lookback : i].low)   (left side)
        candles[i].low < min(candles[i+1 : i+lookback+1].low)  (right side)

    Args:
        candles:  Candle series in ascending timestamp order (oldest first).
        lookback: Number of candles required on each side (default 5).

    Returns:
        List of SwingPoint (LOW type), sorted ascending by index.
        Empty list if fewer than (2 * lookback + 1) candles provided.
    """
    if lookback < 1:
        raise ValueError(f"lookback must be >= 1, got {lookback}")
    if not candles:
        return []

    swing_lows: List[SwingPoint] = []
    n = len(candles)

    for i in range(lookback, n - lookback):
        candidate = candles[i].low
        left_min = min(c.low for c in candles[i - lookback: i])
        right_min = min(c.low for c in candles[i + 1: i + lookback + 1])

        if candidate < left_min and candidate < right_min:
            swing_lows.append(SwingPoint(
                index=i,
                price=candidate,
                swing_type=SwingType.LOW,
                candle=candles[i],
                confirmed=True,
                strength=lookback,
            ))

    return swing_lows


def identify_swing_points(
    candles: List[CandleData],
    lookback: int = 5,
) -> List[SwingPoint]:
    """
    Identify all swing highs and lows in a candle series.

    Convenience function that merges and sorts the results of
    identify_swing_highs() and identify_swing_lows().

    Returns:
        All swing points sorted ascending by index.
    """
    highs = identify_swing_highs(candles, lookback=lookback)
    lows = identify_swing_lows(candles, lookback=lookback)
    return sorted(highs + lows, key=lambda sp: sp.index)


def classify_structure(
    swing_points: List[SwingPoint],
    trend_min_swings: int = 2,
    ranging_tolerance: float = 0.0002,
) -> Tuple[TrendDirection, List[str], List[str]]:
    """
    Classify market structure direction from a list of swing points.

    Labels each swing high as "HH" or "LH" relative to the prior swing high,
    and each swing low as "HL" or "LL" relative to the prior swing low.
    First swing of each type is labeled "UNDEFINED".

    Trend classification rules:
        UP      — last `trend_min_swings` swing highs are all HH AND
                  last `trend_min_swings` swing lows are all HL
        DOWN    — last `trend_min_swings` swing highs are all LH AND
                  last `trend_min_swings` swing lows are all LL
        RANGING — highs approximately equal (within tolerance) AND
                  lows approximately equal (within tolerance)
        NONE    — contradictory, insufficient, or choppy structure

    Args:
        swing_points:      All swing points (mixed HIGH/LOW), sorted ascending.
        trend_min_swings:  Minimum consecutive swing pairs to confirm trend (default 2).
        ranging_tolerance: Max price diff ratio to call highs/lows "equal" (default 0.0002).

    Returns:
        Tuple of (TrendDirection, high_labels, low_labels)
        high_labels and low_labels are parallel to the swing_highs/swing_lows
        sub-lists extracted from swing_points.
    """
    swing_highs = [sp for sp in swing_points if sp.swing_type == SwingType.HIGH]
    swing_lows = [sp for sp in swing_points if sp.swing_type == SwingType.LOW]

    # Label each swing relative to the previous of the same type
    high_labels = _label_sequence(swing_highs, bullish_label="HH", bearish_label="LH")
    low_labels = _label_sequence(swing_lows, bullish_label="HL", bearish_label="LL")

    # Determine direction (_RANGING sentinel → TrendDirection.NONE + is_ranging=True)
    raw = _determine_direction(
        high_labels=high_labels,
        low_labels=low_labels,
        swing_highs=swing_highs,
        swing_lows=swing_lows,
        trend_min_swings=trend_min_swings,
        ranging_tolerance=ranging_tolerance,
    )
    is_ranging = raw is _RANGING
    direction = TrendDirection.NONE if is_ranging else raw

    return direction, high_labels, low_labels, is_ranging


def detect_structure_break(
    candles: List[CandleData],
    direction: str,
) -> Tuple[bool, bool]:
    """
    Detect a Break of Structure (BOS) in either direction.

    A bullish BOS: last close > last swing high price detected in candles.
    A bearish BOS: last close < last swing low price detected in candles.

    Note: This function runs its own swing detection with default lookback=5.
          For custom lookback, use MarketStructureAnalyzer.detect_structure_break().

    Args:
        candles:   Candle series, ascending order.
        direction: Ignored — checks both directions always. Kept for API symmetry.

    Returns:
        Tuple of (broke_up: bool, broke_down: bool).
        Both False if no candles or no swings found.
    """
    if not candles:
        return False, False

    swing_highs = identify_swing_highs(candles, lookback=5)
    swing_lows = identify_swing_lows(candles, lookback=5)
    last_close = candles[-1].close

    broke_up = bool(swing_highs) and last_close > swing_highs[-1].price
    broke_down = bool(swing_lows) and last_close < swing_lows[-1].price

    return broke_up, broke_down


def summarize_structure(
    candles: List[CandleData],
    swing_points: List[SwingPoint],
) -> Dict:
    """
    Return a human-readable summary dictionary of detected market structure.

    Suitable for logging, API output, and debugging. Does not perform any
    new detection — uses the already-identified swing_points.

    Returns dict with keys:
        candle_count, swing_high_count, swing_low_count,
        latest_swing_high_price, latest_swing_low_price,
        direction, confidence, reason
    """
    swing_highs = [sp for sp in swing_points if sp.swing_type == SwingType.HIGH]
    swing_lows = [sp for sp in swing_points if sp.swing_type == SwingType.LOW]

    direction, high_labels, low_labels, is_ranging = classify_structure(swing_points)

    latest_sh = swing_highs[-1].price if swing_highs else None
    latest_sl = swing_lows[-1].price if swing_lows else None

    last_hh = _last_labeled_price(swing_highs, high_labels, "HH")
    last_hl = _last_labeled_price(swing_lows, low_labels, "HL")
    last_lh = _last_labeled_price(swing_highs, high_labels, "LH")
    last_ll = _last_labeled_price(swing_lows, low_labels, "LL")

    confidence = _compute_confidence(
        swing_highs=swing_highs,
        swing_lows=swing_lows,
        high_labels=high_labels,
        low_labels=low_labels,
        direction=direction,
        is_ranging=is_ranging,
    )

    structure_label = "RANGING" if is_ranging else direction.value

    return {
        "candle_count": len(candles),
        "swing_high_count": len(swing_highs),
        "swing_low_count": len(swing_lows),
        "latest_swing_high_price": latest_sh,
        "latest_swing_low_price": latest_sl,
        "last_hh": last_hh,
        "last_hl": last_hl,
        "last_lh": last_lh,
        "last_ll": last_ll,
        "direction": structure_label,
        "confidence": confidence,
        "reason": _build_reason(direction, swing_highs, swing_lows, high_labels, low_labels,
                                is_ranging=is_ranging),
    }


# ---------------------------------------------------------------------------
# MAIN ANALYZER CLASS
# ---------------------------------------------------------------------------

class MarketStructureAnalyzer:
    """
    M03 — Market Structure Analyzer.

    Algorithm:
    1. Scan candle series for swing pivots using N-bar lookback
    2. Label each swing as HH, HL, LH, or LL relative to prior swing of same type
    3. Determine trend direction from last trend_min_swings swing pairs
    4. Detect Break of Structure (BOS)
    5. Optionally persist swing points to database

    Parameters:
        lookback:             N candles each side to confirm a swing (default 5)
        min_swing_size_pips:  Minimum swing size in pips to filter noise (default 20)
        pip_size:             Size of one pip (default 0.0001 for 5-decimal pairs)
        trend_min_swings:     Consecutive pairs required to confirm trend (default 2)
        ranging_tolerance:    Fractional tolerance for "equal" highs/lows (default 0.0002)
    """

    def __init__(
        self,
        lookback: int = 5,
        min_swing_size_pips: float = 0.0,
        pip_size: float = 0.0001,
        trend_min_swings: int = 2,
        ranging_tolerance: float = 0.0002,
    ):
        if lookback < 1:
            raise ValueError(f"lookback must be >= 1, got {lookback}")
        self.lookback = lookback
        self.min_swing_size_pips = min_swing_size_pips
        self.pip_size = pip_size
        self.min_swing_size = min_swing_size_pips * pip_size
        self.trend_min_swings = trend_min_swings
        self.ranging_tolerance = ranging_tolerance

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def analyze(self, candles: List[CandleData]) -> StructureAnalysis:
        """
        Analyze market structure from a candle series.

        Args:
            candles: List of CandleData in ascending timestamp order (oldest first).
                     Minimum required: 2 * lookback + 1 candles.

        Returns:
            StructureAnalysis with:
              - direction (TrendDirection)
              - swing_highs / swing_lows (sorted ascending)
              - high_labels / low_labels (HH/LH and HL/LL respectively)
              - last_hh, last_hl, last_lh, last_ll prices
              - structure_broken_up / structure_broken_down flags
              - confidence float (0.0–1.0)
              - reason string

        Behavior:
            - Returns NONE direction if fewer than (2*lookback+1) candles
            - Returns NONE if no swing points detected
            - Never raises on empty/short input — always returns safe StructureAnalysis
        """
        min_candles = 2 * self.lookback + 1

        if len(candles) < min_candles:
            reason = (
                f"Insufficient candles: {len(candles)} < {min_candles} required "
                f"(lookback={self.lookback})"
            )
            logger.warning("M03 analyze: %s", reason)
            return StructureAnalysis(
                direction=TrendDirection.NONE,
                candles_analyzed=len(candles),
                lookback_used=self.lookback,
                confidence=0.0,
                reason=reason,
            )

        # Step 1: Detect swing pivots
        swing_highs = self._detect_swing_highs(candles)
        swing_lows = self._detect_swing_lows(candles)

        if not swing_highs and not swing_lows:
            reason = "No swing points detected in candle series"
            logger.warning("M03 analyze: %s", reason)
            return StructureAnalysis(
                direction=TrendDirection.NONE,
                candles_analyzed=len(candles),
                lookback_used=self.lookback,
                confidence=0.0,
                reason=reason,
            )

        # Step 2: Label each swing (HH/LH for highs, HL/LL for lows)
        high_labels = _label_sequence(swing_highs, bullish_label="HH", bearish_label="LH")
        low_labels = _label_sequence(swing_lows, bullish_label="HL", bearish_label="LL")

        # Step 3: Classify direction
        # _determine_direction may return the internal _RANGING sentinel for sideways markets.
        # Since TrendDirection has no RANGING value, we map it to NONE + is_ranging=True.
        raw_direction = _determine_direction(
            high_labels=high_labels,
            low_labels=low_labels,
            swing_highs=swing_highs,
            swing_lows=swing_lows,
            trend_min_swings=self.trend_min_swings,
            ranging_tolerance=self.ranging_tolerance,
        )
        is_ranging_flag = raw_direction is _RANGING
        direction = TrendDirection.NONE if is_ranging_flag else raw_direction

        # Step 4: Extract key price levels
        last_hh = _last_labeled_price(swing_highs, high_labels, "HH")
        last_hl = _last_labeled_price(swing_lows, low_labels, "HL")
        last_lh = _last_labeled_price(swing_highs, high_labels, "LH")
        last_ll = _last_labeled_price(swing_lows, low_labels, "LL")

        # Step 5: Detect structure break
        broke_up, broke_down = self.detect_structure_break(candles, StructureAnalysis(
            direction=direction,
            swing_highs=swing_highs,
            swing_lows=swing_lows,
        ))

        # Step 6: Compute confidence
        confidence = _compute_confidence(
            swing_highs=swing_highs,
            swing_lows=swing_lows,
            high_labels=high_labels,
            low_labels=low_labels,
            direction=direction,
            is_ranging=is_ranging_flag,
        )

        reason = _build_reason(direction, swing_highs, swing_lows, high_labels, low_labels,
                               is_ranging=is_ranging_flag)

        logger.debug(
            "M03 analyze: structure_type=%s direction=%s confidence=%.2f highs=%d lows=%d",
            "RANGING" if is_ranging_flag else direction.value,
            direction.value, confidence, len(swing_highs), len(swing_lows),
        )

        return StructureAnalysis(
            direction=direction,
            is_ranging=is_ranging_flag,
            swing_highs=swing_highs,
            swing_lows=swing_lows,
            last_hh=last_hh,
            last_hl=last_hl,
            last_lh=last_lh,
            last_ll=last_ll,
            structure_broken_up=broke_up,
            structure_broken_down=broke_down,
            high_labels=high_labels,
            low_labels=low_labels,
            candles_analyzed=len(candles),
            lookback_used=self.lookback,
            confidence=confidence,
            reason=reason,
        )

    def detect_structure_break(
        self,
        candles: List[CandleData],
        analysis: StructureAnalysis,
    ) -> Tuple[bool, bool]:
        """
        Detect Break of Structure (BOS) against the detected swing levels.

        A bullish BOS: last candle's close > last confirmed swing high price.
        A bearish BOS: last candle's close < last confirmed swing low price.

        Args:
            candles:  Full candle series.
            analysis: Existing StructureAnalysis (uses its swing_highs/swing_lows).

        Returns:
            Tuple of (broke_up: bool, broke_down: bool).
        """
        if not candles or (not analysis.swing_highs and not analysis.swing_lows):
            return False, False

        last_close = candles[-1].close
        last_sh = analysis.swing_highs[-1].price if analysis.swing_highs else None
        last_sl = analysis.swing_lows[-1].price if analysis.swing_lows else None

        broke_up = last_sh is not None and last_close > last_sh
        broke_down = last_sl is not None and last_close < last_sl

        return broke_up, broke_down

    # ------------------------------------------------------------------
    # PERSISTENCE
    # ------------------------------------------------------------------

    def persist_swing_points(
        self,
        swing_points: List[SwingPoint],
        session,
    ) -> int:
        """
        Persist detected swing points to the database.

        Uses the SwingPoint ORM model from src.db.models.
        Deduplicates by (symbol, timeframe, timestamp, swing_type) before insert.
        Already-existing rows are skipped (no-duplicate guarantee).

        Args:
            swing_points: List of SwingPoint dataclass instances.
            session:      SQLAlchemy Session (caller is responsible for commit).

        Returns:
            Number of new rows inserted.
        """
        from src.db.models import SwingPoint as SwingPointORM

        if not swing_points:
            return 0

        inserted = 0
        for sp in swing_points:
            # Check for existing record (idempotent)
            existing = session.query(SwingPointORM).filter_by(
                symbol=sp.candle.symbol,
                timeframe=sp.candle.timeframe,
                timestamp=sp.candle.timestamp,
                swing_type=sp.swing_type,
            ).first()

            if existing is None:
                orm_obj = SwingPointORM(
                    symbol=sp.candle.symbol,
                    timeframe=sp.candle.timeframe,
                    timestamp=sp.candle.timestamp,
                    price=sp.price,
                    swing_type=sp.swing_type,
                    lookback=sp.strength,
                )
                session.add(orm_obj)
                inserted += 1

        logger.debug(
            "persist_swing_points: inserted %d / %d swing points",
            inserted, len(swing_points),
        )
        return inserted

    def get_swing_points(
        self,
        symbol: str,
        timeframe: str,
        session,
        swing_type: Optional[str] = None,
    ) -> List[SwingPoint]:
        """
        Retrieve persisted swing points from the database.

        Args:
            symbol:     Instrument symbol (e.g. "EURUSD").
            timeframe:  Timeframe string (e.g. "D1").
            session:    SQLAlchemy Session.
            swing_type: Optional filter — SwingType.HIGH or SwingType.LOW.

        Returns:
            List of SwingPoint dataclass instances, sorted ascending by timestamp.
            Empty list if no records found.
        """
        from src.db.models import SwingPoint as SwingPointORM
        from src.db.session import get_session

        query = session.query(SwingPointORM).filter_by(
            symbol=symbol.upper(),
            timeframe=timeframe.upper(),
        )
        if swing_type is not None:
            query = query.filter_by(swing_type=swing_type)

        rows = query.order_by(SwingPointORM.timestamp).all()

        # Convert ORM rows back to SwingPoint dataclass (candle field reconstructed minimally)
        results = []
        for row in rows:
            # We need a minimal CandleData for the candle field
            # (price fields are not stored in SwingPoint, so we use placeholders)
            dummy_candle = CandleData(
                symbol=row.symbol,
                timeframe=row.timeframe,
                timestamp=row.timestamp,
                open=row.price,
                high=row.price if row.swing_type == SwingType.HIGH else row.price,
                low=row.price if row.swing_type == SwingType.LOW else row.price,
                close=row.price,
                volume=0.0,
            )
            results.append(SwingPoint(
                index=-1,   # Index in original series not available from DB
                price=row.price,
                swing_type=row.swing_type,
                candle=dummy_candle,
                confirmed=True,
                strength=row.lookback,
            ))
        return results

    # ------------------------------------------------------------------
    # INTERNAL HELPERS (private)
    # ------------------------------------------------------------------

    def _detect_swing_highs(self, candles: List[CandleData]) -> List[SwingPoint]:
        """Internal — delegates to module-level identify_swing_highs with min_swing_size filter."""
        highs = identify_swing_highs(candles, lookback=self.lookback)
        if self.min_swing_size > 0:
            highs = self._filter_by_min_size(highs, candles)
        return highs

    def _detect_swing_lows(self, candles: List[CandleData]) -> List[SwingPoint]:
        """Internal — delegates to module-level identify_swing_lows with min_swing_size filter."""
        lows = identify_swing_lows(candles, lookback=self.lookback)
        if self.min_swing_size > 0:
            lows = self._filter_by_min_size(lows, candles)
        return lows

    def _filter_by_min_size(
        self,
        swing_points: List[SwingPoint],
        candles: List[CandleData],
    ) -> List[SwingPoint]:
        """
        Filter out swing points that are too close to the previous same-type swing.
        Prevents noise spikes from being flagged as meaningful structure.
        """
        if not swing_points or self.min_swing_size <= 0:
            return swing_points

        filtered = [swing_points[0]]
        for sp in swing_points[1:]:
            prev = filtered[-1]
            if abs(sp.price - prev.price) >= self.min_swing_size:
                filtered.append(sp)
        return filtered

    def _classify_swing_highs(
        self,
        swing_highs: List[SwingPoint],
    ) -> List[Tuple[SwingPoint, str]]:
        """Label swing highs as HH or LH. First entry is UNDEFINED."""
        return _label_with_points(swing_highs, bullish_label="HH", bearish_label="LH")

    def _classify_swing_lows(
        self,
        swing_lows: List[SwingPoint],
    ) -> List[Tuple[SwingPoint, str]]:
        """Label swing lows as HL or LL. First entry is UNDEFINED."""
        return _label_with_points(swing_lows, bullish_label="HL", bearish_label="LL")

    def _determine_direction(
        self,
        classified_highs: List[Tuple[SwingPoint, str]],
        classified_lows: List[Tuple[SwingPoint, str]],
    ) -> TrendDirection:
        """Delegate to module-level _determine_direction."""
        high_labels = [lbl for _, lbl in classified_highs]
        low_labels = [lbl for _, lbl in classified_lows]
        swing_highs = [sp for sp, _ in classified_highs]
        swing_lows = [sp for sp, _ in classified_lows]
        return _determine_direction(
            high_labels=high_labels,
            low_labels=low_labels,
            swing_highs=swing_highs,
            swing_lows=swing_lows,
            trend_min_swings=self.trend_min_swings,
            ranging_tolerance=self.ranging_tolerance,
        )


# ---------------------------------------------------------------------------
# PRIVATE MODULE-LEVEL HELPERS
# ---------------------------------------------------------------------------

def _label_sequence(
    swing_points: List[SwingPoint],
    bullish_label: str,
    bearish_label: str,
) -> List[str]:
    """
    Label each swing point relative to the previous one.

    For highs: bullish_label="HH", bearish_label="LH"
    For lows:  bullish_label="HL", bearish_label="LL"

    First point always gets "UNDEFINED".
    Equal prices get the bearish label (conservative — not a new high/low).
    """
    if not swing_points:
        return []
    labels = ["UNDEFINED"]
    for i in range(1, len(swing_points)):
        if swing_points[i].price > swing_points[i - 1].price:
            labels.append(bullish_label)
        else:
            labels.append(bearish_label)
    return labels


def _label_with_points(
    swing_points: List[SwingPoint],
    bullish_label: str,
    bearish_label: str,
) -> List[Tuple[SwingPoint, str]]:
    """Return list of (SwingPoint, label) tuples — used by class-level classify helpers."""
    labels = _label_sequence(swing_points, bullish_label, bearish_label)
    return list(zip(swing_points, labels))


# Internal sentinel returned by _determine_direction to signal ranging market.
# Callers must convert to (TrendDirection.NONE, is_ranging=True).
_RANGING = "_RANGING"


def _determine_direction(
    high_labels: List[str],
    low_labels: List[str],
    swing_highs: List[SwingPoint],
    swing_lows: List[SwingPoint],
    trend_min_swings: int = 2,
    ranging_tolerance: float = 0.0002,
):
    """
    Classify market direction from labelled swing sequences.

    Priority order:
        1. UP       — last trend_min_swings high_labels are all "HH" AND
                      last trend_min_swings low_labels are all "HL"
                      → returns TrendDirection.UP
        2. DOWN     — last trend_min_swings high_labels are all "LH" AND
                      last trend_min_swings low_labels are all "LL"
                      → returns TrendDirection.DOWN
        3. RANGING  — swing highs approximately equal AND swing lows approximately equal
                      → returns internal _RANGING sentinel
                      Callers convert to (TrendDirection.NONE, is_ranging=True)
        4. NONE     — everything else (choppy, contradictory, insufficient)
                      → returns TrendDirection.NONE
    """
    # Need at least trend_min_swings+1 points to have trend_min_swings labelled pairs
    # (first label is UNDEFINED, so we need at least trend_min_swings non-UNDEFINED labels)
    if (len(high_labels) < trend_min_swings + 1
            or len(low_labels) < trend_min_swings + 1):
        # Try ranging even with few points if we have at least 2 of each
        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            if (
                _is_approximately_equal(swing_highs, ranging_tolerance)
                and _is_approximately_equal(swing_lows, ranging_tolerance)
            ):
                return _RANGING
        return TrendDirection.NONE

    # Actionable (non-UNDEFINED) labels
    recent_high_labels = [lbl for lbl in high_labels if lbl != "UNDEFINED"][
        -trend_min_swings:
    ]
    recent_low_labels = [lbl for lbl in low_labels if lbl != "UNDEFINED"][
        -trend_min_swings:
    ]

    if len(recent_high_labels) < trend_min_swings or len(recent_low_labels) < trend_min_swings:
        return TrendDirection.NONE

    all_hh = all(lbl == "HH" for lbl in recent_high_labels)
    all_hl = all(lbl == "HL" for lbl in recent_low_labels)
    all_lh = all(lbl == "LH" for lbl in recent_high_labels)
    all_ll = all(lbl == "LL" for lbl in recent_low_labels)

    if all_hh and all_hl:
        return TrendDirection.UP

    if all_lh and all_ll:
        return TrendDirection.DOWN

    # Check for ranging: all highs within tolerance of each other,
    # and all lows within tolerance of each other
    if (
        len(swing_highs) >= 2
        and len(swing_lows) >= 2
        and _is_approximately_equal(swing_highs, ranging_tolerance)
        and _is_approximately_equal(swing_lows, ranging_tolerance)
    ):
        return _RANGING

    return TrendDirection.NONE


def _is_approximately_equal(
    swing_points: List[SwingPoint],
    tolerance: float,
) -> bool:
    """
    Return True if all swing point prices are within `tolerance` fraction of the mean.

    Used for RANGING classification: highs that are all similar, lows that are all similar.
    Tolerance is fractional (e.g. 0.0002 = 0.02% spread around mean price).
    """
    if len(swing_points) < 2:
        return False
    prices = [sp.price for sp in swing_points]
    mean_price = sum(prices) / len(prices)
    if mean_price == 0:
        return False
    return all(abs(p - mean_price) / mean_price <= tolerance for p in prices)


def _last_labeled_price(
    swing_points: List[SwingPoint],
    labels: List[str],
    target_label: str,
) -> Optional[float]:
    """Return the price of the most recent swing point with the given label."""
    for sp, lbl in zip(reversed(swing_points), reversed(labels)):
        if lbl == target_label:
            return sp.price
    return None


def _compute_confidence(
    swing_highs: List[SwingPoint],
    swing_lows: List[SwingPoint],
    high_labels: List[str],
    low_labels: List[str],
    direction: TrendDirection,
    is_ranging: bool = False,
) -> float:
    """
    Compute a 0.0–1.0 confidence score for the structure classification.

    Higher score when:
    - More swing points are detected
    - Label consistency is high (all HH, or all HL, etc.)
    - Direction is clearly UP or DOWN (not NONE)
    - Ranging markets get moderate confidence (0.3–0.6)
    """
    if direction == TrendDirection.NONE and not is_ranging:
        return 0.0

    # Swing point count component (caps at 1.0 above 6 pairs)
    pair_count = min(len(swing_highs), len(swing_lows))
    count_score = min(pair_count / 6.0, 1.0)

    # Label consistency component
    if direction == TrendDirection.UP:
        hh_pct = _label_fraction(high_labels, "HH")
        hl_pct = _label_fraction(low_labels, "HL")
        consistency = (hh_pct + hl_pct) / 2.0
    elif direction == TrendDirection.DOWN:
        lh_pct = _label_fraction(high_labels, "LH")
        ll_pct = _label_fraction(low_labels, "LL")
        consistency = (lh_pct + ll_pct) / 2.0
    elif is_ranging:
        consistency = 0.5   # Moderate confidence for ranging
    else:
        return 0.0

    return round((count_score * 0.4 + consistency * 0.6), 3)


def _label_fraction(labels: List[str], target: str) -> float:
    """Fraction of non-UNDEFINED labels that match target."""
    actionable = [lbl for lbl in labels if lbl != "UNDEFINED"]
    if not actionable:
        return 0.0
    return sum(1 for lbl in actionable if lbl == target) / len(actionable)


def _build_reason(
    direction: TrendDirection,
    swing_highs: List[SwingPoint],
    swing_lows: List[SwingPoint],
    high_labels: List[str],
    low_labels: List[str],
    is_ranging: bool = False,
) -> str:
    """Build a short human-readable explanation of the classification."""
    n_highs = len(swing_highs)
    n_lows = len(swing_lows)

    if is_ranging:
        return (
            f"Ranging: highs and lows approximately equal "
            f"({n_highs} highs, {n_lows} lows)"
        )

    if direction == TrendDirection.NONE:
        if n_highs == 0 and n_lows == 0:
            return "No swing points detected"
        if n_highs < 2 or n_lows < 2:
            return (
                f"Insufficient swing points: {n_highs} highs, {n_lows} lows "
                "(need ≥2 of each)"
            )
        return "Contradictory structure (mixed HH/LH or HL/LL)"

    if direction == TrendDirection.UP:
        hh_count = sum(1 for lbl in high_labels if lbl == "HH")
        hl_count = sum(1 for lbl in low_labels if lbl == "HL")
        return (
            f"Uptrend: {hh_count} higher high(s), {hl_count} higher low(s) "
            f"from {n_highs} swing highs and {n_lows} swing lows"
        )

    if direction == TrendDirection.DOWN:
        lh_count = sum(1 for lbl in high_labels if lbl == "LH")
        ll_count = sum(1 for lbl in low_labels if lbl == "LL")
        return (
            f"Downtrend: {lh_count} lower high(s), {ll_count} lower low(s) "
            f"from {n_highs} swing highs and {n_lows} swing lows"
        )

    return "Unknown direction"
