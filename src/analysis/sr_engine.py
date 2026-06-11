"""
M05 — Support & Resistance Engine
Identifies key S/R levels from swing points for trade entry confluence.
The Candlestick Trading Bible: Trade at key levels for high-probability setups.

Phase 1 algorithm:
  1. Accept swing highs / swing lows from M03 (as SwingPoint objects or plain prices)
  2. Build a zone (± zone_width/2) around each swing price
  3. Count touches: how many candle bodies/wicks entered each zone
  4. Detect role reversals: former resistance below current price → support
  5. Merge nearby zones within merge_threshold into a single consolidated level
  6. Score each level (0–10): touch count + recency bonus + role-reversal bonus
  7. Add 21 SMA as a dynamic S/R level (from M04, optional)
  8. Resolve nearest_support / nearest_resistance relative to current price
  9. Optionally persist to DB via SRLevel ORM, idempotent upsert on (symbol, tf, price±tol)
  10. Return ranked SRAnalysis

Level scoring (0–10):
  - touch_count * 1.5 capped at 6.0 (base)
  - +2.0 recency bonus if level was tested in last `recency_window` candles
  - +2.0 role-reversal bonus (resistance→support or vice versa)
  Maximum: 10.0

TQS level component (0–25):
  - Role-reversal STRONG:  25
  - STRONG (score >= 7):   22
  - MODERATE (score >= 4): 18
  - WEAK (score < 4):      12
  - No nearby level:        5

Phase 2 (deferred):
  - Fibonacci retracements (M06 — remains disabled)
  - Supply & Demand zones

Status: Full implementation — Phase 1 Sprint 4.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

from src.types import CandleData, LevelData, LevelType as DtoLevelType

logger = logging.getLogger("candlestickbot.analysis.sr_engine")


# ---------------------------------------------------------------------------
# Local enumerations (kept from Phase 0 stub — NOT the same as DtoLevelType)
# ---------------------------------------------------------------------------

class LevelType(str, Enum):
    """Internal S/R level type for M05 analysis."""
    SUPPORT                   = "SUPPORT"
    RESISTANCE                = "RESISTANCE"
    RESISTANCE_TURNED_SUPPORT = "RESISTANCE_TURNED_SUPPORT"
    SUPPORT_TURNED_RESISTANCE = "SUPPORT_TURNED_RESISTANCE"
    SMA21                     = "SMA21"
    # FIBONACCI intentionally omitted — Phase 2 only


class LevelStrength(str, Enum):
    WEAK     = "WEAK"       # score < 4
    MODERATE = "MODERATE"   # score 4–6.9
    STRONG   = "STRONG"     # score >= 7


# ---------------------------------------------------------------------------
# Scoring constants
# ---------------------------------------------------------------------------
_TOUCH_PER_POINT   = 1.5    # Points per touch (capped at 4 touches = 6.0)
_TOUCH_CAP         = 6.0    # Maximum points from touches alone
_RECENCY_BONUS     = 2.0    # Bonus when level was tested recently
_RTS_BONUS         = 2.0    # Bonus for role-reversal levels
_SCORE_MAX         = 10.0


# ---------------------------------------------------------------------------
# SRLevel — M05's internal level object
# ---------------------------------------------------------------------------

@dataclass
class SRLevel:
    """
    A detected Support/Resistance level with zone boundaries and scoring.

    Used internally by M05 and consumed by M08 Strategy Engine for
    confluence scoring.  Call `.to_level_data()` for the shared DTO.
    """
    price:          float
    level_type:     LevelType
    strength_score: float          # 0–10
    touch_count:    int
    zone_high:      float
    zone_low:       float
    is_resistance_turned_support: bool = False
    last_tested_index: int = 0     # candle index of most recent touch
    formed_index:      int = 0     # candle index when level was first formed
    symbol:    str = ""
    timeframe: str = ""

    @property
    def strength(self) -> LevelStrength:
        if self.strength_score >= 7.0:
            return LevelStrength.STRONG
        if self.strength_score >= 4.0:
            return LevelStrength.MODERATE
        return LevelStrength.WEAK

    @property
    def zone_midpoint(self) -> float:
        return (self.zone_high + self.zone_low) / 2.0

    @property
    def zone_width(self) -> float:
        return self.zone_high - self.zone_low

    def contains_price(self, price: float) -> bool:
        """Return True when price falls within [zone_low, zone_high]."""
        return self.zone_low <= price <= self.zone_high

    def distance_to(self, price: float, pip_size: float = 0.0001) -> float:
        """Distance from price to level midpoint in pips."""
        return abs(price - self.price) / pip_size

    def to_level_data(self) -> LevelData:
        """
        Convert to shared ``LevelData`` DTO (src.types).

        Mapping rules:
          SUPPORT / RESISTANCE_TURNED_SUPPORT → LevelType.SWING_SR
          RESISTANCE / SUPPORT_TURNED_RESISTANCE → LevelType.SWING_SR
          SMA21 → LevelType.SMA_21
        """
        dto_type_map = {
            LevelType.SUPPORT:                   DtoLevelType.SWING_SR,
            LevelType.RESISTANCE:                DtoLevelType.SWING_SR,
            LevelType.RESISTANCE_TURNED_SUPPORT: DtoLevelType.SWING_SR,
            LevelType.SUPPORT_TURNED_RESISTANCE: DtoLevelType.SWING_SR,
            LevelType.SMA21:                     DtoLevelType.SMA_21,
        }
        dto_type = dto_type_map.get(self.level_type, DtoLevelType.SWING_SR)
        return LevelData(
            price=self.price,
            level_type=dto_type,
            strength_score=self.strength_score,
            touch_count=self.touch_count,
            zone_high=self.zone_high,
            zone_low=self.zone_low,
            is_resistance_turned_support=self.is_resistance_turned_support,
        )


# ---------------------------------------------------------------------------
# SRAnalysis — M05's result container
# ---------------------------------------------------------------------------

@dataclass
class SRAnalysis:
    """
    Complete S/R analysis result.

    The ``levels`` list is sorted by strength_score descending.
    ``nearest_support`` and ``nearest_resistance`` are the closest qualified
    levels on each side of the current price.
    """
    levels:               List[SRLevel] = field(default_factory=list)
    sma21_level:          Optional[SRLevel] = None
    nearest_support:      Optional[SRLevel] = None
    nearest_resistance:   Optional[SRLevel] = None
    current_price:        float = 0.0
    candles_analyzed:     int   = 0

    @property
    def support_levels(self) -> List[SRLevel]:
        """All support-type levels (including role-reversed)."""
        return [
            lv for lv in self.levels
            if lv.level_type in (LevelType.SUPPORT, LevelType.RESISTANCE_TURNED_SUPPORT)
        ]

    @property
    def resistance_levels(self) -> List[SRLevel]:
        """All resistance-type levels (including role-reversed)."""
        return [
            lv for lv in self.levels
            if lv.level_type in (LevelType.RESISTANCE, LevelType.SUPPORT_TURNED_RESISTANCE)
        ]

    @property
    def strong_levels(self) -> List[SRLevel]:
        """Levels with strength == STRONG."""
        return [lv for lv in self.levels if lv.strength == LevelStrength.STRONG]

    @property
    def all_levels_sorted(self) -> List[SRLevel]:
        """All levels sorted by price ascending."""
        return sorted(self.levels, key=lambda lv: lv.price)

    def levels_near_price(
        self,
        price: float,
        threshold_pips: float = 30.0,
        pip_size: float = 0.0001,
    ) -> List[SRLevel]:
        """Return levels within `threshold_pips` pips of `price`."""
        threshold = threshold_pips * pip_size
        return [lv for lv in self.levels if abs(lv.price - price) <= threshold]


# ---------------------------------------------------------------------------
# Module-level stateless helpers (required by spec)
# ---------------------------------------------------------------------------

def identify_support_levels(
    candles: List[CandleData],
    swing_lows: List[Union[float, object]],    # float prices or SwingPoint objects
    zone_width_pips: float = 10.0,
    pip_size: float = 0.0001,
    merge_pips: float = 15.0,
    recency_window: int = 20,
) -> List[SRLevel]:
    """
    Build support levels from M03 swing lows.

    Args:
        candles:          Full candle series (used for touch counting).
        swing_lows:       List of swing-low prices (float) or SwingPoint objects.
        zone_width_pips:  Half-width of each level zone in pips.
        pip_size:         One pip in price units (default 0.0001).
        merge_pips:       Merge levels closer than this many pips.
        recency_window:   Number of recent candles for recency bonus.

    Returns:
        List of SRLevel objects sorted by strength_score descending.
    """
    prices = _extract_prices(swing_lows)
    return _build_levels(
        prices, LevelType.SUPPORT, candles,
        zone_width_pips, pip_size, merge_pips, recency_window,
    )


def identify_resistance_levels(
    candles: List[CandleData],
    swing_highs: List[Union[float, object]],
    zone_width_pips: float = 10.0,
    pip_size: float = 0.0001,
    merge_pips: float = 15.0,
    recency_window: int = 20,
) -> List[SRLevel]:
    """
    Build resistance levels from M03 swing highs.

    Args:
        candles:          Full candle series (used for touch counting).
        swing_highs:      List of swing-high prices (float) or SwingPoint objects.
        zone_width_pips:  Half-width of each level zone in pips.
        pip_size:         One pip in price units.
        merge_pips:       Merge levels closer than this many pips.
        recency_window:   Number of recent candles for recency bonus.

    Returns:
        List of SRLevel objects sorted by strength_score descending.
    """
    prices = _extract_prices(swing_highs)
    return _build_levels(
        prices, LevelType.RESISTANCE, candles,
        zone_width_pips, pip_size, merge_pips, recency_window,
    )


def score_level(
    level: SRLevel,
    candles: List[CandleData],
    touch_tolerance_pips: float = 5.0,
    pip_size: float = 0.0001,
    recency_window: int = 20,
) -> float:
    """
    Recompute (or refine) the strength score for an existing SRLevel.

    This allows rescoring after additional candles are added without
    rebuilding the full level set.

    Returns:
        Updated strength_score (0.0–10.0).
    """
    tolerance = touch_tolerance_pips * pip_size
    touches = _count_zone_touches(
        candles, level.zone_low, level.zone_high, level.level_type
    )
    recent = candles[-recency_window:] if len(candles) >= recency_window else candles
    recently_tested = any(
        abs(c.close - level.price) <= tolerance
        or abs(c.low  - level.price) <= tolerance
        or abs(c.high - level.price) <= tolerance
        for c in recent
    )
    base  = min(touches * _TOUCH_PER_POINT, _TOUCH_CAP)
    bonus = (_RECENCY_BONUS if recently_tested else 0.0) + (
        _RTS_BONUS if level.is_resistance_turned_support else 0.0
    )
    return min(base + bonus, _SCORE_MAX)


def find_nearest_level(
    price: float,
    levels: List[SRLevel],
    direction: str = "BOTH",
) -> Optional[SRLevel]:
    """
    Return the SRLevel nearest to ``price``.

    Args:
        price:     Reference price.
        levels:    List of SRLevel objects.
        direction: ``"ABOVE"`` — only levels above price;
                   ``"BELOW"`` — only levels below price;
                   ``"BOTH"``  — closest regardless of side (default).

    Returns:
        Nearest SRLevel or None if list is empty / no match.
    """
    if not levels:
        return None
    if direction == "ABOVE":
        candidates = [lv for lv in levels if lv.price > price]
    elif direction == "BELOW":
        candidates = [lv for lv in levels if lv.price < price]
    else:
        candidates = list(levels)
    if not candidates:
        return None
    return min(candidates, key=lambda lv: abs(lv.price - price))


def classify_zone(price: float, level: SRLevel) -> str:
    """
    Classify a price relative to a level zone.

    Returns:
        ``"INSIDE"`` — price is within (or exactly on the boundary of) the level's zone
        ``"ABOVE"``  — price is above the zone (level is below)
        ``"BELOW"``  — price is below the zone (level is above)

    Floating-point tolerance: boundaries are checked within 1e-9 to avoid
    rounding artefacts (e.g. 1.1000 - 0.0010 == 1.0990000000000002).
    """
    _EPS = 1e-9
    if (level.zone_low - _EPS) <= price <= (level.zone_high + _EPS):
        return "INSIDE"
    if price > level.zone_high:
        return "ABOVE"
    return "BELOW"


def detect_role_reversals(
    support_levels: List[SRLevel],
    resistance_levels: List[SRLevel],
    current_price: float,
    tolerance_pips: float = 5.0,
    pip_size: float = 0.0001,
) -> Tuple[List[SRLevel], List[SRLevel]]:
    """
    Detect resistance-turned-support (RTS) and support-turned-resistance (STR).

    RTS: A former resistance level is now BELOW current price → it has been
         broken through and may now act as support.
    STR: A former support level is now ABOVE current price → it has been
         broken through and may now act as resistance.

    Args:
        support_levels:    Detected support levels.
        resistance_levels: Detected resistance levels.
        current_price:     Latest close.
        tolerance_pips:    Pip tolerance for "broken through" detection.
        pip_size:          Pip size.

    Returns:
        Tuple (updated_supports, updated_resistances) with role-reversal
        flags and level_type updated in-place.
    """
    tol = tolerance_pips * pip_size

    updated_supports = list(support_levels)
    updated_resistances = list(resistance_levels)

    # Former resistances now below current price → promoted to RTS support
    for i, res in enumerate(updated_resistances):
        if res.price < current_price - tol:
            # Was resistance, now below — role reversed to support
            updated_resistances[i] = SRLevel(
                price=res.price,
                level_type=LevelType.RESISTANCE_TURNED_SUPPORT,
                strength_score=min(res.strength_score + _RTS_BONUS, _SCORE_MAX),
                touch_count=res.touch_count,
                zone_high=res.zone_high,
                zone_low=res.zone_low,
                is_resistance_turned_support=True,
                last_tested_index=res.last_tested_index,
                formed_index=res.formed_index,
                symbol=res.symbol,
                timeframe=res.timeframe,
            )

    # Former supports now above current price → promoted to STR resistance
    for i, sup in enumerate(updated_supports):
        if sup.price > current_price + tol:
            updated_supports[i] = SRLevel(
                price=sup.price,
                level_type=LevelType.SUPPORT_TURNED_RESISTANCE,
                strength_score=min(sup.strength_score + _RTS_BONUS, _SCORE_MAX),
                touch_count=sup.touch_count,
                zone_high=sup.zone_high,
                zone_low=sup.zone_low,
                is_resistance_turned_support=False,
                last_tested_index=sup.last_tested_index,
                formed_index=sup.formed_index,
                symbol=sup.symbol,
                timeframe=sup.timeframe,
            )

    return updated_supports, updated_resistances


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class SREngine:
    """
    M05 — Support & Resistance Engine.

    Algorithm:
      1. Accept swing highs/lows from M03 (SwingPoint objects or float prices)
      2. Build zone-based S/R levels with touch counting
      3. Detect role reversals relative to current price
      4. Merge nearby levels within merge_threshold
      5. Score every level (0–10)
      6. Add 21 SMA level if provided
      7. Resolve nearest_support / nearest_resistance
      8. Optionally persist to DB (idempotent upsert on symbol+timeframe+price)

    Accepts M03 output in two forms:
      - SwingPoint objects (from StructureAnalysis.swing_highs/swing_lows)
      - Plain float prices (for simpler callers)
      - SwingPointData DTOs (from MarketStructure DTO)

    TQS Scoring (0–25 pts):
      25 — Role-reversal STRONG level
      22 — STRONG level (score >= 7)
      18 — MODERATE level (score 4–6.9)
      12 — WEAK level
       5 — No qualifying nearby level
    """

    ZONE_WIDTH_PIPS:      float = 10.0
    MERGE_PIPS:           float = 15.0
    MAX_LEVELS:           int   = 10
    NEARBY_THRESHOLD_PIPS: float = 30.0
    RECENCY_WINDOW:       int   = 20

    def __init__(
        self,
        zone_width_pips: float = 10.0,
        merge_pips: float = 15.0,
        pip_size: float = 0.0001,
        max_levels: int = 10,
        nearby_threshold_pips: float = 30.0,
        recency_window: int = 20,
        rts_tolerance_pips: float = 5.0,
    ):
        if zone_width_pips <= 0:
            raise ValueError(f"zone_width_pips must be > 0, got {zone_width_pips}")
        if pip_size <= 0:
            raise ValueError(f"pip_size must be > 0, got {pip_size}")
        self.zone_width_pips      = zone_width_pips
        self.merge_pips           = merge_pips
        self.pip_size             = pip_size
        self.zone_width           = zone_width_pips * pip_size
        self.merge_threshold      = merge_pips * pip_size
        self.max_levels           = max_levels
        self.nearby_threshold     = nearby_threshold_pips * pip_size
        self.recency_window       = recency_window
        self.rts_tolerance        = rts_tolerance_pips * pip_size

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def analyze(
        self,
        candles: List[CandleData],
        swing_highs: Optional[List] = None,
        swing_lows: Optional[List] = None,
        sma21: Optional[float] = None,
    ) -> SRAnalysis:
        """
        Identify and score all S/R levels from M03 swing data.

        Args:
            candles:     Full candle series (oldest-first, ascending).
                         Required for touch counting and recency.
            swing_highs: Swing high prices or SwingPoint/SwingPointData objects
                         from M03 StructureAnalysis.  Pass None to skip.
            swing_lows:  Swing low prices or SwingPoint/SwingPointData objects.
            sma21:       Current 21 SMA value for dynamic S/R level (optional).

        Returns:
            SRAnalysis with ranked levels and nearest support/resistance.
            Never raises on empty candles — always returns a safe result.
        """
        if not candles:
            logger.debug("M05 analyze: no candles provided")
            return SRAnalysis(candles_analyzed=0)

        current_price = candles[-1].close

        # ── Step 1: Build raw levels from swing points ────────────────────
        support_lvls    = self._build_levels_from_swings(
            _extract_prices(swing_lows or []),
            LevelType.SUPPORT, candles,
        )
        resistance_lvls = self._build_levels_from_swings(
            _extract_prices(swing_highs or []),
            LevelType.RESISTANCE, candles,
        )

        # ── Step 2: Role reversals ────────────────────────────────────────
        support_lvls, resistance_lvls = detect_role_reversals(
            support_lvls, resistance_lvls,
            current_price,
            tolerance_pips=self.rts_tolerance / self.pip_size,
            pip_size=self.pip_size,
        )

        # ── Step 3: Combine, sort by strength, cap at max_levels ─────────
        all_levels = support_lvls + resistance_lvls
        all_levels.sort(key=lambda lv: lv.strength_score, reverse=True)
        all_levels = all_levels[: self.max_levels * 2]  # keep up to 2× for both sides

        # ── Step 4: 21 SMA dynamic level ─────────────────────────────────
        sma21_level: Optional[SRLevel] = None
        if sma21 is not None and sma21 > 0.0:
            sma21_level = self._build_sma_level(sma21, candles, current_price)
            all_levels.append(sma21_level)

        # ── Step 5: Nearest support / resistance ─────────────────────────
        support_candidates    = [lv for lv in all_levels
                                 if lv.level_type in (LevelType.SUPPORT,
                                                       LevelType.RESISTANCE_TURNED_SUPPORT)]
        resistance_candidates = [lv for lv in all_levels
                                  if lv.level_type in (LevelType.RESISTANCE,
                                                        LevelType.SUPPORT_TURNED_RESISTANCE)]
        # Also consider SMA level as both if present
        if sma21_level:
            if sma21_level.price < current_price:
                support_candidates.append(sma21_level)
            else:
                resistance_candidates.append(sma21_level)

        nearest_support    = find_nearest_level(current_price, support_candidates, "BELOW")
        nearest_resistance = find_nearest_level(current_price, resistance_candidates, "ABOVE")

        logger.debug(
            "M05 analyze: %d levels (S=%d R=%d) price=%.5f",
            len(all_levels),
            len(support_candidates),
            len(resistance_candidates),
            current_price,
        )

        return SRAnalysis(
            levels=all_levels,
            sma21_level=sma21_level,
            nearest_support=nearest_support,
            nearest_resistance=nearest_resistance,
            current_price=current_price,
            candles_analyzed=len(candles),
        )

    def calculate_tqs_level_score(
        self,
        candle: CandleData,
        nearest_support: Optional[SRLevel],
        nearest_resistance: Optional[SRLevel],
        direction: str,
    ) -> int:
        """
        Compute TQS level component score (0–25 points).

        A pattern candle must be AT or NEAR the relevant level.
        Direction 'LONG' → uses nearest_support.
        Direction 'SHORT' → uses nearest_resistance.

        Scoring:
          25 — pattern at role-reversal level
          22 — STRONG level
          18 — MODERATE level
          12 — WEAK level
           5 — no qualifying level within nearby_threshold
        """
        relevant = nearest_support if direction == "LONG" else nearest_resistance

        if relevant is None:
            return 5

        # Distance from pattern candle to level
        candle_price = candle.low if direction == "LONG" else candle.high
        if abs(candle_price - relevant.price) > self.nearby_threshold:
            return 5

        if relevant.is_resistance_turned_support:
            return 25
        strength = relevant.strength
        if strength == LevelStrength.STRONG:
            return 22
        if strength == LevelStrength.MODERATE:
            return 18
        return 12

    def persist_levels(
        self,
        levels: List[SRLevel],
        session,
        symbol: str,
        timeframe: str,
        price_tolerance: float = 0.0,
    ) -> int:
        """
        Upsert S/R levels to the ``sr_levels`` database table.

        Deduplication key: (symbol, timeframe, price within price_tolerance).
        If a matching row already exists it is updated; otherwise inserted.

        Args:
            levels:          SRLevel objects to persist.
            session:         Live SQLAlchemy Session.
            symbol:          Instrument symbol.
            timeframe:       Chart timeframe.
            price_tolerance: Price window for "same level" dedup (default 0.0
                             means exact price match; use pip_size for fuzzy).

        Returns:
            Number of rows inserted or updated.
        """
        from src.db.models import SRLevel as SRLevelORM  # local import to avoid cycle

        if not levels:
            return 0

        tol = price_tolerance if price_tolerance > 0 else self.pip_size

        inserted = 0
        for lv in levels:
            # Look for existing row with same symbol/timeframe/price (±tol)
            existing = (
                session.query(SRLevelORM)
                .filter(
                    SRLevelORM.symbol    == symbol,
                    SRLevelORM.timeframe == timeframe,
                    SRLevelORM.price.between(lv.price - tol, lv.price + tol),
                )
                .first()
            )

            # Map internal direction string for the ORM column
            if lv.level_type in (LevelType.SUPPORT, LevelType.RESISTANCE_TURNED_SUPPORT):
                direction_str = "SUPPORT"
            elif lv.level_type in (LevelType.RESISTANCE, LevelType.SUPPORT_TURNED_RESISTANCE):
                direction_str = "RESISTANCE"
            else:
                direction_str = "BOTH"   # SMA21

            now = datetime.now(timezone.utc)

            if existing:
                # Update existing
                existing.strength_score = int(round(lv.strength_score))
                existing.touch_count    = lv.touch_count
                existing.zone_high      = lv.zone_high
                existing.zone_low       = lv.zone_low
                existing.last_tested    = now
                existing.is_active      = True
                existing.is_rts         = lv.is_resistance_turned_support
                existing.direction      = direction_str
            else:
                orm_row = SRLevelORM(
                    symbol=symbol,
                    timeframe=timeframe,
                    price=lv.price,
                    level_type="SWING_SR" if lv.level_type != LevelType.SMA21 else "SMA_21",
                    direction=direction_str,
                    strength_score=int(round(lv.strength_score)),
                    touch_count=lv.touch_count,
                    zone_high=lv.zone_high,
                    zone_low=lv.zone_low,
                    first_seen=now,
                    last_tested=now,
                    is_active=True,
                    is_rts=lv.is_resistance_turned_support,
                    age_bars=lv.formed_index,
                )
                session.add(orm_row)

            inserted += 1

        session.flush()
        return inserted

    def get_levels(
        self,
        symbol: str,
        timeframe: str,
        session,
        active_only: bool = True,
    ) -> List[SRLevel]:
        """
        Retrieve persisted S/R levels from the database.

        Args:
            symbol:      Instrument symbol.
            timeframe:   Chart timeframe.
            session:     Live SQLAlchemy Session.
            active_only: When True (default), only return is_active=True rows.

        Returns:
            List of SRLevel objects, sorted by price ascending.
        """
        from src.db.models import SRLevel as SRLevelORM

        query = session.query(SRLevelORM).filter(
            SRLevelORM.symbol == symbol,
            SRLevelORM.timeframe == timeframe,
        )
        if active_only:
            query = query.filter(SRLevelORM.is_active.is_(True))

        rows = query.order_by(SRLevelORM.price.asc()).all()

        result: List[SRLevel] = []
        for row in rows:
            direction_str = row.direction.upper() if row.direction else "SUPPORT"
            if direction_str == "SUPPORT":
                lt = (LevelType.RESISTANCE_TURNED_SUPPORT if row.is_rts
                      else LevelType.SUPPORT)
            elif direction_str == "RESISTANCE":
                lt = LevelType.RESISTANCE
            else:
                lt = LevelType.SMA21

            result.append(SRLevel(
                price=row.price,
                level_type=lt,
                strength_score=float(row.strength_score),
                touch_count=row.touch_count,
                zone_high=row.zone_high,
                zone_low=row.zone_low,
                is_resistance_turned_support=bool(row.is_rts),
                symbol=row.symbol,
                timeframe=row.timeframe,
            ))

        return result

    # ------------------------------------------------------------------
    # PRIVATE HELPERS
    # ------------------------------------------------------------------

    def _build_levels_from_swings(
        self,
        prices: List[float],
        level_type: LevelType,
        candles: List[CandleData],
    ) -> List[SRLevel]:
        """Convert a list of swing prices into scored SRLevel objects."""
        if not prices:
            return []

        n = len(candles)
        levels: List[SRLevel] = []

        for idx, price in enumerate(prices):
            half  = self.zone_width / 2.0
            z_hi  = price + half
            z_lo  = price - half

            touches = _count_zone_touches(candles, z_lo, z_hi, level_type)
            score   = _compute_score(touches, price, candles, self.zone_width,
                                     self.recency_window)

            # Find approximate candle index (for metadata)
            formed_idx       = idx   # simple proxy; refined below if SwingPoint passed
            last_tested_idx  = 0

            levels.append(SRLevel(
                price=price,
                level_type=level_type,
                strength_score=score,
                touch_count=touches,
                zone_high=z_hi,
                zone_low=z_lo,
                formed_index=formed_idx,
                last_tested_index=last_tested_idx,
            ))

        # Merge nearby levels, sort by strength, cap at max_levels
        levels = self._merge_nearby_levels(levels)
        levels.sort(key=lambda lv: lv.strength_score, reverse=True)
        return levels[: self.max_levels]

    def _build_sma_level(
        self,
        sma21: float,
        candles: List[CandleData],
        current_price: float,
    ) -> SRLevel:
        """Create a dynamic S/R level at the 21 SMA value."""
        half   = self.zone_width / 2.0
        z_hi   = sma21 + half
        z_lo   = sma21 - half
        # SMA level type depends on position relative to price
        touches = _count_zone_touches(candles, z_lo, z_hi, LevelType.SUPPORT)
        score   = _compute_score(touches, sma21, candles, self.zone_width,
                                 self.recency_window)
        return SRLevel(
            price=sma21,
            level_type=LevelType.SMA21,
            strength_score=score,
            touch_count=touches,
            zone_high=z_hi,
            zone_low=z_lo,
        )

    def _merge_nearby_levels(self, levels: List[SRLevel]) -> List[SRLevel]:
        """Merge levels within merge_threshold of each other (price-sorted)."""
        if len(levels) <= 1:
            return levels

        sorted_lvls = sorted(levels, key=lambda lv: lv.price)
        merged: List[SRLevel] = [sorted_lvls[0]]

        for lv in sorted_lvls[1:]:
            prev = merged[-1]
            if abs(lv.price - prev.price) <= self.merge_threshold:
                # Merge: weighted average price, summed touches, best score + bonus
                new_price    = (prev.price * prev.touch_count + lv.price * lv.touch_count) \
                               / max(prev.touch_count + lv.touch_count, 1)
                new_touches  = prev.touch_count + lv.touch_count
                new_score    = min(max(prev.strength_score, lv.strength_score) + 0.5,
                                   _SCORE_MAX)
                merged[-1] = SRLevel(
                    price=new_price,
                    level_type=prev.level_type,
                    strength_score=new_score,
                    touch_count=new_touches,
                    zone_high=max(prev.zone_high, lv.zone_high),
                    zone_low=min(prev.zone_low, lv.zone_low),
                    is_resistance_turned_support=prev.is_resistance_turned_support,
                    last_tested_index=max(prev.last_tested_index, lv.last_tested_index),
                    formed_index=min(prev.formed_index, lv.formed_index),
                    symbol=prev.symbol,
                    timeframe=prev.timeframe,
                )
            else:
                merged.append(lv)

        return merged


# ---------------------------------------------------------------------------
# Private pure-function helpers (module-level)
# ---------------------------------------------------------------------------

def _build_levels(
    prices: List[float],
    level_type: LevelType,
    candles: List[CandleData],
    zone_width_pips: float = 10.0,
    pip_size: float = 0.0001,
    merge_pips: float = 15.0,
    recency_window: int = 20,
) -> List[SRLevel]:
    """
    Module-level helper: build, score, merge, and sort SRLevel objects from
    a list of prices.  Used by ``identify_support_levels`` /
    ``identify_resistance_levels`` so those functions work without an
    ``SREngine`` instance.
    """
    if not prices:
        return []

    zone_width      = zone_width_pips * pip_size
    merge_threshold = merge_pips * pip_size
    levels: List[SRLevel] = []

    for price in prices:
        half  = zone_width / 2.0
        z_hi  = price + half
        z_lo  = price - half

        touches = _count_zone_touches(candles, z_lo, z_hi, level_type)
        score   = _compute_score(touches, price, candles, zone_width, recency_window)

        levels.append(SRLevel(
            price=price,
            level_type=level_type,
            strength_score=score,
            touch_count=touches,
            zone_high=z_hi,
            zone_low=z_lo,
        ))

    # Merge nearby levels (simple greedy pass on price-sorted list)
    sorted_lvls = sorted(levels, key=lambda lv: lv.price)
    merged: List[SRLevel] = [sorted_lvls[0]]
    for lv in sorted_lvls[1:]:
        prev = merged[-1]
        if abs(lv.price - prev.price) <= merge_threshold:
            new_touches  = prev.touch_count + lv.touch_count
            new_price    = (
                prev.price * prev.touch_count + lv.price * lv.touch_count
            ) / max(new_touches, 1)
            new_score    = min(
                max(prev.strength_score, lv.strength_score) + 0.5, _SCORE_MAX
            )
            merged[-1] = SRLevel(
                price=new_price,
                level_type=prev.level_type,
                strength_score=new_score,
                touch_count=new_touches,
                zone_high=max(prev.zone_high, lv.zone_high),
                zone_low=min(prev.zone_low, lv.zone_low),
            )
        else:
            merged.append(lv)

    merged.sort(key=lambda lv: lv.strength_score, reverse=True)
    return merged


def _extract_prices(items: List) -> List[float]:
    """
    Normalise a mixed list of SwingPoint objects, SwingPointData DTOs,
    or plain floats/ints into a list of float prices.
    """
    prices: List[float] = []
    for item in items:
        if isinstance(item, (int, float)):
            prices.append(float(item))
        elif hasattr(item, "price"):
            prices.append(float(item.price))
        # silently skip anything else
    return prices


def _count_zone_touches(
    candles: List[CandleData],
    zone_low: float,
    zone_high: float,
    level_type: LevelType,
) -> int:
    """
    Count candles that entered a price zone.

    For support zones: count candles whose LOW entered [zone_low, zone_high].
    For resistance zones: count candles whose HIGH entered [zone_low, zone_high].
    Also counts bodies (open/close) that overlap the zone.
    """
    count = 0
    for c in candles:
        if level_type in (LevelType.RESISTANCE, LevelType.SUPPORT_TURNED_RESISTANCE):
            # Resistance touched from below — candle high enters zone
            if zone_low <= c.high <= zone_high:
                count += 1
            elif zone_low <= c.open <= zone_high or zone_low <= c.close <= zone_high:
                count += 1
        else:
            # Support touched from above — candle low enters zone
            if zone_low <= c.low <= zone_high:
                count += 1
            elif zone_low <= c.open <= zone_high or zone_low <= c.close <= zone_high:
                count += 1
    return max(count, 1)   # every swing price counts as at least 1 touch


def _compute_score(
    touch_count: int,
    price: float,
    candles: List[CandleData],
    zone_width: float,
    recency_window: int = 20,
) -> float:
    """
    Compute 0–10 strength score from touches + recency.

    Formula:
      base  = min(touch_count * 1.5, 6.0)
      bonus = +2.0 if any of last recency_window candles is within zone_width
    """
    base        = min(touch_count * _TOUCH_PER_POINT, _TOUCH_CAP)
    recent      = candles[-recency_window:] if len(candles) >= recency_window else candles
    recently    = any(
        abs(c.close - price) <= zone_width
        or abs(c.low  - price) <= zone_width
        or abs(c.high - price) <= zone_width
        for c in recent
    )
    bonus = _RECENCY_BONUS if recently else 0.0
    return min(base + bonus, _SCORE_MAX)
