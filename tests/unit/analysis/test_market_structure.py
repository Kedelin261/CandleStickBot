"""
Tests for M03 — Market Structure Engine.

Coverage:
  - identify_swing_highs / identify_swing_lows / identify_swing_points
  - classify_structure (HH/LH/HL/LL labelling)
  - is_higher_high / is_higher_low / is_lower_high / is_lower_low helpers
  - detect_structure_break (BOS detection)
  - summarize_structure (summary dict)
  - MarketStructureAnalyzer.analyze() — UP / DOWN / RANGING / NONE
  - MarketStructureAnalyzer with custom lookback
  - SwingPoint persistence (insert, dedup, retrieval)
  - Edge cases: empty input, insufficient candles, flat/equal prices,
    unsorted input, single-point series, mixed/contradictory structure
  - to_market_structure() DTO conversion
  - all_swing_points / latest_swing_high / latest_swing_low properties
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import List

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.analysis.market_structure import (
    MarketStructureAnalyzer,
    StructureAnalysis,
    SwingPoint,
    SwingType,
    classify_structure,
    detect_structure_break,
    identify_swing_highs,
    identify_swing_lows,
    identify_swing_points,
    is_higher_high,
    is_higher_low,
    is_lower_high,
    is_lower_low,
    summarize_structure,
)
from src.db.models import Base
from src.db.session import init_db
from src.types import CandleData, MarketStructure, SwingPointData, TrendDirection


# ===========================================================================
# DATA BUILDERS — deterministic zigzag series with clear swing structure
# ===========================================================================

BASE_DATE = datetime(2024, 1, 1)
BASE_SYMBOL = "EURUSD"
BASE_TF = "D1"


def _candle(
    i: int,
    open_: float,
    high: float,
    low: float,
    close: float,
    symbol: str = BASE_SYMBOL,
    timeframe: str = BASE_TF,
) -> CandleData:
    return CandleData(
        symbol=symbol,
        timeframe=timeframe,
        timestamp=BASE_DATE + timedelta(days=i),
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=1000.0,
        spread=1.5,
    )


def make_zigzag_up(
    n_swings: int = 4,
    lookback: int = 3,
    start_price: float = 1.1000,
    swing_size: float = 0.0060,
    step: float = 0.0020,
    symbol: str = BASE_SYMBOL,
    timeframe: str = BASE_TF,
) -> List[CandleData]:
    """
    Build a bullish zigzag candle series: Higher Highs + Higher Lows.

    Each swing consists of `lookback + 1` candles in one direction and
    `lookback` candles in the opposite pullback direction so that at least
    `n_swings` clean pivots are detectable.

    Pattern per swing cycle (up then down):
        lookback+1 rising candles (creates swing HIGH at the top)
        lookback   falling candles (creates swing LOW at the bottom)

    With each cycle:
        swing HIGH increases by `swing_size + step * cycle`
        swing LOW  increases by `step * cycle`   (higher lows)
    """
    candles: List[CandleData] = []
    day = 0
    price = start_price

    for cycle in range(n_swings):
        top = start_price + swing_size * (cycle + 1) + step * cycle
        bottom = start_price + step * cycle

        # Rising leg: lookback+1 candles towards top
        leg_up = lookback + 1
        for j in range(leg_up):
            p = price + (top - price) * (j + 1) / leg_up
            candles.append(_candle(day, price, p + 0.0005, price - 0.0003, p,
                                   symbol=symbol, timeframe=timeframe))
            price = p
            day += 1

        # Falling leg: lookback candles back down towards bottom
        leg_down = lookback
        for j in range(leg_down):
            p = price - (price - bottom) * (j + 1) / leg_down
            candles.append(_candle(day, price, price + 0.0003, p - 0.0005, p,
                                   symbol=symbol, timeframe=timeframe))
            price = p
            day += 1

    return candles


def make_zigzag_down(
    n_swings: int = 4,
    lookback: int = 3,
    start_price: float = 1.1200,
    swing_size: float = 0.0060,
    step: float = 0.0020,
    symbol: str = BASE_SYMBOL,
    timeframe: str = BASE_TF,
) -> List[CandleData]:
    """
    Build a bearish zigzag candle series: Lower Highs + Lower Lows.
    """
    candles: List[CandleData] = []
    day = 0
    price = start_price

    for cycle in range(n_swings):
        bottom = start_price - swing_size * (cycle + 1) - step * cycle
        top = start_price - step * cycle

        # Falling leg
        leg_down = lookback + 1
        for j in range(leg_down):
            p = price - (price - bottom) * (j + 1) / leg_down
            candles.append(_candle(day, price, price + 0.0003, p - 0.0005, p,
                                   symbol=symbol, timeframe=timeframe))
            price = p
            day += 1

        # Rising bounce
        leg_up = lookback
        for j in range(leg_up):
            p = price + (top - price) * (j + 1) / leg_up
            candles.append(_candle(day, price, p + 0.0005, price - 0.0003, p,
                                   symbol=symbol, timeframe=timeframe))
            price = p
            day += 1

    return candles


def make_ranging_series(
    n_cycles: int = 6,
    lookback: int = 3,
    center: float = 1.1000,
    amplitude: float = 0.0050,
    symbol: str = BASE_SYMBOL,
    timeframe: str = BASE_TF,
) -> List[CandleData]:
    """
    Build a sideways ranging candle series with detectable swing points.

    Creates clear oscillations around a center price so that swing highs
    cluster near center+amplitude and swing lows cluster near center-amplitude.
    The tiny per-cycle offset (< 0.02% of price) keeps prices within the
    ranging_tolerance used by classify_structure.
    """
    candles: List[CandleData] = []
    day = 0
    price = center
    # Each leg needs lookback+1 candles to allow swing detection on both sides
    leg = lookback + 1

    for cycle in range(n_cycles):
        # Tiny alternating offset keeps all highs/lows within ~0.02% of mean
        # (ranging_tolerance default = 0.0002, i.e. 0.02% of price)
        alt = 0.0001 if cycle % 2 == 0 else -0.0001
        top = center + amplitude + alt
        bottom = center - amplitude + alt

        # Up leg: create rising candles reaching the top
        for j in range(leg):
            frac = (j + 1) / leg
            p = price + (top - price) * frac
            candles.append(_candle(
                day, price,
                p + 0.0003,           # high slightly above close
                price - 0.0002,       # low slightly below open
                p,
                symbol=symbol, timeframe=timeframe,
            ))
            price = p
            day += 1

        # Down leg: create falling candles reaching the bottom
        for j in range(leg):
            frac = (j + 1) / leg
            p = price - (price - bottom) * frac
            candles.append(_candle(
                day, price,
                price + 0.0002,       # high slightly above open
                p - 0.0003,           # low slightly below close
                p,
                symbol=symbol, timeframe=timeframe,
            ))
            price = p
            day += 1

    return candles


def make_minimal_swing_high_series(lookback: int = 5) -> List[CandleData]:
    """
    Minimal series with exactly one detectable swing high at the center.

    Structure: lookback flat candles, one peak, lookback flat candles.
    """
    n = lookback * 2 + 1
    candles = []
    base = 1.1000
    for i in range(n):
        is_peak = i == lookback
        high = base + 0.0050 if is_peak else base + 0.0010
        low = base - 0.0010
        candles.append(_candle(i, base, high, low, base))
    return candles


def make_minimal_swing_low_series(lookback: int = 5) -> List[CandleData]:
    """
    Minimal series with exactly one detectable swing low at the center.
    """
    n = lookback * 2 + 1
    candles = []
    base = 1.1000
    for i in range(n):
        is_trough = i == lookback
        low = base - 0.0050 if is_trough else base - 0.0010
        high = base + 0.0010
        candles.append(_candle(i, base, high, low, base))
    return candles


# ===========================================================================
# FIXTURES
# ===========================================================================

@pytest.fixture
def analyzer():
    """Default analyzer with lookback=3 for faster test series."""
    return MarketStructureAnalyzer(lookback=3)


@pytest.fixture
def analyzer5():
    """Analyzer with default lookback=5."""
    return MarketStructureAnalyzer(lookback=5)


@pytest.fixture
def bullish_candles():
    return make_zigzag_up(n_swings=4, lookback=3)


@pytest.fixture
def bearish_candles():
    return make_zigzag_down(n_swings=4, lookback=3)


@pytest.fixture
def ranging_series():
    return make_ranging_series(n_cycles=6, lookback=3)


@pytest.fixture
def db_session():
    """In-memory SQLite session for persistence tests."""
    engine = create_engine("sqlite:///:memory:")
    init_db(engine)
    factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    session = factory()
    yield session
    session.rollback()
    session.close()
    engine.dispose()


# ===========================================================================
# 1. SWING HIGH DETECTION
# ===========================================================================

class TestIdentifySwingHighs:
    def test_single_peak_detected(self):
        """Exactly one swing high when series has one clear peak."""
        candles = make_minimal_swing_high_series(lookback=3)
        highs = identify_swing_highs(candles, lookback=3)
        assert len(highs) == 1

    def test_peak_at_correct_index(self):
        """Swing high index points to the candle with the highest high."""
        candles = make_minimal_swing_high_series(lookback=3)
        highs = identify_swing_highs(candles, lookback=3)
        assert highs[0].index == 3   # Middle of 7-candle series (lookback=3)

    def test_swing_type_is_high(self):
        """All returned points have swing_type == HIGH."""
        candles = make_minimal_swing_high_series(lookback=3)
        highs = identify_swing_highs(candles, lookback=3)
        assert all(sp.swing_type == SwingType.HIGH for sp in highs)

    def test_price_equals_candle_high(self):
        """Swing high price matches candle.high at that index."""
        candles = make_minimal_swing_high_series(lookback=3)
        highs = identify_swing_highs(candles, lookback=3)
        for sp in highs:
            assert sp.price == candles[sp.index].high

    def test_empty_input_returns_empty(self):
        """Empty candle list returns empty result."""
        assert identify_swing_highs([], lookback=3) == []

    def test_too_short_returns_empty(self):
        """Series shorter than 2*lookback+1 returns empty."""
        candles = make_minimal_swing_high_series(lookback=3)[:5]  # 5 < 7
        assert identify_swing_highs(candles, lookback=3) == []

    def test_monotonic_rise_no_swing_high(self):
        """A strictly increasing series has no swing highs (no candles higher on BOTH sides)."""
        candles = [_candle(i, 1.1 + i * 0.001, 1.11 + i * 0.001, 1.09 + i * 0.001, 1.105 + i * 0.001)
                   for i in range(20)]
        highs = identify_swing_highs(candles, lookback=3)
        assert len(highs) == 0

    def test_invalid_lookback_raises(self):
        """lookback < 1 raises ValueError."""
        candles = make_minimal_swing_high_series(lookback=3)
        with pytest.raises(ValueError, match="lookback must be >= 1"):
            identify_swing_highs(candles, lookback=0)

    def test_lookback1_detects_local_peaks(self):
        """lookback=1 detects every local max (one candle each side)."""
        candles = [
            _candle(0, 1.10, 1.105, 1.095, 1.100),
            _candle(1, 1.10, 1.115, 1.095, 1.110),  # peak
            _candle(2, 1.10, 1.108, 1.095, 1.102),
        ]
        highs = identify_swing_highs(candles, lookback=1)
        assert len(highs) == 1
        assert highs[0].index == 1

    def test_multiple_zigzag_highs_detected(self):
        """Multiple swing highs detected in zigzag up series."""
        candles = make_zigzag_up(n_swings=4, lookback=3)
        highs = identify_swing_highs(candles, lookback=3)
        assert len(highs) >= 2

    def test_sorted_ascending_by_index(self):
        """Returned swing highs are always sorted ascending by index."""
        candles = make_zigzag_up(n_swings=4, lookback=3)
        highs = identify_swing_highs(candles, lookback=3)
        indices = [sp.index for sp in highs]
        assert indices == sorted(indices)

    def test_candle_reference_preserved(self):
        """SwingPoint.candle refers to the correct CandleData object."""
        candles = make_minimal_swing_high_series(lookback=3)
        highs = identify_swing_highs(candles, lookback=3)
        for sp in highs:
            assert sp.candle is candles[sp.index]


# ===========================================================================
# 2. SWING LOW DETECTION
# ===========================================================================

class TestIdentifySwingLows:
    def test_single_trough_detected(self):
        candles = make_minimal_swing_low_series(lookback=3)
        lows = identify_swing_lows(candles, lookback=3)
        assert len(lows) == 1

    def test_trough_at_correct_index(self):
        candles = make_minimal_swing_low_series(lookback=3)
        lows = identify_swing_lows(candles, lookback=3)
        assert lows[0].index == 3

    def test_swing_type_is_low(self):
        candles = make_minimal_swing_low_series(lookback=3)
        lows = identify_swing_lows(candles, lookback=3)
        assert all(sp.swing_type == SwingType.LOW for sp in lows)

    def test_price_equals_candle_low(self):
        candles = make_minimal_swing_low_series(lookback=3)
        lows = identify_swing_lows(candles, lookback=3)
        for sp in lows:
            assert sp.price == candles[sp.index].low

    def test_empty_returns_empty(self):
        assert identify_swing_lows([], lookback=3) == []

    def test_monotonic_fall_no_swing_low(self):
        candles = [_candle(i, 1.2 - i * 0.001, 1.21 - i * 0.001, 1.19 - i * 0.001, 1.205 - i * 0.001)
                   for i in range(20)]
        assert len(identify_swing_lows(candles, lookback=3)) == 0

    def test_multiple_zigzag_lows_detected(self):
        candles = make_zigzag_up(n_swings=4, lookback=3)
        lows = identify_swing_lows(candles, lookback=3)
        assert len(lows) >= 2

    def test_sorted_ascending_by_index(self):
        candles = make_zigzag_down(n_swings=4, lookback=3)
        lows = identify_swing_lows(candles, lookback=3)
        indices = [sp.index for sp in lows]
        assert indices == sorted(indices)


# ===========================================================================
# 3. IDENTIFY SWING POINTS (COMBINED)
# ===========================================================================

class TestIdentifySwingPoints:
    def test_returns_both_types(self):
        candles = make_zigzag_up(n_swings=4, lookback=3)
        pts = identify_swing_points(candles, lookback=3)
        types = {sp.swing_type for sp in pts}
        assert SwingType.HIGH in types
        assert SwingType.LOW in types

    def test_sorted_ascending_by_index(self):
        candles = make_zigzag_up(n_swings=4, lookback=3)
        pts = identify_swing_points(candles, lookback=3)
        indices = [sp.index for sp in pts]
        assert indices == sorted(indices)

    def test_empty_returns_empty(self):
        assert identify_swing_points([], lookback=3) == []

    def test_count_equals_sum_of_individual(self):
        candles = make_zigzag_up(n_swings=4, lookback=3)
        highs = identify_swing_highs(candles, lookback=3)
        lows = identify_swing_lows(candles, lookback=3)
        pts = identify_swing_points(candles, lookback=3)
        assert len(pts) == len(highs) + len(lows)


# ===========================================================================
# 4. COMPARISON HELPERS
# ===========================================================================

class TestComparisonHelpers:
    """Tests for is_higher_high, is_higher_low, is_lower_high, is_lower_low."""

    def _sp(self, price: float, swing_type: str = SwingType.HIGH) -> SwingPoint:
        c = CandleData("EURUSD", "D1", BASE_DATE, price, price, price, price)
        return SwingPoint(index=0, price=price, swing_type=swing_type, candle=c)

    def test_is_higher_high_true(self):
        assert is_higher_high(self._sp(1.11), self._sp(1.10)) is True

    def test_is_higher_high_false(self):
        assert is_higher_high(self._sp(1.10), self._sp(1.11)) is False

    def test_is_higher_high_equal_is_false(self):
        assert is_higher_high(self._sp(1.10), self._sp(1.10)) is False

    def test_is_higher_low_true(self):
        assert is_higher_low(self._sp(1.09, SwingType.LOW), self._sp(1.08, SwingType.LOW)) is True

    def test_is_higher_low_false(self):
        assert is_higher_low(self._sp(1.08, SwingType.LOW), self._sp(1.09, SwingType.LOW)) is False

    def test_is_lower_high_true(self):
        assert is_lower_high(self._sp(1.10), self._sp(1.11)) is True

    def test_is_lower_high_false(self):
        assert is_lower_high(self._sp(1.11), self._sp(1.10)) is False

    def test_is_lower_low_true(self):
        assert is_lower_low(self._sp(1.08, SwingType.LOW), self._sp(1.09, SwingType.LOW)) is True

    def test_is_lower_low_false(self):
        assert is_lower_low(self._sp(1.09, SwingType.LOW), self._sp(1.08, SwingType.LOW)) is False


# ===========================================================================
# 5. CLASSIFY STRUCTURE
# ===========================================================================

class TestClassifyStructure:
    def test_bullish_returns_up(self, bullish_candles):
        pts = identify_swing_points(bullish_candles, lookback=3)
        direction, high_labels, low_labels, _ = classify_structure(pts)
        assert direction == TrendDirection.UP

    def test_bearish_returns_down(self, bearish_candles):
        pts = identify_swing_points(bearish_candles, lookback=3)
        direction, _, _, _ = classify_structure(pts)
        assert direction == TrendDirection.DOWN

    def test_empty_swing_points_returns_none(self):
        direction, hl, ll, _ = classify_structure([])
        assert direction == TrendDirection.NONE
        assert hl == []
        assert ll == []

    def test_only_highs_returns_none(self, bullish_candles):
        """Only swing highs (no lows) → NONE."""
        pts = identify_swing_highs(bullish_candles, lookback=3)
        direction, _, _, _ = classify_structure(pts)
        assert direction == TrendDirection.NONE

    def test_high_labels_hh_in_uptrend(self, bullish_candles):
        pts = identify_swing_points(bullish_candles, lookback=3)
        _, high_labels, _, _ = classify_structure(pts)
        # All non-UNDEFINED high labels should be "HH"
        actionable = [lbl for lbl in high_labels if lbl != "UNDEFINED"]
        assert all(lbl == "HH" for lbl in actionable)

    def test_low_labels_hl_in_uptrend(self, bullish_candles):
        pts = identify_swing_points(bullish_candles, lookback=3)
        _, _, low_labels, _ = classify_structure(pts)
        actionable = [lbl for lbl in low_labels if lbl != "UNDEFINED"]
        assert all(lbl == "HL" for lbl in actionable)

    def test_high_labels_lh_in_downtrend(self, bearish_candles):
        pts = identify_swing_points(bearish_candles, lookback=3)
        _, high_labels, _, _ = classify_structure(pts)
        actionable = [lbl for lbl in high_labels if lbl != "UNDEFINED"]
        assert all(lbl == "LH" for lbl in actionable)

    def test_low_labels_ll_in_downtrend(self, bearish_candles):
        pts = identify_swing_points(bearish_candles, lookback=3)
        _, _, low_labels, _ = classify_structure(pts)
        actionable = [lbl for lbl in low_labels if lbl != "UNDEFINED"]
        assert all(lbl == "LL" for lbl in actionable)

    def test_first_label_is_undefined(self, bullish_candles):
        """First swing of each type must be labeled UNDEFINED (no prior for comparison)."""
        pts = identify_swing_points(bullish_candles, lookback=3)
        _, high_labels, low_labels, _ = classify_structure(pts)
        if high_labels:
            assert high_labels[0] == "UNDEFINED"
        if low_labels:
            assert low_labels[0] == "UNDEFINED"


# ===========================================================================
# 6. ANALYZE — MAIN PUBLIC API
# ===========================================================================

class TestAnalyzeUptrend:
    def test_direction_is_up(self, analyzer, bullish_candles):
        result = analyzer.analyze(bullish_candles)
        assert result.direction == TrendDirection.UP

    def test_swing_highs_detected(self, analyzer, bullish_candles):
        result = analyzer.analyze(bullish_candles)
        assert len(result.swing_highs) >= 2

    def test_swing_lows_detected(self, analyzer, bullish_candles):
        result = analyzer.analyze(bullish_candles)
        assert len(result.swing_lows) >= 2

    def test_last_hh_is_populated(self, analyzer, bullish_candles):
        result = analyzer.analyze(bullish_candles)
        assert result.last_hh is not None
        assert result.last_hh > 0

    def test_last_hl_is_populated(self, analyzer, bullish_candles):
        result = analyzer.analyze(bullish_candles)
        assert result.last_hl is not None

    def test_last_lh_is_none_in_uptrend(self, analyzer, bullish_candles):
        """In a clean uptrend there are no LH labels, so last_lh is None."""
        result = analyzer.analyze(bullish_candles)
        assert result.last_lh is None

    def test_last_ll_is_none_in_uptrend(self, analyzer, bullish_candles):
        result = analyzer.analyze(bullish_candles)
        assert result.last_ll is None

    def test_confidence_above_zero(self, analyzer, bullish_candles):
        result = analyzer.analyze(bullish_candles)
        assert result.confidence > 0.0

    def test_candles_analyzed_correct(self, analyzer, bullish_candles):
        result = analyzer.analyze(bullish_candles)
        assert result.candles_analyzed == len(bullish_candles)

    def test_not_undefined(self, analyzer, bullish_candles):
        result = analyzer.analyze(bullish_candles)
        assert result.direction != TrendDirection.NONE

    def test_is_trending_true(self, analyzer, bullish_candles):
        result = analyzer.analyze(bullish_candles)
        assert result.is_trending is True

    def test_is_ranging_flag_false_in_uptrend(self, analyzer, bullish_candles):
        result = analyzer.analyze(bullish_candles)
        assert result.is_ranging is False

    def test_hh_higher_than_hl(self, analyzer, bullish_candles):
        result = analyzer.analyze(bullish_candles)
        if result.last_hh and result.last_hl:
            assert result.last_hh > result.last_hl


class TestAnalyzeDowntrend:
    def test_direction_is_down(self, analyzer, bearish_candles):
        result = analyzer.analyze(bearish_candles)
        assert result.direction == TrendDirection.DOWN

    def test_last_lh_is_populated(self, analyzer, bearish_candles):
        result = analyzer.analyze(bearish_candles)
        assert result.last_lh is not None

    def test_last_ll_is_populated(self, analyzer, bearish_candles):
        result = analyzer.analyze(bearish_candles)
        assert result.last_ll is not None

    def test_last_hh_is_none_in_downtrend(self, analyzer, bearish_candles):
        result = analyzer.analyze(bearish_candles)
        assert result.last_hh is None

    def test_last_hl_is_none_in_downtrend(self, analyzer, bearish_candles):
        result = analyzer.analyze(bearish_candles)
        assert result.last_hl is None

    def test_confidence_above_zero(self, analyzer, bearish_candles):
        result = analyzer.analyze(bearish_candles)
        assert result.confidence > 0.0

    def test_is_trending_true(self, analyzer, bearish_candles):
        result = analyzer.analyze(bearish_candles)
        assert result.is_trending is True

    def test_ll_lower_than_lh(self, analyzer, bearish_candles):
        result = analyzer.analyze(bearish_candles)
        if result.last_lh and result.last_ll:
            assert result.last_ll < result.last_lh


class TestAnalyzeRanging:
    def test_structure_type_is_ranging(self, analyzer, ranging_series):
        """Ranging series: structure_type returns 'RANGING' (direction.is_ranging == True)."""
        result = analyzer.analyze(ranging_series)
        assert result.structure_type == "RANGING"

    def test_direction_is_none_for_ranging(self, analyzer, ranging_series):
        """TrendDirection is NONE for ranging (no UP/DOWN directional bias)."""
        result = analyzer.analyze(ranging_series)
        assert result.direction == TrendDirection.NONE

    def test_is_ranging_flag_true(self, analyzer, ranging_series):
        result = analyzer.analyze(ranging_series)
        assert result.is_ranging is True

    def test_swing_highs_detected(self, analyzer, ranging_series):
        result = analyzer.analyze(ranging_series)
        assert len(result.swing_highs) >= 2

    def test_swing_lows_detected(self, analyzer, ranging_series):
        result = analyzer.analyze(ranging_series)
        assert len(result.swing_lows) >= 2

    def test_is_trending_false(self, analyzer, ranging_series):
        result = analyzer.analyze(ranging_series)
        assert result.is_trending is False


class TestAnalyzeInsufficientData:
    def test_empty_returns_none(self, analyzer):
        result = analyzer.analyze([])
        assert result.direction == TrendDirection.NONE

    def test_too_short_returns_none(self, analyzer):
        """Fewer than 2*lookback+1 candles → NONE direction."""
        candles = make_zigzag_up(n_swings=1, lookback=3)[:5]
        result = analyzer.analyze(candles)
        assert result.direction == TrendDirection.NONE

    def test_too_short_candles_analyzed_correct(self, analyzer):
        short = [_candle(i, 1.1, 1.11, 1.09, 1.10) for i in range(3)]
        result = analyzer.analyze(short)
        assert result.candles_analyzed == 3

    def test_exactly_min_candles_does_not_crash(self, analyzer):
        """Exactly 2*lookback+1 candles — may return NONE but must not raise."""
        min_n = 2 * analyzer.lookback + 1
        candles = [_candle(i, 1.1, 1.11, 1.09, 1.10) for i in range(min_n)]
        result = analyzer.analyze(candles)
        assert result.direction in (TrendDirection.NONE, TrendDirection.UP, TrendDirection.DOWN)
        assert result.structure_type in ("NONE", "UP", "DOWN", "RANGING")

    def test_no_swing_points_returns_none(self, analyzer):
        """Flat candles above minimum length — no swings detected → NONE."""
        candles = [_candle(i, 1.1000, 1.1010, 1.0990, 1.1000) for i in range(20)]
        result = analyzer.analyze(candles)
        assert result.direction == TrendDirection.NONE

    def test_reason_populated_on_none(self, analyzer):
        result = analyzer.analyze([])
        assert isinstance(result.reason, str)
        assert len(result.reason) > 0


# ===========================================================================
# 7. LOOKBACK PARAMETER BEHAVIOR
# ===========================================================================

class TestLookbackBehavior:
    def test_smaller_lookback_detects_more_swings(self):
        """lookback=2 should detect at least as many swings as lookback=4 for same data."""
        candles = make_zigzag_up(n_swings=4, lookback=3)
        highs_small = identify_swing_highs(candles, lookback=2)
        highs_large = identify_swing_highs(candles, lookback=4)
        # Smaller lookback is more sensitive
        assert len(highs_small) >= len(highs_large)

    def test_lookback_stored_in_strength(self):
        candles = make_minimal_swing_high_series(lookback=4)
        highs = identify_swing_highs(candles, lookback=4)
        if highs:
            assert highs[0].strength == 4

    def test_custom_lookback_5_works(self):
        candles = make_zigzag_up(n_swings=5, lookback=5)
        analyzer = MarketStructureAnalyzer(lookback=5)
        result = analyzer.analyze(candles)
        # Should detect structure without raising
        assert result.direction in (TrendDirection.UP, TrendDirection.NONE, TrendDirection.DOWN)
        assert result.structure_type in ("UP", "DOWN", "RANGING", "NONE")

    def test_lookback_1_detects_fine_grained_pivots(self):
        candles = make_zigzag_up(n_swings=4, lookback=3)
        highs = identify_swing_highs(candles, lookback=1)
        assert len(highs) >= 2

    def test_invalid_lookback_raises(self):
        with pytest.raises(ValueError):
            MarketStructureAnalyzer(lookback=0)


# ===========================================================================
# 8. MIXED / CONTRADICTORY STRUCTURE
# ===========================================================================

class TestContradictoryStructure:
    def test_mixed_hh_ll_returns_none_or_ranging(self):
        """HH on highs but LL on lows → contradictory → NONE or RANGING, never UP/DOWN."""
        candles = make_zigzag_up(n_swings=2, lookback=3) + make_zigzag_down(n_swings=2, lookback=3)
        analyzer = MarketStructureAnalyzer(lookback=3)
        result = analyzer.analyze(candles)
        # Mixed structure — could be RANGING or NONE, must not be UP or DOWN
        assert result.structure_type in ("NONE", "RANGING")
        assert result.direction in (TrendDirection.NONE,)

    def test_single_swing_returns_none(self):
        """Only one swing of each type (no pair to classify) → NONE."""
        candles = make_minimal_swing_high_series(lookback=3)
        # Only has 1 swing high, 0 swing lows → NONE
        analyzer = MarketStructureAnalyzer(lookback=3)
        result = analyzer.analyze(candles)
        assert result.direction == TrendDirection.NONE


# ===========================================================================
# 9. DETECT STRUCTURE BREAK
# ===========================================================================

class TestDetectStructureBreak:
    def test_close_above_last_swing_high_breaks_up(self, analyzer, bullish_candles):
        result = analyzer.analyze(bullish_candles)
        if not result.swing_highs:
            pytest.skip("No swing highs detected")
        last_high = result.swing_highs[-1].price
        # Append a candle that closes well above last swing high
        breakout_candle = _candle(
            len(bullish_candles),
            last_high, last_high + 0.005, last_high - 0.001, last_high + 0.004
        )
        extended = bullish_candles + [breakout_candle]
        broke_up, broke_down = analyzer.detect_structure_break(extended, result)
        assert broke_up is True

    def test_close_below_last_swing_low_breaks_down(self, analyzer, bearish_candles):
        result = analyzer.analyze(bearish_candles)
        if not result.swing_lows:
            pytest.skip("No swing lows detected")
        last_low = result.swing_lows[-1].price
        breakdown_candle = _candle(
            len(bearish_candles),
            last_low, last_low + 0.001, last_low - 0.005, last_low - 0.004
        )
        extended = bearish_candles + [breakdown_candle]
        broke_up, broke_down = analyzer.detect_structure_break(extended, result)
        assert broke_down is True

    def test_no_break_returns_false_false(self, analyzer, bullish_candles):
        result = analyzer.analyze(bullish_candles)
        if not result.swing_highs:
            pytest.skip("No swing highs")
        last_high = result.swing_highs[-1].price
        # Candle that stays inside the range
        inside_candle = _candle(
            len(bullish_candles),
            last_high - 0.005, last_high - 0.001, last_high - 0.008, last_high - 0.002
        )
        extended = bullish_candles + [inside_candle]
        broke_up, broke_down = analyzer.detect_structure_break(extended, result)
        assert broke_up is False

    def test_empty_candles_returns_false_false(self, analyzer):
        result = StructureAnalysis(direction=TrendDirection.NONE)
        broke_up, broke_down = analyzer.detect_structure_break([], result)
        assert broke_up is False
        assert broke_down is False

    def test_module_level_detect_structure_break(self, bullish_candles):
        """Module-level detect_structure_break function works independently."""
        broke_up, broke_down = detect_structure_break(bullish_candles, "UP")
        assert isinstance(broke_up, bool)
        assert isinstance(broke_down, bool)


# ===========================================================================
# 10. SUMMARIZE STRUCTURE
# ===========================================================================

class TestSummarizeStructure:
    def test_returns_dict(self, bullish_candles):
        pts = identify_swing_points(bullish_candles, lookback=3)
        summary = summarize_structure(bullish_candles, pts)
        assert isinstance(summary, dict)

    def test_has_required_keys(self, bullish_candles):
        pts = identify_swing_points(bullish_candles, lookback=3)
        summary = summarize_structure(bullish_candles, pts)
        for key in ("candle_count", "swing_high_count", "swing_low_count",
                    "direction", "confidence", "reason"):
            assert key in summary, f"Missing key: {key}"

    def test_direction_matches_analyze(self, analyzer, bullish_candles):
        result = analyzer.analyze(bullish_candles)
        pts = result.all_swing_points
        summary = summarize_structure(bullish_candles, pts)
        assert summary["direction"] == result.structure_type

    def test_candle_count_correct(self, bullish_candles):
        pts = identify_swing_points(bullish_candles, lookback=3)
        summary = summarize_structure(bullish_candles, pts)
        assert summary["candle_count"] == len(bullish_candles)

    def test_empty_swing_points(self):
        candles = [_candle(i, 1.1, 1.11, 1.09, 1.10) for i in range(5)]
        summary = summarize_structure(candles, [])
        assert summary["swing_high_count"] == 0
        assert summary["swing_low_count"] == 0


# ===========================================================================
# 11. STRUCTURE ANALYSIS PROPERTIES
# ===========================================================================

class TestStructureAnalysisProperties:
    def test_all_swing_points_sorted(self, analyzer, bullish_candles):
        result = analyzer.analyze(bullish_candles)
        pts = result.all_swing_points
        indices = [sp.index for sp in pts]
        assert indices == sorted(indices)

    def test_all_swing_points_contains_both_types(self, analyzer, bullish_candles):
        result = analyzer.analyze(bullish_candles)
        pts = result.all_swing_points
        if result.swing_highs and result.swing_lows:
            types = {sp.swing_type for sp in pts}
            assert SwingType.HIGH in types
            assert SwingType.LOW in types

    def test_latest_swing_high(self, analyzer, bullish_candles):
        result = analyzer.analyze(bullish_candles)
        if result.swing_highs:
            assert result.latest_swing_high is result.swing_highs[-1]

    def test_latest_swing_low(self, analyzer, bullish_candles):
        result = analyzer.analyze(bullish_candles)
        if result.swing_lows:
            assert result.latest_swing_low is result.swing_lows[-1]

    def test_latest_swing_high_none_when_empty(self):
        result = StructureAnalysis(direction=TrendDirection.NONE)
        assert result.latest_swing_high is None

    def test_latest_swing_low_none_when_empty(self):
        result = StructureAnalysis(direction=TrendDirection.NONE)
        assert result.latest_swing_low is None

    def test_is_defined_true_for_up(self, analyzer, bullish_candles):
        result = analyzer.analyze(bullish_candles)
        if result.structure_type == "UP":
            assert result.is_defined is True

    def test_is_defined_false_for_none(self, analyzer):
        result = analyzer.analyze([])
        assert result.is_defined is False


# ===========================================================================
# 12. TO_MARKET_STRUCTURE DTO CONVERSION
# ===========================================================================

class TestToMarketStructure:
    def test_returns_market_structure_type(self, analyzer, bullish_candles):
        result = analyzer.analyze(bullish_candles)
        ms = result.to_market_structure()
        assert isinstance(ms, MarketStructure)

    def test_regime_matches_direction(self, analyzer, bullish_candles):
        result = analyzer.analyze(bullish_candles)
        ms = result.to_market_structure()
        assert ms.regime == result.direction

    def test_swing_highs_dtos_are_swing_point_data(self, analyzer, bullish_candles):
        result = analyzer.analyze(bullish_candles)
        ms = result.to_market_structure()
        for sh in ms.swing_highs:
            assert isinstance(sh, SwingPointData)
            assert sh.swing_type == SwingType.HIGH

    def test_swing_lows_dtos_are_swing_point_data(self, analyzer, bullish_candles):
        result = analyzer.analyze(bullish_candles)
        ms = result.to_market_structure()
        for sl in ms.swing_lows:
            assert isinstance(sl, SwingPointData)
            assert sl.swing_type == SwingType.LOW

    def test_symbol_and_timeframe_preserved(self, analyzer):
        candles = make_zigzag_up(n_swings=4, lookback=3, symbol="GBPUSD", timeframe="H4")
        result = analyzer.analyze(candles)
        ms = result.to_market_structure()
        if result.swing_highs or result.swing_lows:
            assert ms.symbol == "GBPUSD"
            assert ms.timeframe == "H4"

    def test_last_hh_copied(self, analyzer, bullish_candles):
        result = analyzer.analyze(bullish_candles)
        ms = result.to_market_structure()
        assert ms.last_hh == result.last_hh

    def test_empty_analysis_gives_empty_symbol(self):
        result = StructureAnalysis(direction=TrendDirection.NONE)
        ms = result.to_market_structure()
        assert ms.symbol == ""
        assert ms.timeframe == ""


# ===========================================================================
# 13. SWING POINT PERSISTENCE
# ===========================================================================

class TestSwingPointPersistence:
    def test_persist_inserts_rows(self, analyzer, bullish_candles, db_session):
        result = analyzer.analyze(bullish_candles)
        all_pts = result.all_swing_points
        if not all_pts:
            pytest.skip("No swing points detected")
        count = analyzer.persist_swing_points(all_pts, db_session)
        db_session.commit()
        assert count == len(all_pts)

    def test_persist_no_duplicates_on_second_call(self, analyzer, bullish_candles, db_session):
        result = analyzer.analyze(bullish_candles)
        all_pts = result.all_swing_points
        if not all_pts:
            pytest.skip("No swing points detected")
        count1 = analyzer.persist_swing_points(all_pts, db_session)
        db_session.commit()
        count2 = analyzer.persist_swing_points(all_pts, db_session)
        db_session.commit()
        assert count2 == 0   # Second insert should find all rows already exist

    def test_persist_empty_list_returns_zero(self, analyzer, db_session):
        count = analyzer.persist_swing_points([], db_session)
        assert count == 0

    def test_get_swing_points_returns_persisted(self, analyzer, bullish_candles, db_session):
        result = analyzer.analyze(bullish_candles)
        all_pts = result.all_swing_points
        if not all_pts:
            pytest.skip("No swing points detected")
        analyzer.persist_swing_points(all_pts, db_session)
        db_session.commit()

        symbol = bullish_candles[0].symbol
        tf = bullish_candles[0].timeframe
        retrieved = analyzer.get_swing_points(symbol, tf, db_session)
        assert len(retrieved) == len(all_pts)

    def test_get_swing_points_filter_by_type(self, analyzer, bullish_candles, db_session):
        result = analyzer.analyze(bullish_candles)
        all_pts = result.all_swing_points
        if not result.swing_highs:
            pytest.skip("No swing highs")
        analyzer.persist_swing_points(all_pts, db_session)
        db_session.commit()

        symbol = bullish_candles[0].symbol
        tf = bullish_candles[0].timeframe
        highs = analyzer.get_swing_points(symbol, tf, db_session, swing_type=SwingType.HIGH)
        assert all(sp.swing_type == SwingType.HIGH for sp in highs)
        assert len(highs) == len(result.swing_highs)

    def test_get_swing_points_empty_db_returns_empty(self, analyzer, db_session):
        result = analyzer.get_swing_points("EURUSD", "D1", db_session)
        assert result == []

    def test_persisted_prices_correct(self, analyzer, bullish_candles, db_session):
        result = analyzer.analyze(bullish_candles)
        all_pts = result.all_swing_points
        if not all_pts:
            pytest.skip("No swing points detected")
        analyzer.persist_swing_points(all_pts, db_session)
        db_session.commit()

        symbol = bullish_candles[0].symbol
        tf = bullish_candles[0].timeframe
        retrieved = analyzer.get_swing_points(symbol, tf, db_session)
        original_prices = sorted(sp.price for sp in all_pts)
        retrieved_prices = sorted(sp.price for sp in retrieved)
        assert original_prices == pytest.approx(retrieved_prices, rel=1e-6)


# ===========================================================================
# 14. SYMBOL / TIMEFRAME ISOLATION
# ===========================================================================

class TestSymbolIsolation:
    def test_different_symbols_detected_independently(self):
        """Analyzer works on any symbol, not just EURUSD."""
        candles_eur = make_zigzag_up(n_swings=4, lookback=3, symbol="EURUSD")
        candles_gbp = make_zigzag_up(n_swings=4, lookback=3, symbol="GBPUSD")
        analyzer = MarketStructureAnalyzer(lookback=3)
        r_eur = analyzer.analyze(candles_eur)
        r_gbp = analyzer.analyze(candles_gbp)
        assert r_eur.direction == r_gbp.direction  # Same structure, different symbol

    def test_different_timeframes_work(self):
        candles_d1 = make_zigzag_up(n_swings=4, lookback=3, timeframe="D1")
        candles_h4 = make_zigzag_up(n_swings=4, lookback=3, timeframe="H4")
        analyzer = MarketStructureAnalyzer(lookback=3)
        r_d1 = analyzer.analyze(candles_d1)
        r_h4 = analyzer.analyze(candles_h4)
        # Both should return valid (non-crash) results
        assert r_d1.direction in list(TrendDirection)
        assert r_h4.direction in list(TrendDirection)


# ===========================================================================
# 15. SWING POINT DTO CONVERSION
# ===========================================================================

class TestSwingPointToDto:
    def test_to_dto_returns_swing_point_data(self):
        c = _candle(0, 1.1000, 1.1100, 1.0950, 1.1050)
        sp = SwingPoint(index=0, price=1.1100, swing_type=SwingType.HIGH, candle=c, strength=5)
        dto = sp.to_dto()
        assert isinstance(dto, SwingPointData)

    def test_to_dto_price_preserved(self):
        c = _candle(0, 1.1000, 1.1100, 1.0950, 1.1050)
        sp = SwingPoint(index=0, price=1.1100, swing_type=SwingType.HIGH, candle=c, strength=5)
        dto = sp.to_dto()
        assert dto.price == 1.1100

    def test_to_dto_swing_type_preserved(self):
        c = _candle(0, 1.1000, 1.1100, 1.0950, 1.1050)
        sp = SwingPoint(index=0, price=1.1100, swing_type=SwingType.HIGH, candle=c, strength=5)
        dto = sp.to_dto()
        assert dto.swing_type == SwingType.HIGH

    def test_to_dto_timestamp_preserved(self):
        c = _candle(5, 1.1000, 1.1100, 1.0950, 1.1050)
        sp = SwingPoint(index=5, price=1.1100, swing_type=SwingType.HIGH, candle=c, strength=5)
        dto = sp.to_dto()
        assert dto.timestamp == c.timestamp
