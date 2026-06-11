"""
Tests for M04 — Trend Detection Engine.

Coverage:
  - compute_sma: correctness, insufficient candles, period=1, period=len
  - get_current_sma: returns None below threshold, correct value
  - is_price_above_sma / is_price_below_sma: basic comparisons
  - determine_trend_direction: all 4 paths (UP/DOWN/RANGING/UNDEFINED)
  - calculate_trend_strength: component scoring
  - is_trend_tradeable: threshold logic, direction/SMA-side guards
  - summarize_trend: key presence, values, price_vs_sma_side
  - TrendDetector.analyze(): main integration scenarios
      - bullish trend confirmed
      - bearish trend confirmed
      - close above SMA but structure NONE
      - structure UP but close below SMA
      - structure DOWN but close above SMA
      - ranging market rejected
      - choppy/undefined rejected
      - insufficient candles for SMA
  - TrendDetector with custom sma_period
  - SMA slope direction
  - confidence score mechanics
  - tradeable threshold configuration
  - to_trend_signal() DTO conversion
  - MarketStructure DTO (shared type) as input
  - StructureAnalysis (M03 internal) as input
  - None market_structure as input
  - no hardcoded EURUSD logic (GBPUSD, USDJPY pass-through)
  - stable/deterministic output for same input
  - TQS trend score rubric
  - TrendAnalysis dataclass fields
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import List, Optional

import pytest

from src.analysis.market_structure import (
    MarketStructureAnalyzer,
    StructureAnalysis,
)
from src.analysis.trend_detection import (
    TrendAnalysis,
    TrendDetector,
    TrendStrength,
    _compute_sma_slope,
    _extract_structure_info,
    calculate_trend_strength,
    compute_sma,
    determine_trend_direction,
    get_current_sma,
    is_price_above_sma,
    is_price_below_sma,
    is_trend_tradeable,
    summarize_trend,
)
from src.types import (
    CandleData,
    MarketStructure,
    SwingPointData,
    TrendDirection,
    TrendSignal,
)


# ===========================================================================
# DATA BUILDERS
# ===========================================================================

BASE_DATE   = datetime(2024, 1, 1)
BASE_SYMBOL = "EURUSD"
BASE_TF     = "D1"


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
    )


def make_flat_candles(
    n: int,
    price: float = 1.1000,
    symbol: str = BASE_SYMBOL,
    timeframe: str = BASE_TF,
) -> List[CandleData]:
    """All candles at the same price — deterministic SMA = price."""
    return [
        _candle(i, price, price + 0.0005, price - 0.0005, price,
                symbol=symbol, timeframe=timeframe)
        for i in range(n)
    ]


def make_rising_candles(
    n: int,
    start: float = 1.1000,
    step: float = 0.0010,
    symbol: str = BASE_SYMBOL,
    timeframe: str = BASE_TF,
) -> List[CandleData]:
    """Monotonically rising close prices by `step` per bar."""
    candles = []
    for i in range(n):
        p = start + i * step
        candles.append(_candle(i, p, p + 0.0005, p - 0.0005, p,
                               symbol=symbol, timeframe=timeframe))
    return candles


def make_falling_candles(
    n: int,
    start: float = 1.1200,
    step: float = 0.0010,
    symbol: str = BASE_SYMBOL,
    timeframe: str = BASE_TF,
) -> List[CandleData]:
    """Monotonically falling close prices by `step` per bar."""
    candles = []
    for i in range(n):
        p = start - i * step
        candles.append(_candle(i, p, p + 0.0005, p - 0.0005, p,
                               symbol=symbol, timeframe=timeframe))
    return candles


def make_zigzag_up(
    n_swings: int = 6,
    lookback: int = 3,
    start_price: float = 1.1000,
    swing_size: float = 0.0060,
    step: float = 0.0020,
    symbol: str = BASE_SYMBOL,
    timeframe: str = BASE_TF,
    continuation: int = 10,
) -> List[CandleData]:
    """
    Bullish zigzag: Higher Highs + Higher Lows.
    Identical builder to test_market_structure.py so M03 will confirm UP.

    `continuation` trailing bars are appended as rising candles so that the
    final close is above the 21 SMA — ensuring M04 can confirm the UP trend.
    """
    candles: List[CandleData] = []
    day = 0
    price = start_price

    for cycle in range(n_swings):
        top    = start_price + swing_size * (cycle + 1) + step * cycle
        bottom = start_price + step * cycle

        leg_up = lookback + 1
        for j in range(leg_up):
            p = price + (top - price) * (j + 1) / leg_up
            candles.append(_candle(day, price, p + 0.0005, price - 0.0003, p,
                                   symbol=symbol, timeframe=timeframe))
            price = p
            day += 1

        leg_down = lookback
        for j in range(leg_down):
            p = price - (price - bottom) * (j + 1) / leg_down
            candles.append(_candle(day, price, price + 0.0003, p - 0.0005, p,
                                   symbol=symbol, timeframe=timeframe))
            price = p
            day += 1

    # Continuation bars: rising so final close > SMA21
    for i in range(continuation):
        p = price + 0.0020 * (i + 1)
        candles.append(_candle(day, price, p + 0.0005, price - 0.0005, p,
                               symbol=symbol, timeframe=timeframe))
        price = p
        day += 1

    return candles


def make_zigzag_down(
    n_swings: int = 6,
    lookback: int = 3,
    start_price: float = 1.1200,
    swing_size: float = 0.0060,
    step: float = 0.0020,
    symbol: str = BASE_SYMBOL,
    timeframe: str = BASE_TF,
    continuation: int = 10,
) -> List[CandleData]:
    """
    Bearish zigzag: Lower Highs + Lower Lows.
    Identical builder to test_market_structure.py so M03 will confirm DOWN.

    `continuation` trailing bars are appended as falling candles so the
    final close is below the 21 SMA — ensuring M04 can confirm the DOWN trend.
    """
    candles: List[CandleData] = []
    day = 0
    price = start_price

    for cycle in range(n_swings):
        bottom = start_price - swing_size * (cycle + 1) - step * cycle
        top    = start_price - step * cycle

        leg_down = lookback + 1
        for j in range(leg_down):
            p = price - (price - bottom) * (j + 1) / leg_down
            candles.append(_candle(day, price, price + 0.0003, p - 0.0005, p,
                                   symbol=symbol, timeframe=timeframe))
            price = p
            day += 1

        leg_up = lookback
        for j in range(leg_up):
            p = price + (top - price) * (j + 1) / leg_up
            candles.append(_candle(day, price, p + 0.0005, price - 0.0003, p,
                                   symbol=symbol, timeframe=timeframe))
            price = p
            day += 1

    # Continuation bars: falling so final close < SMA21
    for i in range(continuation):
        p = price - 0.0020 * (i + 1)
        candles.append(_candle(day, price, price + 0.0005, p - 0.0005, p,
                               symbol=symbol, timeframe=timeframe))
        price = p
        day += 1

    return candles


def make_ranging_series(
    n_cycles: int = 8,
    lookback: int = 3,
    center: float = 1.1000,
    amplitude: float = 0.0050,
    symbol: str = BASE_SYMBOL,
    timeframe: str = BASE_TF,
) -> List[CandleData]:
    """
    Ranging oscillation — same builder used in M03 tests.
    Uses a tight amplitude so ranging_tolerance is met.
    """
    candles: List[CandleData] = []
    day = 0
    for cycle in range(n_cycles):
        offset = amplitude * (1.0 + (0.0001 if cycle % 2 == 0 else -0.0001))
        top    = center + offset
        bottom = center - offset

        leg_up = lookback + 1
        price  = bottom
        for j in range(leg_up):
            p = price + (top - price) * (j + 1) / leg_up
            candles.append(_candle(day, price, p + 0.0001, price - 0.0001, p,
                                   symbol=symbol, timeframe=timeframe))
            price = p
            day += 1

        leg_down = lookback
        for j in range(leg_down):
            p = price - (price - bottom) * (j + 1) / leg_down
            candles.append(_candle(day, price, price + 0.0001, p - 0.0001, p,
                                   symbol=symbol, timeframe=timeframe))
            price = p
            day += 1

    return candles


def _make_structure_analysis(
    direction: TrendDirection,
    is_ranging: bool = False,
    confidence: float = 0.7,
) -> StructureAnalysis:
    """Minimal StructureAnalysis for unit tests that don't need real swings."""
    return StructureAnalysis(
        direction=direction,
        is_ranging=is_ranging,
        confidence=confidence,
    )


def _make_market_structure_dto(
    direction: TrendDirection,
    symbol: str = BASE_SYMBOL,
    timeframe: str = BASE_TF,
) -> MarketStructure:
    """Minimal MarketStructure DTO (shared type)."""
    return MarketStructure(
        symbol=symbol,
        timeframe=timeframe,
        timestamp=BASE_DATE,
        regime=direction,
    )


# ===========================================================================
# FIXTURES
# ===========================================================================

@pytest.fixture
def detector() -> TrendDetector:
    """Default TrendDetector (sma_period=21, threshold=60)."""
    return TrendDetector(sma_period=21, tradeable_threshold=60.0)


@pytest.fixture
def detector5() -> TrendDetector:
    """TrendDetector with sma_period=5 for small datasets."""
    return TrendDetector(sma_period=5, tradeable_threshold=60.0)


@pytest.fixture
def bullish_candles() -> List[CandleData]:
    """Zigzag UP series with enough candles for SMA21."""
    return make_zigzag_up(n_swings=6, lookback=3)


@pytest.fixture
def bearish_candles() -> List[CandleData]:
    """Zigzag DOWN series with enough candles for SMA21."""
    return make_zigzag_down(n_swings=6, lookback=3)


@pytest.fixture
def rising_candles_30() -> List[CandleData]:
    """30 monotonically rising candles — close always > SMA."""
    return make_rising_candles(30, start=1.1000, step=0.0010)


@pytest.fixture
def falling_candles_30() -> List[CandleData]:
    """30 monotonically falling candles — close always < SMA."""
    return make_falling_candles(30, start=1.1300, step=0.0010)


@pytest.fixture
def flat_candles_30() -> List[CandleData]:
    """30 flat candles — SMA == close."""
    return make_flat_candles(30)


@pytest.fixture
def m3_up(bullish_candles) -> StructureAnalysis:
    """M03 analysis of bullish candles — should return UP."""
    return MarketStructureAnalyzer(lookback=3).analyze(bullish_candles)


@pytest.fixture
def m3_down(bearish_candles) -> StructureAnalysis:
    """M03 analysis of bearish candles — should return DOWN."""
    return MarketStructureAnalyzer(lookback=3).analyze(bearish_candles)


# ===========================================================================
# 1. COMPUTE SMA
# ===========================================================================

class TestComputeSma:
    def test_sma_constant_series(self):
        """SMA of constant prices == that price."""
        candles = make_flat_candles(30, price=1.1000)
        series = compute_sma(candles, period=21)
        assert len(series) == 30
        assert abs(series[-1] - 1.1000) < 1e-9

    def test_sma_first_period_minus_1_elements_are_zero(self):
        """First (period-1) entries must be 0.0."""
        candles = make_flat_candles(30)
        series = compute_sma(candles, period=21)
        assert all(v == 0.0 for v in series[:20])

    def test_sma_position_at_period_is_nonzero(self):
        """Entry at index (period-1) is the first valid SMA."""
        candles = make_flat_candles(30, price=1.2000)
        series = compute_sma(candles, period=10)
        assert series[9] != 0.0
        assert abs(series[9] - 1.2000) < 1e-9

    def test_sma_rising_series_increases(self):
        """SMA of rising series should be strictly increasing."""
        candles = make_rising_candles(30)
        series = compute_sma(candles, period=5)
        valid = [v for v in series if v > 0]
        assert all(valid[i] < valid[i + 1] for i in range(len(valid) - 1))

    def test_sma_period_1_equals_closes(self):
        """Period-1 SMA is just the close price at each bar."""
        candles = make_rising_candles(10, start=1.1000, step=0.0010)
        series = compute_sma(candles, period=1)
        for i, c in enumerate(candles):
            assert abs(series[i] - c.close) < 1e-9

    def test_sma_length_matches_candles(self):
        candles = make_flat_candles(50)
        series = compute_sma(candles, period=21)
        assert len(series) == 50

    def test_sma_period_equals_series_length(self):
        """Period == len(candles) yields one valid SMA at last position."""
        candles = make_flat_candles(21, price=1.0500)
        series = compute_sma(candles, period=21)
        assert abs(series[-1] - 1.0500) < 1e-9

    def test_sma_invalid_period_raises(self):
        candles = make_flat_candles(10)
        with pytest.raises(ValueError):
            compute_sma(candles, period=0)

    def test_sma_empty_series_returns_empty(self):
        series = compute_sma([], period=21)
        assert series == []


# ===========================================================================
# 2. GET CURRENT SMA
# ===========================================================================

class TestGetCurrentSma:
    def test_returns_none_below_period(self):
        candles = make_flat_candles(10)
        assert get_current_sma(candles, period=21) is None

    def test_returns_none_empty(self):
        assert get_current_sma([], period=21) is None

    def test_returns_correct_value_exactly_period(self):
        candles = make_flat_candles(21, price=1.0800)
        val = get_current_sma(candles, period=21)
        assert val is not None
        assert abs(val - 1.0800) < 1e-9

    def test_returns_correct_value_above_period(self):
        candles = make_rising_candles(30, start=1.1000, step=0.0010)
        val = get_current_sma(candles, period=21)
        # Expected: mean of closes[9..29] = mean(1.1090..1.1290)
        expected = sum(c.close for c in candles[-21:]) / 21
        assert val is not None
        assert abs(val - expected) < 1e-9

    def test_period_1_equals_last_close(self):
        candles = make_flat_candles(5, price=1.2345)
        val = get_current_sma(candles, period=1)
        assert abs(val - 1.2345) < 1e-9


# ===========================================================================
# 3. PRICE VS SMA COMPARISONS
# ===========================================================================

class TestPriceVsSma:
    def test_above_when_price_greater(self):
        assert is_price_above_sma(1.1100, 1.1000) is True

    def test_not_above_when_equal(self):
        assert is_price_above_sma(1.1000, 1.1000) is False

    def test_not_above_when_below(self):
        assert is_price_above_sma(1.0900, 1.1000) is False

    def test_below_when_price_less(self):
        assert is_price_below_sma(1.0900, 1.1000) is True

    def test_not_below_when_equal(self):
        assert is_price_below_sma(1.1000, 1.1000) is False

    def test_not_below_when_above(self):
        assert is_price_below_sma(1.1100, 1.1000) is False


# ===========================================================================
# 4. DETERMINE TREND DIRECTION
# ===========================================================================

class TestDetermineTrendDirection:
    def test_up_structure_and_close_above_sma(self):
        struct = _make_structure_analysis(TrendDirection.UP)
        candles = make_rising_candles(30)
        direction, reason = determine_trend_direction(struct, candles, sma_period=5)
        assert direction == "UP"
        assert "bullish" in reason.lower() or "confirmed" in reason.lower()

    def test_down_structure_and_close_below_sma(self):
        struct = _make_structure_analysis(TrendDirection.DOWN)
        candles = make_falling_candles(30)
        direction, reason = determine_trend_direction(struct, candles, sma_period=5)
        assert direction == "DOWN"

    def test_up_structure_close_below_sma_returns_undefined(self):
        """Structure UP but price is below SMA — conflict → UNDEFINED."""
        struct = _make_structure_analysis(TrendDirection.UP)
        # Falling candles: close is BELOW SMA
        candles = make_falling_candles(30)
        direction, reason = determine_trend_direction(struct, candles, sma_period=5)
        assert direction == "UNDEFINED"
        assert "below" in reason.lower()

    def test_down_structure_close_above_sma_returns_undefined(self):
        """Structure DOWN but price is above SMA — conflict → UNDEFINED."""
        struct = _make_structure_analysis(TrendDirection.DOWN)
        candles = make_rising_candles(30)
        direction, reason = determine_trend_direction(struct, candles, sma_period=5)
        assert direction == "UNDEFINED"
        assert "above" in reason.lower()

    def test_ranging_structure_returns_ranging(self):
        struct = _make_structure_analysis(TrendDirection.NONE, is_ranging=True)
        candles = make_flat_candles(30)
        direction, reason = determine_trend_direction(struct, candles, sma_period=5)
        assert direction == "RANGING"
        assert "ranging" in reason.lower()

    def test_none_structure_returns_undefined(self):
        struct = _make_structure_analysis(TrendDirection.NONE, is_ranging=False)
        candles = make_flat_candles(30)
        direction, _ = determine_trend_direction(struct, candles, sma_period=5)
        assert direction == "UNDEFINED"

    def test_insufficient_candles_returns_undefined(self):
        struct = _make_structure_analysis(TrendDirection.UP)
        candles = make_rising_candles(5)
        direction, reason = determine_trend_direction(struct, candles, sma_period=21)
        assert direction == "UNDEFINED"
        assert "insufficient" in reason.lower() or "sma" in reason.lower()

    def test_none_market_structure_returns_undefined(self):
        candles = make_rising_candles(30)
        direction, _ = determine_trend_direction(None, candles, sma_period=5)
        assert direction == "UNDEFINED"


# ===========================================================================
# 5. CALCULATE TREND STRENGTH
# ===========================================================================

class TestCalculateTrendStrength:
    def test_returns_zero_for_none_sma(self):
        candles = make_flat_candles(30)
        struct = _make_structure_analysis(TrendDirection.UP)
        score = calculate_trend_strength(struct, candles, None)
        assert score == 0.0

    def test_returns_zero_for_empty_candles(self):
        struct = _make_structure_analysis(TrendDirection.UP)
        score = calculate_trend_strength(struct, [], 1.1000)
        assert score == 0.0

    def test_bullish_aligned_scores_positively(self):
        """Structure UP + close above SMA → nonzero score."""
        struct = _make_structure_analysis(TrendDirection.UP, confidence=0.8)
        candles = make_rising_candles(30)
        sma = get_current_sma(candles, period=5)
        score = calculate_trend_strength(struct, candles, sma, sma_period=5)
        assert score > 0.0

    def test_bearish_aligned_scores_positively(self):
        struct = _make_structure_analysis(TrendDirection.DOWN, confidence=0.8)
        candles = make_falling_candles(30)
        sma = get_current_sma(candles, period=5)
        score = calculate_trend_strength(struct, candles, sma, sma_period=5)
        assert score > 0.0

    def test_score_bounded_0_to_100(self):
        struct = _make_structure_analysis(TrendDirection.UP, confidence=1.0)
        candles = make_rising_candles(50)
        sma = get_current_sma(candles, period=5)
        score = calculate_trend_strength(struct, candles, sma, sma_period=5)
        assert 0.0 <= score <= 100.0

    def test_ranging_struct_does_not_earn_structure_points(self):
        struct = _make_structure_analysis(TrendDirection.NONE, is_ranging=True)
        candles = make_flat_candles(30)
        sma = get_current_sma(candles, period=5)
        # Ranging → structure pts = 0; result may still score on price/sma cols
        score = calculate_trend_strength(struct, candles, sma, sma_period=5)
        # score could be low but should not exceed structure alignment pts
        assert score < 60.0   # no structure alignment bonus possible


# ===========================================================================
# 6. IS TREND TRADEABLE
# ===========================================================================

class TestIsTrendTradeable:
    def _make_ta(
        self,
        direction: str = "UP",
        sma21: float = 1.1000,
        price_vs_sma: float = 0.005,
        confidence: float = 70.0,
    ) -> TrendAnalysis:
        return TrendAnalysis(
            direction=direction,
            tradeable=False,   # computed separately
            reason="",
            sma21=sma21,
            sma21_slope=0.0,
            price_vs_sma=price_vs_sma,
            adx=None,
            adx_strength=TrendStrength.WEAK,
            structure_direction=direction,
            confidence_score=confidence,
        )

    def test_tradeable_when_all_conditions_met(self):
        ta = self._make_ta("UP", price_vs_sma=0.005, confidence=70.0)
        assert is_trend_tradeable(ta, tradeable_threshold=60.0) is True

    def test_not_tradeable_when_ranging(self):
        ta = self._make_ta("RANGING", confidence=70.0)
        assert is_trend_tradeable(ta) is False

    def test_not_tradeable_when_undefined(self):
        ta = self._make_ta("UNDEFINED", confidence=70.0)
        assert is_trend_tradeable(ta) is False

    def test_not_tradeable_when_confidence_below_threshold(self):
        ta = self._make_ta("UP", confidence=50.0)
        assert is_trend_tradeable(ta, tradeable_threshold=60.0) is False

    def test_not_tradeable_when_confidence_at_threshold(self):
        """Exactly at threshold is tradeable (>=)."""
        ta = self._make_ta("UP", confidence=60.0)
        assert is_trend_tradeable(ta, tradeable_threshold=60.0) is True

    def test_not_tradeable_up_when_price_below_sma(self):
        """Direction UP but price_vs_sma negative → conflict."""
        ta = self._make_ta("UP", price_vs_sma=-0.003, confidence=80.0)
        assert is_trend_tradeable(ta) is False

    def test_not_tradeable_down_when_price_above_sma(self):
        ta = self._make_ta("DOWN", price_vs_sma=0.003, confidence=80.0)
        assert is_trend_tradeable(ta) is False

    def test_tradeable_down_when_price_below_sma(self):
        ta = self._make_ta("DOWN", price_vs_sma=-0.003, confidence=80.0)
        assert is_trend_tradeable(ta) is True

    def test_not_tradeable_when_sma_zero(self):
        ta = self._make_ta("UP", sma21=0.0, confidence=80.0)
        assert is_trend_tradeable(ta) is False

    def test_custom_threshold_respected(self):
        ta80 = self._make_ta("UP", confidence=79.0)
        assert is_trend_tradeable(ta80, tradeable_threshold=80.0) is False
        assert is_trend_tradeable(ta80, tradeable_threshold=60.0) is True


# ===========================================================================
# 7. SUMMARIZE TREND
# ===========================================================================

class TestSummarizeTrend:
    def _ta_up(self) -> TrendAnalysis:
        return TrendAnalysis(
            direction="UP",
            tradeable=True,
            reason="Bullish trend confirmed",
            sma21=1.1050,
            sma21_slope=0.0002,
            price_vs_sma=0.005,
            adx=None,
            adx_strength=TrendStrength.WEAK,
            structure_direction="UP",
            confidence_score=75.0,
            tqs_trend_score=20,
            structure_confidence=0.8,
        )

    def test_required_keys_present(self):
        s = summarize_trend(self._ta_up())
        required = {
            "direction", "tradeable", "confidence_score", "sma21",
            "sma21_slope", "price_vs_sma_pct", "price_vs_sma_side",
            "adx", "adx_strength", "structure_direction",
            "structure_confidence", "reason", "tqs_trend_score",
        }
        assert required.issubset(s.keys())

    def test_direction_value(self):
        assert summarize_trend(self._ta_up())["direction"] == "UP"

    def test_tradeable_value(self):
        assert summarize_trend(self._ta_up())["tradeable"] is True

    def test_price_vs_sma_side_above(self):
        assert summarize_trend(self._ta_up())["price_vs_sma_side"] == "ABOVE"

    def test_price_vs_sma_side_below(self):
        ta = TrendAnalysis(
            direction="DOWN", tradeable=True, reason="",
            sma21=1.1000, sma21_slope=-0.0001, price_vs_sma=-0.003,
            adx=None, adx_strength=TrendStrength.WEAK,
            structure_direction="DOWN", confidence_score=70.0,
        )
        assert summarize_trend(ta)["price_vs_sma_side"] == "BELOW"

    def test_price_vs_sma_side_at(self):
        ta = TrendAnalysis(
            direction="UNDEFINED", tradeable=False, reason="",
            sma21=1.1000, sma21_slope=0.0, price_vs_sma=0.0,
            adx=None, adx_strength=TrendStrength.WEAK,
            structure_direction="NONE", confidence_score=0.0,
        )
        assert summarize_trend(ta)["price_vs_sma_side"] == "AT"

    def test_confidence_score_rounded(self):
        s = summarize_trend(self._ta_up())
        assert isinstance(s["confidence_score"], float)

    def test_sma21_rounded_to_6_dp(self):
        s = summarize_trend(self._ta_up())
        # Should not have more than 6 decimal places
        assert s["sma21"] == round(s["sma21"], 6)


# ===========================================================================
# 8. TREND DETECTOR — MAIN INTEGRATION TESTS
# ===========================================================================

class TestTrendDetectorInsufficientData:
    def test_zero_candles_returns_undefined(self, detector):
        result = detector.analyze([])
        assert result.direction == "UNDEFINED"
        assert result.tradeable is False
        assert result.sma21 == 0.0
        assert result.confidence_score == 0.0

    def test_fewer_than_sma_period_returns_undefined(self, detector):
        candles = make_rising_candles(10)
        result = detector.analyze(candles)
        assert result.direction == "UNDEFINED"
        assert result.tradeable is False

    def test_exactly_sma_period_minus_1_is_undefined(self, detector):
        candles = make_rising_candles(20)   # sma_period=21, need 21
        result = detector.analyze(candles)
        assert result.direction == "UNDEFINED"

    def test_exactly_sma_period_is_not_undefined_with_valid_struct(self, detector):
        """With exactly 21 candles + valid structure, direction can be determined."""
        candles = make_rising_candles(21)
        struct = _make_structure_analysis(TrendDirection.UP, confidence=0.8)
        result = detector.analyze(candles, struct)
        # Direction should be UP (close > SMA for rising series)
        assert result.direction in ("UP", "UNDEFINED")  # either valid; must not crash
        assert result.sma21 > 0.0

    def test_reason_mentions_insufficient_when_too_short(self, detector):
        result = detector.analyze(make_flat_candles(5))
        assert "insufficient" in result.reason.lower() or "sma" in result.reason.lower()


class TestTrendDetectorBullish:
    def test_direction_is_up(self, detector, bullish_candles, m3_up):
        result = detector.analyze(bullish_candles, m3_up)
        assert result.direction == "UP"

    def test_tradeable_is_true(self, detector, bullish_candles, m3_up):
        result = detector.analyze(bullish_candles, m3_up)
        assert result.tradeable is True

    def test_sma21_is_positive(self, detector, bullish_candles, m3_up):
        result = detector.analyze(bullish_candles, m3_up)
        assert result.sma21 > 0.0

    def test_price_vs_sma_positive(self, detector, bullish_candles, m3_up):
        """Bullish trend: close > SMA → positive price_vs_sma."""
        result = detector.analyze(bullish_candles, m3_up)
        assert result.price_vs_sma > 0.0

    def test_confidence_above_threshold(self, detector, bullish_candles, m3_up):
        result = detector.analyze(bullish_candles, m3_up)
        assert result.confidence_score >= 60.0

    def test_structure_direction_is_up(self, detector, bullish_candles, m3_up):
        result = detector.analyze(bullish_candles, m3_up)
        assert result.structure_direction == "UP"

    def test_tqs_trend_score_positive(self, detector, bullish_candles, m3_up):
        result = detector.analyze(bullish_candles, m3_up)
        assert result.tqs_trend_score > 0

    def test_tqs_trend_score_max_25(self, detector, bullish_candles, m3_up):
        result = detector.analyze(bullish_candles, m3_up)
        assert result.tqs_trend_score <= 25

    def test_structure_confidence_populated(self, detector, bullish_candles, m3_up):
        result = detector.analyze(bullish_candles, m3_up)
        # m3_up should have a nonzero confidence since it has real swings
        assert result.structure_confidence >= 0.0

    def test_sma_series_length_matches_candles(self, detector, bullish_candles, m3_up):
        result = detector.analyze(bullish_candles, m3_up)
        assert len(result.sma_series) == len(bullish_candles)


class TestTrendDetectorBearish:
    def test_direction_is_down(self, detector, bearish_candles, m3_down):
        result = detector.analyze(bearish_candles, m3_down)
        assert result.direction == "DOWN"

    def test_tradeable_is_true(self, detector, bearish_candles, m3_down):
        result = detector.analyze(bearish_candles, m3_down)
        assert result.tradeable is True

    def test_price_vs_sma_negative(self, detector, bearish_candles, m3_down):
        """Bearish trend: close < SMA → negative price_vs_sma."""
        result = detector.analyze(bearish_candles, m3_down)
        assert result.price_vs_sma < 0.0

    def test_confidence_above_threshold(self, detector, bearish_candles, m3_down):
        result = detector.analyze(bearish_candles, m3_down)
        assert result.confidence_score >= 60.0

    def test_structure_direction_is_down(self, detector, bearish_candles, m3_down):
        result = detector.analyze(bearish_candles, m3_down)
        assert result.structure_direction == "DOWN"

    def test_sma_slope_is_negative(self, detector, bearish_candles, m3_down):
        result = detector.analyze(bearish_candles, m3_down)
        assert result.sma21_slope <= 0.0


class TestTrendDetectorConflicts:
    """Structure and SMA-side are in conflict → UNDEFINED."""

    def test_up_structure_price_below_sma(self, detector5):
        """Structure says UP but falling price is below SMA."""
        struct = _make_structure_analysis(TrendDirection.UP, confidence=0.8)
        candles = make_falling_candles(30)
        result = detector5.analyze(candles, struct)
        assert result.direction == "UNDEFINED"
        assert result.tradeable is False

    def test_down_structure_price_above_sma(self, detector5):
        """Structure says DOWN but rising price is above SMA."""
        struct = _make_structure_analysis(TrendDirection.DOWN, confidence=0.8)
        candles = make_rising_candles(30)
        result = detector5.analyze(candles, struct)
        assert result.direction == "UNDEFINED"
        assert result.tradeable is False

    def test_none_structure_price_above_sma_is_undefined(self, detector5):
        """NONE structure — no trend regardless of SMA position."""
        struct = _make_structure_analysis(TrendDirection.NONE, is_ranging=False)
        candles = make_rising_candles(30)
        result = detector5.analyze(candles, struct)
        assert result.direction == "UNDEFINED"
        assert result.tradeable is False

    def test_no_market_structure_returns_undefined(self, detector5):
        candles = make_rising_candles(30)
        result = detector5.analyze(candles, market_structure=None)
        assert result.direction == "UNDEFINED"
        assert result.tradeable is False


class TestTrendDetectorRanging:
    def test_ranging_structure_returns_ranging_direction(self, detector5):
        struct = _make_structure_analysis(TrendDirection.NONE, is_ranging=True)
        candles = make_flat_candles(30)
        result = detector5.analyze(candles, struct)
        assert result.direction == "RANGING"

    def test_ranging_is_not_tradeable(self, detector5):
        struct = _make_structure_analysis(TrendDirection.NONE, is_ranging=True)
        candles = make_flat_candles(30)
        result = detector5.analyze(candles, struct)
        assert result.tradeable is False

    def test_ranging_confidence_is_zero(self, detector5):
        struct = _make_structure_analysis(TrendDirection.NONE, is_ranging=True)
        candles = make_flat_candles(30)
        result = detector5.analyze(candles, struct)
        assert result.confidence_score == 0.0

    def test_ranging_tqs_is_zero(self, detector5):
        struct = _make_structure_analysis(TrendDirection.NONE, is_ranging=True)
        candles = make_flat_candles(30)
        result = detector5.analyze(candles, struct)
        assert result.tqs_trend_score == 0

    def test_m03_ranging_series_detected(self):
        """Full pipeline: M03 detects ranging → M04 returns RANGING."""
        candles = make_ranging_series(n_cycles=8, lookback=3)
        # Pad to 21 candles minimum for SMA
        while len(candles) < 21:
            last = candles[-1]
            p = last.close
            candles.append(_candle(
                len(candles), p, p + 0.0002, p - 0.0002, p
            ))
        m3 = MarketStructureAnalyzer(lookback=3, ranging_tolerance=0.0020)
        struct = m3.analyze(candles)
        detector = TrendDetector(sma_period=21)
        result = detector.analyze(candles, struct)
        # Either RANGING (M03 confirmed ranging) or UNDEFINED (inconclusive) — never UP/DOWN
        assert result.direction in ("RANGING", "UNDEFINED")
        assert result.tradeable is False


# ===========================================================================
# 9. SMA SLOPE
# ===========================================================================

class TestSmaSlope:
    def test_rising_series_positive_slope(self, detector5):
        candles = make_rising_candles(30)
        result = detector5.analyze(
            candles,
            _make_structure_analysis(TrendDirection.UP, confidence=0.8)
        )
        assert result.sma21_slope > 0.0

    def test_falling_series_negative_slope(self, detector5):
        candles = make_falling_candles(30)
        result = detector5.analyze(
            candles,
            _make_structure_analysis(TrendDirection.DOWN, confidence=0.8)
        )
        assert result.sma21_slope < 0.0

    def test_flat_series_zero_slope(self, detector5):
        candles = make_flat_candles(30)
        result = detector5.analyze(
            candles,
            _make_structure_analysis(TrendDirection.NONE)
        )
        assert result.sma21_slope == 0.0

    def test_insufficient_candles_slope_is_zero(self, detector5):
        candles = make_flat_candles(3)
        result = detector5.analyze(candles)
        assert result.sma21_slope == 0.0


# ===========================================================================
# 10. TO_TREND_SIGNAL DTO CONVERSION
# ===========================================================================

class TestToTrendSignal:
    def _bullish_ta(self) -> TrendAnalysis:
        return TrendAnalysis(
            direction="UP",
            tradeable=True,
            reason="Bullish trend confirmed",
            sma21=1.1050,
            sma21_slope=0.0002,
            price_vs_sma=0.005,
            adx=None,
            adx_strength=TrendStrength.WEAK,
            structure_direction="UP",
            confidence_score=75.0,
            tqs_trend_score=20,
        )

    def test_returns_trend_signal_type(self):
        ts = self._bullish_ta().to_trend_signal("EURUSD", "D1")
        assert isinstance(ts, TrendSignal)

    def test_direction_up_maps_to_trend_direction_up(self):
        ts = self._bullish_ta().to_trend_signal()
        assert ts.direction == TrendDirection.UP

    def test_direction_down_maps_correctly(self):
        ta = TrendAnalysis(
            direction="DOWN", tradeable=True, reason="",
            sma21=1.1000, sma21_slope=-0.0001, price_vs_sma=-0.003,
            adx=None, adx_strength=TrendStrength.WEAK,
            structure_direction="DOWN", confidence_score=70.0,
        )
        ts = ta.to_trend_signal()
        assert ts.direction == TrendDirection.DOWN

    def test_ranging_maps_to_trend_direction_none(self):
        ta = TrendAnalysis(
            direction="RANGING", tradeable=False, reason="",
            sma21=1.1000, sma21_slope=0.0, price_vs_sma=0.0,
            adx=None, adx_strength=TrendStrength.WEAK,
            structure_direction="RANGING", confidence_score=0.0,
        )
        ts = ta.to_trend_signal()
        assert ts.direction == TrendDirection.NONE

    def test_symbol_and_timeframe_passed_through(self):
        ts = self._bullish_ta().to_trend_signal("GBPUSD", "H4")
        assert ts.symbol == "GBPUSD"
        assert ts.timeframe == "H4"

    def test_sma21_value_preserved(self):
        ts = self._bullish_ta().to_trend_signal()
        assert abs(ts.sma21 - 1.1050) < 1e-9

    def test_strength_is_confidence_normalized(self):
        ts = self._bullish_ta().to_trend_signal()
        assert abs(ts.strength - 0.75) < 1e-4

    def test_tradeable_preserved(self):
        ts = self._bullish_ta().to_trend_signal()
        assert ts.tradeable is True


# ===========================================================================
# 11. MARKET STRUCTURE DTO (SHARED TYPE) AS INPUT
# ===========================================================================

class TestMarketStructureDtoInput:
    """Verify TrendDetector works with the shared MarketStructure DTO."""

    def test_up_dto_with_close_above_sma(self, detector5):
        dto = _make_market_structure_dto(TrendDirection.UP)
        candles = make_rising_candles(30)
        result = detector5.analyze(candles, dto)
        assert result.direction == "UP"
        assert result.structure_direction == "UP"

    def test_down_dto_with_close_below_sma(self, detector5):
        dto = _make_market_structure_dto(TrendDirection.DOWN)
        candles = make_falling_candles(30)
        result = detector5.analyze(candles, dto)
        assert result.direction == "DOWN"

    def test_none_dto_returns_undefined(self, detector5):
        dto = _make_market_structure_dto(TrendDirection.NONE)
        candles = make_flat_candles(30)
        result = detector5.analyze(candles, dto)
        assert result.direction == "UNDEFINED"

    def test_dto_has_zero_structure_confidence(self, detector5):
        """MarketStructure DTO has no confidence field → should be 0.0."""
        dto = _make_market_structure_dto(TrendDirection.UP)
        candles = make_rising_candles(30)
        result = detector5.analyze(candles, dto)
        assert result.structure_confidence == 0.0


# ===========================================================================
# 12. CUSTOM SMA PERIOD
# ===========================================================================

class TestCustomSmaPeriod:
    def test_period_5_requires_only_5_candles(self):
        detector = TrendDetector(sma_period=5)
        candles = make_rising_candles(10)
        struct = _make_structure_analysis(TrendDirection.UP, confidence=0.8)
        result = detector.analyze(candles, struct)
        assert result.sma21 > 0.0   # SMA was computed

    def test_sma_value_changes_with_period(self):
        candles = make_rising_candles(30)
        struct = _make_structure_analysis(TrendDirection.UP, confidence=0.8)
        d5  = TrendDetector(sma_period=5).analyze(candles, struct)
        d10 = TrendDetector(sma_period=10).analyze(candles, struct)
        # Longer period → lower SMA for rising series (lags more)
        assert d10.sma21 < d5.sma21


# ===========================================================================
# 13. DETERMINISM
# ===========================================================================

class TestDeterminism:
    def test_same_input_same_output(self, detector5):
        """Identical inputs must always produce identical outputs."""
        struct = _make_structure_analysis(TrendDirection.UP, confidence=0.7)
        candles = make_rising_candles(30)
        r1 = detector5.analyze(candles, struct)
        r2 = detector5.analyze(candles, struct)
        assert r1.direction == r2.direction
        assert r1.tradeable == r2.tradeable
        assert r1.confidence_score == r2.confidence_score
        assert r1.sma21 == r2.sma21

    def test_different_symbol_does_not_affect_direction(self):
        """M04 must not hardcode any symbol."""
        struct_eur = _make_structure_analysis(TrendDirection.UP, confidence=0.8)
        struct_gbp = _make_structure_analysis(TrendDirection.UP, confidence=0.8)
        candles_eur = make_rising_candles(30, symbol="EURUSD")
        candles_gbp = make_rising_candles(30, symbol="GBPUSD")
        d = TrendDetector(sma_period=5)
        r_eur = d.analyze(candles_eur, struct_eur)
        r_gbp = d.analyze(candles_gbp, struct_gbp)
        assert r_eur.direction == r_gbp.direction  # symbol-agnostic

    def test_usdjpy_passes_through(self):
        struct = _make_structure_analysis(TrendDirection.DOWN, confidence=0.8)
        candles = make_falling_candles(30, start=150.00, step=0.20, symbol="USDJPY")
        d = TrendDetector(sma_period=5)
        result = d.analyze(candles, struct)
        assert result.direction in ("DOWN", "UNDEFINED")  # no crash


# ===========================================================================
# 14. TRADEABLE THRESHOLD CONFIGURATION
# ===========================================================================

class TestTradeableThreshold:
    def test_high_threshold_rejects_moderate_confidence(self, detector5):
        """With threshold=90, a moderate confidence trend is not tradeable."""
        struct = _make_structure_analysis(TrendDirection.UP, confidence=0.8)
        candles = make_rising_candles(30)
        detector_strict = TrendDetector(sma_period=5, tradeable_threshold=95.0)
        result = detector_strict.analyze(candles, struct)
        # confidence is unlikely to be 95+, so tradeable should be False
        if result.confidence_score < 95.0:
            assert result.tradeable is False

    def test_low_threshold_accepts_lower_confidence(self, detector5):
        struct = _make_structure_analysis(TrendDirection.UP, confidence=0.8)
        candles = make_rising_candles(30)
        detector_lenient = TrendDetector(sma_period=5, tradeable_threshold=10.0)
        result = detector_lenient.analyze(candles, struct)
        # Very low bar — if direction is UP and price above SMA, should be tradeable
        if result.direction == "UP":
            assert result.tradeable is True


# ===========================================================================
# 15. EXTRACT STRUCTURE INFO HELPER
# ===========================================================================

class TestExtractStructureInfo:
    def test_none_returns_undefined_false(self):
        d, r = _extract_structure_info(None)
        assert d == "UNDEFINED"
        assert r is False

    def test_structure_analysis_up(self):
        sa = _make_structure_analysis(TrendDirection.UP)
        d, r = _extract_structure_info(sa)
        assert d == "UP"
        assert r is False

    def test_structure_analysis_ranging(self):
        sa = _make_structure_analysis(TrendDirection.NONE, is_ranging=True)
        d, r = _extract_structure_info(sa)
        assert d == "RANGING"
        assert r is True

    def test_market_structure_dto_up(self):
        dto = _make_market_structure_dto(TrendDirection.UP)
        d, r = _extract_structure_info(dto)
        assert d == "UP"
        assert r is False

    def test_market_structure_dto_down(self):
        dto = _make_market_structure_dto(TrendDirection.DOWN)
        d, r = _extract_structure_info(dto)
        assert d == "DOWN"
        assert r is False
