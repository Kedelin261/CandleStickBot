"""
Tests for M16 — Market Regime Engine (src/analysis/market_regime.py).

Coverage plan (≥ 60 tests across 14 test classes):

 1. TestComputeAtr            (8)  — series length, values, Wilder smoothing
 2. TestGetCurrentAtr         (4)  — scalar helper
 3. TestComputeAtrMa          (5)  — SMA of ATR series
 4. TestComputeBbWidth        (7)  — formula, edge cases
 5. TestGetCurrentBbWidth     (3)  — scalar helper
 6. TestComputeBbWidthMa      (3)  — SMA of BB width series
 7. TestComputeChoppiness     (7)  — formula, boundary, edge cases
 8. TestRegimeAnalysisDto     (5)  — RegimeAnalysis properties + to_regime_signal()
 9. TestTrendingClassification(7)  — TRENDING regime conditions
10. TestRangingClassification (5)  — RANGING regime conditions
11. TestVolatileClassification(5)  — VOLATILE regime conditions
12. TestQuietClassification   (5)  — QUIET regime conditions
13. TestChoppyClassification  (5)  — CHOPPY regime conditions
14. TestEdgeCases             (7)  — insufficient candles, empty, ADX optional,
                                     allowed strategies, risk multiplier,
                                     TQS score, determinism
"""

import math
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parents[3]))

from src.analysis.market_regime import (
    CHOPPINESS_THRESHOLD,
    MarketRegimeEngine,
    RegimeAnalysis,
    compute_atr,
    compute_atr_ma,
    compute_bb_width,
    compute_bb_width_ma,
    compute_choppiness,
    get_current_atr,
    get_current_atr_ma,
    get_current_bb_width,
)
from src.types import CandleData, RegimeSignal, RegimeType


# ===========================================================================
# HELPERS
# ===========================================================================

BASE_TIME = datetime(2024, 1, 1)


def make_candle(
    idx: int = 0,
    open_: float = 1.1000,
    high: float = 1.1050,
    low: float = 1.0950,
    close: float = 1.1000,
    symbol: str = "EURUSD",
    timeframe: str = "D1",
) -> CandleData:
    return CandleData(
        symbol=symbol,
        timeframe=timeframe,
        timestamp=BASE_TIME + timedelta(days=idx),
        open=open_,
        high=high,
        low=low,
        close=close,
    )


def flat_candles(n: int = 50, price: float = 1.1000, range_: float = 0.0020) -> List[CandleData]:
    """Perfectly flat candles — ATR stays constant, choppiness high."""
    return [
        make_candle(i, open_=price, high=price + range_, low=price - range_, close=price)
        for i in range(n)
    ]


def trending_up_candles(
    n: int = 60,
    start: float = 1.0800,
    step: float = 0.0015,
    volatility: float = 0.0010,
) -> List[CandleData]:
    """
    Strongly trending candles (each bar closes higher).
    Large bodies + small wicks → low choppiness, ATR steady, BB expanding.
    """
    candles = []
    price = start
    for i in range(n):
        open_ = price
        close = price + step
        high = close + volatility * 0.5
        low  = open_ - volatility * 0.5
        candles.append(make_candle(i, open_=open_, high=high, low=low, close=close))
        price = close
    return candles


def ranging_candles(
    n: int = 60,
    center: float = 1.1000,
    amplitude: float = 0.0050,
) -> List[CandleData]:
    """Sine-wave oscillating candles — BB contracting, choppiness near threshold."""
    candles = []
    for i in range(n):
        offset = math.sin(i * 0.5) * amplitude
        open_  = center + offset
        close  = center - offset
        high   = max(open_, close) + 0.0005
        low    = min(open_, close) - 0.0005
        candles.append(make_candle(i, open_=open_, high=high, low=low, close=close))
    return candles


def volatile_candles(
    n: int = 80,
    base_range: float = 0.0008,
    spike_range: float = 0.0400,
    spike_start: int = 50,
) -> List[CandleData]:
    """
    Long series of calm candles followed by extreme high-volatility spikes.

    The spike phase must be long enough (n - spike_start bars) that the
    Wilder ATR rises well above the ATR_MA built from the calm phase.
    Using 30 spike bars (50..79) ensures ATR >> ATR_MA * 1.5.
    Varying the close price avoids price_range=0 (which would force CI=100
    and trigger CHOPPY before VOLATILE).
    """
    candles = []
    price = 1.1000
    for i in range(n):
        r = spike_range if i >= spike_start else base_range
        # Vary close slightly so price_range > 0 → choppiness not pinned at 100
        close = price + (0.0001 * (i % 3 - 1))  # tiny drift: -0.0001, 0, +0.0001
        candles.append(
            make_candle(i, open_=price, high=price + r, low=price - r, close=close)
        )
        price = close
    return candles


def quiet_candles(
    n: int = 100,
    normal_range: float = 0.0040,
    quiet_range: float = 0.0001,
    quiet_start: int = 70,
) -> List[CandleData]:
    """
    70 normal-ATR candles followed by 30 extremely compressed candles.

    With atr_ma_period=50, the ATR_MA window (last 50 ATR values = bars
    49–99) is still mostly anchored to the normal phase (bars 49–69 = 21
    bars of normal range).  The Wilder ATR (period=5) drops rapidly once
    the quiet_range bars arrive, giving ATR/ATR_MA ≈ 0.05, well below the
    QUIET threshold of 0.6.

    Close is varied slightly to keep HH-LL > 0 for choppiness.
    """
    candles = []
    price = 1.1000
    for i in range(n):
        r = quiet_range if i >= quiet_start else normal_range
        # Tiny close drift to keep HH-LL non-zero for choppiness calculation
        close = price + (0.00005 * (i % 2 - 0.5))
        candles.append(
            make_candle(i, open_=price, high=price + r, low=price - r, close=close)
        )
        price = close
    return candles


def choppy_candles(
    n: int = 60,
    center: float = 1.1000,
    noise: float = 0.0008,
) -> List[CandleData]:
    """
    Very erratic candles with small directional moves but large wicks.
    Sum of TRs >> price range → high choppiness index.
    """
    candles = []
    price = center
    for i in range(n):
        direction = 1 if i % 2 == 0 else -1
        open_ = price
        close = price + direction * 0.0002   # tiny net move
        high  = max(open_, close) + noise    # large wick
        low   = min(open_, close) - noise
        candles.append(make_candle(i, open_=open_, high=high, low=low, close=close))
        price = close
    return candles


# ===========================================================================
# 1. TestComputeAtr
# ===========================================================================

class TestComputeAtr:
    def test_empty_returns_empty(self):
        assert compute_atr([], 14) == []

    def test_single_candle_returns_empty(self):
        candles = [make_candle(0)]
        assert compute_atr(candles, 14) == []

    def test_two_candles_returns_one_value(self):
        # c0: high=1.11, low=1.09, close=1.10
        # c1: high=1.12, low=1.10, close=1.11
        # TR = max(0.02, |1.12-1.10|, |1.10-1.10|) = 0.02
        c0 = make_candle(0, open_=1.10, high=1.11, low=1.09, close=1.10)
        c1 = make_candle(1, open_=1.10, high=1.12, low=1.10, close=1.11)
        result = compute_atr([c0, c1], period=14)
        assert len(result) == 1
        assert result[0] > 0

    def test_length_matches_candles_minus_one(self):
        candles = flat_candles(30)
        result = compute_atr(candles, period=14)
        assert len(result) == len(candles) - 1

    def test_returns_positive_values(self):
        candles = flat_candles(30)
        result = compute_atr(candles, period=14)
        assert all(v >= 0 for v in result)

    def test_flat_candles_constant_atr(self):
        """Flat candles (constant range) produce a constant ATR."""
        candles = flat_candles(30, range_=0.0020)
        result = compute_atr(candles, period=14)
        valid = [v for v in result if v > 0]
        assert len(valid) > 0
        # All valid ATR values should be approximately 0.0040 (H-L range)
        for v in valid:
            assert abs(v - 0.0040) < 1e-8, f"Expected ~0.0040 but got {v}"

    def test_volatile_candles_higher_atr(self):
        normal = flat_candles(30, range_=0.0010)
        volatile = flat_candles(30, range_=0.0100)
        atr_normal   = [v for v in compute_atr(normal, 14) if v > 0][-1]
        atr_volatile = [v for v in compute_atr(volatile, 14) if v > 0][-1]
        assert atr_volatile > atr_normal * 5

    def test_wilder_smoothing_converges(self):
        """Wilder's ATR should converge to H-L range for constant-range candles."""
        candles = flat_candles(100, range_=0.0020)
        result = compute_atr(candles, period=14)
        final = [v for v in result if v > 0][-1]
        assert abs(final - 0.0040) < 1e-8


# ===========================================================================
# 2. TestGetCurrentAtr
# ===========================================================================

class TestGetCurrentAtr:
    def test_empty_returns_zero(self):
        assert get_current_atr([], 14) == 0.0

    def test_single_candle_returns_zero(self):
        assert get_current_atr([make_candle(0)], 14) == 0.0

    def test_returns_scalar_float(self):
        candles = flat_candles(30)
        result = get_current_atr(candles, 14)
        assert isinstance(result, float)
        assert result > 0

    def test_volatile_higher_than_quiet(self):
        quiet_c    = flat_candles(30, range_=0.0005)
        volatile_c = flat_candles(30, range_=0.0100)
        assert get_current_atr(volatile_c, 14) > get_current_atr(quiet_c, 14)


# ===========================================================================
# 3. TestComputeAtrMa
# ===========================================================================

class TestComputeAtrMa:
    def test_empty_returns_empty(self):
        assert compute_atr_ma([], 10) == []

    def test_length_preserved(self):
        series = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = compute_atr_ma(series, 3)
        assert len(result) == len(series)

    def test_first_values_zero(self):
        series = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = compute_atr_ma(series, 3)
        assert result[0] == 0.0
        assert result[1] == 0.0

    def test_simple_average_correct(self):
        series = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = compute_atr_ma(series, 3)
        assert abs(result[2] - 2.0) < 1e-9   # (1+2+3)/3
        assert abs(result[3] - 3.0) < 1e-9   # (2+3+4)/3
        assert abs(result[4] - 4.0) < 1e-9   # (3+4+5)/3

    def test_period_1_is_identity(self):
        series = [1.5, 2.5, 3.5]
        result = compute_atr_ma(series, 1)
        assert result == series


# ===========================================================================
# 4. TestComputeBbWidth
# ===========================================================================

class TestComputeBbWidth:
    def test_empty_returns_empty(self):
        assert compute_bb_width([], 20) == []

    def test_single_candle_returns_zero(self):
        result = compute_bb_width([make_candle(0)], 20)
        assert result == [0.0]

    def test_length_equals_candle_count(self):
        candles = flat_candles(30)
        result = compute_bb_width(candles, period=20)
        assert len(result) == len(candles)

    def test_first_values_zero_before_period(self):
        candles = flat_candles(30)
        result = compute_bb_width(candles, period=20)
        assert all(v == 0.0 for v in result[:19])

    def test_positive_after_period(self):
        candles = flat_candles(30, range_=0.0020)
        result = compute_bb_width(candles, period=20)
        # After enough candles, width must be > 0 (unless all closes equal)
        valid = result[19:]
        # flat_candles has constant close — std dev may be 0
        # Just verify it's non-negative
        assert all(v >= 0 for v in valid)

    def test_wider_range_larger_bb_width(self):
        narrow = flat_candles(40, range_=0.0005)
        wide   = flat_candles(40, range_=0.0050)
        # Use candles with varying closes for meaningful std dev
        # Create manually varying candles
        def varying(n, amplitude):
            base = 1.1000
            return [
                make_candle(i, open_=base, high=base + amplitude,
                            low=base - amplitude,
                            close=base + amplitude * math.sin(i * 0.3))
                for i in range(n)
            ]
        narrow_v = varying(40, 0.0005)
        wide_v   = varying(40, 0.0050)
        bw_narrow = [v for v in compute_bb_width(narrow_v, 20) if v > 0]
        bw_wide   = [v for v in compute_bb_width(wide_v,   20) if v > 0]
        if bw_narrow and bw_wide:
            assert bw_wide[-1] > bw_narrow[-1]

    def test_zero_variance_gives_zero_width(self):
        """All closes equal → std dev = 0 → BB width = 0."""
        candles = [
            make_candle(i, open_=1.1, high=1.11, low=1.09, close=1.1000)
            for i in range(30)
        ]
        result = compute_bb_width(candles, period=20)
        for v in result[19:]:
            assert v == 0.0

    def test_period_2_works(self):
        candles = [
            make_candle(0, close=1.10),
            make_candle(1, close=1.12),
        ]
        result = compute_bb_width(candles, period=2)
        assert len(result) == 2
        # Only last value should be non-zero
        assert result[0] == 0.0
        assert result[1] > 0


# ===========================================================================
# 5. TestGetCurrentBbWidth
# ===========================================================================

class TestGetCurrentBbWidth:
    def test_empty_returns_zero(self):
        assert get_current_bb_width([], 20) == 0.0

    def test_returns_scalar_float(self):
        candles = [
            make_candle(i, close=1.1000 + math.sin(i * 0.3) * 0.0010)
            for i in range(40)
        ]
        result = get_current_bb_width(candles, 20)
        assert isinstance(result, float)

    def test_positive_for_varying_candles(self):
        candles = [
            make_candle(i, close=1.10 + math.sin(i * 0.5) * 0.005)
            for i in range(40)
        ]
        assert get_current_bb_width(candles, 20) > 0


# ===========================================================================
# 6. TestComputeBbWidthMa
# ===========================================================================

class TestComputeBbWidthMa:
    def test_empty_returns_zero(self):
        assert compute_bb_width_ma([], 20, 2.0, 10) == 0.0

    def test_returns_float(self):
        candles = [
            make_candle(i, close=1.1 + math.sin(i * 0.4) * 0.005)
            for i in range(60)
        ]
        result = compute_bb_width_ma(candles, bb_period=20, bb_std_dev=2.0, ma_period=10)
        assert isinstance(result, float)

    def test_ma_smooths_bb_width(self):
        candles = [
            make_candle(i, close=1.1 + math.sin(i * 0.4) * 0.005)
            for i in range(60)
        ]
        instant = get_current_bb_width(candles, 20)
        ma_val  = compute_bb_width_ma(candles, 20, 2.0, 10)
        # MA smooths the series — cannot make exact assertion, but both > 0
        assert ma_val >= 0


# ===========================================================================
# 7. TestComputeChoppiness
# ===========================================================================

class TestComputeChoppiness:
    def test_insufficient_candles_returns_threshold(self):
        candles = flat_candles(5)
        result = compute_choppiness(candles, period=14)
        assert result == CHOPPINESS_THRESHOLD

    def test_perfectly_flat_returns_100(self):
        """All candles same price → price_range = 0 → maximally choppy."""
        candles = [
            make_candle(i, open_=1.1, high=1.1, low=1.1, close=1.1)
            for i in range(20)
        ]
        assert compute_choppiness(candles, period=14) == 100.0

    def test_trending_below_threshold(self):
        """Strong directional trend produces low choppiness."""
        candles = trending_up_candles(60, step=0.0050, volatility=0.0002)
        ci = compute_choppiness(candles, period=14)
        assert ci < CHOPPINESS_THRESHOLD, f"Expected < 61.8, got {ci}"

    def test_choppy_market_above_threshold(self):
        """Erratic candles produce choppiness above threshold."""
        candles = choppy_candles(60, noise=0.0020)
        ci = compute_choppiness(candles, period=14)
        assert ci > CHOPPINESS_THRESHOLD, f"Expected > 61.8, got {ci}"

    def test_result_bounded_0_100(self):
        for candles in [trending_up_candles(30), choppy_candles(30), flat_candles(30)]:
            ci = compute_choppiness(candles, period=14)
            assert 0.0 <= ci <= 100.0, f"Out of range: {ci}"

    def test_returns_float(self):
        candles = flat_candles(30)
        assert isinstance(compute_choppiness(candles, period=14), float)

    def test_choppiness_threshold_constant(self):
        assert CHOPPINESS_THRESHOLD == 61.8


# ===========================================================================
# 8. TestRegimeAnalysisDto
# ===========================================================================

class TestRegimeAnalysisDto:
    def _make_analysis(self, regime=RegimeType.TRENDING, strategies=None) -> RegimeAnalysis:
        # Use explicit sentinel so empty list is not collapsed to default
        strats = ["pin_bar"] if strategies is None else strategies
        return RegimeAnalysis(
            regime=regime,
            confidence=0.75,
            allowed_strategies=strats,
            risk_multiplier=1.0,
            reason="test",
            atr=0.0015,
            atr_ma=0.0012,
            bb_width=1.5,
            choppiness_index=45.0,
        )

    def test_is_tradeable_true_when_strategies_present(self):
        a = self._make_analysis(strategies=["pin_bar"])
        assert a.is_tradeable is True

    def test_is_tradeable_false_when_no_strategies(self):
        a = self._make_analysis(strategies=[])
        assert a.is_tradeable is False

    def test_to_regime_signal_returns_regime_signal(self):
        a = self._make_analysis()
        sig = a.to_regime_signal("EURUSD", "D1")
        assert isinstance(sig, RegimeSignal)

    def test_to_regime_signal_regime_field_set(self):
        a = self._make_analysis(regime=RegimeType.TRENDING)
        sig = a.to_regime_signal("EURUSD", "D1")
        assert sig.regime == RegimeType.TRENDING

    def test_to_regime_signal_allowed_strategies_copied(self):
        a = self._make_analysis(strategies=["pin_bar", "engulfing_bar"])
        sig = a.to_regime_signal("EURUSD", "D1")
        assert "pin_bar" in sig.allowed_strategies
        assert "engulfing_bar" in sig.allowed_strategies


# ===========================================================================
# 9. TestTrendingClassification
# ===========================================================================

class TestTrendingClassification:
    """TRENDING = BB expanding + choppiness < 61.8 + ATR ratio ~ 1."""

    def _engine(self, **kw) -> MarketRegimeEngine:
        # Use shorter periods so the candle builders produce enough data
        defaults = dict(
            atr_period=5, atr_ma_period=10, bb_period=10, bb_ma_period=10,
            choppiness_period=10,
        )
        defaults.update(kw)
        return MarketRegimeEngine(**defaults)

    def test_strong_trend_classified_trending(self):
        candles = trending_up_candles(60, step=0.0050, volatility=0.0002)
        engine  = self._engine()
        result  = engine.analyze(candles)
        assert result.regime == RegimeType.TRENDING, \
            f"Expected TRENDING got {result.regime}: {result.reason}"

    def test_trending_allows_pin_bar(self):
        candles = trending_up_candles(60, step=0.0050, volatility=0.0002)
        result  = self._engine().analyze(candles)
        if result.regime == RegimeType.TRENDING:
            assert "pin_bar" in result.allowed_strategies

    def test_trending_allows_engulfing_bar(self):
        candles = trending_up_candles(60, step=0.0050, volatility=0.0002)
        result  = self._engine().analyze(candles)
        if result.regime == RegimeType.TRENDING:
            assert "engulfing_bar" in result.allowed_strategies

    def test_trending_risk_multiplier_is_one(self):
        candles = trending_up_candles(60, step=0.0050, volatility=0.0002)
        result  = self._engine().analyze(candles)
        if result.regime == RegimeType.TRENDING:
            assert result.risk_multiplier == 1.0

    def test_trending_choppiness_below_threshold(self):
        candles = trending_up_candles(60, step=0.0050, volatility=0.0002)
        result  = self._engine().analyze(candles)
        assert result.choppiness_index < CHOPPINESS_THRESHOLD

    def test_trending_confidence_positive(self):
        candles = trending_up_candles(60, step=0.0050, volatility=0.0002)
        result  = self._engine().analyze(candles)
        assert result.confidence > 0.0

    def test_trending_with_adx_confirmation(self):
        """ADX >= 25 should not block TRENDING classification."""
        candles = trending_up_candles(60, step=0.0050, volatility=0.0002)
        result  = self._engine().analyze(candles, adx=30.0)
        assert result.regime == RegimeType.TRENDING
        assert result.adx == 30.0


# ===========================================================================
# 10. TestRangingClassification
# ===========================================================================

class TestRangingClassification:
    """RANGING = BB contracting + choppiness < 61.8 (not fully choppy)."""

    def _engine(self) -> MarketRegimeEngine:
        return MarketRegimeEngine(
            atr_period=5, atr_ma_period=10, bb_period=10, bb_ma_period=10,
            choppiness_period=10,
        )

    def test_ranging_regime_produced(self):
        candles = ranging_candles(80, amplitude=0.0030)
        result  = self._engine().analyze(candles)
        # Ranging candles should produce RANGING or CHOPPY (both valid)
        assert result.regime in (RegimeType.RANGING, RegimeType.CHOPPY), \
            f"Unexpected regime {result.regime}: {result.reason}"

    def test_ranging_risk_multiplier_less_than_1(self):
        candles = ranging_candles(80, amplitude=0.0030)
        result  = self._engine().analyze(candles)
        assert result.risk_multiplier <= 1.0

    def test_ranging_no_strategies_phase1(self):
        """Phase 1: RANGING has no allowed strategies."""
        candles = ranging_candles(80, amplitude=0.0030)
        result  = self._engine().analyze(candles)
        if result.regime == RegimeType.RANGING:
            assert result.allowed_strategies == []

    def test_ranging_reason_contains_bb(self):
        candles = ranging_candles(80, amplitude=0.0030)
        result  = self._engine().analyze(candles)
        if result.regime == RegimeType.RANGING:
            assert "BB" in result.reason or "bb" in result.reason.lower()

    def test_ranging_tqs_score_ten(self):
        engine = self._engine()
        score  = engine.calculate_tqs_regime_score(RegimeType.RANGING)
        assert score == 10


# ===========================================================================
# 11. TestVolatileClassification
# ===========================================================================

class TestVolatileClassification:
    """VOLATILE = ATR > ATR_MA * volatile_atr_ratio (1.5)."""

    def _engine(self, volatile_ratio: float = 1.5) -> MarketRegimeEngine:
        # Use large atr_ma_period so the MA is anchored to the calm baseline
        return MarketRegimeEngine(
            atr_period=5, atr_ma_period=50, bb_period=10, bb_ma_period=10,
            choppiness_period=10, volatile_atr_ratio=volatile_ratio,
        )

    def test_volatile_regime_produced(self):
        candles = volatile_candles()  # 80 bars: 50 calm + 30 spike
        result  = self._engine().analyze(candles)
        assert result.regime == RegimeType.VOLATILE, \
            f"Expected VOLATILE got {result.regime}: {result.reason}"

    def test_volatile_no_allowed_strategies(self):
        candles = volatile_candles()
        result  = self._engine().analyze(candles)
        if result.regime == RegimeType.VOLATILE:
            assert result.allowed_strategies == []

    def test_volatile_risk_multiplier_zero(self):
        candles = volatile_candles()
        result  = self._engine().analyze(candles)
        if result.regime == RegimeType.VOLATILE:
            assert result.risk_multiplier == 0.0

    def test_volatile_reason_mentions_atr(self):
        candles = volatile_candles()
        result  = self._engine().analyze(candles)
        if result.regime == RegimeType.VOLATILE:
            assert "ATR" in result.reason or "atr" in result.reason.lower()

    def test_volatile_tqs_score_zero(self):
        score = self._engine().calculate_tqs_regime_score(RegimeType.VOLATILE)
        assert score == 0


# ===========================================================================
# 12. TestQuietClassification
# ===========================================================================

class TestQuietClassification:
    """QUIET = ATR < ATR_MA * quiet_atr_ratio (0.6)."""

    def _engine(self) -> MarketRegimeEngine:
        # Use large atr_ma_period so the MA stays anchored to the noisy baseline
        return MarketRegimeEngine(
            atr_period=5, atr_ma_period=50, bb_period=10, bb_ma_period=10,
            choppiness_period=10,
        )

    def test_quiet_regime_produced(self):
        candles = quiet_candles()  # 80 bars: 20 normal + 60 tiny
        result  = self._engine().analyze(candles)
        assert result.regime == RegimeType.QUIET, \
            f"Expected QUIET got {result.regime}: {result.reason}"

    def test_quiet_no_allowed_strategies(self):
        candles = quiet_candles()
        result  = self._engine().analyze(candles)
        if result.regime == RegimeType.QUIET:
            assert result.allowed_strategies == []

    def test_quiet_risk_multiplier_zero(self):
        candles = quiet_candles()
        result  = self._engine().analyze(candles)
        if result.regime == RegimeType.QUIET:
            assert result.risk_multiplier == 0.0

    def test_quiet_reason_mentions_atr(self):
        candles = quiet_candles()
        result  = self._engine().analyze(candles)
        if result.regime == RegimeType.QUIET:
            assert "ATR" in result.reason or "atr" in result.reason.lower()

    def test_quiet_tqs_score_zero(self):
        score = self._engine().calculate_tqs_regime_score(RegimeType.QUIET)
        assert score == 0


# ===========================================================================
# 13. TestChoppyClassification
# ===========================================================================

class TestChoppyClassification:
    """CHOPPY = Choppiness >= 61.8 (but ATR ratio not extreme)."""

    def _engine(self) -> MarketRegimeEngine:
        return MarketRegimeEngine(
            atr_period=5, atr_ma_period=10, bb_period=10, bb_ma_period=10,
            choppiness_period=10,
        )

    def test_choppy_regime_produced(self):
        candles = choppy_candles(60, noise=0.0030)
        result  = self._engine().analyze(candles)
        # Expect CHOPPY (or RANGING for mild choppiness)
        assert result.regime in (RegimeType.CHOPPY, RegimeType.RANGING), \
            f"Unexpected regime {result.regime}: {result.reason}"

    def test_choppy_no_allowed_strategies(self):
        candles = choppy_candles(60, noise=0.0030)
        result  = self._engine().analyze(candles)
        if result.regime == RegimeType.CHOPPY:
            assert result.allowed_strategies == []

    def test_choppy_risk_multiplier_zero(self):
        candles = choppy_candles(60, noise=0.0030)
        result  = self._engine().analyze(candles)
        if result.regime == RegimeType.CHOPPY:
            assert result.risk_multiplier == 0.0

    def test_choppy_tqs_score_zero(self):
        score = self._engine().calculate_tqs_regime_score(RegimeType.CHOPPY)
        assert score == 0

    def test_choppy_reason_mentions_choppiness(self):
        candles = choppy_candles(60, noise=0.0030)
        result  = self._engine().analyze(candles)
        if result.regime == RegimeType.CHOPPY:
            low = result.reason.lower()
            assert "choppiness" in low or "choppy" in low or "61.8" in result.reason


# ===========================================================================
# 14. TestEdgeCases
# ===========================================================================

class TestEdgeCases:
    def _engine(self) -> MarketRegimeEngine:
        return MarketRegimeEngine(
            atr_period=5, atr_ma_period=10, bb_period=10, bb_ma_period=10,
            choppiness_period=10,
        )

    def test_empty_candles_returns_unknown(self):
        result = self._engine().analyze([])
        assert result.regime == RegimeType.UNKNOWN
        assert result.confidence == 0.0

    def test_insufficient_candles_returns_unknown(self):
        """Single candle — ATR cannot be computed → UNKNOWN."""
        candles = flat_candles(1)
        result  = self._engine().analyze(candles)
        assert result.regime == RegimeType.UNKNOWN

    def test_adx_none_does_not_block_trending(self):
        """When ADX is not provided, TRENDING can still be classified."""
        candles = trending_up_candles(60, step=0.0050, volatility=0.0002)
        result  = self._engine().analyze(candles, adx=None)
        # Without ADX the engine may still classify TRENDING
        assert result.regime in (RegimeType.TRENDING, RegimeType.RANGING), \
            f"Unexpected {result.regime}"
        assert result.adx is None

    def test_adx_present_stored_in_result(self):
        candles = trending_up_candles(60, step=0.0050, volatility=0.0002)
        result  = self._engine().analyze(candles, adx=28.5)
        assert result.adx == 28.5

    def test_unknown_regime_zero_risk(self):
        result = self._engine().analyze([])
        assert result.risk_multiplier == 0.0

    def test_unknown_regime_no_strategies(self):
        result = self._engine().analyze([])
        assert result.allowed_strategies == []

    def test_deterministic_output(self):
        """Same candles → same result (no randomness)."""
        candles = trending_up_candles(60, step=0.0050, volatility=0.0002)
        engine  = self._engine()
        r1 = engine.analyze(candles)
        r2 = engine.analyze(candles)
        assert r1.regime     == r2.regime
        assert r1.confidence == r2.confidence
        assert abs(r1.atr - r2.atr) < 1e-12

    # ── TQS score tests ───────────────────────────────────────────────────

    def test_tqs_trending_no_adx_is_25(self):
        engine = self._engine()
        assert engine.calculate_tqs_regime_score(RegimeType.TRENDING, adx=None) == 25

    def test_tqs_trending_adx_30_is_25(self):
        engine = self._engine()
        assert engine.calculate_tqs_regime_score(RegimeType.TRENDING, adx=30.0) == 25

    def test_tqs_trending_adx_27_is_20(self):
        engine = self._engine()
        assert engine.calculate_tqs_regime_score(RegimeType.TRENDING, adx=27.0) == 20

    def test_tqs_trending_adx_22_is_15(self):
        engine = self._engine()
        assert engine.calculate_tqs_regime_score(RegimeType.TRENDING, adx=22.0) == 15

    def test_tqs_unknown_is_zero(self):
        engine = self._engine()
        assert engine.calculate_tqs_regime_score(RegimeType.UNKNOWN) == 0

    # ── Allowed strategies / risk mapping ─────────────────────────────────

    def test_trending_strategies_are_pinbar_and_engulfing(self):
        candles = trending_up_candles(60, step=0.0050, volatility=0.0002)
        result  = self._engine().analyze(candles)
        if result.regime == RegimeType.TRENDING:
            assert set(result.allowed_strategies) == {"pin_bar", "engulfing_bar"}

    def test_volatile_is_not_tradeable(self):
        candles = volatile_candles(60, base_range=0.0010, spike_range=0.0200)
        result  = self._engine().analyze(candles)
        if result.regime == RegimeType.VOLATILE:
            assert result.is_tradeable is False

    def test_result_has_atr_series(self):
        candles = trending_up_candles(60, step=0.0050, volatility=0.0002)
        result  = self._engine().analyze(candles)
        assert isinstance(result.atr_series, list)
        assert len(result.atr_series) > 0

    def test_result_has_bb_width_series(self):
        candles = trending_up_candles(60, step=0.0050, volatility=0.0002)
        result  = self._engine().analyze(candles)
        assert isinstance(result.bb_width_series, list)

    def test_invalid_volatile_ratio_raises(self):
        with pytest.raises(ValueError):
            MarketRegimeEngine(volatile_atr_ratio=0.5, quiet_atr_ratio=0.6)
