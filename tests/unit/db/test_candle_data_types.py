"""
Tests for shared CandleData domain types and derived properties.
Ensures candle math is correct before any pattern detection.
"""

import pytest
from datetime import datetime, timezone

from src.data.types import CandleData, Direction, TrendDirection, RegimeType, TQSResult, TradeTier


class TestCandleDataProperties:
    """Test all derived candle properties used in pattern detection."""

    def make_candle(self, open_, high, low, close) -> CandleData:
        return CandleData(
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            open=open_, high=high, low=low, close=close,
            volume=1000.0, symbol="EURUSD", timeframe="D1"
        )

    def test_bullish_candle_is_bullish(self):
        c = self.make_candle(1.0990, 1.1050, 1.0980, 1.1030)
        assert c.is_bullish is True
        assert c.is_bearish is False

    def test_bearish_candle_is_bearish(self):
        c = self.make_candle(1.1030, 1.1050, 1.0980, 1.0990)
        assert c.is_bearish is True
        assert c.is_bullish is False

    def test_body_size_calculation(self):
        c = self.make_candle(1.1000, 1.1050, 1.0950, 1.1020)
        assert abs(c.body_size - 0.0020) < 1e-8

    def test_total_range_calculation(self):
        c = self.make_candle(1.1000, 1.1050, 1.0950, 1.1020)
        assert abs(c.total_range - 0.0100) < 1e-8

    def test_upper_wick_bullish(self):
        # Bullish: close=1.1020, high=1.1050 → upper_wick = 1.1050 - 1.1020 = 0.0030
        c = self.make_candle(1.1000, 1.1050, 1.0950, 1.1020)
        assert abs(c.upper_wick - 0.0030) < 1e-8

    def test_lower_wick_bullish(self):
        # Bullish: open=1.1000, low=1.0950 → lower_wick = 1.1000 - 1.0950 = 0.0050
        c = self.make_candle(1.1000, 1.1050, 1.0950, 1.1020)
        assert abs(c.lower_wick - 0.0050) < 1e-8

    def test_upper_wick_bearish(self):
        # Bearish: open=1.1020, high=1.1050 → upper_wick = 1.1050 - 1.1020 = 0.0030
        c = self.make_candle(1.1020, 1.1050, 1.0950, 1.1000)
        assert abs(c.upper_wick - 0.0030) < 1e-8

    def test_lower_wick_bearish(self):
        # Bearish: close=1.1000, low=1.0950 → lower_wick = 1.1000 - 1.0950 = 0.0050
        c = self.make_candle(1.1020, 1.1050, 1.0950, 1.1000)
        assert abs(c.lower_wick - 0.0050) < 1e-8

    def test_midpoint_calculation(self):
        c = self.make_candle(1.1000, 1.1100, 1.0900, 1.1050)
        assert abs(c.midpoint - 1.1000) < 1e-8  # (1.1100 + 1.0900) / 2

    def test_doji_detection(self):
        # Open == Close → perfect doji
        c = self.make_candle(1.1000, 1.1050, 1.0950, 1.1000)
        assert c.is_doji is True

    def test_non_doji(self):
        c = self.make_candle(1.1000, 1.1050, 1.0950, 1.1030)
        assert c.is_doji is False

    def test_candle_repr(self):
        c = self.make_candle(1.1000, 1.1050, 1.0950, 1.1020)
        repr_str = repr(c)
        assert "EURUSD" in repr_str
        assert "D1" in repr_str


class TestTQSResult:
    """Test Trade Quality Score computation (Section 2a)."""

    def test_premium_tier_score_80_plus(self):
        tqs = TQSResult(trend_score=25, level_score=20, pattern_score=20, regime_score=25)
        assert tqs.total == 90
        assert tqs.tier == TradeTier.PREMIUM

    def test_standard_tier_score_60_to_79(self):
        tqs = TQSResult(trend_score=18, level_score=15, pattern_score=15, regime_score=20)
        assert tqs.total == 68
        assert tqs.tier == TradeTier.STANDARD

    def test_reject_tier_below_60(self):
        tqs = TQSResult(trend_score=10, level_score=10, pattern_score=10, regime_score=20)
        assert tqs.total == 50
        assert tqs.tier == TradeTier.REJECT

    def test_auto_reject_when_regime_zero(self):
        """Regime score = 0 (VOLATILE/CHOPPY) → automatic REJECT regardless of other scores."""
        tqs = TQSResult(trend_score=25, level_score=25, pattern_score=25, regime_score=0)
        assert tqs.total == 75  # Would be STANDARD by score alone
        assert tqs.is_tradeable is False  # But not tradeable due to regime=0

    def test_standard_trade_is_tradeable(self):
        tqs = TQSResult(trend_score=18, level_score=15, pattern_score=15, regime_score=20)
        assert tqs.is_tradeable is True

    def test_reject_trade_not_tradeable(self):
        tqs = TQSResult(trend_score=5, level_score=5, pattern_score=5, regime_score=25)
        assert tqs.total == 40
        assert tqs.is_tradeable is False

    def test_boundary_score_60_is_standard(self):
        """Exactly 60 = STANDARD (boundary condition)."""
        tqs = TQSResult(trend_score=15, level_score=15, pattern_score=15, regime_score=15)
        assert tqs.total == 60
        assert tqs.tier == TradeTier.STANDARD

    def test_boundary_score_80_is_premium(self):
        """Exactly 80 = PREMIUM (boundary condition)."""
        tqs = TQSResult(trend_score=20, level_score=20, pattern_score=20, regime_score=20)
        assert tqs.total == 80
        assert tqs.tier == TradeTier.PREMIUM

    def test_tqs_repr(self):
        tqs = TQSResult(trend_score=20, level_score=15, pattern_score=23, regime_score=25)
        repr_str = repr(tqs)
        assert "83" in repr_str
        assert "PREMIUM" in repr_str


class TestRegimeType:
    """Ensure all 5 regime types are defined."""

    def test_all_regimes_defined(self):
        regimes = [r.value for r in RegimeType]
        assert "TRENDING" in regimes
        assert "RANGING" in regimes
        assert "VOLATILE" in regimes
        assert "QUIET" in regimes
        assert "CHOPPY" in regimes

    def test_volatile_choppy_have_no_strategies(self):
        """VOLATILE and CHOPPY have risk_multiplier 0.0 — no trades allowed."""
        from src.data.types import RegimeSignal, StrategyType
        for regime_type in [RegimeType.VOLATILE, RegimeType.CHOPPY]:
            sig = RegimeSignal(
                regime=regime_type,
                confidence=0.9,
                allowed_strategies=[],
                risk_multiplier=0.0
            )
            assert sig.is_tradeable is False

    def test_trending_is_tradeable(self):
        from src.data.types import RegimeSignal, StrategyType
        sig = RegimeSignal(
            regime=RegimeType.TRENDING,
            confidence=0.9,
            allowed_strategies=[StrategyType.PIN_BAR, StrategyType.ENGULFING],
            risk_multiplier=1.0
        )
        assert sig.is_tradeable is True
