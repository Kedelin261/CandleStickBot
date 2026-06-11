"""
Tests for M05 — Support & Resistance Engine.

Coverage:
  - identify_support_levels: basic detection, zone boundaries, scoring
  - identify_resistance_levels: basic detection
  - score_level: touch count, recency, role-reversal bonus
  - find_nearest_level: ABOVE / BELOW / BOTH directions
  - classify_zone: INSIDE / ABOVE / BELOW
  - detect_role_reversals: RTS and STR promotion, flag set
  - SRLevel: zone helpers, strength property, to_level_data() DTO
  - SRAnalysis: support/resistance/strong properties, levels_near_price
  - SREngine.analyze(): empty candles, swing lows/highs, SMA21 level,
      nearest support/resistance, role reversals, count limits
  - SREngine.calculate_tqs_level_score(): all strength bands
  - SREngine.persist_levels(): insert, update/dedup, get_levels()
  - No Fibonacci / no strategy logic
  - Symbol-agnostic (GBPUSD / USDJPY pass-through)
  - Determinism (same input → same output)
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import List

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.analysis.sr_engine import (
    LevelStrength,
    LevelType,
    SRAnalysis,
    SREngine,
    SRLevel,
    classify_zone,
    detect_role_reversals,
    find_nearest_level,
    identify_resistance_levels,
    identify_support_levels,
    score_level,
)
from src.db.models import Base
from src.db.session import init_db
from src.types import CandleData, LevelData, LevelType as DtoLevelType


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
    n: int = 30,
    price: float = 1.1000,
    symbol: str = BASE_SYMBOL,
    timeframe: str = BASE_TF,
) -> List[CandleData]:
    return [
        _candle(i, price, price + 0.0005, price - 0.0005, price,
                symbol=symbol, timeframe=timeframe)
        for i in range(n)
    ]


def make_ranging_candles(
    n: int = 40,
    center: float = 1.1000,
    amplitude: float = 0.0050,
    symbol: str = BASE_SYMBOL,
    timeframe: str = BASE_TF,
) -> List[CandleData]:
    """Oscillates between center±amplitude — good for touch counting."""
    candles = []
    for i in range(n):
        offset = amplitude if i % 2 == 0 else -amplitude
        p = center + offset
        candles.append(_candle(i, center, max(p, center) + 0.0001,
                               min(p, center) - 0.0001, p,
                               symbol=symbol, timeframe=timeframe))
    return candles


def make_sr_level(
    price: float,
    level_type: LevelType = LevelType.SUPPORT,
    strength_score: float = 5.0,
    touch_count: int = 3,
    zone_half: float = 0.0005,
    is_rts: bool = False,
) -> SRLevel:
    return SRLevel(
        price=price,
        level_type=level_type,
        strength_score=strength_score,
        touch_count=touch_count,
        zone_high=price + zone_half,
        zone_low=price - zone_half,
        is_resistance_turned_support=is_rts,
    )


# ===========================================================================
# FIXTURES
# ===========================================================================

@pytest.fixture
def engine() -> SREngine:
    return SREngine(
        zone_width_pips=5.0,
        merge_pips=10.0,
        pip_size=0.0001,
        max_levels=10,
        nearby_threshold_pips=30.0,
        recency_window=20,
    )


@pytest.fixture
def flat_candles() -> List[CandleData]:
    return make_flat_candles(40, price=1.1000)


@pytest.fixture
def ranging_candles() -> List[CandleData]:
    return make_ranging_candles(40)


@pytest.fixture
def db_session():
    """In-memory SQLite session for persistence tests."""
    db_engine = create_engine("sqlite:///:memory:")
    init_db(db_engine)
    factory = sessionmaker(bind=db_engine, autocommit=False, autoflush=False)
    session = factory()
    yield session
    session.rollback()
    session.close()


# ===========================================================================
# 1. IDENTIFY SUPPORT LEVELS
# ===========================================================================

class TestIdentifySupportLevels:
    def test_returns_list(self, flat_candles):
        lows = [1.0950, 1.0960]
        result = identify_support_levels(flat_candles, lows)
        assert isinstance(result, list)

    def test_empty_swing_lows_returns_empty(self, flat_candles):
        assert identify_support_levels(flat_candles, []) == []

    def test_single_level_created(self, flat_candles):
        result = identify_support_levels(flat_candles, [1.0950])
        assert len(result) >= 1

    def test_level_type_is_support(self, flat_candles):
        result = identify_support_levels(flat_candles, [1.0950])
        assert result[0].level_type == LevelType.SUPPORT

    def test_zone_boundaries_correct(self, flat_candles):
        result = identify_support_levels(flat_candles, [1.0950],
                                          zone_width_pips=10.0, pip_size=0.0001)
        lv = result[0]
        assert abs(lv.zone_high - (lv.price + 0.0005)) < 1e-8
        assert abs(lv.zone_low  - (lv.price - 0.0005)) < 1e-8

    def test_nearby_levels_merged(self, flat_candles):
        # Two lows only 3 pips apart → should merge into one
        result = identify_support_levels(
            flat_candles, [1.0950, 1.0953],
            zone_width_pips=5.0, merge_pips=10.0, pip_size=0.0001,
        )
        assert len(result) == 1

    def test_distant_levels_not_merged(self, flat_candles):
        result = identify_support_levels(
            flat_candles, [1.0900, 1.0950],
            zone_width_pips=5.0, merge_pips=5.0, pip_size=0.0001,
        )
        assert len(result) == 2

    def test_accepts_float_prices(self, flat_candles):
        result = identify_support_levels(flat_candles, [1.0950, 1.0960])
        assert all(isinstance(lv, SRLevel) for lv in result)

    def test_accepts_swing_point_objects(self, flat_candles):
        """SwingPoint-like objects with .price attribute."""
        class FakeSP:
            def __init__(self, price): self.price = price
        result = identify_support_levels(flat_candles, [FakeSP(1.0950), FakeSP(1.0960)])
        assert len(result) >= 1

    def test_strength_score_bounded(self, flat_candles):
        result = identify_support_levels(flat_candles, [1.0950])
        assert 0.0 <= result[0].strength_score <= 10.0


# ===========================================================================
# 2. IDENTIFY RESISTANCE LEVELS
# ===========================================================================

class TestIdentifyResistanceLevels:
    def test_returns_list(self, flat_candles):
        result = identify_resistance_levels(flat_candles, [1.1050])
        assert isinstance(result, list)

    def test_level_type_is_resistance(self, flat_candles):
        result = identify_resistance_levels(flat_candles, [1.1050])
        assert result[0].level_type == LevelType.RESISTANCE

    def test_empty_highs_returns_empty(self, flat_candles):
        assert identify_resistance_levels(flat_candles, []) == []

    def test_nearby_highs_merged(self, flat_candles):
        result = identify_resistance_levels(
            flat_candles, [1.1050, 1.1053],
            zone_width_pips=5.0, merge_pips=10.0, pip_size=0.0001,
        )
        assert len(result) == 1


# ===========================================================================
# 3. SCORE LEVEL
# ===========================================================================

class TestScoreLevel:
    def test_returns_float_in_range(self, flat_candles):
        lv = make_sr_level(1.0950, touch_count=2)
        s = score_level(lv, flat_candles)
        assert 0.0 <= s <= 10.0

    def test_higher_touch_count_higher_score(self, flat_candles):
        lv1 = make_sr_level(1.0950, touch_count=1)
        lv2 = make_sr_level(1.0950, touch_count=4)
        assert score_level(lv2, flat_candles) >= score_level(lv1, flat_candles)

    def test_rts_flag_adds_bonus(self, flat_candles):
        lv_normal = make_sr_level(1.0950, is_rts=False, touch_count=2)
        lv_rts    = make_sr_level(1.0950, is_rts=True,  touch_count=2)
        assert score_level(lv_rts, flat_candles) >= score_level(lv_normal, flat_candles)

    def test_score_capped_at_10(self, flat_candles):
        lv = make_sr_level(1.0950, touch_count=100, is_rts=True)
        assert score_level(lv, flat_candles) <= 10.0


# ===========================================================================
# 4. FIND NEAREST LEVEL
# ===========================================================================

class TestFindNearestLevel:
    def _levels(self) -> List[SRLevel]:
        return [
            make_sr_level(1.0900),
            make_sr_level(1.0950),
            make_sr_level(1.1050),
            make_sr_level(1.1100),
        ]

    def test_find_nearest_below(self):
        nearest = find_nearest_level(1.1000, self._levels(), "BELOW")
        assert nearest is not None
        assert nearest.price == pytest.approx(1.0950, abs=1e-9)

    def test_find_nearest_above(self):
        nearest = find_nearest_level(1.1000, self._levels(), "ABOVE")
        assert nearest is not None
        assert nearest.price == pytest.approx(1.1050, abs=1e-9)

    def test_find_nearest_both(self):
        nearest = find_nearest_level(1.1000, self._levels(), "BOTH")
        assert nearest is not None
        assert nearest.price in (1.0950, 1.1050)  # equidistant; either valid

    def test_returns_none_when_empty(self):
        assert find_nearest_level(1.1000, [], "BOTH") is None

    def test_returns_none_when_no_match(self):
        levels = [make_sr_level(1.0900)]
        assert find_nearest_level(1.1000, levels, "ABOVE") is None

    def test_single_level_below(self):
        levels = [make_sr_level(1.0950)]
        result = find_nearest_level(1.1000, levels, "BELOW")
        assert result is not None
        assert result.price == pytest.approx(1.0950, abs=1e-9)


# ===========================================================================
# 5. CLASSIFY ZONE
# ===========================================================================

class TestClassifyZone:
    def _lv(self) -> SRLevel:
        return make_sr_level(1.1000, zone_half=0.0010)

    def test_price_inside_zone(self):
        assert classify_zone(1.1005, self._lv()) == "INSIDE"

    def test_price_above_zone(self):
        assert classify_zone(1.1020, self._lv()) == "ABOVE"

    def test_price_below_zone(self):
        assert classify_zone(1.0980, self._lv()) == "BELOW"

    def test_price_at_zone_high(self):
        assert classify_zone(1.1010, self._lv()) == "INSIDE"

    def test_price_at_zone_low(self):
        assert classify_zone(1.0990, self._lv()) == "INSIDE"


# ===========================================================================
# 6. DETECT ROLE REVERSALS
# ===========================================================================

class TestDetectRoleReversals:
    def test_resistance_below_price_becomes_rts(self):
        """A former resistance level now below current price → RTS."""
        resistance = [make_sr_level(1.0950, level_type=LevelType.RESISTANCE)]
        support    = []
        sup_out, res_out = detect_role_reversals(support, resistance,
                                                  current_price=1.1000,
                                                  tolerance_pips=5.0)
        assert len(res_out) == 1
        assert res_out[0].level_type == LevelType.RESISTANCE_TURNED_SUPPORT
        assert res_out[0].is_resistance_turned_support is True

    def test_support_above_price_becomes_str(self):
        """A former support level now above current price → STR."""
        support    = [make_sr_level(1.1100, level_type=LevelType.SUPPORT)]
        resistance = []
        sup_out, res_out = detect_role_reversals(support, resistance,
                                                  current_price=1.1000,
                                                  tolerance_pips=5.0)
        assert sup_out[0].level_type == LevelType.SUPPORT_TURNED_RESISTANCE

    def test_no_reversal_when_resistance_above_price(self):
        """Resistance above current price stays as resistance."""
        resistance = [make_sr_level(1.1100, level_type=LevelType.RESISTANCE)]
        support    = []
        _, res_out = detect_role_reversals(support, resistance,
                                            current_price=1.1000)
        assert res_out[0].level_type == LevelType.RESISTANCE

    def test_rts_score_boosted(self):
        """RTS levels should have a higher score than the original."""
        orig_score = 4.0
        resistance = [make_sr_level(1.0950, level_type=LevelType.RESISTANCE,
                                    strength_score=orig_score)]
        _, res_out = detect_role_reversals([], resistance, current_price=1.1000)
        assert res_out[0].strength_score >= orig_score

    def test_empty_lists_safe(self):
        sup_out, res_out = detect_role_reversals([], [], current_price=1.1000)
        assert sup_out == []
        assert res_out == []


# ===========================================================================
# 7. SRLEVEL HELPERS
# ===========================================================================

class TestSRLevelHelpers:
    def test_contains_price_true(self):
        lv = make_sr_level(1.1000, zone_half=0.0010)
        assert lv.contains_price(1.1005) is True

    def test_contains_price_false(self):
        lv = make_sr_level(1.1000, zone_half=0.0010)
        assert lv.contains_price(1.1020) is False

    def test_zone_midpoint(self):
        lv = make_sr_level(1.1000, zone_half=0.0010)
        assert abs(lv.zone_midpoint - 1.1000) < 1e-9

    def test_zone_width(self):
        lv = make_sr_level(1.1000, zone_half=0.0010)
        assert abs(lv.zone_width - 0.0020) < 1e-9

    def test_strength_strong(self):
        lv = make_sr_level(1.1000, strength_score=8.0)
        assert lv.strength == LevelStrength.STRONG

    def test_strength_moderate(self):
        lv = make_sr_level(1.1000, strength_score=5.0)
        assert lv.strength == LevelStrength.MODERATE

    def test_strength_weak(self):
        lv = make_sr_level(1.1000, strength_score=2.0)
        assert lv.strength == LevelStrength.WEAK

    def test_to_level_data_returns_level_data(self):
        lv = make_sr_level(1.1000)
        ld = lv.to_level_data()
        assert isinstance(ld, LevelData)

    def test_to_level_data_support_maps_to_swing_sr(self):
        lv = make_sr_level(1.1000, level_type=LevelType.SUPPORT)
        ld = lv.to_level_data()
        assert ld.level_type == DtoLevelType.SWING_SR

    def test_to_level_data_sma21_maps_to_sma_21(self):
        lv = make_sr_level(1.1000, level_type=LevelType.SMA21)
        ld = lv.to_level_data()
        assert ld.level_type == DtoLevelType.SMA_21

    def test_to_level_data_price_preserved(self):
        lv = make_sr_level(1.0987)
        ld = lv.to_level_data()
        assert abs(ld.price - 1.0987) < 1e-9

    def test_distance_to_in_pips(self):
        lv = make_sr_level(1.1000)
        assert abs(lv.distance_to(1.1020) - 20.0) < 1e-6


# ===========================================================================
# 8. SRANALYSIS PROPERTIES
# ===========================================================================

class TestSRAnalysisProperties:
    def _analysis(self) -> SRAnalysis:
        return SRAnalysis(
            levels=[
                make_sr_level(1.0950, level_type=LevelType.SUPPORT, strength_score=8.0),
                make_sr_level(1.0960, level_type=LevelType.RESISTANCE_TURNED_SUPPORT, strength_score=5.0),
                make_sr_level(1.1050, level_type=LevelType.RESISTANCE, strength_score=6.0),
                make_sr_level(1.1100, level_type=LevelType.SUPPORT_TURNED_RESISTANCE, strength_score=3.0),
            ],
            current_price=1.1000,
        )

    def test_support_levels_correct(self):
        a = self._analysis()
        types = {lv.level_type for lv in a.support_levels}
        assert LevelType.SUPPORT in types
        assert LevelType.RESISTANCE_TURNED_SUPPORT in types
        assert LevelType.RESISTANCE not in types

    def test_resistance_levels_correct(self):
        a = self._analysis()
        types = {lv.level_type for lv in a.resistance_levels}
        assert LevelType.RESISTANCE in types
        assert LevelType.SUPPORT_TURNED_RESISTANCE in types
        assert LevelType.SUPPORT not in types

    def test_strong_levels_filter(self):
        a = self._analysis()
        strong = a.strong_levels
        assert all(lv.strength == LevelStrength.STRONG for lv in strong)

    def test_levels_near_price(self):
        a = self._analysis()
        near = a.levels_near_price(1.1000, threshold_pips=60.0)
        prices = [lv.price for lv in near]
        assert 1.0950 in prices
        assert 1.1050 in prices

    def test_all_levels_sorted_ascending(self):
        a = self._analysis()
        prices = [lv.price for lv in a.all_levels_sorted]
        assert prices == sorted(prices)


# ===========================================================================
# 9. SRENGINE.ANALYZE — CORE INTEGRATION
# ===========================================================================

class TestSREngineAnalyze:
    def test_empty_candles_returns_safe_result(self, engine):
        result = engine.analyze([])
        assert isinstance(result, SRAnalysis)
        assert result.levels == []
        assert result.candles_analyzed == 0

    def test_no_swings_returns_empty_levels(self, engine, flat_candles):
        result = engine.analyze(flat_candles)
        assert result.levels == []

    def test_current_price_set(self, engine, flat_candles):
        result = engine.analyze(flat_candles, swing_lows=[1.0950])
        assert abs(result.current_price - flat_candles[-1].close) < 1e-9

    def test_candles_analyzed_set(self, engine, flat_candles):
        result = engine.analyze(flat_candles, swing_lows=[1.0950])
        assert result.candles_analyzed == len(flat_candles)

    def test_support_levels_detected(self, engine, flat_candles):
        result = engine.analyze(flat_candles, swing_lows=[1.0950, 1.0930])
        assert len(result.support_levels) >= 1

    def test_resistance_levels_detected(self, engine, flat_candles):
        result = engine.analyze(flat_candles, swing_highs=[1.1050, 1.1070])
        assert len(result.resistance_levels) >= 1

    def test_sma21_level_created(self, engine, flat_candles):
        result = engine.analyze(flat_candles, sma21=1.0980)
        assert result.sma21_level is not None
        assert abs(result.sma21_level.price - 1.0980) < 1e-9
        assert result.sma21_level.level_type == LevelType.SMA21

    def test_no_sma21_level_when_not_provided(self, engine, flat_candles):
        result = engine.analyze(flat_candles, swing_lows=[1.0950])
        assert result.sma21_level is None

    def test_nearest_support_is_below_price(self, engine, flat_candles):
        result = engine.analyze(
            flat_candles,
            swing_lows=[1.0950, 1.0930],
            swing_highs=[1.1050],
        )
        if result.nearest_support:
            assert result.nearest_support.price < result.current_price

    def test_nearest_resistance_is_above_price(self, engine, flat_candles):
        result = engine.analyze(
            flat_candles,
            swing_lows=[1.0950],
            swing_highs=[1.1050, 1.1070],
        )
        if result.nearest_resistance:
            assert result.nearest_resistance.price > result.current_price

    def test_levels_sorted_by_strength(self, engine, flat_candles):
        """Levels list should be descending by strength_score."""
        result = engine.analyze(
            flat_candles,
            swing_lows=[1.0930, 1.0950, 1.0970],
            swing_highs=[1.1050, 1.1070],
        )
        scores = [lv.strength_score for lv in result.levels]
        assert scores == sorted(scores, reverse=True)

    def test_max_levels_respected(self, flat_candles):
        eng = SREngine(max_levels=2, pip_size=0.0001)
        many_lows  = [1.0900 + i * 0.0020 for i in range(8)]
        result = eng.analyze(flat_candles, swing_lows=many_lows)
        # Each side capped at max_levels, total ≤ max_levels * 2
        assert len(result.support_levels) <= 2

    def test_symbol_agnostic_gbpusd(self):
        eng = SREngine(pip_size=0.0001)
        candles = make_flat_candles(30, price=1.2700, symbol="GBPUSD")
        result = eng.analyze(candles, swing_lows=[1.2650], swing_highs=[1.2750])
        assert result.current_price == pytest.approx(1.2700, abs=1e-9)

    def test_usdjpy_large_pip_size(self):
        eng = SREngine(pip_size=0.01)   # USDJPY uses 0.01 pip size
        candles = make_flat_candles(30, price=150.00, symbol="USDJPY")
        result = eng.analyze(candles, swing_lows=[149.50], swing_highs=[150.50])
        assert isinstance(result, SRAnalysis)

    def test_deterministic_output(self, engine, flat_candles):
        r1 = engine.analyze(flat_candles, swing_lows=[1.0950], swing_highs=[1.1050])
        r2 = engine.analyze(flat_candles, swing_lows=[1.0950], swing_highs=[1.1050])
        assert len(r1.levels) == len(r2.levels)
        if r1.levels and r2.levels:
            assert r1.levels[0].price == pytest.approx(r2.levels[0].price)

    def test_rts_level_in_results(self, engine, flat_candles):
        """Resistance below current price should become RTS in the result."""
        # current_price = 1.1000; swing high at 1.0950 (below price → RTS)
        result = engine.analyze(
            flat_candles,
            swing_highs=[1.0950],   # below current price → RTS
        )
        rts_levels = [lv for lv in result.levels
                      if lv.is_resistance_turned_support]
        assert len(rts_levels) >= 1

    def test_ranging_candles_produce_multiple_levels(self):
        """Ranging market with many swing touches → multiple graded levels."""
        eng = SREngine(zone_width_pips=8.0, merge_pips=12.0, pip_size=0.0001)
        candles = make_ranging_candles(40, center=1.1000, amplitude=0.0050)
        result = eng.analyze(
            candles,
            swing_lows=[1.0950, 1.0955],
            swing_highs=[1.1050, 1.1055],
        )
        # After merging, should still have at least one level on each side
        assert len(result.support_levels) >= 1
        assert len(result.resistance_levels) >= 1


# ===========================================================================
# 10. TQS LEVEL SCORING
# ===========================================================================

class TestTqsLevelScore:
    def test_strong_level_scores_22(self, engine, flat_candles):
        support = make_sr_level(1.0995, strength_score=8.0)
        score = engine.calculate_tqs_level_score(
            flat_candles[-1], support, None, "LONG"
        )
        assert score == 22

    def test_moderate_level_scores_18(self, engine, flat_candles):
        support = make_sr_level(1.0995, strength_score=5.0)
        score = engine.calculate_tqs_level_score(
            flat_candles[-1], support, None, "LONG"
        )
        assert score == 18

    def test_weak_level_scores_12(self, engine, flat_candles):
        support = make_sr_level(1.0995, strength_score=2.0)
        score = engine.calculate_tqs_level_score(
            flat_candles[-1], support, None, "LONG"
        )
        assert score == 12

    def test_rts_scores_25(self, engine, flat_candles):
        support = make_sr_level(1.0995, strength_score=8.0, is_rts=True)
        score = engine.calculate_tqs_level_score(
            flat_candles[-1], support, None, "LONG"
        )
        assert score == 25

    def test_no_level_scores_5(self, engine, flat_candles):
        score = engine.calculate_tqs_level_score(
            flat_candles[-1], None, None, "LONG"
        )
        assert score == 5

    def test_too_far_level_scores_5(self, engine, flat_candles):
        """Level more than nearby_threshold away → 5."""
        far_support = make_sr_level(1.0500, strength_score=8.0)
        score = engine.calculate_tqs_level_score(
            flat_candles[-1], far_support, None, "LONG"
        )
        assert score == 5

    def test_short_uses_resistance(self, engine, flat_candles):
        resistance = make_sr_level(1.1005, level_type=LevelType.RESISTANCE, strength_score=8.0)
        score = engine.calculate_tqs_level_score(
            flat_candles[-1], None, resistance, "SHORT"
        )
        assert score == 22


# ===========================================================================
# 11. PERSISTENCE
# ===========================================================================

class TestPersistence:
    def test_persist_inserts_rows(self, engine, flat_candles, db_session):
        lows = [1.0950, 1.0930]
        result = engine.analyze(flat_candles, swing_lows=lows)
        inserted = engine.persist_levels(
            result.support_levels, db_session, BASE_SYMBOL, BASE_TF
        )
        assert inserted > 0

    def test_persist_empty_list_returns_zero(self, engine, db_session):
        inserted = engine.persist_levels([], db_session, BASE_SYMBOL, BASE_TF)
        assert inserted == 0

    def test_persist_idempotent(self, engine, flat_candles, db_session):
        """Persisting the same levels twice should not duplicate rows."""
        lows = [1.0950]
        result = engine.analyze(flat_candles, swing_lows=lows)
        engine.persist_levels(result.support_levels, db_session, BASE_SYMBOL, BASE_TF)
        db_session.commit()
        # Second call — should update, not insert duplicate
        inserted2 = engine.persist_levels(
            result.support_levels, db_session, BASE_SYMBOL, BASE_TF,
            price_tolerance=engine.pip_size,
        )
        assert inserted2 > 0  # returns count of upserts (existing + new)
        # Verify row count in DB
        from src.db.models import SRLevel as ORM
        count = db_session.query(ORM).filter_by(
            symbol=BASE_SYMBOL, timeframe=BASE_TF
        ).count()
        assert count == len(result.support_levels)

    def test_get_levels_returns_persisted(self, engine, flat_candles, db_session):
        result = engine.analyze(flat_candles, swing_lows=[1.0950])
        engine.persist_levels(result.support_levels, db_session, BASE_SYMBOL, BASE_TF)
        db_session.commit()
        retrieved = engine.get_levels(BASE_SYMBOL, BASE_TF, db_session)
        assert len(retrieved) >= 1
        assert all(isinstance(lv, SRLevel) for lv in retrieved)

    def test_get_levels_empty_db_returns_empty(self, engine, db_session):
        result = engine.get_levels("GBPUSD", "H4", db_session)
        assert result == []
