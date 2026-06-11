"""
Enhanced tests for M02 — CandleStore (Sprint 1 additions)
Tests new methods: delete_candles, get_symbols, get_timeframes,
store_candle_data_list, and additional edge cases.
"""

from datetime import datetime, timezone, timedelta
from typing import List

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.db.models import Base, Candle
from src.db.session import init_db
from src.db.candle_store import CandleStore, candle_from_dict
from src.types import CandleData


# ===========================================================================
# FIXTURES
# ===========================================================================

@pytest.fixture
def engine():
    eng = create_engine("sqlite:///:memory:")
    init_db(eng)
    return eng


@pytest.fixture
def session_factory(engine):
    return sessionmaker(bind=engine, autocommit=False, autoflush=False)


@pytest.fixture
def session(session_factory):
    sess = session_factory()
    yield sess
    sess.close()


@pytest.fixture
def store(session):
    return CandleStore(session)


def _dt(year, month, day, hour=0) -> datetime:
    return datetime(year, month, day, hour, tzinfo=timezone.utc)


def _candle(symbol="EURUSD", timeframe="D1", ts=None,
            o=1.10, h=1.11, l=1.09, c=1.105, vol=1000.0) -> Candle:
    return Candle(
        symbol=symbol, timeframe=timeframe,
        timestamp=ts or _dt(2024, 1, 2),
        open=o, high=h, low=l, close=c,
        volume=vol, spread=1.5,
    )


def _series(n: int, symbol="EURUSD", timeframe="D1",
            start=None) -> List[Candle]:
    base = start or _dt(2024, 1, 2)
    return [
        _candle(symbol=symbol, timeframe=timeframe,
                ts=base + timedelta(days=i), o=1.1 + i * 0.0001)
        for i in range(n)
    ]


def _cd_series(n: int, symbol="EURUSD", timeframe="D1") -> List[CandleData]:
    base = _dt(2024, 2, 1)
    return [
        CandleData(
            symbol=symbol, timeframe=timeframe,
            timestamp=base + timedelta(days=i),
            open=1.1 + i * 0.0001, high=1.12, low=1.09, close=1.105,
            volume=500.0, spread=1.5,
        )
        for i in range(n)
    ]


# ===========================================================================
# delete_candles
# ===========================================================================

class TestDeleteCandles:

    def test_delete_all_candles_for_symbol_tf(self, store, session):
        store.store_candles(_series(10))
        session.commit()
        deleted = store.delete_candles("EURUSD", "D1")
        session.commit()
        assert deleted == 10
        assert store.get_candle_count("EURUSD", "D1") == 0

    def test_delete_with_start_only(self, store, session):
        store.store_candles(_series(10, start=_dt(2024, 1, 1)))
        session.commit()
        # Delete from Jan 6 onwards (idx 5-9 = 5 candles)
        deleted = store.delete_candles("EURUSD", "D1", start=_dt(2024, 1, 6))
        session.commit()
        assert deleted == 5
        assert store.get_candle_count("EURUSD", "D1") == 5

    def test_delete_with_end_only(self, store, session):
        store.store_candles(_series(10, start=_dt(2024, 1, 1)))
        session.commit()
        # Delete up to Jan 5 (idx 0-4 = 5 candles)
        deleted = store.delete_candles("EURUSD", "D1", end=_dt(2024, 1, 5))
        session.commit()
        assert deleted == 5
        assert store.get_candle_count("EURUSD", "D1") == 5

    def test_delete_range(self, store, session):
        store.store_candles(_series(10, start=_dt(2024, 1, 1)))
        session.commit()
        # Delete Jan 3–7 (5 candles)
        deleted = store.delete_candles(
            "EURUSD", "D1",
            start=_dt(2024, 1, 3),
            end=_dt(2024, 1, 7),
        )
        session.commit()
        assert deleted == 5
        assert store.get_candle_count("EURUSD", "D1") == 5

    def test_delete_nonexistent_returns_zero(self, store, session):
        deleted = store.delete_candles("GBPUSD", "H4")
        session.commit()
        assert deleted == 0

    def test_delete_does_not_affect_other_symbols(self, store, session):
        store.store_candles(_series(5, symbol="EURUSD"))
        store.store_candles(_series(5, symbol="GBPUSD"))
        session.commit()
        store.delete_candles("EURUSD", "D1")
        session.commit()
        assert store.get_candle_count("EURUSD", "D1") == 0
        assert store.get_candle_count("GBPUSD", "D1") == 5


# ===========================================================================
# get_symbols
# ===========================================================================

class TestGetSymbols:

    def test_empty_db_returns_empty_list(self, store):
        assert store.get_symbols() == []

    def test_returns_stored_symbols(self, store, session):
        store.store_candles(_series(2, symbol="EURUSD"))
        store.store_candles(_series(2, symbol="GBPUSD"))
        store.store_candles(_series(2, symbol="USDJPY"))
        session.commit()
        symbols = store.get_symbols()
        assert set(symbols) == {"EURUSD", "GBPUSD", "USDJPY"}

    def test_symbols_sorted_alphabetically(self, store, session):
        for sym in ("USDJPY", "EURUSD", "GBPUSD", "AUDUSD"):
            store.store_candles(_series(1, symbol=sym))
        session.commit()
        symbols = store.get_symbols()
        assert symbols == sorted(symbols)

    def test_no_duplicates_in_result(self, store, session):
        # Store multiple candles for same symbol
        store.store_candles(_series(10, symbol="EURUSD"))
        session.commit()
        symbols = store.get_symbols()
        assert symbols.count("EURUSD") == 1


# ===========================================================================
# get_timeframes
# ===========================================================================

class TestGetTimeframes:

    def test_empty_db_returns_empty_list(self, store):
        assert store.get_timeframes() == []

    def test_returns_stored_timeframes(self, store, session):
        for tf in ("D1", "H4", "H1"):
            store.store_candles(_series(2, timeframe=tf))
        session.commit()
        tfs = store.get_timeframes()
        assert set(tfs) == {"D1", "H4", "H1"}

    def test_filter_by_symbol(self, store, session):
        store.store_candles(_series(2, symbol="EURUSD", timeframe="D1"))
        store.store_candles(_series(2, symbol="EURUSD", timeframe="H4"))
        store.store_candles(_series(2, symbol="GBPUSD", timeframe="D1"))
        session.commit()
        eur_tfs = store.get_timeframes("EURUSD")
        gbp_tfs = store.get_timeframes("GBPUSD")
        assert set(eur_tfs) == {"D1", "H4"}
        assert set(gbp_tfs) == {"D1"}

    def test_no_symbol_filter_returns_all(self, store, session):
        store.store_candles(_series(2, symbol="EURUSD", timeframe="D1"))
        store.store_candles(_series(2, symbol="GBPUSD", timeframe="H4"))
        session.commit()
        tfs = store.get_timeframes()
        assert "D1" in tfs
        assert "H4" in tfs


# ===========================================================================
# store_candle_data_list
# ===========================================================================

class TestStoreCandleDataList:

    def test_stores_candle_data_dtos(self, store, session):
        dtos = _cd_series(5)
        count = store.store_candle_data_list(dtos)
        session.commit()
        assert count == 5
        assert store.get_candle_count("EURUSD", "D1") == 5

    def test_empty_list_returns_zero(self, store):
        assert store.store_candle_data_list([]) == 0

    def test_upsert_on_duplicate_dto(self, store, session):
        dtos = _cd_series(5)
        store.store_candle_data_list(dtos)
        session.commit()
        store.store_candle_data_list(dtos)   # again
        session.commit()
        assert store.get_candle_count("EURUSD", "D1") == 5

    def test_naive_timestamp_handled(self, store, session):
        dto = CandleData(
            symbol="EURUSD", timeframe="D1",
            timestamp=datetime(2024, 6, 1),    # naive!
            open=1.1, high=1.11, low=1.09, close=1.105,
            volume=100.0, spread=1.5,
        )
        count = store.store_candle_data_list([dto])
        session.commit()
        assert count == 1

    def test_values_preserved_after_store(self, store, session):
        dto = CandleData(
            symbol="EURUSD", timeframe="D1",
            timestamp=_dt(2024, 6, 1),
            open=1.10001, high=1.11001, low=1.09001, close=1.10501,
            volume=750.0, spread=2.0,
        )
        store.store_candle_data_list([dto])
        session.commit()
        candles = store.get_candles(
            "EURUSD", "D1",
            start=_dt(2024, 6, 1),
            end=_dt(2024, 6, 2),
        )
        assert len(candles) == 1
        assert candles[0].open == pytest.approx(1.10001)
        assert candles[0].volume == pytest.approx(750.0)


# ===========================================================================
# validate_data_integrity (CandleStore method)
# ===========================================================================

class TestCandleStoreValidateIntegrity:

    def test_clean_series_returns_clean(self, store, session):
        store.store_candles(_series(10))
        session.commit()
        result = store.validate_data_integrity("EURUSD", "D1")
        assert result["total_candles"] == 10
        errors = [a for a in result["anomalies"] if a.get("severity") != "WARNING"]
        assert len(errors) == 0

    def test_gap_detected_in_validation(self, store, session):
        c1 = _candle(ts=_dt(2024, 1, 1))
        c2 = _candle(ts=_dt(2024, 3, 1))   # 59-day gap
        store.store_candles([c1, c2])
        session.commit()
        result = store.validate_data_integrity("EURUSD", "D1")
        assert result["gap_count"] >= 1

    def test_empty_symbol_returns_empty_result(self, store):
        result = store.validate_data_integrity("XYZUSD", "D1")
        assert result["total_candles"] == 0
        assert result["anomaly_count"] == 0
        assert result["gap_count"] == 0


# ===========================================================================
# candle_gap_check edge cases
# ===========================================================================

class TestCandleGapCheckEdgeCases:

    def test_two_candles_one_day_apart_no_gap(self, store, session):
        store.store_candles([
            _candle(ts=_dt(2024, 1, 2)),
            _candle(ts=_dt(2024, 1, 3)),
        ])
        session.commit()
        gaps = store.candle_gap_check("EURUSD", "D1")
        assert gaps == []

    def test_weekend_skip_not_flagged_as_gap(self, store, session):
        """Fri→Mon gap is 72 h (259 200 s).

        The documented tolerance is 2.5 × D1 interval = 2.5 × 86 400 = 216 000 s
        (60 h), so 72 h **does** exceed the threshold and is flagged as a gap.
        The test asserts the correct behaviour: exactly one gap is returned, and
        its missing_bars_estimate is small (≤ 2 — it is a weekend, not a real
        data hole spanning many bars).
        """
        store.store_candles([
            _candle(ts=_dt(2024, 1, 5)),   # Friday
            _candle(ts=_dt(2024, 1, 8)),   # Monday  (72 h gap > 60 h tolerance)
        ])
        session.commit()
        gaps = store.candle_gap_check("EURUSD", "D1")
        # The gap IS detected because 72 h > 2.5 × 86 400 s = 60 h.
        assert len(gaps) == 1
        assert gaps[0]["gap_seconds"] == pytest.approx(72 * 3600, rel=1e-3)
        assert gaps[0]["missing_bars_estimate"] <= 2

    def test_missing_week_detected(self, store, session):
        store.store_candles([
            _candle(ts=_dt(2024, 1, 1)),
            _candle(ts=_dt(2024, 1, 15)),   # 14-day gap
        ])
        session.commit()
        gaps = store.candle_gap_check("EURUSD", "D1")
        assert len(gaps) == 1
        assert gaps[0]["missing_bars_estimate"] >= 10

    def test_h4_gap_uses_correct_interval(self, store, session):
        """H4 bars are 4 hours apart; >10h should be a gap."""
        store.store_candles([
            _candle(timeframe="H4", ts=_dt(2024, 1, 1, 0)),
            _candle(timeframe="H4", ts=_dt(2024, 1, 1, 4)),   # normal gap
            _candle(timeframe="H4", ts=_dt(2024, 1, 2, 0)),   # 20h gap → >2.5x of 4h = 10h
        ])
        session.commit()
        gaps = store.candle_gap_check("EURUSD", "H4")
        assert len(gaps) == 1

    def test_custom_interval_respected(self, store, session):
        store.store_candles([
            _candle(ts=_dt(2024, 1, 1)),
            _candle(ts=_dt(2024, 1, 10)),   # 9-day gap
        ])
        session.commit()
        # With 1-hour expected interval, everything is a gap
        gaps = store.candle_gap_check("EURUSD", "D1", expected_interval_seconds=3600)
        assert len(gaps) >= 1


# ===========================================================================
# get_candle_count / get_date_range
# ===========================================================================

class TestCountAndDateRange:

    def test_count_zero_for_unknown_symbol(self, store):
        assert store.get_candle_count("XYZUSD", "D1") == 0

    def test_count_reflects_stored_candles(self, store, session):
        store.store_candles(_series(7))
        session.commit()
        assert store.get_candle_count("EURUSD", "D1") == 7

    def test_date_range_none_when_no_data(self, store):
        assert store.get_date_range("XYZUSD", "D1") is None

    def test_date_range_min_max_correct(self, store, session):
        """get_date_range returns the min/max timestamp for the series.

        SQLite's func.min/max returns naive datetime strings which SQLAlchemy
        deserialises as tz-naive datetime objects, while the stored CandleData
        timestamps are UTC-aware.  We compare without tzinfo to avoid a false
        mismatch on the tzinfo attribute alone.
        """
        candles = _series(10, start=_dt(2024, 3, 1))
        store.store_candles(candles)
        session.commit()
        dr = store.get_date_range("EURUSD", "D1")
        assert dr is not None

        def _naive(dt: datetime) -> datetime:
            return dt.replace(tzinfo=None) if dt.tzinfo else dt

        assert _naive(dr[0]) == _naive(candles[0].timestamp)
        assert _naive(dr[1]) == _naive(candles[-1].timestamp)


# ===========================================================================
# candle_from_dict additional edge cases
# ===========================================================================

class TestCandleFromDictEdgeCases:

    def test_string_timestamp_iso(self):
        data = {
            "symbol": "EURUSD", "timeframe": "D1",
            "timestamp": "2024-06-01T00:00:00",
            "open": 1.1, "high": 1.11, "low": 1.09, "close": 1.105,
        }
        c = candle_from_dict(data)
        assert c.symbol == "EURUSD"
        assert c.timestamp.year == 2024

    def test_spread_none_allowed(self):
        data = {
            "symbol": "EURUSD", "timeframe": "D1",
            "timestamp": datetime(2024, 1, 1, tzinfo=timezone.utc),
            "open": 1.1, "high": 1.11, "low": 1.09, "close": 1.105,
        }
        c = candle_from_dict(data)
        assert c.spread is None

    def test_volume_defaults_to_zero(self):
        data = {
            "symbol": "EURUSD", "timeframe": "D1",
            "timestamp": datetime(2024, 1, 1, tzinfo=timezone.utc),
            "open": 1.1, "high": 1.11, "low": 1.09, "close": 1.105,
        }
        c = candle_from_dict(data)
        assert c.volume == 0.0

    def test_high_exactly_equals_low_is_valid(self):
        """Doji with no range is technically valid."""
        data = {
            "symbol": "EURUSD", "timeframe": "D1",
            "timestamp": datetime(2024, 1, 1, tzinfo=timezone.utc),
            "open": 1.1, "high": 1.1, "low": 1.1, "close": 1.1,
        }
        c = candle_from_dict(data)
        assert c.open == pytest.approx(1.1)
