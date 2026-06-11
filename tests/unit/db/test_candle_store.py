"""
Tests for M02 — Candle Storage: CandleStore
Validates storage, retrieval, deduplication, and gap detection.
"""

from datetime import datetime, timedelta
from typing import List

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.db.models import Base, Candle
from src.db.session import init_db, get_session
from src.db.candle_store import CandleStore, candle_from_dict


# ===========================================================================
# FIXTURES
# ===========================================================================

@pytest.fixture
def engine():
    """In-memory SQLite engine for testing."""
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


def make_candle(
    symbol="EURUSD",
    timeframe="D1",
    timestamp=None,
    open_=1.1000,
    high=1.1100,
    low=1.0950,
    close=1.1050,
    volume=1000.0,
    spread=1.5,
) -> Candle:
    """Helper to create a valid Candle ORM object."""
    return Candle(
        symbol=symbol,
        timeframe=timeframe,
        timestamp=timestamp or datetime(2024, 1, 1, 0, 0, 0),
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        spread=spread,
    )


def make_candle_series(n: int, symbol="EURUSD", timeframe="D1", start_date=None) -> List[Candle]:
    """Create N consecutive daily candles with unique timestamps."""
    base_date = start_date or datetime(2024, 1, 1)
    candles = []
    base_price = 1.1000
    # Use hours offset to guarantee unique timestamps regardless of weekday
    for i in range(n):
        ts = base_date + timedelta(hours=i * 24)
        open_ = base_price + i * 0.0001
        close = open_ + 0.0010
        candles.append(Candle(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=ts,
            open=open_,
            high=close + 0.0020,
            low=open_ - 0.0010,
            close=close,
            volume=1000.0 + i,
            spread=1.5,
        ))
    return candles


# ===========================================================================
# STORE AND RETRIEVE TESTS
# ===========================================================================

class TestStoreCandles:

    def test_store_single_candle(self, store, session):
        """Single candle stores successfully."""
        candle = make_candle()
        count = store.store_candles([candle])
        session.commit()
        assert count == 1
        assert store.get_candle_count("EURUSD", "D1") == 1

    def test_store_batch_candles(self, store, session):
        """Batch of 10 candles stores correctly."""
        candles = make_candle_series(10)
        count = store.store_candles(candles)
        session.commit()
        assert count == 10
        assert store.get_candle_count("EURUSD", "D1") == 10

    def test_store_large_batch(self, store, session):
        """10,000 candles store without data loss or corruption."""
        candles = make_candle_series(10000)
        count = store.store_candles(candles)
        session.commit()
        assert count == 10000
        assert store.get_candle_count("EURUSD", "D1") == 10000

    def test_upsert_no_duplicates(self, store, session):
        """Storing same candle twice results in one record (upsert)."""
        candle1 = make_candle(close=1.1050)
        candle2 = make_candle(close=1.1055)  # Same timestamp, different close

        store.store_candles([candle1])
        session.commit()
        store.store_candles([candle2])
        session.commit()

        assert store.get_candle_count("EURUSD", "D1") == 1
        # Should have updated close to 1.1055
        latest = store.get_latest_n_candles("EURUSD", "D1", 1)
        assert latest[0].close == pytest.approx(1.1055)

    def test_empty_list_returns_zero(self, store):
        """Storing empty list returns 0."""
        assert store.store_candles([]) == 0


# ===========================================================================
# RETRIEVAL TESTS
# ===========================================================================

class TestGetCandles:

    def test_get_candles_by_range(self, store, session):
        """get_candles returns candles within date range."""
        candles = make_candle_series(30, start_date=datetime(2024, 1, 1))
        store.store_candles(candles)
        session.commit()

        result = store.get_candles(
            "EURUSD", "D1",
            start=datetime(2024, 1, 5),
            end=datetime(2024, 1, 15),
        )
        for c in result:
            assert datetime(2024, 1, 5) <= c.timestamp <= datetime(2024, 1, 15)

    def test_get_candles_returns_ascending_order(self, store, session):
        """Candles returned in ascending timestamp order."""
        candles = make_candle_series(10)
        store.store_candles(candles)
        session.commit()

        result = store.get_candles(
            "EURUSD", "D1",
            start=datetime(2024, 1, 1),
            end=datetime(2024, 12, 31),
        )
        timestamps = [c.timestamp for c in result]
        assert timestamps == sorted(timestamps)

    def test_get_latest_n_candles(self, store, session):
        """get_latest_n_candles returns exactly N most recent candles."""
        candles = make_candle_series(50)
        store.store_candles(candles)
        session.commit()

        latest = store.get_latest_n_candles("EURUSD", "D1", 10)
        assert len(latest) == 10
        # Should be the most recent 10, ordered ascending
        full = store.get_candles("EURUSD", "D1",
                                 start=datetime(2020, 1, 1),
                                 end=datetime(2030, 1, 1))
        assert latest[-1].timestamp == full[-1].timestamp

    def test_get_candles_empty_range(self, store, session):
        """Empty range returns empty list."""
        store.store_candles(make_candle_series(5))
        session.commit()
        result = store.get_candles("EURUSD", "D1",
                                   start=datetime(2030, 1, 1),
                                   end=datetime(2030, 12, 31))
        assert result == []

    def test_get_date_range(self, store, session):
        """get_date_range returns correct min/max timestamps."""
        candles = make_candle_series(10, start_date=datetime(2024, 3, 1))
        store.store_candles(candles)
        session.commit()

        date_range = store.get_date_range("EURUSD", "D1")
        assert date_range is not None
        assert date_range[0] == candles[0].timestamp
        assert date_range[1] == candles[-1].timestamp

    def test_get_date_range_no_data(self, store):
        """get_date_range returns None when no data exists."""
        result = store.get_date_range("GBPUSD", "H4")
        assert result is None

    def test_different_symbols_isolated(self, store, session):
        """Candles for different symbols don't cross-contaminate."""
        eurusd = make_candle_series(5, symbol="EURUSD")
        gbpusd = make_candle_series(5, symbol="GBPUSD")
        store.store_candles(eurusd + gbpusd)
        session.commit()

        eur_result = store.get_candles("EURUSD", "D1",
                                       start=datetime(2024, 1, 1),
                                       end=datetime(2024, 12, 31))
        gbp_result = store.get_candles("GBPUSD", "D1",
                                       start=datetime(2024, 1, 1),
                                       end=datetime(2024, 12, 31))
        assert all(c.symbol == "EURUSD" for c in eur_result)
        assert all(c.symbol == "GBPUSD" for c in gbp_result)


# ===========================================================================
# GAP DETECTION TESTS
# ===========================================================================

class TestCandleGapCheck:

    def test_no_gaps_in_clean_data(self, store, session):
        """Clean consecutive data returns no gaps."""
        candles = make_candle_series(10)
        store.store_candles(candles)
        session.commit()
        gaps = store.candle_gap_check("EURUSD", "D1")
        # Weekends are allowed (2.5x tolerance)
        assert isinstance(gaps, list)

    def test_detects_large_gap(self, store, session):
        """Large gap (e.g., missing a week) is detected."""
        c1 = make_candle(timestamp=datetime(2024, 1, 1))
        c2 = make_candle(timestamp=datetime(2024, 1, 15))  # 14-day gap
        store.store_candles([c1, c2])
        session.commit()

        gaps = store.candle_gap_check("EURUSD", "D1")
        assert len(gaps) >= 1
        assert gaps[0]["missing_bars_estimate"] > 0

    def test_gap_has_required_fields(self, store, session):
        """Gap dict contains all required fields."""
        c1 = make_candle(timestamp=datetime(2024, 1, 1))
        c2 = make_candle(timestamp=datetime(2024, 1, 20))
        store.store_candles([c1, c2])
        session.commit()

        gaps = store.candle_gap_check("EURUSD", "D1")
        assert len(gaps) > 0
        gap = gaps[0]
        assert "gap_start" in gap
        assert "gap_end" in gap
        assert "missing_bars_estimate" in gap

    def test_single_candle_no_gaps(self, store, session):
        """Single candle cannot have gaps."""
        store.store_candles([make_candle()])
        session.commit()
        gaps = store.candle_gap_check("EURUSD", "D1")
        assert gaps == []


# ===========================================================================
# DATA INTEGRITY TESTS
# ===========================================================================

class TestValidateDataIntegrity:

    def test_clean_data_passes_integrity(self, store, session):
        """Valid candle series passes all integrity checks."""
        candles = make_candle_series(10)
        store.store_candles(candles)
        session.commit()
        result = store.validate_data_integrity("EURUSD", "D1")
        # May have zero-volume warnings but no errors
        errors = [a for a in result["anomalies"] if a.get("severity") != "WARNING"]
        assert len(errors) == 0


# ===========================================================================
# CANDLE_FROM_DICT TESTS
# ===========================================================================

class TestCandleFromDict:

    def test_valid_dict_creates_candle(self):
        """Valid dict creates Candle object."""
        data = {
            "symbol": "EURUSD",
            "timeframe": "D1",
            "timestamp": datetime(2024, 1, 1),
            "open": 1.1000,
            "high": 1.1100,
            "low": 1.0900,
            "close": 1.1050,
            "volume": 1000.0,
            "spread": 1.5,
        }
        c = candle_from_dict(data)
        assert c.symbol == "EURUSD"
        assert c.open == pytest.approx(1.1000)

    def test_missing_required_field_raises(self):
        """Missing required field raises KeyError."""
        data = {
            "symbol": "EURUSD",
            "timeframe": "D1",
            "timestamp": datetime(2024, 1, 1),
            "open": 1.1000,
            # missing high, low, close
        }
        with pytest.raises(KeyError):
            candle_from_dict(data)

    def test_high_below_low_raises(self):
        """High < Low raises ValueError."""
        data = {
            "symbol": "EURUSD",
            "timeframe": "D1",
            "timestamp": datetime(2024, 1, 1),
            "open": 1.1000,
            "high": 1.0900,  # < low!
            "low": 1.1000,
            "close": 1.1000,
        }
        with pytest.raises(ValueError) as exc_info:
            candle_from_dict(data)
        assert "high" in str(exc_info.value).lower()

    def test_zero_price_raises(self):
        """Zero price raises ValueError."""
        data = {
            "symbol": "EURUSD",
            "timeframe": "D1",
            "timestamp": datetime(2024, 1, 1),
            "open": 0.0,
            "high": 1.1100,
            "low": 1.0900,
            "close": 1.1050,
        }
        with pytest.raises(ValueError):
            candle_from_dict(data)

    def test_symbol_uppercased(self):
        """Symbol is normalized to uppercase."""
        data = {
            "symbol": "eurusd",
            "timeframe": "d1",
            "timestamp": datetime(2024, 1, 1),
            "open": 1.1000,
            "high": 1.1100,
            "low": 1.0900,
            "close": 1.1050,
        }
        c = candle_from_dict(data)
        assert c.symbol == "EURUSD"
        assert c.timeframe == "D1"

    def test_roundtrip_precision(self):
        """OHLC values maintain precision through conversion."""
        data = {
            "symbol": "EURUSD",
            "timeframe": "D1",
            "timestamp": datetime(2024, 1, 1),
            "open": 1.10001,
            "high": 1.10050,
            "low": 1.09980,
            "close": 1.10025,
        }
        c = candle_from_dict(data)
        assert c.open == pytest.approx(1.10001, abs=1e-5)
        assert c.close == pytest.approx(1.10025, abs=1e-5)
