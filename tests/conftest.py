"""
Shared pytest fixtures for CandleStickBot test suite.
"""

import sys
from pathlib import Path

# Ensure src is importable without pip install
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime, timedelta
from typing import List

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.db.models import Base, Candle
from src.db.session import init_db
from src.types import CandleData, Direction


# ===========================================================================
# DATABASE FIXTURES
# ===========================================================================

@pytest.fixture(scope="function")
def db_engine():
    """In-memory SQLite engine, recreated per test function."""
    engine = create_engine("sqlite:///:memory:")
    init_db(engine)
    yield engine
    engine.dispose()


@pytest.fixture(scope="function")
def db_session(db_engine):
    """SQLAlchemy session for testing with auto-rollback."""
    factory = sessionmaker(bind=db_engine, autocommit=False, autoflush=False)
    session = factory()
    yield session
    session.rollback()
    session.close()


# ===========================================================================
# CANDLE DATA FACTORIES
# ===========================================================================

def make_candle_data(
    symbol: str = "EURUSD",
    timeframe: str = "D1",
    timestamp: datetime = None,
    open_: float = 1.1000,
    high: float = 1.1100,
    low: float = 1.0950,
    close: float = 1.1050,
    volume: float = 1000.0,
    spread: float = 1.5,
) -> CandleData:
    """Create a CandleData DTO."""
    return CandleData(
        symbol=symbol,
        timeframe=timeframe,
        timestamp=timestamp or datetime(2024, 1, 15),
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        spread=spread,
    )


def make_bullish_candle(timestamp=None, base_price=1.1000) -> CandleData:
    """Bullish candle (close > open)."""
    return make_candle_data(
        timestamp=timestamp or datetime(2024, 1, 15),
        open_=base_price,
        high=base_price + 0.0100,
        low=base_price - 0.0020,
        close=base_price + 0.0080,
    )


def make_bearish_candle(timestamp=None, base_price=1.1000) -> CandleData:
    """Bearish candle (close < open)."""
    return make_candle_data(
        timestamp=timestamp or datetime(2024, 1, 15),
        open_=base_price,
        high=base_price + 0.0020,
        low=base_price - 0.0100,
        close=base_price - 0.0080,
    )


def make_doji_candle(timestamp=None, price=1.1000) -> CandleData:
    """Doji candle (open == close)."""
    return make_candle_data(
        timestamp=timestamp or datetime(2024, 1, 15),
        open_=price,
        high=price + 0.0050,
        low=price - 0.0050,
        close=price,
    )


def make_uptrend_candles(n: int = 20, start_price: float = 1.0800) -> List[CandleData]:
    """
    Series of candles forming a clear uptrend (HH+HL pattern).
    Used for M03/M04 trend detection tests.
    """
    candles = []
    base = datetime(2024, 1, 1)
    price = start_price
    for i in range(n):
        open_ = price
        close = price + 0.0020 + (i * 0.0001)  # Steadily rising
        high = close + 0.0015
        low = open_ - 0.0010
        candles.append(CandleData(
            symbol="EURUSD",
            timeframe="D1",
            timestamp=base + timedelta(days=i),
            open=open_,
            high=high,
            low=low,
            close=close,
        ))
        price = close - 0.0005  # Small pullback then resume up
    return candles


def make_downtrend_candles(n: int = 20, start_price: float = 1.1200) -> List[CandleData]:
    """
    Series of candles forming a clear downtrend (LH+LL pattern).
    Used for M03/M04 trend detection tests.
    """
    candles = []
    base = datetime(2024, 1, 1)
    price = start_price
    for i in range(n):
        open_ = price
        close = price - 0.0020 - (i * 0.0001)
        high = open_ + 0.0010
        low = close - 0.0015
        candles.append(CandleData(
            symbol="EURUSD",
            timeframe="D1",
            timestamp=base + timedelta(days=i),
            open=open_,
            high=high,
            low=low,
            close=close,
        ))
        price = close + 0.0005  # Small bounce then resume down
    return candles


def make_ranging_candles(n: int = 20, center: float = 1.1000, range_size: float = 0.0100) -> List[CandleData]:
    """
    Sideways/ranging candles oscillating around a center price.
    """
    import math
    candles = []
    base = datetime(2024, 1, 1)
    for i in range(n):
        # Oscillate using sine wave
        offset = math.sin(i * 0.5) * range_size * 0.5
        open_ = center + offset
        close = center - offset  # Opposite direction
        high = max(open_, close) + 0.0010
        low = min(open_, close) - 0.0010
        candles.append(CandleData(
            symbol="EURUSD",
            timeframe="D1",
            timestamp=base + timedelta(days=i),
            open=open_,
            high=high,
            low=low,
            close=close,
        ))
    return candles


# ===========================================================================
# PYTEST FIXTURES FROM FACTORIES
# ===========================================================================

@pytest.fixture
def bullish_candle():
    return make_bullish_candle()


@pytest.fixture
def bearish_candle():
    return make_bearish_candle()


@pytest.fixture
def doji_candle():
    return make_doji_candle()


@pytest.fixture
def uptrend_candles():
    return make_uptrend_candles(20)


@pytest.fixture
def downtrend_candles():
    return make_downtrend_candles(20)


@pytest.fixture
def ranging_candles():
    return make_ranging_candles(20)
