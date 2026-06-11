"""
Tests for M02 — Candle Storage: Database models and operations.
Phase 0 pass criteria:
  - Store and retrieve candles; verify no data loss or corruption
  - Gap detection correctly identifies missing bars
  - Duplicate timestamp handling (upsert, no errors)
"""

import pytest
from datetime import datetime, timezone, timedelta
from sqlalchemy.orm import Session

from src.db.models import (
    Base, Candle, Trade, TradeSignal, SwingPoint, SRLevel,
    StrategyPerformance, AuditEvent, BotState,
    TradeDirectionEnum, TradeTierEnum, RegimeTypeEnum, ExecutionModeEnum,
    SignalStatusEnum, ExitReasonEnum
)
from src.db.database import DatabaseManager


# ===========================================================================
# FIXTURES
# ===========================================================================

@pytest.fixture
def db_manager(tmp_path):
    """In-memory SQLite database for testing."""
    db = DatabaseManager("sqlite:///:memory:")
    yield db
    db.drop_all_tables()


@pytest.fixture
def session(db_manager):
    """Database session with auto-rollback after each test."""
    with db_manager.get_session() as s:
        yield s


def make_candle(
    symbol="EURUSD", timeframe="D1",
    ts=None, open_=1.1000, high=1.1050, low=1.0950, close=1.1020,
    volume=1000.0, spread=1.5
) -> Candle:
    """Factory for test candles."""
    if ts is None:
        ts = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    return Candle(
        symbol=symbol,
        timeframe=timeframe,
        timestamp=ts,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        spread=spread,
    )


# ===========================================================================
# DATABASE INITIALIZATION TESTS
# ===========================================================================

class TestDatabaseInit:

    def test_database_creates_successfully(self, db_manager):
        """Database manager must initialize without errors."""
        assert db_manager is not None

    def test_health_check_passes(self, db_manager):
        """Health check must return True for active database."""
        assert db_manager.health_check() is True

    def test_all_tables_created(self, db_manager):
        """All expected tables must be created."""
        stats = db_manager.get_table_stats()
        expected_tables = [
            "candles", "trades", "trade_signals", "swing_points",
            "sr_levels", "pattern_detections", "strategy_performance",
            "audit_events", "bot_state",
        ]
        for table in expected_tables:
            assert table in stats, f"Table '{table}' not found in stats"

    def test_initial_table_counts_are_zero(self, db_manager):
        """All tables must start empty."""
        stats = db_manager.get_table_stats()
        for table, count in stats.items():
            assert count == 0, f"Table '{table}' has {count} rows but should be empty"


# ===========================================================================
# CANDLE STORAGE TESTS (M02 CORE)
# ===========================================================================

class TestCandleStorage:

    def test_store_single_candle(self, db_manager):
        """Store one candle and retrieve it without data loss."""
        candle = make_candle()
        with db_manager.get_session() as session:
            session.add(candle)

        with db_manager.get_session() as session:
            retrieved = session.query(Candle).filter_by(
                symbol="EURUSD", timeframe="D1"
            ).first()
            assert retrieved is not None
            assert retrieved.open == 1.1000
            assert retrieved.high == 1.1050
            assert retrieved.low == 1.0950
            assert retrieved.close == 1.1020
            assert retrieved.volume == 1000.0
            assert retrieved.spread == 1.5

    def test_store_batch_candles(self, db_manager):
        """Store 100 candles in batch — verify count and no corruption."""
        candles = []
        base_ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        for i in range(100):
            ts = base_ts + timedelta(days=i)
            candles.append(make_candle(ts=ts, close=1.1000 + i * 0.0001))

        with db_manager.get_session() as session:
            session.add_all(candles)

        with db_manager.get_session() as session:
            count = session.query(Candle).count()
            assert count == 100

    def test_round_trip_lossless(self, db_manager):
        """OHLCV values must survive storage/retrieval exactly."""
        original_values = {
            "open": 1.12345,
            "high": 1.12456,
            "low": 1.12234,
            "close": 1.12389,
            "volume": 12345.67,
            "spread": 1.8,
        }
        candle = make_candle(
            open_=original_values["open"],
            high=original_values["high"],
            low=original_values["low"],
            close=original_values["close"],
            volume=original_values["volume"],
            spread=original_values["spread"],
        )
        with db_manager.get_session() as session:
            session.add(candle)

        with db_manager.get_session() as session:
            retrieved = session.query(Candle).first()
            assert abs(retrieved.open - original_values["open"]) < 1e-8
            assert abs(retrieved.high - original_values["high"]) < 1e-8
            assert abs(retrieved.low - original_values["low"]) < 1e-8
            assert abs(retrieved.close - original_values["close"]) < 1e-8
            assert abs(retrieved.volume - original_values["volume"]) < 1e-4

    def test_duplicate_candle_raises_error(self, db_manager):
        """Duplicate (symbol, timeframe, timestamp) must raise IntegrityError."""
        from sqlalchemy.exc import IntegrityError
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        candle1 = make_candle(ts=ts)
        candle2 = make_candle(ts=ts, close=1.2000)  # Same timestamp, different close

        with pytest.raises(IntegrityError):
            with db_manager.get_session() as session:
                session.add(candle1)
                session.add(candle2)

    def test_candle_ordering_by_timestamp(self, db_manager):
        """Candles must be retrievable in chronological order."""
        base_ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        candles = [
            make_candle(ts=base_ts + timedelta(days=i), close=1.1 + i * 0.001)
            for i in [2, 0, 4, 1, 3]  # Deliberately out of order
        ]
        with db_manager.get_session() as session:
            session.add_all(candles)

        with db_manager.get_session() as session:
            retrieved = (
                session.query(Candle)
                .order_by(Candle.timestamp.asc())
                .all()
            )
            timestamps = [c.timestamp for c in retrieved]
            assert timestamps == sorted(timestamps)

    def test_filter_by_symbol_and_timeframe(self, db_manager):
        """Filter candles by symbol and timeframe must return only matching."""
        base_ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        with db_manager.get_session() as session:
            session.add(make_candle(symbol="EURUSD", timeframe="D1", ts=base_ts))
            session.add(make_candle(symbol="GBPUSD", timeframe="D1",
                                    ts=base_ts + timedelta(days=1)))
            session.add(make_candle(symbol="EURUSD", timeframe="H4",
                                    ts=base_ts + timedelta(days=2)))

        with db_manager.get_session() as session:
            eurusd_d1 = session.query(Candle).filter_by(
                symbol="EURUSD", timeframe="D1"
            ).all()
            assert len(eurusd_d1) == 1
            assert eurusd_d1[0].symbol == "EURUSD"

    def test_get_latest_n_candles(self, db_manager):
        """get latest N candles returns most recent N in order."""
        base_ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        with db_manager.get_session() as session:
            for i in range(20):
                session.add(make_candle(
                    ts=base_ts + timedelta(days=i),
                    close=1.1 + i * 0.001
                ))

        with db_manager.get_session() as session:
            latest_5 = (
                session.query(Candle)
                .filter_by(symbol="EURUSD", timeframe="D1")
                .order_by(Candle.timestamp.desc())
                .limit(5)
                .all()
            )
            assert len(latest_5) == 5
            # Most recent first
            assert latest_5[0].timestamp > latest_5[1].timestamp


# ===========================================================================
# SWING POINT STORAGE
# ===========================================================================

class TestSwingPointStorage:

    def test_store_swing_high(self, db_manager):
        """Swing high must be stored and retrievable."""
        swing = SwingPoint(
            symbol="EURUSD",
            timeframe="D1",
            timestamp=datetime(2024, 1, 15, tzinfo=timezone.utc),
            price=1.1050,
            swing_type="HIGH",
            lookback=5,
        )
        with db_manager.get_session() as session:
            session.add(swing)

        with db_manager.get_session() as session:
            result = session.query(SwingPoint).filter_by(swing_type="HIGH").first()
            assert result is not None
            assert result.price == 1.1050

    def test_store_multiple_swing_points(self, db_manager):
        """Multiple swing points for trend analysis."""
        base_ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        with db_manager.get_session() as session:
            for i in range(4):
                session.add(SwingPoint(
                    symbol="EURUSD", timeframe="D1",
                    timestamp=base_ts + timedelta(days=i * 10),
                    price=1.1 + i * 0.005,
                    swing_type="HIGH" if i % 2 == 0 else "LOW",
                    lookback=5,
                ))

        with db_manager.get_session() as session:
            count = session.query(SwingPoint).count()
            assert count == 4


# ===========================================================================
# TRADE SIGNAL AND TRADE STORAGE
# ===========================================================================

class TestTradeStorage:

    def test_store_trade_signal(self, db_manager):
        """TradeSignal must be stored with all required fields."""
        import uuid
        from src.db.models import TrendDirectionEnum
        signal = TradeSignal(
            signal_id=str(uuid.uuid4()),
            symbol="EURUSD",
            timeframe="D1",
            strategy="PIN_BAR",
            direction=TradeDirectionEnum.LONG,
            timestamp=datetime(2024, 6, 1, tzinfo=timezone.utc),
            entry_price=1.1000,
            stop_price=1.0950,
            target_price=1.1100,
            rr_ratio=2.0,
            tqs_total=75,
            tqs_trend=18,
            tqs_level=15,
            tqs_pattern=20,
            tqs_regime=22,
            trade_tier=TradeTierEnum.STANDARD,
            regime=RegimeTypeEnum.TRENDING,
            trend_direction=TrendDirectionEnum.UP,
            execution_mode=ExecutionModeEnum.PAPER,
            status=SignalStatusEnum.PENDING,
        )
        with db_manager.get_session() as session:
            session.add(signal)

        with db_manager.get_session() as session:
            retrieved = session.query(TradeSignal).first()
            assert retrieved.tqs_total == 75
            assert retrieved.trade_tier == TradeTierEnum.STANDARD
            assert retrieved.rr_ratio == 2.0

    def test_store_trade(self, db_manager):
        """Trade record must be stored with full lifecycle data."""
        import uuid
        trade = Trade(
            trade_id=str(uuid.uuid4()),
            symbol="EURUSD",
            timeframe="D1",
            strategy="ENGULFING",
            direction=TradeDirectionEnum.LONG,
            execution_mode=ExecutionModeEnum.PAPER,
            timestamp_entry=datetime(2024, 6, 1, tzinfo=timezone.utc),
            entry_price=1.1000,
            fill_price=1.1001,
            lot_size=0.01,
            risk_amount=100.0,
            sl_price=1.0950,
            tp_price=1.1100,
            rr_ratio=2.0,
            is_open=True,
        )
        with db_manager.get_session() as session:
            session.add(trade)

        with db_manager.get_session() as session:
            retrieved = session.query(Trade).first()
            assert retrieved.is_open is True
            assert retrieved.entry_price == 1.1000


# ===========================================================================
# BOT STATE TESTS
# ===========================================================================

class TestBotState:

    def test_bot_state_creation(self, db_manager):
        """Bot state singleton must be storable."""
        state = BotState(
            id=1,
            execution_mode=ExecutionModeEnum.BACKTEST,
            is_halted=False,
            day_realized_loss=0.0,
            week_realized_loss=0.0,
            current_drawdown_pct=0.0,
            consecutive_losses=0,
        )
        with db_manager.get_session() as session:
            session.add(state)

        with db_manager.get_session() as session:
            retrieved = session.query(BotState).first()
            assert retrieved.is_halted is False
            assert retrieved.consecutive_losses == 0

    def test_kill_switch_state(self, db_manager):
        """Kill switch activation must persist correctly."""
        state = BotState(
            id=1,
            execution_mode=ExecutionModeEnum.PAPER,
            is_halted=True,
            halt_reason="Max drawdown reached: 10.5%",
            current_drawdown_pct=10.5,
            consecutive_losses=3,
        )
        with db_manager.get_session() as session:
            session.add(state)

        with db_manager.get_session() as session:
            retrieved = session.query(BotState).first()
            assert retrieved.is_halted is True
            assert "drawdown" in retrieved.halt_reason.lower()


# ===========================================================================
# STRATEGY PERFORMANCE TESTS
# ===========================================================================

class TestStrategyPerformance:

    def test_strategy_performance_stored(self, db_manager):
        """Strategy performance scorecard must be storable."""
        perf = StrategyPerformance(
            strategy_name="PIN_BAR",
            symbol="EURUSD",
            timeframe="D1",
            execution_mode=ExecutionModeEnum.PAPER,
            total_trades=50,
            win_count=22,
            loss_count=28,
            win_rate=0.44,
            profit_factor=1.61,
            expectancy_r=0.48,
            is_enabled=True,
        )
        with db_manager.get_session() as session:
            session.add(perf)

        with db_manager.get_session() as session:
            retrieved = session.query(StrategyPerformance).filter_by(
                strategy_name="PIN_BAR"
            ).first()
            assert retrieved.profit_factor == 1.61
            assert retrieved.total_trades == 50
            assert retrieved.is_enabled is True
