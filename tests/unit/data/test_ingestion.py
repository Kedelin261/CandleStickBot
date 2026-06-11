"""
Tests for M01 — Data Ingestion Engine
Phase 1 Sprint 1: Data Pipeline

Covers:
    - normalize_candle_dict: timestamp parsing, field mapping, type coercion
    - validate_ohlcv:        all OHLC rules, volume/spread constraints
    - validate_data_integrity: duplicate detection, sort checking
    - deduplicate_candles:   last-wins de-dup, timestamp sorting
    - DataIngestionEngine:   MT5 unavailable graceful failure, connection state
    - CSVDataLoader:         full CSV round-trip (string and file), bad rows
    - FetchResult:           error flag, metadata
    - candle_data_to_orm / candle_orm_to_data: round-trip conversions
"""

import csv
import io
import textwrap
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List

import pytest

from src.data.ingestion import (
    DataIngestionEngine,
    DataSource,
    CSVDataLoader,
    ConnectionStatus,
    FetchResult,
    MT5Config,
    normalize_candle_dict,
    validate_ohlcv,
    validate_data_integrity,
    deduplicate_candles,
    sort_candles_ascending,
    candle_data_to_orm,
    candle_orm_to_data,
    _parse_timestamp,
)
from src.types import CandleData
from src.db.candle_store import CandleStore


# ===========================================================================
# HELPERS
# ===========================================================================

def _dt(year: int, month: int, day: int, hour: int = 0) -> datetime:
    return datetime(year, month, day, hour, tzinfo=timezone.utc)


def _make_cd(
    symbol="EURUSD", timeframe="D1",
    ts: datetime = None, o=1.1000, h=1.1100, l=1.0900, c=1.1050,
    vol=1000.0, spread=1.5,
) -> CandleData:
    return CandleData(
        symbol=symbol, timeframe=timeframe,
        timestamp=ts or _dt(2024, 1, 2),
        open=o, high=h, low=l, close=c,
        volume=vol, spread=spread,
    )


def _make_series(n: int, start: datetime = None) -> List[CandleData]:
    base = start or _dt(2024, 1, 2)
    return [
        _make_cd(ts=base + timedelta(days=i), o=1.1+i*0.0001)
        for i in range(n)
    ]


# ===========================================================================
# _parse_timestamp
# ===========================================================================

class TestParseTimestamp:

    def test_datetime_aware_passthrough(self):
        dt = _dt(2024, 3, 15, 12)
        result = _parse_timestamp(dt)
        assert result == dt

    def test_datetime_naive_becomes_utc(self):
        dt_naive = datetime(2024, 3, 15, 12)
        result = _parse_timestamp(dt_naive)
        assert result.tzinfo == timezone.utc
        assert result.year == 2024

    def test_unix_int_epoch(self):
        result = _parse_timestamp(0)
        assert result == datetime(1970, 1, 1, tzinfo=timezone.utc)

    def test_unix_float_epoch(self):
        result = _parse_timestamp(1_704_153_600.0)  # 2024-01-02 00:00:00 UTC
        assert result.year == 2024

    def test_string_iso_with_time(self):
        result = _parse_timestamp("2024-06-15 09:30:00")
        assert result == datetime(2024, 6, 15, 9, 30, tzinfo=timezone.utc)

    def test_string_iso_date_only(self):
        result = _parse_timestamp("2024-01-01")
        assert result == datetime(2024, 1, 1, tzinfo=timezone.utc)

    def test_string_iso_T_separator(self):
        result = _parse_timestamp("2024-06-15T09:30:00")
        assert result.hour == 9

    def test_string_uk_format(self):
        result = _parse_timestamp("15/06/2024 09:30:00")
        assert result.day == 15

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError, match="Cannot parse timestamp"):
            _parse_timestamp("not_a_date")

    def test_unsupported_type_raises(self):
        with pytest.raises(ValueError, match="Unsupported timestamp type"):
            _parse_timestamp([2024, 1, 1])  # type: ignore


# ===========================================================================
# normalize_candle_dict
# ===========================================================================

class TestNormalizeCandleDict:

    def _base_row(self) -> dict:
        return {
            "timestamp": "2024-01-02",
            "open":  "1.1000",
            "high":  "1.1100",
            "low":   "1.0900",
            "close": "1.1050",
            "volume": "500",
            "spread": "1.5",
        }

    def test_basic_normalization(self):
        row = self._base_row()
        result = normalize_candle_dict(row, "eurusd", "d1")
        assert result["symbol"] == "EURUSD"
        assert result["timeframe"] == "D1"
        assert isinstance(result["timestamp"], datetime)
        assert result["open"] == pytest.approx(1.1)
        assert result["volume"] == pytest.approx(500.0)
        assert result["spread"] == pytest.approx(1.5)

    def test_symbol_and_timeframe_uppercased(self):
        result = normalize_candle_dict(self._base_row(), "gbpusd", "h4")
        assert result["symbol"] == "GBPUSD"
        assert result["timeframe"] == "H4"

    def test_string_prices_coerced_to_float(self):
        result = normalize_candle_dict(self._base_row(), "EURUSD", "D1")
        for field in ("open", "high", "low", "close"):
            assert isinstance(result[field], float)

    def test_missing_volume_defaults_to_zero(self):
        row = self._base_row()
        del row["volume"]
        result = normalize_candle_dict(row, "EURUSD", "D1")
        assert result["volume"] == 0.0

    def test_negative_volume_clamped_to_zero(self):
        row = {**self._base_row(), "volume": "-50"}
        result = normalize_candle_dict(row, "EURUSD", "D1")
        assert result["volume"] == 0.0

    def test_missing_spread_becomes_none(self):
        row = self._base_row()
        del row["spread"]
        result = normalize_candle_dict(row, "EURUSD", "D1")
        assert result["spread"] is None

    def test_negative_spread_clamped_to_zero(self):
        row = {**self._base_row(), "spread": "-1"}
        result = normalize_candle_dict(row, "EURUSD", "D1")
        assert result["spread"] == 0.0

    def test_alt_timestamp_col_name(self):
        row = {
            "time": "2024-06-01",
            "open": 1.1, "high": 1.11, "low": 1.09, "close": 1.105,
        }
        result = normalize_candle_dict(row, "EURUSD", "D1", timestamp_col="time")
        assert result["timestamp"].year == 2024

    def test_missing_timestamp_raises_key_error(self):
        row = {"open": 1.1, "high": 1.11, "low": 1.09, "close": 1.105}
        with pytest.raises(KeyError, match="Timestamp column"):
            normalize_candle_dict(row, "EURUSD", "D1")

    def test_missing_ohlc_field_raises_key_error(self):
        row = {"timestamp": "2024-01-01", "open": 1.1, "high": 1.11, "low": 1.09}
        with pytest.raises(KeyError, match="close"):
            normalize_candle_dict(row, "EURUSD", "D1")

    def test_tick_volume_alias(self):
        row = {
            "timestamp": "2024-01-02",
            "open": 1.1, "high": 1.11, "low": 1.09, "close": 1.105,
            "tick_volume": "750",
        }
        result = normalize_candle_dict(row, "EURUSD", "D1")
        assert result["volume"] == 750.0

    def test_datetime_object_as_timestamp(self):
        row = {
            "timestamp": _dt(2024, 5, 1),
            "open": 1.1, "high": 1.11, "low": 1.09, "close": 1.105,
        }
        result = normalize_candle_dict(row, "EURUSD", "D1")
        assert result["timestamp"] == _dt(2024, 5, 1)

    def test_unix_epoch_as_timestamp(self):
        row = {
            "time": 1_704_153_600,   # 2024-01-02 00:00:00 UTC
            "open": 1.1, "high": 1.11, "low": 1.09, "close": 1.105,
        }
        result = normalize_candle_dict(row, "EURUSD", "D1", timestamp_col="time")
        assert result["timestamp"].year == 2024


# ===========================================================================
# validate_ohlcv
# ===========================================================================

class TestValidateOHLCV:

    def _valid(self) -> dict:
        return {
            "open": 1.1000, "high": 1.1100, "low": 1.0900, "close": 1.1050,
            "volume": 500.0, "spread": 1.5,
            "timestamp": _dt(2024, 1, 1),
        }

    def test_valid_candle_returns_no_errors(self):
        assert validate_ohlcv(self._valid()) == []

    def test_high_below_low_returns_error(self):
        row = {**self._valid(), "high": 1.08, "low": 1.09}
        errors = validate_ohlcv(row)
        assert any("high" in e and "low" in e for e in errors)

    def test_high_below_open_returns_error(self):
        row = {**self._valid(), "high": 1.09, "open": 1.10}
        errors = validate_ohlcv(row)
        assert any("open" in e or "high" in e for e in errors)

    def test_high_below_close_returns_error(self):
        row = {**self._valid(), "high": 1.09, "close": 1.10}
        errors = validate_ohlcv(row)
        assert len(errors) > 0

    def test_zero_open_returns_error(self):
        row = {**self._valid(), "open": 0.0}
        errors = validate_ohlcv(row)
        assert any("open" in e for e in errors)

    def test_negative_high_returns_error(self):
        row = {**self._valid(), "high": -1.0}
        errors = validate_ohlcv(row)
        assert len(errors) > 0

    def test_negative_volume_returns_error(self):
        row = {**self._valid(), "volume": -1.0}
        errors = validate_ohlcv(row)
        assert any("volume" in e for e in errors)

    def test_negative_spread_returns_error(self):
        row = {**self._valid(), "spread": -0.5}
        errors = validate_ohlcv(row)
        assert any("spread" in e for e in errors)

    def test_non_datetime_timestamp_returns_error(self):
        row = {**self._valid(), "timestamp": "2024-01-01"}
        errors = validate_ohlcv(row)
        assert any("timestamp" in e for e in errors)

    def test_non_numeric_price_returns_error(self):
        row = {**self._valid(), "open": "NOT_A_NUMBER"}
        errors = validate_ohlcv(row)
        assert len(errors) > 0

    def test_missing_field_returns_error(self):
        row = self._valid()
        del row["close"]
        errors = validate_ohlcv(row)
        assert any("close" in e for e in errors)

    def test_zero_volume_is_allowed(self):
        """Volume = 0 is valid — some brokers omit tick volume."""
        row = {**self._valid(), "volume": 0.0}
        assert validate_ohlcv(row) == []

    def test_none_spread_is_allowed(self):
        """Spread = None is valid — optional field."""
        row = {**self._valid(), "spread": None}
        assert validate_ohlcv(row) == []

    def test_low_above_open_returns_error(self):
        row = {**self._valid(), "low": 1.15, "high": 1.16}
        errors = validate_ohlcv(row)
        assert len(errors) > 0

    def test_all_equal_ohlc_is_valid(self):
        """Doji candle: all prices equal is valid."""
        row = {**self._valid(), "open": 1.1, "high": 1.1, "low": 1.1, "close": 1.1}
        assert validate_ohlcv(row) == []


# ===========================================================================
# validate_data_integrity (module-level function)
# ===========================================================================

class TestValidateDataIntegrityFunction:

    def test_clean_series_is_valid(self):
        candles = _make_series(10)
        result = validate_data_integrity(candles)
        assert result["is_valid"] is True
        assert result["invalid"] == 0
        assert result["duplicate_count"] == 0
        assert result["sort_issues"] == 0

    def test_duplicate_timestamp_detected(self):
        candles = _make_series(5)
        candles.append(_make_cd(ts=candles[2].timestamp))  # duplicate at idx 2
        result = validate_data_integrity(candles)
        assert result["duplicate_count"] >= 1
        assert result["is_valid"] is False

    def test_out_of_order_detected(self):
        candles = _make_series(5)
        # Swap last two to break ordering
        candles[-1], candles[-2] = candles[-2], candles[-1]
        result = validate_data_integrity(candles)
        assert result["sort_issues"] >= 1
        assert result["is_valid"] is False

    def test_invalid_ohlc_detected(self):
        bad = _make_cd(h=1.08, l=1.09)   # high < low
        result = validate_data_integrity([bad])
        assert result["invalid"] == 1
        assert result["is_valid"] is False

    def test_empty_list_is_valid(self):
        result = validate_data_integrity([])
        assert result["is_valid"] is True
        assert result["total"] == 0

    def test_counts_are_consistent(self):
        candles = _make_series(10)
        result = validate_data_integrity(candles)
        assert result["total"] == 10
        assert result["valid"] + result["invalid"] == 10


# ===========================================================================
# deduplicate_candles
# ===========================================================================

class TestDeduplicateCandles:

    def test_no_duplicates_unchanged_length(self):
        candles = _make_series(5)
        deduped = deduplicate_candles(candles)
        assert len(deduped) == 5

    def test_duplicate_removed_last_wins(self):
        base_ts = _dt(2024, 1, 2)
        c1 = _make_cd(ts=base_ts, c=1.100)
        c2 = _make_cd(ts=base_ts, c=1.200)  # same timestamp, different close
        result = deduplicate_candles([c1, c2])
        assert len(result) == 1
        assert result[0].close == pytest.approx(1.200)

    def test_output_sorted_ascending(self):
        candles = _make_series(5)
        shuffled = [candles[4], candles[0], candles[2], candles[1], candles[3]]
        result = deduplicate_candles(shuffled)
        timestamps = [c.timestamp for c in result]
        assert timestamps == sorted(timestamps)

    def test_multiple_duplicates_each_keeps_last(self):
        t1 = _dt(2024, 1, 1)
        t2 = _dt(2024, 1, 2)
        candles = [
            _make_cd(ts=t1, c=1.0), _make_cd(ts=t2, c=2.0),
            _make_cd(ts=t1, c=1.5), _make_cd(ts=t2, c=2.5),
        ]
        result = deduplicate_candles(candles)
        assert len(result) == 2
        prices = {c.timestamp: c.close for c in result}
        assert prices[t1] == pytest.approx(1.5)
        assert prices[t2] == pytest.approx(2.5)


# ===========================================================================
# sort_candles_ascending
# ===========================================================================

class TestSortCandlesAscending:

    def test_unsorted_becomes_sorted(self):
        candles = _make_series(5)
        shuffled = list(reversed(candles))
        result = sort_candles_ascending(shuffled)
        ts = [c.timestamp for c in result]
        assert ts == sorted(ts)

    def test_already_sorted_unchanged(self):
        candles = _make_series(3)
        result = sort_candles_ascending(candles)
        assert [c.timestamp for c in result] == [c.timestamp for c in candles]


# ===========================================================================
# DataIngestionEngine — MT5 unavailable
# ===========================================================================

class TestDataIngestionEngineMT5Unavailable:
    """
    Tests for DataIngestionEngine when MetaTrader5 is not installed.
    MT5 is unavailable in this CI environment; we test graceful degradation.
    """

    @pytest.fixture
    def engine(self):
        cfg = MT5Config(login=12345, password="pass", server="Demo-Server")
        return DataIngestionEngine(cfg)

    def test_initial_status_is_disconnected(self, engine):
        assert engine.status == ConnectionStatus.DISCONNECTED

    def test_is_connected_false_initially(self, engine):
        assert engine.is_connected is False

    def test_connect_returns_false_when_mt5_unavailable(self, engine):
        """MT5 package absent → connect() always returns False."""
        result = engine.connect()
        # Either MT5 is available (connected) or unavailable (False)
        # In test environment (Linux, no MT5), expect False
        if not DataIngestionEngine._mt5_available:
            assert result is False
            assert engine.status == ConnectionStatus.ERROR

    def test_fetch_when_not_connected_returns_failure(self, engine):
        result = engine.fetch_candles("EURUSD", "D1", count=10)
        assert result.success is False
        assert result.candles_fetched == 0
        assert result.has_errors is True
        assert any("connect" in e.lower() for e in result.errors)

    def test_fetch_latest_when_not_connected(self, engine):
        result = engine.fetch_latest("EURUSD", "D1", n=50)
        assert result.success is False
        assert result.source == DataSource.MT5

    def test_fetch_historical_when_not_connected(self, engine):
        result = engine.fetch_historical("EURUSD", "D1", start=_dt(2024, 1, 1))
        assert result.success is False

    def test_disconnect_is_safe_when_not_connected(self, engine):
        """disconnect() must not raise even if never connected."""
        engine.disconnect()  # should not raise
        assert engine.status == ConnectionStatus.DISCONNECTED

    def test_get_symbol_info_returns_none_when_not_connected(self, engine):
        assert engine.get_symbol_info("EURUSD") is None

    def test_get_server_time_returns_none_when_not_connected(self, engine):
        assert engine.get_server_time() is None

    def test_fetch_result_has_error_flag(self, engine):
        result = engine.fetch_candles("EURUSD", "D1")
        assert result.has_errors is True

    def test_mt5_available_attribute_is_bool(self):
        assert isinstance(DataIngestionEngine._mt5_available, bool)


# ===========================================================================
# DataIngestionEngine — unknown timeframe
# ===========================================================================

class TestDataIngestionEngineTimeframeValidation:
    """
    When connected (mock), unknown timeframes should fail gracefully.
    We monkey-patch _status to CONNECTED to test the timeframe check path.
    """

    @pytest.fixture
    def engine_connected(self):
        cfg = MT5Config(login=0, password="x", server="x")
        eng = DataIngestionEngine(cfg)
        eng._status = ConnectionStatus.CONNECTED
        eng._mt5 = None  # no real MT5 object needed for TF validation
        return eng

    def test_unknown_timeframe_returns_failure(self, engine_connected):
        result = engine_connected.fetch_candles("EURUSD", "INVALID_TF", count=10)
        assert result.success is False
        assert any("timeframe" in e.lower() or "INVALID_TF" in e for e in result.errors)

    def test_known_timeframe_d1_proceeds(self, engine_connected):
        """D1 is a valid timeframe — the engine should try the MT5 call,
        which then fails because _mt5 is None, not because of the timeframe."""
        result = engine_connected.fetch_candles("EURUSD", "D1", count=10)
        # Fails because _mt5 is None, not due to TF validation
        assert result.success is False
        # Error should NOT mention unknown timeframe
        assert not any("unknown timeframe" in e.lower() for e in result.errors)


# ===========================================================================
# FetchResult
# ===========================================================================

class TestFetchResult:

    def test_has_errors_true_when_errors_present(self):
        r = FetchResult(
            success=False, symbol="EURUSD", timeframe="D1",
            candles_fetched=0, candles_stored=0, source=DataSource.MT5,
            errors=["some error"],
        )
        assert r.has_errors is True

    def test_has_errors_false_when_no_errors(self):
        r = FetchResult(
            success=True, symbol="EURUSD", timeframe="D1",
            candles_fetched=5, candles_stored=5, source=DataSource.CSV,
            errors=[],
        )
        assert r.has_errors is False

    def test_default_errors_is_empty_list(self):
        r = FetchResult(
            success=True, symbol="EURUSD", timeframe="D1",
            candles_fetched=1, candles_stored=1, source=DataSource.MOCK,
        )
        assert r.errors == []


# ===========================================================================
# CSVDataLoader — load_csv_string (no files needed)
# ===========================================================================

class TestCSVDataLoaderString:
    """Test CSVDataLoader using in-memory CSV strings."""

    @pytest.fixture
    def loader(self):
        return CSVDataLoader(candle_store=None)

    def _basic_csv(self, rows: int = 5) -> str:
        header = "timestamp,open,high,low,close,volume,spread\n"
        lines = []
        base = datetime(2024, 1, 2)
        for i in range(rows):
            ts = (base + timedelta(days=i)).strftime("%Y-%m-%d")
            o = round(1.1000 + i * 0.0001, 5)
            h = round(o + 0.0100, 5)
            l = round(o - 0.0050, 5)
            c = round(o + 0.0050, 5)
            lines.append(f"{ts},{o},{h},{l},{c},1000,1.5")
        return header + "\n".join(lines)

    def test_basic_csv_loads_correct_count(self, loader):
        result = loader.load_csv_string(self._basic_csv(5), "EURUSD", "D1")
        assert result.success is True
        assert result.candles_fetched == 5
        assert result.candles_stored == 5

    def test_symbol_and_timeframe_assigned(self, loader):
        result = loader.load_csv_string(self._basic_csv(3), "GBPUSD", "H4")
        assert result.symbol == "GBPUSD"
        assert result.timeframe == "H4"

    def test_source_is_csv(self, loader):
        result = loader.load_csv_string(self._basic_csv(2), "EURUSD", "D1")
        assert result.source == DataSource.CSV

    def test_fetch_start_and_end_populated(self, loader):
        result = loader.load_csv_string(self._basic_csv(5), "EURUSD", "D1")
        assert result.fetch_start is not None
        assert result.fetch_end is not None
        assert result.fetch_end > result.fetch_start

    def test_no_errors_for_clean_csv(self, loader):
        result = loader.load_csv_string(self._basic_csv(5), "EURUSD", "D1")
        assert result.errors == []

    def test_empty_csv_returns_zero_candles(self, loader):
        result = loader.load_csv_string(
            "timestamp,open,high,low,close\n", "EURUSD", "D1"
        )
        assert result.candles_fetched == 0

    def test_invalid_row_skipped_with_error(self, loader):
        csv_content = (
            "timestamp,open,high,low,close\n"
            "2024-01-02,1.1,1.11,1.09,1.105\n"
            "INVALID_ROW,not,numbers,here,x\n"
            "2024-01-04,1.12,1.13,1.11,1.125\n"
        )
        result = loader.load_csv_string(csv_content, "EURUSD", "D1")
        assert result.candles_fetched == 2      # 2 valid
        assert len(result.errors) >= 1          # 1 bad row

    def test_high_below_low_row_skipped(self, loader):
        csv_content = (
            "timestamp,open,high,low,close\n"
            "2024-01-02,1.1,1.08,1.09,1.105\n"  # high < low!
        )
        result = loader.load_csv_string(csv_content, "EURUSD", "D1")
        assert result.candles_fetched == 0
        assert len(result.errors) >= 1

    def test_zero_price_row_skipped(self, loader):
        csv_content = (
            "timestamp,open,high,low,close\n"
            "2024-01-02,0.0,1.11,1.09,1.105\n"   # open = 0
        )
        result = loader.load_csv_string(csv_content, "EURUSD", "D1")
        assert result.candles_fetched == 0
        assert len(result.errors) >= 1

    def test_missing_ohlc_row_skipped(self, loader):
        csv_content = (
            "timestamp,open,high,low\n"           # no close column
            "2024-01-02,1.1,1.11,1.09\n"
        )
        result = loader.load_csv_string(csv_content, "EURUSD", "D1")
        assert result.candles_fetched == 0
        assert len(result.errors) >= 1

    def test_duplicate_timestamps_deduplicated(self, loader):
        csv_content = (
            "timestamp,open,high,low,close\n"
            "2024-01-02,1.1000,1.1100,1.0900,1.1050\n"
            "2024-01-02,1.2000,1.2100,1.1900,1.2050\n"  # duplicate date
            "2024-01-03,1.1100,1.1200,1.1000,1.1150\n"
        )
        result = loader.load_csv_string(csv_content, "EURUSD", "D1")
        assert result.candles_fetched == 2  # 3 parsed, 1 deduped → 2

    def test_alt_timestamp_column_name(self, loader):
        csv_content = (
            "date,open,high,low,close\n"
            "2024-01-02,1.1,1.11,1.09,1.105\n"
        )
        result = loader.load_csv_string(
            csv_content, "EURUSD", "D1", timestamp_col="date"
        )
        assert result.candles_fetched == 1

    def test_tab_delimiter(self, loader):
        csv_content = (
            "timestamp\topen\thigh\tlow\tclose\n"
            "2024-01-02\t1.1\t1.11\t1.09\t1.105\n"
        )
        result = loader.load_csv_string(
            csv_content, "EURUSD", "D1", delimiter="\t"
        )
        assert result.candles_fetched == 1

    def test_volume_optional(self, loader):
        csv_content = (
            "timestamp,open,high,low,close\n"
            "2024-01-02,1.1,1.11,1.09,1.105\n"
        )
        result = loader.load_csv_string(csv_content, "EURUSD", "D1")
        assert result.candles_fetched == 1

    def test_large_csv_correct_count(self, loader):
        result = loader.load_csv_string(self._basic_csv(500), "EURUSD", "D1")
        assert result.candles_fetched == 500


# ===========================================================================
# CSVDataLoader — load_csv (file-based)
# ===========================================================================

class TestCSVDataLoaderFile:

    @pytest.fixture
    def loader(self):
        return CSVDataLoader(candle_store=None)

    def test_nonexistent_file_returns_failure(self, loader):
        result = loader.load_csv("/nonexistent/path/data.csv", "EURUSD", "D1")
        assert result.success is False
        assert result.candles_fetched == 0
        assert any("not found" in e.lower() or "nonexistent" in e for e in result.errors)

    def test_valid_file_loads_correctly(self, loader, tmp_path):
        csv_path = tmp_path / "eurusd_d1.csv"
        csv_path.write_text(
            "timestamp,open,high,low,close,volume\n"
            "2024-01-02,1.1000,1.1100,1.0900,1.1050,1000\n"
            "2024-01-03,1.1050,1.1150,1.0950,1.1100,1100\n"
        )
        result = loader.load_csv(str(csv_path), "EURUSD", "D1")
        assert result.success is True
        assert result.candles_fetched == 2

    def test_path_object_accepted(self, loader, tmp_path):
        csv_path = tmp_path / "data.csv"
        csv_path.write_text(
            "timestamp,open,high,low,close\n"
            "2024-03-15,1.08,1.09,1.07,1.085\n"
        )
        result = loader.load_csv(csv_path, "EURUSD", "D1")   # Path object
        assert result.candles_fetched == 1

    def test_utf8_bom_handled(self, loader, tmp_path):
        """Some MT4/MT5 exports write UTF-8 BOM at the start.

        Write with encoding='utf-8-sig' which prepends exactly one BOM byte.
        The loader opens with 'utf-8-sig' which strips it, leaving a clean
        'timestamp' column header.
        """
        csv_path = tmp_path / "bom.csv"
        # Do NOT embed a manual '\ufeff' — utf-8-sig already adds the BOM.
        csv_path.write_text(
            "timestamp,open,high,low,close\n"
            "2024-01-02,1.1,1.11,1.09,1.105\n",
            encoding="utf-8-sig",
        )
        result = loader.load_csv(str(csv_path), "EURUSD", "D1")
        assert result.candles_fetched == 1


# ===========================================================================
# CSVDataLoader + CandleStore integration
# ===========================================================================

class TestCSVLoaderWithCandleStore:
    """Tests that CSVDataLoader correctly persists candles via CandleStore."""

    @pytest.fixture
    def db_setup(self):
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from src.db.models import Base
        from src.db.session import init_db
        eng = create_engine("sqlite:///:memory:")
        init_db(eng)
        factory = sessionmaker(bind=eng, autocommit=False, autoflush=False)
        session = factory()
        store = CandleStore(session)
        loader = CSVDataLoader(candle_store=store)
        yield store, loader, session
        session.close()

    # Import at top of fixture body (avoid module-level import)
    def _import_candle_store(self):
        from src.db.candle_store import CandleStore
        return CandleStore

    def _csv(self, n=5) -> str:
        lines = ["timestamp,open,high,low,close,volume,spread"]
        base = datetime(2024, 1, 2)
        for i in range(n):
            ts = (base + timedelta(days=i)).strftime("%Y-%m-%d")
            o = round(1.1 + i * 0.0001, 5)
            lines.append(f"{ts},{o},{round(o+0.01,5)},{round(o-0.005,5)},{round(o+0.005,5)},1000,1.5")
        return "\n".join(lines)

    def test_candles_stored_in_db(self, db_setup):
        store, loader, session = db_setup
        result = loader.load_csv_string(self._csv(5), "EURUSD", "D1")
        session.commit()
        assert result.candles_stored == 5
        assert store.get_candle_count("EURUSD", "D1") == 5

    def test_upsert_on_reload(self, db_setup):
        store, loader, session = db_setup
        loader.load_csv_string(self._csv(5), "EURUSD", "D1")
        session.commit()
        loader.load_csv_string(self._csv(5), "EURUSD", "D1")
        session.commit()
        # Upsert → still 5, not 10
        assert store.get_candle_count("EURUSD", "D1") == 5

    def test_retrieval_order_is_ascending(self, db_setup):
        store, loader, session = db_setup
        loader.load_csv_string(self._csv(10), "EURUSD", "D1")
        session.commit()
        candles = store.get_candles(
            "EURUSD", "D1",
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 12, 31, tzinfo=timezone.utc),
        )
        timestamps = [c.timestamp for c in candles]
        assert timestamps == sorted(timestamps)

    def test_latest_n_candles_returns_most_recent(self, db_setup):
        store, loader, session = db_setup
        loader.load_csv_string(self._csv(20), "EURUSD", "D1")
        session.commit()
        latest = store.get_latest_n_candles("EURUSD", "D1", 5)
        assert len(latest) == 5
        # The last candle in latest should be the most recent overall
        all_candles = store.get_candles(
            "EURUSD", "D1",
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 12, 31, tzinfo=timezone.utc),
        )
        assert latest[-1].timestamp == all_candles[-1].timestamp

    def test_date_range_returned_correctly(self, db_setup):
        store, loader, session = db_setup
        loader.load_csv_string(self._csv(10), "EURUSD", "D1")
        session.commit()
        dr = store.get_date_range("EURUSD", "D1")
        assert dr is not None
        assert dr[0] < dr[1]

    def test_gap_detection_with_large_gap(self, db_setup):
        store, loader, session = db_setup
        # Two candles far apart
        csv_content = (
            "timestamp,open,high,low,close\n"
            "2024-01-02,1.1,1.11,1.09,1.105\n"
            "2024-03-01,1.2,1.21,1.19,1.205\n"   # ~59 day gap
        )
        loader.load_csv_string(csv_content, "EURUSD", "D1")
        session.commit()
        gaps = store.candle_gap_check("EURUSD", "D1")
        assert len(gaps) >= 1
        assert gaps[0]["missing_bars_estimate"] > 0

    def test_symbol_isolation(self, db_setup):
        store, loader, session = db_setup
        loader.load_csv_string(self._csv(5), "EURUSD", "D1")
        loader.load_csv_string(self._csv(3), "GBPUSD", "D1")
        session.commit()
        assert store.get_candle_count("EURUSD", "D1") == 5
        assert store.get_candle_count("GBPUSD", "D1") == 3


# ===========================================================================
# candle_data_to_orm / candle_orm_to_data conversions
# ===========================================================================

class TestCandleConversions:

    def test_candle_data_to_orm_fields_preserved(self):
        cd = _make_cd(ts=_dt(2024, 6, 1), o=1.10, h=1.11, l=1.09, c=1.105)
        orm = candle_data_to_orm(cd)
        assert orm.symbol    == "EURUSD"
        assert orm.timeframe == "D1"
        assert orm.open      == pytest.approx(1.10)
        assert orm.high      == pytest.approx(1.11)
        assert orm.low       == pytest.approx(1.09)
        assert orm.close     == pytest.approx(1.105)
        assert orm.volume    == pytest.approx(1000.0)
        assert orm.spread    == pytest.approx(1.5)

    def test_candle_data_to_orm_naive_ts_becomes_utc(self):
        naive_ts = datetime(2024, 6, 1)   # no tzinfo
        cd = _make_cd(ts=naive_ts)
        orm = candle_data_to_orm(cd)
        assert orm.timestamp.tzinfo is not None

    def test_candle_orm_to_data_round_trip(self):
        cd = _make_cd(ts=_dt(2024, 6, 1), o=1.10, h=1.11, l=1.09, c=1.105, vol=500.0)
        orm = candle_data_to_orm(cd)
        cd2 = candle_orm_to_data(orm)
        assert cd2.symbol    == "EURUSD"
        assert cd2.open      == pytest.approx(1.10)
        assert cd2.volume    == pytest.approx(500.0)

    def test_none_volume_in_orm_defaults_to_zero(self):
        from src.db.models import Candle
        orm = Candle(
            symbol="EURUSD", timeframe="D1",
            timestamp=_dt(2024, 1, 1),
            open=1.1, high=1.11, low=1.09, close=1.105,
            volume=None, spread=None,
        )
        cd = candle_orm_to_data(orm)
        assert cd.volume == 0.0
