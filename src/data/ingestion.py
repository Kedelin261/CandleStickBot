"""
M01 — Data Ingestion Engine
Fetches OHLCV data from MT5 (when available) or CSV files.
Normalizes, validates, and stores candles via CandleStore (M02).

Phase 1 scope: EURUSD D1 primary, multi-timeframe agnostic internally.
MT5 is optional — if the package is unavailable the engine gracefully
falls back to CSV ingestion without raising an ImportError.

Architecture Note:
  - This module ONLY fetches/stores data. It never places orders.
  - All MT5 interaction is guarded by try/import so the rest of the
    system works in test environments without a broker terminal.

Version: 3.1 (Phase 1 Sprint 1)
"""

from __future__ import annotations

import csv
import io
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from src.db.candle_store import CandleStore, candle_from_dict
from src.db.models import Candle
from src.types import CandleData

logger = logging.getLogger("candlestickbot.data.ingestion")


# ---------------------------------------------------------------------------
# ENUMERATIONS
# ---------------------------------------------------------------------------

class DataSource(str, Enum):
    MT5  = "MT5"
    CSV  = "CSV"
    MOCK = "MOCK"


class ConnectionStatus(str, Enum):
    DISCONNECTED = "DISCONNECTED"
    CONNECTING   = "CONNECTING"
    CONNECTED    = "CONNECTED"
    ERROR        = "ERROR"


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

@dataclass
class MT5Config:
    """MT5 connection configuration (credentials never logged)."""
    login:            int
    password:         str
    server:           str
    terminal_path:    Optional[str] = None
    timeout_ms:       int           = 60_000
    retry_attempts:   int           = 3
    retry_delay_sec:  float         = 2.0


# ---------------------------------------------------------------------------
# RESULT DTO
# ---------------------------------------------------------------------------

@dataclass
class FetchResult:
    """Outcome of a data fetch/load operation."""
    success:         bool
    symbol:          str
    timeframe:       str
    candles_fetched: int
    candles_stored:  int
    source:          DataSource
    errors:          List[str]         = field(default_factory=list)
    gaps_detected:   int               = 0
    fetch_start:     Optional[datetime] = None
    fetch_end:       Optional[datetime] = None

    @property
    def has_errors(self) -> bool:
        return bool(self.errors)


# ---------------------------------------------------------------------------
# VALIDATION HELPERS
# ---------------------------------------------------------------------------

# Accepted date/time formats for CSV timestamp parsing (tried in order)
_TS_FORMATS: List[str] = [
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y",
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y",
]

# MT5 integer timeframe codes
_MT5_TF_MAP: Dict[str, int] = {
    "M1":  1,
    "M5":  5,
    "M15": 15,
    "M30": 30,
    "H1":  16385,
    "H4":  16388,
    "D1":  16408,
    "W1":  32769,
    "MN1": 49153,
}


def _parse_timestamp(value: Any, fmt: Optional[str] = None) -> datetime:
    """
    Parse a timestamp value into a timezone-aware datetime (UTC).

    Accepts:
        - datetime (with or without tzinfo)
        - int/float (Unix epoch)
        - str (tries multiple formats)

    Raises:
        ValueError: if the value cannot be parsed.
    """
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)

    if isinstance(value, str):
        value = value.strip()
        formats = [fmt] if fmt else _TS_FORMATS
        for f in formats:
            if f is None:
                continue
            try:
                dt = datetime.strptime(value, f)
                return dt.replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        raise ValueError(f"Cannot parse timestamp '{value}' — tried formats: {formats}")

    raise ValueError(f"Unsupported timestamp type: {type(value)}")


def normalize_candle_dict(
    raw: Dict[str, Any],
    symbol: str,
    timeframe: str,
    timestamp_col: str = "timestamp",
    date_format: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Normalize a raw dictionary row into the canonical candle data contract.

    Canonical fields after normalization:
        symbol, timeframe, timestamp (UTC datetime),
        open, high, low, close, volume (≥ 0), spread (≥ 0 or None)

    Args:
        raw:           Raw row dict (e.g. from CSV reader or MT5 structured array).
        symbol:        Symbol to assign (will be uppercased).
        timeframe:     Timeframe to assign (will be uppercased).
        timestamp_col: Name of the column that contains the timestamp.
        date_format:   Optional strptime format string for parsing timestamps.

    Returns:
        Normalized dict ready for candle_from_dict().

    Raises:
        KeyError:   If a required OHLC field is absent.
        ValueError: If OHLC values are invalid.
    """
    # ---------- timestamp ----------
    ts_raw = raw.get(timestamp_col) or raw.get("time") or raw.get("date")
    if ts_raw is None:
        raise KeyError(
            f"Timestamp column '{timestamp_col}' not found in row: {list(raw.keys())}"
        )
    timestamp = _parse_timestamp(ts_raw, date_format)

    # ---------- OHLCV ----------
    def _float(key: str) -> float:
        val = raw.get(key)
        if val is None:
            raise KeyError(f"Required OHLC field '{key}' missing from row")
        try:
            return float(val)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Field '{key}' = {val!r} is not numeric") from exc

    open_  = _float("open")
    high   = _float("high")
    low    = _float("low")
    close  = _float("close")

    volume_raw = raw.get("volume") or raw.get("tick_volume") or raw.get("vol") or 0
    try:
        volume = float(volume_raw)
    except (TypeError, ValueError):
        volume = 0.0
    if volume < 0:
        volume = 0.0

    spread_raw = raw.get("spread")
    spread = None
    if spread_raw is not None:
        try:
            spread = float(spread_raw)
            if spread < 0:
                spread = 0.0
        except (TypeError, ValueError):
            spread = None

    return {
        "symbol":    symbol.upper(),
        "timeframe": timeframe.upper(),
        "timestamp": timestamp,
        "open":      open_,
        "high":      high,
        "low":       low,
        "close":     close,
        "volume":    volume,
        "spread":    spread,
    }


def validate_ohlcv(row: Dict[str, Any]) -> List[str]:
    """
    Validate a normalized candle dict.  Returns a list of error strings
    (empty list means the candle is valid).

    Rules:
        1. open, high, low, close must be numeric and > 0.
        2. high >= open, close, low.
        3. low  <= open, close, high.
        4. volume >= 0.
        5. spread >= 0 (if present).
        6. timestamp must be a datetime.
    """
    errors: List[str] = []

    for field in ("open", "high", "low", "close"):
        v = row.get(field)
        if v is None:
            errors.append(f"Missing field: {field}")
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            errors.append(f"Field '{field}' is not numeric: {v!r}")
            continue
        if fv <= 0:
            errors.append(f"Field '{field}' must be > 0, got {fv}")

    if not errors:
        high  = float(row["high"])
        low   = float(row["low"])
        open_ = float(row["open"])
        close = float(row["close"])

        if high < low:
            errors.append(f"high ({high}) < low ({low})")
        if high < open_:
            errors.append(f"high ({high}) < open ({open_})")
        if high < close:
            errors.append(f"high ({high}) < close ({close})")
        if low > open_:
            errors.append(f"low ({low}) > open ({open_})")
        if low > close:
            errors.append(f"low ({low}) > close ({close})")

    vol = row.get("volume")
    if vol is not None:
        try:
            if float(vol) < 0:
                errors.append(f"volume must be >= 0, got {vol}")
        except (TypeError, ValueError):
            errors.append(f"volume is not numeric: {vol!r}")

    spread = row.get("spread")
    if spread is not None:
        try:
            if float(spread) < 0:
                errors.append(f"spread must be >= 0, got {spread}")
        except (TypeError, ValueError):
            errors.append(f"spread is not numeric: {spread!r}")

    if not isinstance(row.get("timestamp"), datetime):
        errors.append("timestamp must be a datetime object")

    return errors


def deduplicate_candles(candles: List[CandleData]) -> List[CandleData]:
    """
    Remove duplicate candles (same symbol/timeframe/timestamp).
    When duplicates exist, keeps the LAST occurrence (most recently seen wins).
    Returns list sorted ascending by timestamp.
    """
    seen: Dict[Tuple[str, str, datetime], CandleData] = {}
    for c in candles:
        key = (c.symbol.upper(), c.timeframe.upper(), c.timestamp)
        seen[key] = c  # last-wins
    return sorted(seen.values(), key=lambda c: c.timestamp)


def sort_candles_ascending(candles: List[CandleData]) -> List[CandleData]:
    """Sort candles oldest-first by timestamp."""
    return sorted(candles, key=lambda c: c.timestamp)


# ---------------------------------------------------------------------------
# MT5 DATA INGESTION ENGINE
# ---------------------------------------------------------------------------

class DataIngestionEngine:
    """
    M01 — MetaTrader 5 Data Ingestion Engine.

    Responsibilities
    ----------------
    - Connect / disconnect from the MT5 terminal.
    - Fetch historical OHLCV bars (by count or date range).
    - Normalize raw MT5 rates arrays → CandleData DTOs.
    - Validate OHLCV integrity before persistence.
    - Store candles via CandleStore (M02).
    - Log all operations via AuditLogger (M13) when provided.

    MT5 availability
    ----------------
    The MetaTrader5 Python package only works on Windows with a running
    terminal.  The engine checks for its availability at import time; if
    not installed ``connect()`` returns False and every fetch method
    returns a failed ``FetchResult`` with a clear error message.  This
    allows tests and development on Linux/macOS without MT5.

    Phase 1 scope: EURUSD D1 primary (no artificial limitation here).
    """

    # Detect MT5 availability once at class definition time
    _mt5_available: bool = False
    try:
        import MetaTrader5 as _MT5_MODULE  # type: ignore[import]
        _mt5_available = True
    except (ImportError, ModuleNotFoundError):
        _MT5_MODULE = None  # type: ignore[assignment]

    def __init__(
        self,
        config: MT5Config,
        candle_store: Optional[CandleStore] = None,
        audit_logger=None,
    ) -> None:
        self.config       = config
        self.candle_store = candle_store
        self.audit_logger = audit_logger
        self._status      = ConnectionStatus.DISCONNECTED
        self._mt5         = None   # live reference to MetaTrader5 module

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    @property
    def status(self) -> ConnectionStatus:
        return self._status

    @property
    def is_connected(self) -> bool:
        return self._status == ConnectionStatus.CONNECTED

    def connect(self) -> bool:
        """
        Connect to the MT5 terminal.

        Returns True on success, False on failure.
        When the MetaTrader5 package is not installed this always returns
        False with status ERROR — use CSVDataLoader as fallback.
        """
        if not self._mt5_available:
            logger.warning(
                "MetaTrader5 package not available — running without MT5 connectivity. "
                "Use CSVDataLoader for data ingestion in this environment."
            )
            self._status = ConnectionStatus.ERROR
            return False

        self._status = ConnectionStatus.CONNECTING
        mt5 = self.__class__._MT5_MODULE  # type: ignore[attr-defined]

        init_kwargs: Dict[str, Any] = {
            "login":    self.config.login,
            "password": self.config.password,
            "server":   self.config.server,
            "timeout":  self.config.timeout_ms,
        }
        if self.config.terminal_path:
            init_kwargs["path"] = self.config.terminal_path

        for attempt in range(1, self.config.retry_attempts + 1):
            try:
                ok = mt5.initialize(**init_kwargs)
            except Exception as exc:
                logger.error("MT5 initialize() raised exception: %s", exc)
                ok = False

            if ok:
                self._mt5 = mt5
                self._status = ConnectionStatus.CONNECTED
                logger.info(
                    "Connected to MT5 server=%s login=%s",
                    self.config.server,
                    self.config.login,
                )
                if self.audit_logger:
                    self.audit_logger.log_system_event(
                        "MT5_CONNECTED",
                        {"server": self.config.server, "login": self.config.login},
                    )
                return True

            err = mt5.last_error()
            logger.warning(
                "MT5 connect attempt %d/%d failed: %s",
                attempt,
                self.config.retry_attempts,
                err,
            )
            if attempt < self.config.retry_attempts:
                import time
                time.sleep(self.config.retry_delay_sec)

        self._status = ConnectionStatus.ERROR
        logger.error(
            "MT5 connection failed after %d attempts", self.config.retry_attempts
        )
        return False

    def disconnect(self) -> None:
        """Shut down MT5 connection and release resources."""
        if self._mt5 is not None:
            try:
                self._mt5.shutdown()
            except Exception:
                pass
            self._mt5 = None
        self._status = ConnectionStatus.DISCONNECTED
        logger.info("MT5 disconnected")

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    def fetch_candles(
        self,
        symbol: str,
        timeframe: str,
        count: int = 500,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> FetchResult:
        """
        Fetch OHLCV bars from MT5.

        If both *start* and *end* are provided uses
        ``copy_rates_range``; otherwise fetches the last *count* bars
        with ``copy_rates_from_pos``.

        Returns a FetchResult (success=False if not connected or MT5 error).
        """
        if not self.is_connected:
            return FetchResult(
                success=False, symbol=symbol, timeframe=timeframe,
                candles_fetched=0, candles_stored=0, source=DataSource.MT5,
                errors=["Not connected to MT5 — call connect() first"],
            )

        mt5_tf = _MT5_TF_MAP.get(timeframe.upper())
        if mt5_tf is None:
            return FetchResult(
                success=False, symbol=symbol, timeframe=timeframe,
                candles_fetched=0, candles_stored=0, source=DataSource.MT5,
                errors=[f"Unknown timeframe '{timeframe}'. Supported: {list(_MT5_TF_MAP)}"],
            )

        try:
            if start is not None and end is not None:
                rates = self._mt5.copy_rates_range(symbol, mt5_tf, start, end)
            else:
                rates = self._mt5.copy_rates_from_pos(symbol, mt5_tf, 0, count)
        except Exception as exc:
            return FetchResult(
                success=False, symbol=symbol, timeframe=timeframe,
                candles_fetched=0, candles_stored=0, source=DataSource.MT5,
                errors=[f"MT5 fetch error: {exc}"],
            )

        if rates is None or len(rates) == 0:
            err = self._mt5.last_error() if self._mt5 else "unknown"
            return FetchResult(
                success=False, symbol=symbol, timeframe=timeframe,
                candles_fetched=0, candles_stored=0, source=DataSource.MT5,
                errors=[f"MT5 returned no data: {err}"],
            )

        # Normalize and validate
        candle_dicts, errors = self._normalize_rates(symbol, timeframe, rates)
        valid_dicts = [d for d in candle_dicts if not validate_ohlcv(d)]
        invalid_count = len(candle_dicts) - len(valid_dicts)
        if invalid_count:
            logger.warning(
                "Discarded %d invalid candles for %s/%s", invalid_count, symbol, timeframe
            )

        candle_objs = [candle_from_dict(d) for d in valid_dicts]

        stored = 0
        if self.candle_store and candle_objs:
            stored = self.candle_store.store_candles(candle_objs)

        gaps = self.candle_store.candle_gap_check(symbol, timeframe) if self.candle_store else []

        if self.audit_logger:
            self.audit_logger.log_data_fetch(
                symbol, timeframe, len(candle_objs), "MT5",
                success=True, error=None
            )

        return FetchResult(
            success=True,
            symbol=symbol,
            timeframe=timeframe,
            candles_fetched=len(candle_objs),
            candles_stored=stored,
            source=DataSource.MT5,
            errors=errors,
            gaps_detected=len(gaps),
            fetch_start=candle_objs[0].timestamp if candle_objs else None,
            fetch_end=candle_objs[-1].timestamp if candle_objs else None,
        )

    def fetch_latest(self, symbol: str, timeframe: str, n: int = 100) -> FetchResult:
        """Fetch the N most recent bars (convenience wrapper)."""
        return self.fetch_candles(symbol, timeframe, count=n)

    def fetch_historical(
        self,
        symbol:    str,
        timeframe: str,
        start:     datetime,
        end:       Optional[datetime] = None,
    ) -> FetchResult:
        """Backfill historical data from *start* to *end* (or now)."""
        end = end or datetime.now(timezone.utc)
        return self.fetch_candles(symbol, timeframe, start=start, end=end)

    # backward-compat alias
    fetch_historical_candles = fetch_historical

    # ------------------------------------------------------------------
    # Symbol info / server time
    # ------------------------------------------------------------------

    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Return MT5 symbol properties dict or None if not connected."""
        if not self.is_connected or self._mt5 is None:
            return None
        try:
            info = self._mt5.symbol_info(symbol)
            return info._asdict() if info else None
        except Exception:
            return None

    def get_server_time(self) -> Optional[datetime]:
        """Return current MT5 server time (UTC) or None if not connected."""
        if not self.is_connected or self._mt5 is None:
            return None
        try:
            tick = self._mt5.symbol_info_tick("EURUSD")
            if tick:
                return datetime.fromtimestamp(tick.time, tz=timezone.utc)
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalize_rates(
        self,
        symbol:    str,
        timeframe: str,
        rates,               # numpy structured array from MT5
    ) -> Tuple[List[Dict], List[str]]:
        """Convert MT5 rates array to list of normalized dicts. Returns (dicts, errors)."""
        results: List[Dict] = []
        errors:  List[str]  = []

        for row in rates:
            try:
                normalized = normalize_candle_dict(
                    {
                        "time":        row["time"],
                        "open":        row["open"],
                        "high":        row["high"],
                        "low":         row["low"],
                        "close":       row["close"],
                        "tick_volume": row.get("tick_volume", 0),
                        "spread":      row.get("spread", 0),
                    },
                    symbol=symbol,
                    timeframe=timeframe,
                    timestamp_col="time",
                )
                results.append(normalized)
            except (KeyError, ValueError) as exc:
                errors.append(str(exc))

        return results, errors


# ---------------------------------------------------------------------------
# CSV DATA LOADER
# ---------------------------------------------------------------------------

class CSVDataLoader:
    """
    Load OHLCV candles from CSV files — primary fallback when MT5 is unavailable.

    Supported CSV formats
    ---------------------
    Required columns (case-insensitive, flexible naming):
        timestamp / time / date  — bar open time
        open, high, low, close   — OHLC prices
    Optional columns:
        volume / tick_volume / vol
        spread

    All recognized date/time formats are listed in ``_TS_FORMATS``.
    Rows that fail validation are skipped and counted as errors.
    """

    def __init__(self, candle_store: Optional[CandleStore] = None) -> None:
        self.candle_store = candle_store

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_csv(
        self,
        filepath: Union[str, Path],
        symbol:        str,
        timeframe:     str,
        timestamp_col: str           = "timestamp",
        date_format:   Optional[str] = None,
        delimiter:     str           = ",",
        skip_rows:     int           = 0,
    ) -> FetchResult:
        """
        Parse a CSV file and store candles in CandleStore.

        Args:
            filepath:      Path to the CSV file (str or Path).
            symbol:        Symbol to assign all loaded candles.
            timeframe:     Timeframe to assign.
            timestamp_col: Header name of the timestamp column.
            date_format:   strptime format override (auto-detected if None).
            delimiter:     CSV field delimiter (default ",").
            skip_rows:     Number of header rows to skip beyond the first.

        Returns:
            FetchResult describing success/failure and counts.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            return FetchResult(
                success=False, symbol=symbol, timeframe=timeframe,
                candles_fetched=0, candles_stored=0, source=DataSource.CSV,
                errors=[f"File not found: {filepath}"],
            )

        try:
            with open(filepath, newline="", encoding="utf-8-sig") as fh:
                return self._parse_csv_stream(
                    fh, symbol, timeframe, timestamp_col, date_format, delimiter, skip_rows
                )
        except OSError as exc:
            return FetchResult(
                success=False, symbol=symbol, timeframe=timeframe,
                candles_fetched=0, candles_stored=0, source=DataSource.CSV,
                errors=[f"Cannot open file '{filepath}': {exc}"],
            )

    def load_csv_string(
        self,
        content:       str,
        symbol:        str,
        timeframe:     str,
        timestamp_col: str           = "timestamp",
        date_format:   Optional[str] = None,
        delimiter:     str           = ",",
    ) -> FetchResult:
        """
        Parse CSV content from a string (useful for testing without real files).

        Args:
            content: Raw CSV text.
            (others same as load_csv)
        """
        stream = io.StringIO(content)
        return self._parse_csv_stream(
            stream, symbol, timeframe, timestamp_col, date_format, delimiter, 0
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_csv_stream(
        self,
        stream,
        symbol:        str,
        timeframe:     str,
        timestamp_col: str,
        date_format:   Optional[str],
        delimiter:     str,
        skip_rows:     int,
    ) -> FetchResult:
        """Core parsing logic shared by load_csv and load_csv_string."""
        errors:  List[str]  = []
        candles: List[Candle] = []

        reader = csv.DictReader(stream, delimiter=delimiter)

        # Skip extra header rows if requested
        for _ in range(skip_rows):
            try:
                next(reader)
            except StopIteration:
                break

        # Normalise header names (lowercase, strip whitespace)
        def _norm_headers(row: Dict) -> Dict:
            return {k.lower().strip(): v for k, v in row.items()}

        row_num = 1
        for raw_row in reader:
            row_num += 1
            row = _norm_headers(raw_row)

            # Allow flexible timestamp column name
            ts_col = timestamp_col.lower().strip()
            if ts_col not in row:
                # Try common aliases
                for alias in ("time", "date", "datetime", "open_time", "bar_time"):
                    if alias in row:
                        ts_col = alias
                        break

            try:
                normalized = normalize_candle_dict(
                    row,
                    symbol=symbol,
                    timeframe=timeframe,
                    timestamp_col=ts_col,
                    date_format=date_format,
                )
            except (KeyError, ValueError) as exc:
                errors.append(f"Row {row_num}: normalization error — {exc}")
                continue

            validation_errors = validate_ohlcv(normalized)
            if validation_errors:
                errors.append(
                    f"Row {row_num}: validation failed — {'; '.join(validation_errors)}"
                )
                continue

            try:
                candle = candle_from_dict(normalized)
                candles.append(candle)
            except (KeyError, ValueError) as exc:
                errors.append(f"Row {row_num}: candle_from_dict error — {exc}")
                continue

        # Remove in-memory duplicates before persistence
        candles = _dedup_candle_objs(candles)

        stored = 0
        if self.candle_store and candles:
            stored = self.candle_store.store_candles(candles)
        elif not self.candle_store:
            stored = len(candles)   # caller gets count even without persistence

        success = len(candles) > 0 or not errors

        if errors:
            logger.warning(
                "CSV load for %s/%s: %d rows OK, %d errors",
                symbol, timeframe, len(candles), len(errors)
            )

        return FetchResult(
            success=success,
            symbol=symbol,
            timeframe=timeframe,
            candles_fetched=len(candles),
            candles_stored=stored,
            source=DataSource.CSV,
            errors=errors,
            fetch_start=candles[0].timestamp if candles else None,
            fetch_end=candles[-1].timestamp if candles else None,
        )


# ---------------------------------------------------------------------------
# CANDLE DATA INTEGRITY VALIDATOR (standalone, module-level)
# ---------------------------------------------------------------------------

def validate_data_integrity(candles: List[CandleData]) -> Dict[str, Any]:
    """
    Validate a list of CandleData DTOs before storage.

    Checks:
        1. All required fields present and numeric.
        2. OHLC relationships valid (H≥O,C,L; L≤O,C,H).
        3. Volume and spread are non-negative.
        4. No duplicate timestamps for same symbol/timeframe.
        5. Candles are sorted ascending by timestamp.

    Args:
        candles: List of CandleData objects.

    Returns:
        Dict with keys:
            total, valid, invalid, duplicate_count, sort_issues,
            errors (list of dicts with index + messages)
    """
    errors: List[Dict] = []
    seen_ts: Dict[Tuple[str, str, datetime], int] = {}
    prev_ts: Optional[datetime] = None
    sort_issues = 0

    for idx, c in enumerate(candles):
        row_errors: List[str] = []

        # Convert to dict for reuse of validate_ohlcv
        row: Dict[str, Any] = {
            "open":      c.open,
            "high":      c.high,
            "low":       c.low,
            "close":     c.close,
            "volume":    c.volume,
            "spread":    c.spread,
            "timestamp": c.timestamp,
        }
        row_errors.extend(validate_ohlcv(row))

        # Duplicate check
        key = (c.symbol.upper(), c.timeframe.upper(), c.timestamp)
        if key in seen_ts:
            row_errors.append(
                f"Duplicate timestamp {c.timestamp} (first seen at index {seen_ts[key]})"
            )
        else:
            seen_ts[key] = idx

        # Sort check
        if prev_ts is not None and c.timestamp < prev_ts:
            sort_issues += 1
            row_errors.append(
                f"Out-of-order: {c.timestamp} < previous {prev_ts}"
            )
        prev_ts = c.timestamp

        if row_errors:
            errors.append({"index": idx, "timestamp": c.timestamp, "errors": row_errors})

    valid   = len(candles) - len(errors)
    invalid = len(errors)

    return {
        "total":           len(candles),
        "valid":           valid,
        "invalid":         invalid,
        "duplicate_count": len(candles) - len(seen_ts),
        "sort_issues":     sort_issues,
        "is_valid":        invalid == 0 and sort_issues == 0,
        "errors":          errors,
    }


# ---------------------------------------------------------------------------
# CANDLE DATA ↔ ORM CONVERSION HELPERS
# ---------------------------------------------------------------------------

def candle_data_to_orm(c: CandleData) -> Candle:
    """Convert a CandleData DTO to a Candle ORM object (not persisted)."""
    return Candle(
        symbol    = c.symbol.upper(),
        timeframe = c.timeframe.upper(),
        timestamp = c.timestamp if c.timestamp.tzinfo else
                    c.timestamp.replace(tzinfo=timezone.utc),
        open      = c.open,
        high      = c.high,
        low       = c.low,
        close     = c.close,
        volume    = c.volume,
        spread    = c.spread,
    )


def candle_orm_to_data(c: Candle) -> CandleData:
    """Convert a Candle ORM object back to a CandleData DTO."""
    return CandleData(
        symbol    = c.symbol,
        timeframe = c.timeframe,
        timestamp = c.timestamp,
        open      = c.open,
        high      = c.high,
        low       = c.low,
        close     = c.close,
        volume    = c.volume if c.volume is not None else 0.0,
        spread    = c.spread,
    )


# ---------------------------------------------------------------------------
# INTERNAL DEDUP HELPER
# ---------------------------------------------------------------------------

def _dedup_candle_objs(candles: List[Candle]) -> List[Candle]:
    """
    Remove duplicate Candle ORM objects (same symbol/timeframe/timestamp).
    Last-occurrence wins.  Returns a new list sorted ascending by timestamp.
    """
    seen: Dict[Tuple[str, str, datetime], Candle] = {}
    for c in candles:
        key = (c.symbol.upper(), c.timeframe.upper(), c.timestamp)
        seen[key] = c
    return sorted(seen.values(), key=lambda c: c.timestamp)


# ---------------------------------------------------------------------------
# FACTORY FUNCTION
# ---------------------------------------------------------------------------

def create_data_ingestion_engine(
    mt5_login:      int,
    mt5_password:   str,
    mt5_server:     str,
    terminal_path:  Optional[str]         = None,
    candle_store:   Optional[CandleStore] = None,
    audit_logger                          = None,
) -> DataIngestionEngine:
    """
    Factory: create a configured (but not yet connected) DataIngestionEngine.

    Args:
        mt5_login:     MT5 account login number.
        mt5_password:  MT5 account password.
        mt5_server:    MT5 broker server name.
        terminal_path: Optional path to MT5 terminal executable.
        candle_store:  CandleStore for persistence (can be None for dry-run).
        audit_logger:  AuditLogger instance (optional).
    """
    config = MT5Config(
        login         = mt5_login,
        password      = mt5_password,
        server        = mt5_server,
        terminal_path = terminal_path,
    )
    return DataIngestionEngine(
        config       = config,
        candle_store = candle_store,
        audit_logger = audit_logger,
    )
