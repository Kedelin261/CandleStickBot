"""
M02 — Candle Storage: CandleStore
Core CRUD operations for OHLCV candle data.
Version: 3.1 (Phase 1 Sprint 1)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import and_, delete, distinct, func, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from .models import Candle


class CandleStore:
    """
    Provides all CRUD operations for OHLCV candle data.

    Responsibilities (M02):
    - store_candles: Upsert batch of candles (no duplicates)
    - get_candles: Range query returning ordered list
    - get_latest_n_candles: Most recent N candles for live analysis
    - candle_gap_check: Detect missing bars in stored data
    """

    def __init__(self, session: Session):
        self.session = session

    # -----------------------------------------------------------------------
    # WRITE OPERATIONS
    # -----------------------------------------------------------------------

    def store_candles(self, candles: List[Candle]) -> int:
        """
        Upsert a batch of candles. Duplicate (symbol, timeframe, timestamp)
        tuples are updated rather than raising an error.

        Args:
            candles: List of Candle ORM objects

        Returns:
            Number of candles stored/updated
        """
        if not candles:
            return 0

        # Build upsert data
        rows = [
            {
                "symbol":    c.symbol,
                "timeframe": c.timeframe,
                "timestamp": c.timestamp,
                "open":      c.open,
                "high":      c.high,
                "low":       c.low,
                "close":     c.close,
                "volume":    c.volume,
                "spread":    c.spread,
            }
            for c in candles
        ]

        # SQLite upsert (INSERT OR REPLACE semantics)
        stmt = sqlite_insert(Candle).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["symbol", "timeframe", "timestamp"],
            set_={
                "open":   stmt.excluded.open,
                "high":   stmt.excluded.high,
                "low":    stmt.excluded.low,
                "close":  stmt.excluded.close,
                "volume": stmt.excluded.volume,
                "spread": stmt.excluded.spread,
            },
        )
        self.session.execute(stmt)
        return len(rows)

    # -----------------------------------------------------------------------
    # READ OPERATIONS
    # -----------------------------------------------------------------------

    def get_candles(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> List[Candle]:
        """
        Retrieve candles for a symbol/timeframe within a date range.
        Returns candles ordered by timestamp ascending.

        Args:
            symbol: Trading pair (e.g. "EURUSD")
            timeframe: Timeframe string (e.g. "D1", "H4")
            start: Start datetime (inclusive)
            end: End datetime (inclusive)

        Returns:
            List of Candle objects ordered by timestamp ASC
        """
        stmt = (
            select(Candle)
            .where(
                and_(
                    Candle.symbol    == symbol,
                    Candle.timeframe == timeframe,
                    Candle.timestamp >= start,
                    Candle.timestamp <= end,
                )
            )
            .order_by(Candle.timestamp.asc())
        )
        return list(self.session.scalars(stmt).all())

    def get_latest_n_candles(
        self,
        symbol: str,
        timeframe: str,
        n: int,
    ) -> List[Candle]:
        """
        Retrieve the N most recent candles for a symbol/timeframe.
        Returns candles ordered by timestamp ascending (oldest first).

        Args:
            symbol: Trading pair
            timeframe: Timeframe string
            n: Number of candles to retrieve

        Returns:
            List of n Candle objects, oldest first
        """
        # Subquery: get N most recent
        subq = (
            select(Candle)
            .where(
                and_(
                    Candle.symbol    == symbol,
                    Candle.timeframe == timeframe,
                )
            )
            .order_by(Candle.timestamp.desc())
            .limit(n)
            .subquery()
        )
        # Outer query: reorder ascending
        stmt = (
            select(Candle)
            .join(subq, Candle.id == subq.c.id)
            .order_by(Candle.timestamp.asc())
        )
        return list(self.session.scalars(stmt).all())

    def get_candle_count(self, symbol: str, timeframe: str) -> int:
        """
        Return total number of stored candles for a symbol/timeframe.

        Args:
            symbol: Trading pair
            timeframe: Timeframe string

        Returns:
            Count of stored candles
        """
        stmt = (
            select(func.count(Candle.id))
            .where(
                and_(
                    Candle.symbol    == symbol,
                    Candle.timeframe == timeframe,
                )
            )
        )
        return self.session.scalar(stmt) or 0

    def get_date_range(
        self,
        symbol: str,
        timeframe: str,
    ) -> Optional[Tuple[datetime, datetime]]:
        """
        Return the (earliest, latest) timestamps for stored candles.

        Args:
            symbol: Trading pair
            timeframe: Timeframe string

        Returns:
            Tuple of (min_timestamp, max_timestamp) or None if no data
        """
        stmt = select(
            func.min(Candle.timestamp),
            func.max(Candle.timestamp),
        ).where(
            and_(
                Candle.symbol    == symbol,
                Candle.timeframe == timeframe,
            )
        )
        result = self.session.execute(stmt).one()
        if result[0] is None:
            return None
        return (result[0], result[1])

    # -----------------------------------------------------------------------
    # VALIDATION
    # -----------------------------------------------------------------------

    def candle_gap_check(
        self,
        symbol: str,
        timeframe: str,
        expected_interval_seconds: Optional[int] = None,
    ) -> List[Dict]:
        """
        Detect gaps in candle data (missing bars).

        Compares consecutive candle timestamps and flags gaps larger than
        the expected bar interval (with a 10% tolerance for weekends/holidays).

        Args:
            symbol: Trading pair
            timeframe: Timeframe string
            expected_interval_seconds: Expected seconds between bars.
                If None, inferred from timeframe string.

        Returns:
            List of gap dicts: [{gap_start, gap_end, missing_bars_estimate}]
        """
        if expected_interval_seconds is None:
            expected_interval_seconds = _timeframe_to_seconds(timeframe)

        candles = (
            self.session.scalars(
                select(Candle)
                .where(
                    and_(
                        Candle.symbol    == symbol,
                        Candle.timeframe == timeframe,
                    )
                )
                .order_by(Candle.timestamp.asc())
            )
            .all()
        )

        if len(candles) < 2:
            return []

        gaps = []
        tolerance = expected_interval_seconds * 2.5  # Allow weekends (2.5x = Fri→Mon)

        for i in range(1, len(candles)):
            delta = (
                candles[i].timestamp - candles[i - 1].timestamp
            ).total_seconds()

            if delta > tolerance:
                missing_estimate = max(0, int(delta / expected_interval_seconds) - 1)
                gaps.append({
                    "gap_start":             candles[i - 1].timestamp,
                    "gap_end":               candles[i].timestamp,
                    "gap_seconds":           delta,
                    "missing_bars_estimate": missing_estimate,
                    "symbol":                symbol,
                    "timeframe":             timeframe,
                })

        return gaps

    def delete_candles(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> int:
        """
        Delete candles for a symbol/timeframe, optionally within a date range.

        Args:
            symbol:    Trading pair
            timeframe: Timeframe string
            start:     If provided, delete only candles at or after this time
            end:       If provided, delete only candles at or before this time

        Returns:
            Number of rows deleted
        """
        conditions = [
            Candle.symbol    == symbol,
            Candle.timeframe == timeframe,
        ]
        if start is not None:
            conditions.append(Candle.timestamp >= start)
        if end is not None:
            conditions.append(Candle.timestamp <= end)

        stmt = delete(Candle).where(and_(*conditions))
        result = self.session.execute(stmt)
        return result.rowcount  # type: ignore[return-value]

    def get_symbols(self) -> List[str]:
        """Return a sorted list of all distinct symbols in the candles table."""
        stmt = select(distinct(Candle.symbol)).order_by(Candle.symbol)
        return list(self.session.scalars(stmt).all())

    def get_timeframes(self, symbol: Optional[str] = None) -> List[str]:
        """
        Return a sorted list of distinct timeframes.
        If *symbol* is provided, only timeframes for that symbol are returned.
        """
        stmt = select(distinct(Candle.timeframe))
        if symbol:
            stmt = stmt.where(Candle.symbol == symbol)
        stmt = stmt.order_by(Candle.timeframe)
        return list(self.session.scalars(stmt).all())

    def store_candle_data_list(self, candle_data_list) -> int:
        """
        Convenience method: accept a list of CandleData DTOs, convert to ORM
        objects, and upsert them.  Avoids requiring callers to import Candle.

        Args:
            candle_data_list: List of src.types.CandleData instances

        Returns:
            Number of candles stored/updated
        """
        from datetime import timezone as _tz

        candles: List[Candle] = []
        for cd in candle_data_list:
            ts = cd.timestamp
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=_tz.utc)
            candles.append(
                Candle(
                    symbol    = cd.symbol.upper(),
                    timeframe = cd.timeframe.upper(),
                    timestamp = ts,
                    open      = cd.open,
                    high      = cd.high,
                    low       = cd.low,
                    close     = cd.close,
                    volume    = cd.volume,
                    spread    = cd.spread,
                )
            )
        return self.store_candles(candles)

    def validate_data_integrity(
        self,
        symbol: str,
        timeframe: str,
    ) -> Dict:
        """
        Run comprehensive data integrity checks.

        Checks:
        - Duplicate timestamps
        - Zero-volume bars
        - Negative OHLC values
        - High < Low anomalies
        - Gaps in data

        Args:
            symbol: Trading pair
            timeframe: Timeframe string

        Returns:
            Dict with check results and list of anomalies
        """
        candles = (
            self.session.scalars(
                select(Candle)
                .where(
                    and_(
                        Candle.symbol    == symbol,
                        Candle.timeframe == timeframe,
                    )
                )
                .order_by(Candle.timestamp.asc())
            )
            .all()
        )

        anomalies = []
        seen_timestamps = set()

        for c in candles:
            # Duplicate timestamps
            ts_key = (c.symbol, c.timeframe, c.timestamp)
            if ts_key in seen_timestamps:
                anomalies.append({
                    "type":      "DUPLICATE_TIMESTAMP",
                    "timestamp": c.timestamp,
                    "candle_id": c.id,
                })
            seen_timestamps.add(ts_key)

            # High < Low (invalid OHLC)
            if c.high < c.low:
                anomalies.append({
                    "type":      "HIGH_BELOW_LOW",
                    "timestamp": c.timestamp,
                    "high":      c.high,
                    "low":       c.low,
                })

            # Negative prices
            if c.open <= 0 or c.high <= 0 or c.low <= 0 or c.close <= 0:
                anomalies.append({
                    "type":      "NEGATIVE_OR_ZERO_PRICE",
                    "timestamp": c.timestamp,
                })

            # Zero volume (warning, not error — some brokers omit)
            if c.volume == 0:
                anomalies.append({
                    "type":      "ZERO_VOLUME",
                    "timestamp": c.timestamp,
                    "severity":  "WARNING",
                })

        gaps = self.candle_gap_check(symbol, timeframe)

        return {
            "symbol":        symbol,
            "timeframe":     timeframe,
            "total_candles": len(candles),
            "anomaly_count": len(anomalies),
            "gap_count":     len(gaps),
            "anomalies":     anomalies,
            "gaps":          gaps,
            "is_clean":      len(anomalies) == 0 and len(gaps) == 0,
        }


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _timeframe_to_seconds(timeframe: str) -> int:
    """
    Convert timeframe string to expected interval in seconds.

    Args:
        timeframe: e.g. "M1", "H4", "D1", "W1"

    Returns:
        Seconds per bar
    """
    mapping = {
        "M1":  60,
        "M5":  300,
        "M15": 900,
        "M30": 1800,
        "H1":  3600,
        "H4":  14400,
        "D1":  86400,
        "W1":  604800,
        "MN1": 2592000,
    }
    return mapping.get(timeframe.upper(), 86400)


def candle_from_dict(data: Dict) -> Candle:
    """
    Create a Candle ORM object from a normalized data dictionary.
    Data contract: {timestamp, open, high, low, close, volume, spread, symbol, timeframe}

    Args:
        data: Dictionary with OHLCV fields

    Returns:
        Candle ORM object (not yet persisted)

    Raises:
        KeyError: If required fields are missing
        ValueError: If OHLC values are invalid
    """
    required = {"timestamp", "open", "high", "low", "close", "symbol", "timeframe"}
    missing = required - set(data.keys())
    if missing:
        raise KeyError(f"Missing required candle fields: {missing}")

    c = Candle(
        symbol    = str(data["symbol"]).upper(),
        timeframe = str(data["timeframe"]).upper(),
        timestamp = data["timestamp"] if isinstance(data["timestamp"], datetime)
                    else datetime.fromisoformat(str(data["timestamp"])),
        open      = float(data["open"]),
        high      = float(data["high"]),
        low       = float(data["low"]),
        close     = float(data["close"]),
        volume    = float(data.get("volume", 0.0)),
        spread    = float(data["spread"]) if data.get("spread") is not None else None,
    )

    # Validate OHLC sanity
    if c.high < c.low:
        raise ValueError(f"Candle high ({c.high}) < low ({c.low}) at {c.timestamp}")
    if c.open <= 0 or c.high <= 0 or c.low <= 0 or c.close <= 0:
        raise ValueError(f"Zero or negative price in candle at {c.timestamp}")

    return c
