"""
M02 — Candle Storage: Database connection and session management.
SQLAlchemy 2.0 engine setup with SQLite (dev) and PostgreSQL (prod) support.
Version: 3.1 (Phase 0)
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from .models import Base


# ===========================================================================
# ENGINE FACTORY
# ===========================================================================

def create_db_engine(db_url: str, echo: bool = False):
    """
    Create SQLAlchemy engine with appropriate configuration.
    SQLite: Uses StaticPool for thread safety; enables WAL mode and foreign keys.
    PostgreSQL: Uses default connection pool with SSL support.

    Args:
        db_url: Database URL (sqlite:///... or postgresql://...)
        echo: If True, log all SQL statements (DEBUG use only)
    """
    if db_url.startswith("sqlite"):
        # Extract path and ensure directory exists
        if ":///" in db_url:
            db_path = db_url.split("///")[1]
            if db_path and db_path != ":memory:":
                Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        engine = create_engine(
            db_url,
            echo=echo,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,  # Required for SQLite + multithreading
        )

        # Enable WAL mode and foreign keys for SQLite
        @event.listens_for(engine, "connect")
        def set_sqlite_pragmas(dbapi_conn, connection_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")   # Better concurrency
            cursor.execute("PRAGMA foreign_keys=ON")    # Enforce FK constraints
            cursor.execute("PRAGMA synchronous=NORMAL") # Performance balance
            cursor.close()

    else:
        # PostgreSQL / production
        engine = create_engine(
            db_url,
            echo=echo,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,   # Reconnect if connection is stale
        )

    return engine


def init_database(db_url: str, echo: bool = False):
    """
    Initialize database: create engine, create all tables if not exist.

    Args:
        db_url: Database URL
        echo: If True, log SQL statements

    Returns:
        (engine, SessionLocal) tuple
    """
    engine = create_db_engine(db_url, echo=echo)
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    return engine, SessionLocal


# ===========================================================================
# DATABASE MANAGER
# ===========================================================================

class DatabaseManager:
    """
    Manages database connection lifecycle for CandleStickBot.
    Provides session factory and context manager for safe transaction handling.
    """

    def __init__(self, db_url: str, echo: bool = False):
        self.db_url = db_url
        self.engine, self._SessionLocal = init_database(db_url, echo=echo)

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """
        Provide a transactional session scope.
        Automatically commits on success, rolls back on exception.

        Usage:
            with db.get_session() as session:
                session.add(candle)
                # commit is automatic on exit
        """
        session = self._SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get_session_factory(self) -> sessionmaker:
        """Return the session factory for manual session management."""
        return self._SessionLocal

    def create_all_tables(self) -> None:
        """Create all tables (idempotent — safe to call multiple times)."""
        Base.metadata.create_all(self.engine)

    def drop_all_tables(self) -> None:
        """
        Drop all tables. USE WITH EXTREME CAUTION.
        Only for testing and development reset scenarios.
        """
        Base.metadata.drop_all(self.engine)

    def health_check(self) -> bool:
        """
        Verify database connection is healthy.
        Returns True if connection succeeds, False otherwise.
        """
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    def get_table_stats(self) -> dict:
        """
        Return row counts for all key tables.
        Used for monitoring and health checks.
        """
        tables = [
            "candles", "trades", "trade_signals", "swing_points",
            "sr_levels", "pattern_detections", "strategy_performance",
            "audit_events", "bot_state",
        ]
        stats = {}
        try:
            with self.engine.connect() as conn:
                for table in tables:
                    try:
                        result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                        stats[table] = result.scalar()
                    except Exception:
                        stats[table] = "error"
        except Exception as e:
            return {"error": str(e)}
        return stats


# ===========================================================================
# CONVENIENCE FACTORY
# ===========================================================================

_db_manager: DatabaseManager | None = None


def get_database(db_url: str | None = None) -> DatabaseManager:
    """
    Get or create the global DatabaseManager instance.
    Uses CSBOT__SYSTEM__DB_URL environment variable or provided URL.
    Default: SQLite at data/candlestickbot.db

    Args:
        db_url: Database URL (uses env var or default if not provided)
    """
    global _db_manager
    if _db_manager is None:
        if db_url is None:
            db_url = os.environ.get(
                "CSBOT__SYSTEM__DB_URL",
                "sqlite:///data/candlestickbot.db"
            )
        _db_manager = DatabaseManager(db_url)
    return _db_manager
