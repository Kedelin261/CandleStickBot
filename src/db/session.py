"""
M02 — Candle Storage: Database session management
SQLAlchemy engine creation, session factory, and initialization.
Version: 3.1 (Phase 0)
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from .models import Base


# ---------------------------------------------------------------------------
# ENGINE CREATION
# ---------------------------------------------------------------------------

def create_db_engine(
    db_url: Optional[str] = None,
    echo: bool = False,
    pool_size: int = 5,
    max_overflow: int = 10,
) -> Engine:
    """
    Create SQLAlchemy engine from URL.

    For SQLite (dev): enables WAL mode and foreign key enforcement.
    For PostgreSQL (prod): configures connection pool.

    Args:
        db_url: Database connection URL. Defaults to env var DATABASE_URL
                or SQLite at data/candlestickbot.db
        echo: If True, log all SQL statements (DEBUG mode only)
        pool_size: Connection pool size (PostgreSQL only)
        max_overflow: Max overflow connections (PostgreSQL only)

    Returns:
        Configured SQLAlchemy Engine
    """
    url = db_url or os.environ.get(
        "CSBOT__SYSTEM__DB_URL",
        "sqlite:///data/candlestickbot.db"
    )

    # Ensure data directory exists for SQLite
    if url.startswith("sqlite:///"):
        db_path = url.replace("sqlite:///", "")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    if url.startswith("sqlite"):
        engine = create_engine(
            url,
            echo=echo,
            connect_args={
                "check_same_thread": False,
                "timeout": 30,
            },
        )
        # Enable WAL mode and foreign keys for SQLite
        @event.listens_for(engine, "connect")
        def _sqlite_setup(dbapi_conn, connection_record):
            dbapi_conn.execute("PRAGMA journal_mode=WAL")
            dbapi_conn.execute("PRAGMA foreign_keys=ON")
            dbapi_conn.execute("PRAGMA synchronous=NORMAL")
            dbapi_conn.execute("PRAGMA cache_size=-64000")  # 64MB cache

    else:
        # PostgreSQL / other
        engine = create_engine(
            url,
            echo=echo,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_pre_ping=True,  # Reconnect on stale connections
        )

    return engine


def create_session_factory(engine: Engine) -> sessionmaker:
    """
    Create a SQLAlchemy session factory.

    Args:
        engine: SQLAlchemy Engine instance

    Returns:
        sessionmaker factory
    """
    return sessionmaker(
        bind=engine,
        autocommit=False,
        autoflush=False,
        expire_on_commit=False,  # Keep attributes accessible after commit
    )


# ---------------------------------------------------------------------------
# DATABASE INITIALIZATION
# ---------------------------------------------------------------------------

def init_db(engine: Engine) -> None:
    """
    Create all tables if they don't exist.
    Safe to call multiple times (CREATE IF NOT EXISTS semantics).

    Args:
        engine: SQLAlchemy Engine instance
    """
    Base.metadata.create_all(bind=engine)


def drop_all_tables(engine: Engine) -> None:
    """
    Drop all tables. USE WITH EXTREME CAUTION.
    Only for testing and development reset.

    Args:
        engine: SQLAlchemy Engine instance
    """
    Base.metadata.drop_all(bind=engine)


# ---------------------------------------------------------------------------
# SESSION CONTEXT MANAGER
# ---------------------------------------------------------------------------

@contextmanager
def get_session(session_factory: sessionmaker) -> Generator[Session, None, None]:
    """
    Context manager for database sessions with automatic commit/rollback.

    Usage:
        with get_session(session_factory) as session:
            session.add(candle)
            # commit happens automatically on exit

    Args:
        session_factory: SQLAlchemy sessionmaker instance

    Yields:
        Active Session instance

    Raises:
        Exception: Re-raises any exception after rolling back the transaction
    """
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ---------------------------------------------------------------------------
# DATABASE SINGLETON (for simple usage)
# ---------------------------------------------------------------------------

_engine: Optional[Engine] = None
_session_factory: Optional[sessionmaker] = None


def get_engine(db_url: Optional[str] = None, echo: bool = False) -> Engine:
    """
    Get or create the global database engine (singleton pattern).
    """
    global _engine
    if _engine is None:
        _engine = create_db_engine(db_url=db_url, echo=echo)
    return _engine


def get_session_factory(db_url: Optional[str] = None) -> sessionmaker:
    """
    Get or create the global session factory (singleton pattern).
    """
    global _session_factory
    if _session_factory is None:
        engine = get_engine(db_url=db_url)
        _session_factory = create_session_factory(engine)
    return _session_factory


def setup_database(db_url: Optional[str] = None, echo: bool = False) -> sessionmaker:
    """
    Full database setup: create engine, init tables, return session factory.
    Call once at application startup.

    Args:
        db_url: Database URL (defaults to SQLite dev db)
        echo: Enable SQL echo logging

    Returns:
        Configured session factory
    """
    global _engine, _session_factory
    _engine = create_db_engine(db_url=db_url, echo=echo)
    init_db(_engine)
    _session_factory = create_session_factory(_engine)
    return _session_factory


def check_db_health(engine: Engine) -> bool:
    """
    Verify database connection is healthy.

    Args:
        engine: SQLAlchemy Engine instance

    Returns:
        True if connection is healthy, False otherwise
    """
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
