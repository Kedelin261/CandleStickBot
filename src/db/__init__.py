"""
M02 — Candle Storage
SQLAlchemy ORM models and database management for CandleStickBot.
"""

from .models import (
    Base,
    Candle,
    SwingPoint,
    SRLevel,
    PatternDetection,
    TradeSignal,
    Trade,
    StrategyPerformance,
    MonthlyFailureReport,
    AuditEvent,
    BotState,
    # Enums
    TradeDirectionEnum,
    TradeTierEnum,
    RegimeTypeEnum,
    TrendDirectionEnum,
    ExitReasonEnum,
    LossCategory,
    SignalStatusEnum,
    ExecutionModeEnum,
)
from .database import DatabaseManager, get_database, init_database

__all__ = [
    "Base",
    "Candle",
    "SwingPoint",
    "SRLevel",
    "PatternDetection",
    "TradeSignal",
    "Trade",
    "StrategyPerformance",
    "MonthlyFailureReport",
    "AuditEvent",
    "BotState",
    "TradeDirectionEnum",
    "TradeTierEnum",
    "RegimeTypeEnum",
    "TrendDirectionEnum",
    "ExitReasonEnum",
    "LossCategory",
    "SignalStatusEnum",
    "ExecutionModeEnum",
    "DatabaseManager",
    "get_database",
    "init_database",
]
