"""
M13 — Logging / Audit Module
Structured logging of every decision and state change.
All decisions logged with full context for review and improvement.
Version: 3.1 (Phase 0)
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

import structlog


class EventType(str, Enum):
    DATA_FETCH = "DATA_FETCH"
    DATA_GAP_DETECTED = "DATA_GAP_DETECTED"
    DATA_VALIDATION_FAIL = "DATA_VALIDATION_FAIL"
    SWING_POINT_DETECTED = "SWING_POINT_DETECTED"
    TREND_CLASSIFIED = "TREND_CLASSIFIED"
    REGIME_CLASSIFIED = "REGIME_CLASSIFIED"
    LEVEL_DETECTED = "LEVEL_DETECTED"
    PATTERN_DETECTED = "PATTERN_DETECTED"
    PATTERN_REJECTED = "PATTERN_REJECTED"
    TRADE_EVALUATED = "TRADE_EVALUATED"
    TQS_COMPUTED = "TQS_COMPUTED"
    TRADE_REJECTED = "TRADE_REJECTED"
    TRADE_RECOMMENDED = "TRADE_RECOMMENDED"
    PORTFOLIO_CHECK = "PORTFOLIO_CHECK"
    CORRELATION_BLOCK = "CORRELATION_BLOCK"
    HEAT_LIMIT_REACHED = "HEAT_LIMIT_REACHED"
    RISK_CHECK = "RISK_CHECK"
    POSITION_SIZED = "POSITION_SIZED"
    RR_REJECTED = "RR_REJECTED"
    DAILY_LIMIT_REACHED = "DAILY_LIMIT_REACHED"
    WEEKLY_LIMIT_REACHED = "WEEKLY_LIMIT_REACHED"
    KILL_SWITCH_TRIGGERED = "KILL_SWITCH_TRIGGERED"
    KILL_SWITCH_RESET = "KILL_SWITCH_RESET"
    SPREAD_FILTERED = "SPREAD_FILTERED"
    SESSION_FILTERED = "SESSION_FILTERED"
    NEWS_FILTERED = "NEWS_FILTERED"
    ORDER_PLACED = "ORDER_PLACED"
    ORDER_MODIFIED = "ORDER_MODIFIED"
    ORDER_CANCELLED = "ORDER_CANCELLED"
    POSITION_CLOSED = "POSITION_CLOSED"
    ORDER_FAILED = "ORDER_FAILED"
    SIGNAL_QUEUED = "SIGNAL_QUEUED"
    SIGNAL_APPROVED = "SIGNAL_APPROVED"
    SIGNAL_REJECTED_MANUAL = "SIGNAL_REJECTED_MANUAL"
    SIGNAL_EXPIRED = "SIGNAL_EXPIRED"
    STRATEGY_SCORECARD_UPDATED = "STRATEGY_SCORECARD_UPDATED"
    STRATEGY_DEGRADATION_ALERT = "STRATEGY_DEGRADATION_ALERT"
    STRATEGY_ENABLED = "STRATEGY_ENABLED"
    STRATEGY_DISABLED = "STRATEGY_DISABLED"
    LOSS_CLASSIFIED = "LOSS_CLASSIFIED"
    MONTHLY_REPORT_GENERATED = "MONTHLY_REPORT_GENERATED"
    SYSTEMATIC_ERROR_ALERT = "SYSTEMATIC_ERROR_ALERT"
    MODE_CHANGED = "MODE_CHANGED"
    PROMOTION_CRITERIA_MET = "PROMOTION_CRITERIA_MET"
    PROMOTION_CRITERIA_NOT_MET = "PROMOTION_CRITERIA_NOT_MET"
    BOT_STARTED = "BOT_STARTED"
    BOT_STOPPED = "BOT_STOPPED"
    CONFIG_LOADED = "CONFIG_LOADED"
    ERROR = "ERROR"


def setup_logger(log_level="INFO", log_dir=None, log_to_console=True, log_to_file=True):
    if log_dir is None:
        log_dir = os.path.join(os.getcwd(), "logs")
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    root_logger = logging.getLogger("candlestickbot")
    root_logger.setLevel(numeric_level)
    root_logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ"
    )

    if log_to_console:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(numeric_level)
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)

    if log_to_file:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d")
        fh = logging.FileHandler(os.path.join(log_dir, f"candlestickbot_{ts}.log"), encoding="utf-8")
        fh.setLevel(numeric_level)
        fh.setFormatter(formatter)
        root_logger.addHandler(fh)
        ah = logging.FileHandler(os.path.join(log_dir, f"audit_{ts}.jsonl"), encoding="utf-8")
        ah.setLevel(logging.DEBUG)
        ah.setFormatter(formatter)
        root_logger.addHandler(ah)

    structlog.configure(
        processors=[
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.dev.ConsoleRenderer(),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=False,
    )
    return structlog.get_logger("candlestickbot")


class AuditLogger:
    def __init__(self, log_level="INFO", log_dir=None):
        self._logger = setup_logger(log_level=log_level, log_dir=log_dir)
        self._log_dir = log_dir or "logs"

    def log_data_fetch(self, symbol, timeframe, count, source, success, error=None):
        self._logger.info("Data fetch", event_type=EventType.DATA_FETCH.value,
                          module="M01", symbol=symbol, timeframe=timeframe,
                          count=count, source=source, success=success, error=error)

    def log_data_gap(self, symbol, timeframe, gap_start, gap_end):
        self._logger.warning("Data gap", event_type=EventType.DATA_GAP_DETECTED.value,
                             module="M02", symbol=symbol, timeframe=timeframe,
                             gap_start=gap_start, gap_end=gap_end)

    def log_trend_classified(self, symbol, timeframe, direction, tradeable, reason, ma_value, adx=None):
        self._logger.info("Trend classified", event_type=EventType.TREND_CLASSIFIED.value,
                          module="M04", symbol=symbol, timeframe=timeframe,
                          direction=direction, tradeable=tradeable, reason=reason,
                          ma_value=ma_value, adx=adx)

    def log_regime_classified(self, symbol, timeframe, regime, confidence,
                               allowed_strategies, risk_multiplier, indicators):
        self._logger.info("Regime classified", event_type=EventType.REGIME_CLASSIFIED.value,
                          module="M16", symbol=symbol, timeframe=timeframe, regime=regime,
                          confidence=confidence, allowed_strategies=allowed_strategies,
                          risk_multiplier=risk_multiplier, indicators=indicators)

    def log_pattern_detected(self, symbol, timeframe, pattern_type, direction,
                              quality_score, candle_data, trade_id=None):
        self._logger.info("Pattern detected", event_type=EventType.PATTERN_DETECTED.value,
                          module="M07", symbol=symbol, timeframe=timeframe,
                          pattern_type=pattern_type, direction=direction,
                          quality_score=quality_score, candle_data=candle_data, trade_id=trade_id)

    def log_pattern_rejected(self, symbol, timeframe, pattern_type, reason, candle_data):
        self._logger.debug("Pattern rejected", event_type=EventType.PATTERN_REJECTED.value,
                           module="M07", symbol=symbol, timeframe=timeframe,
                           pattern_type=pattern_type, reason=reason, candle_data=candle_data)

    def log_tqs_computed(self, symbol, timeframe, strategy, tqs_total,
                         trend_score, level_score, pattern_score, regime_score, tier, trade_id=None):
        self._logger.info("TQS computed", event_type=EventType.TQS_COMPUTED.value,
                          module="M08", symbol=symbol, timeframe=timeframe,
                          strategy=strategy, tqs_total=tqs_total,
                          components={"trend": trend_score, "level": level_score,
                                      "pattern": pattern_score, "regime": regime_score},
                          tier=tier, trade_id=trade_id)

    def log_trade_rejected(self, symbol, timeframe, strategy, reason, gate, tqs=None, trade_id=None):
        self._logger.info("Trade rejected", event_type=EventType.TRADE_REJECTED.value,
                          module="M08", symbol=symbol, timeframe=timeframe,
                          strategy=strategy, reason=reason, gate=gate,
                          tqs=tqs, trade_id=trade_id)

    def log_trade_recommended(self, symbol, timeframe, strategy, direction,
                               entry, stop, target, rr_ratio, tqs, tier, lots, trade_id):
        self._logger.info("Trade recommended", event_type=EventType.TRADE_RECOMMENDED.value,
                          module="M08", symbol=symbol, timeframe=timeframe,
                          strategy=strategy, direction=direction, entry=entry,
                          stop=stop, target=target, rr_ratio=rr_ratio,
                          tqs=tqs, tier=tier, lots=lots, trade_id=trade_id)

    def log_risk_check(self, result, reason, trade_id, risk_amount=None, lots=None, details=None):
        level = "info" if result else "warning"
        getattr(self._logger, level)(
            "Risk check", event_type=EventType.RISK_CHECK.value, module="M09",
            result="APPROVED" if result else "REJECTED", reason=reason, trade_id=trade_id,
            risk_amount=risk_amount, lots=lots, details=details or {})

    def log_kill_switch_triggered(self, reason, account_state):
        self._logger.critical("KILL SWITCH TRIGGERED", event_type=EventType.KILL_SWITCH_TRIGGERED.value,
                              module="M09", reason=reason, account_state=account_state)

    def log_kill_switch_reset(self, reset_by):
        self._logger.warning("Kill switch reset", event_type=EventType.KILL_SWITCH_RESET.value,
                             module="M09", reset_by=reset_by)

    def log_daily_limit_reached(self, loss_pct, limit_pct):
        self._logger.warning("Daily limit", event_type=EventType.DAILY_LIMIT_REACHED.value,
                             module="M09", loss_pct=loss_pct, limit_pct=limit_pct)

    def log_weekly_limit_reached(self, loss_pct, limit_pct):
        self._logger.warning("Weekly limit", event_type=EventType.WEEKLY_LIMIT_REACHED.value,
                             module="M09", loss_pct=loss_pct, limit_pct=limit_pct)

    def log_order_event(self, event_type, order_id, symbol, direction, lots,
                        entry=None, sl=None, tp=None, fill_price=None,
                        slippage=None, trade_id=None, reason=None):
        ev = event_type.value if hasattr(event_type, 'value') else str(event_type)
        self._logger.info(f"Order: {ev}", event_type=ev, module="M10",
                          order_id=order_id, symbol=symbol, direction=direction,
                          lots=lots, entry=entry, sl=sl, tp=tp,
                          fill_price=fill_price, slippage=slippage,
                          trade_id=trade_id, reason=reason)

    def log_loss_classified(self, trade_id, strategy, category, pnl_r, context):
        self._logger.info("Loss classified", event_type=EventType.LOSS_CLASSIFIED.value,
                          module="M19", trade_id=trade_id, strategy=strategy,
                          category=category, pnl_r=pnl_r, context=context)

    def log_systematic_error_alert(self, category, pct_of_losses, threshold):
        self._logger.warning("Systematic error alert",
                             event_type=EventType.SYSTEMATIC_ERROR_ALERT.value, module="M19",
                             category=category, pct_of_losses=pct_of_losses, threshold=threshold)

    def log_bot_started(self, mode, config_summary):
        self._logger.info("Bot started", event_type=EventType.BOT_STARTED.value,
                          module="SYSTEM", mode=mode, config_summary=config_summary)

    def log_bot_stopped(self, reason, runtime_seconds):
        self._logger.info("Bot stopped", event_type=EventType.BOT_STOPPED.value,
                          module="SYSTEM", reason=reason, runtime_seconds=runtime_seconds)

    def log_config_loaded(self, path, phase, mode):
        self._logger.info("Config loaded", event_type=EventType.CONFIG_LOADED.value,
                          module="M15", config_path=path, phase=phase, mode=mode)

    def log_error(self, module, error, context=None, trade_id=None):
        self._logger.error(f"Error in {module}", event_type=EventType.ERROR.value,
                           module=module, error=error, context=context or {}, trade_id=trade_id)

    def log_mode_changed(self, old_mode, new_mode, authorized_by):
        self._logger.warning("Mode changed", event_type=EventType.MODE_CHANGED.value,
                             module="M10", old_mode=old_mode, new_mode=new_mode,
                             authorized_by=authorized_by)


_audit_logger: Optional[AuditLogger] = None


def get_audit_logger(log_level="INFO", log_dir=None):
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger(log_level=log_level, log_dir=log_dir)
    return _audit_logger
