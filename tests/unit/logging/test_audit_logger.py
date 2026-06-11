"""
Tests for M13 — Logging / Audit Module.
Phase 0 pass criteria:
  - Logger outputs to file and console without errors
  - All required event types are defined
  - All log methods execute without exceptions
"""

import pytest
from pathlib import Path
from src.logging.audit_logger import AuditLogger, EventType, get_audit_logger


@pytest.fixture
def logger(tmp_path):
    return AuditLogger(log_level="DEBUG", log_dir=str(tmp_path / "logs"))


class TestEventTypes:
    def test_all_required_event_types_defined(self):
        required = [
            "KILL_SWITCH_TRIGGERED", "TRADE_RECOMMENDED", "TRADE_REJECTED",
            "TQS_COMPUTED", "REGIME_CLASSIFIED", "TREND_CLASSIFIED",
            "PATTERN_DETECTED", "ORDER_PLACED", "POSITION_CLOSED",
            "LOSS_CLASSIFIED", "DAILY_LIMIT_REACHED", "WEEKLY_LIMIT_REACHED",
            "BOT_STARTED", "CONFIG_LOADED",
        ]
        event_names = [e.value for e in EventType]
        for name in required:
            assert name in event_names, f"Missing event type: {name}"


class TestLoggingFunctionality:

    def test_logger_init(self, logger):
        assert logger is not None

    def test_log_dir_created(self, tmp_path):
        log_dir = tmp_path / "new_logs" / "subdir"
        AuditLogger(log_level="INFO", log_dir=str(log_dir))
        assert log_dir.exists()

    def test_data_fetch_logs(self, logger):
        logger.log_data_fetch("EURUSD", "D1", 100, "MT5", True)

    def test_data_fetch_failure_logs(self, logger):
        logger.log_data_fetch("EURUSD", "D1", 0, "MT5", False, error="Connection refused")

    def test_trend_classified_logs(self, logger):
        logger.log_trend_classified("EURUSD", "D1", "UP", True,
                                    "HH+HL confirmed; above 21 SMA", 1.0950, adx=28.5)

    def test_regime_classified_logs(self, logger):
        logger.log_regime_classified(
            "EURUSD", "D1", "TRENDING", 0.85,
            ["PIN_BAR", "ENGULFING"], 1.0,
            {"adx": 28.5, "atr": 0.0080, "choppiness": 45.2}
        )

    def test_pattern_detected_logs(self, logger):
        logger.log_pattern_detected(
            "EURUSD", "D1", "PIN_BAR_BULLISH", "LONG", 8,
            {"body_size": 0.0020, "lower_wick": 0.0080}, "trade-001"
        )

    def test_tqs_computed_logs(self, logger):
        logger.log_tqs_computed(
            "EURUSD", "D1", "PIN_BAR",
            83, 20, 15, 23, 25, "PREMIUM", "trade-001"
        )

    def test_trade_rejected_logs(self, logger):
        logger.log_trade_rejected(
            "EURUSD", "D1", "PIN_BAR",
            "TQS score 45 below minimum 60", "TQS", 45, "trade-002"
        )

    def test_trade_recommended_logs(self, logger):
        logger.log_trade_recommended(
            "EURUSD", "D1", "PIN_BAR", "LONG",
            1.1000, 1.0950, 1.1100, 2.0, 83, "PREMIUM", 0.01, "trade-001"
        )

    def test_risk_check_pass_logs(self, logger):
        logger.log_risk_check(True, "All checks passed", "trade-001",
                              risk_amount=100.0, lots=0.01)

    def test_risk_check_fail_logs(self, logger):
        logger.log_risk_check(False, "Daily loss limit reached", "trade-002")

    def test_kill_switch_triggered_logs(self, logger):
        logger.log_kill_switch_triggered(
            "Drawdown 10.5% exceeded 10.0%",
            {"balance": 9000.0, "equity": 8950.0, "drawdown_pct": 10.5}
        )

    def test_order_event_logs(self, logger):
        logger.log_order_event(
            EventType.ORDER_PLACED, "ORD-001", "EURUSD", "LONG", 0.01,
            entry=1.1000, sl=1.0950, tp=1.1100,
            fill_price=1.1001, slippage=0.1, trade_id="trade-001"
        )

    def test_loss_classified_logs(self, logger):
        logger.log_loss_classified(
            "trade-003", "ENGULFING", "NORMAL_STATISTICAL_LOSS",
            -1.0, {"regime": "TRENDING", "tqs": 72}
        )

    def test_daily_limit_reached_logs(self, logger):
        logger.log_daily_limit_reached(3.2, 3.0)

    def test_weekly_limit_reached_logs(self, logger):
        logger.log_weekly_limit_reached(6.5, 6.0)

    def test_bot_started_logs(self, logger):
        logger.log_bot_started("backtest", {"symbols": ["EURUSD"], "phase": 1})

    def test_config_loaded_logs(self, logger):
        logger.log_config_loaded("config/default_config.yaml", 1, "backtest")

    def test_error_logs(self, logger):
        logger.log_error("M07", "Pattern detection failed", {"candles": 2})

    def test_mode_changed_logs(self, logger):
        logger.log_mode_changed("backtest", "paper", "admin")

    def test_systematic_error_alert_logs(self, logger):
        logger.log_systematic_error_alert("BAD_REGIME", 0.35, 0.30)

    def test_kill_switch_reset_logs(self, logger):
        logger.log_kill_switch_reset("admin")

    def test_data_gap_logs(self, logger):
        logger.log_data_gap("EURUSD", "D1", "2024-01-05", "2024-01-08")

    def test_bot_stopped_logs(self, logger):
        logger.log_bot_stopped("Normal shutdown", 3600.0)


class TestLogFileOutput:

    def test_log_files_created(self, tmp_path):
        log_dir = tmp_path / "test_logs"
        logger = AuditLogger(log_level="INFO", log_dir=str(log_dir))
        logger.log_bot_started("backtest", {})
        assert log_dir.exists()
        log_files = list(log_dir.glob("*.log")) + list(log_dir.glob("*.jsonl"))
        assert len(log_files) >= 1


class TestSingleton:

    def test_get_audit_logger_returns_instance(self, tmp_path):
        import src.logging.audit_logger as m
        original = m._audit_logger
        m._audit_logger = None
        log = get_audit_logger(log_dir=str(tmp_path / "logs"))
        assert log is not None
        m._audit_logger = original
