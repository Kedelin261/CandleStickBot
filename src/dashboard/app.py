"""
M14 — Dashboard / Monitoring (Minimal Phase 0 Stub)
Provides basic system status and health monitoring interface.
The Candlestick Trading Bible: Know what your system is doing at all times.

Phase 1 Scope (minimal):
  - System status endpoint (health check)
  - Current mode display (backtest/paper/live)
  - Kill switch status
  - Basic account summary

Phase 2+ (deferred):
  - Real-time trade monitoring dashboard
  - Performance charts and analytics
  - Alert management
  - Trade journal viewer

Technology: Flask or FastAPI (Phase 2)
Phase 1: Simple status output to console/log only.

Status: MINIMAL STUB — Phase 0. Expanded in Phase 2.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional

logger = logging.getLogger("candlestickbot.dashboard")


class SystemStatus(str, Enum):
    """Overall system health status."""
    HEALTHY = "HEALTHY"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    OFFLINE = "OFFLINE"


@dataclass
class StatusReport:
    """
    System status snapshot for monitoring.
    Phase 1: output to log. Phase 2: serve via REST API.
    """
    timestamp: str
    status: SystemStatus
    mode: str                   # backtest / paper / live
    phase: int                  # 1 or 2
    kill_switch_active: bool
    kill_switch_reason: Optional[str]

    # Account info
    account_balance: Optional[float] = None
    account_equity: Optional[float] = None
    drawdown_pct: Optional[float] = None

    # Trading stats
    open_trades: int = 0
    trades_today: int = 0
    consecutive_losses: int = 0

    # System health
    data_feed_ok: bool = False
    db_connection_ok: bool = False
    config_loaded: bool = False

    # Alerts
    alerts: list = None

    def __post_init__(self):
        if self.alerts is None:
            self.alerts = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "status": self.status.value,
            "mode": self.mode,
            "phase": self.phase,
            "kill_switch": {
                "active": self.kill_switch_active,
                "reason": self.kill_switch_reason,
            },
            "account": {
                "balance": self.account_balance,
                "equity": self.account_equity,
                "drawdown_pct": self.drawdown_pct,
            },
            "trading": {
                "open_trades": self.open_trades,
                "trades_today": self.trades_today,
                "consecutive_losses": self.consecutive_losses,
            },
            "health": {
                "data_feed": self.data_feed_ok,
                "database": self.db_connection_ok,
                "config": self.config_loaded,
            },
            "alerts": self.alerts,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def to_console_summary(self) -> str:
        """Format status as a readable console string."""
        ks = "🔴 ACTIVE" if self.kill_switch_active else "✅ CLEAR"
        status_icon = {
            SystemStatus.HEALTHY: "✅",
            SystemStatus.WARNING: "⚠️",
            SystemStatus.CRITICAL: "🚨",
            SystemStatus.OFFLINE: "⛔",
        }.get(self.status, "❓")

        lines = [
            f"{'=' * 50}",
            f" CandleStickBot Status Report",
            f"{'=' * 50}",
            f" Time:        {self.timestamp}",
            f" Status:      {status_icon} {self.status.value}",
            f" Mode:        {self.mode.upper()} (Phase {self.phase})",
            f" Kill Switch: {ks}",
        ]

        if self.kill_switch_reason:
            lines.append(f" KS Reason:   {self.kill_switch_reason}")

        if self.account_balance is not None:
            lines.extend([
                f"",
                f" Account:",
                f"   Balance:    ${self.account_balance:,.2f}",
                f"   Equity:     ${self.account_equity:,.2f}",
                f"   Drawdown:   {self.drawdown_pct:.1f}%",
            ])

        lines.extend([
            f"",
            f" Trading:",
            f"   Open Trades:      {self.open_trades}",
            f"   Trades Today:     {self.trades_today}",
            f"   Consec. Losses:   {self.consecutive_losses}",
        ])

        if self.alerts:
            lines.append(f"")
            lines.append(f" ⚠️  Alerts ({len(self.alerts)}):")
            for alert in self.alerts[:5]:  # Show max 5
                lines.append(f"   - {alert}")

        lines.append(f"{'=' * 50}")
        return "\n".join(lines)


class DashboardMonitor:
    """
    M14 — Dashboard Monitor.

    Phase 1: Minimal implementation — generates status reports to log.
    Phase 2: Full REST API with HTML dashboard.

    Responsibilities:
    - Collect system status from all modules
    - Log periodic status reports
    - Generate health check responses
    - Surface alerts and warnings
    """

    REPORT_INTERVAL_SECONDS = 300  # Log status every 5 minutes

    def __init__(
        self,
        config=None,
        risk_engine=None,       # M09 for kill switch state
        db_session=None,        # For DB health check
        audit_logger=None,      # M13 for logging
    ):
        self.config = config
        self.risk_engine = risk_engine
        self.db_session = db_session
        self.audit_logger = audit_logger
        self._last_report_time: Optional[datetime] = None

    def get_status(self) -> StatusReport:
        """
        Generate current system status report.

        Collects data from all available modules and returns
        a StatusReport object suitable for logging or API response.
        """
        now = datetime.now(timezone.utc).isoformat()

        # Check kill switch state
        ks_active = False
        ks_reason = None
        if self.risk_engine:
            ks_active = self.risk_engine.kill_switch_active
            if hasattr(self.risk_engine, "state"):
                state = self.risk_engine.state
                ks_reason = state.kill_switch_reason.value if state.kill_switch_reason else None

        # Check DB health
        db_ok = False
        if self.db_session:
            try:
                self.db_session.execute("SELECT 1")
                db_ok = True
            except Exception:
                db_ok = False

        # Config loaded?
        config_ok = self.config is not None

        # Determine overall status
        if ks_active:
            overall_status = SystemStatus.CRITICAL
        elif not db_ok or not config_ok:
            overall_status = SystemStatus.WARNING
        else:
            overall_status = SystemStatus.HEALTHY

        # Mode and phase from config
        mode = "backtest"
        phase = 1
        if self.config:
            if hasattr(self.config, "execution"):
                mode = str(self.config.execution.mode.value)
            if hasattr(self.config, "system"):
                phase = self.config.system.phase

        alerts = []
        if ks_active:
            alerts.append(f"KILL SWITCH ACTIVE: {ks_reason}")
        if not db_ok:
            alerts.append("Database connection failed")

        return StatusReport(
            timestamp=now,
            status=overall_status,
            mode=mode,
            phase=phase,
            kill_switch_active=ks_active,
            kill_switch_reason=ks_reason,
            db_connection_ok=db_ok,
            config_loaded=config_ok,
            alerts=alerts,
        )

    def log_status(self, force: bool = False) -> None:
        """
        Log a status report if interval has elapsed (or forced).

        Args:
            force: If True, log regardless of interval.
        """
        now = datetime.now(timezone.utc)
        elapsed = (
            (now - self._last_report_time).total_seconds()
            if self._last_report_time else float("inf")
        )

        if not force and elapsed < self.REPORT_INTERVAL_SECONDS:
            return

        status = self.get_status()
        self._last_report_time = now

        if status.status == SystemStatus.CRITICAL:
            logger.critical(f"System status: {status.to_json()}")
        elif status.status == SystemStatus.WARNING:
            logger.warning(f"System status: {status.to_json()}")
        else:
            logger.info(f"System status: {status.to_json()}")

    def health_check(self) -> Dict[str, Any]:
        """
        Simple health check endpoint response.
        Returns minimal JSON suitable for monitoring systems.

        Phase 2: This becomes a real HTTP endpoint.
        """
        status = self.get_status()
        return {
            "ok": status.status == SystemStatus.HEALTHY,
            "status": status.status.value,
            "kill_switch": status.kill_switch_active,
            "timestamp": status.timestamp,
        }


def create_dashboard(
    config=None,
    risk_engine=None,
    db_session=None,
    audit_logger=None,
) -> DashboardMonitor:
    """Factory function: create and return a DashboardMonitor."""
    return DashboardMonitor(
        config=config,
        risk_engine=risk_engine,
        db_session=db_session,
        audit_logger=audit_logger,
    )
