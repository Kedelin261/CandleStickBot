"""
M09 — Risk Management Engine
The most critical module — protects capital above all else.
The Candlestick Trading Bible: Risk management is the ONLY thing that keeps you alive.

Risk Rules (from config/spec):
  - Default risk: 1.0% per trade
  - Premium risk (TQS >= 80, opt-in): 1.5% (DISABLED by default)
  - Hard cap: 2.0% absolute maximum (cannot be overridden)
  - Minimum R:R: 2.0:1
  - Daily loss limit: 3.0% of account
  - Weekly loss limit: 6.0% of account
  - Kill switch — drawdown: 10.0% from peak equity
  - Kill switch — consecutive losses: 7
  - Kill switch — both daily AND weekly limits hit simultaneously

Position Sizing Formula:
  lot_size = (account_balance * risk_pct / 100) / (stop_distance_pips * pip_value)

Kill Switch:
  - Triggers immediately on ANY of: 10% drawdown OR 7 consecutive losses
  - OR both daily AND weekly limits simultaneously reached
  - Once triggered: NO new trades until manually reset by authorized user
  - Full kill switch state persisted to AccountSnapshot (DB)

Phase 1: Backtest + Paper modes only (no live execution).
Status: STUB — Phase 0 scaffold. Full implementation in Phase 1 Sprint 3.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

from src.types import (
    AccountState,
    TradeRecommendation,
    RiskApprovedOrder,
    RiskRejection,
    TQSComponents,
)

logger = logging.getLogger("candlestickbot.risk.engine")


class RiskCheckResult(str, Enum):
    """Risk check outcome."""
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    KILL_SWITCH_ACTIVE = "KILL_SWITCH_ACTIVE"


class KillSwitchReason(str, Enum):
    """Why the kill switch was triggered."""
    DRAWDOWN = "DRAWDOWN"
    CONSECUTIVE_LOSSES = "CONSECUTIVE_LOSSES"
    DAILY_AND_WEEKLY = "DAILY_AND_WEEKLY"
    MANUAL = "MANUAL"


@dataclass
class RiskState:
    """
    Current risk state of the trading system.
    Persisted to AccountSnapshot table after each update.
    """
    kill_switch_active: bool = False
    kill_switch_reason: Optional[KillSwitchReason] = None
    consecutive_losses: int = 0
    trades_today: int = 0
    losses_today: int = 0
    daily_pnl_pct: float = 0.0        # Today's P&L as % of balance
    weekly_pnl_pct: float = 0.0       # This week's P&L as % of balance
    drawdown_from_peak_pct: float = 0.0
    peak_equity: float = 0.0
    current_balance: float = 0.0
    current_equity: float = 0.0
    open_trade_count: int = 0

    @property
    def is_trading_allowed(self) -> bool:
        """True if all kill switch conditions are clear."""
        return not self.kill_switch_active


@dataclass
class KillSwitchEvent:
    """Record of a kill switch trigger event."""
    reason: KillSwitchReason
    trigger_value: float        # The value that triggered (drawdown%, loss count, etc.)
    threshold: float            # The threshold that was breached
    account_balance: float
    account_equity: float
    description: str


class RiskEngine:
    """
    M09 — Risk Management Engine.

    Responsibilities:
    1. Kill switch monitoring — check BEFORE evaluating any new signal
    2. Daily/weekly loss limit enforcement
    3. Position sizing via Kelly-fraction or fixed-fractional method
    4. Maximum open trade count check
    5. Account heat check (total open risk %)
    6. Post-trade state update (after fill or close)

    All risk checks are logged via M13 AuditLogger with full context.
    No trade is executed without passing ALL risk checks.

    Phase 1: Fixed fractional position sizing only.
             Kelly criterion deferred to Phase 2.
    """

    def __init__(
        self,
        # Core risk parameters
        risk_per_trade_pct: float = 1.0,
        max_risk_per_trade_pct: float = 2.0,
        premium_risk_pct: float = 1.0,        # Set to 1.5 to enable premium risk
        premium_risk_enabled: bool = False,   # Disabled by default
        min_rr_ratio: float = 2.0,
        max_open_trades: int = 3,
        max_account_heat_pct: float = 6.0,

        # Loss limits
        daily_loss_limit_pct: float = 3.0,
        weekly_loss_limit_pct: float = 6.0,

        # Kill switch
        kill_switch_drawdown_pct: float = 10.0,
        kill_switch_consecutive_losses: int = 7,

        # Symbol parameters
        pip_size: float = 0.0001,
        pip_value_per_lot: float = 10.0,      # Standard lot pip value in account currency

        # Audit
        audit_logger=None,
    ):
        # Enforce hard cap (cannot exceed 2.0% regardless of config)
        if max_risk_per_trade_pct > 2.0:
            raise ValueError(
                f"max_risk_per_trade_pct {max_risk_per_trade_pct} exceeds hard cap of 2.0%"
            )

        self.risk_per_trade_pct = risk_per_trade_pct
        self.max_risk_per_trade_pct = max_risk_per_trade_pct
        self.premium_risk_pct = premium_risk_pct
        self.premium_risk_enabled = premium_risk_enabled
        self.min_rr_ratio = min_rr_ratio
        self.max_open_trades = max_open_trades
        self.max_account_heat_pct = max_account_heat_pct
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.weekly_loss_limit_pct = weekly_loss_limit_pct
        self.kill_switch_drawdown_pct = kill_switch_drawdown_pct
        self.kill_switch_consecutive_losses = kill_switch_consecutive_losses
        self.pip_size = pip_size
        self.pip_value_per_lot = pip_value_per_lot
        self.audit_logger = audit_logger

        # Internal state
        self._state = RiskState()

    @property
    def state(self) -> RiskState:
        """Current risk state (read-only snapshot)."""
        return self._state

    @property
    def kill_switch_active(self) -> bool:
        return self._state.kill_switch_active

    def check_and_approve(
        self,
        recommendation: TradeRecommendation,
        account: AccountState,
    ) -> Tuple[RiskCheckResult, Optional[RiskApprovedOrder], Optional[RiskRejection]]:
        """
        Run all risk checks and size the position if approved.

        Gate chain:
        1. Kill switch active?
        2. Daily loss limit reached?
        3. Weekly loss limit reached?
        4. Max open trades reached?
        5. Account heat (total risk) too high?
        6. Minimum R:R check
        7. Position sizing

        Args:
            recommendation: TradeRecommendation from M08
            account: Current account state

        Returns:
            Tuple of (result, approved_order_or_None, rejection_or_None)
        """
        # Gate 1: Kill switch
        if self._state.kill_switch_active:
            rejection = RiskRejection(
                recommendation=recommendation,
                reason="Kill switch active — trading halted",
                check_type="KILL_SWITCH",
            )
            return RiskCheckResult.KILL_SWITCH_ACTIVE, None, rejection

        # Gate 2: Daily loss limit
        if abs(account.daily_pnl_pct) >= self.daily_loss_limit_pct and account.daily_pnl_pct < 0:
            rejection = RiskRejection(
                recommendation=recommendation,
                reason=f"Daily loss limit {self.daily_loss_limit_pct}% reached",
                check_type="DAILY_LIMIT",
            )
            return RiskCheckResult.REJECTED, None, rejection

        # Gate 3: Weekly loss limit
        if abs(account.weekly_pnl_pct) >= self.weekly_loss_limit_pct and account.weekly_pnl_pct < 0:
            rejection = RiskRejection(
                recommendation=recommendation,
                reason=f"Weekly loss limit {self.weekly_loss_limit_pct}% reached",
                check_type="WEEKLY_LIMIT",
            )
            return RiskCheckResult.REJECTED, None, rejection

        # Gate 4: Max open trades
        if self._state.open_trade_count >= self.max_open_trades:
            rejection = RiskRejection(
                recommendation=recommendation,
                reason=f"Max open trades {self.max_open_trades} reached",
                check_type="MAX_OPEN_TRADES",
            )
            return RiskCheckResult.REJECTED, None, rejection

        # Gate 5: R:R check
        if recommendation.rr_ratio < self.min_rr_ratio:
            rejection = RiskRejection(
                recommendation=recommendation,
                reason=f"R:R {recommendation.rr_ratio:.2f} < minimum {self.min_rr_ratio}",
                check_type="RR_RATIO",
            )
            return RiskCheckResult.REJECTED, None, rejection

        # Gate 6: Position sizing
        risk_pct = self._determine_risk_pct(recommendation.tqs)
        lot_size = self._calculate_lot_size(
            account_balance=account.balance,
            risk_pct=risk_pct,
            entry=recommendation.entry_price,
            stop=recommendation.stop_price,
        )

        if lot_size <= 0:
            rejection = RiskRejection(
                recommendation=recommendation,
                reason="Calculated lot size is zero or negative",
                check_type="POSITION_SIZE",
            )
            return RiskCheckResult.REJECTED, None, rejection

        risk_amount_usd = account.balance * risk_pct / 100.0
        approved = RiskApprovedOrder(
            recommendation=recommendation,
            lot_size=lot_size,
            risk_pct=risk_pct,
            risk_amount_usd=risk_amount_usd,
        )
        return RiskCheckResult.APPROVED, approved, None

    def _determine_risk_pct(self, tqs: Optional[TQSComponents]) -> float:
        """
        Determine risk percentage based on TQS tier.

        Phase 1 defaults:
          - Standard (TQS 60-79): 1.0%
          - Premium (TQS >= 80): 1.0% (unless premium_risk_enabled=True → 1.5%)

        Hard cap: 2.0% absolute maximum.
        """
        if tqs is None:
            return self.risk_per_trade_pct

        if tqs.tier == "PREMIUM" and self.premium_risk_enabled:
            risk = self.premium_risk_pct
        else:
            risk = self.risk_per_trade_pct

        # Enforce hard cap
        return min(risk, self.max_risk_per_trade_pct)

    def _calculate_lot_size(
        self,
        account_balance: float,
        risk_pct: float,
        entry: float,
        stop: float,
        lot_min: float = 0.01,
        lot_step: float = 0.01,
        lot_max: float = 100.0,
    ) -> float:
        """
        Calculate position size using fixed fractional method.

        Formula:
            risk_amount = balance * risk_pct / 100
            stop_distance_pips = |entry - stop| / pip_size
            lot_size = risk_amount / (stop_distance_pips * pip_value_per_lot)

        Args:
            account_balance: Current account balance
            risk_pct: Risk percentage (e.g., 1.0 for 1%)
            entry: Entry price
            stop: Stop loss price
            lot_min: Broker minimum lot size (default 0.01)
            lot_step: Lot size increment (default 0.01)
            lot_max: Maximum allowed lot size

        Returns:
            Lot size rounded to lot_step, min lot_min, max lot_max.
            Returns 0 if stop distance is zero.
        """
        stop_distance = abs(entry - stop)
        if stop_distance <= 0:
            return 0.0

        stop_pips = stop_distance / self.pip_size
        risk_amount = account_balance * risk_pct / 100.0
        raw_lots = risk_amount / (stop_pips * self.pip_value_per_lot)

        # Round to lot_step
        lots = round(raw_lots / lot_step) * lot_step
        lots = max(lot_min, min(lot_max, lots))
        return round(lots, 2)

    def check_kill_switch(self, account: AccountState) -> Optional[KillSwitchEvent]:
        """
        Check if any kill switch condition is breached.

        Called after each trade close or account update.

        Returns:
            KillSwitchEvent if triggered, None if all clear.
        """
        # Check 1: Drawdown from peak
        drawdown = account.drawdown_from_peak_pct
        if drawdown >= self.kill_switch_drawdown_pct:
            event = KillSwitchEvent(
                reason=KillSwitchReason.DRAWDOWN,
                trigger_value=drawdown,
                threshold=self.kill_switch_drawdown_pct,
                account_balance=account.balance,
                account_equity=account.equity,
                description=f"Drawdown {drawdown:.1f}% exceeded {self.kill_switch_drawdown_pct}% limit",
            )
            self._trigger_kill_switch(event)
            return event

        # Check 2: Consecutive losses
        if self._state.consecutive_losses >= self.kill_switch_consecutive_losses:
            event = KillSwitchEvent(
                reason=KillSwitchReason.CONSECUTIVE_LOSSES,
                trigger_value=float(self._state.consecutive_losses),
                threshold=float(self.kill_switch_consecutive_losses),
                account_balance=account.balance,
                account_equity=account.equity,
                description=(
                    f"{self._state.consecutive_losses} consecutive losses "
                    f"reached {self.kill_switch_consecutive_losses} limit"
                ),
            )
            self._trigger_kill_switch(event)
            return event

        # Check 3: Both daily AND weekly limits hit simultaneously
        daily_breached = abs(account.daily_pnl_pct) >= self.daily_loss_limit_pct
        weekly_breached = abs(account.weekly_pnl_pct) >= self.weekly_loss_limit_pct
        if daily_breached and weekly_breached:
            event = KillSwitchEvent(
                reason=KillSwitchReason.DAILY_AND_WEEKLY,
                trigger_value=max(abs(account.daily_pnl_pct), abs(account.weekly_pnl_pct)),
                threshold=min(self.daily_loss_limit_pct, self.weekly_loss_limit_pct),
                account_balance=account.balance,
                account_equity=account.equity,
                description="Both daily AND weekly loss limits breached simultaneously",
            )
            self._trigger_kill_switch(event)
            return event

        return None  # All clear

    def _trigger_kill_switch(self, event: KillSwitchEvent) -> None:
        """Activate kill switch and log the event."""
        self._state.kill_switch_active = True
        self._state.kill_switch_reason = event.reason
        logger.critical(
            "KILL SWITCH TRIGGERED",
            extra={
                "reason": event.reason.value,
                "description": event.description,
                "trigger_value": event.trigger_value,
                "threshold": event.threshold,
            }
        )
        if self.audit_logger:
            self.audit_logger.log_kill_switch_triggered(
                reason=event.description,
                account_state={
                    "balance": event.account_balance,
                    "equity": event.account_equity,
                    "reason": event.reason.value,
                }
            )

    def reset_kill_switch(self, authorized_by: str) -> None:
        """
        Manually reset kill switch. Requires explicit authorization.

        Args:
            authorized_by: User/system that authorized the reset
        """
        if not self._state.kill_switch_active:
            logger.warning("Kill switch reset called but not active")
            return

        self._state.kill_switch_active = False
        self._state.kill_switch_reason = None
        logger.warning(f"Kill switch RESET by: {authorized_by}")

        if self.audit_logger:
            self.audit_logger.log_kill_switch_reset(reset_by=authorized_by)

    def update_after_trade_close(
        self,
        pnl_r: float,
        account: AccountState,
    ) -> None:
        """
        Update risk state after a trade closes.

        Args:
            pnl_r: Trade P&L in R-multiples (+1.0 = 1R win, -1.0 = 1R loss)
            account: Updated account state after trade close

        Called by M10 TradeExecutor after every trade close.
        """
        if pnl_r < 0:
            self._state.consecutive_losses += 1
            self._state.losses_today += 1
        else:
            self._state.consecutive_losses = 0  # Reset on any win

        self._state.trades_today += 1
        self._state.daily_pnl_pct = account.daily_pnl_pct
        self._state.weekly_pnl_pct = account.weekly_pnl_pct
        self._state.drawdown_from_peak_pct = account.drawdown_from_peak_pct
        self._state.current_balance = account.balance
        self._state.current_equity = account.equity

        # Check kill switch after each trade
        self.check_kill_switch(account)

    def reset_daily_state(self) -> None:
        """Reset daily counters. Called at start of each trading day."""
        self._state.trades_today = 0
        self._state.losses_today = 0
        self._state.daily_pnl_pct = 0.0
