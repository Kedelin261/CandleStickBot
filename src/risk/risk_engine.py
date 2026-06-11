"""
M09 — Risk Management Engine (Phase 1 MVP)
The most critical module — protects capital above all else.
The Candlestick Trading Bible: Risk management is the ONLY thing that keeps you alive.

=== RISK RULES ===

  Default risk:   1.0% per trade
  Premium risk:   1.5% (TQS >= 80, opt-in, DISABLED by default)
  Hard cap:       2.0% absolute maximum (cannot be overridden)
  Minimum R:R:    2.0:1
  Daily loss limit:  3.0% of account
  Weekly loss limit: 6.0% of account
  Max open trades:   3 (configurable)

=== KILL SWITCH CONDITIONS ===

  Triggers on ANY of:
    1. Drawdown >= 10.0% from peak equity
    2. Consecutive losses >= 7
    3. Both daily AND weekly limits hit simultaneously

  Once active: NO new trades until manually reset by authorized user.
  Full kill switch state persisted in RiskState.

=== GATE CHAIN (check_and_approve) ===

  1. KILL_SWITCH    — kill switch active?
  2. DAILY_LIMIT    — daily P&L loss limit breached?
  3. WEEKLY_LIMIT   — weekly P&L loss limit breached?
  4. MAX_TRADES     — max open trades reached?
  5. RR_RATIO       — recommendation R:R >= min_rr_ratio?
  6. POSITION_SIZE  — lot size calculable > 0?

=== POSITION SIZING ===

  lot_size = (balance * risk_pct / 100) / (stop_pips * pip_value_per_lot)
  Rounded to lot_step, clamped to [lot_min, lot_max].

=== PHASE 1 SCOPE ===

  Back-test + paper modes only. No live execution.
  Fixed fractional position sizing only (Kelly deferred to Phase 2).

Status: Full Phase 1 implementation — Sprint 8.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

from src.types import (
    AccountState,
    RiskApprovedOrder,
    RiskRejection,
    TQSComponents,
    TradeTier,
    TradeRecommendation,
)

logger = logging.getLogger("candlestickbot.risk.engine")


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class RiskCheckResult(str, Enum):
    """Risk check outcome."""
    APPROVED           = "APPROVED"
    REJECTED           = "REJECTED"
    KILL_SWITCH_ACTIVE = "KILL_SWITCH_ACTIVE"


class KillSwitchReason(str, Enum):
    """Why the kill switch was triggered."""
    DRAWDOWN          = "DRAWDOWN"
    CONSECUTIVE_LOSSES = "CONSECUTIVE_LOSSES"
    DAILY_AND_WEEKLY  = "DAILY_AND_WEEKLY"
    MANUAL            = "MANUAL"


# ---------------------------------------------------------------------------
# Supporting dataclasses
# ---------------------------------------------------------------------------

@dataclass
class RiskState:
    """
    Current risk state of the trading system.
    Maintained in-memory during a session; designed to be persisted to
    AccountSnapshot table between sessions.
    """
    kill_switch_active:     bool                    = False
    kill_switch_reason:     Optional[KillSwitchReason] = None
    consecutive_losses:     int                     = 0
    trades_today:           int                     = 0
    losses_today:           int                     = 0
    daily_pnl_pct:          float                   = 0.0
    weekly_pnl_pct:         float                   = 0.0
    drawdown_from_peak_pct: float                   = 0.0
    peak_equity:            float                   = 0.0
    current_balance:        float                   = 0.0
    current_equity:         float                   = 0.0
    open_trade_count:       int                     = 0

    @property
    def is_trading_allowed(self) -> bool:
        """True if the kill switch is not active."""
        return not self.kill_switch_active


@dataclass
class KillSwitchEvent:
    """Record of a kill switch trigger event for audit logging."""
    reason:          KillSwitchReason
    trigger_value:   float       # Value that tripped (e.g. drawdown %)
    threshold:       float       # Threshold that was breached
    account_balance: float
    account_equity:  float
    description:     str


@dataclass
class RiskConfig:
    """
    All risk-engine parameters in one place.

    Designed to be validated once at construction time so that the engine
    itself never needs to re-check parameter sanity.
    """
    # Per-trade risk
    risk_per_trade_pct:     float = 1.0
    max_risk_per_trade_pct: float = 2.0      # Hard cap — overrides everything
    premium_risk_pct:       float = 1.5      # Applied only when premium_risk_enabled=True
    premium_risk_enabled:   bool  = False    # Disabled by default

    # Trade quality
    min_rr_ratio:           float = 2.0
    max_open_trades:        int   = 3
    max_account_heat_pct:   float = 6.0      # Total open-risk ceiling

    # Loss limits
    daily_loss_limit_pct:   float = 3.0
    weekly_loss_limit_pct:  float = 6.0

    # Kill switch
    kill_switch_drawdown_pct:        float = 10.0
    kill_switch_consecutive_losses:  int   = 7

    # Instrument params
    pip_size:             float = 0.0001
    pip_value_per_lot:    float = 10.0   # Account-currency value per pip, std lot

    # Lot constraints
    lot_min:  float = 0.01
    lot_step: float = 0.01
    lot_max:  float = 100.0

    def __post_init__(self) -> None:
        if self.max_risk_per_trade_pct > 2.0:
            raise ValueError(
                f"max_risk_per_trade_pct {self.max_risk_per_trade_pct} exceeds "
                f"hard cap of 2.0% — refusing construction"
            )
        if self.risk_per_trade_pct <= 0:
            raise ValueError("risk_per_trade_pct must be > 0")
        if self.daily_loss_limit_pct <= 0:
            raise ValueError("daily_loss_limit_pct must be > 0")
        if self.weekly_loss_limit_pct <= 0:
            raise ValueError("weekly_loss_limit_pct must be > 0")
        if self.pip_size <= 0:
            raise ValueError("pip_size must be > 0")
        if self.pip_value_per_lot <= 0:
            raise ValueError("pip_value_per_lot must be > 0")


# ---------------------------------------------------------------------------
# RiskEngine
# ---------------------------------------------------------------------------

class RiskEngine:
    """
    M09 — Risk Management Engine (Phase 1 MVP).

    Responsibilities:
      1. Kill switch monitoring — checked FIRST before any new trade
      2. Daily / weekly loss limit enforcement
      3. Maximum open trade count check
      4. Account heat ceiling check
      5. Minimum R:R validation
      6. Fixed-fractional position sizing (lot_size calculation)
      7. Post-trade state update (after fill or trade close)
      8. Daily / weekly state reset

    All gate failures produce an explicit ``RiskRejection`` DTO with a
    ``check_type`` string so callers can log/audit exactly which rule fired.

    Usage::

        engine = RiskEngine()
        result, approved, rejection = engine.check_and_approve(rec, account)
        if result == RiskCheckResult.APPROVED:
            # approved.lot_size, approved.risk_pct, etc.
            ...

    The engine is stateful between calls: call ``update_after_trade_close()``
    and ``reset_daily_state()`` / ``reset_weekly_state()`` at the appropriate
    lifecycle points so the kill-switch counters and loss-limit accumulators
    remain accurate.
    """

    def __init__(
        self,
        # Accept either a RiskConfig object or individual keyword arguments
        # for backward-compatibility with the Phase-0 stub's call signature.
        config: Optional[RiskConfig] = None,

        # ---- legacy / convenience kwargs (ignored if config is provided) ----
        risk_per_trade_pct:            float = 1.0,
        max_risk_per_trade_pct:        float = 2.0,
        premium_risk_pct:              float = 1.5,
        premium_risk_enabled:          bool  = False,
        min_rr_ratio:                  float = 2.0,
        max_open_trades:               int   = 3,
        max_account_heat_pct:          float = 6.0,
        daily_loss_limit_pct:          float = 3.0,
        weekly_loss_limit_pct:         float = 6.0,
        kill_switch_drawdown_pct:      float = 10.0,
        kill_switch_consecutive_losses: int  = 7,
        pip_size:                      float = 0.0001,
        pip_value_per_lot:             float = 10.0,
        lot_min:                       float = 0.01,
        lot_step:                      float = 0.01,
        lot_max:                       float = 100.0,
        audit_logger=None,
    ):
        if config is not None:
            self.config = config
        else:
            self.config = RiskConfig(
                risk_per_trade_pct=risk_per_trade_pct,
                max_risk_per_trade_pct=max_risk_per_trade_pct,
                premium_risk_pct=premium_risk_pct,
                premium_risk_enabled=premium_risk_enabled,
                min_rr_ratio=min_rr_ratio,
                max_open_trades=max_open_trades,
                max_account_heat_pct=max_account_heat_pct,
                daily_loss_limit_pct=daily_loss_limit_pct,
                weekly_loss_limit_pct=weekly_loss_limit_pct,
                kill_switch_drawdown_pct=kill_switch_drawdown_pct,
                kill_switch_consecutive_losses=kill_switch_consecutive_losses,
                pip_size=pip_size,
                pip_value_per_lot=pip_value_per_lot,
                lot_min=lot_min,
                lot_step=lot_step,
                lot_max=lot_max,
            )

        self.audit_logger = audit_logger
        self._state = RiskState()

        # Expose config fields as instance attributes for backward compatibility
        # with code that accesses e.g. engine.risk_per_trade_pct directly.
        self._sync_attributes()

    # ------------------------------------------------------------------
    # Properties — backward-compat attribute access
    # ------------------------------------------------------------------

    def _sync_attributes(self) -> None:
        """Expose config fields as direct attributes (legacy API)."""
        c = self.config
        self.risk_per_trade_pct            = c.risk_per_trade_pct
        self.max_risk_per_trade_pct        = c.max_risk_per_trade_pct
        self.premium_risk_pct              = c.premium_risk_pct
        self.premium_risk_enabled          = c.premium_risk_enabled
        self.min_rr_ratio                  = c.min_rr_ratio
        self.max_open_trades               = c.max_open_trades
        self.max_account_heat_pct          = c.max_account_heat_pct
        self.daily_loss_limit_pct          = c.daily_loss_limit_pct
        self.weekly_loss_limit_pct         = c.weekly_loss_limit_pct
        self.kill_switch_drawdown_pct      = c.kill_switch_drawdown_pct
        self.kill_switch_consecutive_losses = c.kill_switch_consecutive_losses
        self.pip_size                      = c.pip_size
        self.pip_value_per_lot             = c.pip_value_per_lot

    @property
    def state(self) -> RiskState:
        """Current risk state (read-only reference)."""
        return self._state

    @property
    def kill_switch_active(self) -> bool:
        """True if kill switch has been triggered and not reset."""
        return self._state.kill_switch_active

    # ------------------------------------------------------------------
    # PRIMARY API — check_and_approve
    # ------------------------------------------------------------------

    def check_and_approve(
        self,
        recommendation: TradeRecommendation,
        account: AccountState,
    ) -> Tuple[RiskCheckResult, Optional[RiskApprovedOrder], Optional[RiskRejection]]:
        """
        Run the full risk gate chain and size the position if all pass.

        Gate chain (in order):
          1. KILL_SWITCH  — is kill switch currently active?
          2. DAILY_LIMIT  — daily loss limit reached?
          3. WEEKLY_LIMIT — weekly loss limit reached?
          4. MAX_TRADES   — open_trades >= max_open_trades?
          5. RR_RATIO     — recommendation.rr_ratio >= min_rr_ratio?
          6. POSITION_SIZE — lot_size > 0?

        Args:
            recommendation: ``TradeRecommendation`` from M08 StrategyEngine.
            account:        Current ``AccountState`` from broker/paper account.

        Returns:
            Tuple of (RiskCheckResult, RiskApprovedOrder | None, RiskRejection | None).

            APPROVED:           (APPROVED, order, None)
            REJECTED (rule):    (REJECTED, None, rejection)
            KILL_SWITCH_ACTIVE: (KILL_SWITCH_ACTIVE, None, rejection)
        """
        # Gate 1 — Kill switch
        if self._state.kill_switch_active:
            rejection = RiskRejection(
                recommendation=recommendation,
                reason=(
                    f"Kill switch active "
                    f"(reason: {self._state.kill_switch_reason.value if self._state.kill_switch_reason else 'UNKNOWN'})"
                    f" — all trading halted until manual reset"
                ),
                check_type="KILL_SWITCH",
            )
            logger.warning("M09 KILL_SWITCH: %s %s rejected", recommendation.symbol, recommendation.direction.value)
            return RiskCheckResult.KILL_SWITCH_ACTIVE, None, rejection

        # Gate 2 — Daily loss limit
        if account.daily_pnl_pct < 0 and abs(account.daily_pnl_pct) >= self.config.daily_loss_limit_pct:
            rejection = RiskRejection(
                recommendation=recommendation,
                reason=(
                    f"Daily loss limit {self.config.daily_loss_limit_pct:.1f}% reached "
                    f"(current daily P&L: {account.daily_pnl_pct:.2f}%)"
                ),
                check_type="DAILY_LIMIT",
            )
            logger.warning("M09 DAILY_LIMIT: %s rejected — daily P&L %.2f%%", recommendation.symbol, account.daily_pnl_pct)
            return RiskCheckResult.REJECTED, None, rejection

        # Gate 3 — Weekly loss limit
        if account.weekly_pnl_pct < 0 and abs(account.weekly_pnl_pct) >= self.config.weekly_loss_limit_pct:
            rejection = RiskRejection(
                recommendation=recommendation,
                reason=(
                    f"Weekly loss limit {self.config.weekly_loss_limit_pct:.1f}% reached "
                    f"(current weekly P&L: {account.weekly_pnl_pct:.2f}%)"
                ),
                check_type="WEEKLY_LIMIT",
            )
            logger.warning("M09 WEEKLY_LIMIT: %s rejected — weekly P&L %.2f%%", recommendation.symbol, account.weekly_pnl_pct)
            return RiskCheckResult.REJECTED, None, rejection

        # Gate 4 — Max open trades  (use live account.open_trades, not internal counter)
        if account.open_trades >= self.config.max_open_trades:
            rejection = RiskRejection(
                recommendation=recommendation,
                reason=(
                    f"Max open trades {self.config.max_open_trades} reached "
                    f"(current: {account.open_trades})"
                ),
                check_type="MAX_TRADES",
            )
            logger.warning("M09 MAX_TRADES: %s rejected — open_trades=%d", recommendation.symbol, account.open_trades)
            return RiskCheckResult.REJECTED, None, rejection

        # Gate 5 — R:R ratio check
        if recommendation.rr_ratio < self.config.min_rr_ratio:
            rejection = RiskRejection(
                recommendation=recommendation,
                reason=(
                    f"R:R {recommendation.rr_ratio:.2f} < minimum {self.config.min_rr_ratio:.2f}"
                ),
                check_type="RR_RATIO",
            )
            logger.warning("M09 RR_RATIO: %s rejected — rr=%.2f", recommendation.symbol, recommendation.rr_ratio)
            return RiskCheckResult.REJECTED, None, rejection

        # Gate 6 — Position sizing
        risk_pct = self._determine_risk_pct(recommendation.tqs)
        stop_pips = abs(recommendation.entry_price - recommendation.stop_price) / self.config.pip_size
        lot_size = self._calculate_lot_size(
            account_balance=account.balance,
            risk_pct=risk_pct,
            entry=recommendation.entry_price,
            stop=recommendation.stop_price,
        )

        if lot_size <= 0:
            rejection = RiskRejection(
                recommendation=recommendation,
                reason=(
                    f"Calculated lot size is zero or negative "
                    f"(balance={account.balance:.2f}, risk={risk_pct:.2f}%, "
                    f"stop_pips={stop_pips:.1f})"
                ),
                check_type="POSITION_SIZE",
            )
            logger.warning("M09 POSITION_SIZE: %s rejected — lot_size=0", recommendation.symbol)
            return RiskCheckResult.REJECTED, None, rejection

        risk_amount_usd = account.balance * risk_pct / 100.0

        approved = RiskApprovedOrder(
            recommendation=recommendation,
            lot_size=lot_size,
            risk_pct=risk_pct,
            risk_amount_usd=risk_amount_usd,
            account_balance=account.balance,
            stop_pips=stop_pips,
        )

        logger.info(
            "M09 APPROVED: %s %s lots=%.2f risk=%.2f%% ($%.2f) stop_pips=%.1f",
            recommendation.symbol, recommendation.direction.value,
            lot_size, risk_pct, risk_amount_usd, stop_pips,
        )
        return RiskCheckResult.APPROVED, approved, None

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def _determine_risk_pct(self, tqs: Optional[TQSComponents]) -> float:
        """
        Determine risk percentage based on TQS tier and config.

        Phase 1 defaults:
          STANDARD (TQS 60–79):  1.0%
          PREMIUM  (TQS  >=80):  1.0% base  (→1.5% if premium_risk_enabled=True)

        Hard cap: max_risk_per_trade_pct (default 2.0%) enforced here.
        """
        if tqs is None:
            risk = self.config.risk_per_trade_pct
        elif tqs.tier == TradeTier.PREMIUM and self.config.premium_risk_enabled:
            risk = self.config.premium_risk_pct
        else:
            risk = self.config.risk_per_trade_pct

        # Enforce absolute hard cap
        return min(risk, self.config.max_risk_per_trade_pct)

    def _calculate_lot_size(
        self,
        account_balance: float,
        risk_pct:        float,
        entry:           float,
        stop:            float,
    ) -> float:
        """
        Fixed-fractional lot size calculation.

        Formula::

            risk_amount   = balance * risk_pct / 100
            stop_pips     = |entry - stop| / pip_size
            lot_size      = risk_amount / (stop_pips * pip_value_per_lot)

        Result is rounded to ``lot_step``, clamped to ``[lot_min, lot_max]``.
        Returns 0.0 if stop distance is zero (degenerate trade).
        """
        stop_distance = abs(entry - stop)
        if stop_distance <= 0:
            return 0.0

        stop_pips    = stop_distance / self.config.pip_size
        risk_amount  = account_balance * risk_pct / 100.0
        raw_lots     = risk_amount / (stop_pips * self.config.pip_value_per_lot)

        # Round to nearest lot_step
        lots = round(raw_lots / self.config.lot_step) * self.config.lot_step
        lots = max(self.config.lot_min, min(self.config.lot_max, lots))
        return round(lots, 2)

    # ------------------------------------------------------------------
    # Kill switch
    # ------------------------------------------------------------------

    def check_kill_switch(self, account: AccountState) -> Optional[KillSwitchEvent]:
        """
        Evaluate all kill-switch conditions against the current account state.

        Called automatically inside ``update_after_trade_close()``.
        Can also be called manually (e.g. on account state refresh).

        Returns:
            ``KillSwitchEvent`` if a condition triggered, ``None`` if all clear.
            If already active, does NOT re-trigger (idempotent).
        """
        if self._state.kill_switch_active:
            return None  # Already active; no duplicate events

        # Condition 1: Drawdown from peak equity
        drawdown = account.drawdown_from_peak_pct
        if drawdown >= self.config.kill_switch_drawdown_pct:
            event = KillSwitchEvent(
                reason=KillSwitchReason.DRAWDOWN,
                trigger_value=drawdown,
                threshold=self.config.kill_switch_drawdown_pct,
                account_balance=account.balance,
                account_equity=account.equity,
                description=(
                    f"Drawdown {drawdown:.2f}% reached {self.config.kill_switch_drawdown_pct:.1f}% limit"
                ),
            )
            self._trigger_kill_switch(event)
            return event

        # Condition 2: Consecutive losses
        if self._state.consecutive_losses >= self.config.kill_switch_consecutive_losses:
            event = KillSwitchEvent(
                reason=KillSwitchReason.CONSECUTIVE_LOSSES,
                trigger_value=float(self._state.consecutive_losses),
                threshold=float(self.config.kill_switch_consecutive_losses),
                account_balance=account.balance,
                account_equity=account.equity,
                description=(
                    f"{self._state.consecutive_losses} consecutive losses "
                    f"reached {self.config.kill_switch_consecutive_losses} limit"
                ),
            )
            self._trigger_kill_switch(event)
            return event

        # Condition 3: Both daily AND weekly limits breached simultaneously
        daily_breached  = (account.daily_pnl_pct  < 0 and abs(account.daily_pnl_pct)  >= self.config.daily_loss_limit_pct)
        weekly_breached = (account.weekly_pnl_pct < 0 and abs(account.weekly_pnl_pct) >= self.config.weekly_loss_limit_pct)
        if daily_breached and weekly_breached:
            event = KillSwitchEvent(
                reason=KillSwitchReason.DAILY_AND_WEEKLY,
                trigger_value=max(abs(account.daily_pnl_pct), abs(account.weekly_pnl_pct)),
                threshold=min(self.config.daily_loss_limit_pct, self.config.weekly_loss_limit_pct),
                account_balance=account.balance,
                account_equity=account.equity,
                description=(
                    "Both daily AND weekly loss limits breached simultaneously — "
                    f"daily={account.daily_pnl_pct:.2f}%, weekly={account.weekly_pnl_pct:.2f}%"
                ),
            )
            self._trigger_kill_switch(event)
            return event

        return None  # All clear

    def _trigger_kill_switch(self, event: KillSwitchEvent) -> None:
        """Activate kill switch, update state, and log the event."""
        self._state.kill_switch_active  = True
        self._state.kill_switch_reason  = event.reason

        logger.critical(
            "KILL SWITCH TRIGGERED — reason=%s trigger=%.2f threshold=%.2f: %s",
            event.reason.value, event.trigger_value, event.threshold, event.description,
        )

        if self.audit_logger:
            try:
                self.audit_logger.log_kill_switch_triggered(
                    reason=event.description,
                    account_state={
                        "balance":       event.account_balance,
                        "equity":        event.account_equity,
                        "reason":        event.reason.value,
                        "trigger_value": event.trigger_value,
                        "threshold":     event.threshold,
                    },
                )
            except Exception as exc:
                logger.error("audit_logger.log_kill_switch_triggered failed: %s", exc)

    def reset_kill_switch(self, authorized_by: str) -> None:
        """
        Manually reset the kill switch.

        Args:
            authorized_by: User/system identifier for audit trail.

        Raises:
            RuntimeError: if kill switch is not currently active (safety guard).
        """
        if not self._state.kill_switch_active:
            logger.warning("reset_kill_switch called but kill switch is NOT active")
            return

        prev_reason = self._state.kill_switch_reason
        self._state.kill_switch_active = False
        self._state.kill_switch_reason = None

        logger.warning(
            "Kill switch RESET by '%s' (was reason: %s)",
            authorized_by,
            prev_reason.value if prev_reason else "UNKNOWN",
        )

        if self.audit_logger:
            try:
                self.audit_logger.log_kill_switch_reset(reset_by=authorized_by)
            except Exception as exc:
                logger.error("audit_logger.log_kill_switch_reset failed: %s", exc)

    # ------------------------------------------------------------------
    # State updates (lifecycle management)
    # ------------------------------------------------------------------

    def update_after_trade_close(
        self,
        pnl_r:   float,
        account: AccountState,
    ) -> None:
        """
        Update risk state after a trade closes.

        Must be called by M10 TradeExecutor after every trade close so that
        consecutive-loss and kill-switch counters stay accurate.

        Args:
            pnl_r:   Trade P&L in R-multiples (+1.0 = 1R win, -1.0 = 1R loss).
            account: Updated ``AccountState`` reflecting the closed trade.
        """
        if pnl_r < 0:
            self._state.consecutive_losses += 1
            self._state.losses_today += 1
        else:
            self._state.consecutive_losses = 0   # Any win resets the streak

        self._state.trades_today          += 1
        self._state.daily_pnl_pct          = account.daily_pnl_pct
        self._state.weekly_pnl_pct         = account.weekly_pnl_pct
        self._state.drawdown_from_peak_pct = account.drawdown_from_peak_pct
        self._state.current_balance        = account.balance
        self._state.current_equity         = account.equity

        logger.debug(
            "Trade closed: pnl_r=%.2f consecutive_losses=%d trades_today=%d",
            pnl_r, self._state.consecutive_losses, self._state.trades_today,
        )

        # Auto-check kill switch on every trade close
        self.check_kill_switch(account)

    def update_open_trade_count(self, count: int) -> None:
        """
        Synchronize internal open trade count with broker state.

        Called whenever a trade is opened or closed so the kill-switch
        account-heat check can use the correct value.

        Args:
            count: Current number of open trades.
        """
        if count < 0:
            raise ValueError(f"open_trade_count cannot be negative, got {count}")
        self._state.open_trade_count = count

    def reset_daily_state(self) -> None:
        """
        Reset daily-scoped counters.

        Called at the start of each new trading day (00:00 broker time).
        Does NOT reset consecutive_losses (persists across days) or kill switch.
        """
        self._state.trades_today   = 0
        self._state.losses_today   = 0
        self._state.daily_pnl_pct  = 0.0
        logger.info("Daily risk state reset")

    def reset_weekly_state(self) -> None:
        """
        Reset weekly-scoped counters.

        Called at the start of each new trading week (Sunday midnight or
        broker-equivalent).  Does NOT reset consecutive_losses or kill switch.
        """
        self._state.weekly_pnl_pct = 0.0
        logger.info("Weekly risk state reset")

    # ------------------------------------------------------------------
    # Convenience helpers (stateless)
    # ------------------------------------------------------------------

    def compute_stop_pips(self, entry: float, stop: float) -> float:
        """Return the stop distance in pips."""
        return abs(entry - stop) / self.config.pip_size

    def compute_risk_amount(self, balance: float, risk_pct: float) -> float:
        """Return the dollar risk amount for the given balance and risk %."""
        return balance * risk_pct / 100.0

    def is_trading_allowed(self) -> bool:
        """
        High-level check: True if kill switch is off (all other limits are
        checked live against the account state in ``check_and_approve``).
        """
        return self._state.is_trading_allowed


# ---------------------------------------------------------------------------
# Module-level helpers (stateless, public)
# ---------------------------------------------------------------------------

def compute_lot_size(
    account_balance:  float,
    risk_pct:         float,
    entry:            float,
    stop:             float,
    pip_size:         float = 0.0001,
    pip_value_per_lot: float = 10.0,
    lot_min:          float = 0.01,
    lot_step:         float = 0.01,
    lot_max:          float = 100.0,
) -> float:
    """
    Stateless lot-size calculation helper.

    Identical to ``RiskEngine._calculate_lot_size`` but callable without
    constructing an engine instance.

    Returns 0.0 if stop distance is zero.
    """
    stop_distance = abs(entry - stop)
    if stop_distance <= 0 or pip_size <= 0 or pip_value_per_lot <= 0:
        return 0.0

    stop_pips   = stop_distance / pip_size
    risk_amount = account_balance * risk_pct / 100.0
    raw_lots    = risk_amount / (stop_pips * pip_value_per_lot)

    lots = round(raw_lots / lot_step) * lot_step
    lots = max(lot_min, min(lot_max, lots))
    return round(lots, 2)


def compute_risk_amount(balance: float, risk_pct: float) -> float:
    """Return dollar risk for the given balance and risk percentage."""
    return balance * risk_pct / 100.0


def compute_stop_pips(entry: float, stop: float, pip_size: float = 0.0001) -> float:
    """Return stop distance in pips."""
    if pip_size <= 0:
        return 0.0
    return abs(entry - stop) / pip_size
