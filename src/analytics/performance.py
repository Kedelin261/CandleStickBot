"""
M18 — Strategy Analytics & Performance Tracking
Computes per-strategy scorecards and generates monthly reports.
The Candlestick Trading Bible: Track your results to improve your edge.

Phase 1 Scope:
  - Per-strategy scorecard: total trades, win rate, PF, avg R, max DD
  - Monthly summary report generation
  - Strategy degradation detection (alert when edge deteriorating)
  - Promotion criteria evaluation

Promotion Criteria (AND logic — both required):
  - >= 50 completed trades (not just signals)
  - >= 3 calendar months in current mode
  - Both conditions must be met simultaneously

Promotion: Backtest → Paper → Live (each step requires criteria)

Optimization Governance:
  - Baseline must pass (PF >= 1.1) BEFORE optimization is allowed
  - Phase 1: Optimization engine disabled regardless

Status: STUB — Phase 0 scaffold. Full implementation in Phase 1 Sprint 5.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("candlestickbot.analytics.performance")


@dataclass
class StrategyScorecard:
    """
    Per-strategy performance scorecard.
    Generated from completed trades for a specific strategy.
    """
    strategy: str
    period_start: datetime
    period_end: datetime

    # Trade counts
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    breakeven_trades: int = 0

    # P&L metrics
    total_pnl_r: float = 0.0      # Total P&L in R-multiples
    gross_profit_r: float = 0.0
    gross_loss_r: float = 0.0

    # Derived metrics
    win_rate: float = 0.0
    profit_factor: float = 0.0
    expectancy_r: float = 0.0
    avg_win_r: float = 0.0
    avg_loss_r: float = 0.0
    max_consecutive_losses: int = 0
    max_drawdown_pct: float = 0.0

    # TQS analysis
    avg_tqs_winners: float = 0.0
    avg_tqs_losers: float = 0.0

    # Promotion progress
    months_in_mode: int = 0
    meets_promotion_criteria: bool = False

    def calculate(self, trades: list) -> "StrategyScorecard":
        """Calculate scorecard from trade list."""
        # TODO: Implementation in Phase 1 Sprint 5
        logger.warning("StrategyScorecard.calculate() — STUB")
        self.total_trades = len(trades)
        return self


@dataclass
class PromotionCheck:
    """Result of promotion criteria evaluation."""
    strategy: str
    current_mode: str
    target_mode: str

    # Criteria values
    completed_trades: int = 0
    min_trades_required: int = 50       # AND condition
    calendar_months: int = 0
    min_months_required: int = 3        # AND condition

    # Results
    trades_criteria_met: bool = False
    months_criteria_met: bool = False
    all_criteria_met: bool = False      # True ONLY if BOTH criteria met

    reason: str = ""


class PerformanceAnalytics:
    """
    M18 — Performance Analytics Engine.

    Phase 1: Calculate scorecard from completed trades.
    Phase 2: Full monthly reporting, visualization, strategy comparison.
    """

    def __init__(
        self,
        min_promotion_trades: int = 50,
        min_promotion_months: int = 3,
        degradation_window: int = 20,     # Rolling window for degradation check
        degradation_pf_threshold: float = 1.0,  # Alert if rolling PF < 1.0
        audit_logger=None,
        db_session=None,
    ):
        self.min_promotion_trades = min_promotion_trades
        self.min_promotion_months = min_promotion_months
        self.degradation_window = degradation_window
        self.degradation_pf_threshold = degradation_pf_threshold
        self.audit_logger = audit_logger
        self.db_session = db_session

    def calculate_scorecard(
        self,
        strategy: str,
        trades: list,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None,
    ) -> StrategyScorecard:
        """
        Calculate performance scorecard for a strategy.

        Args:
            strategy: Strategy name (e.g., "PIN_BAR")
            trades: List of completed Trade objects or trade dicts
            period_start: Analysis period start
            period_end: Analysis period end

        Returns:
            StrategyScorecard with all metrics calculated.
        """
        # TODO: Full implementation in Phase 1 Sprint 5
        logger.warning("PerformanceAnalytics.calculate_scorecard() — STUB")
        return StrategyScorecard(
            strategy=strategy,
            period_start=period_start or datetime.utcnow(),
            period_end=period_end or datetime.utcnow(),
            total_trades=len(trades),
        )

    def check_promotion_criteria(
        self,
        strategy: str,
        current_mode: str,
        completed_trades: int,
        mode_start_date: datetime,
    ) -> PromotionCheck:
        """
        Evaluate promotion criteria (AND logic — both conditions required).

        Promotion requires:
          - >= 50 completed trades (hard minimum)
          - >= 3 full calendar months in current mode
          - BOTH must be true simultaneously

        Args:
            strategy: Strategy name
            current_mode: Current execution mode
            completed_trades: Number of completed trades in current mode
            mode_start_date: When current mode was activated

        Returns:
            PromotionCheck with criteria evaluation.
        """
        # Target mode
        mode_progression = {"backtest": "paper", "paper": "live"}
        target_mode = mode_progression.get(current_mode, "unknown")

        # Calculate months since mode start
        now = datetime.utcnow()
        months = (
            (now.year - mode_start_date.year) * 12
            + (now.month - mode_start_date.month)
        )

        trades_ok = completed_trades >= self.min_promotion_trades
        months_ok = months >= self.min_promotion_months
        all_ok = trades_ok and months_ok  # AND logic — both required

        reason_parts = []
        if not trades_ok:
            reason_parts.append(
                f"Need {self.min_promotion_trades - completed_trades} more trades "
                f"({completed_trades}/{self.min_promotion_trades})"
            )
        if not months_ok:
            reason_parts.append(
                f"Need {self.min_promotion_months - months} more months "
                f"({months}/{self.min_promotion_months})"
            )

        if all_ok:
            reason = "All promotion criteria met ✓"
        else:
            reason = "; ".join(reason_parts)

        return PromotionCheck(
            strategy=strategy,
            current_mode=current_mode,
            target_mode=target_mode,
            completed_trades=completed_trades,
            min_trades_required=self.min_promotion_trades,
            calendar_months=months,
            min_months_required=self.min_promotion_months,
            trades_criteria_met=trades_ok,
            months_criteria_met=months_ok,
            all_criteria_met=all_ok,
            reason=reason,
        )

    def check_strategy_degradation(
        self,
        trades: list,
        window: Optional[int] = None,
    ) -> Tuple[bool, str]:
        """
        Check if strategy edge is degrading over recent trades.

        Uses rolling window profit factor: if last N trades PF < threshold,
        alert for review.

        Args:
            trades: All completed trades (time-ordered)
            window: Number of recent trades to analyze (default: degradation_window)

        Returns:
            Tuple of (is_degrading, reason_string)
        """
        # TODO: Implementation in Phase 1 Sprint 5
        return False, "STUB — not yet implemented"
