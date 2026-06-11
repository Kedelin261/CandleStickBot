"""
M11 — Backtesting Engine
Historical simulation of the trading system using stored candle data.
The Candlestick Trading Bible: Test before you trust.

Backtesting Methodology:
  - Walk-forward simulation: process candles in chronological order
  - No lookahead bias: each bar evaluation uses only past data
  - Realistic slippage: configurable pip slippage on fill
  - Conservative exit logic: SL takes precedence over TP if both hit same bar
  - Full trade log: every signal, rejection, and trade recorded

Phase 1 Scope:
  - EURUSD D1 only
  - Pin Bar + Engulfing Bar strategies
  - Date range: configurable (default: 5 years)
  - Minimum data requirement: 200 candles (for indicator warm-up)

Results Output:
  - Performance metrics (M18): Total trades, win rate, PF, avg R, max DD
  - Trade log: Full entry/exit/TQS/reason for each trade
  - BacktestResult record: stored to DB for comparison

Optimization Governance (Phase 2):
  - Baseline must pass before optimization is allowed
  - Minimum baseline: PF >= 1.1, win rate >= 40%, max DD <= 20%
  - Optimization DISABLED in Phase 1

Status: STUB — Phase 0 scaffold. Full implementation in Phase 1 Sprint 4.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger("candlestickbot.backtesting.engine")


class BacktestStatus(str, Enum):
    """Backtest run status."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass
class BacktestConfig:
    """Configuration for a backtest run."""
    symbol: str = "EURUSD"
    timeframe: str = "D1"
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    initial_balance: float = 10000.0
    commission_per_lot: float = 7.0     # Round-trip commission in USD
    slippage_pips: float = 0.5
    active_strategies: List[str] = field(
        default_factory=lambda: ["PIN_BAR", "ENGULFING_BAR"]
    )
    run_id: Optional[str] = None
    notes: str = ""


@dataclass
class TradeRecord:
    """Single trade record in backtest results."""
    trade_id: str
    symbol: str
    timeframe: str
    strategy: str
    direction: str
    entry_date: datetime
    exit_date: Optional[datetime]
    entry_price: float
    exit_price: float
    stop_price: float
    target_price: float
    lot_size: float
    pnl_pips: float
    pnl_usd: float
    pnl_r: float
    exit_reason: str           # "SL", "TP", "MANUAL"
    tqs_total: int
    tqs_tier: str
    tqs_trend: int
    tqs_level: int
    tqs_pattern: int
    tqs_regime: int


@dataclass
class BacktestMetrics:
    """
    Aggregated performance metrics from a completed backtest.
    Mirrors StrategySummary ORM model in db/models.py.
    """
    # Trade counts
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    breakeven_trades: int = 0

    # P&L metrics
    gross_profit_usd: float = 0.0
    gross_loss_usd: float = 0.0
    net_profit_usd: float = 0.0
    net_profit_pct: float = 0.0

    # Efficiency metrics
    win_rate: float = 0.0       # 0.0-1.0
    profit_factor: float = 0.0  # Gross profit / Gross loss
    average_win_r: float = 0.0  # Average win in R-multiples
    average_loss_r: float = 0.0  # Average loss in R-multiples
    average_rr_actual: float = 0.0  # Actual average R:R
    expectancy_r: float = 0.0   # Expected value per trade in R

    # Drawdown
    max_drawdown_pct: float = 0.0
    max_drawdown_usd: float = 0.0
    max_consecutive_losses: int = 0
    max_consecutive_wins: int = 0

    # System quality
    sharpe_ratio: float = 0.0   # Risk-adjusted return
    calmar_ratio: float = 0.0   # Annual return / Max drawdown

    # Baseline check
    passes_baseline: bool = False  # PF >= 1.1 AND win_rate >= 0.4 AND max_DD <= 20%

    def calculate(self, trades: List[TradeRecord]) -> "BacktestMetrics":
        """Calculate all metrics from trade list."""
        if not trades:
            return self

        self.total_trades = len(trades)
        wins = [t for t in trades if t.pnl_r > 0]
        losses = [t for t in trades if t.pnl_r < 0]
        breakevens = [t for t in trades if t.pnl_r == 0]

        self.winning_trades = len(wins)
        self.losing_trades = len(losses)
        self.breakeven_trades = len(breakevens)

        self.gross_profit_usd = sum(t.pnl_usd for t in wins)
        self.gross_loss_usd = abs(sum(t.pnl_usd for t in losses))
        self.net_profit_usd = sum(t.pnl_usd for t in trades)

        self.win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0.0

        self.profit_factor = (
            self.gross_profit_usd / self.gross_loss_usd
            if self.gross_loss_usd > 0
            else float("inf")
        )

        self.average_win_r = sum(t.pnl_r for t in wins) / len(wins) if wins else 0.0
        self.average_loss_r = sum(t.pnl_r for t in losses) / len(losses) if losses else 0.0

        # Expectancy: (win_rate * avg_win_R) - ((1 - win_rate) * abs(avg_loss_R))
        self.expectancy_r = (
            self.win_rate * self.average_win_r
            - (1 - self.win_rate) * abs(self.average_loss_r)
        )

        # Baseline pass check
        self.passes_baseline = (
            self.profit_factor >= 1.1
            and self.win_rate >= 0.40
            and self.max_drawdown_pct <= 20.0
        )

        return self

    @property
    def summary_dict(self) -> Dict:
        """Return metrics as a dictionary for logging/reporting."""
        return {
            "total_trades": self.total_trades,
            "win_rate": f"{self.win_rate:.1%}",
            "profit_factor": f"{self.profit_factor:.2f}",
            "net_profit_usd": f"{self.net_profit_usd:.2f}",
            "expectancy_r": f"{self.expectancy_r:.2f}",
            "max_drawdown_pct": f"{self.max_drawdown_pct:.1f}%",
            "passes_baseline": self.passes_baseline,
        }


@dataclass
class BacktestResult:
    """Complete backtest run result."""
    config: BacktestConfig
    metrics: BacktestMetrics = field(default_factory=BacktestMetrics)
    trades: List[TradeRecord] = field(default_factory=list)
    status: BacktestStatus = BacktestStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    run_id: str = ""

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


class BacktestEngine:
    """
    M11 — Backtesting Engine.

    Walk-forward simulation with no lookahead bias.

    Processing loop (per candle):
    1. Update indicator windows (warm-up: skip first 200 bars)
    2. Check open trades for SL/TP exits (call TradeExecutor)
    3. Check kill switch (risk state after each close)
    4. Evaluate current bar via StrategyEngine → EvaluationResult
    5. If recommendation → submit to RiskEngine → TradeExecutor
    6. Advance to next candle

    Phase 1 constraints:
    - No live mode (backtest and paper only)
    - Single symbol, single timeframe
    - No portfolio management (Phase 2)
    - No optimization (Phase 2)
    """

    # Minimum bars needed before trading (indicator warm-up period)
    WARMUP_BARS = 200

    def __init__(
        self,
        config: Optional[BacktestConfig] = None,
        candle_store=None,        # M02 CandleStore
        strategy_engine=None,     # M08 StrategyEngine
        risk_engine=None,         # M09 RiskEngine
        trade_executor=None,      # M10 TradeExecutor
        analytics_engine=None,    # M18 AnalyticsEngine
        audit_logger=None,        # M13 AuditLogger
        db_session=None,
    ):
        self.config = config or BacktestConfig()
        self.candle_store = candle_store
        self.strategy_engine = strategy_engine
        self.risk_engine = risk_engine
        self.trade_executor = trade_executor
        self.analytics_engine = analytics_engine
        self.audit_logger = audit_logger
        self.db_session = db_session

    def run(self, run_id: Optional[str] = None) -> BacktestResult:
        """
        Execute the backtest simulation.

        Args:
            run_id: Optional unique identifier for this run.
                    Auto-generated if not provided.

        Returns:
            BacktestResult with full metrics and trade log.
        """
        import uuid
        run_id = run_id or str(uuid.uuid4())[:8]
        result = BacktestResult(
            config=self.config,
            run_id=run_id,
            status=BacktestStatus.PENDING,
        )

        # TODO: Full implementation in Phase 1 Sprint 4
        # Step 1: Load candles from CandleStore
        # Step 2: Validate data completeness
        # Step 3: Initialize components
        # Step 4: Walk forward through candles
        # Step 5: Calculate metrics
        # Step 6: Persist results

        logger.warning("BacktestEngine.run() — STUB")
        result.status = BacktestStatus.FAILED
        result.error_message = "STUB — not yet implemented"
        return result

    def _walk_forward(
        self,
        candles: list,
        result: BacktestResult,
    ) -> None:
        """
        Core walk-forward simulation loop.

        For each candle at index i (after warm-up):
        1. Build window: candles[0:i+1] (no lookahead)
        2. Check exits for open trades
        3. Evaluate new signals if no max trades
        4. Log everything
        """
        # TODO: Implementation in Phase 1 Sprint 4
        pass

    def _check_drawdown_curve(self, trades: List[TradeRecord], initial_balance: float) -> float:
        """
        Calculate maximum drawdown from trade equity curve.

        Returns maximum drawdown as percentage of peak equity.
        """
        if not trades:
            return 0.0

        equity = initial_balance
        peak = initial_balance
        max_dd = 0.0

        for trade in sorted(trades, key=lambda t: t.entry_date):
            equity += trade.pnl_usd
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak * 100.0
            max_dd = max(max_dd, dd)

        return max_dd

    def _check_consecutive_losses(self, trades: List[TradeRecord]) -> int:
        """
        Find maximum consecutive losing streak in trade sequence.
        """
        if not trades:
            return 0

        max_streak = 0
        current_streak = 0

        for trade in sorted(trades, key=lambda t: t.entry_date):
            if trade.pnl_r < 0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0

        return max_streak
