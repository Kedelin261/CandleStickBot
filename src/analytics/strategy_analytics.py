"""
M18 — Strategy Analytics Engine (Phase 1 MVP)
Tracks strategy performance independently for Pin Bar and Engulfing Bar.
The Candlestick Trading Bible: Track your results to improve your edge.

=== PHASE 1 STRATEGIES ===
  PIN_BAR        — tracked under "pin_bar"
  ENGULFING_BAR  — tracked under "engulfing_bar"

  NOT tracked: inside_bar, false_breakout (Phase 2)

=== METRICS (per strategy, per symbol/timeframe key) ===
  total_trades, win_count, loss_count, win_rate
  gross_profit_r, gross_loss_r, net_profit_r
  profit_factor, expectancy_r
  avg_winner_r, avg_loser_r
  max_drawdown_pct
  max_consecutive_wins, max_consecutive_losses
  sharpe_ratio (requires >= 5 trades)
  recent_30_profit_factor, recent_90_profit_factor

=== ENABLE / DISABLE ===
  Strategies can be disabled with a reason string.
  is_strategy_enabled() returns False if disabled.
  Disabled strategies still record trades but emit a warning.

=== DEGRADATION DETECTION ===
  is_strategy_degrading() fires if rolling-window PF < degradation_pf_threshold
  (default window=20 trades, threshold=0.80).

Status: Full Phase 1 implementation — Sprint 9.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from src.types import StrategyName, TradeTier

logger = logging.getLogger("candlestickbot.analytics.strategy_analytics")

# ---------------------------------------------------------------------------
# Phase 1 constants
# ---------------------------------------------------------------------------

_PHASE1_STRATEGIES = frozenset({
    StrategyName.PIN_BAR.value,       # "pin_bar"
    StrategyName.ENGULFING_BAR.value, # "engulfing_bar"
})

_MIN_SHARPE_TRADES = 5      # Minimum trades before Sharpe is meaningful
_MIN_DEGRADATION_TRADES = 5 # Minimum trades before degradation fires


# ---------------------------------------------------------------------------
# StrategyPerformanceRecord — one closed trade
# ---------------------------------------------------------------------------

@dataclass
class StrategyPerformanceRecord:
    """
    Record of a single completed trade for analytics purposes.

    This is the canonical input to the analytics engine.
    Created by M10 PaperTradeExecutor when a trade closes.
    """
    trade_id:         str
    strategy_name:    str          # "pin_bar" | "engulfing_bar"
    symbol:           str
    timeframe:        str
    direction:        str          # "LONG" | "SHORT"
    entry_timestamp:  datetime
    exit_timestamp:   datetime
    entry_price:      float
    exit_price:       float
    stop_loss:        float
    take_profit:      float
    pnl_pips:         float        # + = profit, - = loss
    pnl_usd:          float
    r_multiple:       float        # + = winner, - = loser (e.g. +2.0, -1.0)
    trade_quality_score: int = 0   # TQS total (0-100)
    trade_tier:       str = "STANDARD"  # TradeTier.value
    regime_at_entry:  str = "TRENDING"
    exit_reason:      str = "TP_HIT"    # TP_HIT | SL_HIT | MANUAL | EXPIRED

    @property
    def is_winner(self) -> bool:
        return self.r_multiple > 0

    @property
    def is_loser(self) -> bool:
        return self.r_multiple < 0

    @property
    def is_breakeven(self) -> bool:
        return self.r_multiple == 0.0


# ---------------------------------------------------------------------------
# StrategySummary — rolling aggregate for one strategy key
# ---------------------------------------------------------------------------

@dataclass
class StrategySummary:
    """
    Aggregate performance summary for one (strategy_name, symbol, timeframe) key.
    Recomputed by update_strategy_summary() after each new record.
    """
    strategy_name: str
    symbol:        str
    timeframe:     str

    # Counts
    total_trades:          int   = 0
    win_count:             int   = 0
    loss_count:            int   = 0
    breakeven_count:       int   = 0

    # R-multiples
    gross_profit_r:        float = 0.0
    gross_loss_r:          float = 0.0   # stored as positive magnitude
    net_profit_r:          float = 0.0

    # Derived
    win_rate:              float = 0.0
    profit_factor:         float = 0.0
    expectancy_r:          float = 0.0
    avg_winner_r:          float = 0.0
    avg_loser_r:           float = 0.0   # stored as positive magnitude
    max_drawdown_pct:      float = 0.0
    max_consecutive_wins:  int   = 0
    max_consecutive_losses: int  = 0
    sharpe_ratio:          Optional[float] = None

    # Rolling windows (recent N trades)
    recent_30_profit_factor: Optional[float] = None
    recent_90_profit_factor: Optional[float] = None

    # Enable / disable
    is_enabled:    bool          = True
    disable_reason: Optional[str] = None
    last_updated:  Optional[datetime] = None


# ---------------------------------------------------------------------------
# StrategyScorecard — human-readable summary card
# ---------------------------------------------------------------------------

@dataclass
class StrategyScorecard:
    """
    Formatted scorecard derived from a StrategySummary.
    Produced by get_strategy_scorecard().
    """
    strategy_name:    str
    symbol:           str
    timeframe:        str
    generated_at:     datetime

    total_trades:     int   = 0
    win_rate_pct:     float = 0.0    # as percentage 0-100
    profit_factor:    float = 0.0
    expectancy_r:     float = 0.0
    avg_winner_r:     float = 0.0
    avg_loser_r:      float = 0.0
    max_drawdown_pct: float = 0.0
    max_con_wins:     int   = 0
    max_con_losses:   int   = 0
    sharpe_ratio:     Optional[float] = None
    recent_30_pf:     Optional[float] = None
    recent_90_pf:     Optional[float] = None
    is_enabled:       bool  = True
    is_degrading:     bool  = False

    def __str__(self) -> str:
        lines = [
            f"=== {self.strategy_name.upper()} | {self.symbol}/{self.timeframe} ===",
            f"  Trades:      {self.total_trades}",
            f"  Win Rate:    {self.win_rate_pct:.1f}%",
            f"  Prof Factor: {self.profit_factor:.2f}",
            f"  Expectancy:  {self.expectancy_r:+.2f}R",
            f"  Avg Win:     {self.avg_winner_r:.2f}R",
            f"  Avg Loss:   -{self.avg_loser_r:.2f}R",
            f"  Max DD:     -{self.max_drawdown_pct:.1f}%",
            f"  MaxConWin:   {self.max_con_wins}",
            f"  MaxConLoss:  {self.max_con_losses}",
            f"  Sharpe:      {self.sharpe_ratio:.2f}" if self.sharpe_ratio is not None else "  Sharpe:      n/a",
            f"  30T PF:      {self.recent_30_pf:.2f}" if self.recent_30_pf is not None else "  30T PF:      n/a",
            f"  90T PF:      {self.recent_90_pf:.2f}" if self.recent_90_pf is not None else "  90T PF:      n/a",
            f"  Enabled:     {'YES' if self.is_enabled else 'NO'}",
            f"  Degrading:   {'YES ⚠' if self.is_degrading else 'NO'}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# StrategyAnalyticsEngine
# ---------------------------------------------------------------------------

class StrategyAnalyticsEngine:
    """
    M18 — Strategy Analytics Engine (Phase 1 MVP).

    Tracks Pin Bar and Engulfing Bar performance separately.
    Stateless between sessions if no db_session provided —
    all data lives in self._records and self._summaries.

    Usage::

        engine = StrategyAnalyticsEngine()
        engine.record_trade(record)
        summary = engine.get_strategy_summary("pin_bar", "EURUSD", "H1")
        scorecard = engine.get_strategy_scorecard("pin_bar", "EURUSD", "H1")
    """

    def __init__(
        self,
        degradation_window:       int   = 20,
        degradation_pf_threshold: float = 0.80,
        min_sharpe_trades:        int   = _MIN_SHARPE_TRADES,
        db_session=None,
    ):
        self.degradation_window       = degradation_window
        self.degradation_pf_threshold = degradation_pf_threshold
        self.min_sharpe_trades        = min_sharpe_trades
        self.db_session               = db_session

        # Internal storage keyed by (strategy_name, symbol, timeframe)
        self._records:   Dict[tuple, List[StrategyPerformanceRecord]] = defaultdict(list)
        self._summaries: Dict[tuple, StrategySummary]                  = {}
        self._disabled:  Dict[str, Optional[str]] = {}  # strategy_name → reason | None

    # ------------------------------------------------------------------
    # Key helper
    # ------------------------------------------------------------------

    @staticmethod
    def _key(strategy_name: str, symbol: str, timeframe: str) -> tuple:
        return (strategy_name.lower(), symbol.upper(), timeframe.upper())

    # ------------------------------------------------------------------
    # PRIMARY API — record_trade
    # ------------------------------------------------------------------

    def record_trade(self, record: StrategyPerformanceRecord) -> None:
        """
        Record a completed trade outcome and refresh the strategy summary.

        Validates that the strategy is a Phase 1 strategy.
        Disabled strategies still get recorded (with a warning).

        Args:
            record: StrategyPerformanceRecord from M10 PaperTradeExecutor.
        """
        name = record.strategy_name.lower()

        if name not in _PHASE1_STRATEGIES:
            logger.warning(
                "M18: strategy '%s' is not a Phase 1 strategy — trade %s skipped",
                name, record.trade_id,
            )
            return

        if not self.is_strategy_enabled(name):
            logger.warning(
                "M18: strategy '%s' is disabled — recording trade %s anyway (audit)",
                name, record.trade_id,
            )

        key = self._key(name, record.symbol, record.timeframe)
        self._records[key].append(record)
        self.update_strategy_summary(name, record.symbol, record.timeframe)

        logger.debug(
            "M18 recorded: %s %s %s r=%.2f total=%d",
            name, record.symbol, record.timeframe,
            record.r_multiple, len(self._records[key]),
        )

    # ------------------------------------------------------------------
    # update_strategy_summary
    # ------------------------------------------------------------------

    def update_strategy_summary(
        self,
        strategy_name: str,
        symbol:        str,
        timeframe:     str,
    ) -> StrategySummary:
        """
        Recompute and cache the StrategySummary for a given key.

        Called automatically by record_trade(); can also be called manually.
        """
        key = self._key(strategy_name, symbol, timeframe)
        records = self._records.get(key, [])

        summary = self._summaries.get(key) or StrategySummary(
            strategy_name=strategy_name.lower(),
            symbol=symbol.upper(),
            timeframe=timeframe.upper(),
        )

        # Recompute everything from scratch (deterministic)
        winners = [r for r in records if r.is_winner]
        losers  = [r for r in records if r.is_loser]

        summary.total_trades     = len(records)
        summary.win_count        = len(winners)
        summary.loss_count       = len(losers)
        summary.breakeven_count  = len(records) - len(winners) - len(losers)

        summary.gross_profit_r   = sum(r.r_multiple for r in winners)
        summary.gross_loss_r     = abs(sum(r.r_multiple for r in losers))
        summary.net_profit_r     = summary.gross_profit_r - summary.gross_loss_r

        summary.win_rate = (
            summary.win_count / summary.total_trades
            if summary.total_trades > 0 else 0.0
        )

        summary.profit_factor = (
            summary.gross_profit_r / summary.gross_loss_r
            if summary.gross_loss_r > 0
            else (float("inf") if summary.gross_profit_r > 0 else 0.0)
        )

        summary.expectancy_r = (
            summary.net_profit_r / summary.total_trades
            if summary.total_trades > 0 else 0.0
        )

        summary.avg_winner_r = (
            summary.gross_profit_r / summary.win_count
            if summary.win_count > 0 else 0.0
        )
        summary.avg_loser_r = (
            summary.gross_loss_r / summary.loss_count
            if summary.loss_count > 0 else 0.0
        )

        summary.max_drawdown_pct     = self._compute_max_drawdown(records)
        summary.max_consecutive_wins = self._max_streak(records, winning=True)
        summary.max_consecutive_losses = self._max_streak(records, winning=False)
        summary.sharpe_ratio         = self._compute_sharpe(records)

        summary.recent_30_profit_factor = self._rolling_pf(records, 30)
        summary.recent_90_profit_factor = self._rolling_pf(records, 90)

        # Preserve enable/disable state
        summary.is_enabled    = self.is_strategy_enabled(strategy_name)
        summary.disable_reason = self._disabled.get(strategy_name.lower())
        summary.last_updated  = datetime.utcnow()

        self._summaries[key] = summary
        return summary

    # ------------------------------------------------------------------
    # Getters
    # ------------------------------------------------------------------

    def get_strategy_summary(
        self,
        strategy_name: str,
        symbol:        str,
        timeframe:     str,
    ) -> StrategySummary:
        """
        Return the current StrategySummary for a key.
        Creates an empty one if no trades have been recorded.
        """
        key = self._key(strategy_name, symbol, timeframe)
        if key not in self._summaries:
            name = strategy_name.lower()
            enabled = name not in self._disabled
            reason  = self._disabled.get(name)
            return StrategySummary(
                strategy_name=name,
                symbol=symbol.upper(),
                timeframe=timeframe.upper(),
                is_enabled=enabled,
                disable_reason=reason,
            )
        return self._summaries[key]

    def get_strategy_scorecard(
        self,
        strategy_name: str,
        symbol:        str,
        timeframe:     str,
    ) -> StrategyScorecard:
        """
        Return a formatted StrategyScorecard derived from the summary.
        """
        s = self.get_strategy_summary(strategy_name, symbol, timeframe)
        return StrategyScorecard(
            strategy_name=s.strategy_name,
            symbol=s.symbol,
            timeframe=s.timeframe,
            generated_at=datetime.utcnow(),
            total_trades=s.total_trades,
            win_rate_pct=round(s.win_rate * 100, 2),
            profit_factor=s.profit_factor,
            expectancy_r=s.expectancy_r,
            avg_winner_r=s.avg_winner_r,
            avg_loser_r=s.avg_loser_r,
            max_drawdown_pct=s.max_drawdown_pct,
            max_con_wins=s.max_consecutive_wins,
            max_con_losses=s.max_consecutive_losses,
            sharpe_ratio=s.sharpe_ratio,
            recent_30_pf=s.recent_30_profit_factor,
            recent_90_pf=s.recent_90_profit_factor,
            is_enabled=s.is_enabled,
            is_degrading=self.is_strategy_degrading(strategy_name, symbol, timeframe),
        )

    def get_all_records(
        self,
        strategy_name: str,
        symbol:        str,
        timeframe:     str,
    ) -> List[StrategyPerformanceRecord]:
        """Return all records for a given key (oldest first)."""
        key = self._key(strategy_name, symbol, timeframe)
        return list(self._records.get(key, []))

    # ------------------------------------------------------------------
    # Enable / Disable
    # ------------------------------------------------------------------

    def enable_strategy(self, strategy_name: str) -> None:
        """
        Re-enable a previously disabled strategy.
        No-op if already enabled.
        """
        name = strategy_name.lower()
        if name in self._disabled:
            del self._disabled[name]
        # Refresh all summaries for this strategy
        for key, summary in self._summaries.items():
            if key[0] == name:
                summary.is_enabled    = True
                summary.disable_reason = None
        logger.info("M18: strategy '%s' enabled", name)

    def disable_strategy(self, strategy_name: str, reason: str = "") -> None:
        """
        Disable a strategy with an optional reason.
        Disabled strategies still record trades but signal callers not to trade.
        """
        name = strategy_name.lower()
        self._disabled[name] = reason or None
        # Refresh all summaries for this strategy
        for key, summary in self._summaries.items():
            if key[0] == name:
                summary.is_enabled    = False
                summary.disable_reason = reason or None
        logger.info("M18: strategy '%s' disabled — %s", name, reason)

    def is_strategy_enabled(self, strategy_name: str) -> bool:
        """Return True if strategy is enabled (not explicitly disabled)."""
        return strategy_name.lower() not in self._disabled

    # ------------------------------------------------------------------
    # Degradation detection
    # ------------------------------------------------------------------

    def is_strategy_degrading(
        self,
        strategy_name: str,
        symbol:        str,
        timeframe:     str,
    ) -> bool:
        """
        Return True if the rolling-window profit factor has dropped below
        degradation_pf_threshold.

        Requires at least _MIN_DEGRADATION_TRADES in the window.
        """
        key = self._key(strategy_name, symbol, timeframe)
        records = self._records.get(key, [])
        window = records[-self.degradation_window:]
        if len(window) < _MIN_DEGRADATION_TRADES:
            return False
        pf = _profit_factor_of(window)
        return pf < self.degradation_pf_threshold

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------

    def rank_strategies_by_metric(
        self,
        metric: str,
        symbol:     str = "EURUSD",
        timeframe:  str = "H1",
        descending: bool = True,
    ) -> List[Tuple[str, float]]:
        """
        Rank all tracked Phase 1 strategies by a named metric.

        Supported metrics: "profit_factor", "win_rate", "expectancy_r",
        "total_trades", "net_profit_r", "sharpe_ratio", "max_drawdown_pct".

        Returns list of (strategy_name, metric_value) sorted by value.
        """
        results = []
        for strat in _PHASE1_STRATEGIES:
            s = self.get_strategy_summary(strat, symbol, timeframe)
            val = getattr(s, metric, None)
            if val is None:
                val = 0.0
            results.append((strat, float(val)))

        results.sort(key=lambda x: x[1], reverse=descending)
        return results

    # ------------------------------------------------------------------
    # Comparison table
    # ------------------------------------------------------------------

    def get_strategy_comparison_table(
        self,
        symbol:    str = "EURUSD",
        timeframe: str = "H1",
    ) -> Dict[str, Dict]:
        """
        Return a dict of {strategy_name: {metric: value}} for both Phase 1 strategies.
        Useful for display / reporting.
        """
        table = {}
        for strat in sorted(_PHASE1_STRATEGIES):
            s = self.get_strategy_summary(strat, symbol, timeframe)
            table[strat] = {
                "total_trades":     s.total_trades,
                "win_rate":         round(s.win_rate, 4),
                "profit_factor":    s.profit_factor,
                "expectancy_r":     round(s.expectancy_r, 4),
                "avg_winner_r":     round(s.avg_winner_r, 4),
                "avg_loser_r":      round(s.avg_loser_r, 4),
                "max_drawdown_pct": round(s.max_drawdown_pct, 4),
                "sharpe_ratio":     s.sharpe_ratio,
                "recent_30_pf":     s.recent_30_profit_factor,
                "recent_90_pf":     s.recent_90_profit_factor,
                "is_enabled":       s.is_enabled,
            }
        return table

    # ------------------------------------------------------------------
    # Private computation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_max_drawdown(
        records: List[StrategyPerformanceRecord],
    ) -> float:
        """
        Compute maximum peak-to-trough drawdown as a percentage of peak equity.

        Equity curve is built from cumulative R-multiples (starting at 0).
        Returns the maximum percentage drop from any peak to any subsequent trough.
        """
        if not records:
            return 0.0

        equity = 0.0
        peak   = 0.0
        max_dd = 0.0

        for r in records:
            equity += r.r_multiple
            if equity > peak:
                peak = equity
            if peak > 0:
                dd = (peak - equity) / peak * 100.0
                if dd > max_dd:
                    max_dd = dd

        return round(max_dd, 4)

    @staticmethod
    def _max_streak(
        records: List[StrategyPerformanceRecord],
        winning: bool,
    ) -> int:
        """Compute max consecutive wins (winning=True) or losses (winning=False)."""
        if not records:
            return 0
        max_streak = 0
        cur_streak = 0
        for r in records:
            if (winning and r.is_winner) or (not winning and r.is_loser):
                cur_streak += 1
                max_streak = max(max_streak, cur_streak)
            else:
                cur_streak = 0
        return max_streak

    def _compute_sharpe(
        self,
        records: List[StrategyPerformanceRecord],
    ) -> Optional[float]:
        """
        Compute annualised Sharpe ratio from R-multiple series.

        Uses: Sharpe = mean(R) / std(R)  (simplified, no risk-free rate).
        Returns None if fewer than min_sharpe_trades records or std == 0.
        """
        if len(records) < self.min_sharpe_trades:
            return None

        r_series = [r.r_multiple for r in records]
        n    = len(r_series)
        mean = sum(r_series) / n
        variance = sum((x - mean) ** 2 for x in r_series) / (n - 1) if n > 1 else 0.0
        std  = math.sqrt(variance)

        if std == 0:
            return None

        return round(mean / std, 4)

    @staticmethod
    def _rolling_pf(
        records: List[StrategyPerformanceRecord],
        window:  int,
    ) -> Optional[float]:
        """
        Compute profit factor over the last ``window`` trades.
        Returns None if fewer than 1 record in window.
        """
        recent = records[-window:]
        if not recent:
            return None
        return _profit_factor_of(recent)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _profit_factor_of(records: List[StrategyPerformanceRecord]) -> float:
    """Compute profit factor for a record list. Returns 0.0 if no losers and no winners."""
    gross_profit = sum(r.r_multiple for r in records if r.is_winner)
    gross_loss   = abs(sum(r.r_multiple for r in records if r.is_loser))
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def make_record(
    strategy_name:  str,
    r_multiple:     float,
    trade_id:       str = "",
    symbol:         str = "EURUSD",
    timeframe:      str = "H1",
    direction:      str = "LONG",
    entry_price:    float = 1.1000,
    exit_price:     float = 1.1040,
    stop_loss:      float = 1.0980,
    take_profit:    float = 1.1040,
    pnl_pips:       float = 0.0,
    pnl_usd:        float = 0.0,
    tqs:            int   = 70,
    tier:           str   = "STANDARD",
    regime:         str   = "TRENDING",
    exit_reason:    str   = "TP_HIT",
    timestamp:      Optional[datetime] = None,
) -> StrategyPerformanceRecord:
    """
    Convenience factory for creating StrategyPerformanceRecord in tests.
    """
    import uuid
    ts = timestamp or datetime.utcnow()
    return StrategyPerformanceRecord(
        trade_id=trade_id or str(uuid.uuid4()),
        strategy_name=strategy_name,
        symbol=symbol,
        timeframe=timeframe,
        direction=direction,
        entry_timestamp=ts,
        exit_timestamp=ts,
        entry_price=entry_price,
        exit_price=exit_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        pnl_pips=pnl_pips,
        pnl_usd=pnl_usd,
        r_multiple=r_multiple,
        trade_quality_score=tqs,
        trade_tier=tier,
        regime_at_entry=regime,
        exit_reason=exit_reason,
    )
