"""
Sprint 12 — Phase 1 Paper Trading Pipeline Runner
==================================================
Orchestrates all Phase 1 modules into a complete end-to-end paper trading
simulation.  No business logic is reimplemented here; every calculation
delegates to the appropriate engine.

Flow per candle (signal bar = last candle in the lookback window):
    M03 MarketStructureAnalyzer
    → M04 TrendDetector
    → M05 SREngine
    → M16 MarketRegimeEngine
    → M07 PatternEngine  (via StrategyEngine)
    → M08 StrategyEngine
    → M09 RiskEngine
    → M10 PaperTradeExecutor  (place order)
    → Simulate future candles until TP / SL / end-of-data
    → _finalize_closed_order()  ← single finalization gate (Sprint 12.1)
        → PipelineResult  (always)
        → M18 StrategyAnalyticsEngine  (always)
        → M19 TradeReviewEngine  (losses only)

Phase 1 scope:
    - EURUSD Daily only
    - Bullish / Bearish Pin Bar
    - Bullish / Bearish Engulfing
    - No Fibonacci, Inside Bar, False Breakout, Portfolio, Correlation
    - No MT5, broker, or live-execution code

Sprint 12.1 change:
    Unified finalization pathway.  Previously TP/SL closures mutated the
    PaperOrder directly inside _try_close_on_candle() and bypassed M18/M19.
    Now every closed trade — regardless of exit type — flows through the
    private _finalize_closed_order() method, which is the single place
    responsible for updating PipelineResult, M18, and M19.  This guarantees
    PipelineResult and M18 always agree on total trades, wins, losses, P&L.
"""

from __future__ import annotations

import logging
import textwrap
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from src.analysis.market_regime import MarketRegimeEngine
from src.analysis.market_structure import MarketStructureAnalyzer
from src.analysis.sr_engine import SREngine
from src.analysis.trend_detection import TrendDetector
from src.analytics.strategy_analytics import StrategyAnalyticsEngine
from src.analytics.trade_review import TradeContext, TradeReviewEngine
from src.data.types import CandleData
from src.execution.paper_executor import (
    ExitReason,
    PaperExecutorConfig,
    PaperOrder,
    PaperOrderStatus,
    PaperTradeExecutor,
)
from src.patterns.pattern_engine import PatternEngine
from src.risk.risk_engine import RiskCheckResult, RiskConfig, RiskEngine
from src.strategy.strategy_engine import StrategyConfig, StrategyEngine
from src.types import AccountState, LossCategory

logger = logging.getLogger("candlestickbot.integration.pipeline")

# Minimum lookback window required before any analysis runs.
_MIN_LOOKBACK: int = 30


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """
    Configuration for one complete pipeline run.

    symbol            : Instrument to trade (Phase 1: 'EURUSD')
    timeframe         : Chart timeframe    (Phase 1: 'D1')
    initial_balance   : Starting account balance in USD
    slippage_pips     : Fixed adverse slippage applied to every fill
    enable_pin_bar    : Whether Pin Bar signals are processed
    enable_engulfing  : Whether Engulfing Bar signals are processed
    risk_enabled      : Whether M09 position sizing / risk checks are active
    analytics_enabled : Whether M18 analytics updates are wired
    review_enabled    : Whether M19 trade review classification is wired
    max_candles       : Maximum candles to process (0 = unlimited)
    minimum_tqs       : Minimum TQS total score to place a trade
    minimum_rr        : Minimum risk-reward ratio to place a trade
    lookback_window   : Number of candles fed as context per evaluation
    """

    symbol:            str   = "EURUSD"
    timeframe:         str   = "D1"
    initial_balance:   float = 10_000.0
    slippage_pips:     float = 1.0
    enable_pin_bar:    bool  = True
    enable_engulfing:  bool  = True
    risk_enabled:      bool  = True
    analytics_enabled: bool  = True
    review_enabled:    bool  = True
    max_candles:       int   = 0       # 0 = process all
    minimum_tqs:       float = 0.0     # override StrategyConfig.min_tqs_score
    minimum_rr:        float = 2.0
    lookback_window:   int   = 50


# ---------------------------------------------------------------------------
# Strategy-level breakdown container
# ---------------------------------------------------------------------------

@dataclass
class StrategyBreakdown:
    """Per-strategy performance counts."""
    strategy_name: str
    trades:  int   = 0
    wins:    int   = 0
    losses:  int   = 0

    @property
    def profit_factor(self) -> float:
        if self.losses == 0:
            return float("inf") if self.wins > 0 else 0.0
        if self.wins == 0:
            return 0.0
        return self.wins / self.losses  # simplified: trade counts as proxy

    def to_dict(self) -> dict:
        return {
            "trades":        self.trades,
            "wins":          self.wins,
            "losses":        self.losses,
            "profit_factor": self.profit_factor,
        }


# ---------------------------------------------------------------------------
# Pipeline result
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """
    Complete output of a pipeline run.
    """
    # Run information
    symbol:           str
    timeframe:        str
    started_at:       datetime
    completed_at:     Optional[datetime] = None
    candles_processed: int  = 0
    trades_generated:  int  = 0   # strategy engine recommendations
    trades_approved:   int  = 0   # passed risk engine
    trades_rejected:   int  = 0   # blocked by risk engine
    trades_executed:   int  = 0   # placed in paper executor
    error_message:     Optional[str] = None

    # Performance
    initial_balance:   float = 0.0
    final_balance:     float = 0.0
    wins:              int   = 0
    losses:            int   = 0
    win_rate:          float = 0.0
    net_profit_usd:    float = 0.0
    gross_profit:      float = 0.0
    gross_loss:        float = 0.0
    expectancy:        float = 0.0   # in R-multiples
    profit_factor:     float = 0.0
    max_drawdown:      float = 0.0   # percent

    # Strategy breakdown
    pin_bar:     StrategyBreakdown = field(
        default_factory=lambda: StrategyBreakdown("pin_bar"))
    engulfing:   StrategyBreakdown = field(
        default_factory=lambda: StrategyBreakdown("engulfing_bar"))

    # Trade review summary (loss counts by category)
    bad_signal:          int = 0
    bad_regime:          int = 0
    bad_level:           int = 0
    bad_execution:       int = 0
    normal_statistical:  int = 0

    # Internal audit
    _closed_orders: List[PaperOrder] = field(default_factory=list, repr=False)
    _r_multiples:   List[float]      = field(default_factory=list, repr=False)
    _pnl_usd_series: List[float]    = field(default_factory=list, repr=False)  # Sprint 15 FIX-3

    def record_closed_order(self, order: PaperOrder) -> None:
        """Update result counters from a freshly-closed PaperOrder."""
        self._closed_orders.append(order)
        self.trades_executed += 1

        r = order.r_multiple or 0.0
        self._r_multiples.append(r)
        self._pnl_usd_series.append(order.pnl_usd or 0.0)  # Sprint 15 FIX-3

        strat = order.strategy_name.lower()
        breakdown = self.pin_bar if "pin" in strat else self.engulfing
        breakdown.trades += 1

        if order.is_winner:
            self.wins += 1
            breakdown.wins += 1
            self.gross_profit += r
        elif order.is_loser:
            self.losses += 1
            breakdown.losses += 1
            self.gross_loss += abs(r)

        # Update balance from pnl_usd
        self.final_balance += (order.pnl_usd or 0.0)

    def finalise(self) -> None:
        """Compute derived metrics after all trades are recorded."""
        total = self.wins + self.losses
        self.win_rate      = (self.wins / total * 100) if total > 0 else 0.0
        self.net_profit_usd = self.final_balance - self.initial_balance
        self.profit_factor  = (
            self.gross_profit / self.gross_loss
            if self.gross_loss > 0
            else (float("inf") if self.gross_profit > 0 else 0.0)
        )
        if self._r_multiples:
            self.expectancy = sum(self._r_multiples) / len(self._r_multiples)
        # Max drawdown: peak-to-trough on cumulative R equity curve
        self.max_drawdown = _compute_max_drawdown_equity(   # Sprint 15 FIX-3
            self._pnl_usd_series, self.initial_balance
        )
        self.completed_at = datetime.now(timezone.utc)

    def record_review_result(self, category: LossCategory) -> None:
        """Tally a loss classification from M19."""
        if category == LossCategory.BAD_SIGNAL:
            self.bad_signal += 1
        elif category == LossCategory.BAD_REGIME:
            self.bad_regime += 1
        elif category == LossCategory.BAD_LEVEL:
            self.bad_level += 1
        elif category == LossCategory.BAD_EXECUTION:
            self.bad_execution += 1
        elif category == LossCategory.NORMAL_STATISTICAL:
            self.normal_statistical += 1


def _compute_max_drawdown(r_multiples: List[float]) -> float:
    """
    LEGACY — R-multiple equity curve (peak seeded at 0.0).
    Kept for backward compatibility; PipelineResult now uses
    _compute_max_drawdown_equity() (Sprint 15 FIX-3).
    """
    if not r_multiples:
        return 0.0
    peak = 0.0
    equity = 0.0
    max_dd = 0.0
    for r in r_multiples:
        equity += r
        if equity > peak:
            peak = equity
        dd = (peak - equity) / max(abs(peak), 1e-9) * 100
        if dd > max_dd:
            max_dd = dd
    return max_dd


def _compute_max_drawdown_equity(
    pnl_usd_series: List[float],
    initial_balance: float,
) -> float:
    """
    Sprint 15 FIX-3 — Account-equity max drawdown (peak seeded at initial_balance).

    equity[i] = initial_balance + sum(pnl_usd_series[:i+1])
    peak       = running maximum of equity
    dd%        = (peak - equity) / peak * 100

    Guarantees:
      - Empty series → 0.0%
      - First-trade loss → small %, not astronomical
      - Peak is always ≥ initial_balance at start, so denominator is safe
    """
    if not pnl_usd_series:
        return 0.0
    peak    = initial_balance
    equity  = initial_balance
    max_dd  = 0.0
    for pnl in pnl_usd_series:
        equity += pnl
        if equity > peak:
            peak = equity
        if peak > 0:
            dd = (peak - equity) / peak * 100.0
            if dd > max_dd:
                max_dd = dd
    return max_dd


# ---------------------------------------------------------------------------
# Pipeline Runner
# ---------------------------------------------------------------------------

class PipelineRunner:
    """
    Phase 1 Paper Trading Pipeline.

    Usage::

        runner = PipelineRunner(config)
        result = runner.run(candles)
        print(runner.generate_run_report(result))

    Sprint 12.1: All trade closes — TP hit, SL hit, end-of-data manual close —
    flow through _finalize_closed_order(), ensuring PipelineResult and M18
    always agree on every metric.
    """

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self._cfg = config or PipelineConfig()
        self._setup_engines()

    # ------------------------------------------------------------------
    # Engine setup
    # ------------------------------------------------------------------

    def _setup_engines(self) -> None:
        cfg = self._cfg

        # M03 Market Structure
        self._structure = MarketStructureAnalyzer(
            lookback=5,
            pip_size=0.0001,
        )

        # M04 Trend Detection
        self._trend = TrendDetector(sma_period=21)

        # M05 Support/Resistance
        self._sr = SREngine(pip_size=0.0001)

        # M16 Market Regime
        self._regime = MarketRegimeEngine()

        # M07 Pattern Engine (embedded inside StrategyEngine but also standalone)
        self._pattern = PatternEngine(pip_size=0.0001)

        # M08 Strategy Engine — wired with all analysis engines
        strategy_config = StrategyConfig(
            min_tqs_score=max(cfg.minimum_tqs, 0.0),
            min_rr_ratio=cfg.minimum_rr,
        )
        self._strategy = StrategyEngine(
            config=strategy_config,
            trend_detector=self._trend,
            sr_engine=self._sr,
            regime_engine=self._regime,
            pattern_engine=self._pattern,
        )

        # M09 Risk Engine
        risk_config = RiskConfig(min_rr_ratio=cfg.minimum_rr)
        self._risk = RiskEngine(config=risk_config)

        # M18 Analytics
        self._analytics = StrategyAnalyticsEngine()

        # M19 Trade Review
        self._review = TradeReviewEngine()

        # M10 Paper Executor — wired to M18 + M19
        # NOTE: The executor's own M18/M19 push (inside close_order) is NOT
        # used for TP/SL simulation closes, because the pipeline drives those
        # directly via _finalize_closed_order().  The executor's push is only
        # invoked for end-of-data closes via executor.close_order(), but we
        # redirect that path too (see run()) to avoid double-recording.
        exec_config = PaperExecutorConfig(
            default_slippage_pips=cfg.slippage_pips,
            pip_size=0.0001,
        )
        self._executor = PaperTradeExecutor(
            analytics_engine=self._analytics,
            review_engine=self._review,
            config=exec_config,
        )

        logger.debug("M12: pipeline engines initialised for %s %s",
                     cfg.symbol, cfg.timeframe)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, candles: List[CandleData]) -> PipelineResult:
        """
        Process *candles* through the complete Phase 1 pipeline.

        Parameters
        ----------
        candles : list of CandleData, oldest first.

        Returns
        -------
        PipelineResult populated with performance metrics.

        Sprint 12.1: every closed trade flows through _finalize_closed_order().
        _try_close_on_candle() is now a pure detection helper that returns
        (hit: bool, exit_price: float, exit_reason: str) without mutating
        the order or touching analytics.
        """
        cfg = self._cfg
        result = PipelineResult(
            symbol=cfg.symbol,
            timeframe=cfg.timeframe,
            started_at=datetime.now(timezone.utc),
            initial_balance=cfg.initial_balance,
            final_balance=cfg.initial_balance,
        )

        if not candles:
            result.error_message = "No candles provided"
            result.finalise()
            return result

        # Limit candle count if configured
        if cfg.max_candles > 0:
            candles = candles[: cfg.max_candles]

        # Need at least _MIN_LOOKBACK candles to compute any analysis
        if len(candles) < _MIN_LOOKBACK:
            result.error_message = (
                f"Insufficient candles: need ≥{_MIN_LOOKBACK}, got {len(candles)}"
            )
            result.candles_processed = len(candles)
            result.finalise()
            return result

        # Build initial account state
        account = self._make_account(cfg.initial_balance)

        # Slide a lookback window over the candle series.
        # The last candle in each window is the "signal bar".
        window = cfg.lookback_window
        open_order: Optional[PaperOrder] = None   # at most one open trade

        for i in range(window, len(candles) + 1):
            context_candles = candles[i - window: i]
            signal_candle   = context_candles[-1]
            result.candles_processed = i

            # ---- if we have an open order, check for TP/SL hit ----
            if open_order is not None:
                hit, exit_price, exit_reason = self._try_close_on_candle(
                    open_order, signal_candle
                )
                if hit:
                    account = self._finalize_closed_order(
                        open_order, exit_price, exit_reason, result, account
                    )
                    open_order = None
                    continue   # skip signal evaluation on close candle

            # ---- no open trade: evaluate for signal ----
            if open_order is not None:
                continue   # guard; should never reach here

            # Step 1-4: pre-compute analysis (feed to strategy engine)
            try:
                structure_a = self._structure.analyze(context_candles)
                trend_a     = self._trend.analyze(
                    context_candles,
                    market_structure=structure_a.to_market_structure(),
                )
                sr_a        = self._sr.analyze(       # Sprint 15 FIX-1/FIX-2
                    context_candles,
                    swing_highs=structure_a.swing_highs,   # RC-1: wire M03→M05
                    swing_lows=structure_a.swing_lows,     # RC-1: wire M03→M05
                    sma21=trend_a.sma21,                   # RC-2: wire M04→M05
                )
                regime_a    = self._regime.analyze(
                    context_candles, adx=trend_a.adx
                )
            except Exception as exc:
                logger.debug("M12: analysis error at candle %d: %s", i, exc)
                continue

            # Step 5-6: Strategy Engine
            try:
                rec_result = self._strategy.evaluate_candle(
                    context_candles,
                    trend=trend_a,
                    sr=sr_a,
                    regime=regime_a,
                )
            except Exception as exc:
                logger.debug("M12: strategy error at candle %d: %s", i, exc)
                continue

            if not rec_result.is_recommended:
                continue   # no signal

            recommendation = rec_result.recommendation

            # Filter by enabled strategies BEFORE incrementing counter (Sprint 15 FIX-5 / RC-7)
            strat_val = recommendation.strategy.value.lower()
            if "pin" in strat_val and not cfg.enable_pin_bar:
                continue
            if "engulf" in strat_val and not cfg.enable_engulfing:
                continue

            result.trades_generated += 1  # RC-7: moved after strategy-enable filter

            # Step 7: Risk Engine
            if not cfg.risk_enabled:
                # Bypass risk; create a synthetic approval
                from src.types import RiskApprovedOrder
                approved = RiskApprovedOrder(
                    recommendation=recommendation,
                    lot_size=0.10,
                    risk_pct=1.0,
                    risk_amount_usd=cfg.initial_balance * 0.01,
                    account_balance=account.balance,
                    stop_pips=self._risk.compute_stop_pips(
                        recommendation.entry_price, recommendation.stop_price
                    ),
                )
                result.trades_approved += 1
            else:
                check, approved, rejection = self._risk.check_and_approve(
                    recommendation, account
                )
                if check != RiskCheckResult.APPROVED or approved is None:
                    result.trades_rejected += 1
                    logger.debug("M12: risk rejected — %s",
                                 rejection.reason if rejection else "unknown")
                    continue
                result.trades_approved += 1

            # Step 8: Place paper order
            try:
                open_order = self._executor.place_paper_order(approved)
                self._risk.update_open_trade_count(1)
                logger.debug(
                    "M12: order placed %s %s @ %.5f SL=%.5f TP=%.5f",
                    open_order.strategy_name,
                    open_order.direction,
                    open_order.filled_price,
                    open_order.stop_loss,
                    open_order.take_profit,
                )
            except Exception as exc:
                logger.warning("M12: executor error: %s", exc)
                result.trades_rejected += 1
                continue

        # End of candle loop — close any open order at final candle price
        if open_order is not None and open_order.is_open:
            last_price = candles[-1].close
            try:
                account = self._finalize_closed_order(
                    open_order,
                    exit_price=last_price,
                    exit_reason=ExitReason.MANUAL_CLOSE,
                    result=result,
                    account=account,
                )
            except Exception as exc:
                logger.warning("M12: end-of-data close error: %s", exc)

        result.finalise()
        return result

    def reset(self) -> None:
        """
        Reset all engine state and executor session.
        Allows the same PipelineRunner to process a new candle batch.
        """
        self._executor.reset_session()
        self._analytics = StrategyAnalyticsEngine()
        self._review    = TradeReviewEngine()
        cfg = self._cfg
        exec_config = PaperExecutorConfig(
            default_slippage_pips=cfg.slippage_pips,
            pip_size=0.0001,
        )
        self._executor = PaperTradeExecutor(
            analytics_engine=self._analytics,
            review_engine=self._review,
            config=exec_config,
        )
        logger.debug("M12: pipeline reset")

    def get_analytics_engine(self) -> StrategyAnalyticsEngine:
        """Return M18 engine (read access for tests and reporting)."""
        return self._analytics

    def get_review_engine(self) -> TradeReviewEngine:
        """Return M19 engine (read access for tests and reporting)."""
        return self._review

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_run_report(self, result: PipelineResult) -> str:
        """
        Generate a human-readable run report from a PipelineResult.

        Returns a multi-section text string.
        """
        sep  = "=" * 60
        sep2 = "-" * 60
        total_trades = result.wins + result.losses

        lines = [
            sep,
            "  CANDLESTICKBOT — PHASE 1 PAPER TRADING RUN REPORT",
            sep,
            "",
            "── EXECUTIVE SUMMARY ──────────────────────────────────",
            f"  Symbol       : {result.symbol}",
            f"  Timeframe    : {result.timeframe}",
            f"  Run started  : {_fmt_dt(result.started_at)}",
            f"  Run completed: {_fmt_dt(result.completed_at)}",
            f"  Candles proc : {result.candles_processed}",
            "",
            "── TRADE STATISTICS ────────────────────────────────────",
            f"  Generated    : {result.trades_generated}",
            f"  Approved     : {result.trades_approved}",
            f"  Rejected     : {result.trades_rejected}",
            f"  Executed     : {result.trades_executed}",
            f"  Wins         : {result.wins}",
            f"  Losses       : {result.losses}",
            f"  Win Rate     : {result.win_rate:.1f}%",
            "",
            "── PERFORMANCE METRICS ─────────────────────────────────",
            f"  Initial Bal  : ${result.initial_balance:,.2f}",
            f"  Final Bal    : ${result.final_balance:,.2f}",
            f"  Net P&L      : ${result.net_profit_usd:+,.2f}",
            f"  Gross Profit : {result.gross_profit:.2f}R",
            f"  Gross Loss   : {result.gross_loss:.2f}R",
            f"  Profit Factor: {_fmt_pf(result.profit_factor)}",
            f"  Expectancy   : {result.expectancy:+.3f}R",
            f"  Max Drawdown : {result.max_drawdown:.2f}%",
            "",
            "── STRATEGY BREAKDOWN ──────────────────────────────────",
        ]

        for bd in (result.pin_bar, result.engulfing):
            lines += [
                f"  {bd.strategy_name.upper()}",
                f"    Trades   : {bd.trades}",
                f"    Wins     : {bd.wins}",
                f"    Losses   : {bd.losses}",
                f"    PF       : {_fmt_pf(bd.profit_factor)}",
            ]

        lines += [
            "",
            "── ANALYTICS SUMMARY ───────────────────────────────────",
        ]
        for strat in ("pin_bar", "engulfing_bar"):
            try:
                s = self._analytics.get_strategy_summary(
                    strat, result.symbol, result.timeframe
                )
                lines.append(
                    f"  {strat}: trades={s.total_trades} "
                    f"WR={s.win_rate*100:.1f}% "
                    f"PF={_fmt_pf(s.profit_factor)} "
                    f"exp={s.expectancy_r:+.3f}R "
                    f"DD={s.max_drawdown_pct:.1f}%"
                )
            except Exception:
                pass

        lines += [
            "",
            "── FAILURE ANALYSIS ────────────────────────────────────",
            f"  BAD_SIGNAL          : {result.bad_signal}",
            f"  BAD_REGIME          : {result.bad_regime}",
            f"  BAD_LEVEL           : {result.bad_level}",
            f"  BAD_EXECUTION       : {result.bad_execution}",
            f"  NORMAL_STATISTICAL  : {result.normal_statistical}",
            "",
            "── RISK SUMMARY ────────────────────────────────────────",
            f"  Kill switch active : {self._risk.kill_switch_active}",
            f"  Consec. losses     : {self._risk._state.consecutive_losses}",
            "",
            sep,
        ]

        if result.error_message:
            lines.insert(3, f"  *** ERROR: {result.error_message} ***")
            lines.insert(4, "")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Sprint 12.1 — Unified finalization gate
    # ------------------------------------------------------------------

    def _finalize_closed_order(
        self,
        order:       PaperOrder,
        exit_price:  float,
        exit_reason: str,
        result:      PipelineResult,
        account:     AccountState,
    ) -> AccountState:
        """
        The single finalization gate for every closed trade.

        Responsibilities (in order):
          1. Stamp the order as CLOSED with exit fields and computed metrics.
          2. Push to M18 StrategyAnalyticsEngine (always).
          3. Push to M19 TradeReviewEngine (losses only).
          4. Tally M19 category in PipelineResult (losses only).
          5. Record the order in PipelineResult.
          6. Update the account state.
          7. Notify RiskEngine of the close.

        This method is the ONLY place where steps 1-7 happen, ensuring
        PipelineResult and M18 always agree on every metric.

        Parameters
        ----------
        order       : The PaperOrder to finalize (must be is_open == True).
        exit_price  : Price at which the trade exits.
        exit_reason : One of ExitReason constants.
        result      : PipelineResult accumulator to update.
        account     : Current AccountState (read-only; a new one is returned).

        Returns
        -------
        Updated AccountState after incorporating this trade's P&L.
        """
        # --- 1. Stamp order fields ---
        order.exit_price  = exit_price
        order.exit_reason = exit_reason
        order.closed_at   = datetime.now(timezone.utc)
        order.status      = PaperOrderStatus.CLOSED

        r, pnl = _calc_r_pnl(order, exit_price)
        order.r_multiple = r
        order.pnl_usd    = pnl

        logger.debug(
            "M12: finalize %s @ %.5f — R=%.2f PnL=$%.2f (%s)",
            order.order_id, exit_price, r, pnl, exit_reason,
        )

        # --- 2. Push to M18 (always) ---
        try:
            self._executor._push_to_analytics(order)
        except Exception as exc:
            logger.warning(
                "M12: M18 push failed for %s: %s", order.order_id, exc
            )

        # --- 3. Push to M19 (losses only) ---
        if order.is_loser:
            try:
                self._executor._push_to_review(order)
            except Exception as exc:
                logger.warning(
                    "M12: M19 push failed for %s: %s", order.order_id, exc
                )

            # --- 4. Tally M19 category in PipelineResult ---
            review_results = self._review.get_all_results()
            if review_results:
                result.record_review_result(review_results[-1].category)

        # --- 5. Record in PipelineResult ---
        result.record_closed_order(order)

        # --- 6. Update account ---
        new_account = self._update_account(account, order)

        # --- 7. Notify RiskEngine ---
        self._risk.update_after_trade_close(order.r_multiple or 0.0, new_account)
        self._risk.update_open_trade_count(0)

        return new_account

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_account(balance: float) -> AccountState:
        """Create a fresh AccountState at the given balance."""
        return AccountState(
            balance=balance,
            equity=balance,
            margin=0.0,
            free_margin=balance,
            open_pnl=0.0,
            peak_equity=balance,
            day_open_balance=balance,
            week_open_balance=balance,
            open_trades=0,
        )

    @staticmethod
    def _update_account(account: AccountState, order: PaperOrder) -> AccountState:
        """Return a new AccountState after a trade closes."""
        pnl   = order.pnl_usd or 0.0
        new_bal = account.balance + pnl
        new_eq  = new_bal
        return AccountState(
            balance=new_bal,
            equity=new_eq,
            margin=0.0,
            free_margin=new_bal,
            open_pnl=0.0,
            peak_equity=max(account.peak_equity, new_eq),
            day_open_balance=account.day_open_balance,
            week_open_balance=account.week_open_balance,
            open_trades=0,
        )

    @staticmethod
    def _try_close_on_candle(
        order:  PaperOrder,
        candle: CandleData,
    ) -> Tuple[bool, float, str]:
        """
        Check whether *candle* hits the order's TP or SL.

        Sprint 12.1: this method is now a PURE DETECTION HELPER.
        It does NOT mutate the order and does NOT touch analytics.
        All mutation happens in _finalize_closed_order().

        For LONG:  TP hit if candle.high  >= take_profit
                   SL hit if candle.low   <= stop_loss
        For SHORT: TP hit if candle.low   <= take_profit
                   SL hit if candle.high  >= stop_loss

        Returns
        -------
        (hit: bool, exit_price: float, exit_reason: str)
        hit=False returns (False, 0.0, "").
        TP is checked before SL (favourable outcome first).
        """
        if order.direction == "LONG":
            if candle.high >= order.take_profit:
                return True, order.take_profit, ExitReason.TP_HIT
            if candle.low <= order.stop_loss:
                return True, order.stop_loss, ExitReason.SL_HIT
        else:  # SHORT
            if candle.low <= order.take_profit:
                return True, order.take_profit, ExitReason.TP_HIT
            if candle.high >= order.stop_loss:
                return True, order.stop_loss, ExitReason.SL_HIT
        return False, 0.0, ""


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _calc_r_pnl(order: PaperOrder, exit_price: float) -> Tuple[float, float]:
    """Return (r_multiple, pnl_usd) for an order closed at exit_price."""
    entry = order.filled_price
    sl    = order.stop_loss

    if order.direction == "LONG":
        pnl_pips  = (exit_price - entry) / 0.0001
        risk_pips = abs(entry - sl)      / 0.0001
    else:
        pnl_pips  = (entry - exit_price) / 0.0001
        risk_pips = abs(sl - entry)      / 0.0001

    r_multiple = pnl_pips / risk_pips if risk_pips > 0 else 0.0
    pip_value  = 10.0  # standard EURUSD
    pnl_usd    = pnl_pips * order.lot_size * pip_value
    return r_multiple, pnl_usd


def _fmt_dt(dt: Optional[datetime]) -> str:
    if dt is None:
        return "N/A"
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")


def _fmt_pf(pf: float) -> str:
    if pf == float("inf"):
        return "∞"
    return f"{pf:.2f}"
