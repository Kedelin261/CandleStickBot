"""
Sprint 13 — Backtest Runner & Strategy Validation Lab
======================================================
Wraps PipelineRunner to provide three evaluation modes:

    1. Pin Bar only
    2. Engulfing Bar only
    3. Combined (both strategies)

Does NOT duplicate any strategy, risk, execution, or analytics logic.
All computation is delegated to the existing Phase 1 pipeline.

Public API
----------
BacktestConfig          — configuration dataclass
BacktestResult          — rich result dataclass (extends PipelineResult data)
BacktestRunner          — orchestrates run modes and report calls
StrategyValidationLab   — runs all three modes and ranks strategies

No Phase 2 features. No optimization. No live execution. No broker APIs.
No Fibonacci, Inside Bar, False Breakout, Portfolio, Correlation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import io

from src.backtesting.data_loader import (
    DataQualityReport,
    load_candles_from_csv,
)
from src.data.types import CandleData
from src.integration.pipeline_runner import (
    PipelineConfig,
    PipelineResult,
    PipelineRunner,
    StrategyBreakdown,
    _compute_max_drawdown,
)

logger = logging.getLogger("candlestickbot.backtesting.runner")


# ---------------------------------------------------------------------------
# BacktestConfig
# ---------------------------------------------------------------------------

@dataclass
class BacktestConfig:
    """
    Configuration for a single backtest run.

    symbol          : Instrument  (Phase 1: 'EURUSD')
    timeframe       : Chart frame (Phase 1: 'D1')
    initial_balance : Starting equity in USD
    slippage_pips   : Fixed adverse slippage on each fill
    start_date      : Filter candles after this date (UTC, inclusive)
    end_date        : Filter candles before this date (UTC, inclusive)
    enable_pin_bar  : Include Pin Bar signals
    enable_engulfing: Include Engulfing Bar signals
    risk_enabled    : Apply M09 position sizing and risk checks
    analytics_enabled: Wire M18 StrategyAnalyticsEngine
    review_enabled  : Wire M19 TradeReviewEngine
    minimum_tqs     : Minimum TQS score threshold (0 = no filter)
    minimum_rr      : Minimum risk-reward ratio (default 2.0)
    lookback_window : Candle context window per evaluation
    """
    symbol:            str              = "EURUSD"
    timeframe:         str              = "D1"
    initial_balance:   float            = 10_000.0
    slippage_pips:     float            = 1.0
    start_date:        Optional[datetime] = None
    end_date:          Optional[datetime] = None
    enable_pin_bar:    bool             = True
    enable_engulfing:  bool             = True
    risk_enabled:      bool             = True
    analytics_enabled: bool             = True
    review_enabled:    bool             = True
    minimum_tqs:       float            = 0.0
    minimum_rr:        float            = 2.0
    lookback_window:   int              = 50

    def to_pipeline_config(self) -> PipelineConfig:
        """Convert to a PipelineConfig for PipelineRunner."""
        return PipelineConfig(
            symbol=self.symbol,
            timeframe=self.timeframe,
            initial_balance=self.initial_balance,
            slippage_pips=self.slippage_pips,
            enable_pin_bar=self.enable_pin_bar,
            enable_engulfing=self.enable_engulfing,
            risk_enabled=self.risk_enabled,
            analytics_enabled=self.analytics_enabled,
            review_enabled=self.review_enabled,
            minimum_tqs=self.minimum_tqs,
            minimum_rr=self.minimum_rr,
            lookback_window=self.lookback_window,
        )


# ---------------------------------------------------------------------------
# StrategyStats  (per-strategy breakdown inside BacktestResult)
# ---------------------------------------------------------------------------

@dataclass
class StrategyStats:
    """
    Detailed per-strategy performance extracted from PipelineResult
    and the M18 StrategySummary.
    """
    strategy_name:        str
    trades:               int   = 0
    wins:                 int   = 0
    losses:               int   = 0
    profit_factor:        float = 0.0
    expectancy_r:         float = 0.0
    avg_winner_r:         float = 0.0
    avg_loser_r:          float = 0.0
    max_consecutive_wins:  int  = 0
    max_consecutive_losses: int = 0
    max_drawdown_pct:     float = 0.0

    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses
        return self.wins / total if total > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "strategy_name":          self.strategy_name,
            "trades":                 self.trades,
            "wins":                   self.wins,
            "losses":                 self.losses,
            "win_rate":               round(self.win_rate, 4),
            "profit_factor":          self.profit_factor,
            "expectancy_r":           self.expectancy_r,
            "avg_winner_r":           self.avg_winner_r,
            "avg_loser_r":            self.avg_loser_r,
            "max_consecutive_wins":   self.max_consecutive_wins,
            "max_consecutive_losses": self.max_consecutive_losses,
            "max_drawdown_pct":       self.max_drawdown_pct,
        }


# ---------------------------------------------------------------------------
# BacktestResult
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    """
    Complete output of one backtest run.

    Populated by BacktestRunner after PipelineRunner.run() completes.
    Combines PipelineResult data with data-quality information and
    richer per-strategy stats from M18.
    """
    # ---- identity ----
    symbol:        str
    timeframe:     str
    strategy_mode: str              # "pin_bar_only" | "engulfing_only" | "combined"
    data_source:   str              = "<unknown>"
    date_range:    Tuple[Optional[datetime], Optional[datetime]] = (None, None)
    started_at:    Optional[datetime] = None
    completed_at:  Optional[datetime] = None

    # ---- trade counts ----
    trades_generated:  int = 0
    trades_approved:   int = 0
    trades_rejected:   int = 0
    trades_executed:   int = 0
    wins:              int = 0
    losses:            int = 0

    # ---- performance ----
    initial_balance:   float = 10_000.0
    final_balance:     float = 10_000.0
    net_profit_usd:    float = 0.0
    gross_profit:      float = 0.0   # in R
    gross_loss:        float = 0.0   # in R
    profit_factor:     float = 0.0
    expectancy_r:      float = 0.0
    average_r:         float = 0.0
    win_rate:          float = 0.0
    max_drawdown_pct:  float = 0.0
    max_consecutive_wins:  int = 0
    max_consecutive_losses: int = 0
    candles_processed: int  = 0

    # ---- per-strategy breakdown ----
    pin_bar:   StrategyStats = field(
        default_factory=lambda: StrategyStats("pin_bar"))
    engulfing: StrategyStats = field(
        default_factory=lambda: StrategyStats("engulfing_bar"))

    # ---- trade review (M19) ----
    bad_signal:         int = 0
    bad_regime:         int = 0
    bad_level:          int = 0
    bad_execution:      int = 0
    normal_statistical: int = 0

    # ---- data quality ----
    total_candles_loaded:   int = 0
    duplicate_candles:      int = 0
    missing_candles:        int = 0
    invalid_rows:           int = 0

    # ---- error ----
    error_message: Optional[str] = None

    # ---- internal audit ----
    _pipeline_result: Optional[PipelineResult] = field(
        default=None, repr=False
    )

    @property
    def is_successful(self) -> bool:
        return self.error_message is None

    @property
    def passes_baseline(self) -> bool:
        """
        Sprint 15 FIX-4 — Canonical baseline criteria (§5 of Sprint 15 spec):
          PF > 1.10  AND  Expectancy > 0  AND  Max DD < 25%  AND  N >= 30
        Previously: PF >= 1.1, WR >= 40%, DD <= 20%, N >= 10.
        Changed to align measurement layer with experiment design (RC-5).
        """
        if self.trades_executed < 30:   # RC-5: was 10, spec requires 30
            return False
        return (
            self.profit_factor  >  1.10    # RC-5: strict >, was >=
            and self.expectancy_r > 0.0    # RC-5: replaced WR gate
            and self.max_drawdown_pct < 25.0  # RC-5: < 25%, was <= 20%
        )

    def to_dict(self) -> dict:
        return {
            "symbol":            self.symbol,
            "timeframe":         self.timeframe,
            "strategy_mode":     self.strategy_mode,
            "data_source":       self.data_source,
            "trades_executed":   self.trades_executed,
            "wins":              self.wins,
            "losses":            self.losses,
            "win_rate":          round(self.win_rate, 4),
            "profit_factor":     self.profit_factor,
            "expectancy_r":      self.expectancy_r,
            "average_r":         self.average_r,
            "max_drawdown_pct":  self.max_drawdown_pct,
            "net_profit_usd":    self.net_profit_usd,
            "passes_baseline":   self.passes_baseline,
            "max_consecutive_wins":   self.max_consecutive_wins,
            "max_consecutive_losses": self.max_consecutive_losses,
        }


# ---------------------------------------------------------------------------
# ValidationReport
# ---------------------------------------------------------------------------

@dataclass
class ValidationReport:
    """
    Output of StrategyValidationLab — ranks all three strategy modes.
    """
    pin_bar_result:   BacktestResult
    engulfing_result: BacktestResult
    combined_result:  BacktestResult
    generated_at:     datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Rankings (populated by _rank())
    best_strategy:    str = ""
    worst_strategy:   str = ""
    highest_pf:       float = 0.0
    highest_pf_mode:  str = ""
    highest_expectancy: float = 0.0
    highest_exp_mode: str = ""
    lowest_drawdown:  float = 0.0
    lowest_dd_mode:   str = ""
    strategy_rankings: List[str] = field(default_factory=list)
    recommendations:  List[str] = field(default_factory=list)

    def _all_results(self) -> List[Tuple[str, BacktestResult]]:
        return [
            ("pin_bar_only",    self.pin_bar_result),
            ("engulfing_only",  self.engulfing_result),
            ("combined",        self.combined_result),
        ]

    def rank(self) -> "ValidationReport":
        """Compute rankings and populate recommendation fields."""
        scored: List[Tuple[float, str, BacktestResult]] = []
        for mode, res in self._all_results():
            score = _composite_score(res)
            scored.append((score, mode, res))

        scored.sort(key=lambda x: x[0], reverse=True)

        self.strategy_rankings = [m for _, m, _ in scored]
        self.best_strategy     = scored[0][1]  if scored else ""
        self.worst_strategy    = scored[-1][1] if scored else ""

        # Highest PF
        pf_sorted = sorted(self._all_results(),
                           key=lambda t: t[1].profit_factor, reverse=True)
        if pf_sorted:
            self.highest_pf_mode = pf_sorted[0][0]
            self.highest_pf      = pf_sorted[0][1].profit_factor

        # Highest expectancy
        exp_sorted = sorted(self._all_results(),
                            key=lambda t: t[1].expectancy_r, reverse=True)
        if exp_sorted:
            self.highest_exp_mode    = exp_sorted[0][0]
            self.highest_expectancy  = exp_sorted[0][1].expectancy_r

        # Lowest drawdown (among results with ≥1 trade)
        dd_candidates = [(m, r) for m, r in self._all_results()
                         if r.trades_executed > 0]
        if dd_candidates:
            dd_sorted = sorted(dd_candidates,
                               key=lambda t: t[1].max_drawdown_pct)
            self.lowest_dd_mode  = dd_sorted[0][0]
            self.lowest_drawdown = dd_sorted[0][1].max_drawdown_pct

        self.recommendations = _build_recommendations(self)
        return self


# ---------------------------------------------------------------------------
# BacktestRunner
# ---------------------------------------------------------------------------

class BacktestRunner:
    """
    Phase 1 Backtesting Framework.

    Wraps PipelineRunner to provide named evaluation modes and CSV loading.
    Does NOT reimplement any trading or analytics logic.

    Usage
    -----
    runner = BacktestRunner(BacktestConfig())
    result = runner.run_combined("path/to/EURUSD_D1.csv")
    print(runner.generate_scorecard(result))
    """

    def __init__(self, config: Optional[BacktestConfig] = None) -> None:
        self._cfg = config or BacktestConfig()

    # ------------------------------------------------------------------
    # Named run modes
    # ------------------------------------------------------------------

    def run_pin_bar_only(
        self,
        source: Union[str, Path, io.StringIO, List[CandleData]],
    ) -> BacktestResult:
        """Backtest with Pin Bar signals only."""
        cfg = _override(self._cfg, enable_pin_bar=True, enable_engulfing=False)
        return self._run(source, cfg, mode="pin_bar_only")

    def run_engulfing_only(
        self,
        source: Union[str, Path, io.StringIO, List[CandleData]],
    ) -> BacktestResult:
        """Backtest with Engulfing Bar signals only."""
        cfg = _override(self._cfg, enable_pin_bar=False, enable_engulfing=True)
        return self._run(source, cfg, mode="engulfing_only")

    def run_combined(
        self,
        source: Union[str, Path, io.StringIO, List[CandleData]],
    ) -> BacktestResult:
        """Backtest with both Pin Bar and Engulfing Bar signals."""
        cfg = _override(self._cfg, enable_pin_bar=True, enable_engulfing=True)
        return self._run(source, cfg, mode="combined")

    def run_from_candles(
        self,
        candles: List[CandleData],
        mode:    str = "combined",
    ) -> BacktestResult:
        """
        Run a backtest directly on a pre-built List[CandleData].

        mode : "pin_bar_only" | "engulfing_only" | "combined"
        """
        if mode == "pin_bar_only":
            return self.run_pin_bar_only(candles)
        if mode == "engulfing_only":
            return self.run_engulfing_only(candles)
        return self.run_combined(candles)

    # ------------------------------------------------------------------
    # Report generators (thin wrappers over reports.py)
    # ------------------------------------------------------------------

    def generate_scorecard(self, result: BacktestResult) -> str:
        """Return a multi-section strategy scorecard string."""
        from src.backtesting.reports import generate_scorecard
        return generate_scorecard(result)

    def generate_comparison_report(
        self, results: List[BacktestResult]
    ) -> str:
        """Return a side-by-side comparison of multiple BacktestResults."""
        from src.backtesting.reports import generate_comparison_report
        return generate_comparison_report(results)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(
        self,
        source: Union[str, Path, io.StringIO, List[CandleData]],
        cfg:    BacktestConfig,
        mode:   str,
    ) -> BacktestResult:
        """Core dispatch: load candles if needed, run pipeline, build result."""
        quality: Optional[DataQualityReport] = None
        data_source = "<candles>"

        # Load from CSV if source is a path or StringIO
        if isinstance(source, (str, Path, io.StringIO)):
            try:
                candles, quality = load_candles_from_csv(
                    source,
                    symbol=cfg.symbol,
                    timeframe=cfg.timeframe,
                    start_date=cfg.start_date,
                    end_date=cfg.end_date,
                )
                data_source = quality.source_path
            except (ValueError, OSError) as exc:
                return _error_result(mode, cfg, str(exc),
                                     source_path=str(source))
        else:
            candles = list(source)
            # Apply date filter even when candles are passed directly
            if cfg.start_date:
                candles = [c for c in candles if c.timestamp >= cfg.start_date]
            if cfg.end_date:
                candles = [c for c in candles if c.timestamp <= cfg.end_date]

        if not candles:
            return _error_result(
                mode, cfg,
                "No candles available after date filtering",
                source_path=data_source,
                quality=quality,
            )

        # Run the Phase 1 pipeline
        pipeline_cfg = cfg.to_pipeline_config()
        runner = PipelineRunner(pipeline_cfg)
        pipeline_result = runner.run(candles)

        # Build the rich BacktestResult
        result = _build_backtest_result(
            pipeline_result=pipeline_result,
            pipeline_runner=runner,
            mode=mode,
            cfg=cfg,
            data_source=data_source,
            quality=quality,
            candles=candles,
        )
        return result


# ---------------------------------------------------------------------------
# StrategyValidationLab
# ---------------------------------------------------------------------------

class StrategyValidationLab:
    """
    Runs all three strategy modes and produces a ranked ValidationReport.

    Usage
    -----
    lab = StrategyValidationLab(BacktestConfig())
    report = lab.run("path/to/EURUSD_D1.csv")
    print(lab.generate_validation_report(report))
    """

    def __init__(self, config: Optional[BacktestConfig] = None) -> None:
        self._cfg = config or BacktestConfig()

    def run(
        self,
        source: Union[str, Path, io.StringIO, List[CandleData]],
    ) -> ValidationReport:
        """
        Execute all three modes on the same data and return a ranked
        ValidationReport.
        """
        runner = BacktestRunner(self._cfg)

        logger.info("StrategyValidationLab: running pin_bar_only")
        pb  = runner.run_pin_bar_only(source)

        logger.info("StrategyValidationLab: running engulfing_only")
        eng = runner.run_engulfing_only(source)

        logger.info("StrategyValidationLab: running combined")
        com = runner.run_combined(source)

        report = ValidationReport(
            pin_bar_result=pb,
            engulfing_result=eng,
            combined_result=com,
        )
        report.rank()
        return report

    def generate_validation_report(self, report: ValidationReport) -> str:
        """Return a formatted multi-section validation report string."""
        from src.backtesting.reports import generate_validation_report
        return generate_validation_report(report)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _override(cfg: BacktestConfig, **kwargs) -> BacktestConfig:
    """Return a shallow copy of cfg with the given fields overridden."""
    import dataclasses
    d = dataclasses.asdict(cfg)
    d.update(kwargs)
    return BacktestConfig(**d)


def _build_backtest_result(
    pipeline_result: PipelineResult,
    pipeline_runner: PipelineRunner,
    mode:            str,
    cfg:             BacktestConfig,
    data_source:     str,
    quality:         Optional[DataQualityReport],
    candles:         List[CandleData],
) -> BacktestResult:
    """Assemble a BacktestResult from PipelineResult + M18 data."""
    pr = pipeline_result

    # Date range from actual candles processed
    date_range: Tuple[Optional[datetime], Optional[datetime]] = (None, None)
    if candles:
        date_range = (candles[0].timestamp, candles[-1].timestamp)

    result = BacktestResult(
        symbol=pr.symbol,
        timeframe=pr.timeframe,
        strategy_mode=mode,
        data_source=data_source,
        date_range=date_range,
        started_at=pr.started_at,
        completed_at=pr.completed_at,
        # trade counts
        trades_generated=pr.trades_generated,
        trades_approved=pr.trades_approved,
        trades_rejected=pr.trades_rejected,
        trades_executed=pr.trades_executed,
        wins=pr.wins,
        losses=pr.losses,
        # performance
        initial_balance=pr.initial_balance,
        final_balance=pr.final_balance,
        net_profit_usd=pr.net_profit_usd,
        gross_profit=pr.gross_profit,
        gross_loss=pr.gross_loss,
        profit_factor=pr.profit_factor,
        expectancy_r=pr.expectancy,
        win_rate=pr.win_rate / 100.0 if pr.win_rate > 1 else pr.win_rate,
        max_drawdown_pct=pr.max_drawdown,
        candles_processed=pr.candles_processed,
        # trade review
        bad_signal=pr.bad_signal,
        bad_regime=pr.bad_regime,
        bad_level=pr.bad_level,
        bad_execution=pr.bad_execution,
        normal_statistical=pr.normal_statistical,
        # error
        error_message=pr.error_message,
        # internal
        _pipeline_result=pr,
    )

    # average_r
    if pr._r_multiples:
        result.average_r = sum(pr._r_multiples) / len(pr._r_multiples)

    # consecutive streaks from closed orders
    if pr._closed_orders:
        result.max_consecutive_wins, result.max_consecutive_losses = (
            _compute_streaks(pr._closed_orders)
        )

    # data quality
    if quality:
        result.total_candles_loaded = quality.total_candles
        result.duplicate_candles    = quality.duplicate_candles
        result.missing_candles      = quality.missing_candles
        result.invalid_rows         = quality.invalid_rows
    else:
        result.total_candles_loaded = len(candles)

    # per-strategy stats from M18
    analytics = pipeline_runner.get_analytics_engine()
    result.pin_bar   = _strategy_stats_from_m18(
        analytics, "pin_bar", pr.symbol, pr.timeframe, pr.pin_bar
    )
    result.engulfing = _strategy_stats_from_m18(
        analytics, "engulfing_bar", pr.symbol, pr.timeframe, pr.engulfing
    )

    return result


def _strategy_stats_from_m18(
    analytics,
    strategy_name: str,
    symbol:        str,
    timeframe:     str,
    breakdown:     StrategyBreakdown,
) -> StrategyStats:
    """Build StrategyStats merging M18 summary with PipelineResult breakdown."""
    try:
        s = analytics.get_strategy_summary(strategy_name, symbol, timeframe)
        return StrategyStats(
            strategy_name=strategy_name,
            trades=s.total_trades,
            wins=s.win_count,
            losses=s.loss_count,
            profit_factor=s.profit_factor,
            expectancy_r=s.expectancy_r,
            avg_winner_r=s.avg_winner_r,
            avg_loser_r=s.avg_loser_r,
            max_consecutive_wins=s.max_consecutive_wins,
            max_consecutive_losses=s.max_consecutive_losses,
            max_drawdown_pct=s.max_drawdown_pct,
        )
    except Exception:
        # Fallback to PipelineResult breakdown if M18 unavailable
        return StrategyStats(
            strategy_name=strategy_name,
            trades=breakdown.trades,
            wins=breakdown.wins,
            losses=breakdown.losses,
            profit_factor=breakdown.profit_factor,
        )


def _compute_streaks(
    orders,
) -> Tuple[int, int]:
    """Return (max_consecutive_wins, max_consecutive_losses) from closed orders."""
    max_w = max_l = cur_w = cur_l = 0
    for o in orders:
        if o.is_winner:
            cur_w += 1
            cur_l  = 0
            max_w  = max(max_w, cur_w)
        elif o.is_loser:
            cur_l += 1
            cur_w  = 0
            max_l  = max(max_l, cur_l)
    return max_w, max_l


def _error_result(
    mode:        str,
    cfg:         BacktestConfig,
    message:     str,
    source_path: str = "<unknown>",
    quality:     Optional[DataQualityReport] = None,
) -> BacktestResult:
    """Return a BacktestResult that communicates a load or run error."""
    r = BacktestResult(
        symbol=cfg.symbol,
        timeframe=cfg.timeframe,
        strategy_mode=mode,
        data_source=source_path,
        error_message=message,
        initial_balance=cfg.initial_balance,
        final_balance=cfg.initial_balance,
    )
    if quality:
        r.total_candles_loaded = quality.total_candles
        r.duplicate_candles    = quality.duplicate_candles
        r.invalid_rows         = quality.invalid_rows
    return r


def _composite_score(result: BacktestResult) -> float:
    """
    Simple composite score for ranking strategies.
    Higher is better.

    Combines: expectancy_r (primary), profit_factor, negative drawdown.
    Returns 0.0 when no trades were executed.
    """
    if result.trades_executed == 0:
        return 0.0
    pf  = min(result.profit_factor, 5.0)   # cap to avoid inf dominating
    exp = result.expectancy_r
    dd  = result.max_drawdown_pct / 100.0
    return (exp * 2.0) + (pf * 0.5) - (dd * 1.0)


def _build_recommendations(report: ValidationReport) -> List[str]:
    """Generate plain-language recommendations from ranked results."""
    recs: List[str] = []
    results = dict(report._all_results())

    best_r = results.get(report.best_strategy)
    if best_r and best_r.trades_executed >= 5:
        pf  = best_r.profit_factor
        exp = best_r.expectancy_r
        if pf >= 1.5 and exp > 0.2:
            recs.append(
                f"{report.best_strategy} shows a promising edge "
                f"(PF={pf:.2f}, Exp={exp:+.3f}R). "
                "Consider increasing sample size with more historical data."
            )
        elif pf >= 1.1:
            recs.append(
                f"{report.best_strategy} has marginal edge (PF={pf:.2f}). "
                "Insufficient evidence of consistent edge — continue monitoring."
            )
        else:
            recs.append(
                "No strategy shows a statistically meaningful edge. "
                "Review signal confluence requirements before live deployment."
            )
    else:
        recs.append(
            "Insufficient trades to evaluate edge. "
            "Use a longer historical dataset (recommend ≥ 500 candles)."
        )

    # Drawdown warning
    for mode, res in report._all_results():
        if res.max_drawdown_pct > 20.0:
            recs.append(
                f"{mode} max drawdown is {res.max_drawdown_pct:.1f}% — "
                "exceeds 20% baseline threshold."
            )

    # Combined vs individual
    com = results.get("combined")
    pb  = results.get("pin_bar_only")
    eng = results.get("engulfing_only")
    if com and pb and eng:
        if (com.profit_factor > pb.profit_factor
                and com.profit_factor > eng.profit_factor):
            recs.append(
                "Combined system outperforms individual strategies — "
                "diversification benefit confirmed."
            )
        elif (com.profit_factor < pb.profit_factor
              or com.profit_factor < eng.profit_factor):
            recs.append(
                "Individual strategies outperform combined system — "
                "consider running strategies independently."
            )

    return recs
