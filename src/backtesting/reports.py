"""
Sprint 13 — Report Generators
==============================
Pure formatting functions that produce human-readable strings from
BacktestResult and ValidationReport objects.

No trading logic here — only presentation.

Public API
----------
generate_scorecard(result)           → str
generate_comparison_report(results)  → str
generate_validation_report(report)   → str
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from src.backtesting.backtest_runner import BacktestResult, ValidationReport

_SEP  = "=" * 60
_SEP2 = "-" * 60
_SEP3 = "─" * 60


# ---------------------------------------------------------------------------
# Scorecard
# ---------------------------------------------------------------------------

def generate_scorecard(result: BacktestResult) -> str:
    """
    Generate a multi-section strategy scorecard for a single BacktestResult.

    Sections
    --------
    Header | Trade Statistics | Performance Metrics |
    Strategy Breakdown | Review Analysis | Data Quality
    """
    mode_label = _mode_label(result.strategy_mode)
    lines = [
        _SEP,
        f"  {mode_label} — STRATEGY SCORECARD",
        _SEP,
        "",
    ]

    # Error short-circuit
    if result.error_message:
        lines += [
            f"  *** ERROR: {result.error_message} ***",
            "",
            f"  Symbol    : {result.symbol}",
            f"  Timeframe : {result.timeframe}",
            f"  Source    : {result.data_source}",
            "",
            _SEP,
        ]
        return "\n".join(lines)

    # ── Executive ────────────────────────────────────────────────
    lines += [
        "── EXECUTIVE SUMMARY ──────────────────────────────────",
        f"  Symbol        : {result.symbol}",
        f"  Timeframe     : {result.timeframe}",
        f"  Strategy mode : {result.strategy_mode}",
        f"  Data source   : {result.data_source}",
        f"  Date range    : {_fmt_range(result.date_range)}",
        f"  Run completed : {_fmt_dt(result.completed_at)}",
        f"  Candles proc  : {result.candles_processed}",
        "",
        "── TRADE STATISTICS ────────────────────────────────────",
        f"  Generated     : {result.trades_generated}",
        f"  Approved      : {result.trades_approved}",
        f"  Rejected      : {result.trades_rejected}",
        f"  Executed      : {result.trades_executed}",
        f"  Wins          : {result.wins}",
        f"  Losses        : {result.losses}",
        f"  Win Rate      : {result.win_rate * 100:.1f}%",
        f"  Max consec W  : {result.max_consecutive_wins}",
        f"  Max consec L  : {result.max_consecutive_losses}",
        "",
        "── PERFORMANCE METRICS ─────────────────────────────────",
        f"  Initial Bal   : ${result.initial_balance:,.2f}",
        f"  Final Bal     : ${result.final_balance:,.2f}",
        f"  Net P&L       : ${result.net_profit_usd:+,.2f}",
        f"  Gross Profit  : {result.gross_profit:.3f}R",
        f"  Gross Loss    : {result.gross_loss:.3f}R",
        f"  Profit Factor : {_fmt_pf(result.profit_factor)}",
        f"  Expectancy    : {result.expectancy_r:+.3f}R",
        f"  Average R     : {result.average_r:+.3f}R",
        f"  Max Drawdown  : {result.max_drawdown_pct:.2f}%",
        f"  Baseline pass : {'✅ YES' if result.passes_baseline else '❌ NO'}",
        "",
        "── STRATEGY BREAKDOWN ──────────────────────────────────",
    ]

    for stats in (result.pin_bar, result.engulfing):
        if stats.trades == 0:
            lines.append(f"  {stats.strategy_name.upper():20s} — no trades")
            continue
        lines += [
            f"  {stats.strategy_name.upper()}",
            f"    Trades   : {stats.trades}",
            f"    Wins     : {stats.wins}",
            f"    Losses   : {stats.losses}",
            f"    Win Rate : {stats.win_rate * 100:.1f}%",
            f"    PF       : {_fmt_pf(stats.profit_factor)}",
            f"    Exp      : {stats.expectancy_r:+.3f}R",
            f"    Avg Win  : {stats.avg_winner_r:+.3f}R",
            f"    Avg Loss : {stats.avg_loser_r:+.3f}R",
            f"    Max DD   : {stats.max_drawdown_pct:.2f}%",
        ]

    lines += [
        "",
        "── REVIEW ANALYSIS (M19) ───────────────────────────────",
        f"  BAD_SIGNAL          : {result.bad_signal}",
        f"  BAD_REGIME          : {result.bad_regime}",
        f"  BAD_LEVEL           : {result.bad_level}",
        f"  BAD_EXECUTION       : {result.bad_execution}",
        f"  NORMAL_STATISTICAL  : {result.normal_statistical}",
        "",
        "── DATA QUALITY ────────────────────────────────────────",
        f"  Candles loaded : {result.total_candles_loaded}",
        f"  Duplicates     : {result.duplicate_candles}",
        f"  Missing gaps   : {result.missing_candles}",
        f"  Invalid rows   : {result.invalid_rows}",
        "",
        _SEP,
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Comparison report
# ---------------------------------------------------------------------------

def generate_comparison_report(results: List[BacktestResult]) -> str:
    """
    Side-by-side comparison of multiple BacktestResult objects.

    Works with 1–N results.  Columns are truncated to terminal width.
    """
    if not results:
        return "No results to compare."

    lines = [
        _SEP,
        "  STRATEGY COMPARISON REPORT",
        _SEP,
        "",
    ]

    col_w = 18   # width per result column
    modes = [_mode_label(r.strategy_mode) for r in results]

    # Header row
    lines.append(_row("METRIC", modes, col_w))
    lines.append(_row("", ["─" * (col_w - 2)] * len(results), col_w,
                       left_w=22))

    def row(label, values):
        lines.append(_row(label, values, col_w))

    row("Symbol",       [r.symbol for r in results])
    row("Timeframe",    [r.timeframe for r in results])
    row("",             [""] * len(results))
    row("Candles proc", [str(r.candles_processed) for r in results])
    row("Trades exec",  [str(r.trades_executed) for r in results])
    row("Wins",         [str(r.wins) for r in results])
    row("Losses",       [str(r.losses) for r in results])
    row("Win Rate",     [f"{r.win_rate * 100:.1f}%" for r in results])
    row("",             [""] * len(results))
    row("Net P&L",      [f"${r.net_profit_usd:+,.0f}" for r in results])
    row("Profit Factor",[_fmt_pf(r.profit_factor) for r in results])
    row("Expectancy",   [f"{r.expectancy_r:+.3f}R" for r in results])
    row("Average R",    [f"{r.average_r:+.3f}R" for r in results])
    row("Max Drawdown", [f"{r.max_drawdown_pct:.2f}%" for r in results])
    row("Max Con Wins", [str(r.max_consecutive_wins) for r in results])
    row("Max Con Loss", [str(r.max_consecutive_losses) for r in results])
    row("",             [""] * len(results))
    row("Baseline",     [("✅" if r.passes_baseline else "❌") for r in results])
    row("",             [""] * len(results))
    row("BAD_SIGNAL",   [str(r.bad_signal) for r in results])
    row("BAD_REGIME",   [str(r.bad_regime) for r in results])
    row("BAD_LEVEL",    [str(r.bad_level) for r in results])
    row("BAD_EXEC",     [str(r.bad_execution) for r in results])
    row("NORMAL_STAT",  [str(r.normal_statistical) for r in results])

    lines.append("")
    lines.append(_SEP)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Validation report
# ---------------------------------------------------------------------------

def generate_validation_report(report: ValidationReport) -> str:
    """
    Full Strategy Validation Lab report including ranking, metrics, and
    plain-language recommendations.
    """
    lines = [
        _SEP,
        "  STRATEGY VALIDATION LAB — PHASE 1 REPORT",
        _SEP,
        "",
        f"  Generated  : {_fmt_dt(report.generated_at)}",
        f"  Symbol     : {report.pin_bar_result.symbol}",
        f"  Timeframe  : {report.pin_bar_result.timeframe}",
        "",
        "── STRATEGY RANKINGS ───────────────────────────────────",
    ]

    for rank, mode in enumerate(report.strategy_rankings, start=1):
        res = _get_result(report, mode)
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, "  ")
        lines.append(
            f"  {medal} #{rank}  {_mode_label(mode):20s}"
            f"  PF={_fmt_pf(res.profit_factor):6s}"
            f"  Exp={res.expectancy_r:+.3f}R"
            f"  WR={res.win_rate * 100:.1f}%"
            f"  DD={res.max_drawdown_pct:.1f}%"
            f"  Trades={res.trades_executed}"
        )

    lines += [
        "",
        "── KEY FINDINGS ────────────────────────────────────────",
        f"  Best strategy    : {_mode_label(report.best_strategy)}",
        f"  Worst strategy   : {_mode_label(report.worst_strategy)}",
        f"  Highest PF       : {_fmt_pf(report.highest_pf)} "
        f"({_mode_label(report.highest_pf_mode)})",
        f"  Highest Exp      : {report.highest_expectancy:+.3f}R "
        f"({_mode_label(report.highest_exp_mode)})",
        f"  Lowest Drawdown  : {report.lowest_drawdown:.2f}% "
        f"({_mode_label(report.lowest_dd_mode)})",
        "",
    ]

    # Compact per-mode table
    all_results = [
        ("pin_bar_only",   report.pin_bar_result),
        ("engulfing_only", report.engulfing_result),
        ("combined",       report.combined_result),
    ]
    lines.append("── SIDE-BY-SIDE METRICS ────────────────────────────────")
    lines.append(generate_comparison_report(
        [r for _, r in all_results]
    ))

    lines += [
        "",
        "── RECOMMENDATIONS ─────────────────────────────────────",
    ]
    for i, rec in enumerate(report.recommendations, start=1):
        lines.append(f"  {i}. {rec}")

    lines += ["", _SEP]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _mode_label(mode: str) -> str:
    return {
        "pin_bar_only":   "PIN BAR",
        "engulfing_only": "ENGULFING BAR",
        "combined":       "COMBINED",
    }.get(mode, mode.upper())


def _fmt_pf(pf: float) -> str:
    if pf != pf:          # NaN guard
        return "N/A"
    if pf == float("inf"):
        return "∞"
    return f"{pf:.2f}"


def _fmt_dt(dt: Optional[datetime]) -> str:
    if dt is None:
        return "N/A"
    return dt.strftime("%Y-%m-%d %H:%M UTC")


def _fmt_range(date_range) -> str:
    start, end = date_range
    s = start.strftime("%Y-%m-%d") if start else "N/A"
    e = end.strftime("%Y-%m-%d")   if end   else "N/A"
    return f"{s} → {e}"


def _row(label: str, values: List[str], col_w: int, left_w: int = 22) -> str:
    cells = "".join(str(v)[:col_w - 1].ljust(col_w) for v in values)
    return f"  {label[:left_w - 2]:{left_w - 2}s}  {cells}"


def _get_result(report: ValidationReport, mode: str) -> BacktestResult:
    return {
        "pin_bar_only":   report.pin_bar_result,
        "engulfing_only": report.engulfing_result,
        "combined":       report.combined_result,
    }[mode]
