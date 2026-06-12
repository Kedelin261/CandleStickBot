"""
M19 — Trade Review Engine
Phase 1 implementation of the post-trade loss classification and monthly
failure reporting system.

Classification priority order (first match wins):
    OVERRIDDEN → BAD_REGIME → BAD_SIGNAL → BAD_LEVEL → BAD_EXECUTION
    → NORMAL_STATISTICAL_LOSS

Loss categories are drawn from src.types.LossCategory (canonical source of
truth). The Phase 0 stub in src/trade_review/classifier.py uses different
names and must NOT be imported here.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from src.types import LossCategory

logger = logging.getLogger("candlestickbot.analytics.trade_review")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ReviewConfig:
    """
    Configurable thresholds for the TradeReviewEngine.

    pattern_quality_threshold : minimum acceptable pattern quality score
        (0-100).  Trades with score < this → BAD_SIGNAL candidate.
    level_strength_threshold  : minimum acceptable level strength score
        (0-100).  Trades with score < this → BAD_LEVEL candidate.
    max_slippage_pips         : maximum acceptable fill slippage in pips.
        Trades with |slippage| > this → BAD_EXECUTION candidate.
    min_stop_pips             : minimum acceptable stop distance in pips.
        Trades with stop < this → BAD_EXECUTION candidate.
    degradation_threshold     : category frequency ≥ this fraction triggers
        a systematic issue flag (default 0.30 = 30 %).
    degradation_window        : number of recent losses to examine when
        checking for systematic issues.
    """

    pattern_quality_threshold: float = 50.0
    level_strength_threshold:  float = 50.0
    max_slippage_pips:         float = 2.0
    min_stop_pips:             float = 5.0
    degradation_threshold:     float = 0.30
    degradation_window:        int   = 20


# ---------------------------------------------------------------------------
# Input context
# ---------------------------------------------------------------------------

@dataclass
class TradeContext:
    """
    Contextual metadata supplied by the caller when requesting loss
    classification.  All numeric scores run 0–100 unless noted.
    """

    pattern_quality_score: float          # M07 pattern quality (0-100)
    level_strength_score:  float          # M05 level strength (0-100)
    regime:                str            # e.g. "TRENDING", "CHOPPY", …
    fill_slippage_pips:    float  = 0.0   # actual − requested entry, abs pips
    stop_distance_pips:    float  = 10.0  # distance from entry to stop in pips
    was_overridden:        bool   = False  # True if a manual override occurred

    # Additional text for the reason string
    notes: str = ""


# ---------------------------------------------------------------------------
# Output DTOs
# ---------------------------------------------------------------------------

@dataclass
class TradeReviewResult:
    """Result of classifying a single losing trade."""

    trade_id:           str
    strategy_name:      str
    category:           LossCategory
    reason:             str
    recommended_action: str
    severity:           str          # "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
    reviewed_at:        datetime


@dataclass
class MonthlyFailureReport:
    """
    Aggregated loss-category report for a calendar month.

    *month* is an ISO-format year-month string, e.g. ``"2025-01"``.
    """

    month:               str
    total_losses:        int
    category_counts:     Dict[str, int]
    category_percentages: Dict[str, float]
    top_issue:           Optional[str]
    recommended_action:  str
    systematic_issue_flag: bool


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_GOOD_REGIMES: frozenset = frozenset({"TRENDING"})

_SEVERITY_MAP: Dict[LossCategory, str] = {
    LossCategory.OVERRIDDEN:           "CRITICAL",
    LossCategory.BAD_REGIME:           "HIGH",
    LossCategory.BAD_SIGNAL:           "MEDIUM",
    LossCategory.BAD_LEVEL:            "MEDIUM",
    LossCategory.BAD_EXECUTION:        "HIGH",
    LossCategory.NORMAL_STATISTICAL:   "LOW",
    LossCategory.UNCATEGORIZED:         "LOW",
}

_RECOMMENDED_ACTIONS: Dict[LossCategory, str] = {
    LossCategory.OVERRIDDEN: (
        "Investigate manual override; ensure override logic is documented "
        "and does not compromise system rules."
    ),
    LossCategory.BAD_REGIME: (
        "Review regime detection sensitivity; consider raising the regime "
        "quality threshold or waiting for clearer trends."
    ),
    LossCategory.BAD_SIGNAL: (
        "Raise the pattern quality threshold in ReviewConfig; audit recent "
        "marginal patterns for this strategy."
    ),
    LossCategory.BAD_LEVEL: (
        "Review S/R level scoring; consider raising the minimum level "
        "strength threshold."
    ),
    LossCategory.BAD_EXECUTION: (
        "Check execution pathway for slippage causes; widen minimum stop "
        "distance or reduce traded size in volatile sessions."
    ),
    LossCategory.NORMAL_STATISTICAL: (
        "No action required — trade met all system rules. "
        "Normal statistical variance."
    ),
    LossCategory.UNCATEGORIZED: (
        "Trade loss has not yet been classified. Review and assign a "
        "specific loss category."
    ),
}


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class TradeReviewEngine:
    """
    M19 Trade Review Engine.

    Classifies individual losing trades, maintains an in-memory audit trail,
    and produces monthly failure reports.

    Storage is in-memory by default.  The engine is intentionally stateless
    regarding DB writes so that unit tests need no database fixtures.
    """

    def __init__(self, config: Optional[ReviewConfig] = None) -> None:
        self._config = config or ReviewConfig()
        # List of (TradeReviewResult, strategy_name) for all reviewed trades
        self._results: List[TradeReviewResult] = []

    # ------------------------------------------------------------------
    # Public API — classification
    # ------------------------------------------------------------------

    def classify_loss(
        self,
        trade_id:      str,
        strategy_name: str,
        context:       TradeContext,
    ) -> TradeReviewResult:
        """
        Classify a single losing trade and store the result.

        Classification priority:
            1. OVERRIDDEN          – manual override detected
            2. BAD_REGIME          – entered in non-trending regime
            3. BAD_SIGNAL          – pattern quality too low
            4. BAD_LEVEL           – level strength too low
            5. BAD_EXECUTION       – slippage too high or stop too tight
            6. NORMAL_STATISTICAL  – trade was valid; loss is variance
        """
        category, reason = self._classify(context)
        severity         = _SEVERITY_MAP[category]
        action           = _RECOMMENDED_ACTIONS[category]

        result = TradeReviewResult(
            trade_id=trade_id,
            strategy_name=strategy_name,
            category=category,
            reason=reason,
            recommended_action=action,
            severity=severity,
            reviewed_at=_now_utc(),
        )
        self._results.append(result)
        logger.debug(
            "M19: trade %s classified as %s (strategy=%s)",
            trade_id,
            category.value,
            strategy_name,
        )
        return result

    def review_trade(
        self,
        trade_id:      str,
        strategy_name: str,
        context:       TradeContext,
    ) -> TradeReviewResult:
        """Alias for classify_loss — explicit 'review' entry-point."""
        return self.classify_loss(trade_id, strategy_name, context)

    # ------------------------------------------------------------------
    # Public API — reporting
    # ------------------------------------------------------------------

    def generate_monthly_report(
        self,
        month: str,
        strategy_name: Optional[str] = None,
    ) -> MonthlyFailureReport:
        """
        Aggregate all reviewed losses for *month* (``"YYYY-MM"`` format)
        into a MonthlyFailureReport.

        If *strategy_name* is given, only results for that strategy are
        included.
        """
        results = self._filter_by_month_and_strategy(month, strategy_name)
        return self._build_report(month, results)

    def get_top_loss_category(
        self,
        last_n: int = 20,
        strategy_name: Optional[str] = None,
    ) -> Optional[LossCategory]:
        """
        Return the most frequent loss category in the last *last_n* results.

        Returns ``None`` if no losses have been recorded.
        """
        results = self._recent_results(last_n, strategy_name)
        if not results:
            return None
        counts: Dict[LossCategory, int] = defaultdict(int)
        for r in results:
            counts[r.category] += 1
        return max(counts, key=lambda k: counts[k])

    def suggest_parameter_adjustment(
        self,
        report: MonthlyFailureReport,
    ) -> str:
        """
        Return a human-readable parameter adjustment suggestion based on
        the top issue in *report*.
        """
        if not report.top_issue or report.total_losses == 0:
            return "Insufficient data for parameter adjustment suggestions."

        try:
            cat = LossCategory(report.top_issue)
        except ValueError:
            return f"Unknown loss category '{report.top_issue}'."

        pct = report.category_percentages.get(report.top_issue, 0.0)
        base = _RECOMMENDED_ACTIONS[cat]
        return (
            f"{report.top_issue} accounts for {pct:.1f}% of losses this month. "
            f"{base}"
        )

    def flag_systematic_issue(
        self,
        category: LossCategory,
        threshold: Optional[float] = None,
        window: Optional[int] = None,
    ) -> bool:
        """
        Return ``True`` when *category* exceeds *threshold* fraction of
        losses in the last *window* results.

        Uses ``ReviewConfig`` defaults if *threshold* / *window* omitted.
        """
        threshold = threshold if threshold is not None else self._config.degradation_threshold
        window    = window    if window    is not None else self._config.degradation_window

        recent = self._recent_results(window)
        if not recent:
            return False
        count = sum(1 for r in recent if r.category == category)
        return (count / len(recent)) >= threshold

    def get_failure_breakdown(
        self,
        strategy_name: Optional[str] = None,
    ) -> Dict[str, int]:
        """
        Return a dict mapping each LossCategory value to its total count
        across all recorded results, optionally filtered by *strategy_name*.
        """
        results = (
            self._results
            if strategy_name is None
            else [r for r in self._results
                  if r.strategy_name.lower() == strategy_name.lower()]
        )
        counts: Dict[str, int] = {cat.value: 0 for cat in LossCategory}
        for r in results:
            counts[r.category.value] += 1
        return counts

    def get_all_results(self) -> List[TradeReviewResult]:
        """Return a shallow copy of all stored TradeReviewResult objects."""
        return list(self._results)

    def reset(self) -> None:
        """Clear all stored results (useful in tests)."""
        self._results.clear()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _classify(
        self, context: TradeContext
    ) -> Tuple[LossCategory, str]:
        """Apply the priority-ordered classification rules."""
        cfg = self._config

        # 1. OVERRIDDEN
        if context.was_overridden:
            return (
                LossCategory.OVERRIDDEN,
                "Trade outcome was affected by a manual override.",
            )

        # 2. BAD_REGIME
        regime_upper = context.regime.upper().strip()
        if regime_upper not in _GOOD_REGIMES:
            return (
                LossCategory.BAD_REGIME,
                f"Trade entered during '{context.regime}' regime "
                f"(acceptable: {sorted(_GOOD_REGIMES)}).",
            )

        # 3. BAD_SIGNAL
        if context.pattern_quality_score < cfg.pattern_quality_threshold:
            return (
                LossCategory.BAD_SIGNAL,
                f"Pattern quality score {context.pattern_quality_score:.1f} "
                f"is below threshold {cfg.pattern_quality_threshold:.1f}.",
            )

        # 4. BAD_LEVEL
        if context.level_strength_score < cfg.level_strength_threshold:
            return (
                LossCategory.BAD_LEVEL,
                f"Level strength score {context.level_strength_score:.1f} "
                f"is below threshold {cfg.level_strength_threshold:.1f}.",
            )

        # 5. BAD_EXECUTION
        if (
            abs(context.fill_slippage_pips) > cfg.max_slippage_pips
            or context.stop_distance_pips < cfg.min_stop_pips
        ):
            slippage_msg = (
                f"slippage {abs(context.fill_slippage_pips):.1f} pips "
                f"> max {cfg.max_slippage_pips:.1f}"
                if abs(context.fill_slippage_pips) > cfg.max_slippage_pips
                else f"stop {context.stop_distance_pips:.1f} pips "
                     f"< min {cfg.min_stop_pips:.1f}"
            )
            return (
                LossCategory.BAD_EXECUTION,
                f"Execution quality issue: {slippage_msg}.",
            )

        # 6. NORMAL_STATISTICAL_LOSS
        return (
            LossCategory.NORMAL_STATISTICAL,
            "Trade met all system rules; loss attributed to normal "
            "statistical variance.",
        )

    def _filter_by_month_and_strategy(
        self,
        month: str,
        strategy_name: Optional[str],
    ) -> List[TradeReviewResult]:
        """Return results matching *month* (YYYY-MM) and optional strategy."""
        out: List[TradeReviewResult] = []
        for r in self._results:
            # Normalise timezone-aware vs naive comparison
            ts = r.reviewed_at
            if ts.tzinfo is not None:
                ym = ts.strftime("%Y-%m")
            else:
                ym = ts.strftime("%Y-%m")
            if ym != month:
                continue
            if strategy_name is not None:
                if r.strategy_name.lower() != strategy_name.lower():
                    continue
            out.append(r)
        return out

    def _build_report(
        self,
        month: str,
        results: List[TradeReviewResult],
    ) -> MonthlyFailureReport:
        total = len(results)
        counts: Dict[str, int] = {cat.value: 0 for cat in LossCategory}
        for r in results:
            counts[r.category.value] += 1

        pcts: Dict[str, float] = {}
        for cat_val, cnt in counts.items():
            pcts[cat_val] = round((cnt / total * 100), 2) if total > 0 else 0.0

        top_issue: Optional[str] = None
        if total > 0:
            top_issue = max(counts, key=lambda k: counts[k])
            # If all counts are 0 (shouldn't happen if total > 0, but guard)
            if counts[top_issue] == 0:
                top_issue = None

        systematic = False
        if total > 0 and top_issue:
            try:
                top_cat = LossCategory(top_issue)
                # Only flag systematic when a non-normal/non-uncategorized
                # category dominates the window
                if top_cat not in (
                    LossCategory.NORMAL_STATISTICAL,
                    LossCategory.UNCATEGORIZED,
                ):
                    systematic = self.flag_systematic_issue(
                        top_cat,
                        threshold=self._config.degradation_threshold,
                        window=max(total, self._config.degradation_window),
                    )
            except ValueError:
                pass

        action = (
            self.suggest_parameter_adjustment(
                MonthlyFailureReport(
                    month=month,
                    total_losses=total,
                    category_counts=counts,
                    category_percentages=pcts,
                    top_issue=top_issue,
                    recommended_action="",
                    systematic_issue_flag=systematic,
                )
            )
            if total > 0
            else "No losses recorded for this period."
        )

        return MonthlyFailureReport(
            month=month,
            total_losses=total,
            category_counts=counts,
            category_percentages=pcts,
            top_issue=top_issue,
            recommended_action=action,
            systematic_issue_flag=systematic,
        )

    def _recent_results(
        self,
        n: int,
        strategy_name: Optional[str] = None,
    ) -> List[TradeReviewResult]:
        results = (
            self._results
            if strategy_name is None
            else [r for r in self._results
                  if r.strategy_name.lower() == strategy_name.lower()]
        )
        return results[-n:] if n < len(results) else list(results)
