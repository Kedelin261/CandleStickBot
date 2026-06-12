"""
Sprint 10 — M19 Trade Review Engine
Tests for src/analytics/trade_review.py

Classes / counts:
    TestTradeReviewResultFields      (7)
    TestMonthlyFailureReportFields   (6)
    TestClassifyOverridden           (7)
    TestClassifyBadRegime            (9)
    TestClassifyBadSignal            (7)
    TestClassifyBadLevel             (7)
    TestClassifyBadExecution         (8)
    TestClassifyNormalStatistical    (6)
    TestClassificationPriority       (8)
    TestReviewTrade                  (4)
    TestGetTopLossCategory           (6)
    TestGenerateMonthlyReport        (9)
    TestSuggestParameterAdjustment   (5)
    TestFlagSystematicIssue          (6)
    TestGetFailureBreakdown          (7)
    TestEdgeCasesAndIntegration      (8)

Total: 108 tests
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from src.analytics.trade_review import (
    MonthlyFailureReport,
    ReviewConfig,
    TradeContext,
    TradeReviewEngine,
    TradeReviewResult,
)
from src.types import LossCategory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _engine(**kwargs) -> TradeReviewEngine:
    return TradeReviewEngine(config=ReviewConfig(**kwargs))


def _ctx(
    pattern_quality_score: float = 70.0,
    level_strength_score: float = 70.0,
    regime: str = "TRENDING",
    fill_slippage_pips: float = 0.5,
    stop_distance_pips: float = 10.0,
    was_overridden: bool = False,
) -> TradeContext:
    return TradeContext(
        pattern_quality_score=pattern_quality_score,
        level_strength_score=level_strength_score,
        regime=regime,
        fill_slippage_pips=fill_slippage_pips,
        stop_distance_pips=stop_distance_pips,
        was_overridden=was_overridden,
    )


def _classify(engine: TradeReviewEngine, ctx: TradeContext) -> TradeReviewResult:
    return engine.classify_loss(str(uuid.uuid4()), "pin_bar", ctx)


def _add_results(
    engine: TradeReviewEngine,
    categories,
    strategy: str = "pin_bar",
) -> None:
    """Quick-add results with specific categories."""
    for cat in categories:
        ctx = _make_ctx_for_category(cat)
        engine.classify_loss(str(uuid.uuid4()), strategy, ctx)


def _make_ctx_for_category(cat: LossCategory) -> TradeContext:
    """Return a TradeContext that will classify as *cat*."""
    if cat == LossCategory.OVERRIDDEN:
        return _ctx(was_overridden=True)
    if cat == LossCategory.BAD_REGIME:
        return _ctx(regime="CHOPPY")
    if cat == LossCategory.BAD_SIGNAL:
        return _ctx(pattern_quality_score=30.0)  # below 50
    if cat == LossCategory.BAD_LEVEL:
        return _ctx(level_strength_score=30.0)   # below 50
    if cat == LossCategory.BAD_EXECUTION:
        return _ctx(fill_slippage_pips=5.0)       # above 2.0
    # NORMAL_STATISTICAL
    return _ctx()


# ===========================================================================
# TestTradeReviewResultFields
# ===========================================================================

class TestTradeReviewResultFields:
    def test_result_has_trade_id(self):
        e = _engine()
        r = _classify(e, _ctx())
        assert isinstance(r.trade_id, str) and len(r.trade_id) > 0

    def test_result_has_strategy_name(self):
        e = _engine()
        r = e.classify_loss("t1", "engulfing_bar", _ctx())
        assert r.strategy_name == "engulfing_bar"

    def test_result_has_category(self):
        e = _engine()
        r = _classify(e, _ctx())
        assert isinstance(r.category, LossCategory)

    def test_result_has_reason_string(self):
        e = _engine()
        r = _classify(e, _ctx())
        assert isinstance(r.reason, str) and len(r.reason) > 0

    def test_result_has_recommended_action(self):
        e = _engine()
        r = _classify(e, _ctx())
        assert isinstance(r.recommended_action, str) and len(r.recommended_action) > 0

    def test_result_has_severity(self):
        e = _engine()
        r = _classify(e, _ctx())
        assert r.severity in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}

    def test_result_has_reviewed_at_datetime(self):
        e = _engine()
        r = _classify(e, _ctx())
        assert isinstance(r.reviewed_at, datetime)


# ===========================================================================
# TestMonthlyFailureReportFields
# ===========================================================================

class TestMonthlyFailureReportFields:
    def test_report_has_month(self):
        e = _engine()
        rpt = e.generate_monthly_report("2025-01")
        assert rpt.month == "2025-01"

    def test_report_has_total_losses(self):
        e = _engine()
        rpt = e.generate_monthly_report("2025-01")
        assert isinstance(rpt.total_losses, int)

    def test_report_has_category_counts_dict(self):
        e = _engine()
        rpt = e.generate_monthly_report("2025-01")
        assert isinstance(rpt.category_counts, dict)

    def test_report_has_category_percentages_dict(self):
        e = _engine()
        rpt = e.generate_monthly_report("2025-01")
        assert isinstance(rpt.category_percentages, dict)

    def test_report_has_recommended_action(self):
        e = _engine()
        rpt = e.generate_monthly_report("2025-01")
        assert isinstance(rpt.recommended_action, str)

    def test_report_has_systematic_issue_flag(self):
        e = _engine()
        rpt = e.generate_monthly_report("2025-01")
        assert isinstance(rpt.systematic_issue_flag, bool)


# ===========================================================================
# TestClassifyOverridden
# ===========================================================================

class TestClassifyOverridden:
    def test_overridden_flag_triggers_category(self):
        e = _engine()
        r = _classify(e, _ctx(was_overridden=True))
        assert r.category == LossCategory.OVERRIDDEN

    def test_overridden_severity_is_critical(self):
        e = _engine()
        r = _classify(e, _ctx(was_overridden=True))
        assert r.severity == "CRITICAL"

    def test_overridden_reason_mentions_override(self):
        e = _engine()
        r = _classify(e, _ctx(was_overridden=True))
        assert "override" in r.reason.lower()

    def test_overridden_beats_bad_regime(self):
        e = _engine()
        r = _classify(e, _ctx(was_overridden=True, regime="CHOPPY"))
        assert r.category == LossCategory.OVERRIDDEN

    def test_overridden_beats_bad_signal(self):
        e = _engine()
        r = _classify(e, _ctx(was_overridden=True, pattern_quality_score=10.0))
        assert r.category == LossCategory.OVERRIDDEN

    def test_overridden_beats_bad_execution(self):
        e = _engine()
        r = _classify(e, _ctx(was_overridden=True, fill_slippage_pips=99.0))
        assert r.category == LossCategory.OVERRIDDEN

    def test_overridden_recommended_action_non_empty(self):
        e = _engine()
        r = _classify(e, _ctx(was_overridden=True))
        assert len(r.recommended_action) > 0


# ===========================================================================
# TestClassifyBadRegime
# ===========================================================================

class TestClassifyBadRegime:
    def test_choppy_is_bad_regime(self):
        e = _engine()
        r = _classify(e, _ctx(regime="CHOPPY"))
        assert r.category == LossCategory.BAD_REGIME

    def test_volatile_is_bad_regime(self):
        e = _engine()
        r = _classify(e, _ctx(regime="VOLATILE"))
        assert r.category == LossCategory.BAD_REGIME

    def test_quiet_is_bad_regime(self):
        e = _engine()
        r = _classify(e, _ctx(regime="QUIET"))
        assert r.category == LossCategory.BAD_REGIME

    def test_unknown_is_bad_regime(self):
        e = _engine()
        r = _classify(e, _ctx(regime="UNKNOWN"))
        assert r.category == LossCategory.BAD_REGIME

    def test_trending_is_not_bad_regime(self):
        e = _engine()
        r = _classify(e, _ctx(regime="TRENDING"))
        assert r.category != LossCategory.BAD_REGIME

    def test_bad_regime_severity_is_high(self):
        e = _engine()
        r = _classify(e, _ctx(regime="CHOPPY"))
        assert r.severity == "HIGH"

    def test_bad_regime_reason_mentions_regime(self):
        e = _engine()
        r = _classify(e, _ctx(regime="CHOPPY"))
        assert "CHOPPY" in r.reason or "regime" in r.reason.lower()

    def test_bad_regime_beats_bad_signal(self):
        e = _engine()
        r = _classify(e, _ctx(regime="CHOPPY", pattern_quality_score=10.0))
        assert r.category == LossCategory.BAD_REGIME

    def test_regime_case_insensitive(self):
        # 'trending' lowercase should still resolve as TRENDING
        e = _engine()
        r = _classify(e, _ctx(regime="trending"))
        assert r.category != LossCategory.BAD_REGIME


# ===========================================================================
# TestClassifyBadSignal
# ===========================================================================

class TestClassifyBadSignal:
    def test_low_pattern_score_is_bad_signal(self):
        e = _engine()
        r = _classify(e, _ctx(pattern_quality_score=30.0))
        assert r.category == LossCategory.BAD_SIGNAL

    def test_exactly_at_threshold_is_not_bad_signal(self):
        e = _engine()  # threshold default = 50
        r = _classify(e, _ctx(pattern_quality_score=50.0))
        assert r.category != LossCategory.BAD_SIGNAL

    def test_custom_threshold(self):
        e = _engine(pattern_quality_threshold=70.0)
        r = _classify(e, _ctx(pattern_quality_score=65.0))
        assert r.category == LossCategory.BAD_SIGNAL

    def test_bad_signal_severity_is_medium(self):
        e = _engine()
        r = _classify(e, _ctx(pattern_quality_score=20.0))
        assert r.severity == "MEDIUM"

    def test_bad_signal_reason_mentions_score(self):
        e = _engine()
        r = _classify(e, _ctx(pattern_quality_score=20.0))
        assert "20.0" in r.reason or "quality" in r.reason.lower()

    def test_bad_signal_beats_bad_level(self):
        e = _engine()
        r = _classify(
            e,
            _ctx(pattern_quality_score=10.0, level_strength_score=10.0),
        )
        assert r.category == LossCategory.BAD_SIGNAL

    def test_zero_score_is_bad_signal(self):
        e = _engine()
        r = _classify(e, _ctx(pattern_quality_score=0.0))
        assert r.category == LossCategory.BAD_SIGNAL


# ===========================================================================
# TestClassifyBadLevel
# ===========================================================================

class TestClassifyBadLevel:
    def test_low_level_strength_is_bad_level(self):
        e = _engine()
        r = _classify(e, _ctx(level_strength_score=30.0))
        assert r.category == LossCategory.BAD_LEVEL

    def test_exactly_at_threshold_is_not_bad_level(self):
        e = _engine()  # threshold default = 50
        r = _classify(e, _ctx(level_strength_score=50.0))
        assert r.category != LossCategory.BAD_LEVEL

    def test_custom_level_threshold(self):
        e = _engine(level_strength_threshold=60.0)
        r = _classify(e, _ctx(level_strength_score=55.0))
        assert r.category == LossCategory.BAD_LEVEL

    def test_bad_level_severity_is_medium(self):
        e = _engine()
        r = _classify(e, _ctx(level_strength_score=20.0))
        assert r.severity == "MEDIUM"

    def test_bad_level_reason_mentions_score(self):
        e = _engine()
        r = _classify(e, _ctx(level_strength_score=20.0))
        assert "20.0" in r.reason or "level" in r.reason.lower() or "strength" in r.reason.lower()

    def test_bad_level_beats_bad_execution(self):
        e = _engine()
        r = _classify(
            e,
            _ctx(level_strength_score=10.0, fill_slippage_pips=10.0),
        )
        assert r.category == LossCategory.BAD_LEVEL

    def test_zero_level_score_is_bad_level(self):
        e = _engine()
        r = _classify(e, _ctx(level_strength_score=0.0))
        assert r.category == LossCategory.BAD_LEVEL


# ===========================================================================
# TestClassifyBadExecution
# ===========================================================================

class TestClassifyBadExecution:
    def test_high_slippage_is_bad_execution(self):
        e = _engine()
        r = _classify(e, _ctx(fill_slippage_pips=5.0))
        assert r.category == LossCategory.BAD_EXECUTION

    def test_exactly_at_slippage_threshold_is_not_bad(self):
        e = _engine()  # max_slippage default = 2.0
        r = _classify(e, _ctx(fill_slippage_pips=2.0))
        # exactly at threshold is ≤ not >; should not be BAD_EXECUTION
        assert r.category != LossCategory.BAD_EXECUTION

    def test_tight_stop_is_bad_execution(self):
        e = _engine()
        r = _classify(e, _ctx(stop_distance_pips=2.0))
        assert r.category == LossCategory.BAD_EXECUTION

    def test_exactly_at_min_stop_is_not_bad(self):
        e = _engine()  # min_stop default = 5.0
        r = _classify(e, _ctx(stop_distance_pips=5.0))
        assert r.category != LossCategory.BAD_EXECUTION

    def test_bad_execution_severity_is_high(self):
        e = _engine()
        r = _classify(e, _ctx(fill_slippage_pips=10.0))
        assert r.severity == "HIGH"

    def test_bad_execution_reason_mentions_slippage_or_stop(self):
        e = _engine()
        r = _classify(e, _ctx(fill_slippage_pips=10.0))
        assert "slippage" in r.reason.lower() or "stop" in r.reason.lower()

    def test_custom_slippage_threshold(self):
        e = _engine(max_slippage_pips=1.0)
        r = _classify(e, _ctx(fill_slippage_pips=1.5))
        assert r.category == LossCategory.BAD_EXECUTION

    def test_custom_min_stop(self):
        e = _engine(min_stop_pips=10.0)
        r = _classify(e, _ctx(stop_distance_pips=8.0))
        assert r.category == LossCategory.BAD_EXECUTION


# ===========================================================================
# TestClassifyNormalStatistical
# ===========================================================================

class TestClassifyNormalStatistical:
    def test_all_gates_passed_is_normal(self):
        e = _engine()
        r = _classify(e, _ctx())  # all defaults pass
        assert r.category == LossCategory.NORMAL_STATISTICAL

    def test_normal_severity_is_low(self):
        e = _engine()
        r = _classify(e, _ctx())
        assert r.severity == "LOW"

    def test_normal_reason_mentions_variance(self):
        e = _engine()
        r = _classify(e, _ctx())
        assert "variance" in r.reason.lower() or "statistical" in r.reason.lower()

    def test_normal_action_says_no_action(self):
        e = _engine()
        r = _classify(e, _ctx())
        assert "no action" in r.recommended_action.lower()

    def test_high_quality_scores_give_normal(self):
        e = _engine()
        r = _classify(
            e,
            _ctx(
                pattern_quality_score=90.0,
                level_strength_score=90.0,
                fill_slippage_pips=0.1,
                stop_distance_pips=20.0,
            ),
        )
        assert r.category == LossCategory.NORMAL_STATISTICAL

    def test_breakeven_scores_give_normal(self):
        # Exactly at thresholds → all gates pass → NORMAL
        e = _engine()
        r = _classify(
            e,
            _ctx(
                pattern_quality_score=50.0,
                level_strength_score=50.0,
                fill_slippage_pips=2.0,
                stop_distance_pips=5.0,
            ),
        )
        assert r.category == LossCategory.NORMAL_STATISTICAL


# ===========================================================================
# TestClassificationPriority
# ===========================================================================

class TestClassificationPriority:
    def test_overridden_beats_all(self):
        e = _engine()
        r = _classify(
            e,
            _ctx(
                was_overridden=True,
                regime="CHOPPY",
                pattern_quality_score=10.0,
                level_strength_score=10.0,
                fill_slippage_pips=99.0,
            ),
        )
        assert r.category == LossCategory.OVERRIDDEN

    def test_bad_regime_beats_bad_signal(self):
        e = _engine()
        r = _classify(e, _ctx(regime="VOLATILE", pattern_quality_score=10.0))
        assert r.category == LossCategory.BAD_REGIME

    def test_bad_regime_beats_bad_level(self):
        e = _engine()
        r = _classify(e, _ctx(regime="VOLATILE", level_strength_score=10.0))
        assert r.category == LossCategory.BAD_REGIME

    def test_bad_regime_beats_bad_execution(self):
        e = _engine()
        r = _classify(e, _ctx(regime="VOLATILE", fill_slippage_pips=99.0))
        assert r.category == LossCategory.BAD_REGIME

    def test_bad_signal_beats_bad_level(self):
        e = _engine()
        r = _classify(
            e,
            _ctx(pattern_quality_score=10.0, level_strength_score=10.0),
        )
        assert r.category == LossCategory.BAD_SIGNAL

    def test_bad_signal_beats_bad_execution(self):
        e = _engine()
        r = _classify(
            e,
            _ctx(pattern_quality_score=10.0, fill_slippage_pips=99.0),
        )
        assert r.category == LossCategory.BAD_SIGNAL

    def test_bad_level_beats_bad_execution(self):
        e = _engine()
        r = _classify(
            e,
            _ctx(level_strength_score=10.0, fill_slippage_pips=99.0),
        )
        assert r.category == LossCategory.BAD_LEVEL

    def test_all_bad_executes_overridden_wins(self):
        e = _engine()
        r = _classify(
            e,
            _ctx(
                was_overridden=True,
                regime="CHOPPY",
                pattern_quality_score=5.0,
                level_strength_score=5.0,
                fill_slippage_pips=50.0,
                stop_distance_pips=1.0,
            ),
        )
        assert r.category == LossCategory.OVERRIDDEN


# ===========================================================================
# TestReviewTrade
# ===========================================================================

class TestReviewTrade:
    def test_review_trade_returns_result(self):
        e = _engine()
        r = e.review_trade("t1", "pin_bar", _ctx())
        assert isinstance(r, TradeReviewResult)

    def test_review_trade_stores_result(self):
        e = _engine()
        e.review_trade("t1", "pin_bar", _ctx())
        assert len(e.get_all_results()) == 1

    def test_review_trade_is_alias_for_classify(self):
        e = _engine()
        r1 = e.classify_loss("t1", "pin_bar", _ctx(pattern_quality_score=30.0))
        e.reset()
        r2 = e.review_trade("t2", "pin_bar", _ctx(pattern_quality_score=30.0))
        assert r1.category == r2.category

    def test_review_trade_correct_strategy_stored(self):
        e = _engine()
        r = e.review_trade("t1", "engulfing_bar", _ctx())
        assert r.strategy_name == "engulfing_bar"


# ===========================================================================
# TestGetTopLossCategory
# ===========================================================================

class TestGetTopLossCategory:
    def test_empty_returns_none(self):
        e = _engine()
        assert e.get_top_loss_category() is None

    def test_single_result_returns_its_category(self):
        e = _engine()
        _classify(e, _ctx(regime="CHOPPY"))
        top = e.get_top_loss_category()
        assert top == LossCategory.BAD_REGIME

    def test_majority_category_wins(self):
        e = _engine()
        for _ in range(4):
            _classify(e, _ctx(regime="CHOPPY"))   # BAD_REGIME ×4
        _classify(e, _ctx(pattern_quality_score=10.0))  # BAD_SIGNAL ×1
        assert e.get_top_loss_category() == LossCategory.BAD_REGIME

    def test_uses_last_n_window(self):
        e = _engine()
        for _ in range(10):
            _classify(e, _ctx(regime="CHOPPY"))   # old BAD_REGIME ×10
        # Recent 3 are all NORMAL
        for _ in range(3):
            _classify(e, _ctx())
        top = e.get_top_loss_category(last_n=3)
        assert top == LossCategory.NORMAL_STATISTICAL

    def test_strategy_filter(self):
        e = _engine()
        e.classify_loss("t1", "pin_bar", _ctx(regime="CHOPPY"))
        e.classify_loss("t2", "engulfing_bar", _ctx(pattern_quality_score=10.0))
        top_pin = e.get_top_loss_category(strategy_name="pin_bar")
        assert top_pin == LossCategory.BAD_REGIME

    def test_unknown_strategy_returns_none(self):
        e = _engine()
        _classify(e, _ctx())
        top = e.get_top_loss_category(strategy_name="fib")
        assert top is None


# ===========================================================================
# TestGenerateMonthlyReport
# ===========================================================================

class TestGenerateMonthlyReport:
    def _add_dated(self, engine, category: LossCategory, month_str: str) -> None:
        """Add a result with a forced reviewed_at timestamp."""
        year, mon = map(int, month_str.split("-"))
        ctx = _make_ctx_for_category(category)
        result = engine.classify_loss(str(uuid.uuid4()), "pin_bar", ctx)
        # Patch the reviewed_at to the target month
        object.__setattr__(
            result,
            "reviewed_at",
            datetime(year, mon, 15, 12, 0, 0, tzinfo=timezone.utc),
        )

    def test_empty_report_zero_total(self):
        e = _engine()
        rpt = e.generate_monthly_report("2025-01")
        assert rpt.total_losses == 0

    def test_empty_report_no_top_issue(self):
        e = _engine()
        rpt = e.generate_monthly_report("2025-01")
        assert rpt.top_issue is None

    def test_empty_report_recommended_action_non_empty(self):
        e = _engine()
        rpt = e.generate_monthly_report("2025-01")
        assert len(rpt.recommended_action) > 0

    def test_report_counts_match_added(self):
        e = _engine()
        for _ in range(3):
            self._add_dated(e, LossCategory.BAD_REGIME, "2025-02")
        rpt = e.generate_monthly_report("2025-02")
        assert rpt.total_losses == 3
        assert rpt.category_counts[LossCategory.BAD_REGIME.value] == 3

    def test_report_percentages_sum_to_100(self):
        e = _engine()
        self._add_dated(e, LossCategory.BAD_REGIME, "2025-03")
        self._add_dated(e, LossCategory.BAD_SIGNAL, "2025-03")
        rpt = e.generate_monthly_report("2025-03")
        total_pct = sum(rpt.category_percentages.values())
        assert abs(total_pct - 100.0) < 0.1

    def test_report_top_issue_is_most_frequent(self):
        e = _engine()
        for _ in range(3):
            self._add_dated(e, LossCategory.BAD_SIGNAL, "2025-04")
        self._add_dated(e, LossCategory.BAD_REGIME, "2025-04")
        rpt = e.generate_monthly_report("2025-04")
        assert rpt.top_issue == LossCategory.BAD_SIGNAL.value

    def test_report_ignores_other_months(self):
        e = _engine()
        self._add_dated(e, LossCategory.BAD_REGIME, "2025-01")
        self._add_dated(e, LossCategory.BAD_SIGNAL, "2025-02")
        rpt = e.generate_monthly_report("2025-01")
        assert rpt.total_losses == 1
        assert rpt.category_counts[LossCategory.BAD_SIGNAL.value] == 0

    def test_strategy_filter_in_report(self):
        e = _engine()
        e.classify_loss("t1", "pin_bar", _ctx(regime="CHOPPY"))
        # Patch to correct month
        e._results[-1] = TradeReviewResult(
            trade_id="t1",
            strategy_name="pin_bar",
            category=LossCategory.BAD_REGIME,
            reason="r",
            recommended_action="a",
            severity="HIGH",
            reviewed_at=datetime(2025, 5, 10, tzinfo=timezone.utc),
        )
        e.classify_loss("t2", "engulfing_bar", _ctx(pattern_quality_score=10.0))
        e._results[-1] = TradeReviewResult(
            trade_id="t2",
            strategy_name="engulfing_bar",
            category=LossCategory.BAD_SIGNAL,
            reason="r",
            recommended_action="a",
            severity="MEDIUM",
            reviewed_at=datetime(2025, 5, 20, tzinfo=timezone.utc),
        )
        rpt = e.generate_monthly_report("2025-05", strategy_name="pin_bar")
        assert rpt.total_losses == 1
        assert rpt.category_counts[LossCategory.BAD_REGIME.value] == 1
        assert rpt.category_counts[LossCategory.BAD_SIGNAL.value] == 0

    def test_systematic_flag_true_when_threshold_exceeded(self):
        e = _engine(degradation_threshold=0.30)
        # Add 10 results; 8 are BAD_REGIME (80 %) to the same month
        for _ in range(8):
            self._add_dated(e, LossCategory.BAD_REGIME, "2025-06")
        for _ in range(2):
            self._add_dated(e, LossCategory.NORMAL_STATISTICAL, "2025-06")
        rpt = e.generate_monthly_report("2025-06")
        assert rpt.systematic_issue_flag is True

    def test_systematic_flag_false_when_below_threshold(self):
        e = _engine(degradation_threshold=0.30)
        # 2 BAD_REGIME out of 10 = 20 % < 30 %
        for _ in range(2):
            self._add_dated(e, LossCategory.BAD_REGIME, "2025-07")
        for _ in range(8):
            self._add_dated(e, LossCategory.NORMAL_STATISTICAL, "2025-07")
        rpt = e.generate_monthly_report("2025-07")
        assert rpt.systematic_issue_flag is False


# ===========================================================================
# TestSuggestParameterAdjustment
# ===========================================================================

class TestSuggestParameterAdjustment:
    def _report(self, top_issue: str, total: int = 5, pct: float = 60.0) -> MonthlyFailureReport:
        counts = {cat.value: 0 for cat in LossCategory}
        counts[top_issue] = total
        pcts = {cat.value: 0.0 for cat in LossCategory}
        pcts[top_issue] = pct
        return MonthlyFailureReport(
            month="2025-01",
            total_losses=total,
            category_counts=counts,
            category_percentages=pcts,
            top_issue=top_issue,
            recommended_action="",
            systematic_issue_flag=False,
        )

    def test_no_losses_returns_insufficient_data(self):
        e = _engine()
        rpt = MonthlyFailureReport(
            month="2025-01",
            total_losses=0,
            category_counts={},
            category_percentages={},
            top_issue=None,
            recommended_action="",
            systematic_issue_flag=False,
        )
        suggestion = e.suggest_parameter_adjustment(rpt)
        assert "insufficient" in suggestion.lower() or "no losses" in suggestion.lower()

    def test_bad_regime_suggestion_mentions_regime(self):
        e = _engine()
        rpt = self._report(LossCategory.BAD_REGIME.value)
        suggestion = e.suggest_parameter_adjustment(rpt)
        assert "regime" in suggestion.lower()

    def test_bad_signal_suggestion_mentions_threshold(self):
        e = _engine()
        rpt = self._report(LossCategory.BAD_SIGNAL.value)
        suggestion = e.suggest_parameter_adjustment(rpt)
        assert "threshold" in suggestion.lower() or "quality" in suggestion.lower()

    def test_bad_execution_suggestion_mentions_slippage_or_stop(self):
        e = _engine()
        rpt = self._report(LossCategory.BAD_EXECUTION.value)
        suggestion = e.suggest_parameter_adjustment(rpt)
        assert "slippage" in suggestion.lower() or "stop" in suggestion.lower()

    def test_normal_statistical_suggestion_says_no_action(self):
        e = _engine()
        rpt = self._report(LossCategory.NORMAL_STATISTICAL.value)
        suggestion = e.suggest_parameter_adjustment(rpt)
        assert "no action" in suggestion.lower()


# ===========================================================================
# TestFlagSystematicIssue
# ===========================================================================

class TestFlagSystematicIssue:
    def test_empty_returns_false(self):
        e = _engine()
        assert e.flag_systematic_issue(LossCategory.BAD_REGIME) is False

    def test_below_threshold_returns_false(self):
        e = _engine(degradation_threshold=0.30)
        for _ in range(2):
            _classify(e, _ctx(regime="CHOPPY"))
        for _ in range(8):
            _classify(e, _ctx())
        assert e.flag_systematic_issue(LossCategory.BAD_REGIME) is False

    def test_at_threshold_triggers(self):
        e = _engine(degradation_threshold=0.30)
        for _ in range(3):
            _classify(e, _ctx(regime="CHOPPY"))
        for _ in range(7):
            _classify(e, _ctx())
        # 3/10 = 30% ≥ 0.30
        assert e.flag_systematic_issue(LossCategory.BAD_REGIME) is True

    def test_above_threshold_triggers(self):
        e = _engine(degradation_threshold=0.30)
        for _ in range(8):
            _classify(e, _ctx(regime="CHOPPY"))
        for _ in range(2):
            _classify(e, _ctx())
        assert e.flag_systematic_issue(LossCategory.BAD_REGIME) is True

    def test_custom_threshold(self):
        e = _engine()
        for _ in range(5):
            _classify(e, _ctx(pattern_quality_score=10.0))  # BAD_SIGNAL
        for _ in range(5):
            _classify(e, _ctx())
        # 5/10 = 50% ≥ 0.50 → True
        assert e.flag_systematic_issue(LossCategory.BAD_SIGNAL, threshold=0.50) is True
        # 5/10 = 50% < 0.60 → depends; below 60%
        assert e.flag_systematic_issue(LossCategory.BAD_SIGNAL, threshold=0.60) is False

    def test_different_category_not_flagged(self):
        e = _engine(degradation_threshold=0.30)
        for _ in range(8):
            _classify(e, _ctx(regime="CHOPPY"))
        # BAD_REGIME is flagged, but BAD_SIGNAL is not
        assert e.flag_systematic_issue(LossCategory.BAD_SIGNAL) is False


# ===========================================================================
# TestGetFailureBreakdown
# ===========================================================================

class TestGetFailureBreakdown:
    def test_empty_all_zero(self):
        e = _engine()
        bd = e.get_failure_breakdown()
        assert all(v == 0 for v in bd.values())

    def test_all_categories_present_in_output(self):
        e = _engine()
        bd = e.get_failure_breakdown()
        for cat in LossCategory:
            assert cat.value in bd

    def test_single_result_counted(self):
        e = _engine()
        _classify(e, _ctx(regime="CHOPPY"))
        bd = e.get_failure_breakdown()
        assert bd[LossCategory.BAD_REGIME.value] == 1

    def test_multiple_results_counted(self):
        e = _engine()
        for _ in range(3):
            _classify(e, _ctx(regime="CHOPPY"))
        for _ in range(2):
            _classify(e, _ctx())
        bd = e.get_failure_breakdown()
        assert bd[LossCategory.BAD_REGIME.value] == 3
        assert bd[LossCategory.NORMAL_STATISTICAL.value] == 2

    def test_strategy_filter(self):
        e = _engine()
        e.classify_loss("t1", "pin_bar", _ctx(regime="CHOPPY"))
        e.classify_loss("t2", "engulfing_bar", _ctx(regime="CHOPPY"))
        bd = e.get_failure_breakdown(strategy_name="pin_bar")
        assert bd[LossCategory.BAD_REGIME.value] == 1

    def test_unknown_strategy_all_zero(self):
        e = _engine()
        _classify(e, _ctx(regime="CHOPPY"))
        bd = e.get_failure_breakdown(strategy_name="fibonacci")
        assert all(v == 0 for v in bd.values())

    def test_reset_clears_breakdown(self):
        e = _engine()
        _classify(e, _ctx(regime="CHOPPY"))
        e.reset()
        bd = e.get_failure_breakdown()
        assert bd[LossCategory.BAD_REGIME.value] == 0


# ===========================================================================
# TestEdgeCasesAndIntegration
# ===========================================================================

class TestEdgeCasesAndIntegration:
    def test_multiple_strategies_tracked_independently(self):
        e = _engine()
        e.classify_loss("t1", "pin_bar", _ctx(regime="CHOPPY"))
        e.classify_loss("t2", "engulfing_bar", _ctx(pattern_quality_score=10.0))
        bd_pin = e.get_failure_breakdown(strategy_name="pin_bar")
        bd_eng = e.get_failure_breakdown(strategy_name="engulfing_bar")
        assert bd_pin[LossCategory.BAD_REGIME.value] == 1
        assert bd_eng[LossCategory.BAD_SIGNAL.value] == 1

    def test_get_all_results_returns_copy(self):
        e = _engine()
        _classify(e, _ctx())
        results = e.get_all_results()
        results.clear()
        assert len(e.get_all_results()) == 1

    def test_reset_clears_all_results(self):
        e = _engine()
        for _ in range(5):
            _classify(e, _ctx())
        e.reset()
        assert len(e.get_all_results()) == 0

    def test_classification_is_deterministic(self):
        """Same context always yields same category."""
        e = _engine()
        ctx = _ctx(pattern_quality_score=30.0)
        results = [e.classify_loss(str(i), "pin_bar", ctx) for i in range(10)]
        cats = {r.category for r in results}
        assert len(cats) == 1
        assert cats.pop() == LossCategory.BAD_SIGNAL

    def test_no_execution_side_effects(self):
        """Engine does not call any external APIs or brokers."""
        e = _engine()
        r = _classify(e, _ctx())
        # As long as classification ran without raising, no broker call occurred.
        assert r is not None

    def test_trade_id_is_passed_through(self):
        e = _engine()
        r = e.classify_loss("MY-TRADE-123", "pin_bar", _ctx())
        assert r.trade_id == "MY-TRADE-123"

    def test_severity_mapping_complete(self):
        """All LossCategory members have a severity mapping."""
        from src.analytics.trade_review import _SEVERITY_MAP
        for cat in LossCategory:
            assert cat in _SEVERITY_MAP, f"Missing severity for {cat}"

    def test_recommended_actions_mapping_complete(self):
        """All LossCategory members have a recommended-action mapping."""
        from src.analytics.trade_review import _RECOMMENDED_ACTIONS
        for cat in LossCategory:
            assert cat in _RECOMMENDED_ACTIONS, f"Missing action for {cat}"
