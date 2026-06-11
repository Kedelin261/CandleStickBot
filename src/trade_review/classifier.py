"""
M19 — Trade Review & Loss Classification
Mandatory post-trade analysis for all losses.
The Candlestick Trading Bible: Every loss is a lesson — classify it to improve.

Loss Categories (5 types):
  1. NORMAL_STATISTICAL_LOSS
     - Trade was correct per system rules, market didn't cooperate
     - No action required — part of normal variance
     - Expected: ~60% of losses

  2. SETUP_QUALITY_ERROR
     - Pattern quality below optimal, but above minimum
     - Consider raising quality thresholds
     - Flag for review

  3. REGIME_MISMATCH
     - Correct pattern, wrong market regime
     - Trend was ranging when we traded trending signal
     - Review regime detection sensitivity

  4. RULE_VIOLATION
     - Trade taken despite rule breach (manual override, config bug)
     - Highest priority review — system integrity compromised
     - Immediate investigation required

  5. SYSTEMATIC_ERROR
     - Multiple losses with same classification (>30% of losses in window)
     - Indicates edge erosion or strategy flaw
     - Full strategy review required

Alert Threshold:
  - If >30% of recent losses are same non-NORMAL category → systematic_error_alert

Phase 1 Scope:
  - Manual loss classification (review interface)
  - Automated category suggestion based on TQS components
  - Systematic error detection over rolling 20-trade window

Status: STUB — Phase 0 scaffold. Full implementation in Phase 1 Sprint 5.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("candlestickbot.trade_review.classifier")


class LossCategory(str, Enum):
    """Loss classification categories."""
    NORMAL_STATISTICAL_LOSS = "NORMAL_STATISTICAL_LOSS"
    SETUP_QUALITY_ERROR = "SETUP_QUALITY_ERROR"
    REGIME_MISMATCH = "REGIME_MISMATCH"
    RULE_VIOLATION = "RULE_VIOLATION"
    SYSTEMATIC_ERROR = "SYSTEMATIC_ERROR"


@dataclass
class LossClassification:
    """
    Classification result for a losing trade.
    Stored to TradeReview table in db/models.py.
    """
    trade_id: str
    strategy: str
    category: LossCategory
    auto_suggested: bool       # True if auto-classified, False if manual override
    pnl_r: float
    notes: str = ""

    # Context captured at time of trade
    tqs_total: int = 0
    tqs_trend: int = 0
    tqs_level: int = 0
    tqs_pattern: int = 0
    tqs_regime: int = 0
    regime: str = ""
    market_context: str = ""

    @property
    def requires_investigation(self) -> bool:
        """True for categories requiring manual investigation."""
        return self.category in (
            LossCategory.RULE_VIOLATION,
            LossCategory.SYSTEMATIC_ERROR,
        )


@dataclass
class SystematicErrorAlert:
    """Alert when systematic error pattern detected."""
    category: LossCategory
    occurrences: int
    total_losses_in_window: int
    percentage: float
    threshold: float
    window_size: int
    requires_review: bool = True

    @property
    def description(self) -> str:
        return (
            f"Systematic error: {self.category.value} appears in "
            f"{self.percentage:.1%} of last {self.window_size} losses "
            f"(threshold: {self.threshold:.1%})"
        )


class LossClassifier:
    """
    M19 — Trade Review Loss Classifier.

    Responsibilities:
    1. Suggest category for each loss based on TQS breakdown
    2. Store classifications in TradeReview table
    3. Monitor rolling window for systematic patterns
    4. Trigger alerts when systematic errors detected

    Auto-classification logic:
    - TQS regime score was 0: → REGIME_MISMATCH
    - TQS pattern score < 10 (low quality): → SETUP_QUALITY_ERROR
    - TQS trend score was 0 but trade was taken: → RULE_VIOLATION
    - Otherwise: → NORMAL_STATISTICAL_LOSS
    """

    SYSTEMATIC_ERROR_THRESHOLD = 0.30   # 30% of losses in same non-normal category
    REVIEW_WINDOW = 20                   # Rolling window for systematic detection

    def __init__(
        self,
        systematic_threshold: float = 0.30,
        review_window: int = 20,
        audit_logger=None,
        db_session=None,
    ):
        self.systematic_threshold = systematic_threshold
        self.review_window = review_window
        self.audit_logger = audit_logger
        self.db_session = db_session

    def suggest_category(
        self,
        tqs_trend: int,
        tqs_level: int,
        tqs_pattern: int,
        tqs_regime: int,
        pnl_r: float,
    ) -> Tuple[LossCategory, str]:
        """
        Auto-suggest loss category based on TQS component breakdown.

        Decision tree:
        1. If regime score == 0: REGIME_MISMATCH
        2. If trend score == 0: RULE_VIOLATION (trade shouldn't have been taken)
        3. If pattern score < 10: SETUP_QUALITY_ERROR (low quality setup)
        4. Otherwise: NORMAL_STATISTICAL_LOSS

        Args:
            tqs_trend: Trend component score (0-25)
            tqs_level: Level component score (0-25)
            tqs_pattern: Pattern component score (0-25)
            tqs_regime: Regime component score (0-25)
            pnl_r: Trade P&L in R-multiples (should be negative for a loss)

        Returns:
            Tuple of (LossCategory, explanation_string)
        """
        if tqs_regime == 0:
            return (
                LossCategory.REGIME_MISMATCH,
                f"Regime score was 0 — trade taken in unsuitable regime"
            )

        if tqs_trend == 0:
            return (
                LossCategory.RULE_VIOLATION,
                f"Trend score was 0 — trade should have been rejected by trend gate"
            )

        if tqs_pattern < 10:
            return (
                LossCategory.SETUP_QUALITY_ERROR,
                f"Pattern quality score {tqs_pattern}/25 indicates marginal setup"
            )

        return (
            LossCategory.NORMAL_STATISTICAL_LOSS,
            f"All gates passed (TQS: trend={tqs_trend}, level={tqs_level}, "
            f"pattern={tqs_pattern}, regime={tqs_regime}). Statistical variance."
        )

    def classify_loss(
        self,
        trade_id: str,
        strategy: str,
        tqs_trend: int,
        tqs_level: int,
        tqs_pattern: int,
        tqs_regime: int,
        pnl_r: float,
        manual_override: Optional[LossCategory] = None,
        notes: str = "",
    ) -> LossClassification:
        """
        Classify a losing trade and store the result.

        Args:
            trade_id: Unique trade identifier
            strategy: Strategy name
            tqs_trend/level/pattern/regime: TQS component scores
            pnl_r: Trade P&L in R-multiples
            manual_override: If provided, use this category instead of auto
            notes: Optional reviewer notes

        Returns:
            LossClassification record.
        """
        if manual_override:
            category = manual_override
            auto_suggested = False
            reason = f"Manual classification: {manual_override.value}"
        else:
            category, reason = self.suggest_category(
                tqs_trend, tqs_level, tqs_pattern, tqs_regime, pnl_r
            )
            auto_suggested = True

        classification = LossClassification(
            trade_id=trade_id,
            strategy=strategy,
            category=category,
            auto_suggested=auto_suggested,
            pnl_r=pnl_r,
            notes=notes or reason,
            tqs_total=tqs_trend + tqs_level + tqs_pattern + tqs_regime,
            tqs_trend=tqs_trend,
            tqs_level=tqs_level,
            tqs_pattern=tqs_pattern,
            tqs_regime=tqs_regime,
        )

        if self.audit_logger:
            self.audit_logger.log_loss_classified(
                trade_id=trade_id,
                strategy=strategy,
                category=category.value,
                pnl_r=pnl_r,
                context={
                    "tqs_total": classification.tqs_total,
                    "reason": reason,
                    "auto": auto_suggested,
                },
            )

        return classification

    def check_systematic_errors(
        self,
        recent_classifications: List[LossClassification],
    ) -> Optional[SystematicErrorAlert]:
        """
        Check for systematic error patterns in recent loss classifications.

        If any non-NORMAL category appears in >30% of the review window,
        trigger a systematic error alert.

        Args:
            recent_classifications: Last N loss classifications (time-ordered)

        Returns:
            SystematicErrorAlert if systematic pattern detected, None if normal.
        """
        window = recent_classifications[-self.review_window:]
        if len(window) < 5:  # Need at least 5 losses for meaningful analysis
            return None

        # Count non-normal categories
        category_counts: Dict[LossCategory, int] = {}
        for c in window:
            if c.category != LossCategory.NORMAL_STATISTICAL_LOSS:
                category_counts[c.category] = category_counts.get(c.category, 0) + 1

        total = len(window)
        for category, count in category_counts.items():
            pct = count / total
            if pct >= self.systematic_threshold:
                alert = SystematicErrorAlert(
                    category=category,
                    occurrences=count,
                    total_losses_in_window=total,
                    percentage=pct,
                    threshold=self.systematic_threshold,
                    window_size=self.review_window,
                )

                if self.audit_logger:
                    self.audit_logger.log_systematic_error_alert(
                        category=category.value,
                        pct_of_losses=pct,
                        threshold=self.systematic_threshold,
                    )

                return alert

        return None
