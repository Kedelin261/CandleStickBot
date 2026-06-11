"""
M08 — Strategy Engine (Phase 1 MVP)
Coordinates M04 Trend, M05 S/R, M16 Market Regime, M07 Pattern Recognition
into a single gated trade recommendation pipeline.

The Candlestick Trading Bible: The strategy is the integration of all signals.

=== PHASE 1 STRATEGIES ===

  PIN_BAR      — Bullish or Bearish Pin Bar (M07)
  ENGULFING_BAR — Bullish or Bearish Engulfing Bar (M07)

  NOT implemented: Inside Bar, False Breakout, Fibonacci, Supply/Demand
  NOT implemented: Risk Engine, Position Sizing, Order Execution

=== GATE CHAIN (ordered, each can short-circuit) ===

  1. SIGNAL GATE   — a qualifying Phase 1 pattern must exist on the signal bar
  2. TREND GATE    — trade direction must align with M04 confirmed trend
  3. REGIME GATE   — regime must be TRENDING (RANGING rejected in Phase 1)
  4. LEVEL GATE    — signal bar must be within level_tolerance of an S/R level
  5. TQS GATE      — composite TQS score must reach min_tqs_score (default 60)
  6. RR GATE       — risk/reward ratio must be >= min_rr_ratio (default 2.0)

=== TQS COMPONENTS (0–25 each, total 0–100) ===

  Trend score   — delegated to M04 TrendAnalysis.tqs_trend_score
  Level score   — delegated to M05 SREngine.calculate_tqs_level_score()
  Pattern score — delegated to M07 PatternEngine.calculate_tqs_pattern_score()
  Regime score  — delegated to M16 RegimeAnalysis.tqs_regime_score

=== TQS TIERS ===

  REJECT:   TQS < 60   — no trade
  STANDARD: TQS 60–79  — trade at standard risk
  PREMIUM:  TQS >= 80  — eligible for premium risk (if enabled, default off)

=== ENTRY / STOP / TARGET ===

  Entry (aggressive, Phase 1 default):
    Bullish: candle.high  (break above pin bar / engulfing top)
    Bearish: candle.low   (break below pin bar / engulfing bottom)

  Stop:
    Bullish Pin Bar:      candle.low  - buffer_pips
    Bearish Pin Bar:      candle.high + buffer_pips
    Bullish Engulfing:    candle.low  - buffer_pips
    Bearish Engulfing:    candle.high + buffer_pips

  Take Profit (cascading priority):
    1. Next valid S/R level in trade direction within 10R window
    2. Fallback: entry ± risk_distance * 2.0 (guarantees 2R)

=== OUTPUT ===

  TradeRecommendation (src.types) — all gates passed
  EvaluationResult               — full audit trail per candle

Status: Full Phase 1 implementation — Sprint 7.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional, Union

from src.analysis.market_regime import MarketRegimeEngine, RegimeAnalysis
from src.analysis.sr_engine import SRAnalysis, SREngine, SRLevel
from src.analysis.trend_detection import TrendAnalysis, TrendDetector
from src.patterns.pattern_engine import (
    PatternEngine,
    PatternResult,
    detect_patterns,
)
from src.types import (
    CandleData,
    Direction,
    LevelData,
    PatternSignal,
    PatternType,
    RegimeSignal,
    RegimeType,
    StrategyName,
    TQSComponents,
    TradeTier,
    TradeRecommendation,
    TrendDirection,
    TrendSignal,
)

logger = logging.getLogger("candlestickbot.strategy.strategy_engine")

# ---------------------------------------------------------------------------
# Phase 1 constants
# ---------------------------------------------------------------------------

_PHASE1_PATTERNS = frozenset({
    "PIN_BAR_BULLISH",
    "PIN_BAR_BEARISH",
    "ENGULFING_BULLISH",
    "ENGULFING_BEARISH",
})

_PHASE1_ALLOWED_REGIMES = frozenset({
    RegimeType.TRENDING,
    # RANGING: rejected in Phase 1, deferred to Phase 2
})

_BUFFER_PIPS_DEFAULT: float = 2.0    # pips added to stop loss
_MIN_RR_DEFAULT:      float = 2.0
_MIN_TQS_DEFAULT:     int   = 60
_LEVEL_TOLERANCE_DEFAULT: float = 30.0  # pips
_MIN_PATTERN_QUALITY:     int   = 5


# ---------------------------------------------------------------------------
# StrategyConfig — all tunable parameters in one place
# ---------------------------------------------------------------------------

@dataclass
class StrategyConfig:
    """
    Configuration for the Phase 1 Strategy Engine.

    All distances in pips (pip_size scales them to price units).
    """
    # TQS thresholds
    min_tqs_score:          int   = _MIN_TQS_DEFAULT
    premium_tqs_threshold:  int   = 80

    # Gates — can individually be disabled for testing
    trend_gate_enabled:     bool  = True
    regime_gate_enabled:    bool  = True
    level_gate_enabled:     bool  = True
    tqs_gate_enabled:       bool  = True
    rr_gate_enabled:        bool  = True

    # Trade parameters
    min_rr_ratio:           float = _MIN_RR_DEFAULT
    buffer_pips:            float = _BUFFER_PIPS_DEFAULT
    pip_size:               float = 0.0001

    # Level proximity tolerance for the Level Gate
    level_tolerance_pips:   float = _LEVEL_TOLERANCE_DEFAULT

    # Minimum pattern quality score (1-10) for the Pattern Gate
    min_pattern_quality:    int   = _MIN_PATTERN_QUALITY

    # Maximum R:R window for TP search (limits unreachable targets)
    max_rr_tp_window:       float = 10.0


# ---------------------------------------------------------------------------
# GateResult — per-gate verdict
# ---------------------------------------------------------------------------

@dataclass
class GateResult:
    """Verdict from a single gate evaluation."""
    gate:    str        # "SIGNAL" | "TREND" | "REGIME" | "LEVEL" | "TQS" | "RR"
    passed:  bool
    reason:  str = ""


# ---------------------------------------------------------------------------
# TradeRecommendationResult — the full M08 output for one signal candle
# ---------------------------------------------------------------------------

@dataclass
class TradeRecommendationResult:
    """
    Full evaluation result produced by StrategyEngine.evaluate_candle().

    Contains:
      - The phase-1 pattern that triggered evaluation
      - Per-gate verdicts (audit trail)
      - TQS components
      - Final TradeRecommendation (or None + rejection_reason)
    """
    # Identification
    symbol:     str
    timeframe:  str
    timestamp:  datetime

    # Pattern that was evaluated
    pattern:    Optional[PatternResult] = None

    # Analysis snapshots
    trend:      Optional[TrendAnalysis]  = None
    regime:     Optional[RegimeAnalysis] = None
    sr:         Optional[SRAnalysis]     = None

    # Gate verdicts (in order)
    gates:      List[GateResult] = field(default_factory=list)

    # TQS breakdown
    trend_score:   int = 0
    level_score:   int = 0
    pattern_score: int = 0
    regime_score:  int = 0
    tqs_total:     int = 0
    tqs_tier:      TradeTier = TradeTier.REJECT

    # Final output
    recommendation:    Optional[TradeRecommendation] = None
    rejection_reason:  Optional[str] = None
    rejection_gate:    str = ""

    # Nearest level used for target / level score
    nearest_level: Optional[SRLevel] = None

    @property
    def is_recommended(self) -> bool:
        return self.recommendation is not None

    def to_dict(self) -> dict:
        return {
            "symbol":           self.symbol,
            "timeframe":        self.timeframe,
            "timestamp":        self.timestamp.isoformat() if self.timestamp else None,
            "pattern_type":     self.pattern.pattern_type if self.pattern else None,
            "direction":        self.pattern.direction if self.pattern else None,
            "trend_score":      self.trend_score,
            "level_score":      self.level_score,
            "pattern_score":    self.pattern_score,
            "regime_score":     self.regime_score,
            "tqs_total":        self.tqs_total,
            "tqs_tier":         self.tqs_tier.value,
            "recommended":      self.is_recommended,
            "rejection_reason": self.rejection_reason,
            "rejection_gate":   self.rejection_gate,
            "entry_price":      self.recommendation.entry_price if self.recommendation else None,
            "stop_price":       self.recommendation.stop_price if self.recommendation else None,
            "target_price":     self.recommendation.target_price if self.recommendation else None,
            "rr_ratio":         self.recommendation.rr_ratio if self.recommendation else None,
        }


# ---------------------------------------------------------------------------
# StrategyEngine — the core M08 class
# ---------------------------------------------------------------------------

class StrategyEngine:
    """
    M08 — Strategy Engine (Phase 1 MVP).

    Responsibilities:
      - Detect Phase 1 patterns (M07) on the signal bar
      - Gate trade direction against M04 trend
      - Gate regime against M16 (TRENDING only in Phase 1)
      - Gate confluence against M05 S/R levels
      - Compute TQS (0–100) and gate on minimum threshold
      - Compute entry/stop/target and gate on minimum R:R
      - Return TradeRecommendationResult (recommendation + full audit)

    Usage::

        engine = StrategyEngine()
        result = engine.evaluate_candle(
            candles=candles,
            trend=trend_analysis,
            sr=sr_analysis,
            regime=regime_analysis,
        )
        if result.is_recommended:
            rec = result.recommendation

    The engine is stateless between calls.
    All injected engines (M04/M05/M16/M07) are optional — if not supplied,
    the engine runs in "pre-computed mode" where analysis objects are passed
    directly to evaluate_candle().
    """

    def __init__(
        self,
        config:          Optional[StrategyConfig]       = None,
        trend_detector:  Optional[TrendDetector]        = None,
        sr_engine:       Optional[SREngine]             = None,
        regime_engine:   Optional[MarketRegimeEngine]   = None,
        pattern_engine:  Optional[PatternEngine]        = None,
    ):
        self.config         = config or StrategyConfig()
        self.trend_detector = trend_detector
        self.sr_engine      = sr_engine
        self.regime_engine  = regime_engine
        self.pattern_engine = pattern_engine or PatternEngine(
            pip_size=self.config.pip_size,
        )

    # ------------------------------------------------------------------
    # PRIMARY API — evaluate_candle
    # ------------------------------------------------------------------

    def evaluate_candle(
        self,
        candles:  List[CandleData],
        trend:    Optional[TrendAnalysis]  = None,
        sr:       Optional[SRAnalysis]     = None,
        regime:   Optional[RegimeAnalysis] = None,
        level:    Optional[float]          = None,
    ) -> TradeRecommendationResult:
        """
        Evaluate the last candle in ``candles`` for a Phase 1 trade setup.

        Args:
            candles: Full candle series (oldest first).  The LAST candle is
                     the signal bar under evaluation.
            trend:   Pre-computed M04 TrendAnalysis (optional; if not supplied
                     the injected TrendDetector is called).
            sr:      Pre-computed M05 SRAnalysis (optional; if not supplied
                     the injected SREngine is called).
            regime:  Pre-computed M16 RegimeAnalysis (optional; if not supplied
                     the injected MarketRegimeEngine is called).
            level:   Optional price level for pin bar quality scoring.

        Returns:
            TradeRecommendationResult with full audit trail.
            Never raises — on any error returns a rejected result.
        """
        if not candles:
            return self._reject("", "", None, "SIGNAL", "No candles provided")

        signal_candle = candles[-1]
        symbol    = signal_candle.symbol
        timeframe = signal_candle.timeframe
        timestamp = signal_candle.timestamp

        result = TradeRecommendationResult(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=timestamp,
        )

        # ----------------------------------------------------------
        # Step 0: Run injected engines if analysis not pre-computed
        # ----------------------------------------------------------
        try:
            trend  = trend  or (self.trend_detector.analyze(candles)  if self.trend_detector  else None)
            regime = regime or (self.regime_engine.analyze(candles)    if self.regime_engine   else None)
            sr     = sr     or (self.sr_engine.analyze(candles)        if self.sr_engine       else None)
        except Exception as exc:
            logger.warning("M08: analysis engine error — %s", exc)
            return self._reject(symbol, timeframe, timestamp, "DATA",
                                f"Analysis engine error: {exc}")

        result.trend  = trend
        result.regime = regime
        result.sr     = sr

        # ----------------------------------------------------------
        # Gate 1 — SIGNAL GATE: detect Phase 1 pattern on signal bar
        # ----------------------------------------------------------
        patterns = detect_patterns(
            [signal_candle],
            min_tail_ratio=self.pattern_engine.min_tail_ratio,
            strict_engulfing=self.pattern_engine.strict_engulfing,
            level=level,
            pip_size=self.config.pip_size,
        )
        # Also check the last two candles for engulfing (needs prev candle)
        if len(candles) >= 2:
            eng_patterns = detect_patterns(
                [candles[-2], signal_candle],
                min_tail_ratio=self.pattern_engine.min_tail_ratio,
                strict_engulfing=self.pattern_engine.strict_engulfing,
                level=level,
                pip_size=self.config.pip_size,
            )
            # Collect only patterns whose timestamp matches the signal candle
            for ep in eng_patterns:
                if ep.timestamp == signal_candle.timestamp:
                    # Avoid duplication
                    if not any(p.pattern_type == ep.pattern_type for p in patterns):
                        patterns.append(ep)

        # Filter to Phase 1 types with minimum quality
        phase1 = [
            p for p in patterns
            if p.pattern_type in _PHASE1_PATTERNS
            and p.quality_score >= self.config.min_pattern_quality
        ]

        if not phase1:
            g = GateResult("SIGNAL", False,
                           f"No qualifying Phase 1 pattern on signal bar "
                           f"(min_quality={self.config.min_pattern_quality})")
            result.gates.append(g)
            result.rejection_reason = g.reason
            result.rejection_gate   = "SIGNAL"
            return result

        # Best pattern by quality score (tie → keep first)
        best_pattern = max(phase1, key=lambda p: p.quality_score)
        result.pattern = best_pattern
        result.gates.append(GateResult("SIGNAL", True,
                                       f"Pattern: {best_pattern.pattern_type} "
                                       f"quality={best_pattern.quality_score}"))

        direction = best_pattern.direction  # "LONG" or "SHORT"

        # ----------------------------------------------------------
        # Gate 2 — TREND GATE
        # ----------------------------------------------------------
        if self.config.trend_gate_enabled:
            trend_gate = self._check_trend_gate(trend, direction)
            result.gates.append(trend_gate)
            if not trend_gate.passed:
                result.rejection_reason = trend_gate.reason
                result.rejection_gate   = "TREND"
                return result
        else:
            result.gates.append(GateResult("TREND", True, "disabled"))

        # ----------------------------------------------------------
        # Gate 3 — REGIME GATE
        # ----------------------------------------------------------
        if self.config.regime_gate_enabled:
            regime_gate = self._check_regime_gate(regime, best_pattern.pattern_type)
            result.gates.append(regime_gate)
            if not regime_gate.passed:
                result.rejection_reason = regime_gate.reason
                result.rejection_gate   = "REGIME"
                return result
        else:
            result.gates.append(GateResult("REGIME", True, "disabled"))

        # ----------------------------------------------------------
        # Gate 4 — LEVEL GATE
        # ----------------------------------------------------------
        nearest_level: Optional[SRLevel] = None
        if self.config.level_gate_enabled:
            level_gate, nearest_level = self._check_level_gate(
                signal_candle, sr, direction
            )
            result.gates.append(level_gate)
            result.nearest_level = nearest_level
            if not level_gate.passed:
                result.rejection_reason = level_gate.reason
                result.rejection_gate   = "LEVEL"
                return result
        else:
            result.gates.append(GateResult("LEVEL", True, "disabled"))
            if sr:
                nearest_level = (
                    sr.nearest_support if direction == "LONG"
                    else sr.nearest_resistance
                )
            result.nearest_level = nearest_level

        # ----------------------------------------------------------
        # TQS calculation
        # ----------------------------------------------------------
        trend_score   = trend.tqs_trend_score if trend else 0
        regime_score  = regime.tqs_regime_score if regime else 0
        pattern_score = self.pattern_engine.calculate_tqs_pattern_score(best_pattern)
        level_score   = self._compute_level_score(
            signal_candle, sr, direction
        )

        tqs_total = trend_score + level_score + pattern_score + regime_score
        tqs_tier  = _classify_tier(tqs_total)

        result.trend_score   = trend_score
        result.level_score   = level_score
        result.pattern_score = pattern_score
        result.regime_score  = regime_score
        result.tqs_total     = tqs_total
        result.tqs_tier      = tqs_tier

        # ----------------------------------------------------------
        # Gate 5 — TQS GATE
        # ----------------------------------------------------------
        if self.config.tqs_gate_enabled:
            tqs_gate = GateResult(
                "TQS",
                tqs_total >= self.config.min_tqs_score,
                f"TQS={tqs_total} "
                f"({'pass' if tqs_total >= self.config.min_tqs_score else 'fail'}, "
                f"min={self.config.min_tqs_score})",
            )
            result.gates.append(tqs_gate)
            if not tqs_gate.passed:
                result.rejection_reason = (
                    f"TQS {tqs_total} < minimum {self.config.min_tqs_score}"
                )
                result.rejection_gate = "TQS"
                return result
        else:
            result.gates.append(GateResult("TQS", True, "disabled"))

        # ----------------------------------------------------------
        # Entry / Stop / Target calculation
        # ----------------------------------------------------------
        entry, stop = self._compute_entry_stop(signal_candle, best_pattern)
        target, rr  = self._compute_target(
            entry, stop, direction, sr, nearest_level
        )

        # ----------------------------------------------------------
        # Gate 6 — R:R GATE
        # ----------------------------------------------------------
        if self.config.rr_gate_enabled:
            rr_gate = GateResult(
                "RR",
                rr >= self.config.min_rr_ratio,
                f"R:R={rr:.2f} "
                f"({'pass' if rr >= self.config.min_rr_ratio else 'fail'}, "
                f"min={self.config.min_rr_ratio})",
            )
            result.gates.append(rr_gate)
            if not rr_gate.passed:
                result.rejection_reason = (
                    f"R:R {rr:.2f} < minimum {self.config.min_rr_ratio}"
                )
                result.rejection_gate = "RR"
                return result
        else:
            result.gates.append(GateResult("RR", True, "disabled"))

        # ----------------------------------------------------------
        # Build TradeRecommendation
        # ----------------------------------------------------------
        tqs_components = TQSComponents(
            trend_score=float(trend_score),
            level_score=float(level_score),
            pattern_score=float(pattern_score),
            regime_score=float(regime_score),
        )

        strategy_name = (
            StrategyName.PIN_BAR
            if "PIN_BAR" in best_pattern.pattern_type
            else StrategyName.ENGULFING_BAR
        )

        direction_enum = Direction.LONG if direction == "LONG" else Direction.SHORT

        # Build pattern signal for the recommendation
        try:
            pattern_signal: Optional[PatternSignal] = best_pattern.to_pattern_signal()
        except Exception:
            pattern_signal = None

        # Build regime signal
        regime_signal: Optional[RegimeSignal] = None
        if regime:
            try:
                regime_signal = regime.to_regime_signal(symbol, timeframe, timestamp)
            except Exception:
                pass

        # Build trend signal
        trend_signal_dto: Optional[TrendSignal] = None
        if trend:
            try:
                trend_signal_dto = trend.to_trend_signal(symbol, timeframe, timestamp)
            except Exception:
                pass

        # Convert nearest level to LevelData DTO
        level_dto: Optional[LevelData] = None
        if nearest_level:
            try:
                level_dto = nearest_level.to_level_data()
            except Exception:
                pass

        rec = TradeRecommendation(
            strategy=strategy_name,
            symbol=symbol,
            timeframe=timeframe,
            direction=direction_enum,
            entry_price=entry,
            stop_price=stop,
            target_price=target,
            rr_ratio=rr,
            tqs=tqs_components,
            pattern=pattern_signal,
            trend=trend_signal_dto,
            regime=regime_signal,
            level=level_dto,
            timestamp=timestamp,
        )

        result.recommendation = rec
        logger.info(
            "M08 RECOMMEND: %s %s %s E=%.5f SL=%.5f TP=%.5f RR=%.2f TQS=%d [%s]",
            strategy_name.value, direction, symbol,
            entry, stop, target, rr, tqs_total, tqs_tier.value,
        )
        return result

    # ------------------------------------------------------------------
    # PUBLIC CONVENIENCE: evaluate multiple candles as a batch
    # ------------------------------------------------------------------

    def evaluate_series(
        self,
        candles:  List[CandleData],
        trend:    Optional[TrendAnalysis]  = None,
        sr:       Optional[SRAnalysis]     = None,
        regime:   Optional[RegimeAnalysis] = None,
    ) -> List[TradeRecommendationResult]:
        """
        Scan every candle in the series as a potential signal bar.

        Returns only recommended results (gates all passed).
        Useful for back-testing a candle sequence.

        The first 21 candles (SMA warm-up) are skipped.
        """
        results: List[TradeRecommendationResult] = []
        min_idx = max(21, 1)
        for i in range(min_idx, len(candles)):
            window = candles[: i + 1]
            r = self.evaluate_candle(window, trend=trend, sr=sr, regime=regime)
            if r.is_recommended:
                results.append(r)
        return results

    # ------------------------------------------------------------------
    # PUBLIC: calculate TQS components from raw inputs
    # ------------------------------------------------------------------

    def calculate_tqs(
        self,
        trend_score:   int,
        level_score:   int,
        pattern_score: int,
        regime_score:  int,
    ) -> TQSComponents:
        """
        Build a TQSComponents object from raw component scores.

        Used by callers that compute scores externally (e.g. tests).
        """
        return TQSComponents(
            trend_score=float(trend_score),
            level_score=float(level_score),
            pattern_score=float(pattern_score),
            regime_score=float(regime_score),
        )

    def classify_tier(self, tqs_total: int) -> TradeTier:
        """Classify a TQS total into REJECT / STANDARD / PREMIUM."""
        return _classify_tier(tqs_total)

    # ------------------------------------------------------------------
    # PRIVATE — Gate checks
    # ------------------------------------------------------------------

    def _check_trend_gate(
        self,
        trend: Optional[TrendAnalysis],
        direction: str,
    ) -> GateResult:
        """
        Trend Gate: trade direction must align with a confirmed tradeable trend.

        Phase 1 default: only trend-continuation trades.
          LONG  requires UP trend with tradeable=True
          SHORT requires DOWN trend with tradeable=True
        """
        if trend is None:
            return GateResult(
                "TREND", False,
                "No trend analysis available — trend gate failed",
            )

        if not trend.tradeable:
            return GateResult(
                "TREND", False,
                f"Trend not tradeable: {trend.reason} "
                f"(direction={trend.direction}, confidence={trend.confidence_score:.1f})",
            )

        if direction == "LONG" and trend.direction != "UP":
            return GateResult(
                "TREND", False,
                f"LONG trade requires UP trend, got {trend.direction}",
            )

        if direction == "SHORT" and trend.direction != "DOWN":
            return GateResult(
                "TREND", False,
                f"SHORT trade requires DOWN trend, got {trend.direction}",
            )

        return GateResult(
            "TREND", True,
            f"Trend confirmed: {trend.direction} "
            f"(confidence={trend.confidence_score:.1f}, TQS={trend.tqs_trend_score})",
        )

    def _check_regime_gate(
        self,
        regime: Optional[RegimeAnalysis],
        pattern_type: str,
    ) -> GateResult:
        """
        Regime Gate (Phase 1): only TRENDING regime is allowed.

        RANGING / VOLATILE / QUIET / CHOPPY / UNKNOWN → reject.
        Strategy name must be in allowed_strategies list.
        """
        if regime is None:
            return GateResult(
                "REGIME", False,
                "No regime analysis available — regime gate failed",
            )

        if regime.regime not in _PHASE1_ALLOWED_REGIMES:
            return GateResult(
                "REGIME", False,
                f"Regime {regime.regime.value} not allowed in Phase 1 "
                f"(allowed: {[r.value for r in _PHASE1_ALLOWED_REGIMES]}). "
                f"Reason: {regime.reason}",
            )

        strategy_name = (
            "pin_bar" if "PIN_BAR" in pattern_type else "engulfing_bar"
        )
        if strategy_name not in regime.allowed_strategies:
            return GateResult(
                "REGIME", False,
                f"Strategy '{strategy_name}' not in allowed list "
                f"{regime.allowed_strategies} for regime {regime.regime.value}",
            )

        return GateResult(
            "REGIME", True,
            f"Regime {regime.regime.value} allows {strategy_name} "
            f"(TQS={regime.tqs_regime_score})",
        )

    def _check_level_gate(
        self,
        candle:    CandleData,
        sr:        Optional[SRAnalysis],
        direction: str,
    ) -> tuple[GateResult, Optional[SRLevel]]:
        """
        Level Gate: signal candle must be within level_tolerance_pips of
        the nearest relevant S/R level (or 21 SMA).

        Direction LONG  → check nearest_support (price below / at support).
        Direction SHORT → check nearest_resistance (price above / at resistance).

        Returns (GateResult, nearest_level_or_None).
        """
        tolerance = self.config.level_tolerance_pips * self.config.pip_size

        if sr is None:
            return (
                GateResult("LEVEL", False,
                           "No S/R analysis available — level gate failed"),
                None,
            )

        # Also include SMA level as a valid level
        candidates: List[SRLevel] = []
        if direction == "LONG":
            if sr.nearest_support:
                candidates.append(sr.nearest_support)
        else:
            if sr.nearest_resistance:
                candidates.append(sr.nearest_resistance)

        # Add SMA level if present and on the correct side
        if sr.sma21_level:
            if direction == "LONG" and sr.sma21_level.price <= candle.close:
                candidates.append(sr.sma21_level)
            elif direction == "SHORT" and sr.sma21_level.price >= candle.close:
                candidates.append(sr.sma21_level)

        if not candidates:
            return (
                GateResult(
                    "LEVEL", False,
                    f"No {'support' if direction == 'LONG' else 'resistance'} "
                    f"level available near signal bar",
                ),
                None,
            )

        # Find the candidate closest to the signal bar's tail extreme
        tail_price = candle.low if direction == "LONG" else candle.high
        nearest = min(candidates, key=lambda lv: abs(lv.price - tail_price))
        dist = abs(nearest.price - tail_price)

        if dist > tolerance:
            return (
                GateResult(
                    "LEVEL", False,
                    f"Nearest level @ {nearest.price:.5f} is {dist/self.config.pip_size:.1f} pips "
                    f"from tail ({tail_price:.5f}), tolerance={self.config.level_tolerance_pips} pips",
                ),
                nearest,
            )

        return (
            GateResult(
                "LEVEL", True,
                f"Level @ {nearest.price:.5f} within "
                f"{dist/self.config.pip_size:.1f} pips of tail",
            ),
            nearest,
        )

    # ------------------------------------------------------------------
    # PRIVATE — TQS level score (wraps SREngine method)
    # ------------------------------------------------------------------

    def _compute_level_score(
        self,
        candle:    CandleData,
        sr:        Optional[SRAnalysis],
        direction: str,
    ) -> int:
        """
        Compute TQS level component (0–25) using same rubric as M05.

        Uses SREngine.calculate_tqs_level_score() if sr_engine injected,
        else replicates the logic inline to avoid circular dependency.
        """
        if sr is None:
            return 5  # no level data → minimum score

        if self.sr_engine:
            try:
                return self.sr_engine.calculate_tqs_level_score(
                    candle,
                    sr.nearest_support,
                    sr.nearest_resistance,
                    direction,
                )
            except Exception:
                pass

        # Inline fallback (same rubric as M05)
        from src.analysis.sr_engine import LevelStrength
        relevant = (
            sr.nearest_support if direction == "LONG"
            else sr.nearest_resistance
        )
        if relevant is None:
            return 5

        nearby_threshold = self.config.level_tolerance_pips * self.config.pip_size
        candle_price = candle.low if direction == "LONG" else candle.high
        if abs(candle_price - relevant.price) > nearby_threshold:
            return 5

        if relevant.is_resistance_turned_support:
            return 25
        if relevant.strength == LevelStrength.STRONG:
            return 22
        if relevant.strength == LevelStrength.MODERATE:
            return 18
        return 12

    # ------------------------------------------------------------------
    # PRIVATE — Entry / Stop calculation
    # ------------------------------------------------------------------

    def _compute_entry_stop(
        self,
        candle:  CandleData,
        pattern: PatternResult,
    ) -> tuple[float, float]:
        """
        Compute aggressive entry and stop-loss prices.

        Entry:
          LONG  → candle.high  (break above pattern)
          SHORT → candle.low

        Stop:
          LONG  → candle.low  - buffer_pips * pip_size
          SHORT → candle.high + buffer_pips * pip_size
        """
        buf = self.config.buffer_pips * self.config.pip_size
        if pattern.direction == "LONG":
            entry = candle.high
            stop  = candle.low - buf
        else:
            entry = candle.low
            stop  = candle.high + buf
        return entry, stop

    # ------------------------------------------------------------------
    # PRIVATE — Take profit / R:R calculation
    # ------------------------------------------------------------------

    def _compute_target(
        self,
        entry:         float,
        stop:          float,
        direction:     str,
        sr:            Optional[SRAnalysis],
        nearest_level: Optional[SRLevel],
    ) -> tuple[float, float]:
        """
        Compute take-profit target and actual R:R ratio.

        Priority:
          1. Next valid S/R level in trade direction within max_rr_tp_window
          2. Fallback: entry ± risk_distance * min_rr_ratio (guarantees 2R)

        Returns (target_price, rr_ratio).
        """
        risk_distance = abs(entry - stop)
        if risk_distance <= 0:
            return entry, 0.0

        # Fallback target (guarantees min R:R)
        fallback_target = (
            entry + risk_distance * self.config.min_rr_ratio
            if direction == "LONG"
            else entry - risk_distance * self.config.min_rr_ratio
        )

        if sr is None:
            return fallback_target, self.config.min_rr_ratio

        # Search for a better target from S/R levels
        max_window = risk_distance * self.config.max_rr_tp_window
        best_target: Optional[float] = None
        best_rr:     float           = 0.0

        # All levels sorted by price
        sorted_levels = sorted(sr.levels, key=lambda lv: lv.price)

        for lv in sorted_levels:
            if direction == "LONG":
                # Level must be above entry
                if lv.price <= entry:
                    continue
                dist_to_level = lv.price - entry
                if dist_to_level > max_window:
                    continue
                rr = dist_to_level / risk_distance
                if rr >= self.config.min_rr_ratio and rr > best_rr:
                    best_target = lv.price
                    best_rr     = rr
                    break  # Take nearest qualifying level above entry
            else:
                # Level must be below entry
                if lv.price >= entry:
                    continue
                dist_to_level = entry - lv.price
                if dist_to_level > max_window:
                    continue
                rr = dist_to_level / risk_distance
                if rr >= self.config.min_rr_ratio and rr > best_rr:
                    best_target = lv.price
                    best_rr     = rr

        if best_target is not None:
            actual_rr = abs(best_target - entry) / risk_distance
            return best_target, actual_rr

        # Fallback
        return fallback_target, self.config.min_rr_ratio

    # ------------------------------------------------------------------
    # PRIVATE — helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _reject(
        symbol: str,
        timeframe: str,
        timestamp: Optional[datetime],
        gate:      str,
        reason:    str,
    ) -> TradeRecommendationResult:
        return TradeRecommendationResult(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=timestamp or datetime.now(timezone.utc),
            gates=[GateResult(gate, False, reason)],
            rejection_reason=reason,
            rejection_gate=gate,
        )


# ---------------------------------------------------------------------------
# Module-level helpers (public, stateless)
# ---------------------------------------------------------------------------

def classify_tier(tqs_total: int) -> TradeTier:
    """Classify a TQS total (0–100) into REJECT / STANDARD / PREMIUM."""
    return _classify_tier(tqs_total)


def compute_tqs(
    trend_score:   int,
    level_score:   int,
    pattern_score: int,
    regime_score:  int,
) -> TQSComponents:
    """
    Build a TQSComponents DTO from the four component scores.

    Each component is 0–25.  Total is their sum (0–100).
    """
    return TQSComponents(
        trend_score=float(trend_score),
        level_score=float(level_score),
        pattern_score=float(pattern_score),
        regime_score=float(regime_score),
    )


def is_phase1_pattern(pattern_type: str) -> bool:
    """Return True if pattern_type is a supported Phase 1 pattern."""
    return pattern_type in _PHASE1_PATTERNS


def is_regime_allowed(regime: RegimeType) -> bool:
    """Return True if the regime allows Phase 1 trading."""
    return regime in _PHASE1_ALLOWED_REGIMES


def compute_entry_stop(
    candle:     CandleData,
    direction:  str,
    buffer_pips: float = _BUFFER_PIPS_DEFAULT,
    pip_size:   float  = 0.0001,
) -> tuple[float, float]:
    """
    Compute aggressive entry and stop-loss for a pattern candle.

    Returns (entry_price, stop_price).
    """
    buf = buffer_pips * pip_size
    if direction == "LONG":
        return candle.high, candle.low - buf
    else:
        return candle.low, candle.high + buf


def compute_rr(entry: float, stop: float, target: float) -> float:
    """
    Compute R:R ratio.  Returns 0.0 if risk is zero.

    R:R = abs(target - entry) / abs(stop - entry)
    """
    risk = abs(stop - entry)
    if risk <= 0:
        return 0.0
    return abs(target - entry) / risk


# ---------------------------------------------------------------------------
# Private
# ---------------------------------------------------------------------------

def _classify_tier(tqs_total: int) -> TradeTier:
    if tqs_total >= 80:
        return TradeTier.PREMIUM
    if tqs_total >= 60:
        return TradeTier.STANDARD
    return TradeTier.REJECT
