"""
Append Sprint 14 diagnostic findings to the validation lab report.
Documents the root cause of 0 trades and the corrected-wiring evidence.
"""
import sys
sys.path.insert(0, '.')

from pathlib import Path

DIAGNOSTIC_SECTION = """

============================================================
  SPRINT 14 DIAGNOSTIC INVESTIGATION
============================================================

── ROOT CAUSE: M12 Pipeline SR-Wiring Gap ──────────────────
  The unmodified PipelineRunner.run() generates 0 trades on
  3,240 candles of real EURUSD D1 data (2014-01-01 to 2026-06-12).

  ROOT CAUSE (confirmed):
    pipeline_runner.py line 436:
      sr_a = self._sr.analyze(context_candles)
                         ↑ NO swing_highs / swing_lows passed

    SREngine.analyze() with no swing points:
      → _build_levels_from_swings([]) returns []
      → SRAnalysis.support_levels  == []
      → SRAnalysis.resistance_levels == []

    Level Gate (strategy_engine.py):
      → "No support/resistance level available" → FAIL
      → 0 trades regardless of window size or data length

  CONSTRAINT: pipeline_runner.py is M12 — cannot be modified
              per Sprint 14 hard constraint (no Phase 1 changes).

── DIAGNOSTIC EVIDENCE (correctly wired M03→M05) ────────────
  When struct_a.swing_highs / swing_lows ARE passed to sr.analyze():

  Window │  SR>0 rate  │  Tradeable  │  Trending   │  All-3-gates
  ────── │ ─────────── │ ─────────── │ ─────────── │ ─────────────
    50   │  100.0%     │   3.1%      │  56.7%      │   1.3%
   100   │  100.0%     │  14.2%      │  57.5%      │   8.3%
   200   │  100.0%     │  14.6%      │  57.1%      │   8.6%
   300   │  100.0%     │  14.1%      │  56.3%      │   8.0%
   500   │  100.0%     │  14.6%      │  57.1%      │   8.4%

  Key findings from diagnostic (window=200, all 3,040 positions):
    Candles evaluated  : 3,040
    Strategy signals   : 3  (0.10% of positions)
    All SHORT direction (no LONG signals in this sample)

  Pattern detection (standalone, no gates):
    Pin Bar detected   : 1,152 / 3,240  (35.6% of candles)
    SR produces levels : 100% when swing points wired correctly

── GATE-BY-GATE ANALYSIS (sampled 638 positions, window=50) ──
  Gate         │ Pass Rate │ Root Cause (when failing)
  ──────────── │ ───────── │ ─────────────────────────────────────────
  Signal Gate  │  35.6%    │ Pattern not detected on signal bar
  Trend Gate   │   3.1%    │ TrendAnalysis.tradeable=False (ADX<threshold)
               │           │ → improves to 14.2% at window=100
  Regime Gate  │  56.7%    │ RANGING / CHOPPY regimes rejected (correct)
  Level Gate   │   0.0%    │ SR wiring gap in M12 pipeline
               │           │ → 100% pass rate when wiring fixed

── SPRINT 14 BASELINE ANSWER ─────────────────────────────────

  QUESTION: "Does the current unoptimized Phase 1 system show
             evidence of a baseline edge?"

  ANSWER:   CANNOT DETERMINE (system structurally blocked)

  REASON:
    The Phase 1 system produces 0 trades in its current unmodified
    state. This is NOT evidence of a poor edge — it is evidence of
    an integration gap in M12 PipelineRunner where M05 SREngine
    never receives swing points from M03 MarketStructureAnalyzer.

    The individual components ARE functional:
      ✅ M03 MarketStructureAnalyzer — detects swing highs/lows correctly
      ✅ M05 SREngine — produces S/R levels when swing points are provided
      ✅ M07 PatternEngine — detects 1,152 pin bars (35.6% of candles)
      ✅ M16 MarketRegimeEngine — classifies TRENDING 56.7% of positions
      ✅ M04 TrendDetector — tradeable signal ~14% at window≥100
      ❌ M12 PipelineRunner — does not wire M03 output → M05 input

  REQUIRED FIX (Sprint 15 candidate):
    Change pipeline_runner.py line 436 from:
      sr_a = self._sr.analyze(context_candles)
    To:
      sr_a = self._sr.analyze(
          context_candles,
          swing_highs=structure_a.swing_highs,
          swing_lows=structure_a.swing_lows,
      )

  POST-FIX EXPECTATION:
    With window=200 and corrected wiring:
      ~8% of positions pass all 3 gates
      3,040 positions → ~250 trade candidates
      Sufficient sample for baseline edge measurement

============================================================
"""

val_path = Path('reports/validation_lab_report.txt')
existing = val_path.read_text()
val_path.write_text(existing + DIAGNOSTIC_SECTION)
print(f"Appended diagnostic section to {val_path}")

# Also write a standalone diagnostic summary
diag_path = Path('reports/diagnostic_findings.txt')
diag_path.write_text(DIAGNOSTIC_SECTION.strip())
print(f"Wrote standalone diagnostic to {diag_path}")
