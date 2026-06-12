# Sprint 15 — Remediation Summary

## FIX-1: M03→M05 swing wiring (RC-1)
**File:** `src/integration/pipeline_runner.py` line ~476
**Change:** `sr_a = self._sr.analyze(context_candles)` →
  `self._sr.analyze(context_candles, swing_highs=structure_a.swing_highs, swing_lows=structure_a.swing_lows, sma21=trend_a.sma21)`
**Evidence:** PHASE0_BLUEPRINT.md signature, README §core philosophy, funnel showing 0 RECOMMENDED pre-fix.
**Effect:** V0→0 trades; V2→4 trades.

## FIX-2: sma21 wiring decision (RC-2)
**Decision:** ADOPTED as official baseline (V2). Spec language is unambiguous.
**Evidence:** README line 17: "Patterns must form at key S/R levels or the 21 SMA."
PHASE0_BLUEPRINT.md: `analyze(candles, swing_highs, swing_lows, sma21)`.
**V1 appendix:** swings-only yields 3 trades; V2 (official) yields 4 trades.

## FIX-3: Account-equity max drawdown (RC-3)
**File:** `src/integration/pipeline_runner.py`
**Change:** Added `_compute_max_drawdown_equity(pnl_usd_series, initial_balance)`.
`PipelineResult` now tracks `_pnl_usd_series: List[float]` and `finalise()` calls new function.
**Golden validation:** `[+100,−200,+50]` from $10k → 1.9802% (§2 canonical). 42/42 golden tests pass.

## FIX-4: passes_baseline canonical criteria (RC-5)
**File:** `src/backtesting/backtest_runner.py`
**Change:** PF≥1.10, WR≥40%, DD≤20%, N≥10 → PF>1.10 AND Exp>0 AND DD<25% AND N≥30.
**Tests updated:** test_backtest_runner.py, test_reports.py, test_sprint14.py — all with docstrings citing RC-5.

## FIX-5: trades_generated counter (RC-7)
**File:** `src/integration/pipeline_runner.py`
**Change:** `result.trades_generated += 1` moved after strategy-enable filter.
**Effect:** Engulfing-only mode no longer inflates counter with pin-bar recommendations.

## No Other Production Changes
All other modules (M03–M19) are read-only. No strategy logic, parameters, thresholds, or dataset was modified.
