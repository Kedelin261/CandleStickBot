# Sprint 15 — Root Cause Report

## RC-1 (CONFIRMED) — M03→M05 S/R Wiring Gap
**File:** `src/integration/pipeline_runner.py:436` (pre-fix)
**Evidence:** `sr_a = self._sr.analyze(context_candles)` — swing_highs/lows omitted.
`SREngine._build_levels_from_swings([])` → `[]` → Level Gate fails 100%.
**Funnel proof:** V0 diagnostic shows LEVEL rejected=11 (of 12 reaching that gate), RECOMMENDED=0.
**Fix:** FIX-1 wires `swing_highs=structure_a.swing_highs, swing_lows=structure_a.swing_lows`.

## RC-2 (CONFIRMED) — SMA21 Dynamic Level Dead
**Evidence:** Same call site omitted `sma21`. `TrendAnalysis.sma21` computed every window but never reached M05.
README: "Patterns must form at key S/R levels or the 21 SMA." PHASE0_BLUEPRINT.md: `analyze(candles, swing_highs, swing_lows, sma21)`.
**Fix:** FIX-2 wires `sma21=trend_a.sma21`. Result: V1→3 trades, V2→4 trades.

## RC-3 (CONFIRMED) — Max Drawdown Metric Broken
**File:** `_compute_max_drawdown()` seeded `peak = 0.0` over R-multiples.
First losing trade: `dd = (0−(−1))/1e-9 × 100 ≈ 1×10¹¹%`. Invisible in Sprint 14 (0 trades).
**Fix:** FIX-3 creates `_compute_max_drawdown_equity(pnl_usd_series, initial_balance)`, seeded at `initial_balance`. Golden: `[+100,−200,+50]` from $10k → 1.9802%.

## RC-4 (CONFIRMED) — Dataset SYNTHETIC_SUSPECT
34.8% exact dojis (real rate <1%), 100% zero volume, perfectly uniform DOW, 2014-01-01 holiday bar, 4–10 pip spot-check deviations. See `reports/sprint15_data_authenticity.txt`.

## RC-5 (CONFIRMED) — passes_baseline Criteria Mismatch
Old: PF≥1.10, WR≥40%, DD≤20%, N≥10. Canonical (§5): PF>1.10, Exp>0, DD<25%, N≥30.
**Fix:** FIX-4 updates `BacktestResult.passes_baseline` to canonical criteria.

## RC-6 (CONFIRMED) — Sprint 14 Diagnostic Used Non-Production Path
Sprint 14's "1,152/3,240 pin bars (35.6%)" used relaxed parameters. Production path (PatternEngine defaults, min_quality=5) detects ~165 pin bars (~5%) and only 5 engulfing bars in 12.5 years. All Sprint 15 diagnostics use production code paths only.

## RC-7 (CONFIRMED) — trades_generated Counter Semantics
Counter incremented before strategy-enable filter. In "Engulfing Only" mode, pin-bar recommendations inflated `trades_generated`. **Fix:** FIX-5 moves counter after filter.
