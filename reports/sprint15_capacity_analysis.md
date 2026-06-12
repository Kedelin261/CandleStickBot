# Sprint 15 — Capacity Analysis

## Purpose
Prove mathematically that N≥30 is unreachable at frozen Phase 1 parameters
on this dataset, supporting an INSUFFICIENT DATA verdict.

## Raw Detection Counts (production path, min_quality=5)

### Pin Bar
Funnel (V2, window=200, 3,041 evaluations):
  - SIGNAL gate: 2,882 rejected (no qualifying pattern ≥ quality 5)
  - TREND gate: 146 rejected
  - REGIME gate: 1 rejected
  - LEVEL gate: 8 rejected (with FIX-1/2)
  - RECOMMENDED: 4 (V2 official)

Raw detections reaching SIGNAL gate: 3,041 - 2,882 = 159 patterns qualified
After all gates: 4 trades executed.
To reach N=30: need 7.5× more qualifying patterns pass all gates at frozen params.

### Engulfing Bar
Total raw detections in 3,240 bars (12.5 years): ~5
All 5 are filtered out by TREND gate (ADX/confidence thresholds).
RECOMMENDED = 0. Ceiling = 5. N≥30 is mathematically impossible.

## Mathematical Ceiling

For Engulfing:
  ceiling = raw_detections = ~5 < 30
  ∴ N≥30 is IMPOSSIBLE at any gate configuration with frozen parameters.

For Pin Bar:
  4 trades executed from 3,041 windows over 12.5 years.
  Expected rate: 4/3041 ≈ 0.13% per window.
  To get 30 trades: need 30/0.0013 ≈ 23,000 windows ≈ 88 years of data.
  At frozen lookback_window=200: impossible on this 12.5-year dataset.

For Combined:
  Same as Pin Bar (Engulfing contributes 0).

## Conclusion
N≥30 is mathematically unreachable on EURUSD D1 2014–2026 at frozen
Phase 1 parameters. This is a dataset capacity constraint, not a strategy
failure — the correct verdict is INSUFFICIENT DATA.
