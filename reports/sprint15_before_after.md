# Sprint 15 — Before/After Comparison

Configuration (identical for both): BacktestConfig(symbol='EURUSD', timeframe='D1',
initial_balance=10_000.0, slippage_pips=1.0, lookback_window=200,
minimum_tqs=0.0, minimum_rr=2.0)

NOTE: minimum_tqs=0.0 means the TQS≥60 gate was DISABLED in the Sprint 14
configuration (provenance: scripts/run_sprint14_backtests.py). This baseline
therefore measures the gate stack without the TQS filter.

## Sprint 14 (V0 — pre-fix, commit cbb4092)
| Mode       | Generated | Executed | Net P&L | Passes |
|------------|-----------|----------|---------|--------|
| Pin Bar    | 0         | 0        | $0.00   | NO     |
| Engulfing  | 0         | 0        | $0.00   | NO     |
| Combined   | 0         | 0        | $0.00   | NO     |

Root cause: RC-1 (SR wiring gap) + RC-3 (DD broken, invisible) + RC-5 (wrong criteria)

## Sprint 15 V1 Appendix (swings-only, no sma21)
| Mode       | Generated | Executed | W/L  | Net P&L  | Passes |
|------------|-----------|----------|------|----------|--------|
| Pin Bar    | 3         | 3        | 1/2  | +$24     | NO (N<30) |
| Engulfing  | 0         | 0        | 0/0  | $0.00    | NO     |
| Combined   | 3         | 3        | 1/2  | +$24     | NO (N<30) |

## Sprint 15 V2 — OFFICIAL (swings + sma21, commit 4b1e75c)
| Mode       | Generated | Executed | W/L  | PF     | Exp    | DD       | Net P&L  | Passes |
|------------|-----------|----------|------|--------|--------|----------|----------|--------|
| Pin Bar    | 4         | 4        | 0/4  | 0.000  | -1.00R | 3.97%    | -$397.27 | NO     |
| Engulfing  | 0         | 0        | 0/0  | 0.000  | 0.00R  | 0.00%    | $0.00    | NO     |
| Combined   | 4         | 4        | 0/4  | 0.000  | -1.00R | 3.97%    | -$397.27 | NO     |

**Primary failure reason: N=4 < 30 (INSUFFICIENT DATA)**

The measurement system is now repaired. The verdict is INSUFFICIENT DATA,
not a negative edge verdict, because sample size requirements are unmet.
