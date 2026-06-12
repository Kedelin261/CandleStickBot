# Sprint 15 — Final Determination

## VERDICT: INSUFFICIENT DATA

### §7 Evidence Standard Applied

**Criterion 1 — N < 30 on official run:**
  Pin Bar:   4 trades executed (need ≥30) ❌
  Engulfing: 0 trades executed             ❌
  Combined:  4 trades executed             ❌

**Criterion 2 — Data classified SYNTHETIC_SUSPECT:**
  34.8% exact dojis, 100% zero volume, uniform DOW, holiday bar included.
  See reports/sprint15_data_authenticity.txt. ❌

Both criteria independently trigger INSUFFICIENT DATA.

### Measurement System Validation (Completed)
- FIX-1+2: SR wiring repaired — Level Gate now functional ✅
- FIX-3: Equity drawdown validated by golden tests (1.9802% canonical) ✅
- FIX-4: passes_baseline uses canonical criteria (N≥30, PF>1.10, Exp>0, DD<25%) ✅
- FIX-5: trades_generated counts only strategy-enabled trades ✅
- 2054 tests passing (0 failed) ✅
- Funnel conservation verified ✅

### Official V2 Results (FIX-1+2, window=200, tqs=0, rr=2)
| Mode       | Executed | PF    | Exp    | DD     | Passes |
|------------|----------|-------|--------|--------|--------|
| Pin Bar    | 4        | 0.000 | -1.00R | 3.97%  | NO     |
| Engulfing  | 0        | 0.000 | 0.00R  | 0.00%  | NO     |
| Combined   | 4        | 0.000 | -1.00R | 3.97%  | NO     |

NOTE: minimum_tqs=0.0 — TQS gate was disabled in the official Sprint 14
config. This baseline measures the gate stack without the TQS filter.

### Capacity Analysis (Stage 7)
Engulfing ceiling: ~5 raw detections in 12.5 years → N≥30 mathematically impossible.
Pin Bar: 4/3041 windows = 0.13% rate → requires ~88 years of data at frozen params.
N≥30 is unreachable on this dataset at frozen Phase 1 parameters.

This is a property of the frozen parameters + this dataset — NOT a failure
of the now-validated measurement system.

### Decisions Required from the Project Owner

1. **Source a verified dataset.** Replace EURUSD_D1_2014_2026.csv with a
   real broker feed (e.g., FXCM, Dukascopy, MetaTrader export). Verify:
   doji rate <1%, non-zero volume, correct holiday gaps, spot-check closes.

2. **Revisit dataset provenance.** Determine how the current CSV was
   generated. If it is synthetic/resampled, a real feed is mandatory for
   any statistically valid verdict.

3. **Consider Phase 2 H4 timeframe.** Daily data generates ~4 trades per
   12.5 years through this gate stack. H4 data over the same period would
   provide ~6× more bars and potentially reach N≥30. This is a design
   decision (strategy modification) — implement only after owner review.

4. **Optionally enable TQS gate.** The Sprint 14 config used minimum_tqs=0.0
   (disabled). A re-run with minimum_tqs=60 (the intended threshold) would
   test whether the TQS filter changes the gate distribution.

5. **Do NOT tune strategy parameters** to increase trade count on the
   current dataset — that would invalidate the scientific baseline and
   constitute data-snooping.
