# Sprint 16 — Capacity Analysis
## Status: PLACEHOLDER — BLOCKED-ON-DATA

**Generated:** 2026-06-12  
**Real dataset:** NOT YET SUPPLIED  

---

## Purpose

Determine whether N ≥ 30 is reachable on real EURUSD D1 data at frozen
Phase 1 parameters, and if so, how many years of history are required.

This answers the Sprint 17 decision: branch 17A (extend history) vs
17B/C (different timeframe/symbol).

---

## Legacy Dataset Capacity (for comparison — INVALID for verdict)

The legacy dataset (SYNTHETIC_SUSPECT) produced:
- 4 trades executed over 12.44 years = **0.32 trades/year**
- To reach N=30: **~94 years of this synthetic data**
- **Conclusion on legacy data**: N≥30 is mathematically impossible

**This number is not useful for real-data projections** because the legacy
dataset's 34.8% doji rate fundamentally distorts signal detection. Real
projections must come from real data.

---

## Real Dataset Capacity Analysis (PENDING)

*This section will be populated after the Stage 3 run on VERIFIED_REAL data.*

### Projected metrics to compute:

1. **Observed recommendation rate (real data)**
   ```
   rate_per_bar = N_recommended / N_bars
   rate_per_year = rate_per_bar × 252  (approx trading days/year)
   ```

2. **Years required for N=30**
   ```
   years_for_30 = 30 / rate_per_year
   ```

3. **Available history capacity** (from named source)
   - Dukascopy EURUSD: ~20+ years from 2003 → ~5,000+ bars
   - MT5 broker export: varies by broker (typically 5–20 years)
   - HistData: ~20+ years from 2001

4. **Projection table**
   
   | History length | Projected N | Verdict possible? |
   |---|---|---|
   | 5 years | [PENDING] | [PENDING] |
   | 10 years | [PENDING] | [PENDING] |
   | 20 years | [PENDING] | [PENDING] |
   | Full Dukascopy (~23yr) | [PENDING] | [PENDING] |

---

## Sprint 17 Branch Pre-Selection (from capacity analysis)

The appropriate Sprint 17 branch depends on the real-data capacity number:

### Branch 17A — Extend real D1 history
**Trigger:** projected N ≥ 30 within the named source's available years  
**Action:** Owner acquires full available history; re-run Stage 3  
**Assessment:** [PENDING — depends on real rate]

### Branch 17B — H4 measurement
**Trigger:** D1 capacity insufficient AND source offers H4 data  
**Scope:** Phase 2 — requires owner sign-off  
**Caution:** Phase 1 patterns/gates were designed and tuned on D1 bars.
H4 bars have different OHLC structure; cross-timeframe validity is untested.  
**Assessment:** [PENDING]

### Branch 17C — Multi-pair D1 (non-JPY)
**Trigger:** D1 single-pair insufficient  
**Scope:** Symbol plumbing required  
**Known limitation:** `pip_size=0.0001` is hardcoded in
`PipelineRunner._setup_engines` — JPY pairs excluded until config plumbing
is added. Non-JPY candidates: GBPUSD, AUDUSD, NZDUSD  
**Assessment:** [PENDING]

### Branch 17D — Parameter governance review
**Trigger:** 17A–C all insufficient or rejected  
**Scope:** Strategy modification — last resort  
**Requirements:** formal pre-registration, train/validation split, owner sign-off  
**Assessment:** Premature to evaluate until 17A–C are measured

---

## Status

| Component | Status |
|---|---|
| Legacy capacity numbers | ✅ Available (useless for real verdict) |
| Real recommendation rate | ❌ BLOCKED — awaiting real data |
| Years-for-30 projection | ❌ BLOCKED |
| Sprint 17 branch recommendation | ❌ BLOCKED |

**Owner action:** Supply `data/EURUSD_D1_REAL_*.csv` to unblock this analysis.
