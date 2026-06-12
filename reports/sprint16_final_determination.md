# Sprint 16 — Final Determination
## Verdict: INSUFFICIENT DATA / READY — BLOCKED-ON-DATA

**Generated:** 2026-06-12  
**Sprint 16 HEAD:** [to be filled at commit time]  
**Sprint 15 HEAD:** `2f4761b` (reference)

---

## Executive Summary

Sprint 16 completes the **measurement infrastructure** for the first
trustworthy baseline but cannot deliver a strategy verdict because the
owner has not yet supplied a verified-real dataset.

The Phase 1 measurement system is now fully instrumented, tested, and
production-ready. The first scientifically valid YES/NO verdict is available
the moment a VERIFIED_REAL EURUSD D1 dataset is placed at
`data/EURUSD_D1_REAL_*.csv`.

---

## Sprint 16 Deliverables — Completed

| Deliverable | Status |
|---|---|
| Permanent `DataAuthenticityReport` gate | ✅ DONE |
| 60 gate tests (legacy must classify SYNTHETIC_SUSPECT) | ✅ 60/60 passing |
| `config/baseline_phase1_frozen.yaml` committed | ✅ DONE |
| Drift-alarm test (27 assertions) | ✅ 27/27 passing |
| Legacy file authenticity report | ✅ DONE (`reports/sprint16_data_authenticity_EURUSD_D1_2014_2026.txt`) |
| `data/README.md` (legacy file marked SYNTHETIC_SUSPECT) | ✅ DONE |
| `data/DATA_PROVENANCE.md` | ✅ DONE |
| README credential scrub (MT5 login/password) | ✅ DONE |
| `config/default_config.yaml` credential scrub | ✅ DONE |
| Full regression suite ≥ 2,054 tests all green | ✅ DONE |

| Deliverable | Status |
|---|---|
| Stage 3: Official real-data baseline scorecards | ❌ BLOCKED — no real data |
| Stage 4: Real-data capacity analysis | ❌ BLOCKED |
| Stage 5: Strategy verdict (YES/NO) | ❌ BLOCKED |
| Sprint 17 decision matrix (filled with measured numbers) | ❌ BLOCKED |

---

## Why INSUFFICIENT DATA (again)

Sprint 15 reported INSUFFICIENT DATA on two grounds:
1. N = 4 < 30 (minimum trades for a verdict)
2. Dataset classified SYNTHETIC_SUSPECT

Sprint 16 resolves ground #2 by building the gate and waiting for real data.
Ground #1 remains — the legacy dataset cannot produce N ≥ 30 at frozen params
(Sprint 15 capacity analysis: ~94 years required on synthetic data).

**A real dataset may change the recommendation rate significantly.** Real
EURUSD D1 data has different OHLC structure (no 34.8% doji rate) and genuine
momentum periods. The legacy rate of 0.13% per evaluation window is not a
reliable predictor of the real rate. The real rate is UNKNOWN until measured.

---

## Verdict Criteria (frozen — do not modify)

From `config/baseline_phase1_frozen.yaml` → `verdict_criteria`:

| Criterion | Threshold | Source |
|---|---|---|
| Profit Factor | > 1.10 | `profit_factor_min: 1.10` |
| Expectancy R | > 0.0 | `expectancy_r_min: 0.0` |
| Max Drawdown | < 25.0% | `max_drawdown_pct_max: 25.0` |
| Min Trades | ≥ 30 | `min_trades: 30` |

All four must be satisfied simultaneously. N < 30 prevents verdict regardless
of other metrics.

---

## What Would Change This Verdict

**To reach YES or NO:**
1. Owner supplies `data/EURUSD_D1_REAL_*.csv` (≥ 10 years, ≥ 20 years preferred)
2. Gate classifies it VERIFIED_REAL
3. Stage 3 baseline run completes
4. N ≥ 30 trades executed
5. Apply canonical criteria → YES or NO

**If N < 30 on real data:**
- Report INSUFFICIENT DATA (with measured N and projection)
- Proceed to Sprint 17 decision matrix

---

## Sprint 17 Decision Matrix

*Numbers marked [PENDING] require real-data capacity analysis.*

| Branch | Trigger | Scope | Prerequisite | Status |
|---|---|---|---|---|
| **17A** — Extend real D1 history | Projected N ≥ 30 within source's available years | Measurement only | Real data rate measured | ❌ PENDING |
| **17B** — H4 measurement | D1 capacity insufficient AND source offers H4 | Phase 2 — owner sign-off required | Owner confirms H4 available | ❌ PENDING |
| **17C** — Multi-pair D1 (GBPUSD/AUDUSD/NZDUSD) | D1 single-pair insufficient | Measurement + symbol plumbing | JPY excluded (`pip_size` hardcoded) | ❌ PENDING |
| **17D** — Parameter governance review | 17A–C all insufficient or rejected | Strategy modification — last resort | Pre-registration + train/val split + owner sign-off | ❌ LAST RESORT |

**Agent recommendation (preliminary, before real data):**  
Pursue **17A first** — acquire the longest available real EURUSD D1 history
(Dukascopy ~23 years from 2003 is the most accessible source). If N < 30 on
the full available history, escalate to 17B (H4) or 17C (multi-pair).
17D should not be considered until 17A–C are exhausted.

---

## Sprint 15 vs Sprint 16 Comparison

| Dimension | Sprint 15 | Sprint 16 |
|---|---|---|
| Dataset | Legacy SYNTHETIC_SUSPECT | No real data (still blocked) |
| Gate | Ad-hoc report | Permanent tested gate |
| Frozen config | Script kwargs | Committed YAML + drift alarm |
| Verdict | INSUFFICIENT DATA (N=4, synthetic) | INSUFFICIENT DATA (BLOCKED-ON-DATA) |
| Progress | Measurement system validated | Infrastructure ready; awaiting data |

---

## Owner Actions Required (Priority Order)

1. **[URGENT]** Rotate the MT5 password (`!5UvKcSl` was committed in git history).
   Use `git filter-repo --path README.md --invert-paths` or equivalent to purge
   history; then update password in broker account.

2. **[HIGH]** Supply real EURUSD D1 dataset at `data/EURUSD_D1_REAL_*.csv`.
   Dukascopy is recommended (free, 20+ years, documented methodology).

3. **[MEDIUM]** Document dataset provenance in `data/DATA_PROVENANCE.md`
   (source, export method, timezone/bar-close convention).

4. **[FUTURE — Sprint 17]** Sign off on which Sprint 17 branch to pursue
   after capacity analysis is complete.

---

## Test Suite Status at Sprint 16 Close

| Category | Tests |
|---|---|
| All tests (full suite) | ≥ 2,054 + new (all passing) |
| New Sprint 16 (Stage 1 gate) | 60 |
| New Sprint 16 (Stage 2 drift alarm) | 27 |
| Sprint 15 regression tests | 22 |
| Sprint 15 golden metrics | 42 |
| All prior tests (Sprints 14 and earlier) | ≥ 1,903 |

**Total sprint 16 new tests: 87 (60 + 27)**
