# Data Provenance — CandleStickBot

This document records the origin, export method, and classification of every
dataset ever used in this project. Files classified `SYNTHETIC_SUSPECT` are
excluded from all official verdicts (they may be retained for archival
reproducibility of earlier sprints).

---

## `EURUSD_D1_2014_2026.csv`

| Field | Value |
|---|---|
| **File** | `data/EURUSD_D1_2014_2026.csv` |
| **Symbol** | EURUSD |
| **Timeframe** | D1 |
| **Rows** | 3,240 |
| **Date range** | 2014-01-01 → 2026-06-12 |
| **Coverage** | ~12.44 years |
| **Authenticity** | ❌ `SYNTHETIC_SUSPECT` |
| **Gate report** | `reports/sprint16_data_authenticity_EURUSD_D1_2014_2026.txt` |

### Classification Evidence

| Signal | Trigger | Value |
|---|---|---|
| H1 Doji rate | ❌ HARD FAIL | 34.75% (threshold ≥ 2%) |
| H2 Calendar-impossible bars | ❌ HARD FAIL | 16 bars (weekends/holidays) |
| H3 Future bars | ✅ OK | 0 |
| H4 Structural integrity | ✅ OK | Chronological, no duplicates |
| H5 Spot checks | N/A | No spot-check references provided |
| S1 Zero volume | ⚠️ WARN | 100% zero volume |
| S2 DOW distribution | ⚠️ WARN | Max deviation ±1 from mean 648 |
| S3 Holiday gaps | ⚠️ WARN | 0.1 gaps/year (expect ~6–12) |

A real EURUSD D1 feed has < 1% doji rate, ~6–12 holiday gaps per year,
non-uniform DOW distribution, and non-zero volume. This file fails on all
four dimensions — it is not suitable for measuring real-world strategy
performance.

### Origin (Unknown)

The provenance of this file before Sprint 14 is undocumented. It was the
only data file present in the repository at project inception. Based on the
authenticity signals (perfect DOW uniformity, calendar-day bars including
Jan 1 and Dec 25, 34.75% exact open==close bars), it appears to be a
**procedurally generated or heavily reconstructed dataset** — not a genuine
broker or data-provider export.

**Owner action:** If you know the origin of this file, document it here.
This information is useful for understanding Sprints 14–15 reproducibility.

### Usage History

| Sprint | Used for | Notes |
|---|---|---|
| Sprint 14 | Initial baseline exploration | Produced 0 trades (pre-FIX-1/2) |
| Sprint 14 | Root cause investigation | SR wiring bug discovered |
| Sprint 15 | Official V2 baseline (legacy) | Produced 4 trades (0W/4L, post-FIX) |
| Sprint 15 | Data authenticity audit | First SYNTHETIC_SUSPECT classification |
| Sprint 16 | Gate validation (must-fail test) | Confirms gate correctly rejects it |
| **All future** | **EXCLUDED** | Authenticity gate will reject it |

---

## `EURUSD_D1_REAL_<source>_<startyear>_<endyear>.csv` — AWAITED

*Entry to be filled by the project owner when the real dataset is supplied.*

| Field | Value |
|---|---|
| **File** | `data/EURUSD_D1_REAL_[TBD].csv` |
| **Symbol** | EURUSD |
| **Timeframe** | D1 |
| **Source** | [OWNER TO FILL: e.g., Dukascopy, MT5 broker export, HistData] |
| **Export method** | [OWNER TO FILL: e.g., MT5 History Export, Dukascopy web download] |
| **Export date** | [OWNER TO FILL] |
| **Timezone** | [OWNER TO FILL: e.g., UTC, NY 17:00, broker server time] |
| **Bar-close convention** | [OWNER TO FILL: e.g., "daily bar closes at NY 17:00 EST"] |
| **Authenticity** | [To be determined by gate after file is supplied] |

---

## Adding New Entries

When a new dataset is added:

1. Run `run_authenticity_check()` and save the report to `reports/`.
2. Fill in the table above with: file name, symbol, timeframe, rows,
   date range, source, export method, export date, timezone, bar-close
   convention, gate result, and gate report path.
3. If `VERIFIED_REAL`, it may be used for official scorecards.
4. If `SYNTHETIC_SUSPECT` or `MIXED_RECONSTRUCTED`, mark as excluded and
   document why it was retained (archival/reproducibility only).
