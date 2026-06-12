# CandleStickBot — Data Directory

## Overview

This directory contains market data files used for backtesting.
Every file used in official baseline runs MUST pass the
`DataAuthenticityReport` gate before any results are considered valid.

**Gate module:** `src/backtesting/data_authenticity.py`  
**Gate tests:** `tests/backtesting/test_sprint16_data_authenticity.py` (60 tests)

---

## File Inventory

### `EURUSD_D1_2014_2026.csv`

| Property | Value |
|---|---|
| **Classification** | ❌ `SYNTHETIC_SUSPECT` |
| **Gate report** | `reports/sprint16_data_authenticity_EURUSD_D1_2014_2026.txt` |
| **Hard fails** | H1 (doji rate 34.75%), H2 (16 calendar-impossible bars) |
| **Soft warnings** | S1 (100% zero volume), S2 (uniform DOW), S3 (0.1 gaps/year) |
| **Excluded from** | ALL verdicts, ALL official baseline scorecards |
| **Used for** | Sprint 14–15 legacy reproduction only (archival) |

**Do not use this file for any forward-looking measurement.** Results on
this file are documented in Sprint 14–15 reports for reproducibility but
carry no scientific weight about real-world strategy performance.

See `data/DATA_PROVENANCE.md` for full origin documentation.

---

### `EURUSD_D1_REAL_<source>_<startyear>_<endyear>.csv` — AWAITED

This file does not yet exist. The project owner must supply a verified-real
dataset matching this naming convention before Sprint 16 Stage 3 can proceed.

**Requirements:**
- Symbol: EURUSD, Timeframe: D1
- Minimum 10 years (20+ years strongly preferred)
- Columns: `date,open,high,low,close[,volume]`
- Raw broker/provider export — no edits, no gap-filling
- Must classify as `VERIFIED_REAL` to be used for official scorecards

See `reports/sprint16_stage3_blocked_on_data.md` for the full runbook.

---

## Adding New Data Files

1. Place the file in this directory with format `<SYMBOL>_<TF>_<SOURCE>_<YEAR>_<YEAR>.csv`
2. Run the authenticity gate:
   ```python
   from src.backtesting.data_authenticity import run_authenticity_check
   rpt = run_authenticity_check(
       'data/<filename>.csv',
       'reports/sprint16_data_authenticity_<filename>.txt',
   )
   print(rpt.summary())
   ```
3. If `rpt.classification == 'VERIFIED_REAL'`, add a provenance entry to
   `data/DATA_PROVENANCE.md` and proceed to backtesting.
4. If `SYNTHETIC_SUSPECT` or `MIXED_RECONSTRUCTED`, do NOT run official
   scorecards on the file. Investigate the flagged signals and either source
   a better file or obtain owner sign-off for APPENDIX-labeled runs.
