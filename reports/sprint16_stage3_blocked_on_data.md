# Sprint 16 — Stage 3: Official Real-Data Baseline
## Status: READY / BLOCKED-ON-DATA

**Generated:** 2026-06-12  
**Gate status:** Infrastructure COMPLETE — awaiting owner-supplied dataset

---

## Gate Prerequisites (Stage 1 → Stage 3 handoff)

Before any official baseline run, the candidate dataset must pass the
`DataAuthenticityReport` gate implemented in `src/backtesting/data_authenticity.py`.

| Requirement | Status |
|---|---|
| Gate implemented and tested | ✅ DONE (60 tests, 60/60 passing) |
| Legacy file correctly classified SYNTHETIC_SUSPECT | ✅ CONFIRMED |
| Frozen config artifact committed | ✅ DONE (`config/baseline_phase1_frozen.yaml`) |
| Drift-alarm test in place | ✅ DONE (27 tests, 27/27 passing) |
| Real dataset present at `data/EURUSD_D1_REAL_*.csv` | ❌ NOT YET SUPPLIED |

---

## Blocking Condition

The project owner has not yet supplied a candidate real-data CSV file.
No dataset exists at any path matching `data/EURUSD_D1_REAL_*.csv`.

**This sprint cannot proceed to Stage 3 until the owner places a verified
real dataset at the expected path.** All Stage 1–2 infrastructure is complete
and ready to receive the file.

---

## Owner Action Required

Place the candidate file at:
```
data/EURUSD_D1_REAL_<source>_<startyear>_<endyear>.csv
```

**Required file properties:**
- Symbol: EURUSD, Timeframe: D1
- Minimum 10 years of history (20+ years strongly preferred for N≥30 capacity)
- Columns: `date,open,high,low,close[,volume]` — ascending, single header
- Raw broker/provider export — no hand edits, no gap-filling, no smoothing

**Acceptable sources (in order of preference):**
1. Owner's broker MT5 export — matches execution venue
2. Dukascopy historical data (EURUSD from ~2003, ~20+ years available)
3. HistData.com M1 aggregated to D1 (document bar-close session boundary)

**Also required:**
- `data/DATA_PROVENANCE.md` entry documenting: source name, export method,
  export date, timezone/daily bar-close convention

---

## Runbook: After Owner Supplies Dataset

Once the file is placed, execute in order:

### Step 1 — Authenticity gate
```bash
cd /home/user/CandleStickBot
python3 -c "
from src.backtesting.data_authenticity import run_authenticity_check
from datetime import date
import glob, os

files = glob.glob('data/EURUSD_D1_REAL_*.csv')
if not files:
    print('ERROR: No real data file found')
else:
    path = files[0]
    name = os.path.basename(path).replace('.csv', '')
    rpt = run_authenticity_check(
        path,
        f'reports/sprint16_data_authenticity_{name}.txt',
        run_date=date.today(),
        spot_checks=None,
    )
    print(rpt.summary())
    print(f'Classification: {rpt.classification}')
"
```

**If classification = SYNTHETIC_SUSPECT:** Dataset rejected. Owner must supply
a different file. Do NOT proceed to backtesting on a rejected dataset.

**If classification = VERIFIED_REAL:** Proceed to Step 2.

**If classification = MIXED_RECONSTRUCTED:** Owner must assess warnings.
Proceed only with explicit owner sign-off, clearly labeled APPENDIX.

### Step 2 — Official baseline runs (OFFICIAL variant, tqs=0)
```bash
# Run Pin Bar Only
python3 scripts/run_baseline.py \
  --csv data/EURUSD_D1_REAL_*.csv \
  --strategy pin_bar \
  --config config/baseline_phase1_frozen.yaml \
  --variant official \
  --output reports/sprint16_pin_bar_scorecard.txt

# Run Engulfing Only  
python3 scripts/run_baseline.py \
  --csv data/EURUSD_D1_REAL_*.csv \
  --strategy engulfing \
  --config config/baseline_phase1_frozen.yaml \
  --variant official \
  --output reports/sprint16_engulfing_scorecard.txt

# Run Combined
python3 scripts/run_baseline.py \
  --csv data/EURUSD_D1_REAL_*.csv \
  --strategy combined \
  --config config/baseline_phase1_frozen.yaml \
  --variant official \
  --output reports/sprint16_combined_scorecard.txt
```

### Step 3 — Appendix runs (appendix_tqs60 variant)
Same commands as Step 2, `--variant appendix_tqs60`, output to
`reports/sprint16_*_scorecard_tqs60_appendix.txt`.

### Step 4 — Verdict
Apply canonical criteria from `config/baseline_phase1_frozen.yaml`:
- PF > 1.10 AND Exp > 0 AND DD < 25% AND N ≥ 30 → **YES or NO**
- N < 30 → **INSUFFICIENT DATA** (proceed to capacity analysis)

---

## Expected Real-Data Calibration Points

The following are expected properties of a genuinely real EURUSD D1 feed.
If the supplied file deviates significantly from these, the gate may reject it.

| Property | Expected value | Legacy file (SYNTHETIC_SUSPECT) |
|---|---|---|
| Exact doji rate (open==close 5dp) | < 1% | 34.75% ❌ |
| Calendar-impossible bars | 0 | 16 ❌ |
| Zero-volume bars | < 5% | 100% ❌ |
| DOW distribution uniformity | Non-uniform (holiday gaps) | Max dev ±1 ❌ |
| Holiday gaps per year | ~6–12 | 0.1 ❌ |
| Years of history | ≥ 10 (20+ preferred) | 12.44 |
| Total bars (20yr) | ~5,000–5,200 | N/A |

**Known spot-check anchors for EURUSD D1:**
- Early Jan 2021: ~1.2100–1.2250
- 2022-09-27 (cycle low area): ~0.9556–0.9620
- Early Jan 2017: ~1.0400–1.0550
- Spring 2014 (Apr–May): ~1.3700–1.3950

---

## Infrastructure Readiness Summary

| Component | Status |
|---|---|
| `src/backtesting/data_authenticity.py` | ✅ Implemented, 60 tests passing |
| `config/baseline_phase1_frozen.yaml` | ✅ Committed |
| `tests/backtesting/test_sprint16_frozen_config.py` | ✅ 27 tests passing |
| `data/README.md` | ✅ Created |
| `data/DATA_PROVENANCE.md` | ✅ Created (legacy entry) |
| `reports/sprint16_data_authenticity_EURUSD_D1_2014_2026.txt` | ✅ Generated |
| Baseline run scripts | ⏳ Pending (will be written when data arrives) |
| Stage 3 scorecards | ❌ BLOCKED — awaiting real data |
| Stage 4 capacity analysis | ❌ BLOCKED — awaiting real data |
| Stage 5 verdict | ❌ BLOCKED — awaiting real data |
