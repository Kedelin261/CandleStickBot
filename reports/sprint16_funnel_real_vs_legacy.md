# Sprint 16 — Funnel Comparison: Real Data vs Legacy Data
## Status: PLACEHOLDER — BLOCKED-ON-DATA

**Generated:** 2026-06-12  
**Real dataset:** NOT YET SUPPLIED  
**Legacy dataset:** `data/EURUSD_D1_2014_2026.csv` (SYNTHETIC_SUSPECT)

---

## Purpose

This report will quantify how much the synthetic legacy dataset distorted
Sprint 15's measurements. It answers: "Were the Sprint 15 numbers trustworthy
signals of the strategy's real-world performance?"

Expected finding: significant distortion. A dataset with 34.8% exact dojis
(real rate <1%) produces a different signal detection landscape than a real
feed. The legacy 0.13% recommendation rate may not be representative.

---

## Legacy Dataset Funnel (Sprint 15 V2, for reference)

```
Dataset: EURUSD_D1_2014_2026.csv (SYNTHETIC_SUSPECT)
Bars: 3,240  |  Lookback: 200  |  TQS: 0.0  |  RR: 2.0  |  Slippage: 1.0 pip

Pin Bar funnel:
  Evaluations  : 3,041
  SIGNAL reject: 2,882   (94.8%)  — no qualifying bar ≥ quality 5
  TREND  reject:   147   (4.8%)
  REGIME reject:     1   (0.0%)
  LEVEL  reject:     8   (0.3%)   [with FIX-1/2 SR wiring]
  RECOMMENDED  :     4   (0.13% of evaluations)

Engulfing funnel:
  RECOMMENDED  :     0

Combined:
  Trades executed: 4  (Pin Bar only)
  Win rate: 0/4 = 0.0%
  Net P&L: −$397.27
  Max DD:  3.97%
  Profit Factor: 0.0 (no wins)
  Expectancy: negative

Recommendation rate: 4 / 3,041 ≈ 0.13% per window
Recommendations per 1,000 bars: 4/3241 × 1000 ≈ 1.23 / 1,000 bars
```

---

## Real Dataset Funnel (PENDING)

*This section will be populated after the owner supplies a VERIFIED_REAL
dataset and the Stage 3 run is executed.*

```
Dataset: [PENDING]
Bars: [PENDING]  |  Same frozen config as legacy

Pin Bar funnel:
  Evaluations  : [PENDING]
  SIGNAL reject: [PENDING]
  TREND  reject: [PENDING]
  REGIME reject: [PENDING]
  LEVEL  reject: [PENDING]
  RECOMMENDED  : [PENDING]

Engulfing funnel: [PENDING]

Combined:
  Trades executed: [PENDING]
  ...

Recommendation rate: [PENDING]
Recommendations per 1,000 bars: [PENDING]
```

---

## Expected Comparison Dimensions

Once real data is available, this report will document:

1. **Raw pattern detection rate change**  
   Hypothesis: real data has fewer exact-doji bars → pattern detector sees
   more varied OHLC structure → signal quality distribution shifts.
   Direction unknown — real feeds may produce more OR fewer signals depending
   on OHLC variance matching pattern definitions.

2. **Per-gate pass rate change**  
   TREND gate (ADX) and REGIME gate behavior may differ significantly on
   real data where momentum periods are anchored to actual macro events
   vs synthetic flat periods.

3. **Recommendation rate per 1,000 bars**  
   This is the key capacity number. Legacy: ~1.23/1,000 bars.
   Real data may deviate by an order of magnitude in either direction —
   this is scientifically expected and should be reported without anchoring
   on the legacy figure.

4. **Distortion quantification**  
   `distortion_ratio = real_rate / legacy_rate`
   If `distortion_ratio >> 1`: legacy underestimated signal frequency.
   If `distortion_ratio << 1`: legacy overestimated (or real data is harder).
   Either result is a valid scientific finding.

---

## Status

| Component | Status |
|---|---|
| Legacy funnel numbers | ✅ Available from Sprint 15 |
| Real data funnel | ❌ BLOCKED — awaiting VERIFIED_REAL dataset |
| Comparison table | ❌ BLOCKED |
| Distortion quantification | ❌ BLOCKED |

**Owner action:** Supply `data/EURUSD_D1_REAL_*.csv` to unblock this report.
