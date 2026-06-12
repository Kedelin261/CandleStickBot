"""
Sprint 14 SR Engine investigation v2 — test with M03 swing data wired to M05.
This tests what the pipeline SHOULD do: pass structure_a.swing_highs/lows to sr.analyze().
Also compares pipeline's current call (no swing data) vs. the properly wired call.
"""
import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import csv
from src.data.types import CandleData
from src.analysis.sr_engine import SREngine
from src.analysis.trend_detection import TrendDetector
from src.analysis.market_regime import MarketRegimeEngine, RegimeType
from src.analysis.market_structure import MarketStructureAnalyzer

# Load CSV
candles = []
with open('data/EURUSD_D1_2014_2026.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        candles.append(CandleData(
            timestamp=row['date'],
            open=float(row['open']),
            high=float(row['high']),
            low=float(row['low']),
            close=float(row['close']),
            volume=float(row['volume']),
            symbol='EURUSD',
            timeframe='D1',
        ))

print(f"Loaded {len(candles)} candles")
print()

sr_engine   = SREngine(pip_size=0.0001)
td_engine   = TrendDetector(sma_period=21)
mr_engine   = MarketRegimeEngine()
ms_analyzer = MarketStructureAnalyzer()

print("="*80)
print("PART 1: SREngine called WITHOUT swing data (current pipeline behaviour)")
print("="*80)

sample_step = 5
window = 200
sr_ok = 0; total = 0
for i in range(window, len(candles), sample_step):
    ctx = candles[i - window: i]
    sr_result = sr_engine.analyze(ctx)   # no swing data
    n_levels = len(sr_result.support_levels) + len(sr_result.resistance_levels)
    total += 1
    if n_levels > 0: sr_ok += 1
print(f"Window={window}, samples={total}: SR levels>0 in {sr_ok}/{total} ({100*sr_ok/max(total,1):.1f}%)")

print()
print("="*80)
print("PART 2: SREngine called WITH M03 swing data (correctly wired)")
print("="*80)
print()
print(f"{'Window':>8} | {'SR>0':>22} | {'Tradeable':>20} | {'TRENDING':>20} | {'All-3-gates':>20} | {'Avg SR levels':>15}")
print("-" * 120)

for window in [50, 100, 150, 200, 300, 500]:
    sr_ok = 0; trend_ok = 0; regime_ok = 0; all3 = 0; total = 0
    sr_total_levels = 0

    for i in range(window, len(candles), sample_step):
        ctx = candles[i - window: i]

        try:
            structure_a = ms_analyzer.analyze(ctx)
            # Wire M03 swings to M05 — pass swing_highs and swing_lows
            sr_result   = sr_engine.analyze(
                ctx,
                swing_highs=[sp.candle.high for sp in structure_a.swing_highs],
                swing_lows=[sp.candle.low for sp in structure_a.swing_lows],
            )
            td_result   = td_engine.analyze(ctx, market_structure=structure_a.to_market_structure())
            mr_result   = mr_engine.analyze(ctx, adx=td_result.adx)
        except Exception as e:
            continue

        n_levels  = len(sr_result.support_levels) + len(sr_result.resistance_levels)
        has_sr    = n_levels > 0
        tradeable = getattr(td_result, 'tradeable', False)
        is_trend  = getattr(mr_result, 'regime', None) == RegimeType.TRENDING

        total      += 1
        sr_total_levels += n_levels
        if has_sr:                               sr_ok     += 1
        if tradeable:                            trend_ok  += 1
        if is_trend:                             regime_ok += 1
        if has_sr and tradeable and is_trend:    all3      += 1

    def pct(n, t=total):
        return f"{n}/{t} ({100*n/max(t,1):.1f}%)"

    avg_levels = sr_total_levels / max(total, 1)
    print(f"{window:>8} | {pct(sr_ok):>28} | {pct(trend_ok):>26} | {pct(regime_ok):>26} | {pct(all3):>24} | {avg_levels:>15.2f}")

print()
print("="*80)
print("PART 3: Swing point count vs. window size (to understand swing detection)")
print("="*80)
print()

for window in [50, 100, 200]:
    swing_h_counts = []
    swing_l_counts = []
    for i in range(window, len(candles), sample_step):
        ctx = candles[i - window: i]
        s = ms_analyzer.analyze(ctx)
        swing_h_counts.append(len(s.swing_highs))
        swing_l_counts.append(len(s.swing_lows))
    
    total = len(swing_h_counts)
    any_swing = sum(1 for h,l in zip(swing_h_counts, swing_l_counts) if h>0 or l>0)
    avg_h = sum(swing_h_counts)/max(total,1)
    avg_l = sum(swing_l_counts)/max(total,1)
    print(f"Window={window:3d}: avg_swing_highs={avg_h:.2f} avg_swing_lows={avg_l:.2f} "
          f"any_swing={any_swing}/{total} ({100*any_swing/max(total,1):.1f}%)")
